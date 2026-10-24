"""HTTP 服务层：报告数据与模型目录清单的路由注册。

只做参数校验、线程调度与错误映射；报告内容由 :mod:`report` 组装。
仅在 ComfyUI 已加载 ``server`` 且 ``PromptServer.instance`` 存在时注册，
测试或旧版环境下静默跳过。
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

import folder_paths
from aiohttp import web

from .folders import list_dir_files, list_model_dirs, resolve_model_path
from .report import ReportError, build_report, prepare_report

logger = logging.getLogger(__package__.split(".")[0])

WEB_REPORT_DIR = Path(__file__).resolve().parent.parent / "web" / "report"

_REPORT_STATUS = {
    "MODEL_NOT_FOUND": 404,
    "MODEL_LOAD_FAILED": 422,
    "ANALYSIS_FAILED": 500,
}


class _RequestError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _json(data, status=200):
    return web.json_response(
        data,
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def _error(status, code, message):
    return _json({"error": message, "code": code}, status=status)


def _normalize_model_name(name):
    return os.path.normcase(name.replace("\\", "/"))


def _match_model_name(model_name, files):
    """在目录文件清单里找到实际条目；容忍 URL 中的分隔符差异。"""
    if model_name in files:
        return model_name
    wanted = _normalize_model_name(model_name)
    return next(
        (entry for entry in files if _normalize_model_name(entry) == wanted),
        None,
    )


def _allowed_roots(model_dir):
    """该目录允许的根路径：models/<dir> 与 extra_model_paths 注册路径。"""
    roots = []
    models_root = getattr(folder_paths, "models_dir", None)
    if models_root:
        roots.append(os.path.join(models_root, model_dir))
    registered = getattr(folder_paths, "folder_names_and_paths", {}).get(model_dir)
    if registered:
        roots.extend(registered[0])
    return [os.path.realpath(root) for root in roots]


def _inside_roots(path, roots):
    if not roots:
        return True  # 没有可用的 models_dir：交给 resolve_model_path 的白名单
    target = os.path.normcase(os.path.realpath(path))
    for root in roots:
        root = os.path.normcase(root)
        try:
            if os.path.commonpath([root, target]) == root:
                return True
        except ValueError:
            continue
    return False


def _parse_request(query):
    model_dir = (query.get("dir") or "").strip()
    model_name = (query.get("model") or "").strip()
    keys_raw = query.get("keys", "1")

    if model_dir not in list_model_dirs():
        raise _RequestError(400, "INVALID_DIR", "Invalid model directory")

    canonical = _match_model_name(model_name, list_dir_files(model_dir))
    if canonical is None:
        raise _RequestError(
            400, "INVALID_MODEL", "Model is not in the selected directory"
        )

    if keys_raw not in ("0", "1"):
        raise _RequestError(400, "INVALID_KEYS", "keys must be 0 or 1")

    path = resolve_model_path(model_dir, canonical)
    if not path:
        raise _RequestError(404, "MODEL_NOT_FOUND", "Model not found")
    if not _inside_roots(path, _allowed_roots(model_dir)):
        raise _RequestError(403, "FORBIDDEN", "Model path is outside the allowed roots")

    return model_dir, canonical, keys_raw == "1"


async def handle_folder_files(_request):
    try:
        folders = {name: list_dir_files(name) for name in list_model_dirs()}
    except Exception:
        logger.exception("Failed to list model files")
        return _error(500, "INTERNAL_ERROR", "Failed to list model files")
    return _json(folders)


async def handle_report_page(_request):
    index = WEB_REPORT_DIR / "index.html"
    if not index.is_file():
        return _error(404, "REPORT_PAGE_MISSING", "Report page is not available")
    return web.FileResponse(index, headers={"Cache-Control": "no-cache"})


async def handle_report_data(request):
    try:
        model_dir, model_name, include_keys = _parse_request(request.query)
    except _RequestError as e:
        return _error(e.status, e.code, e.message)

    try:
        data = await asyncio.to_thread(
            build_report, model_dir, model_name, include_keys
        )
    except ReportError as e:
        return _error(_REPORT_STATUS.get(e.code, 400), e.code, e.message)
    except Exception:
        logger.exception("Failed to build report for %s/%s", model_dir, model_name)
        return _error(500, "INTERNAL_ERROR", "Failed to build report")

    return _json(data)


async def handle_report_prepare(request):
    """报告按钮的显式刷新：重建报告并预热缓存，不执行整张工作流。"""
    try:
        payload = await request.json()
    except Exception:
        return _error(400, "INVALID_BODY", "Expected a JSON object")
    if not isinstance(payload, dict):
        return _error(400, "INVALID_BODY", "Expected a JSON object")

    try:
        model_dir, model_name, _ = _parse_request(payload)
    except _RequestError as e:
        return _error(e.status, e.code, e.message)

    try:
        data = await asyncio.to_thread(
            prepare_report, model_dir, model_name
        )
    except ReportError as e:
        return _error(_REPORT_STATUS.get(e.code, 400), e.code, e.message)
    except Exception:
        logger.exception("Failed to prepare report for %s/%s", model_dir, model_name)
        return _error(500, "INTERNAL_ERROR", "Failed to prepare report")

    return _json({"ok": True, "schema_version": data.get("schema_version")})


def register_routes():
    """注册路由；ComfyUI server 不可用时返回 False。"""
    if "server" not in sys.modules:
        return False
    try:
        from server import PromptServer
    except Exception:
        return False

    instance = getattr(PromptServer, "instance", None)
    routes = getattr(instance, "routes", None)
    if routes is None:
        return False

    try:
        routes.get("/model_toolkit/report")(handle_report_page)
        routes.get("/model_toolkit/folder_files")(handle_folder_files)
        routes.get("/model_toolkit/report/data")(handle_report_data)
        routes.post("/model_toolkit/report/prepare")(handle_report_prepare)
        if WEB_REPORT_DIR.is_dir():
            routes.static("/model_toolkit/report/assets", str(WEB_REPORT_DIR))
    except Exception as e:
        logger.warning(f"Failed to register model toolkit routes: {e}")
        return False
    return True
