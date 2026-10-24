"""模型报告数据层：把一次分析整理成可 JSON 序列化的唯一数据源。

页面（P2）只消费这里的 dict；本模块不碰 HTTP、不渲染。缓存键包含文件
realpath、mtime、大小与是否包含键名清单，同名文件被覆盖后自动失效。
"""
import logging
import math
import os
from functools import lru_cache

from . import structure as structure_mod
from . import templates
from .analysis import AnalysisMixin
from .architectures import identify, report_hint
from .core import ModelUtils
from .folders import resolve_model_path
from .precision import PrecisionMixin

logger = logging.getLogger(__package__.split(".")[0])

SCHEMA_VERSION = "1.0"

_COMPONENT_ORDER = ("UNET", "CLIP", "VAE", "EMA", "Other")


class ReportError(Exception):
    """报告构建失败；``code`` 供服务层映射成错误响应。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def build_report(model_dir, model_name, include_keys=True):
    """构建模型报告。

    返回的 dict 可被缓存复用，调用方应视为只读。找不到模型或无法加载时
    抛出 :class:`ReportError`。
    """
    model_path = resolve_model_path(model_dir, model_name)
    if not model_path:
        raise ReportError("MODEL_NOT_FOUND", f"Model not found: {model_dir}/{model_name}")

    try:
        stat = os.stat(model_path)
    except OSError as e:
        raise ReportError("MODEL_NOT_FOUND", f"Model not found: {model_dir}/{model_name}") from e

    return _build_cached(
        os.path.realpath(model_path),
        stat.st_mtime_ns,
        stat.st_size,
        bool(include_keys),
        model_dir,
        model_name,
    )


def clear_cache():
    """清空进程内报告缓存（测试与文件监控用）。"""
    _build_cached.cache_clear()


def prepare_report(model_dir, model_name):
    """重新构建报告并写入缓存，供报告按钮的显式刷新使用。

    先清空缓存再构建，确保模型文件被同名覆盖但 mtime/大小未变时，
    页面也不会命中旧报告。
    """
    clear_cache()
    return build_report(model_dir, model_name, True)


@lru_cache(maxsize=8)
def _build_cached(real_path, mtime_ns, size, include_keys, model_dir, model_name):
    model_data = ModelUtils.load_model_index(real_path)
    if not model_data:
        raise ReportError("MODEL_LOAD_FAILED", "Failed to load model file")

    try:
        return _assemble(real_path, size, include_keys, model_dir, model_name, model_data)
    except ReportError:
        raise
    except Exception as e:
        logger.exception("Failed to build model report")
        raise ReportError("ANALYSIS_FAILED", f"Analysis failed: {e}") from e


def _assemble(real_path, size, include_keys, model_dir, model_name, model_data):
    total_params = sum(meta.numel() for meta in model_data.values())
    if total_params <= 0:
        raise ReportError("MODEL_LOAD_FAILED", "Model contains no tensors")

    keys = sorted(model_data)
    groups = ModelUtils.group_model_keys(model_data)
    metadata = _read_metadata(real_path, model_data)
    architecture = _architecture(model_data, keys, metadata)
    components = _components(model_data, groups, total_params)
    precision = _precision(model_data, groups)

    report = {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "file_name": model_name,
            "model_dir": model_dir,
            "size_bytes": size,
            "size_formatted": ModelUtils.format_size(size),
            "total_params": total_params,
            "total_params_formatted": ModelUtils.format_params(total_params),
            "key_count": len(model_data),
            "component_count": len(components),
            "primary_precision": precision["primary"],
            "architecture": architecture["type"],
        },
        "architecture": architecture,
        "components": components,
        "precision": precision,
        "metadata": metadata,
        "structure": structure_mod.extract(
            model_data, groups, architecture["type"]
        ),
    }
    if not include_keys:
        report["structure"]["unattributed"].pop("keys", None)
    if include_keys:
        report["keys"] = _key_entries(model_data, groups)
    return report


def _read_metadata(real_path, model_data):
    metadata = ModelUtils.read_safetensors_metadata(real_path)
    if not metadata:
        metadata = {
            key: value for key, value in model_data.items() if "_metadata" in key
        }
    return {str(key): _json_safe(metadata[key]) for key in sorted(metadata)}


def _architecture(model_data, keys, metadata):
    spec, coverage = templates.detect(keys)
    family = identify(keys, metadata)
    heuristic, _debug = AnalysisMixin().analyze_architecture(model_data)

    arch_type = spec.display_name if spec else heuristic["type"]
    if arch_type == "Unknown" and family is not None:
        arch_type = family.display

    return {
        "type": arch_type,
        "family": family.family if family else None,
        "capability": family.capability if family else "unknown",
        "note": family.note if family else "",
        "template": spec.identifier if spec else None,
        "template_coverage": round(coverage, 4),
        "hint": report_hint(family),
        "extra_info": list(heuristic.get("extra_info", [])),
    }


def _components(model_data, groups, total_params):
    names = [name for name in _COMPONENT_ORDER if name in groups]
    names += sorted(name for name in groups if name not in _COMPONENT_ORDER)

    components = {}
    for name in names:
        if not groups[name]:
            continue
        params = sum(model_data[key].numel() for key in groups[name])
        components[name] = {
            "params": params,
            "percentage": round(params / total_params * 100, 2),
            "key_count": len(groups[name]),
        }
    return components


def _precision(model_data, groups):
    stats = PrecisionMixin().analyze_model_precision(model_data)
    distributions = [
        {
            "label": label,
            "params": count,
            "percentage": round(percentage, 2),
        }
        for label, (count, percentage) in sorted(
            stats["distributions"].items(), key=lambda item: item[1][0], reverse=True
        )
    ]

    per_component = PrecisionMixin().analyze_component_precision(model_data, groups)
    component_precision = {
        name: [
            {"label": label, "params": count, "percentage": round(percentage, 2)}
            for label, (count, percentage) in sorted(
                per_component.get(name, {}).items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
        ]
        for name in _COMPONENT_ORDER
        if name in per_component
    }
    for name in sorted(per_component):
        if name not in component_precision:
            component_precision[name] = [
                {"label": label, "params": count, "percentage": round(percentage, 2)}
                for label, (count, percentage) in sorted(
                    per_component[name].items(),
                    key=lambda item: item[1][0],
                    reverse=True,
                )
            ]

    return {
        "primary": stats["primary_precision"],
        "distributions": distributions,
        "components": component_precision,
    }


def _key_entries(model_data, groups):
    component_of = {
        key: name for name, keys in groups.items() for key in keys
    }
    return [
        {
            "name": key,
            "shape": list(meta.shape),
            "dtype": str(meta.dtype),
            "params": meta.numel(),
            "component": component_of.get(key, "Other"),
        }
        for key, meta in sorted(model_data.items())
    ]


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return str(value)
