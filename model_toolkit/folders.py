"""模型目录枚举与路径解析：驱动模型检查器的「模型目录」下拉。

设计要点：
- 下拉列的是 models/ 下第一层子目录的磁盘实况，而不是
  folder_names_and_paths 的键——后者包含 custom_nodes/datasets 等
  非模型目录，又漏掉 ipadapter/onnx 等未注册目录（它们同样在 models/ 下）。
- 已注册目录（含 unet/clip 等 map_legacy 历史别名）交给 folder_paths，
  沿用它的扩展名过滤与多路径注册；未注册目录自行递归扫描。
- 默认目录固定为 checkpoints，并始终排在选项第一位；不再提供「全部」。
- 选择某个目录后，文件名严格只在该目录内解析，不做跨目录回退。
"""
import logging
import os

import folder_paths

logger = logging.getLogger(__package__.split(".")[0])

DEFAULT_DIR = "checkpoints"

# folder_paths.supported_pt_extensions 取不到时（极旧版本）的兜底扩展名
_FALLBACK_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".pkl", ".pt2")


def _models_root():
    """ComfyUI 的 models/ 根目录；取不到返回 None。"""
    return getattr(folder_paths, "models_dir", None)


def _supported_extensions():
    """模型文件扩展名（小写）；优先复用 folder_paths 的集合。"""
    extensions = getattr(folder_paths, "supported_pt_extensions", None)
    if extensions:
        return {ext.lower() for ext in extensions}
    return set(_FALLBACK_EXTENSIONS)


def _is_plain_dir_name(dir_name):
    """只接受 models/ 下的一级目录名，拒绝路径分隔符与相对路径。"""
    return (
        isinstance(dir_name, str)
        and bool(dir_name)
        and dir_name not in (".", "..")
        and "/" not in dir_name
        and "\\" not in dir_name
    )


def _is_safe_model_name(model_name):
    """允许子目录，但拒绝绝对路径与 .. 穿越（模型名可能来自工作流 JSON）。"""
    if not isinstance(model_name, str) or not model_name:
        return False
    normalized = model_name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return False
    return ".." not in normalized.split("/")


def list_model_dirs():
    """模型目录下拉的选项：checkpoints 固定第一，其余 models/ 一级目录排序。"""
    root = _models_root()
    if not root or not os.path.isdir(root):
        return [DEFAULT_DIR]
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(root)
            if entry.is_dir() and entry.name != DEFAULT_DIR
        )
    except OSError as e:
        logger.warning(f"Failed to scan model directories: {e}")
        return [DEFAULT_DIR]
    return [DEFAULT_DIR, *names]


def list_dir_files(dir_name):
    """目录下的模型文件（相对路径，递归，排序）。

    已注册目录（含 unet/clip 别名）走 folder_paths.get_filename_list；
    未注册目录自行递归扫描（folder_paths 对未注册名会抛 KeyError）。
    """
    if not dir_name:
        return []
    try:
        return list(folder_paths.get_filename_list(dir_name))
    except Exception:
        pass  # 未注册目录：落到下面的磁盘扫描

    root = _models_root()
    if not root:
        return []
    base = os.path.join(root, dir_name)
    extensions = _supported_extensions()
    found = []
    for dirpath, _dirs, files in os.walk(base):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in extensions:
                found.append(os.path.relpath(os.path.join(dirpath, file_name), base))
    return sorted(found)


def _lookup(dir_name, model_name):
    """在单个目录内解析文件路径；找不到或越界返回 None。"""
    try:
        path = folder_paths.get_full_path(dir_name, model_name)
    except Exception:
        path = None
    if path:
        return path

    root = _models_root()
    if not root:
        return None
    base = os.path.realpath(os.path.join(root, dir_name))
    candidate = os.path.realpath(os.path.join(base, model_name))
    try:
        inside = os.path.commonpath(
            [os.path.normcase(base), os.path.normcase(candidate)]
        ) == os.path.normcase(base)
    except ValueError:
        inside = False
    if inside and os.path.isfile(candidate):
        return candidate
    return None


def resolve_model_path(model_dir, model_name):
    """把「目录 + 文件名」解析为磁盘路径；失败返回 None。

    指定目录时只在该目录内找，绝不回退到其他目录（避免选了 loras
    却悄悄解析到 checkpoints 的同名文件）。
    """
    if not _is_plain_dir_name(model_dir) or not _is_safe_model_name(model_name):
        return None
    return _lookup(model_dir, model_name)
