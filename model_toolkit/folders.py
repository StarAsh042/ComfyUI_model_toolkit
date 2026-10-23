"""模型目录枚举与路径解析：驱动模型检查器的「模型目录」下拉。

设计要点：
- 下拉列的是 models/ 下第一层子目录的磁盘实况，而不是
  folder_names_and_paths 的键——后者包含 custom_nodes/datasets 等
  非模型目录，又漏掉 ipadapter/onnx 等未注册目录（它们同样在 models/ 下）。
- 已注册目录（含 unet/clip 等 map_legacy 历史别名）交给 folder_paths，
  沿用它的扩展名过滤与多路径注册；未注册目录自行递归扫描。
- 「（全部）」合并所有目录的文件，与前端 js/model_dir_filter.js 的
  合并去重顺序保持一致（都按目录枚举顺序），同名文件解析结果可预期。
"""
import logging
import os

import folder_paths

logger = logging.getLogger(__package__.split(".")[0])

# 「（全部）」哨兵值：合并所有目录。Python 与前端 JS 必须保持一致
ALL_DIRS = "（全部）"

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


def list_model_dirs():
    """模型目录下拉的选项：['（全部）', ...models/ 第一层子目录（排序）]。"""
    root = _models_root()
    if not root or not os.path.isdir(root):
        return [ALL_DIRS]
    try:
        names = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    except OSError as e:
        logger.warning(f"扫描模型目录失败: {e}")
        return [ALL_DIRS]
    return [ALL_DIRS, *names]


def list_dir_files(dir_name):
    """目录下的模型文件（相对路径，递归，排序）。

    已注册目录（含 unet/clip 别名）走 folder_paths.get_filename_list；
    未注册目录自行递归扫描（folder_paths 对未注册名会抛 KeyError）。
    """
    if not dir_name or dir_name == ALL_DIRS:
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
    """在单个目录内解析文件路径；找不到返回 None。"""
    try:
        path = folder_paths.get_full_path(dir_name, model_name)
    except Exception:
        path = None
    if path:
        return path
    root = _models_root()
    if root:
        candidate = os.path.join(root, dir_name, model_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_model_path(model_dir, model_name):
    """把「目录 + 文件名」解析为磁盘路径；失败返回 None。

    指定目录时只在该目录内找（与前端联动下拉同一语义，避免选了 loras
    却悄悄解析到 checkpoints 的同名文件）；「（全部）」按目录枚举顺序
    回退查找，与前端合并去重顺序一致。
    """
    if not model_name:
        return None
    if model_dir and model_dir != ALL_DIRS:
        return _lookup(model_dir, model_name)
    for dir_name in list_model_dirs():
        if dir_name == ALL_DIRS:
            continue
        path = _lookup(dir_name, model_name)
        if path:
            return path
    return None
