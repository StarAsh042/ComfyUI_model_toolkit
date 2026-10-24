"""CI 用的最小 ``folder_paths`` 替身。

GitHub Runner 上没有 ComfyUI 与真实模型；CI 通过
``COMFYUI_DIR=tests/ci`` 让 ``tests/_env.py`` 把这个目录加入 ``sys.path``，
从而导入本替身而不是 ComfyUI 的真实模块。只实现插件与测试用到的 API，
并创建空模型目录和一个占位文件，供目录枚举类测试使用。
"""
import os
import tempfile

models_dir = os.path.join(tempfile.gettempdir(), "model_toolkit_ci_models")
output_directory = os.path.join(tempfile.gettempdir(), "model_toolkit_ci_output")
supported_pt_extensions = {".safetensors", ".ckpt", ".pt"}

base_models = ("checkpoints", "loras", "diffusion_models", "onnx")
for _name in base_models:
    os.makedirs(os.path.join(models_dir, _name), exist_ok=True)
os.makedirs(output_directory, exist_ok=True)
open(os.path.join(models_dir, "checkpoints", "placeholder.safetensors"), "wb").close()

folder_names_and_paths = {
    name: ([os.path.join(models_dir, name)], set(supported_pt_extensions))
    for name in ("checkpoints", "loras", "diffusion_models")
}


def get_folder_paths(name):
    if name not in folder_names_and_paths:
        raise KeyError(name)
    return list(folder_names_and_paths[name][0])


def get_filename_list(name):
    if name not in folder_names_and_paths:
        raise KeyError(name)
    base_dirs, extensions = folder_names_and_paths[name]
    found = []
    for base in base_dirs:
        for dirpath, _dirs, files in os.walk(base):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() in extensions:
                    relative = os.path.relpath(os.path.join(dirpath, file_name), base)
                    found.append(relative)
    return sorted(set(found))


def get_full_path(folder, name):
    if folder not in folder_names_and_paths or not name:
        return None
    for base in folder_names_and_paths[folder][0]:
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def add_model_folder_path(name, path, is_default=False):
    folder_names_and_paths[name] = (
        [os.path.abspath(path)],
        set(supported_pt_extensions),
    )
