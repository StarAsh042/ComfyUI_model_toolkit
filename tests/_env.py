"""测试环境引导：把仓库父目录与 ComfyUI 根目录加入 sys.path。"""
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMFYUI_DIR = pathlib.Path(os.environ.get("COMFYUI_DIR", r"D:\AIGC\ComfyUI"))

for _p in (str(REPO_ROOT.parent), str(COMFYUI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
