"""
Model Toolkit - 用于分析和修改Stable Diffusion模型的工具包
"""

import logging
import os
import sys

import folder_paths

from .model_toolkit.core import ModelUtils

# 简化日志配置（ComfyUI 已配置 root logger 时不覆盖宿主设置）
if not logging.getLogger().handlers:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# 版本兼容性检查（必须先于 folder_paths API 调用）
if not hasattr(folder_paths, 'add_model_folder_path'):
    logger.warning("Legacy ComfyUI detected; some features may be limited")

if not hasattr(folder_paths, 'get_filename_list'):
    logger.error("Incompatible ComfyUI version; please upgrade to v1.0+")

# 提取产物目录：组件提取器写这里，替换器也只从这里选（修复 B2）
if hasattr(folder_paths, 'add_model_folder_path'):
    models_root = getattr(folder_paths, "models_dir", None) or folder_paths.output_directory
    extracted_components_path = os.path.join(models_root, "model_toolkit_extracted")
    os.makedirs(extracted_components_path, exist_ok=True)
    folder_paths.add_model_folder_path("model_toolkit_extracted", extracted_components_path)
    # add_model_folder_path 对新目录使用空扩展集（= 放行所有文件），
    # 这里显式限定为模型文件，避免模板/杂项文件混进替换器下拉
    folder_paths.folder_names_and_paths["model_toolkit_extracted"] = (
        folder_paths.folder_names_and_paths["model_toolkit_extracted"][0],
        {".safetensors", ".ckpt", ".pt"},
    )

# 前端扩展目录（ComfyUI 约定）：模型检查器的「模型目录」联动下拉
WEB_DIRECTORY = "./js"


def _register_folder_files_route():
    """注册 GET /model_toolkit/folder_files：返回各模型目录的文件清单。

    模型检查器的「模型目录」下拉需要按目录切换模型列表，而 INPUT_TYPES
    只在启动时求值一次，服务端做不了控件间联动——文件清单经该路由提供给
    前端 js/model_dir_filter.js。

    只在 ComfyUI 已加载 server 模块（即真实运行环境）时注册：main.py 先
    创建 PromptServer 再导入自定义节点，顺序有保证；测试或旧版 ComfyUI 下
    静默跳过，避免拖入整个 ComfyUI 依赖导致导入失败。
    """
    if "server" not in sys.modules:
        return
    from server import PromptServer
    from aiohttp import web

    if getattr(PromptServer, "instance", None) is None:
        return

    from .model_toolkit.folders import list_dir_files, list_model_dirs

    try:

        @PromptServer.instance.routes.get("/model_toolkit/folder_files")
        async def _folder_files(_request):
            folders = {
                dir_name: list_dir_files(dir_name)
                for dir_name in list_model_dirs()
            }
            return web.json_response(folders)

    except Exception as e:
        logger.warning(f"Failed to register the model directory file-list route (directory filtering will be unavailable): {e}")


_register_folder_files_route()

# 导入节点类
from .nodes.checkpoint_inspector import CheckpointInspector
from .nodes.component_extractor import ComponentExtractor
from .nodes.component_replacer import ComponentReplacer

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "ModelInspector": CheckpointInspector,
    "ComponentExtractor": ComponentExtractor,
    "ComponentReplacer": ComponentReplacer,
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelInspector": "🔍 Model Inspector",
    "ComponentExtractor": "📤 Component Extractor",
    "ComponentReplacer": "🔄 Component Replacer",
}

# 版本信息
__version__ = "1.3.0" 