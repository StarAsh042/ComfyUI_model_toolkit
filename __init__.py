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

# 服务层集中注册：模型目录文件清单 + 报告数据接口。
# 只在真实 ComfyUI 环境生效；测试或旧版环境下 register_routes 内部跳过。
from .model_toolkit.server import register_routes

register_routes()

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
__version__ = "2.0.0" 