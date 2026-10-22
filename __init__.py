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
    logger.warning("检测到旧版本ComfyUI，部分功能可能受限")

if not hasattr(folder_paths, 'get_filename_list'):
    logger.error("不兼容的ComfyUI版本，请升级到v1.0+")

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

# 导入节点类
from .nodes.checkpoint_inspector import CheckpointInspector
from .nodes.component_extractor import ComponentExtractor
from .nodes.component_replacer import ComponentReplacer
from .nodes.unet_inspector import UNetInspector

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "ModelInspector": CheckpointInspector,
    "ComponentExtractor": ComponentExtractor,
    "ComponentReplacer": ComponentReplacer,
    "UNetInspector": UNetInspector
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelInspector": "🔍 模型检查器",
    "ComponentExtractor": "📤 组件提取器",
    "ComponentReplacer": "🔄 组件替换器", 
    "UNetInspector": "🔬 UNET检查器"
}

# 版本信息
__version__ = "1.2.0" 