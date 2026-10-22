"""
Model Toolkit - 用于分析和修改Stable Diffusion模型的工具包
"""

import logging
import os
import sys

import folder_paths

from .model_toolkit.core import ModelUtils

# 简化日志配置
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

# 设置组件路径（旧版本缺少 API 时跳过注册，保证导入不崩）
components_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "components")
if hasattr(folder_paths, 'add_model_folder_path'):
    folder_paths.add_model_folder_path("components", components_path)

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
__version__ = "1.0.0" 