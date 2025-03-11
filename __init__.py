"""
Model Toolkit - 用于分析和修改Stable Diffusion模型的工具包
"""

import os
import torch
import safetensors.torch
from collections import defaultdict
import folder_paths
import logging
import sys

# 简化日志配置
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 设置组件路径
components_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "components")
folder_paths.add_model_folder_path("components", components_path)

# 获取当前目录路径
dir_path = os.path.dirname(__file__)

class ModelUtils:
    """模型工具类 - 提供模型处理的核心功能"""
    
    COMPONENT_TYPES = {
        'UNET': 'diffusion_model',
        'VAE': ['first_stage_model', 'encoder', 'decoder', 'quant_conv'],
        'CLIP': ['cond_stage_model', 'text_model', 'transformer'],
        'EMA': 'model_ema'
    }
    
    ARCH_FEATURES = {
        'SDXL': 'text_model_2',
        'SD2.x': 'v_pred'
    }
    
    @staticmethod
    def format_size(size_in_bytes):
        """格式化文件大小"""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(size_in_bytes)
        unit_index = 0
        
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
            
        return f"{size:.2f} {units[unit_index]}"

    @staticmethod
    def format_params(num):
        """格式化参数数量"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        else:
            return f"{num/1e6:.2f}M"

    @staticmethod
    def load_model(model_path):
        """加载模型文件"""
        try:
            if model_path.endswith('.safetensors'):
                model_data = safetensors.torch.load_file(model_path, device="cpu")
            else:
                model_data = torch.load(model_path, map_location="cpu")
                if "state_dict" in model_data:
                    model_data = model_data["state_dict"]
            return model_data
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return None

    @classmethod
    def group_model_keys(cls, model_data):
        """将模型键值按组件分组"""
        groups = defaultdict(list)
        
        for key in model_data.keys():
            grouped = False
            for comp_name, patterns in cls.COMPONENT_TYPES.items():
                if isinstance(patterns, list):
                    if any(p in key for p in patterns):
                        groups[comp_name].append(key)
                        grouped = True
                        break
                elif patterns in key:
                    groups[comp_name].append(key)
                    grouped = True
                    break
            
            if not grouped:
                groups['其他'].append(key)
        
        return groups

    @classmethod
    def analyze_architecture(cls, model_data):
        """分析模型架构"""
        if not model_data:
            return "未知"
        
        for arch_type, feature in cls.ARCH_FEATURES.items():
            if any(feature in k for k in model_data.keys()):
                return arch_type
                
        return "SD1.x"

    @staticmethod
    def save_component(component_data, output_path):
        """保存模型组件"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_path.endswith('.safetensors'):
                safetensors.torch.save_file(component_data, output_path)
            else:
                torch.save({"state_dict": component_data}, output_path)
            return True
        except Exception as e:
            logger.error(f"保存组件时出错: {str(e)}")
            return False

    @staticmethod
    def validate_compatibility(source_comp, target_comp):
        """验证组件兼容性"""
        try:
            for key in source_comp:
                if key not in target_comp:
                    return False
                if source_comp[key].shape != target_comp[key].shape:
                    return False
            return True
        except Exception as e:
            logger.error(f"验证兼容性时出错: {str(e)}")
            return False

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

# 版本兼容性检查
if not hasattr(folder_paths, 'add_model_folder_path'):
    logger.warning("检测到旧版本ComfyUI，部分功能可能受限")

if not hasattr(folder_paths, 'get_filename_list'):
    logger.error("不兼容的ComfyUI版本，请升级到v1.0+")

# 版本信息
__version__ = "1.0.0" 