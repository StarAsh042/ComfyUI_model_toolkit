"""
Model Toolkit 核心工具：模型加载 / 保存、键分组、架构识别与兼容性校验。
"""
import logging
import os
from collections import defaultdict

import safetensors.torch
import torch

# 与包级 logger 同名，保证日志输出（logger name）不变
logger = logging.getLogger(__package__.split(".")[0])


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
