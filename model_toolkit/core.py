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
    
    # 组件键前缀。顺序即优先级：CLIP 必须排在 VAE 之前，
    # 否则文本编码器的 encoder.* 键会被 VAE 抢走（B1）。
    COMPONENT_PREFIXES = {
        'CLIP': ('cond_stage_model.', 'conditioner.', 'text_encoder.', 'text_encoder_2.'),
        'VAE': ('first_stage_model.', 'vae.'),
        'EMA': ('model_ema.',),
    }

    # UNET 前缀候选：按命中键数投票选出（参照 ComfyUI unet_prefix_from_state_dict）
    UNET_PREFIX_CANDIDATES = (
        'model.diffusion_model.',
        'diffusion_model.',
        'model.model.',
        'net.',
    )
    
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
    def read_safetensors_metadata(model_path):
        """读取 safetensors 头部的 __metadata__（不加载权重）。

        非 safetensors 文件返回空 dict，由调用方回退到键扫描。
        """
        if not model_path or not str(model_path).lower().endswith((".safetensors", ".sft")):
            return {}
        try:
            with safetensors.safe_open(str(model_path), framework="pt", device="cpu") as f:
                return dict(f.metadata() or {})
        except Exception as e:
            logger.error(f"读取模型元信息时出错: {str(e)}")
            return {}

    @staticmethod
    def load_model(model_path):
        """加载模型文件"""
        try:
            if model_path.endswith('.safetensors'):
                model_data = safetensors.torch.load_file(model_path, device="cpu")
            else:
                # 显式 weights_only=True：拒绝 pickle 任意对象（torch<2.6 默认 False）
                model_data = torch.load(model_path, map_location="cpu", weights_only=True)
                if "state_dict" in model_data:
                    model_data = model_data["state_dict"]
            return model_data
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return None

    @staticmethod
    def detect_unet_prefix(keys):
        """按键数投票选出 UNET 前缀；命中不超过 5 个键时返回 None。"""
        counts = {candidate: 0 for candidate in ModelUtils.UNET_PREFIX_CANDIDATES}
        for key in keys:
            for candidate in counts:
                if key.startswith(candidate):
                    counts[candidate] += 1
                    break
        best = max(counts, key=counts.get)
        return best if counts[best] > 5 else None

    @classmethod
    def classify_key(cls, key, unet_prefix=None):
        """把单个键归入组件；无法判定时归入 '其他'。"""
        for comp_name, prefixes in cls.COMPONENT_PREFIXES.items():
            if key.startswith(prefixes):
                return comp_name
        if unet_prefix and key.startswith(unet_prefix):
            return 'UNET'
        # 提取产物（键已去掉完整前缀）的兜底规则
        if key.startswith(('encoder.', 'decoder.', 'quant_conv.')):
            return 'VAE'
        if key.startswith(('text_model.', 'transformer.')):
            return 'CLIP'
        if 'diffusion_model.' in key:
            return 'UNET'
        return '其他'

    @classmethod
    def group_model_keys(cls, model_data):
        """将模型键值按组件分组（前缀优先，避免子串误判）"""
        keys = list(model_data.keys())
        unet_prefix = cls.detect_unet_prefix(keys)
        groups = defaultdict(list)

        for key in keys:
            groups[cls.classify_key(key, unet_prefix)].append(key)

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

    @classmethod
    def replace_component_keys(cls, target_data, replacement_data, component):
        """按组件归属与张量形状把替换组件的键写入目标模型（原地修改）。

        修复 B7/B8：component 参数决定可替换的键集合，
        replacement 中不属于该组件、或形状不匹配的键一律跳过。

        Returns:
            dict: replaced（写入键数）/ missing（非本组件键数）/
                  shape_mismatch（形状不匹配键数）/ target_keys（目标组件键数）
        """
        target_keys = set(cls.group_model_keys(target_data).get(component, []))
        replaced = missing = shape_mismatch = 0

        for key, value in replacement_data.items():
            if key not in target_keys:
                missing += 1
                continue
            target_value = target_data.get(key)
            if target_value is None or tuple(target_value.shape) != tuple(value.shape):
                shape_mismatch += 1
                continue
            target_data[key] = value
            replaced += 1

        return {
            "replaced": replaced,
            "missing": missing,
            "shape_mismatch": shape_mismatch,
            "target_keys": len(target_keys),
        }

    @staticmethod
    def validate_compatibility(source_comp, target_comp):
        """验证组件兼容性（全部键必须存在且形状一致）。

        注意：替换节点已改用 replace_component_keys 做逐键校验，
        本方法保留给需要「整份组件是否完全匹配」判断的调用方。
        """
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
