"""
Checkpoint Inspector Node - 模型检查器节点
用于分析Stable Diffusion模型的结构和组件
"""

import os
import torch
import safetensors.torch
import folder_paths
from .. import ModelUtils
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 解决线程警告
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import io
import re
from datetime import datetime
import logging
import numpy as np
import traceback
import math
import matplotlib.gridspec as gridspec

# 添加中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先使用系统字体
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class CheckpointInspector:
    """提供完整的模型分析功能，包括：
    - 模型架构检测 (SD1.x/SD2.x/SDXL)
    - 参数统计和精度分析
    - 可视化报告生成
    - 组件结构分析
    """
    CATEGORY = "model_toolkit/分析"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": (folder_paths.get_filename_list("checkpoints"),),
                "显示细节": (["基础", "高级"], {"default": "基础"}),
                "显示图表": ("BOOLEAN", {"default": True})
            }
        }
    
    def __init__(self):
        self.components = self.load_components()
        self.last_analyzed_model = None  # 添加模型缓存
        self.cached_result = None
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("分析报告", "可视化图表")
    FUNCTION = "analyze_model"
    
    def load_components(self):
        """加载组件定义文件"""
        components = {}
        components_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "components")
        
        # 扩展支持的组件类型
        component_types = {
            'UNET': [
                'UNET-v1-SD.txt', 'UNET-v2-SD.txt', 'UNET-XL-SD.txt',
                'UNET-v1-EMA.txt', 'UNET-Flux-SD.txt'
            ],
            'VAE': ['VAE-v1-SD.txt'],
            'CLIP': [
                'CLIP-v1-SD.txt', 'CLIP-v2-SD.txt', 'CLIP-XL-SD.txt',
                'CLIP-v2-WD.txt'
            ],
            'EMA': ['UNET-v1-EMA.txt', 'UNET-v1-Pix2Pix-EMA.txt'] # 添加EMA组件
        }
        
        for comp_type, files in component_types.items():
            components[comp_type] = {}
            for filename in files:
                try:
                    filepath = os.path.join(components_path, filename)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            keys = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                            model_type = filename.split('-')[1]  # 提取v1/v2/XL等标识
                            components[comp_type][model_type] = set(keys)
                except Exception as e:
                    print(f"加载组件文件{filename}时出错: {str(e)}")
                    
        return components
    
    def analyze_model(self, 模型, 显示细节, 显示图表):
        """分析模型结构(主方法)"""
        try:
            current_state = (模型, 显示细节, 显示图表)
            if self.last_analyzed_model == current_state and self.cached_result:
                return self.cached_result
            self.last_analyzed_model = current_state
            
            def get_tensor_precision(tensor):
                """确定张量的精度类型"""
                dtype = tensor.dtype
                if dtype == torch.float16:
                    return "FP16 (半精度)"
                elif dtype == torch.bfloat16:
                    return "BF16 (Brain浮点)"
                elif dtype == torch.float32:
                    return "FP32 (单精度)"
                elif dtype == torch.float64:
                    return "FP64 (双精度)"
                elif dtype == torch.int8:
                    return "INT8 (量化)"
                elif dtype == torch.uint8:
                    return "UINT8"
                else:
                    return f"{dtype}"
            
            model_path = folder_paths.get_full_path("checkpoints", 模型)
            model_data = ModelUtils.load_model(model_path)
            
            if not model_data:
                return ("❌ 无法加载模型文件", torch.zeros((1, 1, 3)))

            result = []
            result.append(f"📄 模型文件: {模型}")
            result.append(f"📊 模型大小: {ModelUtils.format_size(os.path.getsize(model_path))}")
            total_params = sum(t.numel() for t in model_data.values())
            
            precision_stats = self.analyze_model_precision(model_data)
            
            result.append(f"🔢 参数总量: {self.format_params(total_params)} [{precision_stats['primary_precision']}]")
            
            arch_result = self.analyze_architecture(model_data)
            if isinstance(arch_result, tuple):
                arch_info = arch_result[0]
            else:
                arch_info = arch_result
            
            result.append(f"\n🏗️ 模型架构: {arch_info['type']}")
            if arch_info.get('extra_info'):
                for info in arch_info['extra_info']:
                    result.append(f"  • {info}")
            
            groups = ModelUtils.group_model_keys(model_data)
            
            comp_params = defaultdict(lambda: (0, 0.0))
            for comp_name, keys in groups.items():
                params = sum(model_data[k].numel() for k in keys if k in model_data)
                comp_params[comp_name] = (params, params / total_params * 100)
            
            for k in ['UNET', 'CLIP', 'VAE']:
                if k not in comp_params:
                    comp_params[k] = (0, 0.0)
            
            # 计算其他参数
            other_params = comp_params.get('其他', (0, 0.0))[0]
            if other_params > 0:
                other_percent = other_params / total_params * 100
                comp_params['其他'] = (other_params, other_percent)
            
            # 计算EMA参数
            ema_keys = []
            
            # 方法1: 所有可能的EMA命名模式
            ema_patterns = ['model_ema', 'ema.', '.ema', '_ema', 'ema_', 'EMA']
            for pattern in ema_patterns:
                ema_keys.extend([k for k in model_data.keys() if pattern in k])
            
            # 方法2: 检查components目录中的所有EMA文件
            components_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "components")
            try:
                for filename in os.listdir(components_dir):
                    if 'EMA' in filename and filename.endswith('.txt'):
                        ema_file_path = os.path.join(components_dir, filename)
                        with open(ema_file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    # 匹配所有包含此模式的键
                                    matching_keys = [k for k in model_data.keys() if line in k]
                                    ema_keys.extend(matching_keys)
            except Exception as e:
                logger.error(f"加载EMA组件文件时出错: {str(e)}")
            
            # 移除重复项
            ema_keys = list(set(ema_keys))
            ema_params = sum(model_data[k].numel() for k in ema_keys if k in model_data)
            
            # 调整UNET参数，排除EMA部分
            if ema_params > 0:
                # 计算重叠部分：同时属于UNET和EMA的参数
                unet_keys = groups.get('UNET', [])
                overlap_keys = [k for k in ema_keys if k in unet_keys]
                overlap_params = sum(model_data[k].numel() for k in overlap_keys if k in model_data)
                
                # 从UNET中减去重叠部分
                if 'UNET' in comp_params:
                    unet_params, unet_percent = comp_params['UNET']
                    adj_unet_params = unet_params - overlap_params
                    adj_unet_percent = adj_unet_params / total_params * 100
                    comp_params['UNET'] = (adj_unet_params, adj_unet_percent)
                
                # 添加EMA作为独立组件
                comp_params['EMA'] = (ema_params, ema_params / total_params * 100)
                
                # 更新文本报告
                result.append(f"\n⚡ EMA参数:")
                result.append(f"- EMA参数量: {self.format_params(ema_params)} ({ema_params/total_params*100:.1f}%)")
                result.append(f"- EMA键数量: {len(ema_keys)}")
            
            # 获取未归类键名
            classified_keys = set()
            for group_name, keys in groups.items():
                classified_keys.update(keys)
            
            unknown_keys = [k for k in model_data.keys() if k not in classified_keys and k not in ema_keys]
            unknown_params = sum(model_data[k].numel() for k in unknown_keys if k in model_data)
            
            # 如果有未归类参数，添加到结果
            if unknown_params > 0:
                comp_params['未知'] = (unknown_params, unknown_params/total_params*100)
                result.append(f"\n🔍 未归类参数:")
                result.append(f"- 未归类参数量: {self.format_params(unknown_params)} ({unknown_params/total_params*100:.1f}%)")
                result.append(f"- 未归类键数量: {len(unknown_keys)}")
            
            # 如果有其他参数，添加到结果
            if other_params > 0:
                result.append(f"\n📦 其他参数:")
                result.append(f"- 其他参数量: {self.format_params(other_params)} ({other_percent:.1f}%)")
                result.append(f"- 其他键数量: {len(groups.get('其他', []))}")
            
            # 元数据提取
            metadata = {}
            for k in model_data.keys():
                if k.startswith('_metadata') or '_metadata' in k:
                    metadata[k] = model_data[k]
            
            if metadata:
                result.append(f"\n📌 模型元信息:")
                for key, value in metadata.items():
                    if hasattr(value, 'tolist'):
                        try:
                            value = value.tolist()
                        except:
                            value = str(value)
                    result.append(f"- {key}: {value}")
            else:
                result.append(f"\n📌 模型未包含元信息")

            # 详细分析（基础/高级模式）
            if 显示细节 == "基础":
                # UNET分析
                if 'UNET' in groups and groups['UNET']:
                    unet_results = self.analyze_unet_structure(model_data, groups['UNET'], total_params)
                    result.extend(unet_results)
                
                # VAE分析
                if 'VAE' in groups and groups['VAE']:
                    vae_results = self.analyze_vae_structure(model_data, groups['VAE'], total_params)
                    result.extend(vae_results)
                
                # CLIP分析
                if 'CLIP' in groups and groups['CLIP']:
                    clip_results = self.analyze_clip_structure(model_data, groups['CLIP'], total_params, arch_info['type'])
                    result.extend(clip_results)

            elif 显示细节 == "高级":
                result.append("\n🔍 模型所有键名:")
                all_keys = set(model_data.keys())
                
                for comp_name, keys in groups.items():
                    if keys:
                        result.append(f"\n=== {comp_name} 键名 ===")
                        for key in sorted(keys):
                            if key in model_data:
                                result.append(f"  {key}")
                                all_keys.discard(key)

                if all_keys:
                    result.append("\n=== 未分类键名 ===")
                    for key in sorted(all_keys):
                        result.append(f"  {key}")

            # 创建分析数据字典
            analysis_data = {
                'model_name': 模型,
                'model_size': ModelUtils.format_size(os.path.getsize(model_path)),
                'total_params': total_params,
                'total_params_formatted': self.format_params(total_params),
                'primary_precision': precision_stats['primary_precision'],
                'precision_distributions': precision_stats.get('distributions', {}),
                'components': {}, 
                'architecture': arch_info
            }
            
            # 填充组件数据
            for comp_name, (params, percentage) in comp_params.items():
                analysis_data['components'][comp_name] = params
            
            # UNET细分图数据计算
            unet_parts = {}
            if 'UNET' in groups and groups['UNET']:
                # 计算各块的参数量占比 - 排除EMA参数
                input_keys = [k for k in groups['UNET'] if any(p in k for p in ['input_blocks', 'down_blocks']) and k not in ema_keys]
                middle_keys = [k for k in groups['UNET'] if any(p in k for p in ['middle_block', 'mid_block', 'bottleneck']) and k not in ema_keys]
                output_keys = [k for k in groups['UNET'] if any(p in k for p in ['output_blocks', 'up_blocks']) and k not in ema_keys]
                
                input_params = sum(model_data[k].numel() for k in input_keys if k in model_data)
                middle_params = sum(model_data[k].numel() for k in middle_keys if k in model_data)
                output_params = sum(model_data[k].numel() for k in output_keys if k in model_data)
                
                total_unet_params = input_params + middle_params + output_params
                if total_unet_params > 0:
                    unet_parts = {
                        '输入块': input_params / total_unet_params,
                        '中间块': middle_params / total_unet_params,
                        '输出块': output_params / total_unet_params
                    }
                    analysis_data['unet_parts'] = unet_parts
            
            # 生成图表
            image_tensor = torch.zeros((1, 1, 3))
            if 显示图表:
                try:
                    chart_data = self.generate_visual_report(analysis_data, 模型, arch_info['type'])
                    if chart_data:
                        image_tensor = self.convert_image(chart_data)
                except Exception as e:
                    logger.error(f"生成图表失败: {str(e)}")
            
            self.cached_result = ("\n".join(result), image_tensor)
            return self.cached_result
            
        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            return (f"❌ 分析失败: {str(e)}", torch.zeros((1, 1, 3)))

    def analyze_architecture(self, model_data):
        """支持检测以下架构：
        - SD1.x (含EMA变体)
        - SD2.x (通过v_pred检测)
        - SDXL (通过双CLIP检测)
        """
        result = {'type': '未知', 'extra_info': []}
        total_params = sum(t.numel() for t in model_data.values())

        # 调试标记 - 收集关键信息
        debug_info = {}
        
        # 1. 收集关键结构信息
        # ====================
        
        # 检测输入/输出块数量
        block_indices = set()
        for k in [key for key in model_data.keys() if 'input_blocks' in key or 'down_blocks' in key]:
            try:
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part in ['input_blocks', 'down_blocks'] and i+1 < len(parts):
                        if parts[i+1].isdigit():
                            block_idx = int(parts[i+1])
                            block_indices.add(block_idx)
            except Exception as e:
                print(f"块索引解析错误: {e}")
        
        block_count = len(block_indices)
        max_block_idx = max(block_indices) if block_indices else -1
        debug_info['block_count'] = block_count
        debug_info['max_block_idx'] = max_block_idx
        
        # 2. UNET输入通道检测策略 
        # =======================
        
        # 尝试多种可能的输入层位置
        unet_input_patterns = [
            # SDXL格式
            'model.diffusion_model.input_blocks.0.0.weight',
            'diffusion_model.input_blocks.0.0.weight',
            'model.diffusion_model.input_blocks.0.0.conv.weight',
            'diffusion_model.input_blocks.0.0.conv.weight',
            # SD格式
            'model.model.diffusion_model.input_blocks.0.0.weight',
            'model.model.diffusion_model.input_blocks.0.0.conv.weight',
            # 稳定性改进格式
            'model.input_blocks.0.0.weight',
            'model.down_blocks.0.resnets.0.conv1.weight',
            # 最后尝试更通用的模式
            'input_blocks.0',
            'down_blocks.0'
        ]
        
        # 搜索可能的输入层
        found_input_keys = []
        for pattern in unet_input_patterns:
            matching_keys = [k for k in model_data.keys() if pattern in k]
            if matching_keys:
                found_input_keys.extend(matching_keys)
        
        # 提取实际通道数
        input_channels = None
        channel_key = None
        
        for key in found_input_keys:
            if key in model_data:
                tensor = model_data[key]
                # 检查是否为卷积权重 
                if len(tensor.shape) == 4:
                    # 卷积权重形状: [out_channels, in_channels, kernel_h, kernel_w]
                    input_channels = tensor.shape[1]  # 第二维是输入通道
                    channel_key = key
                    break
        
        debug_info['input_channel_detection'] = {
            'found_keys': len(found_input_keys),
            'channel_key': channel_key,
            'input_channels': input_channels
        }
        
        # 3. CLIP模型检测
        # ==============
        
        # 检测text_model_1
        has_clip_l = any(
            k for k in model_data.keys() 
            if 'text_model.encoder' in k or 'cond_stage_model.transformer' in k
        )
        
        # 检测text_model_2 (SDXL特有)
        has_clip_g = any(
            k for k in model_data.keys() 
            if 'text_model_2' in k or 'conditioner.embedders.1' in k
        )
        
        debug_info['clip_detection'] = {
            'has_clip_l': has_clip_l,
            'has_clip_g': has_clip_g
        }
        
        # 4. 架构特征分析
        # =============
        
        # SD1.x特征
        sd1_features = {
            'block_count': block_count <= 12 and block_count >= 9,
            'input_channels': input_channels == 4,
            'param_count': total_params < 2e9,
            'single_clip': has_clip_l and not has_clip_g
        }
        
        # SD2.x特征
        sd2_features = {
            'block_count': block_count <= 12 and block_count >= 9,
            'input_channels': input_channels == 4,
            'param_count': total_params < 2e9, 
            'v_pred': any('v_pred' in k for k in model_data.keys()),
            'single_clip': has_clip_l and not has_clip_g
        }
        
        # SDXL特征
        sdxl_features = {
            'block_count': block_count >= 9,
            'dual_clip': has_clip_l and has_clip_g,
            'param_count': total_params > 2.5e9
        }
        
        debug_info['architecture_scores'] = {
            'sd1': sum(1 for v in sd1_features.values() if v),
            'sd2': sum(1 for v in sd2_features.values() if v),
            'sdxl': sum(1 for v in sdxl_features.values() if v)
        }
        
        # 5. 架构判定
        # ==========
        
        # 记录决策日志
        decision_log = []
        
        # SDXL判定（修改后）
        if sdxl_features['dual_clip'] and (sdxl_features['block_count'] or sdxl_features['param_count']):
            result['type'] = "SDXL"
            decision_log.append("发现双CLIP结构，SDXL特征明显")
            
            # 检查UNET输入通道
            if input_channels == 9:
                result['extra_info'].append(f"UNET输入通道数: {input_channels} (6个条件通道 + 1个噪声通道 + 2个分辨率通道)")
            elif input_channels == 4:
                result['extra_info'].append(f"UNET输入通道数: {input_channels} (3个条件通道 + 1个噪声通道) [注意: 与标准SDXL的9通道不符]")
                decision_log.append("输入通道数(4)异常，可能是特殊转换版本")
            else:
                result['extra_info'].append(f"UNET输入通道数: {input_channels} [非标准配置]")
            
            # 修改：不再显示EMA参数，改为显示CLIP和VAE的通道信息
            result['extra_info'].append("CLIP文本输入维度: 768 (最大长度: 77)")
            result['extra_info'].append("VAE输入通道数: 3 (RGB图像输入)")
            
            if channel_key:
                decision_log.append(f"检测到的输入层: {channel_key}")
            
            return result, debug_info
        
        # SD2.x判定
        if sd2_features['v_pred']:
            result['type'] = "SD2.x"
            decision_log.append("发现v_pred预测器，识别为SD2.x")
            result['extra_info'].append(f"UNET输入通道数: {input_channels} (3个条件通道 + 1个噪声通道)")
            result['extra_info'].append("检测到v_pred预测器")
            return result, debug_info
        
        # SD1.x判定（修改后）
        if sd1_features['block_count'] and sd1_features['input_channels']:
            # 不再输出EMA数量，而是直接显示UNET、VAE和CLIP的通道信息
            result['type'] = "SD1.x-EMA" if len([k for k in model_data.keys() if 'model_ema' in k]) > 0 else "SD1.x"
            result['extra_info'].append(f"UNET输入通道数: {input_channels} (3个条件通道 + 1个噪声通道)")
            result['extra_info'].append("VAE输入通道数: 3 (RGB图像输入)")
            result['extra_info'].append("CLIP文本输入维度: 768 (最大长度: 77)")
            return result, debug_info
        
        # 未能确定类型
        result['type'] = "未知"
        result['extra_info'].append(f"无法确定架构类型")
        if input_channels:
            result['extra_info'].append(f"UNET输入通道数: {input_channels}")
        
        return result, debug_info

    def format_params(self, num):
        """格式化参数数量，1B以上用B单位，否则用M单位"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        else:
            return f"{num/1e6:.2f}M"

    def analyze_unet_structure(self, model_data, unet_keys, total_params):
        """分析UNET结构"""
        result = []
        result.append("\nUNET结构:")
        
        key_groups = {
            '输入块': [k for k in unet_keys if any(p in k for p in ['input_blocks', 'down_blocks'])],
            '中间块': [k for k in unet_keys if any(p in k for p in ['middle_block', 'mid_block', 'bottleneck'])],
            '输出块': [k for k in unet_keys if any(p in k for p in ['output_blocks', 'up_blocks'])]
        }
        
        def extract_indices(keys, pattern):
            """提取特定模式的键索引"""
            indices = set()
            for k in keys:
                if pattern in k:
                    parts = k.split(pattern)
                    if len(parts) > 1:
                        next_part = parts[1].lstrip('.')
                        if '.' in next_part:
                            index_str = next_part.split('.')[0]
                            if index_str.isdigit():
                                indices.add(int(index_str))
            return sorted(list(indices))
        
        total_unet_params = 0
        for group_name, keys in key_groups.items():
            params = sum(model_data[k].numel() for k in keys if k in model_data)
            percentage = params / total_params * 100
            total_unet_params += params
            
            if group_name == '输入块':
                indices = extract_indices(keys, 'input_blocks')
                if not indices:
                    indices = extract_indices(keys, 'down_blocks')
            elif group_name == '输出块':
                indices = extract_indices(keys, 'output_blocks')
                if not indices:
                    indices = extract_indices(keys, 'up_blocks')
            else:
                indices = []
            
            if indices:
                max_idx = max(indices)
                result.append(f"- {group_name}: {max_idx + 1} (索引: 0-{max_idx})")
            else:
                count = 1 if group_name == '中间块' else 0
                result.append(f"- {group_name}: {count}")
            
            result.append(f"  参数量: {self.format_params(params)} ({percentage:.1f}%)")
        
        result.append(f"- UNET总参数量: {self.format_params(total_unet_params)} ({total_unet_params/total_params*100:.1f}%)")
        
        return result

    def analyze_vae_structure(self, model_data, vae_keys, total_params):
        """分析VAE结构"""
        result = []
        result.append("\nVAE结构:")
        
        encoders = [k for k in vae_keys if 'encoder' in k]
        decoders = [k for k in vae_keys if 'decoder' in k]
        
        encoder_params = sum(model_data[k].numel() for k in encoders if k in model_data)
        decoder_params = sum(model_data[k].numel() for k in decoders if k in model_data)
        total_vae_params = encoder_params + decoder_params
        
        enc_percent = encoder_params / total_params * 100
        dec_percent = decoder_params / total_params * 100
        vae_percent = total_vae_params / total_params * 100
        
        vae_input = next((k for k in vae_keys if 'encoder.conv_in.weight' in k), None)
        if vae_input and vae_input in model_data:
            shape = model_data[vae_input].shape
            in_channels = shape[1] if len(shape) >= 2 else 0
            result.append(f"- 输入通道数: {in_channels}")
        
        result.append(f"- 编码器键数量: {len(encoders)}")
        result.append(f"- 编码器参数量: {self.format_params(encoder_params)} ({enc_percent:.1f}%)")
        result.append(f"- 解码器键数量: {len(decoders)}")
        result.append(f"- 解码器参数量: {self.format_params(decoder_params)} ({dec_percent:.1f}%)")
        result.append(f"- VAE总参数量: {self.format_params(total_vae_params)} ({vae_percent:.1f}%)")
        
        return result

    def analyze_clip_structure(self, model_data, clip_keys, total_params, arch_type):
        """分析CLIP结构"""
        result = []
        result.append("\nCLIP结构:")
        
        # 更加精确地识别text_model_1和text_model_2
        text_model_patterns = {
            'text_model_1': [
                'text_model.encoder', 
                'text_model.embeddings',
                'conditioner.embedders.0',
                'cond_stage_model'
            ],
            'text_model_2': [
                'text_model_2',
                'conditioner.embedders.1'
            ]
        }
        
        text_model_1_keys = []
        text_model_2_keys = []
        
        # 收集text_model_1和text_model_2的键
        for k in clip_keys:
            if any(pattern in k for pattern in text_model_patterns['text_model_1']):
                text_model_1_keys.append(k)
            elif any(pattern in k for pattern in text_model_patterns['text_model_2']):
                text_model_2_keys.append(k)
        
        # 确定是否有text_model_2或是SDXL模型类型
        has_text_model_2 = len(text_model_2_keys) > 0 or arch_type == "SDXL"
        
        # 添加text_model_1信息
        if text_model_1_keys:
            model_params = sum(model_data[k].numel() for k in text_model_1_keys if k in model_data)
            model_percent = model_params / total_params * 100
            result.append(f"- text_model_1键数量: {len(text_model_1_keys)}")
            result.append(f"- text_model_1参数量: {self.format_params(model_params)} ({model_percent:.1f}%)")
        
        # 添加text_model_2信息（如果存在）
        if text_model_2_keys:
            model_params = sum(model_data[k].numel() for k in text_model_2_keys if k in model_data)
            model_percent = model_params / total_params * 100
            result.append(f"- text_model_2键数量: {len(text_model_2_keys)}")
            result.append(f"- text_model_2参数量: {self.format_params(model_params)} ({model_percent:.1f}%)")
        # 对于SDXL类型，但没检测到text_model_2键的情况
        elif arch_type == "SDXL" and not text_model_2_keys:
            result.append(f"- text_model_2: 未检测到 (可能是SDXL变种)")
        
        # 总CLIP参数量
        total_clip_params = sum(model_data[k].numel() for k in clip_keys if k in model_data)
        total_clip_percent = total_clip_params / total_params * 100
        result.append(f"- CLIP总参数量: {self.format_params(total_clip_params)} ({total_clip_percent:.1f}%)")
        
        return result

    def analyze_model_precision(self, model_data):
        """支持检测：
        - FP16/FP32/BF16
        - INT8/INT4量化
        - 混合精度模型
        """
        precision_counts = defaultdict(int)
        total_params = 0
        
        # 精度类型映射表
        precision_map = {
            'torch.float32': 'FP32 (单精度)',
            'torch.float16': 'FP16 (半精度)',
            'torch.bfloat16': 'BF16 (脑浮点)',
            'torch.float8_e4m3fn': 'FP8 (E4M3)',
            'torch.float8_e5m2': 'FP8 (E5M2)',
            'torch.int8': 'INT8 (量化)',
            'torch.int4': 'INT4 (量化)',
            'torch.uint8': 'UINT8 (量化)',
        }
        
        # 精度友好名称
        precision_friendly = {
            'float32': 'FP32 (单精度)',
            'float16': 'FP16 (半精度)',
            'bfloat16': 'BF16 (脑浮点)',
            'float8': 'FP8',
            'int8': 'INT8 (量化)',
            'int4': 'INT4 (量化)',
            'uint8': 'UINT8 (量化)',
        }
        
        # 遍历所有张量统计精度分布
        for key, tensor in model_data.items():
            param_count = tensor.numel()
            total_params += param_count
            
            # 获取数据类型并映射到友好名称
            dtype_str = str(tensor.dtype)
            
            # 提取数据类型的核心部分 (例如从"torch.float32"提取"float32")
            if 'torch.' in dtype_str:
                dtype_core = dtype_str.split('torch.')[1]
            else:
                dtype_core = dtype_str
            
            # 查找友好名称，如果没有则使用原始类型名
            if dtype_core in precision_friendly:
                precision_type = precision_friendly[dtype_core]
            elif dtype_str in precision_map:
                precision_type = precision_map[dtype_str]
            else:
                precision_type = f"其他 ({dtype_str})"
            
            precision_counts[precision_type] += param_count
        
        # 计算百分比并确定主要精度类型
        precision_distributions = {}
        primary_precision = "混合精度"
        max_count = 0
        
        for prec_type, count in precision_counts.items():
            percentage = (count / total_params) * 100
            precision_distributions[prec_type] = (count, percentage)
            
            # 更新主要精度类型(超过90%的参数使用同一精度)
            if count > max_count and percentage > 90:
                max_count = count
                primary_precision = prec_type
        
        # 如果没有明确主要精度，检查是否为混合精度训练典型组合
        if primary_precision == "混合精度":
            fp16_percentage = precision_distributions.get('FP16 (半精度)', (0, 0))[1]
            fp32_percentage = precision_distributions.get('FP32 (单精度)', (0, 0))[1]
            
            if fp16_percentage > 50 and fp32_percentage > 0:
                primary_precision = "混合精度 (主要FP16)"
            elif 'INT8 (量化)' in precision_distributions and precision_distributions['INT8 (量化)'][1] > 50:
                primary_precision = "量化模型 (INT8)"
        
        return {
            'distributions': precision_distributions,
            'primary_precision': primary_precision,
            'total_params': total_params
        }

    def analyze_component_precision(self, model_data, groups):
        """分析每个组件的精度分布"""
        precision_stats = {}
        
        # 精度友好名称
        precision_friendly = {
            'float32': 'FP32 (单精度)',
            'float16': 'FP16 (半精度)',
            'bfloat16': 'BF16 (脑浮点)',
            'float8': 'FP8',
            'int8': 'INT8 (量化)',
            'int4': 'INT4 (量化)',
            'uint8': 'UINT8 (量化)'
        }
        
        # 处理每个组件
        for comp_name, keys in groups.items():
            comp_precision = defaultdict(lambda: (0, 0.0))
            comp_total = 0
            
            # 分析组件内每个张量的精度
            for key in keys:
                if key in model_data:
                    tensor = model_data[key]
                    param_count = tensor.numel()
                    comp_total += param_count
                    
                    # 获取精度类型名称
                    dtype_str = str(tensor.dtype)
                    if 'torch.' in dtype_str:
                        dtype_core = dtype_str.split('torch.')[1]
                    else:
                        dtype_core = dtype_str
                    
                    # 查找友好名称
                    if dtype_core in precision_friendly:
                        precision_type = precision_friendly[dtype_core]
                    else:
                        precision_type = f"其他 ({dtype_str})"
                    
                    # 更新计数
                    curr_count, curr_pct = comp_precision[precision_type]
                    comp_precision[precision_type] = (curr_count + param_count, 0)  # 百分比稍后计算
            
            # 计算百分比
            if comp_total > 0:
                for prec_type in comp_precision:
                    count, _ = comp_precision[prec_type]
                    comp_precision[prec_type] = (count, count / comp_total * 100)
                
                precision_stats[comp_name] = dict(comp_precision)
        
        return precision_stats

    def generate_visual_report(self, analysis_data, model_name, arch_type):
        """生成可视化报告"""
        try:
            # 设置图表风格
            plt.style.use('dark_background')
            # 使用稍大一些的画布
            plt.figure(figsize=(20, 12), dpi=100) 
            
            # 添加标题
            plt.suptitle(
                f"{model_name} 结构分析 - {arch_type}\n"
                f"总参数量: {analysis_data['total_params_formatted']} | "
                f"主要精度: {analysis_data['primary_precision']}",
                color='white', fontsize=20, y=0.98
            )
            
            # 使用3x3网格布局
            grid = plt.GridSpec(3, 3, hspace=0.4, wspace=0.3)
            
            # 左上：组件比例圆环图
            ax_pie = plt.subplot(grid[0:2, 0])
            self.draw_component_pie(ax_pie, analysis_data)
            
            # 右侧：详细参数条形图
            ax_bars = plt.subplot(grid[:, 1:3])
            self.draw_param_bars(ax_bars, analysis_data, arch_type)
            
            # 左下：UNET结构
            ax_unet = plt.subplot(grid[2, 0])
            self.draw_unet_structure(ax_unet, analysis_data)
            
            # 添加水印
            plt.figtext(0.02, 0.01, "由 ComfyUI Model Toolkit 生成", color='gray', alpha=0.5)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 保存为临时内存中的图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='#1E1E1E')
            plt.close()
            
            # 使用PIL调整图像到精确尺寸
            buf.seek(0)
            img = Image.open(buf)
            img_resized = img.resize((2048, 1280), Image.Resampling.LANCZOS)
            
            # 保存调整后的图像
            buf_resized = io.BytesIO()
            img_resized.save(buf_resized, format='PNG')
            buf_resized.seek(0)
            
            return f"data:image/png;base64,{base64.b64encode(buf_resized.getvalue()).decode()}"
        except Exception as e:
            logger.error(f"生成图表出错: {str(e)}")
            return None

    def draw_component_pie(self, ax, data):
        """绘制组件构成比例饼图"""
        components = data.get('components', {})
        if not components:
            ax.text(0.5, 0.5, '无组件数据', ha='center', va='center', color='white')
            return
        
        # 获取组件参数量和比例，过滤掉未知和占比很小的组件
        names = []
        values = []
        explode = []
        
        # 基础组件优先显示
        base_components = ['UNET', 'VAE', 'CLIP', 'EMA']
        for comp in base_components:
            if comp in components and components[comp] > 0:
                names.append(comp)
                values.append(components[comp])
                explode.append(0.05 if comp == 'UNET' else 0)
        
        # 添加其他组件
        for comp, value in components.items():
            if comp not in base_components and value > 0:
                if comp == '未知' and value/sum(components.values()) < 0.01:
                    continue  # 跳过占比极小的未知组件
                names.append(comp)
                values.append(value)
                explode.append(0)
        
        # 颜色映射
        colors = {
            'UNET': '#4C72B0', 
            'VAE': '#55A868', 
            'CLIP': '#C44E52', 
            'EMA': '#DD8452',
            '未知': '#937860',
            '其他': '#8172B0'
        }
        pie_colors = [colors.get(name, '#cccccc') for name in names]
        
        # 使用环形图替代饼图，并移除内部标签
        wedges, texts = ax.pie(
            values, 
            autopct=None,  # 移除内部百分比标签
            explode=explode, 
            startangle=90, 
            colors=pie_colors,
            wedgeprops={'width': 0.5, 'edgecolor': '#1E1E1E', 'linewidth': 0.5}
        )
        
        # 添加图例到左侧
        ax.legend(
            wedges, 
            [f"{name} ({self.format_params(val)}, {val/sum(values)*100:.1f}%)" for name, val in zip(names, values)],
            loc="center right", 
            bbox_to_anchor=(0, 0.5),  # 图例位置改到左侧
            fontsize=9,
            framealpha=0.5,
            labelcolor='white'
        )
        
        ax.set_title("组件参数分布", color='white', fontsize=14)

    def draw_unet_structure(self, ax, data):
        """修改后的UNET结构图使用B单位坐标系"""
        unet_parts = data.get('unet_parts', {})
        if not unet_parts:
            ax.text(0.5, 0.5, '无UNET结构数据', ha='center', va='center', color='white')
            return
        
        # 计算绝对参数量
        total_unet_params = data['components'].get('UNET', 0)
        abs_parts = {}
        for part, ratio in unet_parts.items():
            abs_parts[part] = ratio * total_unet_params
        
        # 排序并创建标签
        labels = []
        values = []
        for part, params in sorted(abs_parts.items(), key=lambda x: -x[1]):
            labels.append(part)
            values.append(params)
        
        # 创建水平条形图
        bars = ax.barh(labels, values, color=['#2E5D87', '#3B8EA5', '#4BB3D3'])
        
        # 添加参数量标签 - 使用动态单位（保持不变）
        ax.bar_label(bars, [self.format_params(v) for v in values], 
                    padding=3, color='white', fontsize=10)
        
        # 设置X轴为参数量（使用0.5B为最小间隔单位）
        max_value = max(values) * 1.1
        max_value_in_b = max_value / 1e9
        
        # 确定最大刻度值（向上取整到0.5B的倍数）
        max_tick_value_in_b = math.ceil(max_value_in_b * 2) / 2
        
        # 生成0.5B间隔的刻度
        tick_values_in_b = [i * 0.5 for i in range(int(max_tick_value_in_b * 2) + 1)]
        x_ticks = [v * 1e9 for v in tick_values_in_b]
        
        ax.set_xticks(x_ticks)
        
        # 使用B单位坐标轴刻度(0.5B, 1B, 1.5B等)
        x_tick_labels = []
        for value_in_b in tick_values_in_b:
            if value_in_b == int(value_in_b):
                # 整数B值
                x_tick_labels.append(f"{int(value_in_b)}B")
            else:
                # 0.5B的倍数
                x_tick_labels.append(f"{value_in_b:.1f}B")
        
        ax.set_xticklabels(x_tick_labels, color='white')
        ax.set_xlim(0, max_value)
        
        ax.set_title("UNET内部结构", color='white', fontsize=14)

    def draw_param_bars(self, ax, data, arch_type):
        """修改后的柱状图使用整数单位"""
        # 基础组件 + 更多可能的组件
        components = ['UNET', 'VAE', 'CLIP']
        
        # 如果存在EMA，添加到组件列表
        if data['components'].get('EMA', 0) > 0:
            components.append('EMA')
        
        # 添加未知分类
        if data['components'].get('未知', 0) > 0:
            components.append('未知')
        
        # 添加其他分类
        if data['components'].get('其他', 0) > 0:
            components.append('其他')
        
        params = [data['components'].get(c, 0) for c in components]
        
        # 过滤掉数值为0的组件
        valid_indices = [i for i, p in enumerate(params) if p > 0]
        valid_components = [components[i] for i in valid_indices]
        valid_params = [params[i] for i in valid_indices]
        
        # 颜色映射
        colors = {
            'UNET': '#4C72B0', 
            'VAE': '#55A868', 
            'CLIP': '#C44E52', 
            'EMA': '#DD8452',
            '未知': '#937860',
            '其他': '#8172B0'
        }
        valid_colors = [colors.get(comp, '#cccccc') for comp in valid_components]
        
        # 创建简洁标签
        bars = ax.bar(valid_components, valid_params, color=valid_colors)
        
        # 添加参数数值标签 - 使用动态单位（保持不变）
        ax.bar_label(bars, [self.format_params(p) for p in valid_params], 
                    color='white', fontsize=10)
        
        # 设置Y轴为参数量（使用0.5B为最小间隔单位）
        max_value = max(valid_params) * 1.1 if valid_params else 1
        max_value_in_b = max_value / 1e9
        
        # 确定最大刻度值（向上取整到0.5B的倍数）
        max_tick_value_in_b = math.ceil(max_value_in_b * 2) / 2
        
        # 生成0.5B间隔的刻度
        tick_values_in_b = [i * 0.5 for i in range(int(max_tick_value_in_b * 2) + 1)]
        y_ticks = [v * 1e9 for v in tick_values_in_b]
        
        ax.set_yticks(y_ticks)
        
        # 使用B单位坐标轴刻度(0.5B, 1B, 1.5B等)
        y_tick_labels = []
        for value_in_b in tick_values_in_b:
            if value_in_b == int(value_in_b):
                # 整数B值
                y_tick_labels.append(f"{int(value_in_b)}B")
            else:
                # 0.5B的倍数
                y_tick_labels.append(f"{value_in_b:.1f}B")
        
        ax.set_yticklabels(y_tick_labels, color='white')
        ax.set_ylim(0, max_value)
        
        # 美化图表
        ax.set_ylabel("参数量", color='white')
        ax.set_title(f"{arch_type} 模型组件参数分布", color='white', fontsize=16)
        ax.tick_params(axis='x', labelcolor='white', labelrotation=0)
        ax.grid(True, axis='y', alpha=0.3)

    def convert_image(self, base64_img):
        """增强图像转换可靠性"""
        try:
            if not base64_img:
                return torch.zeros((1, 1, 3))
            # 解析base64数据
            if ',' in base64_img:
                header, data = base64_img.split(",", 1)
            else:
                data = base64_img
            img_data = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_data))
            # 保持原始比例调整尺寸
            max_size = 2048
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            return img_tensor.unsqueeze(0)
        except Exception as e:
            logger.error(f"图像转换失败: {str(e)}")
            return torch.zeros((1, 512, 512, 3))

    def update_progress(self, progress, desc):
        """更新进度提示"""
        if hasattr(self, "progress_bar"):
            self.progress_bar.update(progress, desc)
        else:
            logger.info(f"[进度 {progress}%] {desc}")

    def generate_text_report(self, analysis_result):
        """生成文本格式的详细分析报告"""
        report = []
        # 基础信息
        report.append(f"📄 模型文件: {analysis_result['model_name']}")
        report.append(f"📊 模型大小: {analysis_result['model_size']}")
        report.append(f"🔢 参数总量: {analysis_result['total_params']}")
        
        # 架构信息
        report.append(f"\n🏗️ 模型架构: {analysis_result['architecture']['type']}")
        for info in analysis_result['architecture'].get('extra_info', []):
            report.append(f"  • {info}")
        
        # 组件分析
        for comp in ['UNET', 'VAE', 'CLIP']:
            if comp in analysis_result['components']:
                report.append(f"\n{comp}结构:")
                report.append(f"- 参数量: {analysis_result['components'][comp]['params']}")
                report.append(f"- 占比: {analysis_result['components'][comp]['percentage']}%")
        
        # 精度分析
        report.append("\n🔧 精度分析:")
        report.append(f"- 主要精度: {analysis_result['precision']['primary']}")
        for dist in analysis_result['precision']['distribution']:
            report.append(f"- {dist}: {analysis_result['precision']['distribution'][dist]}%")
        
        # 添加EMA参数统计到文本报告
        if ema_params > 0:
            report.append(f"\n⚡ EMA参数:")
            report.append(f"- 参数量: {self.format_params(ema_params)} ({ema_params/total_params*100:.1f}%)")
            report.append(f"- 键数量: {len(ema_keys)}")
        
        return "\n".join(report) 