"""检查器的分析逻辑（纯函数式方法，无节点状态依赖）。"""
import logging
from collections import defaultdict

logger = logging.getLogger(__package__.split(".")[0])


class AnalysisMixin:
    def analyze_architecture(self, model_data):
        """支持检测以下架构：
        - SD1.x (含EMA变体)
        - SD2.x (通过v_pred检测)
        - SDXL (通过双CLIP检测)
        """
        result = {'type': 'Unknown', 'extra_info': []}
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
                print(f"Failed to parse block index: {e}")
        
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
            decision_log.append("Dual CLIP detected: SDXL features present")
            
            # 检查UNET输入通道
            if input_channels == 9:
                result['extra_info'].append(f"UNET input channels: {input_channels} (6 conditioning + 1 noise + 2 resolution channels)")
            elif input_channels == 4:
                result['extra_info'].append(f"UNET input channels: {input_channels} (3 conditioning + 1 noise channels) [note: standard SDXL uses 9 channels]")
                decision_log.append("Input channels (4) are unusual; possible converted variant")
            else:
                result['extra_info'].append(f"UNET input channels: {input_channels} [non-standard configuration]")
            
            # 修改：不再显示EMA参数，改为显示CLIP和VAE的通道信息
            result['extra_info'].append("CLIP text input dim: 768 (max length: 77)")
            result['extra_info'].append("VAE input channels: 3 (RGB image input)")
            
            if channel_key:
                decision_log.append(f"Detected input layer: {channel_key}")
            
            return result, debug_info
        
        # SD2.x判定
        if sd2_features['v_pred']:
            result['type'] = "SD2.x"
            decision_log.append("v_pred predictor detected: identified as SD2.x")
            result['extra_info'].append(f"UNET input channels: {input_channels} (3 conditioning + 1 noise channels)")
            result['extra_info'].append("v_pred predictor detected")
            return result, debug_info
        
        # SD1.x判定（修改后）
        if sd1_features['block_count'] and sd1_features['input_channels']:
            # 不再输出EMA数量，而是直接显示UNET、VAE和CLIP的通道信息
            result['type'] = "SD1.x-EMA" if len([k for k in model_data.keys() if 'model_ema' in k]) > 0 else "SD1.x"
            result['extra_info'].append(f"UNET input channels: {input_channels} (3 conditioning + 1 noise channels)")
            result['extra_info'].append("VAE input channels: 3 (RGB image input)")
            result['extra_info'].append("CLIP text input dim: 768 (max length: 77)")
            return result, debug_info
        
        # 未能确定类型
        result['type'] = "Unknown"
        result['extra_info'].append("Unable to determine architecture type")
        if input_channels:
            result['extra_info'].append(f"UNET input channels: {input_channels}")
        
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
        result.append("\nUNET structure:")
        
        key_groups = {
            'Input blocks': [k for k in unet_keys if any(p in k for p in ['input_blocks', 'down_blocks'])],
            'Middle block': [k for k in unet_keys if any(p in k for p in ['middle_block', 'mid_block', 'bottleneck'])],
            'Output blocks': [k for k in unet_keys if any(p in k for p in ['output_blocks', 'up_blocks'])]
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
            
            if group_name == 'Input blocks':
                indices = extract_indices(keys, 'input_blocks')
                if not indices:
                    indices = extract_indices(keys, 'down_blocks')
            elif group_name == 'Output blocks':
                indices = extract_indices(keys, 'output_blocks')
                if not indices:
                    indices = extract_indices(keys, 'up_blocks')
            else:
                indices = []
            
            if indices:
                max_idx = max(indices)
                result.append(f"- {group_name}: {max_idx + 1} (indices: 0-{max_idx})")
            else:
                count = 1 if group_name == 'Middle block' else 0
                result.append(f"- {group_name}: {count}")
            
            result.append(f"  Parameters: {self.format_params(params)} ({percentage:.1f}%)")
        
        result.append(f"- UNET total parameters: {self.format_params(total_unet_params)} ({total_unet_params/total_params*100:.1f}%)")
        
        return result

    def analyze_vae_structure(self, model_data, vae_keys, total_params):
        """分析VAE结构"""
        result = []
        result.append("\nVAE structure:")
        
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
            result.append(f"- Input channels: {in_channels}")
        
        result.append(f"- Encoder keys: {len(encoders)}")
        result.append(f"- Encoder parameters: {self.format_params(encoder_params)} ({enc_percent:.1f}%)")
        result.append(f"- Decoder keys: {len(decoders)}")
        result.append(f"- Decoder parameters: {self.format_params(decoder_params)} ({dec_percent:.1f}%)")
        result.append(f"- VAE total parameters: {self.format_params(total_vae_params)} ({vae_percent:.1f}%)")
        
        return result

    def analyze_clip_structure(self, model_data, clip_keys, total_params, arch_type):
        """分析CLIP结构"""
        result = []
        result.append("\nCLIP structure:")
        
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
            result.append(f"- text_model_1 keys: {len(text_model_1_keys)}")
            result.append(f"- text_model_1 parameters: {self.format_params(model_params)} ({model_percent:.1f}%)")
        
        # 添加text_model_2信息（如果存在）
        if text_model_2_keys:
            model_params = sum(model_data[k].numel() for k in text_model_2_keys if k in model_data)
            model_percent = model_params / total_params * 100
            result.append(f"- text_model_2 keys: {len(text_model_2_keys)}")
            result.append(f"- text_model_2 parameters: {self.format_params(model_params)} ({model_percent:.1f}%)")
        # 对于SDXL类型，但没检测到text_model_2键的情况
        elif arch_type == "SDXL" and not text_model_2_keys:
            result.append(f"- text_model_2: not detected (possible SDXL variant)")
        
        # 总CLIP参数量
        total_clip_params = sum(model_data[k].numel() for k in clip_keys if k in model_data)
        total_clip_percent = total_clip_params / total_params * 100
        result.append(f"- CLIP total parameters: {self.format_params(total_clip_params)} ({total_clip_percent:.1f}%)")
        
        return result


