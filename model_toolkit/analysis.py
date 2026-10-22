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
