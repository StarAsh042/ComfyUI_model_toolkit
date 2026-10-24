"""精度统计（按张量 dtype 汇总参数量与占比）。"""
import logging
from collections import defaultdict

logger = logging.getLogger(__package__.split(".")[0])


class PrecisionMixin:
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
            'torch.float32': 'FP32 (Single precision)',
            'torch.float16': 'FP16 (Half precision)',
            'torch.bfloat16': 'BF16 (Brain float)',
            'torch.float8_e4m3fn': 'FP8 (E4M3)',
            'torch.float8_e5m2': 'FP8 (E5M2)',
            'torch.int8': 'INT8 (Quantized)',
            'torch.int4': 'INT4 (Quantized)',
            'torch.uint8': 'UINT8 (Quantized)',
        }

        # 精度友好名称
        precision_friendly = {
            'float32': 'FP32 (Single precision)',
            'float16': 'FP16 (Half precision)',
            'bfloat16': 'BF16 (Brain float)',
            'float8': 'FP8',
            'int8': 'INT8 (Quantized)',
            'int4': 'INT4 (Quantized)',
            'uint8': 'UINT8 (Quantized)',
        }

        # 遍历所有张量统计精度分布
        for tensor in model_data.values():
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
                precision_type = f"Other ({dtype_str})"

            precision_counts[precision_type] += param_count

        # 计算百分比并确定主要精度类型
        precision_distributions = {}
        primary_precision = "Mixed precision"
        max_count = 0

        for prec_type, count in precision_counts.items():
            percentage = (count / total_params) * 100
            precision_distributions[prec_type] = (count, percentage)

            # 更新主要精度类型(超过90%的参数使用同一精度)
            if count > max_count and percentage > 90:
                max_count = count
                primary_precision = prec_type

        # 如果没有明确主要精度，检查是否为混合精度训练典型组合
        if primary_precision == "Mixed precision":
            fp16_percentage = precision_distributions.get('FP16 (Half precision)', (0, 0))[1]
            fp32_percentage = precision_distributions.get('FP32 (Single precision)', (0, 0))[1]

            if fp16_percentage > 50 and fp32_percentage > 0:
                primary_precision = "Mixed precision (mostly FP16)"
            elif 'INT8 (Quantized)' in precision_distributions and precision_distributions['INT8 (Quantized)'][1] > 50:
                primary_precision = "Quantized model (INT8)"

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
            'float32': 'FP32 (Single precision)',
            'float16': 'FP16 (Half precision)',
            'bfloat16': 'BF16 (Brain float)',
            'float8': 'FP8',
            'int8': 'INT8 (Quantized)',
            'int4': 'INT4 (Quantized)',
            'uint8': 'UINT8 (Quantized)'
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
                        precision_type = f"Other ({dtype_str})"

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
