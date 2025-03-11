"""
UNET Inspector Node - UNET检查器节点
用于分析UNET模型的结构和参数
"""

import os
import folder_paths
from .. import ModelUtils

class UNetInspector:
    """UNET检查器节点类"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """定义节点输入参数"""
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), {"default": ""}),
                "show_details": (["简略", "详细"], {"default": "简略"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("分析结果",)
    FUNCTION = "inspect_unet"
    CATEGORY = "model_toolkit/分析"
    
    def inspect_unet(self, model_name, show_details):
        """分析UNET结构
        
        Args:
            model_name (str): 模型文件名
            show_details (str): 显示详细程度
            
        Returns:
            tuple[str]: 分析结果
        """
        try:
            # 加载模型
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            model_data = ModelUtils.load_model(model_path)
            
            if not model_data:
                return ("❌ 无法加载模型文件",)
            
            # 获取UNET组件
            groups = ModelUtils.group_model_keys(model_data)
            if 'UNET' not in groups or not groups['UNET']:
                return ("❌ 未找到UNET组件",)
            
            unet_data = {k: model_data[k] for k in groups['UNET']}
            
            # 基础信息
            result = []
            result.append(f"📄 模型文件: {model_name}")
            result.append(f"🔢 UNET参数量: {ModelUtils.format_params(sum(t.numel() for t in unet_data.values()))}")
            
            # 分析块结构
            blocks = {
                'input': [k for k in unet_data.keys() if 'input_blocks' in k],
                'middle': [k for k in unet_data.keys() if 'middle_block' in k],
                'output': [k for k in unet_data.keys() if 'output_blocks' in k]
            }
            
            # 计算参数分布
            total_params = sum(unet_data[k].numel() for k in unet_data.keys())
            
            result.append("\n🏗️ 块结构:")
            for block_name, keys in blocks.items():
                if keys:
                    params = sum(unet_data[k].numel() for k in keys)
                    block_count = len(set(k.split('.')[1] for k in keys))
                    percentage = params / total_params * 100
                    result.append(f"  • {block_name}块: {block_count}个")
                    result.append(f"    参数量: {ModelUtils.format_params(params)} ({percentage:.1f}%)")
            
            if show_details == "详细":
                # 分析注意力层
                attn_keys = [k for k in unet_data.keys() if 'attn' in k]
                if attn_keys:
                    result.append("\n🔍 注意力层:")
                    attn_params = sum(unet_data[k].numel() for k in attn_keys)
                    attn_percentage = attn_params / total_params * 100
                    result.append(f"  • 数量: {len(attn_keys)}")
                    result.append(f"  • 参数量: {ModelUtils.format_params(attn_params)} ({attn_percentage:.1f}%)")
                
                # 分析时间嵌入
                time_keys = [k for k in unet_data.keys() if 'time_embed' in k]
                if time_keys:
                    result.append("\n⏱️ 时间嵌入:")
                    for k in time_keys:
                        result.append(f"  • {k}: shape={tuple(unet_data[k].shape)}")
                
                # 检查是否为Flux架构
                flux_keys = [k for k in unet_data.keys() if 'flux_' in k]
                if flux_keys:
                    result.append("\n⚡ Flux特有组件:")
                    for k in flux_keys:
                        result.append(f"  • {k}: shape={tuple(unet_data[k].shape)}")
            
            return ("\n".join(result),)
            
        except Exception as e:
            return (f"❌ 分析UNET时出错: {str(e)}",) 