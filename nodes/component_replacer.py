"""
Component Replacer Node - 组件替换器节点
用于替换Stable Diffusion模型中的特定组件
"""

import os
import folder_paths
from .. import ModelUtils

class ComponentReplacer:
    """组件替换器节点类"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """定义节点输入参数"""
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), {"default": ""}),
                "component_name": (["UNET", "VAE", "CLIP"], {"default": "UNET"}),
                "replacement_name": (folder_paths.get_filename_list("components"), {"default": ""}),
                "output_name": ("STRING", {
                    "default": "modified_model",
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("替换结果",)
    FUNCTION = "replace_component"
    CATEGORY = "model_toolkit/修改"
    
    def replace_component(self, model_name, component, replacement_name, output_name):
        """替换模型组件
        
        Args:
            model_name (str): 目标模型文件名
            component (str): 要替换的组件类型
            replacement_name (str): 替换组件文件名
            output_name (str): 输出文件名
            
        Returns:
            tuple[str]: 替换结果
        """
        try:
            # 加载目标模型
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            model_data = ModelUtils.load_model(model_path)
            
            if not model_data:
                return ("❌ 无法加载目标模型",)
            
            # 加载替换组件
            replacement_path = folder_paths.get_full_path("components", replacement_name)
            replacement_data = ModelUtils.load_model(replacement_path)
            
            if not replacement_data:
                return ("❌ 无法加载替换组件",)
            
            # 验证兼容性
            if not ModelUtils.validate_compatibility(replacement_data, model_data):
                return ("❌ 组件不兼容，请检查组件类型和形状",)
            
            # 替换组件
            for k, v in replacement_data.items():
                model_data[k] = v
                
            # 自动生成输出路径
            if not output_name:
                output_name = f"{os.path.splitext(model_name)[0]}_replaced"
            
            # 确保有正确的扩展名
            if not output_name.endswith(('.safetensors', '.ckpt', '.pt')):
                output_name += '.safetensors'
            
            # 保存修改后的模型
            save_path = os.path.join(folder_paths.output_directory, output_name)
            if ModelUtils.save_component(model_data, save_path):
                return (f"✅ 成功保存修改后的模型到: {save_path}",)
            else:
                return ("❌ 保存模型失败",)
                
        except Exception as e:
            return (f"❌ 替换失败: {str(e)}",) 