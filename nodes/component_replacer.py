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
                "component": (["UNET", "VAE", "CLIP"], {"default": "UNET"}),
                "replacement_name": (folder_paths.get_filename_list("model_toolkit_extracted"), {"default": ""}),
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
            
            # 加载替换组件（只从提取产物目录选，避免把模板 txt 当模型加载）
            replacement_path = folder_paths.get_full_path("model_toolkit_extracted", replacement_name)
            if not replacement_path:
                return (f"❌ 未找到替换组件: {replacement_name}（请先运行「📤 组件提取器」生成）",)

            replacement_data = ModelUtils.load_model(replacement_path)

            if not replacement_data:
                return (f"❌ 无法加载替换组件: {replacement_path}",)

            # 按组件归属 + 形状替换（B7/B8：component 决定可写键集合）
            stats = ModelUtils.replace_component_keys(model_data, replacement_data, component)
            if stats["target_keys"] == 0:
                return (f"❌ 目标模型中没有 {component} 组件，无法替换",)

            if stats["replaced"] == 0:
                return (
                    f"❌ 没有可替换的 {component} 键："
                    f"形状不匹配 {stats['shape_mismatch']} 个，非本组件键 {stats['missing']} 个",
                )
                
            # 自动生成输出路径
            if not output_name:
                output_name = f"{os.path.splitext(model_name)[0]}_replaced"
            
            # 确保有正确的扩展名
            if not output_name.endswith(('.safetensors', '.ckpt', '.pt')):
                output_name += '.safetensors'
            
            # 保存修改后的模型
            save_path = os.path.join(folder_paths.output_directory, output_name)
            if ModelUtils.save_component(model_data, save_path):
                detail = f"替换 {stats['replaced']}/{stats['target_keys']} 个 {component} 键"
                if stats["shape_mismatch"]:
                    detail += f"，跳过形状不匹配 {stats['shape_mismatch']} 个"
                if stats["missing"]:
                    detail += f"，忽略非 {component} 键 {stats['missing']} 个"
                return (f"✅ {detail}，已保存到: {save_path}",)
            else:
                return ("❌ 保存模型失败",)
                
        except Exception as e:
            return (f"❌ 替换失败: {str(e)}",) 