"""
Component Extractor Node - 组件提取器节点
用于从Stable Diffusion模型中提取特定组件
"""

import os
import folder_paths
from .. import ModelUtils

class ComponentExtractor:
    """组件提取器节点类"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """定义节点输入参数"""
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), {"default": ""}),
                "component": (["UNET", "VAE", "CLIP"], {"default": "UNET"}),
                "output_name": ("STRING", {
                    "default": "extracted_component",
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提取结果",)
    FUNCTION = "extract_component"
    CATEGORY = "model_toolkit/修改"
    
    def extract_component(self, model_name, component, output_name):
        """从模型中提取组件
        
        Args:
            model_name (str): 模型文件名
            component (str): 组件类型
            output_name (str): 输出文件名
            
        Returns:
            tuple[str]: 提取结果
        """
        try:
            # 加载模型
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            model_data = ModelUtils.load_model(model_path)
            
            if not model_data:
                return ("❌ 无法加载模型文件",)
                
            # 获取组件键值
            groups = ModelUtils.group_model_keys(model_data)
            if component not in groups or not groups[component]:
                return (f"❌ 未找到{component}组件",)
                
            # 提取组件数据
            extracted = {k: model_data[k] for k in groups[component]}
            
            # 自动生成输出路径
            if not output_name:
                output_name = f"{os.path.splitext(model_name)[0]}_{component.lower()}"
            
            # 确保有正确的扩展名
            if not output_name.endswith(('.safetensors', '.ckpt', '.pt')):
                output_name += '.safetensors'
            
            # 保存组件
            save_path = os.path.join(folder_paths.output_directory, output_name)
            if ModelUtils.save_component(extracted, save_path):
                return (f"✅ 成功保存{component}组件到: {save_path}",)
            else:
                return ("❌ 保存组件失败",)
                
        except Exception as e:
            return (f"❌ 提取失败: {str(e)}",)

    def load_component_definition(self, component):
        # This method should be implemented to load the component definition
        # based on the component type. It's a placeholder and should be replaced
        # with the actual implementation.
        pass

    def load_model(self, model_path):
        # This method should be implemented to load the model from the given path
        # using the utils.model_utils.load_model function. It's a placeholder and should
        # be replaced with the actual implementation.
        pass

    def save_component(self, component_data, save_path):
        # This method should be implemented to save the component data to the given path
        # using the utils.model_utils.save_component function. It's a placeholder and should
        # be replaced with the actual implementation.
        pass 