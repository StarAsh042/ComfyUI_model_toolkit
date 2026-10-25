"""
Component Replacer Node - 组件替换器节点
用于替换Stable Diffusion模型中的特定组件
"""

import os

import folder_paths

from .. import ModelUtils
from ..model_toolkit.output_paths import UnsafeOutputPathError, resolve_output_path


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
    RETURN_NAMES = ("Replacement Result",)
    FUNCTION = "replace_component"
    CATEGORY = "model_toolkit/Editing"

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
            # 输出路径先校验：越界输入不得进入模型加载/保存流程
            default_output_name = f"{os.path.splitext(model_name)[0]}_replaced"
            try:
                save_path = resolve_output_path(
                    folder_paths.output_directory, output_name, default_output_name
                )
            except UnsafeOutputPathError as e:
                return (f"❌ Invalid output_name: {e}",)

            # 加载目标模型
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            model_data = ModelUtils.load_model(model_path)

            if not model_data:
                return ("❌ Failed to load target model",)

            # 加载替换组件（只从提取产物目录选，避免把模板 txt 当模型加载）
            replacement_path = folder_paths.get_full_path("model_toolkit_extracted", replacement_name)
            if not replacement_path:
                return (f"❌ Replacement component not found: {replacement_name} (run the 📤 Component Extractor first)",)

            replacement_data = ModelUtils.load_model(replacement_path)

            if not replacement_data:
                return (f"❌ Failed to load replacement component: {replacement_path}",)

            # 按组件归属 + 形状替换（B7/B8：component 决定可写键集合）
            stats = ModelUtils.replace_component_keys(model_data, replacement_data, component)
            if stats["target_keys"] == 0:
                return (f"❌ Target model has no {component} component to replace",)

            if stats["replaced"] == 0:
                return (
                    f"❌ No {component} keys could be replaced: "
                    f"{stats['shape_mismatch']} shape mismatches, "
                    f"{stats['missing']} keys outside this component",
                )

            # 保存修改后的模型
            if ModelUtils.save_component(model_data, save_path):
                detail = f"Replaced {stats['replaced']}/{stats['target_keys']} {component} keys"
                if stats["shape_mismatch"]:
                    detail += f", skipped {stats['shape_mismatch']} shape mismatches"
                if stats["missing"]:
                    detail += f", ignored {stats['missing']} non-{component} keys"
                return (f"✅ {detail}, saved to: {save_path}",)
            else:
                return ("❌ Failed to save model",)

        except Exception as e:
            return (f"❌ Replacement failed: {str(e)}",)
