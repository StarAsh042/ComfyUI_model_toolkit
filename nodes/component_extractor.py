"""
Component Extractor Node - 组件提取器节点
用于从Stable Diffusion模型中提取特定组件
"""

import os

import folder_paths

from .. import ModelUtils
from ..model_toolkit import templates
from ..model_toolkit.output_paths import UnsafeOutputPathError, resolve_output_path


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
    RETURN_NAMES = ("Extraction Result",)
    FUNCTION = "extract_component"
    CATEGORY = "model_toolkit/Editing"

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
            # 输出路径先校验：越界输入不得进入模型加载/保存流程
            default_output_name = f"{os.path.splitext(model_name)[0]}_{component.lower()}"
            try:
                save_dir = folder_paths.get_folder_paths("model_toolkit_extracted")[0]
            except (KeyError, IndexError, AttributeError):
                save_dir = folder_paths.output_directory

            try:
                save_path = resolve_output_path(save_dir, output_name, default_output_name)
            except UnsafeOutputPathError as e:
                return (f"❌ Invalid output_name: {e}",)

            # 加载模型
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            model_data = ModelUtils.load_model(model_path)

            if not model_data:
                return ("❌ Failed to load model file",)

            # 获取组件键值
            groups = ModelUtils.group_model_keys(model_data)
            if component not in groups or not groups[component]:
                return (f"❌ {component} component not found",)

            # 提取组件数据
            extracted = {k: model_data[k] for k in groups[component]}

            # 模板契约核对：已知架构时告诉用户组件是否完整
            spec, _ = templates.detect(model_data.keys())
            note = ""
            if spec:
                hit, total = spec.contract(component, extracted.keys())
                note = f", template {spec.identifier} matched {hit}/{total} keys"

            # 保存组件
            if ModelUtils.save_component(extracted, save_path):
                return (f"✅ Saved {component} component ({len(extracted)} keys{note}) to: {save_path}",)
            else:
                return ("❌ Failed to save component",)

        except Exception as e:
            return (f"❌ Extraction failed: {str(e)}",)
