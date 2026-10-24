"""架构结构提取入口。

v1 只实现 UNet（LDM / diffusers 两种命名）；DiT / LLM 等架构回退到
generic，页面显示通用组件视图。新增特化视图时在这里接入新的提取器。
"""
from . import generic, unet


def extract(model_data, groups, arch_name=None):
    unet_keys = list(groups.get("UNET", []))
    clip_keys = list(groups.get("CLIP", []))
    parsed = unet.extract(model_data, unet_keys, arch_name, clip_keys)
    if parsed is not None:
        return parsed
    return generic.extract(model_data, groups, arch_name)
