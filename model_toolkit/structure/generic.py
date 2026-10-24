"""无法做架构特化时的组件级回退。"""
from .common import SOURCE_WEIGHTS, sum_params


def extract(model_data, groups, arch_name=None):
    """返回组件级结构，页面据此显示通用视图与「特化视图规划中」提示。"""
    components = [
        {
            "name": name,
            "params": sum_params(model_data, keys),
            "key_count": len(keys),
            "source": SOURCE_WEIGHTS,
        }
        for name, keys in groups.items()
        if keys
    ]
    total_params = sum(meta.numel() for meta in model_data.values())
    unet_params = sum_params(model_data, groups.get("UNET", []))

    return {
        "kind": "generic",
        "naming": None,
        "architecture": arch_name,
        "capabilities": {},
        "components": components,
        "unattributed": {"params": 0, "key_count": 0, "keys": []},
        "totals": {
            "params": total_params,
            "key_count": len(model_data),
            "unet_params": unet_params,
        },
        "note": "Architecture-specific view is not available for this model yet.",
    }
