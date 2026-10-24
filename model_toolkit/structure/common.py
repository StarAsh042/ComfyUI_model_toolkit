"""结构提取共用工具：键名归一化、参数汇总与节点构造。

所有数值都带 ``source``：权重直接算出的标 ``weights``，由约定推导的
（如 stride=2）标 ``convention``，权重里不存在的（如注意力头数）不猜。
"""
from collections import Counter

SOURCE_WEIGHTS = "weights"
SOURCE_CONVENTION = "convention"
SOURCE_TEMPLATE = "template"
SOURCE_USER = "user"
SOURCE_UNKNOWN = "unknown"

# 归一化时只保留从这些命名空间开始的键；前面的 checkpoint 包装前缀
# （model.diffusion_model. 等）单独记录，不参与结构解析。
BASE_TOKENS = (
    "input_blocks.",
    "down_blocks.",
    "middle_block.",
    "mid_block.",
    "output_blocks.",
    "up_blocks.",
    "conv_in.",
    "conv_out.",
    "time_embedding.",
    "time_embed.",
    "label_emb.",
    "add_embedding.",
    "out.",
)


def split_prefix(key):
    """把完整键拆成（结构键，包装前缀）。"""
    positions = [key.find(token) for token in BASE_TOKENS]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return key, ""
    pos = min(positions)
    return key[pos:], key[:pos]


def detect_prefix(keys):
    """取出现次数最多的包装前缀；纯结构键（提取产物）返回空串。"""
    prefixes = Counter(prefix for _, prefix in (split_prefix(key) for key in keys) if prefix)
    return prefixes.most_common(1)[0][0] if prefixes else ""


def sum_params(meta, keys):
    return sum(meta[key].numel() for key in keys if key in meta)


def make_transition(kind, keys, meta):
    """下采样 / 上采样节点；stride/factor 是网络约定，不在权重中。"""
    data = {
        "kind": kind,
        "params": sum_params(meta, keys),
        "key_count": len(keys),
        "source": SOURCE_WEIGHTS,
    }
    if kind == "downsample":
        data["stride"] = 2
        data["stride_source"] = SOURCE_CONVENTION
    else:
        data["factor"] = 2
        data["factor_source"] = SOURCE_CONVENTION
    return data


def make_stage(stage_id, direction, level, blocks, transition, channels=None, skip_from=None):
    """阶段节点；params/key_count 汇总子块，供 UI 与不变量测试使用。"""
    params = sum(block["params"] for block in blocks)
    key_count = sum(block["key_count"] for block in blocks)
    if transition is not None:
        params += transition["params"]
        key_count += transition["key_count"]

    data = {
        "id": stage_id,
        "direction": direction,
        "level": level,
        "resolution_ratio": round(1 / (2 ** level), 6) if level is not None else None,
        "resolution_source": SOURCE_CONVENTION,
        "channels": channels,
        "channels_source": SOURCE_WEIGHTS if channels is not None else SOURCE_UNKNOWN,
        "blocks": blocks,
        "params": params,
        "key_count": key_count,
        "transition": transition,
    }
    if skip_from is not None:
        data["skip_from"] = skip_from
        data["skip_source"] = SOURCE_CONVENTION
    return data


def make_io_entry(keys, meta):
    """输入卷积 / 输出卷积 / 时间嵌入的统计。"""
    entry = {
        "params": sum_params(meta, keys),
        "key_count": len(keys),
        "source": SOURCE_WEIGHTS,
    }
    for key in keys:
        shape = meta[key].shape
        if not key.endswith(".weight") or len(shape) not in (2, 4):
            continue
        if len(shape) == 4:
            entry["out_channels"] = shape[0]
            entry["in_channels"] = shape[1]
        else:
            entry["out_features"] = shape[0]
            entry["in_features"] = shape[1]
        break
    return entry


def base_size_for(arch_name):
    """Base resolution is a convention when no user choice exists.

    SD1.x defaults to 512 and SDXL to 1024 in common inference setups; both
    remain user-selectable in the report UI, so the value is labelled
    ``convention`` rather than ``weights``.
    """
    if not arch_name:
        return None
    name = str(arch_name).upper()
    if "SDXL" in name:
        return 1024
    if "SD1" in name or "SD2" in name:
        return 512
    return None
