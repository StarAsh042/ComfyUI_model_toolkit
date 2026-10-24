"""UNet 结构提取：从权重键名还原 down / middle / up 阶段。

同时支持两种命名：

- LDM / CompVis / kohya：``input_blocks.*`` / ``middle_block.*`` / ``output_blocks.*``
- diffusers：``down_blocks.*`` / ``mid_block.*`` / ``up_blocks.*``

层数与通道数全部从键名和形状推导，不硬编码 SD1/SDXL 的块数。
注意力头数不在权重中，统一返回 ``null`` 并标注 ``unknown``。
"""
import re
from collections import OrderedDict, defaultdict

from .common import (
    SOURCE_CONVENTION,
    SOURCE_UNKNOWN,
    SOURCE_WEIGHTS,
    base_size_for,
    detect_prefix,
    make_io_entry,
    make_stage,
    make_transition,
    split_prefix,
    sum_params,
)

_INPUT_RE = re.compile(r"^input_blocks\.(\d+)\.(\d+)\.")
_OUTPUT_RE = re.compile(r"^output_blocks\.(\d+)\.(\d+)\.")
_OUTPUT_CONV_RE = re.compile(r"^output_blocks\.\d+\.\d+\.conv\.")
_MIDDLE_RE = re.compile(r"^middle_block\.(\d+)\.")

_DOWN_RE = re.compile(r"^down_blocks\.(\d+)\.(resnets|attentions|downsamplers)\.(\d+)\.")
_MID_DIFF_RE = re.compile(r"^mid_block\.(resnets|attentions)\.(\d+)\.")
_UP_RE = re.compile(r"^up_blocks\.(\d+)\.(resnets|attentions|upsamplers)\.(\d+)\.")

# diffusers 同一 stage 内的布局顺序：resnets 在 attentions 之前
_DIFFUSERS_KIND_ORDER = {"resnets": 0, "attentions": 1}

_TRANSFORMER_RE = re.compile(r"transformer_blocks\.(\d+)\.")

_RESBLOCK_RULES_LDM = (
    ("in_layers.0", "norm"),
    ("in_layers.2", "conv"),
    ("emb_layers", "time_mlp"),
    ("out_layers.0", "norm"),
    ("out_layers.3", "conv"),
    ("skip_connection", "skip_proj"),
)

_RESBLOCK_RULES_DIFFUSERS = (
    ("norm1", "norm"),
    ("conv1", "conv"),
    ("time_emb_proj", "time_mlp"),
    ("norm2", "norm"),
    ("conv2", "conv"),
    ("conv_shortcut", "skip_proj"),
)

_TRANSFORMER_RULES = (
    ("attn1", "self_attn"),
    ("attn2", "cross_attn"),
    ("ff.", "ff"),
    ("norm", "norm"),
    ("proj_in", "proj_in"),
    ("proj_out", "proj_out"),
)


def extract(model_data, unet_keys, arch_name=None, clip_keys=()):
    """解析 UNET 组件；命名无法识别时返回 None，由调用方回退到 generic。"""
    if not unet_keys:
        return None

    unet_keys = sorted(unet_keys)
    clip_keys = sorted(clip_keys)
    short_by_key = {key: split_prefix(key)[0] for key in unet_keys}
    ldm_keys = sum(
        1
        for short in short_by_key.values()
        if short.startswith(("input_blocks.", "middle_block.", "output_blocks."))
    )
    diffusers_keys = sum(
        1
        for short in short_by_key.values()
        if short.startswith(("down_blocks.", "mid_block.", "up_blocks."))
    )

    if ldm_keys and ldm_keys >= diffusers_keys:
        return _extract_ldm(model_data, unet_keys, short_by_key, arch_name, clip_keys)
    if diffusers_keys:
        return _extract_diffusers(
            model_data, unet_keys, short_by_key, arch_name, clip_keys
        )
    return None


def _block_kind(shorts):
    if any(
        "transformer_blocks" in short or "attn1." in short or "attn2." in short
        for short in shorts
    ):
        return "transformer"
    if any(
        marker in short
        for short in shorts
        for marker in ("in_layers", "out_layers", "emb_layers", "resnets")
    ):
        return "resblock"
    return "unknown"


def _children(kind, keys, short_by_key, meta, naming):
    if kind == "resblock":
        rules = _RESBLOCK_RULES_LDM if naming == "ldm" else _RESBLOCK_RULES_DIFFUSERS
    elif kind == "transformer":
        rules = _TRANSFORMER_RULES
    else:
        rules = ()

    buckets = OrderedDict()
    for key in keys:
        short = short_by_key[key]
        label = next((label for token, label in rules if token in short), "other")
        buckets.setdefault(label, []).append(key)

    return [
        {
            "kind": label,
            "params": sum_params(meta, bucket_keys),
            "key_count": len(bucket_keys),
        }
        for label, bucket_keys in buckets.items()
    ]


def _shape_dim(keys, short_by_key, meta, suffix, dim):
    for key in keys:
        if short_by_key[key].endswith(suffix):
            shape = meta[key].shape
            if len(shape) > dim:
                return shape[dim]
    return None


def _attention_info(keys, short_by_key, meta):
    shorts = [short_by_key[key] for key in keys]
    self_attention = any("attn1." in short for short in shorts)
    cross_attention = any("attn2." in short for short in shorts)
    if not self_attention and not cross_attention:
        return None

    inner_dim = _shape_dim(keys, short_by_key, meta, ".attn1.to_q.weight", 0)
    if inner_dim is None:
        inner_dim = _shape_dim(keys, short_by_key, meta, ".attn2.to_q.weight", 0)
    context_dim = (
        _shape_dim(keys, short_by_key, meta, ".attn2.to_k.weight", 1)
        if cross_attention
        else None
    )
    transformer_blocks = len(
        {
            match.group(1)
            for short in shorts
            for match in [_TRANSFORMER_RE.search(short)]
            if match
        }
    )

    return {
        "self": self_attention,
        "cross": cross_attention,
        "inner_dim": inner_dim,
        "context_dim": context_dim,
        "heads": None,
        "heads_source": SOURCE_UNKNOWN,
        "transformer_blocks": transformer_blocks or None,
    }


def _channels(keys, short_by_key, meta):
    for key in keys:
        shape = meta[key].shape
        short = short_by_key[key]
        if len(shape) == 4 and (
            short.endswith(".out_layers.3.weight") or short.endswith(".conv2.weight")
        ):
            return shape[0]
    for key in keys:
        shape = meta[key].shape
        short = short_by_key[key]
        if len(shape) == 4 and short.endswith(".proj_in.weight"):
            return shape[1]
    return None


def _make_block(block_id, keys, short_by_key, meta, naming):
    kind = _block_kind([short_by_key[key] for key in keys])
    return {
        "id": block_id,
        "kind": kind,
        "params": sum_params(meta, keys),
        "key_count": len(keys),
        "children": _children(kind, keys, short_by_key, meta, naming),
        "attention": _attention_info(keys, short_by_key, meta)
        if kind == "transformer"
        else None,
        "source": SOURCE_WEIGHTS,
    }


def _io_entries(io_keys, meta):
    return {
        name: make_io_entry(keys, meta)
        for name, keys in io_keys.items()
        if keys
    }


def _extras(label_keys, meta):
    return [
        {
            "id": kind,
            "kind": kind,
            "params": sum_params(meta, keys),
            "key_count": len(keys),
            "source": SOURCE_WEIGHTS,
        }
        for kind, keys in label_keys.items()
        if keys
    ]


def _capabilities(io, extras, all_blocks, clip_keys, model_data):
    attention = [block.get("attention") for block in all_blocks]
    has_clip = bool(clip_keys)
    return {
        "time_embedding": "time_embed" in io,
        "self_attention": any(info and info.get("self") for info in attention),
        "text_cross_attention": any(info and info.get("cross") for info in attention),
        "size_conditioning": any(
            extra["kind"] in ("label_embed", "add_embedding") for extra in extras
        ),
        "dual_text_encoder": (
            any(
                "text_model_2" in key or "conditioner.embedders.1" in key
                for key in clip_keys
            )
            if has_clip
            else None
        ),
        "dual_text_encoder_source": SOURCE_WEIGHTS if has_clip else SOURCE_UNKNOWN,
        "refiner": False,
        "v_pred": any("v_pred" in key for key in model_data),
    }


def _latent(io, arch_name):
    conv_in = io.get("conv_in") or {}
    in_channels = conv_in.get("in_channels")
    base_size = base_size_for(arch_name)
    return {
        "in_channels": in_channels,
        "in_channels_source": (
            SOURCE_WEIGHTS if in_channels is not None else SOURCE_UNKNOWN
        ),
        "base_size": base_size,
        "base_size_source": SOURCE_CONVENTION if base_size else SOURCE_UNKNOWN,
        "latent_size": base_size // 8 if base_size else None,
        "latent_size_source": SOURCE_CONVENTION if base_size else SOURCE_UNKNOWN,
    }


def _unattributed(keys, meta):
    return {
        "params": sum_params(meta, keys),
        "key_count": len(keys),
        "keys": sorted(keys),
    }


def _totals(model_data, unet_keys, io, extras, stages, middle, up_stages, unattributed):
    total_params = sum_params(model_data, unet_keys)
    by_stage = [
        {"id": stage["id"], "direction": stage["direction"], "params": stage["params"]}
        for stage in [*stages, middle, *up_stages]
    ]
    return {
        "params": total_params,
        "key_count": len(unet_keys),
        "attributed_params": total_params - unattributed["params"],
        "attributed_key_count": len(unet_keys) - unattributed["key_count"],
        "io_params": sum(entry["params"] for entry in io.values()),
        "extras_params": sum(entry["params"] for entry in extras),
        "unattributed_params": unattributed["params"],
        "by_stage": by_stage,
    }


def _finish(
    naming,
    unet_keys,
    arch_name,
    io,
    extras,
    down_stages,
    middle,
    up_stages,
    unattributed,
    capabilities,
    latent,
    totals,
):
    return {
        "kind": "unet",
        "naming": naming,
        "prefix": detect_prefix(unet_keys),
        "architecture": arch_name,
        "latent": latent,
        "capabilities": capabilities,
        "io": io,
        "stages": down_stages,
        "middle": middle,
        "up": up_stages,
        "extras": extras,
        "unattributed": unattributed,
        "totals": totals,
    }


def _extract_ldm(model_data, unet_keys, short_by_key, arch_name, clip_keys):
    io_keys = {"conv_in": [], "conv_out": [], "time_embed": []}
    label_keys = {"label_embed": []}
    down_blocks = defaultdict(list)
    down_transitions = defaultdict(list)
    middle_blocks = defaultdict(list)
    up_blocks = defaultdict(list)
    up_transitions = defaultdict(list)
    unattributed = []

    for key in unet_keys:
        short = short_by_key[key]

        if short.startswith("input_blocks."):
            match = _INPUT_RE.match(short)
            if not match:
                unattributed.append(key)
                continue
            index, sub = int(match.group(1)), int(match.group(2))
            if index == 0:
                io_keys["conv_in"].append(key)
            elif ".op." in short:
                down_transitions[index].append(key)
            else:
                down_blocks[(index, sub)].append(key)
        elif short.startswith("output_blocks."):
            match = _OUTPUT_RE.match(short)
            if not match:
                unattributed.append(key)
                continue
            index, sub = int(match.group(1)), int(match.group(2))
            if _OUTPUT_CONV_RE.match(short):
                up_transitions[index].append(key)
            else:
                up_blocks[(index, sub)].append(key)
        elif short.startswith("middle_block."):
            match = _MIDDLE_RE.match(short)
            if not match:
                unattributed.append(key)
                continue
            middle_blocks[int(match.group(1))].append(key)
        elif short.startswith("time_embed."):
            io_keys["time_embed"].append(key)
        elif short.startswith("label_emb."):
            label_keys["label_embed"].append(key)
        elif short.startswith("out.") or short.startswith("conv_out."):
            io_keys["conv_out"].append(key)
        else:
            unattributed.append(key)

    down_slots = {}
    level = 0
    for index in sorted({n for n, _ in down_blocks} | set(down_transitions)):
        slot = down_slots.setdefault(level, {"blocks": [], "transition": None})
        for sub in sorted(sub for (n, sub) in down_blocks if n == index):
            slot["blocks"].append((index, sub, down_blocks[(index, sub)]))
        if index in down_transitions:
            slot["transition"] = down_transitions[index]
            level += 1

    down_stages = []
    for lvl, slot in sorted(down_slots.items()):
        blocks = [
            _make_block(
                f"input_blocks.{index}.{sub}",
                keys,
                short_by_key,
                model_data,
                "ldm",
            )
            for index, sub, keys in slot["blocks"]
        ]
        transition = (
            make_transition("downsample", slot["transition"], model_data)
            if slot["transition"]
            else None
        )
        stage_keys = [key for _, _, keys in slot["blocks"] for key in keys]
        if slot["transition"]:
            stage_keys.extend(slot["transition"])
        down_stages.append(
            make_stage(
                f"down.{lvl}",
                "down",
                lvl,
                blocks,
                transition,
                channels=_channels(stage_keys, short_by_key, model_data),
            )
        )

    middle_items = [
        _make_block(
            f"middle_block.{sub}",
            middle_blocks[sub],
            short_by_key,
            model_data,
            "ldm",
        )
        for sub in sorted(middle_blocks)
    ]
    max_level = max((stage["level"] for stage in down_stages), default=0)
    middle = make_stage(
        "middle",
        "middle",
        max_level,
        middle_items,
        None,
        channels=_channels(
            [key for keys in middle_blocks.values() for key in keys],
            short_by_key,
            model_data,
        ),
    )

    up_slots = {}
    up_level = max_level
    order = 0
    for index in sorted({n for n, _ in up_blocks} | set(up_transitions)):
        slot = up_slots.setdefault(
            up_level, {"blocks": [], "transition": None, "order": order}
        )
        for sub in sorted(sub for (n, sub) in up_blocks if n == index):
            slot["blocks"].append((index, sub, up_blocks[(index, sub)]))
        if index in up_transitions:
            slot["transition"] = up_transitions[index]
            up_level = max(up_level - 1, 0)
            order += 1

    up_stages = []
    for lvl, slot in sorted(up_slots.items(), key=lambda item: item[1]["order"]):
        blocks = [
            _make_block(
                f"output_blocks.{index}.{sub}",
                keys,
                short_by_key,
                model_data,
                "ldm",
            )
            for index, sub, keys in slot["blocks"]
        ]
        transition = (
            make_transition("upsample", slot["transition"], model_data)
            if slot["transition"]
            else None
        )
        stage_keys = [key for _, _, keys in slot["blocks"] for key in keys]
        if slot["transition"]:
            stage_keys.extend(slot["transition"])
        up_stages.append(
            make_stage(
                f"up.{slot['order']}",
                "up",
                lvl,
                blocks,
                transition,
                channels=_channels(stage_keys, short_by_key, model_data),
                skip_from=f"down.{lvl}" if lvl in down_slots else None,
            )
        )

    io = _io_entries(io_keys, model_data)
    extras = _extras(label_keys, model_data)
    unattributed_entry = _unattributed(unattributed, model_data)
    all_blocks = [
        block for stage in [*down_stages, middle, *up_stages] for block in stage["blocks"]
    ]
    return _finish(
        "ldm",
        unet_keys,
        arch_name,
        io,
        extras,
        down_stages,
        middle,
        up_stages,
        unattributed_entry,
        _capabilities(io, extras, all_blocks, clip_keys, model_data),
        _latent(io, arch_name),
        _totals(
            model_data,
            unet_keys,
            io,
            extras,
            down_stages,
            middle,
            up_stages,
            unattributed_entry,
        ),
    )


def _extract_diffusers(model_data, unet_keys, short_by_key, arch_name, clip_keys):
    io_keys = {"conv_in": [], "conv_out": [], "time_embed": []}
    label_keys = {"add_embedding": []}
    down_blocks = defaultdict(list)
    down_transitions = defaultdict(list)
    middle_blocks = defaultdict(list)
    up_blocks = defaultdict(list)
    up_transitions = defaultdict(list)
    unattributed = []

    for key in unet_keys:
        short = short_by_key[key]

        if short.startswith("conv_in."):
            io_keys["conv_in"].append(key)
        elif short.startswith("conv_out."):
            io_keys["conv_out"].append(key)
        elif short.startswith("time_embedding."):
            io_keys["time_embed"].append(key)
        elif short.startswith("add_embedding."):
            label_keys["add_embedding"].append(key)
        elif short.startswith("down_blocks."):
            match = _DOWN_RE.match(short)
            if not match:
                unattributed.append(key)
                continue
            index, kind, sub = (
                int(match.group(1)),
                match.group(2),
                int(match.group(3)),
            )
            if kind == "downsamplers":
                down_transitions[index].append(key)
            else:
                down_blocks[(index, kind, sub)].append(key)
        elif short.startswith("mid_block."):
            match = _MID_DIFF_RE.match(short)
            if not match:
                unattributed.append(key)
                continue
            middle_blocks[(match.group(1), int(match.group(2)))].append(key)
        elif short.startswith("up_blocks."):
            match = _UP_RE.match(short)
            if not match:
                unattributed.append(key)
                continue
            index, kind, sub = (
                int(match.group(1)),
                match.group(2),
                int(match.group(3)),
            )
            if kind == "upsamplers":
                up_transitions[index].append(key)
            else:
                up_blocks[(index, kind, sub)].append(key)
        else:
            unattributed.append(key)

    down_stages = []
    down_levels = set()
    for index in sorted({n for n, _, _ in down_blocks} | set(down_transitions)):
        down_levels.add(index)
        blocks = []
        stage_keys = []
        for sub, kind in sorted(
            ((sub, kind) for (n, kind, sub) in down_blocks if n == index),
            key=lambda item: (item[0], _DIFFUSERS_KIND_ORDER.get(item[1], 9)),
        ):
            keys = down_blocks[(index, kind, sub)]
            stage_keys.extend(keys)
            blocks.append(
                _make_block(
                    f"down_blocks.{index}.{kind}.{sub}",
                    keys,
                    short_by_key,
                    model_data,
                    "diffusers",
                )
            )
        transition = None
        if index in down_transitions:
            stage_keys.extend(down_transitions[index])
            transition = make_transition(
                "downsample", down_transitions[index], model_data
            )
        down_stages.append(
            make_stage(
                f"down.{index}",
                "down",
                index,
                blocks,
                transition,
                channels=_channels(stage_keys, short_by_key, model_data),
            )
        )

    middle_items = [
        _make_block(
            f"mid_block.{kind}.{sub}",
            middle_blocks[(kind, sub)],
            short_by_key,
            model_data,
            "diffusers",
        )
        for kind, sub in sorted(
            middle_blocks,
            key=lambda item: (_DIFFUSERS_KIND_ORDER.get(item[0], 9), item[1]),
        )
    ]
    max_level = max(down_levels, default=0)
    middle = make_stage(
        "middle",
        "middle",
        max_level,
        middle_items,
        None,
        channels=_channels(
            [key for keys in middle_blocks.values() for key in keys],
            short_by_key,
            model_data,
        ),
    )

    up_indices = sorted({n for n, _, _ in up_blocks} | set(up_transitions))
    up_stages = []
    for order, index in enumerate(up_indices):
        level = max_level - index if down_levels else len(up_indices) - 1 - order
        blocks = []
        stage_keys = []
        for sub, kind in sorted(
            ((sub, kind) for (n, kind, sub) in up_blocks if n == index),
            key=lambda item: (item[0], _DIFFUSERS_KIND_ORDER.get(item[1], 9)),
        ):
            keys = up_blocks[(index, kind, sub)]
            stage_keys.extend(keys)
            blocks.append(
                _make_block(
                    f"up_blocks.{index}.{kind}.{sub}",
                    keys,
                    short_by_key,
                    model_data,
                    "diffusers",
                )
            )
        transition = None
        if index in up_transitions:
            stage_keys.extend(up_transitions[index])
            transition = make_transition(
                "upsample", up_transitions[index], model_data
            )
        up_stages.append(
            make_stage(
                f"up.{order}",
                "up",
                level,
                blocks,
                transition,
                channels=_channels(stage_keys, short_by_key, model_data),
                skip_from=f"down.{level}" if level in down_levels else None,
            )
        )

    io = _io_entries(io_keys, model_data)
    extras = _extras(label_keys, model_data)
    unattributed_entry = _unattributed(unattributed, model_data)
    all_blocks = [
        block for stage in [*down_stages, middle, *up_stages] for block in stage["blocks"]
    ]
    return _finish(
        "diffusers",
        unet_keys,
        arch_name,
        io,
        extras,
        down_stages,
        middle,
        up_stages,
        unattributed_entry,
        _capabilities(io, extras, all_blocks, clip_keys, model_data),
        _latent(io, arch_name),
        _totals(
            model_data,
            unet_keys,
            io,
            extras,
            down_stages,
            middle,
            up_stages,
            unattributed_entry,
        ),
    )
