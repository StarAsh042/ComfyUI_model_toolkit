"""架构识别（家族级）与能力矩阵。

判据参考 ComfyUI 0.42.4 的 ``comfy/model_detection.py``（GPL-3.0，与本项目许可一致），
只取「某个特征键是否存在」这类结构事实，不复制其检测实现。

与 ``templates.py`` 的分工：

- ``templates``：已知架构的**逐键组件归属**（只有 SD1/SDXL 等有模板）；
- ``architectures``：**认得出更多架构**，并说明本插件的三组件统计是否适用于它。

识别到 ``partial`` / ``unsupported`` 时，报告里会给出一行提示，避免把 T5 参数漏算成
「CLIP 只有这么点」这种误导性数字。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    """一个架构（或其家族）的识别结果与能力说明。"""

    id: str
    display: str
    family: str  # sd / mmdit / video / cascade / audio
    capability: str  # complete / partial / unsupported
    note: str
    verified: bool = False  # 是否用本机真实模型验证过


SD = Architecture(
    id="sd",
    display="SD family",
    family="sd",
    capability="complete",
    note="",
    verified=True,
)

FLUX = Architecture(
    id="flux",
    display="Flux family",
    family="mmdit",
    capability="partial",
    note="MMDiT backbone; the text encoder includes T5, so CLIP statistics only cover CLIP-L/G",
)

SD3 = Architecture(
    id="sd3",
    display="SD3 family",
    family="mmdit",
    capability="partial",
    note="Includes a T5-XXL text encoder, so CLIP statistics are incomplete",
)

AURAFLOW = Architecture(
    id="auraflow",
    display="AuraFlow",
    family="mmdit",
    capability="partial",
    note="MMDiT backbone; the text encoder includes T5, so CLIP statistics are incomplete",
)

PIXART = Architecture(
    id="pixart",
    display="PixArt family",
    family="mmdit",
    capability="partial",
    note="Uses a T5 text encoder, so CLIP statistics do not apply",
)

HUNYUANDIT = Architecture(
    id="hunyuan_dit",
    display="HunyuanDiT",
    family="mmdit",
    capability="partial",
    note="Dual text encoders (including T5), so CLIP statistics are incomplete",
)

CASCADE = Architecture(
    id="stable_cascade",
    display="Stable Cascade",
    family="cascade",
    capability="partial",
    note="Multi-stage cascade model; a single UNET does not represent the full model size",
)

LTXV = Architecture(
    id="ltxv",
    display="LTXV",
    family="video",
    capability="unsupported",
    note="Video diffusion model with a 3D VAE and temporal layers, so image statistics do not apply",
)

SVD = Architecture(
    id="svd",
    display="Stable Video Diffusion",
    family="video",
    capability="unsupported",
    note="Video diffusion model with temporal layers, so static image statistics do not apply",
)

MINIMAX_H3 = Architecture(
    id="minimax_h3",
    display="MiniMax H3",
    family="video",
    capability="unsupported",
    note="Joint audio-video model, so three-component statistics do not apply",
)

# 元数据（modelspec.architecture）→ 架构；只有少数导出工具会写这个字段
_METADATA_HINTS = {
    "stable-diffusion-v1": SD,
    "stable-diffusion-xl": SD,
    "stable-diffusion-2": SD,
    "flux": FLUX,
    "stable-diffusion-3": SD3,
}

# 顺序敏感：先匹配特征更强的
_KEY_RULES = (
    (("video_patch_proj.", "audio_patch_proj."), MINIMAX_H3),
    (("double_blocks.",), FLUX),
    (("joint_blocks.",), SD3),
    (("joint_transformer_blocks.",), AURAFLOW),
    (("mlp_t5.",), HUNYUANDIT),
    (("adaln_single.", "pos_embed.proj."), PIXART),
    (("adaln_single.",), LTXV),
    (("clf.1.weight",), CASCADE),
    (("time_stack.",), SVD),
    (("model.diffusion_model.input_blocks.", "diffusion_model.input_blocks."), SD),
)


def _has_all(keys, needles):
    remaining = set(needles)
    for key in keys:
        for needle in list(remaining):
            if needle in key:
                remaining.discard(needle)
        if not remaining:
            return True
    return False


def identify(model_keys, metadata=None):
    """识别架构；无法判断时返回 None。

    参数 ``metadata`` 为可选的 ``__metadata__`` 字典（少数导出工具会写
    ``modelspec.architecture``），优先于键规则。
    """
    if metadata:
        hint = str(metadata.get("modelspec.architecture", "")).lower()
        for prefix, arch in _METADATA_HINTS.items():
            if hint.startswith(prefix):
                return arch

    keys = list(model_keys)
    for needles, arch in _KEY_RULES:
        if _has_all(keys, needles):
            return arch
    return None


def report_hint(arch):
    """给报告用的提示行；``complete`` 架构不需要提示。"""
    if arch is None or arch.capability == "complete":
        return None
    level = "⚠️" if arch.capability == "partial" else "⛔"
    return f"{level} Architecture note: {arch.display} — {arch.note}"
