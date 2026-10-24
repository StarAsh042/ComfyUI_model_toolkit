"""架构模板：组件划分与逐键形状契约。

数据来源：sd_mecha 1.1.7 的内置 ``model_configs``（MIT，见同目录 LICENSE-sd-mecha）。
上游用的是原始 checkpoint 键，组件名与本项目不同（``diffuser``/``clip_l``）。

运行时读取编译后的 ``components/architectures/index.json``：
YAML 是源数据（便于与上游 diff），但纯 Python 解析这 10 份要数秒，
JSON 只要几十毫秒。改过 YAML 后必须运行 ``tools/build_template_index.py``。

注意：上游的 ``ema`` 段存在键名丢点的问题（``model_ema.diffusion_modelinput_blocks00bias``），
对真实 checkpoint 命中率近乎 0，因此 **不用于匹配**；EMA 仍由 ``model_ema.`` 前缀规则识别。
"""
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__package__.split(".")[0])

ARCH_DIR = Path(__file__).resolve().parent.parent / "components" / "architectures"
INDEX_PATH = ARCH_DIR / "index.json"

# sd-mecha 组件名 → 本项目组件名
COMPONENT_ALIASES = {
    "diffuser": "UNET",
    "clip_l": "CLIP",
    "clip_g": "CLIP",  # 本项目节点只区分 CLIP，SDXL 的双编码器合并为同一组件
    "vae": "VAE",
    "ema": "EMA",
}

# 模型文件通常不保存、本项目也不分析的组件
IGNORED_COMPONENTS = {"scheduler"}

# 只用于匹配的组件（ema 段数据损坏，见模块 docstring）
UNTRUSTED_COMPONENTS = {"EMA"}

# 本项目节点能表达的组件：模板若含其它组件（T5XXL / BLOCKS 等），不参与自动识别
REPRESENTABLE_COMPONENTS = {"UNET", "VAE", "CLIP", "EMA"}

# 模板（文件名 stem）→ 报告里显示的架构名
DISPLAY_NAMES = {
    "sd1-ldm": "SD1.x",
    "sd1-kohya": "SD1.x (kohya)",
    "sdxl-sgm": "SDXL",
    "sdxl-kohya": "SDXL (kohya)",
    "sdxl-kohya_but_diffusers": "SDXL (kohya/diffusers)",
    "sdxl-diffusers_unet_only": "SDXL (diffusers UNet)",
    "flux-flux": "Flux",
    "sd3-comfyui": "SD3",
    "sd1-supermerger_blocks": "SD1.x (SuperMerger blocks)",
    "sdxl-supermerger_blocks": "SDXL (SuperMerger blocks)",
}

DTYPE_ALIASES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class ArchitectureSpec:
    """一份架构模板：组件名 → {原始键: 元信息}。"""

    def __init__(self, identifier, components, source=None):
        self.identifier = identifier
        self.components = components
        self.source = source
        # 含 T5XXL / BLOCKS 等本项目无法表达的组件时，不参与自动识别
        self.supported = set(components).issubset(REPRESENTABLE_COMPONENTS)
        self._trusted = None

    @property
    def display_name(self):
        return DISPLAY_NAMES.get(self.identifier, self.identifier)

    def keys(self, component, required_only=False):
        """某个组件的键集合（EMA 段不可信，恒返回空）。

        required_only=True 时排除上游标注 optional 的键（不同模型未必带这些）。
        """
        if component.upper() in UNTRUSTED_COMPONENTS:
            return set()
        entries = self.components.get(component.upper(), {})
        if required_only:
            return {key for key, meta in entries.items() if not meta.get("optional")}
        return set(entries)

    def trusted_keys(self):
        if self._trusted is None:
            trusted = set()
            for component in self.components:
                trusted |= self.keys(component)
            self._trusted = trusted
        return self._trusted

    def coverage(self, model_keys):
        """模型键被模板覆盖的比例（模板认不认得这个模型）。"""
        model_keys = set(model_keys)
        if not model_keys:
            return 0.0
        return len(self.trusted_keys() & model_keys) / len(model_keys)

    def assignment(self, keys):
        """逐键判定组件：返回 {键: 组件名}，只包含模板收录且可信的键。"""
        wanted = set(keys)
        result = {}
        for component in self.components:
            if component.upper() in UNTRUSTED_COMPONENTS:
                continue
            for key in self.components[component]:
                if key in wanted:
                    result[key] = component
        return result

    def contract(self, component, present_keys):
        """组件契约核对：返回 (命中必需键数, 必需键总数)。"""
        expected = self.keys(component, required_only=True)
        return len(expected & set(present_keys)), len(expected)

    def __repr__(self):
        return f"ArchitectureSpec({self.identifier}, {len(self.components)} components)"


def _parse_component(entries):
    return {
        key: {
            "shape": tuple(meta[0] or ()),
            "dtype": DTYPE_ALIASES.get(str(meta[1])),
            "optional": bool(meta[2]),
        }
        for key, meta in (entries or {}).items()
    }


def load_specs(index_path=INDEX_PATH):
    """读取编译索引；文件缺失或损坏时返回空表（节点回退到前缀规则）。"""
    path = Path(index_path)
    if not path.is_file():
        logger.error(f"Architecture template index missing: {path} (run tools/build_template_index.py)")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load architecture template index: {str(e)}")
        return {}

    specs = {}
    for name, payload in (data.get("templates") or {}).items():
        components = payload.get("components", payload)
        specs[name] = ArchitectureSpec(
            name,
            {component: _parse_component(entries) for component, entries in components.items()},
            source=path.name,
        )
    return specs


_SPECS = None


def get_specs():
    """进程内缓存：模板是只读数据，加载一次即可。"""
    global _SPECS
    if _SPECS is None:
        _SPECS = load_specs()
    return _SPECS


def detect(model_keys, specs=None, min_coverage=0.95):
    """按覆盖率选出架构模板；只考虑节点能表达的组件。

    同一架构可能有多份模板（如 SDXL 的 sgm / kohya），取覆盖率最高者；
    覆盖率相同时按 identifier 字典序先到先得（加载顺序稳定）。
    """
    specs = get_specs() if specs is None else specs
    best, best_coverage = None, 0.0
    for identifier in sorted(specs):
        spec = specs[identifier]
        if not spec.supported:
            continue
        coverage = spec.coverage(model_keys)
        if coverage > best_coverage:
            best, best_coverage = spec, coverage
    if best is not None and best_coverage >= min_coverage:
        return best, best_coverage
    return None, best_coverage


def component_of(spec, key):
    """模板判定某个键属于哪个组件；未收录返回 None。"""
    for component in spec.components:
        if key in spec.components[component]:
            return component
    return None
