"""架构模板：组件划分与逐键形状契约。

数据来源：sd_mecha 1.1.7 的内置 ``model_configs``（MIT，见同目录 LICENSE-sd-mecha）。
上游用的是原始 checkpoint 键，组件名与本项目不同（``diffuser``/``clip_l``），
本模块负责名称与 dtype 的适配，并用「模型键被模板覆盖率」做架构判定。

注意：上游的 ``ema`` 段存在键名丢点的问题（``model_ema.diffusion_modelinput_blocks00bias``），
对真实 checkpoint 命中率近乎 0，因此 **不用于匹配**；EMA 仍由 ``model_ema.`` 前缀规则识别。
"""
import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__package__.split(".")[0])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "components" / "architectures"

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

# 模板 identifier → 报告里显示的架构名
DISPLAY_NAMES = {
    "sd1-ldm": "SD1.x",
    "sdxl-sgm": "SDXL",
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

    def coverage(self, model_keys):
        """模型键被模板覆盖的比例（模板认不认得这个模型）。"""
        model_keys = set(model_keys)
        if not model_keys:
            return 0.0
        return len(self.trusted_keys() & model_keys) / len(model_keys)

    def contract(self, component, present_keys):
        """组件契约核对：返回 (命中必需键数, 必需键总数)。"""
        expected = self.keys(component, required_only=True)
        return len(expected & set(present_keys)), len(expected)

    def __repr__(self):
        return f"ArchitectureSpec({self.identifier}, {len(self.components)} components)"


def _parse_component(entries):
    parsed = {}
    for key, meta in (entries or {}).items():
        meta = meta or {}
        parsed[key] = {
            "shape": tuple(meta.get("shape") or ()),
            "dtype": DTYPE_ALIASES.get(str(meta.get("dtype")), None),
            "optional": bool(meta.get("optional", False)),
        }
    return parsed


def load_specs(directory=TEMPLATES_DIR):
    """读取目录下所有模板；坏文件跳过并记日志，不影响其余模板。"""
    specs = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.error(f"读取架构模板 {path.name} 失败: {str(e)}")
            continue

        identifier = data.get("identifier") or path.stem
        components = {}
        for raw_name, entries in (data.get("components") or {}).items():
            if raw_name.lower() in IGNORED_COMPONENTS:
                continue
            name = COMPONENT_ALIASES.get(raw_name.lower(), raw_name.upper())
            # 多个上游组件可能映射到同一个本项目组件（如 clip_l + clip_g → CLIP）
            components.setdefault(name, {}).update(_parse_component(entries))
        specs[identifier] = ArchitectureSpec(identifier, components, source=path.name)
    return specs


_SPECS = None


def get_specs():
    """进程内缓存：模板是只读数据，加载一次即可。"""
    global _SPECS
    if _SPECS is None:
        _SPECS = load_specs()
    return _SPECS


def detect(model_keys, specs=None, min_coverage=0.95):
    """按覆盖率选出架构模板；达不到阈值时返回 (None, 最佳覆盖率)。"""
    specs = get_specs() if specs is None else specs
    best, best_coverage = None, 0.0
    for spec in specs.values():
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
