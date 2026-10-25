"""把 components/architectures/*.yaml 编译成运行时索引 index.json。

YAML 是**源数据**（便于与上游 diff），运行时只读编译后的 JSON：
纯 Python 的 YAML 解析这 8 份文件要数秒，而 JSON 只要几十毫秒。

用法：
    python tools/build_template_index.py            # 生成/更新 index.json
    python tools/build_template_index.py --check    # 只校验是否与 YAML 同步（CI/本地均可）

输出约定：每行以 ASCII 令牌开头（INDEX_SYNC_OK / INDEX_MISSING /
INDEX_OUT_OF_SYNC / INDEX_WRITTEN / INDEX_TOOL_ERROR），因此 cp936、GBK 等
非 UTF-8 控制台不会因打印符号抛 UnicodeEncodeError 而假失败。
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from model_toolkit import templates  # noqa: E402

ARCH_DIR = REPO_ROOT / "components" / "architectures"
INDEX_PATH = ARCH_DIR / "index.json"
UPSTREAM = "sd_mecha 1.1.7 model_configs"


class ToolError(RuntimeError):
    """索引工具无法继续；消息以 INDEX_TOOL_ERROR 开头。"""


def _emit(text):
    """输出一行；目标代码页无法编码时退化为 ASCII 转义，绝不因编码中断。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def _index_label():
    try:
        return INDEX_PATH.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return INDEX_PATH.name


def _require_c_loader():
    """缺少 libyaml 加速器时明确失败，不静默回退到纯 Python 解析。"""
    loader = getattr(yaml, "CSafeLoader", None)
    if loader is None:
        raise ToolError(
            "INDEX_TOOL_ERROR CSafeLoader unavailable; install PyYAML with libyaml support"
        )
    return loader


def build_index(loader):
    result = {}
    for path in sorted(ARCH_DIR.glob("*.yaml")):
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=loader) or {}
        identifier = data.get("identifier") or path.stem
        components = {}
        for raw_name, entries in (data.get("components") or {}).items():
            if raw_name.lower() in templates.IGNORED_COMPONENTS:
                continue
            name = templates.COMPONENT_ALIASES.get(raw_name.lower(), raw_name.upper())
            bucket = components.setdefault(name, {})
            for key, meta in (entries or {}).items():
                meta = meta or {}
                bucket[key] = [
                    list(meta.get("shape") or []),
                    str(meta.get("dtype")),
                    bool(meta.get("optional", False)),
                ]
        result[path.stem] = {
            "identifier": identifier,
            "components": {
                name: dict(sorted(entries.items()))
                for name, entries in sorted(components.items())
            },
        }
    return {
        "version": 1,
        "generated_from": UPSTREAM,
        "templates": dict(sorted(result.items())),
    }


def serialise(index):
    return json.dumps(index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def main(argv):
    check = "--check" in argv
    label = _index_label()
    try:
        loader = _require_c_loader()
        if check:
            if not INDEX_PATH.is_file():
                _emit(f"INDEX_MISSING {label}; run python tools/build_template_index.py")
                return 1
            index = build_index(loader)
            if INDEX_PATH.read_text(encoding="utf-8") != serialise(index):
                _emit(
                    f"INDEX_OUT_OF_SYNC {label}; "
                    "regenerate with python tools/build_template_index.py"
                )
                return 1
            _emit(f"INDEX_SYNC_OK {label} matches {len(index['templates'])} YAML templates")
            return 0

        index = build_index(loader)
        blob = serialise(index)
        INDEX_PATH.write_text(blob, encoding="utf-8")
        size_kb = len(blob.encode("utf-8")) / 1024
        _emit(
            f"INDEX_WRITTEN {label} ({len(index['templates'])} templates, {size_kb:.0f} KB)"
        )
        for name, payload in index["templates"].items():
            counts = ", ".join(f"{c}:{len(e)}" for c, e in payload["components"].items())
            _emit(f"  {name:32s} {counts}")
        return 0
    except ToolError as error:
        _emit(str(error))
        return 1
    except (OSError, yaml.YAMLError) as error:
        _emit(f"INDEX_TOOL_ERROR {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
