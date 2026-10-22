"""把 components/architectures/*.yaml 编译成运行时索引 index.json。

YAML 是**源数据**（便于与上游 diff），运行时只读编译后的 JSON：
纯 Python 的 YAML 解析这 8 份文件要数秒，而 JSON 只要几十毫秒。

用法：
    python tools/build_template_index.py            # 生成/更新 index.json
    python tools/build_template_index.py --check    # 只校验是否与 YAML 同步（CI/本地均可）
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


def build_index():
    result = {}
    for path in sorted(ARCH_DIR.glob("*.yaml")):
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader) or {}
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
    index = build_index()
    blob = serialise(index)

    if "--check" in argv:
        if not INDEX_PATH.is_file():
            print("✗ index.json 不存在，请先运行 python tools/build_template_index.py")
            return 1
        if INDEX_PATH.read_text(encoding="utf-8") != blob:
            print("✗ index.json 与 YAML 不同步，请重新生成")
            return 1
        print(f"✓ index.json 与 {len(index['templates'])} 份 YAML 同步")
        return 0

    INDEX_PATH.write_text(blob, encoding="utf-8")
    size_kb = len(blob.encode("utf-8")) / 1024
    print(f"✓ 已生成 {INDEX_PATH}（{len(index['templates'])} 份模板，{size_kb:.0f} KB）")
    for name, payload in index["templates"].items():
        counts = ", ".join(f"{c}:{len(e)}" for c, e in payload["components"].items())
        print(f"  {name:32s} {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
