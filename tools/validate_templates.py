"""用真实模型校验架构模板：覆盖率、架构判定、组件契约。

用法：
    python tools/validate_templates.py <模型路径> [<模型路径> ...]

只读 safetensors 头部，不加载权重。退出码：全部模型都能被模板识别为 0，否则 1。
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 直接导入子包，避免触发外层 __init__（那里依赖 ComfyUI 的 folder_paths）
from model_toolkit import templates  # noqa: E402


def read_keys(model_path):
    with open(model_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    return list(header)


def main(argv):
    if not argv:
        print(__doc__)
        return 2

    failures = 0
    for model_path in argv:
        path = Path(model_path)
        if not path.is_file():
            print(f"✗ {model_path}: 文件不存在")
            failures += 1
            continue

        keys = read_keys(path)
        spec, coverage = templates.detect(keys)
        tag = spec.identifier if spec else "未识别"
        print(f"\n{path.name} | {len(keys)} 键 | 判定 {tag} | 覆盖率 {coverage:.1%}")

        if spec is None:
            failures += 1
            continue

        for component in ("UNET", "VAE", "CLIP"):
            expected = spec.keys(component, required_only=True)
            present = len(expected & set(keys))
            flag = "✓" if present == len(expected) else "!"
            print(f"  {flag} {component:5s} 模板 {len(expected):5d} 键 | 模型内命中 {present:5d}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
