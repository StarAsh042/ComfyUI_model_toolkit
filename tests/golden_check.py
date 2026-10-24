"""黄金样本比对（G9）。

用法（ComfyUI 自带 Python，torch 2.14.0+cpu）：
    & "D:\\AIGC\\ComfyUI\\.ext\\python.exe" tests\\golden_check.py           # 比对
    & "D:\\AIGC\\ComfyUI\\.ext\\python.exe" tests\\golden_check.py --write   # 重建基线

2.0.0 起节点零输出，基线从「文本/图像哈希」切换为报告 JSON 的规范化哈希：
- `report`：完整报告（含键名清单）；
- `report_no_keys`：`include_keys=False` 的页面数据；
- `replacer_error`：固定错误路径的返回文本。
环境（ComfyUI/Python/torch）记录在 golden.json 里，环境不一致时只告警。
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

import _env  # noqa: F401  设置 sys.path
import ComfyUI_model_toolkit as pkg
from ComfyUI_model_toolkit.model_toolkit.report import build_report

# Windows 控制台默认 GBK，报告里的 emoji 无法编码；统一按 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GOLDEN = pathlib.Path(__file__).with_name("golden.json")
MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"


def _report_sha(report: dict) -> str:
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _report_summary(report: dict) -> dict:
    summary = report["summary"]
    return {
        "file_name": summary["file_name"],
        "model_dir": summary["model_dir"],
        "architecture": summary["architecture"],
        "key_count": summary["key_count"],
        "component_count": summary["component_count"],
        "total_params": summary["total_params"],
    }


def _environment() -> dict:
    import torch

    version_file = _env.COMFYUI_DIR / "comfyui_version.py"
    match = re.search(
        r'__version__\s*=\s*"([^"]+)"', version_file.read_text(encoding="utf-8")
    )
    return {
        "comfyui": match.group(1) if match else "unknown",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def _measure() -> dict:
    full = build_report("checkpoints", MODEL, include_keys=True)
    compact = build_report("checkpoints", MODEL, include_keys=False)
    scenarios = {
        "report": {
            "summary": _report_summary(full),
            "has_keys": "keys" in full,
            "sha256": _report_sha(full),
        },
        "report_no_keys": {
            "summary": _report_summary(compact),
            "has_keys": "keys" in compact,
            "sha256": _report_sha(compact),
        },
    }

    (replacer_text,) = pkg.ComponentReplacer().replace_component(
        MODEL, "UNET", "UNET-v1-SD.txt", "golden_check"
    )
    scenarios["replacer_error"] = {"text": replacer_text}

    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="重建 golden.json 基线")
    args = parser.parse_args()

    current = {"model": MODEL, "environment": _environment(), "scenarios": _measure()}

    if args.write:
        GOLDEN.write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已写入基线: {GOLDEN}")
        return 0

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if expected["model"] != current["model"]:
        print(f"FAIL 模型不一致: {expected['model']} != {current['model']}")
        return 1
    if expected["environment"] != current["environment"]:
        print(f"WARN 环境不一致: {expected['environment']} -> {current['environment']}")

    failed = 0
    for name, want in expected["scenarios"].items():
        got = current["scenarios"].get(name)
        if got == want:
            print(f"[PASS] {name}")
        else:
            failed += 1
            print(f"[FAIL] {name}")
            print(f"   期望: {want}")
            print(f"   实际: {got}")
    print(f"\n{len(expected['scenarios']) - failed}/{len(expected['scenarios'])} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
