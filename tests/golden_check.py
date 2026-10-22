"""黄金样本比对（G9）。

用法（ComfyUI 自带 Python，torch 2.14.0+cpu）：
    & "D:\\AIGC\\ComfyUI\\.ext\\python.exe" tests\\golden_check.py           # 比对
    & "D:\\AIGC\\ComfyUI\\.ext\\python.exe" tests\\golden_check.py --write   # 重建基线（重构期间禁止）

比对内容：文本报告的长度与 SHA256、图像张量的 shape/dtype/完整字节 SHA256、
固定错误路径的返回文本。环境（ComfyUI/Python/torch）记录在 golden.json 里，
环境不一致时只告警，不静默通过。
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

import _env  # noqa: F401  设置 sys.path

import ComfyUI_model_toolkit as pkg

GOLDEN = pathlib.Path(__file__).with_name("golden.json")
MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_sha(tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


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
    scenarios = {}

    text, image = pkg.CheckpointInspector().analyze_model(MODEL, "基础", False)
    scenarios["basic"] = {
        "text_len": len(text),
        "text_sha256": _text_sha(text),
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_sha256": _tensor_sha(image),
    }

    text, image = pkg.CheckpointInspector().analyze_model(MODEL, "高级", True)
    scenarios["advanced_chart"] = {
        "text_len": len(text),
        "text_sha256": _text_sha(text),
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_sha256": _tensor_sha(image),
    }

    (unet_text,) = pkg.UNetInspector().inspect_unet(MODEL, "简略")
    scenarios["unet"] = {"text_len": len(unet_text), "text_sha256": _text_sha(unet_text)}

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
