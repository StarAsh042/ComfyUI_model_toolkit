"""行为特征测试：锁定当前行为。

`test_group_model_keys_clip_encoder_stays_in_clip` 与
`RealCheckpointGroupingTest` 锁定的是 B1 修复后的正确分类：
CLIP 文本编码器键必须归 CLIP，不能因键含 'encoder' 被 VAE 抢走。
"""
import tempfile
import unittest
from pathlib import Path

import _env  # noqa: F401  设置 sys.path

import torch

from ComfyUI_model_toolkit import ModelUtils
from ComfyUI_model_toolkit.model_toolkit.analysis import AnalysisMixin
from ComfyUI_model_toolkit.model_toolkit.precision import PrecisionMixin
from ComfyUI_model_toolkit.nodes.component_extractor import ComponentExtractor
from ComfyUI_model_toolkit.nodes.component_replacer import ComponentReplacer


def _safetensors_keys(path):
    """只读 safetensors 文件头，返回键名列表（不加载权重）。"""
    import json
    import struct

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return [key for key in header if key != "__metadata__"]


def _sd1_like_state_dict():
    """构造一个最小 SD1.x 形态的 state_dict（键名与真实 checkpoint 一致）。"""
    sd = {}
    for i in range(12):
        channels = 4 if i == 0 else 8
        sd[f"model.diffusion_model.input_blocks.{i}.0.weight"] = torch.zeros(
            (8, channels, 3, 3), dtype=torch.float16
        )
    sd["model.diffusion_model.middle_block.0.weight"] = torch.zeros(
        (8, 8, 3, 3), dtype=torch.float16
    )
    for i in range(12):
        sd[f"model.diffusion_model.output_blocks.{i}.0.weight"] = torch.zeros(
            (8, 8, 3, 3), dtype=torch.float16
        )
    sd["cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"] = (
        torch.zeros((8, 8), dtype=torch.float16)
    )
    sd["first_stage_model.decoder.conv_in.weight"] = torch.zeros(
        (8, 4, 3, 3), dtype=torch.float16
    )
    return sd


class GroupModelKeysTest(unittest.TestCase):
    def test_group_model_keys_clip_encoder_stays_in_clip(self):
        """CLIP 文本编码器键必须归 CLIP，不能被 VAE 的 encoder 规则抢走（B1 修复）。"""
        sd = _sd1_like_state_dict()
        groups = ModelUtils.group_model_keys(sd)

        self.assertIn("model.diffusion_model.input_blocks.0.0.weight", groups["UNET"])
        self.assertIn("first_stage_model.decoder.conv_in.weight", groups["VAE"])
        self.assertIn(
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            groups["CLIP"],
        )
        self.assertNotIn(
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            groups["VAE"],
        )

    def test_group_model_keys_other_bucket(self):
        groups = ModelUtils.group_model_keys({"unrelated.key": torch.zeros(1)})
        self.assertEqual(groups["Other"], ["unrelated.key"])

    def test_group_model_keys_extracted_vae_without_prefix(self):
        """提取后的 VAE 文件键以 encoder./decoder. 开头，仍应归 VAE。"""
        groups = ModelUtils.group_model_keys(
            {
                "encoder.conv_in.weight": torch.zeros(1),
                "decoder.conv_out.weight": torch.zeros(1),
            }
        )
        self.assertEqual(len(groups["VAE"]), 2)


class RealCheckpointGroupingTest(unittest.TestCase):
    """用真实 checkpoint 的键名（只读文件头，不加载权重）验证分组数量。"""

    MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"

    def test_group_counts(self):
        import folder_paths

        path = folder_paths.get_full_path("checkpoints", self.MODEL)
        self.assertTrue(path, "测试模型不存在")
        keys = _safetensors_keys(path)
        groups = ModelUtils.group_model_keys(dict.fromkeys(keys))
        self.assertEqual(len(groups["UNET"]), 686)
        self.assertEqual(len(groups["CLIP"]), 197)
        self.assertEqual(len(groups["VAE"]), 248)
        self.assertEqual(len(groups["EMA"]), 2)
        self.assertEqual(len(groups["Other"]), 0)


class FormattingTest(unittest.TestCase):
    def test_format_size(self):
        self.assertEqual(ModelUtils.format_size(500), "500.00 B")
        self.assertEqual(ModelUtils.format_size(1024), "1.00 KB")
        self.assertEqual(ModelUtils.format_size(1536), "1.50 KB")

    def test_format_params(self):
        self.assertEqual(ModelUtils.format_params(999_999), "1.00M")
        self.assertEqual(ModelUtils.format_params(1_000_000), "1.00M")
        self.assertEqual(ModelUtils.format_params(1_500_000_000), "1.50B")


class LoadSaveTest(unittest.TestCase):
    def test_load_model_missing_file_returns_none(self):
        self.assertIsNone(ModelUtils.load_model("Z:/definitely_missing.safetensors"))

    def test_load_model_text_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_a_model.txt"
            path.write_text("hello", encoding="utf-8")
            self.assertIsNone(ModelUtils.load_model(str(path)))

    def test_save_and_load_component_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "component.safetensors"
            saved = ModelUtils.save_component({"a": torch.ones(2)}, str(path))
            self.assertTrue(saved)
            loaded = ModelUtils.load_model(str(path))
            self.assertEqual(sorted(loaded), ["a"])
            self.assertEqual(tuple(loaded["a"].shape), (2,))


class AnalysisTest(unittest.TestCase):
    def test_analyze_architecture_sd1(self):
        sd = _sd1_like_state_dict()
        result, _debug = AnalysisMixin().analyze_architecture(sd)
        self.assertEqual(result["type"], "SD1.x")
        self.assertTrue(any("UNET input channels: 4" in info for info in result["extra_info"]))

    def test_analyze_model_precision(self):
        sd = {
            "a": torch.zeros(100, dtype=torch.float16),
            "b": torch.zeros(10, dtype=torch.float32),
        }
        stats = PrecisionMixin().analyze_model_precision(sd)
        self.assertEqual(stats["primary_precision"], "FP16 (Half precision)")
        self.assertEqual(
            list(stats["distributions"]),
            ["FP16 (Half precision)", "FP32 (Single precision)"],
        )

    def test_analyze_unet_structure(self):
        sd = _sd1_like_state_dict()
        unet_keys = [k for k in sd if "diffusion_model" in k]
        total = sum(t.numel() for t in sd.values())
        lines = AnalysisMixin().analyze_unet_structure(sd, unet_keys, total)
        text = "\n".join(lines)
        self.assertIn("- Input blocks: 12 (indices: 0-11)", text)
        self.assertIn("- Middle block: 1", text)
        self.assertIn("- Output blocks: 12 (indices: 0-11)", text)

    def test_analyze_vae_structure(self):
        sd = _sd1_like_state_dict()
        vae_keys = [k for k in sd if "first_stage_model" in k]
        total = sum(t.numel() for t in sd.values())
        lines = AnalysisMixin().analyze_vae_structure(sd, vae_keys, total)
        text = "\n".join(lines)
        self.assertIn("- Encoder keys: 0", text)
        self.assertIn("- Decoder keys: 1", text)


class ErrorPathTest(unittest.TestCase):
    def test_extractor_missing_model(self):
        (text,) = ComponentExtractor().extract_component(
            "no_such_model.safetensors", "UNET", "out"
        )
        self.assertEqual(text, "❌ Failed to load model file")

    def test_replacer_missing_model(self):
        (text,) = ComponentReplacer().replace_component(
            "no_such_model.safetensors", "UNET", "UNET-v1-SD.txt", "out"
        )
        self.assertEqual(text, "❌ Failed to load target model")


class MetadataTest(unittest.TestCase):
    """B4：safetensors 元信息必须从头部读取（不加载权重）。"""

    MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"

    def test_read_safetensors_metadata(self):
        import folder_paths

        path = folder_paths.get_full_path("checkpoints", self.MODEL)
        self.assertTrue(path, "测试模型不存在")
        metadata = ModelUtils.read_safetensors_metadata(path)
        self.assertEqual(metadata.get("format"), "pt")

    def test_non_safetensors_returns_empty(self):
        self.assertEqual(ModelUtils.read_safetensors_metadata("model.ckpt"), {})
        self.assertEqual(ModelUtils.read_safetensors_metadata(None), {})


class OldComfyUICompatTest(unittest.TestCase):
    """B5：旧版 ComfyUI（无 add_model_folder_path）不应在导入期崩溃。"""

    def test_import_without_add_model_folder_path(self):
        import os
        import subprocess
        import sys
        import tempfile

        import _env

        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "folder_paths.py")
            with open(stub, "w", encoding="utf-8") as f:
                f.write(
                    "def get_filename_list(name):\n"
                    "    return []\n"
                    "\n"
                    "output_directory = ''\n"
                )
            code = "import ComfyUI_model_toolkit as m; print('IMPORT_OK', m.__version__)"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([tmp, str(_env.REPO_ROOT.parent)])
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertIn("IMPORT_OK", proc.stdout)
            self.assertIn("Legacy ComfyUI detected", proc.stdout)


class NonTensorPayload:
    """用于验证反序列化安全的占位对象（非张量）。"""


class TorchLoadSafetyTest(unittest.TestCase):
    """T7：torch.load 必须显式 weights_only=True，拒绝非张量负载。"""

    def test_torch_state_dict_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"state_dict": {"a": torch.ones(2)}}, path)
            loaded = ModelUtils.load_model(str(path))
            self.assertEqual(sorted(loaded), ["a"])

    def test_rejects_non_tensor_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.pt"
            torch.save({"state_dict": {"a": NonTensorPayload()}}, path)
            self.assertIsNone(ModelUtils.load_model(str(path)))


if __name__ == "__main__":
    unittest.main()
