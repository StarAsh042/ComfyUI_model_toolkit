"""行为特征测试：锁定重构前的现有行为。

注意：其中 `test_group_model_keys_misclassifies_clip_encoder` 锁定的是一处
已知分类缺陷（CLIP 文本编码器键落进 VAE 组，见 CODE-REVIEW B1）。
重构必须保持行为不变；修 bug 时需同时更新本测试。
"""
import tempfile
import unittest
from pathlib import Path

import _env  # noqa: F401  设置 sys.path

import torch

from ComfyUI_model_toolkit import ModelUtils
from ComfyUI_model_toolkit.nodes.checkpoint_inspector import CheckpointInspector
from ComfyUI_model_toolkit.nodes.component_extractor import ComponentExtractor
from ComfyUI_model_toolkit.nodes.component_replacer import ComponentReplacer


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
    def test_group_model_keys_misclassifies_clip_encoder(self):
        """锁定现状（B1）：含 'encoder' 的 CLIP 键被归到 VAE，CLIP 只剩非 encoder 键。"""
        sd = _sd1_like_state_dict()
        groups = ModelUtils.group_model_keys(sd)

        self.assertIn("model.diffusion_model.input_blocks.0.0.weight", groups["UNET"])
        self.assertIn("first_stage_model.decoder.conv_in.weight", groups["VAE"])
        # B1 现状：CLIP 的 encoder 键落在 VAE 组
        self.assertIn(
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            groups["VAE"],
        )
        self.assertEqual(groups["CLIP"], [])

    def test_group_model_keys_other_bucket(self):
        groups = ModelUtils.group_model_keys({"unrelated.key": torch.zeros(1)})
        self.assertEqual(groups["其他"], ["unrelated.key"])


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
        result, _debug = CheckpointInspector().analyze_architecture(sd)
        self.assertEqual(result["type"], "SD1.x")
        self.assertTrue(any("UNET输入通道数: 4" in info for info in result["extra_info"]))

    def test_analyze_model_precision(self):
        sd = {
            "a": torch.zeros(100, dtype=torch.float16),
            "b": torch.zeros(10, dtype=torch.float32),
        }
        stats = CheckpointInspector().analyze_model_precision(sd)
        self.assertEqual(stats["primary_precision"], "FP16 (半精度)")
        self.assertEqual(
            list(stats["distributions"]), ["FP16 (半精度)", "FP32 (单精度)"]
        )

    def test_analyze_unet_structure(self):
        sd = _sd1_like_state_dict()
        unet_keys = [k for k in sd if "diffusion_model" in k]
        total = sum(t.numel() for t in sd.values())
        lines = CheckpointInspector().analyze_unet_structure(sd, unet_keys, total)
        text = "\n".join(lines)
        self.assertIn("- 输入块: 12 (索引: 0-11)", text)
        self.assertIn("- 中间块: 1", text)
        self.assertIn("- 输出块: 12 (索引: 0-11)", text)

    def test_analyze_vae_structure(self):
        sd = _sd1_like_state_dict()
        vae_keys = [k for k in sd if "first_stage_model" in k]
        total = sum(t.numel() for t in sd.values())
        lines = CheckpointInspector().analyze_vae_structure(sd, vae_keys, total)
        text = "\n".join(lines)
        self.assertIn("- 编码器键数量: 0", text)
        self.assertIn("- 解码器键数量: 1", text)


class ErrorPathTest(unittest.TestCase):
    def test_extractor_missing_model(self):
        (text,) = ComponentExtractor().extract_component(
            "no_such_model.safetensors", "UNET", "out"
        )
        self.assertEqual(text, "❌ 无法加载模型文件")

    def test_replacer_missing_model(self):
        (text,) = ComponentReplacer().replace_component(
            "no_such_model.safetensors", "UNET", "UNET-v1-SD.txt", "out"
        )
        self.assertEqual(text, "❌ 无法加载目标模型")


if __name__ == "__main__":
    unittest.main()
