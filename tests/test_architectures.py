"""架构识别（家族级）与能力矩阵。"""
import unittest

import _env  # noqa: F401  设置 sys.path
from ComfyUI_model_toolkit.model_toolkit import architectures as arch


def _keys(*patterns):
    """按特征片段造键，模拟真实键名结构。"""
    return [f"model.diffusion_model.{p}" for p in patterns]


class IdentifyTest(unittest.TestCase):
    def test_sd_family_is_complete(self):
        result = arch.identify(_keys("input_blocks.0.0.weight"))

        self.assertEqual(result.id, "sd")
        self.assertEqual(result.capability, "complete")
        self.assertIsNone(arch.report_hint(result))

    def test_flux_is_partial(self):
        result = arch.identify(_keys("double_blocks.0.img_attn.norm.query_norm.scale"))

        self.assertEqual(result.id, "flux")
        self.assertEqual(result.family, "mmdit")
        self.assertEqual(result.capability, "partial")
        self.assertIn("T5", arch.report_hint(result))

    def test_sd3_is_partial(self):
        result = arch.identify(_keys("joint_blocks.0.context_block.attn.qkv.weight"))

        self.assertEqual(result.id, "sd3")
        self.assertIn("CLIP", arch.report_hint(result))

    def test_auraflow_and_hunyuan_and_pixart(self):
        cases = {
            "joint_transformer_blocks.0.attn.add_k_proj.weight": "auraflow",
            "mlp_t5.0.weight": "hunyuan_dit",
            "adaln_single.emb.timestep_embedder.linear_1.bias": "ltxv",  # 无 pos_embed → LTXV
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(arch.identify(_keys(key)).id, expected)

    def test_pixart_requires_pos_embed(self):
        keys = _keys(
            "adaln_single.emb.timestep_embedder.linear_1.bias",
            "pos_embed.proj.bias",
        )

        self.assertEqual(arch.identify(keys).id, "pixart")

    def test_cascade_and_svd(self):
        self.assertEqual(arch.identify(_keys("clf.1.weight")).id, "stable_cascade")
        self.assertEqual(arch.identify(_keys("1.time_stack.0.attn1.to_q.weight")).id, "svd")

    def test_audio_video_union_is_unsupported(self):
        result = arch.identify(_keys("video_patch_proj.weight", "audio_patch_proj.weight"))

        self.assertEqual(result.id, "minimax_h3")
        self.assertEqual(result.capability, "unsupported")
        self.assertTrue(arch.report_hint(result).startswith("⛔"))

    def test_unknown_returns_none(self):
        self.assertIsNone(arch.identify(["foo.bar", "baz.qux"]))
        self.assertIsNone(arch.report_hint(None))

    def test_metadata_hint_wins(self):
        result = arch.identify(
            _keys("input_blocks.0.0.weight"),
            metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"},
        )

        self.assertEqual(result.id, "sd")

    def test_verification_flags(self):
        """只有 SD 家族经过本机真实模型验证，其余标记为未验证。"""
        self.assertTrue(arch.SD.verified)
        self.assertFalse(arch.FLUX.verified)
        self.assertFalse(arch.SD3.verified)


if __name__ == "__main__":
    unittest.main()
