"""UNet 结构提取：从权重键名还原阶段、块与注意力信息（P1）。

三层验证：

- 合成 LDM / diffusers 键名：锁定两种命名的解析规则；
- 结构不变量：每个 UNET 键恰好归入一个节点，参数与键数守恒；
- 本机真实 SD1 / SDXL：只读 safetensors 头部，锁定阶段数与上下文维度。
"""
import unittest

import _env  # noqa: F401  设置 sys.path
import folder_paths
import torch
from ComfyUI_model_toolkit.model_toolkit.core import ModelUtils, TensorMeta
from ComfyUI_model_toolkit.model_toolkit.structure import extract

PREFIX = "model.diffusion_model."

SD1_MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"
SDXL_MODEL = r"Mine\SDXL\Alpha_Anime_2D_XL_v1_fp16.safetensors"


def _meta(shape):
    return TensorMeta(shape, torch.float16)


def _add_ldm_resblock(sd, base, cin, cout):
    """LDM/CompVis ResBlock 的六类子权重。"""
    prefix = f"{PREFIX}{base}."
    sd[prefix + "in_layers.0.weight"] = _meta((cin,))
    sd[prefix + "in_layers.2.weight"] = _meta((cout, cin, 3, 3))
    sd[prefix + "emb_layers.1.weight"] = _meta((cout, 32))
    sd[prefix + "out_layers.0.weight"] = _meta((cout,))
    sd[prefix + "out_layers.3.weight"] = _meta((cout, cout, 3, 3))
    sd[prefix + "skip_connection.weight"] = _meta((cout, cin, 1, 1))


def _add_ldm_attention(sd, base, dim=16, context=12):
    prefix = f"{PREFIX}{base}."
    sd[prefix + "proj_in.weight"] = _meta((dim, dim, 1, 1))
    sd[prefix + "transformer_blocks.0.attn1.to_q.weight"] = _meta((dim, dim))
    sd[prefix + "transformer_blocks.0.attn1.to_k.weight"] = _meta((dim, dim))
    sd[prefix + "transformer_blocks.0.attn2.to_q.weight"] = _meta((dim, dim))
    sd[prefix + "transformer_blocks.0.attn2.to_k.weight"] = _meta((dim, context))
    sd[prefix + "transformer_blocks.0.ff.net.0.proj.weight"] = _meta((dim * 4, dim))
    sd[prefix + "proj_out.weight"] = _meta((dim, dim, 1, 1))


def _ldm_state_dict():
    """最小 LDM UNet：3 个 down 阶段、middle 3 块、3 个 up 阶段。"""
    sd = {}
    sd[PREFIX + "input_blocks.0.0.weight"] = _meta((8, 4, 3, 3))
    sd[PREFIX + "input_blocks.0.0.bias"] = _meta((8,))

    _add_ldm_resblock(sd, "input_blocks.1.0", 8, 8)
    sd[PREFIX + "input_blocks.3.0.op.weight"] = _meta((8, 8, 3, 3))

    _add_ldm_resblock(sd, "input_blocks.4.0", 8, 16)
    sd[PREFIX + "input_blocks.6.0.op.weight"] = _meta((16, 16, 3, 3))
    _add_ldm_attention(sd, "input_blocks.7.0")

    _add_ldm_resblock(sd, "middle_block.0", 16, 16)
    _add_ldm_attention(sd, "middle_block.1")
    _add_ldm_resblock(sd, "middle_block.2", 16, 16)

    _add_ldm_resblock(sd, "output_blocks.0.0", 16, 16)
    _add_ldm_resblock(sd, "output_blocks.1.0", 16, 16)
    sd[PREFIX + "output_blocks.2.1.conv.weight"] = _meta((16, 16, 3, 3))
    _add_ldm_resblock(sd, "output_blocks.3.0", 16, 8)
    _add_ldm_resblock(sd, "output_blocks.4.0", 8, 8)
    sd[PREFIX + "output_blocks.5.2.conv.weight"] = _meta((8, 8, 3, 3))
    _add_ldm_resblock(sd, "output_blocks.6.0", 8, 8)
    _add_ldm_resblock(sd, "output_blocks.7.0", 8, 8)
    _add_ldm_resblock(sd, "output_blocks.8.0", 8, 8)

    sd[PREFIX + "out.0.weight"] = _meta((8,))
    sd[PREFIX + "out.2.weight"] = _meta((4, 8, 3, 3))
    sd[PREFIX + "out.2.bias"] = _meta((4,))
    sd[PREFIX + "time_embed.0.weight"] = _meta((8, 32))
    sd[PREFIX + "time_embed.2.weight"] = _meta((32, 8))
    sd[PREFIX + "label_emb.0.0.weight"] = _meta((8, 8))

    # 未识别键：必须进入 unattributed，而不是被静默丢弃
    sd[PREFIX + "some_unknown.weight"] = _meta((3,))
    return sd


def _add_diffusers_resblock(sd, base, cin, cout):
    prefix = f"{PREFIX}{base}."
    sd[prefix + "norm1.weight"] = _meta((cin,))
    sd[prefix + "conv1.weight"] = _meta((cout, cin, 3, 3))
    sd[prefix + "time_emb_proj.weight"] = _meta((cout, 32))
    sd[prefix + "norm2.weight"] = _meta((cout,))
    sd[prefix + "conv2.weight"] = _meta((cout, cout, 3, 3))
    if cin != cout:
        sd[prefix + "conv_shortcut.weight"] = _meta((cout, cin, 1, 1))


def _add_diffusers_attention(sd, base, dim=16, context=12):
    prefix = f"{PREFIX}{base}.transformer_blocks.0."
    sd[prefix + "attn1.to_q.weight"] = _meta((dim, dim))
    sd[prefix + "attn2.to_q.weight"] = _meta((dim, dim))
    sd[prefix + "attn2.to_k.weight"] = _meta((dim, context))
    sd[prefix + "ff.net.0.proj.weight"] = _meta((dim * 4, dim))


def _diffusers_state_dict():
    sd = {}
    sd[PREFIX + "conv_in.weight"] = _meta((8, 4, 3, 3))
    sd[PREFIX + "time_embedding.linear_1.weight"] = _meta((32, 8))
    sd[PREFIX + "time_embedding.linear_2.weight"] = _meta((8, 32))

    _add_diffusers_resblock(sd, "down_blocks.0.resnets.0", 8, 8)
    sd[PREFIX + "down_blocks.0.downsamplers.0.conv.weight"] = _meta((8, 8, 3, 3))
    _add_diffusers_resblock(sd, "down_blocks.1.resnets.0", 8, 16)
    _add_diffusers_attention(sd, "down_blocks.1.attentions.0")
    sd[PREFIX + "down_blocks.1.downsamplers.0.conv.weight"] = _meta((16, 16, 3, 3))
    _add_diffusers_resblock(sd, "down_blocks.2.resnets.0", 16, 16)

    _add_diffusers_resblock(sd, "mid_block.resnets.0", 16, 16)
    _add_diffusers_attention(sd, "mid_block.attentions.0")

    _add_diffusers_resblock(sd, "up_blocks.0.resnets.0", 16, 16)
    sd[PREFIX + "up_blocks.0.upsamplers.0.conv.weight"] = _meta((16, 16, 3, 3))
    _add_diffusers_resblock(sd, "up_blocks.1.resnets.0", 16, 8)
    sd[PREFIX + "up_blocks.1.upsamplers.0.conv.weight"] = _meta((8, 8, 3, 3))
    _add_diffusers_resblock(sd, "up_blocks.2.resnets.0", 8, 8)

    sd[PREFIX + "conv_out.weight"] = _meta((4, 8, 3, 3))
    sd[PREFIX + "add_embedding.linear_1.weight"] = _meta((8, 8))
    sd[PREFIX + "add_embedding.linear_2.weight"] = _meta((8, 8))
    return sd


def _all_stages(result):
    return [*result["stages"], result["middle"], *result["up"]]


class InvariantMixin:
    def assert_structure_invariants(self, sd, result):
        """每个 UNET 键恰好归属一次：参数与键数守恒。"""
        totals = result["totals"]
        unet_keys = [key for key in sd if key.startswith(PREFIX)]

        self.assertEqual(totals["params"], sum(sd[key].numel() for key in unet_keys))
        self.assertEqual(totals["key_count"], len(unet_keys))

        io_params = sum(entry["params"] for entry in result["io"].values())
        io_keys = sum(entry["key_count"] for entry in result["io"].values())
        extras_params = sum(entry["params"] for entry in result["extras"])
        extras_keys = sum(entry["key_count"] for entry in result["extras"])
        stage_params = sum(stage["params"] for stage in _all_stages(result))
        stage_keys = sum(stage["key_count"] for stage in _all_stages(result))
        unattributed = result["unattributed"]

        self.assertEqual(
            totals["params"],
            io_params + extras_params + stage_params + unattributed["params"],
        )
        self.assertEqual(
            totals["key_count"],
            io_keys + extras_keys + stage_keys + unattributed["key_count"],
        )
        self.assertEqual(totals["attributed_params"], totals["params"] - unattributed["params"])
        self.assertEqual(
            totals["attributed_key_count"],
            totals["key_count"] - unattributed["key_count"],
        )
        self.assertEqual(totals["io_params"], io_params)
        self.assertEqual(totals["extras_params"], extras_params)
        self.assertEqual(totals["unattributed_params"], unattributed["params"])

        by_stage = [
            {"id": stage["id"], "direction": stage["direction"], "params": stage["params"]}
            for stage in _all_stages(result)
        ]
        self.assertEqual(totals["by_stage"], by_stage)

        for stage in _all_stages(result):
            expected = sum(block["params"] for block in stage["blocks"])
            expected_keys = sum(block["key_count"] for block in stage["blocks"])
            if stage["transition"] is not None:
                expected += stage["transition"]["params"]
                expected_keys += stage["transition"]["key_count"]
            self.assertEqual(stage["params"], expected)
            self.assertEqual(stage["key_count"], expected_keys)
            for block in stage["blocks"]:
                self.assertEqual(
                    block["params"], sum(child["params"] for child in block["children"])
                )
                self.assertEqual(
                    block["key_count"],
                    sum(child["key_count"] for child in block["children"]),
                )


class SyntheticLdmTest(InvariantMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sd = _ldm_state_dict()
        cls.groups = ModelUtils.group_model_keys(cls.sd)
        cls.result = extract(cls.sd, cls.groups, "SD1.x")

    def test_naming_prefix_and_kind(self):
        self.assertEqual(self.result["kind"], "unet")
        self.assertEqual(self.result["naming"], "ldm")
        self.assertEqual(self.result["prefix"], PREFIX)

    def test_down_middle_up_stages(self):
        self.assertEqual([stage["level"] for stage in self.result["stages"]], [0, 1, 2])
        self.assertEqual(
            [stage["transition"] is not None for stage in self.result["stages"]],
            [True, True, False],
        )
        self.assertEqual(
            [block["kind"] for block in self.result["middle"]["blocks"]],
            ["resblock", "transformer", "resblock"],
        )
        self.assertEqual([stage["level"] for stage in self.result["up"]], [2, 1, 0])
        self.assertEqual(
            [stage["transition"] is not None for stage in self.result["up"]],
            [True, True, False],
        )
        self.assertEqual(
            [stage["skip_from"] for stage in self.result["up"]],
            ["down.2", "down.1", "down.0"],
        )

    def test_transition_is_labelled_convention(self):
        downsample = self.result["stages"][0]["transition"]
        upsample = self.result["up"][0]["transition"]
        self.assertEqual(downsample["kind"], "downsample")
        self.assertEqual(downsample["stride"], 2)
        self.assertEqual(downsample["stride_source"], "convention")
        self.assertEqual(upsample["kind"], "upsample")
        self.assertEqual(upsample["factor"], 2)
        self.assertEqual(upsample["factor_source"], "convention")
        self.assertEqual(self.result["stages"][0]["resolution_ratio"], 1.0)
        self.assertEqual(self.result["up"][0]["resolution_ratio"], 0.25)

    def test_attention_heads_are_not_guessed(self):
        info = self.result["stages"][2]["blocks"][0]["attention"]
        self.assertTrue(info["self"])
        self.assertTrue(info["cross"])
        self.assertEqual(info["inner_dim"], 16)
        self.assertEqual(info["context_dim"], 12)
        self.assertEqual(info["transformer_blocks"], 1)
        self.assertIsNone(info["heads"])
        self.assertEqual(info["heads_source"], "unknown")

    def test_io_and_latent(self):
        conv_in = self.result["io"]["conv_in"]
        self.assertEqual((conv_in["in_channels"], conv_in["out_channels"]), (4, 8))
        conv_out = self.result["io"]["conv_out"]
        self.assertEqual((conv_out["in_channels"], conv_out["out_channels"]), (8, 4))
        self.assertEqual(self.result["latent"]["in_channels"], 4)
        self.assertEqual(self.result["latent"]["base_size"], 512)
        self.assertEqual(self.result["latent"]["latent_size"], 64)

    def test_extras_and_capabilities(self):
        self.assertEqual([extra["id"] for extra in self.result["extras"]], ["label_embed"])
        capabilities = self.result["capabilities"]
        self.assertTrue(capabilities["time_embedding"])
        self.assertTrue(capabilities["self_attention"])
        self.assertTrue(capabilities["text_cross_attention"])
        self.assertTrue(capabilities["size_conditioning"])
        self.assertIsNone(capabilities["dual_text_encoder"])
        self.assertEqual(capabilities["dual_text_encoder_source"], "unknown")
        self.assertFalse(capabilities["refiner"])

    def test_unknown_key_is_reported(self):
        self.assertEqual(self.result["unattributed"]["key_count"], 1)
        self.assertEqual(
            self.result["unattributed"]["keys"], [PREFIX + "some_unknown.weight"]
        )

    def test_every_key_attributed_once(self):
        self.assert_structure_invariants(self.sd, self.result)


class SyntheticDiffusersTest(InvariantMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sd = _diffusers_state_dict()
        cls.groups = ModelUtils.group_model_keys(cls.sd)
        cls.result = extract(cls.sd, cls.groups, "SDXL")

    def test_naming_is_diffusers(self):
        self.assertEqual(self.result["naming"], "diffusers")
        self.assertEqual(self.result["prefix"], PREFIX)

    def test_down_middle_up_stages(self):
        self.assertEqual([stage["level"] for stage in self.result["stages"]], [0, 1, 2])
        self.assertEqual(
            [stage["transition"] is not None for stage in self.result["stages"]],
            [True, True, False],
        )
        self.assertEqual(
            [stage["transition"] is not None for stage in self.result["up"]],
            [True, True, False],
        )
        self.assertEqual([stage["level"] for stage in self.result["up"]], [2, 1, 0])
        self.assertEqual(
            [stage["skip_from"] for stage in self.result["up"]],
            ["down.2", "down.1", "down.0"],
        )

    def test_attention_and_extras(self):
        info = self.result["stages"][1]["blocks"][1]["attention"]
        self.assertEqual(info["context_dim"], 12)
        self.assertIsNone(info["heads"])
        self.assertEqual([extra["kind"] for extra in self.result["extras"]], ["add_embedding"])
        self.assertTrue(self.result["capabilities"]["size_conditioning"])
        self.assertEqual(self.result["latent"]["base_size"], 1024)

    def test_every_key_attributed_once(self):
        self.assert_structure_invariants(self.sd, self.result)


class GenericFallbackTest(unittest.TestCase):
    def test_no_unet_falls_back_to_components(self):
        model_data = {
            "vae.decoder.conv_in.weight": _meta((8, 4, 3, 3)),
            "vae.encoder.conv_out.weight": _meta((8, 8, 3, 3)),
        }
        groups = ModelUtils.group_model_keys(model_data)
        result = extract(model_data, groups, "Flux")

        self.assertEqual(result["kind"], "generic")
        self.assertEqual([entry["name"] for entry in result["components"]], ["VAE"])
        self.assertEqual(result["totals"]["params"], 288 + 576)
        self.assertEqual(result["totals"]["unet_params"], 0)
        self.assertEqual(result["unattributed"], {"params": 0, "key_count": 0, "keys": []})
        self.assertIn("not available", result["note"])


@unittest.skipUnless(
    _env.has_real_model(SD1_MODEL) and _env.has_real_model(SDXL_MODEL),
    "需要本机 SD1/SDXL 真实模型",
)
class RealCheckpointStructureTest(InvariantMixin, unittest.TestCase):
    """真实模型只读头部；模型缺失时跳过（CI 无模型）。"""

    @classmethod
    def _extract(cls, name, arch_name):
        path = folder_paths.get_full_path("checkpoints", name)
        assert path, f"测试模型不存在: {name}"
        model_data = ModelUtils.load_model_index(path)
        groups = ModelUtils.group_model_keys(model_data)
        return model_data, groups, extract(model_data, groups, arch_name)

    def test_sd1_structure(self):
        model_data, groups, result = self._extract(SD1_MODEL, "SD1.x")

        self.assertEqual(result["naming"], "ldm")
        self.assertEqual(result["prefix"], PREFIX)
        self.assertEqual([stage["level"] for stage in result["stages"]], [0, 1, 2, 3])
        self.assertEqual([stage["level"] for stage in result["up"]], [3, 2, 1, 0])
        self.assertEqual(
            [stage["transition"] is not None for stage in result["stages"]],
            [True, True, True, False],
        )
        self.assertEqual(
            [stage["transition"] is not None for stage in result["up"]],
            [True, True, True, False],
        )
        self.assertEqual(len(result["middle"]["blocks"]), 3)

        info = result["middle"]["blocks"][1]["attention"]
        self.assertEqual(info["context_dim"], 768)
        self.assertIsNone(info["heads"])
        self.assertEqual(result["io"]["conv_in"].get("in_channels"), 4)
        self.assertFalse(result["capabilities"]["dual_text_encoder"])
        self.assertFalse(result["capabilities"]["size_conditioning"])

        self.assertEqual(result["totals"]["key_count"], 686)
        self.assertEqual(len(groups["UNET"]), 686)
        self.assertEqual(result["totals"]["unattributed_params"], 0)
        self.assert_structure_invariants(model_data, result)

    def test_sdxl_structure(self):
        model_data, groups, result = self._extract(SDXL_MODEL, "SDXL")

        self.assertEqual(result["naming"], "ldm")
        self.assertEqual([stage["level"] for stage in result["stages"]], [0, 1, 2])
        self.assertEqual([stage["level"] for stage in result["up"]], [2, 1, 0])
        self.assertEqual(
            [stage["transition"] is not None for stage in result["stages"]],
            [True, True, False],
        )
        self.assertEqual(
            [stage["transition"] is not None for stage in result["up"]],
            [True, True, False],
        )

        info = result["middle"]["blocks"][1]["attention"]
        self.assertEqual(info["context_dim"], 2048)
        self.assertIsNone(info["heads"])

        extras = {extra["id"]: extra for extra in result["extras"]}
        self.assertEqual(extras["label_embed"]["params"], 5_245_440)
        self.assertEqual(extras["label_embed"]["key_count"], 4)
        self.assertTrue(result["capabilities"]["size_conditioning"])
        self.assertTrue(result["capabilities"]["dual_text_encoder"])
        self.assertEqual(result["latent"]["base_size"], 1024)
        self.assertEqual(result["latent"]["latent_size"], 128)

        self.assertEqual(result["totals"]["key_count"], 1680)
        self.assert_structure_invariants(model_data, result)


if __name__ == "__main__":
    unittest.main()
