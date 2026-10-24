"""架构模板（YAML）：加载、组件名适配、覆盖率判定与分组接入。"""
import unittest

import _env  # noqa: F401  设置 sys.path
import torch
from ComfyUI_model_toolkit import ModelUtils
from ComfyUI_model_toolkit.model_toolkit import templates


class LoadTest(unittest.TestCase):
    def test_all_upstream_configs_vendored(self):
        """上游 10 份架构配置全部纳入（singleton-mecha 是内部标记，不是架构）。"""
        specs = templates.get_specs()
        self.assertEqual(len(specs), 10)

    def test_specs_loaded(self):
        specs = templates.get_specs()
        self.assertIn("sd1-ldm", specs)
        self.assertIn("sdxl-sgm", specs)

    def test_kohya_and_variant_templates_present(self):
        specs = templates.get_specs()
        self.assertEqual(len(specs["sd1-kohya"].keys("UNET")), 686)
        self.assertEqual(len(specs["sdxl-kohya"].keys("UNET")), 1680)
        # 上游两份 kohya 变体的 identifier 相同，索引按文件名区分
        self.assertIn("sdxl-kohya_but_diffusers", specs)

    def test_templates_with_unrepresentable_components_are_not_supported(self):
        """Flux/SD3 含 T5XXL、supermerger 只有 BLOCKS：数据保留，但不参与自动识别。"""
        specs = templates.get_specs()
        for name in ("flux-flux", "sd3-comfyui", "sd1-supermerger_blocks", "sdxl-supermerger_blocks"):
            with self.subTest(template=name):
                self.assertFalse(specs[name].supported)

        for name in ("sd1-ldm", "sd1-kohya", "sdxl-sgm", "sdxl-kohya"):
            with self.subTest(template=name):
                self.assertTrue(specs[name].supported)

    def test_unsupported_template_never_wins_detection(self):
        flux = templates.get_specs()["flux-flux"]

        spec, _ = templates.detect(list(flux.trusted_keys()))

        self.assertNotEqual(getattr(spec, "identifier", None), "flux-flux")

    def test_sd1_component_alias(self):
        sd1 = templates.get_specs()["sd1-ldm"]
        self.assertEqual(len(sd1.keys("UNET")), 686)  # diffuser
        self.assertEqual(len(sd1.keys("VAE")), 248)  # vae
        self.assertEqual(len(sd1.keys("CLIP")), 197)  # clip_l

    def test_sdxl_dual_clip_merged(self):
        sdxl = templates.get_specs()["sdxl-sgm"]
        # clip_l 197 + clip_g 390 合并为本项目的 CLIP
        self.assertEqual(len(sdxl.keys("CLIP")), 587)

    def test_scheduler_component_ignored(self):
        for spec in templates.get_specs().values():
            self.assertNotIn("SCHEDULER", spec.components)

    def test_ema_segment_not_trusted(self):
        """上游 ema 段键名丢点（model_ema.diffusion_modelinput_blocks00bias），不可用于匹配。"""
        sd1 = templates.get_specs()["sd1-ldm"]
        self.assertEqual(len(sd1.components.get("EMA", {})), 688)
        self.assertEqual(sd1.keys("EMA"), set())
        self.assertNotIn(
            "model_ema.decay",
            sd1.trusted_keys(),
        )

    def test_optional_keys_excluded_from_contract(self):
        """上游标 optional 的键（如 v_pred / ztsnr）不参与契约核对。"""
        sdxl = templates.get_specs()["sdxl-sgm"]
        optional = {
            key
            for key, meta in sdxl.components["UNET"].items()
            if meta["optional"]
        }

        self.assertTrue(optional, "sd1/sdxl 模板应包含 optional 键")
        self.assertTrue(optional.isdisjoint(sdxl.keys("UNET", required_only=True)))

    def test_shape_and_dtype_adaptation(self):
        vae = templates.get_specs()["sd1-ldm"].components["VAE"]
        meta = vae["first_stage_model.decoder.conv_in.weight"]

        self.assertEqual(meta["shape"], (512, 4, 3, 3))
        self.assertIs(meta["dtype"], torch.float32)  # YAML 里写的是 "float32"


class DetectTest(unittest.TestCase):
    def test_detect_sd1_by_coverage(self):
        sd1 = templates.get_specs()["sd1-ldm"]
        keys = list(sd1.trusted_keys())

        spec, coverage = templates.detect(keys)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.identifier, "sd1-ldm")
        self.assertGreater(coverage, 0.99)

    def test_detect_sdxl_by_coverage(self):
        sdxl = templates.get_specs()["sdxl-sgm"]
        keys = list(sdxl.trusted_keys())

        spec, coverage = templates.detect(keys)

        self.assertEqual(spec.identifier, "sdxl-sgm")
        self.assertGreater(coverage, 0.99)

    def test_unknown_keys_fall_below_threshold(self):
        spec, coverage = templates.detect(["foo.bar", "baz.qux"])

        self.assertIsNone(spec)
        self.assertEqual(coverage, 0.0)


class GroupingTest(unittest.TestCase):
    def test_grouping_uses_template_for_known_architecture(self):
        sd1 = templates.get_specs()["sd1-ldm"]
        model_data = {key: None for key in sd1.trusted_keys()}

        groups = ModelUtils.group_model_keys(model_data)

        self.assertEqual(len(groups["UNET"]), 686)
        self.assertEqual(len(groups["VAE"]), 248)
        self.assertEqual(len(groups["CLIP"]), 197)
        self.assertEqual(groups["Other"], [])

    def test_ema_still_uses_prefix_rule(self):
        """模板的 ema 段不可信，EMA 必须继续由 model_ema.* 前缀识别。"""
        sd1 = templates.get_specs()["sd1-ldm"]
        model_data = {key: None for key in sd1.trusted_keys()}
        model_data["model_ema.decay"] = None
        model_data["model_ema.num_updates"] = None

        groups = ModelUtils.group_model_keys(model_data)

        self.assertEqual(sorted(groups["EMA"]), ["model_ema.decay", "model_ema.num_updates"])

    def test_unknown_model_falls_back_to_prefix_rules(self):
        model_data = {
            "diffusion_model.input_blocks.0.0.weight": None,
            "first_stage_model.decoder.conv_in.weight": None,
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight": None,
        }

        groups = ModelUtils.group_model_keys(model_data)

        self.assertEqual(len(groups["UNET"]), 1)
        self.assertEqual(len(groups["VAE"]), 1)
        self.assertEqual(len(groups["CLIP"]), 1)


if __name__ == "__main__":
    unittest.main()
