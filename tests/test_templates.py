"""架构模板（YAML）：加载、组件名适配、模板召回/额外键判定与分组接入。"""
import unittest

import _env  # noqa: F401  设置 sys.path
import torch
from ComfyUI_model_toolkit import ModelUtils
from ComfyUI_model_toolkit.model_toolkit import templates


def _synthetic_spec(identifier, key_count):
    """合成只有 UNET 的模板：key_count 个必需键，用于命中/召回阈值边界。"""
    components = {
        "UNET": {
            f"synth.{index}.weight": {"shape": (), "dtype": None, "optional": False}
            for index in range(key_count)
        }
    }
    return templates.ArchitectureSpec(identifier, components)


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


class CoverageSemanticsTest(unittest.TestCase):
    """coverage 是「必需键召回率」，extra_key_ratio 是「额外键比例」。"""

    def setUp(self):
        self.sd1 = templates.get_specs()["sd1-ldm"]

    def test_required_keys_is_union_without_ema(self):
        """R = 所有非 EMA 组件的 required_only 键并集；EMA 段 688 键不计入。"""
        self.assertEqual(len(self.sd1.components["EMA"]), 688)
        self.assertEqual(len(self.sd1.required_keys), 1130)
        # T 含 optional，R 不含：差集只有 1 个 optional 键
        self.assertEqual(len(self.sd1.trusted_keys() - self.sd1.required_keys), 1)

    def test_optional_key_alone_does_not_raise_recall(self):
        optional = self.sd1.trusted_keys() - self.sd1.required_keys

        self.assertEqual(self.sd1.required_hits(optional), 0)
        self.assertEqual(self.sd1.coverage(optional), 0.0)

    def test_recall_counts_required_hits_only(self):
        keys = set(self.sd1.required_keys)

        self.assertEqual(self.sd1.required_hits(keys), 1130)
        self.assertEqual(self.sd1.coverage(keys), 1.0)
        self.assertEqual(self.sd1.coverage(self.sd1.trusted_keys()), 1.0)

    def test_extra_ratio_excludes_ignored_keys(self):
        """S = M - 明确忽略键；只有真正未知的键进入 U。"""
        keys = set(self.sd1.trusted_keys()) | {"betas", "model_ema.decay", "unknown.key"}

        self.assertAlmostEqual(
            self.sd1.extra_key_ratio(keys), 1 / (len(self.sd1.trusted_keys()) + 1)
        )
        self.assertEqual(self.sd1.extra_key_ratio(self.sd1.trusted_keys()), 0.0)
        # S 为空（全是明确忽略键）时额外键比例为 0
        self.assertEqual(self.sd1.extra_key_ratio(["betas", "model_ema.decay"]), 0.0)


class IgnoredKeysTest(unittest.TestCase):
    """精确忽略：EMA、调度标量、嵌入/LoRA 前缀、等价裸 VAE 键。其余一律算额外键。"""

    def setUp(self):
        self.sd1 = templates.get_specs()["sd1-ldm"]

    def test_scheduler_keys_are_the_twelve_scalars(self):
        self.assertEqual(len(templates.IGNORED_KEYS), 12)
        self.assertEqual(
            templates.IGNORED_KEYS,
            {
                "alphas_cumprod",
                "alphas_cumprod_prev",
                "betas",
                "log_one_minus_alphas_cumprod",
                "posterior_log_variance_clipped",
                "posterior_mean_coef1",
                "posterior_mean_coef2",
                "posterior_variance",
                "sqrt_alphas_cumprod",
                "sqrt_one_minus_alphas_cumprod",
                "sqrt_recip_alphas_cumprod",
                "sqrt_recipm1_alphas_cumprod",
            },
        )

    def test_prefix_rules(self):
        self.assertEqual(
            set(templates.IGNORED_PREFIXES),
            {"model_ema.", "embedding_manager.", "lora_te_", "lora_unet_"},
        )
        for key in (
            "model_ema.decay",
            "model_ema.num_updates",
            "embedding_manager.emb_params",
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            "lora_unet_down_blocks_0_attn1_to_q.lora_up.weight",
        ):
            with self.subTest(key=key):
                self.assertTrue(self.sd1.ignored(key))

    def test_bare_vae_key_ignored_only_when_template_has_prefixed_equivalent(self):
        key = "decoder.conv_in.weight"
        self.assertIn(templates.VAE_KEY_PREFIX + key, self.sd1.keys("VAE"))

        self.assertTrue(self.sd1.ignored(key))
        self.assertFalse(self.sd1.ignored("encoder.made_up.weight"))

    def test_unknown_keys_are_never_ignored_fuzzily(self):
        for key in (
            "diffusion_model.made_up.weight",
            "first_stage_model.encoder.made_up.weight",
            "cond_stage_model.transformer.text_model.made_up.weight",
            "betas.extra",
        ):
            with self.subTest(key=key):
                self.assertFalse(self.sd1.ignored(key))


class RecallThresholdTest(unittest.TestCase):
    """合成 40 必需键模板：38 命中 + 2 未知恰过双 95%，再差一点就失败。"""

    def setUp(self):
        self.spec = _synthetic_spec("synthetic-40", 40)
        self.specs = {"synthetic-40": self.spec}
        self.required = sorted(self.spec.required_keys)

    def test_synthetic_spec_is_supported(self):
        self.assertTrue(self.spec.supported)
        self.assertEqual(len(self.spec.required_keys), 40)

    def test_boundary_passes_at_exactly_95_percent_each(self):
        keys = self.required[:38] + ["unknown.a", "unknown.b"]

        spec, recall = templates.detect(keys, specs=self.specs)

        self.assertIs(spec, self.spec)
        self.assertAlmostEqual(recall, 0.95)
        self.assertEqual(self.spec.required_hits(keys), 38)
        self.assertAlmostEqual(self.spec.extra_key_ratio(keys), 0.05)

    def test_one_missing_required_key_fails(self):
        keys = self.required[:37] + ["unknown.a", "unknown.b"]

        spec, recall = templates.detect(keys, specs=self.specs)

        self.assertIsNone(spec)
        self.assertAlmostEqual(recall, 37 / 40)

    def test_one_more_unknown_key_fails(self):
        keys = self.required[:38] + ["unknown.a", "unknown.b", "unknown.c"]

        spec, recall = templates.detect(keys, specs=self.specs)

        self.assertIsNone(spec)
        self.assertAlmostEqual(recall, 0.95)
        self.assertGreater(self.spec.extra_key_ratio(keys), 0.05)

    def test_hit_floor_is_32(self):
        self.assertEqual(templates.MIN_REQUIRED_HITS, 32)

    def test_31_of_32_hits_fails_despite_high_recall(self):
        spec = _synthetic_spec("synthetic-32", 32)
        keys = sorted(spec.required_keys)[:31]

        detected, recall = templates.detect(keys, specs={"synthetic-32": spec})

        self.assertIsNone(detected)
        self.assertAlmostEqual(recall, 31 / 32)

    def test_32_of_32_hits_passes_hit_floor(self):
        spec = _synthetic_spec("synthetic-32", 32)

        detected, recall = templates.detect(sorted(spec.required_keys), specs={"synthetic-32": spec})

        self.assertIs(detected, spec)
        self.assertEqual(recall, 1.0)

    def test_duplicate_keys_and_empty_input(self):
        keys = self.required[:38] + ["unknown.a", "unknown.b"]

        duplicate = templates.detect(keys + keys[:10], specs=self.specs)
        self.assertEqual(duplicate[0], self.spec)
        self.assertAlmostEqual(duplicate[1], 0.95)
        self.assertAlmostEqual(self.spec.extra_key_ratio(keys + keys[:10]), 0.05)

        self.assertEqual(templates.detect([], specs=self.specs), (None, 0.0))
        self.assertEqual(self.spec.coverage([]), 0.0)
        self.assertEqual(self.spec.required_hits([]), 0)
        self.assertEqual(self.spec.extra_key_ratio([]), 0.0)

    def test_spec_without_components_has_zero_recall(self):
        """R 为空时召回为 0，不会被当成万用模板。"""
        empty = templates.ArchitectureSpec("empty", {})

        self.assertEqual(empty.coverage(["a.b"]), 0.0)
        self.assertEqual(empty.required_hits(["a.b"]), 0)
        self.assertEqual(templates.detect(["a.b"], specs={"empty": empty}), (None, 0.0))


class DetectionBoundaryTest(unittest.TestCase):
    """真实模板形态：整模型可识别，小子集 / 仅 VAE / 未知键不误判。"""

    def setUp(self):
        self.specs = templates.get_specs()

    def test_full_sd1_with_noise_still_recognized(self):
        """完整 SD1 + EMA + 调度标量 + embedded LoRA + 嵌入管理键。"""
        sd1 = self.specs["sd1-ldm"]
        keys = set(sd1.trusted_keys())
        keys |= {"model_ema.decay", "model_ema.num_updates"}
        keys |= templates.IGNORED_KEYS
        keys |= {
            "embedding_manager.emb_params",
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            "lora_unet_down_blocks_0_attn1_to_q.lora_up.weight",
        }

        spec, recall = templates.detect(keys)

        self.assertEqual(spec.identifier, "sd1-ldm")
        self.assertEqual(recall, 1.0)
        self.assertEqual(sd1.extra_key_ratio(keys), 0.0)

    def test_bare_vae_duplicates_are_ignored(self):
        sd1 = self.specs["sd1-ldm"]
        bare = {
            key[len(templates.VAE_KEY_PREFIX):]
            for key in sd1.keys("VAE")
            if key.startswith(templates.VAE_KEY_PREFIX)
        }
        keys = set(sd1.trusted_keys()) | bare

        spec, recall = templates.detect(keys)

        self.assertEqual(spec.identifier, "sd1-ldm")
        self.assertEqual(recall, 1.0)
        self.assertEqual(sd1.extra_key_ratio(keys), 0.0)

    def test_full_sdxl_with_noise_still_recognized(self):
        sdxl = self.specs["sdxl-sgm"]
        keys = set(sdxl.trusted_keys()) | {"betas", "model_ema.decay"}

        spec, recall = templates.detect(keys)

        self.assertEqual(spec.identifier, "sdxl-sgm")
        self.assertEqual(recall, 1.0)
        self.assertEqual(sdxl.extra_key_ratio(keys), 0.0)

    def test_sd1_and_sdxl_do_not_cross_match(self):
        sd1 = self.specs["sd1-ldm"]
        sdxl = self.specs["sdxl-sgm"]

        sd1_hit, _ = templates.detect(sd1.trusted_keys())
        sdxl_hit, _ = templates.detect(sdxl.trusted_keys())

        self.assertEqual(sd1_hit.identifier, "sd1-ldm")
        self.assertEqual(sdxl_hit.identifier, "sdxl-sgm")

    def test_unet_only_template_still_recognized(self):
        unet_only = self.specs["sdxl-diffusers_unet_only"]

        spec, recall = templates.detect(unet_only.trusted_keys())

        self.assertEqual(spec.identifier, "sdxl-diffusers_unet_only")
        self.assertEqual(recall, 1.0)
        self.assertEqual(unet_only.extra_key_ratio(unet_only.trusted_keys()), 0.0)

    def test_missing_optional_keys_keep_full_recall(self):
        sdxl = self.specs["sdxl-sgm"]
        optional = {key for key, meta in sdxl.components["UNET"].items() if meta["optional"]}
        self.assertTrue(optional)
        keys = sdxl.trusted_keys() - optional

        spec, recall = templates.detect(keys)

        self.assertEqual(spec.identifier, "sdxl-sgm")
        self.assertEqual(recall, 1.0)
        self.assertEqual(sdxl.extra_key_ratio(keys), 0.0)

    def test_single_key_is_not_an_architecture(self):
        sd1 = self.specs["sd1-ldm"]

        spec, recall = templates.detect([sorted(sd1.required_keys)[0]])

        self.assertIsNone(spec)
        self.assertLess(recall, 0.95)

    def test_complete_vae_only_is_not_an_architecture(self):
        sd1 = self.specs["sd1-ldm"]

        spec, recall = templates.detect(sd1.keys("VAE"))

        self.assertIsNone(spec)
        self.assertLess(recall, 0.95)

    def test_sixty_unknown_keys_rejected(self):
        keys = [f"mystery.block{index}.weight" for index in range(60)]

        spec, recall = templates.detect(keys)

        self.assertIsNone(spec)
        self.assertEqual(recall, 0.0)
        self.assertEqual(self.specs["sd1-ldm"].extra_key_ratio(keys), 1.0)

    def test_unsupported_templates_never_recognized(self):
        for name in ("flux-flux", "sd3-comfyui"):
            with self.subTest(template=name):
                spec, recall = templates.detect(self.specs[name].trusted_keys())
                self.assertIsNone(spec)
                self.assertLess(recall, 0.95)

        flux = self.specs["flux-flux"]
        self.assertEqual(templates.detect(flux.trusted_keys(), specs={"flux-flux": flux}), (None, 0.0))

    def test_assignment_and_contract_unchanged(self):
        sd1 = self.specs["sd1-ldm"]
        keys = sorted(sd1.trusted_keys())

        mapping = sd1.assignment(keys)
        self.assertEqual(mapping[next(iter(sd1.keys("UNET")))], "UNET")
        self.assertEqual(sd1.contract("UNET", keys), (686, 686))
        self.assertEqual(len(mapping), len(sd1.trusted_keys()))


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
