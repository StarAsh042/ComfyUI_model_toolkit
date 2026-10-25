"""模型报告数据层契约（P1）：JSON 结构、缓存与错误码。

报告是交互页面的唯一数据源，页面只消费这里的 dict。测试锁定：

- JSON 可序列化且不含服务器绝对路径；
- 元信息按数据处理（HTML 转义是页面的事，不在这里改写原值）；
- 组件/UNET 参数守恒；
- LRU 缓存命中与 mtime 失效；
- 失败路径有稳定的错误码。
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import _env  # noqa: F401  设置 sys.path
import folder_paths
import torch
from ComfyUI_model_toolkit.model_toolkit import report
from safetensors.torch import save_file

REAL_MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"
METADATA = {
    "modelspec.architecture": "stable-diffusion-v1",
    "note": "<script>alert(1)</script>",
    "count": "3",
}


def _tiny_state_dict():
    def zeros(shape):
        return torch.zeros(shape, dtype=torch.float16)

    return {
        "model.diffusion_model.input_blocks.0.0.weight": zeros((8, 4, 3, 3)),
        "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": zeros((8, 8, 3, 3)),
        "model.diffusion_model.input_blocks.1.0.out_layers.3.weight": zeros((8, 8, 3, 3)),
        "model.diffusion_model.middle_block.0.in_layers.2.weight": zeros((8, 8, 3, 3)),
        "model.diffusion_model.middle_block.0.out_layers.3.weight": zeros((8, 8, 3, 3)),
        "model.diffusion_model.output_blocks.0.0.in_layers.2.weight": zeros((8, 8, 3, 3)),
        "model.diffusion_model.output_blocks.0.0.out_layers.3.weight": zeros((8, 8, 3, 3)),
        "model.diffusion_model.out.0.weight": zeros((8,)),
        "model.diffusion_model.out.2.weight": zeros((4, 8, 3, 3)),
        "model.diffusion_model.out.2.bias": zeros((4,)),
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": zeros(
            (8, 8)
        ),
        "first_stage_model.decoder.conv_in.weight": zeros((8, 4, 3, 3)),
    }


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


class SyntheticReportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.path = os.path.join(self.tmp, "tiny.safetensors")
        self.state = _tiny_state_dict()
        save_file(self.state, self.path, metadata=METADATA)

        self._orig_resolve = report.resolve_model_path
        report.resolve_model_path = lambda model_dir, model_name: self.path
        self.addCleanup(self._restore_resolve)
        report.clear_cache()
        self.addCleanup(report.clear_cache)

    def _restore_resolve(self):
        report.resolve_model_path = self._orig_resolve

    def _build(self, **kwargs):
        return report.build_report("checkpoints", "tiny.safetensors", **kwargs)

    def test_summary_and_schema(self):
        data = self._build()
        total = sum(tensor.numel() for tensor in self.state.values())

        self.assertEqual(data["schema_version"], report.SCHEMA_VERSION)
        self.assertEqual(data["summary"]["file_name"], "tiny.safetensors")
        self.assertEqual(data["summary"]["model_dir"], "checkpoints")
        self.assertEqual(data["summary"]["key_count"], len(self.state))
        self.assertEqual(data["summary"]["total_params"], total)
        # 12 键的桩模型远低于识别门槛（MIN_REQUIRED_HITS=32，召回需 ≥95%），
        # 按契约不得判为模板架构：类型回退到家族识别，template 为空。
        self.assertEqual(data["summary"]["architecture"], "SD family")
        self.assertEqual(data["summary"]["primary_precision"], "FP16 (Half precision)")
        self.assertEqual(data["architecture"]["type"], "SD family")
        self.assertIsNone(data["architecture"]["template"])

        self.assertEqual(list(data["components"]), ["UNET", "CLIP", "VAE"])
        self.assertEqual(
            sum(component["params"] for component in data["components"].values()), total
        )
        self.assertAlmostEqual(
            sum(component["percentage"] for component in data["components"].values()),
            100.0,
            places=1,
        )
        self.assertEqual(
            sum(component["key_count"] for component in data["components"].values()),
            len(self.state),
        )

    def test_structure_matches_component_totals(self):
        data = self._build()
        structure = data["structure"]

        self.assertEqual(structure["kind"], "unet")
        self.assertEqual(structure["naming"], "ldm")
        self.assertEqual(
            structure["totals"]["params"], data["components"]["UNET"]["params"]
        )
        self.assertEqual(
            structure["totals"]["key_count"], data["components"]["UNET"]["key_count"]
        )
        self.assertEqual(structure["unattributed"]["params"], 0)
        self.assertEqual(structure["unattributed"]["key_count"], 0)

    def test_key_entries_cover_every_key(self):
        data = self._build(include_keys=True)
        entries = {entry["name"]: entry for entry in data["keys"]}

        self.assertEqual(set(entries), set(self.state))
        self.assertEqual(
            entries["model.diffusion_model.input_blocks.0.0.weight"]["component"],
            "UNET",
        )
        self.assertEqual(entries["first_stage_model.decoder.conv_in.weight"]["component"], "VAE")
        self.assertEqual(
            entries[
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
            ]["component"],
            "CLIP",
        )
        self.assertEqual(
            entries["model.diffusion_model.out.2.weight"]["shape"], [4, 8, 3, 3]
        )

    def test_include_keys_false_removes_key_listings(self):
        data = self._build(include_keys=False)

        self.assertNotIn("keys", data)
        self.assertNotIn("keys", data["structure"]["unattributed"])
        self.assertEqual(
            data["structure"]["totals"], self._build()["structure"]["totals"]
        )

    def test_metadata_is_preserved_as_data(self):
        data = self._build()

        self.assertEqual(data["metadata"]["note"], METADATA["note"])
        self.assertEqual(data["metadata"]["count"], "3")
        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
        self.assertIn("<script>alert(1)</script>", text)

    def test_no_server_absolute_paths(self):
        data = self._build()
        text = json.dumps(data, ensure_ascii=False, allow_nan=False)

        self.assertNotIn(os.path.realpath(self.path), text)
        for value in _strings(data):
            self.assertNotIn(self.tmp, value)

    def test_json_roundtrip(self):
        data = self._build()
        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
        restored = json.loads(text)

        self.assertEqual(restored["summary"]["total_params"], data["summary"]["total_params"])
        self.assertEqual(restored["structure"]["totals"], data["structure"]["totals"])

    def test_cache_returns_same_object(self):
        first = self._build()
        second = self._build()

        self.assertIs(first, second)

        report.clear_cache()
        self.assertIsNot(self._build(), first)

    def test_cache_invalidated_when_file_changes(self):
        first = self._build()
        old_mtime = os.stat(self.path).st_mtime_ns

        changed = _tiny_state_dict()
        changed["extra.weight"] = torch.zeros(2, dtype=torch.float16)
        save_file(changed, self.path, metadata=METADATA)
        os.utime(self.path, ns=(old_mtime, old_mtime + 1_000_000))

        second = self._build()
        self.assertIsNot(first, second)
        self.assertEqual(second["summary"]["key_count"], len(changed))

    def test_include_keys_is_part_of_cache_key(self):
        with_keys = self._build(include_keys=True)
        without_keys = self._build(include_keys=False)

        self.assertIsNot(with_keys, without_keys)
        self.assertIn("keys", with_keys)
        self.assertNotIn("keys", without_keys)
        self.assertEqual(
            with_keys["summary"]["total_params"], without_keys["summary"]["total_params"]
        )

    def test_model_not_found(self):
        report.resolve_model_path = lambda model_dir, model_name: None
        with self.assertRaises(report.ReportError) as caught:
            self._build()
        self.assertEqual(caught.exception.code, "MODEL_NOT_FOUND")

    def test_load_failed(self):
        bad = os.path.join(self.tmp, "broken.safetensors")
        with open(bad, "wb") as f:
            f.write(b"this is not a safetensors header")
        report.resolve_model_path = lambda model_dir, model_name: bad

        with self.assertRaises(report.ReportError) as caught:
            self._build()
        self.assertEqual(caught.exception.code, "MODEL_LOAD_FAILED")

    def test_analysis_failure_is_wrapped(self):
        with mock.patch.object(report, "_assemble", side_effect=RuntimeError("boom")):
            with self.assertRaises(report.ReportError) as caught:
                self._build(include_keys=False)
        self.assertEqual(caught.exception.code, "ANALYSIS_FAILED")
        self.assertIn("boom", caught.exception.message)


@unittest.skipUnless(_env.has_real_model(REAL_MODEL), "需要本机真实模型")
class RealReportTest(unittest.TestCase):
    """真实模型：只读头部，验证数据层与结构提取的集成。"""

    @classmethod
    def setUpClass(cls):
        report.clear_cache()
        cls.data = report.build_report("checkpoints", REAL_MODEL)
        cls.path = folder_paths.get_full_path("checkpoints", REAL_MODEL)

    @classmethod
    def tearDownClass(cls):
        report.clear_cache()

    def test_real_summary(self):
        summary = self.data["summary"]
        self.assertEqual(summary["file_name"], REAL_MODEL)
        self.assertEqual(summary["model_dir"], "checkpoints")
        self.assertEqual(summary["key_count"], 1133)
        self.assertEqual(
            list(self.data["components"]), ["UNET", "CLIP", "VAE", "EMA"]
        )
        self.assertEqual(self.data["components"]["UNET"]["key_count"], 686)

    def test_real_architecture(self):
        architecture = self.data["architecture"]
        self.assertEqual(architecture["type"], "SD1.x")
        self.assertEqual(architecture["template"], "sd1-ldm")
        self.assertGreater(architecture["template_coverage"], 0.99)

    def test_real_structure(self):
        structure = self.data["structure"]
        self.assertEqual(structure["kind"], "unet")
        self.assertEqual(structure["naming"], "ldm")
        self.assertEqual(len(structure["stages"]), 4)
        self.assertEqual(len(structure["up"]), 4)
        self.assertEqual(
            structure["totals"]["params"], self.data["components"]["UNET"]["params"]
        )
        self.assertEqual(structure["unattributed"]["params"], 0)

    def test_real_report_has_no_absolute_paths(self):
        self.assertTrue(self.path)
        text = json.dumps(self.data, ensure_ascii=False, allow_nan=False)
        for value in _strings(self.data):
            self.assertNotIn(os.path.realpath(self.path), value)
            self.assertNotIn(os.path.dirname(os.path.realpath(self.path)), value)
        self.assertNotIn(os.path.dirname(self.path), text)

    def test_real_key_entries(self):
        self.assertEqual(len(self.data["keys"]), 1133)


if __name__ == "__main__":
    unittest.main()
