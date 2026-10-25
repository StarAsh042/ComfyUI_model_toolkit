"""组件替换器行为测试：来源目录注册、按组件过滤、严格全量与 metadata。"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _env  # noqa: F401  设置 sys.path
import folder_paths
import safetensors.torch
import torch
from ComfyUI_model_toolkit import ModelUtils
from ComfyUI_model_toolkit.nodes.component_replacer import ComponentReplacer

UNET_KEY = "model.diffusion_model.input_blocks.0.0.weight"
UNET_EXTRA_KEY = "model.diffusion_model.input_blocks.1.0.weight"
VAE_KEY = "first_stage_model.decoder.conv_in.weight"
VAE_KEY_2 = "first_stage_model.decoder.conv_out.weight"
CLIP_KEY = "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight"


def _target():
    """最小可分类的目标模型：三个组件各一个键。"""
    return {
        UNET_KEY: torch.zeros(4, 4),
        VAE_KEY: torch.zeros(3, 3),
        CLIP_KEY: torch.zeros(2, 2),
    }


def _raw_safetensors_header(path):
    """读取 safetensors 原始头部 JSON（用于断言 __metadata__ 是否真实存在）。"""
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(header_size).decode("utf-8"))


def _metadata_header(path):
    return _raw_safetensors_header(path).get("__metadata__")


class ReplaceComponentKeysTest(unittest.TestCase):
    def test_only_selected_component_is_replaced(self):
        """B7/B8：component 必须真正生效，不得覆盖其它组件的同名键。"""
        target = _target()
        new_unet = torch.ones(4, 4)
        new_vae = torch.ones(3, 3)
        new_clip = torch.ones(2, 2)
        replacement = {UNET_KEY: new_unet, VAE_KEY: new_vae, CLIP_KEY: new_clip}

        stats = ModelUtils.replace_component_keys(target, replacement, "VAE")

        self.assertEqual(stats["replaced"], 1)
        self.assertIs(target[VAE_KEY], new_vae)
        self.assertIsNot(target[UNET_KEY], new_unet)
        self.assertIsNot(target[CLIP_KEY], new_clip)

    def test_component_without_replacement_keys_reports_zero(self):
        target = _target()
        stats = ModelUtils.replace_component_keys(target, {UNET_KEY: torch.ones(4, 4)}, "VAE")

        self.assertEqual(stats["replaced"], 0)
        self.assertEqual(stats["uncovered"], 1)
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(stats["target_keys"], 1)

    def test_shape_mismatch_is_skipped_not_written(self):
        target = _target()
        wrong = torch.ones(3, 4)
        stats = ModelUtils.replace_component_keys(target, {VAE_KEY: wrong}, "VAE")

        self.assertEqual(stats["replaced"], 0)
        self.assertEqual(stats["shape_mismatch"], 1)
        self.assertEqual(stats["uncovered"], 0)
        self.assertEqual(tuple(target[VAE_KEY].shape), (3, 3))

    def test_target_keys_are_partitioned_exactly_once(self):
        """replaced / shape_mismatch / uncovered 互斥且覆盖全部目标键。"""
        target = {
            UNET_KEY: torch.zeros(4, 4),
            VAE_KEY: torch.zeros(3, 3),
            VAE_KEY_2: torch.zeros(2, 2),
            "first_stage_model.encoder.conv_in.weight": torch.zeros(1, 1),
        }
        good = torch.ones(3, 3)
        wrong = torch.ones(2, 3)
        extra = torch.ones(9)
        replacement = {
            VAE_KEY: good,
            VAE_KEY_2: wrong,
            UNET_EXTRA_KEY: extra,
        }
        replacement_keys = set(replacement)

        stats = ModelUtils.replace_component_keys(target, replacement, "VAE")

        self.assertEqual(stats["target_keys"], 3)
        self.assertEqual(stats["replaced"], 1)
        self.assertEqual(stats["shape_mismatch"], 1)
        self.assertEqual(stats["uncovered"], 1)
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(
            stats["replaced"] + stats["shape_mismatch"] + stats["uncovered"],
            stats["target_keys"],
        )
        self.assertIs(target[VAE_KEY], good)
        self.assertTrue(torch.all(target[VAE_KEY_2] == 0), "形状不匹配的键不得写入")
        self.assertEqual(set(replacement), replacement_keys, "replacement_data 不得被修改")
        self.assertIs(replacement[UNET_EXTRA_KEY], extra)

    def test_missing_target_component(self):
        """目标模型没有该组件时返回 target_keys=0，而不是抛异常。"""
        replacement = {VAE_KEY: torch.ones(3, 3), UNET_EXTRA_KEY: torch.ones(4, 4)}
        stats = ModelUtils.replace_component_keys(
            {UNET_KEY: torch.zeros(4, 4)}, replacement, "VAE"
        )

        self.assertEqual(stats["target_keys"], 0)
        self.assertEqual(stats["replaced"], 0)
        self.assertEqual(stats["shape_mismatch"], 0)
        self.assertEqual(stats["uncovered"], 0)
        self.assertEqual(stats["missing"], len(replacement))
        self.assertEqual(sorted(replacement), sorted([VAE_KEY, UNET_EXTRA_KEY]))


class ExtractedFolderTest(unittest.TestCase):
    def test_extracted_folder_registered_with_model_extensions(self):
        """B2 根因：新建目录的扩展集为空会放行所有文件（含模板 .txt）。"""
        self.assertIn("model_toolkit_extracted", folder_paths.folder_names_and_paths)
        paths, exts = folder_paths.folder_names_and_paths["model_toolkit_extracted"]

        self.assertTrue(paths, "提取目录至少要有一个路径")
        self.assertEqual(set(exts), {".safetensors", ".sft", ".ckpt", ".pt"})


class ReplaceFlowTest(unittest.TestCase):
    """端到端：用微型 safetensors 跑完整个节点，验证写入内容。"""

    def test_end_to_end_replace_writes_only_target_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_path = os.path.join(tmp, "target.safetensors")
            repl_path = os.path.join(tmp, "vae.safetensors")
            safetensors.torch.save_file(_target(), target_path)
            safetensors.torch.save_file({VAE_KEY: torch.ones(3, 3)}, repl_path)

            original_get_full_path = folder_paths.get_full_path
            original_output_dir = folder_paths.output_directory
            folder_paths.get_full_path = lambda folder, name: {
                "checkpoints": target_path,
                "model_toolkit_extracted": repl_path,
            }.get(folder)
            folder_paths.output_directory = tmp
            try:
                (text,) = ComponentReplacer().replace_component(
                    "target.safetensors", "VAE", "vae.safetensors", "result"
                )
            finally:
                folder_paths.get_full_path = original_get_full_path
                folder_paths.output_directory = original_output_dir

            self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)

            saved = safetensors.torch.load_file(os.path.join(tmp, "result.safetensors"))
            self.assertTrue(torch.all(saved[VAE_KEY] == 1), "VAE 键应被替换")
            self.assertTrue(torch.all(saved[UNET_KEY] == 0), "UNET 键不应被改动")
            self.assertTrue(torch.all(saved[CLIP_KEY] == 0), "CLIP 键不应被改动")


class StrictReplacementFlowTest(unittest.TestCase):
    """默认严格全量：不完整替换必须在 save_component 之前拒绝。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.output_dir = self.root / "output"
        self.target_path = self.root / "target.safetensors"
        self.replacement_path = self.root / "replacement.safetensors"

    def _save_target(self, state, metadata=None, path=None):
        path = path or self.target_path
        safetensors.torch.save_file(state, str(path), metadata=metadata)
        return path

    def _save_replacement(self, state, metadata=None):
        safetensors.torch.save_file(
            state, str(self.replacement_path), metadata=metadata
        )

    def _save_sentinel(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "result.safetensors"
        safetensors.torch.save_file({"sentinel": torch.ones(1)}, str(path))
        return path

    def _run(self, component="VAE", target_path=None):
        resolved = {
            "checkpoints": str(target_path or self.target_path),
            "model_toolkit_extracted": str(self.replacement_path),
        }
        with (
            mock.patch.object(
                folder_paths,
                "get_full_path",
                side_effect=lambda folder, name: resolved.get(folder),
            ),
            mock.patch.object(
                folder_paths, "output_directory", str(self.output_dir)
            ),
        ):
            return ComponentReplacer().replace_component(
                "target.safetensors", component, "replacement.safetensors", "result"
            )[0]

    def _assert_sentinel_unchanged(self, path):
        self.assertEqual(
            sorted(safetensors.torch.load_file(str(path))), ["sentinel"]
        )

    def test_partial_replacement_is_rejected_before_save(self):
        state = _target()
        state[VAE_KEY_2] = torch.zeros(2, 2)
        self._save_target(state)
        self._save_replacement({VAE_KEY: torch.ones(3, 3)})
        sentinel = self._save_sentinel()

        with mock.patch.object(
            ModelUtils, "save_component", wraps=ModelUtils.save_component
        ) as save_mock:
            text = self._run()

        self.assertFalse(save_mock.called, "不完整替换不得调用 save_component")
        self.assertEqual(
            text,
            "❌ Incomplete VAE replacement (1/2 target keys replaced): "
            "0 shape mismatches, 1 target keys missing from replacement, "
            "0 replacement keys not in target VAE; output not written",
        )
        self._assert_sentinel_unchanged(sentinel)

    def test_shape_mismatch_is_rejected_before_save(self):
        self._save_target(_target())
        self._save_replacement({VAE_KEY: torch.ones(3, 4)})
        sentinel = self._save_sentinel()

        with mock.patch.object(
            ModelUtils, "save_component", wraps=ModelUtils.save_component
        ) as save_mock:
            text = self._run()

        self.assertFalse(save_mock.called)
        self.assertEqual(
            text,
            "❌ Incomplete VAE replacement (0/1 target keys replaced): "
            "1 shape mismatches, 0 target keys missing from replacement, "
            "0 replacement keys not in target VAE; output not written",
        )
        self._assert_sentinel_unchanged(sentinel)

    def test_target_without_component_keeps_old_error_and_sentinel(self):
        self._save_target({UNET_KEY: torch.zeros(4, 4)})
        self._save_replacement({VAE_KEY: torch.ones(3, 3)})
        sentinel = self._save_sentinel()

        with mock.patch.object(
            ModelUtils, "save_component", wraps=ModelUtils.save_component
        ) as save_mock:
            text = self._run()

        self.assertFalse(save_mock.called)
        self.assertEqual(text, "❌ Target model has no VAE component to replace")
        self._assert_sentinel_unchanged(sentinel)

    def test_rejection_does_not_create_output_directory(self):
        state = _target()
        state[VAE_KEY_2] = torch.zeros(2, 2)
        self._save_target(state)
        self._save_replacement({VAE_KEY: torch.ones(3, 3)})

        text = self._run()

        self.assertTrue(text.startswith("❌ Incomplete VAE replacement"), text)
        self.assertFalse(self.output_dir.exists(), "拒绝后不得创建保存目录")

    def test_full_coverage_with_extra_replacement_key_succeeds(self):
        self._save_target(_target())
        self._save_replacement(
            {VAE_KEY: torch.ones(3, 3), UNET_EXTRA_KEY: torch.ones(4, 4)}
        )

        text = self._run()

        expected_path = self.output_dir / "result.safetensors"
        self.assertEqual(
            text,
            "✅ Replaced 1/1 VAE keys, ignored 1 non-VAE keys, "
            f"saved to: {expected_path}",
        )
        self.assertNotIn("skipped", text)
        saved = safetensors.torch.load_file(str(expected_path))
        self.assertTrue(torch.all(saved[VAE_KEY] == 1))
        self.assertTrue(torch.all(saved[UNET_KEY] == 0))
        self.assertTrue(torch.all(saved[CLIP_KEY] == 0))
        self.assertNotIn(UNET_EXTRA_KEY, saved)
        self.assertNotIn("__metadata__", _raw_safetensors_header(expected_path))

    def test_target_metadata_kept_and_checksum_labels_dropped(self):
        target_metadata = {
            "format": "pt",
            "modelspec.architecture": "stable-diffusion-v1",
            "sha256": "target-digest",
            "File_Hash": "target-hash",
            "model-checksum": "target-checksum",
            "sha2": "target-sha2",
            "sha-2": "target-sha-2",
            "sha2-256": "target-sha2-256",
            "shasum": "target-shasum",
            "blake": "target-blake",
            "blake3": "target-blake3",
            "sha": "sha2-256-looking-value",
            "shape": "sha256-looking-value",
            "shard": "md5-value",
            "description": "sha256 is only a value",
        }
        self._save_target(_target(), metadata=target_metadata)
        self._save_replacement(
            {VAE_KEY: torch.ones(3, 3)},
            metadata={"source": "replacement", "sha256": "replacement-digest"},
        )

        text = self._run()

        self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
        output_path = self.output_dir / "result.safetensors"
        header = _metadata_header(output_path)
        self.assertEqual(
            header,
            {
                "format": "pt",
                "modelspec.architecture": "stable-diffusion-v1",
                "sha": "sha2-256-looking-value",
                "shape": "sha256-looking-value",
                "shard": "md5-value",
                "description": "sha256 is only a value",
            },
        )
        for key in (
            "sha256",
            "File_Hash",
            "model-checksum",
            "sha2",
            "sha-2",
            "sha2-256",
            "shasum",
            "blake",
            "blake3",
        ):
            self.assertNotIn(key, header, f"校验字段 {key} 不得写入输出 metadata")
        for key in ("sha", "shape", "shard"):
            self.assertIn(key, header, f"安全字段 {key} 必须保留")

    def test_target_without_metadata_or_with_digest_only_has_no_header(self):
        for label, metadata in (
            ("no_metadata", None),
            ("digest_only", {"sha256": "only-digest"}),
        ):
            with self.subTest(label=label):
                self.output_dir = self.root / f"output_{label}"
                self._save_target(_target(), metadata=metadata)
                self._save_replacement({VAE_KEY: torch.ones(3, 3)})

                text = self._run()

                self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
                output_path = self.output_dir / "result.safetensors"
                self.assertNotIn(
                    "__metadata__", _raw_safetensors_header(output_path)
                )

    def test_ckpt_target_writes_no_safetensors_metadata(self):
        ckpt_path = self.root / "target.ckpt"
        torch.save({"state_dict": _target()}, ckpt_path)
        self._save_replacement({VAE_KEY: torch.ones(3, 3)})

        text = self._run(target_path=ckpt_path)

        self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
        output_path = self.output_dir / "result.safetensors"
        self.assertNotIn("__metadata__", _raw_safetensors_header(output_path))

    def test_existing_output_metadata_is_not_merged(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "result.safetensors"
        safetensors.torch.save_file(
            {"stale": torch.ones(1)},
            str(output_path),
            metadata={"format": "old", "stale": "yes"},
        )
        self._save_target(_target())
        self._save_replacement({VAE_KEY: torch.ones(3, 3)})

        text = self._run()

        self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
        self.assertNotIn("__metadata__", _raw_safetensors_header(output_path))
        self.assertNotIn("stale", safetensors.torch.load_file(str(output_path)))


if __name__ == "__main__":
    unittest.main()
