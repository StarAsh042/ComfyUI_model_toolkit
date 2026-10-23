"""P4 + T10：分析只读 safetensors 头部（不加载权重）、文件指纹驱动的缓存失效。"""
import os
import tempfile
import time
import unittest

import _env  # noqa: F401  设置 sys.path

import folder_paths
import safetensors.torch
import torch

from ComfyUI_model_toolkit import ModelUtils
from ComfyUI_model_toolkit.nodes.checkpoint_inspector import CheckpointInspector


def _tiny_state_dict():
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4, 4, dtype=torch.float16),
        "model.diffusion_model.middle_block.0.weight": torch.zeros(2, 2, dtype=torch.float16),
        "model.diffusion_model.output_blocks.0.0.weight": torch.zeros(3, 3, dtype=torch.float32),
        "first_stage_model.decoder.conv_in.weight": torch.zeros(3, 3, dtype=torch.float32),
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.zeros(2, 2),
    }


class HeaderIndexTest(unittest.TestCase):
    def test_index_matches_full_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.safetensors")
            state = _tiny_state_dict()
            safetensors.torch.save_file(state, path)

            full = ModelUtils.load_model(path)
            index = ModelUtils.load_model_index(path)

            self.assertEqual(list(index), list(full))
            for key in full:
                self.assertEqual(tuple(index[key].shape), tuple(full[key].shape))
                self.assertEqual(index[key].dtype, full[key].dtype)
                self.assertEqual(index[key].numel(), full[key].numel())
                self.assertEqual(index[key].element_size(), full[key].element_size())

    def test_index_excludes_metadata_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.safetensors")
            safetensors.torch.save_file(
                _tiny_state_dict(), path, metadata={"format": "pt"}
            )

            index = ModelUtils.load_model_index(path)

            self.assertNotIn("__metadata__", index)
            self.assertIn("first_stage_model.decoder.conv_in.weight", index)


class AnalyzeWithoutLoadingWeightsTest(unittest.TestCase):
    """P4 的核心断言：分析节点不允许再调用 load_model（全量加载）。"""

    def _patch(self, tmp, path):
        self._orig_get_full_path = folder_paths.get_full_path
        self._orig_load_model = ModelUtils.load_model
        folder_paths.get_full_path = lambda folder, name: path
        ModelUtils.load_model = staticmethod(
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("分析节点不得全量加载模型"))
        )

    def _restore(self):
        folder_paths.get_full_path = self._orig_get_full_path
        ModelUtils.load_model = self._orig_load_model

    def test_inspector_reads_header_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.safetensors")
            safetensors.torch.save_file(_tiny_state_dict(), path)
            self._patch(tmp, path)
            try:
                text, image = CheckpointInspector().analyze_model("tiny.safetensors", "基础", False)
            finally:
                self._restore()

            self.assertIn("📄 模型文件: tiny.safetensors", text)
            self.assertIn("模型架构", text)
            self.assertEqual(tuple(image.shape), (1, 1, 3))


class ModelFingerprintTest(unittest.TestCase):
    def test_fingerprint_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fp.safetensors")
            safetensors.torch.save_file({"a": torch.zeros(2)}, path)
            first = ModelUtils.model_fingerprint(path)

            time.sleep(0.01)
            safetensors.torch.save_file({"a": torch.zeros(3)}, path)
            second = ModelUtils.model_fingerprint(path)

            self.assertTrue(first)
            self.assertNotEqual(first, second)

    def test_fingerprint_stable_for_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fp.safetensors")
            safetensors.torch.save_file({"a": torch.zeros(2)}, path)

            self.assertEqual(ModelUtils.model_fingerprint(path), ModelUtils.model_fingerprint(path))

    def test_missing_file_returns_empty(self):
        self.assertEqual(ModelUtils.model_fingerprint(None), "")
        self.assertEqual(ModelUtils.model_fingerprint(""), "")
        self.assertEqual(ModelUtils.model_fingerprint("Z:/no/such/model.safetensors"), "")


class IsChangedTest(unittest.TestCase):
    def test_analysis_nodes_expose_is_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.safetensors")
            safetensors.torch.save_file(_tiny_state_dict(), path)

            original = folder_paths.get_full_path
            folder_paths.get_full_path = lambda folder, name: path
            try:
                first = CheckpointInspector.IS_CHANGED("tiny.safetensors", "基础", True)
                second = CheckpointInspector.IS_CHANGED("tiny.safetensors", "基础", True)
            finally:
                folder_paths.get_full_path = original

            self.assertEqual(first, second)
            self.assertTrue(first)


if __name__ == "__main__":
    unittest.main()
