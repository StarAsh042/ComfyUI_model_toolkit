"""output_name 路径安全回归：两个组件节点都不得写出保存目录。

恶意输入必须在保存前被明确拒绝；合法默认名、合法显式名、扩展名与同名
覆盖语义保持不变。Windows 盘符 / UNC / 反斜杠规则在本机 Windows 实测。
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _env  # noqa: F401  设置 sys.path
import folder_paths
import safetensors.torch
import torch
from ComfyUI_model_toolkit.nodes.component_extractor import ComponentExtractor
from ComfyUI_model_toolkit.nodes.component_replacer import ComponentReplacer

UNET_KEY = "model.diffusion_model.input_blocks.0.0.weight"
VAE_KEY = "first_stage_model.decoder.conv_in.weight"
CLIP_KEY = "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight"

NODES = ("extractor", "replacer")


def _target():
    """最小可分类模型：三个组件各一个键。"""
    return {
        UNET_KEY: torch.zeros(4, 4),
        VAE_KEY: torch.zeros(3, 3),
        CLIP_KEY: torch.zeros(2, 2),
    }


class OutputNameSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.extracted_dir = self.root / "extracted"
        self.output_dir = self.root / "output"
        self.outside_dir = self.root / "outside"
        for path in (self.extracted_dir, self.output_dir, self.outside_dir):
            path.mkdir()

        self.target_path = self.root / "target.safetensors"
        self.replacement_path = self.root / "vae.safetensors"
        safetensors.torch.save_file(_target(), str(self.target_path))
        safetensors.torch.save_file({VAE_KEY: torch.ones(3, 3)}, str(self.replacement_path))

        self.baseline_tree = self._tree()

    def _patch_paths(self):
        """把节点用到的目录/文件解析固定到临时目录。"""
        resolved = {
            "checkpoints": str(self.target_path),
            "model_toolkit_extracted": str(self.replacement_path),
        }
        return (
            mock.patch.object(
                folder_paths,
                "get_full_path",
                side_effect=lambda folder, name: resolved.get(folder),
            ),
            mock.patch.object(
                folder_paths,
                "get_folder_paths",
                return_value=[str(self.extracted_dir)],
            ),
            mock.patch.object(folder_paths, "output_directory", str(self.output_dir)),
        )

    def _run(self, node, output_name, model_name="target.safetensors"):
        get_full_path, get_folder_paths, output_directory = self._patch_paths()
        with get_full_path, get_folder_paths, output_directory:
            if node == "extractor":
                text = ComponentExtractor().extract_component(model_name, "VAE", output_name)
            else:
                text = ComponentReplacer().replace_component(
                    model_name, "VAE", "vae.safetensors", output_name
                )
        return text[0]

    def _run_both(self, output_name):
        return {node: self._run(node, output_name) for node in NODES}

    def _save_dir(self, node):
        return self.extracted_dir if node == "extractor" else self.output_dir

    def _tree(self):
        """临时根下的全部条目（含空目录），用于断言没有目录外产物。"""
        return sorted(
            str(path.relative_to(self.root)).replace("\\", "/")
            for path in self.root.rglob("*")
        )

    def _assert_rejected(self, node, output_name):
        text = self._run(node, output_name)
        self.assertTrue(text.startswith("❌ Invalid output_name:"), text)
        self.assertEqual(self._tree(), self.baseline_tree, "拒绝后不得产生任何新文件/目录")

    def test_rejects_forward_slash_traversal(self):
        for node in NODES:
            with self.subTest(node=node):
                self._assert_rejected(node, "../evil.safetensors")

    def test_rejects_backslash_traversal(self):
        for node in NODES:
            with self.subTest(node=node):
                self._assert_rejected(node, r"..\evil.safetensors")

    def test_rejects_absolute_path(self):
        absolute = str(self.outside_dir / "evil.safetensors")
        for node, text in self._run_both(absolute).items():
            with self.subTest(node=node):
                self.assertTrue(text.startswith("❌ Invalid output_name:"), text)
                self.assertEqual(self._tree(), self.baseline_tree)
        self.assertFalse((self.outside_dir / "evil.safetensors").exists())

    @unittest.skipUnless(os.name == "nt", "UNC 是 Windows 路径规则")
    def test_rejects_unc_path(self):
        for node in NODES:
            with self.subTest(node=node):
                self._assert_rejected(node, r"\\server\share\evil.safetensors")

    @unittest.skipUnless(os.name == "nt", "盘符相对路径是 Windows 路径规则")
    def test_rejects_drive_relative_path(self):
        for node in NODES:
            with self.subTest(node=node):
                self._assert_rejected(node, r"C:evil.safetensors")

    def test_rejects_subdirectories(self):
        for node in NODES:
            for name in ("sub/evil.safetensors", r"sub\evil.safetensors"):
                with self.subTest(node=node, name=name):
                    self._assert_rejected(node, name)

    def test_rejects_directory_references(self):
        for node in NODES:
            for name in (".", ".."):
                with self.subTest(node=node, name=name):
                    self._assert_rejected(node, name)

    def test_accepts_default_names(self):
        text = self._run("extractor", "")
        self.assertTrue(text.startswith("✅ Saved VAE component"), text)
        extracted = safetensors.torch.load_file(
            str(self.extracted_dir / "target_vae.safetensors")
        )
        self.assertEqual(set(extracted), {VAE_KEY})

        text = self._run("replacer", "")
        self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
        replaced = safetensors.torch.load_file(
            str(self.output_dir / "target_replaced.safetensors")
        )
        self.assertEqual(set(replaced), {UNET_KEY, VAE_KEY, CLIP_KEY})

    def test_default_name_keeps_nested_model_layout(self):
        """默认命名沿用模型名（可含子目录），但仍落在保存目录内。"""
        nested_model = os.path.join("nested", "target.safetensors")

        text = self._run("extractor", "", model_name=nested_model)
        self.assertTrue(text.startswith("✅ Saved VAE component"), text)
        self.assertTrue((self.extracted_dir / "nested" / "target_vae.safetensors").is_file())

        text = self._run("replacer", "", model_name=nested_model)
        self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
        self.assertTrue((self.output_dir / "nested" / "target_replaced.safetensors").is_file())

    def test_default_name_still_rejects_escape_from_model_name(self):
        """默认名同样要过 realpath 包含性检查，模型名不能把它带出保存目录。"""
        escaping_model = os.path.join("..", "target.safetensors")
        for node in NODES:
            with self.subTest(node=node):
                text = self._run(node, "", model_name=escaping_model)
                self.assertTrue(text.startswith("❌ Invalid output_name:"), text)
                self.assertEqual(self._tree(), self.baseline_tree)

    def test_accepts_explicit_names_and_keeps_extension_rule(self):
        text = self._run("extractor", "chosen.safetensors")
        self.assertTrue(text.startswith("✅ Saved VAE component"), text)
        self.assertTrue((self.extracted_dir / "chosen.safetensors").is_file())
        self.assertFalse((self.extracted_dir / "chosen.safetensors.safetensors").exists())

        text = self._run("replacer", "chosen")
        self.assertTrue(text.startswith("✅ Replaced 1/1 VAE keys"), text)
        self.assertTrue((self.output_dir / "chosen.safetensors").is_file())
        self.assertFalse((self.output_dir / "chosen.safetensors.safetensors").exists())

    def test_existing_output_file_is_overwritten(self):
        stale = {"stale": torch.ones(1)}
        extract_path = self.extracted_dir / "result.safetensors"
        replace_path = self.output_dir / "result.safetensors"
        safetensors.torch.save_file(stale, str(extract_path))
        safetensors.torch.save_file(stale, str(replace_path))

        self._run("extractor", "result")
        self._run("replacer", "result")

        extracted = safetensors.torch.load_file(str(extract_path))
        replaced = safetensors.torch.load_file(str(replace_path))
        self.assertNotIn("stale", extracted)
        self.assertNotIn("stale", replaced)

    def test_rejects_symlink_resolving_outside_save_dir(self):
        outside_file = self.outside_dir / "outside.safetensors"
        safetensors.torch.save_file({"secret": torch.ones(1)}, str(outside_file))

        for node in NODES:
            link = self._save_dir(node) / f"{node}_link.safetensors"
            try:
                os.symlink(outside_file, link)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"当前环境无法创建符号链接: {e}")
            with self.subTest(node=node):
                before = self._tree()
                text = self._run(node, f"{node}_link")
                self.assertTrue(text.startswith("❌ Invalid output_name:"), text)
                self.assertEqual(self._tree(), before, "拒绝后不得改动任何条目")

        saved = safetensors.torch.load_file(str(outside_file))
        self.assertIn("secret", saved, "目录外文件不得被覆盖")


if __name__ == "__main__":
    unittest.main()
