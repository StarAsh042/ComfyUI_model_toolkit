"""组件替换器行为测试：来源目录注册、按组件过滤、形状校验（B2 / B7 / B8）。"""
import unittest

import _env  # noqa: F401  设置 sys.path
import folder_paths
import torch
from ComfyUI_model_toolkit import ModelUtils

UNET_KEY = "model.diffusion_model.input_blocks.0.0.weight"
VAE_KEY = "first_stage_model.decoder.conv_in.weight"
CLIP_KEY = "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight"


def _target():
    """最小可分类的目标模型：三个组件各一个键。"""
    return {
        UNET_KEY: torch.zeros(4, 4),
        VAE_KEY: torch.zeros(3, 3),
        CLIP_KEY: torch.zeros(2, 2),
    }


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
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(stats["target_keys"], 1)

    def test_shape_mismatch_is_skipped_not_written(self):
        target = _target()
        wrong = torch.ones(3, 4)
        stats = ModelUtils.replace_component_keys(target, {VAE_KEY: wrong}, "VAE")

        self.assertEqual(stats["replaced"], 0)
        self.assertEqual(stats["shape_mismatch"], 1)
        self.assertEqual(tuple(target[VAE_KEY].shape), (3, 3))

    def test_missing_target_component(self):
        """目标模型没有该组件时返回 target_keys=0，而不是抛异常。"""
        stats = ModelUtils.replace_component_keys(
            {UNET_KEY: torch.zeros(4, 4)}, {VAE_KEY: torch.ones(3, 3)}, "VAE"
        )

        self.assertEqual(stats["target_keys"], 0)
        self.assertEqual(stats["replaced"], 0)


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
        import os
        import tempfile

        import safetensors.torch
        from ComfyUI_model_toolkit.nodes.component_replacer import ComponentReplacer

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


if __name__ == "__main__":
    unittest.main()
