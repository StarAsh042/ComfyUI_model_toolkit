"""模型目录枚举与路径解析（模型检查器「模型目录」下拉）。"""
import os
import tempfile
import unittest

import _env  # noqa: F401  设置 sys.path

import folder_paths

from ComfyUI_model_toolkit.model_toolkit.folders import (
    ALL_DIRS,
    list_dir_files,
    list_model_dirs,
    resolve_model_path,
)

# 本机真实模型（与黄金样本同款）：位于 checkpoints 第二层 Mine/SD1/ 下
REAL_MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"


class ListModelDirsTest(unittest.TestCase):
    def test_all_dirs_is_first(self):
        """「（全部）」必须是第一项，作为下拉默认值。"""
        self.assertEqual(list_model_dirs()[0], ALL_DIRS)

    def test_contains_registered_folders(self):
        dirs = list_model_dirs()
        for name in ("checkpoints", "loras", "diffusion_models"):
            self.assertIn(name, dirs)

    def test_unregistered_folder_listed(self):
        """未注册目录（如 onnx）也在 models/ 下，应出现在列表里。"""
        root = getattr(folder_paths, "models_dir", None)
        if root and os.path.isdir(os.path.join(root, "onnx")):
            self.assertIn("onnx", list_model_dirs())

    def test_missing_models_dir_degrades(self):
        """models_dir 不可用时只返回「（全部）」，不炸。"""
        orig = folder_paths.models_dir
        folder_paths.models_dir = None
        try:
            self.assertEqual(list_model_dirs(), [ALL_DIRS])
        finally:
            folder_paths.models_dir = orig


class ListDirFilesTest(unittest.TestCase):
    def test_all_dirs_returns_empty(self):
        self.assertEqual(list_dir_files(ALL_DIRS), [])

    def test_checkpoints_returns_files(self):
        files = list_dir_files("checkpoints")
        self.assertIsInstance(files, list)
        self.assertTrue(files, "checkpoints 目录应有模型文件")

    def test_unregistered_dir_scanned_from_disk(self):
        """未注册目录（folder_paths 里没有）走磁盘递归扫描。"""
        orig = folder_paths.models_dir
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "models")
            os.makedirs(os.path.join(base, "ipadapter", "sub"))
            open(os.path.join(base, "ipadapter", "a.safetensors"), "w").close()
            open(os.path.join(base, "ipadapter", "sub", "b.ckpt"), "w").close()
            open(os.path.join(base, "ipadapter", "note.txt"), "w").close()
            folder_paths.models_dir = base
            try:
                files = list_dir_files("ipadapter")
            finally:
                folder_paths.models_dir = orig
        self.assertEqual(
            sorted(files), ["a.safetensors", os.path.join("sub", "b.ckpt")]
        )


class ResolveModelPathTest(unittest.TestCase):
    """用假的 get_full_path 精确控制每个目录里有什么，不依赖磁盘实况。"""

    def setUp(self):
        self._orig = folder_paths.get_full_path
        self._files = {
            ("loras", "shared.safetensors"): "L:/loras/shared.safetensors",
            ("checkpoints", "only_ckpt.safetensors"): "C:/ckpt/only_ckpt.safetensors",
        }
        folder_paths.get_full_path = lambda folder, name: self._files.get((folder, name))

    def tearDown(self):
        folder_paths.get_full_path = self._orig

    def test_specific_dir_hit(self):
        self.assertEqual(
            resolve_model_path("loras", "shared.safetensors"), "L:/loras/shared.safetensors"
        )

    def test_specific_dir_is_strict(self):
        """选了 loras 就只在 loras 里找：checkpoints 的同名文件不得被悄悄解析。"""
        self.assertIsNone(resolve_model_path("loras", "only_ckpt.safetensors"))

    def test_all_dirs_falls_back(self):
        """「（全部）」按目录枚举顺序回退，only_ckpt 最终在 checkpoints 命中。"""
        self.assertEqual(
            resolve_model_path(ALL_DIRS, "only_ckpt.safetensors"), "C:/ckpt/only_ckpt.safetensors"
        )

    def test_missing_returns_none(self):
        self.assertIsNone(resolve_model_path(ALL_DIRS, "no_such.safetensors"))
        self.assertIsNone(resolve_model_path("checkpoints", "no_such.safetensors"))
        self.assertIsNone(resolve_model_path("checkpoints", ""))


class RealModelResolutionTest(unittest.TestCase):
    """用本机真实模型验证解析链路（模型不存在时测试失败，与既有约定一致）。"""

    def test_all_dirs_resolves_real_model(self):
        path = resolve_model_path(ALL_DIRS, REAL_MODEL)
        self.assertTrue(path and os.path.isfile(path))

    def test_checkpoints_dir_resolves_real_model(self):
        path = resolve_model_path("checkpoints", REAL_MODEL)
        self.assertTrue(path and os.path.isfile(path))

    def test_wrong_dir_returns_none(self):
        self.assertIsNone(resolve_model_path("loras", REAL_MODEL))


if __name__ == "__main__":
    unittest.main()
