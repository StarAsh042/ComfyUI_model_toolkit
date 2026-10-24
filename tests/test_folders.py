"""模型目录枚举与路径解析（模型检查器「模型目录」下拉）。"""
import os
import tempfile
import unittest

import _env  # noqa: F401  设置 sys.path
import folder_paths
from ComfyUI_model_toolkit.model_toolkit.folders import (
    DEFAULT_DIR,
    list_dir_files,
    list_model_dirs,
    resolve_model_path,
)

# 本机真实模型（与黄金样本同款）：位于 checkpoints 第二层 Mine/SD1/ 下
REAL_MODEL = r"Mine\SD1\Alpha_Anime_2.5D_fp16_v7.safetensors"


class ListModelDirsTest(unittest.TestCase):
    def test_checkpoints_is_first(self):
        """没有「全部」选项：checkpoints 固定第一，作为下拉默认值。"""
        dirs = list_model_dirs()
        self.assertEqual(dirs[0], DEFAULT_DIR)
        self.assertEqual(len(dirs), len(set(dirs)))

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
        """models_dir 不可用时只返回 checkpoints，不炸。"""
        orig = folder_paths.models_dir
        folder_paths.models_dir = None
        try:
            self.assertEqual(list_model_dirs(), [DEFAULT_DIR])
        finally:
            folder_paths.models_dir = orig


class ListDirFilesTest(unittest.TestCase):
    def test_empty_dir_name_returns_empty(self):
        self.assertEqual(list_dir_files(""), [])

    def test_checkpoints_returns_files(self):
        files = list_dir_files(DEFAULT_DIR)
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

    def test_missing_returns_none(self):
        self.assertIsNone(resolve_model_path("checkpoints", "no_such.safetensors"))
        self.assertIsNone(resolve_model_path("checkpoints", ""))
        self.assertIsNone(resolve_model_path("", "only_ckpt.safetensors"))

    def test_traversal_and_absolute_names_rejected(self):
        """工作流 JSON 里的模型名不可信：拒绝 .. 与绝对路径。"""
        for name in (
            "../loras/shared.safetensors",
            "..\\loras\\shared.safetensors",
            "C:/ckpt/only_ckpt.safetensors",
            "/etc/passwd",
        ):
            with self.subTest(name=name):
                self.assertIsNone(resolve_model_path("checkpoints", name))


@unittest.skipUnless(_env.has_real_model(REAL_MODEL), "需要本机真实模型")
class RealModelResolutionTest(unittest.TestCase):
    """用本机真实模型验证解析链路（模型不存在时跳过）。"""

    def test_default_dir_resolves_real_model(self):
        path = resolve_model_path(DEFAULT_DIR, REAL_MODEL)
        self.assertTrue(path and os.path.isfile(path))

    def test_wrong_dir_returns_none(self):
        self.assertIsNone(resolve_model_path("loras", REAL_MODEL))


if __name__ == "__main__":
    unittest.main()
