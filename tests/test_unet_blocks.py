"""UNET 检查器块数量统计（B3 修复）。"""
import unittest

import _env  # noqa: F401  设置 sys.path

from ComfyUI_model_toolkit.nodes.unet_inspector import _count_blocks


class BlockCountTest(unittest.TestCase):
    def test_full_checkpoint_prefix(self):
        keys = [
            f"model.diffusion_model.input_blocks.{i}.0.weight" for i in range(12)
        ]
        self.assertEqual(_count_blocks(keys, ("input_blocks", "down_blocks")), 12)

    def test_middle_block(self):
        keys = [
            "model.diffusion_model.middle_block.0.weight",
            "model.diffusion_model.middle_block.1.weight",
        ]
        self.assertEqual(_count_blocks(keys, ("middle_block",)), 2)

    def test_extracted_unet_without_prefix(self):
        keys = [f"output_blocks.{i}.0.weight" for i in range(12)]
        self.assertEqual(_count_blocks(keys, ("output_blocks", "up_blocks")), 12)

    def test_empty(self):
        self.assertEqual(_count_blocks([], ("input_blocks",)), 0)


if __name__ == "__main__":
    unittest.main()
