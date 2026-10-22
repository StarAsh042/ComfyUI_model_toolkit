"""节点契约特征测试（G6 基线）。

只记录重构前的现状：节点名、显示名、CATEGORY、输入顺序 / 类型 / 默认值、
RETURN_TYPES / RETURN_NAMES / FUNCTION / OUTPUT_NODE / IS_CHANGED。
重构不得改变任何一项；改了就说明越界（见《代码重构_ComfyUI插件.md》第八节）。
"""
import unittest

import _env  # noqa: F401  设置 sys.path

import folder_paths

import ComfyUI_model_toolkit as pkg


def _describe_inputs(cls):
    """把 INPUT_TYPES 归一化成可比较的结构。"""
    described = []
    for section, params in cls.INPUT_TYPES().items():
        for key, value in params.items():
            spec = value[0] if isinstance(value, tuple) and value else value
            opts = {}
            if isinstance(value, tuple) and len(value) > 1 and isinstance(value[1], dict):
                opts = value[1]
            default = opts.get("default", None)
            multiline = opts.get("multiline", None)
            if isinstance(spec, (list, tuple)):
                if list(spec) == folder_paths.get_filename_list("checkpoints"):
                    kind, choices = "folder:checkpoints", None
                elif list(spec) == folder_paths.get_filename_list("components"):
                    kind, choices = "folder:components", None
                elif list(spec) == folder_paths.get_filename_list("model_toolkit_extracted"):
                    kind, choices = "folder:model_toolkit_extracted", None
                else:
                    kind, choices = "choices", list(spec)
            else:
                kind, choices = spec, None
            described.append((section, key, kind, choices, default, multiline))
    return described


EXPECTED = {
    "ModelInspector": {
        "display": "🔍 模型检查器",
        "category": "model_toolkit/分析",
        "return_types": ("STRING", "IMAGE"),
        "return_names": ("分析报告", "可视化图表"),
        "function": "analyze_model",
        "output_node": False,
        "has_is_changed": True,
        "inputs": [
            ("required", "模型", "folder:checkpoints", None, None, None),
            ("required", "显示细节", "choices", ["基础", "高级"], "基础", None),
            ("required", "显示图表", "BOOLEAN", None, True, None),
        ],
    },
    "ComponentExtractor": {
        "display": "📤 组件提取器",
        "category": "model_toolkit/修改",
        "return_types": ("STRING",),
        "return_names": ("提取结果",),
        "function": "extract_component",
        "output_node": False,
        "has_is_changed": False,
        "inputs": [
            ("required", "model_name", "folder:checkpoints", None, "", None),
            ("required", "component", "choices", ["UNET", "VAE", "CLIP"], "UNET", None),
            ("required", "output_name", "STRING", None, "extracted_component", False),
        ],
    },
    "ComponentReplacer": {
        "display": "🔄 组件替换器",
        "category": "model_toolkit/修改",
        "return_types": ("STRING",),
        "return_names": ("替换结果",),
        "function": "replace_component",
        "output_node": False,
        "has_is_changed": False,
        "inputs": [
            ("required", "model_name", "folder:checkpoints", None, "", None),
            ("required", "component", "choices", ["UNET", "VAE", "CLIP"], "UNET", None),
            ("required", "replacement_name", "folder:model_toolkit_extracted", None, "", None),
            ("required", "output_name", "STRING", None, "modified_model", False),
        ],
    },
    "UNetInspector": {
        "display": "🔬 UNET检查器",
        "category": "model_toolkit/分析",
        "return_types": ("STRING",),
        "return_names": ("分析结果",),
        "function": "inspect_unet",
        "output_node": False,
        "has_is_changed": True,
        "inputs": [
            ("required", "model_name", "folder:checkpoints", None, "", None),
            ("required", "show_details", "choices", ["简略", "详细"], "简略", None),
        ],
    },
}


class ContractTest(unittest.TestCase):
    def test_mapping_keys_and_order(self):
        self.assertEqual(
            list(pkg.NODE_CLASS_MAPPINGS),
            ["ModelInspector", "ComponentExtractor", "ComponentReplacer", "UNetInspector"],
        )
        self.assertEqual(
            list(pkg.NODE_DISPLAY_NAME_MAPPINGS),
            ["ModelInspector", "ComponentExtractor", "ComponentReplacer", "UNetInspector"],
        )

    def test_version(self):
        self.assertEqual(pkg.__version__, "1.0.0")

    def test_components_folder_contents(self):
        files = folder_paths.get_filename_list("components")
        self.assertEqual(len(files), 21)
        self.assertIn("UNET-v1-SD.txt", files)
        self.assertIn("VAE-v1-SD.txt", files)

    def test_node_contracts(self):
        for name, expected in EXPECTED.items():
            with self.subTest(node=name):
                cls = pkg.NODE_CLASS_MAPPINGS[name]
                self.assertEqual(pkg.NODE_DISPLAY_NAME_MAPPINGS[name], expected["display"])
                self.assertEqual(cls.CATEGORY, expected["category"])
                self.assertEqual(tuple(cls.RETURN_TYPES), expected["return_types"])
                self.assertEqual(tuple(cls.RETURN_NAMES), expected["return_names"])
                self.assertEqual(cls.FUNCTION, expected["function"])
                self.assertEqual(getattr(cls, "OUTPUT_NODE", False), expected["output_node"])
                self.assertEqual(hasattr(cls, "IS_CHANGED"), expected["has_is_changed"])
                self.assertEqual(_describe_inputs(cls), expected["inputs"])


if __name__ == "__main__":
    unittest.main()
