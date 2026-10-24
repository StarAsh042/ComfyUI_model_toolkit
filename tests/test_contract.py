"""节点契约特征测试（G6 基线）。

只记录重构前的现状：节点名、显示名、CATEGORY、输入顺序 / 类型 / 默认值、
RETURN_TYPES / RETURN_NAMES / FUNCTION / OUTPUT_NODE / IS_CHANGED。
重构不得改变任何一项；改了就说明越界（见《代码重构_ComfyUI插件.md》第八节）。

1.3.0 起的预期变更（见 CHANGELOG）：删除 UNetInspector；模型检查器
新增 model_dir 输入（顶部目录选择，默认 checkpoints）；界面与输出改为英文。
"""
import unittest
from pathlib import Path

import _env  # noqa: F401  设置 sys.path

import folder_paths

import ComfyUI_model_toolkit as pkg
from ComfyUI_model_toolkit.model_toolkit.folders import list_model_dirs


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
                elif list(spec) == folder_paths.get_filename_list("model_toolkit_extracted"):
                    kind, choices = "folder:model_toolkit_extracted", None
                elif list(spec) == list_model_dirs():
                    kind, choices = "folders:models", None
                else:
                    kind, choices = "choices", list(spec)
            else:
                kind, choices = spec, None
            described.append((section, key, kind, choices, default, multiline))
    return described


EXPECTED = {
    "ModelInspector": {
        "display": "🔍 Model Inspector",
        "category": "model_toolkit/Analysis",
        "return_types": ("STRING", "IMAGE"),
        "return_names": ("Analysis Report", "Visualization Chart"),
        "function": "analyze_model",
        "output_node": False,
        "has_is_changed": True,
        "inputs": [
            ("required", "model_dir", "folders:models", None, "checkpoints", None),
            ("required", "model_name", "folder:checkpoints", None, None, None),
            ("required", "show_details", "choices", ["Basic", "Advanced"], "Basic", None),
            ("required", "show_charts", "BOOLEAN", None, True, None),
        ],
    },
    "ComponentExtractor": {
        "display": "📤 Component Extractor",
        "category": "model_toolkit/Editing",
        "return_types": ("STRING",),
        "return_names": ("Extraction Result",),
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
        "display": "🔄 Component Replacer",
        "category": "model_toolkit/Editing",
        "return_types": ("STRING",),
        "return_names": ("Replacement Result",),
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
}


class ContractTest(unittest.TestCase):
    def test_mapping_keys_and_order(self):
        self.assertEqual(
            list(pkg.NODE_CLASS_MAPPINGS),
            ["ModelInspector", "ComponentExtractor", "ComponentReplacer"],
        )
        self.assertEqual(
            list(pkg.NODE_DISPLAY_NAME_MAPPINGS),
            ["ModelInspector", "ComponentExtractor", "ComponentReplacer"],
        )

    def test_version(self):
        self.assertEqual(pkg.__version__, "1.3.0")

    def test_architecture_templates_present(self):
        """模板已升级为 YAML：目录里不应再有旧的键清单 txt。"""
        templates_dir = Path(__file__).resolve().parent.parent / "components" / "architectures"
        self.assertTrue((templates_dir / "sd1-ldm.yaml").is_file())
        self.assertTrue((templates_dir / "sdxl-sgm.yaml").is_file())
        self.assertEqual(list(templates_dir.parent.glob("*.txt")), [])

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
