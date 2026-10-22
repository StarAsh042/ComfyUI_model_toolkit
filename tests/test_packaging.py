"""依赖声明约束：插件不得声明 torch（由 ComfyUI 宿主提供）。"""
import re
import unittest
from pathlib import Path

import _env  # noqa: F401  设置 sys.path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"

REQUIRED = {"numpy", "matplotlib", "pillow", "safetensors"}


def _declared_packages():
    names = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if match:
            names.append(match.group(1).lower())
    return names


class RequirementsTest(unittest.TestCase):
    def test_does_not_declare_torch(self):
        """torch 必须由 ComfyUI 提供：声明会导致 Manager 安装时重装/降级宿主 torch。"""
        self.assertNotIn("torch", _declared_packages())

    def test_required_runtime_dependencies_present(self):
        declared = set(_declared_packages())
        self.assertTrue(
            REQUIRED.issubset(declared),
            f"缺少运行期依赖: {sorted(REQUIRED - declared)}",
        )

    def test_every_line_is_pinned_with_lower_bound(self):
        for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            with self.subTest(line=line):
                self.assertRegex(line, r">=|==", "依赖应写明版本下限")


if __name__ == "__main__":
    unittest.main()
