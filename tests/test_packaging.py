"""依赖、发布元数据与运行期第三方导入的约束。

运行期只允许导入 ``aiohttp`` / ``safetensors`` / ``torch``；``torch`` 由
ComfyUI 宿主提供，其余必须在 ``requirements.txt`` 声明。``pyyaml`` 只服务于
模板索引工具与测试，不是插件运行依赖。
"""
import ast
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import _env  # noqa: F401  设置 sys.path
import ComfyUI_model_toolkit as pkg

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
CI_REQUIREMENTS = REPO_ROOT / "tests" / "requirements-ci.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

REQUIRED = {"safetensors", "aiohttp"}

REMOVED_IN_2_0 = {"numpy", "matplotlib", "pillow"}

TOOLS_ONLY = {"pyyaml"}

HOST_MODULES = {"folder_paths", "server", "execution", "comfy", "comfy_extras", "utils"}

LOCAL_PACKAGES = {"model_toolkit", "nodes", "ComfyUI_model_toolkit"}

IMPORT_ALIASES = {"yaml": "pyyaml"}

THIRD_PARTY_ALLOWED = {"aiohttp", "safetensors", "torch"}

HOST_PROVIDED = {"torch"}


def _declared_packages(path=REQUIREMENTS):
    names = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if match:
            names.append(match.group(1).lower())
    return names


def _canonical(module_name):
    return IMPORT_ALIASES.get(module_name, module_name)


def _package_sources():
    sources = [REPO_ROOT / "__init__.py"]
    sources.extend(sorted((REPO_ROOT / "model_toolkit").rglob("*.py")))
    sources.extend(sorted((REPO_ROOT / "nodes").glob("*.py")))
    return sources


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

    def test_chart_dependencies_removed_in_2_0(self):
        """2.0.0 图表改由浏览器渲染，不再声明 matplotlib/Pillow/numpy。"""
        declared = set(_declared_packages())
        self.assertEqual(
            declared & REMOVED_IN_2_0,
            set(),
            f"2.0.0 不应再声明: {sorted(declared & REMOVED_IN_2_0)}",
        )

    def test_tooling_dependency_not_shipped(self):
        """pyyaml 只用于索引工具/测试，不得进入插件运行依赖。"""
        self.assertEqual(set(_declared_packages()) & TOOLS_ONLY, set())

    def test_every_line_is_pinned_with_lower_bound(self):
        for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            with self.subTest(line=line):
                self.assertRegex(line, r">=|==", "依赖应写明版本下限")

    def test_removing_aiohttp_line_is_detected(self):
        """在临时副本上删除 aiohttp 行，声明集合必须随之缺失（不污染仓库）。"""
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "requirements.txt"
            kept = [
                line
                for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
                if not line.strip().lower().startswith("aiohttp")
            ]
            copy.write_text("\n".join(kept) + "\n", encoding="utf-8")
            declared = set(_declared_packages(copy))
            self.assertNotIn("aiohttp", declared)
            with self.assertRaises(AssertionError):
                self.assertTrue(REQUIRED.issubset(declared))


class CiRequirementsTest(unittest.TestCase):
    def test_keeps_test_only_numpy(self):
        self.assertIn("numpy", _declared_packages(CI_REQUIREMENTS))

    def test_declares_pyyaml_for_tooling(self):
        self.assertIn("pyyaml", _declared_packages(CI_REQUIREMENTS))

    def test_aiohttp_is_not_a_test_dependency(self):
        declared = set(_declared_packages(CI_REQUIREMENTS))
        self.assertNotIn("aiohttp", declared)


class RuntimeImportClosureTest(unittest.TestCase):
    def test_scans_all_package_modules(self):
        self.assertEqual(len(_package_sources()), 19)

    def test_third_party_imports_match_declarations(self):
        third_party = set()
        for path in _package_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level or not node.module:
                        continue  # 相对导入属于本包
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    root = module.split(".", 1)[0]
                    if root in sys.stdlib_module_names:
                        continue
                    if root in HOST_MODULES or root in LOCAL_PACKAGES:
                        continue
                    third_party.add(_canonical(root))

        self.assertEqual(
            third_party,
            THIRD_PARTY_ALLOWED,
            f"运行期第三方导入集合发生变化: {sorted(third_party)}",
        )
        declared = set(_declared_packages())
        for name in sorted(third_party - HOST_PROVIDED):
            with self.subTest(package=name):
                self.assertIn(name, declared, f"{name} 第三方导入未在 requirements.txt 声明")


class PyprojectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    def test_version_matches_package(self):
        self.assertEqual(self.config["project"]["version"], pkg.__version__)

    def test_dependencies_follow_requirements(self):
        self.assertEqual(self.config["project"]["dynamic"], ["dependencies"])
        self.assertEqual(
            self.config["tool"]["setuptools"]["dynamic"]["dependencies"]["file"],
            ["requirements.txt"],
        )

    def test_registry_metadata_present(self):
        comfy = self.config["tool"]["comfy"]
        self.assertTrue(comfy["PublisherId"])
        self.assertTrue(comfy["DisplayName"])

    def test_dev_ruff_pinned_exactly(self):
        dev = self.config["project"]["optional-dependencies"]["dev"]
        ruff = [dep for dep in dev if dep.lower().startswith("ruff")]
        self.assertEqual(ruff, ["ruff==0.17.2"])

    def test_dev_declares_pyyaml(self):
        dev = self.config["project"]["optional-dependencies"]["dev"]
        self.assertIn("pyyaml>=5.4.0", dev)

    def test_build_system_uses_pep639_capable_setuptools(self):
        build = self.config["build-system"]
        self.assertEqual(build["build-backend"], "setuptools.build_meta")
        self.assertIn("setuptools>=77.0.1", build["requires"])
        self.assertIn("wheel", build["requires"])

    def test_license_expression_without_license_classifier(self):
        project = self.config["project"]
        self.assertEqual(project["license"], "GPL-3.0-only")
        self.assertEqual(project["requires-python"], ">=3.11")
        classifiers = project.get("classifiers", [])
        self.assertIn("Programming Language :: Python :: 3.11", classifiers)
        self.assertIn("Programming Language :: Python :: 3.12", classifiers)
        self.assertFalse(
            [item for item in classifiers if item.startswith("License ::")],
            "PEP 639 下不得再使用 License:: classifier",
        )


class RuffVersionTest(unittest.TestCase):
    def test_ci_ruff_pinned_exactly(self):
        text = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pip install ruff==0.17.2", text)
        self.assertNotRegex(text, r"(?m)pip install ruff\s*$")


if __name__ == "__main__":
    unittest.main()
