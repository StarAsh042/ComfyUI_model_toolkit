"""发布一致性守卫的静态契约与本地索引工具行为（不访问网络）。

断言对象：
- ``publish.yml``：触发条件/权限不变、workflow 级串行、只允许 main 发布、
  版本守卫位于发布 action 之前且不读 secret / 不以 tag 为基线；
  守卫先校验节点存在再读版本，404/3xx/网络/JSON 错误 fail-closed
  （用 stub urlopen 执行守卫脚本，不访问网络）；
- ``ci.yml``：模板索引同步检查与 Ruff 精确版本；
- ``tools/build_template_index.py``：``--check`` 只读、ASCII 令牌、跨代码页不崩溃。
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH = REPO_ROOT / ".github" / "workflows" / "publish.yml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TOOL = REPO_ROOT / "tools" / "build_template_index.py"
INDEX = REPO_ROOT / "components" / "architectures" / "index.json"

GUARD_STEP = "Assert version changed since last publish"


def _load(path):
    """PyYAML 会把 on 解析成布尔 True，调用方用 _triggers 统一取值。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow):
    return workflow.get("on", workflow.get(True))


def _steps(job):
    return [step.get("name") for step in job.get("steps", [])]


def _step(job, name):
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"缺少步骤: {name}")


def _heredoc_python(run):
    """取出 ``run`` 中 ``<<'PY'`` 与终止行之间的 Python 源码。"""
    lines = run.splitlines()
    begin = next(i for i, line in enumerate(lines) if "<<'PY'" in line) + 1
    end = next(i for i, line in enumerate(lines) if i > begin and line.strip() == "PY")
    return "\n".join(lines[begin:end])


class PublishWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = _load(PUBLISH)
        cls.text = PUBLISH.read_text(encoding="utf-8")
        cls.job = cls.workflow["jobs"]["publish-node"]
        cls.guard = _step(cls.job, GUARD_STEP)["run"]

    def test_triggers_unchanged(self):
        triggers = _triggers(self.workflow)
        self.assertIsNotNone(triggers, "on 触发器缺失（注意 YAML 的 on 键）")
        self.assertEqual(set(triggers), {"workflow_dispatch", "push"})
        self.assertEqual(triggers["push"]["branches"], ["main"])
        self.assertEqual(triggers["push"]["paths"], ["pyproject.toml"])

    def test_permissions_unchanged(self):
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})

    def test_workflow_concurrency_serialises_without_cancel(self):
        concurrency = self.workflow.get("concurrency")
        self.assertIsNotNone(concurrency, "缺少 workflow 级 concurrency")
        self.assertEqual(
            concurrency["group"], "${{ github.workflow }}-${{ github.ref }}"
        )
        self.assertIs(concurrency["cancel-in-progress"], False)

    def test_publish_only_on_main_for_owner(self):
        condition = self.job["if"]
        self.assertIn("StarAsh042", condition)
        self.assertIn("github.ref == 'refs/heads/main'", condition)
        self.assertNotIn("refs/tags", condition)

    def test_guard_runs_after_version_check_before_publish(self):
        names = _steps(self.job)
        self.assertEqual(names.count(GUARD_STEP), 1, "守卫步骤名必须唯一")
        self.assertGreater(names.index(GUARD_STEP), names.index("Verify version consistency"))
        self.assertLess(names.index(GUARD_STEP), names.index("Publish Custom Node"))

    def test_guard_uses_repository_tail_for_url(self):
        self.assertIn("${GITHUB_REPOSITORY#*/}", self.guard)
        self.assertIn("https://api.comfy.org/nodes/", self.guard)
        self.assertIn("{node_id}/versions", self.guard)
        self.assertNotIn("ComfyUI_model_toolkit/versions", self.guard)

    def test_guard_reads_pyproject_version(self):
        self.assertIn("pyproject.toml", self.guard)
        self.assertIn('["project"]["version"]', self.guard)

    def test_guard_fetches_with_stdlib_timeout_and_redirect(self):
        self.assertIn("urllib.request.urlopen", self.guard)
        self.assertIn("timeout=20", self.guard)
        self.assertIn("def triple(", self.guard)
        self.assertIn("max(parsed)", self.guard)
        self.assertIn('entry["version"]', self.guard)

    def test_guard_exit_codes_and_tokens(self):
        for token in (
            "VERSION_UNCHANGED",
            "NO_BASELINE_FIRST_PUBLISH",
            "BASELINE_UNKNOWN_NODE",
            "REGISTRY_UNREACHABLE",
            "VERSION_NOT_SEMVER",
            "VERSION_BELOW_PUBLISHED",
            "VERSION_AHEAD",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.guard)

        self.assertRegex(
            self.guard, r'if current in published:[\s\S]{0,160}VERSION_UNCHANGED[\s\S]{0,80}, 1\)'
        )
        self.assertRegex(
            self.guard,
            r'if not entries:[\s\S]{0,160}NO_BASELINE_FIRST_PUBLISH[\s\S]{0,80}, 0\)',
        )
        self.assertRegex(
            self.guard, r"404[\s\S]{0,160}BASELINE_UNKNOWN_NODE[\s\S]{0,80}, 1\)"
        )
        self.assertRegex(
            self.guard, r'SEMVER = re\.compile\(r"\^\\d\+\\\.\\d\+\\\.\\d\+\$"\)'
        )
        self.assertRegex(
            self.guard, r"current_triple < latest:[\s\S]{0,200}VERSION_BELOW_PUBLISHED"
        )
        self.assertRegex(
            self.guard, r"VERSION_AHEAD[\s\S]{0,200}0,"
        )
        self.assertGreaterEqual(self.guard.count("REGISTRY_UNREACHABLE"), 4)

    def test_guard_has_no_secret_or_write_side_effects(self):
        self.assertNotIn("secrets.", self.guard)
        self.assertNotIn("REGISTRY_ACCESS_TOKEN", self.guard)
        self.assertNotIn("GITHUB_REF", self.guard)
        for command in ("git push", "git tag", "git commit", "gh release"):
            with self.subTest(command=command):
                self.assertNotIn(command, self.text)


class _StubResponse:
    """最小 urlopen 响应替身：支持 ``with`` 与 read()。"""

    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _UnexpectedRequest(BaseException):
    """守卫发出未编排的请求时冒泡，不被守卫的 ``except Exception`` 吞掉。"""


class ReleaseGuardBehaviorTest(unittest.TestCase):
    """执行 publish.yml 中的守卫脚本，用 stub urlopen 覆盖 fail-closed 路径。"""

    NODE = "ComfyUI_model_toolkit"
    METADATA_URL = f"https://api.comfy.org/nodes/{NODE}"
    VERSIONS_URL = f"{METADATA_URL}/versions"

    @classmethod
    def setUpClass(cls):
        job = _load(PUBLISH)["jobs"]["publish-node"]
        cls.guard = _step(job, GUARD_STEP)["run"]
        cls.namespace = {"__name__": "release_guard_under_test"}
        exec(compile(_heredoc_python(cls.guard), "<release-guard>", "exec"), cls.namespace)

    @staticmethod
    def _metadata():
        return _StubResponse(
            200, json.dumps({"id": ReleaseGuardBehaviorTest.NODE}).encode("utf-8")
        )

    @staticmethod
    def _versions(*versions):
        payload = {"versions": [{"version": version} for version in versions]}
        return _StubResponse(200, json.dumps(payload).encode("utf-8"))

    def _run_guard(self, node_id, *responses, version="2.0.0"):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append((request.full_url, timeout))
            if len(calls) > len(responses):
                raise _UnexpectedRequest(request.full_url)
            outcome = responses[len(calls) - 1]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        opened = urllib.request._opener
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "pyproject.toml").write_text(
                f'[project]\nversion = "{version}"\n', encoding="utf-8"
            )
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.object(sys, "argv", ["guard", node_id]), \
                        mock.patch.object(urllib.request, "urlopen", fake_urlopen), \
                        contextlib.redirect_stdout(io.StringIO()) as stdout:
                    with self.assertRaises(SystemExit) as raised:
                        self.namespace["main"]()
            finally:
                os.chdir(previous_cwd)
                urllib.request._opener = opened
        return raised.exception.code, stdout.getvalue(), calls

    def _assert_calls(self, calls, *urls):
        self.assertEqual([url for url, _ in calls], list(urls))
        self.assertEqual([timeout for _, timeout in calls], [20] * len(urls))

    def test_known_node_with_empty_versions_allows_first_publish(self):
        code, output, calls = self._run_guard(
            self.NODE, self._metadata(), self._versions()
        )
        self.assertEqual(code, 0, output)
        self.assertIn("RIG-A NO_BASELINE_FIRST_PUBLISH", output)
        self._assert_calls(calls, self.METADATA_URL, self.VERSIONS_URL)

    def test_unknown_node_404_blocks_before_versions(self):
        error = urllib.error.HTTPError(self.METADATA_URL, 404, "Not Found", None, None)
        code, output, calls = self._run_guard(self.NODE, error)
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A BASELINE_UNKNOWN_NODE", output)
        self._assert_calls(calls, self.METADATA_URL)

    def test_lowercase_redirect_302_blocks_without_following(self):
        lower = f"https://api.comfy.org/nodes/{self.NODE.lower()}"
        error = urllib.error.HTTPError(
            lower, 302, "Found", {"Location": self.METADATA_URL}, None
        )
        code, output, calls = self._run_guard(self.NODE.lower(), error)
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        # 只请求了小写 URL，没有跟随 302 到规范大小写 URL
        self._assert_calls(calls, lower)

    def test_redirect_handler_refuses_to_follow(self):
        handler = self.namespace["_NoRedirect"]()
        request = urllib.request.Request(
            f"https://api.comfy.org/nodes/{self.NODE.lower()}"
        )
        followed = handler.redirect_request(
            request, None, 302, "Found", {"Location": self.METADATA_URL}, self.METADATA_URL
        )
        self.assertIsNone(followed)

    def test_guard_installs_no_redirect_opener(self):
        self.assertIn("class _NoRedirect(urllib.request.HTTPRedirectHandler)", self.guard)
        self.assertIn("urllib.request.install_opener", self.guard)
        self.assertIn("urllib.request.build_opener(_NoRedirect)", self.guard)

    def test_newer_version_is_ahead(self):
        code, output, calls = self._run_guard(
            self.NODE, self._metadata(), self._versions("1.9.9")
        )
        self.assertEqual(code, 0, output)
        self.assertIn("RIG-A VERSION_AHEAD 2.0.0 > 1.9.9", output)
        self._assert_calls(calls, self.METADATA_URL, self.VERSIONS_URL)

    def test_same_version_blocks(self):
        code, output, _ = self._run_guard(
            self.NODE, self._metadata(), self._versions("1.9.0", "2.0.0")
        )
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A VERSION_UNCHANGED 2.0.0 already published", output)

    def test_version_below_published_blocks(self):
        code, output, _ = self._run_guard(
            self.NODE, self._metadata(), self._versions("1.9.0", "2.1.0")
        )
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A VERSION_BELOW_PUBLISHED 2.0.0 < 2.1.0", output)

    def test_metadata_network_error_blocks(self):
        code, output, calls = self._run_guard(self.NODE, urllib.error.URLError("boom"))
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        self.assertIn("URLError", output)
        self._assert_calls(calls, self.METADATA_URL)

    def test_metadata_timeout_blocks(self):
        code, output, calls = self._run_guard(self.NODE, TimeoutError("timed out"))
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        self._assert_calls(calls, self.METADATA_URL)

    def test_versions_network_error_blocks(self):
        code, output, calls = self._run_guard(
            self.NODE, self._metadata(), urllib.error.URLError("boom")
        )
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        self._assert_calls(calls, self.METADATA_URL, self.VERSIONS_URL)

    def test_versions_404_is_unreachable_not_unknown_node(self):
        error = urllib.error.HTTPError(self.VERSIONS_URL, 404, "Not Found", None, None)
        code, output, _ = self._run_guard(self.NODE, self._metadata(), error)
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        self.assertNotIn("BASELINE_UNKNOWN_NODE", output)

    def test_non_200_status_blocks(self):
        code, output, _ = self._run_guard(self.NODE, _StubResponse(302, b""))
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        self.assertIn("HTTP 302", output)

    def test_metadata_invalid_json_blocks(self):
        code, output, calls = self._run_guard(self.NODE, _StubResponse(200, b"<html>"))
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)
        self._assert_calls(calls, self.METADATA_URL)

    def test_versions_invalid_json_blocks(self):
        code, output, _ = self._run_guard(
            self.NODE, self._metadata(), _StubResponse(200, b"not-json")
        )
        self.assertEqual(code, 1, output)
        self.assertIn("RIG-A REGISTRY_UNREACHABLE", output)


class CiWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = _load(CI)
        cls.text = CI.read_text(encoding="utf-8")
        cls.test_job = cls.workflow["jobs"]["test"]

    def test_python_matrix_unchanged(self):
        matrix = self.test_job["strategy"]["matrix"]
        self.assertEqual(matrix["python-version"], ["3.11", "3.12"])

    def test_template_index_check_between_install_and_tests(self):
        names = _steps(self.test_job)
        index_step = _step(self.test_job, "Check template index is in sync")["run"]
        self.assertEqual(index_step.strip(), "python tools/build_template_index.py --check")
        self.assertGreater(
            names.index("Check template index is in sync"), names.index("Install dependencies")
        )
        self.assertLess(
            names.index("Check template index is in sync"), names.index("Run Python tests")
        )

    def test_ruff_is_pinned_exactly(self):
        lint = self.workflow["jobs"]["lint"]
        commands = "\n".join(
            step.get("run", "") for step in lint["steps"]
        )
        self.assertIn("pip install ruff==0.17.2", commands)
        self.assertNotRegex(commands, r"(?m)pip install ruff\s*$")

    def test_aggregate_job_unchanged(self):
        aggregate = self.workflow["jobs"]["ci"]
        self.assertEqual(aggregate["needs"], ["test", "frontend", "lint"])
        self.assertEqual(aggregate["if"], "always()")


class TemplateIndexToolTest(unittest.TestCase):
    def _run(self, args, encoding, cwd=REPO_ROOT, extra_env=None):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = encoding
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, str(cwd / "tools" / "build_template_index.py"), *args],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding=encoding,
            errors="replace",
        )

    def _copy_repo(self, tmp):
        root = Path(tmp) / "repo"
        ignore = shutil.ignore_patterns("__pycache__")
        shutil.copytree(
            REPO_ROOT / "model_toolkit", root / "model_toolkit", ignore=ignore
        )
        shutil.copytree(
            REPO_ROOT / "components" / "architectures",
            root / "components" / "architectures",
            ignore=ignore,
        )
        (root / "tools").mkdir()
        shutil.copy2(TOOL, root / "tools" / "build_template_index.py")
        return root

    def test_check_is_read_only_and_sync_utf8(self):
        before = hashlib.sha256(INDEX.read_bytes()).hexdigest()
        result = self._run(["--check"], "utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("INDEX_SYNC_OK"), result.stdout)
        after = hashlib.sha256(INDEX.read_bytes()).hexdigest()
        self.assertEqual(before, after, "--check 不得写 index.json")

    def test_check_sync_cp936_does_not_crash(self):
        result = self._run(["--check"], "cp936")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("INDEX_SYNC_OK"), result.stdout)
        self.assertNotIn("UnicodeEncodeError", result.stderr)

    def test_missing_index_reports_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_repo(tmp)
            (root / "components" / "architectures" / "index.json").unlink()
            result = self._run(["--check"], "utf-8", cwd=root)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(result.stdout.startswith("INDEX_MISSING"), result.stdout)

    def test_drift_reports_token_and_regenerate_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_repo(tmp)
            (root / "components" / "architectures" / "index.json").write_text(
                "{}\n", encoding="utf-8"
            )
            result = self._run(["--check"], "utf-8", cwd=root)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(result.stdout.startswith("INDEX_OUT_OF_SYNC"), result.stdout)
            self.assertIn("regenerate", result.stdout)

    def test_missing_c_loader_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_repo(tmp)
            fake = Path(tmp) / "fake"
            fake.mkdir()
            (fake / "yaml.py").write_text(
                "class YAMLError(Exception):\n    pass\n", encoding="utf-8"
            )
            result = self._run(
                ["--check"], "utf-8", cwd=root, extra_env={"PYTHONPATH": str(fake)}
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(result.stdout.startswith("INDEX_TOOL_ERROR"), result.stdout)
            self.assertIn("CSafeLoader", result.stdout)

    def test_tool_uses_ascii_tokens(self):
        source = TOOL.read_text(encoding="utf-8")
        for token in (
            "INDEX_SYNC_OK",
            "INDEX_MISSING",
            "INDEX_OUT_OF_SYNC",
            "INDEX_TOOL_ERROR",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
