"""服务层路由：参数白名单、路径包含性、错误码与线程调度。

不启动真实 HTTP 服务：直接调用处理函数并检查 aiohttp 响应，注册流程用
假的 ``server.PromptServer`` 验证。真实 ComfyUI 下的路由由根包调用
``register_routes()`` 完成。
"""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import _env  # noqa: F401  设置 sys.path

import folder_paths
from aiohttp import web

from ComfyUI_model_toolkit.model_toolkit import report
from ComfyUI_model_toolkit.model_toolkit import server as service


class FakeRoutes:
    def __init__(self):
        self.gets = {}
        self.posts = {}
        self.statics = {}

    def get(self, path):
        def register(handler):
            self.gets[path] = handler
            return handler

        return register

    def post(self, path):
        def register(handler):
            self.posts[path] = handler
            return handler

        return register

    def static(self, prefix, path):
        self.statics[prefix] = path


def _request(**query):
    return types.SimpleNamespace(query=query)


class _JsonRequest:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    async def json(self):
        if self._error:
            raise self._error
        return self._payload


def _body(response):
    return json.loads(response.body.decode("utf-8"))


class RouteRegistrationTest(unittest.TestCase):
    def test_registers_report_routes(self):
        routes = FakeRoutes()
        prompt_server = type(
            "PromptServer", (), {"instance": types.SimpleNamespace(routes=routes)}
        )
        fake = types.ModuleType("server")
        fake.PromptServer = prompt_server

        with mock.patch.dict(sys.modules, {"server": fake}):
            self.assertTrue(service.register_routes())

        self.assertEqual(
            set(routes.gets),
            {
                "/model_toolkit/report",
                "/model_toolkit/folder_files",
                "/model_toolkit/report/data",
            },
        )
        self.assertIs(routes.gets["/model_toolkit/report"], service.handle_report_page)
        self.assertIs(routes.gets["/model_toolkit/report/data"], service.handle_report_data)
        self.assertIs(routes.gets["/model_toolkit/folder_files"], service.handle_folder_files)
        self.assertEqual(
            routes.posts,
            {"/model_toolkit/report/prepare": service.handle_report_prepare},
        )

    def test_skips_without_server_module(self):
        saved = sys.modules.pop("server", None)
        try:
            self.assertFalse(service.register_routes())
        finally:
            if saved is not None:
                sys.modules["server"] = saved

    def test_skips_without_instance(self):
        fake = types.ModuleType("server")
        fake.PromptServer = type("PromptServer", (), {"instance": None})
        with mock.patch.dict(sys.modules, {"server": fake}):
            self.assertFalse(service.register_routes())


class FolderFilesRouteTest(unittest.TestCase):
    def test_returns_per_directory_lists(self):
        with mock.patch.object(
            service, "list_model_dirs", return_value=["checkpoints", "loras"]
        ), mock.patch.object(
            service,
            "list_dir_files",
            side_effect=lambda name: ["a.safetensors"] if name == "checkpoints" else [],
        ):
            response = asyncio.run(service.handle_folder_files(_request()))

        self.assertEqual(response.status, 200)
        self.assertEqual(
            _body(response), {"checkpoints": ["a.safetensors"], "loras": []}
        )
        self.assertEqual(response.headers["Cache-Control"], "no-store")


class ReportPageRouteTest(unittest.TestCase):
    def test_missing_page_returns_json_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(service, "WEB_REPORT_DIR", Path(tmp)):
                response = asyncio.run(service.handle_report_page(_request()))

        self.assertEqual(response.status, 404)
        self.assertEqual(_body(response)["code"], "REPORT_PAGE_MISSING")

    def test_serves_index_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "index.html").write_text("<html></html>", encoding="utf-8")
            with mock.patch.object(service, "WEB_REPORT_DIR", Path(tmp)):
                response = asyncio.run(service.handle_report_page(_request()))

        self.assertIsInstance(response, web.FileResponse)
        self.assertEqual(response.status, 200)


class ReportDataRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.models_root = self._tmp.name
        self.model_path = os.path.join(
            self.models_root, "checkpoints", "a.safetensors"
        )
        self.files = ["a.safetensors", os.path.join("sub", "b.safetensors")]

        patches = [
            mock.patch.object(service, "list_model_dirs", return_value=["checkpoints"]),
            mock.patch.object(service, "list_dir_files", return_value=self.files),
            mock.patch.object(
                service, "resolve_model_path", return_value=self.model_path
            ),
            mock.patch.object(
                service, "build_report", return_value={"schema_version": "1.0"}
            ),
            mock.patch.object(folder_paths, "models_dir", self.models_root),
            mock.patch.object(folder_paths, "folder_names_and_paths", {}),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.build = service.build_report

    def _call(self, **query):
        return asyncio.run(service.handle_report_data(_request(**query)))

    def test_success_defaults_to_include_keys(self):
        response = self._call(dir="checkpoints", model="a.safetensors")

        self.assertEqual(response.status, 200)
        self.assertEqual(_body(response), {"schema_version": "1.0"})
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.build.assert_called_once_with("checkpoints", "a.safetensors", True)

    def test_keys_zero_disables_key_listing(self):
        response = self._call(dir="checkpoints", model="a.safetensors", keys="0")

        self.assertEqual(response.status, 200)
        self.build.assert_called_once_with("checkpoints", "a.safetensors", False)

    def test_accepts_url_separators(self):
        response = self._call(dir="checkpoints", model="sub/b.safetensors")

        self.assertEqual(response.status, 200)
        self.build.assert_called_once_with(
            "checkpoints", os.path.join("sub", "b.safetensors"), True
        )

    def test_invalid_directory(self):
        response = self._call(dir="loras", model="a.safetensors")

        self.assertEqual(response.status, 400)
        self.assertEqual(_body(response)["code"], "INVALID_DIR")
        self.build.assert_not_called()

    def test_model_must_be_in_selected_directory(self):
        for model_name in ("missing.safetensors", "../a.safetensors", "C:/a.safetensors"):
            with self.subTest(model_name=model_name):
                response = self._call(dir="checkpoints", model=model_name)
                self.assertEqual(response.status, 400)
                self.assertEqual(_body(response)["code"], "INVALID_MODEL")
        self.build.assert_not_called()

    def test_invalid_keys_parameter(self):
        response = self._call(dir="checkpoints", model="a.safetensors", keys="yes")

        self.assertEqual(response.status, 400)
        self.assertEqual(_body(response)["code"], "INVALID_KEYS")
        self.build.assert_not_called()

    def test_resolution_race_returns_not_found(self):
        with mock.patch.object(service, "resolve_model_path", return_value=None):
            response = self._call(dir="checkpoints", model="a.safetensors")

        self.assertEqual(response.status, 404)
        self.assertEqual(_body(response)["code"], "MODEL_NOT_FOUND")

    def test_path_outside_allowed_roots_is_forbidden(self):
        with tempfile.TemporaryDirectory() as other:
            outside = os.path.join(other, "a.safetensors")
            with mock.patch.object(
                service, "resolve_model_path", return_value=outside
            ):
                response = self._call(dir="checkpoints", model="a.safetensors")

        self.assertEqual(response.status, 403)
        self.assertEqual(_body(response)["code"], "FORBIDDEN")
        self.assertNotIn(other, response.text)
        self.build.assert_not_called()

    def test_registered_extra_path_is_allowed(self):
        """extra_model_paths 的目录在 models_dir 外，但仍属于合法白名单。"""
        with tempfile.TemporaryDirectory() as extra:
            extra_model = os.path.join(extra, "a.safetensors")
            with mock.patch.object(
                folder_paths,
                "folder_names_and_paths",
                {"checkpoints": ([extra], set())},
            ), mock.patch.object(
                service, "resolve_model_path", return_value=extra_model
            ):
                response = self._call(dir="checkpoints", model="a.safetensors")

        self.assertEqual(response.status, 200)

    def test_report_errors_map_to_status(self):
        cases = (
            ("MODEL_NOT_FOUND", 404),
            ("MODEL_LOAD_FAILED", 422),
            ("ANALYSIS_FAILED", 500),
        )
        for code, status in cases:
            with self.subTest(code=code):
                self.build.side_effect = report.ReportError(code, "failed")
                response = self._call(dir="checkpoints", model="a.safetensors")
                self.assertEqual(response.status, status)
                self.assertEqual(_body(response), {"error": "failed", "code": code})
        self.build.side_effect = None

    def test_unexpected_error_hides_details(self):
        self.build.side_effect = RuntimeError("secret stack trace")
        response = self._call(dir="checkpoints", model="a.safetensors")

        self.assertEqual(response.status, 500)
        self.assertEqual(_body(response)["code"], "INTERNAL_ERROR")
        self.assertNotIn("secret stack trace", response.text)
        self.build.side_effect = None


class ReportPrepareRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.models_root = self._tmp.name
        self.model_path = os.path.join(
            self.models_root, "checkpoints", "a.safetensors"
        )

        patches = [
            mock.patch.object(service, "list_model_dirs", return_value=["checkpoints"]),
            mock.patch.object(
                service, "list_dir_files", return_value=["a.safetensors"]
            ),
            mock.patch.object(
                service, "resolve_model_path", return_value=self.model_path
            ),
            mock.patch.object(
                service, "prepare_report", return_value={"schema_version": "1.0"}
            ),
            mock.patch.object(folder_paths, "models_dir", self.models_root),
            mock.patch.object(folder_paths, "folder_names_and_paths", {}),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.prepare = service.prepare_report

    def _call(self, payload=None, error=None):
        return asyncio.run(
            service.handle_report_prepare(_JsonRequest(payload, error))
        )

    def test_success_rebuilds_report(self):
        response = self._call({"dir": "checkpoints", "model": "a.safetensors"})

        self.assertEqual(response.status, 200)
        self.assertEqual(
            _body(response), {"ok": True, "schema_version": "1.0"}
        )
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.prepare.assert_called_once_with("checkpoints", "a.safetensors")

    def test_invalid_body(self):
        cases = (
            None,
            ["checkpoints", "a.safetensors"],
            "dir=checkpoints",
        )
        for payload in cases:
            with self.subTest(payload=payload):
                response = self._call(payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(_body(response)["code"], "INVALID_BODY")
        self.prepare.assert_not_called()

    def test_invalid_json_body(self):
        response = self._call(error=ValueError("bad json"))

        self.assertEqual(response.status, 400)
        self.assertEqual(_body(response)["code"], "INVALID_BODY")
        self.prepare.assert_not_called()

    def test_invalid_model_is_rejected_before_prepare(self):
        response = self._call(
            {"dir": "checkpoints", "model": "../a.safetensors"}
        )

        self.assertEqual(response.status, 400)
        self.assertEqual(_body(response)["code"], "INVALID_MODEL")
        self.prepare.assert_not_called()

    def test_report_errors_map_to_status(self):
        self.prepare.side_effect = report.ReportError(
            "MODEL_LOAD_FAILED", "failed"
        )
        response = self._call({"dir": "checkpoints", "model": "a.safetensors"})

        self.assertEqual(response.status, 422)
        self.assertEqual(
            _body(response),
            {"error": "failed", "code": "MODEL_LOAD_FAILED"},
        )

    def test_unexpected_error_hides_details(self):
        self.prepare.side_effect = RuntimeError("secret stack trace")
        response = self._call({"dir": "checkpoints", "model": "a.safetensors"})

        self.assertEqual(response.status, 500)
        self.assertEqual(_body(response)["code"], "INTERNAL_ERROR")
        self.assertNotIn("secret stack trace", response.text)


if __name__ == "__main__":
    unittest.main()
