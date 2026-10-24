import test from "node:test";
import assert from "node:assert/strict";

import {
    REPORT_PAGE_PATH,
    REPORT_PREPARE_PATH,
    buildPreparePayload,
    buildReportUrl,
} from "../../js/report_url.js";

test("报告 URL 指向页面路由并带目录与模型", () => {
    const url = new URL(
        buildReportUrl("http://127.0.0.1:8188", "loras", "style/ink.safetensors")
    );
    assert.equal(url.origin, "http://127.0.0.1:8188");
    assert.equal(url.pathname, REPORT_PAGE_PATH);
    assert.equal(url.searchParams.get("dir"), "loras");
    assert.equal(url.searchParams.get("model"), "style/ink.safetensors");
});

test("模型名中的反斜杠、空格与特殊字符被编码且可还原", () => {
    const model = "Mine\\SD1\\Alpha Anime #2.safetensors";
    const raw = buildReportUrl("http://localhost:8188", "checkpoints", model);
    assert.ok(!raw.includes(" "), "空格应被编码");
    assert.ok(!raw.includes("#"), "井号应被编码");
    const url = new URL(raw);
    assert.equal(url.searchParams.get("model"), model);
    assert.equal(url.searchParams.get("dir"), "checkpoints");
});

test("空目录回退到 checkpoints", () => {
    const url = new URL(buildReportUrl("http://localhost:8188", "", "model.safetensors"));
    assert.equal(url.searchParams.get("dir"), "checkpoints");
});

test("origin 末尾斜杠不会产生双斜杠", () => {
    const url = new URL(
        buildReportUrl("http://localhost:8188/", "checkpoints", "model.safetensors")
    );
    assert.equal(url.pathname, "/model_toolkit/report");
});

test("没有 origin 时返回站内绝对路径", () => {
    const url = buildReportUrl("", "checkpoints", "model.safetensors");
    assert.ok(url.startsWith(`${REPORT_PAGE_PATH}?`));
});

test("报告准备接口使用 /api 前缀并携带目录与模型", () => {
    assert.equal(REPORT_PREPARE_PATH, "/api/model_toolkit/report/prepare");
    assert.deepEqual(
        buildPreparePayload("loras", "style/ink.safetensors"),
        { dir: "loras", model: "style/ink.safetensors" }
    );
});

test("报告准备载荷对空值回退默认目录", () => {
    assert.deepEqual(buildPreparePayload("", ""), {
        dir: "checkpoints",
        model: "",
    });
});
