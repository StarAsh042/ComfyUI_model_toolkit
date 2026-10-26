import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { reportToMarkdown } from "../../web/report/report_logic.js";

function read(relativePath) {
    return readFileSync(new URL(`../../${relativePath}`, import.meta.url), "utf8");
}

const REPORT = {
    schema_version: "1.0",
    summary: {
        file_name: "model.safetensors",
        model_dir: "checkpoints",
        total_params: 4,
        total_params_formatted: "4",
        key_count: 4,
        component_count: 1,
        primary_precision: "FP16",
        size_formatted: "1 MB",
        architecture: "SD1.x",
    },
    architecture: { type: "SD1.x", family: "sd1", template: null },
    components: { UNET: { params: 4, percentage: 100, key_count: 4 } },
    precision: { distributions: [{ label: "FP16", params: 4, percentage: 100 }] },
    metadata: {},
    structure: { kind: "generic" },
};

test("设计文档删除 PNG 承诺并明确 SVG 不提供位图", () => {
    const design = read("docs/design/interactive-report.md");
    assert.equal(/\bPNG\b/.test(design), false);
    assert.match(design, /SVG（当前体系结构图；不提供位图导出）/);
});

test("设计文档 §8.4 描述分页平面表且 §8.5 无 Markdown 可选键名", () => {
    const design = read("docs/design/interactive-report.md");
    const start = design.indexOf("### 8.4");
    const middle = design.indexOf("### 8.5");
    const end = design.indexOf("### 8.6");
    assert.ok(start > 0 && middle > start && end > middle);

    const browser = design.slice(start, middle);
    assert.match(browser, /平面表/);
    assert.match(browser, /分页/);
    assert.match(browser, /resolveKeyView/);
    assert.match(browser, /No key list/);
    assert.match(browser, /formatKeysStatus/);

    const exports = design.slice(middle, end);
    assert.match(exports, /current view/);
    assert.match(exports, /all keys/);
    assert.match(exports, /跨分页/);
    assert.match(exports, /不含键名/);
    assert.match(exports, /NO_REPORT/);
    assert.equal(design.includes("可选键名"), false);
});

test("README 说明两个 CSV 范围与 Markdown 不含键名", () => {
    const readme = read("README.md");
    assert.match(readme, /Export keys CSV · current view/);
    assert.match(readme, /Export keys CSV · all keys/);
    assert.match(readme, /跨分页取全集/);
    assert.match(readme, /不含键名章节/);
    assert.match(readme, /Advanced/);
    assert.match(readme, /全量键名/);
});

test("Markdown 导出不含键名章节", () => {
    const markdown = reportToMarkdown(REPORT);
    assert.ok(markdown.startsWith("# Model report:"));
    assert.ok(markdown.includes("## Components"));
    assert.ok(!markdown.includes("## Keys"));
    assert.ok(!markdown.includes("Weight keys"));
});
