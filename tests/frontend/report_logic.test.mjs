import test from "node:test";
import assert from "node:assert/strict";

import {
    blockLabel,
    capabilitySummary,
    componentChartData,
    dtypesOf,
    escapeCsvCell,
    filterKeys,
    formatCount,
    formatParams,
    formatPercent,
    isAttentionKey,
    keysToCsv,
    paginate,
    parseReportParams,
    precisionChartData,
    ratioLabel,
    reportToMarkdown,
    safeFilename,
    shapeLabel,
    shortPrecisionLabel,
    sortKeys,
    sourceDescription,
    sourceLabel,
} from "../../web/report/report_logic.js";

const KEYS = [
    {
        name: "input_blocks.1.0.in_layers.2.weight",
        shape: [320, 320, 3, 3],
        dtype: "torch.float16",
        params: 921600,
        component: "UNET",
    },
    {
        name: "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
        shape: [320, 768],
        dtype: "torch.float16",
        params: 245760,
        component: "UNET",
    },
    {
        name: "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
        shape: [49408, 768],
        dtype: "torch.float32",
        params: 37945344,
        component: "CLIP",
    },
    {
        name: "first_stage_model.decoder.conv_in.weight",
        shape: [512, 4, 3, 3],
        dtype: "torch.float32",
        params: 18432,
        component: "VAE",
    },
];

const REPORT = {
    schema_version: "1.0",
    summary: {
        file_name: "<script>alert(1)</script>.safetensors",
        model_dir: "checkpoints",
        size_bytes: 2147483648,
        size_formatted: "2.00 GB",
        total_params: 100000000,
        total_params_formatted: "100.00M",
        key_count: 4,
        component_count: 3,
        primary_precision: "FP16 (Half precision)",
        architecture: "SD1.x",
    },
    architecture: {
        type: "SD1.x",
        family: "sd1",
        template: "sd1-ldm",
        template_coverage: 1,
        hint: "",
        note: "",
    },
    components: {
        UNET: { params: 80, percentage: 80, key_count: 2 },
        CLIP: { params: 20, percentage: 20, key_count: 2 },
    },
    precision: {
        primary: "FP16 (Half precision)",
        distributions: [
            { label: "FP16 (Half precision)", params: 100, percentage: 100 },
        ],
        components: {},
    },
    metadata: {
        "modelspec.architecture": "SD1.x",
        note: "<script>alert(1)</script> | pipe",
    },
    structure: {
        kind: "generic",
        note: "Architecture-specific view is not available for this model yet.",
    },
    keys: KEYS,
};

test("解析查询参数并对目录回退到 checkpoints", () => {
    assert.deepEqual(parseReportParams("?model=a.safetensors"), {
        dir: "checkpoints",
        model: "a.safetensors",
    });
    const params = parseReportParams(
        "?dir=loras&model=" + encodeURIComponent("Mine\\SD1\\a.safetensors")
    );
    assert.equal(params.dir, "loras");
    assert.equal(params.model, "Mine\\SD1\\a.safetensors");
});

test("数字与占比格式化", () => {
    assert.equal(formatCount(1066235386), "1,066,235,386");
    assert.equal(formatCount(null), "—");
    assert.equal(formatParams(1066235386), "1.07B");
    assert.equal(formatParams(860000000), "860M");
    assert.equal(formatParams(1500), "1.5K");
    assert.equal(formatParams(512), "512");
    assert.equal(formatPercent(80), "80%");
    assert.equal(formatPercent(33.333), "33.33%");
});

test("分辨率为 2 的幂时显示分数", () => {
    assert.equal(ratioLabel(1), "1/1");
    assert.equal(ratioLabel(0.5), "1/2");
    assert.equal(ratioLabel(0.125), "1/8");
    assert.equal(ratioLabel(null), "—");
    assert.equal(ratioLabel(0.333), "0.333");
});

test("形状标签与 dtype 清单", () => {
    assert.equal(shapeLabel([320, 320, 3, 3]), "[320, 320, 3, 3]");
    assert.equal(shapeLabel([]), "—");
    assert.deepEqual(dtypesOf(KEYS), ["torch.float16", "torch.float32"]);
});

test("注意力键识别覆盖 LDM 与 diffusers 命名", () => {
    for (const name of [
        "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    ]) {
        assert.ok(isAttentionKey(name), name);
    }
    assert.equal(isAttentionKey("input_blocks.1.0.in_layers.2.weight"), false);
});

test("按搜索、组件、dtype 与注意力过滤", () => {
    assert.equal(filterKeys(KEYS, { query: "TRANSFORMER" }).length, 2);
    assert.equal(filterKeys(KEYS, { component: "UNET" }).length, 2);
    assert.equal(filterKeys(KEYS, { dtype: "torch.float32" }).length, 2);
    assert.deepEqual(
        filterKeys(KEYS, { component: "UNET", attentionOnly: true }).map((key) => key.params),
        [245760]
    );
    assert.equal(filterKeys(KEYS, {}).length, KEYS.length);
});

test("排序不修改原数组且支持参数升降序", () => {
    const byParams = sortKeys(KEYS, "params-desc");
    assert.deepEqual(byParams.map((key) => key.params), [37945344, 921600, 245760, 18432]);
    assert.equal(KEYS[0].name, "input_blocks.1.0.in_layers.2.weight");
    assert.deepEqual(sortKeys(KEYS, "name").map((key) => key.component), ["CLIP", "VAE", "UNET", "UNET"]);
});

test("分页夹取边界并给出显示范围", () => {
    const first = paginate(KEYS, 0, 2);
    assert.deepEqual(first.items.map((key) => key.params), [921600, 245760]);
    assert.equal(first.start, 0);
    assert.equal(first.end, 2);
    assert.equal(first.pageCount, 2);

    const clamped = paginate(KEYS, 99, 3);
    assert.equal(clamped.page, 1);
    assert.equal(clamped.start, 3);
    assert.equal(clamped.end, 4);

    assert.deepEqual(paginate([], 3, 10), {
        items: [],
        page: 0,
        pageCount: 1,
        total: 0,
        start: 0,
        end: 0,
    });
});

test("图表数据按参数量降序并过滤零值", () => {
    assert.deepEqual(componentChartData(REPORT.components).map((item) => item.name), [
        "UNET",
        "CLIP",
    ]);
    assert.deepEqual(
        componentChartData({ A: { params: 0 }, B: { params: 5 } }).map((item) => item.name),
        ["B"]
    );
    const precision = precisionChartData([
        { label: "FP16 (Half precision)", params: 10, percentage: 50 },
        { label: "BF16 (Brain float)", params: 0, percentage: 0 },
        { label: "FP8 (E4M3)", params: 10, percentage: 50 },
    ]);
    assert.deepEqual(precision.map((item) => item.name), ["FP16", "FP8 E4M3"]);
    assert.equal(shortPrecisionLabel("FP32 (Single precision)"), "FP32");
});

test("能力检测把未知值标成 Unknown，不伪装成 No", () => {
    const caps = capabilitySummary({
        time_embedding: true,
        self_attention: false,
        dual_text_encoder: null,
        refiner: false,
    });
    const byKey = Object.fromEntries(caps.map((item) => [item.key, item]));
    assert.equal(byKey.time_embedding.value, "Yes");
    assert.equal(byKey.self_attention.value, "No");
    assert.equal(byKey.dual_text_encoder.value, "Unknown");
    assert.equal(byKey.refiner.value, "No");
});

test("块、来源标签有可读名称", () => {
    assert.equal(blockLabel("resblock"), "ResBlock");
    assert.equal(blockLabel("cross_attn"), "Cross-attention");
    assert.equal(sourceLabel("convention"), "Convention");
    assert.match(sourceDescription("unknown"), /not stored/i);
});

test("CSV 转义阻止公式注入并处理引号换行", () => {
    assert.equal(escapeCsvCell("=cmd|' /C calc'!A0"), "'=cmd|' /C calc'!A0");
    assert.equal(escapeCsvCell("+1"), "'+1");
    assert.equal(escapeCsvCell("-1"), "'-1");
    assert.equal(escapeCsvCell("@x"), "'@x");
    assert.equal(escapeCsvCell('a,"b"'), '"a,""b"""');
    assert.equal(escapeCsvCell("a\nb"), '"a\nb"');
});

test("键名 CSV 含表头且每个键一行", () => {
    const csv = keysToCsv(KEYS);
    const lines = csv.split("\n");
    assert.equal(lines[0], "Key,Component,Shape,Dtype,Params");
    assert.equal(lines.length, KEYS.length + 1);
    assert.ok(csv.includes("first_stage_model.decoder.conv_in.weight,VAE,\"[512, 4, 3, 3]\""));
});

test("Markdown 导出包含概览、组件与架构，并转义 HTML 与竖线", () => {
    const markdown = reportToMarkdown(REPORT);
    assert.ok(markdown.startsWith("# Model report:"));
    assert.ok(markdown.includes("## Components"));
    assert.ok(markdown.includes("## Architecture"));
    assert.ok(markdown.includes("UNET"));
    assert.ok(markdown.includes("&lt;script&gt;alert(1)&lt;/script&gt;"));
    assert.ok(!markdown.includes("<script>alert(1)</script>"));
    assert.ok(markdown.includes("\\| pipe"));
});

test("导出文件名去掉路径并清理非法字符", () => {
    assert.equal(
        safeFilename("..\\Mine/SD1\\my model?.safetensors", "report.json"),
        "my model_.report.json"
    );
    assert.equal(safeFilename("", "report.md"), "model.report.md");
});
