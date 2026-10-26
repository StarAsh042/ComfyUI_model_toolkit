import test from "node:test";
import assert from "node:assert/strict";

import {
    blockLabel,
    buildCsvExport,
    capabilitySummary,
    componentChartData,
    csvExportFileName,
    csvMenuLabels,
    dtypesOf,
    escapeCsvCell,
    filterKeys,
    formatCount,
    formatKeysStatus,
    formatParams,
    formatPercent,
    isAttentionKey,
    keysToCsv,
    keysUnderPrefixes,
    normalizePrefix,
    paginate,
    parseReportParams,
    precisionChartData,
    ratioLabel,
    reportToMarkdown,
    resolveKeyView,
    safeFilename,
    shapeLabel,
    shortPrecisionLabel,
    sortKeys,
    sourceDescription,
    sourceLabel,
} from "../../web/report/report_logic.js";
import * as unetView from "../../web/report/views/unet.js";

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
    assert.ok(!markdown.includes("## Keys"));
});

test("导出文件名去掉路径并清理非法字符", () => {
    assert.equal(
        safeFilename("..\\Mine/SD1\\my model?.safetensors", "report.json"),
        "my model_.report.json"
    );
    assert.equal(safeFilename("", "report.md"), "model.report.md");

    const longStem = "b".repeat(120);
    const truncated = safeFilename(`${longStem}.safetensors`, "report.json");
    assert.equal(truncated, `${"b".repeat(80)}.report.json`);
    assert.equal(truncated.length, 80 + ".report.json".length);
    assert.ok(truncated.length <= 80 + 1 + "report.json".length);

    const longExtension = "x".repeat(40);
    assert.equal(
        safeFilename("model.safetensors", longExtension),
        `model.${longExtension}`
    );
});

const RESOLVE_KEYS = Array.from({ length: 260 }, (_, index) => {
    const attention = index % 5 === 0;
    return {
        name: `${index % 2 === 0 ? "encoder" : "decoder"}.block.${index}.${
            attention ? "transformer_blocks.0.attn2.to_k" : "resnet.conv"
        }.weight`,
        component: index % 3 === 0 ? "CLIP" : "UNET",
        dtype: index % 4 === 0 ? "torch.float32" : "torch.float16",
        params: index,
    };
});

const EXPORT_REPORT = { summary: { file_name: "my model.safetensors" } };

test("resolveKeyView 应用全部筛选、块前缀和当前排序并保留跨分页全集", () => {
    const base = resolveKeyView(RESOLVE_KEYS, {}, []);
    assert.equal(base.totalKeys, 260);
    assert.equal(base.viewCount, 260);
    assert.equal(base.narrowed, false);
    assert.equal(resolveKeyView(RESOLVE_KEYS, { query: "attn2" }, []).viewCount, 52);
    assert.equal(resolveKeyView(RESOLVE_KEYS, { component: "CLIP" }, []).viewCount, 87);
    assert.equal(
        resolveKeyView(RESOLVE_KEYS, { dtype: "torch.float32" }, []).viewCount,
        65
    );
    assert.equal(
        resolveKeyView(RESOLVE_KEYS, { attentionOnly: true }, []).viewCount,
        52
    );
    assert.equal(resolveKeyView(RESOLVE_KEYS, {}, ["encoder."]).viewCount, 130);

    const view = resolveKeyView(RESOLVE_KEYS, { sort: "params-desc" }, ["encoder."]);
    assert.equal(view.viewCount, 130);
    assert.equal(view.prefixActive, true);
    assert.equal(view.narrowed, true);
    assert.equal(view.rows.length, view.viewCount);
    assert.ok(view.rows.every((key) => key.name.startsWith("encoder.")));
    assert.equal(view.rows[0].params, 258);
    assert.equal(view.rows.at(-1).params, 0);
    for (let index = 1; index < view.rows.length; index += 1) {
        assert.ok(
            view.rows[index - 1].params >= view.rows[index].params,
            "rows must follow the current sort"
        );
    }

    const firstPage = paginate(view.rows, 0, 100);
    assert.equal(firstPage.items.length, 100);
    const secondPage = paginate(view.rows, 1, 100);
    assert.equal(secondPage.items.length, 30);
    assert.deepEqual(
        secondPage.items.map((key) => key.name),
        view.rows.slice(100, 130).map((key) => key.name)
    );
});

test("resolveKeyView 的空前缀不限制视图，而低层 keysUnderPrefixes 仍返回空", () => {
    const view = resolveKeyView(KEYS, {}, []);
    assert.equal(view.viewCount, KEYS.length);
    assert.equal(view.prefixActive, false);
    assert.equal(view.narrowed, false);
    assert.deepEqual(
        view.rows.map((key) => key.name),
        sortKeys(KEYS, "name").map((key) => key.name)
    );

    const blanks = resolveKeyView(KEYS, {}, ["", null]);
    assert.equal(blanks.viewCount, KEYS.length);
    assert.equal(blanks.prefixActive, false);

    assert.equal(keysUnderPrefixes(KEYS, []).length, 0);
    assert.equal(keysUnderPrefixes(KEYS, [""]).length, 0);
    assert.equal(keysUnderPrefixes(KEYS, ["cond_stage_model"]).length, 1);
    assert.equal(normalizePrefix("cond_stage_model"), "cond_stage_model.");

    assert.deepEqual(Object.keys(unetView).sort(), [
        "UNET_METRICS",
        "UnetView",
        "absoluteStageSize",
        "defaultBaseSize",
        "findContextDim",
        "keyPrefixForBlock",
        "keysUnderPrefixes",
        "layoutUnet",
        "stageKeyPrefixes",
        "stageRoleLabel",
    ]);
    assert.equal("normalizePrefix" in unetView, false);
    assert.strictEqual(unetView.keysUnderPrefixes, keysUnderPrefixes);
    assert.equal(
        unetView.keysUnderPrefixes(KEYS, ["cond_stage_model"]).length,
        1
    );
});

test("resolveKeyView 与 buildCsvExport 对非法输入不抛异常", () => {
    assert.deepEqual(resolveKeyView(null, null, null), {
        rows: [],
        viewCount: 0,
        totalKeys: 0,
        prefixActive: false,
        narrowed: false,
    });
    assert.deepEqual(resolveKeyView({}, "nope", "prefix"), {
        rows: [],
        viewCount: 0,
        totalKeys: 0,
        prefixActive: false,
        narrowed: false,
    });
    assert.equal(buildCsvExport().code, "NO_REPORT");
    assert.equal(
        buildCsvExport({
            report: { summary: { file_name: null } },
            keys: "nope",
            view: null,
            filters: null,
            scope: "all",
        }).code,
        "NO_KEY_LIST"
    );
    assert.equal(formatKeysStatus(null, null), "No key list");
    assert.deepEqual(csvMenuLabels(null), {
        currentView: "Export keys CSV · current view (0 of 0)",
        all: "Export keys CSV · all keys (0)",
    });
    assert.equal(csvExportFileName(null, "nope"), "model.keys.csv");
    assert.doesNotThrow(() => keysUnderPrefixes(null, null));
    assert.deepEqual(keysUnderPrefixes(null, null), []);
});

test("CSV 行数为 recordCount + 1、无尾换行并阻止公式注入", () => {
    const view = resolveKeyView(KEYS, { sort: "params-desc" }, []);
    const result = buildCsvExport({
        report: EXPORT_REPORT,
        view,
        scope: "view",
        keys: KEYS,
        filters: { sort: "params-desc" },
    });
    assert.equal(result.ok, true);
    assert.equal(result.recordCount, KEYS.length);
    assert.equal(result.csv.split("\n").length, result.recordCount + 1);
    assert.equal(result.csv.endsWith("\n"), false);
    assert.equal(result.csv.split("\n")[0], "Key,Component,Shape,Dtype,Params");
    assert.deepEqual(
        result.rows.map((key) => key.params),
        [37945344, 921600, 245760, 18432]
    );
    assert.ok(result.csv.split("\n")[1].startsWith("cond_stage_model."));
    assert.equal(result.notice, "Exported 4 keys");

    const filteredView = resolveKeyView(KEYS, { component: "UNET" }, []);
    const filtered = buildCsvExport({
        report: EXPORT_REPORT,
        view: filteredView,
        scope: "view",
        keys: KEYS,
        filters: { component: "UNET" },
    });
    assert.equal(filtered.recordCount, 2);
    assert.equal(filtered.notice, "Exported 2 of 4 keys");

    const all = buildCsvExport({
        report: EXPORT_REPORT,
        view,
        scope: "all",
        keys: KEYS,
        filters: {},
    });
    assert.equal(all.recordCount, KEYS.length);
    assert.equal(all.notice, "Exported 4 keys");

    const injectionKeys = [
        {
            name: "=SUM(A1:A2)",
            component: "+CMD",
            shape: [2],
            dtype: "-1",
            params: 2,
        },
        {
            name: "@user",
            component: "normal",
            shape: [1],
            dtype: "torch.float16",
            params: 1,
        },
    ];
    const injectionView = resolveKeyView(injectionKeys, {}, []);
    const injection = buildCsvExport({
        report: EXPORT_REPORT,
        view: injectionView,
        scope: "view",
        keys: injectionKeys,
        filters: {},
    });
    assert.equal(injection.recordCount, 2);
    assert.equal(injection.csv.split("\n").length, 3);
    assert.ok(injection.csv.includes("'=SUM(A1:A2)"));
    assert.ok(injection.csv.includes("'+CMD"));
    assert.ok(injection.csv.includes("'-1"));
    assert.ok(injection.csv.includes("'@user"));
});

test("formatKeysStatus 精确输出四种状态并使用千分位与 U+2013", () => {
    const allRows = Array.from({ length: 2182 }, () => ({}));
    const allView = {
        rows: allRows,
        viewCount: 2182,
        totalKeys: 2182,
        narrowed: false,
    };
    assert.equal(
        formatKeysStatus(allView, paginate(allRows, 0, 100)),
        "Showing 1\u2013100 of 2,182 keys"
    );

    const narrowedRows = Array.from({ length: 1234 }, () => ({}));
    const narrowedView = {
        rows: narrowedRows,
        viewCount: 1234,
        totalKeys: 2182,
        narrowed: true,
    };
    const narrowedStatus = formatKeysStatus(
        narrowedView,
        paginate(narrowedRows, 0, 100)
    );
    assert.equal(narrowedStatus, "Showing 1\u2013100 of 1,234 of 2,182 keys");
    assert.ok(narrowedStatus.includes("\u2013"));
    assert.ok(!narrowedStatus.includes("-"));

    assert.equal(
        formatKeysStatus(
            { rows: [], viewCount: 0, totalKeys: 2182, narrowed: true },
            paginate([], 0, 100)
        ),
        "Showing 0 of 2,182 keys"
    );
    assert.equal(
        formatKeysStatus(
            { rows: [], viewCount: 0, totalKeys: 0, narrowed: false },
            paginate([], 0, 100)
        ),
        "No key list"
    );
});

test("csvMenuLabels 精确输出当前视图与全部键文案", () => {
    assert.deepEqual(csvMenuLabels({ viewCount: 1234, totalKeys: 2182 }), {
        currentView: "Export keys CSV · current view (1,234 of 2,182)",
        all: "Export keys CSV · all keys (2,182)",
    });
    assert.deepEqual(csvMenuLabels({ viewCount: 5, totalKeys: 5 }), {
        currentView: "Export keys CSV · current view (5 of 5)",
        all: "Export keys CSV · all keys (5)",
    });
});

test("csvExportFileName 生成 view/all 文件名并把 stem 截断到 80 字符", () => {
    assert.equal(
        csvExportFileName("..\\Mine/SD1\\my model?.safetensors", "view"),
        "my model_.keys.csv"
    );
    assert.equal(
        csvExportFileName("..\\Mine/SD1\\my model?.safetensors", "all"),
        "my model_.all-keys.csv"
    );
    assert.equal(csvExportFileName("a:b?.safetensors", "view"), "a_b_.keys.csv");
    assert.equal(csvExportFileName("", "all"), "model.all-keys.csv");

    const longStem = "a".repeat(100);
    assert.equal(
        csvExportFileName(`${longStem}.safetensors`, "view"),
        `${"a".repeat(80)}.keys.csv`
    );
});

test("buildCsvExport 的失败计划为空载荷且不产生文件", () => {
    const view = resolveKeyView(KEYS, {}, []);
    const failures = [
        {
            result: buildCsvExport({
                report: null,
                view,
                scope: "view",
                keys: KEYS,
                filters: {},
            }),
            code: "NO_REPORT",
            notice: "No report is loaded.",
        },
        {
            result: buildCsvExport({
                report: EXPORT_REPORT,
                view: { rows: [], viewCount: 0, totalKeys: KEYS.length },
                scope: "view",
                keys: KEYS,
                filters: {},
            }),
            code: "EMPTY_VIEW",
            notice: "No keys match the current filters.",
        },
        {
            result: buildCsvExport({
                report: EXPORT_REPORT,
                view,
                scope: "all",
                keys: [],
                filters: {},
            }),
            code: "NO_KEY_LIST",
            notice: "This report does not include a key list.",
        },
    ];
    for (const { result, code, notice } of failures) {
        assert.equal(result.ok, false);
        assert.equal(result.code, code);
        assert.deepEqual(result.rows, []);
        assert.equal(result.csv, "");
        assert.equal(result.fileName, "");
        assert.equal(result.recordCount, 0);
        assert.equal(result.notice, notice);
    }

    const all = buildCsvExport({
        report: EXPORT_REPORT,
        view: { rows: [], viewCount: 0, totalKeys: KEYS.length },
        scope: "all",
        keys: KEYS,
        filters: { sort: "params-desc" },
    });
    assert.equal(all.ok, true);
    assert.equal(all.recordCount, KEYS.length);
    assert.equal(all.fileName, "my model.all-keys.csv");
    assert.deepEqual(
        all.rows.map((key) => key.params),
        [37945344, 921600, 245760, 18432]
    );
});

test("resolveKeyView 与 buildCsvExport 同参幂等且不修改入参", () => {
    const keys = structuredClone(KEYS);
    const filters = {
        query: "",
        component: "all",
        dtype: "all",
        attentionOnly: false,
        sort: "params-desc",
    };
    const prefixes = ["model.diffusion_model."];
    const snapshot = JSON.stringify({ keys, filters, prefixes });

    const first = resolveKeyView(keys, filters, prefixes);
    const second = resolveKeyView(keys, filters, prefixes);
    assert.deepEqual(first, second);
    assert.equal(JSON.stringify({ keys, filters, prefixes }), snapshot);
    assert.notStrictEqual(first.rows, keys);

    const args = {
        report: EXPORT_REPORT,
        view: first,
        scope: "view",
        keys,
        filters,
    };
    const exportA = buildCsvExport(args);
    const exportB = buildCsvExport(args);
    assert.deepEqual(exportA, exportB);
    assert.strictEqual(exportA.rows, first.rows);
    assert.equal(JSON.stringify({ keys, filters, prefixes }), snapshot);

    const allA = buildCsvExport({ ...args, scope: "all" });
    const allB = buildCsvExport({ ...args, scope: "all" });
    assert.deepEqual(allA, allB);
    assert.equal(JSON.stringify(keys), JSON.stringify(KEYS));
});

test("resolveKeyView 对 100k 键全量排序在 600 ms 内完成", () => {
    const keys = Array.from({ length: 100000 }, (_, index) => ({
        name: `block.${index % 1000}.${index}`,
        component: index % 3 === 0 ? "CLIP" : "UNET",
        dtype: index % 4 === 0 ? "torch.float32" : "torch.float16",
        params: index,
    }));
    const started = performance.now();
    const view = resolveKeyView(keys, {}, []);
    const elapsed = performance.now() - started;
    assert.equal(view.totalKeys, 100000);
    assert.equal(view.viewCount, 100000);
    assert.equal(view.rows.length, 100000);
    assert.equal(view.narrowed, false);
    assert.equal(view.prefixActive, false);

    assert.deepEqual(
        view.rows.slice(0, 3).map((key) => key.name),
        ["block.0.0", "block.0.1000", "block.0.2000"]
    );
    assert.ok(elapsed <= 600, `resolveKeyView took ${elapsed.toFixed(1)} ms`);
});

test("buildCsvExport 生成 20k 行 CSV 在 1500 ms 内完成", () => {
    const keys = Array.from({ length: 20000 }, (_, index) => ({
        name: `block.${index}.weight`,
        component: "UNET",
        shape: [index + 1],
        dtype: "torch.float16",
        params: index,
    }));
    const started = performance.now();
    const result = buildCsvExport({
        report: EXPORT_REPORT,
        view: null,
        scope: "all",
        keys,
        filters: { sort: "name" },
    });
    const elapsed = performance.now() - started;
    assert.equal(result.ok, true);
    assert.equal(result.recordCount, 20000);
    assert.equal(result.csv.split("\n").length, 20001);
    assert.ok(elapsed <= 1500, `buildCsvExport took ${elapsed.toFixed(1)} ms`);
});
