import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
    DEFAULT_DIR,
    installConfigureMigration,
    migrateNamedWidgetsValues,
    migrateWidgetsValues,
} from "../../js/widget_migration.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const FIXTURE = path.join(ROOT, "tests", "fixtures", "workflow_1.2.0_model_inspector.json");

test("1.2.0 基础模式迁移为两控件", () => {
    assert.deepEqual(
        migrateWidgetsValues(["sd15.safetensors", "基础", true]),
        [DEFAULT_DIR, "sd15.safetensors"]
    );
});

test("1.2.0 高级模式丢弃 show_charts", () => {
    assert.deepEqual(
        migrateWidgetsValues(["sdxl.safetensors", "高级", false]),
        [DEFAULT_DIR, "sdxl.safetensors"]
    );
});

test("旧 1.3.0 的「（全部）」改为 checkpoints", () => {
    assert.deepEqual(
        migrateWidgetsValues(["（全部）", "sd15.safetensors", "基础", true]),
        [DEFAULT_DIR, "sd15.safetensors"]
    );
});

test("旧 1.3.0 选择其它目录时保留目录", () => {
    assert.deepEqual(
        migrateWidgetsValues(["loras", "style.safetensors", "Advanced", false]),
        ["loras", "style.safetensors"]
    );
});

test("当前两值格式保持不变", () => {
    const current = [DEFAULT_DIR, "sd15.safetensors"];
    assert.strictEqual(migrateWidgetsValues(current), current);
});

test("非旧版数组不迁移", () => {
    const candidates = [
        [DEFAULT_DIR, "sd15.safetensors", "Advanced"],
        ["sd15.safetensors", "Basic", "true"],
        ["sd15.safetensors", "Advanced", 1],
        [DEFAULT_DIR, "sd15.safetensors", "verbose", true],
        [],
    ];
    for (const values of candidates) {
        assert.strictEqual(migrateWidgetsValues(values), values);
    }
});

test("非数组输入原样返回", () => {
    for (const values of [null, undefined, "sd15", { model_name: "sd15.safetensors" }]) {
        assert.strictEqual(migrateWidgetsValues(values), values);
    }
});

test("named 形态迁移目录并删除已移除控件", () => {
    assert.deepEqual(
        migrateNamedWidgetsValues({
            model_dir: "（全部）",
            model_name: "sd15.safetensors",
            show_details: "高级",
            show_charts: true,
        }),
        {
            model_dir: DEFAULT_DIR,
            model_name: "sd15.safetensors",
        }
    );
});

test("named 形态只有旧控件时全部删除", () => {
    assert.deepEqual(migrateNamedWidgetsValues({ show_details: "基础" }), {});
});

test("named 形态无需迁移时返回原对象", () => {
    const named = { model_dir: "loras", model_name: "style.safetensors" };
    assert.strictEqual(migrateNamedWidgetsValues(named), named);
});

test("迁移在 configure 恢复控件之前执行", () => {
    const restored = [];
    function FakeNodeType() {}
    FakeNodeType.prototype.configure = function (info) {
        restored.push({
            values: [...info.widgets_values],
            named: { ...info.widgets_values_named },
        });
    };

    installConfigureMigration(FakeNodeType);
    new FakeNodeType().configure({
        widgets_values: ["sd15.safetensors", "基础", true],
        widgets_values_named: { model_dir: "（全部）", show_details: "基础" },
    });

    assert.deepEqual(restored, [
        {
            values: [DEFAULT_DIR, "sd15.safetensors"],
            named: { model_dir: DEFAULT_DIR },
        },
    ]);
});

test("1.2.0 工作流夹具迁移后为当前格式", async () => {
    const workflow = JSON.parse(await readFile(FIXTURE, "utf8"));
    const migrated = migrateWidgetsValues(workflow.nodes[0].widgets_values);

    assert.deepEqual(migrated, [
        DEFAULT_DIR,
        "Mine\\SD1\\Alpha_Anime_2.5D_fp16_v7.safetensors",
    ]);
});
