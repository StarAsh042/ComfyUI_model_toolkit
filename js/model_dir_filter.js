/**
 * 模型检查器「模型目录」联动下拉。
 *
 * 背景：INPUT_TYPES 只在服务端启动时求值一次，model_name 的静态选项
 * 无法随 model_dir 变化；因此文件清单由 /api/model_toolkit/folder_files
 * 提供，在前端按所选目录过滤。
 *
 * - 「（全部）」= 按目录顺序合并去重（与 Python resolve_model_path 的回退顺序一致）
 * - 具体目录 = 该目录的文件（递归，与 ComfyUI 自己的下拉一致）
 * - 路由不可用（旧版 ComfyUI）时退回静态列表，节点仍可按 checkpoints 使用
 */
import { app } from "../../scripts/app.js";

const NODE_TYPE = "ModelInspector";
const ALL_DIRS = "（全部）";
const FILES_URL = "/api/model_toolkit/folder_files";

/** 与 Python resolve_model_path 的「（全部）」回退顺序一致：按目录顺序去重合并。 */
function mergeFolderFiles(mapping) {
    const seen = new Set();
    const merged = [];
    for (const files of Object.values(mapping)) {
        for (const file of files) {
            if (!seen.has(file)) {
                seen.add(file);
                merged.push(file);
            }
        }
    }
    return merged;
}

async function fetchFolderFiles() {
    try {
        const resp = await fetch(FILES_URL);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        return null;
    }
}

async function applyModelDirFilter(node, resetIfInvalid) {
    const dirWidget = node.widgets?.find((w) => w.name === "model_dir");
    const nameWidget = node.widgets?.find((w) => w.name === "model_name");
    if (!dirWidget || !nameWidget) return;

    // 并发保护：异步返回时若已有更新的交互，丢弃本次结果
    node.__modelDirSeq = (node.__modelDirSeq || 0) + 1;
    const seq = node.__modelDirSeq;

    const mapping = await fetchFolderFiles();
    if (seq !== node.__modelDirSeq) return;

    const selected = dirWidget.value;
    let values;
    if (mapping) {
        values = selected === ALL_DIRS ? mergeFolderFiles(mapping) : (mapping[selected] || []);
    } else {
        values = nameWidget.options.values || [];
    }

    const previous = nameWidget.__modelDirAllValues || [];
    const current = nameWidget.value;

    // 交互切换目录时，已选模型不在新目录里才重置；工作流加载时不重置
    if (resetIfInvalid && current && !values.includes(current)) {
        nameWidget.value = values.length ? values[0] : "";
    }
    // 旧工作流里曾有效的已选值保留在列表中，避免加载后选项丢失
    if (
        nameWidget.value &&
        !values.includes(nameWidget.value) &&
        previous.includes(nameWidget.value)
    ) {
        values = [nameWidget.value, ...values];
    }

    nameWidget.__modelDirAllValues = values;
    nameWidget.options.values = values;
}

app.registerExtension({
    name: "model_toolkit.ModelDirFilter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const dirWidget = node.widgets?.find((w) => w.name === "model_dir");
            if (dirWidget) {
                const prevCallback = dirWidget.callback;
                dirWidget.callback = function (...args) {
                    if (prevCallback) prevCallback.apply(this, args);
                    applyModelDirFilter(node, true);
                };
            }
            applyModelDirFilter(node, false);
            return result;
        };

        // 工作流加载：widget 值恢复后再过滤（不重置，保留存档里的选择）
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            applyModelDirFilter(this, false);
            return result;
        };
    },
});

// ComfyUI 的刷新按钮会用 /object_info 的全量列表覆盖 options.values，
// 之后按当前目录重新过滤一次
const origRefresh = app.refreshComboInNodes?.bind(app);
if (origRefresh) {
    app.refreshComboInNodes = async function (...args) {
        const result = await origRefresh(...args);
        for (const node of app.graph._nodes) {
            if (node.comfyClass === NODE_TYPE) {
                applyModelDirFilter(node, false);
            }
        }
        return result;
    };
}
