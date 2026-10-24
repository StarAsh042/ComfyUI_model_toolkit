/**
 * 模型检查器「模型目录」联动下拉。
 *
 * 背景：INPUT_TYPES 只在服务端启动时求值一次，model_name 的静态选项
 * 无法随 model_dir 变化；因此文件清单由 /api/model_toolkit/folder_files
 * 提供，在前端按所选目录严格过滤：下拉里只出现当前目录内的模型。
 *
 * 路由不可用（旧版 ComfyUI）时退回静态 checkpoints 列表；选择其它目录
 * 时列表为空，绝不把 checkpoints 的文件混进别的目录。
 */
import { app } from "../../scripts/app.js";
import { DEFAULT_DIR, installConfigureMigration } from "./widget_migration.js";

const NODE_TYPE = "ModelInspector";
const FILES_URL = "/api/model_toolkit/folder_files";

async function fetchFolderFiles() {
    try {
        const resp = await fetch(FILES_URL);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        return null;
    }
}

async function applyModelDirFilter(node) {
    const dirWidget = node.widgets?.find((w) => w.name === "model_dir");
    const nameWidget = node.widgets?.find((w) => w.name === "model_name");
    if (!dirWidget || !nameWidget) return;

    // 并发保护：异步返回时若已有更新的交互，丢弃本次结果
    node.__modelDirSeq = (node.__modelDirSeq || 0) + 1;
    const seq = node.__modelDirSeq;

    const mapping = await fetchFolderFiles();
    if (seq !== node.__modelDirSeq) return;

    const selected = dirWidget.value || DEFAULT_DIR;
    let values;
    if (mapping) {
        values = Array.isArray(mapping[selected]) ? mapping[selected] : [];
    } else {
        values = selected === DEFAULT_DIR ? [...(node.__modelDirStaticValues || [])] : [];
    }

    // 严格模式：已选模型不属于当前目录时立即清掉，不复用旧目录的值
    if (!nameWidget.value || !values.includes(nameWidget.value)) {
        nameWidget.value = values.length ? values[0] : "";
    }
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
            const nameWidget = node.widgets?.find((w) => w.name === "model_name");
            if (nameWidget) {
                // 记录服务端下发的静态列表，作为路由不可用时的 checkpoints 兜底
                node.__modelDirStaticValues = [...(nameWidget.options?.values || [])];
            }

            const dirWidget = node.widgets?.find((w) => w.name === "model_dir");
            if (dirWidget) {
                const prevCallback = dirWidget.callback;
                dirWidget.callback = function (...args) {
                    if (prevCallback) prevCallback.apply(this, args);
                    applyModelDirFilter(node);
                };
            }
            applyModelDirFilter(node);
            return result;
        };

        // 旧工作流：在 configure 恢复控件前归一化控件值
        installConfigureMigration(nodeType);

        // 工作流加载：widget 值恢复后再按目录过滤
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            applyModelDirFilter(this);
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
                applyModelDirFilter(node);
            }
        }
        return result;
    };
}
