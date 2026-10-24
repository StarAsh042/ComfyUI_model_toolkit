/**
 * 工作流控件兼容迁移：
 * - 2.0.0 删除 show_details / show_charts，节点只剩 model_dir + model_name；
 * - 1.2.0 → 2.0.0：三值数组保留 model_name，补 model_dir=checkpoints；
 * - 1.3.0 → 2.0.0：四值数组保留前两项，丢弃两个已删除的控件。
 *
 * ComfyUI 按位置保存 widgets_values；新版前端还会额外保存
 * widgets_values_named。两种形态都要在节点 restore 前归一化。
 */

export const DEFAULT_DIR = "checkpoints";

const DETAIL_ALIASES = new Map([
    ["基础", "Basic"],
    ["高级", "Advanced"],
]);

const DETAIL_MODES = new Set(["Basic", "Advanced"]);

const LEGACY_WIDGETS = ["show_details", "show_charts"];

function normalizeDetail(value) {
    return DETAIL_ALIASES.get(value) || value;
}

function normalizeDir(value) {
    return value === "（全部）" ? DEFAULT_DIR : value;
}

export function migrateWidgetsValues(values) {
    if (!Array.isArray(values)) return values;

    // 1.2.0：三值数组 [model_name, show_details, show_charts]
    if (values.length === 3) {
        const [modelName, rawDetail, showCharts] = values;
        const detail = normalizeDetail(rawDetail);
        if (!DETAIL_MODES.has(detail) || typeof showCharts !== "boolean") {
            return values;
        }
        return [DEFAULT_DIR, modelName];
    }

    // 1.3.x：四值数组 [model_dir, model_name, show_details, show_charts]
    if (values.length === 4) {
        const [rawDir, modelName, rawDetail] = values;
        const detail = normalizeDetail(rawDetail);
        if (!DETAIL_MODES.has(detail)) return values;
        return [normalizeDir(rawDir), modelName];
    }

    return values;
}

export function migrateNamedWidgetsValues(named) {
    if (!named || typeof named !== "object" || Array.isArray(named)) return named;

    let changed = false;
    const migrated = { ...named };

    if ("model_dir" in migrated) {
        const dir = normalizeDir(migrated.model_dir);
        if (dir !== migrated.model_dir) {
            migrated.model_dir = dir;
            changed = true;
        }
    }

    for (const key of LEGACY_WIDGETS) {
        if (key in migrated) {
            delete migrated[key];
            changed = true;
        }
    }

    return changed ? migrated : named;
}

/**
 * 在 configure 恢复控件之前迁移。不能挂在 onConfigure：新版 ComfyUI 在
 * 控件值恢复完成后才调用它，此时旧数组已经写进控件。
 */
export function installConfigureMigration(nodeType) {
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
        if (info) {
            if (Array.isArray(info.widgets_values)) {
                info.widgets_values = migrateWidgetsValues(info.widgets_values);
            }
            if (info.widgets_values_named) {
                info.widgets_values_named = migrateNamedWidgetsValues(
                    info.widgets_values_named
                );
            }
        }
        return configure.apply(this, arguments);
    };
}
