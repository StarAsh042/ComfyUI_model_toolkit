/**
 * 工作流控件兼容迁移：
 * - 1.2.0 → 当前版本：在模型检查器最前面补 model_dir；
 * - 旧 1.3.0（含「（全部）」与中文显示模式）→ 当前版本：
 *   「（全部）」改为 checkpoints，显示模式改为 Basic / Advanced。
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
        return [DEFAULT_DIR, modelName, detail, showCharts];
    }

    // 旧 1.3.0：四值数组，可能带「（全部）」或中文显示模式
    if (values.length === 4) {
        const [rawDir, modelName, rawDetail, showCharts] = values;
        const detail = normalizeDetail(rawDetail);
        if (!DETAIL_MODES.has(detail)) return values;

        const dir = normalizeDir(rawDir);
        if (dir === rawDir && detail === rawDetail) return values;
        return [dir, modelName, detail, showCharts];
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

    if ("show_details" in migrated) {
        const detail = normalizeDetail(migrated.show_details);
        if (detail !== migrated.show_details) {
            migrated.show_details = detail;
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
