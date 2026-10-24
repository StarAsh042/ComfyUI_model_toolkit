/** 报告页 URL 构造（纯函数，供节点按钮与 Node 单测共用）。 */
import { DEFAULT_DIR } from "./widget_migration.js";

export const REPORT_PAGE_PATH = "/model_toolkit/report";
export const REPORT_PREPARE_PATH = "/api/model_toolkit/report/prepare";

export function buildReportUrl(origin, modelDir, modelName) {
    const params = new URLSearchParams();
    params.set("dir", modelDir || DEFAULT_DIR);
    params.set("model", modelName || "");
    const root = String(origin || "").replace(/\/+$/, "");
    return `${root}${REPORT_PAGE_PATH}?${params.toString()}`;
}

export function buildPreparePayload(modelDir, modelName) {
    return {
        dir: modelDir || DEFAULT_DIR,
        model: modelName || "",
    };
}
