/**
 * 模型检查器节点上的「Open Interactive Report」按钮。
 *
 * 点击时先请求服务端重建报告并预热缓存，再打开页面；不把整张工作流
 * 加入队列，避免误触发生图。按钮是 DOM widget 且 serialize=false，
 * 不写入 widgets_values，因此不影响旧工作流的控件位置与迁移结果。
 * 服务层或页面资源不可用时禁用按钮并提示，不影响 ComfyUI 启动。
 */
import { app } from "../../scripts/app.js";
import { DEFAULT_DIR } from "./widget_migration.js";
import {
    REPORT_PREPARE_PATH,
    buildPreparePayload,
    buildReportUrl,
} from "./report_url.js";

const NODE_TYPE = "ModelInspector";
const REPORT_PAGE_URL = "/api/model_toolkit/report";
const STATUS_COLOR = "#94a3b8";
const WARN_COLOR = "#fbbf24";

let availabilityPromise = null;

function checkReportAvailability() {
    if (!availabilityPromise) {
        availabilityPromise = fetch(REPORT_PAGE_URL, {
            method: "HEAD",
            cache: "no-store",
        })
            .then((response) => response.ok || response.status === 405)
            .catch(() => false);
    }
    return availabilityPromise;
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function createReportWidget(node) {
    if (typeof node.addDOMWidget !== "function") return;

    const container = document.createElement("div");
    container.style.cssText =
        "display:flex;flex-direction:column;gap:4px;padding:2px 0;min-height:44px;";

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Open Interactive Report";
    button.title = "Open the interactive model report in a new tab";
    button.style.cssText = [
        "width:100%",
        "padding:8px 10px",
        "border-radius:6px",
        "border:1px solid #38bdf8",
        "background:linear-gradient(180deg,#0ea5e9,#0284c7)",
        "color:#ffffff",
        "font:600 12px/1.2 ui-sans-serif,system-ui,sans-serif",
        "cursor:pointer",
    ].join(";");

    const status = document.createElement("div");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.textContent = "Checking report service…";
    status.style.cssText =
        "font:11px/1.3 ui-sans-serif,system-ui,sans-serif;color:#94a3b8;min-height:14px;";

    const setStatus = (text, color = STATUS_COLOR) => {
        status.textContent = text;
        status.style.color = color;
    };

    let busy = false;
    const setBusy = (value) => {
        busy = value;
        button.disabled = value;
        button.textContent = value ? "Preparing Report…" : "Open Interactive Report";
    };

    button.addEventListener("click", async () => {
        if (busy) return;

        const dirWidget = findWidget(node, "model_dir");
        const nameWidget = findWidget(node, "model_name");
        const modelDir = dirWidget?.value || DEFAULT_DIR;
        const modelName = nameWidget?.value || "";
        if (!modelName) {
            setStatus("Select a model first.", WARN_COLOR);
            return;
        }

        // 先同步打开空白页，避免 await 之后被浏览器当成弹窗拦截。
        const tab = window.open("about:blank", "_blank");
        if (!tab) {
            setStatus("Allow pop-ups for ComfyUI to open the report.", WARN_COLOR);
            return;
        }

        setBusy(true);
        setStatus("Rebuilding the report from the current model file…");

        try {
            const response = await fetch(REPORT_PREPARE_PATH, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildPreparePayload(modelDir, modelName)),
                cache: "no-store",
            });
            let payload = null;
            try {
                payload = await response.json();
            } catch (error) {
                payload = null;
            }
            if (!response.ok || payload?.ok !== true) {
                throw new Error(
                    payload?.error ||
                        `Report preparation failed (HTTP ${response.status})`
                );
            }

            if (tab.closed) {
                setStatus("The report tab was closed before loading.", WARN_COLOR);
                return;
            }
            tab.location.href = buildReportUrl(
                window.location.origin,
                modelDir,
                modelName
            );
            try {
                tab.opener = null;
            } catch (error) {
                // 同源新标签页通常可写；跨源时保持默认行为即可
            }
            setStatus("Report opened in a new tab.");
        } catch (error) {
            try {
                tab.close();
            } catch (closeError) {
                // 标签页可能已被用户关闭
            }
            setStatus(
                error instanceof Error ? error.message : "Failed to prepare the report.",
                WARN_COLOR
            );
        } finally {
            setBusy(false);
        }
    });

    container.append(button, status);

    const widget = node.addDOMWidget(
        "mt_open_report",
        "MODEL_TOOLKIT_REPORT_BUTTON",
        container,
        { serialize: false, hideOnZoom: false }
    );
    widget.serialize = false;
    if (widget.options) widget.options.serialize = false;

    checkReportAvailability().then((available) => {
        if (available) {
            setStatus("Opens in a new tab.");
            return;
        }
        button.disabled = true;
        button.textContent = "Interactive Report Unavailable";
        button.style.opacity = "0.55";
        button.style.cursor = "not-allowed";
        setStatus(
            "Update or restart ComfyUI so the model toolkit routes are available.",
            WARN_COLOR
        );
    });
}

app.registerExtension({
    name: "model_toolkit.ReportButton",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            createReportWidget(this);
            return result;
        };
    },
});
