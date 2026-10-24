import {
    capabilitySummary,
    componentChartData,
    dtypesOf,
    filterKeys,
    formatCount,
    formatParams,
    formatPercent,
    isAttentionKey,
    keysToCsv,
    paginate,
    parseReportParams,
    precisionChartData,
    reportToMarkdown,
    safeFilename,
    shapeLabel,
    sortKeys,
    sourceDescription,
    sourceLabel,
} from "./report_logic.js";
import { UnetView, keysUnderPrefixes } from "./views/unet.js";
import { mountPrinciples } from "./principles/principles.js";

const API_PATH = "/api/model_toolkit/report/data";
const WIDE_STORAGE_KEY = "model_toolkit.report.wide";
const TABS = ["overview", "architecture", "principles", "keys", "precision", "metadata", "json"];
const TEXT = "#cbd5e1";
const MUTED = "#94a3b8";
const GRID = "rgba(148, 163, 184, 0.16)";
const PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#fb7185", "#22d3ee", "#f472b6"];

const state = {
    report: null,
    keys: [],
    filteredKeys: [],
    page: 0,
    pageSize: 100,
    filters: {
        query: "",
        component: "all",
        dtype: "all",
        attentionOnly: false,
        sort: "name",
    },
    charts: {},
    activeTab: null,
    unetView: null,
    prefixFilter: [],
    prefixLabel: "",
};

class ReportLoadError extends Error {
    constructor(message, code) {
        super(message);
        this.code = code;
    }
}

function $(id) {
    return document.getElementById(id);
}

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
}

function clear(node) {
    if (node) node.replaceChildren();
}

function showLoading() {
    $("loading-state").hidden = false;
    $("error-state").hidden = true;
    $("report-view").hidden = true;
    for (const id of ["export-json", "export-csv", "export-markdown", "copy-json"]) {
        $(id).disabled = true;
    }
}

function showError(message, code, options = {}) {
    $("loading-state").hidden = true;
    $("report-view").hidden = true;
    $("error-state").hidden = false;
    $("error-message").textContent = message;
    $("error-code").textContent = code || "";
    $("retry-button").hidden = options.retry === false;
    document.title = "Report unavailable · Model Toolkit";
    for (const id of ["export-json", "export-csv", "export-markdown", "copy-json"]) {
        $(id).disabled = true;
    }
}

async function loadReport() {
    disposeCharts();
    showLoading();

    const params = parseReportParams(window.location.search);
    if (!params.model) {
        showError(
            "No model was selected. Open this page from a Model Inspector node.",
            "MISSING_PARAMS",
            { retry: false }
        );
        return;
    }

    const query = new URLSearchParams({
        dir: params.dir,
        model: params.model,
        keys: "1",
    });

    try {
        const response = await fetch(`${API_PATH}?${query.toString()}`, {
            cache: "no-store",
            headers: { Accept: "application/json" },
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch (error) {
            payload = null;
        }

        if (!response.ok) {
            throw new ReportLoadError(
                payload?.error || `Request failed with status ${response.status}.`,
                payload?.code || `HTTP_${response.status}`
            );
        }
        if (!payload || !payload.summary) {
            throw new ReportLoadError(
                "The report response is incomplete.",
                "INVALID_RESPONSE"
            );
        }

        state.report = payload;
        state.keys = Array.isArray(payload.keys) ? payload.keys : [];
        renderReport();
    } catch (error) {
        showError(
            error?.message || "Failed to load the model report.",
            error?.code || "UNKNOWN_ERROR"
        );
    }
}

function renderReport() {
    const report = state.report;
    $("loading-state").hidden = true;
    $("error-state").hidden = true;
    $("report-view").hidden = false;

    document.title = `${report.summary.file_name} · Model Report`;
    $("model-title").textContent = report.summary.file_name;
    $("architecture-badge").textContent =
        report.architecture?.type || report.summary.architecture || "Unknown";
    $("model-subtitle").textContent = [
        report.summary.model_dir,
        report.summary.size_formatted,
        `${formatCount(report.summary.key_count)} keys`,
        report.summary.primary_precision,
    ]
        .filter(Boolean)
        .join(" · ");
    $("schema-version").textContent = report.schema_version || "—";

    renderMeta();
    renderSummaryCards();
    renderCapabilities();
    renderOverview();
    renderArchitecture();
    renderPrinciples();
    renderKeys();
    renderPrecision();
    renderMetadata();
    renderRawJson();

    for (const id of ["export-json", "export-csv", "export-markdown"]) {
        $(id).disabled = false;
    }

    const hash = window.location.hash.replace(/^#/, "");
    const fallback = report.structure?.kind === "unet" ? "architecture" : "overview";
    activateTab(TABS.includes(hash) ? hash : fallback, false);
}

function renderMeta() {
    const report = state.report;
    const architecture = report.architecture || {};
    const parts = [];

    if (architecture.template) {
        parts.push(
            `Template ${architecture.template} · ${formatPercent(
                (architecture.template_coverage || 0) * 100
            )} coverage`
        );
    }
    if (architecture.family) parts.push(`Family ${architecture.family}`);
    parts.push(`${formatCount(report.summary.total_params)} parameters`);
    parts.push(
        report.structure?.kind === "unet"
            ? "UNet-specific structure"
            : "Generic component structure"
    );

    const container = $("summary-meta");
    clear(container);
    parts.forEach((text, index) => {
        if (index) container.append(element("span", "meta-sep", "·"));
        container.append(element("span", "meta-item", text));
    });
}

function metricCard(label, value, hint) {
    const card = element("div", "card metric");
    card.append(
        element("div", "metric-label", label),
        element("div", "metric-value", value),
        element("div", "metric-hint", hint)
    );
    if (hint) card.title = hint;
    return card;
}

function renderSummaryCards() {
    const report = state.report;
    const architecture = report.architecture || {};
    const precision = report.precision || {};
    const structure = report.structure || {};
    const componentNames = Object.keys(report.components || {});
    const componentHint = componentNames.join(", ") || "No components detected";
    const structureHint =
        structure.kind === "unet"
            ? structure.naming === "ldm"
                ? "LDM / CompVis key naming"
                : "Diffusers key naming"
            : "Generic component view";

    const cards = [
        metricCard(
            "Architecture",
            architecture.type || report.summary.architecture || "Unknown",
            architecture.template
                ? `Template ${architecture.template}`
                : architecture.family
                  ? `Family ${architecture.family}`
                  : "Detected from key names"
        ),
        metricCard(
            "Total parameters",
            report.summary.total_params_formatted,
            `${formatCount(report.summary.total_params)} parameters`
        ),
        metricCard(
            "Primary precision",
            report.summary.primary_precision || "Unknown",
            `${(precision.distributions || []).length} precision types`
        ),
        metricCard(
            "Model size",
            report.summary.size_formatted,
            `${formatCount(report.summary.key_count)} weight tensors`
        ),
        metricCard(
            "Components",
            formatCount(report.summary.component_count),
            componentHint
        ),
        metricCard("Structure view", structure.kind === "unet" ? "UNet" : "Generic", structureHint),
    ];

    const container = $("summary-cards");
    clear(container);
    for (const card of cards) container.append(card);
}

function renderCapabilities() {
    const capabilities = state.report.structure?.capabilities;
    const block = $("capabilities-block");
    if (!capabilities || !Object.keys(capabilities).length) {
        block.hidden = true;
        return;
    }

    block.hidden = false;
    const container = $("capabilities");
    clear(container);
    for (const item of capabilitySummary(capabilities)) {
        const chip = element(
            "span",
            `capability ${item.value === "Yes" ? "yes" : item.value === "No" ? "no" : "unknown"}`
        );
        chip.append(
            element("span", "capability-label", item.label),
            element("span", "capability-value", item.value)
        );
        chip.title = item.description;
        container.append(chip);
    }
}

function numericCell(value, title) {
    const cell = element("td", "num", value);
    if (title) cell.title = title;
    return cell;
}

function renderOverview() {
    const report = state.report;
    const components = report.components || {};
    const body = $("components-body");
    clear(body);

    for (const [name, info] of Object.entries(components)) {
        const row = document.createElement("tr");
        const nameCell = document.createElement("td");
        nameCell.append(element("span", "component-badge", name));
        row.append(
            nameCell,
            numericCell(formatParams(info.params), `${formatCount(info.params)} parameters`),
            numericCell(formatPercent(info.percentage)),
            numericCell(formatCount(info.key_count))
        );
        body.append(row);
    }

    if (!Object.keys(components).length) {
        const row = document.createElement("tr");
        const cell = element("td", "empty", "No components were detected.");
        cell.colSpan = 4;
        row.append(cell);
        body.append(row);
    }

    $("components-summary").textContent = `${Object.keys(components).length} components · ${formatParams(
        report.summary.total_params
    )} total`;

    const notes = $("architecture-notes");
    clear(notes);
    const architecture = report.architecture || {};
    if (architecture.hint) notes.append(element("p", "note", architecture.hint));
    if (architecture.note) notes.append(element("p", "note", architecture.note));
    for (const info of architecture.extra_info || []) {
        notes.append(element("p", "note", info));
    }
    if (!notes.childElementCount) {
        notes.append(element("p", "muted", "No additional architecture notes for this model."));
    }
    const browse = element("button", "button", "Browse keys");
    browse.type = "button";
    browse.addEventListener("click", () => activateTab("keys"));
    notes.append(browse);
}

function structureChip(label, value, source) {
    const chip = element("span", "structure-chip");
    chip.append(
        element("span", "structure-chip-label", label),
        element("span", "structure-chip-value", value)
    );
    if (source) {
        const badge = element("span", "source", sourceLabel(source));
        badge.title = sourceDescription(source);
        chip.append(badge);
    }
    return chip;
}

function findContextDim(structure) {
    const stages = [...(structure.stages || []), structure.middle, ...(structure.up || [])];
    for (const stage of stages) {
        for (const block of stage?.blocks || []) {
            const context = block?.attention?.context_dim;
            if (context) return context;
        }
    }
    return null;
}

function renderStructureSummary(structure) {
    const card = element("div", "panel-card");
    const head = element("div", "panel-head");
    head.append(
        element("h2", null, "Structure summary"),
        element("span", "muted", "Every value is labelled by where it comes from")
    );
    const chips = element("div", "chips");
    const latent = structure.latent || {};
    chips.append(
        structureChip("Key naming", structure.naming === "ldm" ? "LDM / CompVis" : "Diffusers"),
        structureChip("Prefix", structure.prefix || "(none)"),
        structureChip(
            "Latent channels",
            latent.in_channels ?? "Unknown",
            latent.in_channels_source
        ),
        structureChip(
            "Base size",
            latent.base_size ? `${latent.base_size} px` : "Unknown",
            latent.base_size_source
        ),
        structureChip(
            "Latent size",
            latent.latent_size ? `${latent.latent_size} px at 1/8` : "Unknown",
            latent.latent_size_source
        ),
        structureChip("Text context", findContextDim(structure) ?? "None"),
        structureChip("UNet parameters", formatParams(structure.totals?.params))
    );
    card.append(head, chips);
    return card;
}

function ioDetail(entry) {
    if (entry?.in_channels != null && entry?.out_channels != null) {
        return `${entry.in_channels} → ${entry.out_channels}`;
    }
    if (entry?.in_features != null && entry?.out_features != null) {
        return `${entry.in_features} → ${entry.out_features} features`;
    }
    return "—";
}

function renderIoTable(structure) {
    const card = element("div", "panel-card");
    const head = element("div", "panel-head");
    head.append(
        element("h2", null, "Inputs, outputs and extras"),
        element("span", "muted", "Outside the down / middle / up path")
    );

    const rows = [];
    const labels = {
        conv_in: "Input convolution",
        conv_out: "Output convolution",
        time_embed: "Time embedding",
    };
    for (const [key, entry] of Object.entries(structure.io || {})) {
        rows.push([
            labels[key] || key,
            ioDetail(entry),
            formatParams(entry.params),
            sourceLabel(entry.source),
        ]);
    }
    for (const extra of structure.extras || []) {
        rows.push([
            extra.kind === "label_embed"
                ? "Label embedding (size conditioning)"
                : extra.kind,
            "—",
            formatParams(extra.params),
            sourceLabel(extra.source),
        ]);
    }
    const unattributed = structure.unattributed || {};
    rows.push([
        "Unattributed",
        `${formatCount(unattributed.key_count || 0)} keys`,
        formatParams(unattributed.params || 0),
        "—",
    ]);

    const wrap = element("div", "table-wrap");
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const label of ["Entry", "Channels / features", "Parameters", "Source"]) {
        headRow.append(element("th", label === "Parameters" ? "num" : null, label));
    }
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    for (const row of rows) {
        const tr = document.createElement("tr");
        row.forEach((value, index) => {
            tr.append(element("td", index === 2 ? "num" : null, value));
        });
        tbody.append(tr);
    }
    table.append(thead, tbody);
    wrap.append(table);
    card.append(head, wrap);
    return card;
}

function renderReconciliation(structure) {
    const totals = structure.totals || {};
    const stageParams = (totals.by_stage || []).reduce(
        (sum, stage) => sum + (Number(stage.params) || 0),
        0
    );
    const ioParams = Number(totals.io_params) || 0;
    const extrasParams = Number(totals.extras_params) || 0;
    const unattributedParams = Number(totals.unattributed_params) || 0;
    const total = Number(totals.params) || 0;
    const calculated = stageParams + ioParams + extrasParams + unattributedParams;
    const balanced = calculated === total;

    const card = element("div", "panel-card reconciliation");
    const head = element("div", "panel-head");
    head.append(element("h2", null, "Parameter reconciliation"));
    head.append(
        element(
            "span",
            `badge ${balanced ? "good" : "warn"}`,
            balanced ? "Balanced" : "Check required"
        )
    );
    const equation = element(
        "p",
        "equation",
        `Stages ${formatParams(stageParams)} + I/O ${formatParams(ioParams)} + extras ${formatParams(
            extrasParams
        )} + unattributed ${formatParams(unattributedParams)} = ${formatParams(calculated)}`
    );
    const totalLine = element(
        "p",
        "muted",
        `UNet total from the report: ${formatParams(total)} (${formatCount(total)} parameters).`
    );
    const note = element(
        "p",
        "muted",
        "Every UNet key is assigned exactly once in the data layer; the automated invariant tests enforce this balance."
    );
    card.append(head, equation, totalLine, note);
    if (!balanced) {
        card.append(
            element(
                "p",
                "callout warn",
                `The calculated sum differs from the reported UNet total by ${formatParams(
                    calculated - total
                )}. Please report this model.`
            )
        );
    }
    return card;
}

function renderGenericComponents() {
    const card = element("div", "panel-card");
    const head = element("div", "panel-head");
    head.append(element("h2", null, "Components"));
    const wrap = element("div", "table-wrap");
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const label of ["Component", "Parameters", "Share", "Keys"]) {
        headRow.append(element("th", label === "Component" ? null : "num", label));
    }
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    for (const [name, info] of Object.entries(state.report.components || {})) {
        const row = document.createElement("tr");
        const nameCell = document.createElement("td");
        nameCell.append(element("span", "component-badge", name));
        row.append(
            nameCell,
            numericCell(formatParams(info.params)),
            numericCell(formatPercent(info.percentage)),
            numericCell(formatCount(info.key_count))
        );
        tbody.append(row);
    }
    table.append(thead, tbody);
    wrap.append(table);
    card.append(head, wrap);
    return card;
}

function renderArchitecture() {
    const container = $("architecture-content");
    clear(container);
    state.unetView = null;
    const structure = state.report.structure || {};

    if (structure.kind !== "unet") {
        container.append(
            element(
                "div",
                "callout warn",
                structure.note ||
                    "An architecture-specific view is not available for this model yet."
            )
        );
        container.append(renderGenericComponents());
        container.append(
            element(
                "div",
                "callout",
                "DiT and LLM specialized views are planned after the UNet view. The component table above is derived from the weights."
            )
        );
        return;
    }

    container.append(renderStructureSummary(structure));
    const viewHost = element("div", "panel-card unet-host");
    container.append(viewHost);
    state.unetView = new UnetView(viewHost, {
        structure,
        keys: state.keys,
        fileName: state.report.summary.file_name,
        onViewKeys: viewKeysInTab,
        onToast: showToast,
    });
    container.append(renderIoTable(structure));
    container.append(renderReconciliation(structure));
}

function renderPrinciples() {
    mountPrinciples($("principles-content"), state.report);
}

function appendHighlighted(parent, text, query) {
    const needle = String(query || "").trim();
    if (!needle) {
        parent.textContent = text;
        return;
    }
    const haystack = text.toLowerCase();
    const lowerNeedle = needle.toLowerCase();
    let start = 0;
    let index = haystack.indexOf(lowerNeedle);
    if (index < 0) {
        parent.textContent = text;
        return;
    }
    while (index >= 0) {
        if (index > start) parent.append(document.createTextNode(text.slice(start, index)));
        parent.append(element("mark", null, text.slice(index, index + needle.length)));
        start = index + needle.length;
        index = haystack.indexOf(lowerNeedle, start);
    }
    if (start < text.length) parent.append(document.createTextNode(text.slice(start)));
}

function renderPrefixBar() {
    const bar = $("keys-prefix-bar");
    if (!bar) return;
    const active = state.prefixFilter.length > 0;
    bar.hidden = !active;
    if (active) {
        $("keys-prefix-label").textContent = state.prefixLabel;
    }
}

function viewKeysInTab(prefixes, label) {
    state.prefixFilter = Array.isArray(prefixes) ? prefixes.filter(Boolean) : [];
    state.prefixLabel = label || "";
    $("keys-search").value = "";
    $("keys-component").value = "all";
    $("keys-dtype").value = "all";
    $("keys-attention").checked = false;
    $("keys-sort").value = "name";
    renderPrefixBar();
    activateTab("keys");
    applyKeyFilters();
}

function clearPrefixFilter() {
    state.prefixFilter = [];
    state.prefixLabel = "";
    renderPrefixBar();
    applyKeyFilters();
}

function renderKeys() {
    const componentSelect = $("keys-component");
    const dtypeSelect = $("keys-dtype");
    componentSelect.length = 1;
    dtypeSelect.length = 1;
    for (const name of Object.keys(state.report.components || {})) {
        componentSelect.append(new Option(name, name));
    }
    for (const dtype of dtypesOf(state.keys)) {
        dtypeSelect.append(new Option(dtype, dtype));
    }

    $("keys-search").value = "";
    $("keys-attention").checked = false;
    $("keys-sort").value = "name";
    state.page = 0;
    state.pageSize = Number($("keys-page-size").value) || 100;
    state.filters = {
        query: "",
        component: "all",
        dtype: "all",
        attentionOnly: false,
        sort: "name",
    };
    state.prefixFilter = [];
    state.prefixLabel = "";
    renderPrefixBar();

    if (!state.keys.length) {
        $("keys-status").textContent = "No key list";
        $("keys-empty").hidden = false;
        $("keys-empty").textContent = "This report does not include a key list.";
        $("keys-table-wrap").hidden = true;
        $("keys-pagination").hidden = true;
        clear($("keys-body"));
        $("keys-page-label").textContent = "";
        return;
    }

    $("keys-empty").hidden = true;
    $("keys-table-wrap").hidden = false;
    $("keys-pagination").hidden = false;
    applyKeyFilters();
}

function applyKeyFilters() {
    if (!state.report) return;
    state.filters = {
        query: $("keys-search").value,
        component: $("keys-component").value,
        dtype: $("keys-dtype").value,
        attentionOnly: $("keys-attention").checked,
        sort: $("keys-sort").value,
    };
    state.page = 0;
    let keys = filterKeys(state.keys, state.filters);
    if (state.prefixFilter.length) keys = keysUnderPrefixes(keys, state.prefixFilter);
    state.filteredKeys = sortKeys(keys, state.filters.sort);
    renderKeyTable();
}

function keyRow(key) {
    const row = document.createElement("tr");

    const keyCell = document.createElement("td");
    const copyButton = element("button", "key-name");
    copyButton.type = "button";
    copyButton.title = "Copy key name";
    appendHighlighted(copyButton, key.name, state.filters.query);
    copyButton.addEventListener("click", async () => {
        const copied = await copyText(key.name);
        showToast(copied ? "Key copied to clipboard" : "Could not copy the key");
    });
    keyCell.append(copyButton);

    const componentCell = document.createElement("td");
    componentCell.append(element("span", "component-badge", key.component || "Other"));

    const shapeCell = element("td", "mono", shapeLabel(key.shape));
    shapeCell.title = Array.isArray(key.shape) ? key.shape.join(" × ") : "";
    const dtypeCell = element("td", "mono", key.dtype || "—");
    const paramsCell = numericCell(formatParams(key.params), `${formatCount(key.params)} parameters`);
    if (isAttentionKey(key.name)) row.classList.add("attention-row");

    row.append(keyCell, componentCell, shapeCell, dtypeCell, paramsCell);
    return row;
}

function renderKeyTable() {
    const page = paginate(state.filteredKeys, state.page, state.pageSize);
    state.page = page.page;

    $("keys-status").textContent = page.total
        ? `Showing ${formatCount(page.start + 1)}–${formatCount(page.end)} of ${formatCount(
              page.total
          )} keys`
        : "No matching keys";
    $("keys-page-label").textContent = `Page ${page.page + 1} of ${page.pageCount}`;
    $("keys-prev").disabled = page.page <= 0;
    $("keys-next").disabled = page.page >= page.pageCount - 1;

    const body = $("keys-body");
    clear(body);
    for (const key of page.items) body.append(keyRow(key));

    const hasRows = page.total > 0;
    $("keys-empty").hidden = hasRows;
    $("keys-empty").textContent = "No keys match the current filters.";
    $("keys-table-wrap").hidden = !hasRows;
    $("keys-pagination").hidden = !hasRows;
}

function changePage(delta) {
    state.page += delta;
    renderKeyTable();
}

function renderPrecision() {
    const precision = state.report.precision || {};
    const distributions = precision.distributions || [];
    const body = $("precision-body");
    clear(body);

    for (const item of distributions) {
        const row = document.createElement("tr");
        row.append(
            element("td", null, item.label),
            numericCell(formatParams(item.params), `${formatCount(item.params)} parameters`),
            numericCell(formatPercent(item.percentage))
        );
        body.append(row);
    }
    if (!distributions.length) {
        const row = document.createElement("tr");
        const cell = element("td", "empty", "No precision data is available.");
        cell.colSpan = 3;
        row.append(cell);
        body.append(row);
    }

    const grid = $("precision-components");
    clear(grid);
    const components = precision.components || {};
    for (const [name, entries] of Object.entries(components)) {
        const card = element("div", "precision-card");
        card.append(element("h3", null, name));
        for (const entry of entries || []) {
            const line = element("div", "precision-line");
            const label = element("div", "precision-row");
            label.append(
                element("span", "precision-label", entry.label),
                element("span", "precision-value", formatPercent(entry.percentage))
            );
            const bar = element("div", "bar");
            const fill = document.createElement("span");
            fill.style.width = `${Math.max(0, Math.min(100, Number(entry.percentage) || 0))}%`;
            bar.append(fill);
            line.title = `${formatCount(entry.params)} parameters`;
            line.append(label, bar);
            card.append(line);
        }
        if (!(entries || []).length) card.append(element("p", "muted", "No data."));
        grid.append(card);
    }
    if (!Object.keys(components).length) {
        grid.append(element("p", "muted", "No per-component precision data is available."));
    }
}

function metadataValue(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "string") return value;
    try {
        return JSON.stringify(value);
    } catch (error) {
        return String(value);
    }
}

function renderMetadata() {
    const entries = Object.entries(state.report.metadata || {});
    const body = $("metadata-body");
    clear(body);
    $("metadata-count").textContent = entries.length ? `${entries.length} entries` : "";
    $("metadata-empty").hidden = entries.length > 0;
    $("metadata-empty").textContent = "This model contains no metadata.";
    $("metadata-table-wrap").hidden = entries.length === 0;

    for (const [key, value] of entries) {
        const row = document.createElement("tr");
        const keyCell = element("td", "mono metadata-key", key);
        const valueCell = element("td", "metadata-value", metadataValue(value));
        row.append(keyCell, valueCell);
        body.append(row);
    }
}

function renderRawJson() {
    $("raw-json").textContent = JSON.stringify(state.report, null, 2);
    $("copy-json").disabled = false;
}

function activateTab(name, updateHash = true) {
    const target = TABS.includes(name) ? name : "overview";
    state.activeTab = target;

    document.querySelectorAll("[data-tab]").forEach((button) => {
        const selected = button.dataset.tab === target;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-selected", selected ? "true" : "false");
        button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll("[data-panel]").forEach((panel) => {
        panel.hidden = panel.dataset.panel !== target;
    });

    if (updateHash) {
        const url = new URL(window.location.href);
        url.hash = target;
        window.history.replaceState(null, "", url);
    }

    if (target === "overview") ensureOverviewChart();
    if (target === "precision") ensurePrecisionChart();
    requestAnimationFrame(() => {
        for (const chart of Object.values(state.charts)) chart?.resize();
        if (target === "architecture") state.unetView?.resize();
    });
}

function disposeCharts() {
    for (const chart of Object.values(state.charts)) {
        try {
            chart.dispose();
        } catch (error) {
            // 已销毁的图表无需处理
        }
    }
    state.charts = {};
    clear($("overview-chart"));
    clear($("precision-chart"));
}

function echartsLib() {
    return window.echarts;
}

function showChartFallback(container, message) {
    if (!container) return;
    clear(container);
    container.append(element("p", "empty", message));
}

function ensureOverviewChart() {
    const container = $("overview-chart");
    if (!container || !state.report) return;
    if (state.charts.overview) {
        state.charts.overview.resize();
        return;
    }
    const echarts = echartsLib();
    const data = componentChartData(state.report.components);
    if (!echarts) {
        showChartFallback(container, "Chart library unavailable. The component table carries the same data.");
        return;
    }
    if (!data.length) {
        showChartFallback(container, "No component data to chart.");
        return;
    }

    const chart = echarts.init(container);
    state.charts.overview = chart;
    const ordered = [...data].reverse();
    chart.setOption({
        backgroundColor: "transparent",
        animationDuration: 400,
        grid: { left: 8, right: 72, top: 8, bottom: 8, containLabel: true },
        tooltip: {
            trigger: "axis",
            axisPointer: { type: "shadow" },
            renderMode: "richText",
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            borderColor: GRID,
            textStyle: { color: TEXT },
            formatter: (items) => {
                const list = Array.isArray(items) ? items : [items];
                return list
                    .map((item) => `${item.name}: ${formatParams(item.value)} parameters`)
                    .join("\n");
            },
        },
        xAxis: {
            type: "value",
            axisLabel: { color: MUTED, formatter: (value) => formatParams(value) },
            splitLine: { lineStyle: { color: GRID } },
            axisLine: { show: false },
            axisTick: { show: false },
        },
        yAxis: {
            type: "category",
            data: ordered.map((item) => item.name),
            axisLabel: { color: TEXT },
            axisLine: { lineStyle: { color: GRID } },
            axisTick: { show: false },
        },
        series: [
            {
                type: "bar",
                data: ordered.map((item) => item.value),
                barMaxWidth: 26,
                itemStyle: {
                    borderRadius: [0, 4, 4, 0],
                    color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                        { offset: 0, color: "#0ea5e9" },
                        { offset: 1, color: "#818cf8" },
                    ]),
                },
                label: {
                    show: true,
                    position: "right",
                    color: MUTED,
                    formatter: (item) => formatParams(item.value),
                },
            },
        ],
    });
}

function ensurePrecisionChart() {
    const container = $("precision-chart");
    if (!container || !state.report) return;
    if (state.charts.precision) {
        state.charts.precision.resize();
        return;
    }
    const echarts = echartsLib();
    const data = precisionChartData(state.report.precision?.distributions);
    if (!echarts) {
        showChartFallback(container, "Chart library unavailable. The table carries the same data.");
        return;
    }
    if (!data.length) {
        showChartFallback(container, "No precision data to chart.");
        return;
    }

    const chart = echarts.init(container);
    state.charts.precision = chart;
    chart.setOption({
        backgroundColor: "transparent",
        color: PALETTE,
        animationDuration: 400,
        tooltip: {
            trigger: "item",
            renderMode: "richText",
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            borderColor: GRID,
            textStyle: { color: TEXT },
            formatter: (item) =>
                `${item.data.label}\n${formatParams(item.value)} parameters (${formatPercent(
                    item.data.percentage
                )})`,
        },
        legend: {
            bottom: 0,
            icon: "circle",
            itemWidth: 8,
            itemHeight: 8,
            textStyle: { color: MUTED },
        },
        series: [
            {
                type: "pie",
                radius: ["42%", "68%"],
                center: ["50%", "46%"],
                avoidLabelOverlap: true,
                itemStyle: { borderColor: "#0b1220", borderWidth: 2 },
                label: { color: TEXT, lineHeight: 16, formatter: "{b}\n{d}%" },
                labelLine: { lineStyle: { color: GRID } },
                data: data.map((item) => ({
                    name: item.name,
                    value: item.value,
                    label: item.label,
                    percentage: item.percentage,
                })),
            },
        ],
    });
}

async function copyText(text) {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (error) {
        // 回退到 execCommand
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    let copied = false;
    try {
        copied = document.execCommand("copy");
    } catch (error) {
        copied = false;
    }
    area.remove();
    return copied;
}

let toastTimer = null;

function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 2200);
}

function downloadText(fileName, text, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.rel = "noopener";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportJson() {
    if (!state.report) return;
    downloadText(
        safeFilename(state.report.summary.file_name, "report.json"),
        JSON.stringify(state.report, null, 2),
        "application/json"
    );
    showToast("JSON exported");
}

function exportCsv() {
    if (!state.report) return;
    downloadText(
        safeFilename(state.report.summary.file_name, "keys.csv"),
        keysToCsv(state.keys),
        "text/csv;charset=utf-8"
    );
    showToast("CSV exported");
}

function exportMarkdown() {
    if (!state.report) return;
    downloadText(
        safeFilename(state.report.summary.file_name, "report.md"),
        reportToMarkdown(state.report),
        "text/markdown;charset=utf-8"
    );
    showToast("Markdown exported");
}

function readWidePreference() {
    try {
        return window.localStorage.getItem(WIDE_STORAGE_KEY) === "1";
    } catch {
        return false;
    }
}

function setWideMode(enabled, { persist = true } = {}) {
    document.body.classList.toggle("wide-mode", enabled);
    const button = $("wide-toggle");
    button.setAttribute("aria-pressed", String(enabled));
    button.textContent = enabled ? "Standard width" : "Wide screen";
    button.title = enabled
        ? "Return to the standard 1440 px content width"
        : "Expand the report to the full window width";
    if (persist) {
        try {
            window.localStorage.setItem(WIDE_STORAGE_KEY, enabled ? "1" : "0");
        } catch {
            // Storage may be unavailable; the toggle still works for this page.
        }
    }
    window.requestAnimationFrame(() => {
        for (const chart of Object.values(state.charts)) chart?.resize();
        state.unetView?.resize();
    });
}

function bindKeyControls() {
    let searchTimer = null;
    $("keys-search").addEventListener("input", () => {
        window.clearTimeout(searchTimer);
        searchTimer = window.setTimeout(applyKeyFilters, 150);
    });
    for (const id of ["keys-component", "keys-dtype", "keys-sort", "keys-attention"]) {
        $(id).addEventListener("change", applyKeyFilters);
    }
    $("keys-page-size").addEventListener("change", () => {
        state.pageSize = Number($("keys-page-size").value) || 100;
        state.page = 0;
        renderKeyTable();
    });
    $("keys-prev").addEventListener("click", () => changePage(-1));
    $("keys-next").addEventListener("click", () => changePage(1));
    $("keys-prefix-clear").addEventListener("click", clearPrefixFilter);
}

function init() {
    setWideMode(readWidePreference(), { persist: false });
    $("wide-toggle").addEventListener("click", () => {
        setWideMode(!document.body.classList.contains("wide-mode"));
    });

    document.querySelectorAll("[data-tab]").forEach((button) => {
        button.addEventListener("click", () => activateTab(button.dataset.tab));
    });

    document.querySelector(".tabs").addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const current = Math.max(0, TABS.indexOf(state.activeTab));
        let next = current;
        if (event.key === "ArrowLeft") next = (current - 1 + TABS.length) % TABS.length;
        if (event.key === "ArrowRight") next = (current + 1) % TABS.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = TABS.length - 1;
        const target = document.querySelector(`[data-tab="${TABS[next]}"]`);
        if (target) {
            activateTab(TABS[next]);
            target.focus();
        }
    });

    $("retry-button").addEventListener("click", loadReport);
    $("copy-link").addEventListener("click", async () => {
        const copied = await copyText(window.location.href);
        showToast(copied ? "Report link copied" : "Could not copy the link");
    });
    $("copy-json").addEventListener("click", async () => {
        if (!state.report) return;
        const copied = await copyText(JSON.stringify(state.report, null, 2));
        showToast(copied ? "Report JSON copied" : "Could not copy the JSON");
    });
    $("export-json").addEventListener("click", exportJson);
    $("export-csv").addEventListener("click", exportCsv);
    $("export-markdown").addEventListener("click", exportMarkdown);
    for (const button of document.querySelectorAll(".export-list button")) {
        button.addEventListener("click", () => {
            button.closest("details")?.removeAttribute("open");
        });
    }

    bindKeyControls();
    window.addEventListener("resize", () => {
        for (const chart of Object.values(state.charts)) chart?.resize();
        state.unetView?.resize();
    });

    loadReport();
}

init();
