/**
 * 报告页纯函数：参数解析、格式化、过滤/排序/分页、导出转义。
 * 不访问 DOM，供 report.js 与 Node 单测共用。
 */

export const DEFAULT_REPORT_DIR = "checkpoints";

export const SORT_MODES = [
    "name",
    "name-desc",
    "params",
    "params-desc",
];

const ATTENTION_MARKERS = [
    "attn1.",
    "attn2.",
    "attentions.",
    "transformer_blocks.",
    "attention",
];

const BLOCK_LABELS = {
    resblock: "ResBlock",
    transformer: "Transformer",
    norm: "Norm",
    conv: "Conv",
    time_mlp: "Time MLP",
    skip_proj: "Skip projection",
    self_attn: "Self-attention",
    cross_attn: "Cross-attention",
    ff: "Feed-forward",
    proj_in: "Projection in",
    proj_out: "Projection out",
    other: "Other",
    unknown: "Unknown",
};

const SOURCE_LABELS = {
    weights: "Weights",
    template: "Template",
    convention: "Convention",
    user: "User",
    unknown: "Unknown",
};

const SOURCE_DESCRIPTIONS = {
    weights: "Derived directly from tensor shapes or key names.",
    template: "Identified by the architecture template.",
    convention: "Derived from a common architecture convention, not stored in the weights.",
    user: "Selected in the report interface.",
    unknown: "Not stored in the weights, so it is not guessed.",
};

const CAPABILITY_FIELDS = [
    ["time_embedding", "Time embedding", "Timestep conditioning weights were found."],
    ["self_attention", "Self-attention", "Self-attention blocks were found in the UNet."],
    ["text_cross_attention", "Text cross-attention", "Cross-attention to a text context was found."],
    ["size_conditioning", "Size conditioning", "Pooled size embeddings (label_emb / add_embedding) were found."],
    ["dual_text_encoder", "Dual text encoder", "A second text encoder was detected in the CLIP component."],
    ["refiner", "Refiner", "No SDXL refiner is included in this base checkpoint."],
    ["v_pred", "v-prediction", "A v_prediction marker was found in the file."],
];

export function parseReportParams(search) {
    const params = new URLSearchParams(search || "");
    const dir = (params.get("dir") || "").trim();
    return {
        dir: dir || DEFAULT_REPORT_DIR,
        model: (params.get("model") || "").trim(),
    };
}

export function formatCount(value) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return Math.round(number).toLocaleString("en-US");
}

function trimFixed(value) {
    return value.toFixed(2).replace(/\.?0+$/, "");
}

export function formatParams(value) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const magnitude = Math.abs(number);
    const units = [
        [1e12, "T"],
        [1e9, "B"],
        [1e6, "M"],
        [1e3, "K"],
    ];
    for (const [size, suffix] of units) {
        if (magnitude >= size) {
            return `${trimFixed(number / size)}${suffix}`;
        }
    }
    return String(Math.round(number));
}

export function formatPercent(value) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${trimFixed(number)}%`;
}

export function ratioLabel(ratio) {
    if (ratio === null || ratio === undefined) return "—";
    const value = Number(ratio);
    if (!Number.isFinite(value) || value <= 0) return "—";
    const denominator = Math.round(1 / value);
    const isPowerOfTwo = denominator >= 1 && (denominator & (denominator - 1)) === 0;
    if (isPowerOfTwo && Math.abs(value - 1 / denominator) < 1e-6) {
        return `1/${denominator}`;
    }
    return value.toFixed(3).replace(/\.?0+$/, "");
}

export function shapeLabel(shape) {
    if (!Array.isArray(shape) || shape.length === 0) return "—";
    return `[${shape.join(", ")}]`;
}

export function isAttentionKey(name) {
    const text = String(name || "").toLowerCase();
    return ATTENTION_MARKERS.some((marker) => text.includes(marker));
}

export function dtypesOf(keys) {
    return [...new Set((keys || []).map((key) => key.dtype).filter(Boolean))].sort();
}

export function filterKeys(keys, filters = {}) {
    const query = String(filters.query || "").trim().toLowerCase();
    const component = filters.component || "all";
    const dtype = filters.dtype || "all";
    const attentionOnly = Boolean(filters.attentionOnly);

    return (keys || []).filter((key) => {
        const name = String(key.name || "");
        if (query && !name.toLowerCase().includes(query)) return false;
        if (component !== "all" && key.component !== component) return false;
        if (dtype !== "all" && key.dtype !== dtype) return false;
        if (attentionOnly && !isAttentionKey(name)) return false;
        return true;
    });
}

export function sortKeys(keys, mode = "name") {
    const items = [...(keys || [])];
    const byName = (a, b) =>
        String(a.name).localeCompare(String(b.name), "en", { numeric: true });
    const byParams = (a, b) => (Number(a.params) || 0) - (Number(b.params) || 0);

    switch (mode) {
        case "name-desc":
            return items.sort((a, b) => byName(b, a));
        case "params":
            return items.sort((a, b) => byParams(a, b) || byName(a, b));
        case "params-desc":
            return items.sort((a, b) => byParams(b, a) || byName(a, b));
        default:
            return items.sort(byName);
    }
}

export function paginate(items, page, pageSize) {
    const total = (items || []).length;
    const size = Math.max(1, Math.floor(Number(pageSize) || 1));
    const pageCount = Math.max(1, Math.ceil(total / size));
    const current = Math.min(Math.max(0, Math.floor(Number(page) || 0)), pageCount - 1);
    const start = total ? current * size : 0;
    const end = Math.min(start + size, total);
    return {
        items: (items || []).slice(start, end),
        page: current,
        pageCount,
        total,
        start,
        end,
    };
}

export function componentChartData(components) {
    return Object.entries(components || {})
        .map(([name, info]) => ({
            name,
            value: Number(info?.params) || 0,
            percentage: Number(info?.percentage) || 0,
        }))
        .filter((item) => item.value > 0)
        .sort((a, b) => b.value - a.value || a.name.localeCompare(b.name));
}

export function shortPrecisionLabel(label) {
    const text = String(label || "");
    const fp8 = text.match(/^FP8 \((E[45]M[23])\)/);
    if (fp8) return `FP8 ${fp8[1]}`;
    const short = text.match(/^(BF16|FP32|FP16|INT8|FP8)/);
    return short ? short[1] : text;
}

export function precisionChartData(distributions) {
    return (distributions || [])
        .map((item) => ({
            name: shortPrecisionLabel(item.label),
            label: item.label,
            value: Number(item.params) || 0,
            percentage: Number(item.percentage) || 0,
        }))
        .filter((item) => item.value > 0)
        .sort((a, b) => b.value - a.value);
}

export function capabilitySummary(capabilities) {
    const caps = capabilities || {};
    return CAPABILITY_FIELDS.map(([key, label, description]) => {
        const raw = caps[key];
        const value = raw === true ? "Yes" : raw === false ? "No" : "Unknown";
        return {
            key,
            label,
            value,
            description:
                value === "Unknown"
                    ? "Not stored in this file, so it is not guessed."
                    : description,
        };
    });
}

export function blockLabel(kind) {
    if (!kind) return "Unknown";
    return BLOCK_LABELS[kind] || String(kind);
}

export function sourceLabel(source) {
    return SOURCE_LABELS[source] || "Unknown";
}

export function sourceDescription(source) {
    return SOURCE_DESCRIPTIONS[source] || SOURCE_DESCRIPTIONS.unknown;
}

export function escapeCsvCell(value) {
    let text = value === null || value === undefined ? "" : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
    if (/[",\n\r]/.test(text)) {
        text = `"${text.replace(/"/g, '""')}"`;
    }
    return text;
}

export function keysToCsv(keys) {
    const rows = [["Key", "Component", "Shape", "Dtype", "Params"]];
    for (const key of keys || []) {
        rows.push([
            key.name,
            key.component,
            shapeLabel(key.shape),
            key.dtype,
            key.params,
        ]);
    }
    return rows.map((row) => row.map(escapeCsvCell).join(",")).join("\n");
}

export function escapeMarkdownCell(value) {
    return String(value === null || value === undefined ? "" : value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\|/g, "\\|")
        .replace(/\r?\n/g, "<br>");
}

function markdownValue(value) {
    if (value !== null && typeof value === "object") {
        try {
            return JSON.stringify(value);
        } catch (error) {
            return String(value);
        }
    }
    return value === null || value === undefined ? "" : String(value);
}

function markdownTable(headers, rows, align = []) {
    const header = `| ${headers.map(escapeMarkdownCell).join(" | ")} |`;
    const separator = `| ${headers
        .map((_, index) => {
            const alignment = align[index];
            if (alignment === "right") return "---:";
            if (alignment === "center") return ":---:";
            return "---";
        })
        .join(" | ")} |`;
    const body = rows.map(
        (row) => `| ${row.map(escapeMarkdownCell).join(" | ")} |`
    );
    return [header, separator, ...body].join("\n");
}

export function reportToMarkdown(report) {
    if (!report || !report.summary) {
        return "# Model report\n\nNo report data is available.\n";
    }

    const summary = report.summary;
    const architecture = report.architecture || {};
    const structure = report.structure || {};
    const lines = [];

    lines.push(`# Model report: ${escapeMarkdownCell(summary.file_name)}`, "");
    lines.push(
        markdownTable(
            ["Field", "Value"],
            [
                ["Directory", summary.model_dir],
                ["Architecture", architecture.type || summary.architecture],
                ["Architecture family", architecture.family || "—"],
                [
                    "Parameters",
                    `${summary.total_params_formatted} (${formatCount(summary.total_params)})`,
                ],
                ["Primary precision", summary.primary_precision],
                ["Keys", formatCount(summary.key_count)],
                ["File size", summary.size_formatted],
                [
                    "Template",
                    architecture.template
                        ? `${architecture.template} (${formatPercent(
                              (architecture.template_coverage || 0) * 100
                          )} coverage)`
                        : "None",
                ],
            ]
        ),
        ""
    );

    lines.push("## Components", "");
    const componentRows = Object.entries(report.components || {}).map(
        ([name, info]) => [
            name,
            formatParams(info.params),
            formatCount(info.params),
            formatPercent(info.percentage),
            formatCount(info.key_count),
        ]
    );
    lines.push(
        componentRows.length
            ? markdownTable(
                  ["Component", "Parameters", "Exact params", "Share", "Keys"],
                  componentRows,
                  ["", "right", "right", "right", "right"]
              )
            : "No components were detected.",
        ""
    );

    if (structure.kind === "unet") {
        lines.push("## Architecture", "");
        const latent = structure.latent || {};
        const capabilities = structure.capabilities || {};
        lines.push(
            markdownTable(
                ["Field", "Value"],
                [
                    ["Key naming", structure.naming === "ldm" ? "LDM / CompVis" : "Diffusers"],
                    ["Base size", latent.base_size ? `${latent.base_size} px (convention)` : "Unknown"],
                    ["Latent channels", latent.in_channels ?? "Unknown"],
                    ["Latent size", latent.latent_size ? `${latent.latent_size} px at 1/8` : "Unknown"],
                    [
                        "Text context",
                        findContextDim([...(structure.stages || []), structure.middle, ...(structure.up || [])]),
                    ],
                    ["Self-attention", capabilities.self_attention ? "Yes" : "No"],
                    ["Text cross-attention", capabilities.text_cross_attention ? "Yes" : "No"],
                    ["Size conditioning", capabilities.size_conditioning ? "Yes" : "No"],
                ]
            ),
            ""
        );

        const stageRows = [];
        for (const stage of [...(structure.stages || []), structure.middle, ...(structure.up || [])]) {
            if (!stage) continue;
            stageRows.push([
                stage.direction,
                stage.id,
                stage.level ?? "—",
                ratioLabel(stage.resolution_ratio),
                stage.channels ?? "Unknown",
                formatParams(stage.params),
                formatCount(stage.key_count),
            ]);
        }
        if (stageRows.length) {
            lines.push("### Stages", "");
            lines.push(
                markdownTable(
                    ["Direction", "Stage", "Level", "Resolution", "Channels", "Params", "Keys"],
                    stageRows,
                    ["", "", "right", "right", "right", "right", "right"]
                ),
                ""
            );
        }
    } else if (structure.note) {
        lines.push("## Architecture", "", escapeMarkdownCell(structure.note), "");
    }

    lines.push("## Precision", "");
    const precisionRows = (report.precision?.distributions || []).map((item) => [
        item.label,
        formatParams(item.params),
        formatPercent(item.percentage),
    ]);
    lines.push(
        precisionRows.length
            ? markdownTable(
                  ["Precision", "Parameters", "Share"],
                  precisionRows,
                  ["", "right", "right"]
              )
            : "No precision data is available.",
        ""
    );

    lines.push("## Metadata", "");
    const metadataRows = Object.entries(report.metadata || {}).map(([key, value]) => [
        key,
        markdownValue(value),
    ]);
    lines.push(
        metadataRows.length
            ? markdownTable(["Key", "Value"], metadataRows)
            : "No metadata is stored in this file.",
        ""
    );

    lines.push(
        `_Generated by ComfyUI Model Toolkit · report schema ${escapeMarkdownCell(
            report.schema_version || ""
        )}._`,
        ""
    );

    return lines.join("\n");
}

function findContextDim(stages) {
    for (const stage of stages) {
        for (const block of stage?.blocks || []) {
            const context = block?.attention?.context_dim;
            if (context) return context;
        }
    }
    return "None";
}

export function reportBaseName(fileName) {
    const base = String(fileName || "")
        .split(/[\\/]/)
        .pop();
    if (!base) return "model";
    const dot = base.lastIndexOf(".");
    const stem = dot > 0 ? base.slice(0, dot) : base;
    return stem || "model";
}

export function safeFilename(fileName, extension) {
    const stem = reportBaseName(fileName)
        .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
        .replace(/[. ]+$/g, "")
        .trim();
    return `${stem || "model"}.${extension}`;
}
