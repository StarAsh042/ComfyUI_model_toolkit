import { formatParams, ratioLabel } from "../report_logic.js";

const SVG_NS = "http://www.w3.org/2000/svg";

const COLORS = {
    slate: { fill: "rgba(148, 163, 184, 0.08)", stroke: "rgba(148, 163, 184, 0.38)", text: "#e2e8f0", sub: "#94a3b8" },
    cyan: { fill: "rgba(56, 189, 248, 0.09)", stroke: "rgba(56, 189, 248, 0.4)", text: "#bae6fd", sub: "#7dd3fc" },
    violet: { fill: "rgba(167, 139, 250, 0.1)", stroke: "rgba(167, 139, 250, 0.42)", text: "#ddd6fe", sub: "#c4b5fd" },
    amber: { fill: "rgba(251, 191, 36, 0.08)", stroke: "rgba(251, 191, 36, 0.36)", text: "#fde68a", sub: "#fcd34d" },
    green: { fill: "rgba(52, 211, 153, 0.08)", stroke: "rgba(52, 211, 153, 0.36)", text: "#a7f3d0", sub: "#6ee7b7" },
    rose: { fill: "rgba(251, 113, 133, 0.08)", stroke: "rgba(251, 113, 133, 0.36)", text: "#fecdd3", sub: "#fda4af" },
    disabled: { fill: "rgba(148, 163, 184, 0.025)", stroke: "rgba(148, 163, 184, 0.16)", text: "#64748b", sub: "#475569" },
};

const MARKER_COLORS = {
    slate: "#94a3b8",
    cyan: "#38bdf8",
    violet: "#a78bfa",
    amber: "#fbbf24",
    green: "#34d399",
    rose: "#fb7185",
};

export const PRINCIPLE_REFERENCES = [
    {
        id: "rombach2022",
        label: "Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022",
        url: "https://arxiv.org/abs/2112.10752",
    },
    {
        id: "podell2023",
        label: "Podell et al., SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, 2023",
        url: "https://arxiv.org/abs/2307.01952",
    },
    {
        id: "ho2020",
        label: "Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020",
        url: "https://arxiv.org/abs/2006.11239",
    },
    {
        id: "vaswani2017",
        label: "Vaswani et al., Attention Is All You Need, NeurIPS 2017",
        url: "https://arxiv.org/abs/1706.03762",
    },
    {
        id: "peebles2023",
        label: "Peebles & Xie, Scalable Diffusion Models with Transformers, 2023 (planned DiT view)",
        url: "https://arxiv.org/abs/2212.09748",
    },
    {
        id: "esser2024",
        label: "Esser et al., Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, 2024 (planned SD3 view)",
        url: "https://arxiv.org/abs/2403.03206",
    },
];

function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attrs)) {
        if (value !== undefined && value !== null) node.setAttribute(name, String(value));
    }
    if (text !== undefined) node.textContent = String(text);
    return node;
}

function clear(node) {
    if (node) node.replaceChildren();
}

function addText(parent, x, y, text, options = {}) {
    const node = svg(
        "text",
        {
            x,
            y,
            fill: options.fill || COLORS.slate.text,
            "font-size": options.size || 10,
            "font-weight": options.weight || 400,
            "text-anchor": options.anchor || "start",
            "font-family": "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            opacity: options.opacity,
        },
        text
    );
    parent.append(node);
    return node;
}

function createCanvas(width, height, label, prefix) {
    const root = svg("svg", {
        class: "principle-svg",
        viewBox: `0 0 ${width} ${height}`,
        width: "100%",
        role: "img",
        "aria-label": label,
    });
    root.append(svg("desc", {}, label));
    const defs = svg("defs");
    const markers = {};
    for (const [tone, color] of Object.entries(MARKER_COLORS)) {
        const id = `${prefix}-${tone}`;
        const marker = svg("marker", {
            id,
            viewBox: "0 0 10 10",
            refX: 9,
            refY: 5,
            markerWidth: 6,
            markerHeight: 6,
            orient: "auto-start-reverse",
        });
        marker.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color }));
        defs.append(marker);
        markers[tone] = id;
    }
    root.append(defs);
    return { root, markers };
}

function drawBox(parent, x, y, w, h, options = {}) {
    const tone = options.disabled ? "disabled" : options.tone || "slate";
    const colors = COLORS[tone];
    const group = svg("g");
    if (options.tooltip) group.append(svg("title", {}, options.tooltip));
    group.append(
        svg("rect", {
            x,
            y,
            width: w,
            height: h,
            rx: options.rx || 8,
            fill: colors.fill,
            stroke: colors.stroke,
            "stroke-width": 1,
            "stroke-dasharray": options.dashed ? "4 3" : null,
        })
    );
    const lines = options.lines || [];
    const lineHeight = 12;
    const total = (1 + lines.length) * lineHeight;
    const center = options.align === "left" ? null : x + w / 2;
    const textX = options.align === "left" ? x + 10 : center;
    const anchor = options.align === "left" ? "start" : "middle";
    let cursor = y + Math.max(12, (h - total) / 2 + 9);
    addText(group, textX, cursor, options.title || "", {
        fill: colors.text,
        size: options.titleSize || 10,
        weight: 700,
        anchor,
    });
    for (const line of lines) {
        cursor += lineHeight;
        addText(group, textX, cursor, line, {
            fill: colors.sub,
            size: options.lineSize || 9,
            anchor,
        });
    }
    parent.append(group);
    return group;
}

function drawArrow(parent, d, tone, markers, options = {}) {
    const markerId = markers[tone] || markers.slate;
    const node = svg("path", {
        d,
        fill: "none",
        stroke: MARKER_COLORS[tone] || MARKER_COLORS.slate,
        "stroke-width": options.width || 1.25,
        "stroke-dasharray": options.dashed ? "4 3" : null,
        "marker-end": options.noMarker ? null : `url(#${markerId})`,
        opacity: options.opacity,
    });
    parent.append(node);
    return node;
}

function drawConnector(parent, points, tone, markers, options = {}) {
    const d = points
        .map((point, index) => `${index === 0 ? "M" : "L"} ${point[0]} ${point[1]}`)
        .join(" ");
    return drawArrow(parent, d, tone, markers, options);
}

function stageDot(parent, x, y, tone, label) {
    parent.append(
        svg("circle", {
            cx: x,
            cy: y,
            r: 3,
            fill: MARKER_COLORS[tone],
        })
    );
    addText(parent, x + 6, y + 3, label, { fill: MARKER_COLORS[tone], size: 8, weight: 700 });
}

function findContextDim(structure) {
    const stages = [...(structure?.stages || []), structure?.middle, ...(structure?.up || [])];
    for (const stage of stages) {
        for (const block of stage?.blocks || []) {
            if (block?.attention?.context_dim) return block.attention.context_dim;
        }
    }
    return null;
}

export function collectPrincipleFacts(report) {
    const structure = report?.structure || {};
    const capabilities = structure.capabilities || {};
    const stages = structure.stages || [];
    const up = structure.up || [];
    const middle = structure.middle || null;
    const allBlocks = [...stages, middle, ...up]
        .filter(Boolean)
        .flatMap((stage) => stage.blocks || []);
    const resblocks = allBlocks.filter((block) => block.kind === "resblock");
    const transformers = allBlocks.filter((block) => block.kind === "transformer");
    const firstResblock = resblocks[0] || null;
    const firstTransformer = transformers[0] || null;
    const latent = structure.latent || {};
    const extras = structure.extras || [];
    const labelEmbed = extras.find((item) => item.kind === "label_embed") || null;
    const timeEmbed = structure.io?.time_embed || null;
    const unet = report?.components?.UNET || null;
    const archType = report?.architecture?.type || report?.summary?.architecture || "Unknown";
    const dualText = capabilities.dual_text_encoder === true;
    const sizeConditioning = capabilities.size_conditioning === true || Boolean(labelEmbed);
    const refiner = capabilities.refiner === true;
    const contextDim = firstTransformer?.attention?.context_dim || findContextDim(structure);
    const transformerBlocks = firstTransformer?.attention?.transformer_blocks;
    const transformerBlockCounts = [
        ...new Set(
            transformers
                .map((block) => Number(block?.attention?.transformer_blocks))
                .filter((value) => Number.isFinite(value))
        ),
    ].sort((a, b) => a - b);
    const transformerBlocksLabel = transformerBlockCounts.length
        ? transformerBlockCounts.length === 1
            ? String(transformerBlockCounts[0])
            : `${transformerBlockCounts[0]}–${transformerBlockCounts[transformerBlockCounts.length - 1]}`
        : null;
    const heads = firstTransformer?.attention?.heads;
    return {
        archType,
        isSDXL: archType === "SDXL",
        naming: structure.naming || "unknown",
        dualTextEncoder: dualText,
        sizeConditioning,
        refiner,
        vPred: capabilities.v_pred === true,
        contextDim,
        textEncoderName: dualText ? "CLIP-L + CLIP-G" : "CLIP",
        latentChannels: latent.in_channels,
        latentSize: latent.latent_size,
        baseSize: latent.base_size,
        downCount: stages.length,
        upCount: up.length,
        skipCount: up.filter((stage) => stage.skip_from).length,
        middleChannels: middle?.channels ?? null,
        middleRatio: middle?.resolution_ratio ?? null,
        downRows: stages.map((stage) => ({
            level: stage.level,
            channels: stage.channels,
            ratio: stage.resolution_ratio,
            cross: (stage.blocks || []).some((block) => block?.attention?.cross),
        })),
        upRows: up.map((stage) => ({
            level: stage.level,
            channels: stage.channels,
            ratio: stage.resolution_ratio,
            cross: (stage.blocks || []).some((block) => block?.attention?.cross),
        })),
        unetParams: structure.totals?.params ?? unet?.params ?? null,
        unetShare: unet?.percentage ?? null,
        unetKeys: structure.totals?.key_count ?? null,
        timeEmbedParams: timeEmbed?.params ?? null,
        timeEmbedFeatures: timeEmbed?.out_features ?? null,
        labelEmbedParams: labelEmbed?.params ?? null,
        resblockParams: firstResblock?.params ?? null,
        resblockCount: resblocks.length,
        transformerParams: firstTransformer?.params ?? null,
        transformerBlocks,
        transformerBlocksLabel,
        heads,
        headsStored: heads !== null && heads !== undefined,
        hasSelfAttention: capabilities.self_attention === true,
        hasCrossAttention: capabilities.text_cross_attention === true,
        unattributedParams: structure.unattributed?.params ?? 0,
    };
}

function diagramLdm(facts) {
    const { root, markers } = createCanvas(640, 280, "Latent diffusion overview", "pldm");
    drawBox(root, 16, 16, 88, 44, { title: "Prompt", tone: "slate" });
    drawBox(root, 124, 16, 130, 44, {
        title: facts.textEncoderName,
        lines: [`${facts.contextDim ?? "?"}-d context`],
        tone: "violet",
        tooltip: "Text encoder detected in the CLIP component.",
    });
    drawArrow(root, "M 104 38 H 124", "slate", markers);
    drawConnector(root, [[254, 38], [356, 38], [356, 84]], "violet", markers);
    addText(root, 360, 54, "context", { fill: COLORS.violet.sub, size: 8 });

    drawBox(root, 16, 106, 72, 44, { title: "Image", tone: "slate" });
    drawBox(root, 108, 106, 88, 44, { title: "VAE encoder", tone: "cyan" });
    drawBox(root, 216, 112, 48, 32, { title: "z", tone: "slate", titleSize: 11 });
    drawBox(root, 286, 84, 140, 124, {
        title: "UNet",
        lines: [
            formatParams(facts.unetParams),
            `${facts.downCount} down · ${facts.upCount} up`,
            `${facts.contextDim ?? "?"}-d ctx`,
        ],
        tone: "violet",
        titleSize: 13,
        tooltip: "The denoising network. All numbers come from the weights.",
    });
    drawArrow(root, "M 88 128 H 108", "slate", markers);
    drawArrow(root, "M 196 128 H 216", "slate", markers);
    drawArrow(root, "M 264 128 H 286", "slate", markers);
    drawArrow(root, "M 300 84 C 300 46, 412 46, 412 84", "violet", markers);
    addText(root, 356, 42, "denoise × N steps", {
        fill: COLORS.violet.sub,
        size: 8,
        anchor: "middle",
    });

    drawBox(root, 450, 112, 48, 32, { title: "z0", tone: "slate", titleSize: 11 });
    drawBox(root, 518, 106, 88, 44, { title: "VAE decoder", tone: "cyan" });
    drawBox(root, 518, 180, 88, 36, { title: "Image", tone: "green" });
    drawArrow(root, "M 426 128 H 450", "slate", markers);
    drawArrow(root, "M 498 128 H 518", "slate", markers);
    drawArrow(root, "M 562 150 V 180", "green", markers);

    drawBox(root, 160, 196, 86, 40, {
        title: "timestep t",
        lines: [`${facts.timeEmbedFeatures ?? "?"}-d embed`],
        tone: "amber",
        tooltip: "The timestep enters every ResBlock.",
    });
    drawConnector(root, [[246, 216], [356, 216], [356, 208]], "amber", markers);
    addText(root, 252, 208, "time", { fill: COLORS.amber.sub, size: 8 });

    drawBox(root, 420, 196, 120, 40, {
        title: "Refiner",
        lines: [facts.refiner ? "separate UNet" : "not in this checkpoint"],
        tone: facts.refiner ? "rose" : "slate",
        disabled: !facts.refiner,
        tooltip: facts.refiner
            ? "A refiner UNet was detected."
            : "The optional SDXL refiner is a separate checkpoint and is not included here.",
    });
    drawArrow(root, "M 356 208 H 420", "slate", markers, { dashed: !facts.refiner, opacity: 0.7 });
    addText(root, 16, 268, "Schematic is original; see Rombach et al. (2022).", {
        fill: COLORS.disabled.sub,
        size: 8,
    });
    return root;
}

function diagramUnet(facts) {
    const { root, markers } = createCanvas(640, 280, "UNet data flow", "punet");
    const rows = facts.downRows;
    const upRows = facts.upRows;
    const columns = Math.max(rows.length, upRows.length, 1);
    const boxW = Math.max(54, Math.min(92, (430 - (columns - 1) * 12) / columns));
    const gap = 12;
    const startX = 30;
    const downY = 58;
    const upY = 188;
    const boxH = 58;
    const columnX = (index) => startX + index * (boxW + gap);

    addText(root, startX, 46, "down path", { fill: COLORS.cyan.sub, size: 9, weight: 700 });
    addText(root, startX, 272, "up path", { fill: COLORS.green.sub, size: 9, weight: 700 });
    addText(root, 620, 22, `text context ${facts.contextDim ?? "?"}-d → CA`, {
        fill: COLORS.violet.sub,
        size: 9,
        anchor: "end",
    });
    addText(root, 620, 36, "timestep → every ResBlock", {
        fill: COLORS.amber.sub,
        size: 9,
        anchor: "end",
    });

    rows.forEach((row, index) => {
        const x = columnX(index);
        drawBox(root, x, downY, boxW, boxH, {
            title: `L${row.level}`,
            lines: [`${row.channels ?? "?"} ch`, ratioLabel(row.ratio)],
            tone: row.cross ? "cyan" : "slate",
        });
        if (row.cross) stageDot(root, x + boxW - 12, downY + 12, "violet", "ctx");
        if (index > 0) {
            drawArrow(root, `M ${x - gap} ${downY + boxH / 2} H ${x}`, "cyan", markers);
        }
    });
    drawArrow(root, `M 8 ${downY + boxH / 2} H ${startX}`, "slate", markers);
    addText(root, 10, downY - 8, "x_t", { fill: COLORS.slate.sub, size: 8 });

    upRows.forEach((row, index) => {
        const level = Number.isInteger(row.level) && row.level >= 0 && row.level < columns
            ? row.level
            : Math.max(columns - 1 - index, 0);
        const x = columnX(level);
        drawBox(root, x, upY, boxW, boxH, {
            title: `L${row.level}`,
            lines: [`${row.channels ?? "?"} ch`, ratioLabel(row.ratio)],
            tone: row.cross ? "green" : "slate",
        });
        if (row.cross) stageDot(root, x + boxW - 12, upY + 12, "violet", "ctx");
    });

    rows.forEach((row, index) => {
        const upRow = upRows.find((item) => item.level === row.level);
        if (!upRow) return;
        const x = columnX(index) + boxW / 2;
        drawArrow(root, `M ${x} ${downY + boxH} C ${x - 10} ${downY + boxH + 28}, ${x - 10} ${upY - 28}, ${x} ${upY}`, "slate", markers, { dashed: true });
    });

    if (facts.downCount) {
        const lastX = columnX(columns - 1) + boxW;
        drawConnector(root, [[lastX, downY + boxH / 2], [lastX + 22, downY + boxH / 2], [lastX + 22, 128], [496, 128]], "violet", markers);
    }
    if (facts.middleChannels != null) {
        drawBox(root, 496, 106, 118, 72, {
            title: "Middle",
            lines: [`${facts.middleChannels} ch`, ratioLabel(facts.middleRatio)],
            tone: "violet",
        });
        if (upRows.length) {
            const rightLevel = Math.min(Math.max(upRows[0].level, 0), columns - 1);
            const rightX = columnX(rightLevel) + boxW;
            drawConnector(root, [[496, 160], [rightX + 22, 160], [rightX + 22, upY + boxH / 2], [rightX, upY + boxH / 2]], "green", markers);
        }
    }

    upRows.forEach((row, index) => {
        const level = Number.isInteger(row.level) && row.level >= 0 && row.level < columns
            ? row.level
            : Math.max(columns - 1 - index, 0);
        if (level <= 0) return;
        const x = columnX(level);
        const previousRight = columnX(level - 1) + boxW;
        drawArrow(root, `M ${x} ${upY + boxH / 2} H ${previousRight}`, "green", markers);
    });

    if (upRows.length) {
        const leftLevel = upRows.reduce((min, row) => Math.min(min, row.level), columns);
        const leftX = columnX(Math.max(0, Math.min(leftLevel, columns - 1)));
        drawArrow(root, `M ${leftX} ${upY + boxH / 2} H 8`, "slate", markers);
        addText(root, 10, upY - 8, "noise", { fill: COLORS.slate.sub, size: 8 });
    }
    addText(root, 16, 30, "Every stage is drawn from the extracted structure, not a fixed template.", {
        fill: COLORS.disabled.sub,
        size: 8,
    });
    return root;
}

function diagramResblock(facts) {
    const { root, markers } = createCanvas(640, 260, "ResBlock and timestep conditioning", "pres");
    const items = [
        { title: "Input", w: 54, tone: "slate" },
        { title: "GroupNorm", w: 78, tone: "slate" },
        { title: "SiLU", w: 56, tone: "slate" },
        { title: "Conv", w: 56, tone: "cyan" },
        { title: "+ t", w: 44, tone: "amber" },
        { title: "GroupNorm", w: 78, tone: "slate" },
        { title: "SiLU", w: 56, tone: "slate" },
        { title: "Conv", w: 56, tone: "cyan" },
        { title: "+ x", w: 44, tone: "green" },
    ];
    const gap = 8;
    const totalWidth = items.reduce((sum, item) => sum + item.w, 0) + gap * (items.length - 1);
    let x = Math.max(10, (640 - totalWidth) / 2);
    const y = 92;
    const centers = [];
    items.forEach((item, index) => {
        drawBox(root, x, y, item.w, 44, { title: item.title, tone: item.tone, titleSize: 9 });
        centers.push(x + item.w / 2);
        if (index > 0) {
            const previous = centers[index - 1];
            drawArrow(root, `M ${previous + items[index - 1].w / 2} ${y + 22} H ${x}`, "slate", markers);
        }
        x += item.w + gap;
    });
    const finalCenter = centers[centers.length - 1];
    drawArrow(root, `M ${finalCenter} ${y + 22} H 632`, "green", markers);
    addText(root, 12, 30, `${formatParams(facts.resblockParams)} in the first ResBlock of this model`, {
        fill: COLORS.slate.sub,
        size: 9,
    });

    drawBox(root, 219, 186, 180, 44, {
        title: "time_embed",
        lines: [`${facts.timeEmbedFeatures ?? "?"}-d · ${formatParams(facts.timeEmbedParams)}`],
        tone: "amber",
    });
    drawArrow(root, `M 309 186 V ${y + 44}`, "amber", markers);
    addText(root, 316, 176, "scale / shift", { fill: COLORS.amber.sub, size: 8 });

    drawConnector(
        root,
        [[centers[0], y], [centers[0], 56], [finalCenter, 56], [finalCenter, y]],
        "green",
        markers,
        { dashed: true }
    );
    addText(root, (centers[0] + finalCenter) / 2, 50, "residual", {
        fill: COLORS.green.sub,
        size: 8,
        anchor: "middle",
    });
    addText(root, 16, 246, "GroupNorm → SiLU → Conv is applied twice; the timestep is injected between the convolutions.", {
        fill: COLORS.disabled.sub,
        size: 8,
    });
    return root;
}

function diagramTransformer(facts) {
    const { root, markers } = createCanvas(640, 300, "Spatial Transformer block", "ptf");
    const chainY = 98;
    drawBox(root, 12, chainY, 50, 44, { title: "Input", tone: "slate", titleSize: 9 });
    drawBox(root, 78, chainY, 78, 44, { title: "GroupNorm", tone: "slate", titleSize: 9 });
    drawBox(root, 172, chainY, 64, 44, { title: "proj_in", tone: "cyan", titleSize: 9 });
    drawBox(root, 252, 58, 250, 130, { title: "", tone: "violet" });
    addText(root, 262, 78, "Spatial Transformer block", { fill: COLORS.violet.text, size: 10, weight: 700 });
    addText(root, 492, 78, `× ${facts.transformerBlocksLabel ?? "?"}`, {
        fill: COLORS.violet.sub,
        size: 9,
        anchor: "end",
    });
    addText(root, 272, 104, "Norm → Self-attention", { fill: COLORS.violet.sub, size: 9 });
    addText(root, 272, 130, "Norm → Cross-attention", { fill: COLORS.violet.sub, size: 9 });
    addText(root, 272, 156, "Norm → Feed-forward", { fill: COLORS.violet.sub, size: 9 });
    drawBox(root, 518, chainY, 64, 44, { title: "proj_out", tone: "cyan", titleSize: 9 });
    drawBox(root, 598, chainY, 30, 44, { title: "+", tone: "green", titleSize: 12 });
    drawArrow(root, "M 62 120 H 78", "slate", markers);
    drawArrow(root, "M 156 120 H 172", "slate", markers);
    drawArrow(root, "M 236 120 H 252", "slate", markers);
    drawArrow(root, "M 502 120 H 518", "slate", markers);
    drawArrow(root, "M 582 120 H 598", "slate", markers);
    drawArrow(root, "M 628 120 H 638", "green", markers);

    drawBox(root, 280, 222, 190, 44, {
        title: "Text context",
        lines: [`${facts.contextDim ?? "?"}-d`],
        tone: "violet",
    });
    drawArrow(root, "M 375 222 V 188", "violet", markers);
    addText(root, 382, 212, "cross-attention", { fill: COLORS.violet.sub, size: 8 });

    drawConnector(root, [[37, 98], [37, 34], [613, 34], [613, 98]], "green", markers, { dashed: true });
    addText(root, 325, 28, "residual", { fill: COLORS.green.sub, size: 8, anchor: "middle" });
    addText(root, 12, 30, `${formatParams(facts.transformerParams)} in the first Transformer block`, {
        fill: COLORS.slate.sub,
        size: 9,
    });
    addText(root, 16, 286, "Heads: not stored in the weights · never guessed", {
        fill: COLORS.disabled.sub,
        size: 9,
    });
    return root;
}

function diagramSdxl(facts) {
    const { root, markers } = createCanvas(640, 320, "SDXL differences linked to this model", "psdxl");
    addText(root, 624, 18, `This model: ${facts.archType}`, {
        fill: facts.isSDXL ? COLORS.green.sub : COLORS.slate.sub,
        size: 9,
        anchor: "end",
        weight: 700,
    });
    drawBox(root, 16, 38, 90, 42, { title: "Prompt", tone: "slate" });
    drawBox(root, 126, 28, 104, 42, {
        title: facts.dualTextEncoder ? "CLIP-L" : "CLIP",
        lines: [`${facts.contextDim ?? "?"}-d`],
        tone: "violet",
    });
    drawBox(root, 246, 28, 104, 42, {
        title: "CLIP-G",
        lines: [facts.dualTextEncoder ? `${facts.contextDim ?? "?"}-d` : "not detected"],
        tone: facts.dualTextEncoder ? "violet" : "slate",
        disabled: !facts.dualTextEncoder,
    });
    drawArrow(root, "M 106 59 H 126", "slate", markers);
    drawArrow(root, "M 106 59 H 116 V 18 H 298 V 28", "slate", markers, {
        dashed: !facts.dualTextEncoder,
        opacity: facts.dualTextEncoder ? 1 : 0.6,
    });
    drawBox(root, 126, 92, 224, 42, {
        title: "Context",
        lines: [facts.dualTextEncoder ? `${facts.contextDim ?? "?"}-d concatenated` : `${facts.contextDim ?? "?"}-d`],
        tone: "violet",
    });
    drawArrow(root, "M 178 70 V 92", "violet", markers);
    drawArrow(root, "M 298 70 V 92", "violet", markers, { dashed: !facts.dualTextEncoder });

    drawBox(root, 368, 28, 120, 42, {
        title: "Pooled text",
        lines: [facts.dualTextEncoder ? "both encoders" : "from CLIP"],
        tone: "amber",
    });
    drawArrow(
        root,
        facts.dualTextEncoder ? "M 350 49 H 368" : "M 230 49 H 238 V 16 H 428 V 28",
        "amber",
        markers
    );
    drawBox(root, 368, 92, 120, 42, {
        title: "Size conditioning",
        lines: [facts.sizeConditioning ? "label_emb detected" : "not in this model"],
        tone: facts.sizeConditioning ? "amber" : "slate",
        disabled: !facts.sizeConditioning,
    });
    drawArrow(root, "M 428 70 V 92", "amber", markers, { dashed: !facts.sizeConditioning });
    drawBox(root, 368, 154, 120, 44, {
        title: "Conditioning",
        lines: ["pooled text + size"],
        tone: facts.sizeConditioning ? "amber" : "slate",
        disabled: !facts.sizeConditioning,
    });
    drawArrow(root, "M 428 134 V 154", "amber", markers, { dashed: !facts.sizeConditioning });

    drawBox(root, 240, 196, 150, 88, {
        title: "UNet",
        lines: [
            formatParams(facts.unetParams),
            `${facts.downCount} down · ${facts.upCount} up`,
            `${facts.contextDim ?? "?"}-d ctx`,
        ],
        tone: "violet",
        titleSize: 13,
    });
    drawConnector(root, [[238, 134], [238, 168], [300, 168], [300, 196]], "violet", markers);
    drawConnector(root, [[428, 198], [428, 214], [390, 214], [390, 196]], "amber", markers);

    drawBox(root, 60, 196, 120, 52, { title: "VAE decoder", tone: "cyan" });
    drawBox(root, 60, 268, 120, 34, { title: "Image", tone: "green" });
    drawArrow(root, "M 240 230 H 180", "cyan", markers);
    drawArrow(root, "M 120 248 V 268", "green", markers);
    drawBox(root, 430, 204, 130, 52, {
        title: "Refiner",
        lines: [facts.refiner ? "separate checkpoint" : "not in this checkpoint"],
        tone: facts.refiner ? "rose" : "slate",
        disabled: !facts.refiner,
    });
    drawArrow(root, "M 390 230 H 430", "slate", markers, {
        dashed: !facts.refiner,
        opacity: facts.refiner ? 1 : 0.55,
    });
    addText(root, 16, 314, "Gray modules belong to SDXL but are not present in this checkpoint.", {
        fill: COLORS.disabled.sub,
        size: 8,
    });
    return root;
}

function card(title, figure, text, facts, referenceId, badge = "Data-linked schematic") {
    const article = document.createElement("article");
    article.className = "panel-card principle-card";
    const head = document.createElement("div");
    head.className = "principle-head";
    const heading = document.createElement("h2");
    heading.textContent = title;
    const tag = document.createElement("span");
    tag.className = "badge";
    tag.textContent = badge;
    head.append(heading, tag);
    article.append(head);

    const figureWrap = document.createElement("div");
    figureWrap.className = "principle-figure";
    figureWrap.append(figure);
    article.append(figureWrap);

    const paragraph = document.createElement("p");
    paragraph.className = "principle-text";
    paragraph.textContent = text;
    article.append(paragraph);

    const factWrap = document.createElement("div");
    factWrap.className = "principle-facts";
    for (const item of facts) {
        const fact = document.createElement("span");
        fact.className = `principle-fact${item.off ? " off" : ""}`;
        const label = document.createElement("span");
        label.className = "principle-fact-label";
        label.textContent = item.label;
        const value = document.createElement("span");
        value.className = "principle-fact-value";
        value.textContent = item.value;
        fact.append(label, value);
        if (item.title) fact.title = item.title;
        factWrap.append(fact);
    }
    article.append(factWrap);

    const reference = PRINCIPLE_REFERENCES.find((item) => item.id === referenceId);
    if (reference) {
        const source = document.createElement("p");
        source.className = "principle-source";
        source.append(document.createTextNode("Schematic is original. Reference: "));
        const link = document.createElement("a");
        link.href = reference.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = reference.label;
        source.append(link);
        article.append(source);
    }
    return article;
}

export function mountPrinciples(container, report) {
    if (!container) return;
    clear(container);
    const facts = collectPrincipleFacts(report);
    const yesNo = (value) => (value ? "Yes" : "No");

    const callout = document.createElement("div");
    callout.className = "callout";
    callout.textContent =
        "Every diagram is an original SVG schematic, not a copied paper or third-party image. Values, stage counts, channel widths and absent modules are linked to the current report; papers are cited below each diagram.";
    container.append(callout);

    const grid = document.createElement("div");
    grid.className = "principle-grid";

    grid.append(
        card(
            "Latent diffusion overview",
            diagramLdm(facts),
            "A latent diffusion model first compresses the image into a smaller VAE latent. CLIP turns the prompt into a context sequence, and the UNet repeatedly predicts noise (or velocity) inside latent space. The VAE decoder turns the final latent back into pixels. Modules that are not in this checkpoint are grayed out.",
            [
                { label: "Text encoders", value: facts.textEncoderName },
                { label: "Context", value: facts.contextDim ? `${facts.contextDim}-d` : "Not detected" },
                { label: "Refiner", value: yesNo(facts.refiner), off: !facts.refiner },
            ],
            "rombach2022"
        ),
        card(
            "UNet data flow",
            diagramUnet(facts),
            "The down path reduces spatial resolution while increasing channels, the middle block mixes the coarsest features, and the up path restores resolution. Skip connections carry same-resolution features from down to up so fine detail is not lost. Text context enters through cross-attention (ctx), and the timestep enters every ResBlock.",
            [
                { label: "Down", value: String(facts.downCount) },
                { label: "Up", value: String(facts.upCount) },
                { label: "Skips", value: String(facts.skipCount) },
                { label: "Middle", value: facts.middleChannels != null ? `${facts.middleChannels} ch` : "—" },
            ],
            "rombach2022"
        ),
        card(
            "ResBlock and timestep conditioning",
            diagramResblock(facts),
            "Each ResBlock keeps a residual path around two convolutions. GroupNorm and SiLU stabilise and activate features, while the timestep embedding is projected and added between the convolutions. The residual add lets the block learn a correction instead of a full mapping.",
            [
                { label: "ResBlock", value: formatParams(facts.resblockParams) },
                { label: "Time embed", value: facts.timeEmbedFeatures ? `${facts.timeEmbedFeatures}-d` : "—" },
                { label: "Residual", value: "Yes" },
            ],
            "ho2020"
        ),
        card(
            "Spatial Transformer",
            diagramTransformer(facts),
            "Before attention, the spatial tensor is projected to the inner dimension. Self-attention mixes positions, cross-attention reads the CLIP context, and the feed-forward network mixes channels. The wrapper projects back to the spatial tensor. Head counts are not stored in these weights, so the diagram does not guess one.",
            [
                { label: "Context", value: facts.contextDim ? `${facts.contextDim}-d` : "Not detected" },
                {
                    label: "Blocks",
                    value: facts.transformerBlocksLabel ?? "—",
                    title: "Range of Spatial Transformer blocks per attention stage in this UNet.",
                },
                {
                    label: "Heads",
                    value: facts.headsStored ? String(facts.heads) : "Not stored in weights",
                    off: !facts.headsStored,
                    title: "Attention head count is not stored in the weights and is not guessed.",
                },
            ],
            "vaswani2017"
        ),
        card(
            "SDXL differences",
            diagramSdxl(facts),
            facts.isSDXL
                ? "SDXL adds a second text encoder and concatenates both context streams, while pooled text and image size form a separate conditioning vector. Its UNet is larger and uses more attention blocks; the optional refiner is a second UNet in a separate checkpoint. Gray modules are not present in this checkpoint."
                : `This checkpoint is ${facts.archType}, so the highlighted path is what it actually contains. The second text encoder, size conditioning and refiner are SDXL modules and are grayed out here. The same data-link rules still apply: anything not detected is never invented.`,
            [
                { label: "Dual encoder", value: yesNo(facts.dualTextEncoder), off: !facts.dualTextEncoder },
                { label: "Size conditioning", value: yesNo(facts.sizeConditioning), off: !facts.sizeConditioning },
                { label: "Refiner", value: yesNo(facts.refiner), off: !facts.refiner },
                { label: "UNet", value: formatParams(facts.unetParams) },
            ],
            "podell2023",
            facts.isSDXL ? "Data-linked schematic" : "This model highlighted"
        )
    );
    container.append(grid);

    const references = document.createElement("div");
    references.className = "panel-card";
    const heading = document.createElement("h2");
    heading.textContent = "References";
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent =
        "The diagrams are original schematics that follow the papers' ideas and notation conventions. They are not copied figures.";
    const list = document.createElement("ul");
    list.className = "references";
    for (const reference of PRINCIPLE_REFERENCES) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = reference.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = reference.label;
        item.append(link);
        list.append(item);
    }
    references.append(heading, note, list);
    container.append(references);
}
