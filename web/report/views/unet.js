import {
    blockLabel,
    formatCount,
    formatParams,
    formatPercent,
    keysUnderPrefixes,
    normalizePrefix,
    ratioLabel,
    safeFilename,
    shapeLabel,
    sourceDescription,
    sourceLabel,
} from "../report_logic.js";

export { keysUnderPrefixes };

const SVG_NS = "http://www.w3.org/2000/svg";
const PAD = 24;
const IO_WIDTH = 176;
const STAGE_WIDTH = 216;
const COL_GAP = 30;
const TOP_BAND = 88;
const ROW_GAP = 112;
const HEADER_H = 64;
const BLOCK_H = 26;
const BLOCK_GAP = 4;
const TIME_BAND_H = 30;
const IO_PILL_WIDTH = IO_WIDTH - 34;

export const UNET_METRICS = Object.freeze({
    PAD,
    IO_WIDTH,
    STAGE_WIDTH,
    COL_GAP,
    TOP_BAND,
    ROW_GAP,
    HEADER_H,
    BLOCK_H,
    BLOCK_GAP,
    TIME_BAND_H,
    IO_PILL_WIDTH,
});

let instanceSeq = 0;

export function defaultBaseSize(structure) {
    const size = Number(structure?.latent?.base_size);
    return Number.isFinite(size) && size > 0 ? Math.round(size) : 512;
}

export function absoluteStageSize(baseSize, ratio) {
    const base = Number(baseSize);
    const factor = Number(ratio);
    if (!Number.isFinite(base) || !Number.isFinite(factor) || base <= 0 || factor <= 0) {
        return null;
    }
    return Math.round(base * factor);
}

export function stageRoleLabel(stage) {
    if (!stage) return "Stage";
    if (stage.direction === "middle") return `Middle L${stage.level}`;
    if (stage.direction === "up") return `Up L${stage.level}`;
    return `Down L${stage.level}`;
}

export function keyPrefixForBlock(structure, blockId) {
    const prefix = normalizePrefix(structure?.prefix || "");
    const block = String(blockId || "").replace(/^\.+/, "");
    return normalizePrefix(`${prefix}${block}`);
}

export function stageKeyPrefixes(structure, stage) {
    return (stage?.blocks || [])
        .map((block) => keyPrefixForBlock(structure, block.id))
        .filter(Boolean);
}

export function findContextDim(structure) {
    const stages = [...(structure?.stages || []), structure?.middle, ...(structure?.up || [])];
    for (const stage of stages) {
        if (!stage) continue;
        for (const block of stage.blocks || []) {
            if (block?.attention?.context_dim) return block.attention.context_dim;
        }
    }
    return null;
}

function stageHasCross(stage) {
    return (stage?.blocks || []).some((block) => block?.attention?.cross);
}

function stageHasSelf(stage) {
    return (stage?.blocks || []).some((block) => block?.attention?.self);
}

function stageHasRes(stage) {
    return (stage?.blocks || []).some((block) => block.kind === "resblock");
}

function cardHeight(stage, collapsed) {
    if (collapsed) return HEADER_H + 20;
    const rows = Math.max((stage?.blocks || []).length, 1);
    const footer = stage?.transition ? 18 : 0;
    return HEADER_H + rows * (BLOCK_H + BLOCK_GAP) + footer + 8;
}

export function layoutUnet(structure, options = {}) {
    const down = Array.isArray(structure?.stages) ? structure.stages : [];
    const up = Array.isArray(structure?.up) ? structure.up : [];
    const middle = structure?.middle || null;
    const collapsed = options.collapsed instanceof Set
        ? options.collapsed
        : new Set(options.collapsed || []);
    const columns = Math.max(down.length, up.length, 1);
    const columnX = (index) => PAD + IO_WIDTH + index * (STAGE_WIDTH + COL_GAP);

    const makeCard = (stage, role, column) => {
        const isCollapsed = collapsed.has(stage.id);
        return {
            stage,
            role,
            column,
            x: columnX(column),
            y: role === "down" ? TOP_BAND : 0,
            w: STAGE_WIDTH,
            h: cardHeight(stage, isCollapsed),
            collapsed: isCollapsed,
            blockRows: [],
        };
    };

    const downCards = down.map((stage, index) => makeCard(stage, "down", index));
    const upCards = up.map((stage, index) => {
        const level = Number(stage.level);
        const column =
            Number.isInteger(level) && level >= 0 && level < columns
                ? level
                : Math.max(columns - 1 - index, 0);
        return makeCard(stage, "up", column);
    });
    const middleCard = middle ? makeCard(middle, "middle", columns) : null;

    const topRowBottom = Math.max(
        TOP_BAND + 140,
        ...downCards.map((card) => card.y + card.h)
    );
    const bottomRowH = Math.max(140, ...upCards.map((card) => card.h));
    const upY = topRowBottom + ROW_GAP;
    for (const card of upCards) card.y = upY;
    if (middleCard) {
        const available = topRowBottom - TOP_BAND + ROW_GAP + bottomRowH;
        middleCard.y = TOP_BAND + Math.max(0, (available - middleCard.h) / 2);
    }
    for (const card of [...downCards, ...upCards, ...(middleCard ? [middleCard] : [])]) {
        if (card.collapsed) continue;
        card.blockRows = (card.stage.blocks || []).map((block, index) => ({
            block,
            index,
            x: card.x + 8,
            y: card.y + HEADER_H + index * (BLOCK_H + BLOCK_GAP),
            w: STAGE_WIDTH - 16,
            h: BLOCK_H,
        }));
    }

    const width = columnX(columns) + STAGE_WIDTH + PAD;
    const bottomBandY = upY + bottomRowH + 18;
    const height = bottomBandY + TIME_BAND_H + PAD;
    const contextBand = {
        x: columnX(0),
        y: 14,
        w: columnX(columns) + STAGE_WIDTH - columnX(0),
        h: 32,
    };
    const timeBand = {
        x: PAD,
        y: bottomBandY,
        w: width - PAD * 2,
        h: TIME_BAND_H,
    };

    const downById = new Map(downCards.map((card) => [card.stage.id, card]));
    const skipLinks = [];
    upCards.forEach((card, index) => {
        const from = downById.get(card.stage.skip_from);
        if (!from) return;
        const x1 = from.x + from.w / 2;
        const y1 = from.y + from.h;
        const x2 = card.x + card.w / 2;
        const y2 = card.y;
        const bow = (index % 2 === 0 ? 1 : -1) * (16 + index * 4);
        skipLinks.push({
            from: from.stage.id,
            to: card.stage.id,
            x1,
            y1,
            x2,
            y2,
            path: `M ${x1} ${y1} C ${x1 + bow} ${y1 + 40}, ${x2 + bow} ${y2 - 40}, ${x2} ${y2}`,
            labelX: (x1 + x2) / 2 + bow,
            labelY: (y1 + y2) / 2,
        });
    });

    const contextLinks = [];
    for (const card of [...downCards, ...(middleCard ? [middleCard] : [])]) {
        if (!stageHasCross(card.stage)) continue;
        contextLinks.push({
            stage: card.stage.id,
            x1: card.x + card.w / 2,
            y1: contextBand.y + contextBand.h,
            x2: card.x + card.w / 2,
            y2: card.y,
        });
    }

    const railX = PAD + IO_WIDTH - 14;
    const timeRail = downCards.length
        ? {
              x: railX,
              path: `M ${railX} ${timeBand.y} V ${TOP_BAND + 44}`,
              stubs: downCards.map((card) => ({
                  x1: railX,
                  y1: card.y + 44,
                  x2: card.x,
                  y2: card.y + 44,
              })),
          }
        : null;
    const timeLinks = [];
    for (const card of [...upCards, ...(middleCard ? [middleCard] : [])]) {
        if (!stageHasRes(card.stage)) continue;
        timeLinks.push({
            stage: card.stage.id,
            x1: card.x + card.w / 2,
            y1: timeBand.y,
            x2: card.x + card.w / 2,
            y2: card.y + card.h,
        });
    }

    const firstDown = downCards.reduce(
        (best, card) => (!best || card.x < best.x ? card : best),
        null
    );
    const leftmostUp = upCards.reduce(
        (best, card) => (!best || card.x < best.x ? card : best),
        null
    );
    const latentIn = { x: PAD, y: TOP_BAND + 8, w: IO_PILL_WIDTH, h: 38 };
    const convIn = { x: PAD, y: TOP_BAND + 58, w: IO_PILL_WIDTH, h: 38 };
    const convOut = { x: PAD, y: upY + bottomRowH - 92, w: IO_PILL_WIDTH, h: 38 };
    const latentOut = { x: PAD, y: upY + bottomRowH - 46, w: IO_PILL_WIDTH, h: 38 };
    const io = {
        latentIn,
        convIn,
        convOut,
        latentOut,
        links: [],
    };
    io.links.push({
        kind: "latent",
        path: `M ${latentIn.x + latentIn.w / 2} ${latentIn.y + latentIn.h} V ${convIn.y}`,
    });
    if (firstDown) {
        io.links.push({
            kind: "latent",
            path: `M ${convIn.x + convIn.w} ${convIn.y + convIn.h / 2} H ${firstDown.x}`,
        });
    }
    if (leftmostUp) {
        io.links.push({
            kind: "latent",
            path: `M ${leftmostUp.x} ${convOut.y + convOut.h / 2} H ${convOut.x + convOut.w}`,
        });
    }
    io.links.push({
        kind: "latent",
        path: `M ${convOut.x + convOut.w / 2} ${convOut.y + convOut.h} V ${latentOut.y}`,
    });

    return {
        width,
        height,
        columns,
        downCards,
        upCards,
        middleCard,
        cards: [...downCards, ...(middleCard ? [middleCard] : []), ...upCards],
        topRowBottom,
        upY,
        bottomRowH,
        contextBand,
        timeBand,
        skipLinks,
        contextLinks,
        timeRail,
        timeLinks,
        io,
        contextDim: findContextDim(structure),
        dualTextEncoder: structure?.capabilities?.dual_text_encoder === true,
        hasSelfAttention: structure?.capabilities?.self_attention === true,
        hasCrossAttention: structure?.capabilities?.text_cross_attention === true,
        labelEmbed: (structure?.extras || []).find((item) => item.kind === "label_embed") || null,
        vPred: structure?.capabilities?.v_pred === true,
        stageHasCross,
        stageHasSelf,
        stageHasRes,
    };
}

function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attrs)) {
        if (value !== undefined && value !== null) node.setAttribute(name, String(value));
    }
    if (text !== undefined) node.textContent = String(text);
    return node;
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
}

function clear(node) {
    if (node) node.replaceChildren();
}

function sourceTag(source) {
    if (!source) return null;
    const node = el("span", "source", sourceLabel(source));
    node.title = sourceDescription(source);
    return node;
}

function detailChip(label, value, source, title) {
    const chip = el("span", "unet-detail-chip");
    chip.append(el("span", "unet-detail-chip-label", label));
    const valueNode = el("span", "unet-detail-chip-value", value);
    if (title) valueNode.title = title;
    chip.append(valueNode);
    const badge = sourceTag(source);
    if (badge) chip.append(badge);
    return chip;
}

function detailTable(headers, rows, numericColumns = []) {
    const wrap = el("div", "unet-detail-table");
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    headers.forEach((header, index) => {
        headRow.append(el("th", numericColumns.includes(index) ? "num" : null, header));
    });
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    for (const row of rows) {
        const tr = document.createElement("tr");
        row.forEach((value, index) => {
            tr.append(el("td", numericColumns.includes(index) ? "num" : null, value));
        });
        tbody.append(tr);
    }
    table.append(thead, tbody);
    wrap.append(table);
    return wrap;
}

function downloadBlob(fileName, blob) {
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

export class UnetView {
    constructor(host, options = {}) {
        this.host = host;
        this.structure = options.structure || {};
        this.keys = Array.isArray(options.keys) ? options.keys : [];
        this.fileName = options.fileName || "model";
        this.onViewKeys = options.onViewKeys || (() => {});
        this.onToast = options.onToast || (() => {});
        this.baseSize = defaultBaseSize(this.structure);
        this.showParams = true;
        this.showShare = true;
        this.structureOnly = false;
        this.selected = null;
        this.scale = 1;
        this.autoFit = true;
        this.layout = null;
        this.svg = null;
        this.suppressUntil = 0;
        this.instanceId = ++instanceSeq;
        const compact = window.matchMedia("(max-width: 720px)").matches;
        this.collapsed = compact
            ? new Set(this.allStages().map((stage) => stage.id))
            : new Set();
        this.buildDom();
        this.renderSvg();
        this.renderDetail();
        window.requestAnimationFrame(() => this.fit());
    }

    allStages() {
        return [
            ...(this.structure.stages || []),
            ...(this.structure.middle ? [this.structure.middle] : []),
            ...(this.structure.up || []),
        ];
    }

    buildDom() {
        this.host.classList.add("unet-view");
        clear(this.host);

        const head = el("div", "unet-view-head");
        const titles = el("div");
        titles.append(
            el("h2", null, "Interactive UNet architecture"),
            el(
                "p",
                "muted",
                "Self-drawn SVG · every number comes from the weights · click a stage or block to inspect it"
            )
        );
        head.append(titles);
        this.host.append(head);

        const toolbar = el("div", "unet-toolbar");
        const baseField = el("label", "unet-field");
        baseField.append(el("span", "unet-field-label", "Base resolution"));
        const baseWrap = el("span", "unet-number");
        this.baseInput = document.createElement("input");
        this.baseInput.type = "number";
        this.baseInput.min = "64";
        this.baseInput.max = "8192";
        this.baseInput.step = "64";
        this.baseInput.value = String(this.baseSize);
        this.baseInput.setAttribute("aria-label", "Base image resolution in pixels");
        baseWrap.append(this.baseInput, el("span", "unet-unit", "px"));
        baseField.append(baseWrap);
        baseField.title =
            "Absolute stage sizes are image-space pixels. The weights only define relative factors.";
        toolbar.append(baseField);

        const toggles = el("div", "unet-controls");
        this.paramsButton = this.toggleButton("Params", true, (on) => {
            this.showParams = on;
            if (!on) this.showShare = false;
            this.syncToggleState();
            this.renderSvg();
        });
        this.shareButton = this.toggleButton("Share", true, (on) => {
            this.showShare = on;
            this.syncToggleState();
            this.renderSvg();
        });
        this.structureButton = this.toggleButton("Structure only", false, (on) => {
            this.structureOnly = on;
            this.showParams = !on;
            this.showShare = !on;
            this.syncToggleState();
            this.renderSvg();
        });
        toggles.append(this.paramsButton, this.shareButton, this.structureButton);
        toolbar.append(toggles);

        const zoom = el("div", "unet-controls unet-zoom");
        const zoomOut = el("button", "unet-tool", "−");
        zoomOut.type = "button";
        zoomOut.title = "Zoom out";
        zoomOut.addEventListener("click", () => this.setScale(this.scale / 1.15));
        const fit = el("button", "unet-tool", "Fit");
        fit.type = "button";
        fit.title = "Fit the diagram to the panel";
        fit.addEventListener("click", () => this.fit());
        const zoomIn = el("button", "unet-tool", "+");
        zoomIn.type = "button";
        zoomIn.title = "Zoom in";
        zoomIn.addEventListener("click", () => this.setScale(this.scale * 1.15));
        const exportButton = el("button", "unet-tool", "Export SVG");
        exportButton.type = "button";
        exportButton.addEventListener("click", () => this.exportSvg());
        zoom.append(zoomOut, fit, zoomIn, exportButton);
        toolbar.append(zoom);
        this.host.append(toolbar);

        const layout = el("div", "unet-layout");
        const main = el("div", "unet-main");
        this.canvasScroll = el("div", "unet-canvas");
        this.canvasScroll.tabIndex = 0;
        this.canvasScroll.setAttribute(
            "aria-label",
            "Scrollable UNet architecture diagram. Drag to pan, Ctrl or Command plus wheel to zoom."
        );
        this.canvasScroll.addEventListener("pointerdown", (event) => this.beginDrag(event));
        this.canvasScroll.addEventListener("pointermove", (event) => this.moveDrag(event));
        this.canvasScroll.addEventListener("pointerup", (event) => this.endDrag(event));
        this.canvasScroll.addEventListener("pointercancel", (event) => this.endDrag(event));
        this.canvasScroll.addEventListener(
            "wheel",
            (event) => {
                if (!event.ctrlKey && !event.metaKey) return;
                event.preventDefault();
                const factor = event.deltaY < 0 ? 1.1 : 0.9;
                this.setScale(this.scale * factor, event.clientX, event.clientY);
            },
            { passive: false }
        );
        this.drag = null;
        main.append(this.canvasScroll);

        const hint = el(
            "p",
            "unet-hint",
            "Drag to pan · Ctrl/⌘ + wheel to zoom · relative resolutions are weight facts, absolute sizes use the selected base."
        );
        main.append(hint);
        main.append(this.legend());

        this.detail = el("aside", "unet-detail");
        this.detail.setAttribute("aria-live", "polite");
        layout.append(main, this.detail);
        this.host.append(layout);

        this.baseInput.addEventListener("change", () => {
            const value = Number(this.baseInput.value);
            this.baseSize = Number.isFinite(value)
                ? Math.max(64, Math.min(8192, Math.round(value)))
                : defaultBaseSize(this.structure);
            this.baseInput.value = String(this.baseSize);
            this.renderSvg();
            this.renderDetail();
        });
        this.baseInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") this.baseInput.blur();
        });
        this.syncToggleState();
    }

    toggleButton(label, initial, onChange) {
        const button = el("button", "unet-toggle", label);
        button.type = "button";
        button.setAttribute("aria-pressed", String(initial));
        button.classList.toggle("active", initial);
        button.addEventListener("click", () => {
            if (button.disabled) return;
            const next = button.getAttribute("aria-pressed") !== "true";
            button.setAttribute("aria-pressed", String(next));
            button.classList.toggle("active", next);
            onChange(next);
        });
        return button;
    }

    syncToggleState() {
        this.paramsButton.disabled = this.structureOnly;
        this.shareButton.disabled = this.structureOnly || !this.showParams;
        this.shareButton.classList.toggle("muted-control", !this.showParams);
        this.baseInput.disabled = false;
    }

    legend() {
        const legend = el("div", "unet-legend");
        const items = [
            ["resblock", "ResBlock"],
            ["transformer", "Transformer"],
            ["ctx", "ctx = text cross-attention"],
            ["ca", "CA = cross-attention"],
            ["skip", "dashed = skip / conditioning"],
        ];
        for (const [tone, text] of items) {
            const item = el("span", "unet-legend-item");
            const swatch = el("span", `unet-legend-swatch ${tone}`);
            item.append(swatch, el("span", null, text));
            legend.append(item);
        }
        return legend;
    }

    renderSvg() {
        const scrollLeft = this.canvasScroll.scrollLeft;
        const scrollTop = this.canvasScroll.scrollTop;
        const layout = layoutUnet(this.structure, { collapsed: this.collapsed });
        this.layout = layout;
        const prefix = `unet${this.instanceId}`;
        const root = svg("svg", {
            class: "unet-svg",
            viewBox: `0 0 ${layout.width} ${layout.height}`,
            role: "img",
            "aria-label": "UNet architecture diagram with down, middle and up stages",
            "data-unet-root": "true",
        });
        root.append(
            svg(
                "title",
                {},
                "UNet architecture. Down and up stages, skip connections, text and time conditioning."
            )
        );
        root.append(svg("desc", {}, "All parameter counts and key counts come from the model weights."));
        root.append(svg("style", {}, this.svgStyles()));

        const defs = svg("defs");
        const markers = {
            slate: this.marker(defs, `${prefix}-arrow-slate`, "#7c8ba1"),
            context: this.marker(defs, `${prefix}-arrow-context`, "#a78bfa"),
            time: this.marker(defs, `${prefix}-arrow-time`, "#fbbf24"),
            skip: this.marker(defs, `${prefix}-arrow-skip`, "#94a3b8"),
        };
        root.append(defs);

        root.append(
            svg("rect", {
                class: "unet-bg",
                x: 0,
                y: 0,
                width: layout.width,
                height: layout.height,
            })
        );

        root.append(this.renderContextBand(layout));
        root.append(this.renderTimeBand(layout));

        const connectors = svg("g", { class: "unet-connectors" });
        for (const link of layout.io.links) {
            connectors.append(
                svg("path", {
                    class: "unet-link",
                    d: link.path,
                    "marker-end": `url(#${markers.slate})`,
                })
            );
        }
        for (const link of layout.skipLinks) {
            connectors.append(
                svg("path", {
                    class: "unet-link skip",
                    d: link.path,
                    "marker-end": `url(#${markers.skip})`,
                })
            );
            connectors.append(
                svg(
                    "text",
                    {
                        class: "unet-skip-label",
                        x: link.labelX,
                        y: link.labelY,
                        "text-anchor": "middle",
                    },
                    "skip"
                )
            );
        }
        for (const link of layout.contextLinks) {
            connectors.append(
                svg("path", {
                    class: "unet-link context",
                    d: `M ${link.x1} ${link.y1} V ${link.y2}`,
                    "marker-end": `url(#${markers.context})`,
                })
            );
        }
        if (layout.timeRail) {
            connectors.append(
                svg("path", {
                    class: "unet-link time",
                    d: layout.timeRail.path,
                })
            );
            for (const stub of layout.timeRail.stubs) {
                connectors.append(
                    svg("path", {
                        class: "unet-link time",
                        d: `M ${stub.x1} ${stub.y1} H ${stub.x2}`,
                        "marker-end": `url(#${markers.time})`,
                    })
                );
            }
        }
        for (const link of layout.timeLinks) {
            connectors.append(
                svg("path", {
                    class: "unet-link time",
                    d: `M ${link.x1} ${link.y1} V ${link.y2}`,
                    "marker-end": `url(#${markers.time})`,
                })
            );
        }
        root.append(connectors);

        for (const card of layout.cards) {
            root.append(this.renderCard(card));
        }
        root.append(this.renderIo(layout));

        clear(this.canvasScroll);
        this.canvasScroll.append(root);
        this.svg = root;
        this.applyScale(this.scale);
        this.canvasScroll.scrollLeft = scrollLeft;
        this.canvasScroll.scrollTop = scrollTop;
        this.bindSvgEvents(root);
        this.updateSelectionClasses();
    }

    marker(defs, id, color) {
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
        return id;
    }

    svgStyles() {
        return `
            .unet-svg { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
            .unet-svg text { fill: #e2e8f0; }
            .unet-bg { fill: #070d18; }
            .unet-stage { cursor: pointer; }
            .unet-stage-bg { fill: #0b1526; stroke: rgba(148, 163, 184, 0.24); stroke-width: 1; }
            .unet-stage:hover .unet-stage-bg { stroke: rgba(56, 189, 248, 0.62); }
            .unet-stage.selected .unet-stage-bg { stroke: #38bdf8; stroke-width: 2; }
            .unet-stage-title { font-size: 12px; font-weight: 700; letter-spacing: 0.02em; }
            .unet-stage-res { fill: #cbd5e1; font-size: 10px; }
            .unet-stage-meta { fill: #94a3b8; font-size: 10px; }
            .unet-stage-params { fill: #cbd5e1; font-size: 10px; font-weight: 600; }
            .unet-ctx-port { fill: #c4b5fd; font-size: 9px; font-weight: 800; letter-spacing: 0.06em; }
            .unet-block { cursor: pointer; }
            .unet-block-bg { fill: rgba(56, 189, 248, 0.07); stroke: rgba(56, 189, 248, 0.27); stroke-width: 1; }
            .unet-block.transformer .unet-block-bg { fill: rgba(167, 139, 250, 0.1); stroke: rgba(167, 139, 250, 0.4); }
            .unet-block:hover .unet-block-bg { stroke: rgba(226, 232, 240, 0.75); }
            .unet-block.selected .unet-block-bg { stroke: #f8fafc; stroke-width: 1.6; }
            .unet-block-kind { fill: #e2e8f0; font-size: 10px; font-weight: 600; }
            .unet-block-params { fill: #94a3b8; font-size: 9px; }
            .unet-badge-text { font-size: 8px; font-weight: 800; letter-spacing: 0.04em; }
            .unet-badge-sa { fill: rgba(52, 211, 153, 0.18); stroke: rgba(52, 211, 153, 0.45); }
            .unet-badge-sa-text { fill: #a7f3d0; }
            .unet-badge-ca { fill: rgba(251, 113, 133, 0.18); stroke: rgba(251, 113, 133, 0.45); }
            .unet-badge-ca-text { fill: #fecdd3; }
            .unet-transition { fill: #fbbf24; font-size: 9px; }
            .unet-collapse { cursor: pointer; }
            .unet-collapse-bg { fill: rgba(148, 163, 184, 0.1); stroke: rgba(148, 163, 184, 0.3); }
            .unet-collapse-text { fill: #cbd5e1; font-size: 12px; font-weight: 700; }
            .unet-band { fill: rgba(129, 140, 248, 0.09); stroke: rgba(129, 140, 248, 0.34); }
            .unet-band.time { fill: rgba(251, 191, 36, 0.06); stroke: rgba(251, 191, 36, 0.26); }
            .unet-band-label { fill: #ddd6fe; font-size: 11px; font-weight: 600; }
            .unet-band-label.time { fill: #fde68a; }
            .unet-band-sub { fill: #94a3b8; font-size: 9px; }
            .unet-link { fill: none; stroke: rgba(148, 163, 184, 0.55); stroke-width: 1.2; }
            .unet-link.context { stroke: rgba(167, 139, 250, 0.7); stroke-dasharray: 4 3; }
            .unet-link.time { stroke: rgba(251, 191, 36, 0.6); stroke-dasharray: 3 3; }
            .unet-link.skip { stroke: rgba(148, 163, 184, 0.6); stroke-dasharray: 5 4; }
            .unet-skip-label { fill: #7c8ba1; font-size: 8px; letter-spacing: 0.06em; text-transform: uppercase; }
            .unet-io-bg { fill: rgba(148, 163, 184, 0.08); stroke: rgba(148, 163, 184, 0.28); }
            .unet-io-title { fill: #e2e8f0; font-size: 10px; font-weight: 700; }
            .unet-io-sub { fill: #94a3b8; font-size: 9px; }
            .unet-io-bg.accent { fill: rgba(56, 189, 248, 0.08); stroke: rgba(56, 189, 248, 0.3); }
        `;
    }

    renderContextBand(layout) {
        const group = svg("g", { class: "unet-band-group" });
        const band = layout.contextBand;
        group.append(
            svg("rect", {
                class: "unet-band",
                x: band.x,
                y: band.y,
                width: band.w,
                height: band.h,
                rx: 8,
            })
        );
        const dim = layout.contextDim;
        const encoders = layout.dualTextEncoder ? "CLIP-L + CLIP-G" : "CLIP";
        const label = dim
            ? `Text context · ${encoders} · ${dim}-d → cross-attention`
            : `Text context · ${encoders} · cross-attention not detected`;
        group.append(
            svg("text", { class: "unet-band-label", x: band.x + 12, y: band.y + 20 }, label)
        );
        group.append(
            svg(
                "text",
                {
                    class: "unet-band-sub",
                    x: band.x + band.w - 12,
                    y: band.y + 20,
                    "text-anchor": "end",
                },
                "ctx markers mark stages with text cross-attention"
            )
        );
        return group;
    }

    renderTimeBand(layout) {
        const group = svg("g", { class: "unet-band-group" });
        const band = layout.timeBand;
        group.append(
            svg("rect", {
                class: "unet-band time",
                x: band.x,
                y: band.y,
                width: band.w,
                height: band.h,
                rx: 8,
            })
        );
        const embed = this.structure.io?.time_embed || {};
        const features = embed.out_features ? ` · ${embed.out_features}-d` : "";
        group.append(
            svg(
                "text",
                { class: "unet-band-label time", x: band.x + 12, y: band.y + 20 },
                `t → time_embed${features} → every ResBlock`
            )
        );
        const right = layout.labelEmbed
            ? `+ size conditioning · ${this.baseSize}² (label_emb)`
            : "time embedding is added inside each ResBlock";
        group.append(
            svg(
                "text",
                {
                    class: "unet-band-sub",
                    x: band.x + band.w - 12,
                    y: band.y + 20,
                    "text-anchor": "end",
                },
                right
            )
        );
        return group;
    }

    renderCard(card) {
        const prefix = `unet${this.instanceId}`;
        const { stage } = card;
        const group = svg("g", {
            class: "unet-stage",
            "data-stage-id": stage.id,
            tabindex: 0,
            role: "button",
            "aria-label": `${stageRoleLabel(stage)} stage, ${stage.params} parameters`,
        });
        group.append(
            svg("title", {}, `${stage.id} · ${formatCount(stage.params)} parameters · ${formatCount(stage.key_count)} keys`)
        );
        group.append(
            svg("rect", {
                class: "unet-stage-bg",
                x: card.x,
                y: card.y,
                width: card.w,
                height: card.h,
                rx: 10,
            })
        );
        group.append(
            svg(
                "text",
                { class: "unet-stage-title", x: card.x + 12, y: card.y + 20 },
                stageRoleLabel(stage)
            )
        );
        const absolute = absoluteStageSize(this.baseSize, stage.resolution_ratio);
        const resolution = absolute
            ? `${ratioLabel(stage.resolution_ratio)} · ${absolute} px`
            : ratioLabel(stage.resolution_ratio);
        group.append(
            svg(
                "text",
                {
                    class: "unet-stage-res",
                    x: card.x + card.w - 34,
                    y: card.y + 20,
                    "text-anchor": "end",
                },
                resolution
            )
        );
        const blockCount = (stage.blocks || []).length;
        group.append(
            svg(
                "text",
                { class: "unet-stage-meta", x: card.x + 12, y: card.y + 38 },
                `${stage.channels ?? "?"} ch · ${blockCount} block${blockCount === 1 ? "" : "s"}`
            )
        );
        if (layoutHasCross(stage)) {
            group.append(
                svg(
                    "text",
                    {
                        class: "unet-ctx-port",
                        x: card.x + card.w - 12,
                        y: card.y + 38,
                        "text-anchor": "end",
                    },
                    "ctx"
                )
            );
        }
        if (this.showParams && !this.structureOnly) {
            const total = Number(this.structure.totals?.params) || 0;
            const share = total ? formatPercent((Number(stage.params) || 0) / total * 100) : null;
            const text = this.showShare && share
                ? `${formatParams(stage.params)} · ${share}`
                : formatParams(stage.params);
            group.append(
                svg(
                    "text",
                    { class: "unet-stage-params", x: card.x + 12, y: card.y + 55 },
                    text
                )
            );
        } else if (card.collapsed) {
            group.append(
                svg(
                    "text",
                    { class: "unet-stage-params", x: card.x + 12, y: card.y + 55 },
                    `${blockCount} blocks`
                )
            );
        }

        const collapse = svg("g", {
            class: "unet-collapse",
            "data-collapse-id": stage.id,
            tabindex: 0,
            role: "button",
            "aria-label": card.collapsed ? "Expand stage" : "Collapse stage",
            transform: `translate(${card.x + card.w - 24}, ${card.y + 7})`,
        });
        collapse.append(
            svg("rect", { class: "unet-collapse-bg", width: 17, height: 17, rx: 4 })
        );
        collapse.append(
            svg(
                "text",
                {
                    class: "unet-collapse-text",
                    x: 8.5,
                    y: 13,
                    "text-anchor": "middle",
                },
                card.collapsed ? "+" : "−"
            )
        );
        group.append(collapse);

        for (const row of card.blockRows) {
            group.append(this.renderBlock(row, stage));
        }

        if (stage.transition) {
            const factor =
                stage.transition.kind === "downsample"
                    ? `↓ downsample ×${stage.transition.stride || 2}`
                    : `↑ upsample ×${stage.transition.factor || 2}`;
            const source = stage.transition.stride_source || stage.transition.factor_source;
            const label = source ? `${factor} · ${sourceLabel(source)}` : factor;
            const text = svg(
                "text",
                {
                    class: "unet-transition",
                    x: card.x + 12,
                    y: card.y + card.h - 8,
                },
                label
            );
            if (source) text.append(svg("title", {}, sourceDescription(source)));
            group.append(text);
        }
        return group;
    }

    renderBlock(row, stage) {
        const { block } = row;
        const group = svg("g", {
            class: `unet-block ${block.kind || "unknown"}`,
            "data-block-id": block.id,
            tabindex: 0,
            role: "button",
            "aria-label": `${blockLabel(block.kind)} ${block.id}, ${block.params} parameters`,
        });
        group.append(
            svg(
                "title",
                {},
                `${block.id} · ${formatCount(block.params)} parameters · ${formatCount(block.key_count)} keys`
            )
        );
        group.append(
            svg("rect", {
                class: "unet-block-bg",
                x: row.x,
                y: row.y,
                width: row.w,
                height: row.h,
                rx: 6,
            })
        );
        group.append(
            svg(
                "text",
                { class: "unet-block-kind", x: row.x + 7, y: row.y + 17 },
                blockLabel(block.kind)
            )
        );

        let badgeX = row.x + row.w - 6;
        const attention = block.attention;
        if (attention?.cross) {
            badgeX -= 18;
            this.renderBadge(group, badgeX, row.y + 7, "CA", "ca");
            badgeX -= 3;
        }
        if (attention?.self) {
            badgeX -= 18;
            this.renderBadge(group, badgeX, row.y + 7, "SA", "sa");
            badgeX -= 3;
        }
        if (this.showParams && !this.structureOnly) {
            group.append(
                svg(
                    "text",
                    {
                        class: "unet-block-params",
                        x: badgeX - 3,
                        y: row.y + 17,
                        "text-anchor": "end",
                    },
                    formatParams(block.params)
                )
            );
        }
        return group;
    }

    renderBadge(parent, x, y, label, tone) {
        parent.append(
            svg("rect", {
                class: `unet-badge-${tone}`,
                x,
                y,
                width: 18,
                height: 12,
                rx: 3,
            })
        );
        parent.append(
            svg(
                "text",
                {
                    class: `unet-badge-text unet-badge-${tone}-text`,
                    x: x + 9,
                    y: y + 9,
                    "text-anchor": "middle",
                },
                label
            )
        );
    }

    renderIo(layout) {
        const group = svg("g", { class: "unet-io" });
        const latent = this.structure.latent || {};
        const convIn = this.structure.io?.conv_in || {};
        const convOut = this.structure.io?.conv_out || {};
        const latentPx = Math.round(this.baseSize / 8);
        const inChannels = latent.in_channels ?? "?";
        const ioNodes = [
            [
                layout.io.latentIn,
                "x_t",
                `${inChannels} ch · ${latentPx}² latent`,
                "Input latent. The 1/8 latent factor is a convention, not a value stored in the weights.",
            ],
            [
                layout.io.convIn,
                "conv_in",
                convIn.in_channels != null && convIn.out_channels != null
                    ? `${convIn.in_channels} → ${convIn.out_channels} ch`
                    : "Input convolution",
                "Input convolution. Channels come from the tensor shapes.",
            ],
            [
                layout.io.convOut,
                "conv_out",
                convOut.in_channels != null && convOut.out_channels != null
                    ? `${convOut.in_channels} → ${convOut.out_channels} ch`
                    : "Output convolution",
                "Output convolution. Channels come from the tensor shapes.",
            ],
            [
                layout.io.latentOut,
                layout.vPred ? "v / noise" : "predicted noise",
                `${convOut.out_channels ?? inChannels} ch · ${latentPx}²`,
                layout.vPred
                    ? "The model predicts a velocity value (v-prediction)."
                    : "The model predicts noise in latent space.",
            ],
        ];
        for (const [node, title, subtitle, tooltip] of ioNodes) {
            const pill = svg("g");
            pill.append(svg("title", {}, tooltip));
            pill.append(
                svg("rect", {
                    class: `unet-io-bg ${title === "conv_in" || title === "conv_out" ? "accent" : ""}`,
                    x: node.x,
                    y: node.y,
                    width: node.w,
                    height: node.h,
                    rx: 8,
                })
            );
            pill.append(
                svg(
                    "text",
                    { class: "unet-io-title", x: node.x + 10, y: node.y + 16 },
                    title
                )
            );
            pill.append(
                svg(
                    "text",
                    { class: "unet-io-sub", x: node.x + 10, y: node.y + 30 },
                    subtitle
                )
            );
            group.append(pill);
        }
        return group;
    }

    bindSvgEvents(root) {
        root.addEventListener("click", (event) => {
            if (performance.now() < this.suppressUntil) return;
            const target = event.target instanceof Element ? event.target : null;
            if (!target) return;
            const collapse = target.closest("[data-collapse-id]");
            if (collapse) {
                event.stopPropagation();
                this.toggleCollapse(collapse.getAttribute("data-collapse-id"));
                return;
            }
            const block = target.closest("[data-block-id]");
            if (block) {
                this.select("block", block.getAttribute("data-block-id"));
                return;
            }
            const stage = target.closest("[data-stage-id]");
            if (stage) this.select("stage", stage.getAttribute("data-stage-id"));
        });
        root.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            const target = event.target instanceof Element ? event.target : null;
            if (!target) return;
            const collapse = target.closest("[data-collapse-id]");
            if (collapse) {
                event.preventDefault();
                this.toggleCollapse(collapse.getAttribute("data-collapse-id"));
                return;
            }
            const block = target.closest("[data-block-id]");
            if (block) {
                event.preventDefault();
                this.select("block", block.getAttribute("data-block-id"));
                return;
            }
            const stage = target.closest("[data-stage-id]");
            if (stage) {
                event.preventDefault();
                this.select("stage", stage.getAttribute("data-stage-id"));
            }
        });
    }

    beginDrag(event) {
        if (event.button !== 0) return;
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest("[data-stage-id],[data-block-id],[data-collapse-id]")) return;
        this.drag = {
            x: event.clientX,
            y: event.clientY,
            left: this.canvasScroll.scrollLeft,
            top: this.canvasScroll.scrollTop,
            moved: false,
        };
        this.canvasScroll.classList.add("dragging");
        this.canvasScroll.setPointerCapture(event.pointerId);
    }

    moveDrag(event) {
        if (!this.drag) return;
        const dx = event.clientX - this.drag.x;
        const dy = event.clientY - this.drag.y;
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) this.drag.moved = true;
        if (!this.drag.moved) return;
        this.canvasScroll.scrollLeft = this.drag.left - dx;
        this.canvasScroll.scrollTop = this.drag.top - dy;
    }

    endDrag(event) {
        if (!this.drag) return;
        if (this.drag.moved) this.suppressUntil = performance.now() + 250;
        this.drag = null;
        this.canvasScroll.classList.remove("dragging");
        try {
            this.canvasScroll.releasePointerCapture(event.pointerId);
        } catch (error) {
            // pointer capture may already be released
        }
    }

    setScale(next, clientX, clientY) {
        if (!this.layout || !this.svg) return;
        const clamped = Math.max(0.28, Math.min(2.5, next));
        const rect = this.canvasScroll.getBoundingClientRect();
        const centerX = clientX === undefined ? rect.width / 2 : clientX - rect.left;
        const centerY = clientY === undefined ? rect.height / 2 : clientY - rect.top;
        const old = this.scale || 1;
        const worldX = (this.canvasScroll.scrollLeft + centerX) / old;
        const worldY = (this.canvasScroll.scrollTop + centerY) / old;
        this.scale = clamped;
        this.autoFit = false;
        this.applyScale(clamped);
        this.canvasScroll.scrollLeft = worldX * clamped - centerX;
        this.canvasScroll.scrollTop = worldY * clamped - centerY;
    }

    applyScale(scale) {
        if (!this.svg || !this.layout) return;
        this.scale = Math.max(0.28, Math.min(2.5, scale));
        const width = Math.round(this.layout.width * this.scale);
        const height = Math.round(this.layout.height * this.scale);
        this.svg.setAttribute("width", width);
        this.svg.setAttribute("height", height);
        this.svg.style.width = `${width}px`;
        this.svg.style.height = `${height}px`;
    }

    fit() {
        if (!this.layout || !this.svg) return;
        const available = this.canvasScroll.clientWidth;
        if (available < 40) return;
        const scale = Math.min(1.1, Math.max(0.28, (available - 4) / this.layout.width));
        this.autoFit = true;
        this.applyScale(scale);
        const maxHeight = Number.parseFloat(getComputedStyle(this.canvasScroll).maxHeight);
        if (Number.isFinite(maxHeight) && this.layout.height * this.scale > maxHeight - 4) {
            this.applyScale(Math.max(0.28, (maxHeight - 4) / this.layout.height));
        }
        this.canvasScroll.scrollLeft = 0;
        this.canvasScroll.scrollTop = 0;
    }

    resize() {
        if (!this.layout) return;
        if (this.autoFit) this.fit();
        else this.applyScale(this.scale);
    }

    toggleCollapse(stageId) {
        if (this.collapsed.has(stageId)) this.collapsed.delete(stageId);
        else this.collapsed.add(stageId);
        this.renderSvg();
    }

    select(type, id) {
        this.selected = { type, id };
        this.updateSelectionClasses();
        this.renderDetail();
    }

    updateSelectionClasses() {
        if (!this.svg) return;
        const selected = this.selected;
        const blockParent = selected?.type === "block"
            ? this.findCardForBlock(selected.id)?.stage.id
            : null;
        for (const node of this.svg.querySelectorAll(".unet-stage")) {
            const id = node.getAttribute("data-stage-id");
            node.classList.toggle(
                "selected",
                id === selected?.id || id === blockParent
            );
        }
        for (const node of this.svg.querySelectorAll(".unet-block")) {
            node.classList.toggle(
                "selected",
                selected?.type === "block" && node.getAttribute("data-block-id") === selected.id
            );
        }
    }

    findCardForBlock(blockId) {
        return (this.layout?.cards || []).find((card) =>
            (card.stage.blocks || []).some((block) => block.id === blockId)
        ) || null;
    }

    findCardForStage(stageId) {
        return (this.layout?.cards || []).find((card) => card.stage.id === stageId) || null;
    }

    renderDetail() {
        clear(this.detail);
        const selected = this.selected;
        if (selected?.type === "block") {
            const card = this.findCardForBlock(selected.id);
            const block = (card?.stage.blocks || []).find((item) => item.id === selected.id);
            if (card && block) {
                this.renderBlockDetail(card, block);
                return;
            }
        }
        if (selected?.type === "stage") {
            const card = this.findCardForStage(selected.id);
            if (card) {
                this.renderStageDetail(card);
                return;
            }
        }
        this.renderNoSelection();
    }

    renderNoSelection() {
        const wrapper = el("div", "unet-detail-inner");
        wrapper.append(el("h3", null, "Model structure"));
        wrapper.append(
            el(
                "p",
                "muted",
                "Select a stage card or a block row. The detail panel shows the exact sublayers, tensor shapes, dtypes and parameter counts behind the diagram."
            )
        );
        const chips = el("div", "unet-detail-chips");
        chips.append(
            detailChip("Key naming", this.structure.naming === "ldm" ? "LDM / CompVis" : "Diffusers"),
            detailChip("Latent channels", this.structure.latent?.in_channels ?? "Unknown", this.structure.latent?.in_channels_source),
            detailChip("Base resolution", `${this.baseSize} px`, "user", "Chosen in the report, not stored in the weights."),
            detailChip("Relative levels", `${(this.structure.stages || []).length} down · ${(this.structure.up || []).length} up`),
            detailChip("Text context", findContextDim(this.structure) ? `${findContextDim(this.structure)}-d` : "Not detected"),
            detailChip(
                "UNet parameters",
                formatParams(this.structure.totals?.params),
                null,
                `${formatCount(this.structure.totals?.params)} parameters`
            )
        );
        wrapper.append(chips);
        if (this.layout?.labelEmbed) {
            wrapper.append(
                el(
                    "p",
                    "unet-detail-note",
                    `Size conditioning is present: label_emb adds ${formatParams(this.layout.labelEmbed.params)} parameters to the timestep embedding.`
                )
            );
        }
        wrapper.append(
            el(
                "p",
                "unet-detail-note",
                "Absolute resolutions use the selected base. Attention head counts are not stored in the weights and are never guessed."
            )
        );
        this.detail.append(wrapper);
    }

    renderStageDetail(card) {
        const { stage } = card;
        const total = Number(this.structure.totals?.params) || 0;
        const wrapper = el("div", "unet-detail-inner");
        const heading = el("div", "unet-detail-title");
        heading.append(
            el("h3", null, stageRoleLabel(stage)),
            el("span", "mono muted", stage.id)
        );
        wrapper.append(heading);

        const absolute = absoluteStageSize(this.baseSize, stage.resolution_ratio);
        const chips = el("div", "unet-detail-chips");
        chips.append(
            detailChip(
                "Resolution",
                absolute
                    ? `${ratioLabel(stage.resolution_ratio)} · ${absolute} px`
                    : ratioLabel(stage.resolution_ratio),
                stage.resolution_source,
                "Relative resolutions come from the weights; the absolute size uses the selected base resolution."
            ),
            detailChip("Channels", stage.channels ?? "Unknown", stage.channels_source),
            detailChip(
                "Parameters",
                formatParams(stage.params),
                null,
                `${formatCount(stage.params)} parameters`
            ),
            detailChip(
                "Share",
                total ? formatPercent((Number(stage.params) || 0) / total * 100) : "—",
                null,
                "Share of all UNet parameters"
            ),
            detailChip(
                "Keys",
                formatCount(stage.key_count),
                null,
                `${formatCount(stage.key_count)} tensor keys`
            )
        );
        wrapper.append(chips);

        if (stage.transition) {
            const source = stage.transition.stride_source || stage.transition.factor_source;
            const factor =
                stage.transition.kind === "downsample"
                    ? `Downsample ×${stage.transition.stride || 2}`
                    : `Upsample ×${stage.transition.factor || 2}`;
            const line = el("p", "unet-detail-note");
            line.append(
                document.createTextNode(
                    `${factor} · ${formatParams(stage.transition.params)} parameters · `
                ),
                sourceTag(source) || document.createTextNode("source unknown")
            );
            wrapper.append(line);
        }
        if (stage.skip_from) {
            const line = el("p", "unet-detail-note");
            line.append(
                document.createTextNode(`Skip connection from ${stage.skip_from} · `),
                sourceTag(stage.skip_source || "convention") || document.createTextNode("")
            );
            wrapper.append(line);
        }

        const blockHead = el("div", "unet-detail-section-head");
        blockHead.append(
            el("h4", null, "Blocks"),
            el("span", "muted", `${(stage.blocks || []).length} detected`)
        );
        wrapper.append(blockHead);
        const blockList = el("div", "unet-block-links");
        for (const block of stage.blocks || []) {
            const button = el("button", "unet-block-link");
            button.type = "button";
            button.append(
                el("span", "unet-block-link-kind", blockLabel(block.kind)),
                el("span", "unet-block-link-id mono", block.id),
                el("span", "unet-block-link-params", formatParams(block.params))
            );
            button.addEventListener("click", () => this.select("block", block.id));
            blockList.append(button);
        }
        if (!(stage.blocks || []).length) {
            blockList.append(el("p", "muted", "No blocks detected in this stage."));
        }
        wrapper.append(blockList);

        const prefixes = stageKeyPrefixes(this.structure, stage);
        wrapper.append(
            this.viewKeysButton(prefixes, `${stageRoleLabel(stage)} · ${stage.id}`)
        );
        this.detail.append(wrapper);
    }

    renderBlockDetail(card, block) {
        const total = Number(this.structure.totals?.params) || 0;
        const wrapper = el("div", "unet-detail-inner");
        const back = el("button", "unet-back", `← ${stageRoleLabel(card.stage)}`);
        back.type = "button";
        back.addEventListener("click", () => this.select("stage", card.stage.id));
        wrapper.append(back);

        const heading = el("div", "unet-detail-title");
        heading.append(
            el("h3", null, blockLabel(block.kind)),
            el("span", "mono muted", block.id)
        );
        wrapper.append(heading);

        const chips = el("div", "unet-detail-chips");
        chips.append(
            detailChip("Parameters", formatParams(block.params), null, `${formatCount(block.params)} parameters`),
            detailChip(
                "Share",
                total ? formatPercent((Number(block.params) || 0) / total * 100) : "—",
                null,
                "Share of all UNet parameters"
            ),
            detailChip("Keys", formatCount(block.key_count), null, "Tensor keys assigned to this block"),
            detailChip(
                "Source",
                sourceLabel(block.source),
                null,
                sourceDescription(block.source)
            )
        );
        if (block.attention) {
            const attention = block.attention;
            chips.append(
                detailChip("Self-attention", attention.self ? "Yes" : "No"),
                detailChip("Cross-attention", attention.cross ? "Yes" : "No"),
                detailChip("Inner dim", attention.inner_dim ?? "Unknown"),
                detailChip("Context dim", attention.context_dim ?? "Unknown"),
                detailChip("Transformer blocks", formatCount(attention.transformer_blocks)),
                detailChip(
                    "Heads",
                    attention.heads == null ? "Not stored in weights" : formatCount(attention.heads),
                    attention.heads == null ? "unknown" : attention.heads_source,
                    attention.heads == null
                        ? "Attention head count is not stored in the weights and is not guessed."
                        : "Head count was detected in the weights."
                )
            );
        }
        wrapper.append(chips);
        wrapper.append(
            this.viewKeysButton([keyPrefixForBlock(this.structure, block.id)], `${blockLabel(block.kind)} · ${block.id}`)
        );

        const children = block.children || [];
        if (children.length) {
            const head = el("div", "unet-detail-section-head");
            head.append(el("h4", null, "Sublayers"), el("span", "muted", `${children.length} groups`));
            wrapper.append(head);
            const rows = children.map((child) => [
                blockLabel(child.kind),
                formatParams(child.params),
                formatCount(child.key_count),
                block.params ? formatPercent((Number(child.params) || 0) / block.params * 100) : "—",
            ]);
            wrapper.append(detailTable(["Group", "Params", "Keys", "Share"], rows, [1, 2, 3]));
        }

        const prefix = keyPrefixForBlock(this.structure, block.id);
        const keys = keysUnderPrefixes(this.keys, [prefix]);
        const head = el("div", "unet-detail-section-head");
        head.append(
            el("h4", null, "Weight keys"),
            el("span", "muted", `${formatCount(keys.length)} under this block`)
        );
        wrapper.append(head);
        if (keys.length) {
            const rows = keys.map((key) => [
                key.name.startsWith(prefix) ? key.name.slice(prefix.length) : key.name,
                shapeLabel(key.shape),
                key.dtype || "—",
                formatParams(key.params),
            ]);
            const table = detailTable(["Key", "Shape", "Dtype", "Params"], rows, [3]);
            table.classList.add("unet-key-table");
            wrapper.append(table);
        } else {
            wrapper.append(
                el(
                    "p",
                    "muted",
                    "No keys were found under this block prefix. The structure and key list may come from different files."
                )
            );
        }

        this.detail.append(wrapper);
    }

    viewKeysButton(prefixes, label) {
        const button = el("button", "button primary unet-view-keys", "View in keys");
        button.type = "button";
        if (!prefixes.length) {
            button.disabled = true;
            button.title = "No key prefix is available for this selection.";
            return button;
        }
        button.title = `Open the Keys tab filtered to ${label}`;
        button.addEventListener("click", () => {
            this.onViewKeys(prefixes, label);
        });
        return button;
    }

    exportSvg() {
        if (!this.svg || !this.layout) return;
        const clone = this.svg.cloneNode(true);
        clone.setAttribute("xmlns", SVG_NS);
        clone.setAttribute("width", this.layout.width);
        clone.setAttribute("height", this.layout.height);
        clone.setAttribute("viewBox", `0 0 ${this.layout.width} ${this.layout.height}`);
        clone.removeAttribute("data-unet-root");
        const source = new XMLSerializer().serializeToString(clone);
        const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${source}`], {
            type: "image/svg+xml;charset=utf-8",
        });
        downloadBlob(safeFilename(this.fileName, "unet-architecture.svg"), blob);
        this.onToast("UNet architecture SVG exported");
    }
}

function layoutHasCross(stage) {
    return (stage?.blocks || []).some((block) => block?.attention?.cross);
}
