import test from "node:test";
import assert from "node:assert/strict";

import {
    absoluteStageSize,
    defaultBaseSize,
    findContextDim,
    keyPrefixForBlock,
    keysUnderPrefixes,
    layoutUnet,
    stageKeyPrefixes,
    stageRoleLabel,
} from "../../web/report/views/unet.js";

function block(id, kind, params, options = {}) {
    const attention =
        options.self || options.cross
            ? {
                  self: Boolean(options.self),
                  cross: Boolean(options.cross),
                  context_dim: options.cross ? 2048 : null,
                  inner_dim: 320,
                  heads: options.heads ?? null,
                  transformer_blocks: options.transformerBlocks ?? 2,
              }
            : null;
    return {
        id,
        kind,
        params,
        key_count: 4,
        source: "weights",
        children: [
            { kind: "conv", params: Math.round(params * 0.6), key_count: 2 },
            { kind: "norm", params: Math.round(params * 0.4), key_count: 2 },
        ],
        attention,
    };
}

function stage(direction, level, ratio, channels, blocks, skipFrom = null) {
    return {
        id:
            direction === "down"
                ? `down.${level}`
                : direction === "up"
                  ? `up.${level}`
                  : "middle.0",
        direction,
        level,
        resolution_ratio: ratio,
        resolution_source: "weights",
        channels,
        channels_source: "weights",
        params: blocks.reduce((sum, item) => sum + item.params, 0),
        key_count: blocks.reduce((sum, item) => sum + item.key_count, 0),
        skip_from: skipFrom,
        skip_source: skipFrom ? "weights" : null,
        blocks,
    };
}

function makeStructure(levels, options = {}) {
    const down = [];
    const up = [];
    let ratio = 1;
    for (let level = 0; level < levels; level += 1) {
        down.push(
            stage("down", level, ratio, 320 * 2 ** level, [
                block(`input_blocks.${level}.0`, "resblock", 1000 + level, {
                    self: level > 0,
                    cross: true,
                }),
                block(`input_blocks.${level}.1`, "transformer", 2000 + level, {
                    self: true,
                    cross: true,
                    transformerBlocks: options.transformerBlocks ?? 2,
                }),
            ])
        );
        ratio /= 2;
    }
    for (let level = levels - 1; level >= 0; level -= 1) {
        up.push(
            stage(
                "up",
                level,
                1 / 2 ** level,
                320 * 2 ** level,
                [block(`output_blocks.${level}.0`, "resblock", 3000 + level, { self: true, cross: true })],
                `down.${level}`
            )
        );
    }
    const middle = stage("middle", 0, ratio, 320 * 2 ** (levels - 1), [
        block("middle_block.0", "resblock", 4000, { self: true, cross: true }),
        block("middle_block.1", "transformer", 5000, { self: true, cross: true }),
    ]);
    middle.id = "middle.0";
    return {
        kind: "unet",
        naming: "ldm",
        prefix: "model.diffusion_model.",
        latent: { in_channels: 4, base_size: options.baseSize ?? 1024, latent_size: 128 },
        capabilities: {
            self_attention: true,
            text_cross_attention: true,
            dual_text_encoder: options.dualTextEncoder ?? true,
            size_conditioning: options.sizeConditioning ?? true,
            v_pred: false,
        },
        io: {
            conv_in: { out_channels: 320, params: 1000, source: "weights" },
            conv_out: { out_channels: 4, params: 1000, source: "weights" },
            time_embed: { out_features: 1280, params: 5_000_000, source: "weights" },
        },
        extras: options.sizeConditioning === false
            ? []
            : [{ kind: "label_embed", params: 5_245_440, source: "weights" }],
        stages: down,
        middle,
        up,
        totals: { params: 1_000_000, key_count: 42, io_params: 2000, extras_params: 0 },
        unattributed: { params: 0, key_count: 0 },
    };
}

const SDXL = makeStructure(3);
const SD15 = makeStructure(4, {
    baseSize: 512,
    dualTextEncoder: false,
    sizeConditioning: false,
});

const KEYS = [
    { name: "model.diffusion_model.input_blocks.0.0.weight", params: 10 },
    { name: "model.diffusion_model.input_blocks.0.1.weight", params: 20 },
    { name: "model.diffusion_model.input_blocks.10.0.weight", params: 30 },
    { name: "model.diffusion_model.middle_block.0.weight", params: 40 },
    { name: "model.diffusion_model.output_blocks.0.0.weight", params: 50 },
    { name: "cond_stage_model.transformer.weight", params: 60 },
];

test("defaultBaseSize uses the weight fact and falls back to 512", () => {
    assert.equal(defaultBaseSize(SDXL), 1024);
    assert.equal(defaultBaseSize(SD15), 512);
    assert.equal(defaultBaseSize({ latent: {} }), 512);
    assert.equal(defaultBaseSize({}), 512);
});

test("absoluteStageSize multiplies the selected base and rejects invalid input", () => {
    assert.equal(absoluteStageSize(1024, 0.5), 512);
    assert.equal(absoluteStageSize(512, 1), 512);
    assert.equal(absoluteStageSize(1024, 0.25), 256);
    assert.equal(absoluteStageSize(0, 0.5), null);
    assert.equal(absoluteStageSize(1024, null), null);
    assert.equal(absoluteStageSize("nope", 0.5), null);
});

test("stageRoleLabel names down, middle and up stages", () => {
    assert.equal(stageRoleLabel(SDXL.stages[0]), "Down L0");
    assert.equal(stageRoleLabel(SDXL.middle), "Middle L0");
    assert.equal(stageRoleLabel(SDXL.up[0]), "Up L2");
});

test("key prefixes follow the report prefix and normalize the trailing dot", () => {
    assert.equal(
        keyPrefixForBlock(SDXL, "input_blocks.0.0"),
        "model.diffusion_model.input_blocks.0.0."
    );
    assert.equal(
        keyPrefixForBlock({ prefix: "model.diffusion_model" }, "middle_block.0"),
        "model.diffusion_model.middle_block.0."
    );
    assert.deepEqual(stageKeyPrefixes(SDXL, SDXL.stages[0]), [
        "model.diffusion_model.input_blocks.0.0.",
        "model.diffusion_model.input_blocks.0.1.",
    ]);
});

test("keysUnderPrefixes matches only keys below the given prefixes", () => {
    const matched = keysUnderPrefixes(KEYS, [
        "model.diffusion_model.input_blocks.0.0",
        "model.diffusion_model.middle_block.0",
    ]);
    assert.deepEqual(
        matched.map((key) => key.name),
        [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.0.weight",
        ]
    );
    assert.equal(keysUnderPrefixes(KEYS, []).length, 0);
    assert.equal(keysUnderPrefixes([], ["model.diffusion_model."]).length, 0);
});

test("findContextDim reads the cross-attention context from real blocks", () => {
    assert.equal(findContextDim(SDXL), 2048);
    assert.equal(findContextDim({ stages: [{ blocks: [{ kind: "resblock" }] }] }), null);
});

test("layoutUnet aligns down and up stages by level and keeps block rows inside cards", () => {
    const layout = layoutUnet(SDXL);
    assert.equal(layout.columns, 3);
    assert.equal(layout.downCards.length, 3);
    assert.equal(layout.upCards.length, 3);
    assert.ok(layout.middleCard);

    assert.deepEqual(
        layout.upCards.map((card) => card.column),
        [2, 1, 0]
    );
    for (const card of layout.downCards) {
        assert.equal(card.column, card.stage.level);
        assert.equal(card.y, layout.downCards[0].y);
    }
    assert.ok(layout.middleCard.x > layout.downCards[2].x);
    assert.ok(layout.width >= layout.middleCard.x + layout.middleCard.w);

    for (const card of layout.cards) {
        assert.ok(card.blockRows.length > 0, `${card.stage.id} should expose block rows`);
        for (const row of card.blockRows) {
            assert.ok(row.y >= card.y + 64, `${card.stage.id} block row must sit below the header`);
            assert.ok(row.y + row.h <= card.y + card.h, `${card.stage.id} block row must stay in the card`);
        }
    }
});

test("layoutUnet draws one skip per linked up stage, plus context and time links", () => {
    const layout = layoutUnet(SDXL);
    assert.equal(layout.skipLinks.length, 3);
    assert.deepEqual(
        layout.skipLinks.map((link) => [link.from, link.to]),
        [
            ["down.2", "up.2"],
            ["down.1", "up.1"],
            ["down.0", "up.0"],
        ]
    );
    assert.equal(layout.contextLinks.length, 4);
    assert.deepEqual(
        layout.contextLinks.map((link) => link.stage),
        ["down.0", "down.1", "down.2", "middle.0"]
    );
    assert.equal(layout.timeRail.stubs.length, 3);
    assert.deepEqual(
        layout.timeLinks.map((link) => link.stage),
        ["up.2", "up.1", "up.0", "middle.0"]
    );
});

test("collapsed stages drop block rows and shorten their card", () => {
    const expanded = layoutUnet(SDXL);
    const collapsed = layoutUnet(SDXL, { collapsed: new Set(["down.1"]) });
    const expandedCard = expanded.downCards.find((card) => card.stage.id === "down.1");
    const collapsedCard = collapsed.downCards.find((card) => card.stage.id === "down.1");
    assert.ok(collapsedCard.collapsed);
    assert.equal(collapsedCard.blockRows.length, 0);
    assert.ok(collapsedCard.h < expandedCard.h);
    for (const card of collapsed.cards) {
        if (card.stage.id === "down.1") continue;
        assert.ok(card.blockRows.length > 0);
    }
});

test("SD1.5 layout keeps four levels and four skips", () => {
    const layout = layoutUnet(SD15);
    assert.equal(layout.columns, 4);
    assert.equal(layout.skipLinks.length, 4);
    assert.equal(layout.contextLinks.length, 5);
    assert.equal(layout.timeRail.stubs.length, 4);
    assert.ok(layout.upCards.every((card) => card.blockRows.length > 0));
    assert.equal(defaultBaseSize(SD15), 512);
});
