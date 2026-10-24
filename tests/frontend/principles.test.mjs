import test from "node:test";
import assert from "node:assert/strict";

import {
    PRINCIPLE_REFERENCES,
    collectPrincipleFacts,
} from "../../web/report/principles/principles.js";

function block(id, kind, params, attention = null) {
    return { id, kind, params, key_count: 2, source: "weights", attention };
}

function makeReport(overrides = {}) {
    const down = [
        {
            id: "input_blocks.0",
            direction: "down",
            level: 0,
            resolution_ratio: 1,
            channels: 320,
            params: 1000,
            key_count: 10,
            blocks: [
                block("input_blocks.0.0", "resblock", 600),
                block("input_blocks.0.1", "transformer", 400, {
                    self: true,
                    cross: true,
                    inner_dim: 320,
                    context_dim: 2048,
                    transformer_blocks: 10,
                    heads: null,
                }),
            ],
        },
        {
            id: "input_blocks.1",
            direction: "down",
            level: 1,
            resolution_ratio: 0.5,
            channels: 640,
            params: 2000,
            key_count: 20,
            blocks: [block("input_blocks.1.0", "resblock", 2000)],
        },
    ];
    const up = [
        {
            id: "output_blocks.0",
            direction: "up",
            level: 0,
            resolution_ratio: 1,
            channels: 320,
            params: 1500,
            key_count: 15,
            skip_from: "down.0",
            blocks: [block("output_blocks.0.0", "resblock", 1500)],
        },
        {
            id: "output_blocks.1",
            direction: "up",
            level: 1,
            resolution_ratio: 0.5,
            channels: 640,
            params: 1200,
            key_count: 12,
            skip_from: "down.1",
            blocks: [block("output_blocks.1.0", "resblock", 1200)],
        },
    ];
    const middle = {
        id: "middle.0",
        direction: "middle",
        level: 0,
        resolution_ratio: 0.25,
        channels: 1280,
        params: 3000,
        key_count: 30,
        blocks: [
            block("middle_block.0", "resblock", 1500),
            block("middle_block.1", "transformer", 1500, {
                self: true,
                cross: true,
                inner_dim: 1280,
                context_dim: 2048,
                transformer_blocks: 10,
                heads: null,
            }),
        ],
    };
    return {
        summary: { architecture: overrides.archType || "SDXL" },
        architecture: { type: overrides.archType || "SDXL" },
        components: { UNET: { params: 450_000_000, percentage: 90 } },
        structure: {
            kind: "unet",
            naming: "ldm",
            latent: { in_channels: 4, base_size: 1024, latent_size: 128 },
            capabilities: {
                self_attention: true,
                text_cross_attention: true,
                dual_text_encoder: overrides.dualTextEncoder ?? true,
                size_conditioning: overrides.sizeConditioning ?? true,
                refiner: overrides.refiner ?? false,
                v_pred: false,
            },
            io: {
                time_embed: { out_features: 1280, params: 5_000_000, source: "weights" },
            },
            extras:
                overrides.sizeConditioning === false
                    ? []
                    : [{ kind: "label_embed", params: 5_245_440, source: "weights" }],
            stages: down,
            middle,
            up,
            totals: { params: 450_000_000, key_count: 1200 },
            unattributed: { params: 0, key_count: 0 },
        },
    };
}

test("collectPrincipleFacts reads SDXL facts from the report", () => {
    const facts = collectPrincipleFacts(makeReport());
    assert.equal(facts.archType, "SDXL");
    assert.equal(facts.isSDXL, true);
    assert.equal(facts.naming, "ldm");
    assert.equal(facts.dualTextEncoder, true);
    assert.equal(facts.sizeConditioning, true);
    assert.equal(facts.refiner, false);
    assert.equal(facts.vPred, false);
    assert.equal(facts.contextDim, 2048);
    assert.equal(facts.textEncoderName, "CLIP-L + CLIP-G");
    assert.equal(facts.baseSize, 1024);
    assert.equal(facts.downCount, 2);
    assert.equal(facts.upCount, 2);
    assert.equal(facts.skipCount, 2);
    assert.equal(facts.middleChannels, 1280);
    assert.equal(facts.middleRatio, 0.25);
    assert.equal(facts.unetParams, 450_000_000);
    assert.equal(facts.unetShare, 90);
    assert.equal(facts.timeEmbedFeatures, 1280);
    assert.equal(facts.labelEmbedParams, 5_245_440);
    assert.equal(facts.transformerBlocks, 10);
    assert.equal(facts.transformerBlocksLabel, "10");
    assert.equal(facts.heads, null);
    assert.equal(facts.headsStored, false);
    assert.equal(facts.resblockCount, 5);
    assert.equal(facts.resblockParams, 600);
    assert.equal(facts.transformerParams, 400);
});

test("collectPrincipleFacts marks SD1.5 modules as absent without inventing heads", () => {
    const facts = collectPrincipleFacts(
        makeReport({ archType: "SD1.x", dualTextEncoder: false, sizeConditioning: false })
    );
    assert.equal(facts.isSDXL, false);
    assert.equal(facts.dualTextEncoder, false);
    assert.equal(facts.sizeConditioning, false);
    assert.equal(facts.textEncoderName, "CLIP");
    assert.equal(facts.labelEmbedParams, null);
    assert.equal(facts.headsStored, false);
    assert.equal(facts.middleRatio, 0.25);
});

test("collectPrincipleFacts tolerates a report without UNet structure", () => {
    const facts = collectPrincipleFacts({ summary: { architecture: "Unknown" } });
    assert.equal(facts.downCount, 0);
    assert.equal(facts.upCount, 0);
    assert.equal(facts.skipCount, 0);
    assert.equal(facts.contextDim, null);
    assert.equal(facts.unetParams, null);
    assert.equal(facts.headsStored, false);
    assert.equal(facts.unattributedParams, 0);
});

test("principle references are complete, unique and point to papers", () => {
    assert.equal(PRINCIPLE_REFERENCES.length, 6);
    assert.equal(new Set(PRINCIPLE_REFERENCES.map((item) => item.id)).size, 6);
    assert.equal(new Set(PRINCIPLE_REFERENCES.map((item) => item.url)).size, 6);
    for (const reference of PRINCIPLE_REFERENCES) {
        assert.match(reference.url, /^https:\/\/arxiv\.org\/abs\/\d{4}\.\d{4,5}$/);
        assert.ok(reference.label.length > 20);
    }
});
