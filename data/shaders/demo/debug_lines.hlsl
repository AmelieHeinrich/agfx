/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct DebugLinesPushConstants {
    float4x4 viewProj;
    ResourceHandle vertexBuffer;
};

AGFX_PUSH_CONSTANTS(DebugLinesPushConstants, g_Constants);

struct vs_out {
    float4 position : SV_POSITION;
    float3 color : TEXCOORD0;
};

// 24 vertices (12 edges x 2) per box - see DeferredRenderer::SetupDebugBoundingBoxes.
static const uint kVerticesPerBox = 24;

float3 HashColor(uint seed) {
    // Cheap deterministic per-index color so spatially-overlapping-but-distinct boxes (common in
    // Sponza, e.g. coincident front/back curtain faces) stay visually distinguishable instead of
    // blending into what looks like one merged box.
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    float hue = frac((float)seed / 4294967296.0f);
    float3 p = abs(frac(hue + float3(1.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f);
    return saturate(p - 1.0f);
}

vs_out main_vs(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<float3> vertices = AGFXStructuredBuffer<float3>::Create(g_Constants.vertexBuffer);
    float3 worldPos = vertices.Load(vertexID);

    vs_out output;
    output.position = mul(g_Constants.viewProj, float4(worldPos, 1.0f));
    output.color = HashColor(vertexID / kVerticesPerBox);
    return output;
}

float4 main_ps(vs_out input) : SV_TARGET0 {
    return float4(input.color, 1.0f);
}
