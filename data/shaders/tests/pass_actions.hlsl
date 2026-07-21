//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the draw behind the render pass load/store action tests.
//
// Those tests are about what the *pass* does to the attachment around the draw, not about the draw
// itself, so this is deliberately the dullest shader in the suite: a flat-colored triangle from
// SV_VertexID, with no vertex buffer, no bindless resources and no interpolated attributes. Anything
// the golden shows that is not this flat color came from the load op or the store op, which is the
// whole point.
//
// The `fullscreen` switch exists for the DONT_CARE load test specifically. DONT_CARE says the
// attachment's contents at the start of the pass are undefined, so the only pixels that test can
// legally assert on are the ones the draw itself writes -- which means it has to write all of them.
// The other two tests want the opposite: a shape that leaves most of the attachment untouched, so
// that whatever the load op put there is still visible around it.

#include "data/shaders/agfx.h"

struct PassActionsPushConstants
{
    float4 color;    // The flat fill color.
    uint fullscreen; // Non-zero: cover the whole target. Zero: a centered triangle.
    float depth;     // Clip-space z the geometry is emitted at. 0 for the load/store action tests,
                     // which have no depth attachment; the depth-sampling test uses it to put a
                     // known, non-uniform pattern into a depth buffer.
    uint padding0;
    uint padding1;
};

AGFX_PUSH_CONSTANTS(PassActionsPushConstants, g_Constants);

float4 main_vs(uint vertexID : SV_VertexID) : SV_Position
{
    // The fullscreen case is the standard oversized triangle rather than a quad: it covers the
    // whole of NDC with a single primitive and no interior edge, so no pixel can be missed by an
    // edge-rasterization tie-break -- which would otherwise leave an undefined DONT_CARE pixel in
    // the middle of the image and make the test flaky rather than wrong.
    const float2 fullscreen[3] = {
        float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)
    };
    // Centered and well clear of the target's edges, so the corners the load-op tests sample stay
    // untouched by the draw under any reasonable rasterization rule.
    const float2 centered[3] = {
        float2(-0.6f, -0.6f), float2(0.6f, -0.6f), float2(0.0f, 0.7f)
    };

    const float2 position = g_Constants.fullscreen ? fullscreen[vertexID] : centered[vertexID];
    return float4(position, g_Constants.depth, 1.0f);
}

float4 main_ps() : SV_Target0
{
    return g_Constants.color;
}
