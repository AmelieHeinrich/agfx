/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the rasterizer-state tests (wireframe, topology, cull mode, front face,
// fragment discard, push-constant color).
//
// Every one of those draws the exact same geometry with the exact same shader
// (data/shaders/tests/raster.hlsl) and differs only in a handful of agfxRenderPipeline fields and
// the push constants. Copying the full C / C++ / ez setup into each of the six files would be six
// near-identical copies of ~200 lines whose only interesting content is three struct fields — so
// the setup lives here once, parameterized by RasterState, and each test file is left holding only
// the state under test and its golden.
//
// The one entry point deliberately covers all three API flavors: the flavors must produce
// byte-identical images for the same state, and routing them through one function makes that a
// property of the code rather than something to keep in sync by hand across six files.

#include "../test_gpu.h"

namespace agfxtest
{
    constexpr uint32_t kRasterWidth = 128;
    constexpr uint32_t kRasterHeight = 128;
    constexpr agfxTextureFormat kRasterFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief Mirrors RasterPushConstants in data/shaders/tests/raster.hlsl. Field order and
    /// padding must match exactly — this is memcpy'd into the 128-byte push-constant block.
    struct RasterConstants
    {
        float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float discardX = 2.0f;         ///< Above any NDC x, so nothing is clipped by default.
        uint32_t useVertexColor = 1;   ///< Default to the per-vertex gradient.
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
    };

    /// @brief The pipeline state one raster test varies. Defaults describe the baseline draw:
    /// both triangles, solid, unculled, vertex-colored.
    struct RasterState
    {
        agfxFillMode fillMode = AGFX_FILL_MODE_SOLID;
        agfxCullMode cullMode = AGFX_CULL_MODE_NONE;
        agfxFrontFace frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        agfxTopology topology = AGFX_TOPOLOGY_TRIANGLES;
        RasterConstants constants{};
        uint32_t vertexCount = 6;
    };

    /// @brief Renders one RasterState through the given API flavor into `outImage`.
    /// Returns false if device creation, shader compilation, or readback failed; the caller should
    /// AGFX_EXPECT on it before comparing against a golden.
    bool RenderRaster(TestApi api, const RasterState& state, Image& outImage);

    /// @brief Counts pixels that are not the pass's black clear color.
    ///
    /// The golden comparison thresholds on *mean* FLIP error across the whole image, which is the
    /// right call for the tests that fill a large area but says almost nothing about one that lights
    /// a handful of pixels: six stray points out of 16384 move the mean by ~0.0004, so a points test
    /// compared on mean error alone would pass just as well against a GPU that drew nothing at all.
    /// Tests with sparse output pair the golden with an exact count from this.
    uint32_t CountLitPixels(const Image& image);
} // namespace agfxtest
