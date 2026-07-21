/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the render pass action tests (load op LOAD / CLEAR / DONT_CARE, each with
// store op STORE).
//
// Store DONT_CARE is deliberately not covered: it leaves the attachment's contents undefined after
// the pass, so there is nothing a golden could assert, and D3D12 does not support it in the first
// place. That leaves the three load ops as the whole matrix.
//
// The mechanism every one of these tests shares: seed the attachment with a distinct pattern
// *before* the pass, then run a pass whose only interesting property is its load op, and look at
// what happened to the pixels the draw did not touch. That is why the seed and the region helpers
// live here — the assertions are about the background, not about the draw, and each test file is
// left holding one load op and one expectation about that background.

#include "../test_gpu.h"

#include <string>

namespace agfxtest
{
    constexpr uint32_t kPassActionWidth = 128;
    constexpr uint32_t kPassActionHeight = 128;
    constexpr agfxTextureFormat kPassActionFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief The pass's clear color and the draw's fill color, as the exact 8-bit values they must
    /// read back as. Held in 8-bit and converted to float rather than the reverse: an arbitrary
    /// float lands between two 8-bit codes and backends may round it differently, which would turn
    /// an exactness check into a flake. Both are exactly representable.
    constexpr uint8_t kPassActionClearRgba8[4] = {51, 102, 204, 255};
    constexpr uint8_t kPassActionDrawRgba8[4] = {255, 128, 0, 255};

    /// @brief A corner region that the centered triangle never covers, so it holds whatever the load
    /// op left behind. 16x16 rather than a single pixel so that a backend which got the background
    /// right only at the very edge cannot pass.
    constexpr uint32_t kPassActionCornerSize = 16;

    /// @brief What one render pass action test varies.
    struct PassActionState
    {
        agfxLoadOp loadOp = AGFX_LOAD_OPERATION_CLEAR;
        agfxStoreOp storeOp = AGFX_STORE_OPERATION_STORE;
        /// @brief Draw a full-coverage triangle instead of the centered one. Required for the
        /// DONT_CARE test, whose undrawn pixels are undefined and therefore unassertable.
        bool fullscreen = false;
    };

    /// @brief The pattern the attachment is seeded with before the pass. A per-pixel gradient rather
    /// than a flat fill: under LOAD this is what must survive, and a flat background would still
    /// look correct if a backend loaded a shifted or smeared copy of it.
    std::vector<uint8_t> PassActionSeed();

    /// @brief True if every pixel of the given region of `image` equals `rgba`.
    bool RegionEquals(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                      const uint8_t rgba[4]);

    /// @brief True if every pixel of the given region of `image` still holds its PassActionSeed value.
    bool RegionMatchesSeed(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h);

    /// @brief Seeds the attachment, runs the pass per `state`, and writes the result into `outImage`.
    /// @param outError Set to a description when the run failed; empty on success.
    bool RenderPassAction(TestApi api, const PassActionState& state, Image& outImage, std::string& outError);
} // namespace agfxtest
