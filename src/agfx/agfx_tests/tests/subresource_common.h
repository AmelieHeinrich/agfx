/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the tests that point a render pass at one subresource of a multi-subresource
// texture: the clear face/slice/mip family, and the render-to face/slice/mip family.
//
// Both families ask the same question — does what the pass does land on exactly the subresource the
// render target names, and on nothing else? — and differ only in which axis of the subresource index
// is non-zero and in whether the pass draws anything. So the setup lives here once, parameterized by
// SubresourceCase, and each test file carries only its texture shape and its golden.
//
// The interesting half of these tests is the negative half. A pass that affected the whole texture
// instead of the named face/slice/mip produces a *correct-looking* target subresource, so the golden
// alone would pass it. Every other subresource is therefore seeded with a distinct pattern up front
// and checked byte-exact afterwards; that check is what actually holds the mipLevel/arrayLayer fields
// of agfxRenderTargetCreateInfo to account.

#include "../test_gpu.h"

#include <string>

namespace agfxtest
{
    constexpr agfxTextureFormat kSubresourceFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kSubresourceBytesPerPixel = 4;

    /// @brief The clear color, as the exact 8-bit values it must read back as.
    ///
    /// Held in 8-bit and converted to float rather than the other way round: an arbitrary float
    /// like 0.6 lands between two 8-bit codes, and backends are free to round it differently, which
    /// would turn an exactness check into a flake. These values are exactly representable.
    constexpr uint8_t kSubresourceClearRgba8[4] = {64, 153, 230, 255};

    /// @brief The color the render-to tests draw with, likewise exactly representable.
    constexpr uint8_t kSubresourceDrawRgba8[4] = {255, 128, 0, 255};

    /// @brief kSubresourceClearRgba8 as the float RGBA an agfxRenderPassAttachment wants.
    const float* SubresourceClearColorFloat();

    /// @brief The subresource layout one test exercises: a texture shape, plus which single
    /// subresource of it the render pass is pointed at and what the pass does to it.
    struct SubresourceCase
    {
        agfxTextureType type = AGFX_TEXTURE_TYPE_2D;
        uint32_t baseSize = 64;    ///< Width and height of mip 0 (these textures are square).
        uint32_t layerCount = 1;   ///< 6 for a cube, N for an array, 1 for a plain 2D texture.
        uint32_t mipLevels = 1;
        uint32_t targetMip = 0;
        uint32_t targetLayer = 0;

        /// @brief What the pass does to the attachment's existing contents. The clear tests use
        /// CLEAR; the render-to tests use LOAD, so that the seed survives around the triangle and
        /// the draw has to be what changed the image.
        agfxLoadOp loadOp = AGFX_LOAD_OPERATION_CLEAR;

        /// @brief Draw a centered flat-colored triangle (data/shaders/tests/pass_actions.hlsl).
        /// False for the clear tests, whose entire subject is the load op.
        bool drawTriangle = false;

        const char* name = "test subresource pass";
    };

    /// @brief The dimensions of `mip` of a case's texture.
    uint32_t SubresourceMipSize(const SubresourceCase& testCase, uint32_t mip);

    /// @brief The seed one subresource carries before the pass. Exposed so the render-to tests can
    /// assert that the seed survived around the triangle.
    std::vector<uint8_t> SubresourceSeedPixels(uint32_t width, uint32_t height, uint32_t mip, uint32_t layer);

    /// @brief Runs the case's pass against its named subresource through the given API flavor, then
    /// verifies every *other* subresource still holds its seed.
    ///
    /// @param outImage Receives the targeted subresource, for the caller's golden comparison.
    /// @param outError Set to a description when the run failed or a neighbour was disturbed; empty
    ///                 on success. The caller should AGFX_EXPECT_MSG on the returned bool with it.
    bool RunSubresourcePass(TestApi api, const SubresourceCase& testCase, Image& outImage, std::string& outError);
} // namespace agfxtest
