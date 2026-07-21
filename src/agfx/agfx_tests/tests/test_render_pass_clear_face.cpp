/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render pass clear face".
//
// Points a render pass at face 3 (-Y) of a six-face cube map and clears it, with all six faces
// seeded distinctly beforehand. The golden holds the cleared face; the other five faces are checked
// byte-exact against their seeds, which is the half that matters — a backend that ignored
// arrayLayer and cleared the whole cube would produce an identical golden image.
//
// Face 3 rather than face 0 so that a backend which silently clamps the layer to zero fails.

#include "subresource_common.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr const char* kGolden = "render_pass_clear_face.png";

    SubresourceCase Case()
    {
        SubresourceCase testCase;
        testCase.type = AGFX_TEXTURE_TYPE_CUBE;
        testCase.baseSize = kSize;
        testCase.layerCount = 6;
        testCase.targetLayer = 3; // -Y
        testCase.name = "test clear face";
        return testCase;
    }
} // namespace

AGFX_TEST_TEXTURE(RenderPassClearFace, C, kSize, kSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::C, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassClearFace, Cpp, kSize, kSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::Cpp, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassClearFace, Ez, kSize, kSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::Ez, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
