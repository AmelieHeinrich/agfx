/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render to texture face".
//
// The drawing counterpart to the clear-face test: a render pass bound to face 3 (-Y) of a six-face
// cube map draws a centered triangle over the face's seeded contents, with loadOp LOAD so the seed
// survives around it. Two things have to be true at once for this to pass — the triangle landed on
// face 3, and the other five faces are untouched — and the second is checked byte-exact against
// their seeds by the shared scaffolding.
//
// Distinct from the clear test in what it exercises on the backend: a clear can be serviced by a
// fast path that never binds the subresource as a real color attachment, whereas a draw has to route
// fragments to it.

#include "subresource_common.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr const char* kGolden = "render_to_texture_face.png";

    SubresourceCase Case()
    {
        SubresourceCase testCase;
        testCase.type = AGFX_TEXTURE_TYPE_CUBE;
        testCase.baseSize = kSize;
        testCase.layerCount = 6;
        testCase.targetLayer = 3; // -Y
        testCase.loadOp = AGFX_LOAD_OPERATION_LOAD;
        testCase.drawTriangle = true;
        testCase.name = "test render to face";
        return testCase;
    }
} // namespace

#define AGFX_RENDER_TO_FACE_CASE(api)                                                              \
    AGFX_TEST_TEXTURE(RenderToTextureFace, api, kSize, kSize)                                      \
    {                                                                                              \
        Image image;                                                                               \
        std::string error;                                                                         \
        AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::api, Case(), image, error), error.c_str());    \
        ExpectImageMatchesGolden(ctx, kGolden, image);                                              \
    }

AGFX_RENDER_TO_FACE_CASE(C)
AGFX_RENDER_TO_FACE_CASE(Cpp)
AGFX_RENDER_TO_FACE_CASE(Ez)
