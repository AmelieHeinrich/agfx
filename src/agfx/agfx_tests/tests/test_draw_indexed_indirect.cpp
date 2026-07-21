/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw indexed indirect, one command".
//
// The indexed counterpart to test_draw_indirect.cpp: one GPU-authored agfxDrawIndexedCommand, whose
// vertices arrive through an index buffer and a bindless structured buffer rather than from a table
// in the shader. Same single red column, so the two goldens are directly comparable and a difference
// between them is the indexed path rather than the scene.
//
// The whole indirect sequence -- zero the count, dispatch a producer that appends the commands,
// prepare, replay -- lives in indirect_common.h, along with why prepare and execute must be handed
// identical execute infos. Indirect submission is optional, so this skips rather than fails where
// the device does not report support.

#include "indirect_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_indexed_indirect.png";

    IndirectState State()
    {
        IndirectState state;
        state.kind = IndirectKind::DrawIndexed;
        state.commandCount = 1;
        return state;
    }

    void RunIndirectTest(TestContext& ctx, TestApi api)
    {
        Image image;
        const IndirectResult result = RenderIndirect(api, State(), image);
        if (result == IndirectResult::Unsupported) {
            ctx.Skip("device reports no indirect (or mesh shader) support");
            return;
        }
        AGFX_EXPECT_MSG(result == IndirectResult::Ok, "indirect render failed");
        ExpectImageMatchesGolden(ctx, kGolden, image);
    }
} // namespace

AGFX_TEST_TEXTURE(DrawIndexedIndirect, C, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawIndexedIndirect, Cpp, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawIndexedIndirect, Ez, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Ez);
}
