/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw indexed indirect, 3 commands + drawID".
//
// Three GPU-authored agfxDrawIndexedCommands, and the one test in this family where position and
// identity travel by different routes: each command carries its own firstIndex (i * 6) *and* its own
// drawID. Position follows the index offset, color follows drawID, so the two can fail
// independently and visibly -- a firstIndex that does not round trip shifts a column into the wrong
// slot, while a broken drawID leaves the columns in place and recolors them.
//
// The whole indirect sequence -- zero the count, dispatch a producer that appends the commands,
// prepare, replay -- lives in indirect_common.h, along with why prepare and execute must be handed
// identical execute infos. Indirect submission is optional, so this skips rather than fails where
// the device does not report support.

#include "indirect_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_indexed_indirect_multi.png";

    IndirectState State()
    {
        IndirectState state;
        state.kind = IndirectKind::DrawIndexed;
        state.commandCount = 3;
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

AGFX_TEST_TEXTURE(DrawIndexedIndirectMulti, C, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawIndexedIndirectMulti, Cpp, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawIndexedIndirectMulti, Ez, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Ez);
}
