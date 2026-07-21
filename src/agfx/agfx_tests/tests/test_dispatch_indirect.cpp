/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "dispatch indirect".
//
// One GPU-authored agfxDispatchCommand replayed on a compute pass, writing the same three-column
// scene the draw bundles rasterize so the goldens are comparable by eye.
//
// This is the one bundle type with no drawID -- agfxDispatchCommand has no such field, since
// indirect compute is expected to carry its own addressing. So what is under test is narrower and
// entirely about the group counts: the producer computes them from the target size and writes them
// into the command, and the consumer covers the target at [numthreads(8,8,1)]. A group count that
// arrives short leaves a black band rather than a subtly wrong image, and one that never arrives
// leaves the target untouched.
//
// The whole indirect sequence -- zero the count, dispatch a producer that appends the commands,
// prepare, replay -- lives in indirect_common.h, along with why prepare and execute must be handed
// identical execute infos. Indirect submission is optional, so this skips rather than fails where
// the device does not report support.

#include "indirect_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "dispatch_indirect.png";

    IndirectState State()
    {
        IndirectState state;
        state.kind = IndirectKind::Dispatch;
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

AGFX_TEST_TEXTURE(DispatchIndirect, C, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DispatchIndirect, Cpp, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DispatchIndirect, Ez, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Ez);
}
