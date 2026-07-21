/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw mesh tasks indirect, 3 commands + drawID".
//
// Three GPU-authored agfxDrawMeshCommands through a task + mesh pipeline, red/green/blue left to
// right. Each command's drawID is read by the *task* stage and handed to the mesh stage in the
// payload, which is what separates this from test_draw_mesh_indirect_multi.cpp: there the mesh stage
// reads drawID directly, here it never sees the root constant at all. A payload that does not cross
// the stage boundary paints all three columns red.
//
// The whole indirect sequence -- zero the count, dispatch a producer that appends the commands,
// prepare, replay -- lives in indirect_common.h, along with why prepare and execute must be handed
// identical execute infos. Indirect submission is optional, so this skips rather than fails where
// the device does not report support.

#include "indirect_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_task_mesh_indirect_multi.png";

    IndirectState State()
    {
        IndirectState state;
        state.kind = IndirectKind::DrawTaskMesh;
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

AGFX_TEST_TEXTURE(DrawTaskMeshIndirectMulti, C, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawTaskMeshIndirectMulti, Cpp, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawTaskMeshIndirectMulti, Ez, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Ez);
}
