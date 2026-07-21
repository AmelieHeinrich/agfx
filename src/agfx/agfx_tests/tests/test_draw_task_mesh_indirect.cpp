/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw mesh tasks indirect, one command".
//
// One GPU-authored agfxDrawMeshCommand through a task + mesh pipeline. The task stage amplifies
// 1 -> 1 and forwards drawID to the mesh stage through the payload, so the image is the same single
// red column the mesh-only bundle draws -- the difference between the two goldens is only whether
// the task stage is in the pipeline at all.
//
// The payload forwarding is the point: drawID has to survive both the per-command b1 patch and the
// task -> mesh boundary. Letting the mesh stage re-read AGFX_DRAW_ID() itself would have made the
// task stage untestable here.
//
// The whole indirect sequence -- zero the count, dispatch a producer that appends the commands,
// prepare, replay -- lives in indirect_common.h, along with why prepare and execute must be handed
// identical execute infos. Indirect submission is optional, so this skips rather than fails where
// the device does not report support.

#include "indirect_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_task_mesh_indirect.png";

    IndirectState State()
    {
        IndirectState state;
        state.kind = IndirectKind::DrawTaskMesh;
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

AGFX_TEST_TEXTURE(DrawTaskMeshIndirect, C, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawTaskMeshIndirect, Cpp, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawTaskMeshIndirect, Ez, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Ez);
}
