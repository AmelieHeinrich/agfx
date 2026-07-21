/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw mesh indirect, one command".
//
// One GPU-authored agfxDrawMeshCommand through a mesh-only pipeline (no task stage): a single mesh
// group emitting the left column as a quad. The same red column the vertex-shader path draws in
// test_draw_indirect.cpp, deliberately, so the mesh bundle's golden is comparable to it by eye.
//
// Worth noting what this pins down that the non-mesh bundles cannot: on Metal the indirect argument
// state for a DRAW_MESH bundle has to include the object and mesh stages, not just vertex.
//
// The whole indirect sequence -- zero the count, dispatch a producer that appends the commands,
// prepare, replay -- lives in indirect_common.h, along with why prepare and execute must be handed
// identical execute infos. Indirect submission is optional, so this skips rather than fails where
// the device does not report support.

#include "indirect_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_mesh_indirect.png";

    IndirectState State()
    {
        IndirectState state;
        state.kind = IndirectKind::DrawMesh;
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

AGFX_TEST_TEXTURE(DrawMeshIndirect, C, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawMeshIndirect, Cpp, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawMeshIndirect, Ez, kIndirectWidth, kIndirectHeight)
{
    RunIndirectTest(ctx, TestApi::Ez);
}
