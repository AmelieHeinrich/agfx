/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw mesh shader".
//
// Dispatches two mesh groups through a mesh-only pipeline (no task stage); each emits one triangle,
// giving the same two triangles the classic vertex-shader path draws in test_draw_winding.cpp. The
// golden is deliberately comparable to that one by eye: the geometry is identical, so anything that
// differs is the mesh path rather than the scene.
//
// What this pins down beyond "a mesh pipeline runs at all" is that SV_GroupID reaches the shader and
// selects geometry per group. Both triangles come from one dispatch of two groups, on opposite
// halves of the target in different colors -- a backend that dispatched one group renders half the
// image, and one that fed every group the same ID stacks both triangles in one place.
//
// Mesh shading is optional, so the test skips rather than fails where the device does not report
// support; see mesh_common.h.

#include "mesh_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_mesh_shader.png";

    MeshState State()
    {
        MeshState state;
        state.useTaskShader = false;
        return state;
    }

    void RunMeshTest(TestContext& ctx, TestApi api)
    {
        Image image;
        const MeshResult result = RenderMesh(api, State(), image);
        if (result == MeshResult::Unsupported) {
            ctx.Skip("device reports no mesh shader support");
            return;
        }
        AGFX_EXPECT_MSG(result == MeshResult::Ok, "mesh render failed");
        ExpectImageMatchesGolden(ctx, kGolden, image);
    }
} // namespace

AGFX_TEST_TEXTURE(DrawMeshShader, C, kMeshWidth, kMeshHeight)
{
    RunMeshTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawMeshShader, Cpp, kMeshWidth, kMeshHeight)
{
    RunMeshTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawMeshShader, Ez, kMeshWidth, kMeshHeight)
{
    RunMeshTest(ctx, TestApi::Ez);
}
