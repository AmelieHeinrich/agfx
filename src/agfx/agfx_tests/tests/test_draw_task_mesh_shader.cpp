/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw task + mesh shader".
//
// The amplified twin of test_draw_mesh_shader.cpp: the draw dispatches *one* task group, which
// amplifies to two mesh groups via DispatchMesh and hands each one a payload. Everything else --
// target, pass, fragment stage, emitted geometry -- is identical, so the difference between the two
// goldens is exactly the task stage's doing.
//
// Two things are under test that the mesh-only path cannot reach:
//
//   - amplification itself. The draw asks for one group; two triangles appear. A backend that
//     ignored DispatchMesh's group count would render one triangle or none.
//   - the payload crossing the stage boundary. Each triangle keeps its own position (from
//     SV_GroupID) but takes its *colors* from the payload, which mesh.hlsl reverses on purpose. A
//     backend that dropped the payload and fell back to SV_GroupID would render each triangle in
//     its own colors -- which is precisely draw_mesh_shader.png.
//
// So this golden is the mesh-only golden with the two triangles' colors exchanged, and the two
// files differing is itself the assertion that the payload took effect. Routing color rather than
// whole-triangle identity through the payload is what makes that true: see the note on
// EmitTriangle in mesh.hlsl for why the obvious alternative silently tests nothing.
//
// Mesh shading is optional, so the test skips rather than fails where the device does not report
// support; see mesh_common.h.

#include "mesh_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_task_mesh_shader.png";

    MeshState State()
    {
        MeshState state;
        state.useTaskShader = true;
        return state;
    }

    void RunTaskMeshTest(TestContext& ctx, TestApi api)
    {
        Image image;
        const MeshResult result = RenderMesh(api, State(), image);
        if (result == MeshResult::Unsupported) {
            ctx.Skip("device reports no mesh shader support");
            return;
        }
        AGFX_EXPECT_MSG(result == MeshResult::Ok, "task + mesh render failed");
        ExpectImageMatchesGolden(ctx, kGolden, image);
    }
} // namespace

AGFX_TEST_TEXTURE(DrawTaskMeshShader, C, kMeshWidth, kMeshHeight)
{
    RunTaskMeshTest(ctx, TestApi::C);
}

AGFX_TEST_TEXTURE(DrawTaskMeshShader, Cpp, kMeshWidth, kMeshHeight)
{
    RunTaskMeshTest(ctx, TestApi::Cpp);
}

AGFX_TEST_TEXTURE(DrawTaskMeshShader, Ez, kMeshWidth, kMeshHeight)
{
    RunTaskMeshTest(ctx, TestApi::Ez);
}
