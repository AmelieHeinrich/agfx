/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "raytrace a triangle BLAS".
//
// The baseline raytracing test: build a bottom-level acceleration structure over triangle geometry,
// wrap it in a single-instance TLAS, and trace it with
// data/shaders/tests/raytracing.hlsl:main_trace_cs. The golden is the triangle's silhouette with a
// depth ramp in the green channel, so a BLAS that builds into the wrong place, drops its geometry,
// or lands at the wrong depth all produce visibly different images rather than the same "not null".
//
// The one- and multiple-geometry cases are separate tests because a BLAS holding several primitives
// exercises index-buffer offsetting and primitive ordering that a single triangle never touches;
// the multi case stacks its three triangles at different depths so the green ramp orders them.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kOneGeometryGolden = "rt_trace_triangle_one_geometry.png";
    constexpr const char* kMultiGeometryGolden = "rt_trace_triangle_multiple_geometry.png";
} // namespace

AGFX_TEST_TEXTURE(RaytraceTriangleOneGeometry, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas blas = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build triangle BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kOneGeometryGolden, image);
}

AGFX_TEST_TEXTURE(RaytraceTriangleMultipleGeometry, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas blas = RtBuildMultiTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build multi-triangle BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kMultiGeometryGolden, image);
}
