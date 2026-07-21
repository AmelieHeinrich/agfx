/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "raytrace an AABB (procedural) BLAS".
//
// The procedural counterpart of test_rt_trace_triangle.cpp: build a BLAS over axis-aligned bounding
// boxes rather than triangles and trace it with
// data/shaders/tests/raytracing.hlsl:main_trace_aabb_cs.
//
// Procedural geometry takes a genuinely different path through both backends — there is no built-in
// intersector, so traversal only surfaces the AABB as a *candidate* and the shader has to commit it.
// That means a BLAS which builds its boxes into the wrong place produces no committed hit at all
// rather than a displaced silhouette, so the golden is the box's coverage: an empty image is the
// failure mode this catches.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kOneGeometryGolden = "rt_trace_aabb_one_geometry.png";
    constexpr const char* kMultiGeometryGolden = "rt_trace_aabb_multiple_geometry.png";
} // namespace

AGFX_TEST_TEXTURE(RaytraceAABBOneGeometry, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas blas = RtBuildAabbBlas(gpu, RtAabbData(), 1);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build AABB BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_aabb_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kOneGeometryGolden, image);
}

AGFX_TEST_TEXTURE(RaytraceAABBMultipleGeometry, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // Two boxes side by side. The shader tints by primitive index, so a BLAS that collapses them
    // into one primitive — or reports both as index 0 — differs from one that keeps them distinct.
    RtBlas blas = RtBuildAabbBlas(gpu, RtMultiAabbData(), 2);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build multi-AABB BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_aabb_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kMultiGeometryGolden, image);
}
