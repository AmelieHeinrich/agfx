/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "mixed triangle and AABB geometry in one TLAS".
//
// Triangle and procedural geometry reaching the same TLAS. Both backends require the two kinds to
// live in *separate* bottom-level structures — a BLAS holds one geometry type — so the mixing
// happens at the instance level, and this test is really about a TLAS whose instances point at
// bottom-level structures of different kinds.
//
// Traced twice through the same TLAS with two shaders, because a RayQuery only commits one candidate
// type: main_trace_cs commits triangles and ignores procedural candidates, main_trace_aabb_cs does
// the reverse. Two goldens over one scene means an instance that silently resolved to the wrong
// BLAS shows up as geometry appearing in the image where the other kind should be.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kTrianglesGolden = "rt_mixed_geometry_triangles.png";
    constexpr const char* kAabbsGolden = "rt_mixed_geometry_aabbs.png";
} // namespace

AGFX_TEST_TEXTURE(RaytraceMixedGeometryTriangles, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas triangles = RtBuildTriangleBlas(gpu);
    RtBlas aabbs = RtBuildAabbBlas(gpu, RtAabbData(), 1);
    AGFX_EXPECT_MSG(triangles.Valid() && aabbs.Valid(), "failed to build BLASes");

    agfxAccelerationStructureInstance triangleInstance{};
    triangleInstance.blas = triangles.blas;
    triangleInstance.userID = 0;
    triangleInstance.opaque = 1;
    RtTranslationTransform(triangleInstance.transform, -0.4f, 0.0f, 0.3f);

    agfxAccelerationStructureInstance aabbInstance{};
    aabbInstance.blas = aabbs.blas;
    aabbInstance.userID = 1;
    aabbInstance.opaque = 1;
    RtTranslationTransform(aabbInstance.transform, 0.4f, 0.0f, 0.9f);

    RtTlas tlas = RtBuildTlas(gpu, {triangleInstance, aabbInstance});
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kTrianglesGolden, image);
}

AGFX_TEST_TEXTURE(RaytraceMixedGeometryAABBs, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas triangles = RtBuildTriangleBlas(gpu);
    RtBlas aabbs = RtBuildAabbBlas(gpu, RtAabbData(), 1);
    AGFX_EXPECT_MSG(triangles.Valid() && aabbs.Valid(), "failed to build BLASes");

    agfxAccelerationStructureInstance triangleInstance{};
    triangleInstance.blas = triangles.blas;
    triangleInstance.userID = 0;
    triangleInstance.opaque = 1;
    RtTranslationTransform(triangleInstance.transform, -0.4f, 0.0f, 0.3f);

    agfxAccelerationStructureInstance aabbInstance{};
    aabbInstance.blas = aabbs.blas;
    aabbInstance.userID = 1;
    aabbInstance.opaque = 1;
    RtTranslationTransform(aabbInstance.transform, 0.4f, 0.0f, 0.9f);

    RtTlas tlas = RtBuildTlas(gpu, {triangleInstance, aabbInstance});
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_aabb_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kAabbsGolden, image);
}
