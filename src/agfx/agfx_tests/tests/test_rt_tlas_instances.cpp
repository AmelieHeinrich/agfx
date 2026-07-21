/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "TLAS instancing".
//
// Everything above the BLAS: how instances placed in a top-level acceleration structure are
// transformed, identified, and shared.
//
// The three cases are separate because they fail independently. One instance checks the instance
// transform is applied at all. Multiple instances check each gets its *own* transform rather than
// the first or last one broadcast — the giveaway a single-instance test cannot produce. BLAS reuse
// checks one bottom-level structure referenced by several instances is not duplicated or aliased,
// which is the whole economic point of a two-level structure and a real bug surface on both
// backends.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kOneInstanceGolden = "rt_tlas_one_instance.png";
    constexpr const char* kMultipleInstancesGolden = "rt_tlas_multiple_instances.png";
    constexpr const char* kBlasReuseGolden = "rt_tlas_blas_reuse.png";

    agfxAccelerationStructureInstance MakeInstance(agfxAccelerationStructure* blas, float x, float y,
                                                   float z, uint32_t userID)
    {
        agfxAccelerationStructureInstance instance{};
        instance.blas = blas;
        instance.userID = userID;
        instance.opaque = 1;
        RtTranslationTransform(instance.transform, x, y, z);
        return instance;
    }
} // namespace

AGFX_TEST_TEXTURE(RaytraceTLASOneInstance, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas blas = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build triangle BLAS");

    // Offset from the origin so the golden shows the instance transform was applied. An identity
    // transform here would be indistinguishable from the transform being ignored entirely.
    RtTlas tlas = RtBuildTlas(gpu, {MakeInstance(blas.blas, -0.3f, 0.2f, 0.5f, 0)});
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kOneInstanceGolden, image);
}

AGFX_TEST_TEXTURE(RaytraceTLASMultipleInstances, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas blas = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build triangle BLAS");

    // Three copies at distinct offsets and depths. Distinct depths matter: two instances that
    // overlap in the image but sit at different z still differ in the golden's green channel, so a
    // TLAS that places them all at one transform collapses to a single silhouette.
    const std::vector<agfxAccelerationStructureInstance> instances = {
        MakeInstance(blas.blas, -0.5f, 0.3f, 0.2f, 0),
        MakeInstance(blas.blas, 0.0f, -0.2f, 0.8f, 1),
        MakeInstance(blas.blas, 0.5f, 0.3f, 1.4f, 2),
    };

    RtTlas tlas = RtBuildTlas(gpu, instances);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kMultipleInstancesGolden, image);
}

AGFX_TEST_TEXTURE(RaytraceBLASReuseInInstances, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // Two different BLASes, each instanced twice, interleaved in the instance array. If the TLAS
    // build associated instances with the wrong bottom-level structure — or deduplicated them by
    // BLAS pointer — the triangle and multi-triangle shapes would land at each other's transforms,
    // which the golden shows as swapped silhouettes rather than as a missing one.
    RtBlas triangle = RtBuildTriangleBlas(gpu);
    RtBlas multi = RtBuildMultiTriangleBlas(gpu);
    AGFX_EXPECT_MSG(triangle.Valid() && multi.Valid(), "failed to build BLASes");

    const std::vector<agfxAccelerationStructureInstance> instances = {
        MakeInstance(triangle.blas, -0.45f, 0.35f, 0.2f, 0),
        MakeInstance(multi.blas, 0.45f, 0.35f, 0.6f, 1),
        MakeInstance(triangle.blas, 0.45f, -0.45f, 1.0f, 2),
        MakeInstance(multi.blas, -0.45f, -0.45f, 1.4f, 3),
    };

    RtTlas tlas = RtBuildTlas(gpu, instances);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kBlasReuseGolden, image);
}
