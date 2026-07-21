/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "ray hit attributes".
//
// What a committed hit reports, as opposed to whether it happened at all. Traces with
// data/shaders/tests/raytracing.hlsl:main_trace_attributes_cs, which packs barycentrics into red and
// green and the primitive/instance indices into blue.
//
// The attributes share one shader and one image on purpose: they are read from the same committed
// hit, and a backend that gets the hit record layout wrong tends to corrupt several at once.
// Splitting them into one test each would report four failures for one bug while still needing four
// near-identical scenes. What varies between these tests is the *scene* — a scene with one
// primitive cannot show a primitive index is wrong, and a scene with one instance cannot show an
// instance ID is — so each case builds the geometry that makes its attribute observable.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kBarycentricsGolden = "rt_ray_barycentrics.png";
    constexpr const char* kPrimitiveIDGolden = "rt_ray_primitive_id.png";
    constexpr const char* kUserIDGolden = "rt_ray_user_id.png";
    constexpr const char* kNonOpaqueGolden = "rt_ray_non_opaque.png";
} // namespace

AGFX_TEST_TEXTURE(RTRayBarycentrics, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // A single large triangle: barycentrics vary smoothly across it, so the golden is a two-channel
    // gradient. Swapped or transposed barycentrics change the direction of that gradient, which a
    // small triangle covering a handful of texels would not show.
    RtBlas blas = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build triangle BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_attributes_cs", image),
                    "trace failed");

    ExpectImageMatchesGolden(ctx, kBarycentricsGolden, image);
}

AGFX_TEST_TEXTURE(RTRayPrimitiveID, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // Three primitives in one BLAS, so the blue channel steps between them. With a single-primitive
    // BLAS every index is 0 and a backend that always reports 0 would pass.
    RtBlas blas = RtBuildMultiTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build multi-triangle BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_attributes_cs", image),
                    "trace failed");

    ExpectImageMatchesGolden(ctx, kPrimitiveIDGolden, image);
}

AGFX_TEST_TEXTURE(RTRayUserID, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    RtBlas blas = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build triangle BLAS");

    // Instance IDs deliberately non-contiguous and not starting at zero: an implementation that
    // reports the instance's *array position* instead of its userID matches 0,1,2 by accident but
    // cannot match 3,5,6.
    agfxAccelerationStructureInstance first{};
    first.blas = blas.blas;
    first.userID = 3;
    first.opaque = 1;
    RtTranslationTransform(first.transform, -0.5f, 0.3f, 0.2f);

    agfxAccelerationStructureInstance second{};
    second.blas = blas.blas;
    second.userID = 5;
    second.opaque = 1;
    RtTranslationTransform(second.transform, 0.0f, -0.2f, 0.8f);

    agfxAccelerationStructureInstance third{};
    third.blas = blas.blas;
    third.userID = 6;
    third.opaque = 1;
    RtTranslationTransform(third.transform, 0.5f, 0.3f, 1.4f);

    RtTlas tlas = RtBuildTlas(gpu, {first, second, third});
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_attributes_cs", image),
                    "trace failed");

    ExpectImageMatchesGolden(ctx, kUserIDGolden, image);
}

AGFX_TEST_TEXTURE(RTRayNonOpaqueCandidate, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // Geometry flagged non-opaque, traced by the opacity shader, which rejects every non-opaque
    // candidate it is offered. The triangle therefore does *not* appear as a hit — it shows as the
    // blue "candidate seen and skipped" marker instead. That distinguishes three otherwise
    // identical-looking outcomes: the flag honoured (blue), the flag dropped so the hit commits
    // anyway (red), and the geometry missing altogether (black).
    RtBlas blas = RtBuildTriangleBlas(gpu, /*opaque=*/false);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build non-opaque triangle BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas, /*userID=*/0, /*opaque=*/false);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_opacity_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kNonOpaqueGolden, image);
}
