/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "scratch buffer reuse and offsets" and "geometry/instance opacity flags".
//
// Two groups that share a shape: build an acceleration structure some non-default way, then check
// tracing it still produces the ordinary image.
//
// The scratch tests exist because scratch memory is the easiest thing to get subtly wrong. A build
// that reuses a previous build's scratch buffer, or is handed a suballocated offset into a larger
// one, must produce a bit-identical result to a build with its own fresh allocation — so both reuse
// the plain triangle golden rather than one of their own. A golden they alone owned would happily
// bless a corrupted structure.
//
// The opacity tests cover the opaque flag at the two levels it can be set — per-geometry on a BLAS
// and per-instance on a TLAS — and, more importantly, how they interact. They trace with
// main_trace_opacity_cs, which rejects non-opaque candidates, so an honoured flag (blue) and a
// dropped one (red) are different images rather than the same silhouette.
//
// The interaction is the part worth knowing: AGFX maps instance.opaque onto the backends' *force*
// flags, so an opaque instance overrides non-opaque geometry beneath it. Three tests pin the corners
// down — both opaque, geometry non-opaque under an opaque instance, and a non-opaque instance.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    // Deliberately the same golden the plain build produces in test_rt_trace_triangle.cpp: the
    // point of these two tests is that the scratch arrangement changes nothing observable.
    constexpr const char* kTriangleGolden = "rt_trace_triangle_one_geometry.png";

    constexpr const char* kInstanceOverrideGolden = "rt_opacity_instance_overrides_geometry.png";
    constexpr const char* kTlasOpacityGolden = "rt_opacity_tlas_non_opaque.png";
    constexpr const char* kOpaqueGolden = "rt_opacity_opaque.png";

    /// @brief Creates an unbuilt triangle BLAS and its input buffers, leaving the build to the
    /// caller — the scratch tests drive the build themselves, which is the whole point of them.
    RtBlas CreateUnbuiltTriangleBlas(GpuFixture& gpu, agfxAccelerationStructureGeometry& outGeometry,
                                     agfxAccelerationStructureCreateInfo& outInfo)
    {
        RtBlas result;
        result.device = gpu.Device();

        const std::vector<float>& vertices = RtTriangleVertices();
        const std::vector<uint32_t>& indices = RtTriangleIndices();

        result.vertexBuffer = RtCreateInputBuffer(gpu, vertices.data(), vertices.size() * sizeof(float),
                                                  sizeof(float) * 3, "rt vertices");
        result.indexBuffer = RtCreateInputBuffer(gpu, indices.data(), indices.size() * sizeof(uint32_t),
                                                 sizeof(uint32_t), "rt indices");
        if (!result.vertexBuffer || !result.indexBuffer) {
            return result;
        }

        outGeometry = agfxAccelerationStructureGeometry{};
        outGeometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
        outGeometry.opaque = 1;
        outGeometry.triangles.vertexBuffer = result.vertexBuffer;
        outGeometry.triangles.vertexCount = (uint32_t)(vertices.size() / 3);
        outGeometry.triangles.indexBuffer = result.indexBuffer;
        outGeometry.triangles.indexCount = (uint32_t)indices.size();

        outInfo = agfxAccelerationStructureCreateInfo{};
        outInfo.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        outInfo.bottomLevel.geometries = &outGeometry;
        outInfo.bottomLevel.geometryCount = 1;
        outInfo.allowUpdate = 0;
        outInfo.name = "rt scratch blas";

        result.blas = agfxAccelerationStructureCreate(gpu.Device(), &outInfo);
        return result;
    }
} // namespace

AGFX_TEST_TEXTURE(RTScratchBufferReuse, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    agfxAccelerationStructureGeometry firstGeometry{};
    agfxAccelerationStructureCreateInfo firstInfo{};
    RtBlas first = CreateUnbuiltTriangleBlas(gpu, firstGeometry, firstInfo);
    AGFX_EXPECT_MSG(first.Valid(), "failed to create first BLAS");

    agfxAccelerationStructureGeometry secondGeometry{};
    agfxAccelerationStructureCreateInfo secondInfo{};
    RtBlas second = CreateUnbuiltTriangleBlas(gpu, secondGeometry, secondInfo);
    AGFX_EXPECT_MSG(second.Valid(), "failed to create second BLAS");

    agfxAccelerationStructureSizes firstSizes{};
    agfxAccelerationStructureSizes secondSizes{};
    agfxAccelerationStructureGetSizes(device, first.blas, &firstSizes);
    agfxAccelerationStructureGetSizes(device, second.blas, &secondSizes);

    const uint64_t scratchSize =
        firstSizes.scratchBufferSize > secondSizes.scratchBufferSize ? firstSizes.scratchBufferSize
                                                                     : secondSizes.scratchBufferSize;
    agfxBuffer* scratch = RtCreateScratchBuffer(device, scratchSize, "shared scratch");
    AGFX_EXPECT_NOT_NULL(scratch);

    // Both builds go through the one scratch buffer, in separate submissions so the first has fully
    // completed before the second overwrites the memory. The second BLAS is the one traced: if the
    // first build left state behind that the second depends on, or the second's build read stale
    // scratch, the image diverges from the plain-build golden.
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "build first");
        agfxComputePassBuildAccelerationStructure(pass, first.blas, scratch, 0);
        agfxComputePassEnd(pass);
    });

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "build second");
        agfxComputePassBuildAccelerationStructure(pass, second.blas, scratch, 0);
        agfxComputePassEnd(pass);
        agfxCommandBufferAccelerationStructureBarrier(
            cmd, second.blas, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
    });

    agfxBufferDestroy(device, scratch);

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, second.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kTriangleGolden, image);
}

AGFX_TEST_TEXTURE(RTBuildWithOffsetScratchBuffer, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    agfxAccelerationStructureGeometry geometry{};
    agfxAccelerationStructureCreateInfo info{};
    RtBlas blas = CreateUnbuiltTriangleBlas(gpu, geometry, info);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to create BLAS");

    agfxAccelerationStructureSizes sizes{};
    agfxAccelerationStructureGetSizes(device, blas.blas, &sizes);

    // 256 bytes clears every scratch alignment requirement either backend imposes; a build that
    // ignores the offset and writes at the base of the buffer still produces a valid-looking
    // structure, so the golden comparison is what catches it.
    constexpr uint64_t kOffset = 256;
    agfxBuffer* scratch = RtCreateScratchBuffer(device, sizes.scratchBufferSize + kOffset,
                                                "offset scratch");
    AGFX_EXPECT_NOT_NULL(scratch);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "build at offset");
        agfxComputePassBuildAccelerationStructure(pass, blas.blas, scratch, kOffset);
        agfxComputePassEnd(pass);
        agfxCommandBufferAccelerationStructureBarrier(
            cmd, blas.blas, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
    });

    agfxBufferDestroy(device, scratch);

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kTriangleGolden, image);
}

AGFX_TEST_TEXTURE(RTOpaqueGeometry, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // The control for the two tests below: opaque at both levels, so the opacity shader is never
    // offered a non-opaque candidate and commits the hit outright.
    RtBlas blas = RtBuildTriangleBlas(gpu, /*opaque=*/true);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas, /*userID=*/0, /*opaque=*/true);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_opacity_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kOpaqueGolden, image);
}

AGFX_TEST_TEXTURE(RTInstanceOpacityOverridesGeometry, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // Non-opaque geometry inside an opaque instance. The instance wins: AGFX maps instance.opaque
    // onto MTLAccelerationStructureInstanceOptionOpaque and
    // D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE, both of which *force* opacity rather than
    // deferring to the geometry, so the hit commits and the golden is the opaque (red) image.
    //
    // Pinning this down is worth a test precisely because it is counterintuitive — the natural
    // reading is that marking geometry non-opaque makes it non-opaque — and because it means the
    // per-geometry flag is not independently observable from the host once an opaque instance
    // references it. RTTlasOpacity covers the other direction.
    RtBlas blas = RtBuildTriangleBlas(gpu, /*opaque=*/false);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas, /*userID=*/0, /*opaque=*/true);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_opacity_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kInstanceOverrideGolden, image);
}

AGFX_TEST_TEXTURE(RTTlasOpacity, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");

    // The mirror of RTBlasOpacity: opaque geometry, non-opaque instance. Whether an instance can
    // override its BLAS's flag is exactly the behaviour worth pinning down, and running both
    // directions means a backend that honours only one level cannot pass both tests.
    RtBlas blas = RtBuildTriangleBlas(gpu, /*opaque=*/true);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build BLAS");

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas, /*userID=*/0, /*opaque=*/false);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_opacity_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kTlasOpacityGolden, image);
}
