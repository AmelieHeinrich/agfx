/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "acceleration structure update, compact and copy".
//
// The three things that can be done to an already-built acceleration structure. All of them are
// checked by tracing the *result* rather than by inspecting handles, because none of these
// operations has any host-visible effect: a copy that silently produced an empty structure and a
// copy that worked look identical from the CPU.
//
// Copy and compact reuse the plain triangle golden on purpose — both are meant to preserve exactly
// what tracing the source produced, so sharing the source's golden is the assertion. Update gets its
// own golden because it is the one operation intended to *change* the result.

#include "../test_rt_scene.h"

namespace
{
    using namespace agfxtest;

    // Copy and compact must reproduce what the original build produced, byte for byte in the image.
    constexpr const char* kTriangleGolden = "rt_trace_triangle_one_geometry.png";
    constexpr const char* kUpdatedGolden = "rt_as_updated_blas.png";
} // namespace

AGFX_TEST_TEXTURE(RTUpdateBLAS, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    // allowUpdate is required for a refit; without it the update call has nothing to work with.
    RtBlas blas = RtBuildTriangleBlas(gpu, /*opaque=*/true, /*allowUpdate=*/true);
    AGFX_EXPECT_MSG(blas.Valid(), "failed to build updatable BLAS");

    // Rewrite the vertex buffer under the built structure, then refit. The new triangle is both
    // moved and shrunk so an update that silently no-ops leaves the original silhouette, which the
    // golden distinguishes from the refitted one. Vertex *count* is unchanged: a refit may move
    // vertices but not change topology.
    const std::vector<float> movedVertices = {
        0.35f, 0.30f,  0.8f, //
        -0.2f, -0.35f, 0.8f, //
        0.85f, -0.35f, 0.8f, //
    };
    AGFX_EXPECT_MSG(UploadBuffer(device, gpu.Queue(), blas.vertexBuffer, movedVertices.data(),
                                 movedVertices.size() * sizeof(float), AGFX_RESOURCE_STATE_COMMON),
                    "failed to upload moved vertices");

    agfxAccelerationStructureSizes sizes{};
    agfxAccelerationStructureGetSizes(device, blas.blas, &sizes);

    // An update uses updateScratchBufferSize, not scratchBufferSize — they differ, and a refit given
    // the build-sized scratch is a real source of overruns.
    agfxBuffer* updateScratch =
        RtCreateScratchBuffer(device, sizes.updateScratchBufferSize, "update scratch");
    AGFX_EXPECT_NOT_NULL(updateScratch);

    // Refit in place: source and destination are the same structure.
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "update blas");
        agfxComputePassUpdateAccelerationStructure(pass, blas.blas, blas.blas, updateScratch, 0);
        agfxComputePassEnd(pass);
        agfxCommandBufferAccelerationStructureBarrier(
            cmd, blas.blas, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
    });

    agfxBufferDestroy(device, updateScratch);

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, blas.blas);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    AGFX_EXPECT_MSG(RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image), "trace failed");

    ExpectImageMatchesGolden(ctx, kUpdatedGolden, image);
}

AGFX_TEST_TEXTURE(RTCompactBLAS, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    RtBlas source = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(source.Valid(), "failed to build source BLAS");

    // The destination is created from the same description as the source rather than through
    // agfxAccelerationStructureCreateCompacted: the compacted size that entry point wants is not
    // reported by agfxAccelerationStructureGetSizes, so sizing it correctly is not expressible
    // through the public API today. Compacting into a normally-sized structure still exercises the
    // compaction path itself, which is what this test is for.
    agfxAccelerationStructureGeometry geometry{};
    geometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
    geometry.opaque = 1;
    geometry.triangles.vertexBuffer = source.vertexBuffer;
    geometry.triangles.vertexCount = 3;
    geometry.triangles.indexBuffer = source.indexBuffer;
    geometry.triangles.indexCount = 3;

    agfxAccelerationStructureCreateInfo info{};
    info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    info.bottomLevel.geometries = &geometry;
    info.bottomLevel.geometryCount = 1;
    info.allowUpdate = 0;
    info.name = "compacted blas";

    agfxAccelerationStructure* compacted = agfxAccelerationStructureCreate(device, &info);
    AGFX_EXPECT_NOT_NULL(compacted);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "compact blas");
        agfxComputePassCompactAccelerationStructure(pass, source.blas, compacted);
        agfxComputePassEnd(pass);
        agfxCommandBufferAccelerationStructureBarrier(
            cmd, compacted, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
    });

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, compacted);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    const bool traced = RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image);

    // Destroyed before the assertion so a mismatch does not also leak the structure.
    agfxAccelerationStructureDestroy(device, compacted);

    AGFX_EXPECT_MSG(traced, "trace failed");
    ExpectImageMatchesGolden(ctx, kTriangleGolden, image);
}

AGFX_TEST_TEXTURE(RTCopyAccelerationStructure, C, kRtWidth, kRtHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    RtBlas source = RtBuildTriangleBlas(gpu);
    AGFX_EXPECT_MSG(source.Valid(), "failed to build source BLAS");

    agfxAccelerationStructureGeometry geometry{};
    geometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
    geometry.opaque = 1;
    geometry.triangles.vertexBuffer = source.vertexBuffer;
    geometry.triangles.vertexCount = 3;
    geometry.triangles.indexBuffer = source.indexBuffer;
    geometry.triangles.indexCount = 3;

    agfxAccelerationStructureCreateInfo info{};
    info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    info.bottomLevel.geometries = &geometry;
    info.bottomLevel.geometryCount = 1;
    info.allowUpdate = 0;
    info.name = "copied blas";

    // Deliberately never built: the copy is the only thing that puts contents in it, so tracing the
    // triangle golden through it proves the copy actually moved the structure.
    agfxAccelerationStructure* copy = agfxAccelerationStructureCreate(device, &info);
    AGFX_EXPECT_NOT_NULL(copy);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy blas");
        agfxComputePassCopyAccelerationStructure(pass, source.blas, copy);
        agfxComputePassEnd(pass);
        agfxCommandBufferAccelerationStructureBarrier(
            cmd, copy, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
    });

    RtTlas tlas = RtBuildSingleInstanceTlas(gpu, copy);
    AGFX_EXPECT_MSG(tlas.Valid(), "failed to build TLAS");

    Image image;
    const bool traced = RtTraceToImage(gpu, tlas.tlas, "main_trace_cs", image);

    agfxAccelerationStructureDestroy(device, copy);

    AGFX_EXPECT_MSG(traced, "trace failed");
    ExpectImageMatchesGolden(ctx, kTriangleGolden, image);
}
