/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// Tests that TLAS handle addresses are not null and can be used in shaders.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    /// @brief Fills in the triangle geometry for the BLAS.
    ///
    /// The geometry is an out-parameter rather than a local, because
    /// agfxBottomLevelAccelerationStructureCreateInfo only stores a *pointer* to it: a helper that
    /// built the geometry on its own stack and returned the create-info would hand back a dangling
    /// pointer, which the backend then reads at create time.
    void FillBLASInfo(agfxBuffer* vertexBuffer, agfxBuffer* indexBuffer,
                      agfxAccelerationStructureGeometry& outGeometry,
                      agfxBottomLevelAccelerationStructureCreateInfo& outInfo)
    {
        outGeometry = agfxAccelerationStructureGeometry{};
        outGeometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
        outGeometry.opaque = 1;
        outGeometry.triangles.vertexBuffer = vertexBuffer;
        outGeometry.triangles.vertexOffset = 0;
        outGeometry.triangles.vertexCount = 3;
        outGeometry.triangles.indexBuffer = indexBuffer;
        outGeometry.triangles.indexOffset = 0;
        outGeometry.triangles.indexCount = 3;

        outInfo = agfxBottomLevelAccelerationStructureCreateInfo{};
        outInfo.geometries = &outGeometry;
        outInfo.geometryCount = 1;
    }
} // namespace

AGFX_TEST_VALIDATION(TLASAddressNotNull, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    // Create simple triangle geometry
    float vertices[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    uint32_t indices[] = { 0, 1, 2 };

    agfxBufferCreateInfo vertexBufferInfo{};
    vertexBufferInfo.size = sizeof(vertices);
    vertexBufferInfo.stride = sizeof(float) * 3;
    vertexBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ);
    vertexBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxBuffer* vertexBuffer = agfxBufferCreate(device, &vertexBufferInfo);
    AGFX_EXPECT_NOT_NULL(vertexBuffer);

    agfxBufferCreateInfo indexBufferInfo{};
    indexBufferInfo.size = sizeof(indices);
    indexBufferInfo.stride = sizeof(uint32_t);
    indexBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ);
    indexBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxBuffer* indexBuffer = agfxBufferCreate(device, &indexBufferInfo);
    AGFX_EXPECT_NOT_NULL(indexBuffer);

    UploadBuffer(device, gpu.Queue(), vertexBuffer, vertices, sizeof(vertices), AGFX_RESOURCE_STATE_COMMON);
    UploadBuffer(device, gpu.Queue(), indexBuffer, indices, sizeof(indices), AGFX_RESOURCE_STATE_COMMON);

    // Create BLAS
    agfxAccelerationStructureGeometry geometry{};
    agfxAccelerationStructureCreateInfo blasInfo{};
    blasInfo.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    FillBLASInfo(vertexBuffer, indexBuffer, geometry, blasInfo.bottomLevel);
    blasInfo.allowUpdate = 0;
    blasInfo.name = "test blas";

    agfxAccelerationStructure* blas = agfxAccelerationStructureCreate(device, &blasInfo);
    AGFX_EXPECT_NOT_NULL(blas);

    agfxAccelerationStructureSizes blasSizes{};
    agfxAccelerationStructureGetSizes(device, blas, &blasSizes);

    agfxBufferCreateInfo scratchBufferInfo{};
    scratchBufferInfo.size = blasSizes.scratchBufferSize;
    scratchBufferInfo.stride = 1;
    scratchBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    scratchBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxBuffer* scratchBuffer = agfxBufferCreate(device, &scratchBufferInfo);
    AGFX_EXPECT_NOT_NULL(scratchBuffer);

    // Build BLAS
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferAccelerationStructureBarrier(cmd, blas, AGFX_RESOURCE_STATE_COMMON,
                                                      AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "build blas");
        agfxComputePassBuildAccelerationStructure(pass, blas, scratchBuffer, 0);
        agfxComputePassEnd(pass);
    });

    // Create TLAS
    agfxTopLevelAccelerationStructureCreateInfo tlasTopLevel{};
    tlasTopLevel.maxInstanceCount = 1;

    agfxAccelerationStructureCreateInfo tlasInfo{};
    tlasInfo.type = AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInfo.topLevel = tlasTopLevel;
    tlasInfo.allowUpdate = 0;
    tlasInfo.name = "test tlas";

    agfxAccelerationStructure* tlas = agfxAccelerationStructureCreate(device, &tlasInfo);
    AGFX_EXPECT_NOT_NULL(tlas);

    // Add instance to TLAS
    agfxAccelerationStructureInstance instance{};
    instance.blas = blas;
    instance.transform[0] = 1.0f; instance.transform[1] = 0.0f; instance.transform[2] = 0.0f; instance.transform[3] = 0.0f;
    instance.transform[4] = 0.0f; instance.transform[5] = 1.0f; instance.transform[6] = 0.0f; instance.transform[7] = 0.0f;
    instance.transform[8] = 0.0f; instance.transform[9] = 0.0f; instance.transform[10] = 1.0f; instance.transform[11] = 0.0f;
    instance.userID = 0;
    instance.opaque = 1;

    agfxAccelerationStructureAddInstances(tlas, &instance, 1);

    agfxAccelerationStructureSizes tlasSizes{};
    agfxAccelerationStructureGetSizes(device, tlas, &tlasSizes);

    agfxBufferCreateInfo tlasScratchBufferInfo{};
    tlasScratchBufferInfo.size = tlasSizes.scratchBufferSize;
    tlasScratchBufferInfo.stride = 1;
    tlasScratchBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    tlasScratchBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxBuffer* tlasScratchBuffer = agfxBufferCreate(device, &tlasScratchBufferInfo);
    AGFX_EXPECT_NOT_NULL(tlasScratchBuffer);

    // Build TLAS
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferAccelerationStructureBarrier(cmd, tlas, AGFX_RESOURCE_STATE_COMMON,
                                                      AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
        agfxCommandBufferAccelerationStructureBarrier(cmd, blas, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                                      AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 0);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "build tlas");
        agfxComputePassBuildAccelerationStructure(pass, tlas, tlasScratchBuffer, 0);
        agfxComputePassEnd(pass);
    });

    // Verify TLAS handle is not null
    uint64_t tlasHandle = agfxAccelerationStructureGetHandle(tlas);
    AGFX_EXPECT_MSG(tlasHandle != 0, "TLAS handle should not be null");

    // Cleanup
    agfxAccelerationStructureDestroy(device, tlas);
    agfxAccelerationStructureDestroy(device, blas);
    agfxBufferDestroy(device, tlasScratchBuffer);
    agfxBufferDestroy(device, scratchBuffer);
    agfxBufferDestroy(device, indexBuffer);
    agfxBufferDestroy(device, vertexBuffer);
}

AGFX_TEST_VALIDATION(TLASAddressNotNull, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    // Create simple triangle geometry
    float vertices[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    uint32_t indices[] = { 0, 1, 2 };

    agfxBufferCreateInfo vertexBufferInfo{};
    vertexBufferInfo.size = sizeof(vertices);
    vertexBufferInfo.stride = sizeof(float) * 3;
    vertexBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ);
    vertexBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Buffer vertexBuffer = device.CreateBuffer(vertexBufferInfo);
    AGFX_EXPECT_NOT_NULL(vertexBuffer.Get());

    agfxBufferCreateInfo indexBufferInfo{};
    indexBufferInfo.size = sizeof(indices);
    indexBufferInfo.stride = sizeof(uint32_t);
    indexBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ);
    indexBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Buffer indexBuffer = device.CreateBuffer(indexBufferInfo);
    AGFX_EXPECT_NOT_NULL(indexBuffer.Get());

    UploadBuffer(device.Get(), queue, vertexBuffer, vertices, sizeof(vertices),
                 AGFX_RESOURCE_STATE_COMMON);
    UploadBuffer(device.Get(), queue, indexBuffer, indices, sizeof(indices),
                 AGFX_RESOURCE_STATE_COMMON);

    // Create BLAS
    agfxAccelerationStructureGeometry geometry{};
    geometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
    geometry.opaque = 1;
    geometry.triangles.vertexBuffer = vertexBuffer;
    geometry.triangles.vertexOffset = 0;
    geometry.triangles.vertexCount = 3;
    geometry.triangles.indexBuffer = indexBuffer;
    geometry.triangles.indexOffset = 0;
    geometry.triangles.indexCount = 3;

    agfxAccelerationStructureCreateInfo blasInfo{};
    blasInfo.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInfo.bottomLevel.geometries = &geometry;
    blasInfo.bottomLevel.geometryCount = 1;
    blasInfo.allowUpdate = 0;
    blasInfo.name = "test blas";

    agfx::AccelerationStructure blas = device.CreateAccelerationStructure(blasInfo);
    AGFX_EXPECT_NOT_NULL(blas.Get());

    const agfxAccelerationStructureSizes blasSizes = blas.GetSizes();

    agfxBufferCreateInfo scratchBufferInfo{};
    scratchBufferInfo.size = blasSizes.scratchBufferSize;
    scratchBufferInfo.stride = 1;
    scratchBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    scratchBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Buffer scratchBuffer = device.CreateBuffer(scratchBufferInfo);
    AGFX_EXPECT_NOT_NULL(scratchBuffer.Get());

    // Build BLAS
    cmd.Begin();
    cmd.AccelerationStructureBarrier(blas, AGFX_RESOURCE_STATE_COMMON,
                                     AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("build blas");
        pass.BuildAccelerationStructure(blas, scratchBuffer, 0);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    // Create TLAS
    agfxTopLevelAccelerationStructureCreateInfo tlasTopLevel{};
    tlasTopLevel.maxInstanceCount = 1;

    agfxAccelerationStructureCreateInfo tlasInfo{};
    tlasInfo.type = AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInfo.topLevel = tlasTopLevel;
    tlasInfo.allowUpdate = 0;
    tlasInfo.name = "test tlas";

    agfx::AccelerationStructure tlas = device.CreateAccelerationStructure(tlasInfo);
    AGFX_EXPECT_NOT_NULL(tlas.Get());

    // Add instance to TLAS
    agfxAccelerationStructureInstance instance{};
    instance.blas = blas;
    instance.transform[0] = 1.0f; instance.transform[1] = 0.0f; instance.transform[2] = 0.0f; instance.transform[3] = 0.0f;
    instance.transform[4] = 0.0f; instance.transform[5] = 1.0f; instance.transform[6] = 0.0f; instance.transform[7] = 0.0f;
    instance.transform[8] = 0.0f; instance.transform[9] = 0.0f; instance.transform[10] = 1.0f; instance.transform[11] = 0.0f;
    instance.userID = 0;
    instance.opaque = 1;

    tlas.AddInstances(&instance, 1);

    const agfxAccelerationStructureSizes tlasSizes = tlas.GetSizes();

    agfxBufferCreateInfo tlasScratchBufferInfo{};
    tlasScratchBufferInfo.size = tlasSizes.scratchBufferSize;
    tlasScratchBufferInfo.stride = 1;
    tlasScratchBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    tlasScratchBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Buffer tlasScratchBuffer = device.CreateBuffer(tlasScratchBufferInfo);
    AGFX_EXPECT_NOT_NULL(tlasScratchBuffer.Get());

    // Build TLAS
    cmd.Begin();
    cmd.AccelerationStructureBarrier(tlas, AGFX_RESOURCE_STATE_COMMON,
                                     AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, false);
    cmd.AccelerationStructureBarrier(blas, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                     AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("build tlas");
        pass.BuildAccelerationStructure(tlas, tlasScratchBuffer, 0);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 2);
    fence.Wait(2, UINT64_MAX);

    // Verify TLAS handle is not null
    uint64_t tlasHandle = tlas.GetHandle();
    AGFX_EXPECT_MSG(tlasHandle != 0, "TLAS handle should not be null");
}
