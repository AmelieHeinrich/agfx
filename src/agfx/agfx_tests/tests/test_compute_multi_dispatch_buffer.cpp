/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute multi dispatch buffer".
//
// Drives data/shaders/tests/multi_dispatch.hlsl four times back to back within one compute pass,
// with a UAV barrier between each dispatch. Every pass read-modify-writes what the previous one
// wrote and folds its own pass index in, so the result is only correct if the barriers really
// serialize the dispatches — a backend that lets them overlap lands somewhere below the golden, and
// the per-pass index means a dropped or reordered dispatch is distinguishable from a lost one.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kElementCount = 64;
    constexpr uint64_t kBufferSize = kElementCount * sizeof(uint32_t);
    constexpr uint32_t kGroupSize = 64;                  // Matches [numthreads(64,1,1)].
    constexpr uint32_t kGroupCount = (kElementCount + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kPassCount = 4;
    constexpr const char* kGolden = "compute_multi_dispatch_buffer.bin";

    /// @brief Mirrors MultiDispatchPushConstants in multi_dispatch.hlsl.
    struct PushConstants
    {
        uint32_t rwBuffer;
        uint32_t elementCount;
        uint32_t passIndex;
        uint32_t padding;
    };

    agfxBufferCreateInfo BufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kBufferSize;
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "multi dispatch";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(ComputeMultiDispatchBuffer, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("multi_dispatch.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile multi_dispatch.hlsl:main_cs");

    const agfxBufferCreateInfo bufferInfo = BufferInfo();
    agfxBuffer* buffer = agfxBufferCreate(device, &bufferInfo);
    AGFX_EXPECT_NOT_NULL(buffer);

    agfxBufferViewCreateInfo viewInfo{};
    viewInfo.buffer = buffer;
    viewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    viewInfo.offset = 0;
    viewInfo.writeable = 1;
    agfxBufferView* view = agfxBufferViewCreate(device, &viewInfo);
    AGFX_EXPECT_NOT_NULL(view);

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    // Pass 0 reads this, so the chain has to start from a known value rather than from whatever
    // the allocator handed back.
    const std::vector<uint32_t> zeros(kElementCount, 0u);
    const bool seeded = UploadBuffer(device, gpu.Queue(), buffer, zeros.data(), kBufferSize,
                                     AGFX_RESOURCE_STATE_COMMON);

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(view);
    constants.elementCount = kElementCount;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "multi dispatch");
        agfxComputePassSetPipeline(pass, pipeline);
        for (uint32_t i = 0; i < kPassCount; ++i) {
            if (i > 0) {
                agfxComputePassBufferUAVBarrier(pass, buffer);
            }
            constants.passIndex = i;
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kGroupCount, 1, 1);
        }
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), buffer, kBufferSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, pipeline);
    agfxBufferViewDestroy(device, view);
    agfxBufferDestroy(device, buffer);

    AGFX_EXPECT_MSG(seeded, "failed to zero the buffer");
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(ComputeMultiDispatchBuffer, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("multi_dispatch.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile multi_dispatch.hlsl:main_cs");

    agfx::Buffer buffer = device.CreateBuffer(BufferInfo());
    AGFX_EXPECT_NOT_NULL(buffer.Get());

    agfxBufferViewCreateInfo viewInfo{};
    viewInfo.buffer = buffer;
    viewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    viewInfo.offset = 0;
    viewInfo.writeable = 1;
    agfx::BufferView view = device.CreateBufferView(viewInfo);
    AGFX_EXPECT_NOT_NULL(view.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint32_t> zeros(kElementCount, 0u);
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, buffer, zeros.data(), kBufferSize,
                                 AGFX_RESOURCE_STATE_COMMON),
                    "failed to zero the buffer");

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)view.GetHandle();
    constants.elementCount = kElementCount;

    cmd.Begin();
    cmd.BufferBarrier(buffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("multi dispatch");
        pass.SetPipeline(pipeline);
        for (uint32_t i = 0; i < kPassCount; ++i) {
            if (i > 0) {
                pass.BufferUAVBarrier(buffer);
            }
            constants.passIndex = i;
            pass.PushConstants(constants);
            pass.Dispatch(kGroupCount, 1, 1);
        }
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, buffer, kBufferSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(ComputeMultiDispatchBuffer, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("multi_dispatch.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile multi_dispatch.hlsl:main_cs");

    agfx::Device& device = context.GetDevice();

    const std::vector<uint32_t> zeros(kElementCount, 0u);
    agfx::ez::Buffer buffer =
        context.CreateStructuredBuffer(zeros.data(), kBufferSize, sizeof(uint32_t), /*shaderWritable*/ true);

    agfx::BufferView& view = buffer.View(AGFX_BUFFER_VIEW_TYPE_RAW, /*writeable*/ true);
    AGFX_EXPECT_NOT_NULL(view.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(buffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("multi dispatch");
        pass.SetPipeline(pipeline);
        for (uint32_t i = 0; i < kPassCount; ++i) {
            if (i > 0) {
                pass.BufferUAVBarrier(buffer.Raw());
            }

            // Rebuilt per pass: the pass index is part of the constants the shader folds in.
            agfx::ez::ShaderBindings bindings;
            bindings.BindBuffer(view);
            bindings.Write(kElementCount);
            bindings.Write(i);
            bindings.Write(0u); // padding
            pass.PushConstants(bindings.Data(), bindings.Size());

            pass.Dispatch(kGroupCount, 1, 1);
        }
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), buffer.Raw(), kBufferSize,
                                       buffer.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
