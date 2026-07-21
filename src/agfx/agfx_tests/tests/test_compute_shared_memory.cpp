/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute shared memory".
//
// Drives data/shaders/tests/shared_memory.hlsl over four groups: each group loads its slice into
// groupshared storage, reverses it (so every thread reads a word another thread wrote), then runs a
// barriered log-step reduction over the same array. The golden holds both the reversed slices and
// the per-group sums, so a backend that mis-sizes groupshared storage or drops
// GroupMemoryBarrierWithGroupSync fails on the halves independently.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kGroupSize = 64;                  // Matches GROUP_SIZE / [numthreads] in the shader.
    constexpr uint32_t kGroupCount = 4;
    constexpr uint32_t kElementCount = kGroupSize * kGroupCount;
    // The output holds the reversed input, then one sum per group.
    constexpr uint32_t kOutputWords = kElementCount + kGroupCount;
    constexpr uint64_t kInputSize = kElementCount * sizeof(uint32_t);
    constexpr uint64_t kOutputSize = kOutputWords * sizeof(uint32_t);
    constexpr const char* kGolden = "compute_shared_memory.bin";

    /// @brief Mirrors SharedMemoryPushConstants in shared_memory.hlsl.
    struct PushConstants
    {
        uint32_t readBuffer;
        uint32_t rwBuffer;
        uint32_t elementCount;
        uint32_t padding;
    };

    /// @brief Distinct per-index values, so a reversal that silently doesn't happen is visible.
    std::vector<uint32_t> InputPattern()
    {
        std::vector<uint32_t> data(kElementCount);
        for (uint32_t i = 0; i < kElementCount; ++i) {
            data[i] = i * 13u + 5u;
        }
        return data;
    }

    agfxBufferCreateInfo InputInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kInputSize;
        info.stride = sizeof(uint32_t);
        info.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferCreateInfo OutputInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kOutputSize;
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "compute shared memory";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(ComputeSharedMemory, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("shared_memory.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile shared_memory.hlsl:main_cs");

    const agfxBufferCreateInfo inputInfo = InputInfo();
    const agfxBufferCreateInfo outputInfo = OutputInfo();
    agfxBuffer* input = agfxBufferCreate(device, &inputInfo);
    agfxBuffer* output = agfxBufferCreate(device, &outputInfo);
    AGFX_EXPECT_NOT_NULL(input);
    AGFX_EXPECT_NOT_NULL(output);

    agfxBufferViewCreateInfo readViewInfo{};
    readViewInfo.buffer = input;
    readViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    readViewInfo.offset = 0;
    readViewInfo.writeable = 0;
    agfxBufferView* readView = agfxBufferViewCreate(device, &readViewInfo);

    agfxBufferViewCreateInfo writeViewInfo{};
    writeViewInfo.buffer = output;
    writeViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    writeViewInfo.offset = 0;
    writeViewInfo.writeable = 1;
    agfxBufferView* writeView = agfxBufferViewCreate(device, &writeViewInfo);
    AGFX_EXPECT_NOT_NULL(readView);
    AGFX_EXPECT_NOT_NULL(writeView);

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint32_t> pattern = InputPattern();
    const bool seeded = UploadBuffer(device, gpu.Queue(), input, pattern.data(), kInputSize,
                                     AGFX_RESOURCE_STATE_COMMON);

    PushConstants constants{};
    constants.readBuffer = (uint32_t)agfxBufferViewGetHandle(readView);
    constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(writeView);
    constants.elementCount = kElementCount;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, input, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0);
        agfxCommandBufferBufferBarrier(cmd, output, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "compute shared memory");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupCount, 1, 1);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), output, kOutputSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, pipeline);
    agfxBufferViewDestroy(device, writeView);
    agfxBufferViewDestroy(device, readView);
    agfxBufferDestroy(device, output);
    agfxBufferDestroy(device, input);

    AGFX_EXPECT_MSG(seeded, "failed to seed the input buffer");
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(ComputeSharedMemory, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("shared_memory.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile shared_memory.hlsl:main_cs");

    agfx::Buffer input = device.CreateBuffer(InputInfo());
    agfx::Buffer output = device.CreateBuffer(OutputInfo());
    AGFX_EXPECT_NOT_NULL(input.Get());
    AGFX_EXPECT_NOT_NULL(output.Get());

    agfxBufferViewCreateInfo readViewInfo{};
    readViewInfo.buffer = input;
    readViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    readViewInfo.offset = 0;
    readViewInfo.writeable = 0;
    agfx::BufferView readView = device.CreateBufferView(readViewInfo);

    agfxBufferViewCreateInfo writeViewInfo{};
    writeViewInfo.buffer = output;
    writeViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    writeViewInfo.offset = 0;
    writeViewInfo.writeable = 1;
    agfx::BufferView writeView = device.CreateBufferView(writeViewInfo);
    AGFX_EXPECT_NOT_NULL(readView.Get());
    AGFX_EXPECT_NOT_NULL(writeView.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint32_t> pattern = InputPattern();
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, input, pattern.data(), kInputSize,
                                 AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the input buffer");

    PushConstants constants{};
    constants.readBuffer = (uint32_t)readView.GetHandle();
    constants.rwBuffer = (uint32_t)writeView.GetHandle();
    constants.elementCount = kElementCount;

    cmd.Begin();
    cmd.BufferBarrier(input, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
    cmd.BufferBarrier(output, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("compute shared memory");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupCount, 1, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, output, kOutputSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(ComputeSharedMemory, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("shared_memory.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile shared_memory.hlsl:main_cs");

    agfx::Device& device = context.GetDevice();

    const std::vector<uint32_t> pattern = InputPattern();
    agfx::ez::Buffer input = context.CreateStructuredBuffer(pattern.data(), kInputSize, sizeof(uint32_t));

    const std::vector<uint32_t> zeros(kOutputWords, 0u);
    agfx::ez::Buffer output =
        context.CreateStructuredBuffer(zeros.data(), kOutputSize, sizeof(uint32_t), /*shaderWritable*/ true);

    // Separate buffers, so ez's one-view-per-type cache is enough here.
    agfx::BufferView& readView = input.View(AGFX_BUFFER_VIEW_TYPE_RAW, /*writeable*/ false);
    agfx::BufferView& writeView = output.View(AGFX_BUFFER_VIEW_TYPE_RAW, /*writeable*/ true);
    AGFX_EXPECT_NOT_NULL(readView.Get());
    AGFX_EXPECT_NOT_NULL(writeView.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    agfx::ez::ShaderBindings bindings;
    bindings.BindBuffer(readView);
    bindings.BindBuffer(writeView);
    bindings.Write(kElementCount);
    bindings.Write(0u); // padding

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(input, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionBuffer(output, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("compute shared memory");
        pass.SetPipeline(pipeline);
        pass.PushConstants(bindings.Data(), bindings.Size());
        pass.Dispatch(kGroupCount, 1, 1);
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), output.Raw(), kOutputSize,
                                       output.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
