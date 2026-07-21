/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute buffer atomics".
//
// Drives data/shaders/tests/atomics.hlsl: 128 threads across two groups hammer every Interlocked*
// op AGFXRWByteAddressBuffer exposes, each at a single shared address. The chosen operands make
// each result order-independent (a sum, a saturating and/or, an xor that cancels, a min, a max, and
// a compare-exchange only one thread can win), so the golden is stable even though the execution
// order is not — a dropped atomic shows up as a wrong total rather than as a flaky test.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kThreadCount = 128;               // Two full groups, so cross-group atomics count.
    constexpr uint32_t kGroupSize = 64;                  // Matches [numthreads(64,1,1)].
    constexpr uint32_t kGroupCount = kThreadCount / kGroupSize;
    constexpr uint32_t kSlotCount = 8;                   // Slot layout mirrors atomics.hlsl.
    constexpr uint64_t kBufferSize = kSlotCount * sizeof(uint32_t);
    constexpr const char* kGolden = "compute_buffer_atomics.bin";

    /// @brief Mirrors AtomicsPushConstants in atomics.hlsl.
    struct PushConstants
    {
        uint32_t rwBuffer;
        uint32_t threadCount;
        uint32_t padding0;
        uint32_t padding1;
    };

    /// @brief Each slot starts at its operation's identity, so the final value is purely a product
    /// of the atomics rather than of whatever the allocator left behind.
    std::vector<uint32_t> SeedValues()
    {
        std::vector<uint32_t> seed(kSlotCount, 0u);
        seed[1] = 0xFFFFFFFFu; // AND
        seed[4] = 0xFFFFFFFFu; // MIN
        return seed;
    }

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
        info.name = "buffer atomics";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(ComputeBufferAtomics, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("atomics.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile atomics.hlsl:main_cs");

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

    const std::vector<uint32_t> seed = SeedValues();
    const bool seeded = UploadBuffer(device, gpu.Queue(), buffer, seed.data(), kBufferSize,
                                     AGFX_RESOURCE_STATE_COMMON);

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(view);
    constants.threadCount = kThreadCount;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "buffer atomics");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupCount, 1, 1);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), buffer, kBufferSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, pipeline);
    agfxBufferViewDestroy(device, view);
    agfxBufferDestroy(device, buffer);

    AGFX_EXPECT_MSG(seeded, "failed to seed the atomics buffer");
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(ComputeBufferAtomics, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("atomics.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile atomics.hlsl:main_cs");

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

    const std::vector<uint32_t> seed = SeedValues();
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, buffer, seed.data(), kBufferSize,
                                 AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the atomics buffer");

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)view.GetHandle();
    constants.threadCount = kThreadCount;

    cmd.Begin();
    cmd.BufferBarrier(buffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("buffer atomics");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupCount, 1, 1);
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

AGFX_TEST_BUFFER(ComputeBufferAtomics, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("atomics.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile atomics.hlsl:main_cs");

    agfx::Device& device = context.GetDevice();

    // ez seeds at creation time through its own uploader — no hand-rolled staging buffer needed.
    const std::vector<uint32_t> seed = SeedValues();
    agfx::ez::Buffer buffer =
        context.CreateStructuredBuffer(seed.data(), kBufferSize, sizeof(uint32_t), /*shaderWritable*/ true);

    agfx::BufferView& view = buffer.View(AGFX_BUFFER_VIEW_TYPE_RAW, /*writeable*/ true);
    AGFX_EXPECT_NOT_NULL(view.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    agfx::ez::ShaderBindings bindings;
    bindings.BindBuffer(view);
    bindings.Write(kThreadCount);
    bindings.Write(0u); // padding0
    bindings.Write(0u); // padding1

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(buffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("buffer atomics");
        pass.SetPipeline(pipeline);
        pass.PushConstants(bindings.Data(), bindings.Size());
        pass.Dispatch(kGroupCount, 1, 1);
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), buffer.Raw(), kBufferSize,
                                       buffer.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
