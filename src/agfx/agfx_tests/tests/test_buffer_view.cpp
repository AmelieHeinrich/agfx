/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "buffer view byte address".
//
// Drives data/shaders/tests/byte_address.hlsl in two dispatches: the first writes an offset-derived
// pattern through a writeable raw (byte address) view, the second reads it back through a read-only
// raw view of the same buffer and folds groups of four words into the buffer's second half. Both
// halves are then memcmp'd against the golden, so a wrong view offset, a wrong writeable flag, or a
// broken Load*/Store* path all show up as differing bytes rather than a plausible pattern.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kElementCount = 64;               // Words written by phase 1.
    constexpr uint32_t kTotalWords = kElementCount * 2;  // Phase 2 fills the second half.
    constexpr uint64_t kBufferSize = kTotalWords * sizeof(uint32_t);
    constexpr uint32_t kGroupSize = 64;                  // Matches [numthreads(64,1,1)].
    constexpr uint32_t kGroupCount = (kElementCount + kGroupSize - 1) / kGroupSize;
    constexpr const char* kGolden = "buffer_view_byte_address.bin";

    /// @brief Mirrors ByteAddressPushConstants in byte_address.hlsl.
    struct PushConstants
    {
        uint32_t rwBuffer;
        uint32_t readBuffer;
        uint32_t elementCount;
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

    agfxComputePipelineCreateInfo ComputePipelineInfo(const char* name, agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = name;
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(BufferViewByteAddress, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader writeShader = CompileTestShader("byte_address.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    const CompiledShader readShader = CompileTestShader("byte_address.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_readback_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile byte_address.hlsl:main_cs");
    AGFX_EXPECT_MSG(readShader.Valid(), "failed to compile byte_address.hlsl:main_readback_cs");

    // The buffer under test: GPU-only, so the only way to observe its contents is the copy-to-
    // readback path — no upload-heap shortcut can mask a broken view.
    const agfxBufferCreateInfo bufferInfo = BufferInfo();
    agfxBuffer* buffer = agfxBufferCreate(device, &bufferInfo);
    AGFX_EXPECT_NOT_NULL(buffer);

    agfxBufferViewCreateInfo rwViewInfo{};
    rwViewInfo.buffer = buffer;
    rwViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    rwViewInfo.offset = 0;
    rwViewInfo.writeable = 1;
    agfxBufferView* rwView = agfxBufferViewCreate(device, &rwViewInfo);

    agfxBufferViewCreateInfo readViewInfo = rwViewInfo;
    readViewInfo.writeable = 0;
    agfxBufferView* readView = agfxBufferViewCreate(device, &readViewInfo);
    AGFX_EXPECT_NOT_NULL(rwView);
    AGFX_EXPECT_NOT_NULL(readView);

    agfxShaderModule* writeModule = CreateShaderModule(device, writeShader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxShaderModule* readModule = CreateShaderModule(device, readShader, "main_readback_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);

    const agfxComputePipelineCreateInfo writePipelineInfo = ComputePipelineInfo("byte address write", writeModule);
    const agfxComputePipelineCreateInfo readPipelineInfo = ComputePipelineInfo("byte address readback", readModule);
    agfxComputePipeline* writePipeline = agfxComputePipelineCreate(device, &writePipelineInfo);
    agfxComputePipeline* readPipeline = agfxComputePipelineCreate(device, &readPipelineInfo);

    agfxShaderModuleDestroy(device, writeModule);
    agfxShaderModuleDestroy(device, readModule);
    AGFX_EXPECT_NOT_NULL(writePipeline);
    AGFX_EXPECT_NOT_NULL(readPipeline);

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(rwView);
    constants.readBuffer = (uint32_t)agfxBufferViewGetHandle(readView);
    constants.elementCount = kElementCount;

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "byte address");
        agfxComputePassSetPipeline(pass, writePipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupCount, 1, 1);

        // Phase 2 reads what phase 1 wrote, through a different view of the same buffer.
        agfxComputePassBufferUAVBarrier(pass, buffer);

        agfxComputePassSetPipeline(pass, readPipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupCount, 1, 1);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), buffer, kBufferSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, readPipeline);
    agfxComputePipelineDestroy(device, writePipeline);
    agfxBufferViewDestroy(device, readView);
    agfxBufferViewDestroy(device, rwView);
    agfxBufferDestroy(device, buffer);

    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(BufferViewByteAddress, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader writeShader = CompileTestShader("byte_address.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    const CompiledShader readShader = CompileTestShader("byte_address.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_readback_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile byte_address.hlsl:main_cs");
    AGFX_EXPECT_MSG(readShader.Valid(), "failed to compile byte_address.hlsl:main_readback_cs");

    agfx::Buffer buffer = device.CreateBuffer(BufferInfo());
    AGFX_EXPECT_NOT_NULL(buffer.Get());

    agfxBufferViewCreateInfo rwViewInfo{};
    rwViewInfo.buffer = buffer;
    rwViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    rwViewInfo.offset = 0;
    rwViewInfo.writeable = 1;
    agfx::BufferView rwView = device.CreateBufferView(rwViewInfo);

    agfxBufferViewCreateInfo readViewInfo = rwViewInfo;
    readViewInfo.writeable = 0;
    agfx::BufferView readView = device.CreateBufferView(readViewInfo);
    AGFX_EXPECT_NOT_NULL(rwView.Get());
    AGFX_EXPECT_NOT_NULL(readView.Get());

    agfx::ComputePipeline writePipeline;
    agfx::ComputePipeline readPipeline;
    {
        // Modules only need to outlive the pipelines built from them.
        agfx::ShaderModule writeModule(device.Get(),
            CreateShaderModule(device.Get(), writeShader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule readModule(device.Get(),
            CreateShaderModule(device.Get(), readShader, "main_readback_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));

        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("byte address write", writeModule));
        readPipeline = device.CreateComputePipeline(ComputePipelineInfo("byte address readback", readModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(readPipeline.Get());

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)rwView.GetHandle();
    constants.readBuffer = (uint32_t)readView.GetHandle();
    constants.elementCount = kElementCount;

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.BufferBarrier(buffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("byte address");
        pass.SetPipeline(writePipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupCount, 1, 1);

        pass.BufferUAVBarrier(buffer);

        pass.SetPipeline(readPipeline);
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

AGFX_TEST_BUFFER(BufferViewByteAddress, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    const CompiledShader writeShader = CompileTestShader("byte_address.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    const CompiledShader readShader = CompileTestShader("byte_address.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_readback_cs");
    AGFX_EXPECT_MSG(writeShader.Valid(), "failed to compile byte_address.hlsl:main_cs");
    AGFX_EXPECT_MSG(readShader.Valid(), "failed to compile byte_address.hlsl:main_readback_cs");

    agfx::Device& device = context.GetDevice();

    // Zero-initialized so a shader that fails to write leaves obvious zeros rather than garbage.
    const std::vector<uint32_t> zeros(kTotalWords, 0u);
    agfx::ez::Buffer buffer =
        context.CreateStructuredBuffer(zeros.data(), kBufferSize, sizeof(uint32_t), /*shaderWritable*/ true);

    // ez::Buffer::View() caches exactly one view per type, so it can't hand out both a writeable
    // and a read-only raw view of the same buffer — create them off the device directly.
    agfxBufferViewCreateInfo rwViewInfo{};
    rwViewInfo.buffer = buffer.Raw();
    rwViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    rwViewInfo.offset = 0;
    rwViewInfo.writeable = 1;
    agfx::BufferView rwView = device.CreateBufferView(rwViewInfo);

    agfxBufferViewCreateInfo readViewInfo = rwViewInfo;
    readViewInfo.writeable = 0;
    agfx::BufferView readView = device.CreateBufferView(readViewInfo);
    AGFX_EXPECT_NOT_NULL(rwView.Get());
    AGFX_EXPECT_NOT_NULL(readView.Get());

    agfx::ComputePipeline writePipeline;
    agfx::ComputePipeline readPipeline;
    {
        agfx::ShaderModule writeModule(device.Get(),
            CreateShaderModule(device.Get(), writeShader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule readModule(device.Get(),
            CreateShaderModule(device.Get(), readShader, "main_readback_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));

        writePipeline = device.CreateComputePipeline(ComputePipelineInfo("byte address write", writeModule));
        readPipeline = device.CreateComputePipeline(ComputePipelineInfo("byte address readback", readModule));
    }
    AGFX_EXPECT_NOT_NULL(writePipeline.Get());
    AGFX_EXPECT_NOT_NULL(readPipeline.Get());

    agfx::ez::ShaderBindings bindings;
    bindings.BindBuffer(rwView);
    bindings.BindBuffer(readView);
    bindings.Write(kElementCount);
    bindings.Write(0u); // padding

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(buffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("byte address");
        pass.SetPipeline(writePipeline);
        pass.PushConstants(bindings.Data(), bindings.Size());
        pass.Dispatch(kGroupCount, 1, 1);

        pass.BufferUAVBarrier(buffer.Raw());

        pass.SetPipeline(readPipeline);
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
