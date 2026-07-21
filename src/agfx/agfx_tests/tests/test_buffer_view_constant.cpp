/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "buffer view constant".
//
// Drives data/shaders/tests/constant_buffer.hlsl, whose output pattern is derived entirely from
// parameters read through an AGFX_BUFFER_VIEW_TYPE_CONSTANT view rather than from push constants.
// Every field of the constant struct — including the float4 that sits past the leading uints —
// participates in the written value, so a view bound at the wrong offset or as the wrong descriptor
// type yields zeros or garbage across the whole golden instead of one suspicious word.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kElementCount = 64;
    constexpr uint32_t kWordsPerElement = 2;             // The shader Store2()s per element.
    constexpr uint64_t kOutputSize = kElementCount * kWordsPerElement * sizeof(uint32_t);
    constexpr uint32_t kGroupSize = 64;                  // Matches [numthreads(64,1,1)].
    constexpr uint32_t kGroupCount = (kElementCount + kGroupSize - 1) / kGroupSize;
    constexpr const char* kGolden = "buffer_view_constant.bin";

    /// @brief Mirrors Parameters in constant_buffer.hlsl.
    struct Parameters
    {
        uint32_t elementCount;
        uint32_t seed;
        uint32_t multiplier;
        uint32_t bias;
        float vector[4];
    };
    static_assert(sizeof(Parameters) == 32, "Parameters must match the HLSL layout");

    /// @brief Mirrors ConstantPushConstants in constant_buffer.hlsl.
    struct PushConstants
    {
        uint32_t parameters;
        uint32_t rwBuffer;
        uint32_t elementCount;
        uint32_t padding;
    };

    Parameters MakeParameters()
    {
        Parameters params{};
        params.elementCount = kElementCount;
        params.seed = 0x1234u;
        params.multiplier = 37u;
        params.bias = 11u;
        params.vector[0] = 1.25f;
        params.vector[1] = 2.5f;
        params.vector[2] = 3.75f;
        params.vector[3] = 5.0f;
        return params;
    }

    /// @brief Constant buffers are bound at a 256-byte granularity on both backends; oversize the
    /// allocation so the view is legal regardless of how the backend rounds.
    constexpr uint64_t kConstantBufferSize = 256;

    agfxBufferCreateInfo ConstantBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kConstantBufferSize;
        info.stride = sizeof(Parameters);
        info.usage = AGFX_BUFFER_USAGE_CONSTANT;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
        return info;
    }

    agfxBufferCreateInfo OutputBufferInfo()
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
        info.name = "constant buffer view";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(BufferViewConstant, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("constant_buffer.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile constant_buffer.hlsl:main_cs");

    const agfxBufferCreateInfo constantInfo = ConstantBufferInfo();
    const agfxBufferCreateInfo outputInfo = OutputBufferInfo();
    agfxBuffer* constantBuffer = agfxBufferCreate(device, &constantInfo);
    agfxBuffer* output = agfxBufferCreate(device, &outputInfo);
    AGFX_EXPECT_NOT_NULL(constantBuffer);
    AGFX_EXPECT_NOT_NULL(output);

    agfxDeviceMakeResourcesResident(device);

    const Parameters params = MakeParameters();
    void* mapped = agfxBufferMap(constantBuffer);
    if (!mapped) {
        agfxBufferDestroy(device, output);
        agfxBufferDestroy(device, constantBuffer);
        AGFX_FAIL("failed to map the constant buffer");
    }
    memcpy(mapped, &params, sizeof(params));
    agfxBufferUnmap(constantBuffer);

    agfxBufferViewCreateInfo constantViewInfo{};
    constantViewInfo.buffer = constantBuffer;
    constantViewInfo.type = AGFX_BUFFER_VIEW_TYPE_CONSTANT;
    constantViewInfo.offset = 0;
    constantViewInfo.writeable = 0;
    agfxBufferView* constantView = agfxBufferViewCreate(device, &constantViewInfo);

    agfxBufferViewCreateInfo outputViewInfo{};
    outputViewInfo.buffer = output;
    outputViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    outputViewInfo.offset = 0;
    outputViewInfo.writeable = 1;
    agfxBufferView* outputView = agfxBufferViewCreate(device, &outputViewInfo);
    AGFX_EXPECT_NOT_NULL(constantView);
    AGFX_EXPECT_NOT_NULL(outputView);

    agfxShaderModule* module = CreateShaderModule(device, shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(module);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, module);
    AGFX_EXPECT_NOT_NULL(pipeline);

    PushConstants constants{};
    constants.parameters = (uint32_t)agfxBufferViewGetHandle(constantView);
    constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(outputView);
    constants.elementCount = kElementCount;

    agfxDeviceMakeResourcesResident(device);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, output, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "constant buffer view");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroupCount, 1, 1);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), output, kOutputSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, pipeline);
    agfxBufferViewDestroy(device, outputView);
    agfxBufferViewDestroy(device, constantView);
    agfxBufferDestroy(device, output);
    agfxBufferDestroy(device, constantBuffer);

    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(BufferViewConstant, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("constant_buffer.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile constant_buffer.hlsl:main_cs");

    agfx::Buffer constantBuffer = device.CreateBuffer(ConstantBufferInfo());
    agfx::Buffer output = device.CreateBuffer(OutputBufferInfo());
    AGFX_EXPECT_NOT_NULL(constantBuffer.Get());
    AGFX_EXPECT_NOT_NULL(output.Get());

    device.MakeResourcesResident();

    const Parameters params = MakeParameters();
    {
        agfx::MappedBuffer mapping(constantBuffer);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(mapping.Get(), &params, sizeof(params));
    }

    agfxBufferViewCreateInfo constantViewInfo{};
    constantViewInfo.buffer = constantBuffer;
    constantViewInfo.type = AGFX_BUFFER_VIEW_TYPE_CONSTANT;
    constantViewInfo.offset = 0;
    constantViewInfo.writeable = 0;
    agfx::BufferView constantView = device.CreateBufferView(constantViewInfo);

    agfxBufferViewCreateInfo outputViewInfo{};
    outputViewInfo.buffer = output;
    outputViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    outputViewInfo.offset = 0;
    outputViewInfo.writeable = 1;
    agfx::BufferView outputView = device.CreateBufferView(outputViewInfo);
    AGFX_EXPECT_NOT_NULL(constantView.Get());
    AGFX_EXPECT_NOT_NULL(outputView.Get());

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    PushConstants constants{};
    constants.parameters = (uint32_t)constantView.GetHandle();
    constants.rwBuffer = (uint32_t)outputView.GetHandle();
    constants.elementCount = kElementCount;

    device.MakeResourcesResident();

    cmd.Begin();
    cmd.BufferBarrier(output, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("constant buffer view");
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

AGFX_TEST_BUFFER(BufferViewConstant, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    // D3D12 caps a CBV's SizeInBytes at 65536, and the whole ring buffer (budget * framesInFlight) is
    // what gets sized into each view, so keep the per-frame budget tiny -- this test only ever
    // allocates one small Parameters struct per frame.
    contextInfo.dynamicConstantsBudgetPerFrame = 4096;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("constant_buffer.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile constant_buffer.hlsl:main_cs");

    agfx::Device& device = context.GetDevice();

    const std::vector<uint32_t> zeros(kElementCount * kWordsPerElement, 0u);
    agfx::ez::Buffer output =
        context.CreateStructuredBuffer(zeros.data(), kOutputSize, sizeof(uint32_t), /*shaderWritable*/ true);

    agfxBufferViewCreateInfo outputViewInfo{};
    outputViewInfo.buffer = output.Raw();
    outputViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    outputViewInfo.offset = 0;
    outputViewInfo.writeable = 1;
    agfx::BufferView outputView = device.CreateBufferView(outputViewInfo);
    AGFX_EXPECT_NOT_NULL(outputView.Get());

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

        // The ez way to get a constant buffer view: suballocate out of the per-frame ring. This is
        // the same AGFX_BUFFER_VIEW_TYPE_CONSTANT path the C and C++ flavors build by hand.
        agfx::BufferView& constantView = context.AllocateConstants(MakeParameters());

        agfx::ez::ShaderBindings bindings;
        bindings.BindBuffer(constantView);
        bindings.BindBuffer(outputView);
        bindings.Write(kElementCount);
        bindings.Write(0u); // padding

        context.TransitionBuffer(output, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("constant buffer view");
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
