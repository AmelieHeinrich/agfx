/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "UAV barriers".
//
// agfxComputePassBufferUAVBarrier and agfxComputePassTextureUAVBarrier are already load-bearing
// plumbing in several other tests (the multi-dispatch buffer/texture tests, the indirect-bundle
// tests), but always for one resource type at a time -- so a barrier call that silently did nothing
// for buffers, say, could still be masked by the texture path elsewhere passing. This test chains a
// buffer and a texture together: 16 round trips of "read the other resource's last value, write mine
// as value + 1", alternating agfxComputePassBufferUAVBarrier and agfxComputePassTextureUAVBarrier
// between every dispatch. See data/shaders/tests/uav_barriers.hlsl for the exact sequence.
//
// Both final values are known analytically -- the buffer ends at 2*kRoundTrips, the texture one step
// behind at 2*kRoundTrips - 1 -- so the test does not lean on FLIP tolerance to catch a race; either
// value being off by even one means a barrier let a read run ahead of the write it depends on.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 4; // The texture side; small since only its broadcast value matters.
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA32F;
    constexpr uint32_t kRoundTrips = 16;

    /// @brief After kRoundTrips full round trips (texture write, then buffer write), per the
    /// sequence in uav_barriers.hlsl: buffer = 2*kRoundTrips, texture = 2*kRoundTrips - 1.
    constexpr uint32_t kExpectedBuffer = 2 * kRoundTrips;
    constexpr float kExpectedTexture = (float)(2 * kRoundTrips - 1);

    constexpr const char* kGolden = "uav_barriers.hdr";

    /// @brief Mirrors UAVBarrierPushConstants in data/shaders/tests/uav_barriers.hlsl.
    struct PushConstants
    {
        uint32_t rwBuffer = 0;
        uint32_t rwTexture = 0;
        uint32_t width = kSize;
        uint32_t height = kSize;
    };

    agfxBufferCreateInfo BufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = sizeof(uint32_t);
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferViewCreateInfo BufferViewInfo(agfxBuffer* buffer)
    {
        agfxBufferViewCreateInfo info{};
        info.buffer = buffer;
        info.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        info.offset = 0;
        info.writeable = 1;
        return info;
    }

    agfxTextureCreateInfo TextureInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kSize;
        info.height = kSize;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureViewCreateInfo TextureViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.mipLevelCount = 1;
        info.arrayLayerCount = 1;
        info.writeable = 1;
        return info;
    }

    agfxComputePipelineCreateInfo TexturePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "uav barrier: texture from buffer";
        info.computeShader = module;
        info.groupSizeX = 8;
        info.groupSizeY = 8;
        info.groupSizeZ = 1;
        return info;
    }

    agfxComputePipelineCreateInfo BufferPipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "uav barrier: buffer from texture";
        info.computeShader = module;
        info.groupSizeX = 1;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }

    bool RunC(const CompiledShader& texShader, const CompiledShader& bufShader, uint32_t& outBufferResult,
              Image& outImage, std::string& outError)
    {
        GpuFixture gpu;
        if (!gpu.Valid()) {
            outError = "failed to create headless device";
            return false;
        }
        agfxDevice* device = gpu.Device();

        const agfxBufferCreateInfo bufferInfo = BufferInfo();
        agfxBuffer* buffer = agfxBufferCreate(device, &bufferInfo);
        const agfxBufferViewCreateInfo bufferViewInfo = BufferViewInfo(buffer);
        agfxBufferView* bufferView = buffer ? agfxBufferViewCreate(device, &bufferViewInfo) : nullptr;

        const agfxTextureCreateInfo textureInfo = TextureInfo();
        agfxTexture* texture = agfxTextureCreate(device, &textureInfo);
        const agfxTextureViewCreateInfo textureViewInfo = TextureViewInfo(texture);
        agfxTextureView* textureView = texture ? agfxTextureViewCreate(device, &textureViewInfo) : nullptr;

        agfxShaderModule* texModule =
            CreateShaderModule(device, texShader, "main_texture_from_buffer_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
        agfxShaderModule* bufModule =
            CreateShaderModule(device, bufShader, "main_buffer_from_texture_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
        const agfxComputePipelineCreateInfo texPipelineInfo = TexturePipelineInfo(texModule);
        const agfxComputePipelineCreateInfo bufPipelineInfo = BufferPipelineInfo(bufModule);
        agfxComputePipeline* texPipeline = agfxComputePipelineCreate(device, &texPipelineInfo);
        agfxComputePipeline* bufPipeline = agfxComputePipelineCreate(device, &bufPipelineInfo);
        agfxShaderModuleDestroy(device, texModule);
        agfxShaderModuleDestroy(device, bufModule);

        bool ok = buffer && bufferView && texture && textureView && texPipeline && bufPipeline;
        if (!ok) {
            outError = "resource or pipeline creation failed";
        }

        if (ok) {
            agfxDeviceMakeResourcesResident(device);

            const uint32_t zero = 0;
            ok = UploadBuffer(device, gpu.Queue(), buffer, &zero, sizeof(zero), AGFX_RESOURCE_STATE_COMMON);
            if (!ok) {
                outError = "failed to seed the buffer";
            }
        }

        if (ok) {
            PushConstants constants{};
            constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(bufferView);
            constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(textureView);

            gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COMMON,
                                               AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);
                agfxCommandBufferTextureBarrier(cmd, texture, AGFX_RESOURCE_STATE_COMMON,
                                                AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

                agfxComputePass* pass = agfxComputePassBegin(cmd, "uav barrier chain");
                for (uint32_t i = 0; i < kRoundTrips; ++i) {
                    agfxComputePassSetPipeline(pass, texPipeline);
                    agfxComputePassPushConstants(pass, &constants, sizeof(constants));
                    agfxComputePassDispatch(pass, 1, 1, 1);
                    // The next dispatch reads the texture this one just wrote.
                    agfxComputePassTextureUAVBarrier(pass, texture);

                    agfxComputePassSetPipeline(pass, bufPipeline);
                    agfxComputePassPushConstants(pass, &constants, sizeof(constants));
                    agfxComputePassDispatch(pass, 1, 1, 1);
                    // The next iteration's texture dispatch reads the buffer this one just wrote.
                    agfxComputePassBufferUAVBarrier(pass, buffer);
                }
                agfxComputePassEnd(pass);
            });

            std::vector<uint8_t> bufferBytes;
            ok = ReadbackBuffer(device, gpu.Queue(), buffer, sizeof(uint32_t),
                                AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bufferBytes);
            if (ok) {
                memcpy(&outBufferResult, bufferBytes.data(), sizeof(outBufferResult));
                ok = ReadbackTexture2D(device, gpu.Queue(), texture, kSize, kSize, kFormat,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);
            }
            if (!ok) {
                outError = "buffer or texture readback failed";
            }
        }

        if (bufPipeline) agfxComputePipelineDestroy(device, bufPipeline);
        if (texPipeline) agfxComputePipelineDestroy(device, texPipeline);
        if (textureView) agfxTextureViewDestroy(device, textureView);
        if (texture) agfxTextureDestroy(device, texture);
        if (bufferView) agfxBufferViewDestroy(device, bufferView);
        if (buffer) agfxBufferDestroy(device, buffer);
        return ok;
    }

    bool RunCpp(const CompiledShader& texShader, const CompiledShader& bufShader, uint32_t& outBufferResult,
                Image& outImage, std::string& outError)
    {
        agfx::Device device(DefaultDeviceCreateInfo());
        if (!device.Get()) {
            outError = "failed to create headless device";
            return false;
        }

        agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
        agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
        agfx::Fence fence = device.CreateFence();

        agfx::Buffer buffer = device.CreateBuffer(BufferInfo());
        agfx::BufferView bufferView = device.CreateBufferView(BufferViewInfo(buffer));
        agfx::Texture texture = device.CreateTexture(TextureInfo());
        agfx::TextureView textureView = device.CreateTextureView(TextureViewInfo(texture));

        agfx::ComputePipeline texPipeline;
        agfx::ComputePipeline bufPipeline;
        {
            agfx::ShaderModule texModule(device.Get(),
                CreateShaderModule(device.Get(), texShader, "main_texture_from_buffer_cs",
                                   AGFX_SHADER_MODULE_TYPE_COMPUTE));
            agfx::ShaderModule bufModule(device.Get(),
                CreateShaderModule(device.Get(), bufShader, "main_buffer_from_texture_cs",
                                   AGFX_SHADER_MODULE_TYPE_COMPUTE));
            texPipeline = device.CreateComputePipeline(TexturePipelineInfo(texModule));
            bufPipeline = device.CreateComputePipeline(BufferPipelineInfo(bufModule));
        }

        if (!buffer.Get() || !bufferView.Get() || !texture.Get() || !textureView.Get() ||
            !texPipeline.Get() || !bufPipeline.Get()) {
            outError = "resource or pipeline creation failed";
            return false;
        }

        device.MakeResourcesResident();

        const uint32_t zero = 0;
        if (!UploadBuffer(device.Get(), queue, buffer, &zero, sizeof(zero), AGFX_RESOURCE_STATE_COMMON)) {
            outError = "failed to seed the buffer";
            return false;
        }

        PushConstants constants{};
        constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(bufferView);
        constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(textureView);

        cmd.Begin();
        cmd.BufferBarrier(buffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, true);
        cmd.TextureBarrier(texture, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                           AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
        {
            agfx::ComputePass pass = cmd.BeginComputePass("uav barrier chain");
            for (uint32_t i = 0; i < kRoundTrips; ++i) {
                pass.SetPipeline(texPipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Dispatch(1, 1, 1);
                pass.TextureUAVBarrier(texture);

                pass.SetPipeline(bufPipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Dispatch(1, 1, 1);
                pass.BufferUAVBarrier(buffer);
            }
        }
        cmd.End();

        queue.Submit(cmd);
        queue.Signal(fence, 1);
        fence.Wait(1, UINT64_MAX);

        std::vector<uint8_t> bufferBytes;
        if (!ReadbackBuffer(device.Get(), queue, buffer, sizeof(uint32_t),
                            AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bufferBytes)) {
            outError = "buffer readback failed";
            return false;
        }
        memcpy(&outBufferResult, bufferBytes.data(), sizeof(outBufferResult));

        if (!ReadbackTexture2D(device.Get(), queue, texture, kSize, kSize, kFormat,
                               AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage)) {
            outError = "texture readback failed";
            return false;
        }
        return true;
    }

    bool RunEz(const CompiledShader& texShader, const CompiledShader& bufShader, uint32_t& outBufferResult,
               Image& outImage, std::string& outError)
    {
        agfx::ez::ContextCreateInfo contextInfo{};
        contextInfo.deviceInfo = DefaultDeviceCreateInfo();
        contextInfo.windowHandle = nullptr; // headless: no swap chain
        contextInfo.width = kSize;
        contextInfo.height = kSize;
        agfx::ez::Context context(contextInfo);

        agfx::Device& device = context.GetDevice();

        // ez has no structured-buffer sugar; the buffer and its view come straight off the device,
        // same as the sampler in the sampling tests.
        agfx::Buffer buffer = device.CreateBuffer(BufferInfo());
        agfx::BufferView bufferView = device.CreateBufferView(BufferViewInfo(buffer));
        agfx::ez::Texture2D texture = context.CreateTexture2D(
            kSize, kSize, kFormat, (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

        agfx::ComputePipeline texPipeline;
        agfx::ComputePipeline bufPipeline;
        {
            agfx::ShaderModule texModule(device.Get(),
                CreateShaderModule(device.Get(), texShader, "main_texture_from_buffer_cs",
                                   AGFX_SHADER_MODULE_TYPE_COMPUTE));
            agfx::ShaderModule bufModule(device.Get(),
                CreateShaderModule(device.Get(), bufShader, "main_buffer_from_texture_cs",
                                   AGFX_SHADER_MODULE_TYPE_COMPUTE));
            texPipeline = device.CreateComputePipeline(TexturePipelineInfo(texModule));
            bufPipeline = device.CreateComputePipeline(BufferPipelineInfo(bufModule));
        }
        if (!buffer.Get() || !bufferView.Get() || !texPipeline.Get() || !bufPipeline.Get()) {
            outError = "resource or pipeline creation failed";
            return false;
        }

        device.MakeResourcesResident();

        const uint32_t zero = 0;
        if (!UploadBuffer(device.Get(), context.GetGraphicsQueue(), buffer, &zero, sizeof(zero),
                          AGFX_RESOURCE_STATE_COMMON)) {
            outError = "failed to seed the buffer";
            return false;
        }

        PushConstants constants{};
        constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(bufferView);
        constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(texture.UAV());

        {
            agfx::ez::Frame frame = context.BeginFrame();
            context.TransitionTexture(texture, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

            // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("uav barrier chain");
            for (uint32_t i = 0; i < kRoundTrips; ++i) {
                pass.SetPipeline(texPipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Dispatch(1, 1, 1);
                pass.TextureUAVBarrier(texture.Raw());

                pass.SetPipeline(bufPipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Dispatch(1, 1, 1);
                pass.BufferUAVBarrier(buffer);
            }
        }
        context.DrainGPU();

        std::vector<uint8_t> bufferBytes;
        if (!ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), buffer, sizeof(uint32_t),
                            AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bufferBytes)) {
            outError = "buffer readback failed";
            return false;
        }
        memcpy(&outBufferResult, bufferBytes.data(), sizeof(outBufferResult));

        if (!ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), texture.Raw(), kSize, kSize,
                               kFormat, texture.State(), outImage)) {
            outError = "texture readback failed";
            return false;
        }
        return true;
    }

    bool Run(TestApi api, uint32_t& outBufferResult, Image& outImage, std::string& outError)
    {
        outError.clear();

        const CompiledShader texShader =
            CompileTestShader("uav_barriers.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_texture_from_buffer_cs");
        const CompiledShader bufShader =
            CompileTestShader("uav_barriers.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_buffer_from_texture_cs");
        if (!texShader.Valid() || !bufShader.Valid()) {
            outError = "failed to compile uav_barriers.hlsl";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RunC(texShader, bufShader, outBufferResult, outImage, outError);
        case TestApi::Cpp: return RunCpp(texShader, bufShader, outBufferResult, outImage, outError);
        case TestApi::Ez:  return RunEz(texShader, bufShader, outBufferResult, outImage, outError);
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace

#define AGFX_UAV_BARRIER_CASE(api)                                                                 \
    AGFX_TEST_TEXTURE(UAVBarriers, api, kSize, kSize)                                              \
    {                                                                                              \
        uint32_t bufferResult = 0;                                                                 \
        Image image;                                                                               \
        std::string error;                                                                         \
        AGFX_EXPECT_MSG(Run(TestApi::api, bufferResult, image, error), error.c_str());              \
        AGFX_EXPECT_MSG(bufferResult == kExpectedBuffer,                                            \
                        "the buffer's final value shows a hazard: a read ran ahead of a write");    \
        AGFX_EXPECT_MSG(image.Valid() && image.pixels[0] == kExpectedTexture,                       \
                        "the texture's final value shows a hazard: a read ran ahead of a write");   \
        ExpectImageMatchesGolden(ctx, kGolden, image);                                              \
    }

AGFX_UAV_BARRIER_CASE(C)
AGFX_UAV_BARRIER_CASE(Cpp)
AGFX_UAV_BARRIER_CASE(Ez)
