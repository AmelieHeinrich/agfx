/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "pass_actions_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        /// @brief Mirrors PassActionsPushConstants in data/shaders/tests/pass_actions.hlsl. Field
        /// order and padding must match exactly -- this is memcpy'd into the push-constant block.
        struct PassActionsConstants
        {
            float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            uint32_t fullscreen = 0;
            float depth = 0.0f; // no depth attachment in these tests
            uint32_t padding0 = 0;
            uint32_t padding1 = 0;
        };
        static_assert(sizeof(PassActionsConstants) == 32, "PassActionsConstants must match the HLSL layout");

        const float* ClearColorFloat()
        {
            static const float color[4] = {
                kPassActionClearRgba8[0] / 255.0f, kPassActionClearRgba8[1] / 255.0f,
                kPassActionClearRgba8[2] / 255.0f, kPassActionClearRgba8[3] / 255.0f};
            return color;
        }

        PassActionsConstants Constants(const PassActionState& state)
        {
            PassActionsConstants constants;
            for (uint32_t i = 0; i < 4; ++i) {
                constants.color[i] = kPassActionDrawRgba8[i] / 255.0f;
            }
            constants.fullscreen = state.fullscreen ? 1u : 0u;
            return constants;
        }

        agfxTextureCreateInfo TargetInfo()
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = kPassActionFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = kPassActionWidth;
            info.height = kPassActionHeight;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;
            return info;
        }

        agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test pass actions";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            info.cullMode = AGFX_CULL_MODE_NONE; // so a winding difference cannot drop the draw
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kPassActionFormat;
            info.vertexShader = vs;
            info.fragmentShader = ps;
            return info;
        }

        agfxRenderPassCreateInfo PassInfo(const PassActionState& state, agfxRenderTarget* renderTarget)
        {
            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 1;
            info.colorAttachments[0].renderTarget = renderTarget;
            info.colorAttachments[0].loadOp = state.loadOp;
            info.colorAttachments[0].storeOp = state.storeOp;
            memcpy(info.colorAttachments[0].clearColor, ClearColorFloat(), sizeof(float) * 4);
            info.width = kPassActionWidth;
            info.height = kPassActionHeight;
            info.name = "test pass actions";
            return info;
        }

        bool RenderC(const PassActionState& state, const CompiledShader& vsShader,
                     const CompiledShader& psShader, Image& outImage, std::string& outError)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                outError = "failed to create headless device";
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo targetInfo = TargetInfo();
            agfxTexture* target = agfxTextureCreate(device, &targetInfo);

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfxRenderTarget* renderTarget = target ? agfxRenderTargetCreate(device, &rtInfo) : nullptr;

            agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
            agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
            const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(vsModule, psModule);
            agfxRenderPipeline* pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
            agfxShaderModuleDestroy(device, vsModule);
            agfxShaderModuleDestroy(device, psModule);

            bool ok = target && renderTarget && pipeline;
            if (!ok) {
                outError = "texture, render target or pipeline creation failed";
            }

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                const std::vector<uint8_t> seed = PassActionSeed();
                ok = UploadTexture2D(device, gpu.Queue(), target, kPassActionWidth, kPassActionHeight,
                                     kPassActionFormat, seed.data(), AGFX_RESOURCE_STATE_COMMON);
                if (!ok) {
                    outError = "failed to seed the attachment";
                }
            }

            if (ok) {
                const PassActionsConstants constants = Constants(state);
                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(state, renderTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kPassActionWidth,
                                              (float)kPassActionHeight, 0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, kPassActionWidth, kPassActionHeight);
                    agfxRenderPassSetPipeline(pass, pipeline);
                    agfxRenderPassPushConstants(pass, &constants, sizeof(constants));
                    agfxRenderPassDraw(pass, 3, 1, 0, 0);
                    agfxRenderPassEnd(pass);
                });

                ok = ReadbackTexture2D(device, gpu.Queue(), target, kPassActionWidth, kPassActionHeight,
                                       kPassActionFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage);
                if (!ok) {
                    outError = "texture readback failed";
                }
            }

            if (pipeline) {
                agfxRenderPipelineDestroy(device, pipeline);
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            if (target) {
                agfxTextureDestroy(device, target);
            }
            return ok;
        }

        bool RenderCpp(const PassActionState& state, const CompiledShader& vsShader,
                       const CompiledShader& psShader, Image& outImage, std::string& outError)
        {
            agfx::Device device(DefaultDeviceCreateInfo());
            if (!device.Get()) {
                outError = "failed to create headless device";
                return false;
            }

            agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
            agfx::Fence fence = device.CreateFence();

            agfx::Texture target = device.CreateTexture(TargetInfo());

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);

            agfx::RenderPipeline pipeline;
            {
                agfx::ShaderModule vsModule(device.Get(),
                    CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                pipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
            }
            if (!target.Get() || !renderTarget.Get() || !pipeline.Get()) {
                outError = "texture, render target or pipeline creation failed";
                return false;
            }

            device.MakeResourcesResident();

            const std::vector<uint8_t> seed = PassActionSeed();
            if (!UploadTexture2D(device.Get(), queue, target, kPassActionWidth, kPassActionHeight,
                                 kPassActionFormat, seed.data(), AGFX_RESOURCE_STATE_COMMON)) {
                outError = "failed to seed the attachment";
                return false;
            }

            const PassActionsConstants constants = Constants(state);

            cmd.Begin();
            cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(state, renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)kPassActionWidth, (float)kPassActionHeight);
                pass.SetScissor(0, 0, kPassActionWidth, kPassActionHeight);
                pass.SetPipeline(pipeline);
                pass.PushConstants(&constants, sizeof(constants));
                pass.Draw(3);
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            if (!ReadbackTexture2D(device.Get(), queue, target, kPassActionWidth, kPassActionHeight,
                                   kPassActionFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, outImage)) {
                outError = "texture readback failed";
                return false;
            }
            return true;
        }

        bool RenderEz(const PassActionState& state, const CompiledShader& vsShader,
                      const CompiledShader& psShader, Image& outImage, std::string& outError)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = kPassActionWidth;
            contextInfo.height = kPassActionHeight;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();
            agfx::ShaderModule vsModule(device.Get(),
                CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
            agfx::ShaderModule psModule(device.Get(),
                CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
            if (!vsModule.Get() || !psModule.Get()) {
                outError = "shader module creation failed";
                return false;
            }

            agfx::ez::Texture2D target = context.CreateTexture2D(
                kPassActionWidth, kPassActionHeight, kPassActionFormat,
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED));

            const std::vector<uint8_t> seed = PassActionSeed();
            const agfxTextureRegion full{0, 0, 0, kPassActionWidth, kPassActionHeight, 1};
            context.UploadTexture(target, full, 0, 0, seed.data(), (uint32_t)seed.size(),
                                  kPassActionWidth * 4);

            device.MakeResourcesResident();

            // The modules must outlive the draw: ez's PipelineDesc caches on their addresses.
            agfx::ez::PipelineDesc desc;
            desc.name = "test pass actions";
            desc.vertexShader = &vsModule;
            desc.fragmentShader = &psModule;
            desc.cullMode = AGFX_CULL_MODE_NONE;
            desc.depthTestEnable = false;
            desc.depthWriteEnable = false;

            const PassActionsConstants constants = Constants(state);
            agfx::ez::ShaderBindings bindings;
            bindings.Write(&constants, sizeof(constants));

            {
                agfx::ez::Frame frame = context.BeginFrame();
                context.SetRenderTargets({&target}, nullptr, state.loadOp, ClearColorFloat());
                context.SetViewportScissor(0, 0, kPassActionWidth, kPassActionHeight);
                context.SetPipeline(desc);
                context.PushShaderBindings(bindings);
                context.Draw(3);
                context.EndActivePass();
            }
            context.DrainGPU();

            if (!ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                   kPassActionWidth, kPassActionHeight, kPassActionFormat,
                                   target.State(), outImage)) {
                outError = "texture readback failed";
                return false;
            }
            return true;
        }
    } // namespace

    std::vector<uint8_t> PassActionSeed()
    {
        std::vector<uint8_t> pixels((size_t)kPassActionWidth * kPassActionHeight * 4);
        for (uint32_t y = 0; y < kPassActionHeight; ++y) {
            for (uint32_t x = 0; x < kPassActionWidth; ++x) {
                uint8_t* texel = &pixels[((size_t)y * kPassActionWidth + x) * 4];
                texel[0] = (uint8_t)x;
                texel[1] = (uint8_t)y;
                texel[2] = (uint8_t)((x ^ y) & 0xFFu);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    bool RegionEquals(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                      const uint8_t rgba[4])
    {
        if (!image.Valid() || x + w > image.width || y + h > image.height) {
            return false;
        }
        for (uint32_t py = y; py < y + h; ++py) {
            for (uint32_t px = x; px < x + w; ++px) {
                const float* texel = &image.pixels[((size_t)py * image.width + px) * 4];
                for (uint32_t c = 0; c < 4; ++c) {
                    // Byte b reads back as exactly b/255.0f, so this stays an exact comparison.
                    if (texel[c] != rgba[c] / 255.0f) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool RegionMatchesSeed(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
    {
        if (!image.Valid() || x + w > image.width || y + h > image.height) {
            return false;
        }
        const std::vector<uint8_t> seed = PassActionSeed();
        for (uint32_t py = y; py < y + h; ++py) {
            for (uint32_t px = x; px < x + w; ++px) {
                const float* texel = &image.pixels[((size_t)py * image.width + px) * 4];
                const uint8_t* expected = &seed[((size_t)py * kPassActionWidth + px) * 4];
                for (uint32_t c = 0; c < 4; ++c) {
                    if (texel[c] != expected[c] / 255.0f) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool RenderPassAction(TestApi api, const PassActionState& state, Image& outImage, std::string& outError)
    {
        outError.clear();

        const CompiledShader vsShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader psShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        if (!vsShader.Valid() || !psShader.Valid()) {
            outError = "failed to compile pass_actions.hlsl";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RenderC(state, vsShader, psShader, outImage, outError);
        case TestApi::Cpp: return RenderCpp(state, vsShader, psShader, outImage, outError);
        case TestApi::Ez:  return RenderEz(state, vsShader, psShader, outImage, outError);
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace agfxtest
