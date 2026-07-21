/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "subresource_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace agfxtest
{
    namespace
    {
        /// @brief Mirrors PassActionsPushConstants in data/shaders/tests/pass_actions.hlsl. The
        /// render-to tests reuse that shader rather than introducing another flat-color triangle.
        struct DrawPushConstants
        {
            float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            uint32_t fullscreen = 0;
            float depth = 0.0f; // no depth attachment in these tests
            uint32_t padding0 = 0;
            uint32_t padding1 = 0;
        };
        static_assert(sizeof(DrawPushConstants) == 32, "DrawPushConstants must match the HLSL layout");

        DrawPushConstants MakeDrawConstants()
        {
            DrawPushConstants constants;
            for (uint32_t i = 0; i < 4; ++i) {
                constants.color[i] = kSubresourceDrawRgba8[i] / 255.0f;
            }
            constants.fullscreen = 0; // centered triangle: the seed must remain visible around it
            return constants;
        }

        /// @brief What the targeted subresource must be when the pass only cleared it.
        std::vector<uint8_t> ClearedPixels(uint32_t width, uint32_t height)
        {
            std::vector<uint8_t> pixels((size_t)width * height * kSubresourceBytesPerPixel);
            for (size_t i = 0; i < pixels.size(); i += kSubresourceBytesPerPixel) {
                memcpy(&pixels[i], kSubresourceClearRgba8, sizeof(kSubresourceClearRgba8));
            }
            return pixels;
        }

        agfxTextureCreateInfo TargetInfo(const SubresourceCase& testCase)
        {
            agfxTextureCreateInfo info{};
            info.type = testCase.type;
            info.format = kSubresourceFormat;
            info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = testCase.baseSize;
            info.height = testCase.baseSize;
            info.depthOrArrayLayers = testCase.layerCount;
            info.mipLevels = testCase.mipLevels;
            return info;
        }

        agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
        {
            agfxRenderPipelineCreateInfo info{};
            info.name = "test subresource draw";
            info.fillMode = AGFX_FILL_MODE_SOLID;
            info.cullMode = AGFX_CULL_MODE_NONE; // so a winding difference cannot drop the draw
            info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
            info.topology = AGFX_TOPOLOGY_TRIANGLES;
            info.depthTestEnable = 0;
            info.depthWriteEnable = 0;
            info.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
            info.colorAttachmentCount = 1;
            info.colorFormats[0] = kSubresourceFormat;
            info.vertexShader = vs;
            info.fragmentShader = ps;
            return info;
        }

        agfxRenderPassCreateInfo PassInfo(const SubresourceCase& testCase, agfxRenderTarget* renderTarget)
        {
            const uint32_t size = SubresourceMipSize(testCase, testCase.targetMip);

            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 1;
            info.colorAttachments[0].renderTarget = renderTarget;
            info.colorAttachments[0].loadOp = testCase.loadOp;
            info.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
            memcpy(info.colorAttachments[0].clearColor, SubresourceClearColorFloat(), sizeof(float) * 4);
            info.width = size;
            info.height = size;
            info.name = testCase.name;
            return info;
        }

        /// @brief Seeds every subresource of `texture` with its SubresourceSeedPixels pattern.
        bool SeedAll(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                     const SubresourceCase& testCase)
        {
            bool ok = true;
            for (uint32_t mip = 0; mip < testCase.mipLevels; ++mip) {
                const uint32_t size = SubresourceMipSize(testCase, mip);
                for (uint32_t layer = 0; layer < testCase.layerCount; ++layer) {
                    const std::vector<uint8_t> seed = SubresourceSeedPixels(size, size, mip, layer);
                    ok &= UploadTextureSubresource(device, queue, texture, size, size, kSubresourceFormat,
                                                   seed.data(), AGFX_RESOURCE_STATE_COMMON, mip, layer);
                }
            }
            return ok;
        }

        /// @brief Reads back the targeted subresource and every neighbour, checking each against what
        /// it should be. `targetState` is whatever state the pass left the targeted subresource in;
        /// the neighbours were never transitioned, so they are still COMMON.
        bool VerifyAll(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                       const SubresourceCase& testCase, agfxResourceState targetState, Image& outImage,
                       std::string& outError)
        {
            const uint32_t targetSize = SubresourceMipSize(testCase, testCase.targetMip);
            if (!ReadbackTextureSubresource(device, queue, texture, targetSize, targetSize, kSubresourceFormat,
                                            targetState, testCase.targetMip, testCase.targetLayer, outImage)) {
                outError = "readback of the targeted subresource failed";
                return false;
            }

            // A clear-only pass has an exact expected result; a pass that drew does not (the
            // triangle's edges are the rasterizer's business), so that case leans on its golden plus
            // the caller's own spot checks instead.
            if (!testCase.drawTriangle && testCase.loadOp == AGFX_LOAD_OPERATION_CLEAR &&
                !ImageEqualsRgba8(outImage, ClearedPixels(targetSize, targetSize))) {
                outError = "the targeted subresource is not the clear color";
                return false;
            }

            for (uint32_t mip = 0; mip < testCase.mipLevels; ++mip) {
                const uint32_t size = SubresourceMipSize(testCase, mip);
                for (uint32_t layer = 0; layer < testCase.layerCount; ++layer) {
                    if (mip == testCase.targetMip && layer == testCase.targetLayer) {
                        continue;
                    }
                    Image neighbour;
                    if (!ReadbackTextureSubresource(device, queue, texture, size, size, kSubresourceFormat,
                                                    AGFX_RESOURCE_STATE_COMMON, mip, layer, neighbour)) {
                        outError = "readback of a neighbouring subresource failed";
                        return false;
                    }
                    if (!ImageEqualsRgba8(neighbour, SubresourceSeedPixels(size, size, mip, layer))) {
                        outError = "the pass escaped the subresource it was pointed at (mip " +
                                   std::to_string(mip) + ", layer " + std::to_string(layer) + " changed)";
                        return false;
                    }
                }
            }
            return true;
        }

        bool RunC(const SubresourceCase& testCase, const CompiledShader& vsShader,
                  const CompiledShader& psShader, Image& outImage, std::string& outError)
        {
            GpuFixture gpu;
            if (!gpu.Valid()) {
                outError = "failed to create headless device";
                return false;
            }
            agfxDevice* device = gpu.Device();

            const agfxTextureCreateInfo targetInfo = TargetInfo(testCase);
            agfxTexture* target = agfxTextureCreate(device, &targetInfo);
            if (!target) {
                outError = "texture creation failed";
                return false;
            }

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN; // inherit the texture's format
            rtInfo.mipLevel = testCase.targetMip;
            rtInfo.arrayLayer = testCase.targetLayer;
            agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);

            agfxRenderPipeline* pipeline = nullptr;
            if (testCase.drawTriangle) {
                agfxShaderModule* vsModule =
                    CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
                agfxShaderModule* psModule =
                    CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
                const agfxRenderPipelineCreateInfo pipelineInfo = PipelineInfo(vsModule, psModule);
                pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
                agfxShaderModuleDestroy(device, vsModule);
                agfxShaderModuleDestroy(device, psModule);
            }

            bool ok = renderTarget != nullptr && (!testCase.drawTriangle || pipeline != nullptr);
            if (!ok) {
                outError = "render target or pipeline creation failed";
            }

            if (ok) {
                agfxDeviceMakeResourcesResident(device);

                ok = SeedAll(device, gpu.Queue(), target, testCase);
                if (!ok) {
                    outError = "failed to seed a subresource";
                }
            }

            if (ok) {
                const uint32_t size = SubresourceMipSize(testCase, testCase.targetMip);
                const DrawPushConstants constants = MakeDrawConstants();
                gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                    // Only the subresource under test is transitioned; the neighbours stay COMMON,
                    // which is also what VerifyAll reads them back from.
                    agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                                    AGFX_RESOURCE_STATE_RENDER_TARGET, testCase.targetMip,
                                                    testCase.targetLayer, 0);

                    const agfxRenderPassCreateInfo passInfo = PassInfo(testCase, renderTarget);
                    agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
                    agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)size, (float)size, 0.0f, 1.0f);
                    agfxRenderPassSetScissor(pass, 0, 0, size, size);
                    if (testCase.drawTriangle) {
                        agfxRenderPassSetPipeline(pass, pipeline);
                        agfxRenderPassPushConstants(pass, &constants, sizeof(constants));
                        agfxRenderPassDraw(pass, 3, 1, 0, 0);
                    }
                    agfxRenderPassEnd(pass);
                });

                ok = VerifyAll(device, gpu.Queue(), target, testCase, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               outImage, outError);
            }

            if (pipeline) {
                agfxRenderPipelineDestroy(device, pipeline);
            }
            if (renderTarget) {
                agfxRenderTargetDestroy(device, renderTarget);
            }
            agfxTextureDestroy(device, target);
            return ok;
        }

        bool RunCpp(const SubresourceCase& testCase, const CompiledShader& vsShader,
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

            agfx::Texture target = device.CreateTexture(TargetInfo(testCase));

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = target;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            rtInfo.mipLevel = testCase.targetMip;
            rtInfo.arrayLayer = testCase.targetLayer;
            agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);

            agfx::RenderPipeline pipeline;
            if (testCase.drawTriangle) {
                agfx::ShaderModule vsModule(device.Get(),
                    CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
                agfx::ShaderModule psModule(device.Get(),
                    CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
                pipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
            }

            if (!target.Get() || !renderTarget.Get() || (testCase.drawTriangle && !pipeline.Get())) {
                outError = "texture, render target or pipeline creation failed";
                return false;
            }

            device.MakeResourcesResident();

            if (!SeedAll(device.Get(), queue, target, testCase)) {
                outError = "failed to seed a subresource";
                return false;
            }

            const uint32_t size = SubresourceMipSize(testCase, testCase.targetMip);
            const DrawPushConstants constants = MakeDrawConstants();

            cmd.Begin();
            cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_RENDER_TARGET,
                               testCase.targetMip, testCase.targetLayer, false);
            {
                agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(testCase, renderTarget));
                pass.SetViewport(0.0f, 0.0f, (float)size, (float)size);
                pass.SetScissor(0, 0, size, size);
                if (testCase.drawTriangle) {
                    pass.SetPipeline(pipeline);
                    pass.PushConstants(&constants, sizeof(constants));
                    pass.Draw(3);
                }
            }
            cmd.End();

            queue.Submit(cmd);
            queue.Signal(fence, 1);
            fence.Wait(1, UINT64_MAX);

            return VerifyAll(device.Get(), queue, target, testCase, AGFX_RESOURCE_STATE_RENDER_TARGET,
                             outImage, outError);
        }

        /// @brief Creates the ez texture matching the case's shape. ez types the three shapes
        /// separately, but every call past creation goes through TextureBase, so the rest of the ez
        /// path is shared.
        std::unique_ptr<agfx::ez::detail::TextureBase> CreateEzTarget(agfx::ez::Context& context,
                                                                      const SubresourceCase& testCase)
        {
            const agfxTextureUsage usage =
                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);

            switch (testCase.type) {
            case AGFX_TEXTURE_TYPE_CUBE:
                return std::make_unique<agfx::ez::TextureCube>(
                    context.CreateTextureCube(testCase.baseSize, kSubresourceFormat, usage, testCase.mipLevels));
            case AGFX_TEXTURE_TYPE_2D_ARRAY:
                return std::make_unique<agfx::ez::Texture2DArray>(
                    context.CreateTexture2DArray(testCase.baseSize, testCase.baseSize, testCase.layerCount,
                                                 kSubresourceFormat, usage, testCase.mipLevels));
            default:
                return std::make_unique<agfx::ez::Texture2D>(
                    context.CreateTexture2D(testCase.baseSize, testCase.baseSize, kSubresourceFormat, usage,
                                            nullptr, 0, testCase.mipLevels));
            }
        }

        bool RunEz(const SubresourceCase& testCase, const CompiledShader& vsShader,
                   const CompiledShader& psShader, Image& outImage, std::string& outError)
        {
            agfx::ez::ContextCreateInfo contextInfo{};
            contextInfo.deviceInfo = DefaultDeviceCreateInfo();
            contextInfo.windowHandle = nullptr; // headless: straight into an offscreen target
            contextInfo.width = testCase.baseSize;
            contextInfo.height = testCase.baseSize;
            agfx::ez::Context context(contextInfo);

            agfx::Device& device = context.GetDevice();
            std::unique_ptr<agfx::ez::detail::TextureBase> target = CreateEzTarget(context, testCase);

            agfx::ShaderModule vsModule(device.Get(),
                CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
            agfx::ShaderModule psModule(device.Get(),
                CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
            if (testCase.drawTriangle && (!vsModule.Get() || !psModule.Get())) {
                outError = "shader module creation failed";
                return false;
            }

            for (uint32_t mip = 0; mip < testCase.mipLevels; ++mip) {
                const uint32_t size = SubresourceMipSize(testCase, mip);
                const agfxTextureRegion region{0, 0, 0, size, size, 1};
                for (uint32_t layer = 0; layer < testCase.layerCount; ++layer) {
                    const std::vector<uint8_t> seed = SubresourceSeedPixels(size, size, mip, layer);
                    context.UploadTexture(*target, region, mip, layer, seed.data(), (uint32_t)seed.size(),
                                          size * kSubresourceBytesPerPixel);
                }
            }

            device.MakeResourcesResident();

            // The modules must outlive the draw: ez's PipelineDesc caches on their addresses.
            agfx::ez::PipelineDesc desc;
            desc.name = "test subresource draw";
            desc.vertexShader = &vsModule;
            desc.fragmentShader = &psModule;
            desc.cullMode = AGFX_CULL_MODE_NONE;
            desc.depthTestEnable = false;
            desc.depthWriteEnable = false;

            const DrawPushConstants constants = MakeDrawConstants();
            agfx::ez::ShaderBindings bindings;
            bindings.Write(&constants, sizeof(constants));

            const uint32_t size = SubresourceMipSize(testCase, testCase.targetMip);
            {
                agfx::ez::Frame frame = context.BeginFrame();
                // ez takes the pass dimensions and the transition from the bound subresource, so
                // naming the mip/layer here is the whole of the ez-side setup.
                context.SetRenderTargets({{*target, testCase.targetMip, testCase.targetLayer}}, nullptr,
                                         testCase.loadOp, SubresourceClearColorFloat());
                context.SetViewportScissor(0, 0, size, size);
                if (testCase.drawTriangle) {
                    context.SetPipeline(desc);
                    context.PushShaderBindings(bindings);
                    context.Draw(3);
                }
                context.EndActivePass();
            }
            context.DrainGPU();

            return VerifyAll(device.Get(), context.GetGraphicsQueue(), target->Raw(), testCase,
                             target->StateAt(testCase.targetMip, testCase.targetLayer), outImage, outError);
        }
    } // namespace

    const float* SubresourceClearColorFloat()
    {
        static const float color[4] = {
            kSubresourceClearRgba8[0] / 255.0f, kSubresourceClearRgba8[1] / 255.0f,
            kSubresourceClearRgba8[2] / 255.0f, kSubresourceClearRgba8[3] / 255.0f};
        return color;
    }

    uint32_t SubresourceMipSize(const SubresourceCase& testCase, uint32_t mip)
    {
        const uint32_t size = testCase.baseSize >> mip;
        return size ? size : 1u;
    }

    std::vector<uint8_t> SubresourceSeedPixels(uint32_t width, uint32_t height, uint32_t mip, uint32_t layer)
    {
        std::vector<uint8_t> pixels((size_t)width * height * kSubresourceBytesPerPixel);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint8_t* texel = &pixels[((size_t)y * width + x) * kSubresourceBytesPerPixel];
                texel[0] = (uint8_t)(x * 3u + layer * 41u);
                texel[1] = (uint8_t)(y * 3u + mip * 97u);
                texel[2] = (uint8_t)(layer * 29u + mip * 53u + 7u);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    bool RunSubresourcePass(TestApi api, const SubresourceCase& testCase, Image& outImage, std::string& outError)
    {
        outError.clear();

        // Compiled unconditionally: the clear cases never draw, but keeping one code path here is
        // cheaper than threading "is there a shader" through three API flavors.
        const CompiledShader vsShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
        const CompiledShader psShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
        if (!vsShader.Valid() || !psShader.Valid()) {
            outError = "failed to compile pass_actions.hlsl";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RunC(testCase, vsShader, psShader, outImage, outError);
        case TestApi::Cpp: return RunCpp(testCase, vsShader, psShader, outImage, outError);
        case TestApi::Ez:  return RunEz(testCase, vsShader, psShader, outImage, outError);
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace agfxtest
