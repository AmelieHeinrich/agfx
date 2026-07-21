/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sample depth texture (d32->r32)".
//
// A DEPTH32F texture cannot be sampled as itself: D32 is not a readable format. Reading one back in
// a shader means creating an agfxTextureView over it with format R32F, which is what the demo's
// cascaded shadow maps do (DeferredRenderer::CreateShadowTargets). This test pins that
// reinterpretation down on its own, separately from the comparison-sampling tests that also rely on
// it — those would fail identically if the *view* were broken, and this one distinguishes the two.
//
// The depth buffer is given structure rather than a flat clear: cleared to kFarDepth, then a
// centered triangle is drawn at kNearDepth with depth writes on and the test function ALWAYS. So the
// sampled result must be kNearDepth inside the triangle and kFarDepth around it, and a backend that
// returned zeros, or the wrong channel, or a comparison result instead of the value, fails on the
// corner/center checks rather than only on the golden.
//
// The destination is RGBA32F and the golden is .hdr, because the two depths are exact in float32 but
// land between 8-bit codes; going through an RGBA8 destination would force a tolerance where an
// exact check is available.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr uint32_t kHeight = 64;
    constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroupsX = (kWidth + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kGroupsY = (kHeight + kGroupSize - 1) / kGroupSize;

    constexpr agfxTextureFormat kDepthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;
    constexpr agfxTextureFormat kDepthViewFormat = AGFX_TEXTURE_FORMAT_R32F;
    constexpr agfxTextureFormat kDestFormat = AGFX_TEXTURE_FORMAT_RGBA32F;

    /// @brief Both exact in float32, and far enough apart that a swapped pair is unmistakable.
    constexpr float kFarDepth = 0.75f;
    constexpr float kNearDepth = 0.25f;

    /// @brief A corner region the centered triangle never covers, and a center region it always does.
    constexpr uint32_t kCornerSize = 8;
    constexpr uint32_t kCenterSize = 8;

    constexpr const char* kGolden = "sample_depth_texture.hdr";

    /// @brief Mirrors SamplingComparisonPushConstants in sampling_comparison.hlsl.
    struct SamplePushConstants
    {
        uint32_t source = 0;
        uint32_t samplerId = 0;
        uint32_t destination = 0;
        uint32_t width = kWidth;
        uint32_t height = kHeight;
        uint32_t bandCount = 1;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
        float references[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // unused by main_sample_depth_cs
    };
    static_assert(sizeof(SamplePushConstants) == 48, "SamplePushConstants must match the HLSL layout");

    /// @brief Mirrors PassActionsPushConstants in pass_actions.hlsl.
    struct DrawPushConstants
    {
        float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        uint32_t fullscreen = 0;
        float depth = kNearDepth;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
    };
    static_assert(sizeof(DrawPushConstants) == 32, "DrawPushConstants must match the HLSL layout");

    agfxTextureCreateInfo DepthInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kDepthFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureCreateInfo DestInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kDestFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureViewCreateInfo DepthViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kDepthViewFormat; // the D32 -> R32F reinterpretation under test
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.mipLevelCount = 1;
        info.arrayLayerCount = 1;
        info.writeable = 0;
        return info;
    }

    agfxTextureViewCreateInfo DestViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kDestFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.mipLevelCount = 1;
        info.arrayLayerCount = 1;
        info.writeable = 1;
        return info;
    }

    agfxSamplerCreateInfo SamplerInfo()
    {
        agfxSamplerCreateInfo info{};
        // NEAREST: the depth buffer's two plateaus must come back as themselves, not blended across
        // the triangle's edge.
        info.filter = AGFX_SAMPLER_FILTER_NEAREST;
        info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxAnisotropy = 1.0f;
        // ALWAYS is AGFX's sentinel for "not a comparison sampler"; see test_sampler_comparison.cpp.
        info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
        return info;
    }

    /// @brief A depth-only pipeline: no color attachments, depth writes on, test ALWAYS so the
    /// triangle's depth replaces the cleared value unconditionally.
    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "depth sample seed";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_TRIANGLES;
        info.depthTestEnable = 1;
        info.depthWriteEnable = 1;
        info.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS;
        info.depthFormat = kDepthFormat;
        info.colorAttachmentCount = 0;
        info.vertexShader = vs;
        info.fragmentShader = ps;
        return info;
    }

    agfxRenderPassCreateInfo DepthPassInfo(agfxRenderTarget* depthTarget)
    {
        agfxRenderPassCreateInfo info{};
        info.colorAttachmentCount = 0;
        info.hasDepthAttachment = 1;
        info.depthAttachment.renderTarget = depthTarget;
        info.depthAttachment.loadOp = AGFX_LOAD_OPERATION_CLEAR;
        info.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
        info.depthAttachment.clearDepth = kFarDepth;
        info.width = kWidth;
        info.height = kHeight;
        info.name = "depth sample seed";
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "sample depth";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }

    /// @brief True if every pixel of the region reads back as `expected` exactly.
    bool RegionIs(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h, float expected)
    {
        if (!image.Valid() || x + w > image.width || y + h > image.height) {
            return false;
        }
        for (uint32_t py = y; py < y + h; ++py) {
            for (uint32_t px = x; px < x + w; ++px) {
                if (image.pixels[((size_t)py * image.width + px) * 4] != expected) {
                    return false;
                }
            }
        }
        return true;
    }

    /// @brief The two spot checks both flavors share: far outside the triangle, near at its center.
    void ExpectDepthPattern(TestContext& ctx, const Image& image)
    {
        AGFX_EXPECT_MSG(RegionIs(image, 0, 0, kCornerSize, kCornerSize, kFarDepth),
                        "the sampled depth outside the triangle is not the cleared far depth");
        AGFX_EXPECT_MSG(RegionIs(image, kWidth / 2 - kCenterSize / 2, kHeight / 2 - kCenterSize / 2,
                                 kCenterSize, kCenterSize, kNearDepth),
                        "the sampled depth inside the triangle is not the drawn near depth");
    }
} // namespace

AGFX_TEST_TEXTURE(SampleDepthTexture, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader vsShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    const CompiledShader csShader =
        CompileTestShader("sampling_comparison.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_depth_cs");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile pass_actions.hlsl");
    AGFX_EXPECT_MSG(csShader.Valid(), "failed to compile sampling_comparison.hlsl:main_sample_depth_cs");

    const agfxTextureCreateInfo depthInfo = DepthInfo();
    const agfxTextureCreateInfo destInfo = DestInfo();
    agfxTexture* depth = agfxTextureCreate(device, &depthInfo);
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(depth);
    AGFX_EXPECT_NOT_NULL(dest);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = depth;
    rtInfo.format = kDepthFormat;
    rtInfo.isDepth = 1;
    agfxRenderTarget* depthTarget = agfxRenderTargetCreate(device, &rtInfo);

    const agfxTextureViewCreateInfo depthViewInfo = DepthViewInfo(depth);
    const agfxTextureViewCreateInfo destViewInfo = DestViewInfo(dest);
    agfxTextureView* depthSrv = agfxTextureViewCreate(device, &depthViewInfo);
    agfxTextureView* destUav = agfxTextureViewCreate(device, &destViewInfo);
    AGFX_EXPECT_NOT_NULL(depthSrv);
    AGFX_EXPECT_NOT_NULL(destUav);

    const agfxSamplerCreateInfo samplerInfo = SamplerInfo();
    agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);

    agfxShaderModule* vsModule = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfxShaderModule* psModule = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    const agfxRenderPipelineCreateInfo drawPipelineInfo = PipelineInfo(vsModule, psModule);
    agfxRenderPipeline* drawPipeline = agfxRenderPipelineCreate(device, &drawPipelineInfo);
    agfxShaderModuleDestroy(device, vsModule);
    agfxShaderModuleDestroy(device, psModule);

    agfxShaderModule* csModule =
        CreateShaderModule(device, csShader, "main_sample_depth_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo samplePipelineInfo = ComputePipelineInfo(csModule);
    agfxComputePipeline* samplePipeline = agfxComputePipelineCreate(device, &samplePipelineInfo);
    agfxShaderModuleDestroy(device, csModule);

    AGFX_EXPECT_NOT_NULL(drawPipeline);
    AGFX_EXPECT_NOT_NULL(samplePipeline);

    agfxDeviceMakeResourcesResident(device);

    const DrawPushConstants drawConstants{};
    SamplePushConstants sampleConstants{};
    sampleConstants.source = (uint32_t)agfxTextureViewGetHandle(depthSrv);
    sampleConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    sampleConstants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, depth, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_DEPTH_WRITE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);

        const agfxRenderPassCreateInfo passInfo = DepthPassInfo(depthTarget);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kWidth, kHeight);
        agfxRenderPassSetPipeline(pass, drawPipeline);
        agfxRenderPassPushConstants(pass, &drawConstants, sizeof(drawConstants));
        agfxRenderPassDraw(pass, 3, 1, 0, 0);
        agfxRenderPassEnd(pass);
    });

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        // agglomerate must be true: on the Metal backend a barrier recorded with false is a no-op,
        // which would leave the depth pass and the sampling dispatch unsynchronized.
        agfxCommandBufferTextureBarrier(cmd, depth, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "sample depth");
        agfxComputePassSetPipeline(pass, samplePipeline);
        agfxComputePassPushConstants(pass, &sampleConstants, sizeof(sampleConstants));
        agfxComputePassDispatch(pass, kGroupsX, kGroupsY, 1);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kWidth, kHeight, kDestFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, samplePipeline);
    agfxRenderPipelineDestroy(device, drawPipeline);
    agfxSamplerDestroy(device, sampler);
    agfxTextureViewDestroy(device, destUav);
    agfxTextureViewDestroy(device, depthSrv);
    agfxRenderTargetDestroy(device, depthTarget);
    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, depth);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectDepthPattern(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(SampleDepthTexture, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader vsShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    const CompiledShader csShader =
        CompileTestShader("sampling_comparison.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_depth_cs");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid() && csShader.Valid(), "shader compilation failed");

    agfx::Texture depth = device.CreateTexture(DepthInfo());
    agfx::Texture dest = device.CreateTexture(DestInfo());

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = depth;
    rtInfo.format = kDepthFormat;
    rtInfo.isDepth = 1;
    agfx::RenderTarget depthTarget = device.CreateRenderTarget(rtInfo);

    agfx::TextureView depthSrv = device.CreateTextureView(DepthViewInfo(depth));
    agfx::TextureView destUav = device.CreateTextureView(DestViewInfo(dest));
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());

    agfx::RenderPipeline drawPipeline;
    agfx::ComputePipeline samplePipeline;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        agfx::ShaderModule csModule(device.Get(),
            CreateShaderModule(device.Get(), csShader, "main_sample_depth_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        drawPipeline = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
        samplePipeline = device.CreateComputePipeline(ComputePipelineInfo(csModule));
    }
    AGFX_EXPECT_NOT_NULL(drawPipeline.Get());
    AGFX_EXPECT_NOT_NULL(samplePipeline.Get());

    device.MakeResourcesResident();

    const DrawPushConstants drawConstants{};
    SamplePushConstants sampleConstants{};
    sampleConstants.source = (uint32_t)agfxTextureViewGetHandle(depthSrv);
    sampleConstants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    sampleConstants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

    cmd.Begin();
    cmd.TextureBarrier(depth, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(DepthPassInfo(depthTarget));
        pass.SetViewport(0.0f, 0.0f, (float)kWidth, (float)kHeight);
        pass.SetScissor(0, 0, kWidth, kHeight);
        pass.SetPipeline(drawPipeline);
        pass.PushConstants(&drawConstants, sizeof(drawConstants));
        pass.Draw(3);
    }
    cmd.TextureBarrier(depth, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("sample depth");
        pass.SetPipeline(samplePipeline);
        pass.PushConstants(&sampleConstants, sizeof(sampleConstants));
        pass.Dispatch(kGroupsX, kGroupsY, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    AGFX_EXPECT_MSG(ReadbackTexture2D(device.Get(), queue, dest, kWidth, kHeight, kDestFormat,
                                      AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image),
                    "texture readback failed");
    ExpectDepthPattern(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(SampleDepthTexture, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader vsShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("pass_actions.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    const CompiledShader csShader =
        CompileTestShader("sampling_comparison.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_depth_cs");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid() && csShader.Valid(), "shader compilation failed");

    agfx::ez::Texture2D depth = context.CreateTexture2D(
        kWidth, kHeight, kDepthFormat,
        (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED));
    agfx::ez::Texture2D dest = context.CreateTexture2D(
        kWidth, kHeight, kDestFormat,
        (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

    // ez's TextureBase::SRV() always views a texture as its own format, so it cannot express the
    // D32 -> R32F reinterpretation this test is about. The view is built off the raw texture
    // instead; everything else on this path stays ez.
    agfx::TextureView depthSrv = device.CreateTextureView(DepthViewInfo(depth.Raw()));
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());

    agfx::ShaderModule vsModule(device.Get(),
        CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
    agfx::ShaderModule psModule(device.Get(),
        CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
    agfx::ComputePipeline samplePipeline;
    {
        agfx::ShaderModule csModule(device.Get(),
            CreateShaderModule(device.Get(), csShader, "main_sample_depth_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        samplePipeline = device.CreateComputePipeline(ComputePipelineInfo(csModule));
    }
    AGFX_EXPECT_NOT_NULL(samplePipeline.Get());

    // The modules must outlive the draw: ez's PipelineDesc caches on their addresses.
    agfx::ez::PipelineDesc desc;
    desc.name = "depth sample seed";
    desc.vertexShader = &vsModule;
    desc.fragmentShader = &psModule;
    desc.cullMode = AGFX_CULL_MODE_NONE;
    desc.depthTestEnable = true;
    desc.depthWriteEnable = true;
    desc.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS;

    const DrawPushConstants drawConstants{};
    agfx::ez::ShaderBindings drawBindings;
    drawBindings.Write(&drawConstants, sizeof(drawConstants));

    agfx::ez::ShaderBindings sampleBindings;
    sampleBindings.BindTexture(depthSrv);
    sampleBindings.BindSampler(sampler);
    sampleBindings.BindTexture(dest.UAV());
    sampleBindings.Write(kWidth);
    sampleBindings.Write(kHeight);
    sampleBindings.Write(1u);  // bandCount, unused here
    sampleBindings.Write(0u);  // padding0
    sampleBindings.Write(0u);  // padding1
    sampleBindings.Write(0.0f);
    sampleBindings.Write(0.0f);
    sampleBindings.Write(0.0f);
    sampleBindings.Write(0.0f);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // A depth-only pass: no color targets.
        context.SetRenderTargets({}, &depth, AGFX_LOAD_OPERATION_CLEAR, nullptr,
                                 AGFX_LOAD_OPERATION_CLEAR, kFarDepth);
        context.SetViewportScissor(0, 0, kWidth, kHeight);
        context.SetPipeline(desc);
        context.PushShaderBindings(drawBindings);
        context.Draw(3);
        context.EndActivePass();

        context.TransitionTexture(depth, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        {
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("sample depth");
            pass.SetPipeline(samplePipeline);
            pass.PushConstants(sampleBindings.Data(), sampleBindings.Size());
            pass.Dispatch(kGroupsX, kGroupsY, 1);
        }
    }
    context.DrainGPU();

    Image image;
    AGFX_EXPECT_MSG(ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kWidth,
                                      kHeight, kDestFormat, dest.State(), image),
                    "texture readback failed");
    ExpectDepthPattern(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
