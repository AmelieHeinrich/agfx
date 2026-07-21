/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sample cube texture".
//
// Seeds a 64x64 cube map's six faces with six distinguishable patterns, then samples it through
// sampling.hlsl:main_sample_cube_cs, which walks each face's uv grid and converts (face, uv) into
// the direction that should land back on exactly that face. The six results are stacked into one
// 64x384 golden, band f per face f.
//
// A cube is the one texture type addressed by a direction rather than a coordinate, so what is under
// test is the hardware's face-selection and per-face axis convention agreeing with AGFX's documented
// face order (+X, -X, +Y, -Y, +Z, -Z). The seed makes both failure modes loud:
//
//   - each face carries a different blue level, so a backend that selects the wrong face shows a
//     band with the wrong brightness rather than a subtly different image;
//   - each face's red/green ramp is asymmetric in both axes, so a flipped or transposed s/t
//     convention on one face shows as a reversed gradient in that band alone.
//
// Unlike its 2D-array and 3D siblings, this test seeds from the host rather than with a compute
// dispatch. Those two seed on the GPU because the suite's upload helpers only address array layers
// while their slices are not layers -- but a cube's six faces *are* array layers, so the reason does
// not apply here, and a host upload avoids needing a writeable non-cube view aliased over a cube
// texture (which ez, typing every view from its texture, cannot express anyway).

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

#include <vector>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 64; // 64 * 4 bytes = a 256-byte row pitch, as D3D12 copies require.
    constexpr uint32_t kFaces = 6;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kGroupSize = 8; // Matches [numthreads(8,8,1)].
    constexpr uint32_t kGroups = (kSize + kGroupSize - 1) / kGroupSize;
    constexpr const char* kGolden = "sample_cube.png";

    /// @brief Mirrors SamplingPushConstants in sampling.hlsl.
    struct SamplingPushConstants
    {
        uint32_t source;
        uint32_t samplerId;
        uint32_t destination;
        uint32_t width;
        uint32_t height;
        uint32_t sliceCount;
        float uvScale[2];
        float uvOffset[2];
        float padding0[2];
    };

    /// @brief The cube entry point ignores the uv transform (see the shader), so it stays identity
    /// here rather than being left zeroed, which would read as "sample the same texel everywhere".
    SamplingPushConstants MakeSamplingConstants()
    {
        SamplingPushConstants constants{};
        constants.width = kSize;
        constants.height = kSize;
        constants.sliceCount = kFaces;
        constants.uvScale[0] = 1.0f;
        constants.uvScale[1] = 1.0f;
        return constants;
    }

    /// @brief One face's RGBA8 pixels: red ramps along x, green along y, blue is the face's index.
    /// Asymmetric in both axes on purpose -- a mirrored or transposed face convention reverses a
    /// ramp, which a symmetric pattern would hide.
    std::vector<uint8_t> FacePixels(uint32_t face)
    {
        std::vector<uint8_t> pixels(kSize * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y) {
            for (uint32_t x = 0; x < kSize; ++x) {
                uint8_t* texel = &pixels[(y * kSize + x) * 4];
                texel[0] = (uint8_t)((x * 255) / (kSize - 1));
                texel[1] = (uint8_t)((y * 255) / (kSize - 1));
                texel[2] = (uint8_t)((face * 255) / (kFaces - 1));
                texel[3] = 255;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo SourceInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_CUBE;
        info.format = kFormat;
        info.usage = AGFX_TEXTURE_USAGE_SAMPLED; // Host-seeded, so no STORAGE bit is needed.
        info.width = kSize;
        info.height = kSize;
        info.depthOrArrayLayers = kFaces; // The six faces, as array layers.
        info.mipLevels = 1;
        return info;
    }

    /// @brief The destination is a flat 2D texture holding all six bands stacked, so one golden
    /// covers the whole cube.
    agfxTextureCreateInfo DestInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kSize;
        info.height = kSize * kFaces;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    /// @brief The source is read as a cube, so the view's type must be CUBE and cover all six
    /// layers -- a 2D_ARRAY view here would bind as a TextureCube of the wrong shape in the shader.
    agfxTextureViewCreateInfo CubeViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_CUBE;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = kFaces;
        info.writeable = 0;
        return info;
    }

    agfxTextureViewCreateInfo FlatViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.baseMipLevel = 0;
        info.mipLevelCount = 1;
        info.baseArrayLayer = 0;
        info.arrayLayerCount = 1;
        info.writeable = 1;
        return info;
    }

    agfxSamplerCreateInfo SamplerInfo()
    {
        agfxSamplerCreateInfo info{};
        info.filter = AGFX_SAMPLER_FILTER_LINEAR;
        info.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxAnisotropy = 1.0f;
        info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
        info.minLod = 0.0f;
        info.maxLod = 0.0f;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "cube sample";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }

    /// @brief Uploads all six faces into `texture`, one array layer each. Shared by the three
    /// flavors so a seeding difference can never be what makes their images differ.
    bool SeedFaces(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture)
    {
        for (uint32_t face = 0; face < kFaces; ++face) {
            const std::vector<uint8_t> pixels = FacePixels(face);
            if (!UploadTextureSubresource(device, queue, texture, kSize, kSize, kFormat, pixels.data(),
                                          AGFX_RESOURCE_STATE_COMMON, /*mipLevel*/ 0, /*layer*/ face)) {
                return false;
            }
        }
        return true;
    }
} // namespace

AGFX_TEST_TEXTURE(SampleCube, C, kSize, kSize* kFaces)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_cube_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_cube_cs");

    const agfxTextureCreateInfo sourceInfo = SourceInfo();
    const agfxTextureCreateInfo destInfo = DestInfo();
    agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    const agfxTextureViewCreateInfo sourceSrvInfo = CubeViewInfo(source);
    const agfxTextureViewCreateInfo destUavInfo = FlatViewInfo(dest);
    agfxTextureView* sourceSrv = agfxTextureViewCreate(device, &sourceSrvInfo);
    agfxTextureView* destUav = agfxTextureViewCreate(device, &destUavInfo);
    AGFX_EXPECT_NOT_NULL(sourceSrv);
    AGFX_EXPECT_NOT_NULL(destUav);

    const agfxSamplerCreateInfo samplerInfo = SamplerInfo();
    agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);
    AGFX_EXPECT_NOT_NULL(sampler);

    AGFX_EXPECT_MSG(SeedFaces(device, gpu.Queue(), source), "cube face upload failed");

    agfxShaderModule* sampleModule =
        CreateShaderModule(device, sampleShader, "main_sample_cube_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfo = ComputePipelineInfo(sampleModule);
    agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
    agfxShaderModuleDestroy(device, sampleModule);
    AGFX_EXPECT_NOT_NULL(pipeline);

    agfxDeviceMakeResourcesResident(device);

    SamplingPushConstants constants = MakeSamplingConstants();
    constants.source = (uint32_t)agfxTextureViewGetHandle(sourceSrv);
    constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
    constants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        // Every face must be transitioned, not just layer 0, and agglomerate must be true: a barrier
        // recorded with false is a no-op on the Metal backend.
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "cube sample");
        agfxComputePassSetPipeline(pass, pipeline);
        agfxComputePassPushConstants(pass, &constants, sizeof(constants));
        agfxComputePassDispatch(pass, kGroups, kGroups, kFaces);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kSize, kSize * kFaces, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, pipeline);
    agfxSamplerDestroy(device, sampler);
    agfxTextureViewDestroy(device, destUav);
    agfxTextureViewDestroy(device, sourceSrv);
    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(SampleCube, Cpp, kSize, kSize* kFaces)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_cube_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_cube_cs");

    agfx::Texture source = device.CreateTexture(SourceInfo());
    agfx::Texture dest = device.CreateTexture(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    agfx::TextureView sourceSrv = device.CreateTextureView(CubeViewInfo(source));
    agfx::TextureView destUav = device.CreateTextureView(FlatViewInfo(dest));
    AGFX_EXPECT_NOT_NULL(sourceSrv.Get());
    AGFX_EXPECT_NOT_NULL(destUav.Get());

    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());
    AGFX_EXPECT_NOT_NULL(sampler.Get());

    AGFX_EXPECT_MSG(SeedFaces(device.Get(), queue, source), "cube face upload failed");

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule sampleModule(device.Get(),
            CreateShaderModule(device.Get(), sampleShader, "main_sample_cube_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(sampleModule));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    device.MakeResourcesResident();

    SamplingPushConstants constants = MakeSamplingConstants();
    constants.source = (uint32_t)sourceSrv.GetHandle();
    constants.samplerId = (uint32_t)sampler.GetHandle();
    constants.destination = (uint32_t)destUav.GetHandle();

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON,
                       AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_SUBRESOURCE_ALL_MIPS,
                       AGFX_SUBRESOURCE_ALL_LAYERS, true);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("cube sample");
        pass.SetPipeline(pipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroups, kGroups, kFaces);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, dest, kSize, kSize * kFaces, kFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(SampleCube, Ez, kSize, kSize* kFaces)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kSize;
    contextInfo.height = kSize * kFaces;
    agfx::ez::Context context(contextInfo);

    const CompiledShader sampleShader =
        CompileTestShader("sampling.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_cube_cs");
    AGFX_EXPECT_MSG(sampleShader.Valid(), "failed to compile sampling.hlsl:main_sample_cube_cs");

    agfx::Device& device = context.GetDevice();

    agfx::ez::TextureCube source = context.CreateTextureCube(kSize, kFormat, AGFX_TEXTURE_USAGE_SAMPLED);
    agfx::ez::Texture2D dest = context.CreateTexture2D(
        kSize, kSize * kFaces, kFormat,
        (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED));

    // ez has no sampler sugar; samplers come straight off the device.
    agfx::Sampler sampler = device.CreateSampler(SamplerInfo());
    AGFX_EXPECT_NOT_NULL(sampler.Get());

    AGFX_EXPECT_MSG(SeedFaces(device.Get(), context.GetGraphicsQueue(), source.Raw()),
                    "cube face upload failed");

    agfx::ComputePipeline pipeline;
    {
        agfx::ShaderModule sampleModule(device.Get(),
            CreateShaderModule(device.Get(), sampleShader, "main_sample_cube_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipeline = device.CreateComputePipeline(ComputePipelineInfo(sampleModule));
    }
    AGFX_EXPECT_NOT_NULL(pipeline.Get());

    // ez types every view from its texture, so a cube's SRV() is already the CUBE-typed view the
    // shader wants -- no raw agfxTextureViewCreate needed on this path.
    agfx::ez::ShaderBindings bindings;
    bindings.BindTexture(source.SRV());
    bindings.BindSampler(sampler);
    bindings.BindTexture(dest.UAV());
    bindings.Write(kSize);
    bindings.Write(kSize);
    bindings.Write(kFaces);
    bindings.Write(1.0f); // uvScale.x -- identity; the cube path ignores the transform.
    bindings.Write(1.0f); // uvScale.y
    bindings.Write(0.0f); // uvOffset.x
    bindings.Write(0.0f); // uvOffset.y

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        {
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("cube sample");
            pass.SetPipeline(pipeline);
            pass.PushConstants(bindings.Data(), bindings.Size());
            pass.Dispatch(kGroups, kGroups, kFaces);
        }
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kSize,
                                          kSize * kFaces, kFormat, dest.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
