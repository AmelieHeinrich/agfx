/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx_mipgen.h"
#include "demo_file_utils.h"

#include <agfx_shader/agfx_shader_compiler.h>

#include <algorithm>
#include <string>

namespace {

struct MipGenPushConstants {
    uint32_t srcTex;
    uint32_t dstTex;
    uint32_t dstWidth;
    uint32_t dstHeight;
};

} // namespace

AgfxMipGen::AgfxMipGen(agfxDevice* device)
{
    m_device = device;

    std::string source = ReadFile((std::string(kDataDir) + "shaders/demo/mipgen.hlsl").c_str());

    agfxShaderCompilerOptions options = {};
    options.stage = AGFX_SHADER_STAGE_COMPUTE;
    strncpy(options.entryPoint, "main_cs", sizeof(options.entryPoint) - 1);
    options.sourceCode = source.empty() ? nullptr : const_cast<char*>(&source[0]);
    options.sourceCodeSize = (uint32_t)source.size();

    agfxShaderCompilerResult result = {};
    agfxCompileShader(&options, &result);

    agfxShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.code = result.compiledCode;
    moduleInfo.codeSize = result.compiledSize;
    moduleInfo.entryPoint = "main_cs";
    moduleInfo.type = AGFX_SHADER_MODULE_TYPE_COMPUTE;
    computeShader = agfxShaderModuleCreate(device, &moduleInfo);
    free(result.compiledCode);

    agfxComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.name = "MipGen Pipeline";
    pipelineInfo.computeShader = computeShader;
    pipelineInfo.groupSizeX = 8;
    pipelineInfo.groupSizeY = 8;
    pipelineInfo.groupSizeZ = 1;
    pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
}

AgfxMipGen::~AgfxMipGen()
{
    if (pipeline) agfxComputePipelineDestroy(m_device, pipeline);
    if (computeShader) agfxShaderModuleDestroy(m_device, computeShader);
}

void AgfxMipGen::Generate(agfxDevice* device, agfxCommandBuffer* cmdBuffer, agfxTexture* texture, uint32_t width, uint32_t height, uint32_t mipLevels, std::vector<agfxTextureView*>& outViews)
{
    agfxTextureCreateInfo textureInfo = {};
    agfxTextureGetInfo(texture, &textureInfo);

    // These transitions must agglomerate (the trailing `true`): each mip is written by one pass's
    // dispatch and read by the next pass's, so the hazard is between encoders, which is exactly what
    // the Metal backend's pending-barrier flush at the next pass boundary covers. Passing false here
    // is documented as a no-op on Metal and silently drops the whole chain.
    //
    // Mip 0 was last written by the uploader's staging copy, which leaves it in
    // COPY_DEST (written-to states don't implicitly decay back to COMMON like
    // read-only promoted states do). Every other mip has never been written, so
    // it starts as COMMON. Once a mip has been used as a downsample source it's
    // left in PIXEL_SHADER_RESOURCE, which is the state the *next* iteration
    // must transition it from.
    agfxResourceState srcState = AGFX_RESOURCE_STATE_COPY_DEST;

    for (uint32_t mip = 0; mip + 1 < mipLevels; ++mip) {
        uint32_t dstWidth = std::max(1u, width >> (mip + 1));
        uint32_t dstHeight = std::max(1u, height >> (mip + 1));

        agfxTextureViewCreateInfo srcViewInfo = {};
        srcViewInfo.texture = texture;
        srcViewInfo.format = textureInfo.format;
        srcViewInfo.type = AGFX_TEXTURE_TYPE_2D;
        srcViewInfo.baseMipLevel = mip;
        srcViewInfo.mipLevelCount = 1;
        srcViewInfo.arrayLayerCount = 1;
        srcViewInfo.writeable = false;
        agfxTextureView* srcView = agfxTextureViewCreate(device, &srcViewInfo);

        agfxTextureViewCreateInfo dstViewInfo = srcViewInfo;
        dstViewInfo.baseMipLevel = mip + 1;
        dstViewInfo.writeable = true;
        agfxTextureView* dstView = agfxTextureViewCreate(device, &dstViewInfo);

        agfxDeviceMakeResourcesResident(device);

        agfxCommandBufferTextureBarrier(cmdBuffer, texture, srcState, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mip, 0, true);
        agfxCommandBufferTextureBarrier(cmdBuffer, texture, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, mip + 1, 0, true);

        agfxComputePass* pass = agfxComputePassBegin(cmdBuffer, "MipGen");
        agfxComputePassSetPipeline(pass, pipeline);

        MipGenPushConstants pc = {};
        pc.srcTex = (uint32_t)agfxTextureViewGetHandle(srcView);
        pc.dstTex = (uint32_t)agfxTextureViewGetHandle(dstView);
        pc.dstWidth = dstWidth;
        pc.dstHeight = dstHeight;
        agfxComputePassPushConstants(pass, &pc, sizeof(pc));

        agfxComputePassDispatch(pass, (dstWidth + 7) / 8, (dstHeight + 7) / 8, 1);
        agfxComputePassEnd(pass);

        agfxCommandBufferTextureBarrier(cmdBuffer, texture, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, mip, 0, true);
        agfxCommandBufferTextureBarrier(cmdBuffer, texture, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, mip + 1, 0, true);

        outViews.push_back(srcView);
        outViews.push_back(dstView);

        srcState = AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }
}
