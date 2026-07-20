/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 12:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// dear imgui: Renderer Backend for agfx

#include "imgui_impl_agfx.h"
#ifndef IMGUI_DISABLE

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// A minimal staging-buffer uploader: data is copied into a CPU-visible staging buffer, then
// transferred to the (GPU-only) destination texture via a GPU copy. This is the pattern
// required by non-UMA backends (D3D12's default heap resources aren't CPU-mappable) -- unlike
// agfxBufferMap/agfxTextureReplaceRegion, which only work because Metal4/UMA lets any resource
// be CPU-visible.
struct ImGui_ImplAGFX_Uploader
{
    agfxCommandQueue*           Queue = nullptr;
    agfxCommandBuffer*          CmdBuffer = nullptr;
    agfxFence*                  Fence = nullptr;
    uint64_t                    FenceValue = 0;
    agfxComputePass*            ActivePass = nullptr;
    std::vector<agfxBuffer*>    PendingStagingBuffers;
};

struct ImGui_ImplAGFX_Texture
{
    agfxTexture*        Texture = nullptr;
    agfxTextureView*    View = nullptr;
};

// Backend data stored in io.BackendRendererUserData to allow support for multiple contexts
struct ImGui_ImplAGFX_Data
{
    ImGui_ImplAGFX_InitInfo     InitInfo;
    agfxRenderPipeline*         Pipeline = nullptr;
    agfxShaderModule*           VertexShader = nullptr;
    agfxShaderModule*           FragmentShader = nullptr;
    agfxSampler*                Sampler = nullptr;

    std::vector<agfxBuffer*>     VertexBuffer;
    std::vector<agfxBufferView*> VertexBufferView;
    std::vector<uint64_t>        VertexBufferCapacity;

    std::vector<agfxBuffer*>     IndexBuffer;
    std::vector<uint64_t>        IndexBufferCapacity;

    ImGui_ImplAGFX_Uploader     Uploader;
};

static ImGui_ImplAGFX_Data* ImGui_ImplAGFX_GetBackendData()
{
    return ImGui::GetCurrentContext() ? (ImGui_ImplAGFX_Data*)ImGui::GetIO().BackendRendererUserData : nullptr;
}

//-----------------------------------------------------------------------------------------------
// UPLOADER
//-----------------------------------------------------------------------------------------------

static void ImGui_ImplAGFX_UploaderInit(ImGui_ImplAGFX_Uploader* uploader, agfxDevice* device, agfxCommandQueue* queue)
{
    uploader->Queue = queue;
    uploader->CmdBuffer = agfxCommandBufferCreate(device, queue);
    uploader->Fence = agfxFenceCreate(device);

    agfxCommandBufferReset(uploader->CmdBuffer);
    agfxCommandBufferBegin(uploader->CmdBuffer);
}

static void ImGui_ImplAGFX_UploaderShutdown(ImGui_ImplAGFX_Uploader* uploader, agfxDevice* device)
{
    if (uploader->Fence) agfxFenceDestroy(device, uploader->Fence);
    if (uploader->CmdBuffer) agfxCommandBufferDestroy(device, uploader->CmdBuffer);
    uploader->Fence = nullptr;
    uploader->CmdBuffer = nullptr;
}

static void ImGui_ImplAGFX_UploadTexture(ImGui_ImplAGFX_Uploader* uploader, agfxDevice* device, agfxTexture* dstTexture, const agfxTextureRegion* region, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage)
{
    if (dataSize == 0) return;

    agfxBufferCreateInfo stagingInfo = {};
    stagingInfo.size = dataSize;
    stagingInfo.stride = dataSize;
    stagingInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
    stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
    agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
    agfxBufferSetName(staging, "ImGui Upload Staging Buffer");

    void* dst = agfxBufferMap(staging);
    memcpy(dst, data, dataSize);
    agfxBufferUnmap(staging);

    if (!uploader->ActivePass)
        uploader->ActivePass = agfxComputePassBegin(uploader->CmdBuffer, "ImGui Texture Upload");
    agfxComputePassCopyBufferToTexture(uploader->ActivePass, staging, dstTexture, region, 0, 0, bytesPerRow, bytesPerImage);

    uploader->PendingStagingBuffers.push_back(staging);
}

static void ImGui_ImplAGFX_UploaderFlush(ImGui_ImplAGFX_Uploader* uploader, agfxDevice* device)
{
    if (uploader->PendingStagingBuffers.empty()) return;

    agfxDeviceMakeResourcesResident(device);

    if (uploader->ActivePass) {
        agfxComputePassEnd(uploader->ActivePass);
        uploader->ActivePass = nullptr;
    }

    agfxCommandBufferEnd(uploader->CmdBuffer);
    agfxCommandQueueSubmit(uploader->Queue, &uploader->CmdBuffer, 1);

    uploader->FenceValue++;
    agfxCommandQueueSignal(uploader->Queue, uploader->Fence, uploader->FenceValue);
    agfxFenceWait(uploader->Fence, uploader->FenceValue, UINT64_MAX);

    for (agfxBuffer* staging : uploader->PendingStagingBuffers)
        agfxBufferDestroy(device, staging);
    uploader->PendingStagingBuffers.clear();

    agfxCommandBufferReset(uploader->CmdBuffer);
    agfxCommandBufferBegin(uploader->CmdBuffer);
}

//-----------------------------------------------------------------------------------------------
// TEXTURES
//-----------------------------------------------------------------------------------------------

static void ImGui_ImplAGFX_DestroyFontTexture(agfxDevice* device, ImTextureData* tex)
{
    if (ImGui_ImplAGFX_Texture* backendTex = (ImGui_ImplAGFX_Texture*)tex->BackendUserData) {
        if (backendTex->View) agfxTextureViewDestroy(device, backendTex->View);
        if (backendTex->Texture) agfxTextureDestroy(device, backendTex->Texture);
        delete backendTex;
        tex->BackendUserData = nullptr;
    }
    tex->SetTexID(ImTextureID_Invalid);
    tex->SetStatus(ImTextureStatus_Destroyed);
}

static void ImGui_ImplAGFX_UpdateTexture(ImGui_ImplAGFX_Data* bd, agfxDevice* device, ImTextureData* tex)
{
    if (tex->Status == ImTextureStatus_WantCreate) {
        agfxTextureCreateInfo textureCreateInfo = {};
        textureCreateInfo.type = AGFX_TEXTURE_TYPE_2D;
        textureCreateInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
        textureCreateInfo.usage = AGFX_TEXTURE_USAGE_SAMPLED;
        textureCreateInfo.width = (uint32_t)tex->Width;
        textureCreateInfo.height = (uint32_t)tex->Height;
        textureCreateInfo.depthOrArrayLayers = 1;
        textureCreateInfo.mipLevels = 1;
        agfxTexture* texture = agfxTextureCreate(device, &textureCreateInfo);
        agfxTextureSetName(texture, "ImGui Font Texture");

        agfxTextureRegion region = {};
        region.x = 0;
        region.y = 0;
        region.z = 0;
        region.width = (uint32_t)tex->Width;
        region.height = (uint32_t)tex->Height;
        region.depth = 1;

        uint32_t bytesPerRow = (uint32_t)tex->Width * 4;
        uint32_t dataSize = bytesPerRow * (uint32_t)tex->Height;
        ImGui_ImplAGFX_UploadTexture(&bd->Uploader, device, texture, &region, tex->Pixels, dataSize, bytesPerRow, dataSize);

        agfxTextureViewCreateInfo viewCreateInfo = {};
        viewCreateInfo.texture = texture;
        viewCreateInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
        viewCreateInfo.type = AGFX_TEXTURE_TYPE_2D;
        viewCreateInfo.baseMipLevel = 0;
        viewCreateInfo.mipLevelCount = 1;
        viewCreateInfo.baseArrayLayer = 0;
        viewCreateInfo.arrayLayerCount = 1;
        viewCreateInfo.writeable = false;
        agfxTextureView* view = agfxTextureViewCreate(device, &viewCreateInfo);

        ImGui_ImplAGFX_Texture* backendTex = new ImGui_ImplAGFX_Texture();
        backendTex->Texture = texture;
        backendTex->View = view;

        tex->SetTexID((ImTextureID)(intptr_t)agfxTextureViewGetHandle(view));
        tex->SetStatus(ImTextureStatus_OK);
        tex->BackendUserData = backendTex;
    } else if (tex->Status == ImTextureStatus_WantUpdates) {
        ImGui_ImplAGFX_Texture* backendTex = (ImGui_ImplAGFX_Texture*)tex->BackendUserData;

        for (ImTextureRect& r : tex->Updates) {
            std::vector<uint8_t> packed((size_t)r.w * r.h * 4);
            for (int row = 0; row < (int)r.h; ++row) {
                const void* src = tex->GetPixelsAt(r.x, r.y + row);
                memcpy(packed.data() + (size_t)row * r.w * 4, src, (size_t)r.w * 4);
            }

            agfxTextureRegion region = {};
            region.x = (uint32_t)r.x;
            region.y = (uint32_t)r.y;
            region.z = 0;
            region.width = (uint32_t)r.w;
            region.height = (uint32_t)r.h;
            region.depth = 1;

            uint32_t bytesPerRow = (uint32_t)r.w * 4;
            uint32_t dataSize = (uint32_t)packed.size();
            ImGui_ImplAGFX_UploadTexture(&bd->Uploader, device, backendTex->Texture, &region, packed.data(), dataSize, bytesPerRow, dataSize);
        }
        tex->SetStatus(ImTextureStatus_OK);
    } else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
        ImGui_ImplAGFX_DestroyFontTexture(device, tex);
    }
}

static void ImGui_ImplAGFX_UpdateTextures(ImGui_ImplAGFX_Data* bd, agfxDevice* device, ImDrawData* drawData)
{
    if (drawData->Textures != nullptr)
        for (ImTextureData* tex : *drawData->Textures)
            if (tex->Status != ImTextureStatus_OK)
                ImGui_ImplAGFX_UpdateTexture(bd, device, tex);
    ImGui_ImplAGFX_UploaderFlush(&bd->Uploader, device);
}

//-----------------------------------------------------------------------------------------------
// DEVICE OBJECTS
//-----------------------------------------------------------------------------------------------

static void ImGui_ImplAGFX_CreatePipeline(ImGui_ImplAGFX_Data* bd, agfxDevice* device)
{
    if (bd->Pipeline) agfxRenderPipelineDestroy(device, bd->Pipeline);

    agfxRenderPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.name = "ImGui Pipeline";
    pipelineInfo.supportsIndirect = false;
    pipelineInfo.fillMode = AGFX_FILL_MODE_SOLID;
    pipelineInfo.cullMode = AGFX_CULL_MODE_NONE;
    pipelineInfo.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
    pipelineInfo.topology = AGFX_TOPOLOGY_TRIANGLES;
    pipelineInfo.depthTestEnable = false;
    pipelineInfo.depthWriteEnable = false;
    pipelineInfo.depthClampEnable = false;
    pipelineInfo.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS;
    pipelineInfo.depthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
    pipelineInfo.blendEnable[0] = true;
    pipelineInfo.srcColorBlendFactor[0] = AGFX_BLEND_FACTOR_SRC_ALPHA;
    pipelineInfo.dstColorBlendFactor[0] = AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineInfo.colorBlendOp[0] = AGFX_BLEND_OPERATION_ADD;
    pipelineInfo.srcAlphaBlendFactor[0] = AGFX_BLEND_FACTOR_ONE;
    pipelineInfo.dstAlphaBlendFactor[0] = AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineInfo.alphaBlendOp[0] = AGFX_BLEND_OPERATION_ADD;
    pipelineInfo.colorFormats[0] = bd->InitInfo.ColorAttachmentFormat;
    pipelineInfo.colorAttachmentCount = 1;
    pipelineInfo.vertexShader = bd->VertexShader;
    pipelineInfo.fragmentShader = bd->FragmentShader;
    bd->Pipeline = agfxRenderPipelineCreate(device, &pipelineInfo);
}

bool ImGui_ImplAGFX_CreateDeviceObjects()
{
    ImGui_ImplAGFX_Data* bd = ImGui_ImplAGFX_GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend initialized!");
    agfxDevice* device = bd->InitInfo.Device;

    bd->VertexShader = bd->InitInfo.VertexShaderModule;
    bd->FragmentShader = bd->InitInfo.FragmentShaderModule;

    ImGui_ImplAGFX_CreatePipeline(bd, device);

    agfxSamplerCreateInfo samplerInfo = {};
    samplerInfo.filter = AGFX_SAMPLER_FILTER_LINEAR;
    samplerInfo.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    bd->Sampler = agfxSamplerCreate(device, &samplerInfo);

    return true;
}

void ImGui_ImplAGFX_InvalidateDeviceObjects()
{
    ImGui_ImplAGFX_Data* bd = ImGui_ImplAGFX_GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend initialized!");
    agfxDevice* device = bd->InitInfo.Device;

    if (bd->Sampler) agfxSamplerDestroy(device, bd->Sampler);
    if (bd->Pipeline) agfxRenderPipelineDestroy(device, bd->Pipeline);
    if (bd->VertexShader) agfxShaderModuleDestroy(device, bd->VertexShader);
    if (bd->FragmentShader) agfxShaderModuleDestroy(device, bd->FragmentShader);

    bd->Sampler = nullptr;
    bd->Pipeline = nullptr;
    bd->VertexShader = nullptr;
    bd->FragmentShader = nullptr;
}

void ImGui_ImplAGFX_RecreatePipeline(agfxTextureFormat new_color_format)
{
    ImGui_ImplAGFX_Data* bd = ImGui_ImplAGFX_GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend initialized!");
    bd->InitInfo.ColorAttachmentFormat = new_color_format;
    ImGui_ImplAGFX_CreatePipeline(bd, bd->InitInfo.Device);
}

//-----------------------------------------------------------------------------------------------
// MAIN INTERFACE
//-----------------------------------------------------------------------------------------------

bool ImGui_ImplAGFX_Init(ImGui_ImplAGFX_InitInfo* info)
{
    ImGuiIO& io = ImGui::GetIO();
    IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    ImGui_ImplAGFX_Data* bd = IM_NEW(ImGui_ImplAGFX_Data)();
    bd->InitInfo = *info;
    bd->VertexBuffer.resize(info->FramesInFlight, nullptr);
    bd->VertexBufferView.resize(info->FramesInFlight, nullptr);
    bd->VertexBufferCapacity.resize(info->FramesInFlight, 0);
    bd->IndexBuffer.resize(info->FramesInFlight, nullptr);
    bd->IndexBufferCapacity.resize(info->FramesInFlight, 0);

    io.BackendRendererUserData = (void*)bd;
    io.BackendRendererName = "imgui_impl_agfx";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;

    ImGui_ImplAGFX_UploaderInit(&bd->Uploader, info->Device, info->CommandQueue);

    return ImGui_ImplAGFX_CreateDeviceObjects();
}

void ImGui_ImplAGFX_Shutdown()
{
    ImGui_ImplAGFX_Data* bd = ImGui_ImplAGFX_GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();
    agfxDevice* device = bd->InitInfo.Device;

    ImGuiPlatformIO& platformIO = ImGui::GetPlatformIO();
    for (ImTextureData* tex : platformIO.Textures)
        ImGui_ImplAGFX_DestroyFontTexture(device, tex);

    for (size_t i = 0; i < bd->VertexBuffer.size(); ++i) {
        if (bd->VertexBufferView[i]) agfxBufferViewDestroy(device, bd->VertexBufferView[i]);
        if (bd->VertexBuffer[i]) agfxBufferDestroy(device, bd->VertexBuffer[i]);
        if (bd->IndexBuffer[i]) agfxBufferDestroy(device, bd->IndexBuffer[i]);
    }

    ImGui_ImplAGFX_UploaderShutdown(&bd->Uploader, device);
    ImGui_ImplAGFX_InvalidateDeviceObjects();

    io.BackendRendererName = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
    IM_DELETE(bd);
}

void ImGui_ImplAGFX_NewFrame()
{
    ImGui_ImplAGFX_Data* bd = ImGui_ImplAGFX_GetBackendData();
    IM_ASSERT(bd != nullptr && "Did you call ImGui_ImplAGFX_Init()?");
    IM_UNUSED(bd);
}

void ImGui_ImplAGFX_RenderDrawData(ImDrawData* drawData, agfxRenderPass* renderPass, uint32_t fbWidth, uint32_t fbHeight, uint32_t frameIndex)
{
    if (!drawData || drawData->CmdListsCount == 0 || fbWidth == 0 || fbHeight == 0)
        return;

    ImGui_ImplAGFX_Data* bd = ImGui_ImplAGFX_GetBackendData();
    IM_ASSERT(bd != nullptr && "Did you call ImGui_ImplAGFX_Init()?");
    IM_ASSERT(frameIndex < bd->VertexBuffer.size() && "frame_index is out of bounds of InitInfo.FramesInFlight");
    agfxDevice* device = bd->InitInfo.Device;

    ImGui_ImplAGFX_UpdateTextures(bd, device, drawData);

    uint64_t vtxSize = (uint64_t)drawData->TotalVtxCount * sizeof(ImDrawVert);
    uint64_t idxSize = (uint64_t)drawData->TotalIdxCount * sizeof(ImDrawIdx);
    if (vtxSize == 0 || idxSize == 0)
        return;

    if (vtxSize > bd->VertexBufferCapacity[frameIndex]) {
        if (bd->VertexBufferView[frameIndex]) agfxBufferViewDestroy(device, bd->VertexBufferView[frameIndex]);
        if (bd->VertexBuffer[frameIndex]) agfxBufferDestroy(device, bd->VertexBuffer[frameIndex]);

        uint64_t newCapacity = vtxSize + vtxSize / 2 + (uint64_t)sizeof(ImDrawVert) * 1024;
        agfxBufferCreateInfo bufferInfo = {};
        bufferInfo.size = newCapacity;
        bufferInfo.stride = sizeof(ImDrawVert);
        bufferInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        bufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
        bd->VertexBuffer[frameIndex] = agfxBufferCreate(device, &bufferInfo);
        agfxBufferSetName(bd->VertexBuffer[frameIndex], "ImGui Vertex Buffer");

        agfxBufferViewCreateInfo viewInfo = {};
        viewInfo.buffer = bd->VertexBuffer[frameIndex];
        viewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        viewInfo.offset = 0;
        viewInfo.writeable = false;
        bd->VertexBufferView[frameIndex] = agfxBufferViewCreate(device, &viewInfo);

        bd->VertexBufferCapacity[frameIndex] = newCapacity;
    }

    if (idxSize > bd->IndexBufferCapacity[frameIndex]) {
        if (bd->IndexBuffer[frameIndex]) agfxBufferDestroy(device, bd->IndexBuffer[frameIndex]);

        uint64_t newCapacity = idxSize + idxSize / 2 + (uint64_t)sizeof(ImDrawIdx) * 1024;
        newCapacity = (newCapacity + sizeof(ImDrawIdx) - 1) & ~(uint64_t)(sizeof(ImDrawIdx) - 1);
        agfxBufferCreateInfo bufferInfo = {};
        bufferInfo.size = newCapacity;
        bufferInfo.stride = sizeof(ImDrawIdx);
        bufferInfo.usage = AGFX_BUFFER_USAGE_INDEX;
        bufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
        bd->IndexBuffer[frameIndex] = agfxBufferCreate(device, &bufferInfo);
        agfxBufferSetName(bd->IndexBuffer[frameIndex], "ImGui Index Buffer");

        bd->IndexBufferCapacity[frameIndex] = newCapacity;
    }

    uint8_t* vtxDst = (uint8_t*)agfxBufferMap(bd->VertexBuffer[frameIndex]);
    uint8_t* idxDst = (uint8_t*)agfxBufferMap(bd->IndexBuffer[frameIndex]);
    uint64_t vtxByteOffset = 0;
    uint64_t idxByteOffset = 0;
    for (int i = 0; i < drawData->CmdListsCount; ++i) {
        const ImDrawList* list = drawData->CmdLists[i];
        uint64_t vtxBytes = (uint64_t)list->VtxBuffer.Size * sizeof(ImDrawVert);
        uint64_t idxBytes = (uint64_t)list->IdxBuffer.Size * sizeof(ImDrawIdx);
        if (vtxBytes > 0) memcpy(vtxDst + vtxByteOffset, list->VtxBuffer.Data, vtxBytes);
        if (idxBytes > 0) memcpy(idxDst + idxByteOffset, list->IdxBuffer.Data, idxBytes);
        vtxByteOffset += vtxBytes;
        idxByteOffset += idxBytes;
    }
    agfxBufferUnmap(bd->VertexBuffer[frameIndex]);
    agfxBufferUnmap(bd->IndexBuffer[frameIndex]);

    agfxDeviceMakeResourcesResident(device);

    agfxRenderPassSetViewport(renderPass, 0.0f, 0.0f, (float)fbWidth, (float)fbHeight, 0.0f, 1.0f);
    agfxRenderPassSetPipeline(renderPass, bd->Pipeline);

    ImVec2 clipOff = drawData->DisplayPos;
    ImVec2 clipScale = drawData->FramebufferScale;

    struct ImGui_ImplAGFX_PushConstants {
        float scale[2];
        float translate[2];
        uint32_t vertexOffset;
        uint32_t vertexBuffer;
        uint32_t texture;
        uint32_t textureSampler;
    } pc = {};
    pc.scale[0] = 2.0f / drawData->DisplaySize.x;
    pc.scale[1] = -2.0f / drawData->DisplaySize.y;
    pc.translate[0] = -1.0f - drawData->DisplayPos.x * pc.scale[0];
    pc.translate[1] = 1.0f - drawData->DisplayPos.y * pc.scale[1];
    pc.vertexBuffer = (uint32_t)agfxBufferViewGetHandle(bd->VertexBufferView[frameIndex]);
    pc.textureSampler = (uint32_t)agfxSamplerGetHandle(bd->Sampler);

    int32_t globalVtxOffset = 0;
    int32_t globalIdxOffset = 0;
    for (int i = 0; i < drawData->CmdListsCount; ++i) {
        const ImDrawList* list = drawData->CmdLists[i];
        for (int cmdIndex = 0; cmdIndex < list->CmdBuffer.Size; ++cmdIndex) {
            const ImDrawCmd& cmd = list->CmdBuffer[cmdIndex];
            if (cmd.UserCallback) {
                if (cmd.UserCallback != ImDrawCallback_ResetRenderState)
                    cmd.UserCallback(list, &cmd);
                continue;
            }

            ImVec2 clipMin((cmd.ClipRect.x - clipOff.x) * clipScale.x, (cmd.ClipRect.y - clipOff.y) * clipScale.y);
            ImVec2 clipMax((cmd.ClipRect.z - clipOff.x) * clipScale.x, (cmd.ClipRect.w - clipOff.y) * clipScale.y);
            if (clipMin.x < 0.0f) clipMin.x = 0.0f;
            if (clipMin.y < 0.0f) clipMin.y = 0.0f;
            if (clipMax.x > (float)fbWidth) clipMax.x = (float)fbWidth;
            if (clipMax.y > (float)fbHeight) clipMax.y = (float)fbHeight;
            if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y)
                continue;
            if (cmd.ElemCount == 0)
                continue;

            agfxRenderPassSetScissor(renderPass, (uint32_t)clipMin.x, (uint32_t)clipMin.y, (uint32_t)(clipMax.x - clipMin.x), (uint32_t)(clipMax.y - clipMin.y));

            pc.texture = (uint32_t)(intptr_t)cmd.GetTexID();
            pc.vertexOffset = (uint32_t)(cmd.VtxOffset + globalVtxOffset);
            agfxRenderPassPushConstants(renderPass, &pc, sizeof(pc));

            agfxRenderPassDrawIndexed(renderPass, bd->IndexBuffer[frameIndex], cmd.ElemCount, 1, cmd.IdxOffset + globalIdxOffset, 0, 0);
        }
        globalVtxOffset += list->VtxBuffer.Size;
        globalIdxOffset += list->IdxBuffer.Size;
    }
}

#endif // #ifndef IMGUI_DISABLE
