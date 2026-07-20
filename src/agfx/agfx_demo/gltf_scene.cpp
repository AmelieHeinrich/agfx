/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "gltf_scene.h"
#include "agfx_uploader.h"
#include "agfx_mipgen.h"
#include "demo_file_utils.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

namespace {

glm::vec3 ReadFloat3(const cgltf_accessor* accessor, size_t index)
{
    float out[3] = {0.0f, 0.0f, 0.0f};
    cgltf_accessor_read_float(accessor, index, out, 3);
    return glm::vec3(out[0], out[1], out[2]);
}

glm::vec2 ReadFloat2(const cgltf_accessor* accessor, size_t index)
{
    float out[2] = {0.0f, 0.0f};
    cgltf_accessor_read_float(accessor, index, out, 2);
    return glm::vec2(out[0], out[1]);
}

glm::vec4 ReadFloat4(const cgltf_accessor* accessor, size_t index)
{
    float out[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    cgltf_accessor_read_float(accessor, index, out, 4);
    return glm::vec4(out[0], out[1], out[2], out[3]);
}

std::string DirectoryOf(const char* path)
{
    std::string s(path);
    size_t slash = s.find_last_of('/');
    if (slash == std::string::npos) return ".";
    return s.substr(0, slash);
}

void ComputeTangents(std::vector<SceneVertex>& verts, const std::vector<uint32_t>& localIndices)
{
    std::vector<glm::vec3> tan(verts.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitan(verts.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < localIndices.size(); i += 3) {
        uint32_t i0 = localIndices[i], i1 = localIndices[i + 1], i2 = localIndices[i + 2];
        const SceneVertex& v0 = verts[i0];
        const SceneVertex& v1 = verts[i1];
        const SceneVertex& v2 = verts[i2];

        glm::vec3 edge1 = v1.pos - v0.pos;
        glm::vec3 edge2 = v2.pos - v0.pos;
        glm::vec2 duv1 = v1.uv - v0.uv;
        glm::vec2 duv2 = v2.uv - v0.uv;

        float det = duv1.x * duv2.y - duv2.x * duv1.y;
        if (fabsf(det) < 1e-8f) continue;
        float r = 1.0f / det;

        glm::vec3 tangent = (edge1 * duv2.y - edge2 * duv1.y) * r;
        glm::vec3 bitangent = (edge2 * duv1.x - edge1 * duv2.x) * r;

        tan[i0] += tangent; tan[i1] += tangent; tan[i2] += tangent;
        bitan[i0] += bitangent; bitan[i1] += bitangent; bitan[i2] += bitangent;
    }

    for (size_t i = 0; i < verts.size(); ++i) {
        glm::vec3 n = verts[i].normal;
        glm::vec3 t = tan[i];
        if (glm::length(t) < 1e-8f) {
            t = fabsf(n.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        }
        t = glm::normalize(t - n * glm::dot(n, t));
        float w = glm::dot(glm::cross(n, t), bitan[i]) < 0.0f ? -1.0f : 1.0f;
        verts[i].tangent = glm::vec4(t, w);
    }
}

int ImageIndex(const cgltf_data* data, const cgltf_image* image)
{
    if (!image) return -1;
    return (int)(image - data->images);
}

agfxTexture* CreateSolidTexture(agfxDevice* device, AgfxUploader& uploader, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    uint8_t pixels[4] = {r, g, b, a};

    agfxTextureCreateInfo textureCreateInfo = {};
    textureCreateInfo.type = AGFX_TEXTURE_TYPE_2D;
    textureCreateInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    textureCreateInfo.usage = AGFX_TEXTURE_USAGE_SAMPLED;
    textureCreateInfo.width = 1;
    textureCreateInfo.height = 1;
    textureCreateInfo.depthOrArrayLayers = 1;
    textureCreateInfo.mipLevels = 1;
    agfxTexture* texture = agfxTextureCreate(device, &textureCreateInfo);

    agfxTextureRegion region = {};
    region.width = 1;
    region.height = 1;
    region.depth = 1;
    uploader.UploadTexture(device, texture, &region, 0, 0, pixels, 4, 4, 4);

    return texture;
}

} // namespace

bool GltfScene::Load(agfxDevice* device, agfxCommandQueue* queue, const char* path)
{
    m_device = device;

    AgfxUploader uploader(device, queue);

    cgltf_options options = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path, &data) != cgltf_result_success) {
        std::cerr << "Failed to parse glTF: " << path << std::endl;
        return false;
    }
    if (cgltf_load_buffers(&options, data, path) != cgltf_result_success) {
        std::cerr << "Failed to load glTF buffers: " << path << std::endl;
        cgltf_free(data);
        return false;
    }

    std::string dir = DirectoryOf(path);

    struct PendingMipGen {
        agfxTexture* texture;
        uint32_t width;
        uint32_t height;
        uint32_t mipLevels;
    };
    std::vector<PendingMipGen> pendingMipGens;

    // Load every image referenced by the asset.
    textures.resize(data->images_count);
    for (size_t i = 0; i < data->images_count; ++i) {
        cgltf_image* image = &data->images[i];
        if (!image->uri) {
            std::cerr << "Skipping embedded/data-uri image " << i << " (unsupported)" << std::endl;
            continue;
        }

        std::string fullPath = dir + "/" + image->uri;
        int w, h, channels;
        stbi_uc* pixels = stbi_load(fullPath.c_str(), &w, &h, &channels, 4);
        if (!pixels) {
            std::cerr << "Failed to load texture: " << fullPath << std::endl;
            continue;
        }

        uint32_t mipLevels = (uint32_t)std::floor(std::log2((double)std::max(w, h))) + 1;

        agfxTextureCreateInfo textureCreateInfo = {};
        textureCreateInfo.type = AGFX_TEXTURE_TYPE_2D;
        textureCreateInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
        textureCreateInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_SAMPLED | AGFX_TEXTURE_USAGE_STORAGE);
        textureCreateInfo.width = (uint32_t)w;
        textureCreateInfo.height = (uint32_t)h;
        textureCreateInfo.depthOrArrayLayers = 1;
        textureCreateInfo.mipLevels = mipLevels;
        agfxTexture* texture = agfxTextureCreate(device, &textureCreateInfo);

        agfxTextureRegion region = {};
        region.width = (uint32_t)w;
        region.height = (uint32_t)h;
        region.depth = 1;
        uint32_t bytesPerRow = (uint32_t)w * 4;
        uint32_t dataSize = bytesPerRow * (uint32_t)h;
        uploader.UploadTexture(device, texture, &region, 0, 0, pixels, dataSize, bytesPerRow, dataSize);
        stbi_image_free(pixels);

        agfxTextureViewCreateInfo viewCreateInfo = {};
        viewCreateInfo.texture = texture;
        viewCreateInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
        viewCreateInfo.type = AGFX_TEXTURE_TYPE_2D;
        viewCreateInfo.mipLevelCount = mipLevels;
        viewCreateInfo.arrayLayerCount = 1;
        agfxTextureView* view = agfxTextureViewCreate(device, &viewCreateInfo);

        textures[i].texture = texture;
        textures[i].view = view;
        textures[i].handle = (uint32_t)agfxTextureViewGetHandle(view);

        pendingMipGens.push_back({texture, (uint32_t)w, (uint32_t)h, mipLevels});
    }

    // Fallback textures (white albedo, flat normal) for materials missing a map.
    agfxTexture* whiteTex = CreateSolidTexture(device, uploader, 255, 255, 255, 255);
    agfxTextureViewCreateInfo whiteViewInfo = {};
    whiteViewInfo.texture = whiteTex;
    whiteViewInfo.format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    whiteViewInfo.type = AGFX_TEXTURE_TYPE_2D;
    whiteViewInfo.mipLevelCount = 1;
    whiteViewInfo.arrayLayerCount = 1;
    agfxTextureView* whiteView = agfxTextureViewCreate(device, &whiteViewInfo);
    int whiteTexIndex = (int)textures.size();
    textures.push_back({whiteTex, whiteView, (uint32_t)agfxTextureViewGetHandle(whiteView)});

    agfxTexture* flatNormalTex = CreateSolidTexture(device, uploader, 128, 128, 255, 255);
    agfxTextureViewCreateInfo flatViewInfo = whiteViewInfo;
    flatViewInfo.texture = flatNormalTex;
    agfxTextureView* flatNormalView = agfxTextureViewCreate(device, &flatViewInfo);
    int flatNormalTexIndex = (int)textures.size();
    textures.push_back({flatNormalTex, flatNormalView, (uint32_t)agfxTextureViewGetHandle(flatNormalView)});

    // Materials.
    materials.resize(data->materials_count);
    for (size_t i = 0; i < data->materials_count; ++i) {
        cgltf_material* mat = &data->materials[i];
        SceneMaterial& sceneMat = materials[i];
        sceneMat.doubleSided = mat->double_sided != 0;

        sceneMat.albedoTexIndex = whiteTexIndex;
        if (mat->has_pbr_metallic_roughness && mat->pbr_metallic_roughness.base_color_texture.texture) {
            int idx = ImageIndex(data, mat->pbr_metallic_roughness.base_color_texture.texture->image);
            if (idx >= 0 && textures[idx].texture) sceneMat.albedoTexIndex = idx;
        }

        sceneMat.normalTexIndex = flatNormalTexIndex;
        if (mat->normal_texture.texture) {
            int idx = ImageIndex(data, mat->normal_texture.texture->image);
            if (idx >= 0 && textures[idx].texture) sceneMat.normalTexIndex = idx;
        }

        sceneMat.metallicRoughnessTexIndex = whiteTexIndex;
        if (mat->has_pbr_metallic_roughness) {
            sceneMat.metallicFactor = mat->pbr_metallic_roughness.metallic_factor;
            sceneMat.roughnessFactor = mat->pbr_metallic_roughness.roughness_factor;
            if (mat->pbr_metallic_roughness.metallic_roughness_texture.texture) {
                int idx = ImageIndex(data, mat->pbr_metallic_roughness.metallic_roughness_texture.texture->image);
                if (idx >= 0 && textures[idx].texture) sceneMat.metallicRoughnessTexIndex = idx;
            }
        }
    }

    // Geometry: walk every node, bake world transform, merge primitives into one global vertex/index array.
    std::vector<SceneVertex> allVertices;
    std::vector<uint32_t> allIndices;

    for (size_t nodeIdx = 0; nodeIdx < data->nodes_count; ++nodeIdx) {
        cgltf_node* node = &data->nodes[nodeIdx];
        if (!node->mesh) continue;

        float worldMatrixRaw[16];
        cgltf_node_transform_world(node, worldMatrixRaw);
        glm::mat4 worldMatrix = glm::make_mat4(worldMatrixRaw);

        for (size_t primIdx = 0; primIdx < node->mesh->primitives_count; ++primIdx) {
            cgltf_primitive* prim = &node->mesh->primitives[primIdx];
            if (prim->type != cgltf_primitive_type_triangles) continue;

            const cgltf_accessor* posAccessor = nullptr;
            const cgltf_accessor* normalAccessor = nullptr;
            const cgltf_accessor* uvAccessor = nullptr;
            const cgltf_accessor* tangentAccessor = nullptr;

            for (size_t a = 0; a < prim->attributes_count; ++a) {
                cgltf_attribute* attr = &prim->attributes[a];
                if (attr->type == cgltf_attribute_type_position) posAccessor = attr->data;
                else if (attr->type == cgltf_attribute_type_normal) normalAccessor = attr->data;
                else if (attr->type == cgltf_attribute_type_texcoord && attr->index == 0) uvAccessor = attr->data;
                else if (attr->type == cgltf_attribute_type_tangent) tangentAccessor = attr->data;
            }
            if (!posAccessor) continue;

            size_t vertexCount = posAccessor->count;
            uint32_t vertexOffset = (uint32_t)allVertices.size();

            std::vector<SceneVertex> localVerts(vertexCount);
            for (size_t v = 0; v < vertexCount; ++v) {
                localVerts[v].pos = ReadFloat3(posAccessor, v);
                localVerts[v].normal = normalAccessor ? glm::normalize(ReadFloat3(normalAccessor, v)) : glm::vec3(0, 1, 0);
                localVerts[v].uv = uvAccessor ? ReadFloat2(uvAccessor, v) : glm::vec2(0.0f);
                localVerts[v].tangent = tangentAccessor ? ReadFloat4(tangentAccessor, v) : glm::vec4(1, 0, 0, 1);
            }

            std::vector<uint32_t> localIndices;
            if (prim->indices) {
                localIndices.resize(prim->indices->count);
                for (size_t idx = 0; idx < prim->indices->count; ++idx) {
                    localIndices[idx] = (uint32_t)cgltf_accessor_read_index(prim->indices, idx);
                }
            } else {
                localIndices.resize(vertexCount);
                for (size_t idx = 0; idx < vertexCount; ++idx) localIndices[idx] = (uint32_t)idx;
            }

            if (!tangentAccessor) {
                ComputeTangents(localVerts, localIndices);
            }

            uint32_t indexOffset = (uint32_t)allIndices.size();
            allVertices.insert(allVertices.end(), localVerts.begin(), localVerts.end());
            allIndices.insert(allIndices.end(), localIndices.begin(), localIndices.end());

            ScenePrimitive scenePrim = {};
            scenePrim.vertexOffset = vertexOffset;
            scenePrim.vertexCount = (uint32_t)vertexCount;
            scenePrim.indexOffset = indexOffset;
            scenePrim.indexCount = (uint32_t)localIndices.size();
            scenePrim.worldMatrix = worldMatrix;
            scenePrim.materialIndex = prim->material ? (int)(prim->material - data->materials) : -1;
            primitives.push_back(scenePrim);
        }
    }

    cgltf_free(data);

    if (allVertices.empty()) {
        std::cerr << "glTF scene has no geometry: " << path << std::endl;
        return false;
    }

    // Upload combined vertex/index buffers via staging (GPU-only destination, portable to
    // non-UMA backends where default-heap resources aren't CPU-mappable).
    agfxBufferCreateInfo vbInfo = {};
    vbInfo.size = allVertices.size() * sizeof(SceneVertex);
    vbInfo.stride = sizeof(SceneVertex);
    vbInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
    vbInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    vertexBuffer = agfxBufferCreate(device, &vbInfo);
    agfxBufferSetName(vertexBuffer, "Scene Vertex Buffer");
    uploader.UploadBuffer(device, vertexBuffer, 0, allVertices.data(), vbInfo.size);

    agfxBufferViewCreateInfo vbViewInfo = {};
    vbViewInfo.buffer = vertexBuffer;
    vbViewInfo.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
    vertexBufferView = agfxBufferViewCreate(device, &vbViewInfo);

    agfxBufferCreateInfo ibInfo = {};
    ibInfo.size = allIndices.size() * sizeof(uint32_t);
    ibInfo.stride = sizeof(uint32_t);
    ibInfo.usage = AGFX_BUFFER_USAGE_INDEX;
    ibInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    indexBuffer = agfxBufferCreate(device, &ibInfo);
    agfxBufferSetName(indexBuffer, "Scene Index Buffer");
    uploader.UploadBuffer(device, indexBuffer, 0, allIndices.data(), ibInfo.size);

    agfxSamplerCreateInfo samplerInfo = {};
    samplerInfo.filter = AGFX_SAMPLER_FILTER_LINEAR;
    samplerInfo.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
    samplerInfo.maxLod = 1000.0f; // otherwise lodMaxClamp defaults to 0 and clips sampling to mip 0
    defaultSampler = agfxSamplerCreate(device, &samplerInfo);

    uploader.Flush(device);

    if (!pendingMipGens.empty()) {
        AgfxMipGen mipGen(device);

        agfxCommandBuffer* mipCmdBuffer = agfxCommandBufferCreate(device, queue);
        agfxFence* mipFence = agfxFenceCreate(device);

        std::vector<agfxTextureView*> mipGenViews;

        agfxCommandBufferReset(mipCmdBuffer);
        agfxCommandBufferBegin(mipCmdBuffer);
        for (const PendingMipGen& pending : pendingMipGens) {
            mipGen.Generate(device, mipCmdBuffer, pending.texture, pending.width, pending.height, pending.mipLevels, mipGenViews);
        }
        agfxCommandBufferEnd(mipCmdBuffer);

        agfxCommandQueueSubmit(queue, &mipCmdBuffer, 1);
        agfxCommandQueueSignal(queue, mipFence, 1);
        agfxFenceWait(mipFence, 1, UINT64_MAX);

        // Only safe to free these bindless descriptor slots now that the GPU has
        // actually finished consuming them - see AgfxMipGen::Generate's contract.
        for (agfxTextureView* view : mipGenViews) {
            agfxTextureViewDestroy(device, view);
        }

        agfxFenceDestroy(device, mipFence);
        agfxCommandBufferDestroy(device, mipCmdBuffer);
    }

    agfxDeviceMakeResourcesResident(device);

    return true;
}

void GltfScene::Destroy(agfxDevice* device)
{
    for (SceneTexture& tex : textures) {
        if (tex.view) agfxTextureViewDestroy(device, tex.view);
        if (tex.texture) agfxTextureDestroy(device, tex.texture);
    }
    textures.clear();

    if (defaultSampler) agfxSamplerDestroy(device, defaultSampler);
    if (vertexBufferView) agfxBufferViewDestroy(device, vertexBufferView);
    if (vertexBuffer) agfxBufferDestroy(device, vertexBuffer);
    if (indexBuffer) agfxBufferDestroy(device, indexBuffer);
}
