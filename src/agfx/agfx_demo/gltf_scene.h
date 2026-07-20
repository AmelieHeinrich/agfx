/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>
#include <glm/glm.hpp>
#include <vector>

struct SceneVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec4 tangent; // xyz = tangent, w = handedness
    glm::vec2 uv;
};

struct SceneMaterial {
    int albedoTexIndex = -1;
    int normalTexIndex = -1;
    int metallicRoughnessTexIndex = -1;
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    bool doubleSided = false;
};

struct ScenePrimitive {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    glm::mat4 worldMatrix = glm::mat4(1.0f);
    int materialIndex = -1;
};

struct SceneTexture {
    agfxTexture* texture = nullptr;
    agfxTextureView* view = nullptr;
    uint32_t handle = 0;
};

// One entry per ScenePrimitive - the "GPUScene" instance array consumed by the
// raytraced reflections pass to resolve material/geometry data at a ray hit
// (indexed by RayQuery::CommittedInstanceID(), which equals the TLAS instance's
// userID, which equals the primitive's index into this array).
struct GPUSceneInstance {
    glm::mat4 worldMatrix = glm::mat4(1.0f);
    uint32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    uint32_t albedoTex = 0;
    uint32_t normalTex = 0;
    uint32_t metallicRoughnessTex = 0;
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float _pad0 = 0.0f;
};

class GltfScene {
public:
    std::vector<ScenePrimitive> primitives;
    std::vector<SceneMaterial> materials;
    std::vector<SceneTexture> textures;

    agfxBuffer* vertexBuffer = nullptr;
    agfxBufferView* vertexBufferView = nullptr;
    agfxBuffer* indexBuffer = nullptr;
    agfxBufferView* indexBufferView = nullptr;
    agfxSampler* defaultSampler = nullptr;

    // Raytracing acceleration structures - one BLAS per primitive, one TLAS for
    // the whole scene (static, built once at load time).
    std::vector<agfxAccelerationStructure*> blas;
    agfxAccelerationStructure* tlas = nullptr;
    uint32_t tlasHandle = 0;

    // GPUScene: flat per-instance array, uploaded once at load time.
    agfxBuffer* gpuSceneBuffer = nullptr;
    agfxBufferView* gpuSceneBufferView = nullptr;

    bool Load(agfxDevice* device, agfxCommandQueue* queue, const char* path);

    // Explicit teardown (kept separate from the destructor so main.cpp can control the exact
    // point in the shutdown sequence at which the scene's agfx resources are released).
    void Destroy(agfxDevice* device);

private:
    agfxDevice* m_device = nullptr;
};
