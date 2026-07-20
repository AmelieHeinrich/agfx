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

class GltfScene {
public:
    std::vector<ScenePrimitive> primitives;
    std::vector<SceneMaterial> materials;
    std::vector<SceneTexture> textures;

    agfxBuffer* vertexBuffer = nullptr;
    agfxBufferView* vertexBufferView = nullptr;
    agfxBuffer* indexBuffer = nullptr;
    agfxSampler* defaultSampler = nullptr;

    bool Load(agfxDevice* device, agfxCommandQueue* queue, const char* path);

    // Explicit teardown (kept separate from the destructor so main.cpp can control the exact
    // point in the shutdown sequence at which the scene's agfx resources are released).
    void Destroy(agfxDevice* device);

private:
    agfxDevice* m_device = nullptr;
};
