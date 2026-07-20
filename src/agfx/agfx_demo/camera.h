/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <glm/glm.hpp>

struct Camera {
    glm::vec3 position = glm::vec3(0.0f, 2.0f, 0.0f);
    float yaw = 0.0f;
    float pitch = 0.0f;

    float fovYRadians = glm::radians(60.0f);
    float aspect = 16.0f / 9.0f;
    float nearZ = 0.05f;
    float farZ = 500.0f;

    glm::vec3 Forward() const;
    glm::vec3 Right() const;
    glm::mat4 GetView() const;
    glm::mat4 GetProj() const;

    void Update(const bool* keyState, float mouseDx, float mouseDy, float dt);
};

// Extracts the 6 frustum planes (left, right, bottom, top, near, far) from a view-projection
// matrix, Gribb-Hartmann style. Each plane is (normal.xyz, distance), with the interior of the
// frustum on the positive side. Used by the GPU-driven culling compute pass.
void ExtractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 planes[6]);
