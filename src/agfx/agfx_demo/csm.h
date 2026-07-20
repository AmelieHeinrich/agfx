/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <glm/glm.hpp>
#include <stdint.h>

#include "camera.h"

static const uint32_t kMaxCascades = 4;

struct CascadeInfo {
    glm::mat4 viewProj = glm::mat4(1.0f);
    float splitFar = 0.0f;       // view-space far distance of this cascade
    float texelWorldSize = 0.0f; // world units per shadow-map texel
};

struct CSMSettings {
    uint32_t cascadeCount = 4;
    float splitLambda = 0.75f;
    float shadowDistance = 100.0f;
    uint32_t shadowMapResolution = 2048;

    float lightSizeUV = 0.005f;
    float pcssMaxPenumbraUV = 0.002f;
    float depthBiasConstant = 0.001f;
    bool visualizeCascades = false;

    // Computes cascadeCount split cascades (view-space split depths + texel-snapped,
    // stable light-space ortho view-proj matrices) for the given camera/light direction.
    // outCascades must point to at least kMaxCascades entries.
    void ComputeCascades(const Camera& camera, const glm::vec3& lightDirNormalized, CascadeInfo* outCascades) const;
};
