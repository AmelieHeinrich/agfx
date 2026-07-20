/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "csm.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace {

// Unprojects the 8 NDC corners of the [splitNear, splitFar] sub-frustum into world space,
// then returns the bounding sphere (center + radius) of those corners. A sphere is used
// (rather than an AABB) because its radius is invariant to camera yaw/pitch for a fixed
// (splitNear, splitFar, fov, aspect) — required for stable texel snapping below.
void ComputeFrustumBoundingSphere(const Camera& camera, float splitNear, float splitFar, glm::vec3* outCenter, float* outRadius)
{
    glm::mat4 subProj = glm::perspectiveRH_ZO(camera.fovYRadians, camera.aspect, splitNear, splitFar);
    glm::mat4 invViewProj = glm::inverse(subProj * camera.GetView());

    glm::vec3 corners[8];
    uint32_t idx = 0;
    for (int z = 0; z <= 1; ++z) {
        for (int y = -1; y <= 1; y += 2) {
            for (int x = -1; x <= 1; x += 2) {
                glm::vec4 ndc((float)x, (float)y, (float)z, 1.0f);
                glm::vec4 world = invViewProj * ndc;
                corners[idx++] = glm::vec3(world) / world.w;
            }
        }
    }

    glm::vec3 nearCenter = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
    glm::vec3 farCenter = (corners[4] + corners[5] + corners[6] + corners[7]) * 0.25f;
    glm::vec3 center = (nearCenter + farCenter) * 0.5f;

    float radius = 0.0f;
    for (uint32_t i = 0; i < 8; ++i) {
        radius = std::max(radius, glm::length(corners[i] - center));
    }

    *outCenter = center;
    *outRadius = radius;
}

} // namespace

void CSMSettings::ComputeCascades(const Camera& camera, const glm::vec3& lightDir, CascadeInfo* outCascades) const
{
    const CSMSettings& settings = *this;
    uint32_t cascadeCount = std::min(settings.cascadeCount, kMaxCascades);

    float nearZ = camera.nearZ;
    float farZ = settings.shadowDistance;

    float splitNear = nearZ;
    for (uint32_t i = 0; i < cascadeCount; ++i) {
        float p = (float)(i + 1) / (float)cascadeCount;
        float logSplit = nearZ * std::pow(farZ / nearZ, p);
        float uniSplit = nearZ + (farZ - nearZ) * p;
        float splitFar = glm::mix(uniSplit, logSplit, settings.splitLambda);

        glm::vec3 sphereCenter;
        float sphereRadius;
        ComputeFrustumBoundingSphere(camera, splitNear, splitFar, &sphereCenter, &sphereRadius);

        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(lightDir, up)) > 0.99f) {
            up = glm::vec3(0.0f, 0.0f, 1.0f);
        }

        // Orientation-only view matrix (fixed origin) used to move the sphere center into
        // light space, snap it to whole shadow-map texels, then move it back to world space.
        // This keeps the shadow-map texel grid stable in world space across camera motion,
        // which is what removes shimmering on shadow edges as the camera translates/rotates.
        glm::mat4 lightViewOrientation = glm::lookAtRH(glm::vec3(0.0f), -lightDir, up);
        float texelWorldSize = (sphereRadius * 2.0f) / (float)settings.shadowMapResolution;

        glm::vec3 centerLS = glm::vec3(lightViewOrientation * glm::vec4(sphereCenter, 1.0f));
        centerLS.x = std::floor(centerLS.x / texelWorldSize) * texelWorldSize;
        centerLS.y = std::floor(centerLS.y / texelWorldSize) * texelWorldSize;
        glm::vec3 snappedCenter = glm::vec3(glm::inverse(lightViewOrientation) * glm::vec4(centerLS, 1.0f));

        float zPad = sphereRadius * 2.0f;
        glm::vec3 eye = snappedCenter - lightDir * (sphereRadius + zPad);
        glm::mat4 lightView = glm::lookAtRH(eye, snappedCenter, up);
        glm::mat4 lightProj = glm::orthoRH_ZO(-sphereRadius, sphereRadius, -sphereRadius, sphereRadius, 0.0f, sphereRadius * 2.0f + zPad);

        outCascades[i].viewProj = lightProj * lightView;
        outCascades[i].splitFar = splitFar;
        outCascades[i].texelWorldSize = texelWorldSize;

        splitNear = splitFar;
    }
}
