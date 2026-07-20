/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <SDL3/SDL_scancode.h>

glm::vec3 Camera::Forward() const
{
    return glm::vec3(
        cosf(pitch) * sinf(yaw),
        sinf(pitch),
        cosf(pitch) * cosf(yaw)
    );
}

glm::vec3 Camera::Right() const
{
    return glm::normalize(glm::cross(Forward(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

glm::mat4 Camera::GetView() const
{
    return glm::lookAtRH(position, position + Forward(), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::GetProj() const
{
    return glm::perspectiveRH_ZO(fovYRadians, aspect, nearZ, farZ);
}

void Camera::Update(const bool* keyState, float mouseDx, float mouseDy, float dt)
{
    const float mouseSensitivity = 0.0025f;
    yaw -= mouseDx * mouseSensitivity;
    pitch -= mouseDy * mouseSensitivity;

    const float pitchLimit = glm::radians(89.0f);
    if (pitch > pitchLimit) pitch = pitchLimit;
    if (pitch < -pitchLimit) pitch = -pitchLimit;

    const float moveSpeed = 5.0f;
    glm::vec3 forward = Forward();
    glm::vec3 right = Right();

    glm::vec3 move(0.0f);
    if (keyState[SDL_SCANCODE_W]) move += forward;
    if (keyState[SDL_SCANCODE_S]) move -= forward;
    if (keyState[SDL_SCANCODE_D]) move += right;
    if (keyState[SDL_SCANCODE_A]) move -= right;
    if (keyState[SDL_SCANCODE_E] || keyState[SDL_SCANCODE_SPACE]) move += glm::vec3(0.0f, 1.0f, 0.0f);
    if (keyState[SDL_SCANCODE_Q] || keyState[SDL_SCANCODE_LCTRL]) move -= glm::vec3(0.0f, 1.0f, 0.0f);

    if (glm::length(move) > 0.0f) {
        move = glm::normalize(move);
    }

    float speed = moveSpeed;
    if (keyState[SDL_SCANCODE_LSHIFT]) speed *= 3.0f;

    position += move * speed * dt;
}
