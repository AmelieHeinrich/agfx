/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 12:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <stdint.h>
#include <string>

// Number of frames the demo keeps in flight (swapchain images / per-frame resource slots).
constexpr uint32_t kFramesInFlight = 3;

// Root directory every demo data path (models, shaders) is built from.
constexpr const char* kDataDir = "data/";

// Reads an entire file into memory as a binary blob.
std::string ReadFile(const char* path);
