/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 12:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "demo_file_utils.h"

#include <fstream>

std::string ReadFile(const char* path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    std::string data((size_t)size, '\0');
    stream.read(&data[0], size);
    return data;
}
