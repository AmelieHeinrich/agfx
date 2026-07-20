/**
 * @ Author: Amélie Heinrich
 * @ Create Time: 2026-07-20
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include <agfx_shader/agfx_shader_compiler.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::string ReadFile(const char* path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }

    std::streamsize size = stream.tellg();
    if (size < 0) {
        return {};
    }

    stream.seekg(0, std::ios::beg);

    std::string data((size_t)size, '\0');
    if (size > 0) {
        stream.read(&data[0], size);
    }
    return data;
}

static bool WriteFile(const char* path, const uint8_t* data, uint32_t size)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }

    stream.write((const char*)data, size);
    return stream.good();
}

static std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

static bool ParseStage(const std::string& value, agfxShaderStage* stage)
{
    std::string lowered = ToLower(value);
    if (lowered == "vertex" || lowered == "vs") {
        *stage = AGFX_SHADER_STAGE_VERTEX;
        return true;
    }
    if (lowered == "fragment" || lowered == "pixel" || lowered == "ps") {
        *stage = AGFX_SHADER_STAGE_FRAGMENT;
        return true;
    }
    if (lowered == "compute" || lowered == "cs") {
        *stage = AGFX_SHADER_STAGE_COMPUTE;
        return true;
    }
    if (lowered == "task" || lowered == "as") {
        *stage = AGFX_SHADER_STAGE_TASK;
        return true;
    }
    if (lowered == "mesh" || lowered == "ms") {
        *stage = AGFX_SHADER_STAGE_MESH;
        return true;
    }

    return false;
}

static void PrintUsage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " --input <path> --output <path> --stage <vertex|fragment|compute|task|mesh>"
              << " [--entry <name>] [--debug] [--define <NAME[=VALUE]> ...]"
              << std::endl;
}

int main(int argc, char** argv)
{
    std::string inputPath;
    std::string outputPath;
    std::string entryPoint = "main";
    std::string stageName;
    bool addDebugSymbols = false;
    std::vector<std::string> definesStorage;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" || arg == "-i") {
            if (++i >= argc) {
                PrintUsage(argv[0]);
                return 1;
            }
            inputPath = argv[i];
        } else if (arg == "--output" || arg == "-o") {
            if (++i >= argc) {
                PrintUsage(argv[0]);
                return 1;
            }
            outputPath = argv[i];
        } else if (arg == "--stage" || arg == "-s") {
            if (++i >= argc) {
                PrintUsage(argv[0]);
                return 1;
            }
            stageName = argv[i];
        } else if (arg == "--entry" || arg == "-e") {
            if (++i >= argc) {
                PrintUsage(argv[0]);
                return 1;
            }
            entryPoint = argv[i];
        } else if (arg == "--define" || arg == "-D") {
            if (++i >= argc) {
                PrintUsage(argv[0]);
                return 1;
            }
            definesStorage.emplace_back(argv[i]);
        } else if (arg == "--debug") {
            addDebugSymbols = true;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else if (inputPath.empty()) {
            inputPath = arg;
        } else if (outputPath.empty()) {
            outputPath = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (inputPath.empty() || outputPath.empty() || stageName.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }

    agfxShaderStage stage = AGFX_SHADER_STAGE_VERTEX;
    if (!ParseStage(stageName, &stage)) {
        std::cerr << "Unknown shader stage: " << stageName << std::endl;
        return 1;
    }

    std::string source = ReadFile(inputPath.c_str());
    if (source.empty()) {
        std::cerr << "Failed to read shader source: " << inputPath << std::endl;
        return 1;
    }

    std::vector<char*> definePointers;
    definePointers.reserve(definesStorage.size());
    for (std::string& define : definesStorage) {
        definePointers.push_back(const_cast<char*>(define.c_str()));
    }

    agfxShaderCompilerOptions options = {};
    options.stage = stage;
    std::snprintf(options.entryPoint, sizeof(options.entryPoint), "%s", entryPoint.c_str());
    options.definesCount = (uint32_t)definePointers.size();
    for (size_t i = 0; i < definePointers.size(); ++i) {
        options.defines[i] = definePointers[i];
    }
    options.sourceCode = source.empty() ? nullptr : const_cast<char*>(source.data());
    options.sourceCodeSize = (uint32_t)source.size();
    options.addDebugSymbols = addDebugSymbols ? 1 : 0;

    agfxShaderCompilerResult result = {};
    agfxCompileShader(&options, &result);
    if (result.compiledCode == nullptr || result.compiledSize == 0) {
        std::cerr << "Shader compilation failed" << std::endl;
        return 1;
    }

    if (!WriteFile(outputPath.c_str(), result.compiledCode, result.compiledSize)) {
        std::cerr << "Failed to write compiled shader: " << outputPath << std::endl;
        free(result.compiledCode);
        return 1;
    }

    std::cout << "Compiled " << inputPath << " -> " << outputPath << " (" << result.compiledSize << " bytes)" << std::endl;
    free(result.compiledCode);
    return 0;
}