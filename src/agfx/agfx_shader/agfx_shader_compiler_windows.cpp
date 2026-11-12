/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 18:12:14
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx_shader_compiler.h"
#include <Windows.h>
#include <Unknwn.h>
#include <stdio.h>
#include <vector>
#include <string>
#include "dxc/dxcapi.h"
#include <d3d12shader.h>

template<typename T>
class DxcPtr {
    T* m_ptr = nullptr;
public:
    DxcPtr() = default;
    DxcPtr(const DxcPtr&) = delete;
    DxcPtr& operator=(const DxcPtr&) = delete;
    ~DxcPtr() { if (m_ptr) m_ptr->Release(); }
    T* operator->() const { return m_ptr; }
    explicit operator bool() const { return m_ptr != nullptr; }
    T* Get() const { return m_ptr; }
    T** GetAddressOf() { return &m_ptr; }
};

inline const char* ProfileFromType(agfxShaderStage type)
{
    switch (type)
    {
    case agfxShaderStage::AGFX_SHADER_STAGE_VERTEX:
        return "vs_6_6";
    case agfxShaderStage::AGFX_SHADER_STAGE_FRAGMENT:
        return "ps_6_6";
    case agfxShaderStage::AGFX_SHADER_STAGE_COMPUTE:
        return "cs_6_6";
    case agfxShaderStage::AGFX_SHADER_STAGE_MESH:
        return "ms_6_6";
    case agfxShaderStage::AGFX_SHADER_STAGE_TASK:
        return "as_6_6";
    default:
        return "";
    }
    return "";
}

void agfxCompileShader(agfxShaderCompilerOptions* options, agfxShaderCompilerResult* output)
{
    DxcPtr<IDxcUtils> pUtils;
    DxcPtr<IDxcCompiler3> pCompiler;
    DxcPtr<IDxcIncludeHandler> pIncludeHandler;
    DxcPtr<IDxcResult> pResult;
    DxcPtr<IDxcBlob> pShaderBlob;
    DxcPtr<IDxcBlobUtf8> pErrorsU8;

    wchar_t wideTarget[512] = {0};
    swprintf_s(wideTarget, 512, L"%hs", ProfileFromType(options->stage));

    wchar_t wideEntry[512] = {0};
    swprintf_s(wideEntry, 512, L"%hs", options->entryPoint);

    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(pUtils.GetAddressOf())))) {
        fprintf(stderr, "Failed to create DXC utils!");
        return;
    }
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(pCompiler.GetAddressOf())))) {
        fprintf(stderr, "Failed to create DXC compiler!");
        return;
    }
    if (FAILED(pUtils->CreateDefaultIncludeHandler(pIncludeHandler.GetAddressOf()))) {
        fprintf(stderr, "Failed to create include handler!");
        return;
    }

    DxcBuffer sourceBuffer = {};
    sourceBuffer.Ptr = options->sourceCode;
    sourceBuffer.Size = options->sourceCodeSize;
    sourceBuffer.Encoding = DXC_CP_UTF8;

    std::vector<std::wstring> wideDefines;
    std::vector<LPCWSTR> compileArgs = {};
    compileArgs.push_back(L"-E");
    compileArgs.push_back(wideEntry);
    compileArgs.push_back(L"-T");
    compileArgs.push_back(wideTarget);
    if (options->addDebugSymbols) {
        compileArgs.push_back(L"-Qembed_debug");
        compileArgs.push_back(L"-Zi");
    }
    for (int i = 0; i < options->definesCount; i++) {
        wideDefines.push_back(L"-D" + std::wstring(options->defines[i], options->defines[i] + strlen(options->defines[i])));
        compileArgs.push_back(wideDefines.back().c_str());
    }

    HRESULT hresult = pCompiler->Compile(&sourceBuffer, compileArgs.data(), (uint32_t)compileArgs.size(), pIncludeHandler.Get(), IID_PPV_ARGS(pResult.GetAddressOf()));
    if (FAILED(hresult)) {
        fprintf(stderr, "Failed to compile shader: 0x%08X\n", (unsigned int)hresult);
        return;
    }

    if (SUCCEEDED(pResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(pErrorsU8.GetAddressOf()), nullptr)) && pErrorsU8 && pErrorsU8->GetStringLength() > 0) {
        fprintf(stderr, "Shader compilation warnings/errors:\n%s\n", pErrorsU8->GetStringPointer());
    }

    HRESULT status = S_OK;
    if (FAILED(pResult->GetStatus(&status)) || FAILED(status)) {
        fprintf(stderr, "Shader compilation failed with status: 0x%08X\n", (unsigned int)status);
        return;
    }

    // Get result blob
    if (FAILED(pResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(pShaderBlob.GetAddressOf()), nullptr)) || !pShaderBlob) {
        fprintf(stderr, "Failed to get compiled shader blob!\n");
        return;
    }

    uint8_t* result = (uint8_t*)(options->allocate ? options->allocate : defaultAlloc)(pShaderBlob->GetBufferSize(), options->userData);
    memcpy(result, pShaderBlob->GetBufferPointer(), pShaderBlob->GetBufferSize());
    uint64_t bytecodeSize = pShaderBlob->GetBufferSize();

    output->compiledCode = result;
    output->compiledSize = bytecodeSize;

    // Get reflection if it's compute, mesh or task shader
    if (options->stage == agfxShaderStage::AGFX_SHADER_STAGE_COMPUTE ||
        options->stage == agfxShaderStage::AGFX_SHADER_STAGE_MESH ||
        options->stage == agfxShaderStage::AGFX_SHADER_STAGE_TASK) {
        DxcPtr<IDxcBlob> pReflectionBlob;
        if (SUCCEEDED(pResult->GetOutput(DXC_OUT_REFLECTION, IID_PPV_ARGS(pReflectionBlob.GetAddressOf()), nullptr)) && pReflectionBlob) {
            DxcBuffer reflectionBuffer = {};
            reflectionBuffer.Ptr = pReflectionBlob->GetBufferPointer();
            reflectionBuffer.Size = pReflectionBlob->GetBufferSize();
            reflectionBuffer.Encoding = 0;

            ID3D12ShaderReflection* pShaderReflection = nullptr;
            if (SUCCEEDED(pUtils->CreateReflection(&reflectionBuffer, IID_PPV_ARGS(&pShaderReflection))) && pShaderReflection) {
                uint32_t sizeX = 0, sizeY = 0, sizeZ = 0;
                pShaderReflection->GetThreadGroupSize(&sizeX, &sizeY, &sizeZ);

                switch (options->stage) {
                case agfxShaderStage::AGFX_SHADER_STAGE_COMPUTE:
                    output->tgSizeX = sizeX;
                    output->tgSizeY = sizeY;
                    output->tgSizeZ = sizeZ;
                    break;
                case agfxShaderStage::AGFX_SHADER_STAGE_MESH:
                    output->meshSizeX = sizeX;
                    output->meshSizeY = sizeY;
                    output->meshSizeZ = sizeZ;
                    break;
                case agfxShaderStage::AGFX_SHADER_STAGE_TASK:
                    output->taskSizeX = sizeX;
                    output->taskSizeY = sizeY;
                    output->taskSizeZ = sizeZ;
                    break;
                default:
                    break;
                }

                pShaderReflection->Release();
            }
        }
    }

    return;
}
