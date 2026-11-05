/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 18:12:14
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx_shader_compiler.h"
#include "dxc/WinAdapter.h"
#include "dxc/dxcapi.h"
#include "spirv_reflect/spirv_reflect.h"

#include <dlfcn.h>
#include <stdio.h>
#include <vector>
#include <string>

struct DlHandleGuard {
    void* handle = nullptr;
    ~DlHandleGuard() { if (handle) dlclose(handle); }
};

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
#if defined(__linux__) && defined(DLOPEN)
    const char* libPath = (options->dxCompilerPath && options->dxCompilerPath[0])
        ? options->dxCompilerPath
        : "data/dlls/libdxcompiler.so";

    void* libHandle = dlopen(libPath, RTLD_NOW | RTLD_LOCAL);
    if (!libHandle) {
        fprintf(stderr, "Failed to dlopen DXC library '%s': %s\n", libPath, dlerror());
        return;
    }

    dlerror();
    DxcCreateInstanceProc DxcCreateInstance = (DxcCreateInstanceProc)dlsym(libHandle, "DxcCreateInstance");
    const char* symError = dlerror();
    if (symError || !DxcCreateInstance) {
        fprintf(stderr, "Failed to dlsym DxcCreateInstance: %s\n", symError ? symError : "unknown error");
        dlclose(libHandle);
        return;
    }

    DlHandleGuard libGuard{libHandle};
#endif

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
    compileArgs.push_back(L"-DAGFX_VULKAN");
    compileArgs.push_back(L"-spirv");
    compileArgs.push_back(L"-fspv-extension=SPV_EXT_mesh_shader");
    compileArgs.push_back(L"-fspv-extension=SPV_EXT_descriptor_indexing");
    compileArgs.push_back(L"-fspv-extension=SPV_KHR_ray_query");
    compileArgs.push_back(L"-fspv-extension=SPV_KHR_shader_draw_parameters");
    compileArgs.push_back(L"-fspv-target-env=vulkan1.3");
    compileArgs.push_back(L"-fvk-use-scalar-layout");
    // D3D12's SV_InstanceID is 0-based per draw regardless of firstInstance; Vulkan's InstanceIndex
    // builtin includes it. This makes DXC emit InstanceIndex - BaseInstance so shaders keep the
    // D3D12 contract (the shared test goldens and every AGFX shader assume it).
    compileArgs.push_back(L"-fvk-support-nonzero-base-instance");
    compileArgs.push_back(L"-fvk-bind-resource-heap");
    compileArgs.push_back(L"0");
    compileArgs.push_back(L"0");
    compileArgs.push_back(L"-fvk-bind-sampler-heap");
    compileArgs.push_back(L"1");
    compileArgs.push_back(L"0");
    compileArgs.push_back(L"-I/home/alex/Code/agfx/data/shaders");
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
        SpvReflectShaderModule module = {};
        if (spvReflectCreateShaderModule(output->compiledSize, output->compiledCode, &module) == SPV_REFLECT_RESULT_SUCCESS) {
            const SpvReflectEntryPoint* entry = spvReflectGetEntryPoint(&module, options->entryPoint);
            if (entry) {
                switch (options->stage) {
                case agfxShaderStage::AGFX_SHADER_STAGE_COMPUTE:
                    output->tgSizeX = entry->local_size.x;
                    output->tgSizeY = entry->local_size.y;
                    output->tgSizeZ = entry->local_size.z;
                    break;
                case agfxShaderStage::AGFX_SHADER_STAGE_MESH:
                    output->meshSizeX = entry->local_size.x;
                    output->meshSizeY = entry->local_size.y;
                    output->meshSizeZ = entry->local_size.z;
                    break;
                case agfxShaderStage::AGFX_SHADER_STAGE_TASK:
                    output->taskSizeX = entry->local_size.x;
                    output->taskSizeY = entry->local_size.y;
                    output->taskSizeZ = entry->local_size.z;
                    break;
                default:
                    break;
                }
            }
            spvReflectDestroyShaderModule(&module);
        }
    }
}
