/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 18:12:14
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx_shader_compiler.h"
#include "dxc/WinAdapter.h"
#include "dxc/dxcapi.h"
#include "msc/metal_irconverter.h"

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
    // Create root signature
    IRRootParameter1 root_parameters[2] = {};
    root_parameters[0] = {
        .ParameterType = IRRootParameterType32BitConstants,
        .Constants = {
            .ShaderRegister = 0,
            .RegisterSpace = 0,
            .Num32BitValues = 128 / sizeof(uint32_t)
        },
        .ShaderVisibility = IRShaderVisibilityAll
    };
    root_parameters[1] = {
        .ParameterType = IRRootParameterType32BitConstants,
        .Constants = {
            .ShaderRegister = 1,
            .RegisterSpace = 0,
            .Num32BitValues = 1
        },
        .ShaderVisibility = IRShaderVisibilityAll
    };

    IRVersionedRootSignatureDescriptor root_sig_descriptor = {};
    root_sig_descriptor.version = IRRootSignatureVersion_1_1;
    root_sig_descriptor.desc_1_1.Flags = IRRootSignatureFlags(IRRootSignatureFlagSamplerHeapDirectlyIndexed | IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed);
    root_sig_descriptor.desc_1_1.pParameters = root_parameters;
    root_sig_descriptor.desc_1_1.NumParameters = 2;

    IRError* root_sig_error = nullptr;
    IRRootSignature* root_sig = IRRootSignatureCreateFromDescriptor(&root_sig_descriptor, &root_sig_error);
    if (root_sig_error) {
        auto errorCode = IRErrorGetCode(root_sig_error);

        fprintf(stderr, "Failed to create root signature: %d\n", errorCode);
        IRErrorDestroy(root_sig_error);
        return;
    }

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
        IRRootSignatureDestroy(root_sig);
        return;
    }
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(pCompiler.GetAddressOf())))) {
        fprintf(stderr, "Failed to create DXC compiler!");
        IRRootSignatureDestroy(root_sig);
        return;
    }
    if (FAILED(pUtils->CreateDefaultIncludeHandler(pIncludeHandler.GetAddressOf()))) {
        fprintf(stderr, "Failed to create include handler!");
        IRRootSignatureDestroy(root_sig);
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
    compileArgs.push_back(L"-DAGFX_METAL");
    for (int i = 0; i < options->definesCount; i++) {
        wideDefines.push_back(L"-D" + std::wstring(options->defines[i], options->defines[i] + strlen(options->defines[i])));
        compileArgs.push_back(wideDefines.back().c_str());
    }

    HRESULT hresult = pCompiler->Compile(&sourceBuffer, compileArgs.data(), (uint32_t)compileArgs.size(), pIncludeHandler.Get(), IID_PPV_ARGS(pResult.GetAddressOf()));
    if (FAILED(hresult)) {
        fprintf(stderr, "Failed to compile shader: 0x%08X\n", hresult);
        IRRootSignatureDestroy(root_sig);
        return;
    }

    if (SUCCEEDED(pResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(pErrorsU8.GetAddressOf()), nullptr)) && pErrorsU8 && pErrorsU8->GetStringLength() > 0) {
        fprintf(stderr, "Shader compilation warnings/errors:\n%s\n", pErrorsU8->GetStringPointer());
    }

    HRESULT status = S_OK;
    if (FAILED(pResult->GetStatus(&status)) || FAILED(status)) {
        fprintf(stderr, "Shader compilation failed with status: 0x%08X\n", status);
        IRRootSignatureDestroy(root_sig);
        return;
    }

    // Get result blob
    if (FAILED(pResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(pShaderBlob.GetAddressOf()), nullptr)) || !pShaderBlob) {
        fprintf(stderr, "Failed to get compiled shader blob!\n");
        IRRootSignatureDestroy(root_sig);
        return;
    }

    // Now do MSC
    auto module = IRObjectCreateFromDXIL((const uint8_t*)pShaderBlob->GetBufferPointer(), (size_t)pShaderBlob->GetBufferSize(), IRBytecodeOwnershipNone);

    IRCompiler* compiler = IRCompilerCreate();
    IRCompilerSetEntryPointName(compiler, options->entryPoint);
    IRCompilerSetMinimumDeploymentTarget(compiler, IROperatingSystem_macOS, "16.0");
    IRCompilerSetGlobalRootSignature(compiler, root_sig);
    if (options->usePointTopology) {
        IRCompilerSetInputTopology(compiler, IRInputTopologyPoint);
    }
    IRCompilerSetValidationFlags(compiler, (IRCompilerValidationFlags)(IRCompilerValidationFlagValidateAllResourcesBound | IRCompilerValidationFlagValidateDXIL));

    IRError* compileError = nullptr;
    auto metalIR = IRCompilerAllocCompileAndLink(compiler, options->entryPoint, module, &compileError);
    if (compileError) {
        auto errorCode = IRErrorGetCode(compileError);

        fprintf(stderr, "Metal IR generation failed with code %u\n", errorCode);
        IRErrorDestroy(compileError);
    }
    if (!metalIR) {
        fprintf(stderr, "IRCompilerAllocCompileAndLink returned null\n");
        IRObjectDestroy(module);
        IRCompilerDestroy(compiler);
        IRRootSignatureDestroy(root_sig);
        return;
    }

    IRMetalLibBinary* pMetallib = IRMetalLibBinaryCreate();
    if (!IRObjectGetMetalLibBinary(metalIR, IRObjectGetMetalIRShaderStage(metalIR), pMetallib)) {
        fprintf(stderr, "Failed to get Metal lib binary from compiled shader\n");
        IRMetalLibBinaryDestroy(pMetallib);
        IRObjectDestroy(module);
        IRObjectDestroy(metalIR);
        IRCompilerDestroy(compiler);
        IRRootSignatureDestroy(root_sig);
        return;
    }
    uint64_t bytecodeSize = IRMetalLibGetBytecodeSize(pMetallib);

    uint8_t* result = (uint8_t*)malloc(bytecodeSize);
    if (!result) {
        fprintf(stderr, "Failed to allocate memory for shader bytecode\n");
        IRMetalLibBinaryDestroy(pMetallib);
        IRObjectDestroy(module);
        IRObjectDestroy(metalIR);
        IRCompilerDestroy(compiler);
        IRRootSignatureDestroy(root_sig);
        return;
    }
    IRMetalLibGetBytecode(pMetallib, result);
    output->compiledCode = result;
    output->compiledSize = bytecodeSize;

    // Get reflection if it's compute, mesh or task shader
    if (options->stage == agfxShaderStage::AGFX_SHADER_STAGE_COMPUTE) {
        IRShaderReflection* reflection = IRShaderReflectionCreate();
        IRObjectGetReflection(metalIR, IRShaderStageCompute, reflection);

        IRVersionedCSInfo csInfo = {};
        if (IRShaderReflectionCopyComputeInfo(reflection, IRReflectionVersion_1_0, &csInfo)) {
            output->tgSizeX = csInfo.info_1_0.tg_size[0];
            output->tgSizeY = csInfo.info_1_0.tg_size[1];
            output->tgSizeZ = csInfo.info_1_0.tg_size[2];
        }

        IRShaderReflectionDestroy(reflection);
    } else if (options->stage == agfxShaderStage::AGFX_SHADER_STAGE_MESH) {
        IRShaderReflection* reflection = IRShaderReflectionCreate();
        IRObjectGetReflection(metalIR, IRShaderStageMesh, reflection);

        IRVersionedMSInfo msInfo = {};
        if (IRShaderReflectionCopyMeshInfo(reflection, IRReflectionVersion_1_0, &msInfo)) {
            output->meshSizeX = msInfo.info_1_0.num_threads[0];
            output->meshSizeY = msInfo.info_1_0.num_threads[1];
            output->meshSizeZ = msInfo.info_1_0.num_threads[2];
        }

        IRShaderReflectionDestroy(reflection);
    } else if (options->stage == agfxShaderStage::AGFX_SHADER_STAGE_TASK) {
        IRShaderReflection* reflection = IRShaderReflectionCreate();
        IRObjectGetReflection(metalIR, IRShaderStageAmplification, reflection);

        IRVersionedASInfo asInfo = {};
        if (IRShaderReflectionCopyAmplificationInfo(reflection, IRReflectionVersion_1_0, &asInfo)) {
            output->taskSizeX = asInfo.info_1_0.num_threads[0];
            output->taskSizeY = asInfo.info_1_0.num_threads[1];
            output->taskSizeZ = asInfo.info_1_0.num_threads[2];
        }

        IRShaderReflectionDestroy(reflection);
    }

    IRMetalLibBinaryDestroy(pMetallib);
    IRObjectDestroy(module);
    IRObjectDestroy(metalIR);
    IRCompilerDestroy(compiler);
    IRRootSignatureDestroy(root_sig);
    return;
}
