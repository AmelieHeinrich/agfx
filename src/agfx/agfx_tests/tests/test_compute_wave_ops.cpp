/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "compute wave ops".
//
// Wave (subgroup) width is not something AGFX exposes or pins down -- Metal SIMD groups are commonly
// 32 lanes on Apple silicon, D3D12 warps are commonly 32 on NVIDIA and 64 on AMD -- so this test
// cannot assert a specific lane count without breaking portability. Instead it checks that
// WaveGetLaneCount, WaveGetLaneIndex, WaveActiveSum, WaveActiveMax, WaveIsFirstLane and
// WaveReadLaneFirst all agree with *each other* and with the launch geometry, whatever the wave size
// turns out to be. See data/shaders/tests/wave_ops.hlsl for exactly what each of the 64 threads
// writes.
//
// The cross-checks (VerifyWaveOps):
//  - WaveActiveSum(1) must equal WaveGetLaneCount() for every thread's own wave.
//  - WaveActiveMax(laneIndex) + 1 must equal WaveGetLaneCount() (the highest lane index is the
//    count minus one).
//  - WaveIsFirstLane() must agree with WaveGetLaneIndex() == 0.
//  - WaveReadLaneFirst(dispatchID) must equal the dispatch ID rounded down to a multiple of that
//    thread's own lane count -- true because the dispatch is a single, non-divergent 1D group of
//    exactly 64 threads, so waves are contiguous blocks assigned in dispatch-ID order with no
//    reordering. That is the assumption under every one of these checks, and it is what makes wave
//    membership computable from the host side at all without knowing the lane count in advance.
//
// A backend that stubbed any one of these intrinsics to a constant, or that let control flow (the
// bounds check on the last threads) desynchronize wave membership, fails at least one lane's
// cross-check; a backend that implemented them correctly cannot fail regardless of its actual wave
// width.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kThreadCount = 64; // Matches [numthreads(64,1,1)]; one dispatch, one group.
    constexpr const char* kGolden = "compute_wave_ops.bin";

    /// @brief Mirrors WaveOpsPushConstants in data/shaders/tests/wave_ops.hlsl.
    struct PushConstants
    {
        uint32_t destination = 0;
        uint32_t firstLaneIDBuffer = 0;
        uint32_t threadCount = kThreadCount;
        uint32_t padding0 = 0;
    };

    /// @brief One thread's packed record: laneCount, activeSum, activeMax+1, isFirstMatches.
    struct Record
    {
        uint32_t x, y, z, w;
    };

    agfxBufferCreateInfo RecordBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kThreadCount * sizeof(Record);
        info.stride = sizeof(Record);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferCreateInfo FirstLaneBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kThreadCount * sizeof(uint32_t);
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferViewCreateInfo BufferViewInfo(agfxBuffer* buffer)
    {
        agfxBufferViewCreateInfo info{};
        info.buffer = buffer;
        info.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        info.offset = 0;
        info.writeable = 1;
        return info;
    }

    agfxComputePipelineCreateInfo PipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "compute wave ops";
        info.computeShader = module;
        info.groupSizeX = kThreadCount;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }

    /// @brief The cross-checks described above. `records`/`firstLaneIDs` are kThreadCount entries.
    bool VerifyWaveOps(const std::vector<Record>& records, const std::vector<uint32_t>& firstLaneIDs,
                       std::string& outError)
    {
        if (records.size() != kThreadCount || firstLaneIDs.size() != kThreadCount) {
            outError = "unexpected readback size";
            return false;
        }

        for (uint32_t i = 0; i < kThreadCount; ++i) {
            const Record& r = records[i];
            if (r.x == 0) {
                outError = "thread " + std::to_string(i) + ": WaveGetLaneCount() returned 0";
                return false;
            }
            if (r.y != r.x) {
                outError = "thread " + std::to_string(i) +
                           ": WaveActiveSum(1) does not equal WaveGetLaneCount()";
                return false;
            }
            if (r.z != r.x) {
                outError = "thread " + std::to_string(i) +
                           ": WaveActiveMax(laneIndex) + 1 does not equal WaveGetLaneCount()";
                return false;
            }
            if (r.w != 1) {
                outError = "thread " + std::to_string(i) + ": WaveIsFirstLane() disagrees with lane index 0";
                return false;
            }

            const uint32_t expectedFirst = (i / r.x) * r.x;
            if (firstLaneIDs[i] != expectedFirst) {
                outError = "thread " + std::to_string(i) +
                           ": WaveReadLaneFirst() did not broadcast the wave's first dispatch ID";
                return false;
            }
        }
        return true;
    }

    std::vector<uint8_t> RecordsToBytes(const std::vector<Record>& records)
    {
        std::vector<uint8_t> bytes(records.size() * sizeof(Record));
        memcpy(bytes.data(), records.data(), bytes.size());
        return bytes;
    }

    bool RunC(const CompiledShader& shader, std::vector<Record>& outRecords,
              std::vector<uint32_t>& outFirstLaneIDs, std::string& outError)
    {
        GpuFixture gpu;
        if (!gpu.Valid()) {
            outError = "failed to create headless device";
            return false;
        }
        agfxDevice* device = gpu.Device();

        const agfxBufferCreateInfo recordInfo = RecordBufferInfo();
        const agfxBufferCreateInfo firstLaneInfo = FirstLaneBufferInfo();
        agfxBuffer* recordBuffer = agfxBufferCreate(device, &recordInfo);
        agfxBuffer* firstLaneBuffer = agfxBufferCreate(device, &firstLaneInfo);

        const agfxBufferViewCreateInfo recordViewInfo = BufferViewInfo(recordBuffer);
        const agfxBufferViewCreateInfo firstLaneViewInfo = BufferViewInfo(firstLaneBuffer);
        agfxBufferView* recordView = recordBuffer ? agfxBufferViewCreate(device, &recordViewInfo) : nullptr;
        agfxBufferView* firstLaneView = firstLaneBuffer ? agfxBufferViewCreate(device, &firstLaneViewInfo) : nullptr;

        agfxShaderModule* module =
            CreateShaderModule(device, shader, "main_wave_ops_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
        const agfxComputePipelineCreateInfo pipelineInfo = PipelineInfo(module);
        agfxComputePipeline* pipeline = agfxComputePipelineCreate(device, &pipelineInfo);
        agfxShaderModuleDestroy(device, module);

        bool ok = recordBuffer && firstLaneBuffer && recordView && firstLaneView && pipeline;
        if (!ok) {
            outError = "resource or pipeline creation failed";
        }

        if (ok) {
            agfxDeviceMakeResourcesResident(device);

            PushConstants constants{};
            constants.destination = (uint32_t)agfxBufferViewGetHandle(recordView);
            constants.firstLaneIDBuffer = (uint32_t)agfxBufferViewGetHandle(firstLaneView);

            gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
                agfxCommandBufferBufferBarrier(cmd, recordBuffer, AGFX_RESOURCE_STATE_COMMON,
                                               AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);
                agfxCommandBufferBufferBarrier(cmd, firstLaneBuffer, AGFX_RESOURCE_STATE_COMMON,
                                               AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);
                agfxComputePass* pass = agfxComputePassBegin(cmd, "compute wave ops");
                agfxComputePassSetPipeline(pass, pipeline);
                agfxComputePassPushConstants(pass, &constants, sizeof(constants));
                agfxComputePassDispatch(pass, 1, 1, 1);
                agfxComputePassEnd(pass);
            });

            std::vector<uint8_t> recordBytes;
            std::vector<uint8_t> firstLaneBytes;
            ok = ReadbackBuffer(device, gpu.Queue(), recordBuffer, kThreadCount * sizeof(Record),
                                AGFX_RESOURCE_STATE_UNORDERED_ACCESS, recordBytes) &&
                 ReadbackBuffer(device, gpu.Queue(), firstLaneBuffer, kThreadCount * sizeof(uint32_t),
                                AGFX_RESOURCE_STATE_UNORDERED_ACCESS, firstLaneBytes);
            if (ok) {
                outRecords.resize(kThreadCount);
                outFirstLaneIDs.resize(kThreadCount);
                memcpy(outRecords.data(), recordBytes.data(), recordBytes.size());
                memcpy(outFirstLaneIDs.data(), firstLaneBytes.data(), firstLaneBytes.size());
            } else {
                outError = "buffer readback failed";
            }
        }

        if (pipeline) agfxComputePipelineDestroy(device, pipeline);
        if (firstLaneView) agfxBufferViewDestroy(device, firstLaneView);
        if (recordView) agfxBufferViewDestroy(device, recordView);
        if (firstLaneBuffer) agfxBufferDestroy(device, firstLaneBuffer);
        if (recordBuffer) agfxBufferDestroy(device, recordBuffer);
        return ok;
    }

    bool RunCpp(const CompiledShader& shader, std::vector<Record>& outRecords,
                std::vector<uint32_t>& outFirstLaneIDs, std::string& outError)
    {
        agfx::Device device(DefaultDeviceCreateInfo());
        if (!device.Get()) {
            outError = "failed to create headless device";
            return false;
        }

        agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
        agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
        agfx::Fence fence = device.CreateFence();

        agfx::Buffer recordBuffer = device.CreateBuffer(RecordBufferInfo());
        agfx::Buffer firstLaneBuffer = device.CreateBuffer(FirstLaneBufferInfo());
        agfx::BufferView recordView = device.CreateBufferView(BufferViewInfo(recordBuffer));
        agfx::BufferView firstLaneView = device.CreateBufferView(BufferViewInfo(firstLaneBuffer));

        agfx::ComputePipeline pipeline;
        {
            agfx::ShaderModule module(device.Get(),
                CreateShaderModule(device.Get(), shader, "main_wave_ops_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
            pipeline = device.CreateComputePipeline(PipelineInfo(module));
        }

        if (!recordBuffer.Get() || !firstLaneBuffer.Get() || !recordView.Get() || !firstLaneView.Get() ||
            !pipeline.Get()) {
            outError = "resource or pipeline creation failed";
            return false;
        }

        device.MakeResourcesResident();

        PushConstants constants{};
        constants.destination = (uint32_t)agfxBufferViewGetHandle(recordView);
        constants.firstLaneIDBuffer = (uint32_t)agfxBufferViewGetHandle(firstLaneView);

        cmd.Begin();
        cmd.BufferBarrier(recordBuffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
        cmd.BufferBarrier(firstLaneBuffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
        {
            agfx::ComputePass pass = cmd.BeginComputePass("compute wave ops");
            pass.SetPipeline(pipeline);
            pass.PushConstants(&constants, sizeof(constants));
            pass.Dispatch(1, 1, 1);
        }
        cmd.End();

        queue.Submit(cmd);
        queue.Signal(fence, 1);
        fence.Wait(1, UINT64_MAX);

        std::vector<uint8_t> recordBytes;
        std::vector<uint8_t> firstLaneBytes;
        if (!ReadbackBuffer(device.Get(), queue, recordBuffer, kThreadCount * sizeof(Record),
                            AGFX_RESOURCE_STATE_UNORDERED_ACCESS, recordBytes) ||
            !ReadbackBuffer(device.Get(), queue, firstLaneBuffer, kThreadCount * sizeof(uint32_t),
                            AGFX_RESOURCE_STATE_UNORDERED_ACCESS, firstLaneBytes)) {
            outError = "buffer readback failed";
            return false;
        }
        outRecords.resize(kThreadCount);
        outFirstLaneIDs.resize(kThreadCount);
        memcpy(outRecords.data(), recordBytes.data(), recordBytes.size());
        memcpy(outFirstLaneIDs.data(), firstLaneBytes.data(), firstLaneBytes.size());
        return true;
    }

    bool RunEz(const CompiledShader& shader, std::vector<Record>& outRecords,
               std::vector<uint32_t>& outFirstLaneIDs, std::string& outError)
    {
        agfx::ez::ContextCreateInfo contextInfo{};
        contextInfo.deviceInfo = DefaultDeviceCreateInfo();
        contextInfo.windowHandle = nullptr; // headless: no swap chain
        contextInfo.width = 4;
        contextInfo.height = 4;
        agfx::ez::Context context(contextInfo);

        agfx::Device& device = context.GetDevice();

        // ez has no structured-buffer sugar; buffers and views come straight off the device.
        agfx::Buffer recordBuffer = device.CreateBuffer(RecordBufferInfo());
        agfx::Buffer firstLaneBuffer = device.CreateBuffer(FirstLaneBufferInfo());
        agfx::BufferView recordView = device.CreateBufferView(BufferViewInfo(recordBuffer));
        agfx::BufferView firstLaneView = device.CreateBufferView(BufferViewInfo(firstLaneBuffer));

        agfx::ComputePipeline pipeline;
        {
            agfx::ShaderModule module(device.Get(),
                CreateShaderModule(device.Get(), shader, "main_wave_ops_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
            pipeline = device.CreateComputePipeline(PipelineInfo(module));
        }
        if (!recordBuffer.Get() || !firstLaneBuffer.Get() || !recordView.Get() || !firstLaneView.Get() ||
            !pipeline.Get()) {
            outError = "resource or pipeline creation failed";
            return false;
        }

        device.MakeResourcesResident();

        PushConstants constants{};
        constants.destination = (uint32_t)agfxBufferViewGetHandle(recordView);
        constants.firstLaneIDBuffer = (uint32_t)agfxBufferViewGetHandle(firstLaneView);

        {
            agfx::ez::Frame frame = context.BeginFrame();
            // recordBuffer/firstLaneBuffer are raw agfx::Buffer, not ez-tracked resources (ez has no
            // structured-buffer sugar), so they are barriered directly rather than through
            // Context::TransitionBuffer, which only knows about its own Buffer wrapper.
            agfxCommandBufferBufferBarrier(context.GetCurrentCommandBuffer().Get(), recordBuffer,
                                           AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);
            agfxCommandBufferBufferBarrier(context.GetCurrentCommandBuffer().Get(), firstLaneBuffer,
                                           AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);

            // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("compute wave ops");
            pass.SetPipeline(pipeline);
            pass.PushConstants(&constants, sizeof(constants));
            pass.Dispatch(1, 1, 1);
        }
        context.DrainGPU();

        std::vector<uint8_t> recordBytes;
        std::vector<uint8_t> firstLaneBytes;
        if (!ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), recordBuffer,
                            kThreadCount * sizeof(Record), AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                            recordBytes) ||
            !ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), firstLaneBuffer,
                            kThreadCount * sizeof(uint32_t), AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                            firstLaneBytes)) {
            outError = "buffer readback failed";
            return false;
        }
        outRecords.resize(kThreadCount);
        outFirstLaneIDs.resize(kThreadCount);
        memcpy(outRecords.data(), recordBytes.data(), recordBytes.size());
        memcpy(outFirstLaneIDs.data(), firstLaneBytes.data(), firstLaneBytes.size());
        return true;
    }

    bool Run(TestApi api, std::vector<Record>& outRecords, std::vector<uint32_t>& outFirstLaneIDs,
             std::string& outError)
    {
        outError.clear();

        const CompiledShader shader = CompileTestShader("wave_ops.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_wave_ops_cs");
        if (!shader.Valid()) {
            outError = "failed to compile wave_ops.hlsl";
            return false;
        }

        switch (api) {
        case TestApi::C:   return RunC(shader, outRecords, outFirstLaneIDs, outError);
        case TestApi::Cpp: return RunCpp(shader, outRecords, outFirstLaneIDs, outError);
        case TestApi::Ez:  return RunEz(shader, outRecords, outFirstLaneIDs, outError);
        }
        outError = "unknown API flavor";
        return false;
    }
} // namespace

AGFX_TEST_BUFFER(ComputeWaveOps, C)
{
    std::vector<Record> records;
    std::vector<uint32_t> firstLaneIDs;
    std::string error;
    AGFX_EXPECT_MSG(Run(TestApi::C, records, firstLaneIDs, error), error.c_str());
    std::string checkError;
    AGFX_EXPECT_MSG(VerifyWaveOps(records, firstLaneIDs, checkError), checkError.c_str());
    ExpectBufferMatchesGolden(ctx, kGolden, RecordsToBytes(records));
}

AGFX_TEST_BUFFER(ComputeWaveOps, Cpp)
{
    std::vector<Record> records;
    std::vector<uint32_t> firstLaneIDs;
    std::string error;
    AGFX_EXPECT_MSG(Run(TestApi::Cpp, records, firstLaneIDs, error), error.c_str());
    std::string checkError;
    AGFX_EXPECT_MSG(VerifyWaveOps(records, firstLaneIDs, checkError), checkError.c_str());
    ExpectBufferMatchesGolden(ctx, kGolden, RecordsToBytes(records));
}

AGFX_TEST_BUFFER(ComputeWaveOps, Ez)
{
    std::vector<Record> records;
    std::vector<uint32_t> firstLaneIDs;
    std::string error;
    AGFX_EXPECT_MSG(Run(TestApi::Ez, records, firstLaneIDs, error), error.c_str());
    std::string checkError;
    AGFX_EXPECT_MSG(VerifyWaveOps(records, firstLaneIDs, checkError), checkError.c_str());
    ExpectBufferMatchesGolden(ctx, kGolden, RecordsToBytes(records));
}
