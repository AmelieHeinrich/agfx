/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// The agfx test runner. Discovers everything registered by the AGFX_TEST_* macros, runs the subset
// the command line selects, and writes test_results/results.json for tools/test_report/index.html.

#include "test_compare.h"
#include "test_framework.h"
#include "test_gpu.h"
#include "test_runner.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

namespace agfxtest
{
    namespace
    {
        Options g_options;

        void PrintUsage()
        {
            printf(
                "agfx_tests — the agfx test suite\n"
                "\n"
                "Usage: xmake run agfx_tests [options]\n"
                "\n"
                "  --filter <substr>   Only run tests whose name contains <substr>\n"
                "  --kind <kind>       Only run validation | buffer | texture tests\n"
                "  --api <api>         Only run c | cpp | ez tests\n"
                "  --out <dir>         Results directory (default: test_results)\n"
                "  --golden <dir>      Golden file directory (default: data/tests/golden)\n"
                "  --update-goldens    Write produced output as the golden instead of comparing\n"
                "  --list              List the selected tests and exit\n"
                "  --help              Show this message\n");
        }

        bool ParseArgs(int argc, char** argv, Options& options)
        {
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                const bool hasValue = (i + 1) < argc;

                if (arg == "--help" || arg == "-h") {
                    PrintUsage();
                    return false;
                } else if (arg == "--update-goldens") {
                    options.updateGoldens = true;
                } else if (arg == "--list") {
                    options.listOnly = true;
                } else if (arg == "--filter" && hasValue) {
                    options.filter = argv[++i];
                } else if (arg == "--kind" && hasValue) {
                    options.kindFilter = argv[++i];
                } else if (arg == "--api" && hasValue) {
                    options.apiFilter = argv[++i];
                } else if (arg == "--out" && hasValue) {
                    options.outputDir = argv[++i];
                } else if (arg == "--golden" && hasValue) {
                    options.goldenDir = argv[++i];
                } else {
                    printf("unknown or incomplete argument: %s\n\n", arg.c_str());
                    PrintUsage();
                    return false;
                }
            }
            return true;
        }

        bool Selected(const TestCase& testCase, const Options& options)
        {
            if (!options.filter.empty() && std::string(testCase.name).find(options.filter) == std::string::npos) {
                return false;
            }
            if (!options.kindFilter.empty() && options.kindFilter != ToString(testCase.kind)) {
                return false;
            }
            if (!options.apiFilter.empty() && options.apiFilter != ToString(testCase.api)) {
                return false;
            }
            return true;
        }

        std::string Timestamp()
        {
            const std::time_t now = std::time(nullptr);
            char buffer[64] = {};
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
            return buffer;
        }

        const char* PlatformName()
        {
#if defined(GAME_MAC)
            return "macos";
#elif defined(GAME_WINDOWS)
            return "windows";
#else
            return "unknown";
#endif
        }

        /// @brief The device name to report. Created and torn down on its own so the header is
        /// populated even when every selected test is filtered out or fails early.
        std::string QueryDeviceName()
        {
            const agfxDeviceCreateInfo info = DefaultDeviceCreateInfo();
            agfxDevice* device = agfxDeviceCreate(&info);
            if (!device) {
                return "unknown";
            }
            const std::string name = DeviceName(device);
            agfxDeviceDestroy(device);
            return name;
        }

        nlohmann::json ToJson(const TestCase& testCase, const TestResult& result)
        {
            nlohmann::json entry;
            entry["name"] = testCase.name;
            entry["kind"] = ToString(testCase.kind);
            entry["api"] = ToString(testCase.api);
            entry["status"] = ToString(result.status);
            entry["durationMs"] = result.durationMs;
            entry["message"] = result.message;
            entry["source"] = std::string(testCase.file) + ":" + std::to_string(testCase.line);

            if (result.buffer.valid) {
                nlohmann::json buffer;
                buffer["size"] = result.buffer.size;
                buffer["goldenSize"] = result.buffer.goldenSize;
                buffer["output"] = result.buffer.outputPath;
                buffer["golden"] = result.buffer.goldenPath;
                buffer["diffOffsets"] = result.buffer.diffOffsets;
                buffer["totalDiffCount"] = result.buffer.totalDiffCount;
                entry["buffer"] = std::move(buffer);
            }

            if (result.texture.valid) {
                nlohmann::json texture;
                texture["width"] = result.texture.width;
                texture["height"] = result.texture.height;
                texture["hdr"] = result.texture.hdr;
                texture["output"] = result.texture.outputPath;
                texture["golden"] = result.texture.goldenPath;
                texture["flip"] = result.texture.flipPath;
                texture["meanError"] = result.texture.meanError;
                texture["maxError"] = result.texture.maxError;
                texture["threshold"] = result.texture.threshold;
                entry["texture"] = std::move(texture);
            }

            return entry;
        }
    } // namespace

    const Options& RunnerOptions() { return g_options; }
} // namespace agfxtest

int main(int argc, char** argv)
{
    using namespace agfxtest;
    using Clock = std::chrono::steady_clock;

    if (!ParseArgs(argc, argv, g_options)) {
        return 1;
    }

    std::vector<TestCase>& registry = Registry();
    std::vector<const TestCase*> selected;
    for (const TestCase& testCase : registry) {
        if (Selected(testCase, g_options)) {
            selected.push_back(&testCase);
        }
    }

    if (g_options.listOnly) {
        for (const TestCase* testCase : selected) {
            printf("%-10s %-4s %s\n", ToString(testCase->kind), ToString(testCase->api), testCase->name);
        }
        printf("\n%zu test(s)\n", selected.size());
        return 0;
    }

    if (!EnsureDirectory(g_options.outputDir)) {
        printf("failed to create output directory: %s\n", g_options.outputDir.c_str());
        return 1;
    }
    if (g_options.updateGoldens && !EnsureDirectory(g_options.goldenDir)) {
        printf("failed to create golden directory: %s\n", g_options.goldenDir.c_str());
        return 1;
    }

    const std::string deviceName = QueryDeviceName();
    printf("agfx_tests — %s on %s\n", deviceName.c_str(), PlatformName());
    if (g_options.updateGoldens) {
        printf("!! --update-goldens: writing goldens into %s instead of comparing\n", g_options.goldenDir.c_str());
    }
    printf("running %zu test(s)\n\n", selected.size());

    nlohmann::json tests = nlohmann::json::array();
    uint32_t passed = 0, failed = 0, skipped = 0;

    const Clock::time_point runStart = Clock::now();
    for (const TestCase* testCase : selected) {
        TestContext context(*testCase);

        const Clock::time_point start = Clock::now();
        testCase->fn(context);
        const Clock::time_point end = Clock::now();

        TestResult& result = context.Result();
        result.durationMs = std::chrono::duration<double, std::milli>(end - start).count();

        const char* statusText = "PASS";
        switch (result.status) {
            case TestStatus::Passed:  ++passed;  statusText = "PASS"; break;
            case TestStatus::Failed:  ++failed;  statusText = "FAIL"; break;
            case TestStatus::Skipped: ++skipped; statusText = "SKIP"; break;
        }

        printf("[%s] %-10s %-4s %-28s %7.1f ms\n", statusText, ToString(testCase->kind),
               ToString(testCase->api), testCase->name, result.durationMs);
        if (result.status != TestStatus::Passed && !result.message.empty()) {
            printf("       %s\n", result.message.c_str());
            if (!result.file.empty()) {
                printf("       at %s:%d\n", result.file.c_str(), result.line);
            }
        }

        tests.push_back(ToJson(*testCase, result));
    }
    const double totalMs =
        std::chrono::duration<double, std::milli>(Clock::now() - runStart).count();

    nlohmann::json report;
    report["timestamp"] = Timestamp();
    report["device"] = deviceName;
    report["platform"] = PlatformName();
    report["updateGoldens"] = g_options.updateGoldens;
    report["summary"] = {
        {"total", selected.size()}, {"passed", passed}, {"failed", failed},
        {"skipped", skipped},       {"durationMs", totalMs},
    };
    report["tests"] = std::move(tests);

    const std::string reportPath = g_options.outputDir + "/results.json";
    const std::string serialized = report.dump(2);
    if (!WriteFileBytes(reportPath, serialized.data(), serialized.size())) {
        printf("\nfailed to write %s\n", reportPath.c_str());
        return 1;
    }

    printf("\n%u passed, %u failed, %u skipped in %.1f ms\n", passed, failed, skipped, totalMs);
    printf("wrote %s\n", reportPath.c_str());

    return failed == 0 ? 0 : 1;
}
