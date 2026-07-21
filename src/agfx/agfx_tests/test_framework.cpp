/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "test_framework.h"
#include "test_runner.h"

namespace agfxtest
{
    const char* ToString(TestKind kind)
    {
        switch (kind) {
            case TestKind::Validation: return "validation";
            case TestKind::Buffer:     return "buffer";
            case TestKind::Texture:    return "texture";
        }
        return "unknown";
    }

    const char* ToString(TestApi api)
    {
        switch (api) {
            case TestApi::C:   return "c";
            case TestApi::Cpp: return "cpp";
            case TestApi::Ez:  return "ez";
        }
        return "unknown";
    }

    const char* ToString(TestStatus status)
    {
        switch (status) {
            case TestStatus::Passed:  return "passed";
            case TestStatus::Failed:  return "failed";
            case TestStatus::Skipped: return "skipped";
        }
        return "unknown";
    }

    std::vector<TestCase>& Registry()
    {
        // Function-local static: the registry must outlive nothing in particular, but it must be
        // constructed before the first Registrar runs, whatever order the TUs initialize in.
        static std::vector<TestCase> registry;
        return registry;
    }

    void TestContext::Fail(const char* file, int line, const std::string& message)
    {
        if (mResult.status != TestStatus::Failed) {
            mResult.status = TestStatus::Failed;
            mResult.file = file ? file : "";
            mResult.line = line;
            mResult.message = message;
        } else {
            mResult.message += "\n" + message;
        }
    }

    void TestContext::Skip(const std::string& reason)
    {
        if (mResult.status == TestStatus::Passed) {
            mResult.status = TestStatus::Skipped;
            mResult.message = reason;
        }
    }

    const std::string& TestContext::OutputDir() const
    {
        return RunnerOptions().outputDir;
    }

    bool TestContext::UpdateGoldens() const
    {
        return RunnerOptions().updateGoldens;
    }

    std::string TestContext::ArtifactDir() const
    {
        return std::string("artifacts/") + mCase->name + "." + ToString(mCase->api);
    }
} // namespace agfxtest
