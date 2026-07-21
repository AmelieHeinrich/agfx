/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <string>

namespace agfxtest
{
    /// @brief Parsed command line, shared with test bodies via TestContext.
    struct Options
    {
        std::string filter;              ///< Substring match on the test name; empty matches all.
        std::string kindFilter;          ///< "validation" | "buffer" | "texture"; empty matches all.
        std::string apiFilter;           ///< "c" | "cpp" | "ez"; empty matches all.
        std::string outputDir = "test_results";
        std::string goldenDir = "data/tests/golden";
        bool updateGoldens = false;
        bool listOnly = false;
    };

    /// @brief The options the runner is executing under. Valid for the duration of main().
    const Options& RunnerOptions();
} // namespace agfxtest
