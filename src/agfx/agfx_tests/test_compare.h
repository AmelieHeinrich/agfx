/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include "test_framework.h"

#include <cstdint>
#include <string>
#include <vector>

namespace agfxtest
{
    // --- Filesystem helpers ---------------------------------------------------------------

    /// @brief Creates dirPath and every missing parent. No-op if it already exists.
    bool EnsureDirectory(const std::string& dirPath);

    bool ReadFileBytes(const std::string& path, std::vector<uint8_t>& outBytes);
    bool WriteFileBytes(const std::string& path, const void* data, size_t size);

    /// @brief Reads a text file whole. Used for shader sources.
    bool ReadTextFile(const std::string& path, std::string& outText);

    // --- Image containers -----------------------------------------------------------------

    /// @brief An RGBA image in linear float, the common currency between readback, golden IO and
    /// FLIP. LDR images are stored as their 0..1 normalized values.
    struct Image
    {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> pixels; ///< width * height * 4, RGBA.

        bool Valid() const { return width > 0 && height > 0 && pixels.size() == size_t(width) * height * 4; }
    };

    /// @brief Loads a golden image. `.hdr` goes through stbi_loadf, anything else through stbi_load
    /// (8-bit, values divided by 255).
    bool LoadImage(const std::string& path, Image& outImage);

    /// @brief Writes an image. `.hdr` writes radiance HDR, anything else writes 8-bit PNG.
    bool SaveImage(const std::string& path, const Image& image);

    // --- Comparison -----------------------------------------------------------------------

    /// @brief memcmp with a diff-offset list, filling `payload` for the report. Returns true when
    /// the buffers are byte-identical.
    bool CompareBuffers(const std::vector<uint8_t>& output, const std::vector<uint8_t>& golden,
                        BufferPayload& payload);

    struct FlipResult
    {
        float meanError = 0.0f;
        float maxError = 0.0f;
        Image errorMap; ///< Magma-colormapped, ready to save as PNG.
    };

    /// @brief Runs NVIDIA FLIP over two same-sized images. `hdr` selects FLIP's HDR mode.
    bool RunFlip(const Image& output, const Image& golden, bool hdr, FlipResult& outResult);

    // --- Golden-backed assertions ---------------------------------------------------------
    // These handle the --update-goldens path, write the report artifacts, and fill the payload.
    // They record failures on ctx rather than returning a value the caller must check.

    /// @brief Compares `bytes` against `<goldenDir>/<goldenName>`; with --update-goldens, writes it.
    void ExpectBufferMatchesGolden(TestContext& ctx, const std::string& goldenName,
                                   const std::vector<uint8_t>& bytes);

    /// @brief Compares `image` against `<goldenDir>/<goldenName>` via FLIP; with --update-goldens,
    /// writes it. `goldenName` extension selects PNG vs HDR.
    void ExpectImageMatchesGolden(TestContext& ctx, const std::string& goldenName, const Image& image);
} // namespace agfxtest
