/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "test_compare.h"
#include "test_runner.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

// FLIP.h declares file-static tables; keep it confined to this single translation unit.
#include "FLIP/FLIP.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

#if defined(_WIN32)
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)
    #ifndef S_ISDIR
        #define S_ISDIR(mode) (((mode) & _S_IFDIR) != 0)
    #endif
#endif

namespace agfxtest
{
    namespace
    {
        bool HasExtension(const std::string& path, const char* ext)
        {
            const size_t len = strlen(ext);
            if (path.size() < len) {
                return false;
            }
            std::string tail = path.substr(path.size() - len);
            std::transform(tail.begin(), tail.end(), tail.begin(), [](char c) { return (char)tolower(c); });
            return tail == ext;
        }

        std::string JoinPath(const std::string& a, const std::string& b)
        {
            if (a.empty()) return b;
            if (b.empty()) return a;
            return a.back() == '/' ? a + b : a + "/" + b;
        }

        /// @brief Reinhard tone map plus a gamma encode, for HDR preview images only.
        ///
        /// The report viewer runs in a browser, which cannot decode radiance HDR — pointing it at an
        /// .hdr artifact leaves the panel blank, and saving one straight to PNG clamps every
        /// above-1.0 texel to white. HDR tests therefore emit both: the .hdr as the real data, and
        /// this preview as what the viewer displays. Output and golden go through the identical
        /// mapping, so a visible difference in the preview is a real difference in the data.
        Image ToneMappedPreview(const Image& image)
        {
            Image preview;
            preview.width = image.width;
            preview.height = image.height;
            preview.pixels.resize(image.pixels.size());

            // A fixed exposure, not an auto one: pure Reinhard squeezes everything above 1.0 into a
            // narrow bright band, but an exposure derived from image content would differ between
            // the output and golden panels and could mask a real difference.
            constexpr float kExposure = 0.25f;

            for (size_t i = 0; i < image.pixels.size(); i += 4) {
                for (size_t c = 0; c < 3; ++c) {
                    const float v = std::max(image.pixels[i + c], 0.0f) * kExposure;
                    preview.pixels[i + c] = std::pow(v / (1.0f + v), 1.0f / 2.2f);
                }
                preview.pixels[i + 3] = image.pixels[i + 3]; // alpha passes through untouched
            }
            return preview;
        }
    } // namespace

    bool EnsureDirectory(const std::string& dirPath)
    {
        if (dirPath.empty()) {
            return true;
        }
        // Walk the path creating each component; mkdir on an existing dir is not an error for us.
        std::string partial;
        for (size_t i = 0; i <= dirPath.size(); ++i) {
            if (i == dirPath.size() || dirPath[i] == '/') {
                if (!partial.empty()) {
                    struct stat st{};
                    if (stat(partial.c_str(), &st) != 0) {
                        if (mkdir(partial.c_str(), 0755) != 0) {
                            return false;
                        }
                    } else if (!S_ISDIR(st.st_mode)) {
                        return false;
                    }
                }
            }
            if (i < dirPath.size()) {
                partial.push_back(dirPath[i]);
            }
        }
        return true;
    }

    bool ReadFileBytes(const std::string& path, std::vector<uint8_t>& outBytes)
    {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            return false;
        }
        fseek(f, 0, SEEK_END);
        const long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        outBytes.resize(size > 0 ? (size_t)size : 0);
        const size_t read = outBytes.empty() ? 0 : fread(outBytes.data(), 1, outBytes.size(), f);
        fclose(f);
        return read == outBytes.size();
    }

    bool WriteFileBytes(const std::string& path, const void* data, size_t size)
    {
        const size_t slash = path.find_last_of('/');
        if (slash != std::string::npos && !EnsureDirectory(path.substr(0, slash))) {
            return false;
        }
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) {
            return false;
        }
        const size_t written = size ? fwrite(data, 1, size, f) : 0;
        fclose(f);
        return written == size;
    }

    bool ReadTextFile(const std::string& path, std::string& outText)
    {
        std::vector<uint8_t> bytes;
        if (!ReadFileBytes(path, bytes)) {
            return false;
        }
        outText.assign(bytes.begin(), bytes.end());
        return true;
    }

    bool LoadImage(const std::string& path, Image& outImage)
    {
        int w = 0, h = 0, channels = 0;
        if (HasExtension(path, ".hdr")) {
            float* data = stbi_loadf(path.c_str(), &w, &h, &channels, 4);
            if (!data) {
                return false;
            }
            outImage.width = (uint32_t)w;
            outImage.height = (uint32_t)h;
            outImage.pixels.assign(data, data + size_t(w) * h * 4);
            stbi_image_free(data);
            return true;
        }

        stbi_uc* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
        if (!data) {
            return false;
        }
        outImage.width = (uint32_t)w;
        outImage.height = (uint32_t)h;
        outImage.pixels.resize(size_t(w) * h * 4);
        for (size_t i = 0; i < outImage.pixels.size(); ++i) {
            outImage.pixels[i] = data[i] / 255.0f;
        }
        stbi_image_free(data);
        return true;
    }

    bool SaveImage(const std::string& path, const Image& image)
    {
        if (!image.Valid()) {
            return false;
        }
        const size_t slash = path.find_last_of('/');
        if (slash != std::string::npos && !EnsureDirectory(path.substr(0, slash))) {
            return false;
        }

        if (HasExtension(path, ".hdr")) {
            return stbi_write_hdr(path.c_str(), (int)image.width, (int)image.height, 4, image.pixels.data()) != 0;
        }

        std::vector<uint8_t> bytes(image.pixels.size());
        for (size_t i = 0; i < image.pixels.size(); ++i) {
            bytes[i] = (uint8_t)std::lround(std::clamp(image.pixels[i], 0.0f, 1.0f) * 255.0f);
        }
        return stbi_write_png(path.c_str(), (int)image.width, (int)image.height, 4, bytes.data(),
                              (int)image.width * 4) != 0;
    }

    bool ImageEqualsRgba8(const Image& image, const std::vector<uint8_t>& expected)
    {
        if (!image.Valid() || image.pixels.size() != expected.size()) {
            return false;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (image.pixels[i] != expected[i] / 255.0f) {
                return false;
            }
        }
        return true;
    }

    bool CompareBuffers(const std::vector<uint8_t>& output, const std::vector<uint8_t>& golden,
                        BufferPayload& payload)
    {
        payload.valid = true;
        payload.size = output.size();
        payload.goldenSize = golden.size();
        payload.diffOffsets.clear();
        payload.totalDiffCount = 0;

        const size_t common = std::min(output.size(), golden.size());
        for (size_t i = 0; i < common; ++i) {
            if (output[i] != golden[i]) {
                ++payload.totalDiffCount;
                if (payload.diffOffsets.size() < kMaxRecordedDiffs) {
                    payload.diffOffsets.push_back(i);
                }
            }
        }
        // Bytes past the shorter buffer count as differing too, so a size mismatch is visible in
        // the report's hex view rather than only in the message.
        for (size_t i = common; i < std::max(output.size(), golden.size()); ++i) {
            ++payload.totalDiffCount;
            if (payload.diffOffsets.size() < kMaxRecordedDiffs) {
                payload.diffOffsets.push_back(i);
            }
        }

        return payload.totalDiffCount == 0 && output.size() == golden.size();
    }

    bool RunFlip(const Image& output, const Image& golden, bool hdr, FlipResult& outResult)
    {
        if (!output.Valid() || !golden.Valid() || output.width != golden.width ||
            output.height != golden.height) {
            return false;
        }

        const int w = (int)output.width;
        const int h = (int)output.height;
        const size_t pixelCount = size_t(w) * h;

        // FLIP takes interleaved 3-channel linear RGB; drop alpha.
        std::vector<float> outRgb(pixelCount * 3);
        std::vector<float> refRgb(pixelCount * 3);
        for (size_t i = 0; i < pixelCount; ++i) {
            for (int c = 0; c < 3; ++c) {
                outRgb[i * 3 + c] = output.pixels[i * 4 + c];
                refRgb[i * 3 + c] = golden.pixels[i * 4 + c];
            }
        }

        // Ask for the grayscale error map so we can derive max error ourselves, then colormap it
        // with FLIP's own magma table for the report.
        FLIP::Parameters parameters;
        float meanError = 0.0f;
        float* errorMap = nullptr;
        FLIP::evaluate(refRgb.data(), outRgb.data(), w, h, hdr, parameters,
                       /*applyMagmaMapToOutput*/ false, /*computeMeanFLIPError*/ true, meanError, &errorMap);
        if (!errorMap) {
            return false;
        }

        outResult.meanError = meanError;
        outResult.maxError = 0.0f;
        outResult.errorMap.width = output.width;
        outResult.errorMap.height = output.height;
        outResult.errorMap.pixels.resize(pixelCount * 4);
        for (size_t i = 0; i < pixelCount; ++i) {
            const float e = std::clamp(errorMap[i], 0.0f, 1.0f);
            outResult.maxError = std::max(outResult.maxError, e);

            const FLIP::color3& magma = FLIP::MapMagma[(int)std::lround(e * 255.0f)];
            outResult.errorMap.pixels[i * 4 + 0] = magma.x;
            outResult.errorMap.pixels[i * 4 + 1] = magma.y;
            outResult.errorMap.pixels[i * 4 + 2] = magma.z;
            outResult.errorMap.pixels[i * 4 + 3] = 1.0f;
        }

        // FLIP allocates the error map with new[]; we own it (see the evaluate() docs).
        delete[] errorMap;
        return true;
    }

    void ExpectBufferMatchesGolden(TestContext& ctx, const std::string& goldenName,
                                   const std::vector<uint8_t>& bytes)
    {
        const Options& options = RunnerOptions();
        const std::string goldenPath = JoinPath(options.goldenDir, goldenName);

        BufferPayload& payload = ctx.Result().buffer;
        payload.valid = true;
        payload.size = bytes.size();

        // Always write the produced bytes so the report can show them even on a pass.
        const std::string artifactDir = ctx.ArtifactDir();
        const std::string outputRel = JoinPath(artifactDir, "output.bin");
        if (WriteFileBytes(JoinPath(ctx.OutputDir(), outputRel), bytes.data(), bytes.size())) {
            payload.outputPath = outputRel;
        }

        if (ctx.UpdateGoldens()) {
            if (!WriteFileBytes(goldenPath, bytes.data(), bytes.size())) {
                ctx.Fail(__FILE__, __LINE__, "failed to write golden: " + goldenPath);
            }
            return;
        }

        std::vector<uint8_t> golden;
        if (!ReadFileBytes(goldenPath, golden)) {
            ctx.Fail(__FILE__, __LINE__,
                     "missing golden: " + goldenPath + " (run with --update-goldens to create it)");
            return;
        }

        // Copy the golden next to the output so every artifact path in the report is relative to
        // the results directory and the report stays self-contained.
        const std::string goldenRel = JoinPath(artifactDir, "golden.bin");
        if (WriteFileBytes(JoinPath(ctx.OutputDir(), goldenRel), golden.data(), golden.size())) {
            payload.goldenPath = goldenRel;
        }

        if (!CompareBuffers(bytes, golden, payload)) {
            std::string message = "buffer differs from golden: " + std::to_string(payload.totalDiffCount) +
                                  " differing byte(s)";
            if (bytes.size() != golden.size()) {
                message += "; size " + std::to_string(bytes.size()) + " vs golden " +
                           std::to_string(golden.size());
            }
            if (!payload.diffOffsets.empty()) {
                message += "; first at offset " + std::to_string(payload.diffOffsets.front());
            }
            ctx.Fail(__FILE__, __LINE__, message);
        }
    }

    void ExpectImageMatchesGolden(TestContext& ctx, const std::string& goldenName, const Image& image)
    {
        const Options& options = RunnerOptions();
        const std::string goldenPath = JoinPath(options.goldenDir, goldenName);
        const bool hdr = HasExtension(goldenName, ".hdr");

        TexturePayload& payload = ctx.Result().texture;
        payload.valid = true;
        payload.width = image.width;
        payload.height = image.height;
        payload.hdr = hdr;
        payload.threshold = ctx.Case().threshold;
        payload.goldenPath = goldenPath;

        if (!image.Valid()) {
            ctx.Fail(__FILE__, __LINE__, "produced image is empty or malformed");
            return;
        }

        // Write the produced image before anything can fail, so a missing/mismatched golden can
        // still be diagnosed by looking at what the test actually rendered.
        const std::string artifactDir = ctx.ArtifactDir();
        const char* ext = hdr ? ".hdr" : ".png";
        const std::string outputRel = JoinPath(artifactDir, std::string("output") + ext);
        if (SaveImage(JoinPath(ctx.OutputDir(), outputRel), image)) {
            payload.outputPath = outputRel;
        }

        // For HDR the .hdr above is the data of record, but the viewer can't render it; hand it a
        // tone-mapped PNG instead.
        if (hdr) {
            const std::string previewRel = JoinPath(artifactDir, "output.png");
            if (SaveImage(JoinPath(ctx.OutputDir(), previewRel), ToneMappedPreview(image))) {
                payload.outputPath = previewRel;
            }
        }

        if (ctx.UpdateGoldens()) {
            if (!SaveImage(goldenPath, image)) {
                ctx.Fail(__FILE__, __LINE__, "failed to write golden: " + goldenPath);
            }
            return;
        }

        Image golden;
        if (!LoadImage(goldenPath, golden)) {
            ctx.Fail(__FILE__, __LINE__,
                     "missing golden: " + goldenPath + " (run with --update-goldens to create it)");
            return;
        }
        if (golden.width != image.width || golden.height != image.height) {
            ctx.Fail(__FILE__, __LINE__,
                     "size mismatch: produced " + std::to_string(image.width) + "x" +
                         std::to_string(image.height) + ", golden " + std::to_string(golden.width) +
                         "x" + std::to_string(golden.height));
            return;
        }

        FlipResult flip;
        if (!RunFlip(image, golden, hdr, flip)) {
            ctx.Fail(__FILE__, __LINE__, "FLIP evaluation failed");
            return;
        }

        payload.meanError = flip.meanError;
        payload.maxError = flip.maxError;

        // Complete the triptych the viewer shows: output (already written above), a copy of the
        // golden, and the error map.
        const std::string goldenRel = JoinPath(artifactDir, std::string("golden") + ext);
        const std::string flipRel = JoinPath(artifactDir, "flip.png");
        if (SaveImage(JoinPath(ctx.OutputDir(), goldenRel), golden)) payload.goldenPath = goldenRel;
        if (SaveImage(JoinPath(ctx.OutputDir(), flipRel), flip.errorMap)) payload.flipPath = flipRel;

        // Same story as the output above: point the viewer at a tone-mapped preview of the golden,
        // produced by the identical mapping so the two panels stay comparable by eye.
        if (hdr) {
            const std::string previewRel = JoinPath(artifactDir, "golden.png");
            if (SaveImage(JoinPath(ctx.OutputDir(), previewRel), ToneMappedPreview(golden))) {
                payload.goldenPath = previewRel;
            }
        }

        if (flip.meanError > payload.threshold) {
            ctx.Fail(__FILE__, __LINE__,
                     "FLIP mean error " + std::to_string(flip.meanError) + " exceeds threshold " +
                         std::to_string(payload.threshold) + " (max error " +
                         std::to_string(flip.maxError) + ")");
        }
    }
} // namespace agfxtest
