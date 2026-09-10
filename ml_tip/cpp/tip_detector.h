#pragma once

// Lightweight ONNX Runtime tip detector for Pi / native builds.
// Input: 200×200 grayscale (uint8 or float [0,1]).
// Output: tip (x, y) in pixel coordinates of the 200×200 image ([0, 199]).
//
// Link against onnxruntime. Example CMake snippet is in README.md.

#include <memory>
#include <string>
#include <utility>

namespace smartdarts {
namespace ml_tip {

constexpr int kTipInputSize = 200;

class TipDetector {
 public:
  /// Load model once. Throws std::runtime_error on failure.
  explicit TipDetector(const std::string& model_path);
  ~TipDetector();

  TipDetector(const TipDetector&) = delete;
  TipDetector& operator=(const TipDetector&) = delete;
  TipDetector(TipDetector&&) noexcept;
  TipDetector& operator=(TipDetector&&) noexcept;

  /// Predict from uint8 grayscale 200×200 (row-major, size = 200*200).
  /// Pixels are scaled to [0,1] internally. Returns (x, y) in pixel coords.
  std::pair<float, float> PredictGrayU8(const unsigned char* gray_200x200) const;

  /// Predict from float32 grayscale 200×200 already in [0, 1] (row-major).
  std::pair<float, float> PredictGray01(const float* gray_200x200) const;

  const std::string& model_path() const { return model_path_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string model_path_;
};

}  // namespace ml_tip
}  // namespace smartdarts
