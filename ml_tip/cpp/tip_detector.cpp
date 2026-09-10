#include "tip_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace smartdarts {
namespace ml_tip {

struct TipDetector::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "smartdarts_ml_tip"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::string input_name;
  std::string output_name;
  Ort::AllocatorWithDefaultOptions allocator;

  explicit Impl(const std::string& model_path) {
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(_WIN32)
    const std::wstring wpath(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(env, wpath.c_str(), opts);
#else
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
#endif
    Ort::AllocatedStringPtr in_name =
        session->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr out_name =
        session->GetOutputNameAllocated(0, allocator);
    input_name = in_name.get();
    output_name = out_name.get();
  }

  std::pair<float, float> Run(const std::vector<float>& nchw) const {
    if (nchw.size() != static_cast<size_t>(1 * 1 * kTipInputSize * kTipInputSize)) {
      throw std::runtime_error("TipDetector: unexpected input buffer size");
    }
    std::array<int64_t, 4> shape{1, 1, kTipInputSize, kTipInputSize};
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(nchw.data()),
        nchw.size(),
        shape.data(),
        shape.size());

    const char* in_names[] = {input_name.c_str()};
    const char* out_names[] = {output_name.c_str()};
    auto outputs = session->Run(
        Ort::RunOptions{nullptr},
        in_names,
        &input_tensor,
        1,
        out_names,
        1);
    float* out = outputs.front().GetTensorMutableData<float>();
    float x = out[0];
    float y = out[1];
    const float lo = 0.f;
    const float hi = static_cast<float>(kTipInputSize - 1);
    x = std::min(hi, std::max(lo, x));
    y = std::min(hi, std::max(lo, y));
    return {x, y};
  }
};

TipDetector::TipDetector(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)), model_path_(model_path) {}

TipDetector::~TipDetector() = default;

TipDetector::TipDetector(TipDetector&&) noexcept = default;
TipDetector& TipDetector::operator=(TipDetector&&) noexcept = default;

std::pair<float, float> TipDetector::PredictGrayU8(
    const unsigned char* gray_200x200) const {
  if (!gray_200x200) {
    throw std::invalid_argument("TipDetector::PredictGrayU8: null input");
  }
  std::vector<float> buf(
      static_cast<size_t>(kTipInputSize * kTipInputSize));
  const float inv = 1.f / 255.f;
  for (int i = 0; i < kTipInputSize * kTipInputSize; ++i) {
    buf[static_cast<size_t>(i)] = static_cast<float>(gray_200x200[i]) * inv;
  }
  return impl_->Run(buf);
}

std::pair<float, float> TipDetector::PredictGray01(
    const float* gray_200x200) const {
  if (!gray_200x200) {
    throw std::invalid_argument("TipDetector::PredictGray01: null input");
  }
  std::vector<float> buf(
      gray_200x200,
      gray_200x200 + (kTipInputSize * kTipInputSize));
  return impl_->Run(buf);
}

}  // namespace ml_tip
}  // namespace smartdarts
