#pragma once

/**
 * @file dropout.hpp
 * @brief Ters dropout (Srivastava et al., 2014).
 *
 * Eğitim: y = x ⊙ m / (1-p), m ~ Bernoulli(1-p)
 * Değerlendirme: y = x
 * Geri: dx = dy ⊙ m / (1-p)  (eğitimde), aksi halde dx = dy
 */

#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <cstdint>

namespace nexus_model {

class Dropout : public Module {
 public:
  explicit Dropout(float probability = 0.5f, std::uint64_t seed = 0xD1CEU) : probability_(probability), seed_(seed) {
    if (probability < 0.f || probability > 1.f) throw ModelError("dropout probability must be in [0, 1]");
  }

  Tensor forward(const Tensor& input) override {
    training_at_forward_ = training_;
    output_.resize_like(input);
    if (!training_ || probability_ == 0.f) {
      output_.copy_from(input);
      return output_;
    }
    mask_.resize_like(input);
    seed_ += 0x9E3779B97F4A7C15ull;
    kernels::dropout_mask(mask_.data(), input.numel(), probability_, seed_);
    const float scale = probability_ >= 1.f ? 0.f : 1.f / (1.f - probability_);
    const float* x = input.data();
    const float* m = mask_.data();
    float* y = output_.data();
    for (std::size_t i = 0; i < input.numel(); ++i) y[i] = x[i] * m[i] * scale;
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    if (!training_at_forward_ || probability_ == 0.f) {
      grad_input_.copy_from(grad_output);
      return grad_input_;
    }
    const float scale = probability_ >= 1.f ? 0.f : 1.f / (1.f - probability_);
    const float* dy = grad_output.data();
    const float* m = mask_.data();
    float* dx = grad_input_.data();
    for (std::size_t i = 0; i < grad_output.numel(); ++i) dx[i] = dy[i] * m[i] * scale;
    return grad_input_;
  }

 private:
  float probability_;
  std::uint64_t seed_;
  bool training_at_forward_ = true;
  Tensor mask_, output_, grad_input_;
};

}  // namespace nexus_model
