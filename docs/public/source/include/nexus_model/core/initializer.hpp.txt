#pragma once

/**
 * @file initializer.hpp
 * @brief Ağırlık başlatma. Linear/Conv varsayılanı Kaiming uniform (He, ReLU).
 *
 * Kaiming: He et al., Delving Deep into Rectifiers, 2015.
 * Xavier: Glorot & Bengio, 2010.
 * Orthogonal: Saxe et al., Exact solutions to the nonlinear dynamics of learning in deep linear neural networks, 2014.
 */

#include "nexus_model/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace nexus_model::init {

inline std::mt19937& engine() {
  static thread_local std::mt19937 generator{0xC0FFEEU};
  return generator;
}

inline void manual_seed(std::uint64_t seed) { engine().seed(static_cast<std::mt19937::result_type>(seed)); }

inline void constant_(Tensor& tensor, float value) { tensor.fill(value); }
inline void zeros_(Tensor& tensor) { tensor.zero(); }
inline void ones_(Tensor& tensor) { tensor.fill(1.f); }

inline void uniform_(Tensor& tensor, float low, float high) {
  std::uniform_real_distribution<float> dist(low, high);
  float* data = tensor.data();
  for (std::size_t i = 0; i < tensor.numel(); ++i) data[i] = dist(engine());
}

inline void normal_(Tensor& tensor, float mean, float stddev) {
  std::normal_distribution<float> dist(mean, stddev);
  float* data = tensor.data();
  for (std::size_t i = 0; i < tensor.numel(); ++i) data[i] = dist(engine());
}

inline std::pair<std::size_t, std::size_t> fan_in_out(const Tensor& tensor) {
  if (tensor.rank() < 2) {
    const std::size_t n = std::max<std::size_t>(tensor.numel(), 1);
    return {n, n};
  }
  std::size_t receptive = 1;
  for (std::uint8_t axis = 2; axis < tensor.rank(); ++axis) receptive *= tensor.size(axis);
  return {tensor.size(1) * receptive, tensor.size(0) * receptive};
}

inline void kaiming_uniform_(Tensor& tensor, float negative_slope = 0.f, bool fan_out = false) {
  const auto [fan_in, fan_out_v] = fan_in_out(tensor);
  const float fan = static_cast<float>(fan_out ? fan_out_v : fan_in);
  const float gain = std::sqrt(2.f / (1.f + negative_slope * negative_slope));
  const float bound = std::sqrt(3.f) * gain / std::sqrt(std::max(fan, 1.f));
  uniform_(tensor, -bound, bound);
}

inline void kaiming_normal_(Tensor& tensor, float negative_slope = 0.f, bool fan_out = false) {
  const auto [fan_in, fan_out_v] = fan_in_out(tensor);
  const float fan = static_cast<float>(fan_out ? fan_out_v : fan_in);
  const float gain = std::sqrt(2.f / (1.f + negative_slope * negative_slope));
  const float stddev = gain / std::sqrt(std::max(fan, 1.f));
  normal_(tensor, 0.f, stddev);
}

inline void xavier_uniform_(Tensor& tensor, float gain = 1.f) {
  const auto [fan_in, fan_out_v] = fan_in_out(tensor);
  const float denom = static_cast<float>(fan_in + fan_out_v);
  const float bound = gain * std::sqrt(6.f / std::max(denom, 1.f));
  uniform_(tensor, -bound, bound);
}

inline void xavier_normal_(Tensor& tensor, float gain = 1.f) {
  const auto [fan_in, fan_out_v] = fan_in_out(tensor);
  const float stddev = gain * std::sqrt(2.f / std::max(static_cast<float>(fan_in + fan_out_v), 1.f));
  normal_(tensor, 0.f, stddev);
}

inline void orthogonal_(Tensor& tensor, float gain = 1.f) {
  if (tensor.rank() != 2) throw ModelError("orthogonal_ expects a matrix");
  const std::size_t rows = tensor.size(0);
  const std::size_t cols = tensor.size(1);
  const bool transpose = rows < cols;
  const std::size_t m = transpose ? cols : rows;
  const std::size_t n = transpose ? rows : cols;
  std::vector<float> q(m * n);
  std::normal_distribution<float> dist(0.f, 1.f);
  for (float& value : q) value = dist(engine());
  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t prev = 0; prev < col; ++prev) {
      double dot = 0.0;
      for (std::size_t row = 0; row < m; ++row) {
        dot += static_cast<double>(q[row * n + col]) * q[row * n + prev];
      }
      for (std::size_t row = 0; row < m; ++row) {
        q[row * n + col] -= static_cast<float>(dot) * q[row * n + prev];
      }
    }
    double norm = 0.0;
    for (std::size_t row = 0; row < m; ++row) {
      const double v = q[row * n + col];
      norm += v * v;
    }
    norm = std::sqrt(std::max(norm, 1e-12));
    for (std::size_t row = 0; row < m; ++row) q[row * n + col] = static_cast<float>(q[row * n + col] / norm) * gain;
  }
  float* data = tensor.data();
  if (!transpose) {
    std::copy(q.begin(), q.end(), data);
  } else {
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < cols; ++c) data[r * cols + c] = q[c * rows + r];
    }
  }
}

}  // namespace nexus_model::init
