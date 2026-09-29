#pragma once

/**
 * @file flatten_reshape.hpp
 * @brief Flatten [N, ...] -> [N, -1]. Reshape genel şekil değiştirir.
 * Bellek satır-major olduğu için veri kopyalanmaz; geri geçiş şekli geri yazar.
 */

#include "nexus_model/core/module.hpp"

#include <cstring>
#include <vector>

namespace nexus_model {

class Flatten : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    if (input.rank() < 2) throw ModelError("Flatten expects rank >= 2");
    cached_rank_ = input.rank();
    for (std::uint8_t i = 0; i < cached_rank_; ++i) cached_[i] = input.dims()[i];
    const std::size_t dims[2] = {input.size(0), input.numel() / input.size(0)};
    output_.resize_dims(dims, 2);
    if (input.numel() > 0) std::memcpy(output_.data(), input.data(), input.numel() * sizeof(float));
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_dims(cached_, cached_rank_);
    if (grad_output.numel() > 0) std::memcpy(grad_input_.data(), grad_output.data(), grad_output.numel() * sizeof(float));
    return grad_input_;
  }

 private:
  std::size_t cached_[kMaxRank]{};
  std::uint8_t cached_rank_ = 0;
  Tensor output_, grad_input_;
};

class Reshape : public Module {
 public:
  explicit Reshape(std::initializer_list<std::size_t> shape) : shape_(shape) {}

  Tensor forward(const Tensor& input) override {
    cached_rank_ = input.rank();
    for (std::uint8_t i = 0; i < cached_rank_; ++i) cached_[i] = input.dims()[i];
    if (shape_.size() > kMaxRank) throw ModelError("reshape rank exceeds 8");
    output_ = input.view_as(shape_.data(), static_cast<std::uint8_t>(shape_.size()));
    held_ = input;
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_dims(cached_, cached_rank_);
    if (grad_output.numel() > 0) std::memcpy(grad_input_.data(), grad_output.data(), grad_output.numel() * sizeof(float));
    return grad_input_;
  }

 private:
  std::vector<std::size_t> shape_;
  std::size_t cached_[kMaxRank]{};
  std::uint8_t cached_rank_ = 0;
  Tensor output_, held_, grad_input_;
};

}  // namespace nexus_model
