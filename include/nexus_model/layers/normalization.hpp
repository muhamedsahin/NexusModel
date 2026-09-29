#pragma once

/**
 * @file normalization.hpp
 * @brief BatchNorm, LayerNorm, GroupNorm, InstanceNorm.
 *
 * BatchNorm (Ioffe & Szegedy, 2015): eğitimde mini-batch istatistiği,
 * değerlendirmede `running_mean` / `running_var`. Koşu varyansı yansız
 * kestirimle güncellenir (PyTorch ile aynı); normalizasyon yanlı varyansı kullanır.
 *
 *   x̂ = (x - μ) / sqrt(σ² + ε),  y = γ x̂ + β
 *   dx = (γ / (m sqrt(σ²+ε))) (m dy - Σdy - x̂ Σ(dy ⊙ x̂))   (γ, dy eleman bazında)
 *
 * LayerNorm (Ba, Kiros, Hinton, 2016): son eksen, batch'ten bağımsız.
 * GroupNorm (Wu & He, 2018): kanal grupları. InstanceNorm, groups = C olan
 * GroupNorm'dur (Ulyanov, stil transferi; koşu istatistiği tutulmaz).
 */

#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

namespace nexus_model {

class BatchNorm : public Module {
 public:
  BatchNorm(int num_features, float eps = 1e-5f, float momentum = 0.1f, bool affine = true)
      : features_(num_features), eps_(eps), momentum_(momentum), affine_(affine) {
    if (affine) {
      gamma_ = std::make_shared<Parameter>(Tensor::full({static_cast<std::size_t>(num_features)}, 1.f));
      beta_ = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(num_features)}));
      register_parameter("weight", gamma_);
      register_parameter("bias", beta_);
    }
    running_mean_ = std::make_shared<Tensor>(Tensor::zeros({static_cast<std::size_t>(num_features)}));
    running_var_ = std::make_shared<Tensor>(Tensor::full({static_cast<std::size_t>(num_features)}, 1.f));
    register_buffer("running_mean", running_mean_);
    register_buffer("running_var", running_var_);
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() < 2 || input.size(1) != static_cast<std::size_t>(features_)) {
      throw ModelError("BatchNorm channel dimension mismatch");
    }
    input_.copy_from(input);
    output_.resize_like(input);
    const int outer = static_cast<int>(input.size(0));
    int inner = 1;
    for (std::uint8_t axis = 2; axis < input.rank(); ++axis) inner *= static_cast<int>(input.size(axis));
    inner_ = inner;
    outer_ = outer;
    const std::size_t stat_dims[1] = {static_cast<std::size_t>(features_)};
    save_mean_.resize_dims(stat_dims, 1);
    save_rstd_.resize_dims(stat_dims, 1);
    training_at_forward_ = training_;
    kernels::batch_norm(input_.data(), affine_ ? gamma_->data.data() : nullptr, affine_ ? beta_->data.data() : nullptr,
                        output_.data(), save_mean_.data(), save_rstd_.data(), running_mean_->data(), running_var_->data(),
                        outer, features_, inner, eps_, momentum_, training_, true);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    float* dgamma = (affine_ && gamma_->requires_grad) ? gamma_->grad.data() : nullptr;
    float* dbeta = (affine_ && beta_->requires_grad) ? beta_->grad.data() : nullptr;
    if (!training_at_forward_) {
      const float* dy = grad_output.data();
      float* dx = grad_input_.data();
      const int outer = outer_;
      const int inner = inner_;
      for (int c = 0; c < features_; ++c) {
        const float g = affine_ ? gamma_->data[static_cast<std::size_t>(c)] : 1.f;
        const float scale = g * save_rstd_[static_cast<std::size_t>(c)];
        for (int n = 0; n < outer; ++n) {
          for (int i = 0; i < inner; ++i) {
            const std::size_t idx = (static_cast<std::size_t>(n * features_ + c) * inner) + i;
            dx[idx] = dy[idx] * scale;
            if (dgamma) {
              const float xhat = (input_[idx] - save_mean_[static_cast<std::size_t>(c)]) * save_rstd_[static_cast<std::size_t>(c)];
              dgamma[c] += dy[idx] * xhat;
            }
            if (dbeta) dbeta[c] += dy[idx];
          }
        }
      }
      return grad_input_;
    }
    kernels::batch_norm_bwd(input_.data(), affine_ ? gamma_->data.data() : nullptr, save_mean_.data(), save_rstd_.data(),
                            grad_output.data(), grad_input_.data(), dgamma, dbeta, outer_, features_, inner_);
    return grad_input_;
  }

 private:
  int features_;
  float eps_;
  float momentum_;
  bool affine_;
  bool training_at_forward_ = true;
  int outer_ = 0;
  int inner_ = 1;
  std::shared_ptr<Parameter> gamma_;
  std::shared_ptr<Parameter> beta_;
  std::shared_ptr<Tensor> running_mean_;
  std::shared_ptr<Tensor> running_var_;
  Tensor input_, output_, grad_input_, save_mean_, save_rstd_;
};

class BatchNorm1D : public BatchNorm {
 public:
  using BatchNorm::BatchNorm;
};

class BatchNorm2D : public BatchNorm {
 public:
  using BatchNorm::BatchNorm;
};

class LayerNorm : public Module {
 public:
  explicit LayerNorm(std::size_t normalized_size, float eps = 1e-5f) : normalized_(normalized_size), eps_(eps) {
    gamma_ = std::make_shared<Parameter>(Tensor::full({normalized_size}, 1.f));
    beta_ = std::make_shared<Parameter>(Tensor::zeros({normalized_size}));
    register_parameter("weight", gamma_);
    register_parameter("bias", beta_);
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() < 1 || input.size(input.rank() - 1) != normalized_) {
      throw ModelError("LayerNorm last dimension mismatch");
    }
    input_.copy_from(input);
    output_.resize_like(input);
    const int cols = static_cast<int>(normalized_);
    const int rows = static_cast<int>(input.numel() / normalized_);
    const std::size_t stat[1] = {static_cast<std::size_t>(rows)};
    mean_.resize_dims(stat, 1);
    rstd_.resize_dims(stat, 1);
    kernels::layernorm(input_.data(), gamma_->data.data(), beta_->data.data(), output_.data(), mean_.data(), rstd_.data(),
                       rows, cols, eps_);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    const int cols = static_cast<int>(normalized_);
    const int rows = static_cast<int>(input_.numel() / normalized_);
    float* dgamma = gamma_->requires_grad ? gamma_->grad.data() : nullptr;
    float* dbeta = beta_->requires_grad ? beta_->grad.data() : nullptr;
    kernels::layernorm_bwd(input_.data(), gamma_->data.data(), mean_.data(), rstd_.data(), grad_output.data(),
                           grad_input_.data(), dgamma, dbeta, rows, cols);
    return grad_input_;
  }

 private:
  std::size_t normalized_;
  float eps_;
  std::shared_ptr<Parameter> gamma_;
  std::shared_ptr<Parameter> beta_;
  Tensor input_, output_, grad_input_, mean_, rstd_;
};

class GroupNorm : public Module {
 public:
  GroupNorm(int num_groups, int num_channels, float eps = 1e-5f, bool affine = true)
      : groups_(num_groups), channels_(num_channels), eps_(eps), affine_(affine) {
    if (num_channels % num_groups != 0) throw ModelError("GroupNorm channels must be divisible by groups");
    if (affine) {
      gamma_ = std::make_shared<Parameter>(Tensor::full({static_cast<std::size_t>(num_channels)}, 1.f));
      beta_ = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(num_channels)}));
      register_parameter("weight", gamma_);
      register_parameter("bias", beta_);
    }
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() < 2 || input.size(1) != static_cast<std::size_t>(channels_)) {
      throw ModelError("GroupNorm channel mismatch");
    }
    input_.copy_from(input);
    output_.resize_like(input);
    const int n = static_cast<int>(input.size(0));
    int inner = 1;
    for (std::uint8_t axis = 2; axis < input.rank(); ++axis) inner *= static_cast<int>(input.size(axis));
    inner_ = inner;
    n_ = n;
    const std::size_t stat[1] = {static_cast<std::size_t>(n * groups_)};
    mean_.resize_dims(stat, 1);
    rstd_.resize_dims(stat, 1);
    kernels::group_norm(input_.data(), affine_ ? gamma_->data.data() : nullptr, affine_ ? beta_->data.data() : nullptr,
                        output_.data(), mean_.data(), rstd_.data(), n, channels_, inner, groups_, eps_);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    float* dgamma = (affine_ && gamma_->requires_grad) ? gamma_->grad.data() : nullptr;
    float* dbeta = (affine_ && beta_->requires_grad) ? beta_->grad.data() : nullptr;
    kernels::group_norm_bwd(input_.data(), affine_ ? gamma_->data.data() : nullptr, mean_.data(), rstd_.data(),
                            grad_output.data(), grad_input_.data(), dgamma, dbeta, n_, channels_, inner_, groups_);
    return grad_input_;
  }

 private:
  int groups_;
  int channels_;
  float eps_;
  bool affine_;
  int n_ = 0;
  int inner_ = 1;
  std::shared_ptr<Parameter> gamma_;
  std::shared_ptr<Parameter> beta_;
  Tensor input_, output_, grad_input_, mean_, rstd_;
};

class InstanceNorm : public GroupNorm {
 public:
  explicit InstanceNorm(int num_channels, float eps = 1e-5f, bool affine = true) : GroupNorm(num_channels, num_channels, eps, affine) {}
};

}  // namespace nexus_model
