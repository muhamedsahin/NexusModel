#pragma once

/**
 * @file linear.hpp
 * @brief Tam bağlı katman.
 *
 * İleri:  Y = X W^T + b
 *         X: [..., in], W: [out, in], b: [out], Y: [..., out]
 * Geri:   dX += dY W ,   dW += dY^T X ,   db += Σ dY
 * Çekirdek biriktirir. `dX` katman tamponunda her geri geçişte sıfırlanır;
 * parametre gradyanı `zero_grad` ile temizlenir (PyTorch sözleşmesi).
 */

#include "nexus_model/core/execution_policy.hpp"
#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <vector>

namespace nexus_model {

class Linear : public Module {
 public:
  Linear(std::size_t in_features, std::size_t out_features, bool use_bias = true)
      : in_features_(in_features), out_features_(out_features) {
    if (in_features == 0 || out_features == 0) throw ModelError("Linear features must be positive");
    weight_ = std::make_shared<Parameter>(Tensor::zeros({out_features, in_features}));
    init::kaiming_uniform_(weight_->data);
    register_parameter("weight", weight_);
    if (use_bias) {
      bias_ = std::make_shared<Parameter>(Tensor::zeros({out_features}));
      register_parameter("bias", bias_);
    }
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() == 0 || input.size(input.rank() - 1) != in_features_) {
      throw ModelError("Linear input last dimension must equal in_features");
    }
    input_.copy_from(input);
    std::size_t dims[kMaxRank];
    for (std::uint8_t i = 0; i < input.rank(); ++i) dims[i] = input.dims()[i];
    dims[input.rank() - 1] = out_features_;
    output_.resize_dims(dims, input.rank());
    const int rows = static_cast<int>(input.numel() / in_features_);
    const int in_f = static_cast<int>(in_features_);
    const int out_f = static_cast<int>(out_features_);
    PackedGemmFn gemm = packed_gemm();
    if (matmul_policy() == MatmulPolicy::Hybrid && gemm != nullptr && rows >= 32 && in_f >= 32 && out_f >= 32) {
      // Y = X W^T. W is [out, in]; the packed product wants W^T as [in, out].
      packed_weight_.resize(static_cast<std::size_t>(in_f) * static_cast<std::size_t>(out_f));
      float* weight_t = packed_weight_.data();
      const float* weight = weight_->data.data();
      for (int row = 0; row < out_f; ++row) {
        for (int col = 0; col < in_f; ++col) {
          weight_t[static_cast<std::size_t>(col) * static_cast<std::size_t>(out_f) + static_cast<std::size_t>(row)] =
              weight[static_cast<std::size_t>(row) * static_cast<std::size_t>(in_f) + static_cast<std::size_t>(col)];
        }
      }
      gemm(input_.data(), in_f, weight_t, out_f, output_.data(), out_f, rows, out_f, in_f);
      if (bias_) {
        const float* bias = bias_->data.data();
        for (int row = 0; row < rows; ++row) {
          float* y = output_.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(out_f);
          for (int col = 0; col < out_f; ++col) {
            y[col] += bias[col];
          }
        }
      }
    } else {
      kernels::linear_forward(input_.data(), weight_->data.data(), bias_ ? bias_->data.data() : nullptr, output_.data(),
                              rows, in_f, out_f);
    }
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    grad_input_.zero();
    const int rows = static_cast<int>(input_.numel() / in_features_);
    float* dw = weight_->requires_grad ? weight_->grad.data() : nullptr;
    float* db = (bias_ && bias_->requires_grad) ? bias_->grad.data() : nullptr;
    kernels::linear_backward(input_.data(), weight_->data.data(), grad_output.data(), grad_input_.data(), dw, db, rows,
                             static_cast<int>(in_features_), static_cast<int>(out_features_));
    return grad_input_;
  }

  std::shared_ptr<Parameter> weight() const { return weight_; }
  std::shared_ptr<Parameter> bias() const { return bias_; }

 private:
  std::size_t in_features_;
  std::size_t out_features_;
  std::shared_ptr<Parameter> weight_;
  std::shared_ptr<Parameter> bias_;
  Tensor input_, output_, grad_input_;
  mutable std::vector<float> packed_weight_;
};

}  // namespace nexus_model
