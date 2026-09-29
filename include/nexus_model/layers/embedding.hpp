#pragma once

/**
 * @file embedding.hpp
 * @brief İndeks ile satır seçimi. Geri yayılım yalnızca dokunulan satırlara
 * scatter-add yapar (padding_idx gradyan almaz ve çıkışı sıfırdır).
 */

#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

namespace nexus_model {

class Embedding : public Module {
 public:
  Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1)
      : vocab_(num_embeddings), dim_(embedding_dim), padding_idx_(padding_idx) {
    weight_ = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(num_embeddings), static_cast<std::size_t>(embedding_dim)}));
    init::normal_(weight_->data, 0.f, 1.f);
    if (padding_idx_ >= 0) {
      float* row = weight_->data.data() + static_cast<std::size_t>(padding_idx_) * embedding_dim;
      for (int i = 0; i < embedding_dim; ++i) row[i] = 0.f;
    }
    register_parameter("weight", weight_);
  }

  Tensor forward(const Tensor& indices) override {
    indices_.copy_from(indices);
    const std::size_t rows = indices.numel();
    std::size_t dims[kMaxRank];
    std::uint8_t rank = indices.rank();
    for (std::uint8_t i = 0; i < rank; ++i) dims[i] = indices.dims()[i];
    dims[rank++] = static_cast<std::size_t>(dim_);
    output_.resize_dims(dims, rank);
    kernels::embedding_forward(weight_->data.data(), indices_.data(), output_.data(), static_cast<int>(rows), dim_, vocab_,
                               padding_idx_);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    if (weight_->requires_grad) {
      kernels::embedding_backward(grad_output.data(), indices_.data(), weight_->grad.data(), static_cast<int>(indices_.numel()),
                                  dim_, vocab_, padding_idx_);
    }
    grad_input_.resize_like(indices_);
    grad_input_.zero();
    return grad_input_;
  }

 private:
  int vocab_;
  int dim_;
  int padding_idx_;
  std::shared_ptr<Parameter> weight_;
  Tensor indices_, output_, grad_input_;
};

}  // namespace nexus_model
