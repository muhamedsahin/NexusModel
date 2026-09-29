#pragma once

/**
 * @file activations.hpp
 * @brief Eleman-bazlı aktivasyonlar. İleri ve geri çekirdekler AVX2/AVX-512'ye
 * düşer. Türevler çıkış ya da giriş önbelleğinden hesaplanır; ara tensör yok.
 *
 * ReLU: y = max(x, 0), dx = dy * 1_{x>0}
 * LeakyReLU: y = x>0 ? x : αx
 * ELU: y = x>0 ? x : α(e^x-1), dx = x>0 ? dy : dy (y+α)  (Clevert et al., 2016)
 * GELU: y = x Φ(x), Φ = standart normal cdf (Hendrycks & Gimpel, 2016)
 * GELU-tanh: Hendrycks yaklaşımı
 * Sigmoid: σ(x)=1/(1+e^{-x}), dx = dy σ(1-σ)
 * Tanh: dx = dy (1-tanh^2)
 * Softmax: satır bazında kararlı (max çıkarılır). Jacobian: y ⊙ (dy - ⟨dy,y⟩)
 * LogSoftmax: y = x - logsumexp(x), dx = dy - softmax(x) Σ dy
 * SiLU: x σ(x) (Elfwing et al., 2018)
 * Mish: x tanh(softplus(x)) (Misra, 2019)
 */

#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <cmath>

namespace nexus_model {

class ReLU : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    kernels::relu(cache_.data(), output_.data(), input.numel());
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::relu_bwd(cache_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  Tensor cache_, output_, grad_input_;
};

class LeakyReLU : public Module {
 public:
  explicit LeakyReLU(float negative_slope = 0.01f) : slope_(negative_slope) {}
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    kernels::leaky_relu(cache_.data(), output_.data(), input.numel(), slope_);
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::leaky_relu_bwd(cache_.data(), grad_output.data(), grad_input_.data(), grad_output.numel(), slope_);
    return grad_input_;
  }

 private:
  float slope_;
  Tensor cache_, output_, grad_input_;
};

class ELU : public Module {
 public:
  explicit ELU(float alpha = 1.f) : alpha_(alpha) {}
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    kernels::elu(cache_.data(), output_.data(), input.numel(), alpha_);
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::elu_bwd(cache_.data(), output_.data(), grad_output.data(), grad_input_.data(), grad_output.numel(), alpha_);
    return grad_input_;
  }

 private:
  float alpha_;
  Tensor cache_, output_, grad_input_;
};

class GELU : public Module {
 public:
  explicit GELU(bool tanh_approx = false) : tanh_approx_(tanh_approx) {}
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    if (tanh_approx_) kernels::gelu_tanh(cache_.data(), output_.data(), input.numel());
    else kernels::gelu(cache_.data(), output_.data(), input.numel());
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    if (tanh_approx_) kernels::gelu_tanh_bwd(cache_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    else kernels::gelu_bwd(cache_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  bool tanh_approx_;
  Tensor cache_, output_, grad_input_;
};

class Sigmoid : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    output_.resize_like(input);
    kernels::sigmoid(input.data(), output_.data(), input.numel());
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::sigmoid_bwd(output_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  Tensor output_, grad_input_;
};

class Tanh : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    output_.resize_like(input);
    kernels::tanh_fwd(input.data(), output_.data(), input.numel());
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::tanh_bwd(output_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  Tensor output_, grad_input_;
};

class SiLU : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    kernels::silu(cache_.data(), output_.data(), input.numel());
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::silu_bwd(cache_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  Tensor cache_, output_, grad_input_;
};

class Mish : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    kernels::mish(cache_.data(), output_.data(), input.numel());
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    kernels::mish_bwd(cache_.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  Tensor cache_, output_, grad_input_;
};

class Softmax : public Module {
 public:
  /// Son eksen üzerinde kararlı softmax.
  Tensor forward(const Tensor& input) override {
    if (input.rank() < 1) throw ModelError("softmax expects a non-scalar tensor");
    output_.resize_like(input);
    const int cols = static_cast<int>(input.size(input.rank() - 1));
    const int rows = static_cast<int>(input.numel() / input.size(input.rank() - 1));
    kernels::softmax(input.data(), output_.data(), rows, cols);
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    const int cols = static_cast<int>(output_.size(output_.rank() - 1));
    const int rows = static_cast<int>(output_.numel() / output_.size(output_.rank() - 1));
    kernels::softmax_bwd(output_.data(), grad_output.data(), grad_input_.data(), rows, cols);
    return grad_input_;
  }

 private:
  Tensor output_, grad_input_;
};

class LogSoftmax : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    output_.resize_like(input);
    const int cols = static_cast<int>(input.size(input.rank() - 1));
    const int rows = static_cast<int>(input.numel() / static_cast<std::size_t>(cols));
    kernels::log_softmax(input.data(), output_.data(), rows, cols);
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    const int cols = static_cast<int>(output_.size(output_.rank() - 1));
    const int rows = static_cast<int>(output_.numel() / static_cast<std::size_t>(cols));
    kernels::log_softmax_bwd(output_.data(), grad_output.data(), grad_input_.data(), rows, cols);
    return grad_input_;
  }

 private:
  Tensor output_, grad_input_;
};

/// PReLU: y = x>0 ? x : αx. α öğrenilir (He et al., 2015).
class PReLU : public Module {
 public:
  explicit PReLU(float init = 0.25f) {
    alpha_ = std::make_shared<Parameter>(Tensor::full({1}, init));
    register_parameter("weight", alpha_);
  }
  Tensor forward(const Tensor& input) override {
    cache_.copy_from(input);
    output_.resize_like(input);
    const float a = alpha_->data[0];
    const float* x = cache_.data();
    float* y = output_.data();
    for (std::size_t i = 0; i < input.numel(); ++i) y[i] = x[i] > 0.f ? x[i] : a * x[i];
    return output_;
  }
  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(grad_output);
    const float a = alpha_->data[0];
    const float* x = cache_.data();
    const float* dy = grad_output.data();
    float* dx = grad_input_.data();
    double dalpha = 0.0;
    for (std::size_t i = 0; i < grad_output.numel(); ++i) {
      if (x[i] > 0.f) dx[i] = dy[i];
      else {
        dx[i] = dy[i] * a;
        dalpha += static_cast<double>(dy[i]) * x[i];
      }
    }
    if (alpha_->requires_grad) alpha_->grad[0] += static_cast<float>(dalpha);
    return grad_input_;
  }

 private:
  std::shared_ptr<Parameter> alpha_;
  Tensor cache_, output_, grad_input_;
};

}  // namespace nexus_model
