#pragma once

/**
 * @file residual.hpp
 * @brief y = F(x) + x. Geri yayılımda gradyan her iki kola da gider:
 * dx = dF + dy. He et al., Deep Residual Learning, 2016.
 */

#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <memory>
#include <utility>

namespace nexus_model {

class ResidualBlock : public Module {
 public:
  explicit ResidualBlock(std::shared_ptr<Module> function) : function_(std::move(function)) {
    register_module("function", function_);
  }

  Tensor forward(const Tensor& input) override {
    input_.copy_from(input);
    Tensor fx = function_->forward(input_);
    output_.resize_like(fx);
    kernels::add(fx.data(), input_.data(), output_.data(), input_.numel());
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    Tensor df = function_->backward(grad_output);
    grad_input_.resize_like(grad_output);
    kernels::add(df.data(), grad_output.data(), grad_input_.data(), grad_output.numel());
    return grad_input_;
  }

 private:
  std::shared_ptr<Module> function_;
  Tensor input_, output_, grad_input_;
};

}  // namespace nexus_model
