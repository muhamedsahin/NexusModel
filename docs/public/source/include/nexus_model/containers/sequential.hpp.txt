#pragma once

/**
 * @file sequential.hpp
 * @brief Katmanları sırayla çalıştırır. Geri geçiş ters sıradadır.
 * Ara aktivasyonlar burada tutulmaz; her katman kendi önbelleğini taşır.
 */

#include "nexus_model/core/module.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nexus_model {

class Sequential : public Module {
 public:
  Sequential() = default;
  explicit Sequential(std::vector<std::shared_ptr<Module>> layers) {
    for (std::size_t i = 0; i < layers.size(); ++i) add(layers[i]);
  }

  void add(const std::shared_ptr<Module>& layer) {
    register_module(std::to_string(layers_.size()), layer);
    layers_.push_back(layer);
  }

  Tensor forward(const Tensor& input) override {
    Tensor current = input;
    for (const auto& layer : layers_) current = layer->forward(current);
    return current;
  }

  Tensor backward(const Tensor& grad_output) override {
    Tensor grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) grad = (*it)->backward(grad);
    return grad;
  }

  [[nodiscard]] std::size_t size() const noexcept { return layers_.size(); }
  Module& operator[](std::size_t index) { return *layers_[index]; }

 private:
  std::vector<std::shared_ptr<Module>> layers_;
};

}  // namespace nexus_model
