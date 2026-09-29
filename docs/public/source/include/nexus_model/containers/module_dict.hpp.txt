#pragma once

/**
 * @file module_dict.hpp
 * @brief İsimle erişilen alt modül koleksiyonu (çoklu görev başlıkları için).
 */

#include "nexus_model/core/module.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nexus_model {

class ModuleDict : public Module {
 public:
  void add(std::string name, std::shared_ptr<Module> module) {
    register_module(name, module);
    items_.emplace_back(std::move(name), std::move(module));
  }

  Tensor forward(const Tensor& input) override {
    (void)input;
    throw ModelError("ModuleDict has no implicit forward");
  }
  Tensor backward(const Tensor& grad_output) override {
    (void)grad_output;
    throw ModelError("ModuleDict has no implicit backward");
  }

  std::shared_ptr<Module> operator[](const std::string& name) const {
    for (const auto& [key, module] : items_) {
      if (key == name) return module;
    }
    throw ModelError("ModuleDict has no module named " + name);
  }

 private:
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> items_;
};

}  // namespace nexus_model
