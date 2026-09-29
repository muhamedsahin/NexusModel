#pragma once

/**
 * @file module_list.hpp
 * @brief İndeksle erişilen alt modül listesi. Kendi başına bir ileri fonksiyonu yoktur;
 * çağıran, elemanları açıkça çalıştırır. Parametreler yine de `parameters()` ile toplanır.
 */

#include "nexus_model/core/module.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nexus_model {

class ModuleList : public Module {
 public:
  ModuleList() = default;
  explicit ModuleList(std::vector<std::shared_ptr<Module>> modules) {
    for (auto& module : modules) add(module);
  }

  void add(const std::shared_ptr<Module>& module) {
    register_module(std::to_string(modules_.size()), module);
    modules_.push_back(module);
  }

  Tensor forward(const Tensor& input) override {
    (void)input;
    throw ModelError("ModuleList has no implicit forward; index the child modules");
  }
  Tensor backward(const Tensor& grad_output) override {
    (void)grad_output;
    throw ModelError("ModuleList has no implicit backward");
  }

  [[nodiscard]] std::size_t size() const noexcept { return modules_.size(); }
  Module& operator[](std::size_t index) { return *modules_[index]; }
  std::shared_ptr<Module> at(std::size_t index) const { return modules_.at(index); }

 private:
  std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace nexus_model
