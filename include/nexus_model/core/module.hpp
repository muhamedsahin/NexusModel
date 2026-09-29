#pragma once

/**
 * @file module.hpp
 * @brief Katman tabanı. `forward` çıktıyı, `backward` giriş gradyanını üretir
 * ve parametre gradyanını biriktirir.
 *
 * Alt modüller `register_module` ile bağlanır; `train` / `eval` ve `state_dict`
 * ağacı bu kayıttan yürür. Sıralı konteyner aktivasyonları yeniden önbelleklemez:
 * her katman kendi geri yayılım tamponunu tutar, böylece ara kopya tek memcpy olur.
 */

#include "nexus_model/core/parameter.hpp"
#include "nexus_model/core/serialization.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nexus_model {

class Module {
 public:
  virtual ~Module() = default;

  virtual Tensor forward(const Tensor& input) = 0;
  virtual Tensor backward(const Tensor& grad_output) = 0;

  [[nodiscard]] std::vector<Parameter*> parameters() const {
    std::vector<Parameter*> out;
    collect_parameters(out);
    return out;
  }

  [[nodiscard]] std::vector<std::pair<std::string, Parameter*>> named_parameters(const std::string& prefix = "") const {
    std::vector<std::pair<std::string, Parameter*>> out;
    collect_named(prefix, out);
    return out;
  }

  void zero_grad() {
    for (Parameter* param : parameters()) param->zero_grad();
  }

  void train() { set_mode(true); }
  void eval() { set_mode(false); }
  [[nodiscard]] bool is_training() const noexcept { return training_; }

  [[nodiscard]] StateDict state_dict() const {
    StateDict state;
    write_state("", state);
    return state;
  }

  void load_state_dict(const StateDict& state, bool strict = true) {
    StateDict current = state_dict();
    if (strict) {
      for (const auto& item : state.items) {
        if (!current.items.count(item.first)) throw ModelError("unexpected state_dict key: " + item.first);
      }
      for (const auto& item : current.items) {
        if (!state.items.count(item.first)) throw ModelError("missing state_dict key: " + item.first);
      }
    }
    load_state("", state);
  }

 protected:
  bool training_ = true;

  void register_parameter(std::string name, std::shared_ptr<Parameter> parameter) {
    params_.emplace_back(std::move(name), std::move(parameter));
  }

  void register_buffer(std::string name, std::shared_ptr<Tensor> buffer) {
    buffers_.emplace_back(std::move(name), std::move(buffer));
  }

  void register_module(std::string name, std::shared_ptr<Module> module) {
    children_.emplace_back(std::move(name), std::move(module));
  }

  [[nodiscard]] const std::vector<std::pair<std::string, std::shared_ptr<Module>>>& children() const { return children_; }

 private:
  void set_mode(bool training) {
    training_ = training;
    for (auto& [_, child] : children_) child->set_mode(training);
  }

  void collect_parameters(std::vector<Parameter*>& out) const {
    for (const auto& [_, param] : params_) out.push_back(param.get());
    for (const auto& [_, child] : children_) child->collect_parameters(out);
  }

  void collect_named(const std::string& prefix, std::vector<std::pair<std::string, Parameter*>>& out) const {
    for (const auto& [name, param] : params_) out.emplace_back(prefix + name, param.get());
    for (const auto& [name, child] : children_) child->collect_named(prefix + name + ".", out);
  }

  void write_state(const std::string& prefix, StateDict& state) const {
    for (const auto& [name, param] : params_) state.items.emplace(prefix + name, param->data.clone());
    for (const auto& [name, buffer] : buffers_) state.items.emplace(prefix + name, buffer->clone());
    for (const auto& [name, child] : children_) child->write_state(prefix + name + ".", state);
  }

  void load_state(const std::string& prefix, const StateDict& state) {
    for (auto& [name, param] : params_) {
      const auto it = state.items.find(prefix + name);
      if (it == state.items.end()) continue;
      if (!param->data.same_shape(it->second)) throw ModelError("state_dict shape mismatch for " + prefix + name);
      param->data.copy_from(it->second);
    }
    for (auto& [name, buffer] : buffers_) {
      const auto it = state.items.find(prefix + name);
      if (it == state.items.end()) continue;
      if (!buffer->same_shape(it->second)) throw ModelError("state_dict shape mismatch for " + prefix + name);
      buffer->copy_from(it->second);
    }
    for (auto& [name, child] : children_) child->load_state(prefix + name + ".", state);
  }

  std::vector<std::pair<std::string, std::shared_ptr<Parameter>>> params_;
  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> buffers_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
};

}  // namespace nexus_model
