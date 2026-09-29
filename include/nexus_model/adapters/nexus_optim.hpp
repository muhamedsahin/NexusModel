#pragma once

/**
 * @file nexus_optim.hpp
 * @brief Parameter* listesini NexusOptim ParamGroup'una çevirir.
 *
 * Çekirdek kütüphane NexusOptim'e bağlanmaz. Bu başlık yalnızca örneği veya
 * eğitim döngüsünü derleyen hedefte dahil edilir. İşaretçiler ödünçtür;
 * Parameter depolaması optimizer yaşadığı sürece yerinde kalmalıdır.
 */

#include "nexus_model/core/module.hpp"
#include "nexus_optim/core/param_group.hpp"

#include <vector>

namespace nexus_model {

inline nexus_optim::ParamGroup<float> as_param_group(Module& model, nexus_optim::ParamGroupOptions options = {}) {
  std::vector<float*> params;
  std::vector<float*> grads;
  std::vector<std::size_t> numels;
  for (Parameter* param : model.parameters()) {
    if (param == nullptr || !param->requires_grad) continue;
    params.push_back(param->data.data());
    grads.push_back(param->grad.data());
    numels.push_back(param->data.numel());
  }
  return nexus_optim::make_group<float>(std::move(params), std::move(grads), std::move(numels), options);
}

}  // namespace nexus_model
