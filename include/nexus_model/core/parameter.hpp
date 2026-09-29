#pragma once

/**
 * @file parameter.hpp
 * @brief Öğrenilebilir ağırlık. `grad` geri yayılımda += ile birikir.
 *
 * Aynı modül nesnesi üzerinde eşzamanlı `forward` tanımsızdır (ara tamponlar
 * paylaşılır). Veri paralel eğitim, kopya başına bir modül kullanır; kopyaların
 * gradyanları adım dışında toplanır. Sıcak yol kilit almaz.
 */

#include "nexus_model/core/tensor.hpp"

#include <mutex>

namespace nexus_model {

class Parameter {
 public:
  Tensor data;
  Tensor grad;
  bool requires_grad = true;

  explicit Parameter(Tensor values, bool need_grad = true) : data(std::move(values)), requires_grad(need_grad) {
    grad = Tensor::zeros(data.shape_vec());
  }

  void zero_grad() {
    if (grad.numel() != data.numel()) {
      grad = Tensor::zeros(data.shape_vec());
    } else {
      grad.zero();
    }
  }

  /// Replika birleştirme. Eğitim adımının iç halkasında çağrılmaz.
  void accumulate_grad_locked(const Tensor& extra) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (extra.numel() != grad.numel()) throw ModelError("accumulate_grad size mismatch");
    for (std::size_t i = 0; i < grad.numel(); ++i) grad[i] += extra[i];
  }

 private:
  std::mutex mutex_;
};

}  // namespace nexus_model
