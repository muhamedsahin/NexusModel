#include "nexus_model/core/error.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nexus_model::kernels {
void embedding_forward(const float* table, const float* indices, float* y, int rows, int dim, int vocab,
                       int padding_idx) {
  for (int r = 0; r < rows; ++r) {
    const int idx = static_cast<int>(indices[r]);
    float* dst = y + static_cast<std::size_t>(r) * dim;
    if (idx == padding_idx) {
      std::fill(dst, dst + dim, 0.f);
      continue;
    }
    if (idx < 0 || idx >= vocab) throw ModelError("embedding index out of range");
    const float* src = table + static_cast<std::size_t>(idx) * dim;
    std::copy(src, src + dim, dst);
  }
}

void embedding_backward(const float* grad, const float* indices, float* dtable, int rows, int dim, int vocab,
                        int padding_idx) {
  for (int r = 0; r < rows; ++r) {
    const int idx = static_cast<int>(indices[r]);
    if (idx == padding_idx) continue;
    if (idx < 0 || idx >= vocab) throw ModelError("embedding index out of range");
    axpy(dtable + static_cast<std::size_t>(idx) * dim, grad + static_cast<std::size_t>(r) * dim, 1.f, dim);
  }
}


}  // namespace nexus_model::kernels
