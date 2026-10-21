#include "nexus_model/core/error.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nexus_model::kernels {
void dropout_mask(float* mask, std::size_t n, float p, std::uint64_t seed) {
  if (p <= 0.f) {
    std::fill(mask, mask + n, 1.f);
    return;
  }
  if (p >= 1.f) {
    std::fill(mask, mask + n, 0.f);
    return;
  }
  fill_dropout_mask(mask, n, p, seed);
}


}  // namespace nexus_model::kernels
