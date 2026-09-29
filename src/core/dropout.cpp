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
  std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ull;
  for (std::size_t i = 0; i < n; ++i) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const float u = static_cast<float>(state >> 40) * (1.f / 16777216.f);
    mask[i] = u >= p ? 1.f : 0.f;
  }
}


}  // namespace nexus_model::kernels
