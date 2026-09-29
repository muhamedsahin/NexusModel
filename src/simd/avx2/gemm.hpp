#pragma once
#include <cstddef>
#include <immintrin.h>

namespace nexus_model::kernels::avx2::internal {
// Six rows, sixteen columns: twelve independent FMA chains, no horizontal sums.
// A may be transposed (dw); B is contiguous or a packed weight panel.
template <int Rows>
inline void tile(const float *a, std::size_t ars, std::size_t aks,
                 const float *b, std::size_t bs, float *c, int cs, int depth,
                 bool accumulate, const float *bias = nullptr) {
  __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00;
  __m256 c20 = c00, c21 = c00, c30 = c00, c31 = c00;
  __m256 c40 = c00, c41 = c00, c50 = c00, c51 = c00;
  for (int k = 0; k < depth; ++k) {
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8);
    __m256 av = _mm256_broadcast_ss(a);
    c00 = _mm256_fmadd_ps(av, b0, c00);
    c01 = _mm256_fmadd_ps(av, b1, c01);
    if constexpr (Rows > 1) {
      av = _mm256_broadcast_ss(a + ars);
      c10 = _mm256_fmadd_ps(av, b0, c10);
      c11 = _mm256_fmadd_ps(av, b1, c11);
    }
    if constexpr (Rows > 2) {
      av = _mm256_broadcast_ss(a + 2 * ars);
      c20 = _mm256_fmadd_ps(av, b0, c20);
      c21 = _mm256_fmadd_ps(av, b1, c21);
    }
    if constexpr (Rows > 3) {
      av = _mm256_broadcast_ss(a + 3 * ars);
      c30 = _mm256_fmadd_ps(av, b0, c30);
      c31 = _mm256_fmadd_ps(av, b1, c31);
    }
    if constexpr (Rows > 4) {
      av = _mm256_broadcast_ss(a + 4 * ars);
      c40 = _mm256_fmadd_ps(av, b0, c40);
      c41 = _mm256_fmadd_ps(av, b1, c41);
    }
    if constexpr (Rows > 5) {
      av = _mm256_broadcast_ss(a + 5 * ars);
      c50 = _mm256_fmadd_ps(av, b0, c50);
      c51 = _mm256_fmadd_ps(av, b1, c51);
    }
    a += aks;
    b += bs;
  }
  auto save = [&](float *dst, __m256 lo, __m256 hi) {
    if (accumulate) {
      lo = _mm256_add_ps(lo, _mm256_loadu_ps(dst));
      hi = _mm256_add_ps(hi, _mm256_loadu_ps(dst + 8));
    } else if (bias) {
      lo = _mm256_add_ps(lo, _mm256_loadu_ps(bias));
      hi = _mm256_add_ps(hi, _mm256_loadu_ps(bias + 8));
    }
    _mm256_storeu_ps(dst, lo);
    _mm256_storeu_ps(dst + 8, hi);
  };
  save(c, c00, c01);
  if constexpr (Rows > 1)
    save(c + cs, c10, c11);
  if constexpr (Rows > 2)
    save(c + 2 * cs, c20, c21);
  if constexpr (Rows > 3)
    save(c + 3 * cs, c30, c31);
  if constexpr (Rows > 4)
    save(c + 4 * cs, c40, c41);
  if constexpr (Rows > 5)
    save(c + 5 * cs, c50, c51);
}
// C += A B; supports both dX=dY W and dW=dY^T X without transposing buffers.
void accumulate(const float *a, std::size_t ars, std::size_t aks,
                const float *b, int bs, float *c, int m, int n, int k);
} // namespace nexus_model::kernels::avx2::internal
