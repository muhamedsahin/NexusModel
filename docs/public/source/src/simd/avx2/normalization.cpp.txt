#include "nexus_model/simd/kernel_decls.hpp"
#include <cmath>
#include <immintrin.h>

namespace nexus_model::kernels::avx2 {
namespace {
double sum4(__m256d v) {
  __m128d s =
      _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
  return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}
} // namespace
void layernorm(const float *x, const float *gamma, const float *beta, float *y,
               float *mean, float *rstd, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float *row = x + static_cast<std::size_t>(r) * cols;
    float *out = y + static_cast<std::size_t>(r) * cols;
    __m256d sum = _mm256_setzero_pd();
    int c = 0;
    for (; c + 4 <= cols; c += 4)
      sum = _mm256_add_pd(sum, _mm256_cvtps_pd(_mm_loadu_ps(row + c)));
    double total = sum4(sum);
    for (; c < cols; ++c)
      total += row[c];
    const float m = static_cast<float>(total / cols);
    const __m256d mv = _mm256_set1_pd(m);
    __m256d var = _mm256_setzero_pd();
    c = 0;
    // Two-pass FP64 variance retains stability for large offsets / tiny
    // variance.
    for (; c + 4 <= cols; c += 4) {
      const __m256d d =
          _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + c)), mv);
      var = _mm256_fmadd_pd(d, d, var);
    }
    double variance = sum4(var);
    for (; c < cols; ++c) {
      const double d = static_cast<double>(row[c]) - m;
      variance += d * d;
    }
    const float rs = 1.f / std::sqrt(static_cast<float>(variance / cols) + eps);
    mean[r] = m;
    rstd[r] = rs;
    const __m256 mf = _mm256_set1_ps(m), rv = _mm256_set1_ps(rs);
    c = 0;
    for (; c + 8 <= cols; c += 8) {
      __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + c), mf), rv);
      if (gamma)
        v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + c));
      if (beta)
        v = _mm256_add_ps(v, _mm256_loadu_ps(beta + c));
      _mm256_storeu_ps(out + c, v);
    }
    for (; c < cols; ++c)
      out[c] =
          (row[c] - m) * rs * (gamma ? gamma[c] : 1.f) + (beta ? beta[c] : 0.f);
  }
}
} // namespace nexus_model::kernels::avx2
