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

void layernorm_bwd(const float* x, const float* gamma, const float* mean, const float* rstd, const float* dy, float* dx,
                   float* dgamma, float* dbeta, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    const float* gr = dy + static_cast<std::size_t>(r) * cols;
    float* dxr = dx + static_cast<std::size_t>(r) * cols;
    const float m = mean[r];
    const float rs = rstd[r];
    __m256 sum_dy = _mm256_setzero_ps();
    __m256 sum_dy_xhat = _mm256_setzero_ps();
    const __m256 mv = _mm256_set1_ps(m);
    const __m256 rv = _mm256_set1_ps(rs);
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
      const __m256 g = gamma != nullptr ? _mm256_loadu_ps(gamma + c) : _mm256_set1_ps(1.f);
      const __m256 xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + c), mv), rv);
      const __m256 dys = _mm256_mul_ps(_mm256_loadu_ps(gr + c), g);
      sum_dy = _mm256_add_ps(sum_dy, dys);
      sum_dy_xhat = _mm256_fmadd_ps(dys, xhat, sum_dy_xhat);
      if (dgamma != nullptr) {
        _mm256_storeu_ps(dgamma + c, _mm256_fmadd_ps(_mm256_loadu_ps(gr + c), xhat, _mm256_loadu_ps(dgamma + c)));
      }
      if (dbeta != nullptr) {
        _mm256_storeu_ps(dbeta + c, _mm256_add_ps(_mm256_loadu_ps(dbeta + c), _mm256_loadu_ps(gr + c)));
      }
    }
    float dy_s = 0.f;
    float dy_x = 0.f;
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum_dy);
    for (float value : tmp) dy_s += value;
    _mm256_store_ps(tmp, sum_dy_xhat);
    for (float value : tmp) dy_x += value;
    for (; c < cols; ++c) {
      const float g = gamma != nullptr ? gamma[c] : 1.f;
      const float xhat = (row[c] - m) * rs;
      const float dys = gr[c] * g;
      dy_s += dys;
      dy_x += dys * xhat;
      if (dgamma != nullptr) dgamma[c] += gr[c] * xhat;
      if (dbeta != nullptr) dbeta[c] += gr[c];
    }
    const float inv_m = 1.f / static_cast<float>(cols);
    const __m256 dy_sv = _mm256_set1_ps(dy_s);
    const __m256 dy_xv = _mm256_set1_ps(dy_x);
    const __m256 cols_v = _mm256_set1_ps(static_cast<float>(cols));
    const __m256 inv = _mm256_set1_ps(inv_m);
    c = 0;
    for (; c + 8 <= cols; c += 8) {
      const __m256 g = gamma != nullptr ? _mm256_loadu_ps(gamma + c) : _mm256_set1_ps(1.f);
      const __m256 xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + c), mv), rv);
      const __m256 dys = _mm256_mul_ps(_mm256_loadu_ps(gr + c), g);
      const __m256 inner = _mm256_sub_ps(_mm256_fmsub_ps(cols_v, dys, dy_sv), _mm256_mul_ps(xhat, dy_xv));
      _mm256_storeu_ps(dxr + c, _mm256_mul_ps(_mm256_mul_ps(rv, inv), inner));
    }
    for (; c < cols; ++c) {
      const float g = gamma != nullptr ? gamma[c] : 1.f;
      const float xhat = (row[c] - m) * rs;
      const float dys = gr[c] * g;
      dxr[c] = rs * inv_m * (static_cast<float>(cols) * dys - dy_s - xhat * dy_x);
    }
  }
}
} // namespace nexus_model::kernels::avx2
