#include "nexus_model/simd/kernel_decls.hpp"

#include <immintrin.h>

namespace nexus_model::kernels::avx512 {
namespace {
inline float hsum(__m512 v) {
  __m256 lo = _mm512_castps512_ps256(v);
  __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  __m256 s = _mm256_add_ps(lo, hi);
  __m128 a = _mm256_castps256_ps128(s);
  __m128 b = _mm256_extractf128_ps(s, 1);
  a = _mm_add_ps(a, b);
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
  return _mm_cvtss_f32(a);
}
}  // namespace

float dot(const float* a, const float* b, int n) {
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
  float s = hsum(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

void relu(const float* x, float* y, std::size_t n) {
  const __m512 z = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) _mm512_storeu_ps(y + i, _mm512_max_ps(_mm512_loadu_ps(x + i), z));
  for (; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

void linear_forward(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f) {
  // Reuse the blocked AVX2 GEMM for batches; a wider dot loop is not a GEMM.
  avx2::linear_forward(x,w,bias,y,rows,in_f,out_f);
}
}  // namespace nexus_model::kernels::avx512
