#include "nexus_model/simd/kernel_decls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace nexus_model::kernels::avx2 {
namespace {

inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  lo = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, lo);
  lo = _mm_add_ss(lo, shuf);
  return _mm_cvtss_f32(lo);
}

inline __m256 exp256(__m256 x) {
  const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
  const __m256 c1 = _mm256_set1_ps(0.693359375f);
  const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);
  x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
  __m256 fx = _mm256_fmadd_ps(x, log2ef, _mm256_set1_ps(0.5f));
  __m256 tmp = _mm256_floor_ps(fx);
  __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), _mm256_set1_ps(1.f));
  fx = _mm256_sub_ps(tmp, mask);
  x = _mm256_fnmadd_ps(fx, c1, x);
  x = _mm256_fnmadd_ps(fx, c2, x);
  __m256 z = _mm256_mul_ps(x, x);
  __m256 y = p0;
  y = _mm256_fmadd_ps(y, x, p1);
  y = _mm256_fmadd_ps(y, x, p2);
  y = _mm256_fmadd_ps(y, x, p3);
  y = _mm256_fmadd_ps(y, x, p4);
  y = _mm256_fmadd_ps(y, x, p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.f));
  __m256i emm0 = _mm256_cvtps_epi32(fx);
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
  emm0 = _mm256_slli_epi32(emm0, 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}

}  // namespace

void fill(float* y, float value, std::size_t n) {
  const __m256 v = _mm256_set1_ps(value);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, v);
  for (; i < n; ++i) y[i] = value;
}
void add(const float* a, const float* b, float* y, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  for (; i < n; ++i) y[i] = a[i] + b[i];
}
void mul(const float* a, const float* b, float* y, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  for (; i < n; ++i) y[i] = a[i] * b[i];
}
void scale(const float* x, float alpha, float* y, std::size_t n) {
  const __m256 a = _mm256_set1_ps(alpha);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), a));
  for (; i < n; ++i) y[i] = x[i] * alpha;
}
void axpy(float* y, const float* x, float alpha, int n) {
  const __m256 a = _mm256_set1_ps(alpha);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), a, _mm256_loadu_ps(y + i));
    _mm256_storeu_ps(y + i, acc);
  }
  for (; i < n; ++i) y[i] += alpha * x[i];
}
float dot(const float* a, const float* b, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1=a0, a2=a0, a3=a0;
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    a0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),a0);
    a1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),a1);
    a2=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16),a2);
    a3=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24),a3);
  }
  __m256 acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
  for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  float s = hsum(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
void relu(const float* x, float* y, std::size_t n) {
  const __m256 z = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_max_ps(_mm256_loadu_ps(x + i), z));
  for (; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
void relu_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  const __m256 z = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 mask = _mm256_cmp_ps(xv, z, _CMP_GT_OS);
    _mm256_storeu_ps(dx + i, _mm256_and_ps(_mm256_loadu_ps(dy + i), mask));
  }
  for (; i < n; ++i) dx[i] = x[i] > 0.f ? dy[i] : 0.f;
}
void leaky_relu(const float* x, float* y, std::size_t n, float slope) {
  const __m256 z = _mm256_setzero_ps();
  const __m256 s = _mm256_set1_ps(slope);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 pos = _mm256_max_ps(xv, z);
    __m256 neg = _mm256_mul_ps(_mm256_min_ps(xv, z), s);
    _mm256_storeu_ps(y + i, _mm256_add_ps(pos, neg));
  }
  for (; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : slope * x[i];
}
void leaky_relu_bwd(const float* x, const float* dy, float* dx, std::size_t n, float slope) {
  const __m256 z = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.f);
  const __m256 s = _mm256_set1_ps(slope);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(x + i), z, _CMP_GT_OS);
    __m256 scale = _mm256_blendv_ps(s, one, mask);
    _mm256_storeu_ps(dx + i, _mm256_mul_ps(_mm256_loadu_ps(dy + i), scale));
  }
  for (; i < n; ++i) dx[i] = dy[i] * (x[i] > 0.f ? 1.f : slope);
}
void sigmoid(const float* x, float* y, std::size_t n) {
  std::size_t i = 0;
  const __m256 one = _mm256_set1_ps(1.f);
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), xv));
    _mm256_storeu_ps(y + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
  }
  for (; i < n; ++i) {
    const float e = std::exp(-x[i]);
    y[i] = 1.f / (1.f + e);
  }
}
void sigmoid_bwd(const float* y, const float* dy, float* dx, std::size_t n) {
  const __m256 one = _mm256_set1_ps(1.f);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 yv = _mm256_loadu_ps(y + i);
    __m256 g = _mm256_mul_ps(yv, _mm256_sub_ps(one, yv));
    _mm256_storeu_ps(dx + i, _mm256_mul_ps(_mm256_loadu_ps(dy + i), g));
  }
  for (; i < n; ++i) dx[i] = dy[i] * y[i] * (1.f - y[i]);
}
void tanh_fwd(const float* x, float* y, std::size_t n) {
  // tanh(x) = 2*sigmoid(2x)-1, stable enough inside the clamped exp range.
  std::size_t i = 0;
  const __m256 one = _mm256_set1_ps(1.f);
  const __m256 two = _mm256_set1_ps(2.f);
  for (; i + 8 <= n; i += 8) {
    __m256 e = exp256(_mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(-2.f)));
    __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
    _mm256_storeu_ps(y + i, _mm256_fmsub_ps(two, sig, one));
  }
  for (; i < n; ++i) y[i] = std::tanh(x[i]);
}
void gelu_tanh(const float* x, float* y, std::size_t n) {
  constexpr float kC = 0.7978845608028654f;
  const __m256 c = _mm256_set1_ps(kC);
  const __m256 k = _mm256_set1_ps(0.044715f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.f);
  const __m256 two = _mm256_set1_ps(2.f);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 x3 = _mm256_mul_ps(xv, _mm256_mul_ps(xv, xv));
    __m256 u = _mm256_mul_ps(c, _mm256_fmadd_ps(k, x3, xv));
    __m256 e = exp256(_mm256_mul_ps(u, _mm256_set1_ps(-2.f)));
    __m256 t = _mm256_fmsub_ps(two, _mm256_div_ps(one, _mm256_add_ps(one, e)), one);
    _mm256_storeu_ps(y + i, _mm256_mul_ps(half, _mm256_mul_ps(xv, _mm256_add_ps(one, t))));
  }
  for (; i < n; ++i) {
    const float x3 = x[i] * x[i] * x[i];
    const float u = kC * (x[i] + 0.044715f * x3);
    y[i] = 0.5f * x[i] * (1.f + std::tanh(u));
  }
}
void gelu_tanh_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  detail::gelu_tanh_bwd(x, dy, dx, n);
}
void silu(const float* x, float* y, std::size_t n) {
  std::size_t i = 0;
  const __m256 one = _mm256_set1_ps(1.f);
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), xv));
    __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
    _mm256_storeu_ps(y + i, _mm256_mul_ps(xv, s));
  }
  for (; i < n; ++i) {
    const float e = std::exp(-x[i]);
    y[i] = x[i] / (1.f + e);
  }
}
void silu_bwd(const float* x, const float* dy, float* dx, std::size_t n) { detail::silu_bwd(x, dy, dx, n); }

void softmax(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* out = y + static_cast<std::size_t>(r) * cols;
    __m256 maximum=_mm256_set1_ps(row[0]);
    int k=0;
    for(;k+8<=cols;k+=8) maximum=_mm256_max_ps(maximum,_mm256_loadu_ps(row+k));
    alignas(32) float lanes[8]; _mm256_store_ps(lanes,maximum);
    float m=lanes[0];
    for(int lane=1;lane<8;++lane) m=std::max(m,lanes[lane]);
    for(;k<cols;++k) m=std::max(m,row[k]);
    const __m256 bias = _mm256_set1_ps(-m);
    __m256 sumv = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
      __m256 e = exp256(_mm256_add_ps(_mm256_loadu_ps(row + c), bias));
      _mm256_storeu_ps(out + c, e);
      sumv = _mm256_add_ps(sumv, e);
    }
    float sum = hsum(sumv);
    for (; c < cols; ++c) {
      const float e = std::exp(row[c] - m);
      out[c] = e;
      sum += e;
    }
    const float inv = 1.f / sum;
    scale(out, inv, out, static_cast<std::size_t>(cols));
  }
}

float erf_approx(float x) {
  const float sign = x < 0.f ? -1.f : 1.f;
  const float ax = std::fabs(x);
  const float t = 1.f / (1.f + 0.3275911f * ax);
  const float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  return sign * (1.f - poly * std::exp(-ax * ax));
}

__m256 erf_ps(__m256 x) {
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 sign = _mm256_or_ps(_mm256_and_ps(x, sign_mask), _mm256_set1_ps(1.0f));
  const __m256 ax = _mm256_andnot_ps(sign_mask, x);
  const __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), ax, _mm256_set1_ps(1.0f)));
  __m256 poly = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429f), t, _mm256_set1_ps(-1.453152027f));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.421413741f));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-0.284496736f));
  poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.254829592f));
  poly = _mm256_mul_ps(poly, t);
  const __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(ax, ax)));
  return _mm256_mul_ps(sign, _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(poly, e)));
}

void gelu(const float* x, float* y, std::size_t n) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 scale = _mm256_set1_ps(kInvSqrt2);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 xv = _mm256_loadu_ps(x + i);
    const __m256 cdf = _mm256_mul_ps(half, _mm256_add_ps(one, erf_ps(_mm256_mul_ps(xv, scale))));
    _mm256_storeu_ps(y + i, _mm256_mul_ps(xv, cdf));
  }
  for (; i < n; ++i) y[i] = 0.5f * x[i] * (1.f + erf_approx(x[i] * kInvSqrt2));
}

void gelu_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  constexpr float kInvSqrt2Pi = 0.3989422804014327f;
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 scale = _mm256_set1_ps(kInvSqrt2);
  const __m256 pdf_scale = _mm256_set1_ps(kInvSqrt2Pi);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 xv = _mm256_loadu_ps(x + i);
    const __m256 cdf = _mm256_mul_ps(half, _mm256_add_ps(one, erf_ps(_mm256_mul_ps(xv, scale))));
    const __m256 pdf = _mm256_mul_ps(pdf_scale, exp256(_mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(xv, xv))));
    const __m256 local = _mm256_fmadd_ps(xv, pdf, cdf);
    _mm256_storeu_ps(dx + i, _mm256_mul_ps(_mm256_loadu_ps(dy + i), local));
  }
  for (; i < n; ++i) {
    const float cdf = 0.5f * (1.f + erf_approx(x[i] * kInvSqrt2));
    const float pdf = kInvSqrt2Pi * std::exp(-0.5f * x[i] * x[i]);
    dx[i] = dy[i] * (cdf + x[i] * pdf);
  }
}

void softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* ys = y + static_cast<std::size_t>(r) * cols;
    const float* gs = dy + static_cast<std::size_t>(r) * cols;
    float* os = dx + static_cast<std::size_t>(r) * cols;
    __m256 dotv = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
      dotv = _mm256_fmadd_ps(_mm256_loadu_ps(gs + c), _mm256_loadu_ps(ys + c), dotv);
    }
    float dot = hsum(dotv);
    for (; c < cols; ++c) dot += gs[c] * ys[c];
    const __m256 dot8 = _mm256_set1_ps(dot);
    c = 0;
    for (; c + 8 <= cols; c += 8) {
      const __m256 yv = _mm256_loadu_ps(ys + c);
      const __m256 gv = _mm256_loadu_ps(gs + c);
      _mm256_storeu_ps(os + c, _mm256_mul_ps(yv, _mm256_sub_ps(gv, dot8)));
    }
    for (; c < cols; ++c) os[c] = ys[c] * (gs[c] - dot);
  }
}

void log_softmax(const float* x, float* y, int rows, int cols) {
  detail::log_softmax(x, y, rows, cols);
}

void log_softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* ys = y + static_cast<std::size_t>(r) * cols;
    const float* gs = dy + static_cast<std::size_t>(r) * cols;
    float* os = dx + static_cast<std::size_t>(r) * cols;
    __m256 sumv = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8) sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(gs + c));
    float sum = hsum(sumv);
    for (; c < cols; ++c) sum += gs[c];
    const __m256 sum8 = _mm256_set1_ps(sum);
    c = 0;
    for (; c + 8 <= cols; c += 8) {
      float lane[8];
      _mm256_storeu_ps(lane, _mm256_loadu_ps(ys + c));
      float out[8];
      _mm256_storeu_ps(out, _mm256_loadu_ps(gs + c));
      for (int lane_index = 0; lane_index < 8; ++lane_index) out[lane_index] -= std::exp(lane[lane_index]) * sum;
      _mm256_storeu_ps(os + c, _mm256_loadu_ps(out));
      (void)sum8;
    }
    for (; c < cols; ++c) os[c] = gs[c] - std::exp(ys[c]) * sum;
  }
}

}  // namespace nexus_model::kernels::avx2

namespace nexus_model::kernels {

float contiguous_sum(const float* data, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(data + i));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

void contiguous_affine(const float* x, float* y, int n, float mean, float scale, float shift) {
    const __m256 mean_v = _mm256_set1_ps(mean);
    const __m256 scale_v = _mm256_set1_ps(scale);
    const __m256 shift_v = _mm256_set1_ps(shift);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 value = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(_mm256_sub_ps(value, mean_v), scale_v, shift_v));
    }
    for (; i < n; ++i) {
        y[i] = (x[i] - mean) * scale + shift;
    }
}

bool pool_aligned(const float* x, float* y, int* index, int n, int c, int h, int w, int kernel, bool is_max) {
    if (kernel != 2 && kernel != 3) {
        return false;
    }
    if (h % kernel != 0 || w % kernel != 0) {
        return false;
    }
    const int oh = h / kernel;
    const int ow = w / kernel;
    const float inv = 1.0f / static_cast<float>(kernel * kernel);
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            const float* src = x + (static_cast<std::size_t>(ni * c + ci) * h) * w;
            float* dst = y + (static_cast<std::size_t>(ni * c + ci) * oh) * ow;
            int* idx = index != nullptr ? index + (static_cast<std::size_t>(ni * c + ci) * oh) * ow : nullptr;
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    if (is_max) {
                        float best = src[(oy * kernel) * w + ox * kernel];
                        int best_i = (oy * kernel) * w + ox * kernel;
                        for (int ky = 0; ky < kernel; ++ky) {
                            const int iy = oy * kernel + ky;
                            for (int kx = 0; kx < kernel; ++kx) {
                                const int flat = iy * w + ox * kernel + kx;
                                if (src[flat] > best) {
                                    best = src[flat];
                                    best_i = flat;
                                }
                            }
                        }
                        dst[oy * ow + ox] = best;
                        if (idx != nullptr) {
                            idx[oy * ow + ox] = best_i;
                        }
                    } else {
                        __m256 acc = _mm256_setzero_ps();
                        int count = 0;
                        float sum = 0.0f;
                        for (int ky = 0; ky < kernel; ++ky) {
                            const float* row = src + (oy * kernel + ky) * w + ox * kernel;
                            int kx = 0;
                            for (; kx + 8 <= kernel; kx += 8) {
                                acc = _mm256_add_ps(acc, _mm256_loadu_ps(row + kx));
                                count += 8;
                            }
                            for (; kx < kernel; ++kx) {
                                sum += row[kx];
                            }
                        }
                        if (count > 0) {
                            __m128 lo = _mm256_castps256_ps128(acc);
                            __m128 hi = _mm256_extractf128_ps(acc, 1);
                            lo = _mm_add_ps(lo, hi);
                            lo = _mm_hadd_ps(lo, lo);
                            lo = _mm_hadd_ps(lo, lo);
                            sum += _mm_cvtss_f32(lo);
                        }
                        dst[oy * ow + ox] = sum * inv;
                    }
                }
            }
        }
    }
    return true;
}

void fill_dropout_mask(float* mask, std::size_t n, float p, std::uint64_t seed) {
    alignas(32) std::uint64_t state[4] = {seed ? seed : 0x9E3779B97F4A7C15ull, seed ^ 0xD1B54A32D192ED03ull,
                                          seed ^ 0x94D049BB133111EBull, seed ^ 0xBF58476D1CE4E5B9ull};
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        for (int lane = 0; lane < 4; ++lane) {
            state[lane] ^= state[lane] << 13;
            state[lane] ^= state[lane] >> 7;
            state[lane] ^= state[lane] << 17;
            const float u = static_cast<float>(state[lane] >> 40) * (1.f / 16777216.f);
            mask[i + static_cast<std::size_t>(lane)] = u >= p ? 1.f : 0.f;
        }
    }
    std::uint64_t tail = state[0];
    for (; i < n; ++i) {
        tail ^= tail << 13;
        tail ^= tail >> 7;
        tail ^= tail << 17;
        const float u = static_cast<float>(tail >> 40) * (1.f / 16777216.f);
        mask[i] = u >= p ? 1.f : 0.f;
    }
}

}  // namespace nexus_model::kernels
