#include "nexus_model/core/error.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nexus_model::kernels {
void max_pool2d(const float* x, float* y, int* index, int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph,
                int pw) {
  const int oh = conv_out_size(h, ph, 1, kh, sh);
  const int ow = conv_out_size(w, pw, 1, kw, sw);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      const float* src = x + (static_cast<std::size_t>(ni * c + ci) * h) * w;
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float best = -1e30f;
          int best_i = -1;
          for (int ky = 0; ky < kh; ++ky) {
            for (int kx = 0; kx < kw; ++kx) {
              const int iy = oy * sh - ph + ky;
              const int ix = ox * sw - pw + kx;
              if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
              const int flat = iy * w + ix;
              if (src[flat] > best) {
                best = src[flat];
                best_i = flat;
              }
            }
          }
          const std::size_t oidx = (((static_cast<std::size_t>(ni) * c + ci) * oh) + oy) * ow + ox;
          y[oidx] = best_i >= 0 ? best : 0.f;
          index[oidx] = best_i;
        }
      }
    }
  }
}

void max_pool2d_bwd(const float* dy, const int* index, float* dx, int n, int c, int oh, int ow, int in_spatial) {
  const std::size_t out_n = static_cast<std::size_t>(n) * c * oh * ow;
  for (std::size_t i = 0; i < out_n; ++i) {
    if (index[i] < 0) continue;
    const int ni = static_cast<int>(i / (static_cast<std::size_t>(c) * oh * ow));
    const int rem = static_cast<int>(i % (static_cast<std::size_t>(c) * oh * ow));
    const int ci = rem / (oh * ow);
    dx[(static_cast<std::size_t>(ni * c + ci) * in_spatial) + index[i]] += dy[i];
  }
}

void avg_pool2d(const float* x, float* y, int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph, int pw,
                bool count_include_pad) {
  const int oh = conv_out_size(h, ph, 1, kh, sh);
  const int ow = conv_out_size(w, pw, 1, kw, sw);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      const float* src = x + (static_cast<std::size_t>(ni * c + ci) * h) * w;
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float sum = 0.f;
          int valid = 0;
          for (int ky = 0; ky < kh; ++ky) {
            for (int kx = 0; kx < kw; ++kx) {
              const int iy = oy * sh - ph + ky;
              const int ix = ox * sw - pw + kx;
              if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
              sum += src[iy * w + ix];
              ++valid;
            }
          }
          const int denom = count_include_pad ? kh * kw : std::max(valid, 1);
          const std::size_t oidx = (((static_cast<std::size_t>(ni) * c + ci) * oh) + oy) * ow + ox;
          y[oidx] = valid == 0 ? 0.f : sum / static_cast<float>(denom);
        }
      }
    }
  }
}

void avg_pool2d_bwd(const float* dy, float* dx, int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph,
                    int pw, bool count_include_pad) {
  const int oh = conv_out_size(h, ph, 1, kh, sh);
  const int ow = conv_out_size(w, pw, 1, kw, sw);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      float* dst = dx + (static_cast<std::size_t>(ni * c + ci) * h) * w;
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          int valid = 0;
          for (int ky = 0; ky < kh; ++ky) {
            for (int kx = 0; kx < kw; ++kx) {
              const int iy = oy * sh - ph + ky;
              const int ix = ox * sw - pw + kx;
              if (iy >= 0 && iy < h && ix >= 0 && ix < w) ++valid;
            }
          }
          const int denom = count_include_pad ? kh * kw : std::max(valid, 1);
          const float share = dy[(((static_cast<std::size_t>(ni) * c + ci) * oh) + oy) * ow + ox] / static_cast<float>(denom);
          if (valid == 0) continue;
          for (int ky = 0; ky < kh; ++ky) {
            for (int kx = 0; kx < kw; ++kx) {
              const int iy = oy * sh - ph + ky;
              const int ix = ox * sw - pw + kx;
              if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
              dst[iy * w + ix] += share;
            }
          }
        }
      }
    }
  }
}


}  // namespace nexus_model::kernels
