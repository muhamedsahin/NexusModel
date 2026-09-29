#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace nexus_model::kernels::detail {

void fill(float* y, float value, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = value;
}
void add(const float* a, const float* b, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = a[i] + b[i];
}
void mul(const float* a, const float* b, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = a[i] * b[i];
}
void scale(const float* x, float alpha, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * alpha;
}
void axpy(float* y, const float* x, float alpha, int n) {
  for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}
float dot(const float* a, const float* b, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return static_cast<float>(acc);
}

void relu(const float* x, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
void relu_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dx[i] = x[i] > 0.f ? dy[i] : 0.f;
}
void leaky_relu(const float* x, float* y, std::size_t n, float slope) {
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : slope * x[i];
}
void leaky_relu_bwd(const float* x, const float* dy, float* dx, std::size_t n, float slope) {
  for (std::size_t i = 0; i < n; ++i) dx[i] = dy[i] * (x[i] > 0.f ? 1.f : slope);
}

float sigmoid_scalar(float x) {
  if (x >= 0.f) {
    const float z = std::exp(-x);
    return 1.f / (1.f + z);
  }
  const float z = std::exp(x);
  return z / (1.f + z);
}
float softplus_scalar(float x) {
  if (x > 0.f) return x + std::log1p(std::exp(-x));
  return std::log1p(std::exp(x));
}

void sigmoid(const float* x, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = sigmoid_scalar(x[i]);
}
void sigmoid_bwd(const float* y, const float* dy, float* dx, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dx[i] = dy[i] * y[i] * (1.f - y[i]);
}
void tanh_fwd(const float* x, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = std::tanh(x[i]);
}
void tanh_bwd(const float* y, const float* dy, float* dx, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dx[i] = dy[i] * (1.f - y[i] * y[i]);
}

void gelu(const float* x, float* y, std::size_t n) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  for (std::size_t i = 0; i < n; ++i) y[i] = 0.5f * x[i] * (1.f + std::erf(x[i] * kInvSqrt2));
}
void gelu_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  constexpr float kInvSqrt2Pi = 0.3989422804014327f;
  for (std::size_t i = 0; i < n; ++i) {
    const float cdf = 0.5f * (1.f + std::erf(x[i] * kInvSqrt2));
    const float pdf = kInvSqrt2Pi * std::exp(-0.5f * x[i] * x[i]);
    dx[i] = dy[i] * (cdf + x[i] * pdf);
  }
}

void gelu_tanh(const float* x, float* y, std::size_t n) {
  constexpr float kC = 0.7978845608028654f;
  for (std::size_t i = 0; i < n; ++i) {
    const float x3 = x[i] * x[i] * x[i];
    const float u = kC * (x[i] + 0.044715f * x3);
    y[i] = 0.5f * x[i] * (1.f + std::tanh(u));
  }
}
void gelu_tanh_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  constexpr float kC = 0.7978845608028654f;
  for (std::size_t i = 0; i < n; ++i) {
    const float x2 = x[i] * x[i];
    const float u = kC * (x[i] + 0.044715f * x2 * x[i]);
    const float t = std::tanh(u);
    const float du = kC * (1.f + 3.f * 0.044715f * x2);
    dx[i] = dy[i] * (0.5f * (1.f + t) + 0.5f * x[i] * (1.f - t * t) * du);
  }
}

void silu(const float* x, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * sigmoid_scalar(x[i]);
}
void silu_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float s = sigmoid_scalar(x[i]);
    dx[i] = dy[i] * s * (1.f + x[i] * (1.f - s));
  }
}
void elu(const float* x, float* y, std::size_t n, float alpha) {
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : alpha * (std::exp(x[i]) - 1.f);
}
void elu_bwd(const float* x, const float* y, const float* dy, float* dx, std::size_t n, float alpha) {
  for (std::size_t i = 0; i < n; ++i) dx[i] = dy[i] * (x[i] > 0.f ? 1.f : (y[i] + alpha));
}
void mish(const float* x, float* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * std::tanh(softplus_scalar(x[i]));
}
void mish_bwd(const float* x, const float* dy, float* dx, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float sp = softplus_scalar(x[i]);
    const float t = std::tanh(sp);
    const float s = sigmoid_scalar(x[i]);
    dx[i] = dy[i] * (t + x[i] * (1.f - t * t) * s);
  }
}

void softmax(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* out = y + static_cast<std::size_t>(r) * cols;
    float m = row[0];
    for (int c = 1; c < cols; ++c) m = std::max(m, row[c]);
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      const float e = std::exp(row[c] - m);
      out[c] = e;
      sum += e;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int c = 0; c < cols; ++c) out[c] *= inv;
  }
}
void softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* ys = y + static_cast<std::size_t>(r) * cols;
    const float* gs = dy + static_cast<std::size_t>(r) * cols;
    float* os = dx + static_cast<std::size_t>(r) * cols;
    double dot = 0.0;
    for (int c = 0; c < cols; ++c) dot += static_cast<double>(gs[c]) * static_cast<double>(ys[c]);
    const float d = static_cast<float>(dot);
    for (int c = 0; c < cols; ++c) os[c] = ys[c] * (gs[c] - d);
  }
}
void log_softmax(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* out = y + static_cast<std::size_t>(r) * cols;
    float m = row[0];
    for (int c = 1; c < cols; ++c) m = std::max(m, row[c]);
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp(row[c] - m);
    const float log_z = m + std::log(static_cast<float>(sum));
    for (int c = 0; c < cols; ++c) out[c] = row[c] - log_z;
  }
}
void log_softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* ys = y + static_cast<std::size_t>(r) * cols;
    const float* gs = dy + static_cast<std::size_t>(r) * cols;
    float* os = dx + static_cast<std::size_t>(r) * cols;
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += gs[c];
    const float s = static_cast<float>(sum);
    for (int c = 0; c < cols; ++c) os[c] = gs[c] - std::exp(ys[c]) * s;
  }
}

void linear_forward(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * in_f;
    float* yr = y + static_cast<std::size_t>(r) * out_f;
    for (int o = 0; o < out_f; ++o) {
      yr[o] = dot(xr, w + static_cast<std::size_t>(o) * in_f, in_f) + (bias != nullptr ? bias[o] : 0.f);
    }
  }
}

void linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db, int rows, int in_f,
                     int out_f) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * in_f;
    const float* gr = dy + static_cast<std::size_t>(r) * out_f;
    float* dxr = dx != nullptr ? dx + static_cast<std::size_t>(r) * in_f : nullptr;
    for (int o = 0; o < out_f; ++o) {
      const float g = gr[o];
      if (dxr != nullptr) axpy(dxr, w + static_cast<std::size_t>(o) * in_f, g, in_f);
      if (dw != nullptr) axpy(dw + static_cast<std::size_t>(o) * in_f, xr, g, in_f);
      if (db != nullptr) db[o] += g;
    }
  }
}

void layernorm(const float* x, const float* gamma, const float* beta, float* y, float* mean, float* rstd, int rows,
               int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* out = y + static_cast<std::size_t>(r) * cols;
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += row[c];
    const float m = static_cast<float>(sum / cols);
    double var = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double d = static_cast<double>(row[c]) - m;
      var += d * d;
    }
    const float rs = 1.f / std::sqrt(static_cast<float>(var / cols) + eps);
    mean[r] = m;
    rstd[r] = rs;
    for (int c = 0; c < cols; ++c) {
      const float xhat = (row[c] - m) * rs;
      const float g = gamma != nullptr ? gamma[c] : 1.f;
      const float b = beta != nullptr ? beta[c] : 0.f;
      out[c] = xhat * g + b;
    }
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
    double sum_dy = 0.0;
    double sum_dy_xhat = 0.0;
    for (int c = 0; c < cols; ++c) {
      const float g = gamma != nullptr ? gamma[c] : 1.f;
      const float xhat = (row[c] - m) * rs;
      const float dys = gr[c] * g;
      sum_dy += dys;
      sum_dy_xhat += static_cast<double>(dys) * xhat;
      if (dgamma != nullptr) dgamma[c] += gr[c] * xhat;
      if (dbeta != nullptr) dbeta[c] += gr[c];
    }
    const float inv_m = 1.f / static_cast<float>(cols);
    for (int c = 0; c < cols; ++c) {
      const float g = gamma != nullptr ? gamma[c] : 1.f;
      const float xhat = (row[c] - m) * rs;
      const float dys = gr[c] * g;
      dxr[c] = rs * inv_m * (static_cast<float>(cols) * dys - static_cast<float>(sum_dy) - xhat * static_cast<float>(sum_dy_xhat));
    }
  }
}

}  // namespace nexus_model::kernels::detail
