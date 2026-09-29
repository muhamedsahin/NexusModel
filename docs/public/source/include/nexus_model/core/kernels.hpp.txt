#pragma once

/**
 * @file kernels.hpp
 * @brief Katmanların çağırdığı sayısal çekirdekler.
 *
 * Hepsi satır-major float32 üstünde çalışır. Sembol ilk çağrıda CPU'ya göre
 * AVX-512, AVX2 veya skaler sürüme bağlanır. `linear_backward` dX, dW ve dB
 * içine **biriktirir** (sıfırlamaz); parametre gradyanı `zero_grad` ile temizlenir.
 */

#include <cstddef>
#include <cstdint>

namespace nexus_model {
namespace kernels {

struct KernelTable {
  void (*fill)(float* y, float value, std::size_t n);
  void (*add)(const float* a, const float* b, float* y, std::size_t n);
  void (*mul)(const float* a, const float* b, float* y, std::size_t n);
  void (*scale)(const float* x, float alpha, float* y, std::size_t n);
  void (*axpy)(float* y, const float* x, float alpha, int n);
  float (*dot)(const float* a, const float* b, int n);
  void (*relu)(const float* x, float* y, std::size_t n);
  void (*relu_bwd)(const float* x, const float* dy, float* dx, std::size_t n);
  void (*leaky_relu)(const float* x, float* y, std::size_t n, float slope);
  void (*leaky_relu_bwd)(const float* x, const float* dy, float* dx, std::size_t n, float slope);
  void (*sigmoid)(const float* x, float* y, std::size_t n);
  void (*sigmoid_bwd)(const float* y, const float* dy, float* dx, std::size_t n);
  void (*tanh_fwd)(const float* x, float* y, std::size_t n);
  void (*tanh_bwd)(const float* y, const float* dy, float* dx, std::size_t n);
  void (*gelu)(const float* x, float* y, std::size_t n);
  void (*gelu_bwd)(const float* x, const float* dy, float* dx, std::size_t n);
  void (*gelu_tanh)(const float* x, float* y, std::size_t n);
  void (*gelu_tanh_bwd)(const float* x, const float* dy, float* dx, std::size_t n);
  void (*silu)(const float* x, float* y, std::size_t n);
  void (*silu_bwd)(const float* x, const float* dy, float* dx, std::size_t n);
  void (*elu)(const float* x, float* y, std::size_t n, float alpha);
  void (*elu_bwd)(const float* x, const float* y, const float* dy, float* dx, std::size_t n, float alpha);
  void (*mish)(const float* x, float* y, std::size_t n);
  void (*mish_bwd)(const float* x, const float* dy, float* dx, std::size_t n);
  void (*softmax)(const float* x, float* y, int rows, int cols);
  void (*softmax_bwd)(const float* y, const float* dy, float* dx, int rows, int cols);
  void (*log_softmax)(const float* x, float* y, int rows, int cols);
  void (*log_softmax_bwd)(const float* y, const float* dy, float* dx, int rows, int cols);
  void (*linear_forward)(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f);
  void (*linear_backward)(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db,
                          int rows, int in_f, int out_f);
  void (*layernorm)(const float* x, const float* gamma, const float* beta, float* y, float* mean, float* rstd,
                    int rows, int cols, float eps);
  void (*layernorm_bwd)(const float* x, const float* gamma, const float* mean, const float* rstd, const float* dy,
                        float* dx, float* dgamma, float* dbeta, int rows, int cols);
};

const KernelTable& cpu();
void set_force_scalar(bool force);

inline void fill(float* y, float value, std::size_t n) { cpu().fill(y, value, n); }
inline void add(const float* a, const float* b, float* y, std::size_t n) { cpu().add(a, b, y, n); }
inline void mul(const float* a, const float* b, float* y, std::size_t n) { cpu().mul(a, b, y, n); }
inline void scale(const float* x, float alpha, float* y, std::size_t n) { cpu().scale(x, alpha, y, n); }
inline void axpy(float* y, const float* x, float alpha, int n) { cpu().axpy(y, x, alpha, n); }
inline float dot(const float* a, const float* b, int n) { return cpu().dot(a, b, n); }
inline void relu(const float* x, float* y, std::size_t n) { cpu().relu(x, y, n); }
inline void relu_bwd(const float* x, const float* dy, float* dx, std::size_t n) { cpu().relu_bwd(x, dy, dx, n); }
inline void leaky_relu(const float* x, float* y, std::size_t n, float slope) { cpu().leaky_relu(x, y, n, slope); }
inline void leaky_relu_bwd(const float* x, const float* dy, float* dx, std::size_t n, float slope) {
  cpu().leaky_relu_bwd(x, dy, dx, n, slope);
}
inline void sigmoid(const float* x, float* y, std::size_t n) { cpu().sigmoid(x, y, n); }
inline void sigmoid_bwd(const float* y, const float* dy, float* dx, std::size_t n) { cpu().sigmoid_bwd(y, dy, dx, n); }
inline void tanh_fwd(const float* x, float* y, std::size_t n) { cpu().tanh_fwd(x, y, n); }
inline void tanh_bwd(const float* y, const float* dy, float* dx, std::size_t n) { cpu().tanh_bwd(y, dy, dx, n); }
inline void gelu(const float* x, float* y, std::size_t n) { cpu().gelu(x, y, n); }
inline void gelu_bwd(const float* x, const float* dy, float* dx, std::size_t n) { cpu().gelu_bwd(x, dy, dx, n); }
inline void gelu_tanh(const float* x, float* y, std::size_t n) { cpu().gelu_tanh(x, y, n); }
inline void gelu_tanh_bwd(const float* x, const float* dy, float* dx, std::size_t n) { cpu().gelu_tanh_bwd(x, dy, dx, n); }
inline void silu(const float* x, float* y, std::size_t n) { cpu().silu(x, y, n); }
inline void silu_bwd(const float* x, const float* dy, float* dx, std::size_t n) { cpu().silu_bwd(x, dy, dx, n); }
inline void elu(const float* x, float* y, std::size_t n, float alpha) { cpu().elu(x, y, n, alpha); }
inline void elu_bwd(const float* x, const float* y, const float* dy, float* dx, std::size_t n, float alpha) {
  cpu().elu_bwd(x, y, dy, dx, n, alpha);
}
inline void mish(const float* x, float* y, std::size_t n) { cpu().mish(x, y, n); }
inline void mish_bwd(const float* x, const float* dy, float* dx, std::size_t n) { cpu().mish_bwd(x, dy, dx, n); }
inline void softmax(const float* x, float* y, int rows, int cols) { cpu().softmax(x, y, rows, cols); }
inline void softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols) {
  cpu().softmax_bwd(y, dy, dx, rows, cols);
}
inline void log_softmax(const float* x, float* y, int rows, int cols) { cpu().log_softmax(x, y, rows, cols); }
inline void log_softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols) {
  cpu().log_softmax_bwd(y, dy, dx, rows, cols);
}
inline void linear_forward(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f) {
  cpu().linear_forward(x, w, bias, y, rows, in_f, out_f);
}
inline void linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db, int rows,
                            int in_f, int out_f) {
  cpu().linear_backward(x, w, dy, dx, dw, db, rows, in_f, out_f);
}
inline void layernorm(const float* x, const float* gamma, const float* beta, float* y, float* mean, float* rstd, int rows,
                      int cols, float eps) {
  cpu().layernorm(x, gamma, beta, y, mean, rstd, rows, cols, eps);
}
inline void layernorm_bwd(const float* x, const float* gamma, const float* mean, const float* rstd, const float* dy,
                          float* dx, float* dgamma, float* dbeta, int rows, int cols) {
  cpu().layernorm_bwd(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, rows, cols);
}

// --- her zaman skaler, tek geçişli yapısal çekirdekler (conv / norm / rnn) ---

void batch_norm(const float* x, const float* gamma, const float* beta, float* y, float* save_mean, float* save_rstd,
                float* running_mean, float* running_var, int outer, int channels, int inner, float eps, float momentum,
                bool training, bool track_running);

void batch_norm_bwd(const float* x, const float* gamma, const float* save_mean, const float* save_rstd, const float* dy,
                    float* dx, float* dgamma, float* dbeta, int outer, int channels, int inner);

void group_norm(const float* x, const float* gamma, const float* beta, float* y, float* save_mean, float* save_rstd,
                int n, int channels, int inner, int groups, float eps);
void group_norm_bwd(const float* x, const float* gamma, const float* save_mean, const float* save_rstd, const float* dy,
                    float* dx, float* dgamma, float* dbeta, int n, int channels, int inner, int groups);

struct ConvDesc {
  int n = 0;
  int c = 0;
  int k = 0;
  int spatial_in[3]{};
  int spatial_out[3]{};
  int kernel[3]{};
  int stride[3]{};
  int pad[3]{};
  int dilation[3]{};
  int groups = 1;
  int rank = 2;  // 1, 2 or 3
};

int conv_out_size(int input, int pad, int dilation, int kernel, int stride);
std::size_t conv_col_size(const ConvDesc& d);
void conv_forward(const float* x, const float* w, const float* bias, float* y, float* col, const ConvDesc& d);
void conv_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db, float* col,
                   const ConvDesc& d);

void max_pool2d(const float* x, float* y, int* index, int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph,
                int pw);
void max_pool2d_bwd(const float* dy, const int* index, float* dx, int n, int c, int oh, int ow, int in_spatial);
void avg_pool2d(const float* x, float* y, int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph, int pw,
                bool count_include_pad);
void avg_pool2d_bwd(const float* dy, float* dx, int n, int c, int h, int w, int kh, int kw, int sh, int sw, int ph, int pw,
                    bool count_include_pad);

void embedding_forward(const float* table, const float* indices, float* y, int rows, int dim, int vocab, int padding_idx);
void embedding_backward(const float* grad, const float* indices, float* dtable, int rows, int dim, int vocab,
                        int padding_idx);

void dropout_mask(float* mask, std::size_t n, float p, std::uint64_t seed);

}  // namespace kernels
}  // namespace nexus_model
