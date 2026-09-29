#pragma once

#include <cstddef>

namespace nexus_model::kernels::detail {
void fill(float* y, float value, std::size_t n);
void add(const float* a, const float* b, float* y, std::size_t n);
void mul(const float* a, const float* b, float* y, std::size_t n);
void scale(const float* x, float alpha, float* y, std::size_t n);
void axpy(float* y, const float* x, float alpha, int n);
float dot(const float* a, const float* b, int n);
void relu(const float* x, float* y, std::size_t n);
void relu_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void leaky_relu(const float* x, float* y, std::size_t n, float slope);
void leaky_relu_bwd(const float* x, const float* dy, float* dx, std::size_t n, float slope);
void sigmoid(const float* x, float* y, std::size_t n);
void sigmoid_bwd(const float* y, const float* dy, float* dx, std::size_t n);
void tanh_fwd(const float* x, float* y, std::size_t n);
void tanh_bwd(const float* y, const float* dy, float* dx, std::size_t n);
void gelu(const float* x, float* y, std::size_t n);
void gelu_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void gelu_tanh(const float* x, float* y, std::size_t n);
void gelu_tanh_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void silu(const float* x, float* y, std::size_t n);
void silu_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void elu(const float* x, float* y, std::size_t n, float alpha);
void elu_bwd(const float* x, const float* y, const float* dy, float* dx, std::size_t n, float alpha);
void mish(const float* x, float* y, std::size_t n);
void mish_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void softmax(const float* x, float* y, int rows, int cols);
void softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols);
void log_softmax(const float* x, float* y, int rows, int cols);
void log_softmax_bwd(const float* y, const float* dy, float* dx, int rows, int cols);
void linear_forward(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f);
void linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db, int rows, int in_f,
                     int out_f);
void layernorm(const float* x, const float* gamma, const float* beta, float* y, float* mean, float* rstd, int rows,
               int cols, float eps);
void layernorm_bwd(const float* x, const float* gamma, const float* mean, const float* rstd, const float* dy, float* dx,
                   float* dgamma, float* dbeta, int rows, int cols);
}

namespace nexus_model::kernels::avx2 {
void fill(float* y, float value, std::size_t n);
void add(const float* a, const float* b, float* y, std::size_t n);
void mul(const float* a, const float* b, float* y, std::size_t n);
void scale(const float* x, float alpha, float* y, std::size_t n);
void axpy(float* y, const float* x, float alpha, int n);
float dot(const float* a, const float* b, int n);
void relu(const float* x, float* y, std::size_t n);
void relu_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void leaky_relu(const float* x, float* y, std::size_t n, float slope);
void leaky_relu_bwd(const float* x, const float* dy, float* dx, std::size_t n, float slope);
void sigmoid(const float* x, float* y, std::size_t n);
void sigmoid_bwd(const float* y, const float* dy, float* dx, std::size_t n);
void tanh_fwd(const float* x, float* y, std::size_t n);
void gelu_tanh(const float* x, float* y, std::size_t n);
void gelu_tanh_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void silu(const float* x, float* y, std::size_t n);
void silu_bwd(const float* x, const float* dy, float* dx, std::size_t n);
void softmax(const float* x, float* y, int rows, int cols);
void linear_forward(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f);
void linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db, int rows, int in_f,
                     int out_f);
void layernorm(const float* x, const float* gamma, const float* beta, float* y, float* mean, float* rstd, int rows,
               int cols, float eps);
}

namespace nexus_model::kernels::avx512 {
void relu(const float* x, float* y, std::size_t n);
void linear_forward(const float* x, const float* w, const float* bias, float* y, int rows, int in_f, int out_f);
float dot(const float* a, const float* b, int n);
}
