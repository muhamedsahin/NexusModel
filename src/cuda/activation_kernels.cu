#include "nexus_model/cuda/activation_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace nexus_model::cuda {
namespace {

__global__ void relu_kernel(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

__global__ void gelu_kernel(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  const float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
  y[i] = 0.5f * v * (1.f + tanhf(inner));
}

}  // namespace

void relu(const float* x, float* y, int n) {
  const int threads = 256;
  relu_kernel<<<(n + threads - 1) / threads, threads>>>(x, y, n);
}

void gelu_tanh(const float* x, float* y, int n) {
  const int threads = 256;
  gelu_kernel<<<(n + threads - 1) / threads, threads>>>(x, y, n);
}

}  // namespace nexus_model::cuda
