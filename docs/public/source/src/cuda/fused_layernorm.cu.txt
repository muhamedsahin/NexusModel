#include "nexus_model/cuda/fused_layernorm.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace nexus_model::cuda {
namespace {

__global__ void layernorm_kernel(const float* x, const float* gamma, const float* beta, float* y, int cols, float eps) {
  const int row = blockIdx.x;
  const float* in = x + static_cast<std::size_t>(row) * cols;
  float* out = y + static_cast<std::size_t>(row) * cols;
  float sum = 0.f;
  float sumsq = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += in[i];
    sumsq += in[i] * in[i];
  }
  __shared__ float red[64];
  sum += __shfl_xor_sync(0xffffffff, sum, 16);
  sum += __shfl_xor_sync(0xffffffff, sum, 8);
  sum += __shfl_xor_sync(0xffffffff, sum, 4);
  sum += __shfl_xor_sync(0xffffffff, sum, 2);
  sum += __shfl_xor_sync(0xffffffff, sum, 1);
  sumsq += __shfl_xor_sync(0xffffffff, sumsq, 16);
  sumsq += __shfl_xor_sync(0xffffffff, sumsq, 8);
  sumsq += __shfl_xor_sync(0xffffffff, sumsq, 4);
  sumsq += __shfl_xor_sync(0xffffffff, sumsq, 2);
  sumsq += __shfl_xor_sync(0xffffffff, sumsq, 1);
  if ((threadIdx.x & 31) == 0) {
    red[threadIdx.x >> 5] = sum;
    red[32 + (threadIdx.x >> 5)] = sumsq;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    float q = 0.f;
    const int warps = (blockDim.x + 31) >> 5;
    for (int i = 0; i < warps; ++i) {
      s += red[i];
      q += red[32 + i];
    }
    red[0] = s;
    red[1] = q;
  }
  __syncthreads();
  const float mean = red[0] / static_cast<float>(cols);
  float var = red[1] / static_cast<float>(cols) - mean * mean;
  if (var < 0.f) var = 0.f;
  const float inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float g = gamma != nullptr ? gamma[i] : 1.f;
    const float b = beta != nullptr ? beta[i] : 0.f;
    out[i] = (in[i] - mean) * inv * g + b;
  }
}

}  // namespace

void fused_layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps) {
  layernorm_kernel<<<rows, 128>>>(x, gamma, beta, y, cols, eps);
}

}  // namespace nexus_model::cuda
