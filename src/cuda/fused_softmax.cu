#include "nexus_model/cuda/fused_softmax.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace nexus_model::cuda {
namespace {

__global__ void softmax_kernel(const float* x, float* y, int cols) {
  const int row = blockIdx.x;
  const float* in = x + static_cast<std::size_t>(row) * cols;
  float* out = y + static_cast<std::size_t>(row) * cols;
  float max_v = -FLT_MAX;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    max_v = fmaxf(max_v, in[i]);
  }
  __shared__ float red[32];
  // One warp reduction is enough for the block sizes this launcher uses.
  max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, 16));
  max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, 8));
  max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, 4));
  max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, 2));
  max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, 1));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = max_v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
    const int warps = (blockDim.x + 31) >> 5;
    for (int i = 1; i < warps; ++i) m = fmaxf(m, red[i]);
    red[0] = m;
  }
  __syncthreads();
  max_v = red[0];
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += expf(in[i] - max_v);
  sum += __shfl_xor_sync(0xffffffff, sum, 16);
  sum += __shfl_xor_sync(0xffffffff, sum, 8);
  sum += __shfl_xor_sync(0xffffffff, sum, 4);
  sum += __shfl_xor_sync(0xffffffff, sum, 2);
  sum += __shfl_xor_sync(0xffffffff, sum, 1);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    const int warps = (blockDim.x + 31) >> 5;
    for (int i = 0; i < warps; ++i) s += red[i];
    red[0] = s;
  }
  __syncthreads();
  const float inv = 1.f / red[0];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = expf(in[i] - max_v) * inv;
}

}  // namespace

void fused_softmax(const float* x, float* y, int rows, int cols) {
  softmax_kernel<<<rows, 128>>>(x, y, cols);
}

}  // namespace nexus_model::cuda
