#include "nexus_model/core/kernels.hpp"
#include "nexus_model/cuda/activation_kernels.cuh"
#include "nexus_model/cuda/fused_layernorm.cuh"
#include "nexus_model/cuda/fused_softmax.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void check_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << what << ": cublas status " << static_cast<int>(status) << "\n";
    std::exit(1);
  }
}

struct DeviceBuf {
  float* ptr = nullptr;
  explicit DeviceBuf(std::size_t n) {
    check_cuda(cudaMalloc(&ptr, n * sizeof(float)), "cudaMalloc");
  }
  ~DeviceBuf() { cudaFree(ptr); }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

double gpu_ms(const std::function<void()>& launch, int warm, int iters) {
  for (int i = 0; i < warm; ++i) {
    launch();
    check_cuda(cudaDeviceSynchronize(), "warmup sync");
  }
  cudaEvent_t start{};
  cudaEvent_t stop{};
  check_cuda(cudaEventCreate(&start), "event");
  check_cuda(cudaEventCreate(&stop), "event");
  check_cuda(cudaEventRecord(start), "record");
  for (int i = 0; i < iters; ++i) launch();
  check_cuda(cudaEventRecord(stop), "record");
  check_cuda(cudaEventSynchronize(stop), "sync");
  float ms = 0.f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms) / iters;
}

double cpu_ms(const std::function<void()>& fn, int warm, int iters) {
  for (int i = 0; i < warm; ++i) fn();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count() / iters;
}

void fill_pattern(std::vector<float>& values) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);
  }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

void report(const std::string& name, const std::string& shape, double cpu, double gpu, float err) {
  std::cout << name << "," << shape << "," << cpu << "," << gpu << "," << err << "\n" << std::flush;
  if (err > 2e-3f) {
    std::cerr << name << " GPU sonucu CPU'dan sapıyor: " << err << "\n";
    std::exit(1);
  }
}

void bench_relu(std::size_t n) {
  std::vector<float> host(n), cpu_out(n), gpu_out(n);
  fill_pattern(host);
  nexus_model::kernels::relu(host.data(), cpu_out.data(), n);
  DeviceBuf x(n);
  DeviceBuf y(n);
  check_cuda(cudaMemcpy(x.ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  const double cpu = cpu_ms([&] { nexus_model::kernels::relu(host.data(), cpu_out.data(), n); }, 2, 8);
  const double gpu = gpu_ms([&] { nexus_model::cuda::relu(x.ptr, y.ptr, static_cast<int>(n)); }, 5, 20);
  check_cuda(cudaMemcpy(gpu_out.data(), y.ptr, n * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
  report("relu", std::to_string(n), cpu, gpu, max_abs_diff(cpu_out, gpu_out));
}

void bench_gelu(std::size_t n) {
  std::vector<float> host(n), cpu_out(n), gpu_out(n);
  fill_pattern(host);
  nexus_model::kernels::gelu_tanh(host.data(), cpu_out.data(), n);
  DeviceBuf x(n);
  DeviceBuf y(n);
  check_cuda(cudaMemcpy(x.ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  const double cpu = cpu_ms([&] { nexus_model::kernels::gelu_tanh(host.data(), cpu_out.data(), n); }, 2, 8);
  const double gpu = gpu_ms([&] { nexus_model::cuda::gelu_tanh(x.ptr, y.ptr, static_cast<int>(n)); }, 5, 20);
  check_cuda(cudaMemcpy(gpu_out.data(), y.ptr, n * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
  report("gelu_tanh", std::to_string(n), cpu, gpu, max_abs_diff(cpu_out, gpu_out));
}

void bench_softmax(int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<float> host(n), cpu_out(n), gpu_out(n);
  fill_pattern(host);
  nexus_model::kernels::softmax(host.data(), cpu_out.data(), rows, cols);
  DeviceBuf x(n);
  DeviceBuf y(n);
  check_cuda(cudaMemcpy(x.ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  const double cpu = cpu_ms([&] { nexus_model::kernels::softmax(host.data(), cpu_out.data(), rows, cols); }, 2, 6);
  const double gpu = gpu_ms([&] { nexus_model::cuda::fused_softmax(x.ptr, y.ptr, rows, cols); }, 5, 20);
  check_cuda(cudaMemcpy(gpu_out.data(), y.ptr, n * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
  report("softmax", std::to_string(rows) + "x" + std::to_string(cols), cpu, gpu, max_abs_diff(cpu_out, gpu_out));
}

void bench_layernorm(int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<float> host(n), gamma(cols, 1.f), beta(cols, 0.f), cpu_out(n), gpu_out(n);
  std::vector<float> mean(rows), rstd(rows);
  fill_pattern(host);
  for (int i = 0; i < cols; ++i) gamma[i] = 1.f + 0.01f * static_cast<float>(i % 5);
  nexus_model::kernels::layernorm(host.data(), gamma.data(), beta.data(), cpu_out.data(), mean.data(), rstd.data(), rows, cols, 1e-5f);
  DeviceBuf x(n);
  DeviceBuf g(cols);
  DeviceBuf b(cols);
  DeviceBuf y(n);
  check_cuda(cudaMemcpy(x.ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  check_cuda(cudaMemcpy(g.ptr, gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  check_cuda(cudaMemcpy(b.ptr, beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  const double cpu = cpu_ms([&] {
    nexus_model::kernels::layernorm(host.data(), gamma.data(), beta.data(), cpu_out.data(), mean.data(), rstd.data(), rows, cols, 1e-5f);
  }, 2, 6);
  const double gpu = gpu_ms([&] { nexus_model::cuda::fused_layernorm(x.ptr, g.ptr, b.ptr, y.ptr, rows, cols, 1e-5f); }, 5, 20);
  check_cuda(cudaMemcpy(gpu_out.data(), y.ptr, n * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
  report("layernorm", std::to_string(rows) + "x" + std::to_string(cols), cpu, gpu, max_abs_diff(cpu_out, gpu_out));
}

void bench_gemm(cublasHandle_t handle, int rows, int in_f, int out_f) {
  std::vector<float> x(static_cast<std::size_t>(rows) * in_f);
  std::vector<float> w(static_cast<std::size_t>(out_f) * in_f);
  std::vector<float> cpu_y(static_cast<std::size_t>(rows) * out_f);
  std::vector<float> gpu_y(cpu_y.size());
  fill_pattern(x);
  fill_pattern(w);
  nexus_model::kernels::linear_forward(x.data(), w.data(), nullptr, cpu_y.data(), rows, in_f, out_f);

  DeviceBuf dx(x.size());
  DeviceBuf dw(w.size());
  DeviceBuf dy(cpu_y.size());
  check_cuda(cudaMemcpy(dx.ptr, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  check_cuda(cudaMemcpy(dw.ptr, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d");

  const float alpha = 1.f;
  const float beta = 0.f;
  auto launch = [&] {
    check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_f, rows, in_f, &alpha, dw.ptr, in_f, dx.ptr, in_f, &beta, dy.ptr, out_f), "sgemm");
  };
  const double cpu = cpu_ms([&] { nexus_model::kernels::linear_forward(x.data(), w.data(), nullptr, cpu_y.data(), rows, in_f, out_f); }, 1, 4);
  const double gpu = gpu_ms(launch, 3, 10);
  check_cuda(cudaMemcpy(gpu_y.data(), dy.ptr, gpu_y.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h");
  report("gemm", std::to_string(rows) + "x" + std::to_string(in_f) + "x" + std::to_string(out_f), cpu, gpu, max_abs_diff(cpu_y, gpu_y));
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "device");
  std::cout << "device," << prop.name << ",,\n";
  std::cout << "op,shape,cpu_ms,gpu_ms,max_abs_err\n" << std::flush;

  bench_relu(1 << 20);
  bench_relu(1 << 24);
  bench_gelu(1 << 20);
  bench_gelu(1 << 24);
  bench_softmax(256, 256);
  bench_softmax(1024, 1024);
  bench_layernorm(256, 256);
  bench_layernorm(1024, 1024);

  cublasHandle_t handle{};
  check_cublas(cublasCreate(&handle), "cublasCreate");
  bench_gemm(handle, 32, 256, 256);
  bench_gemm(handle, 1024, 1024, 1024);
  bench_gemm(handle, 4096, 4096, 4096);
  cublasDestroy(handle);
  return 0;
}
