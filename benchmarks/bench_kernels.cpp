#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/tensor.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace nexus_model;
template <class F> void measure(const char *name, F fn) {
  for (int i = 0; i < 20; ++i)
    fn();
  std::vector<double> samples;
  for (int s = 0; s < 30; ++s) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i)
      fn();
    samples.push_back(std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count() /
                      5);
  }
  std::sort(samples.begin(), samples.end());
  std::cout << std::setprecision(12) << name << ','
            << (samples[14] + samples[15]) / 2 << ',' << samples[28] << ',';
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::cout << (i ? ";" : "") << samples[i];
  std::cout << '\n';
}
int main() {
  auto x = Tensor::zeros({128, 512}), w = Tensor::zeros({512, 512}),
       y = Tensor::zeros({128, 512});
  auto dx = x.clone(), dw = w.clone(), mean = Tensor::zeros({128}),
       rs = mean.clone();
  for (std::size_t i = 0; i < x.numel(); ++i)
    x[i] = float(int(i % 101) - 50) / 100.f;
  for (std::size_t i = 0; i < w.numel(); ++i)
    w[i] = float(int(i % 97) - 48) / 100.f;
  std::cout << "operation,median_ms,p95_nearest_rank_ms,samples_ms\n";
  measure("linear_forward_128_512_512", [&] {
    kernels::linear_forward(x.data(), w.data(), nullptr, y.data(), 128, 512,
                            512);
  });
  measure("linear_backward_128_512_512", [&] {
    dx.zero();
    dw.zero();
    kernels::linear_backward(x.data(), w.data(), y.data(), dx.data(), dw.data(),
                             nullptr, 128, 512, 512);
  });
  measure("softmax_128_512",
          [&] { kernels::softmax(x.data(), y.data(), 128, 512); });
  measure("layernorm_128_512", [&] {
    kernels::layernorm(x.data(), nullptr, nullptr, y.data(), mean.data(),
                       rs.data(), 128, 512, 1e-5f);
  });
  std::cerr << "checksum=" << y[0] << '\n';
}
