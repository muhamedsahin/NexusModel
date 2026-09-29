#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nexus_model/core/kernels.hpp>
#include <nexus_model/core/tensor.hpp>
#include <string>
#include <vector>
using namespace nexus_model;
template <class F> void measure(const std::string &id, F fn, Tensor &output) {
  constexpr int warmup = 20, samples = 30, iterations = 5;
  for (int i = 0; i < warmup; ++i)
    fn();
  std::vector<double> times;
  for (int s = 0; s < samples; ++s) {
    const auto a = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      fn();
    const auto b = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(b - a).count() /
                    iterations);
  }
  std::ofstream values("benchmark-data/" + id + ".bin", std::ios::binary);
  values.write(reinterpret_cast<const char *>(output.data()),
               static_cast<std::streamsize>(output.numel() * sizeof(float)));
  std::cout << "{\"id\":\"" << id << "\",\"samples\":[";
  for (std::size_t i = 0; i < times.size(); ++i)
    std::cout << (i ? "," : "") << std::setprecision(12) << times[i];
  std::cout << "]}" << std::endl;
}
void fill_pattern(Tensor &t, int mult, int offset) {
  for (std::size_t i = 0; i < t.numel(); ++i)
    t[i] = static_cast<float>(
               (static_cast<int>(i % 101) * mult + offset) % 101 - 50) /
           100.f;
}
int main(int argc, char **argv) {
  std::filesystem::create_directories("benchmark-data");
  const std::string selected = argc > 1 ? argv[1] : "";
  int measured = 0;
  kernels::set_force_scalar(false);
  for (auto shape : std::vector<std::vector<int>>{{1, 256, 256},
                                                  {7, 129, 33},
                                                  {32, 256, 256},
                                                  {96, 768, 192},
                                                  {128, 512, 512},
                                                  {256, 1024, 1024}}) {
    int m = shape[0], k = shape[1], n = shape[2];
    const std::string id = "gemm_" + std::to_string(m) + "_" +
                           std::to_string(k) + "_" + std::to_string(n);
    if (!selected.empty() && selected != id)
      continue;
    ++measured;
    auto x = Tensor::zeros(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(k)});
    auto w = Tensor::zeros(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(k)});
    auto y = Tensor::zeros(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(n)});
    fill_pattern(x, 17, 3);
    fill_pattern(w, 13, 7);
    measure(
        id,
        [&] {
          kernels::linear_forward(x.data(), w.data(), nullptr, y.data(), m, k,
                                  n);
        },
        y);
  }
  for (int size : {1 << 16, 1 << 20, 1 << 22}) {
    const std::string id = "relu_" + std::to_string(size);
    if (!selected.empty() && selected != id)
      continue;
    ++measured;
    auto x = Tensor::zeros({static_cast<std::size_t>(size)});
    auto y = Tensor::zeros({static_cast<std::size_t>(size)});
    fill_pattern(x, 17, 3);
    measure(id, [&] { kernels::relu(x.data(), y.data(), x.numel()); }, y);
  }
  return measured ? 0 : 1;
}
