#include "nexus_model/layers/attention.hpp"
#include "nexus_model/layers/conv.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_model/activations/activations.hpp"
#include "nexus_model/core/kernels.hpp"

#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>

namespace {

template<class F> double time_ms(const F& fn, int, int) {
  for (int i=0;i<20;++i) fn();
  std::vector<double> samples;
  for (int sample=0;sample<30;++sample) {
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<5;++i) fn();
    samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/5);
  }
  std::sort(samples.begin(),samples.end());
  return (samples[14]+samples[15])/2;
}

}  // namespace

int main() {
  using namespace nexus_model;
  std::cout << "op,shape,ms\n" << std::flush;

  Linear linear(784, 256);
  Tensor x = Tensor::zeros({64, 784});
  Tensor g;
  auto y = linear.forward(x);
  g.resize_like(y);
  g.fill(1.f);
  const double linear_ms = time_ms([&] {
    linear.forward(x);
    linear.backward(g);
  }, 3, 10);
  std::cout << "linear_784_256,B64," << linear_ms << "\n";

  Linear wide(256, 256);
  Tensor xw = Tensor::zeros({32, 256});
  auto yw = wide.forward(xw);
  Tensor gw;
  gw.resize_like(yw);
  gw.fill(1.f);
  kernels::set_force_scalar(true);
  const double wide_scalar = time_ms([&] {
    wide.forward(xw);
    wide.backward(gw);
  }, 1, 4);
  kernels::set_force_scalar(false);
  const double wide_ms = time_ms([&] {
    wide.forward(xw);
    wide.backward(gw);
  }, 1, 4);
  std::cout << "linear_256_scalar,B32," << wide_scalar << "\n";
  std::cout << "linear_256_simd,B32," << wide_ms << "\n";

  ReLU relu;
  Tensor big = Tensor::zeros({1, 1 << 20});
  Tensor gb;
  gb.resize_like(big);
  gb.fill(1.f);
  kernels::set_force_scalar(true);
  const double relu_scalar = time_ms([&] { relu.forward(big); }, 2, 8);
  kernels::set_force_scalar(false);
  const double relu_simd = time_ms([&] { relu.forward(big); }, 2, 8);
  std::cout << "relu_1M_scalar,," << relu_scalar << "\n";
  std::cout << "relu_1M_simd,," << relu_simd << "\n";

  Conv2D conv(16, 32, 3, 1, 1);
  Tensor image = Tensor::zeros({8, 16, 28, 28});
  auto yc = conv.forward(image);
  Tensor gc;
  gc.resize_like(yc);
  gc.fill(1.f);
  const double conv_ms = time_ms([&] {
    conv.forward(image);
    conv.backward(gc);
  }, 1, 3);
  std::cout << "conv2d_16_32_k3,B8_28," << conv_ms << "\n";

  MultiHeadAttention attn(64, 4, 0.f, false);
  Tensor seq = Tensor::zeros({4, 32, 64});
  auto ya = attn.forward(seq);
  Tensor ga;
  ga.resize_like(ya);
  ga.fill(1.f);
  const double attn_ms = time_ms([&] {
    attn.forward(seq);
    attn.backward(ga);
  }, 1, 3);
  std::cout << "mha_64_h4,B4_T32," << attn_ms << "\n";
  return 0;
}
