#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/micro_tape.hpp"
#include "nexus_model/simd/kernel_decls.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = nexus_model::kernels;
namespace {
std::mt19937 rng(20260929);
std::vector<float> values(int n) {
  std::vector<float> v(n);
  std::uniform_real_distribution<float> dist(-.5f, .5f);
  for (auto &x : v)
    x = dist(rng);
  return v;
}
void check(double got, double want, const char *operation) {
  if (!std::isfinite(got) ||
      std::abs(got - want) > 2e-5 + 2e-4 * std::abs(want))
    throw std::runtime_error(std::string(operation) + ": got " +
                             std::to_string(got) + " expected " +
                             std::to_string(want));
}
void linear() {
  for (const auto &shape : std::vector<std::array<int, 3>>{{1, 1, 1},
                                                           {1, 257, 33},
                                                           {3, 31, 17},
                                                           {4, 16, 16},
                                                           {5, 17, 19},
                                                           {6, 128, 32},
                                                           {7, 129, 33},
                                                           {11, 255, 31},
                                                           {13, 257, 49},
                                                           {32, 512, 64},
                                                           {6, 0, 16}}) {
    const auto [m, depth, n] = shape;
    auto x = values(m * depth), w = values(n * depth), bias = values(n),
         dy = values(m * n);
    std::vector<float> y(m * n + 2, 12345.f);
    for (bool biased : {false, true}) {
      k::linear_forward(x.data(), w.data(), biased ? bias.data() : nullptr,
                        y.data() + 1, m, depth, n);
      if (y.front() != 12345.f || y.back() != 12345.f)
        throw std::runtime_error("output guard overwritten");
      for (int r = 0; r < m; ++r)
        for (int o = 0; o < n; ++o) {
          double sum = biased ? bias[o] : 0;
          for (int c = 0; c < depth; ++c)
            sum += double(x[r * depth + c]) * w[o * depth + c];
          check(y[1 + r * n + o], sum, "forward");
        }
    }
    // Every combination of optional gradients, including pre-existing values.
    for (int mask = 0; mask < 8; ++mask) {
      auto dx = values(m * depth), dw = values(n * depth), db = values(n);
      const auto ix = dx, iw = dw, ib = db;
      for (int repeat = 0; repeat < 2; ++repeat)
        k::linear_backward(x.data(), w.data(), dy.data(),
                           mask & 1 ? dx.data() : nullptr,
                           mask & 2 ? dw.data() : nullptr,
                           mask & 4 ? db.data() : nullptr, m, depth, n);
      for (int r = 0; r < m; ++r)
        for (int c = 0; c < depth; ++c) {
          double sum = ix[r * depth + c];
          if (mask & 1)
            for (int o = 0; o < n; ++o)
              sum += 2. * dy[r * n + o] * w[o * depth + c];
          check(dx[r * depth + c], sum, "dx accumulation");
        }
      for (int o = 0; o < n; ++o) {
        double sum = ib[o];
        if (mask & 4)
          for (int r = 0; r < m; ++r)
            sum += 2. * dy[r * n + o];
        check(db[o], sum, "db accumulation");
        for (int c = 0; c < depth; ++c) {
          double total = iw[o * depth + c];
          if (mask & 2)
            for (int r = 0; r < m; ++r)
              total += 2. * dy[r * n + o] * x[r * depth + c];
          check(dw[o * depth + c], total, "dw accumulation");
        }
      }
    }
  }
}
void reductions() {
  for (int n : {1, 3, 7, 8, 15, 16, 31, 129, 512, 1025})
    for (float offset : {0.f, 10000.f}) {
      auto x = values(n * 3), g = values(n), b = values(n), y = x, ref = x;
      for (float &v : x)
        v += offset;
      float mean[3], rs[3], rm[3], rr[3];
      for (bool affine : {false, true}) {
        k::layernorm(x.data(), affine ? g.data() : nullptr,
                     affine ? b.data() : nullptr, y.data(), mean, rs, 3, n,
                     1e-5f);
        k::detail::layernorm(x.data(), affine ? g.data() : nullptr,
                             affine ? b.data() : nullptr, ref.data(), rm, rr, 3,
                             n, 1e-5f);
        for (int i = 0; i < n * 3; ++i)
          check(y[i], ref[i], "layernorm");
        for (int i = 0; i < 3; ++i) {
          check(mean[i], rm[i], "mean");
          check(rs[i], rr[i], "rstd");
        }
      }
      k::softmax(x.data(), y.data(), 3, n);
      k::detail::softmax(x.data(), ref.data(), 3, n);
      for (int i = 0; i < n * 3; ++i)
        check(y[i], ref[i], "softmax");
    }
}
void convolution() {
  k::ConvDesc d;
  d.n = 2;
  d.c = 6;
  d.k = 10;
  d.groups = 2;
  d.rank = 2;
  d.spatial_in[0] = 5;
  d.spatial_in[1] = 7;
  d.spatial_out[0] = 5;
  d.spatial_out[1] = 7;
  d.kernel[0] = 3;
  d.kernel[1] = 3;
  d.stride[0] = d.stride[1] = 1;
  d.pad[0] = d.pad[1] = 1;
  d.dilation[0] = d.dilation[1] = 1;
  auto x = values(2 * 6 * 35), w = values(10 * 27), b = values(10),
       dy = values(2 * 10 * 35);
  std::vector<float> y(dy.size()), dx(x.size(), .25f), dw(w.size(), .25f),
      db(10, .25f), col(k::conv_col_size(d) * 2);
  std::vector<double> ry(y.size()), rx(dx.size(), .25), rw(dw.size(), .25),
      rb(10, .25);
  for (int n = 0; n < 2; ++n)
    for (int o = 0; o < 10; ++o)
      for (int h = 0; h < 5; ++h)
        for (int v = 0; v < 7; ++v) {
          const int yi = (n * 10 + o) * 35 + h * 7 + v;
          ry[yi] = b[o];
          rb[o] += dy[yi];
          for (int c = 0; c < 3; ++c)
            for (int kh = 0; kh < 3; ++kh)
              for (int kw = 0; kw < 3; ++kw) {
                const int ih = h + kh - 1, iv = v + kw - 1;
                if (ih < 0 || ih >= 5 || iv < 0 || iv >= 7)
                  continue;
                const int xi = (n * 6 + (o / 5) * 3 + c) * 35 + ih * 7 + iv,
                          wi = o * 27 + c * 9 + kh * 3 + kw;
                ry[yi] += double(x[xi]) * w[wi];
                rx[xi] += double(dy[yi]) * w[wi];
                rw[wi] += double(dy[yi]) * x[xi];
              }
        }
  k::conv_forward(x.data(), w.data(), b.data(), y.data(), col.data(), d);
  k::conv_backward(x.data(), w.data(), dy.data(), dx.data(), dw.data(),
                   db.data(), col.data(), d);
  for (std::size_t i = 0; i < y.size(); ++i)
    check(y[i], ry[i], "conv forward");
  for (std::size_t i = 0; i < dx.size(); ++i)
    check(dx[i], rx[i], "conv dx");
  for (std::size_t i = 0; i < dw.size(); ++i)
    check(dw[i], rw[i], "conv dw");
  for (int i = 0; i < 10; ++i)
    check(db[i], rb[i], "conv db");
}
void tape_growth() {
  nexus_model::MicroTape tape;
  auto x = nexus_model::Tensor::full({1, 2, 3}, .25f);
  auto w = nexus_model::Tensor::full({1, 4, 3}, .5f);
  const int a = tape.load(x), b = tape.load(w);
  const int product = tape.matmul_bt(a, b);
  int out = product;
  for (int i = 0; i < 100; ++i)
    out = tape.scale(out, 1.f);
  out = tape.add(out, product);
  auto grad = nexus_model::Tensor::full({1, 2, 4}, 1.f);
  tape.backward(out, grad);
  for (std::size_t i = 0; i < grad.numel(); ++i)
    check(tape.value(out)[i], .75, "tape value after growth");
  for (std::size_t i = 0; i < x.numel(); ++i)
    check(tape.grad(a)[i], 4., "both add branches propagate");
  for (std::size_t i = 0; i < w.numel(); ++i)
    check(tape.grad(b)[i], 1., "tape weight gradient");
}
} // namespace
int main() {
  try {
    for (bool scalar : {false, true}) {
      k::set_force_scalar(scalar);
      linear();
      reductions();
      convolution();
      tape_growth();
    }
    k::set_force_scalar(false);
    std::cout << "kernel tails, guards, optional gradients, accumulation, "
                 "grouped conv and stable reductions passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
