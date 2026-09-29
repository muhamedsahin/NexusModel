#include "nexus_model/activations/activations.hpp"
#include "nexus_model/containers/module_dict.hpp"
#include "nexus_model/containers/module_list.hpp"
#include "nexus_model/containers/sequential.hpp"
#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/layers/attention.hpp"
#include "nexus_model/layers/conv.hpp"
#include "nexus_model/layers/dropout.hpp"
#include "nexus_model/layers/embedding.hpp"
#include "nexus_model/layers/flatten_reshape.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_model/layers/normalization.hpp"
#include "nexus_model/layers/pooling.hpp"
#include "nexus_model/layers/recurrent.hpp"
#include "nexus_model/layers/residual.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const char* expr, const char* file, int line) {
  if (!ok) {
    std::cerr << file << ":" << line << "  " << expr << "\n";
    ++g_failed;
  }
}

#define CHECK(expr) ::check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

double sum_of(const nexus_model::Tensor& tensor) {
  double sum = 0.0;
  for (std::size_t i = 0; i < tensor.numel(); ++i) sum += tensor[i];
  return sum;
}

void fill_pattern(nexus_model::Tensor& tensor, float scale) {
  for (std::size_t i = 0; i < tensor.numel(); ++i) {
    tensor[i] = scale * static_cast<float>(static_cast<int>(i % 7) - 3);
  }
}

void shrink(nexus_model::Module& model, float scale) {
  for (auto* param : model.parameters()) {
    for (std::size_t i = 0; i < param->data.numel(); ++i) param->data[i] *= scale;
  }
}

bool close(float got, float expected, float tol) {
  return std::fabs(got - expected) <= tol;
}

void expect_tensor(const nexus_model::Tensor& got, std::initializer_list<float> expected, float tol) {
  CHECK(got.numel() == expected.size());
  if (got.numel() != expected.size()) return;
  std::size_t i = 0;
  for (float value : expected) {
    if (!close(got[i], value, tol)) {
      std::cerr << "  index " << i << " got " << got[i] << " expected " << value << "\n";
      ++g_failed;
    }
    ++i;
  }
}

void check_parameter_grad(nexus_model::Module& model, const nexus_model::Tensor& input, float eps, float tol, int max_coords) {
  model.zero_grad();
  nexus_model::Tensor y = model.forward(input);
  nexus_model::Tensor grad;
  grad.resize_like(y);
  grad.fill(1.f);
  model.backward(grad);
  std::vector<nexus_model::Tensor> analytic;
  auto params = model.parameters();
  analytic.reserve(params.size());
  for (auto* param : params) analytic.push_back(param->grad.clone());
  if (params.empty()) return;
  int checked = 0;
  for (std::size_t p = 0; p < params.size(); ++p) {
    const std::size_t step = std::max<std::size_t>(1, params[p]->data.numel() / static_cast<std::size_t>(max_coords));
    for (std::size_t i = 0; i < params[p]->data.numel(); i += step) {
      const float original = params[p]->data[i];
      params[p]->data[i] = original + eps;
      const double up = sum_of(model.forward(input));
      params[p]->data[i] = original - eps;
      const double down = sum_of(model.forward(input));
      params[p]->data[i] = original;
      const float numerical = static_cast<float>((up - down) / (2.0 * eps));
      if (!close(analytic[p][i], numerical, tol)) {
        std::cerr << "param " << p << " index " << i << " analytic " << analytic[p][i] << " numerical " << numerical << "\n";
        ++g_failed;
      }
      ++checked;
    }
  }
  CHECK(checked > 0);
}

void check_input_grad(nexus_model::Module& model, nexus_model::Tensor input, float eps, float tol) {
  model.zero_grad();
  nexus_model::Tensor y = model.forward(input);
  nexus_model::Tensor grad;
  grad.resize_like(y);
  grad.fill(1.f);
  nexus_model::Tensor dx = model.backward(grad);
  nexus_model::Tensor analytic = dx.clone();
  for (std::size_t i = 0; i < input.numel(); ++i) {
    const float original = input[i];
    // Piecewise activations (ReLU family) are not differentiable at 0.
    if (std::fabs(original) <= 2.f * eps) continue;
    input[i] = original + eps;
    const double up = sum_of(model.forward(input));
    input[i] = original - eps;
    const double down = sum_of(model.forward(input));
    input[i] = original;
    const float numerical = static_cast<float>((up - down) / (2.0 * eps));
    if (!close(analytic[i], numerical, tol)) {
      std::cerr << "input index " << i << " analytic " << analytic[i] << " numerical " << numerical << "\n";
      ++g_failed;
    }
  }
}

void test_linear_reference() {
  nexus_model::Linear layer(2, 3, true);
  float* w = layer.weight()->data.data();
  w[0] = 1; w[1] = 0;
  w[2] = 0; w[3] = 1;
  w[4] = 1; w[5] = 1;
  layer.bias()->data[0] = 0;
  layer.bias()->data[1] = 0;
  layer.bias()->data[2] = 1;
  auto x = nexus_model::Tensor::from_values({2, 2}, {1, 2, 3, 4});
  auto y = layer.forward(x);
  expect_tensor(y, {1, 2, 4, 3, 4, 8}, 1e-5f);
  layer.zero_grad();
  nexus_model::Tensor gy;
  gy.resize_like(y);
  gy.fill(1.f);
  auto dx = layer.backward(gy);
  expect_tensor(layer.weight()->grad, {4, 6, 4, 6, 4, 6}, 1e-4f);
  expect_tensor(layer.bias()->grad, {2, 2, 2}, 1e-4f);
  expect_tensor(dx, {2, 2, 2, 2}, 1e-4f);
}

void test_linear_grad() {
  nexus_model::init::manual_seed(1);
  nexus_model::Linear layer(4, 3);
  shrink(layer, 0.2f);
  auto x = nexus_model::Tensor::zeros({2, 4});
  fill_pattern(x, 0.3f);
  check_parameter_grad(layer, x, 1e-3f, 2e-3f, 32);
  check_input_grad(layer, x, 1e-3f, 2e-3f);
}

void test_activations() {
  auto x = nexus_model::Tensor::from_values({6}, {-2.f, -0.5f, 0.f, 0.2f, 1.f, 2.f});
  nexus_model::ReLU relu;
  check_parameter_grad(relu, x, 1e-3f, 1e-5f, 8);
  check_input_grad(relu, x.clone(), 1e-3f, 1e-4f);
  nexus_model::Sigmoid sigmoid;
  check_input_grad(sigmoid, x.clone(), 1e-3f, 2e-3f);
  nexus_model::Tanh tanh_layer;
  check_input_grad(tanh_layer, x.clone(), 1e-3f, 2e-3f);
  nexus_model::GELU gelu;
  check_input_grad(gelu, x.clone(), 1e-3f, 2e-3f);
  nexus_model::GELU gelu_tanh(true);
  check_input_grad(gelu_tanh, x.clone(), 1e-3f, 3e-3f);
  nexus_model::SiLU silu;
  check_input_grad(silu, x.clone(), 1e-3f, 2e-3f);
  nexus_model::Mish mish;
  check_input_grad(mish, x.clone(), 1e-3f, 3e-3f);
  nexus_model::ELU elu;
  check_input_grad(elu, x.clone(), 1e-3f, 2e-3f);
  nexus_model::LeakyReLU leaky(0.2f);
  check_input_grad(leaky, x.clone(), 1e-3f, 1e-4f);
  nexus_model::Softmax softmax;
  check_input_grad(softmax, x.clone(), 1e-3f, 3e-3f);
  nexus_model::LogSoftmax log_softmax;
  check_input_grad(log_softmax, x.clone(), 1e-3f, 3e-3f);
  nexus_model::PReLU prelu(0.25f);
  check_parameter_grad(prelu, x, 1e-3f, 2e-3f, 4);
  check_input_grad(prelu, x.clone(), 1e-3f, 2e-3f);
}

void test_sequential_and_dropout() {
  nexus_model::init::manual_seed(2);
  auto model = nexus_model::Sequential({
      std::make_shared<nexus_model::Linear>(4, 5),
      std::make_shared<nexus_model::ReLU>(),
      std::make_shared<nexus_model::Dropout>(0.f),
      std::make_shared<nexus_model::Linear>(5, 2),
  });
  shrink(model, 0.3f);
  auto x = nexus_model::Tensor::zeros({3, 4});
  fill_pattern(x, 0.2f);
  check_parameter_grad(model, x, 1e-3f, 3e-3f, 20);
  nexus_model::Dropout drop(0.5f, 7);
  drop.eval();
  auto y = drop.forward(x);
  expect_tensor(y, {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]}, 1e-6f);
  drop.train();
  nexus_model::Dropout always(1.f, 1);
  auto z = always.forward(x);
  CHECK(sum_of(z) == 0.0);
  nexus_model::Dropout never(0.f, 1);
  auto w = never.forward(x);
  CHECK(std::fabs(sum_of(w) - sum_of(x)) < 1e-5);
}

void test_conv_and_pool() {
  nexus_model::Conv2D conv(1, 1, 2, 1, 0);
  float* weight = conv.parameters()[0]->data.data();
  weight[0] = 1.f;
  weight[1] = 0.f;
  weight[2] = 0.f;
  weight[3] = 1.f;
  conv.parameters()[1]->data.zero();
  auto x = nexus_model::Tensor::from_values({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto y = conv.forward(x);
  expect_tensor(y, {6, 8, 12, 14}, 1e-4f);
  nexus_model::init::manual_seed(3);
  nexus_model::Conv2D learned(2, 3, 2, 1, 1);
  shrink(learned, 0.1f);
  auto cx = nexus_model::Tensor::zeros({2, 2, 4, 4});
  fill_pattern(cx, 0.15f);
  check_parameter_grad(learned, cx, 1e-3f, 4e-3f, 16);
  check_input_grad(learned, cx.clone(), 1e-3f, 4e-3f);

  nexus_model::Conv1D conv1(1, 2, 2);
  shrink(conv1, 0.2f);
  auto x1 = nexus_model::Tensor::zeros({2, 1, 5});
  fill_pattern(x1, 0.2f);
  check_parameter_grad(conv1, x1, 1e-3f, 4e-3f, 12);

  nexus_model::Conv3D conv3(1, 1, 2);
  shrink(conv3, 0.2f);
  auto x3 = nexus_model::Tensor::zeros({1, 1, 3, 3, 3});
  fill_pattern(x3, 0.1f);
  check_parameter_grad(conv3, x3, 1e-3f, 5e-3f, 8);

  nexus_model::MaxPool2D pool(2, 2);
  auto px = nexus_model::Tensor::from_values({1, 1, 2, 2}, {1.f, 3.f, 2.f, 0.f});
  auto py = pool.forward(px);
  expect_tensor(py, {3.f}, 1e-5f);
  check_input_grad(pool, px.clone(), 1e-3f, 1e-3f);
  nexus_model::AvgPool2D avg(2, 2);
  auto ay = avg.forward(px);
  expect_tensor(ay, {1.5f}, 1e-5f);
  nexus_model::GlobalAvgPool2D gap;
  auto gy = gap.forward(px);
  expect_tensor(gy, {1.5f}, 1e-5f);
  check_input_grad(gap, px.clone(), 1e-3f, 1e-4f);
}

void test_norm() {
  nexus_model::init::manual_seed(4);
  nexus_model::LayerNorm ln(4);
  shrink(ln, 0.5f);
  auto x = nexus_model::Tensor::zeros({3, 4});
  fill_pattern(x, 0.4f);
  check_parameter_grad(ln, x, 1e-3f, 4e-3f, 16);
  check_input_grad(ln, x.clone(), 1e-3f, 4e-3f);

  nexus_model::BatchNorm2D bn(2);
  shrink(bn, 0.5f);
  auto bx = nexus_model::Tensor::zeros({4, 2, 2, 2});
  fill_pattern(bx, 0.3f);
  check_parameter_grad(bn, bx, 1e-3f, 6e-3f, 8);
  bn.eval();
  auto y_eval = bn.forward(bx);
  CHECK(y_eval.numel() == bx.numel());
  bn.train();

  nexus_model::GroupNorm gn(2, 4);
  shrink(gn, 0.5f);
  auto gx = nexus_model::Tensor::zeros({2, 4, 2, 2});
  fill_pattern(gx, 0.25f);
  check_parameter_grad(gn, gx, 1e-3f, 6e-3f, 8);

  nexus_model::InstanceNorm in(2);
  shrink(in, 0.5f);
  auto ix = nexus_model::Tensor::zeros({2, 2, 2, 2});
  fill_pattern(ix, 0.25f);
  check_input_grad(in, ix, 1e-3f, 6e-3f);
}

void test_embedding_and_flatten() {
  nexus_model::Embedding emb(6, 3, 0);
  CHECK(emb.parameters()[0]->data[0] == 0.f);
  auto idx = nexus_model::Tensor::from_values({4}, {1.f, 2.f, 1.f, 0.f});
  auto y = emb.forward(idx);
  CHECK(y.rank() == 2);
  CHECK(y.size(1) == 3);
  CHECK(y[9] == 0.f && y[10] == 0.f && y[11] == 0.f);
  emb.zero_grad();
  nexus_model::Tensor gy;
  gy.resize_like(y);
  gy.fill(1.f);
  emb.backward(gy);
  CHECK(close(emb.parameters()[0]->grad[3], 2.f, 1e-5f));
  CHECK(close(emb.parameters()[0]->grad[0], 0.f, 1e-5f));

  nexus_model::Flatten flat;
  auto volume = nexus_model::Tensor::zeros({2, 2, 2});
  fill_pattern(volume, 1.f);
  auto fy = flat.forward(volume);
  CHECK(fy.size(0) == 2 && fy.size(1) == 4);
  auto dx = flat.backward(fy);
  CHECK(dx.rank() == 3);
  nexus_model::Reshape reshape({2, 4});
  auto ry = reshape.forward(volume);
  CHECK(ry.size(0) == 2 && ry.size(1) == 4);
}

void test_recurrent() {
  nexus_model::init::manual_seed(5);
  auto x = nexus_model::Tensor::zeros({2, 3, 3});
  fill_pattern(x, 0.2f);
  nexus_model::RNN rnn(3, 4, 1);
  shrink(rnn, 0.15f);
  check_parameter_grad(rnn, x, 1e-3f, 5e-3f, 10);
  nexus_model::LSTM lstm(3, 4, 1);
  shrink(lstm, 0.15f);
  check_parameter_grad(lstm, x, 1e-3f, 6e-3f, 8);
  nexus_model::GRU gru(3, 4, 1);
  shrink(gru, 0.15f);
  check_parameter_grad(gru, x, 1e-3f, 6e-3f, 8);
}

void test_attention() {
  nexus_model::init::manual_seed(6);
  nexus_model::MultiHeadAttention attn(8, 2, 0.f, false);
  shrink(attn, 0.1f);
  auto x = nexus_model::Tensor::zeros({2, 3, 8});
  fill_pattern(x, 0.1f);
  check_parameter_grad(attn, x, 1e-3f, 6e-3f, 8);
  check_input_grad(attn, x.clone(), 1e-3f, 6e-3f);

  nexus_model::TransformerEncoderLayer enc(8, 2, 16, 0.f);
  shrink(enc, 0.1f);
  check_parameter_grad(enc, x, 1e-3f, 8e-3f, 6);

  nexus_model::PositionalEncoding pe(8, 16);
  auto y = pe.forward(x);
  CHECK(y.numel() == x.numel());
  CHECK(std::fabs(y[0] - x[0]) > 0.f || std::fabs(y[1] - x[1]) > 0.f);
}

void test_containers_and_state() {
  auto a = std::make_shared<nexus_model::Linear>(3, 2);
  auto b = std::make_shared<nexus_model::ReLU>();
  nexus_model::Sequential seq({a, b});
  CHECK(seq.parameters().size() == 2);
  auto state = seq.state_dict();
  CHECK(state.items.count("0.weight") == 1);
  CHECK(state.items.count("0.bias") == 1);
  a->weight()->data.zero();
  seq.load_state_dict(state);
  CHECK(a->weight()->data[0] == state.items["0.weight"][0]);
  const std::string path = "nexus_model_state.bin";
  nexus_model::save_state_dict(state, path);
  auto loaded = nexus_model::load_state_dict_file(path);
  CHECK(loaded.items["0.weight"].numel() == state.items["0.weight"].numel());
  bool threw = false;
  try {
    nexus_model::StateDict extra = state;
    extra.items["missing"] = nexus_model::Tensor::zeros({1});
    seq.load_state_dict(extra);
  } catch (const nexus_model::ModelError&) {
    threw = true;
  }
  CHECK(threw);

  nexus_model::ModuleList list({a});
  CHECK(list.parameters().size() == 2);
  nexus_model::ModuleDict dict;
  dict.add("head", a);
  CHECK(dict["head"]->parameters().size() == 2);

  auto block = nexus_model::ResidualBlock(std::make_shared<nexus_model::Linear>(3, 3, false));
  shrink(block, 0.2f);
  auto x = nexus_model::Tensor::zeros({2, 3});
  fill_pattern(x, 0.3f);
  check_input_grad(block, x, 1e-3f, 3e-3f);
}

void test_hot_path_allocations() {
  nexus_model::Linear layer(8, 8);
  auto x = nexus_model::Tensor::zeros({4, 8});
  fill_pattern(x, 0.1f);
  nexus_model::Tensor grad;
  layer.forward(x);
  {
    auto y = layer.forward(x);
    grad.resize_like(y);
    grad.fill(1.f);
    layer.backward(grad);
  }
  const auto before = nexus_model::allocation_count().load();
  layer.forward(x);
  layer.backward(grad);
  const auto after = nexus_model::allocation_count().load();
  CHECK(after == before);

  nexus_model::MultiHeadAttention attn(8, 2, 0.f, true);
  auto ax = nexus_model::Tensor::zeros({2, 4, 8});
  fill_pattern(ax, 0.05f);
  attn.forward(ax);
  {
    auto ay = attn.forward(ax);
    nexus_model::Tensor ag;
    ag.resize_like(ay);
    ag.fill(1.f);
    attn.backward(ag);
  }
  nexus_model::Tensor ag;
  ag.resize_like(attn.forward(ax));
  ag.fill(1.f);
  attn.backward(ag);
  const auto before_attn = nexus_model::allocation_count().load();
  attn.forward(ax);
  attn.backward(ag);
  CHECK(nexus_model::allocation_count().load() == before_attn);
}

void test_simd_matches_scalar() {
  nexus_model::Linear layer(16, 12);
  shrink(layer, 0.05f);
  auto x = nexus_model::Tensor::zeros({8, 16});
  fill_pattern(x, 0.2f);
  auto y_fast = layer.forward(x).clone();
  nexus_model::kernels::set_force_scalar(true);
  auto y_scalar = layer.forward(x).clone();
  nexus_model::kernels::set_force_scalar(false);
  CHECK(y_fast.numel() == y_scalar.numel());
  float max_abs = 0.f;
  for (std::size_t i = 0; i < y_fast.numel(); ++i) max_abs = std::max(max_abs, std::fabs(y_fast[i] - y_scalar[i]));
  CHECK(max_abs < 2e-3f);
}

}  // namespace

int main() {
  test_linear_reference();
  test_linear_grad();
  test_activations();
  test_sequential_and_dropout();
  test_conv_and_pool();
  test_norm();
  test_embedding_and_flatten();
  test_recurrent();
  test_attention();
  test_containers_and_state();
  test_hot_path_allocations();
  test_simd_matches_scalar();
  if (g_failed != 0) {
    std::cerr << g_failed << " checks failed\n";
    return 1;
  }
  std::cout << "all nexus_model checks passed\n";
  return 0;
}
