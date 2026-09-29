#pragma once

/**
 * @file recurrent.hpp
 * @brief RNN, LSTM ve GRU. Girdi düzeni batch-first: [B, T, F].
 *
 * RNN:  h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
 * LSTM: Hochreiter & Schmidhuber, 1997. Kapı sırası i, f, g, o (PyTorch).
 *       c_t = f ⊙ c_{t-1} + i ⊙ g,  h_t = o ⊙ tanh(c_t)
 *       BPTT: Graves, Supervised Sequence Labelling with Recurrent Neural Networks, 2012.
 * GRU:  Cho et al., 2014.
 *       r,z = σ(W x + U h),  n = tanh(W_n x + r ⊙ U_n h),  h' = (1-z)⊙n + z⊙h
 *
 * `truncate_bptt > 0` ise gradyan yalnızca son k adıma akar.
 * Zaman tamponları kurucudan sonra ilk forward'da ayrılır; aynı [B,T] tekrarında ayırma yoktur.
 */

#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace nexus_model {
namespace detail {

inline void pack_time(const float* src, float* dst, int batch, int time, int feature, int t) {
  for (int b = 0; b < batch; ++b) {
    const float* row = src + (static_cast<std::size_t>(b * time + t) * feature);
    std::copy(row, row + feature, dst + static_cast<std::size_t>(b) * feature);
  }
}

inline void scatter_time(const float* src, float* dst, int batch, int time, int feature, int t) {
  for (int b = 0; b < batch; ++b) {
    float* row = dst + (static_cast<std::size_t>(b * time + t) * feature);
    const float* in = src + static_cast<std::size_t>(b) * feature;
    std::copy(in, in + feature, row);
  }
}

inline void add_bias_block(float* bias, int hidden, int gate, float value) {
  for (int i = 0; i < hidden; ++i) bias[gate * hidden + i] += value;
}

}  // namespace detail

class RNN : public Module {
 public:
  RNN(int input_size, int hidden_size, int num_layers = 1, int truncate_bptt = 0)
      : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers), truncate_(truncate_bptt) {
    layers_.resize(static_cast<std::size_t>(num_layers));
    for (int layer = 0; layer < num_layers; ++layer) {
      const int in = layer == 0 ? input_size : hidden_size;
      auto& slot = layers_[static_cast<std::size_t>(layer)];
      slot.w_ih = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(hidden_size), static_cast<std::size_t>(in)}));
      slot.w_hh = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(hidden_size), static_cast<std::size_t>(hidden_size)}));
      slot.b_ih = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(hidden_size)}));
      slot.b_hh = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(hidden_size)}));
      init::kaiming_uniform_(slot.w_ih->data);
      init::orthogonal_(slot.w_hh->data);
      const std::string prefix = "l" + std::to_string(layer) + ".";
      register_parameter(prefix + "weight_ih", slot.w_ih);
      register_parameter(prefix + "weight_hh", slot.w_hh);
      register_parameter(prefix + "bias_ih", slot.b_ih);
      register_parameter(prefix + "bias_hh", slot.b_hh);
    }
  }

  Tensor forward(const Tensor& input) override { return run_forward(input); }
  Tensor backward(const Tensor& grad_output) override { return run_backward(grad_output); }

 protected:
  struct Slot {
    std::shared_ptr<Parameter> w_ih, w_hh, b_ih, b_hh;
    Tensor h;
    int in = 0;
  };

  virtual int gate_count() const { return 1; }
  virtual void activate(const float* pre, float* h, const float* /*hprev*/, int /*batch*/) const {
    kernels::tanh_fwd(pre, h, static_cast<std::size_t>(batch_cache_) * hidden_size_);
  }
  virtual void gate_backward(const float* h, const float* /*hprev*/, const float* dh, float* dpre, int batch) const {
    for (int i = 0; i < batch * hidden_size_; ++i) dpre[i] = dh[i] * (1.f - h[i] * h[i]);
  }

  Tensor run_forward(const Tensor& input) {
    if (input.rank() != 3 || input.size(2) != static_cast<std::size_t>(input_size_)) {
      throw ModelError("RNN-family input must be [B, T, input_size]");
    }
    const int batch = static_cast<int>(input.size(0));
    const int time = static_cast<int>(input.size(1));
    ensure(batch, time);
    source_.copy_from(input);
    const float* seq = source_.data();
    int seq_feature = input_size_;
    for (int layer = 0; layer < num_layers_; ++layer) {
      auto& slot = layers_[static_cast<std::size_t>(layer)];
      for (int t = 0; t < time; ++t) {
        detail::pack_time(seq, step_x_.data(), batch, time, seq_feature, t);
        if (t == 0) step_h_.zero();
        else detail::pack_time(slot.h.data(), step_h_.data(), batch, time, hidden_size_, t - 1);
        kernels::linear_forward(step_x_.data(), slot.w_ih->data.data(), slot.b_ih->data.data(), pre_x_.data(), batch, slot.in, hidden_size_ * gate_count());
        kernels::linear_forward(step_h_.data(), slot.w_hh->data.data(), slot.b_hh->data.data(), pre_h_.data(), batch, hidden_size_, hidden_size_ * gate_count());
        kernels::add(pre_x_.data(), pre_h_.data(), pre_.data(), pre_.numel());
        activate(pre_.data(), step_y_.data(), step_h_.data(), batch);
        detail::scatter_time(step_y_.data(), slot.h.data(), batch, time, hidden_size_, t);
      }
      seq = slot.h.data();
      seq_feature = hidden_size_;
    }
    return layers_.back().h;
  }

  Tensor run_backward(const Tensor& grad_output) {
    const int batch = batch_cache_;
    const int time = time_cache_;
    const int t_start = truncate_ > 0 ? std::max(0, time - truncate_) : 0;
    flow_.copy_from(grad_output);
    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
      auto& slot = layers_[static_cast<std::size_t>(layer)];
      dinput_.resize_like(layer == 0 ? source_ : layers_[static_cast<std::size_t>(layer - 1)].h);
      dinput_.zero();
      dh_next_.zero();
      const float* seq = layer == 0 ? source_.data() : layers_[static_cast<std::size_t>(layer - 1)].h.data();
      const int seq_time = time;
      const int seq_feature = slot.in;
      for (int t = time - 1; t >= t_start; --t) {
        detail::pack_time(seq, step_x_.data(), batch, seq_time, seq_feature, t);
        if (t == 0) step_h_.zero();
        else detail::pack_time(slot.h.data(), step_h_.data(), batch, time, hidden_size_, t - 1);
        detail::pack_time(slot.h.data(), step_y_.data(), batch, time, hidden_size_, t);
        detail::pack_time(flow_.data(), step_dy_.data(), batch, time, hidden_size_, t);
        for (int i = 0; i < batch * hidden_size_; ++i) step_dy_[i] += dh_next_[i];
        gate_backward(step_y_.data(), step_h_.data(), step_dy_.data(), pre_.data(), batch);
        dx_.zero();
        dh_from_hh_.zero();
        float* dw_ih = slot.w_ih->requires_grad ? slot.w_ih->grad.data() : nullptr;
        float* dw_hh = slot.w_hh->requires_grad ? slot.w_hh->grad.data() : nullptr;
        float* db_ih = slot.b_ih->requires_grad ? slot.b_ih->grad.data() : nullptr;
        float* db_hh = slot.b_hh->requires_grad ? slot.b_hh->grad.data() : nullptr;
        kernels::linear_backward(step_x_.data(), slot.w_ih->data.data(), pre_.data(), dx_.data(), dw_ih, db_ih, batch, slot.in,
                                 hidden_size_ * gate_count());
        kernels::linear_backward(step_h_.data(), slot.w_hh->data.data(), pre_.data(), dh_from_hh_.data(), dw_hh, db_hh, batch,
                                 hidden_size_, hidden_size_ * gate_count());
        detail::scatter_time(dx_.data(), dinput_.data(), batch, time, slot.in, t);
        dh_next_.copy_from(dh_from_hh_);
      }
      flow_.resize_like(dinput_);
      flow_.copy_from(dinput_);
    }
    grad_input_.copy_from(dinput_);
    return grad_input_;
  }

  void ensure(int batch, int time) {
    batch_cache_ = batch;
    time_cache_ = time;
    source_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(time), static_cast<std::size_t>(input_size_)});
    for (int layer = 0; layer < num_layers_; ++layer) {
      auto& slot = layers_[static_cast<std::size_t>(layer)];
      slot.in = layer == 0 ? input_size_ : hidden_size_;
      slot.h.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(time), static_cast<std::size_t>(hidden_size_)});
    }
    const int gates = hidden_size_ * gate_count();
    const int widest = std::max(input_size_, hidden_size_);
    step_x_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(widest)});
    step_h_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(hidden_size_)});
    step_y_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(hidden_size_)});
    step_dy_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(hidden_size_)});
    pre_x_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(gates)});
    pre_h_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(gates)});
    pre_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(gates)});
    dx_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(widest)});
    dh_from_hh_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(hidden_size_)});
    dh_next_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(hidden_size_)});
    dh_next_.zero();
  }

  int input_size_;
  int hidden_size_;
  int num_layers_;
  int truncate_;
  int batch_cache_ = 0;
  int time_cache_ = 0;
  std::vector<Slot> layers_;
  Tensor source_, flow_, dinput_, grad_input_;
  Tensor step_x_, step_h_, step_y_, step_dy_, pre_x_, pre_h_, pre_, dx_, dh_from_hh_, dh_next_;
};

class LSTM : public Module {
 public:
  /// Kapı sırası: input, forget, cell, output. Unutma kapısı bias'ı 1 ile başlar.
  LSTM(int input_size, int hidden_size, int num_layers = 1, int truncate_bptt = 0)
      : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers), truncate_(truncate_bptt) {
    slots_.resize(static_cast<std::size_t>(num_layers));
    for (int layer = 0; layer < num_layers; ++layer) {
      const int in = layer == 0 ? input_size : hidden_size;
      auto& slot = slots_[static_cast<std::size_t>(layer)];
      const std::size_t gates = static_cast<std::size_t>(4 * hidden_size);
      slot.w_ih = std::make_shared<Parameter>(Tensor::zeros({gates, static_cast<std::size_t>(in)}));
      slot.w_hh = std::make_shared<Parameter>(Tensor::zeros({gates, static_cast<std::size_t>(hidden_size)}));
      slot.b_ih = std::make_shared<Parameter>(Tensor::zeros({gates}));
      slot.b_hh = std::make_shared<Parameter>(Tensor::zeros({gates}));
      init::kaiming_uniform_(slot.w_ih->data);
      init::orthogonal_(slot.w_hh->data);
      detail::add_bias_block(slot.b_ih->data.data(), hidden_size, 1, 1.f);
      const std::string prefix = "l" + std::to_string(layer) + ".";
      register_parameter(prefix + "weight_ih", slot.w_ih);
      register_parameter(prefix + "weight_hh", slot.w_hh);
      register_parameter(prefix + "bias_ih", slot.b_ih);
      register_parameter(prefix + "bias_hh", slot.b_hh);
    }
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() != 3 || input.size(2) != static_cast<std::size_t>(input_size_)) {
      throw ModelError("LSTM input must be [B, T, input_size]");
    }
    const int batch = static_cast<int>(input.size(0));
    const int time = static_cast<int>(input.size(1));
    const int H = hidden_size_;
    ensure(batch, time);
    source_.copy_from(input);
    const float* seq = source_.data();
    int feat = input_size_;
    for (int layer = 0; layer < num_layers_; ++layer) {
      auto& slot = slots_[static_cast<std::size_t>(layer)];
      const int in = layer == 0 ? input_size_ : H;
      for (int t = 0; t < time; ++t) {
        detail::pack_time(seq, x_.data(), batch, time, feat, t);
        if (t == 0) {
          hp_.zero();
          cp_.zero();
        } else {
          detail::pack_time(slot.h.data(), hp_.data(), batch, time, H, t - 1);
          detail::pack_time(slot.c.data(), cp_.data(), batch, time, H, t - 1);
        }
        kernels::linear_forward(x_.data(), slot.w_ih->data.data(), slot.b_ih->data.data(), pre_x_.data(), batch, in, 4 * H);
        kernels::linear_forward(hp_.data(), slot.w_hh->data.data(), slot.b_hh->data.data(), pre_h_.data(), batch, H, 4 * H);
        kernels::add(pre_x_.data(), pre_h_.data(), pre_.data(), pre_.numel());
        const int n = batch * H;
        for (int b = 0; b < batch; ++b) {
          float* p = pre_.data() + static_cast<std::size_t>(b) * 4 * H;
          kernels::sigmoid(p, is_.data() + static_cast<std::size_t>(b) * H, static_cast<std::size_t>(H));
          kernels::sigmoid(p + H, fs_.data() + static_cast<std::size_t>(b) * H, static_cast<std::size_t>(H));
          kernels::tanh_fwd(p + 2 * H, gs_.data() + static_cast<std::size_t>(b) * H, static_cast<std::size_t>(H));
          kernels::sigmoid(p + 3 * H, os_.data() + static_cast<std::size_t>(b) * H, static_cast<std::size_t>(H));
        }
        for (int i = 0; i < n; ++i) cs_[i] = fs_[i] * cp_[i] + is_[i] * gs_[i];
        kernels::tanh_fwd(cs_.data(), th_.data(), static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) hs_[i] = os_[i] * th_[i];
        detail::scatter_time(is_.data(), slot.i.data(), batch, time, H, t);
        detail::scatter_time(fs_.data(), slot.f.data(), batch, time, H, t);
        detail::scatter_time(gs_.data(), slot.g.data(), batch, time, H, t);
        detail::scatter_time(os_.data(), slot.o.data(), batch, time, H, t);
        detail::scatter_time(cs_.data(), slot.c.data(), batch, time, H, t);
        detail::scatter_time(hs_.data(), slot.h.data(), batch, time, H, t);
      }
      seq = slot.h.data();
      feat = H;
    }
    return slots_.back().h;
  }

  Tensor backward(const Tensor& grad_output) override {
    const int batch = batch_;
    const int time = time_;
    const int H = hidden_size_;
    const int t0 = truncate_ > 0 ? std::max(0, time - truncate_) : 0;
    flow_.copy_from(grad_output);
    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
      auto& slot = slots_[static_cast<std::size_t>(layer)];
      const int in = layer == 0 ? input_size_ : H;
      const float* seq = layer == 0 ? source_.data() : slots_[static_cast<std::size_t>(layer - 1)].h.data();
      dinput_.resize_like(layer == 0 ? source_ : slots_[static_cast<std::size_t>(layer - 1)].h);
      dinput_.zero();
      dh_next_.zero();
      dc_next_.zero();
      for (int t = time - 1; t >= t0; --t) {
        detail::pack_time(seq, x_.data(), batch, time, in, t);
        if (t == 0) {
          hp_.zero();
          cp_.zero();
        } else {
          detail::pack_time(slot.h.data(), hp_.data(), batch, time, H, t - 1);
          detail::pack_time(slot.c.data(), cp_.data(), batch, time, H, t - 1);
        }
        detail::pack_time(slot.i.data(), is_.data(), batch, time, H, t);
        detail::pack_time(slot.f.data(), fs_.data(), batch, time, H, t);
        detail::pack_time(slot.g.data(), gs_.data(), batch, time, H, t);
        detail::pack_time(slot.o.data(), os_.data(), batch, time, H, t);
        detail::pack_time(slot.c.data(), cs_.data(), batch, time, H, t);
        detail::pack_time(flow_.data(), hs_.data(), batch, time, H, t);
        for (int i = 0; i < batch * H; ++i) hs_[i] += dh_next_[i];
        kernels::tanh_fwd(cs_.data(), th_.data(), static_cast<std::size_t>(batch * H));
        for (int i = 0; i < batch * H; ++i) {
          const float dh = hs_[i];
          const float tanh_c = th_[i];
          const float one_t = 1.f - tanh_c * tanh_c;
          const float d_o = dh * tanh_c;
          const float dc = dc_next_[i] + dh * os_[i] * one_t;
          const float d_i = dc * gs_[i];
          const float d_f = dc * cp_[i];
          const float d_g = dc * is_[i];
          dc_prev_[i] = dc * fs_[i];
          pre_[static_cast<std::size_t>(i / H) * 4 * H + (i % H)] = d_i * is_[i] * (1.f - is_[i]);
          pre_[static_cast<std::size_t>(i / H) * 4 * H + H + (i % H)] = d_f * fs_[i] * (1.f - fs_[i]);
          pre_[static_cast<std::size_t>(i / H) * 4 * H + 2 * H + (i % H)] = d_g * (1.f - gs_[i] * gs_[i]);
          pre_[static_cast<std::size_t>(i / H) * 4 * H + 3 * H + (i % H)] = d_o * os_[i] * (1.f - os_[i]);
        }
        dx_.zero();
        dh_from_.zero();
        float* dw_ih = slot.w_ih->requires_grad ? slot.w_ih->grad.data() : nullptr;
        float* dw_hh = slot.w_hh->requires_grad ? slot.w_hh->grad.data() : nullptr;
        float* db_ih = slot.b_ih->requires_grad ? slot.b_ih->grad.data() : nullptr;
        float* db_hh = slot.b_hh->requires_grad ? slot.b_hh->grad.data() : nullptr;
        kernels::linear_backward(x_.data(), slot.w_ih->data.data(), pre_.data(), dx_.data(), dw_ih, db_ih, batch, in, 4 * H);
        kernels::linear_backward(hp_.data(), slot.w_hh->data.data(), pre_.data(), dh_from_.data(), dw_hh, db_hh, batch, H, 4 * H);
        detail::scatter_time(dx_.data(), dinput_.data(), batch, time, in, t);
        dh_next_.copy_from(dh_from_);
        dc_next_.copy_from(dc_prev_);
      }
      flow_.resize_like(dinput_);
      flow_.copy_from(dinput_);
    }
    grad_input_.copy_from(dinput_);
    return grad_input_;
  }

 private:
  struct Slot {
    std::shared_ptr<Parameter> w_ih, w_hh, b_ih, b_hh;
    Tensor i, f, g, o, c, h;
  };

  void ensure(int batch, int time) {
    batch_ = batch;
    time_ = time;
    source_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(time), static_cast<std::size_t>(input_size_)});
    for (auto& slot : slots_) {
      const std::size_t dims[3] = {static_cast<std::size_t>(batch), static_cast<std::size_t>(time), static_cast<std::size_t>(hidden_size_)};
      slot.i.resize_dims(dims, 3);
      slot.f.resize_dims(dims, 3);
      slot.g.resize_dims(dims, 3);
      slot.o.resize_dims(dims, 3);
      slot.c.resize_dims(dims, 3);
      slot.h.resize_dims(dims, 3);
    }
    const int H = hidden_size_;
    const int wide = std::max(input_size_, H);
    x_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(wide)});
    dx_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(wide)});
    hp_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    cp_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    is_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    fs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    gs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    os_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    cs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    hs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    th_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    pre_x_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(4 * H)});
    pre_h_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(4 * H)});
    pre_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(4 * H)});
    dh_next_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    dc_next_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    dc_prev_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    dh_from_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
  }

  int input_size_, hidden_size_, num_layers_, truncate_;
  int batch_ = 0, time_ = 0;
  std::vector<Slot> slots_;
  Tensor source_, flow_, dinput_, grad_input_;
  Tensor x_, dx_, hp_, cp_, is_, fs_, gs_, os_, cs_, hs_, th_;
  Tensor pre_x_, pre_h_, pre_, dh_next_, dc_next_, dc_prev_, dh_from_;
};

class GRU : public Module {
 public:
  GRU(int input_size, int hidden_size, int num_layers = 1, int truncate_bptt = 0)
      : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers), truncate_(truncate_bptt) {
    slots_.resize(static_cast<std::size_t>(num_layers));
    for (int layer = 0; layer < num_layers; ++layer) {
      const int in = layer == 0 ? input_size : hidden_size;
      auto& slot = slots_[static_cast<std::size_t>(layer)];
      const std::size_t gates = static_cast<std::size_t>(3 * hidden_size);
      slot.w_ih = std::make_shared<Parameter>(Tensor::zeros({gates, static_cast<std::size_t>(in)}));
      slot.w_hh = std::make_shared<Parameter>(Tensor::zeros({gates, static_cast<std::size_t>(hidden_size)}));
      slot.b_ih = std::make_shared<Parameter>(Tensor::zeros({gates}));
      slot.b_hh = std::make_shared<Parameter>(Tensor::zeros({gates}));
      init::kaiming_uniform_(slot.w_ih->data);
      init::orthogonal_(slot.w_hh->data);
      const std::string prefix = "l" + std::to_string(layer) + ".";
      register_parameter(prefix + "weight_ih", slot.w_ih);
      register_parameter(prefix + "weight_hh", slot.w_hh);
      register_parameter(prefix + "bias_ih", slot.b_ih);
      register_parameter(prefix + "bias_hh", slot.b_hh);
    }
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() != 3) throw ModelError("GRU input must be [B, T, F]");
    const int batch = static_cast<int>(input.size(0));
    const int time = static_cast<int>(input.size(1));
    const int H = hidden_size_;
    ensure(batch, time);
    source_.copy_from(input);
    const float* seq = source_.data();
    int feat = input_size_;
    for (int layer = 0; layer < num_layers_; ++layer) {
      auto& slot = slots_[static_cast<std::size_t>(layer)];
      const int in = layer == 0 ? input_size_ : H;
      for (int t = 0; t < time; ++t) {
        detail::pack_time(seq, x_.data(), batch, time, feat, t);
        if (t == 0) hp_.zero();
        else detail::pack_time(slot.h.data(), hp_.data(), batch, time, H, t - 1);
        kernels::linear_forward(x_.data(), slot.w_ih->data.data(), slot.b_ih->data.data(), pre_x_.data(), batch, in, 3 * H);
        kernels::linear_forward(hp_.data(), slot.w_hh->data.data(), slot.b_hh->data.data(), pre_h_.data(), batch, H, 3 * H);
        const int n = batch * H;
        for (int b = 0; b < batch; ++b) {
          float* px = pre_x_.data() + static_cast<std::size_t>(b) * 3 * H;
          float* ph = pre_h_.data() + static_cast<std::size_t>(b) * 3 * H;
          for (int h = 0; h < H; ++h) {
            rs_[static_cast<std::size_t>(b) * H + h] = px[h] + ph[h];
            zs_[static_cast<std::size_t>(b) * H + h] = px[H + h] + ph[H + h];
            hn_[static_cast<std::size_t>(b) * H + h] = ph[2 * H + h];
          }
        }
        kernels::sigmoid(rs_.data(), rs_.data(), static_cast<std::size_t>(n));
        kernels::sigmoid(zs_.data(), zs_.data(), static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
          const int b = i / H;
          const int h = i % H;
          const float nx = pre_x_[static_cast<std::size_t>(b) * 3 * H + 2 * H + h];
          ns_[i] = std::tanh(nx + rs_[i] * hn_[i]);
          hs_[i] = (1.f - zs_[i]) * ns_[i] + zs_[i] * hp_[i];
        }
        detail::scatter_time(rs_.data(), slot.r.data(), batch, time, H, t);
        detail::scatter_time(zs_.data(), slot.z.data(), batch, time, H, t);
        detail::scatter_time(ns_.data(), slot.n.data(), batch, time, H, t);
        detail::scatter_time(hn_.data(), slot.hn.data(), batch, time, H, t);
        detail::scatter_time(hs_.data(), slot.h.data(), batch, time, H, t);
      }
      seq = slot.h.data();
      feat = H;
    }
    return slots_.back().h;
  }

  Tensor backward(const Tensor& grad_output) override {
    const int batch = batch_;
    const int time = time_;
    const int H = hidden_size_;
    const int t0 = truncate_ > 0 ? std::max(0, time - truncate_) : 0;
    flow_.copy_from(grad_output);
    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
      auto& slot = slots_[static_cast<std::size_t>(layer)];
      const int in = layer == 0 ? input_size_ : H;
      const float* seq = layer == 0 ? source_.data() : slots_[static_cast<std::size_t>(layer - 1)].h.data();
      dinput_.resize_like(layer == 0 ? source_ : slots_[static_cast<std::size_t>(layer - 1)].h);
      dinput_.zero();
      dh_next_.zero();
      for (int t = time - 1; t >= t0; --t) {
        detail::pack_time(seq, x_.data(), batch, time, in, t);
        if (t == 0) hp_.zero();
        else detail::pack_time(slot.h.data(), hp_.data(), batch, time, H, t - 1);
        detail::pack_time(slot.r.data(), rs_.data(), batch, time, H, t);
        detail::pack_time(slot.z.data(), zs_.data(), batch, time, H, t);
        detail::pack_time(slot.n.data(), ns_.data(), batch, time, H, t);
        detail::pack_time(slot.hn.data(), hn_.data(), batch, time, H, t);
        detail::pack_time(flow_.data(), hs_.data(), batch, time, H, t);
        for (int i = 0; i < batch * H; ++i) hs_[i] += dh_next_[i];
        pre_x_.zero();
        pre_h_.zero();
        for (int i = 0; i < batch * H; ++i) {
          const int b = i / H;
          const int h = i % H;
          const float dh = hs_[i];
          const float dz = dh * (hp_[i] - ns_[i]);
          const float dn = dh * (1.f - zs_[i]);
          const float dh_explicit = dh * zs_[i];
          const float dn_pre = dn * (1.f - ns_[i] * ns_[i]);
          const float dr = dn_pre * hn_[i];
          const float dz_pre = dz * zs_[i] * (1.f - zs_[i]);
          const float dr_pre = dr * rs_[i] * (1.f - rs_[i]);
          float* dxg = pre_x_.data() + static_cast<std::size_t>(b) * 3 * H;
          float* dhg = pre_h_.data() + static_cast<std::size_t>(b) * 3 * H;
          dxg[h] += dr_pre;
          dhg[h] += dr_pre;
          dxg[H + h] += dz_pre;
          dhg[H + h] += dz_pre;
          dxg[2 * H + h] += dn_pre;
          dhg[2 * H + h] += dn_pre * rs_[i];
          dh_from_[i] = dh_explicit;
        }
        dx_.zero();
        float* dw_ih = slot.w_ih->requires_grad ? slot.w_ih->grad.data() : nullptr;
        float* dw_hh = slot.w_hh->requires_grad ? slot.w_hh->grad.data() : nullptr;
        float* db_ih = slot.b_ih->requires_grad ? slot.b_ih->grad.data() : nullptr;
        float* db_hh = slot.b_hh->requires_grad ? slot.b_hh->grad.data() : nullptr;
        kernels::linear_backward(x_.data(), slot.w_ih->data.data(), pre_x_.data(), dx_.data(), dw_ih, db_ih, batch, in, 3 * H);
        kernels::linear_backward(hp_.data(), slot.w_hh->data.data(), pre_h_.data(), dh_from_.data(), dw_hh, db_hh, batch, H, 3 * H);
        detail::scatter_time(dx_.data(), dinput_.data(), batch, time, in, t);
        dh_next_.copy_from(dh_from_);
      }
      flow_.resize_like(dinput_);
      flow_.copy_from(dinput_);
    }
    grad_input_.copy_from(dinput_);
    return grad_input_;
  }

 private:
  struct Slot {
    std::shared_ptr<Parameter> w_ih, w_hh, b_ih, b_hh;
    Tensor r, z, n, hn, h;
  };

  void ensure(int batch, int time) {
    batch_ = batch;
    time_ = time;
    source_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(time), static_cast<std::size_t>(input_size_)});
    for (auto& slot : slots_) {
      const std::size_t dims[3] = {static_cast<std::size_t>(batch), static_cast<std::size_t>(time), static_cast<std::size_t>(hidden_size_)};
      slot.r.resize_dims(dims, 3);
      slot.z.resize_dims(dims, 3);
      slot.n.resize_dims(dims, 3);
      slot.hn.resize_dims(dims, 3);
      slot.h.resize_dims(dims, 3);
    }
    const int H = hidden_size_;
    const int wide = std::max(input_size_, H);
    x_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(wide)});
    dx_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(wide)});
    hp_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    rs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    zs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    ns_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    hn_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    hs_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    pre_x_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(3 * H)});
    pre_h_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(3 * H)});
    dh_next_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
    dh_from_.resize({static_cast<std::size_t>(batch), static_cast<std::size_t>(H)});
  }

  int input_size_, hidden_size_, num_layers_, truncate_;
  int batch_ = 0, time_ = 0;
  std::vector<Slot> slots_;
  Tensor source_, flow_, dinput_, grad_input_;
  Tensor x_, dx_, hp_, rs_, zs_, ns_, hn_, hs_, pre_x_, pre_h_, dh_next_, dh_from_;
};

}  // namespace nexus_model
