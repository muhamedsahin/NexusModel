#pragma once

/**
 * @file attention.hpp
 * @brief Scaled dot-product attention, multi-head attention ve Transformer blokları.
 *
 * Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V
 * Vaswani et al., Attention Is All You Need, 2017.
 *
 * Softmax∘matmul zinciri `MicroTape` ile türetilir (Yol B). Q/K/V ve çıkış
 * projeksiyonları analitik Linear'dır. Encoder: post-norm
 *   x = LN(x + Dropout(Attn(x))),  x = LN(x + Dropout(FFN(x))).
 */

#include "nexus_model/activations/activations.hpp"
#include "nexus_model/core/micro_tape.hpp"
#include "nexus_model/layers/dropout.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_model/layers/normalization.hpp"

#include <cmath>
#include <vector>

namespace nexus_model {
namespace detail {

inline void split_heads(const float* in, float* out, int batch, int time, int heads, int dim) {
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < time; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
          out[(((static_cast<std::size_t>(b) * heads + h) * time + t) * dim) + d] =
              in[(((static_cast<std::size_t>(b) * time + t) * heads + h) * dim) + d];
        }
      }
    }
  }
}

inline void merge_heads(const float* in, float* out, int batch, int time, int heads, int dim) {
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < time; ++t) {
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
          out[(((static_cast<std::size_t>(b) * time + t) * heads + h) * dim) + d] =
              in[(((static_cast<std::size_t>(b) * heads + h) * time + t) * dim) + d];
        }
      }
    }
  }
}

}  // namespace detail

class MultiHeadAttention : public Module {
 public:
  MultiHeadAttention(int embed, int num_heads, float dropout = 0.f, bool causal = false)
      : embed_(embed), heads_(num_heads), dim_(num_heads > 0 ? embed / num_heads : 0), dropout_p_(dropout), causal_(causal) {
    if (num_heads <= 0 || embed % num_heads != 0) throw ModelError("embed must be divisible by num_heads");
    q_ = std::make_shared<Linear>(embed, embed);
    k_ = std::make_shared<Linear>(embed, embed);
    v_ = std::make_shared<Linear>(embed, embed);
    o_ = std::make_shared<Linear>(embed, embed);
    register_module("q_proj", q_);
    register_module("k_proj", k_);
    register_module("v_proj", v_);
    register_module("out_proj", o_);
  }

  Tensor forward(const Tensor& input) override { return forward(input, input, input, nullptr); }

  Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask) {
    if (query.rank() != 3 || key.rank() != 3 || value.rank() != 3) throw ModelError("attention expects [B, T, E]");
    self_ = (&query == &key) && (&key == &value);
    batch_ = static_cast<int>(query.size(0));
    tq_ = static_cast<int>(query.size(1));
    tk_ = static_cast<int>(key.size(1));
    Tensor q = q_->forward(query);
    Tensor k = k_->forward(key);
    Tensor v = v_->forward(value);
    const std::size_t qdims[3] = {static_cast<std::size_t>(batch_ * heads_), static_cast<std::size_t>(tq_), static_cast<std::size_t>(dim_)};
    const std::size_t kdims[3] = {static_cast<std::size_t>(batch_ * heads_), static_cast<std::size_t>(tk_), static_cast<std::size_t>(dim_)};
    qh_.resize_dims(qdims, 3);
    kh_.resize_dims(kdims, 3);
    vh_.resize_dims(kdims, 3);
    detail::split_heads(q.data(), qh_.data(), batch_, tq_, heads_, dim_);
    detail::split_heads(k.data(), kh_.data(), batch_, tk_, heads_, dim_);
    detail::split_heads(v.data(), vh_.data(), batch_, tk_, heads_, dim_);
    tape_.reset();
    iq_ = tape_.load(qh_);
    ik_ = tape_.load(kh_);
    iv_ = tape_.load(vh_);
    int scores = tape_.matmul_bt(iq_, ik_);
    scores = tape_.scale(scores, 1.f / std::sqrt(static_cast<float>(dim_)));
    if (causal_) {
      if (tq_ != tk_) throw ModelError("causal attention requires equal query and key lengths");
      build_causal();
      scores = tape_.add_matrix(scores, causal_mask_);
    }
    if (mask != nullptr) scores = tape_.add_matrix(scores, *mask);
    int probs = tape_.softmax(scores);
    if (training_ && dropout_p_ > 0.f) {
      build_dropout(probs);
      const int mid = tape_.load(drop_mask_);
      probs = tape_.mul(probs, mid);
    }
    ctx_ = tape_.matmul(probs, iv_);
    merged_.resize_like(q);
    detail::merge_heads(tape_.value(ctx_).data(), merged_.data(), batch_, tq_, heads_, dim_);
    return o_->forward(merged_);
  }

  Tensor backward(const Tensor& grad_output) override {
    Tensor g = o_->backward(grad_output);
    gctx_.resize_like(qh_);
    detail::split_heads(g.data(), gctx_.data(), batch_, tq_, heads_, dim_);
    tape_.backward(ctx_, gctx_);
    const std::size_t qshape[3] = {static_cast<std::size_t>(batch_), static_cast<std::size_t>(tq_), static_cast<std::size_t>(embed_)};
    const std::size_t kshape[3] = {static_cast<std::size_t>(batch_), static_cast<std::size_t>(tk_), static_cast<std::size_t>(embed_)};
    gq_.resize_dims(qshape, 3);
    gk_.resize_dims(kshape, 3);
    gv_.resize_dims(kshape, 3);
    detail::merge_heads(tape_.grad(iq_).data(), gq_.data(), batch_, tq_, heads_, dim_);
    detail::merge_heads(tape_.grad(ik_).data(), gk_.data(), batch_, tk_, heads_, dim_);
    detail::merge_heads(tape_.grad(iv_).data(), gv_.data(), batch_, tk_, heads_, dim_);
    Tensor dq = q_->backward(gq_);
    Tensor dk = k_->backward(gk_);
    Tensor dv = v_->backward(gv_);
    if (self_) {
      grad_input_.resize_like(dq);
      const std::size_t n = dq.numel();
      for (std::size_t i = 0; i < n; ++i) grad_input_[i] = dq[i] + dk[i] + dv[i];
      return grad_input_;
    }
    dk_in_.copy_from(dk);
    dv_in_.copy_from(dv);
    grad_input_.copy_from(dq);
    return grad_input_;
  }

  const Tensor& grad_key_input() const { return dk_in_; }
  const Tensor& grad_value_input() const { return dv_in_; }

 private:
  void build_causal() {
    if (causal_t_ == tq_ && causal_mask_.numel() == static_cast<std::size_t>(tq_ * tq_)) return;
    causal_t_ = tq_;
    causal_mask_.resize({static_cast<std::size_t>(tq_), static_cast<std::size_t>(tq_)});
    for (int i = 0; i < tq_; ++i) {
      for (int j = 0; j < tq_; ++j) causal_mask_[static_cast<std::size_t>(i) * tq_ + j] = j > i ? -1.0e9f : 0.f;
    }
  }

  void build_dropout(int probs_id) {
    const Tensor& probs = tape_.value(probs_id);
    drop_mask_.resize_like(probs);
    kernels::dropout_mask(drop_mask_.data(), probs.numel(), dropout_p_, seed_);
    seed_ += 0x9E3779B97F4A7C15ull;
    const float scale = 1.f / (1.f - dropout_p_);
    for (std::size_t i = 0; i < drop_mask_.numel(); ++i) drop_mask_[i] *= scale;
  }

  int embed_, heads_, dim_;
  float dropout_p_;
  bool causal_;
  bool self_ = true;
  int batch_ = 0, tq_ = 0, tk_ = 0, causal_t_ = -1;
  int iq_ = 0, ik_ = 0, iv_ = 0, ctx_ = 0;
  std::uint64_t seed_ = 0xA77EU;
  std::shared_ptr<Linear> q_, k_, v_, o_;
  MicroTape tape_;
  Tensor qh_, kh_, vh_, merged_, gctx_, gq_, gk_, gv_, grad_input_, dk_in_, dv_in_, causal_mask_, drop_mask_;
};

class PositionalEncoding : public Module {
 public:
  PositionalEncoding(int embed, int max_len = 512, bool learnable = false) : embed_(embed), learnable_(learnable) {
    build(max_len);
    if (learnable) {
      weight_ = std::make_shared<Parameter>(table_.clone());
      register_parameter("weight", weight_);
    }
  }

  Tensor forward(const Tensor& input) override {
    if (input.rank() != 3 || input.size(2) != static_cast<std::size_t>(embed_)) {
      throw ModelError("positional encoding expects [B, T, E]");
    }
    const int time = static_cast<int>(input.size(1));
    if (time > length_) build(time);
    const Tensor& pe = learnable_ ? weight_->data : table_;
    output_.resize_like(input);
    const int batch = static_cast<int>(input.size(0));
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < time; ++t) {
        const float* x = input.data() + (static_cast<std::size_t>(b * time + t) * embed_);
        const float* p = pe.data() + static_cast<std::size_t>(t) * embed_;
        float* y = output_.data() + (static_cast<std::size_t>(b * time + t) * embed_);
        for (int e = 0; e < embed_; ++e) y[e] = x[e] + p[e];
      }
    }
    time_ = time;
    batch_ = batch;
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.copy_from(grad_output);
    if (learnable_ && weight_->requires_grad) {
      for (int b = 0; b < batch_; ++b) {
        for (int t = 0; t < time_; ++t) {
          const float* g = grad_output.data() + (static_cast<std::size_t>(b * time_ + t) * embed_);
          float* dp = weight_->grad.data() + static_cast<std::size_t>(t) * embed_;
          for (int e = 0; e < embed_; ++e) dp[e] += g[e];
        }
      }
    }
    return grad_input_;
  }

 private:
  void build(int length) {
    length_ = length;
    table_.resize({static_cast<std::size_t>(length), static_cast<std::size_t>(embed_)});
    for (int pos = 0; pos < length; ++pos) {
      for (int i = 0; i < embed_; ++i) {
        const float angle = static_cast<float>(pos) / std::pow(10000.f, static_cast<float>(i / 2 * 2) / static_cast<float>(embed_));
        table_[static_cast<std::size_t>(pos) * embed_ + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
      }
    }
  }

  int embed_;
  int length_ = 0;
  int time_ = 0;
  int batch_ = 0;
  bool learnable_;
  Tensor table_, output_, grad_input_;
  std::shared_ptr<Parameter> weight_;
};

class TransformerEncoderLayer : public Module {
 public:
  TransformerEncoderLayer(int embed, int heads, int ff = 0, float dropout = 0.1f)
      : attn_(std::make_shared<MultiHeadAttention>(embed, heads, dropout, false)),
        drop1_(std::make_shared<Dropout>(dropout)),
        drop2_(std::make_shared<Dropout>(dropout)),
        ln1_(std::make_shared<LayerNorm>(embed)),
        ln2_(std::make_shared<LayerNorm>(embed)),
        fc1_(std::make_shared<Linear>(embed, ff > 0 ? ff : 4 * embed)),
        fc2_(std::make_shared<Linear>(ff > 0 ? ff : 4 * embed, embed)),
        act_(std::make_shared<GELU>()) {
    register_module("attn", attn_);
    register_module("drop1", drop1_);
    register_module("drop2", drop2_);
    register_module("ln1", ln1_);
    register_module("ln2", ln2_);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
    register_module("act", act_);
  }

  Tensor forward(const Tensor& input) override {
    x0_.copy_from(input);
    Tensor attn = attn_->forward(x0_);
    Tensor dropped = drop1_->forward(attn);
    res1_.resize_like(x0_);
    kernels::add(x0_.data(), dropped.data(), res1_.data(), x0_.numel());
    Tensor normed = ln1_->forward(res1_);
    Tensor hidden = fc1_->forward(normed);
    Tensor activated = act_->forward(hidden);
    Tensor projected = fc2_->forward(activated);
    Tensor dropped2 = drop2_->forward(projected);
    res2_.resize_like(normed);
    kernels::add(normed.data(), dropped2.data(), res2_.data(), normed.numel());
    return ln2_->forward(res2_);
  }

  Tensor backward(const Tensor& grad_output) override {
    Tensor g_res2 = ln2_->backward(grad_output);
    Tensor g_drop2 = drop2_->backward(g_res2);
    Tensor g_fc2 = fc2_->backward(g_drop2);
    Tensor g_act = act_->backward(g_fc2);
    Tensor g_norm_ff = fc1_->backward(g_act);
    acc_.resize_like(g_res2);
    kernels::add(g_res2.data(), g_norm_ff.data(), acc_.data(), g_res2.numel());
    Tensor g_res1 = ln1_->backward(acc_);
    Tensor g_drop1 = drop1_->backward(g_res1);
    Tensor g_attn = attn_->backward(g_drop1);
    grad_input_.resize_like(g_attn);
    kernels::add(g_attn.data(), g_res1.data(), grad_input_.data(), g_attn.numel());
    return grad_input_;
  }

 private:
  std::shared_ptr<MultiHeadAttention> attn_;
  std::shared_ptr<Dropout> drop1_, drop2_;
  std::shared_ptr<LayerNorm> ln1_, ln2_;
  std::shared_ptr<Linear> fc1_, fc2_;
  std::shared_ptr<GELU> act_;
  Tensor x0_, res1_, res2_, acc_, grad_input_;
};

class TransformerDecoderLayer : public Module {
 public:
  TransformerDecoderLayer(int embed, int heads, int ff = 0, float dropout = 0.1f)
      : self_attn_(std::make_shared<MultiHeadAttention>(embed, heads, dropout, true)),
        cross_attn_(std::make_shared<MultiHeadAttention>(embed, heads, dropout, false)),
        drop1_(std::make_shared<Dropout>(dropout)),
        drop2_(std::make_shared<Dropout>(dropout)),
        drop3_(std::make_shared<Dropout>(dropout)),
        ln1_(std::make_shared<LayerNorm>(embed)),
        ln2_(std::make_shared<LayerNorm>(embed)),
        ln3_(std::make_shared<LayerNorm>(embed)),
        fc1_(std::make_shared<Linear>(embed, ff > 0 ? ff : 4 * embed)),
        fc2_(std::make_shared<Linear>(ff > 0 ? ff : 4 * embed, embed)),
        act_(std::make_shared<GELU>()) {
    register_module("self_attn", self_attn_);
    register_module("cross_attn", cross_attn_);
    register_module("drop1", drop1_);
    register_module("drop2", drop2_);
    register_module("drop3", drop3_);
    register_module("ln1", ln1_);
    register_module("ln2", ln2_);
    register_module("ln3", ln3_);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
    register_module("act", act_);
  }

  Tensor forward(const Tensor& input) override {
    if (!have_memory_) throw ModelError("TransformerDecoderLayer requires forward(target, memory)");
    return forward(input, memory_);
  }

  Tensor forward(const Tensor& target, const Tensor& memory) {
    memory_.copy_from(memory);
    have_memory_ = true;
    x0_.copy_from(target);
    Tensor self_out = self_attn_->forward(x0_);
    Tensor d1 = drop1_->forward(self_out);
    res1_.resize_like(x0_);
    kernels::add(x0_.data(), d1.data(), res1_.data(), x0_.numel());
    Tensor n1 = ln1_->forward(res1_);
    Tensor cross = cross_attn_->forward(n1, memory_, memory_, nullptr);
    Tensor d2 = drop2_->forward(cross);
    res2_.resize_like(n1);
    kernels::add(n1.data(), d2.data(), res2_.data(), n1.numel());
    Tensor n2 = ln2_->forward(res2_);
    Tensor hidden = fc1_->forward(n2);
    Tensor activated = act_->forward(hidden);
    Tensor projected = fc2_->forward(activated);
    Tensor d3 = drop3_->forward(projected);
    res3_.resize_like(n2);
    kernels::add(n2.data(), d3.data(), res3_.data(), n2.numel());
    return ln3_->forward(res3_);
  }

  Tensor backward(const Tensor& grad_output) override {
    Tensor g3 = ln3_->backward(grad_output);
    Tensor gd3 = drop3_->backward(g3);
    Tensor gfc2 = fc2_->backward(gd3);
    Tensor gact = act_->backward(gfc2);
    Tensor gn2_ff = fc1_->backward(gact);
    acc_.resize_like(g3);
    kernels::add(g3.data(), gn2_ff.data(), acc_.data(), g3.numel());
    Tensor g2 = ln2_->backward(acc_);
    Tensor gd2 = drop2_->backward(g2);
    Tensor gcross = cross_attn_->backward(gd2);
    acc2_.resize_like(g2);
    kernels::add(gcross.data(), g2.data(), acc2_.data(), g2.numel());
    Tensor g1 = ln1_->backward(acc2_);
    Tensor gd1 = drop1_->backward(g1);
    Tensor gself = self_attn_->backward(gd1);
    grad_input_.resize_like(gself);
    kernels::add(gself.data(), g1.data(), grad_input_.data(), gself.numel());
    return grad_input_;
  }

 private:
  std::shared_ptr<MultiHeadAttention> self_attn_, cross_attn_;
  std::shared_ptr<Dropout> drop1_, drop2_, drop3_;
  std::shared_ptr<LayerNorm> ln1_, ln2_, ln3_;
  std::shared_ptr<Linear> fc1_, fc2_;
  std::shared_ptr<GELU> act_;
  bool have_memory_ = false;
  Tensor memory_, x0_, res1_, res2_, res3_, acc_, acc2_, grad_input_;
};

}  // namespace nexus_model
