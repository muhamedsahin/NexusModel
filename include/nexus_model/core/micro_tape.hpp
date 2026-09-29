#pragma once

/**
 * @file micro_tape.hpp
 * @brief Katman-lokal ters-mod teyp. Yalnızca bir katmanın iç grafiğini tutar.
 *
 * MatrixFlash `Variable::backward` kök gradyanı 1 ile doldurur ve NexusLoss'un
 * dış `dL/dy` vektörünü kabul etmez. Attention gibi softmax∘matmul zincirleri
 * bu teybe yazılır; projeksiyonlar analitik `Linear` olarak kalır.
 *
 * `reset` kapasiteyi bırakmaz. İkinci forward aynı şekilde yığın ayırmaz.
 */

#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/tensor.hpp"

#include <deque>
#include <vector>

namespace nexus_model {

class MicroTape {
 public:
  void reset() {
    used_ = 0;
    ops_.clear();
  }

  int load(const Tensor& value) {
    Slot& slot = allocate_slot(value);
    slot.value.copy_from(value);
    return used_ - 1;
  }

  /// C[b] = A[b] @ B[b]^T. A is [B,M,K], B is [B,N,K], C is [B,M,N].
  int matmul_bt(int a, int b) {
    const Tensor& left = slots_[a].value;
    const Tensor& right = slots_[b].value;
    const int batch = static_cast<int>(left.size(0));
    const int rows = static_cast<int>(left.size(1));
    const int inner = static_cast<int>(left.size(2));
    const int cols = static_cast<int>(right.size(1));
    const int id = blank_like_rank3(batch, rows, cols);
    for (int n = 0; n < batch; ++n) {
      kernels::linear_forward(left.data() + static_cast<std::size_t>(n) * rows * inner,
                              right.data() + static_cast<std::size_t>(n) * cols * inner, nullptr,
                              slots_[id].value.data() + static_cast<std::size_t>(n) * rows * cols, rows, inner, cols);
    }
    ops_.push_back(Op{Op::MatmulBt, id, a, b, 0.f, batch, rows, inner, cols, {}});
    return id;
  }

  /// C[b] = A[b] @ B[b]. A is [B,M,K], B is [B,K,N], C is [B,M,N].
  int matmul(int a, int b) {
    const Tensor& left = slots_[a].value;
    const Tensor& right = slots_[b].value;
    const int batch = static_cast<int>(left.size(0));
    const int rows = static_cast<int>(left.size(1));
    const int inner = static_cast<int>(left.size(2));
    const int cols = static_cast<int>(right.size(2));
    const int id = blank_like_rank3(batch, rows, cols);
    std::size_t scratch_dims[2] = {static_cast<std::size_t>(cols), static_cast<std::size_t>(inner)};
    scratch_.resize_dims(scratch_dims, 2);
    for (int n = 0; n < batch; ++n) {
      transpose_kn(right.data() + static_cast<std::size_t>(n) * inner * cols, scratch_.data(), inner, cols);
      kernels::linear_forward(left.data() + static_cast<std::size_t>(n) * rows * inner, scratch_.data(), nullptr,
                              slots_[id].value.data() + static_cast<std::size_t>(n) * rows * cols, rows, inner, cols);
    }
    ops_.push_back(Op{Op::Matmul, id, a, b, 0.f, batch, rows, inner, cols, {}});
    return id;
  }

  int scale(int a, float factor) {
    const int id = clone_shape(a);
    kernels::scale(slots_[a].value.data(), factor, slots_[id].value.data(), slots_[a].value.numel());
    ops_.push_back(Op{Op::Scale, id, a, -1, factor, 0, 0, 0, 0, {}});
    return id;
  }

  int add(int a, int b) {
    const int id = clone_shape(a);
    kernels::add(slots_[a].value.data(), slots_[b].value.data(), slots_[id].value.data(), slots_[a].value.numel());
    ops_.push_back(Op{Op::Add, id, a, b, 0.f, 0, 0, 0, 0, {}});
    return id;
  }

  int mul(int a, int b) {
    const int id = clone_shape(a);
    kernels::mul(slots_[a].value.data(), slots_[b].value.data(), slots_[id].value.data(), slots_[a].value.numel());
    ops_.push_back(Op{Op::Mul, id, a, b, 0.f, 0, 0, 0, 0, {}});
    return id;
  }

  /// value [R, rows, cols] += bias [rows, cols], bias is constant (no gradient).
  int add_matrix(int a, const Tensor& bias) {
    const Tensor& src = slots_[a].value;
    const int id = clone_shape(a);
    const int matrix = static_cast<int>(bias.numel());
    const int repeats = static_cast<int>(src.numel() / bias.numel());
    float* dst = slots_[id].value.data();
    const float* in = src.data();
    for (int r = 0; r < repeats; ++r) {
      for (int i = 0; i < matrix; ++i) dst[static_cast<std::size_t>(r) * matrix + i] = in[static_cast<std::size_t>(r) * matrix + i] + bias[i];
    }
    ops_.push_back(Op{Op::AddMatrix, id, a, -1, 0.f, repeats, matrix, 0, 0, {}});
    return id;
  }

  int softmax(int a) {
    const Tensor& src = slots_[a].value;
    const int cols = static_cast<int>(src.size(src.rank() - 1));
    const int rows = static_cast<int>(src.numel() / static_cast<std::size_t>(cols));
    const int id = clone_shape(a);
    kernels::softmax(src.data(), slots_[id].value.data(), rows, cols);
    ops_.push_back(Op{Op::Softmax, id, a, -1, 0.f, rows, cols, 0, 0, {}});
    return id;
  }

  void backward(int root, const Tensor& grad) {
    for (int i = 0; i < used_; ++i) {
      slots_[i].grad.resize_like(slots_[i].value);
      slots_[i].grad.zero();
    }
    slots_[root].grad.copy_from(grad);
    for (int op_index = static_cast<int>(ops_.size()) - 1; op_index >= 0; --op_index) apply(ops_[op_index]);
  }

  Tensor& value(int id) { return slots_[id].value; }
  const Tensor& grad(int id) const { return slots_[id].grad; }

 private:
  struct Slot {
    Tensor value;
    Tensor grad;
  };
  struct Op {
    enum Kind { MatmulBt, Matmul, Scale, Add, Mul, AddMatrix, Softmax } kind;
    int out;
    int a;
    int b;
    float scalar;
    int p0, p1, p2, p3;
    Tensor extra;
  };

  Slot& allocate_slot(const Tensor& like) {
    if (used_ == static_cast<int>(slots_.size())) slots_.emplace_back();
    Slot& slot = slots_[used_++];
    slot.value.resize_like(like);
    return slot;
  }

  int clone_shape(int source) {
    if (used_ == static_cast<int>(slots_.size())) slots_.emplace_back();
    Slot& slot = slots_[used_++];
    slot.value.resize_like(slots_[source].value);
    return used_ - 1;
  }

  int blank_like_rank3(int batch, int rows, int cols) {
    if (used_ == static_cast<int>(slots_.size())) slots_.emplace_back();
    Slot& slot = slots_[used_++];
    const std::size_t dims[3] = {static_cast<std::size_t>(batch), static_cast<std::size_t>(rows),
                                 static_cast<std::size_t>(cols)};
    slot.value.resize_dims(dims, 3);
    return used_ - 1;
  }

  static void transpose_kn(const float* src, float* dst, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) dst[static_cast<std::size_t>(c) * rows + r] = src[static_cast<std::size_t>(r) * cols + c];
    }
  }

  void apply(const Op& op) {
    const float* dy = slots_[op.out].grad.data();
    if (op.kind == Op::Scale) {
      kernels::axpy(slots_[op.a].grad.data(), dy, op.scalar, static_cast<int>(slots_[op.a].grad.numel()));
    } else if (op.kind == Op::Add || op.kind == Op::AddMatrix) {
      kernels::axpy(slots_[op.a].grad.data(), dy, 1.f, static_cast<int>(slots_[op.a].grad.numel()));
      if (op.kind == Op::Add)
        kernels::axpy(slots_[op.b].grad.data(), dy, 1.f, static_cast<int>(slots_[op.b].grad.numel()));
    } else if (op.kind == Op::Mul) {
      const std::size_t n = slots_[op.a].value.numel();
      float* da = slots_[op.a].grad.data();
      float* db = slots_[op.b].grad.data();
      const float* a = slots_[op.a].value.data();
      const float* b = slots_[op.b].value.data();
      for (std::size_t i = 0; i < n; ++i) {
        da[i] += dy[i] * b[i];
        db[i] += dy[i] * a[i];
      }
    } else if (op.kind == Op::Softmax) {
      scratch_.resize_like(slots_[op.out].value);
      kernels::softmax_bwd(slots_[op.out].value.data(), dy, scratch_.data(), op.p0, op.p1);
      kernels::axpy(slots_[op.a].grad.data(), scratch_.data(), 1.f, op.p0 * op.p1);
    } else if (op.kind == Op::MatmulBt) {
      const int batch = op.p0;
      const int rows = op.p1;
      const int inner = op.p2;
      const int cols = op.p3;
      for (int n = 0; n < batch; ++n) {
        kernels::linear_backward(slots_[op.a].value.data() + static_cast<std::size_t>(n) * rows * inner,
                                 slots_[op.b].value.data() + static_cast<std::size_t>(n) * cols * inner,
                                 dy + static_cast<std::size_t>(n) * rows * cols,
                                 slots_[op.a].grad.data() + static_cast<std::size_t>(n) * rows * inner,
                                 slots_[op.b].grad.data() + static_cast<std::size_t>(n) * cols * inner, nullptr, rows,
                                 inner, cols);
      }
    } else if (op.kind == Op::Matmul) {
      const int batch = op.p0;
      const int rows = op.p1;
      const int inner = op.p2;
      const int cols = op.p3;
      const std::size_t bt_dims[2] = {static_cast<std::size_t>(cols), static_cast<std::size_t>(inner)};
      scratch_.resize_dims(bt_dims, 2);
      const std::size_t dbt_dims[2] = {static_cast<std::size_t>(cols), static_cast<std::size_t>(inner)};
      scratch_grad_.resize_dims(dbt_dims, 2);
      for (int n = 0; n < batch; ++n) {
        transpose_kn(slots_[op.b].value.data() + static_cast<std::size_t>(n) * inner * cols, scratch_.data(), inner, cols);
        scratch_grad_.zero();
        kernels::linear_backward(slots_[op.a].value.data() + static_cast<std::size_t>(n) * rows * inner, scratch_.data(),
                                 dy + static_cast<std::size_t>(n) * rows * cols,
                                 slots_[op.a].grad.data() + static_cast<std::size_t>(n) * rows * inner, scratch_grad_.data(),
                                 nullptr, rows, inner, cols);
        float* db = slots_[op.b].grad.data() + static_cast<std::size_t>(n) * inner * cols;
        for (int r = 0; r < inner; ++r) {
          for (int c = 0; c < cols; ++c) db[static_cast<std::size_t>(r) * cols + c] += scratch_grad_[static_cast<std::size_t>(c) * inner + r];
        }
      }
    }
  }

  // New nodes must not invalidate references held by the operation being recorded.
  std::deque<Slot> slots_;
  std::vector<Op> ops_;
  int used_ = 0;
  Tensor scratch_;
  Tensor scratch_grad_;
};

}  // namespace nexus_model
