#pragma once

/**
 * @file tensor.hpp
 * @brief 64-bayt hizalı, paylaşımlı depolamalı yoğun tensör.
 *
 * Sıcak yol (`forward` / `backward`) veri kopyalamaz: dönüş değeri depolamayı
 * paylaşan bir başlıktır. `resize_dims`, tamponu tek sahibi varken yeniden
 * kullanır; şekil değişmediyse ayırma yapmaz.
 *
 * Bu tip `matrix_pro::Tensor` ile aynı satır-major float32 düzenini taşır.
 * GPU GEMM istendiğinde `bridge/matrixflash.hpp` host aynasını MatrixFlash'e
 * devreder. MatrixFlash `multiply` yalnızca CUDA olduğundan CPU sıcak yolu
 * bu depoda, AVX2/AVX-512 çekirdekleriyle yürür.
 */

#include "nexus_model/core/error.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#endif

namespace nexus_model {

inline constexpr int kMaxRank = 8;

inline std::atomic<std::uint64_t>& allocation_count() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}

struct AlignedStorage {
  float* data = nullptr;
  std::size_t capacity = 0;

  explicit AlignedStorage(std::size_t n) : capacity(n) {
    if (n == 0) {
      return;
    }
    const std::size_t bytes = n * sizeof(float);
#ifdef _WIN32
    data = static_cast<float*>(_aligned_malloc(bytes, 64));
#else
    const std::size_t aligned_bytes = (bytes + 63u) & ~std::size_t{63};
    data = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
#endif
    if (data == nullptr) {
      throw ModelError("aligned allocation failed");
    }
    allocation_count().fetch_add(1, std::memory_order_relaxed);
  }

  AlignedStorage(const AlignedStorage&) = delete;
  AlignedStorage& operator=(const AlignedStorage&) = delete;

  ~AlignedStorage() {
#ifdef _WIN32
    _aligned_free(data);
#else
    std::free(data);
#endif
  }
};

class Tensor {
 public:
  Tensor() = default;

  explicit Tensor(std::initializer_list<std::size_t> shape) {
    resize_dims(shape.begin(), static_cast<std::uint8_t>(shape.size()));
  }

  explicit Tensor(const std::vector<std::size_t>& shape) {
    resize_dims(shape.data(), static_cast<std::uint8_t>(shape.size()));
  }

  static Tensor zeros(std::initializer_list<std::size_t> shape) {
    Tensor t(shape);
    t.zero();
    return t;
  }

  static Tensor zeros(const std::vector<std::size_t>& shape) {
    Tensor t(shape);
    t.zero();
    return t;
  }

  static Tensor full(std::initializer_list<std::size_t> shape, float value) {
    Tensor t(shape);
    t.fill(value);
    return t;
  }

  static Tensor from_values(std::initializer_list<std::size_t> shape, std::initializer_list<float> values) {
    Tensor t(shape);
    if (values.size() != t.numel()) {
      throw ModelError("from_values: value count does not match shape");
    }
    if (t.numel() > 0) {
      std::copy(values.begin(), values.end(), t.data());
    }
    return t;
  }

  /// Dış belleğe bakan, sahip olmayan görünüm. Ömür çağırana aittir.
  static Tensor borrow(float* data, std::initializer_list<std::size_t> shape) {
    Tensor t;
    t.ptr_ = data;
    t.apply_shape(shape.begin(), static_cast<std::uint8_t>(shape.size()));
    return t;
  }

  [[nodiscard]] std::uint8_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::size_t numel() const noexcept { return numel_; }
  [[nodiscard]] bool empty() const noexcept { return numel_ == 0; }
  [[nodiscard]] const std::size_t* dims() const noexcept { return dims_.data(); }

  [[nodiscard]] std::size_t size(std::size_t axis) const {
    if (axis >= rank_) {
      throw ModelError("axis out of range");
    }
    return dims_[axis];
  }

  [[nodiscard]] std::vector<std::size_t> shape_vec() const {
    return std::vector<std::size_t>(dims_.begin(), dims_.begin() + rank_);
  }

  [[nodiscard]] bool same_shape(const Tensor& other) const noexcept {
    if (rank_ != other.rank_ || numel_ != other.numel_) {
      return false;
    }
    for (std::uint8_t i = 0; i < rank_; ++i) {
      if (dims_[i] != other.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] float* data() noexcept { return ptr_; }
  [[nodiscard]] const float* data() const noexcept { return ptr_; }

  float& operator[](std::size_t i) { return ptr_[i]; }
  float operator[](std::size_t i) const { return ptr_[i]; }

  void zero() {
    if (numel_ > 0 && ptr_ != nullptr) {
      std::memset(ptr_, 0, numel_ * sizeof(float));
    }
  }

  void fill(float value) {
    if (ptr_ == nullptr) {
      return;
    }
    for (std::size_t i = 0; i < numel_; ++i) {
      ptr_[i] = value;
    }
  }

  /// Şekil aynıysa mevcut tampona yazar. Ayırma yalnızca kapasite yetmezse.
  void copy_from(const Tensor& src) {
    resize_like(src);
    if (numel_ > 0) {
      std::memcpy(ptr_, src.ptr_, numel_ * sizeof(float));
    }
  }

  [[nodiscard]] Tensor clone() const {
    Tensor out;
    out.resize_like(*this);
    if (numel_ > 0) {
      std::memcpy(out.ptr_, ptr_, numel_ * sizeof(float));
    }
    return out;
  }

  /// Eleman sayısı aynıysa depolamayı paylaşan görünüm.
  [[nodiscard]] Tensor view_as(const std::size_t* dims, std::uint8_t rank) const {
    std::size_t n = 1;
    for (std::uint8_t i = 0; i < rank; ++i) n *= dims[i];
    if (n != numel_) throw ModelError("reshape numel mismatch");
    Tensor out = *this;
    out.apply_shape(dims, rank);
    return out;
  }

  /// Eleman sayısı aynıysa depolamayı paylaşan görünüm. Tek `-1` boyutu çıkarılır.
  [[nodiscard]] Tensor reshape(std::initializer_list<std::size_t> shape) const {
    std::size_t known = 1;
    int infer = -1;
    std::uint8_t r = 0;
    std::size_t dims[kMaxRank];
    for (std::size_t d : shape) {
      if (r >= kMaxRank) {
        throw ModelError("reshape rank exceeds 8");
      }
      if (d == static_cast<std::size_t>(-1)) {
        if (infer >= 0) {
          throw ModelError("reshape has two inferred dimensions");
        }
        infer = static_cast<int>(r);
        dims[r++] = 1;
      } else {
        known *= d;
        dims[r++] = d;
      }
    }
    if (infer >= 0) {
      if (known == 0 || numel_ % known != 0) {
        throw ModelError("reshape cannot infer dimension");
      }
      dims[infer] = numel_ / known;
    } else {
      std::size_t n = 1;
      for (std::uint8_t i = 0; i < r; ++i) {
        n *= dims[i];
      }
      if (n != numel_) {
        throw ModelError("reshape numel mismatch");
      }
    }
    Tensor out = *this;
    out.apply_shape(dims, r);
    return out;
  }

  void resize_dims(const std::size_t* dims, std::uint8_t rank) {
    std::size_t n = product(dims, rank);
    const bool shape_same = same_dims(dims, rank);
    const bool reusable = storage_ && storage_.use_count() == 1 && offset_ == 0 &&
                          storage_->capacity >= n && ptr_ == storage_->data;
    if (!reusable) {
      if (n > 0) {
        storage_ = std::make_shared<AlignedStorage>(n);
        ptr_ = storage_->data;
      } else {
        storage_.reset();
        ptr_ = nullptr;
      }
      offset_ = 0;
    }
    if (!shape_same || !reusable) {
      apply_shape(dims, rank);
    }
  }

  void resize_like(const Tensor& src) { resize_dims(src.dims_.data(), src.rank_); }

  void resize(std::initializer_list<std::size_t> shape) {
    resize_dims(shape.begin(), static_cast<std::uint8_t>(shape.size()));
  }

 private:
  static std::size_t product(const std::size_t* dims, std::uint8_t rank) {
    if (rank > kMaxRank) {
      throw ModelError("rank exceeds 8");
    }
    std::size_t n = 1;
    for (std::uint8_t i = 0; i < rank; ++i) {
      if (dims[i] != 0 && n > (static_cast<std::size_t>(-1) / dims[i])) {
        throw ModelError("numel overflow");
      }
      n *= dims[i];
    }
    return rank == 0 ? 0 : n;
  }

  bool same_dims(const std::size_t* dims, std::uint8_t rank) const noexcept {
    if (rank_ != rank) {
      return false;
    }
    for (std::uint8_t i = 0; i < rank; ++i) {
      if (dims_[i] != dims[i]) {
        return false;
      }
    }
    return true;
  }

  void apply_shape(const std::size_t* dims, std::uint8_t rank) {
    rank_ = rank;
    numel_ = product(dims, rank);
    for (std::uint8_t i = 0; i < rank; ++i) {
      dims_[i] = dims[i];
    }
    for (std::uint8_t i = rank; i < kMaxRank; ++i) {
      dims_[i] = 0;
    }
  }

  std::shared_ptr<AlignedStorage> storage_;
  float* ptr_ = nullptr;
  std::size_t offset_ = 0;
  std::array<std::size_t, kMaxRank> dims_{};
  std::uint8_t rank_ = 0;
  std::size_t numel_ = 0;
};

}  // namespace nexus_model
