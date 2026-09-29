#include "nexus_model/core/kernels.hpp"
#include "nexus_model/simd/kernel_decls.hpp"
#include <atomic>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace nexus_model::kernels {
namespace {

bool cpu_feature(int leaf, int sub, int reg, int bit) {
  int info[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuidex(info, leaf, sub);
#else
  __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
#endif
  return (info[reg] & (1 << bit)) != 0;
}

bool os_supports_avx() {
  if (!cpu_feature(1,0,2,27) || !cpu_feature(1,0,2,28)) return false;
#if defined(_MSC_VER)
  const unsigned long long xcr = _xgetbv(0);
#else
  unsigned int low, high;
  __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
  const unsigned long long xcr = (static_cast<unsigned long long>(high)<<32)|low;
#endif
  return (xcr & 0x6) == 0x6;
}

[[maybe_unused]]
bool os_supports_avx512() {
  if (!os_supports_avx()) return false;
#if defined(_MSC_VER)
  const unsigned long long xcr = _xgetbv(0);
#else
  unsigned int low, high;
  __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
  const unsigned long long xcr = (static_cast<unsigned long long>(high)<<32)|low;
#endif
  return (xcr & 0xE6) == 0xE6;
}

KernelTable build(bool force_scalar) {
  KernelTable t{};
  t.fill = detail::fill;
  t.add = detail::add;
  t.mul = detail::mul;
  t.scale = detail::scale;
  t.axpy = detail::axpy;
  t.dot = detail::dot;
  t.relu = detail::relu;
  t.relu_bwd = detail::relu_bwd;
  t.leaky_relu = detail::leaky_relu;
  t.leaky_relu_bwd = detail::leaky_relu_bwd;
  t.sigmoid = detail::sigmoid;
  t.sigmoid_bwd = detail::sigmoid_bwd;
  t.tanh_fwd = detail::tanh_fwd;
  t.tanh_bwd = detail::tanh_bwd;
  t.gelu = detail::gelu;
  t.gelu_bwd = detail::gelu_bwd;
  t.gelu_tanh = detail::gelu_tanh;
  t.gelu_tanh_bwd = detail::gelu_tanh_bwd;
  t.silu = detail::silu;
  t.silu_bwd = detail::silu_bwd;
  t.elu = detail::elu;
  t.elu_bwd = detail::elu_bwd;
  t.mish = detail::mish;
  t.mish_bwd = detail::mish_bwd;
  t.softmax = detail::softmax;
  t.softmax_bwd = detail::softmax_bwd;
  t.log_softmax = detail::log_softmax;
  t.log_softmax_bwd = detail::log_softmax_bwd;
  t.linear_forward = detail::linear_forward;
  t.linear_backward = detail::linear_backward;
  t.layernorm = detail::layernorm;
  t.layernorm_bwd = detail::layernorm_bwd;
  if (force_scalar || !os_supports_avx() || !cpu_feature(7, 0, 1, 5) || !cpu_feature(1,0,2,12)) {
    return t;
  }
  t.fill = avx2::fill;
  t.add = avx2::add;
  t.mul = avx2::mul;
  t.scale = avx2::scale;
  t.axpy = avx2::axpy;
  t.dot = avx2::dot;
  t.relu = avx2::relu;
  t.relu_bwd = avx2::relu_bwd;
  t.leaky_relu = avx2::leaky_relu;
  t.leaky_relu_bwd = avx2::leaky_relu_bwd;
  t.sigmoid = avx2::sigmoid;
  t.sigmoid_bwd = avx2::sigmoid_bwd;
  t.tanh_fwd = avx2::tanh_fwd;
  t.gelu_tanh = avx2::gelu_tanh;
  t.gelu_tanh_bwd = avx2::gelu_tanh_bwd;
  t.silu = avx2::silu;
  t.silu_bwd = avx2::silu_bwd;
  t.softmax = avx2::softmax;
  t.linear_forward = avx2::linear_forward;
  t.linear_backward = avx2::linear_backward;
  t.layernorm = avx2::layernorm;
#if defined(NEXUS_MODEL_ENABLE_AVX512)
  if (os_supports_avx512() && cpu_feature(7, 0, 1, 16)) {
    t.relu = avx512::relu;
    t.dot = avx512::dot;
    t.linear_forward = avx512::linear_forward;
  }
#endif
  return t;
}

const KernelTable& automatic_table() {
  static const KernelTable table = build(false);
  return table;
}
const KernelTable& scalar_table() {
  static const KernelTable table = build(true);
  return table;
}
std::atomic<const KernelTable*>& active_table() {
  static std::atomic<const KernelTable*> table{&automatic_table()};
  return table;
}

}  // namespace

const KernelTable& cpu() { return *active_table().load(std::memory_order_acquire); }

void set_force_scalar(bool force) {
  active_table().store(force ? &scalar_table() : &automatic_table(), std::memory_order_release);
}

}  // namespace nexus_model::kernels
