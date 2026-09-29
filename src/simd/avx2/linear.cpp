#include "gemm.hpp"
#include "nexus_model/simd/kernel_decls.hpp"
#include <algorithm>

namespace nexus_model::kernels::avx2 {
namespace internal {
void accumulate(const float *a, std::size_t ars, std::size_t aks,
                const float *b, int bs, float *c, int m, int n, int depth) {
  constexpr int block = 128;
  for (int kb = 0; kb < depth; kb += block) {
    const int count = std::min(block, depth - kb);
    for (int i = 0; i < m; i += 6) {
      int j = 0;
      const float *ap = a + static_cast<std::size_t>(i) * ars +
                        static_cast<std::size_t>(kb) * aks;
      for (; j + 16 <= n; j += 16) {
        const float *bp = b + static_cast<std::size_t>(kb) * bs + j;
        float *cp = c + static_cast<std::size_t>(i) * n + j;
        switch (std::min(6, m - i)) {
        case 6:
          tile<6>(ap, ars, aks, bp, bs, cp, n, count, true);
          break;
        case 5:
          tile<5>(ap, ars, aks, bp, bs, cp, n, count, true);
          break;
        case 4:
          tile<4>(ap, ars, aks, bp, bs, cp, n, count, true);
          break;
        case 3:
          tile<3>(ap, ars, aks, bp, bs, cp, n, count, true);
          break;
        case 2:
          tile<2>(ap, ars, aks, bp, bs, cp, n, count, true);
          break;
        default:
          tile<1>(ap, ars, aks, bp, bs, cp, n, count, true);
          break;
        }
      }
      for (int r = i; r < std::min(i + 6, m); ++r)
        for (int col = j; col < n; ++col) {
          float sum = 0;
          for (int k = kb; k < kb + count; ++k)
            sum += a[static_cast<std::size_t>(r) * ars +
                     static_cast<std::size_t>(k) * aks] *
                   b[static_cast<std::size_t>(k) * bs + col];
          c[static_cast<std::size_t>(r) * n + col] += sum;
        }
    }
  }
}
} // namespace internal

void linear_forward(const float *x, const float *w, const float *bias, float *y,
                    int rows, int in_f, int out_f) {
  // Packing costs more than it saves for tiny batches. Keep the direct path.
  if (rows < 4 || in_f < 16 || out_f < 16) {
    for (int r = 0; r < rows; ++r)
      for (int o = 0; o < out_f; ++o)
        y[static_cast<std::size_t>(r) * out_f + o] =
            dot(x + static_cast<std::size_t>(r) * in_f,
                w + static_cast<std::size_t>(o) * in_f, in_f) +
            (bias ? bias[o] : 0.f);
    return;
  }
  constexpr int block = 128;
  alignas(64) float packed[block * 16]; // 8 KiB, bounded stack storage; no
                                        // heap/cache invalidation.
  int o = 0;
  for (; o + 16 <= out_f; o += 16)
    for (int kb = 0; kb < in_f; kb += block) {
      const int count = std::min(block, in_f - kb);
      for (int j = 0; j < 16; ++j)
        for (int k = 0; k < count; ++k)
          packed[k * 16 + j] =
              w[static_cast<std::size_t>(o + j) * in_f + kb + k];
      for (int r = 0; r < rows; r += 6) {
        const float *a = x + static_cast<std::size_t>(r) * in_f + kb;
        float *c = y + static_cast<std::size_t>(r) * out_f + o;
        const float *b = bias ? bias + o : nullptr;
        switch (std::min(6, rows - r)) {
        case 6:
          internal::tile<6>(a, in_f, 1, packed, 16, c, out_f, count, kb != 0,
                            b);
          break;
        case 5:
          internal::tile<5>(a, in_f, 1, packed, 16, c, out_f, count, kb != 0,
                            b);
          break;
        case 4:
          internal::tile<4>(a, in_f, 1, packed, 16, c, out_f, count, kb != 0,
                            b);
          break;
        case 3:
          internal::tile<3>(a, in_f, 1, packed, 16, c, out_f, count, kb != 0,
                            b);
          break;
        case 2:
          internal::tile<2>(a, in_f, 1, packed, 16, c, out_f, count, kb != 0,
                            b);
          break;
        default:
          internal::tile<1>(a, in_f, 1, packed, 16, c, out_f, count, kb != 0,
                            b);
          break;
        }
      }
    }
  for (int r = 0; r < rows; ++r)
    for (int j = o; j < out_f; ++j)
      y[static_cast<std::size_t>(r) * out_f + j] =
          dot(x + static_cast<std::size_t>(r) * in_f,
              w + static_cast<std::size_t>(j) * in_f, in_f) +
          (bias ? bias[j] : 0.f);
}

void linear_backward(const float *x, const float *w, const float *dy, float *dx,
                     float *dw, float *db, int rows, int in_f, int out_f) {
  if (dx)
    internal::accumulate(dy, out_f, 1, w, in_f, dx, rows, in_f, out_f);
  if (dw)
    internal::accumulate(dy, 1, out_f, x, in_f, dw, out_f, in_f, rows);
  if (db)
    for (int r = 0; r < rows; ++r)
      axpy(db, dy + static_cast<std::size_t>(r) * out_f, 1.f, out_f);
}
} // namespace nexus_model::kernels::avx2
