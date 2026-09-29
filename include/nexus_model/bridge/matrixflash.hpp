#pragma once

/**
 * @file matrixflash.hpp
 * @brief İsteğe bağlı GPU GEMM köprüsü.
 *
 * matrix_pro::multiply çıktısı cihaz belleğindedir ve CPU'da bir GEMM sunmaz.
 * NexusModel'in varsayılan sıcak yolu bu yüzden host AVX GEMM'dir. Bu başlık,
 * NEXUS_MODEL_WITH_MATRIXFLASH açıkken çarpımı MatrixFlash'e bırakmak isteyen
 * üst katman içindir. NexusModel kendi cuBLAS'ını yazmaz.
 *
 * Köprü, çekirdek kütüphaneye bağlanmaz. Üst katman hem MatrixFlash'i hem bu
 * başlığı kendi hedefinde derler.
 */

#if defined(NEXUS_MODEL_WITH_MATRIXFLASH)

#include "matrix_pro/ops/product.hpp"

namespace nexus_model::bridge {

inline matrix_pro::Matrix gemm(const matrix_pro::Matrix& left, const matrix_pro::Matrix& right) {
  return matrix_pro::multiply(left, right);
}

}  // namespace nexus_model::bridge

#endif
