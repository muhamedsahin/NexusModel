#pragma once

/**
 * @file avx2_activation_kernels.hpp
 * @brief AVX2+FMA aktivasyon ve GEMM bildirimleri.
 *
 * Gövde src/simd/avx2_kernels.cpp. Softmax: skalar max, sonra vektör exp
 * (Cephes) ve toplam. In-place güvenlidir.
 */

#include "nexus_model/simd/kernel_decls.hpp"
