#pragma once

/**
 * @file avx512_activation_kernels.hpp
 * @brief AVX-512F bildirimleri. NEXUS_MODEL_ENABLE_AVX512 ile derlenir.
 *
 * Şu an ReLU, dot ve linear_forward. Diğer aktivasyonlar AVX2 tablosunda kalır.
 */

#include "nexus_model/simd/kernel_decls.hpp"
