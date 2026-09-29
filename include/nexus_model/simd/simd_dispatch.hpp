#pragma once

/**
 * @file simd_dispatch.hpp
 * @brief Çalışma zamanı ISA seçimi. AVX-512 varsa o, yoksa AVX2, yoksa skalar.
 *
 * Uygulama src/simd/simd_dispatch.cpp içindedir. Çekirdek gövdeleri
 * avx2_kernels.cpp ve avx512_kernels.cpp dosyalarındadır.
 */

#include "nexus_model/core/kernels.hpp"

namespace nexus_model::simd {

using kernels::set_force_scalar;

}  // namespace nexus_model::simd
