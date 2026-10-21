#pragma once

/**
 * @file execution_policy.hpp
 * @brief Linear GEMM'sini mevcut cekirdekte ya da disaridan takilan bir
 * paketlenmis carpimda kosar. Varsayilan, NexusModel'in kendi AVX yoludur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

namespace nexus_model {

enum class MatmulPolicy { CpuKernel = 0, Hybrid = 1 };

using PackedGemmFn = void (*)(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n,
                              int k);

void set_matmul_policy(MatmulPolicy policy);
[[nodiscard]] MatmulPolicy matmul_policy();
void set_packed_gemm(PackedGemmFn function);
[[nodiscard]] PackedGemmFn packed_gemm();

}  // namespace nexus_model
