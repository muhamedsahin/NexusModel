#pragma once

/**
 * @file fused_layernorm.cuh
 * @brief mean, varyans, normalize, scale ve shift tek çekirdekte.
 */

namespace nexus_model::cuda {

void fused_layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps);

}  // namespace nexus_model::cuda
