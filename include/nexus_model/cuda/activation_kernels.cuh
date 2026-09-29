#pragma once

/**
 * @file activation_kernels.cuh
 * @brief Cihaz üstü ReLU ve GELU (tanh yaklaşımı).
 */

namespace nexus_model::cuda {

void relu(const float* x, float* y, int n);
void gelu_tanh(const float* x, float* y, int n);

}  // namespace nexus_model::cuda
