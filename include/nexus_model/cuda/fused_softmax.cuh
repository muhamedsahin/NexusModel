#pragma once

/**
 * @file fused_softmax.cuh
 * @brief Son boyut üzerinde tek geçişli softmax. Yalnızca NEXUS_MODEL_WITH_CUDA.
 *
 * Host tensörü buraya kopyalanmaz. Çağıran cihaz işaretçisi verir.
 */

namespace nexus_model::cuda {

void fused_softmax(const float* x, float* y, int rows, int cols);

}  // namespace nexus_model::cuda
