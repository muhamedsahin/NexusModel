#pragma once

/**
 * @file nexus_model.hpp
 * @brief Tek include noktası. NexusModel katman, konteyner ve serileştirme API'si.
 *
 * Hesaplama motoru MatrixFlash Pro'dur; bu başlık onu zorunlu kılmaz. CPU sıcak
 * yolu AVX2/AVX-512, isteğe bağlı GPU yolu `NEXUS_MODEL_WITH_CUDA` ve
 * `NEXUS_MODEL_WITH_MATRIXFLASH` ile açılır. Ayrıntı: `core/details.txt`.
 */

#include "nexus_model/activations/activations.hpp"
#include "nexus_model/containers/module_dict.hpp"
#include "nexus_model/containers/module_list.hpp"
#include "nexus_model/containers/sequential.hpp"
#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/micro_tape.hpp"
#include "nexus_model/core/module.hpp"
#include "nexus_model/core/parameter.hpp"
#include "nexus_model/core/serialization.hpp"
#include "nexus_model/core/tensor.hpp"
#include "nexus_model/layers/attention.hpp"
#include "nexus_model/layers/conv.hpp"
#include "nexus_model/layers/dropout.hpp"
#include "nexus_model/layers/embedding.hpp"
#include "nexus_model/layers/flatten_reshape.hpp"
#include "nexus_model/layers/linear.hpp"
#include "nexus_model/layers/normalization.hpp"
#include "nexus_model/layers/pooling.hpp"
#include "nexus_model/layers/recurrent.hpp"
#include "nexus_model/layers/residual.hpp"
