#pragma once

/**
 * @file error.hpp
 * @brief NexusModel hata tipi. Katman sözleşmesi bozulunca açıklayıcı mesaj taşır.
 */

#include <stdexcept>
#include <string>

namespace nexus_model {

class ModelError : public std::runtime_error {
 public:
  explicit ModelError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace nexus_model
