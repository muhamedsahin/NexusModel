#pragma once

/**
 * @file serialization.hpp
 * @brief state_dict ikili biçimi: sihir `NXM1`, isim, şekil, ham float32.
 *
 * Hiyerarşik anahtarlar `0.weight`, `1.bias`, `norm.running_mean` şeklindedir.
 * Eksik veya fazla anahtar `load_state_dict` içinde açıklayıcı hata üretir.
 */

#include "nexus_model/core/error.hpp"
#include "nexus_model/core/tensor.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace nexus_model {

struct StateDict {
  std::map<std::string, Tensor> items;
};

inline void save_state_dict(const StateDict& state, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw ModelError("cannot open state file for writing: " + path);
  const char magic[4] = {'N', 'X', 'M', '1'};
  out.write(magic, 4);
  const std::uint32_t count = static_cast<std::uint32_t>(state.items.size());
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const auto& [name, tensor] : state.items) {
    const std::uint32_t name_len = static_cast<std::uint32_t>(name.size());
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    out.write(name.data(), static_cast<std::streamsize>(name.size()));
    const std::uint32_t rank = tensor.rank();
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    for (std::uint32_t i = 0; i < rank; ++i) {
      const std::uint64_t dim = tensor.dims()[i];
      out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    if (tensor.numel() > 0) {
      out.write(reinterpret_cast<const char*>(tensor.data()),
                static_cast<std::streamsize>(tensor.numel() * sizeof(float)));
    }
  }
  if (!out) throw ModelError("failed while writing state file");
}

inline StateDict load_state_dict_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ModelError("cannot open state file for reading: " + path);
  char magic[4] = {};
  in.read(magic, 4);
  if (std::strncmp(magic, "NXM1", 4) != 0) throw ModelError("state file magic mismatch");
  std::uint32_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  StateDict state;
  for (std::uint32_t n = 0; n < count; ++n) {
    std::uint32_t name_len = 0;
    in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    std::string name(name_len, '\0');
    in.read(name.data(), static_cast<std::streamsize>(name_len));
    std::uint32_t rank = 0;
    in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    if (rank > kMaxRank) throw ModelError("state file rank is invalid");
    std::size_t dims[kMaxRank];
    for (std::uint32_t i = 0; i < rank; ++i) {
      std::uint64_t dim = 0;
      in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
      dims[i] = static_cast<std::size_t>(dim);
    }
    Tensor tensor;
    tensor.resize_dims(dims, static_cast<std::uint8_t>(rank));
    if (tensor.numel() > 0) {
      in.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(tensor.numel() * sizeof(float)));
    }
    state.items.emplace(std::move(name), std::move(tensor));
  }
  if (!in) throw ModelError("truncated state file");
  return state;
}

}  // namespace nexus_model
