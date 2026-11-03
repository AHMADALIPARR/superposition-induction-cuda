// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Flat parameter layout shared by the CPU reference and the CUDA code.
// Every tensor lives at a fixed offset inside one contiguous buffer, so the
// optimizers and the weight I/O can treat the whole model as one array.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sk {

struct ParamEntry {
  std::string name;              // PyTorch state_dict key
  std::vector<int64_t> shape;
  size_t offset = 0;
  size_t size = 0;
  bool trainable_grad = true;    // false: PyTorch leaves .grad = None (skipped by optimizers)
};

struct ParamTable {
  std::vector<ParamEntry> entries;
  size_t total = 0;

  size_t add(const std::string& name, std::vector<int64_t> shape, bool has_grad = true) {
    ParamEntry e;
    e.name = name;
    e.shape = std::move(shape);
    e.size = 1;
    for (auto s : e.shape) e.size *= static_cast<size_t>(s);
    e.offset = total;
    e.trainable_grad = has_grad;
    total += e.size;
    entries.push_back(e);
    return e.offset;
  }
  const ParamEntry& get(const std::string& name) const {
    for (auto& e : entries)
      if (e.name == name) return e;
    throw std::runtime_error("unknown parameter: " + name);
  }
  size_t off(const std::string& name) const { return get(name).offset; }
};

}  // namespace sk
