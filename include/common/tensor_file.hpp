// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// SKT1 tensor container: a flat list of named tensors (see docs/weights-format.md).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sk {

enum class DType : uint32_t { F32 = 0, I32 = 1 };

struct TensorRec {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  std::vector<float> f;    // used when dtype == F32
  std::vector<int32_t> i;  // used when dtype == I32
  size_t numel() const {
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return n;
  }
};

void write_tensor_file(const std::string& path, const std::vector<TensorRec>& recs);
std::vector<TensorRec> read_tensor_file(const std::string& path);
const TensorRec* find_tensor(const std::vector<TensorRec>& recs, const std::string& name);
const TensorRec& need_tensor(const std::vector<TensorRec>& recs, const std::string& name);

TensorRec make_f32(const std::string& name, std::vector<int64_t> shape, const float* data);
TensorRec make_scalar_f32(const std::string& name, float v);
TensorRec make_scalar_i32(const std::string& name, int32_t v);
float scalar_f32(const std::vector<TensorRec>& recs, const std::string& name, float dflt);
int32_t scalar_i32(const std::vector<TensorRec>& recs, const std::string& name, int32_t dflt);

}  // namespace sk
