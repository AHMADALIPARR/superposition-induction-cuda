// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "common/tensor_file.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace sk {

static const char kMagic[8] = {'S', 'K', 'T', '1', 0, 0, 0, 0};

namespace {
struct File {
  FILE* f;
  File(const std::string& p, const char* m) : f(std::fopen(p.c_str(), m)) {
    if (!f) throw std::runtime_error("cannot open " + p);
  }
  ~File() { std::fclose(f); }
  void w(const void* p, size_t n) {
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("write failed");
  }
  void r(void* p, size_t n) {
    if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("truncated tensor file");
  }
};
}  // namespace

void write_tensor_file(const std::string& path, const std::vector<TensorRec>& recs) {
  File f(path, "wb");
  f.w(kMagic, 8);
  uint32_t n = static_cast<uint32_t>(recs.size());
  f.w(&n, 4);
  for (auto& t : recs) {
    uint32_t len = static_cast<uint32_t>(t.name.size());
    f.w(&len, 4);
    f.w(t.name.data(), len);
    uint32_t dt = static_cast<uint32_t>(t.dtype);
    f.w(&dt, 4);
    uint32_t nd = static_cast<uint32_t>(t.shape.size());
    f.w(&nd, 4);
    for (auto s : t.shape) {
      int64_t v = s;
      f.w(&v, 8);
    }
    size_t ne = t.numel();
    if (t.dtype == DType::F32) {
      if (t.f.size() != ne) throw std::runtime_error("size mismatch writing " + t.name);
      f.w(t.f.data(), ne * 4);
    } else {
      if (t.i.size() != ne) throw std::runtime_error("size mismatch writing " + t.name);
      f.w(t.i.data(), ne * 4);
    }
  }
}

std::vector<TensorRec> read_tensor_file(const std::string& path) {
  File f(path, "rb");
  char magic[8];
  f.r(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error(path + ": not an SKT1 file");
  uint32_t n;
  f.r(&n, 4);
  std::vector<TensorRec> out(n);
  for (auto& t : out) {
    uint32_t len;
    f.r(&len, 4);
    if (len > 4096) throw std::runtime_error("bad name length");
    t.name.resize(len);
    f.r(&t.name[0], len);
    uint32_t dt, nd;
    f.r(&dt, 4);
    f.r(&nd, 4);
    if (dt > 1 || nd > 8) throw std::runtime_error("bad record header for " + t.name);
    t.dtype = static_cast<DType>(dt);
    t.shape.resize(nd);
    for (auto& s : t.shape) f.r(&s, 8);
    size_t ne = t.numel();
    if (t.dtype == DType::F32) {
      t.f.resize(ne);
      f.r(t.f.data(), ne * 4);
    } else {
      t.i.resize(ne);
      f.r(t.i.data(), ne * 4);
    }
  }
  return out;
}

const TensorRec* find_tensor(const std::vector<TensorRec>& recs, const std::string& name) {
  for (auto& t : recs)
    if (t.name == name) return &t;
  return nullptr;
}

const TensorRec& need_tensor(const std::vector<TensorRec>& recs, const std::string& name) {
  auto* t = find_tensor(recs, name);
  if (!t) throw std::runtime_error("missing tensor: " + name);
  return *t;
}

TensorRec make_f32(const std::string& name, std::vector<int64_t> shape, const float* data) {
  TensorRec t;
  t.name = name;
  t.shape = std::move(shape);
  t.f.assign(data, data + t.numel());
  return t;
}

TensorRec make_scalar_f32(const std::string& name, float v) {
  TensorRec t;
  t.name = name;
  t.f = {v};
  return t;
}

TensorRec make_scalar_i32(const std::string& name, int32_t v) {
  TensorRec t;
  t.name = name;
  t.dtype = DType::I32;
  t.i = {v};
  return t;
}

float scalar_f32(const std::vector<TensorRec>& recs, const std::string& name, float dflt) {
  auto* t = find_tensor(recs, name);
  if (!t) return dflt;
  if (t->dtype != DType::F32 || t->f.size() != 1) throw std::runtime_error("bad scalar " + name);
  return t->f[0];
}

int32_t scalar_i32(const std::vector<TensorRec>& recs, const std::string& name, int32_t dflt) {
  auto* t = find_tensor(recs, name);
  if (!t) return dflt;
  if (t->dtype != DType::I32 || t->i.size() != 1) throw std::runtime_error("bad scalar " + name);
  return t->i[0];
}

}  // namespace sk
