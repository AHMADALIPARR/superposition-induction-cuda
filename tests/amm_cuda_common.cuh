// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Shared setup for the CUDA-vs-CPU tests of AlexNet-MMEP.
#pragma once
#include <cstdio>
#include <vector>

#include "alexnet_mmep/cpu_ref.hpp"
#include "alexnet_mmep/cuda_model.cuh"
#include "test_util.hpp"

namespace ammtest {

inline amm::Config small67() {
  amm::Config c;
  c.image = 67;
  const int ch[5] = {4, 6, 8, 8, 6};
  for (int i = 0; i < 5; ++i) c.ch[i] = ch[i];
  c.fc = 16;
  c.num_classes = 5;
  c.num_routes = 3;
  return c;
}

inline amm::Config mid99() {
  amm::Config c;
  c.image = 99;
  const int ch[5] = {16, 24, 32, 32, 24};
  for (int i = 0; i < 5; ++i) c.ch[i] = ch[i];
  c.fc = 64;
  c.num_classes = 10;
  c.num_routes = 4;
  return c;
}

// Scale the fc6 router so its entropy stays above the bound (penalty fires).
inline void flatten_router6(const amm::Config& c, const sk::ParamTable& t, std::vector<float>& p) {
  amm::Offsets o(c, t);
  for (size_t i = 0; i < (size_t)c.num_routes * c.fc; ++i) p[o.r6.Wr + i] *= 1e-3f;
  for (int k = 0; k < c.num_routes; ++k) p[o.r6.br + k] *= 1e-3f;
}

// The GPU's discrete decisions (pool argmax, which rows bisect, tau, penalty).
inline amm::Freeze freeze_from(const amm::CudaModel& m) {
  amm::Freeze f;
  f.am1 = m.pool_argmax(1);
  f.am2 = m.pool_argmax(2);
  f.am5 = m.pool_argmax(5);
  f.viol6 = m.viol(6);
  f.viol7 = m.viol(7);
  auto t6 = m.tau(6), t7 = m.tau(7);
  f.tau6.assign(t6.begin(), t6.end());
  f.tau7.assign(t7.begin(), t7.end());
  const float b = m.config().entropy_bound;
  f.pen6 = (float)m.ent6() > b;
  f.pen7 = (float)m.ent7() > b;
  return f;
}

inline std::vector<int> labels_for(sk::Rng& r, int N, int C) {
  std::vector<int> l(N);
  for (auto& v : l) v = (int)(r.uniform() * C) % C;
  return l;
}

}  // namespace ammtest
