// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Shared setup for the CUDA-vs-CPU tests of the superposition model.
#pragma once
#include <cmath>
#include <vector>

#include "sid/cpu_ref.hpp"
#include "sid/cuda_model.cuh"
#include "test_util.hpp"

namespace sidtest {

// Random parameters with a non-orthogonal basis so interference is exercised.
inline std::vector<float> make_params(const sid::Config& c, const sk::ParamTable& t, uint64_t seed,
                                      double basis_noise) {
  auto p = sid::init_params(c, t, seed);
  sid::Offsets o(t);
  sk::Rng r(seed ^ 0xabcdef);
  for (size_t i = 0; i < (size_t)c.num_paths * c.d_model; ++i)
    p[o.path_basis + i] += (float)(basis_noise * r.normal());
  for (int i = 0; i < c.d_model; ++i) {
    p[o.ln_g + i] = (float)(1.0 + 0.2 * r.normal());
    p[o.ln_b + i] = (float)(0.1 * r.normal());
  }
  return p;
}

inline sid::Freeze freeze_from(const sid::CudaModel& m) {
  sid::Freeze f;
  f.iterations = m.iterations();
  f.mask = m.mask();
  return f;
}

}  // namespace sidtest
