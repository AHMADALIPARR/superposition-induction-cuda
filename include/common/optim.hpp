// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Optimizer math, written once and used by the CPU reference.
// The CUDA kernels in src/common/optim.cu apply the same update per element.
// Both follow torch.optim semantics (see docs/math-spec.md).
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/param_table.hpp"

namespace sk {

struct AdamWConfig {
  float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, weight_decay = 0.01f;
};

struct SgdConfig {
  float lr = 0.01f, momentum = 0.9f, weight_decay = 0.0f;
};

// Per-element mask: 1 where PyTorch would have a gradient, 0 where .grad is None.
inline std::vector<unsigned char> grad_mask(const ParamTable& t) {
  std::vector<unsigned char> m(t.total, 1);
  for (auto& e : t.entries)
    if (!e.trainable_grad)
      for (size_t i = 0; i < e.size; ++i) m[e.offset + i] = 0;
  return m;
}

// torch.optim.AdamW (amsgrad=False, maximize=False). step is 1-based.
template <class T>
void adamw_step_host(const AdamWConfig& c, int step, T* p, const T* g, T* m, T* v,
                     const unsigned char* mask, size_t n) {
  const T bc1 = T(1) - std::pow(T(c.beta1), T(step));
  const T bc2 = T(1) - std::pow(T(c.beta2), T(step));
  const T step_size = T(c.lr) / bc1;
  const T bc2_sqrt = std::sqrt(bc2);
  for (size_t i = 0; i < n; ++i) {
    if (mask && !mask[i]) continue;
    p[i] *= T(1) - T(c.lr) * T(c.weight_decay);
    m[i] = T(c.beta1) * m[i] + (T(1) - T(c.beta1)) * g[i];
    v[i] = T(c.beta2) * v[i] + (T(1) - T(c.beta2)) * g[i] * g[i];
    const T denom = std::sqrt(v[i]) / bc2_sqrt + T(c.eps);
    p[i] -= step_size * m[i] / denom;
  }
}

// torch.optim.SGD (dampening=0, nesterov=False). step is 1-based; on step 1 the
// momentum buffer is initialised to the gradient, as in PyTorch.
template <class T>
void sgd_step_host(const SgdConfig& c, int step, T* p, const T* g, T* buf,
                   const unsigned char* mask, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (mask && !mask[i]) continue;
    T d = g[i] + T(c.weight_decay) * p[i];
    if (c.momentum != 0.0f) {
      buf[i] = (step == 1) ? d : T(c.momentum) * buf[i] + d;
      d = buf[i];
    }
    p[i] -= T(c.lr) * d;
  }
}

}  // namespace sk
