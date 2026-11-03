// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Plain C++ reference of SuperpositionedBURTIMMA (forward + backward).
// Loops follow the PyTorch code line by line; used to check the CUDA build.
#pragma once
#include <vector>

#include "common/optim.hpp"
#include "sid/model.hpp"

namespace sid {

template <class T>
struct IterCache {
  std::vector<T> x_in;   // [B,S,d] input of this refinement iteration
  std::vector<T> alpha;  // [B,K]
  std::vector<T> Q, Kp, V;  // [B,S,E]
  std::vector<T> Ki;     // [K,B,H,S,dh] K_induced
  std::vector<T> A;      // [K,B,H,S,S] attention probabilities
  std::vector<T> O;      // [K,B,S,E] attention output before W_O
  std::vector<T> h;      // [K,B,S,d] path outputs before interference
  std::vector<T> Y;      // [K,B,S,d] path outputs after interference
  std::vector<T> sp;     // [B,S,d] superposed
  std::vector<T> t;      // [B,S,d] tanh(collapse_proj(sp))
  std::vector<T> xhat;   // [B,S,d] normalised t
  std::vector<T> rstd;   // [B*S]
  std::vector<T> out;    // [B,S,d] LayerNorm output
};

template <class T>
struct CpuState {
  int B = 0, S = 0;
  std::vector<IterCache<T>> it;
  std::vector<T> norms, vn, G;  // [K], [K,d], [K,K]
  std::vector<int> mask;        // [K,K] 1 where |Gamma_kj| > threshold (j != k)
  T orth = 0;
  int iterations = 0;
  double final_delta = 0;
  std::vector<double> deltas;
  std::vector<T> y;  // [B,S,d] final output (after output_proj)
};

// Optional frozen discrete decisions (used by the gradient checks so finite
// differences do not cross a threshold, and to align CPU with a GPU run).
struct Freeze {
  int iterations = -1;    // >0: run exactly this many iterations
  std::vector<int> mask;  // non-empty: use this interference mask
};

template <class T>
void cpu_forward(const Config& c, const sk::ParamTable& tab, const T* p, const T* x, int B, int S,
                 CpuState<T>& st, const Freeze* fz = nullptr);

// grad must have tab.total elements (overwritten). dx (optional) gets dL/dx.
// Loss seen by the backward: L = <dy, y> + orth_weight * orth_loss.
template <class T>
void cpu_backward(const Config& c, const sk::ParamTable& tab, const T* p, const CpuState<T>& st,
                  const T* dy, T orth_weight, T* grad, T* dx);

// One training step: loss = mean((y - target)^2) + orthogonality_reg * orth.
// Returns the loss before the update.
template <class T>
T cpu_train_step_adamw(const Config& c, const sk::ParamTable& tab, T* p, T* m, T* v, int step,
                       const sk::AdamWConfig& oc, const T* x, const T* target, int B, int S);
template <class T>
T cpu_train_step_sgd(const Config& c, const sk::ParamTable& tab, T* p, T* buf, int step,
                     const sk::SgdConfig& oc, const T* x, const T* target, int B, int S);

}  // namespace sid
