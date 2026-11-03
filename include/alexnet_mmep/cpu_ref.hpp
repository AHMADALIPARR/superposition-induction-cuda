// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Plain C++ reference for AlexNetMMEP: forward, loss, backward and the
// trainer's SGD step. Direct (loop) convolutions, templated on float/double.
// Used as the ground truth for the CUDA tests and checked against finite
// differences in double.
#pragma once
#include <vector>

#include "alexnet_mmep/model.hpp"
#include "common/optim.hpp"

namespace amm {

template <class T>
struct RouterCache {
  std::vector<T> logits, alpha, ent;  // [N,K], [N,K], [N]
  std::vector<T> tz;                  // route_transforms[k](x): [K,N,d]
  std::vector<T> out;                 // [N,d]
  std::vector<T> tau;                 // per row; 1 for rows that kept plain softmax
  std::vector<int> viol;              // per row: entropy of plain softmax > bound
  int bisect_iters = 0;               // bisection iterations run (0 if no row violated)
};

template <class T>
struct CpuState {
  int N = 0;
  Shapes s{};
  std::vector<T> x;                   // input copy [N, in_ch, H, W]
  std::vector<T> z[5], a[5];          // conv pre-activation / SmoothLeaky output
  std::vector<T> p1, p2, p5;          // pool outputs
  std::vector<int> am1, am2, am5;     // pool argmax, index inside the input plane
  std::vector<T> z6, a6, z7, a7;      // fc6 / fc7 pre-activation and activation
  RouterCache<T> r6, r7;
  std::vector<T> logits;              // [N, classes]
  T ent6 = 0, ent7 = 0;               // aux entropy_fc6 / entropy_fc7 (batch means)
};

// Decisions to replay (for finite differences and GPU comparison).
struct Freeze {
  std::vector<int> am1, am2, am5;
  std::vector<int> viol6, viol7;
  std::vector<double> tau6, tau7;
  int pen6 = -1, pen7 = -1;           // -1: decide from the data
};

struct LossInfo {
  double ce = 0, loss = 0, ent6 = 0, ent7 = 0, accuracy = 0;
  int pen6 = 0, pen7 = 0;
};

template <class T>
T smooth_leaky(T x, T a, T b);
template <class T>
T smooth_leaky_grad(T x, T a, T b);  // f'(x)

// constrained_softmax from burt_imma/burt.py on rows of length K.
// Writes alpha, tau (1 for non-violating rows), viol; returns bisection iters.
template <class T>
int constrained_softmax(const T* logits, int rows, int K, T bound, T* alpha, T* tau, int* viol,
                        const int* viol_fz = nullptr, const double* tau_fz = nullptr);

template <class T>
void cpu_forward(const Config& c, const sk::ParamTable& t, const T* p, const T* x, int N,
                 CpuState<T>& st, const Freeze* fz = nullptr);

// CrossEntropyLoss(mean) + the trainer's entropy penalty. Fills dlogits [N,C]
// and dent6/dent7 [N] (gradients of the loss w.r.t. logits and per-row entropy).
template <class T>
LossInfo cpu_loss(const Config& c, const CpuState<T>& st, const int* labels,
                  std::vector<T>& dlogits, std::vector<T>& dent6, std::vector<T>& dent7,
                  const Freeze* fz = nullptr);

// Gradients of the loss defined by (dlogits, dent6, dent7). grad is overwritten
// (size t.total). dx (size N*in_ch*H*W) may be null.
template <class T>
void cpu_backward(const Config& c, const sk::ParamTable& t, const T* p, const CpuState<T>& st,
                  const T* dlogits, const T* dent6, const T* dent7, T* grad, T* dx);

// AlexNetMMEPTrainer.train_step: zero_grad, forward, CE + penalty, backward,
// torch.optim.SGD(momentum, weight_decay). step is 1-based.
template <class T>
LossInfo cpu_train_step_sgd(const Config& c, const sk::ParamTable& t, std::vector<T>& p,
                            std::vector<T>& momentum_buf, int step, const T* x, const int* labels,
                            int N, const sk::SgdConfig& sc);

}  // namespace amm
