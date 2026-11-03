// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CUDA kernels for AlexNetMMEP (host-side launchers). Convolutions are im2col
// + cuBLAS GEMM (no cuDNN); see docs/kernels.md for the layouts.
#pragma once
#include <cuda_runtime.h>

namespace amm {
namespace kern {

// "Wide" im2col for a chunk of nb images starting at x (NCHW, square H):
//   col[(c*k + ky)*k + kx][nl*Ho*Ho + oy*Ho + ox] = x[nl, c, oy*s - p + ky, ox*s - p + kx]
// (zero outside the image).
void im2col_wide(const float* x, int nb, int C, int H, int k, int s, int p, int Ho, float* col);
// Adjoint of im2col_wide: dx[nl,c,iy,ix] = sum of the dcol entries that read it.
// Overwrites dx.
void col2im_wide(const float* dcol, int nb, int C, int H, int k, int s, int p, int Ho, float* dx);
// Z[nl, o, q] = Zw[o, nl*HW + q] + b[o]
void unwide_bias(const float* Zw, const float* b, int nb, int O, int HW, float* Z);
// Dw[o, nl*HW + q] = D[nl, o, q]
void to_wide(const float* D, int nb, int O, int HW, float* Dw);

// SmoothLeaky: a = alpha*z + ((1-alpha)/beta) * (softplus(beta*z) - ln 2)
void smooth_leaky_fwd(const float* z, float* a, size_t n, float alpha, float beta);
// d *= f'(z)  (in place)
void smooth_leaky_bwd(const float* z, float* d, size_t n, float alpha, float beta);

// MaxPool2d(3, 2) over NC planes of H x H -> Ho x Ho; am = argmax index in plane.
void maxpool_fwd(const float* x, int NC, int H, int Ho, float* y, int* am);
// dx = 0, then dx[argmax] += dy (atomics; windows overlap).
void maxpool_bwd(const float* dy, const int* am, int NC, int H, int Ho, float* dx);

// constrained_softmax (burt_imma/burt.py) on [rows, K] logits, one block.
// Outputs alpha [rows,K], tau [rows] (1 where the row kept plain softmax),
// viol [rows], ent [rows] = -sum a log(a + 1e-10), iters [1] = bisection steps.
// scratch: 2*rows floats. viol_fz/tau_fz (device, may be null) replay decisions.
void constrained_softmax(const float* logits, int rows, int K, float bound, float* alpha,
                         float* tau, int* viol, float* ent, int* iters, float* scratch,
                         const int* viol_fz, const float* tau_fz);
// dl = alpha * (dalpha - <alpha, dalpha>) / tau
void constrained_softmax_bwd(const float* alpha, const float* dalpha, const float* tau, int rows,
                             int K, float* dl);

// out[n,i] = sum_k alpha[n,k] * tz[k,n,i]
void router_combine(const float* alpha, const float* tz, int N, int d, int K, float* out);
// dalpha[n,k] = <dout[n], tz[k,n]> - dent[n] * (log(a + 1e-10) + a / (a + 1e-10))
void router_dalpha(const float* dout, const float* tz, const float* alpha, const float* dent,
                   int N, int d, int K, float* dalpha);
// dtz[n,i] = alpha[n,k] * dout[n,i]
void scale_rows(const float* dout, const float* alpha, int k, int K, int N, int d, float* dtz);

// CrossEntropyLoss(mean): loss_rows[n] = lse - l[label], dlogits = (softmax - onehot)/N,
// pred[n] = argmax (first).
void cross_entropy(const float* logits, const int* labels, int N, int C, float* loss_rows,
                   float* dlogits, int* pred);

}  // namespace kern
}  // namespace amm

namespace amm {
namespace kern {
// out[i] += sum_j X[i*N + j]
void row_sum_add(const float* X, int M, int N, float* out);
}  // namespace kern
}  // namespace amm
