// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Model-specific kernels of the Superpositioned Induction Daemon.
// Layout notes are in docs/kernels.md.
#pragma once
#include <cstddef>

namespace sid {
namespace kern {

// Single block: vn = P / max(||P||, eps), Gamma = vn vn^T, mask = |Gamma| > thr
// (j != k; overridden by mask_override when non-null), orth = sum_{i!=j} Gamma^2.
void basis_fwd(const float* P, int K, int d, float eps, float thr, const int* mask_override,
               float* norms, float* vn, float* G, int* mask, float* orth);
// dG += w_orth * 2 Gamma (off-diagonal), then normalize backward into dP (+=).
void basis_bwd(const float* P, const float* norms, const float* vn, const float* G, float* dG,
               int K, int d, float eps, float w_orth, float* dP);

// alpha[b,:] = softmax(x[b,0,:] P^T / tau)
void alpha_fwd(const float* x, const float* P, int B, int S, int d, int K, float tau, float* alpha);
// dl = alpha (dalpha - <alpha, dalpha>) / tau; dx[b,0,:] += dl P; dP += dl^T cls
void alpha_bwd(const float* x, const float* P, const float* alpha, const float* dalpha, int B,
               int S, int d, int K, float tau, float* dx, float* dP);

// Sequential in-place interference + superposition, per element:
//   Y_k = h_k + sum_{j != k, mask_kj} Gamma_kj Y_j(current), sp = sum_k alpha_k Y_k
void interference_fwd(const float* h, const float* alpha, const float* G, const int* mask, int K,
                      int B, size_t per_b, float* Y, float* sp);
// Reverse of the above. Writes dh (all K paths), accumulates dalpha and dG.
void interference_bwd(const float* dsp, const float* h, const float* Y, const float* alpha,
                      const float* G, const int* mask, int K, int B, size_t per_b, float* dh,
                      float* dalpha, float* dG);

// t = tanh(z) (in place), out = LN(t) * g + b; keeps xhat and rstd.
void tanh_ln_fwd(float* t, const float* g, const float* b, int rows, int d, float eps,
                 float* xhat, float* rstd, float* out);
// dz = dLN/dt * (1 - t^2); also dgamma += sum dout*xhat, dbeta += sum dout.
void tanh_ln_bwd(const float* dout, const float* t, const float* xhat, const float* rstd,
                 const float* g, int rows, int d, float* dz, float* dgamma, float* dbeta);

// dy = 2 (y - target) / n
void mse_grad(const float* y, const float* target, size_t n, float* dy);

}  // namespace kern
}  // namespace sid
