// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "common/cuda_util.cuh"
#include "common/device_reduce.cuh"
#include "sid/kernels.cuh"
#include "sid/model.hpp"

namespace sid {
namespace kern {

using sk::block_sum;
using sk::blocks_for;

// ---------------------------------------------------------------- basis ----
__global__ void k_basis_fwd(const float* P, int K, int d, float eps, float thr,
                            const int* mask_override, float* norms, float* vn, float* G, int* mask,
                            float* orth) {
  for (int k = 0; k < K; ++k) {
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += P[k * d + i] * P[k * d + i];
    s = block_sum(s);
    const float n = sqrtf(s);
    const float den = fmaxf(n, eps);
    if (threadIdx.x == 0) norms[k] = n;
    for (int i = threadIdx.x; i < d; i += blockDim.x) vn[k * d + i] = P[k * d + i] / den;
  }
  __syncthreads();
  float o = 0.f;
  for (int a = 0; a < K; ++a)
    for (int b = 0; b < K; ++b) {
      float s = 0.f;
      for (int i = threadIdx.x; i < d; i += blockDim.x) s += vn[a * d + i] * vn[b * d + i];
      s = block_sum(s);
      if (threadIdx.x == 0) {
        G[a * K + b] = s;
        int m = 0;
        if (a != b) {
          o += s * s;
          m = mask_override ? mask_override[a * K + b] : (fabsf(s) > thr ? 1 : 0);
        }
        mask[a * K + b] = m;
      }
    }
  if (threadIdx.x == 0) *orth = o;
}

void basis_fwd(const float* P, int K, int d, float eps, float thr, const int* mask_override,
               float* norms, float* vn, float* G, int* mask, float* orth) {
  k_basis_fwd<<<1, 256>>>(P, K, d, eps, thr, mask_override, norms, vn, G, mask, orth);
  SK_LAUNCH_CHECK();
}

__global__ void k_basis_bwd(const float* P, const float* norms, const float* vn, const float* G,
                            float* dG, int K, int d, float eps, float w_orth, float* dP) {
  __shared__ float sdG[kMaxPaths * kMaxPaths];
  if (threadIdx.x < K * K) {
    int a = threadIdx.x / K, b = threadIdx.x % K;
    float v = dG[threadIdx.x];
    if (a != b) v += w_orth * 2.f * G[threadIdx.x];
    sdG[threadIdx.x] = v;
  }
  __syncthreads();
  for (int k = 0; k < K; ++k) {
    float dot = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      float dv = 0.f;
      for (int j = 0; j < K; ++j) dv += (sdG[k * K + j] + sdG[j * K + k]) * vn[j * d + i];
      dot += vn[k * d + i] * dv;
    }
    dot = block_sum(dot);
    const float n = norms[k];
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      float dv = 0.f;
      for (int j = 0; j < K; ++j) dv += (sdG[k * K + j] + sdG[j * K + k]) * vn[j * d + i];
      if (n > eps)
        dP[k * d + i] += (dv - vn[k * d + i] * dot) / n;
      else
        dP[k * d + i] += dv / eps;
    }
  }
}

void basis_bwd(const float* P, const float* norms, const float* vn, const float* G, float* dG,
               int K, int d, float eps, float w_orth, float* dP) {
  k_basis_bwd<<<1, 256>>>(P, norms, vn, G, dG, K, d, eps, w_orth, dP);
  SK_LAUNCH_CHECK();
}

// ---------------------------------------------------------------- alpha ----
__global__ void k_alpha_fwd(const float* x, const float* P, int S, int d, int K, float tau,
                            float* alpha) {
  const int b = blockIdx.x;
  const float* cls = x + (size_t)b * S * d;
  __shared__ float lg[kMaxPaths];
  for (int k = 0; k < K; ++k) {
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += cls[i] * P[k * d + i];
    s = block_sum(s);
    if (threadIdx.x == 0) lg[k] = s / tau;
  }
  if (threadIdx.x == 0) {
    float mx = -INFINITY;
    for (int k = 0; k < K; ++k) mx = fmaxf(mx, lg[k]);
    float z = 0.f;
    for (int k = 0; k < K; ++k) z += expf(lg[k] - mx);
    for (int k = 0; k < K; ++k) alpha[b * K + k] = expf(lg[k] - mx) / z;
  }
}

void alpha_fwd(const float* x, const float* P, int B, int S, int d, int K, float tau,
               float* alpha) {
  k_alpha_fwd<<<B, 256>>>(x, P, S, d, K, tau, alpha);
  SK_LAUNCH_CHECK();
}

__global__ void k_alpha_bwd(const float* x, const float* P, const float* alpha,
                            const float* dalpha, int S, int d, int K, float tau, float* dx,
                            float* dP) {
  const int b = blockIdx.x;
  __shared__ float dl[kMaxPaths];
  if (threadIdx.x == 0) {
    float dot = 0.f;
    for (int k = 0; k < K; ++k) dot += alpha[b * K + k] * dalpha[b * K + k];
    for (int k = 0; k < K; ++k) dl[k] = alpha[b * K + k] * (dalpha[b * K + k] - dot) / tau;
  }
  __syncthreads();
  const float* cls = x + (size_t)b * S * d;
  float* dcls = dx + (size_t)b * S * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += dl[k] * P[k * d + i];
      atomicAdd(&dP[k * d + i], dl[k] * cls[i]);
    }
    dcls[i] += acc;
  }
}

void alpha_bwd(const float* x, const float* P, const float* alpha, const float* dalpha, int B,
               int S, int d, int K, float tau, float* dx, float* dP) {
  k_alpha_bwd<<<B, 256>>>(x, P, alpha, dalpha, S, d, K, tau, dx, dP);
  SK_LAUNCH_CHECK();
}

// --------------------------------------------------------- interference ----
template <int K>
__global__ void k_interf_fwd(const float* __restrict__ h, const float* __restrict__ alpha,
                             const float* __restrict__ G, const int* __restrict__ mask, int B,
                             size_t per_b, float* __restrict__ Y, float* __restrict__ sp) {
  __shared__ float sG[K * K];
  __shared__ int sM[K * K];
  if (threadIdx.x < K * K) { sG[threadIdx.x] = G[threadIdx.x]; sM[threadIdx.x] = mask[threadIdx.x]; }
  __syncthreads();
  const size_t n = per_b * B;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const int b = (int)(i / per_b);
    float y[K];
#pragma unroll
    for (int k = 0; k < K; ++k) y[k] = h[k * n + i];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < K; ++j)
        if (j != k && sM[k * K + j]) acc += sG[k * K + j] * y[j];
      y[k] += acc;
    }
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      Y[k * n + i] = y[k];
      s += alpha[b * K + k] * y[k];
    }
    sp[i] = s;
  }
}

template <int K>
__global__ void k_interf_bwd(const float* __restrict__ dsp, const float* __restrict__ h,
                             const float* __restrict__ Y, const float* __restrict__ alpha,
                             const float* __restrict__ G, const int* __restrict__ mask, int B,
                             size_t per_b, float* __restrict__ dh, float* __restrict__ dalpha,
                             float* __restrict__ dG) {
  __shared__ float sG[K * K];
  __shared__ int sM[K * K];
  if (threadIdx.x < K * K) { sG[threadIdx.x] = G[threadIdx.x]; sM[threadIdx.x] = mask[threadIdx.x]; }
  __syncthreads();
  const int b = blockIdx.y;
  const size_t n = per_b * B;
  float al[K], dal[K], dg[K * K];
#pragma unroll
  for (int k = 0; k < K; ++k) { al[k] = alpha[b * K + k]; dal[k] = 0.f; }
#pragma unroll
  for (int q = 0; q < K * K; ++q) dg[q] = 0.f;
  for (size_t li = blockIdx.x * (size_t)blockDim.x + threadIdx.x; li < per_b;
       li += (size_t)gridDim.x * blockDim.x) {
    const size_t i = (size_t)b * per_b + li;
    const float ds = dsp[i];
    float g[K], yv[K], hv[K], dhk[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      yv[k] = Y[k * n + i];
      hv[k] = h[k * n + i];
      g[k] = al[k] * ds;
      dal[k] += ds * yv[k];
      dhk[k] = 0.f;
    }
#pragma unroll
    for (int k = K - 1; k >= 0; --k) {
      dhk[k] += g[k];
#pragma unroll
      for (int j = 0; j < K; ++j) {
        if (j == k || !sM[k * K + j]) continue;
        const float gk = sG[k * K + j];
        if (j < k) {
          g[j] += gk * g[k];
          dg[k * K + j] += g[k] * yv[j];
        } else {
          dhk[j] += gk * g[k];
          dg[k * K + j] += g[k] * hv[j];
        }
      }
    }
#pragma unroll
    for (int k = 0; k < K; ++k) dh[k * n + i] = dhk[k];
  }
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float s = block_sum(dal[k]);
    if (threadIdx.x == 0) atomicAdd(&dalpha[b * K + k], s);
  }
#pragma unroll
  for (int q = 0; q < K * K; ++q) {
    if (q / K == q % K) continue;
    float s = block_sum(dg[q]);
    if (threadIdx.x == 0 && s != 0.f) atomicAdd(&dG[q], s);
  }
}

template <int K>
static void interf_fwd_t(const float* h, const float* alpha, const float* G, const int* mask,
                         int B, size_t per_b, float* Y, float* sp) {
  unsigned g = blocks_for(per_b * B);
  if (g > 65535u * 4) g = 65535u * 4;
  k_interf_fwd<K><<<g, 256>>>(h, alpha, G, mask, B, per_b, Y, sp);
  SK_LAUNCH_CHECK();
}
template <int K>
static void interf_bwd_t(const float* dsp, const float* h, const float* Y, const float* alpha,
                         const float* G, const int* mask, int B, size_t per_b, float* dh,
                         float* dalpha, float* dG) {
  unsigned gx = blocks_for(per_b);
  if (gx > 256) gx = 256;  // grid-stride; keeps the dG/dalpha atomics few
  dim3 grid(gx, B);
  k_interf_bwd<K><<<grid, 256>>>(dsp, h, Y, alpha, G, mask, B, per_b, dh, dalpha, dG);
  SK_LAUNCH_CHECK();
}

#define SID_DISPATCH_K(K, CALL)                                         \
  switch (K) {                                                         \
    case 1: CALL(1); break;                                            \
    case 2: CALL(2); break;                                            \
    case 3: CALL(3); break;                                            \
    case 4: CALL(4); break;                                            \
    case 5: CALL(5); break;                                            \
    case 6: CALL(6); break;                                            \
    case 7: CALL(7); break;                                            \
    case 8: CALL(8); break;                                            \
    default: throw std::runtime_error("num_paths must be in 1..8");    \
  }

void interference_fwd(const float* h, const float* alpha, const float* G, const int* mask, int K,
                      int B, size_t per_b, float* Y, float* sp) {
#define C_(KK) interf_fwd_t<KK>(h, alpha, G, mask, B, per_b, Y, sp)
  SID_DISPATCH_K(K, C_)
#undef C_
}

void interference_bwd(const float* dsp, const float* h, const float* Y, const float* alpha,
                      const float* G, const int* mask, int K, int B, size_t per_b, float* dh,
                      float* dalpha, float* dG) {
#define C_(KK) interf_bwd_t<KK>(dsp, h, Y, alpha, G, mask, B, per_b, dh, dalpha, dG)
  SID_DISPATCH_K(K, C_)
#undef C_
}

// ------------------------------------------------------------- tanh + LN ----
__global__ void k_tanh_ln_fwd(float* t, const float* g, const float* b, int d, float eps,
                              float* xhat, float* rstd, float* out) {
  const size_t r = blockIdx.x;
  float* tr = t + r * d;
  float s = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float v = tanhf(tr[i]);
    tr[i] = v;
    s += v;
  }
  const float mean = block_sum(s) / d;
  float v2 = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float df = tr[i] - mean;
    v2 += df * df;
  }
  const float rs = rsqrtf(block_sum(v2) / d + eps);
  if (threadIdx.x == 0) rstd[r] = rs;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float xh = (tr[i] - mean) * rs;
    xhat[r * d + i] = xh;
    out[r * d + i] = xh * g[i] + b[i];
  }
}

void tanh_ln_fwd(float* t, const float* g, const float* b, int rows, int d, float eps,
                 float* xhat, float* rstd, float* out) {
  k_tanh_ln_fwd<<<rows, 256>>>(t, g, b, d, eps, xhat, rstd, out);
  SK_LAUNCH_CHECK();
}

__global__ void k_tanh_ln_bwd(const float* dout, const float* t, const float* xhat,
                              const float* rstd, const float* g, int d, float* dz) {
  const size_t r = blockIdx.x;
  float m1 = 0.f, m2 = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float dxh = dout[r * d + i] * g[i];
    m1 += dxh;
    m2 += dxh * xhat[r * d + i];
  }
  m1 = block_sum(m1) / d;
  m2 = block_sum(m2) / d;
  const float rs = rstd[r];
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float dxh = dout[r * d + i] * g[i];
    float dt = rs * (dxh - m1 - xhat[r * d + i] * m2);
    float tv = t[r * d + i];
    dz[r * d + i] = dt * (1.f - tv * tv);
  }
}

void tanh_ln_bwd(const float* dout, const float* t, const float* xhat, const float* rstd,
                 const float* g, int rows, int d, float* dz, float* dgamma, float* dbeta) {
  k_tanh_ln_bwd<<<rows, 256>>>(dout, t, xhat, rstd, g, d, dz);
  SK_LAUNCH_CHECK();
  sk::col_dot_add(dout, xhat, rows, d, dgamma);
  sk::col_sum_add(dout, rows, d, dbeta);
}

__global__ void k_mse_grad(const float* y, const float* t, size_t n, float* dy) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) dy[i] = 2.f * (y[i] - t[i]) / (float)n;
}

void mse_grad(const float* y, const float* target, size_t n, float* dy) {
  k_mse_grad<<<blocks_for(n), 256>>>(y, target, n, dy);
  SK_LAUNCH_CHECK();
}

}  // namespace kern
}  // namespace sid
