// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "alexnet_mmep/kernels.cuh"
#include "alexnet_mmep/model.hpp"
#include "common/cuda_util.cuh"
#include "common/device_reduce.cuh"

#include <cfloat>

namespace amm {
namespace kern {

using sk::blocks_for;

__global__ void k_im2col(const float* __restrict__ x, int nb, int C, int H, int k, int s, int p,
                         int Ho, float* __restrict__ col) {
  const long long HW = (long long)Ho * Ho, W = nb * HW;
  const long long total = (long long)C * k * k * W;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const long long colj = i % W;
    const long long row = i / W;
    const int kx = row % k, ky = (row / k) % k, c = row / (k * k);
    const int nl = colj / HW, q = colj % HW, oy = q / Ho, ox = q % Ho;
    const int iy = oy * s - p + ky, ix = ox * s - p + kx;
    float v = 0.f;
    if (iy >= 0 && iy < H && ix >= 0 && ix < H) v = x[(((long long)nl * C + c) * H + iy) * H + ix];
    col[i] = v;
  }
}

void im2col_wide(const float* x, int nb, int C, int H, int k, int s, int p, int Ho, float* col) {
  const size_t total = (size_t)C * k * k * nb * Ho * Ho;
  k_im2col<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(x, nb, C, H, k, s, p, Ho, col);
  SK_LAUNCH_CHECK();
}

__global__ void k_col2im(const float* __restrict__ dcol, int nb, int C, int H, int k, int s, int p,
                         int Ho, float* __restrict__ dx) {
  const long long HW = (long long)Ho * Ho, W = nb * HW;
  const long long total = (long long)nb * C * H * H;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const int ix = i % H, iy = (i / H) % H, c = (i / ((long long)H * H)) % C;
    const int nl = i / ((long long)C * H * H);
    float acc = 0.f;
    for (int ky = 0; ky < k; ++ky) {
      const int ty = iy + p - ky;
      if (ty < 0 || ty % s) continue;
      const int oy = ty / s;
      if (oy >= Ho) continue;
      for (int kx = 0; kx < k; ++kx) {
        const int tx = ix + p - kx;
        if (tx < 0 || tx % s) continue;
        const int ox = tx / s;
        if (ox >= Ho) continue;
        acc += dcol[((long long)(c * k + ky) * k + kx) * W + nl * HW + oy * Ho + ox];
      }
    }
    dx[i] = acc;
  }
}

void col2im_wide(const float* dcol, int nb, int C, int H, int k, int s, int p, int Ho, float* dx) {
  const size_t total = (size_t)nb * C * H * H;
  k_col2im<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(dcol, nb, C, H, k, s, p, Ho, dx);
  SK_LAUNCH_CHECK();
}

__global__ void k_unwide_bias(const float* __restrict__ Zw, const float* __restrict__ b, int nb,
                              int O, int HW, float* __restrict__ Z) {
  const long long total = (long long)nb * O * HW;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const int q = i % HW, o = (i / HW) % O, nl = i / ((long long)O * HW);
    Z[i] = Zw[(long long)o * nb * HW + (long long)nl * HW + q] + b[o];
  }
}

void unwide_bias(const float* Zw, const float* b, int nb, int O, int HW, float* Z) {
  const size_t total = (size_t)nb * O * HW;
  k_unwide_bias<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(Zw, b, nb, O, HW, Z);
  SK_LAUNCH_CHECK();
}

__global__ void k_to_wide(const float* __restrict__ D, int nb, int O, int HW, float* __restrict__ Dw) {
  const long long total = (long long)nb * O * HW;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const int q = i % HW, o = (i / HW) % O, nl = i / ((long long)O * HW);
    Dw[(long long)o * nb * HW + (long long)nl * HW + q] = D[i];
  }
}

void to_wide(const float* D, int nb, int O, int HW, float* Dw) {
  const size_t total = (size_t)nb * O * HW;
  k_to_wide<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(D, nb, O, HW, Dw);
  SK_LAUNCH_CHECK();
}

// ------------------------------------------------------------ SmoothLeaky --
__global__ void k_sl_fwd(const float* __restrict__ z, float* __restrict__ a, size_t n, float al,
                         float be) {
  const float ln2 = 0.693147180559945309f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float x = z[i], bx = be * x;
    const float sp = bx > 20.f ? bx : log1pf(expf(bx));
    a[i] = al * x + ((1.f - al) / be) * (sp - ln2);
  }
}
__global__ void k_sl_bwd(const float* __restrict__ z, float* __restrict__ d, size_t n, float al,
                         float be) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float bx = be * z[i];
    float s;
    if (bx > 20.f) {
      s = 1.f;
    } else {
      const float e = expf(bx);
      s = e / (e + 1.f);
    }
    d[i] *= al + ((1.f - al) / be) * s * be;
  }
}
void smooth_leaky_fwd(const float* z, float* a, size_t n, float alpha, float beta) {
  k_sl_fwd<<<std::min<size_t>(blocks_for(n), 65535u * 8), 256>>>(z, a, n, alpha, beta);
  SK_LAUNCH_CHECK();
}
void smooth_leaky_bwd(const float* z, float* d, size_t n, float alpha, float beta) {
  k_sl_bwd<<<std::min<size_t>(blocks_for(n), 65535u * 8), 256>>>(z, d, n, alpha, beta);
  SK_LAUNCH_CHECK();
}

// ---------------------------------------------------------------- maxpool --
__global__ void k_pool_fwd(const float* __restrict__ x, int NC, int H, int Ho,
                           float* __restrict__ y, int* __restrict__ am) {
  const long long total = (long long)NC * Ho * Ho;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const int ox = i % Ho, oy = (i / Ho) % Ho;
    const long long q = i / ((long long)Ho * Ho);
    const float* plane = x + q * H * H;
    int best = (oy * kPoolS) * H + ox * kPoolS;
    float bv = plane[best];
    for (int ky = 0; ky < kPoolK; ++ky)
      for (int kx = 0; kx < kPoolK; ++kx) {
        const int idx = (oy * kPoolS + ky) * H + ox * kPoolS + kx;
        const float v = plane[idx];
        if (v > bv || isnan(v)) { bv = v; best = idx; }
      }
    y[i] = bv;
    am[i] = best;
  }
}
__global__ void k_pool_bwd(const float* __restrict__ dy, const int* __restrict__ am, int NC, int H,
                           int Ho, float* __restrict__ dx) {
  const long long total = (long long)NC * Ho * Ho;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const long long q = i / ((long long)Ho * Ho);
    atomicAdd(&dx[q * H * H + am[i]], dy[i]);
  }
}
void maxpool_fwd(const float* x, int NC, int H, int Ho, float* y, int* am) {
  const size_t total = (size_t)NC * Ho * Ho;
  k_pool_fwd<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(x, NC, H, Ho, y, am);
  SK_LAUNCH_CHECK();
}
void maxpool_bwd(const float* dy, const int* am, int NC, int H, int Ho, float* dx) {
  SK_CUDA(cudaMemset(dx, 0, (size_t)NC * H * H * sizeof(float)));
  const size_t total = (size_t)NC * Ho * Ho;
  k_pool_bwd<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(dy, am, NC, H, Ho, dx);
  SK_LAUNCH_CHECK();
}

// ---------------------------------------------------- constrained softmax --
__device__ float row_softmax_entropy(const float* l, int K, float tau, bool divide, float* out) {
  float m = -FLT_MAX;
  for (int k = 0; k < K; ++k) m = fmaxf(m, divide ? l[k] / tau : l[k]);
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    const float e = expf((divide ? l[k] / tau : l[k]) - m);
    out[k] = e;
    s += e;
  }
  float h = 0.f;
  for (int k = 0; k < K; ++k) {
    out[k] /= s;
    h += out[k] * logf(out[k] + 1e-10f);
  }
  return -h;
}

__global__ void k_csoftmax(const float* __restrict__ logits, int rows, int K, float bound,
                           float* __restrict__ alpha, float* __restrict__ tau, int* __restrict__ viol,
                           float* __restrict__ ent, int* __restrict__ iters, float* __restrict__ lo,
                           float* __restrict__ hi, const int* __restrict__ viol_fz,
                           const float* __restrict__ tau_fz) {
  __shared__ float smax[32];
  float p[kMaxRoutes];
  if (viol_fz) {
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
      const int v = viol_fz[r];
      const float t = v ? tau_fz[r] : 1.f;
      ent[r] = row_softmax_entropy(logits + (size_t)r * K, K, t, v != 0, alpha + (size_t)r * K);
      viol[r] = v;
      tau[r] = t;
    }
    if (threadIdx.x == 0) *iters = 0;
    return;
  }
  int any = 0;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const float h = row_softmax_entropy(logits + (size_t)r * K, K, 1.f, false, alpha + (size_t)r * K);
    ent[r] = h;
    viol[r] = h > bound;
    tau[r] = 1.f;
    lo[r] = 0.01f;
    hi[r] = 1.f;
    any |= viol[r];
  }
  any = __syncthreads_or(any);
  if (!any) {
    if (threadIdx.x == 0) *iters = 0;
    return;
  }
  int it = 0;
  for (int i = 0; i < 32; ++i) {
    ++it;
    float w = 0.f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
      const float mid = (lo[r] + hi[r]) / 2.f;
      const float h = row_softmax_entropy(logits + (size_t)r * K, K, mid, true, p);
      if (h > bound) hi[r] = mid; else lo[r] = mid;
      w = fmaxf(w, hi[r] - lo[r]);
    }
    w = sk::dwarp_max(w);
    if ((threadIdx.x & 31) == 0) smax[threadIdx.x >> 5] = w;
    __syncthreads();
    float bw = 0.f;
    for (int j = 0; j < (int)((blockDim.x + 31) >> 5); ++j) bw = fmaxf(bw, smax[j]);
    __syncthreads();
    if (bw < 1e-6f) break;
  }
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    if (!viol[r]) continue;
    const float t = (lo[r] + hi[r]) / 2.f;
    tau[r] = t;
    ent[r] = row_softmax_entropy(logits + (size_t)r * K, K, t, true, alpha + (size_t)r * K);
  }
  if (threadIdx.x == 0) *iters = it;
}

void constrained_softmax(const float* logits, int rows, int K, float bound, float* alpha,
                         float* tau, int* viol, float* ent, int* iters, float* scratch,
                         const int* viol_fz, const float* tau_fz) {
  if (K > kMaxRoutes) throw std::runtime_error("constrained_softmax: K > kMaxRoutes");
  k_csoftmax<<<1, 256>>>(logits, rows, K, bound, alpha, tau, viol, ent, iters, scratch,
                         scratch + rows, viol_fz, tau_fz);
  SK_LAUNCH_CHECK();
}

__global__ void k_csoftmax_bwd(const float* __restrict__ a, const float* __restrict__ da,
                               const float* __restrict__ tau, int rows, int K, float* __restrict__ dl) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float dot = 0.f;
  for (int k = 0; k < K; ++k) dot += a[(size_t)r * K + k] * da[(size_t)r * K + k];
  const float inv = 1.f / tau[r];
  for (int k = 0; k < K; ++k)
    dl[(size_t)r * K + k] = a[(size_t)r * K + k] * (da[(size_t)r * K + k] - dot) * inv;
}
void constrained_softmax_bwd(const float* alpha, const float* dalpha, const float* tau, int rows,
                             int K, float* dl) {
  k_csoftmax_bwd<<<blocks_for(rows, 128), 128>>>(alpha, dalpha, tau, rows, K, dl);
  SK_LAUNCH_CHECK();
}

// ----------------------------------------------------------------- router --
__global__ void k_router_combine(const float* __restrict__ alpha, const float* __restrict__ tz,
                                 int N, int d, int K, float* __restrict__ out) {
  const long long total = (long long)N * d;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    const int n = i / d;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += alpha[(size_t)n * K + k] * tz[(long long)k * total + i];
    out[i] = acc;
  }
}
void router_combine(const float* alpha, const float* tz, int N, int d, int K, float* out) {
  const size_t total = (size_t)N * d;
  k_router_combine<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(alpha, tz, N, d, K, out);
  SK_LAUNCH_CHECK();
}

__global__ void k_router_dalpha(const float* __restrict__ dout, const float* __restrict__ tz,
                                const float* __restrict__ alpha, const float* __restrict__ dent,
                                int N, int d, int K, float* __restrict__ dalpha) {
  const int n = blockIdx.x, k = blockIdx.y;
  const float* g = dout + (size_t)n * d;
  const float* t = tz + ((size_t)k * N + n) * d;
  float s = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) s += g[i] * t[i];
  s = sk::block_sum(s);
  if (threadIdx.x == 0) {
    const float a = alpha[(size_t)n * K + k];
    dalpha[(size_t)n * K + k] = s + dent[n] * -(logf(a + 1e-10f) + a / (a + 1e-10f));
  }
}
void router_dalpha(const float* dout, const float* tz, const float* alpha, const float* dent,
                   int N, int d, int K, float* dalpha) {
  k_router_dalpha<<<dim3(N, K), 256>>>(dout, tz, alpha, dent, N, d, K, dalpha);
  SK_LAUNCH_CHECK();
}

__global__ void k_scale_rows(const float* __restrict__ dout, const float* __restrict__ alpha, int k,
                             int K, int N, int d, float* __restrict__ dtz) {
  const long long total = (long long)N * d;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x)
    dtz[i] = alpha[(i / d) * K + k] * dout[i];
}
void scale_rows(const float* dout, const float* alpha, int k, int K, int N, int d, float* dtz) {
  const size_t total = (size_t)N * d;
  k_scale_rows<<<std::min<size_t>(blocks_for(total), 65535u * 8), 256>>>(dout, alpha, k, K, N, d, dtz);
  SK_LAUNCH_CHECK();
}

// ---------------------------------------------------------- cross entropy --
__global__ void k_ce(const float* __restrict__ logits, const int* __restrict__ labels, int N, int C,
                     float* __restrict__ loss_rows, float* __restrict__ dl, int* __restrict__ pred) {
  const int n = blockIdx.x;
  const float* l = logits + (size_t)n * C;
  __shared__ float sv[32];
  __shared__ int si[32];
  // argmax (first) and max
  float m = -FLT_MAX;
  int mi = 0x7fffffff;
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    if (l[j] > m) { m = l[j]; mi = j; }
  for (int o = 16; o > 0; o >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, m, o);
    const int oi = __shfl_xor_sync(0xffffffffu, mi, o);
    if (om > m || (om == m && oi < mi)) { m = om; mi = oi; }
  }
  if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = m; si[threadIdx.x >> 5] = mi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
      if (sv[w] > sv[0] || (sv[w] == sv[0] && si[w] < si[0])) { sv[0] = sv[w]; si[0] = si[w]; }
  }
  __syncthreads();
  m = sv[0];
  float s = 0.f;
  for (int j = threadIdx.x; j < C; j += blockDim.x) s += expf(l[j] - m);
  s = sk::block_sum(s);
  const float lse = m + logf(s);
  const int y = labels[n];
  for (int j = threadIdx.x; j < C; j += blockDim.x)
    dl[(size_t)n * C + j] = (expf(l[j] - lse) - (j == y ? 1.f : 0.f)) / (float)N;
  if (threadIdx.x == 0) {
    loss_rows[n] = lse - l[y];
    pred[n] = si[0];
  }
}
void cross_entropy(const float* logits, const int* labels, int N, int C, float* loss_rows,
                   float* dlogits, int* pred) {
  k_ce<<<N, 256>>>(logits, labels, N, C, loss_rows, dlogits, pred);
  SK_LAUNCH_CHECK();
}

__global__ void k_row_sum_add(const float* __restrict__ X, int M, int N, float* __restrict__ out) {
  const int i = blockIdx.x;
  float s = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) s += X[(size_t)i * N + j];
  s = sk::block_sum(s);
  if (threadIdx.x == 0) out[i] += s;
}
void row_sum_add(const float* X, int M, int N, float* out) {
  k_row_sum_add<<<M, 256>>>(X, M, N, out);
  SK_LAUNCH_CHECK();
}

}  // namespace kern
}  // namespace amm
