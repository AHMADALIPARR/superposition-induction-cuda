// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include <cfloat>
#include <cmath>

#include "common/cuda_util.cuh"

namespace sk {

int cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

static cublasOperation_t op(bool t) { return t ? CUBLAS_OP_T : CUBLAS_OP_N; }

void gemm_rm(cublasHandle_t h, bool transA, bool transB, int M, int N, int K, float alpha,
             const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
  if (M == 0 || N == 0) return;
  SK_CUBLAS(cublasSgemm(h, op(transB), op(transA), N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

void gemm_rm_batched(cublasHandle_t h, bool transA, bool transB, int M, int N, int K,
                     float alpha, const float* A, int lda, long long strideA, const float* B,
                     int ldb, long long strideB, float beta, float* C, int ldc,
                     long long strideC, int batch) {
  if (M == 0 || N == 0 || batch == 0) return;
  SK_CUBLAS(cublasSgemmStridedBatched(h, op(transB), op(transA), N, M, K, &alpha, B, ldb,
                                      strideB, A, lda, strideA, &beta, C, ldc, strideC, batch));
}

__global__ void k_add_bias_rows(float* Y, const float* b, long long total, int N) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < total) Y[i] += b[i % N];
}

void add_bias_rows(float* Y, const float* b, int M, int N) {
  long long total = (long long)M * N;
  if (!total) return;
  k_add_bias_rows<<<blocks_for(total), 256>>>(Y, b, total, N);
  SK_LAUNCH_CHECK();
}

void linear_fwd(cublasHandle_t h, const float* X, const float* W, const float* b, float* Y,
                int M, int K, int N) {
  gemm_rm(h, false, true, M, N, K, 1.f, X, K, W, K, 0.f, Y, N);
  if (b) add_bias_rows(Y, b, M, N);
}

void linear_bwd(cublasHandle_t h, const float* X, const float* W, const float* dY, float* dW,
                float* db, float* dX, int M, int K, int N, bool accumulate_dx) {
  // dW[N,K] += dY[M,N]^T X[M,K]
  if (dW) gemm_rm(h, true, false, N, K, M, 1.f, dY, N, X, K, 1.f, dW, K);
  if (db) col_sum_add(dY, M, N, db);
  // dX[M,K] = dY[M,N] W[N,K]
  if (dX) gemm_rm(h, false, false, M, K, N, 1.f, dY, N, W, K, accumulate_dx ? 1.f : 0.f, dX, K);
}

// Column reductions: grid.x over column tiles, grid.y over row chunks, atomics into out.
template <bool DOT>
__global__ void k_col_reduce(const float* X, const float* Y, int M, int N, int rows_per_chunk,
                             float* out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  int r0 = blockIdx.y * rows_per_chunk;
  int r1 = min(M, r0 + rows_per_chunk);
  float acc = 0.f;
  for (int r = r0; r < r1; ++r) {
    size_t idx = (size_t)r * N + j;
    acc += DOT ? X[idx] * Y[idx] : X[idx];
  }
  atomicAdd(&out[j], acc);
}

template <bool DOT>
static void col_reduce(const float* X, const float* Y, int M, int N, float* out) {
  if (!M || !N) return;
  int chunks = (M + 255) / 256;
  if (chunks > 1024) chunks = 1024;
  int rpc = (M + chunks - 1) / chunks;
  chunks = (M + rpc - 1) / rpc;
  dim3 grid((N + 127) / 128, chunks);
  k_col_reduce<DOT><<<grid, 128>>>(X, Y, M, N, rpc, out);
  SK_LAUNCH_CHECK();
}

void col_sum_add(const float* X, int M, int N, float* out) { col_reduce<false>(X, nullptr, M, N, out); }
void col_dot_add(const float* X, const float* Y, int M, int N, float* out) {
  col_reduce<true>(X, Y, M, N, out);
}

struct Dims5 { int d[5]; int perm[5]; };

__global__ void k_permute5(const float* in, float* out, Dims5 p, long long total) {
  long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (o >= total) return;
  // input strides
  long long istr[5];
  istr[4] = 1;
  for (int i = 3; i >= 0; --i) istr[i] = istr[i + 1] * p.d[i + 1];
  long long rem = o, iidx = 0;
  for (int i = 4; i >= 0; --i) {
    int od = p.d[p.perm[i]];
    long long c = rem % od;
    rem /= od;
    iidx += c * istr[p.perm[i]];
  }
  out[o] = in[iidx];
}

void permute5(const float* in, float* out, const int d[5], const int perm[5]) {
  Dims5 p;
  long long total = 1;
  for (int i = 0; i < 5; ++i) { p.d[i] = d[i]; p.perm[i] = perm[i]; total *= d[i]; }
  if (!total) return;
  k_permute5<<<blocks_for(total), 256>>>(in, out, p, total);
  SK_LAUNCH_CHECK();
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// One warp per row.
__global__ void k_softmax_fwd(const float* in, float* out, long long rows, int L) {
  long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * L;
  float* y = out + row * L;
  float m = -FLT_MAX;
  for (int i = lane; i < L; i += 32) m = fmaxf(m, x[i]);
  m = warp_max(m);
  float s = 0.f;
  for (int i = lane; i < L; i += 32) s += expf(x[i] - m);
  s = warp_sum(s);
  float inv = 1.f / s;
  for (int i = lane; i < L; i += 32) y[i] = expf(x[i] - m) * inv;
}

__global__ void k_softmax_bwd(const float* P, const float* dOut, float* dIn, long long rows, int L) {
  long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* p = P + row * L;
  const float* g = dOut + row * L;
  float* d = dIn + row * L;
  float dot = 0.f;
  for (int i = lane; i < L; i += 32) dot += p[i] * g[i];
  dot = warp_sum(dot);
  for (int i = lane; i < L; i += 32) d[i] = p[i] * (g[i] - dot);
}

void softmax_rows_fwd(const float* in, float* out, long long rows, int L) {
  if (!rows) return;
  k_softmax_fwd<<<blocks_for(rows * 32), 256>>>(in, out, rows, L);
  SK_LAUNCH_CHECK();
}
void softmax_rows_bwd(const float* P, const float* dOut, float* dIn, long long rows, int L) {
  if (!rows) return;
  k_softmax_bwd<<<blocks_for(rows * 32), 256>>>(P, dOut, dIn, rows, L);
  SK_LAUNCH_CHECK();
}

__global__ void k_sq_norms(const float* a, const float* b, size_t n, double* out) {
  __shared__ double s0[256], s1[256];
  double x0 = 0, x1 = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    double av = a[i];
    double bv = b ? b[i] : 0.0;
    x0 += (av - bv) * (av - bv);
    x1 += bv * bv;
  }
  s0[threadIdx.x] = x0;
  s1[threadIdx.x] = x1;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) { s0[threadIdx.x] += s0[threadIdx.x + s]; s1[threadIdx.x] += s1[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { atomicAdd(&out[0], s0[0]); atomicAdd(&out[1], s1[0]); }
}

void sq_diff_norms(const float* a, const float* b, size_t n, double* out2_device) {
  SK_CUDA(cudaMemset(out2_device, 0, 2 * sizeof(double)));
  unsigned g = blocks_for(n);
  if (g > 1024) g = 1024;
  if (g == 0) g = 1;
  k_sq_norms<<<g, 256>>>(a, b, n, out2_device);
  SK_LAUNCH_CHECK();
}

__global__ void k_adamw(float lr, float b1, float b2, float eps, float wd, float step_size,
                        float bc2_sqrt, float* p, const float* g, float* m, float* v,
                        const unsigned char* mask, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n || (mask && !mask[i])) return;
  float pi = p[i] * (1.f - lr * wd);
  float gi = g[i];
  float mi = b1 * m[i] + (1.f - b1) * gi;
  float vi = b2 * v[i] + (1.f - b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  float denom = sqrtf(vi) / bc2_sqrt + eps;
  p[i] = pi - step_size * mi / denom;
}

void adamw_step_device(const AdamWConfig& c, int step, float* p, const float* g, float* m,
                       float* v, const unsigned char* mask, size_t n) {
  float bc1 = 1.f - std::pow(c.beta1, (float)step);
  float bc2 = 1.f - std::pow(c.beta2, (float)step);
  k_adamw<<<blocks_for(n), 256>>>(c.lr, c.beta1, c.beta2, c.eps, c.weight_decay, c.lr / bc1,
                                  std::sqrt(bc2), p, g, m, v, mask, n);
  SK_LAUNCH_CHECK();
}

__global__ void k_sgd(float lr, float mom, float wd, int first, float* p, const float* g,
                      float* buf, const unsigned char* mask, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n || (mask && !mask[i])) return;
  float d = g[i] + wd * p[i];
  if (mom != 0.f) {
    float b = first ? d : mom * buf[i] + d;
    buf[i] = b;
    d = b;
  }
  p[i] -= lr * d;
}

void sgd_step_device(const SgdConfig& c, int step, float* p, const float* g, float* buf,
                     const unsigned char* mask, size_t n) {
  k_sgd<<<blocks_for(n), 256>>>(c.lr, c.momentum, c.weight_decay, step == 1 ? 1 : 0, p, g, buf,
                                mask, n);
  SK_LAUNCH_CHECK();
}

}  // namespace sk
