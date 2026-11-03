// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CUDA / cuBLAS helpers: error checks, a device buffer, row-major GEMM wrappers
// and the small generic kernels both models share.
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/optim.hpp"

#define SK_CUDA(x)                                                                  \
  do {                                                                              \
    cudaError_t e_ = (x);                                                           \
    if (e_ != cudaSuccess)                                                          \
      throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e_) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));  \
  } while (0)

#define SK_CUBLAS(x)                                                                 \
  do {                                                                               \
    cublasStatus_t s_ = (x);                                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                 \
      throw std::runtime_error(std::string("cuBLAS error ") + std::to_string((int)s_) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));   \
  } while (0)

#define SK_LAUNCH_CHECK() SK_CUDA(cudaGetLastError())

namespace sk {

// Owning device buffer of T.
template <class T>
struct DBuf {
  T* p = nullptr;
  size_t n = 0;
  DBuf() = default;
  explicit DBuf(size_t count) { resize(count); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  DBuf(DBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  DBuf& operator=(DBuf&& o) noexcept {
    if (this != &o) { release(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
    return *this;
  }
  ~DBuf() { release(); }
  void release() {
    if (p) cudaFree(p);
    p = nullptr;
    n = 0;
  }
  void resize(size_t count) {
    if (count == n) return;
    release();
    if (count) SK_CUDA(cudaMalloc(&p, count * sizeof(T)));
    n = count;
  }
  void zero() {
    if (n) SK_CUDA(cudaMemset(p, 0, n * sizeof(T)));
  }
  void upload(const T* h, size_t count) {
    resize(count);
    if (count) SK_CUDA(cudaMemcpy(p, h, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void upload(const std::vector<T>& h) { upload(h.data(), h.size()); }
  std::vector<T> download() const {
    std::vector<T> h(n);
    if (n) SK_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// Returns the number of CUDA devices, 0 when none (or no driver).
int cuda_device_count();

// ---------------------------------------------------------------------------
// Row-major GEMM on top of cuBLAS (column-major) by computing C^T = B^T A^T.
//   C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C
// lda/ldb/ldc are the row strides of the matrices as stored.
// ---------------------------------------------------------------------------
void gemm_rm(cublasHandle_t h, bool transA, bool transB, int M, int N, int K, float alpha,
             const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

void gemm_rm_batched(cublasHandle_t h, bool transA, bool transB, int M, int N, int K,
                     float alpha, const float* A, int lda, long long strideA, const float* B,
                     int ldb, long long strideB, float beta, float* C, int ldc,
                     long long strideC, int batch);

// nn.Linear: Y[M,N] = X[M,K] W[N,K]^T + b[N]
void linear_fwd(cublasHandle_t h, const float* X, const float* W, const float* b, float* Y,
                int M, int K, int N);
// Accumulates dW += dY^T X, db += colsum(dY); dX = dY W (+ dX if accumulate_dx).
// dX may be null.
void linear_bwd(cublasHandle_t h, const float* X, const float* W, const float* dY, float* dW,
                float* db, float* dX, int M, int K, int N, bool accumulate_dx);

// out[j] += sum_i X[i*N + j]
void col_sum_add(const float* X, int M, int N, float* out);
// out[j] += sum_i X[i*N + j] * Y[i*N + j]
void col_dot_add(const float* X, const float* Y, int M, int N, float* out);
// Y[i, j] += b[j]
void add_bias_rows(float* Y, const float* b, int M, int N);

// Generic 5-D permute. in has dims d[0..4]; out dim i is in dim perm[i].
void permute5(const float* in, float* out, const int d[5], const int perm[5]);

// Row softmax over contiguous rows of length L (rows*L elements).
void softmax_rows_fwd(const float* in, float* out, long long rows, int L);
// dIn = P * (dOut - sum(dOut * P)); dIn may alias dOut.
void softmax_rows_bwd(const float* P, const float* dOut, float* dIn, long long rows, int L);

// sum((a-b)^2) and sum(b^2) in double (b may be null for just sum(a^2)).
void sq_diff_norms(const float* a, const float* b, size_t n, double* out2_device);

// Optimizer updates on device (same math as common/optim.hpp).
void adamw_step_device(const AdamWConfig& c, int step, float* p, const float* g, float* m,
                       float* v, const unsigned char* mask, size_t n);
void sgd_step_device(const SgdConfig& c, int step, float* p, const float* g, float* buf,
                     const unsigned char* mask, size_t n);

inline unsigned blocks_for(size_t n, unsigned t = 256) {
  return static_cast<unsigned>((n + t - 1) / t);
}

}  // namespace sk
