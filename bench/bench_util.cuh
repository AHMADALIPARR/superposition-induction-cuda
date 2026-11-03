// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#pragma once
#include <cstdio>
#include <cstdlib>

#include "common/cuda_util.cuh"

namespace skbench {

inline bool need_gpu() {
  if (sk::cuda_device_count() > 0) {
    cudaDeviceProp pr{};
    SK_CUDA(cudaGetDeviceProperties(&pr, 0));
    std::printf("device: %s (sm_%d%d)\n", pr.name, pr.major, pr.minor);
    return true;
  }
  std::printf("no CUDA device found\n");
  return false;
}

// Mean milliseconds per call of f over iters calls, after warmup calls.
template <class F>
double time_ms(F&& f, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) f();
  SK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  SK_CUDA(cudaEventCreate(&a));
  SK_CUDA(cudaEventCreate(&b));
  SK_CUDA(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) f();
  SK_CUDA(cudaEventRecord(b));
  SK_CUDA(cudaEventSynchronize(b));
  float ms = 0;
  SK_CUDA(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}

}  // namespace skbench
