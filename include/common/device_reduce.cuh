// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Warp / block reductions used inside kernels.
#pragma once
#include <cuda_runtime.h>

namespace sk {

__device__ __forceinline__ float dwarp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ double dwarp_sum(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float dwarp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over the whole block; every thread gets the result. blockDim.x must be a
// multiple of 32 and <= 1024. Safe to call repeatedly.
template <class T>
__device__ T block_sum(T v) {
  __shared__ T sh[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  v = dwarp_sum(v);
  __syncthreads();
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  T r = (threadIdx.x < nw) ? sh[threadIdx.x] : T(0);
  if (wid == 0) {
    r = dwarp_sum(r);
    if (lane == 0) sh[0] = r;
  }
  __syncthreads();
  return sh[0];
}

}  // namespace sk
