// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Load weights and run the superposition model on a random batch.
//   sid_run weights.skt [--batch=2] [--seq=16] [--seed=0]
#include <cmath>
#include <cstdio>

#include "args.hpp"
#include "common/rng.hpp"
#include "sid/cuda_model.cuh"

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  std::string w = a.str("_pos0", "");
  if (w.empty()) {
    std::fprintf(stderr, "usage: sid_run weights.skt [--batch=2] [--seq=16]\n");
    return 2;
  }
  if (sk::cuda_device_count() == 0) {
    std::fprintf(stderr, "no CUDA device found\n");
    return 1;
  }
  sid::Config c;
  sk::ParamTable t;
  auto p = sid::load_weights(w, c, t);
  const int B = a.i("batch", 2), S = a.i("seq", 16);
  sid::CudaModel m(c, p);
  sk::Rng r((uint64_t)a.i("seed", 0));
  std::vector<float> x((size_t)B * S * c.d_model);
  for (auto& v : x) v = (float)r.normal();
  sk::DBuf<float> dx;
  dx.upload(x);
  m.forward(dx.p, B, S, false);
  std::vector<float> y(x.size());
  SK_CUDA(cudaMemcpy(y.data(), m.device_y(), y.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double s2 = 0;
  for (float v : y) s2 += (double)v * v;
  int on = 0;
  for (int v : m.mask()) on += v;
  std::printf("y: [%d, %d, %d], rms %.6f\n", B, S, c.d_model, std::sqrt(s2 / y.size()));
  std::printf("refinement iterations %d (final delta %.3g), interfering pairs %d, orth loss %.6f\n",
              m.iterations(), m.final_delta(), on, m.orth_loss());
  return 0;
}
