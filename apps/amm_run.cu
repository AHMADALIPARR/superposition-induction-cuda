// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Load AlexNet-MMEP weights, run a random batch and print the predictions.
//   amm_run weights.skt [--batch=4] [--seed=0]
#include <cstdio>

#include "alexnet_mmep/cuda_model.cuh"
#include "args.hpp"
#include "common/rng.hpp"

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  std::string w = a.str("_pos0", "");
  if (w.empty()) {
    std::fprintf(stderr, "usage: amm_run weights.skt [--batch=4]\n");
    return 2;
  }
  if (sk::cuda_device_count() == 0) {
    std::fprintf(stderr, "no CUDA device found\n");
    return 1;
  }
  amm::Config c;
  sk::ParamTable t;
  auto p = amm::load_weights(w, c, t);
  const int N = a.i("batch", 4);
  amm::CudaModel m(c, p);
  sk::Rng r((uint64_t)a.i("seed", 0));
  std::vector<float> x((size_t)N * c.in_ch * c.image * c.image);
  for (auto& v : x) v = (float)r.normal();
  sk::DBuf<float> dx;
  dx.upload(x);
  m.forward(dx.p, N);
  std::vector<float> lg((size_t)N * c.num_classes);
  SK_CUDA(cudaMemcpy(lg.data(), m.device_logits(), lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int n = 0; n < N; ++n) {
    int best = 0;
    for (int k = 1; k < c.num_classes; ++k)
      if (lg[(size_t)n * c.num_classes + k] > lg[(size_t)n * c.num_classes + best]) best = k;
    std::printf("image %d: class %d (logit %.4f)\n", n, best, lg[(size_t)n * c.num_classes + best]);
  }
  std::printf("entropy_fc6 %.5f, entropy_fc7 %.5f (bound %.3f), bisection iterations %d/%d\n",
              m.ent6(), m.ent7(), c.entropy_bound, m.bisect_iters(6), m.bisect_iters(7));
  return 0;
}
