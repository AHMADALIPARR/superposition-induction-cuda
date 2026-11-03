// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Throughput of the superposition model: forward and AdamW train step, in tokens/s.
//   sid_bench [--batch=8] [--seq=512] [--d_model=768] [--paths=4] [--heads=8]
//             [--head_dim=64] [--iters=20] [--tf32]
#include "../apps/args.hpp"
#include "bench_util.cuh"
#include "common/rng.hpp"
#include "sid/cuda_model.cuh"

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  if (!skbench::need_gpu()) return 1;
  sid::Config c;
  c.d_model = a.i("d_model", c.d_model);
  c.num_paths = a.i("paths", c.num_paths);
  c.num_heads = a.i("heads", c.num_heads);
  c.head_dim = a.i("head_dim", c.head_dim);
  const int B = a.i("batch", 8), S = a.i("seq", 512), iters = a.i("iters", 20);
  auto t = sid::make_param_table(c);
  sid::CudaModel m(c, sid::init_params(c, t, 1));
  if (a.has("tf32")) m.set_tf32(true);
  sk::Rng r(2);
  std::vector<float> x((size_t)B * S * c.d_model), y(x.size());
  for (auto& v : x) v = (float)r.normal();
  for (auto& v : y) v = (float)r.normal();
  sk::DBuf<float> dx, dt;
  dx.upload(x);
  dt.upload(y);
  std::printf("config: d_model %d, paths %d, heads %d x %d, batch %d, seq %d, %zu params%s\n",
              c.d_model, c.num_paths, c.num_heads, c.head_dim, B, S, t.total,
              a.has("tf32") ? ", TF32" : "");
  double f = skbench::time_ms([&] { m.forward(dx.p, B, S, false); }, 3, iters);
  std::printf("forward:    %8.3f ms  %12.0f tokens/s  (%d refinement iterations)\n", f,
              B * S / (f * 1e-3), m.iterations());
  sk::AdamWConfig oc;
  double s = skbench::time_ms([&] { m.train_step_adamw(dx.p, dt.p, B, S, oc); }, 2, iters);
  std::printf("train step: %8.3f ms  %12.0f tokens/s  (forward + backward + AdamW)\n", s,
              B * S / (s * 1e-3));
  return 0;
}
