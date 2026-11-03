// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Throughput of AlexNet-MMEP: inference and trainer step, in images/s.
//   amm_bench [--batch=64] [--image=224] [--routes=4] [--iters=10] [--tf32]
#include "../apps/args.hpp"
#include "alexnet_mmep/cuda_model.cuh"
#include "bench_util.cuh"
#include "common/rng.hpp"

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  if (!skbench::need_gpu()) return 1;
  amm::Config c;
  c.image = a.i("image", c.image);
  c.num_routes = a.i("routes", c.num_routes);
  const int N = a.i("batch", 64), iters = a.i("iters", 10);
  auto t = amm::make_param_table(c);
  amm::CudaModel m(c, amm::init_params(c, t, 1));
  if (a.has("tf32")) m.set_tf32(true);
  sk::Rng r(2);
  std::vector<float> x((size_t)N * c.in_ch * c.image * c.image);
  for (auto& v : x) v = (float)r.normal();
  std::vector<int> lab(N);
  for (auto& v : lab) v = (int)(r.uniform() * c.num_classes) % c.num_classes;
  sk::DBuf<float> dx;
  sk::DBuf<int> dl;
  dx.upload(x);
  dl.upload(lab);
  std::printf("config: %dx%d, batch %d, routes %d, fc6_in %d, %zu params%s\n", c.image, c.image, N,
              c.num_routes, amm::shapes(c).fc6_in, t.total, a.has("tf32") ? ", TF32" : "");
  double f = skbench::time_ms([&] { m.forward(dx.p, N); }, 2, iters);
  std::printf("inference:  %8.3f ms  %10.1f images/s\n", f, N / (f * 1e-3));
  sk::SgdConfig sc;
  sc.lr = 0.01f; sc.momentum = 0.9f; sc.weight_decay = 5e-4f;
  double s = skbench::time_ms([&] { m.train_step_sgd(dx.p, dl.p, N, sc); }, 1, iters);
  std::printf("train step: %8.3f ms  %10.1f images/s  (forward + CE/penalty + backward + SGD)\n", s,
              N / (s * 1e-3));
  return 0;
}
