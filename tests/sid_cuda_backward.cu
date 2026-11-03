// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CUDA backward vs the CPU backward, and the CUDA train steps vs the CPU ones.
#include <cstdio>

#include "sid_cuda_common.cuh"

using namespace sid;

static bool run(const char* name, Config c, int B, int S, double noise, uint64_t seed) {
  std::printf("[%s] d=%d K=%d H=%d dh=%d B=%d S=%d\n", name, c.d_model, c.num_paths, c.num_heads,
              c.head_dim, B, S);
  auto tab = make_param_table(c);
  auto p = sidtest::make_params(c, tab, seed, noise);
  sk::Rng r(seed + 1);
  const size_t n = (size_t)B * S * c.d_model;
  auto x = sktest::randn<float>(r, n);
  auto dy = sktest::randn<float>(r, n);
  const float ow = 0.25f;
  CudaModel m(c, p);
  sk::DBuf<float> d_x, d_dy, d_dx(n);
  d_x.upload(x);
  d_dy.upload(dy);
  m.forward(d_x.p, B, S, /*train=*/true);
  m.backward(d_dy.p, ow, d_dx.p);
  auto g = m.grads();
  auto dx = d_dx.download();

  std::vector<double> pd(p.begin(), p.end()), xd(x.begin(), x.end()), dyd(dy.begin(), dy.end());
  CpuState<double> st;
  auto fz = sidtest::freeze_from(m);
  cpu_forward(c, tab, pd.data(), xd.data(), B, S, st, &fz);
  std::vector<double> gc(tab.total), dxc(n);
  cpu_backward(c, tab, pd.data(), st, dyd.data(), (double)ow, gc.data(), dxc.data());
  std::printf("  %d iterations\n", m.iterations());
  bool ok = true;
  for (auto& e : tab.entries)
    ok &= sktest::report(e.name, sktest::compare(&g[e.offset], &gc[e.offset], e.size), 5e-3, 1e-6);
  ok &= sktest::report("input x", sktest::compare(dx.data(), dxc.data(), n), 5e-3, 1e-6);

  // SGD train step: parameters after one step must match the CPU step.
  {
    auto target = sktest::randn<float>(r, n);
    sk::SgdConfig oc;
    oc.lr = 0.05f;
    oc.momentum = 0.9f;
    oc.weight_decay = 1e-3f;
    sk::DBuf<float> d_t;
    d_t.upload(target);
    CudaModel m2(c, p);
    float lg = m2.train_step_sgd(d_x.p, d_t.p, B, S, oc);
    auto pg = m2.params();
    std::vector<double> pc(pd), buf(tab.total, 0.0), td(target.begin(), target.end());
    double lc = cpu_train_step_sgd<double>(c, tab, pc.data(), buf.data(), 1, oc, xd.data(), td.data(), B, S);
    std::printf("  sgd step: loss gpu %.6f cpu %.6f\n", lg, lc);
    double l = lg;
    ok &= sktest::report("sgd loss", sktest::compare(&l, &lc, 1), 1e-3);
    ok &= sktest::report("params after sgd step", sktest::compare(pg.data(), pc.data(), pc.size()), 1e-4);
    // AdamW: a few steps on the GPU must reduce the loss.
    CudaModel m3(c, p);
    sk::AdamWConfig ac;
    ac.lr = 1e-3f;
    float l0 = 0, l1 = 0;
    for (int s = 0; s < 8; ++s) {
      float L = m3.train_step_adamw(d_x.p, d_t.p, B, S, ac);
      if (s == 0) l0 = L;
      l1 = L;
    }
    bool dec = l1 < l0;
    std::printf("  adamw: loss %.6f -> %.6f after 8 steps  %s\n", l0, l1, dec ? "ok" : "FAIL");
    ok &= dec;
    auto pcf = m3.params();
    sid::Offsets o(tab);
    for (int k = 0; k < c.num_paths; ++k)
      if (pcf[o.path_confidence + k] != p[o.path_confidence + k]) {
        std::printf("  path_confidence changed (should be skipped like a None grad)  FAIL\n");
        ok = false;
        break;
      }
  }
  return ok;
}

int main() {
  if (sk::cuda_device_count() == 0) {
    std::printf("no CUDA device: skipping\n");
    return sktest::kSkip;
  }
  bool ok = true;
  {
    Config c;
    c.d_model = 32; c.num_paths = 4; c.num_heads = 2; c.head_dim = 8; c.max_iterations = 3;
    ok &= run("small", c, 2, 7, 0.3, 1);
  }
  {
    Config c;
    c.d_model = 40; c.num_paths = 3; c.num_heads = 4; c.head_dim = 6; c.max_iterations = 2;
    c.collapse_temperature = 0.5f;
    ok &= run("small-b", c, 3, 5, 0.25, 9);
  }
  {
    Config c;
    c.d_model = 128; c.num_paths = 4; c.num_heads = 8; c.head_dim = 16; c.max_iterations = 2;
    ok &= run("medium", c, 2, 16, 0.1, 4);
  }
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
