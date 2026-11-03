// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CUDA forward vs the CPU reference (small configs and the default 768/4/8/64).
#include <cstdio>

#include "sid_cuda_common.cuh"

using namespace sid;

static bool run(const char* name, Config c, int B, int S, double noise, uint64_t seed) {
  std::printf("[%s] d=%d K=%d H=%d dh=%d B=%d S=%d\n", name, c.d_model, c.num_paths, c.num_heads,
              c.head_dim, B, S);
  auto tab = make_param_table(c);
  auto p = sidtest::make_params(c, tab, seed, noise);
  sk::Rng r(seed + 1);
  auto x = sktest::randn<float>(r, (size_t)B * S * c.d_model);
  CudaModel m(c, p);
  sk::DBuf<float> dx;
  dx.upload(x);
  m.forward(dx.p, B, S, /*train=*/false);
  const size_t n = x.size();
  std::vector<float> y(n);
  SK_CUDA(cudaMemcpy(y.data(), m.device_y(), n * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU in double, with the GPU's discrete decisions (iterations, mask).
  std::vector<double> pd(p.begin(), p.end()), xd(x.begin(), x.end());
  CpuState<double> st, st_free;
  auto fz = sidtest::freeze_from(m);
  cpu_forward(c, tab, pd.data(), xd.data(), B, S, st, &fz);
  cpu_forward(c, tab, pd.data(), xd.data(), B, S, st_free);
  int on = 0;
  for (int v : fz.mask) on += v;
  std::printf("  gpu: %d iterations, final delta %.6g, %d interfering pairs; cpu (free): %d iterations\n",
              m.iterations(), m.final_delta(), on, st_free.iterations);
  bool ok = true;
  if (st_free.iterations != m.iterations() || st_free.mask != fz.mask)
    std::printf("  note: CPU and GPU took different discrete decisions (threshold boundary)\n");
  auto G = m.gamma();
  ok &= sktest::report("Gamma", sktest::compare(G.data(), st.G.data(), G.size()), 1e-4);
  float orth = m.orth_loss();
  double od = st.orth;
  ok &= sktest::report("orth loss", sktest::compare(&orth, &od, 1), 1e-3, 1e-6);
  ok &= sktest::report("output y", sktest::compare(y.data(), st.y.data(), n), 2e-3);
  return ok;
}

int main() {
  int ndev = sk::cuda_device_count();
  if (ndev == 0) {
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
    c.d_model = 48; c.num_paths = 8; c.num_heads = 3; c.head_dim = 16;
    ok &= run("k8", c, 3, 9, 0.2, 2);
  }
  {
    Config c;  // the PyTorch defaults
    ok &= run("default-768", c, 2, 32, 0.05, 3);
  }
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
