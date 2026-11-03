// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// AlexNet-MMEP CUDA forward vs the CPU reference: two scaled-down configs and
// the full 224x224 network (fc6_in = 9216).
#include "amm_cuda_common.cuh"

using namespace amm;

template <class T>
static bool run(const char* name, Config c, int N, uint64_t seed, bool flat6) {
  Shapes s = shapes(c);
  std::printf("[%s] %dx%d, ch %d/%d/%d/%d/%d, fc %d, routes %d, N=%d, fc6_in=%d\n", name, c.image,
              c.image, c.ch[0], c.ch[1], c.ch[2], c.ch[3], c.ch[4], c.fc, c.num_routes, N, s.fc6_in);
  auto tab = make_param_table(c);
  auto p = init_params(c, tab, seed);
  if (flat6) ammtest::flatten_router6(c, tab, p);
  sk::Rng r(seed + 7);
  auto x = sktest::randn<float>(r, (size_t)N * c.in_ch * c.image * c.image);
  CudaModel m(c, p);
  sk::DBuf<float> dx;
  dx.upload(x);
  m.forward(dx.p, N);
  std::vector<float> logits((size_t)N * c.num_classes);
  SK_CUDA(cudaMemcpy(logits.data(), m.device_logits(), logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));

  std::vector<T> pt(p.begin(), p.end()), xt(x.begin(), x.end());
  CpuState<T> st, st_free;
  auto fz = ammtest::freeze_from(m);
  cpu_forward(c, tab, pt.data(), xt.data(), N, st, &fz);
  cpu_forward(c, tab, pt.data(), xt.data(), N, st_free);
  int v6 = 0, v7 = 0;
  for (int n = 0; n < N; ++n) { v6 += fz.viol6[n]; v7 += fz.viol7[n]; }
  std::printf("  gpu: ent6 %.5f ent7 %.5f, bisect rows %d/%d (iters %d/%d)\n", m.ent6(), m.ent7(),
              v6, v7, m.bisect_iters(6), m.bisect_iters(7));
  if (st_free.am1 != fz.am1 || st_free.am2 != fz.am2 || st_free.am5 != fz.am5 ||
      st_free.r6.viol != fz.viol6 || st_free.r7.viol != fz.viol7)
    std::printf("  note: CPU and GPU took different discrete decisions (ties / bound boundary)\n");
  bool ok = true;
  if (st_free.r6.viol == fz.viol6)
    ok &= sktest::report("tau6 (free CPU bisection)", sktest::compare(fz.tau6.data(), st_free.r6.tau.data(), N), 1e-4);
  ok &= sktest::report("logits", sktest::compare(logits.data(), st.logits.data(), logits.size()), 2e-3);
  double e6 = m.ent6(), e7 = m.ent7();
  double c6 = st.ent6, c7 = st.ent7;
  ok &= sktest::report("ent6", sktest::compare(&e6, &c6, 1), 1e-3, 1e-5);
  ok &= sktest::report("ent7", sktest::compare(&e7, &c7, 1), 1e-3, 1e-5);
  return ok;
}

int main() {
  if (sk::cuda_device_count() == 0) {
    std::printf("no CUDA device: skipping\n");
    return sktest::kSkip;
  }
  bool ok = true;
  ok &= run<double>("small-67", ammtest::small67(), 3, 1, false);
  ok &= run<double>("small-67-penalty", ammtest::small67(), 3, 2, true);
  ok &= run<double>("mid-99", ammtest::mid99(), 4, 3, false);
  ok &= run<float>("alexnet-224", Config{}, 2, 4, false);
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
