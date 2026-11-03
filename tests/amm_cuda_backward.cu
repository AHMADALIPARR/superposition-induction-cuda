// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// AlexNet-MMEP CUDA backward (every parameter and the input) and one SGD step
// vs the CPU reference in double, with the GPU's discrete decisions replayed.
#include "amm_cuda_common.cuh"

using namespace amm;

static bool run(const char* name, Config c, int N, uint64_t seed, bool flat6, double tol) {
  std::printf("[%s] %dx%d, fc %d, routes %d, N=%d\n", name, c.image, c.image, c.fc, c.num_routes, N);
  auto tab = make_param_table(c);
  auto p = init_params(c, tab, seed);
  if (flat6) ammtest::flatten_router6(c, tab, p);
  sk::Rng r(seed + 7);
  auto x = sktest::randn<float>(r, (size_t)N * c.in_ch * c.image * c.image);
  auto labels = ammtest::labels_for(r, N, c.num_classes);
  CudaModel m(c, p);
  sk::DBuf<float> d_x, d_dx, d_dl, d_d6, d_d7;
  sk::DBuf<int> d_lab;
  d_x.upload(x);
  d_lab.upload(labels);
  StepStats gs = m.loss_and_grad(d_x.p, d_lab.p, N);
  auto g_gpu = m.grads();

  std::vector<double> pd(p.begin(), p.end()), xd(x.begin(), x.end());
  auto fz = ammtest::freeze_from(m);
  CpuState<double> st;
  cpu_forward(c, tab, pd.data(), xd.data(), N, st, &fz);
  std::vector<double> dl, d6, d7;
  LossInfo li = cpu_loss(c, st, labels.data(), dl, d6, d7, &fz);
  std::vector<double> g(tab.total), dx(x.size());
  cpu_backward(c, tab, pd.data(), st, dl.data(), d6.data(), d7.data(), g.data(), dx.data());
  std::printf("  gpu loss %.6f (ce %.6f, pen %d/%d)  cpu loss %.6f (ce %.6f)\n", gs.loss, gs.ce,
              gs.pen6, gs.pen7, li.loss, li.ce);
  bool ok = true;
  ok &= sktest::report("loss", sktest::compare(&gs.loss, &li.loss, 1), 1e-4, 1e-6);
  for (auto& e : tab.entries)
    ok &= sktest::report(e.name, sktest::compare(g_gpu.data() + e.offset, g.data() + e.offset, e.size),
                         tol, 1e-6);

  // Input gradient through the public backward(), fed the CPU's dlogits/dent.
  std::vector<float> dlf(dl.begin(), dl.end()), d6f(d6.begin(), d6.end()), d7f(d7.begin(), d7.end());
  d_dl.upload(dlf);
  d_d6.upload(d6f);
  d_d7.upload(d7f);
  d_dx.resize(x.size());
  m.backward(d_dl.p, d_d6.p, d_d7.p, d_dx.p);
  auto dxg = d_dx.download();
  ok &= sktest::report("input (dx)", sktest::compare(dxg.data(), dx.data(), dx.size()), tol, 1e-7);

  // One trainer step: SGD(lr 0.01, momentum 0.9, weight_decay 5e-4).
  sk::SgdConfig sc;
  sc.lr = 0.01f; sc.momentum = 0.9f; sc.weight_decay = 5e-4f;
  m.train_step_sgd(d_x.p, d_lab.p, N, sc);
  auto p1 = m.params();
  std::vector<double> buf(tab.total, 0.0), pref = pd;
  sk::sgd_step_host(sc, 1, pref.data(), g.data(), buf.data(), nullptr, tab.total);
  ok &= sktest::report("params after one SGD step", sktest::compare(p1.data(), pref.data(), p1.size()),
                       1e-4, 1e-6);
  return ok;
}

static bool train_smoke() {
  Config c = ammtest::small67();
  const int N = 8;
  auto tab = make_param_table(c);
  CudaModel m(c, init_params(c, tab, 99));
  sk::Rng r(5);
  auto x = sktest::randn<float>(r, (size_t)N * c.in_ch * c.image * c.image);
  auto labels = ammtest::labels_for(r, N, c.num_classes);
  sk::DBuf<float> d_x;
  sk::DBuf<int> d_lab;
  d_x.upload(x);
  d_lab.upload(labels);
  sk::SgdConfig sc;
  sc.lr = 0.05f; sc.momentum = 0.9f; sc.weight_decay = 5e-4f;
  double first = 0, last = 0;
  for (int i = 0; i < 40; ++i) {
    auto s = m.train_step_sgd(d_x.p, d_lab.p, N, sc);
    if (i == 0) first = s.loss;
    last = s.loss;
  }
  bool ok = last < first;
  std::printf("[train-smoke] loss %.4f -> %.4f over 40 steps on one batch  %s\n", first, last,
              ok ? "ok" : "FAIL");
  return ok;
}

int main() {
  if (sk::cuda_device_count() == 0) {
    std::printf("no CUDA device: skipping\n");
    return sktest::kSkip;
  }
  bool ok = true;
  ok &= run("small-67", ammtest::small67(), 3, 1, false, 2e-3);
  ok &= run("small-67-penalty", ammtest::small67(), 3, 2, true, 2e-3);
  ok &= run("mid-99", ammtest::mid99(), 4, 3, false, 5e-3);
  ok &= train_smoke();
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
