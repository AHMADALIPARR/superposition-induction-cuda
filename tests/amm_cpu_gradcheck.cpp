// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CPU backward of AlexNetMMEP (CE + entropy penalty) vs central finite
// differences (five-point), in double, on a 67x67 input with scaled-down channels.
// Discrete decisions (pool argmax, which rows bisect, the bisected
// temperature, whether each penalty fires) are frozen so the loss is smooth;
// PyTorch also treats the temperature as a constant.
#include <cmath>
#include <cstdio>
#include <vector>

#include "alexnet_mmep/cpu_ref.hpp"
#include "test_util.hpp"

using namespace amm;

static Config small_config() {
  Config c;
  c.image = 67;
  const int ch[5] = {4, 6, 8, 8, 6};
  for (int i = 0; i < 5; ++i) c.ch[i] = ch[i];
  c.fc = 16;
  c.num_classes = 5;
  c.num_routes = 3;
  return c;
}

// Five-point central difference, O(h^4). The bisected temperature can be as
// low as 0.01, which makes the router logits 100x steeper than the rest.
template <class F>
static double fd5(F&& f, double h) {
  const double a = f(h) - f(-h), b = f(2 * h) - f(-2 * h);
  return (8 * a - b) / (12 * h);
}

enum class Mode { Bisect, Penalty, Plain };

static bool run(const char* name, Config c, Mode mode, int N, uint64_t seed) {
  Shapes s = shapes(c);
  std::printf("[%s] %dx%d input, ch %d/%d/%d/%d/%d, fc %d, routes %d, classes %d, N=%d, fc6_in=%d\n",
              name, c.image, c.image, c.ch[0], c.ch[1], c.ch[2], c.ch[3], c.ch[4], c.fc,
              c.num_routes, c.num_classes, N, s.fc6_in);
  if (mode == Mode::Plain) c.entropy_bound = 5.0f;  // > ln(K): no row ever bisects
  sk::Rng r(seed);
  auto tab = make_param_table(c);
  auto pf = init_params(c, tab, seed);
  std::vector<double> p(pf.begin(), pf.end());
  Offsets o(c, tab);
  if (mode == Mode::Penalty) {
    // Nearly tied router logits: even tau = 0.01 cannot reach the bound, so the
    // batch entropy stays above it and the penalty fires with a real gradient.
    for (size_t i = 0; i < (size_t)c.num_routes * c.fc; ++i) p[o.r6.Wr + i] *= 1e-3;
    for (int k = 0; k < c.num_routes; ++k) p[o.r6.br + k] *= 1e-3;
  }
  auto x = sktest::randn<double>(r, (size_t)N * c.in_ch * c.image * c.image);
  std::vector<int> labels(N);
  for (int n = 0; n < N; ++n) labels[n] = (int)(r.uniform() * c.num_classes) % c.num_classes;

  CpuState<double> st;
  cpu_forward(c, tab, p.data(), x.data(), N, st);
  std::vector<double> dl, d6, d7;
  LossInfo li = cpu_loss(c, st, labels.data(), dl, d6, d7);
  int v6 = 0, v7 = 0;
  for (int n = 0; n < N; ++n) { v6 += st.r6.viol[n]; v7 += st.r7.viol[n]; }
  std::printf("  loss %.6f (ce %.6f)  ent6 %.4f ent7 %.4f  bisect rows %d/%d  penalty %d/%d\n",
              li.loss, li.ce, li.ent6, li.ent7, v6, v7, li.pen6, li.pen7);
  bool ok = true;
  if (mode == Mode::Penalty && !li.pen6) { std::printf("  expected the fc6 penalty to fire  FAIL\n"); ok = false; }
  if (mode == Mode::Plain && (v6 || v7)) { std::printf("  expected no bisection  FAIL\n"); ok = false; }
  if (mode == Mode::Bisect && !(v6 && v7)) { std::printf("  expected bisection  FAIL\n"); ok = false; }

  Freeze fz;
  fz.am1 = st.am1; fz.am2 = st.am2; fz.am5 = st.am5;
  fz.viol6 = st.r6.viol; fz.viol7 = st.r7.viol;
  fz.tau6.assign(st.r6.tau.begin(), st.r6.tau.end());
  fz.tau7.assign(st.r7.tau.begin(), st.r7.tau.end());
  fz.pen6 = li.pen6; fz.pen7 = li.pen7;

  std::vector<double> g(tab.total), dx(x.size());
  cpu_backward(c, tab, p.data(), st, dl.data(), d6.data(), d7.data(), g.data(), dx.data());

  auto loss = [&](const std::vector<double>& pp, const std::vector<double>& xx) {
    CpuState<double> s2;
    cpu_forward(c, tab, pp.data(), xx.data(), N, s2, &fz);
    std::vector<double> a, b, e;
    return cpu_loss(c, s2, labels.data(), a, b, e, &fz).loss;
  };
  const double h = 1e-5, tol = 1e-5, atol = 1e-9;
  for (auto& e : tab.entries) {
    auto idx = sktest::probe(r, e.size, 6);
    std::vector<double> fd, an;
    for (size_t i : idx) {
      auto pp = p;
      auto f = [&](double dv) { pp[e.offset + i] = p[e.offset + i] + dv; return loss(pp, x); };
      fd.push_back(fd5(f, h));
      an.push_back(g[e.offset + i]);
    }
    ok &= sktest::report(e.name, sktest::compare(an.data(), fd.data(), fd.size()), tol, atol);
  }
  {
    auto idx = sktest::probe(r, x.size(), 24);
    std::vector<double> fd, an;
    for (size_t i : idx) {
      auto xx = x;
      auto f = [&](double dv) { xx[i] = x[i] + dv; return loss(p, xx); };
      fd.push_back(fd5(f, h));
      an.push_back(dx[i]);
    }
    ok &= sktest::report("input (dx)", sktest::compare(an.data(), fd.data(), fd.size()), tol, atol);
  }
  return ok;
}

int main() {
  bool ok = true;
  // Shape check from the source: nn.Linear(256*6*6, 4096) needs 9216 features.
  for (int img : {224, 227}) {
    Config c;
    c.image = img;
    Shapes s = shapes(c);
    std::printf("[shapes] %d -> conv1 %d pool %d conv2 %d pool %d conv5 %d pool %d -> fc6_in %d %s\n",
                img, s.c[0], s.p1, s.c[1], s.p2, s.c[4], s.p5, s.fc6_in,
                s.fc6_in == 256 * 6 * 6 ? "(= 256*6*6, ok)" : "MISMATCH");
    ok &= s.fc6_in == 256 * 6 * 6;
  }
  ok &= run("bisect", small_config(), Mode::Bisect, 2, 11);
  ok &= run("penalty", small_config(), Mode::Penalty, 3, 12);
  ok &= run("plain-softmax", small_config(), Mode::Plain, 2, 13);
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
