// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CPU backward of the Superpositioned Induction Daemon vs central finite
// differences, in double precision. Discrete decisions (interference mask,
// number of refinement iterations) are frozen so the loss is smooth.
#include <cmath>
#include <cstdio>
#include <vector>

#include "sid/cpu_ref.hpp"
#include "test_util.hpp"

using namespace sid;

// Pick a threshold halfway inside the widest gap of the sorted |Gamma| values,
// so some pairs interfere and others do not, with a clear margin.
static double pick_threshold(const CpuState<double>& st, int K) {
  std::vector<double> g;
  for (int a = 0; a < K; ++a)
    for (int b = 0; b < K; ++b)
      if (a != b) g.push_back(std::fabs(st.G[a * K + b]));
  std::sort(g.begin(), g.end());
  g.erase(std::unique(g.begin(), g.end(), [](double x, double y) { return std::fabs(x - y) < 1e-12; }), g.end());
  double best = 0, thr = 0.1;
  for (size_t i = 0; i + 1 < g.size(); ++i)
    if (g[i + 1] - g[i] > best) { best = g[i + 1] - g[i]; thr = 0.5 * (g[i] + g[i + 1]); }
  return thr;
}

static bool run(const char* name, Config c, int B, int S, uint64_t seed) {
  std::printf("[%s] d=%d K=%d H=%d dh=%d B=%d S=%d iters<=%d\n", name, c.d_model, c.num_paths,
              c.num_heads, c.head_dim, B, S, c.max_iterations);
  sk::Rng r(seed);
  auto tab = make_param_table(c);
  auto pf = init_params(c, tab, seed);
  std::vector<double> p(pf.begin(), pf.end());
  Offsets o(tab);
  // Non-orthogonal basis so Gamma has off-diagonal mass; non-trivial LN affine.
  for (size_t i = 0; i < (size_t)c.num_paths * c.d_model; ++i) p[o.path_basis + i] = r.normal() / std::sqrt((double)c.d_model) + (i % c.d_model == 0 ? 0.5 : 0.0);
  for (int i = 0; i < c.d_model; ++i) { p[o.ln_g + i] = 1.0 + 0.3 * r.normal(); p[o.ln_b + i] = 0.1 * r.normal(); }
  for (size_t i = 0; i < (size_t)c.num_paths * c.num_heads * c.head_dim * c.head_dim; ++i) p[o.W_induction + i] = 0.3 * r.normal();
  auto x = sktest::randn<double>(r, (size_t)B * S * c.d_model);

  CpuState<double> st;
  cpu_forward(c, tab, p.data(), x.data(), B, S, st);
  c.interference_threshold = (float)pick_threshold(st, c.num_paths);
  cpu_forward(c, tab, p.data(), x.data(), B, S, st);
  int on = 0;
  for (int v : st.mask) on += v;
  std::printf("  threshold %.4f -> %d of %d off-diagonal pairs interfere, %d iterations\n",
              c.interference_threshold, on, c.num_paths * (c.num_paths - 1), st.iterations);
  Freeze fz;
  fz.iterations = st.iterations;
  fz.mask = st.mask;

  const size_t n = (size_t)B * S * c.d_model;
  auto w = sktest::randn<double>(r, n);
  const double ow = 0.37;  // orthogonality weight (non-trivial, to test that path)
  auto loss = [&](const std::vector<double>& pp, const std::vector<double>& xx) {
    CpuState<double> s;
    cpu_forward(c, tab, pp.data(), xx.data(), B, S, s, &fz);
    double L = 0;
    for (size_t i = 0; i < n; ++i) L += w[i] * s.y[i];
    return L + ow * s.orth;
  };
  std::vector<double> g(tab.total), dx(n);
  cpu_backward(c, tab, p.data(), st, w.data(), ow, g.data(), dx.data());

  bool ok = true;
  const double h = 1e-5, tol = 1e-5, atol = 1e-8;
  for (auto& e : tab.entries) {
    auto idx = sktest::probe(r, e.size, 24);
    std::vector<double> fd, an;
    for (size_t i : idx) {
      auto pp = p;
      pp[e.offset + i] += h;
      double lp = loss(pp, x);
      pp[e.offset + i] -= 2 * h;
      double lm = loss(pp, x);
      fd.push_back((lp - lm) / (2 * h));
      an.push_back(g[e.offset + i]);
    }
    ok &= sktest::report(e.name, sktest::compare(an.data(), fd.data(), fd.size()), tol, atol);
    if (!e.trainable_grad) {
      double mx = 0;
      for (size_t i = 0; i < e.size; ++i) mx = std::max(mx, std::fabs(g[e.offset + i]));
      if (mx != 0) { std::printf("  %s grad should be exactly 0\n", e.name.c_str()); ok = false; }
    }
  }
  {
    auto idx = sktest::probe(r, n, 40);
    std::vector<double> fd, an;
    for (size_t i : idx) {
      auto xx = x;
      xx[i] += h;
      double lp = loss(p, xx);
      xx[i] -= 2 * h;
      double lm = loss(p, xx);
      fd.push_back((lp - lm) / (2 * h));
      an.push_back(dx[i]);
    }
    ok &= sktest::report("input x", sktest::compare(an.data(), fd.data(), fd.size()), tol, atol);
  }
  return ok;
}

int main() {
  bool ok = true;
  {
    Config c;
    c.d_model = 12; c.num_paths = 4; c.num_heads = 2; c.head_dim = 3; c.max_iterations = 3;
    c.collapse_temperature = 0.7f;
    ok &= run("small-3iter", c, 2, 5, 11);
  }
  {
    Config c;  // early stop path: generous convergence threshold
    c.d_model = 10; c.num_paths = 3; c.num_heads = 2; c.head_dim = 4; c.max_iterations = 5;
    c.convergence_threshold = 0.9f;
    ok &= run("small-earlystop", c, 2, 4, 23);
  }
  {
    Config c;  // K = 1: no interference at all
    c.d_model = 8; c.num_paths = 1; c.num_heads = 1; c.head_dim = 4; c.max_iterations = 2;
    ok &= run("single-path", c, 1, 3, 5);
  }
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
