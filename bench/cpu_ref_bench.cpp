// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Single-thread timings of the plain C++ references (float). These are the
// correctness baselines, not tuned code; they set a floor for the CUDA numbers.
//   cpu_ref_bench [--reps=3]
#include <chrono>
#include <cstdio>

#include "../apps/args.hpp"
#include "alexnet_mmep/cpu_ref.hpp"
#include "common/rng.hpp"
#include "sid/cpu_ref.hpp"

template <class F>
static double best_ms(F&& f, int reps) {
  double best = 1e300;
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

static std::vector<float> randn(sk::Rng& r, size_t n) {
  std::vector<float> v(n);
  for (auto& x : v) x = (float)r.normal();
  return v;
}

static void sid_case(const char* name, sid::Config c, int B, int S, int reps) {
  auto t = sid::make_param_table(c);
  auto p = sid::init_params(c, t, 1);
  sk::Rng r(2);
  auto x = randn(r, (size_t)B * S * c.d_model), y = randn(r, x.size());
  sid::CpuState<float> st;
  double f = best_ms([&] { sid::cpu_forward(c, t, p.data(), x.data(), B, S, st); }, reps);
  const int it = st.iterations;
  std::vector<float> m(t.total, 0.f), v(t.total, 0.f), buf(t.total, 0.f);
  sk::AdamWConfig oc;
  int step = 0;
  double s = best_ms([&] {
    sid::cpu_train_step_adamw(c, t, p.data(), m.data(), v.data(), ++step, oc, x.data(), y.data(), B, S);
  }, reps);
  std::printf("| sid %-14s | B=%d S=%d | %d | %10.1f ms | %10.0f tok/s | %10.1f ms | %10.0f tok/s |\n",
              name, B, S, it, f, B * S / (f * 1e-3), s, B * S / (s * 1e-3));
}

static void amm_case(const char* name, amm::Config c, int N, int reps, bool train) {
  auto t = amm::make_param_table(c);
  std::vector<float> p = amm::init_params(c, t, 1);
  sk::Rng r(2);
  auto x = randn(r, (size_t)N * c.in_ch * c.image * c.image);
  std::vector<int> lab(N);
  for (auto& l : lab) l = (int)(r.uniform() * c.num_classes) % c.num_classes;
  amm::CpuState<float> st;
  double f = best_ms([&] { amm::cpu_forward(c, t, p.data(), x.data(), N, st); }, reps);
  if (!train) {
    std::printf("| amm %-14s | N=%d %dx%d | - | %10.1f ms | %10.2f img/s | %13s | %16s |\n", name, N,
                c.image, c.image, f, N / (f * 1e-3), "-", "-");
    return;
  }
  std::vector<float> buf(t.total, 0.f);
  sk::SgdConfig sc;
  sc.lr = 0.01f; sc.momentum = 0.9f; sc.weight_decay = 5e-4f;
  int step = 0;
  double s = best_ms([&] { amm::cpu_train_step_sgd(c, t, p, buf, ++step, x.data(), lab.data(), N, sc); }, reps);
  std::printf("| amm %-14s | N=%d %dx%d | - | %10.1f ms | %10.2f img/s | %10.1f ms | %10.2f img/s |\n",
              name, N, c.image, c.image, f, N / (f * 1e-3), s, N / (s * 1e-3));
}

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  const int reps = a.i("reps", 3);
  std::printf("| case | shape | refine iters | forward | forward rate | train step | train rate |\n");
  std::printf("| --- | --- | --- | --- | --- | --- | --- |\n");
  {
    sid::Config c;
    c.d_model = 128; c.num_paths = 4; c.num_heads = 4; c.head_dim = 32;
    sid_case("d128", c, 4, 64, reps);
  }
  sid_case("default-768", sid::Config{}, 1, 64, reps);
  {
    amm::Config c;
    c.image = 67;
    const int ch[5] = {4, 6, 8, 8, 6};
    for (int i = 0; i < 5; ++i) c.ch[i] = ch[i];
    c.fc = 16; c.num_classes = 5; c.num_routes = 3;
    amm_case("small-67", c, 8, reps, true);
  }
  amm_case("alexnet-224", amm::Config{}, 1, 1, false);
  return 0;
}
