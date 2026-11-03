// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Small helpers shared by the tests.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common/param_table.hpp"
#include "common/rng.hpp"

namespace sktest {

// Exit code ctest treats as "skipped" (no GPU available).
constexpr int kSkip = 77;

struct Cmp {
  double max_abs = 0, max_rel = 0, ref_max = 0;
  size_t worst = 0;
};

// Relative error is |a-b| / max(|b|, floor), floor = 1e-3 * max|b| to avoid
// blowing up on entries that are essentially zero.
template <class A, class B>
Cmp compare(const A* a, const B* b, size_t n) {
  Cmp c;
  for (size_t i = 0; i < n; ++i) c.ref_max = std::max(c.ref_max, std::fabs((double)b[i]));
  const double floor = std::max(1e-3 * c.ref_max, 1e-12);
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    double r = d / std::max(std::fabs((double)b[i]), floor);
    if (r > c.max_rel) { c.max_rel = r; c.worst = i; }
    c.max_abs = std::max(c.max_abs, d);
  }
  return c;
}

// Passes when the relative error is within tol, or every absolute error is
// within atol (for tensors whose true gradient is ~0, e.g. the key bias, which
// softmax is invariant to).
inline bool report(const std::string& what, const Cmp& c, double tol, double atol = 0.0) {
  bool ok = c.max_rel <= tol || c.max_abs <= atol;
  std::printf("  %-40s max_abs %.3e  max_rel %.3e  (|ref|max %.3e)  %s\n", what.c_str(),
              c.max_abs, c.max_rel, c.ref_max, ok ? "ok" : "FAIL");
  return ok;
}

template <class T>
std::vector<T> randn(sk::Rng& r, size_t n, double s = 1.0) {
  std::vector<T> v(n);
  for (auto& x : v) x = (T)(r.normal() * s);
  return v;
}

// Indices to probe in a tensor: all of them if small, otherwise a random subset.
inline std::vector<size_t> probe(sk::Rng& r, size_t n, size_t max_n) {
  std::vector<size_t> idx;
  if (n <= max_n) {
    for (size_t i = 0; i < n; ++i) idx.push_back(i);
  } else {
    for (size_t i = 0; i < max_n; ++i) idx.push_back((size_t)(r.uniform() * n) % n);
  }
  return idx;
}

}  // namespace sktest
