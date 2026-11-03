// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "sid/model.hpp"

#include <cmath>
#include <stdexcept>

#include "common/rng.hpp"

namespace sid {

void Config::validate() const {
  if (d_model <= 0 || num_paths <= 0 || num_heads <= 0 || head_dim <= 0 || max_iterations <= 0)
    throw std::runtime_error("sid::Config: dimensions must be positive");
  if (num_paths > kMaxPaths)
    throw std::runtime_error("sid::Config: num_paths > kMaxPaths (raise kMaxPaths and rebuild)");
}

static const std::string P = "refinement.heads.";

sk::ParamTable make_param_table(const Config& c) {
  c.validate();
  const int64_t d = c.d_model, K = c.num_paths, H = c.num_heads, dh = c.head_dim, E = H * dh;
  sk::ParamTable t;
  t.add(P + "path_basis", {K, d});
  t.add(P + "W_induction", {K, H, dh, dh});
  t.add(P + "W_Q.weight", {E, d});
  t.add(P + "W_Q.bias", {E});
  t.add(P + "W_K.weight", {E, d});
  t.add(P + "W_K.bias", {E});
  t.add(P + "W_V.weight", {E, d});
  t.add(P + "W_V.bias", {E});
  t.add(P + "W_O.weight", {d, E});
  t.add(P + "W_O.bias", {d});
  t.add(P + "path_confidence", {K}, /*has_grad=*/false);  // unused in forward
  t.add(P + "collapse_proj.weight", {d, d});
  t.add(P + "collapse_proj.bias", {d});
  t.add(P + "ln.weight", {d});
  t.add(P + "ln.bias", {d});
  t.add("output_proj.weight", {d, d});
  t.add("output_proj.bias", {d});
  return t;
}

Offsets::Offsets(const sk::ParamTable& t)
    : path_basis(t.off(P + "path_basis")),
      W_induction(t.off(P + "W_induction")),
      Wq(t.off(P + "W_Q.weight")), bq(t.off(P + "W_Q.bias")),
      Wk(t.off(P + "W_K.weight")), bk(t.off(P + "W_K.bias")),
      Wv(t.off(P + "W_V.weight")), bv(t.off(P + "W_V.bias")),
      Wo(t.off(P + "W_O.weight")), bo(t.off(P + "W_O.bias")),
      path_confidence(t.off(P + "path_confidence")),
      Wc(t.off(P + "collapse_proj.weight")), bc(t.off(P + "collapse_proj.bias")),
      ln_g(t.off(P + "ln.weight")), ln_b(t.off(P + "ln.bias")),
      Wp(t.off("output_proj.weight")), bp(t.off("output_proj.bias")) {}

static void init_linear(sk::Rng& r, float* W, float* b, int out, int in) {
  const double bound = 1.0 / std::sqrt((double)in);
  for (int i = 0; i < out * in; ++i) W[i] = (float)r.uniform(-bound, bound);
  for (int i = 0; i < out; ++i) b[i] = (float)r.uniform(-bound, bound);
}

std::vector<float> init_params(const Config& c, const sk::ParamTable& t, uint64_t seed) {
  sk::Rng r(seed);
  std::vector<float> p(t.total, 0.f);
  Offsets o(t);
  const int d = c.d_model, K = c.num_paths, H = c.num_heads, dh = c.head_dim, E = c.E();
  // nn.init.orthogonal_ on a [K, d] matrix (K <= d): orthonormal rows.
  {
    std::vector<double> v((size_t)K * d);
    for (auto& x : v) x = r.normal();
    for (int k = 0; k < K; ++k) {
      double* vk = &v[(size_t)k * d];
      for (int pass = 0; pass < 2; ++pass)
        for (int j = 0; j < k; ++j) {
          const double* vj = &v[(size_t)j * d];
          double dot = 0;
          for (int i = 0; i < d; ++i) dot += vk[i] * vj[i];
          for (int i = 0; i < d; ++i) vk[i] -= dot * vj[i];
        }
      double n = 0;
      for (int i = 0; i < d; ++i) n += vk[i] * vk[i];
      n = std::sqrt(n);
      for (int i = 0; i < d; ++i) vk[i] /= (n > 0 ? n : 1.0);
    }
    for (size_t i = 0; i < v.size(); ++i) p[o.path_basis + i] = (float)v[i];
  }
  for (size_t i = 0; i < (size_t)K * H * dh * dh; ++i) p[o.W_induction + i] = (float)(r.normal() * 0.02);
  init_linear(r, &p[o.Wq], &p[o.bq], E, d);
  init_linear(r, &p[o.Wk], &p[o.bk], E, d);
  init_linear(r, &p[o.Wv], &p[o.bv], E, d);
  init_linear(r, &p[o.Wo], &p[o.bo], d, E);
  for (int k = 0; k < K; ++k) p[o.path_confidence + k] = 1.0f / K;
  init_linear(r, &p[o.Wc], &p[o.bc], d, d);
  for (int i = 0; i < d; ++i) { p[o.ln_g + i] = 1.f; p[o.ln_b + i] = 0.f; }
  init_linear(r, &p[o.Wp], &p[o.bp], d, d);
  return p;
}

void save_weights(const std::string& path, const Config& c, const sk::ParamTable& t,
                  const std::vector<float>& params) {
  std::vector<sk::TensorRec> recs;
  recs.push_back(sk::make_scalar_i32("config.d_model", c.d_model));
  recs.push_back(sk::make_scalar_i32("config.num_paths", c.num_paths));
  recs.push_back(sk::make_scalar_i32("config.num_heads", c.num_heads));
  recs.push_back(sk::make_scalar_i32("config.head_dim", c.head_dim));
  recs.push_back(sk::make_scalar_i32("config.max_iterations", c.max_iterations));
  recs.push_back(sk::make_scalar_f32("config.interference_threshold", c.interference_threshold));
  recs.push_back(sk::make_scalar_f32("config.collapse_temperature", c.collapse_temperature));
  recs.push_back(sk::make_scalar_f32("config.orthogonality_reg", c.orthogonality_reg));
  recs.push_back(sk::make_scalar_f32("config.convergence_threshold", c.convergence_threshold));
  for (auto& e : t.entries) recs.push_back(sk::make_f32(e.name, e.shape, &params[e.offset]));
  sk::write_tensor_file(path, recs);
}

std::vector<float> load_weights(const std::string& path, Config& c, sk::ParamTable& t) {
  auto recs = sk::read_tensor_file(path);
  c.d_model = sk::scalar_i32(recs, "config.d_model", c.d_model);
  c.num_paths = sk::scalar_i32(recs, "config.num_paths", c.num_paths);
  c.num_heads = sk::scalar_i32(recs, "config.num_heads", c.num_heads);
  c.head_dim = sk::scalar_i32(recs, "config.head_dim", c.head_dim);
  c.max_iterations = sk::scalar_i32(recs, "config.max_iterations", c.max_iterations);
  c.interference_threshold = sk::scalar_f32(recs, "config.interference_threshold", c.interference_threshold);
  c.collapse_temperature = sk::scalar_f32(recs, "config.collapse_temperature", c.collapse_temperature);
  c.orthogonality_reg = sk::scalar_f32(recs, "config.orthogonality_reg", c.orthogonality_reg);
  c.convergence_threshold = sk::scalar_f32(recs, "config.convergence_threshold", c.convergence_threshold);
  t = make_param_table(c);
  std::vector<float> p(t.total, 0.f);
  for (auto& e : t.entries) {
    auto& r = sk::need_tensor(recs, e.name);
    if (r.dtype != sk::DType::F32 || r.shape != e.shape)
      throw std::runtime_error("shape/dtype mismatch for " + e.name);
    std::copy(r.f.begin(), r.f.end(), p.begin() + e.offset);
  }
  return p;
}

}  // namespace sid
