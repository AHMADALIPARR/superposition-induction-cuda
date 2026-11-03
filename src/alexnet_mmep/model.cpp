// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "alexnet_mmep/model.hpp"

#include <cmath>
#include <stdexcept>

#include "common/rng.hpp"
#include "common/tensor_file.hpp"

namespace amm {

static int conv_out(int n, const ConvGeom& g) { return (n + 2 * g.pad - g.k) / g.stride + 1; }
static int pool_out(int n) { return (n - kPoolK) / kPoolS + 1; }

Shapes shapes(const Config& c) {
  Shapes s{};
  s.in = c.image;
  auto chk = [](int v, const char* what) {
    if (v < 1) throw std::runtime_error(std::string("amm: input too small, ") + what + " < 1");
  };
  if (c.image + 2 * kConv[0].pad < kConv[0].k) throw std::runtime_error("amm: input too small for conv1");
  s.c[0] = conv_out(c.image, kConv[0]); chk(s.c[0], "conv1");
  if (s.c[0] < kPoolK) throw std::runtime_error("amm: input too small for pool1");
  s.p1 = pool_out(s.c[0]);
  s.c[1] = conv_out(s.p1, kConv[1]); chk(s.c[1], "conv2");
  if (s.c[1] < kPoolK) throw std::runtime_error("amm: input too small for pool2");
  s.p2 = pool_out(s.c[1]);
  s.c[2] = conv_out(s.p2, kConv[2]);
  s.c[3] = conv_out(s.c[2], kConv[3]);
  s.c[4] = conv_out(s.c[3], kConv[4]); chk(s.c[4], "conv5");
  if (s.c[4] < kPoolK) throw std::runtime_error("amm: input too small for pool5");
  s.p5 = pool_out(s.c[4]);
  s.fc6_in = c.ch[4] * s.p5 * s.p5;
  return s;
}

void Config::validate() const {
  if (in_ch <= 0 || fc <= 0 || num_classes <= 0 || num_routes <= 0)
    throw std::runtime_error("amm::Config: dimensions must be positive");
  for (int i = 0; i < 5; ++i)
    if (ch[i] <= 0) throw std::runtime_error("amm::Config: channels must be positive");
  if (num_routes > kMaxRoutes)
    throw std::runtime_error("amm::Config: num_routes > kMaxRoutes (raise kMaxRoutes and rebuild)");
  Shapes s = shapes(*this);
  if (fc6_in >= 0 && fc6_in != s.fc6_in)
    throw std::runtime_error("amm::Config: fc6_in = " + std::to_string(fc6_in) +
                             " but the conv stack gives " + std::to_string(s.fc6_in) +
                             " for a " + std::to_string(image) + "x" + std::to_string(image) +
                             " input");
}

static void add_router(sk::ParamTable& t, const std::string& p, int d, int K) {
  t.add(p + ".router.weight", {K, d});
  t.add(p + ".router.bias", {K});
  for (int k = 0; k < K; ++k) {
    t.add(p + ".route_transforms." + std::to_string(k) + ".weight", {d, d});
    t.add(p + ".route_transforms." + std::to_string(k) + ".bias", {d});
  }
}

sk::ParamTable make_param_table(const Config& c) {
  c.validate();
  Shapes s = shapes(c);
  sk::ParamTable t;
  int cin = c.in_ch;
  for (int i = 0; i < 5; ++i) {
    const std::string n = "conv" + std::to_string(i + 1);
    t.add(n + ".weight", {c.ch[i], cin, kConv[i].k, kConv[i].k});
    t.add(n + ".bias", {c.ch[i]});
    cin = c.ch[i];
  }
  t.add("fc6.weight", {c.fc, s.fc6_in});
  t.add("fc6.bias", {c.fc});
  add_router(t, "route6", c.fc, c.num_routes);
  t.add("fc7.weight", {c.fc, c.fc});
  t.add("fc7.bias", {c.fc});
  add_router(t, "route7", c.fc, c.num_routes);
  t.add("fc8.weight", {c.num_classes, c.fc});
  t.add("fc8.bias", {c.num_classes});
  return t;
}

static RouterOff router_off(const Config& c, const sk::ParamTable& t, const std::string& p) {
  RouterOff r{};
  r.Wr = t.off(p + ".router.weight");
  r.br = t.off(p + ".router.bias");
  for (int k = 0; k < c.num_routes; ++k) {
    r.W[k] = t.off(p + ".route_transforms." + std::to_string(k) + ".weight");
    r.b[k] = t.off(p + ".route_transforms." + std::to_string(k) + ".bias");
  }
  return r;
}

Offsets::Offsets(const Config& c, const sk::ParamTable& t) {
  for (int i = 0; i < 5; ++i) {
    convW[i] = t.off("conv" + std::to_string(i + 1) + ".weight");
    convB[i] = t.off("conv" + std::to_string(i + 1) + ".bias");
  }
  fc6W = t.off("fc6.weight"); fc6b = t.off("fc6.bias");
  fc7W = t.off("fc7.weight"); fc7b = t.off("fc7.bias");
  fc8W = t.off("fc8.weight"); fc8b = t.off("fc8.bias");
  r6 = router_off(c, t, "route6");
  r7 = router_off(c, t, "route7");
}

std::vector<float> init_params(const Config& c, const sk::ParamTable& t, uint64_t seed) {
  (void)c;
  sk::Rng r(seed);
  std::vector<float> p(t.total, 0.f);
  // Weights and their biases appear as consecutive (weight, bias) pairs.
  for (size_t i = 0; i + 1 < t.entries.size(); i += 2) {
    const auto& W = t.entries[i];
    const auto& b = t.entries[i + 1];
    size_t fan_in = W.size / (size_t)W.shape[0];
    const double bound = 1.0 / std::sqrt((double)fan_in);
    for (size_t j = 0; j < W.size; ++j) p[W.offset + j] = (float)r.uniform(-bound, bound);
    for (size_t j = 0; j < b.size; ++j) p[b.offset + j] = (float)r.uniform(-bound, bound);
  }
  return p;
}

void save_weights(const std::string& path, const Config& c, const sk::ParamTable& t,
                  const std::vector<float>& params) {
  std::vector<sk::TensorRec> recs;
  recs.push_back(sk::make_scalar_i32("config.in_ch", c.in_ch));
  recs.push_back(sk::make_scalar_i32("config.image", c.image));
  for (int i = 0; i < 5; ++i)
    recs.push_back(sk::make_scalar_i32("config.ch" + std::to_string(i + 1), c.ch[i]));
  recs.push_back(sk::make_scalar_i32("config.fc", c.fc));
  recs.push_back(sk::make_scalar_i32("config.num_classes", c.num_classes));
  recs.push_back(sk::make_scalar_i32("config.num_routes", c.num_routes));
  recs.push_back(sk::make_scalar_f32("config.entropy_bound", c.entropy_bound));
  recs.push_back(sk::make_scalar_f32("config.penalty_weight", c.penalty_weight));
  // The activation's registered buffers, under their state_dict names.
  recs.push_back(sk::make_scalar_f32("activation._alpha", c.act_alpha));
  recs.push_back(sk::make_scalar_f32("activation._beta", c.act_beta));
  for (auto& e : t.entries) recs.push_back(sk::make_f32(e.name, e.shape, &params[e.offset]));
  sk::write_tensor_file(path, recs);
}

std::vector<float> load_weights(const std::string& path, Config& c, sk::ParamTable& t) {
  auto recs = sk::read_tensor_file(path);
  c.in_ch = sk::scalar_i32(recs, "config.in_ch", c.in_ch);
  c.image = sk::scalar_i32(recs, "config.image", c.image);
  for (int i = 0; i < 5; ++i) c.ch[i] = sk::scalar_i32(recs, "config.ch" + std::to_string(i + 1), c.ch[i]);
  c.fc = sk::scalar_i32(recs, "config.fc", c.fc);
  c.num_classes = sk::scalar_i32(recs, "config.num_classes", c.num_classes);
  c.num_routes = sk::scalar_i32(recs, "config.num_routes", c.num_routes);
  c.entropy_bound = sk::scalar_f32(recs, "config.entropy_bound", c.entropy_bound);
  c.penalty_weight = sk::scalar_f32(recs, "config.penalty_weight", c.penalty_weight);
  c.act_alpha = sk::scalar_f32(recs, "activation._alpha", c.act_alpha);
  c.act_beta = sk::scalar_f32(recs, "activation._beta", c.act_beta);
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

}  // namespace amm
