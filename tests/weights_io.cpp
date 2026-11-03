// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Save/load round trip of both weight files (config records + every tensor).
#include <cstdio>
#include <cstring>

#include "alexnet_mmep/model.hpp"
#include "sid/model.hpp"
#include "test_util.hpp"

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main() {
  bool ok = true;
  {
    sid::Config c;
    c.d_model = 40; c.num_paths = 3; c.num_heads = 2; c.head_dim = 8;
    c.interference_threshold = 0.25f; c.max_iterations = 7;
    auto t = sid::make_param_table(c);
    auto p = sid::init_params(c, t, 3);
    sid::save_weights("weights_io_sid.skt", c, t, p);
    sid::Config c2;
    sk::ParamTable t2;
    auto p2 = sid::load_weights("weights_io_sid.skt", c2, t2);
    bool k = same(p, p2) && c2.d_model == 40 && c2.num_paths == 3 && c2.max_iterations == 7 &&
             c2.interference_threshold == 0.25f && t2.entries.size() == t.entries.size();
    std::printf("[sid] %zu tensors, %zu floats  %s\n", t.entries.size(), t.total, k ? "ok" : "FAIL");
    ok &= k;
    std::remove("weights_io_sid.skt");
  }
  {
    amm::Config c;
    c.image = 99;
    const int ch[5] = {8, 12, 16, 16, 12};
    for (int i = 0; i < 5; ++i) c.ch[i] = ch[i];
    c.fc = 32; c.num_classes = 7; c.num_routes = 5; c.entropy_bound = 0.3f; c.act_alpha = 0.05f;
    auto t = amm::make_param_table(c);
    auto p = amm::init_params(c, t, 4);
    amm::save_weights("weights_io_amm.skt", c, t, p);
    amm::Config c2;
    sk::ParamTable t2;
    auto p2 = amm::load_weights("weights_io_amm.skt", c2, t2);
    bool k = same(p, p2) && c2.image == 99 && c2.fc == 32 && c2.num_routes == 5 &&
             c2.entropy_bound == 0.3f && c2.act_alpha == 0.05f && c2.ch[4] == 12 &&
             t2.entries.size() == t.entries.size();
    std::printf("[alexnet-mmep] %zu tensors, %zu floats  %s\n", t.entries.size(), t.total,
                k ? "ok" : "FAIL");
    ok &= k;
    std::remove("weights_io_amm.skt");
  }
  {
    amm::Config c;
    auto t = amm::make_param_table(c);
    size_t routers = 0;
    for (auto& e : t.entries)
      if (e.name.rfind("route", 0) == 0) routers += e.size;
    std::printf("[alexnet-mmep 224] %zu parameters (%zu in the two routers, %zu without)\n", t.total,
                routers, t.total - routers);
  }
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
