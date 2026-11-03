// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Write freshly initialised AlexNet-MMEP weights.
//   amm_init out.skt [--image=224] [--classes=1000] [--routes=4] [--bound=0.2] [--seed=0]
#include <cstdio>

#include "alexnet_mmep/model.hpp"
#include "args.hpp"

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  std::string out = a.str("_pos0", "");
  if (out.empty()) {
    std::fprintf(stderr, "usage: amm_init out.skt [--image=224 ...]\n");
    return 2;
  }
  amm::Config c;
  c.image = a.i("image", c.image);
  c.num_classes = a.i("classes", c.num_classes);
  c.num_routes = a.i("routes", c.num_routes);
  c.entropy_bound = (float)a.f("bound", c.entropy_bound);
  c.validate();
  auto t = amm::make_param_table(c);
  auto p = amm::init_params(c, t, (uint64_t)a.i("seed", 0));
  amm::save_weights(out, c, t, p);
  std::printf("wrote %s: %zu tensors, %zu parameters, fc6_in %d\n", out.c_str(), t.entries.size(),
              t.total, amm::shapes(c).fc6_in);
  return 0;
}
