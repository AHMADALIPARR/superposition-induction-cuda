// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Write freshly initialised superposition-model weights.
//   sid_init out.skt [--d_model=768] [--paths=4] [--heads=8] [--head_dim=64]
//                    [--threshold=0.1] [--temperature=1] [--max_iter=5] [--seed=0]
#include <cstdio>

#include "args.hpp"
#include "sid/model.hpp"

int main(int argc, char** argv) {
  skapp::Args a(argc, argv);
  std::string out = a.str("_pos0", "");
  if (out.empty()) {
    std::fprintf(stderr, "usage: sid_init out.skt [--d_model=N ...]\n");
    return 2;
  }
  sid::Config c;
  c.d_model = a.i("d_model", c.d_model);
  c.num_paths = a.i("paths", c.num_paths);
  c.num_heads = a.i("heads", c.num_heads);
  c.head_dim = a.i("head_dim", c.head_dim);
  c.interference_threshold = (float)a.f("threshold", c.interference_threshold);
  c.collapse_temperature = (float)a.f("temperature", c.collapse_temperature);
  c.max_iterations = a.i("max_iter", c.max_iterations);
  c.validate();
  auto t = sid::make_param_table(c);
  auto p = sid::init_params(c, t, (uint64_t)a.i("seed", 0));
  sid::save_weights(out, c, t, p);
  std::printf("wrote %s: %zu tensors, %zu parameters\n", out.c_str(), t.entries.size(), t.total);
  return 0;
}
