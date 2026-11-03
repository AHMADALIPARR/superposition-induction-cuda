// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Superpositioned Induction Daemon: configuration, parameter table, init and I/O.
// Mirrors SuperpositionConfig / SuperpositionedBURTIMMA in docs/source-model.txt.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "common/param_table.hpp"
#include "common/tensor_file.hpp"

namespace sid {

constexpr int kMaxPaths = 8;  // compile-time cap on num_paths for the CUDA kernels

struct Config {
  int d_model = 768;
  int num_paths = 4;     // K
  int num_heads = 8;     // H
  int head_dim = 64;     // d_h
  float interference_threshold = 0.1f;
  float collapse_temperature = 1.0f;  // tau
  float orthogonality_reg = 0.01f;    // weight of orth loss in the train step
  int max_iterations = 5;
  float convergence_threshold = 1e-4f;
  float ln_eps = 1e-5f;
  float normalize_eps = 1e-12f;

  int E() const { return num_heads * head_dim; }
  void validate() const;
};

// Parameter names are the PyTorch state_dict keys of SuperpositionedBURTIMMA.
sk::ParamTable make_param_table(const Config& c);

// Initialisation mirroring the PyTorch constructors (orthogonal path_basis,
// N(0, 0.02) W_induction, nn.Linear's U(-1/sqrt(in), 1/sqrt(in)), LN = (1, 0),
// path_confidence = 1/K). Deterministic for a given seed.
std::vector<float> init_params(const Config& c, const sk::ParamTable& t, uint64_t seed);

// Weight file I/O (SKT1 format, docs/weights-format.md).
void save_weights(const std::string& path, const Config& c, const sk::ParamTable& t,
                  const std::vector<float>& params);
// Reads config.* records (if present) into c, then the parameters.
std::vector<float> load_weights(const std::string& path, Config& c, sk::ParamTable& t);

// Offsets of every tensor, resolved once.
struct Offsets {
  size_t path_basis, W_induction, Wq, bq, Wk, bk, Wv, bv, Wo, bo, path_confidence, Wc, bc,
      ln_g, ln_b, Wp, bp;
  explicit Offsets(const sk::ParamTable& t);
};

}  // namespace sid
