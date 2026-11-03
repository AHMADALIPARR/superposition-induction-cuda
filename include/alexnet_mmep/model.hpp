// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// AlexNet -> BURT-IMMA/MMEP bridge: configuration, shapes, parameter table,
// init and weight I/O. Mirrors AlexNetMMEP in docs/source-alexnet-mmep.txt.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "common/param_table.hpp"

namespace amm {

constexpr int kMaxRoutes = 8;  // compile-time cap on routes per router (CUDA kernels)

// Fixed layer geometry from the source model (kernel, stride, padding).
struct ConvGeom { int k, stride, pad; };
constexpr ConvGeom kConv[5] = {{11, 4, 2}, {5, 1, 2}, {3, 1, 1}, {3, 1, 1}, {3, 1, 1}};
constexpr int kPoolK = 3, kPoolS = 2;  // nn.MaxPool2d(3, 2), ceil_mode=False

struct Config {
  int in_ch = 3;
  int image = 224;                         // square input H = W
  int ch[5] = {96, 256, 384, 384, 256};    // conv1..conv5 output channels
  int fc = 4096;                           // fc6 / fc7 width (and router d)
  int num_classes = 1000;
  int num_routes = 4;
  float entropy_bound = 0.20f;
  float act_alpha = 0.01f;                 // SmoothLeaky alpha
  float act_beta = 1.0f;                   // SmoothLeaky beta
  float penalty_weight = 0.1f;             // 0.1 * (ent - bound)^2 in the trainer
  // The source hard-codes nn.Linear(256*6*6, 4096). -1 derives it from the
  // geometry; any other value must equal the derived size (checked).
  int fc6_in = -1;

  void validate() const;
};

// Spatial sizes through the network, all square.
struct Shapes {
  int in;                  // input H = W
  int c[5];                // conv output H = W
  int p1, p2, p5;          // pool outputs after conv1, conv2, conv5
  int fc6_in;              // ch[4] * p5 * p5
};
Shapes shapes(const Config& c);  // throws if any size collapses to < 1

// Parameter names are the PyTorch state_dict keys of AlexNetMMEP (trainable
// parameters only; the activation buffers _alpha/_beta live in Config).
sk::ParamTable make_param_table(const Config& c);

// nn.Conv2d / nn.Linear default init: kaiming_uniform(a=sqrt(5)) on the
// weight, which is U(-1/sqrt(fan_in), 1/sqrt(fan_in)), and the same bound for
// the bias. Deterministic for a given seed.
std::vector<float> init_params(const Config& c, const sk::ParamTable& t, uint64_t seed);

void save_weights(const std::string& path, const Config& c, const sk::ParamTable& t,
                  const std::vector<float>& params);
std::vector<float> load_weights(const std::string& path, Config& c, sk::ParamTable& t);

struct RouterOff { size_t Wr, br, W[kMaxRoutes], b[kMaxRoutes]; };
struct Offsets {
  size_t convW[5], convB[5];
  size_t fc6W, fc6b, fc7W, fc7b, fc8W, fc8b;
  RouterOff r6, r7;
  Offsets(const Config& c, const sk::ParamTable& t);
};

}  // namespace amm
