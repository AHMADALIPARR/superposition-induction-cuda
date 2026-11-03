// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CUDA implementation of AlexNetMMEP and AlexNetMMEPTrainer.train_step:
// forward, CE + entropy penalty, backward over every parameter, and
// torch.optim.SGD(momentum, weight_decay). fp32 on device, no cuDNN.
#pragma once
#include <vector>

#include "alexnet_mmep/model.hpp"
#include "common/cuda_util.cuh"

namespace amm {

struct StepStats {
  double loss = 0, ce = 0, ent6 = 0, ent7 = 0, accuracy = 0;
  int pen6 = 0, pen7 = 0;
};

class CudaModel {
 public:
  CudaModel(const Config& c, const std::vector<float>& params);
  ~CudaModel();
  CudaModel(const CudaModel&) = delete;
  CudaModel& operator=(const CudaModel&) = delete;

  const Config& config() const { return cfg_; }
  const sk::ParamTable& table() const { return tab_; }
  void set_params(const std::vector<float>& p);
  std::vector<float> params() const { return d_params_.download(); }
  std::vector<float> grads() const { return d_grad_.download(); }
  float* device_params() { return d_params_.p; }
  float* device_grads() { return d_grad_.p; }

  // x: device [N, in_ch, image, image]; must stay valid until backward().
  // Logits in device_logits() [N, num_classes]. Also computes the batch-mean
  // router entropies (ent6(), ent7()).
  void forward(const float* x, int N);
  const float* device_logits() const { return logits_.p; }

  // Gradients of the loss whose derivatives w.r.t. logits and per-row router
  // entropies are dlogits [N,C], dent6 [N], dent7 [N] (all device). Overwrites
  // device_grads(); dx (device) may be null.
  void backward(const float* dlogits, const float* dent6, const float* dent7, float* dx = nullptr);

  // forward + CrossEntropyLoss + 0.1*(ent - bound)^2 per router when the batch
  // mean exceeds the bound + backward. labels: device int [N].
  StepStats loss_and_grad(const float* x, const int* labels, int N);
  // AlexNetMMEPTrainer.train_step. Returns the stats of the forward pass.
  StepStats train_step_sgd(const float* x, const int* labels, int N, const sk::SgdConfig& sc);

  // Decisions and caches of the last forward (downloaded on request).
  double ent6() const { return ent6_; }
  double ent7() const { return ent7_; }
  std::vector<int> pool_argmax(int which) const;  // which = 1, 2, 5
  std::vector<int> viol(int router) const;        // router = 6, 7
  std::vector<float> tau(int router) const;
  int bisect_iters(int router) const;
  int step() const { return step_; }

  void set_tf32(bool on);
  // Upper bound on the im2col workspace (default 256 MiB).
  void set_workspace_bytes(size_t b) { ws_bytes_ = b; }

 private:
  struct Router {
    sk::DBuf<float> logits, alpha, tau, ent, tz, out, scratch, dalpha, dl, dtz;
    sk::DBuf<int> viol, iters;
  };
  void ensure(int N);
  int chunk(int layer) const;
  void conv_fwd(int i, const float* in, int Cin, int Hin, float* z);
  void conv_bwd(int i, const float* in, int Cin, int Hin, const float* dz, float* din);
  void router_fwd(Router& r, const RouterOff& o, const float* x);
  void router_bwd(Router& r, const RouterOff& o, const float* x, const float* dout,
                  const float* dent, float* dx);

  Config cfg_;
  Shapes s_;
  sk::ParamTable tab_;
  Offsets off_;
  cublasHandle_t h_ = nullptr;
  size_t ws_bytes_ = 256u << 20;
  int N_ = 0, step_ = 0;
  const float* x_ = nullptr;
  double ent6_ = 0, ent7_ = 0;
  sk::DBuf<float> d_params_, d_grad_, d_mom_;
  sk::DBuf<float> z_[5], a_[5], p1_, p2_, p5_, z6_, a6_, z7_, a7_, logits_;
  sk::DBuf<int> am1_, am2_, am5_;
  Router r6_, r7_;
  sk::DBuf<float> col_, zw_;                     // im2col workspace
  sk::DBuf<float> g_a_[5], g_pool_, g_p5_, g_fc_a_, g_fc_b_;  // backward scratch
  sk::DBuf<float> dlogits_, loss_rows_, dent6_, dent7_;
  sk::DBuf<int> pred_;
};

}  // namespace amm
