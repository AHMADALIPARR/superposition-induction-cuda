// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// CUDA implementation of SuperpositionedBURTIMMA: forward, backward over all
// parameters, and AdamW / SGD train steps. All tensors are fp32 on device.
#pragma once
#include <memory>
#include <vector>

#include "common/cuda_util.cuh"
#include "sid/model.hpp"

namespace sid {

struct GpuFreeze {
  int iterations = -1;    // >0: run exactly this many refinement iterations
  std::vector<int> mask;  // non-empty [K*K]: use this interference mask
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

  // x: device [B,S,d]. With train = true every iteration's activations are kept
  // for backward(), and x must stay valid until then. Output in device_y().
  void forward(const float* x, int B, int S, bool train, const GpuFreeze* fz = nullptr);
  const float* device_y() const;

  // Gradients of L = <dy, y> + orth_weight * orth_loss w.r.t. all parameters
  // (overwrites device_grads()) and optionally the input x (dx, device).
  void backward(const float* dy, float orth_weight, float* dx = nullptr);

  // loss = mean((y - target)^2) + orthogonality_reg * orth; returns the loss
  // before the update. x and target are device pointers.
  float train_step_adamw(const float* x, const float* target, int B, int S,
                         const sk::AdamWConfig& oc);
  float train_step_sgd(const float* x, const float* target, int B, int S,
                       const sk::SgdConfig& oc);

  // Results of the last forward().
  int iterations() const { return iterations_; }
  double final_delta() const { return final_delta_; }
  const std::vector<double>& deltas() const { return deltas_; }
  std::vector<int> mask() const;
  float orth_loss() const;
  std::vector<float> gamma() const;

  // Use TF32 tensor cores in cuBLAS (off by default for exact fp32).
  void set_tf32(bool on);

 private:
  struct Iter;
  float loss_and_grad(const float* x, const float* target, int B, int S);
  void ensure(int B, int S, bool train);

  Config cfg_;
  sk::ParamTable tab_;
  Offsets off_;
  cublasHandle_t h_ = nullptr;
  sk::DBuf<float> d_params_, d_grad_, d_m_, d_v_, d_buf_;
  sk::DBuf<unsigned char> d_mask_grad_;
  int step_ = 0;
  // basis
  sk::DBuf<float> norms_, vn_, G_, orth_, dG_;
  sk::DBuf<int> mask_, mask_override_;
  // activations
  int B_ = 0, S_ = 0, nsets_ = 0;
  bool train_ = false;
  std::vector<std::unique_ptr<Iter>> its_;
  std::vector<sk::DBuf<float>> outs_;
  std::vector<const float*> xin_;
  sk::DBuf<float> y_;
  sk::DBuf<double> norm2_;
  // backward scratch
  sk::DBuf<float> dout_, dz_, dsp_, dh_, dO_, dOall_, dA_, dVt_, dQt_, dKi_, dKt_, dQ_, dKp_, dV_,
      dxin_, dalpha_, dy_;
  int iterations_ = 0;
  double final_delta_ = 0;
  std::vector<double> deltas_;
};

}  // namespace sid
