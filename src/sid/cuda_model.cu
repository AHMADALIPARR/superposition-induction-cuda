// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "sid/cuda_model.cuh"

#include <cmath>

#include "sid/kernels.cuh"

namespace sid {

using sk::DBuf;
using sk::gemm_rm;
using sk::gemm_rm_batched;

struct CudaModel::Iter {
  DBuf<float> alpha, Q, Kp, V, Qt, Kt, Vt, Ki, A, Oall, O, h, Y, sp, t, xhat, rstd;
};

CudaModel::CudaModel(const Config& c, const std::vector<float>& params)
    : cfg_(c), tab_(make_param_table(c)), off_(tab_) {
  SK_CUBLAS(cublasCreate(&h_));
  SK_CUBLAS(cublasSetMathMode(h_, CUBLAS_DEFAULT_MATH));
  set_params(params);
  d_grad_.resize(tab_.total);
  d_grad_.zero();
  auto gm = sk::grad_mask(tab_);
  d_mask_grad_.upload(gm);
  const int K = c.num_paths, d = c.d_model;
  norms_.resize(K);
  vn_.resize((size_t)K * d);
  G_.resize((size_t)K * K);
  dG_.resize((size_t)K * K);
  orth_.resize(1);
  mask_.resize((size_t)K * K);
  mask_override_.resize((size_t)K * K);
  norm2_.resize(2);
}

CudaModel::~CudaModel() {
  if (h_) cublasDestroy(h_);
}

void CudaModel::set_params(const std::vector<float>& p) {
  if (p.size() != tab_.total) throw std::runtime_error("CudaModel: parameter count mismatch");
  d_params_.upload(p);
}

void CudaModel::set_tf32(bool on) {
  SK_CUBLAS(cublasSetMathMode(h_, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
}

const float* CudaModel::device_y() const { return y_.p; }

std::vector<int> CudaModel::mask() const { return mask_.download(); }
float CudaModel::orth_loss() const { return orth_.download()[0]; }
std::vector<float> CudaModel::gamma() const { return G_.download(); }

void CudaModel::ensure(int B, int S, bool train) {
  const int nsets = train ? cfg_.max_iterations : 1;
  if (B == B_ && S == S_ && nsets == nsets_) return;
  B_ = B;
  S_ = S;
  nsets_ = nsets;
  const size_t d = cfg_.d_model, K = cfg_.num_paths, H = cfg_.num_heads, dh = cfg_.head_dim,
               E = cfg_.E(), BS = (size_t)B * S;
  its_.clear();
  for (int i = 0; i < nsets; ++i) {
    auto it = std::make_unique<Iter>();
    it->alpha.resize(B * K);
    it->Q.resize(BS * E);
    it->Kp.resize(BS * E);
    it->V.resize(BS * E);
    it->Qt.resize(BS * E);
    it->Kt.resize(BS * E);
    it->Vt.resize(BS * E);
    it->Ki.resize(K * BS * E);
    it->A.resize((size_t)B * H * S * K * S);
    it->Oall.resize(K * BS * E);
    it->O.resize(K * BS * E);
    it->h.resize(K * BS * d);
    it->Y.resize(K * BS * d);
    it->sp.resize(BS * d);
    it->t.resize(BS * d);
    it->xhat.resize(BS * d);
    it->rstd.resize(BS);
    its_.push_back(std::move(it));
  }
  outs_.clear();
  const int nouts = train ? cfg_.max_iterations : 2;
  for (int i = 0; i < nouts; ++i) outs_.emplace_back(BS * d);
  y_.resize(BS * d);
  if (train) {
    dout_.resize(BS * d);
    dz_.resize(BS * d);
    dsp_.resize(BS * d);
    dh_.resize(K * BS * d);
    dO_.resize(K * BS * E);
    dOall_.resize(K * BS * E);
    dA_.resize((size_t)B * H * S * K * S);
    dVt_.resize(BS * E);
    dQt_.resize(BS * E);
    dKi_.resize(K * BS * E);
    dKt_.resize(BS * E);
    dQ_.resize(BS * E);
    dKp_.resize(BS * E);
    dV_.resize(BS * E);
    dxin_.resize(BS * d);
    dalpha_.resize(B * K);
    dy_.resize(BS * d);
  }
  (void)dh;
}

void CudaModel::forward(const float* x, int B, int S, bool train, const GpuFreeze* fz) {
  ensure(B, S, train);
  train_ = train;
  const Config& c = cfg_;
  const int d = c.d_model, K = c.num_paths, H = c.num_heads, dh = c.head_dim, E = c.E();
  const int BS = B * S;
  const float* p = d_params_.p;
  const float* P = p + off_.path_basis;
  const int* mo = nullptr;
  if (fz && !fz->mask.empty()) {
    mask_override_.upload(fz->mask);
    mo = mask_override_.p;
  }
  kern::basis_fwd(P, K, d, c.normalize_eps, c.interference_threshold, mo, norms_.p, vn_.p, G_.p,
                  mask_.p, orth_.p);
  const bool fixed = fz && fz->iterations > 0;
  const int maxit = fixed ? fz->iterations : c.max_iterations;
  if (train && maxit > (int)its_.size()) throw std::runtime_error("frozen iterations > max_iterations");
  xin_.assign(maxit, nullptr);
  deltas_.clear();
  const float scale = 1.0f / std::sqrt((float)dh);
  const int d5_bshd[5] = {B, S, H, dh, 1};
  const int perm_swap[5] = {0, 2, 1, 3, 4};
  const float* cur = x;
  const float* last_out = nullptr;
  for (int i = 0; i < maxit; ++i) {
    Iter& it = *its_[i % its_.size()];
    float* out = outs_[i % outs_.size()].p;
    xin_[i] = cur;
    kern::alpha_fwd(cur, P, B, S, d, K, c.collapse_temperature, it.alpha.p);
    sk::linear_fwd(h_, cur, p + off_.Wq, p + off_.bq, it.Q.p, BS, d, E);
    sk::linear_fwd(h_, cur, p + off_.Wk, p + off_.bk, it.Kp.p, BS, d, E);
    sk::linear_fwd(h_, cur, p + off_.Wv, p + off_.bv, it.V.p, BS, d, E);
    sk::permute5(it.Q.p, it.Qt.p, d5_bshd, perm_swap);   // [B,H,S,dh]
    sk::permute5(it.Kp.p, it.Kt.p, d5_bshd, perm_swap);
    sk::permute5(it.V.p, it.Vt.p, d5_bshd, perm_swap);
    // K_induced[b,h,k] = Kt[b,h] @ W_induction[k,h]   -> Ki [B,H,K,S,dh]
    for (int b = 0; b < B; ++b)
      for (int k = 0; k < K; ++k)
        gemm_rm_batched(h_, false, false, S, dh, dh, 1.f, it.Kt.p + (size_t)b * H * S * dh, dh,
                        (long long)S * dh, p + off_.W_induction + (size_t)k * H * dh * dh, dh,
                        (long long)dh * dh, 0.f,
                        it.Ki.p + ((size_t)b * H * K + k) * S * dh, dh, (long long)K * S * dh, H);
    // scores for all K paths at once: [S, K*S] per (b,h)  -> A [B,H,S,K,S]
    gemm_rm_batched(h_, false, true, S, K * S, dh, scale, it.Qt.p, dh, (long long)S * dh, it.Ki.p,
                    dh, (long long)K * S * dh, 0.f, it.A.p, K * S, (long long)S * K * S, B * H);
    sk::softmax_rows_fwd(it.A.p, it.A.p, (long long)B * H * S * K, S);
    // O_all [S*K, dh] = A [S*K, S] @ Vt [S, dh]   -> [B,H,S,K,dh]
    gemm_rm_batched(h_, false, false, S * K, dh, S, 1.f, it.A.p, S, (long long)S * K * S, it.Vt.p,
                    dh, (long long)S * dh, 0.f, it.Oall.p, dh, (long long)S * K * dh, B * H);
    {
      const int dd[5] = {B, H, S, K, dh};
      const int pr[5] = {3, 0, 2, 1, 4};  // -> [K,B,S,H,dh]
      sk::permute5(it.Oall.p, it.O.p, dd, pr);
    }
    sk::linear_fwd(h_, it.O.p, p + off_.Wo, p + off_.bo, it.h.p, K * BS, E, d);
    kern::interference_fwd(it.h.p, it.alpha.p, G_.p, mask_.p, K, B, (size_t)S * d, it.Y.p, it.sp.p);
    sk::linear_fwd(h_, it.sp.p, p + off_.Wc, p + off_.bc, it.t.p, BS, d, d);
    kern::tanh_ln_fwd(it.t.p, p + off_.ln_g, p + off_.ln_b, BS, d, c.ln_eps, it.xhat.p,
                      it.rstd.p, out);
    sk::sq_diff_norms(out, cur, (size_t)BS * d, norm2_.p);
    double nn[2];
    SK_CUDA(cudaMemcpy(nn, norm2_.p, sizeof(nn), cudaMemcpyDeviceToHost));
    const double delta = std::sqrt(nn[0]) / (std::sqrt(nn[1]) + 1e-8);
    deltas_.push_back(delta);
    final_delta_ = delta;
    iterations_ = i + 1;
    last_out = out;
    if (!fixed && delta < c.convergence_threshold) break;
    cur = out;
  }
  sk::linear_fwd(h_, last_out, p + off_.Wp, p + off_.bp, y_.p, BS, d, d);
}

void CudaModel::backward(const float* dy, float orth_weight, float* dx) {
  if (!train_) throw std::runtime_error("CudaModel::backward needs forward(..., train=true)");
  const Config& c = cfg_;
  const int B = B_, S = S_, d = c.d_model, K = c.num_paths, H = c.num_heads, dh = c.head_dim,
            E = c.E(), BS = B * S;
  const float* p = d_params_.p;
  float* g = d_grad_.p;
  const float* P = p + off_.path_basis;
  const float scale = 1.0f / std::sqrt((float)dh);
  d_grad_.zero();
  dG_.zero();
  const int d5_bshd[5] = {B, S, H, dh, 1};
  const int d5_bhsd[5] = {B, H, S, dh, 1};
  const int perm_swap[5] = {0, 2, 1, 3, 4};
  const float* last_out = outs_[(iterations_ - 1) % outs_.size()].p;
  sk::linear_bwd(h_, last_out, p + off_.Wp, dy, g + off_.Wp, g + off_.bp, dout_.p, BS, d, d, false);
  for (int i = iterations_ - 1; i >= 0; --i) {
    Iter& it = *its_[i];
    const float* xin = xin_[i];
    kern::tanh_ln_bwd(dout_.p, it.t.p, it.xhat.p, it.rstd.p, p + off_.ln_g, BS, d, dz_.p,
                      g + off_.ln_g, g + off_.ln_b);
    sk::linear_bwd(h_, it.sp.p, p + off_.Wc, dz_.p, g + off_.Wc, g + off_.bc, dsp_.p, BS, d, d, false);
    dalpha_.zero();
    kern::interference_bwd(dsp_.p, it.h.p, it.Y.p, it.alpha.p, G_.p, mask_.p, K, B,
                           (size_t)S * d, dh_.p, dalpha_.p, dG_.p);
    sk::linear_bwd(h_, it.O.p, p + off_.Wo, dh_.p, g + off_.Wo, g + off_.bo, dO_.p, K * BS, E, d, false);
    {
      const int dd[5] = {K, B, S, H, dh};
      const int pr[5] = {1, 3, 2, 0, 4};  // -> [B,H,S,K,dh]
      sk::permute5(dO_.p, dOall_.p, dd, pr);
    }
    // dA = dO_all @ Vt^T ; dVt = A^T @ dO_all (sums over the K paths)
    gemm_rm_batched(h_, false, true, S * K, S, dh, 1.f, dOall_.p, dh, (long long)S * K * dh, it.Vt.p,
                    dh, (long long)S * dh, 0.f, dA_.p, S, (long long)S * K * S, B * H);
    gemm_rm_batched(h_, true, false, S, dh, S * K, 1.f, it.A.p, S, (long long)S * K * S, dOall_.p,
                    dh, (long long)S * K * dh, 0.f, dVt_.p, dh, (long long)S * dh, B * H);
    sk::softmax_rows_bwd(it.A.p, dA_.p, dA_.p, (long long)B * H * S * K, S);
    // dQt = scale dSc @ Ki ; dKi = scale dSc^T @ Qt  (dSc viewed as [S, K*S])
    gemm_rm_batched(h_, false, false, S, dh, K * S, scale, dA_.p, K * S, (long long)S * K * S,
                    it.Ki.p, dh, (long long)K * S * dh, 0.f, dQt_.p, dh, (long long)S * dh, B * H);
    gemm_rm_batched(h_, true, false, K * S, dh, S, scale, dA_.p, K * S, (long long)S * K * S, it.Qt.p,
                    dh, (long long)S * dh, 0.f, dKi_.p, dh, (long long)K * S * dh, B * H);
    // K_induced = Kt @ W_induction[k,h]
    for (int b = 0; b < B; ++b)
      for (int k = 0; k < K; ++k) {
        const float* dKi_bk = dKi_.p + ((size_t)b * H * K + k) * S * dh;
        // dKt[b,h] (+)= dKi[b,h,k] @ Wi[k,h]^T
        gemm_rm_batched(h_, false, true, S, dh, dh, 1.f, dKi_bk, dh, (long long)K * S * dh,
                        p + off_.W_induction + (size_t)k * H * dh * dh, dh, (long long)dh * dh,
                        k == 0 ? 0.f : 1.f, dKt_.p + (size_t)b * H * S * dh, dh, (long long)S * dh, H);
        // dWi[k,h] += Kt[b,h]^T @ dKi[b,h,k]
        gemm_rm_batched(h_, true, false, dh, dh, S, 1.f, it.Kt.p + (size_t)b * H * S * dh, dh,
                        (long long)S * dh, dKi_bk, dh, (long long)K * S * dh, 1.f,
                        g + off_.W_induction + (size_t)k * H * dh * dh, dh, (long long)dh * dh, H);
      }
    sk::permute5(dQt_.p, dQ_.p, d5_bhsd, perm_swap);  // -> [B,S,H,dh]
    sk::permute5(dKt_.p, dKp_.p, d5_bhsd, perm_swap);
    sk::permute5(dVt_.p, dV_.p, d5_bhsd, perm_swap);
    sk::linear_bwd(h_, xin, p + off_.Wq, dQ_.p, g + off_.Wq, g + off_.bq, dxin_.p, BS, d, E, false);
    sk::linear_bwd(h_, xin, p + off_.Wk, dKp_.p, g + off_.Wk, g + off_.bk, dxin_.p, BS, d, E, true);
    sk::linear_bwd(h_, xin, p + off_.Wv, dV_.p, g + off_.Wv, g + off_.bv, dxin_.p, BS, d, E, true);
    kern::alpha_bwd(xin, P, it.alpha.p, dalpha_.p, B, S, d, K, c.collapse_temperature, dxin_.p,
                    g + off_.path_basis);
    std::swap(dout_, dxin_);
  }
  (void)d5_bshd;
  if (dx) SK_CUDA(cudaMemcpy(dx, dout_.p, sizeof(float) * (size_t)BS * d, cudaMemcpyDeviceToDevice));
  kern::basis_bwd(P, norms_.p, vn_.p, G_.p, dG_.p, K, d, c.normalize_eps, orth_weight,
                  g + off_.path_basis);
  // path_confidence: unused in forward, gradient stays 0 (skipped by the optimizers).
}

float CudaModel::loss_and_grad(const float* x, const float* target, int B, int S) {
  forward(x, B, S, true);
  const size_t n = (size_t)B * S * cfg_.d_model;
  sk::sq_diff_norms(y_.p, target, n, norm2_.p);
  double nn[2];
  SK_CUDA(cudaMemcpy(nn, norm2_.p, sizeof(nn), cudaMemcpyDeviceToHost));
  kern::mse_grad(y_.p, target, n, dy_.p);
  backward(dy_.p, cfg_.orthogonality_reg, nullptr);
  return (float)(nn[0] / (double)n) + cfg_.orthogonality_reg * orth_loss();
}

float CudaModel::train_step_adamw(const float* x, const float* target, int B, int S,
                                  const sk::AdamWConfig& oc) {
  float loss = loss_and_grad(x, target, B, S);
  if (d_m_.n != tab_.total) {
    d_m_.resize(tab_.total);
    d_m_.zero();
    d_v_.resize(tab_.total);
    d_v_.zero();
  }
  ++step_;
  sk::adamw_step_device(oc, step_, d_params_.p, d_grad_.p, d_m_.p, d_v_.p, d_mask_grad_.p, tab_.total);
  return loss;
}

float CudaModel::train_step_sgd(const float* x, const float* target, int B, int S,
                                const sk::SgdConfig& oc) {
  float loss = loss_and_grad(x, target, B, S);
  if (d_buf_.n != tab_.total) {
    d_buf_.resize(tab_.total);
    d_buf_.zero();
  }
  ++step_;
  sk::sgd_step_device(oc, step_, d_params_.p, d_grad_.p, d_buf_.p, d_mask_grad_.p, tab_.total);
  return loss;
}

}  // namespace sid
