// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "alexnet_mmep/cuda_model.cuh"

#include <algorithm>
#include <cmath>

#include "alexnet_mmep/kernels.cuh"

namespace amm {

CudaModel::CudaModel(const Config& c, const std::vector<float>& params)
    : cfg_(c), s_(shapes(c)), tab_(make_param_table(c)), off_(c, tab_) {
  if (params.size() != tab_.total) throw std::runtime_error("amm::CudaModel: wrong parameter count");
  SK_CUBLAS(cublasCreate(&h_));
  SK_CUBLAS(cublasSetMathMode(h_, CUBLAS_DEFAULT_MATH));
  d_params_.upload(params);
  d_grad_.resize(tab_.total);
  d_grad_.zero();
}

CudaModel::~CudaModel() {
  if (h_) cublasDestroy(h_);
}

void CudaModel::set_params(const std::vector<float>& p) {
  if (p.size() != tab_.total) throw std::runtime_error("amm::CudaModel: wrong parameter count");
  d_params_.upload(p);
}

void CudaModel::set_tf32(bool on) {
  SK_CUBLAS(cublasSetMathMode(h_, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
}

static int conv_in_ch(const Config& c, int i) { return i == 0 ? c.in_ch : c.ch[i - 1]; }
static int conv_in_hw(const Shapes& s, int i) {
  const int h[5] = {s.in, s.p1, s.p2, s.c[2], s.c[3]};
  return h[i];
}

int CudaModel::chunk(int i) const {
  const size_t CKK = (size_t)conv_in_ch(cfg_, i) * kConv[i].k * kConv[i].k;
  const size_t HW = (size_t)s_.c[i] * s_.c[i];
  const size_t per = (CKK + cfg_.ch[i]) * HW * sizeof(float);
  size_t nb = std::max<size_t>(1, ws_bytes_ / per);
  return (int)std::min<size_t>(nb, (size_t)N_);
}

void CudaModel::ensure(int N) {
  if (N == N_) return;
  N_ = N;
  const int fc = cfg_.fc, K = cfg_.num_routes;
  size_t colmax = 0, zwmax = 0;
  for (int i = 0; i < 5; ++i) {
    const size_t n = (size_t)N * cfg_.ch[i] * s_.c[i] * s_.c[i];
    z_[i].resize(n);
    a_[i].resize(n);
    g_a_[i].resize(n);
    const int nb = chunk(i);
    const size_t HW = (size_t)s_.c[i] * s_.c[i];
    colmax = std::max(colmax, (size_t)conv_in_ch(cfg_, i) * kConv[i].k * kConv[i].k * nb * HW);
    zwmax = std::max(zwmax, (size_t)cfg_.ch[i] * nb * HW);
  }
  col_.resize(colmax);
  zw_.resize(zwmax);
  p1_.resize((size_t)N * cfg_.ch[0] * s_.p1 * s_.p1);
  am1_.resize(p1_.n);
  p2_.resize((size_t)N * cfg_.ch[1] * s_.p2 * s_.p2);
  am2_.resize(p2_.n);
  p5_.resize((size_t)N * s_.fc6_in);
  am5_.resize(p5_.n);
  g_pool_.resize(std::max(p1_.n, p2_.n));
  g_p5_.resize(p5_.n);
  for (auto* b : {&z6_, &a6_, &z7_, &a7_, &g_fc_a_, &g_fc_b_}) b->resize((size_t)N * fc);
  for (Router* r : {&r6_, &r7_}) {
    r->logits.resize((size_t)N * K);
    r->alpha.resize((size_t)N * K);
    r->dalpha.resize((size_t)N * K);
    r->dl.resize((size_t)N * K);
    r->tau.resize(N);
    r->ent.resize(N);
    r->viol.resize(N);
    r->iters.resize(1);
    r->scratch.resize(2 * (size_t)N);
    r->tz.resize((size_t)K * N * fc);
    r->out.resize((size_t)N * fc);
    r->dtz.resize((size_t)N * fc);
  }
  logits_.resize((size_t)N * cfg_.num_classes);
  dlogits_.resize(logits_.n);
  loss_rows_.resize(N);
  pred_.resize(N);
}

// Z = conv(in) + b, through wide im2col + one GEMM per chunk of images.
void CudaModel::conv_fwd(int i, const float* in, int Cin, int Hin, float* z) {
  const ConvGeom g = kConv[i];
  const int O = cfg_.ch[i], Ho = s_.c[i], HW = Ho * Ho, CKK = Cin * g.k * g.k;
  const int nbmax = chunk(i);
  const float* W = d_params_.p + off_.convW[i];
  const float* b = d_params_.p + off_.convB[i];
  for (int n0 = 0; n0 < N_; n0 += nbmax) {
    const int nb = std::min(nbmax, N_ - n0);
    const int Wd = nb * HW;
    kern::im2col_wide(in + (size_t)n0 * Cin * Hin * Hin, nb, Cin, Hin, g.k, g.stride, g.pad, Ho, col_.p);
    sk::gemm_rm(h_, false, false, O, Wd, CKK, 1.f, W, CKK, col_.p, Wd, 0.f, zw_.p, Wd);
    kern::unwide_bias(zw_.p, b, nb, O, HW, z + (size_t)n0 * O * HW);
  }
}

// dW += dZ col^T, db += rowsum dZ, din = col2im(W^T dZ) (skipped when din is null).
void CudaModel::conv_bwd(int i, const float* in, int Cin, int Hin, const float* dz, float* din) {
  const ConvGeom g = kConv[i];
  const int O = cfg_.ch[i], Ho = s_.c[i], HW = Ho * Ho, CKK = Cin * g.k * g.k;
  const int nbmax = chunk(i);
  const float* W = d_params_.p + off_.convW[i];
  float* dW = d_grad_.p + off_.convW[i];
  float* db = d_grad_.p + off_.convB[i];
  for (int n0 = 0; n0 < N_; n0 += nbmax) {
    const int nb = std::min(nbmax, N_ - n0);
    const int Wd = nb * HW;
    kern::im2col_wide(in + (size_t)n0 * Cin * Hin * Hin, nb, Cin, Hin, g.k, g.stride, g.pad, Ho, col_.p);
    kern::to_wide(dz + (size_t)n0 * O * HW, nb, O, HW, zw_.p);
    sk::gemm_rm(h_, false, true, O, CKK, Wd, 1.f, zw_.p, Wd, col_.p, Wd, 1.f, dW, CKK);
    kern::row_sum_add(zw_.p, O, Wd, db);
    if (din) {
      sk::gemm_rm(h_, true, false, CKK, Wd, O, 1.f, W, CKK, zw_.p, Wd, 0.f, col_.p, Wd);
      kern::col2im_wide(col_.p, nb, Cin, Hin, g.k, g.stride, g.pad, Ho, din + (size_t)n0 * Cin * Hin * Hin);
    }
  }
}

void CudaModel::router_fwd(Router& r, const RouterOff& o, const float* x) {
  const int N = N_, d = cfg_.fc, K = cfg_.num_routes;
  const float* P = d_params_.p;
  sk::linear_fwd(h_, x, P + o.Wr, P + o.br, r.logits.p, N, d, K);
  kern::constrained_softmax(r.logits.p, N, K, cfg_.entropy_bound, r.alpha.p, r.tau.p, r.viol.p,
                            r.ent.p, r.iters.p, r.scratch.p, nullptr, nullptr);
  for (int k = 0; k < K; ++k)
    sk::linear_fwd(h_, x, P + o.W[k], P + o.b[k], r.tz.p + (size_t)k * N * d, N, d, d);
  kern::router_combine(r.alpha.p, r.tz.p, N, d, K, r.out.p);
}

void CudaModel::router_bwd(Router& r, const RouterOff& o, const float* x, const float* dout,
                           const float* dent, float* dx) {
  const int N = N_, d = cfg_.fc, K = cfg_.num_routes;
  const float* P = d_params_.p;
  float* G = d_grad_.p;
  kern::router_dalpha(dout, r.tz.p, r.alpha.p, dent, N, d, K, r.dalpha.p);
  for (int k = 0; k < K; ++k) {
    kern::scale_rows(dout, r.alpha.p, k, K, N, d, r.dtz.p);
    sk::linear_bwd(h_, x, P + o.W[k], r.dtz.p, G + o.W[k], G + o.b[k], dx, N, d, d, k > 0);
  }
  kern::constrained_softmax_bwd(r.alpha.p, r.dalpha.p, r.tau.p, N, K, r.dl.p);
  sk::linear_bwd(h_, x, P + o.Wr, r.dl.p, G + o.Wr, G + o.br, dx, N, d, K, true);
}

void CudaModel::forward(const float* x, int N) {
  ensure(N);
  x_ = x;
  const float al = cfg_.act_alpha, be = cfg_.act_beta;
  const float* P = d_params_.p;
  // conv1 -> act -> pool
  conv_fwd(0, x, cfg_.in_ch, s_.in, z_[0].p);
  kern::smooth_leaky_fwd(z_[0].p, a_[0].p, z_[0].n, al, be);
  kern::maxpool_fwd(a_[0].p, N * cfg_.ch[0], s_.c[0], s_.p1, p1_.p, am1_.p);
  // conv2 -> act -> pool
  conv_fwd(1, p1_.p, cfg_.ch[0], s_.p1, z_[1].p);
  kern::smooth_leaky_fwd(z_[1].p, a_[1].p, z_[1].n, al, be);
  kern::maxpool_fwd(a_[1].p, N * cfg_.ch[1], s_.c[1], s_.p2, p2_.p, am2_.p);
  // conv3..5
  conv_fwd(2, p2_.p, cfg_.ch[1], s_.p2, z_[2].p);
  kern::smooth_leaky_fwd(z_[2].p, a_[2].p, z_[2].n, al, be);
  conv_fwd(3, a_[2].p, cfg_.ch[2], s_.c[2], z_[3].p);
  kern::smooth_leaky_fwd(z_[3].p, a_[3].p, z_[3].n, al, be);
  conv_fwd(4, a_[3].p, cfg_.ch[3], s_.c[3], z_[4].p);
  kern::smooth_leaky_fwd(z_[4].p, a_[4].p, z_[4].n, al, be);
  kern::maxpool_fwd(a_[4].p, N * cfg_.ch[4], s_.c[4], s_.p5, p5_.p, am5_.p);
  // fc6 -> act -> route6 -> fc7 -> act -> route7 -> fc8
  sk::linear_fwd(h_, p5_.p, P + off_.fc6W, P + off_.fc6b, z6_.p, N, s_.fc6_in, cfg_.fc);
  kern::smooth_leaky_fwd(z6_.p, a6_.p, z6_.n, al, be);
  router_fwd(r6_, off_.r6, a6_.p);
  sk::linear_fwd(h_, r6_.out.p, P + off_.fc7W, P + off_.fc7b, z7_.p, N, cfg_.fc, cfg_.fc);
  kern::smooth_leaky_fwd(z7_.p, a7_.p, z7_.n, al, be);
  router_fwd(r7_, off_.r7, a7_.p);
  sk::linear_fwd(h_, r7_.out.p, P + off_.fc8W, P + off_.fc8b, logits_.p, N, cfg_.fc, cfg_.num_classes);
  // aux entropies (batch means)
  auto e6 = r6_.ent.download(), e7 = r7_.ent.download();
  double s6 = 0, s7 = 0;
  for (int n = 0; n < N; ++n) { s6 += e6[n]; s7 += e7[n]; }
  ent6_ = s6 / N;
  ent7_ = s7 / N;
}

void CudaModel::backward(const float* dlogits, const float* dent6, const float* dent7, float* dx) {
  if (!x_) throw std::runtime_error("amm::CudaModel::backward before forward");
  const int N = N_, fc = cfg_.fc;
  const float al = cfg_.act_alpha, be = cfg_.act_beta;
  const float* P = d_params_.p;
  float* G = d_grad_.p;
  d_grad_.zero();
  sk::linear_bwd(h_, r7_.out.p, P + off_.fc8W, dlogits, G + off_.fc8W, G + off_.fc8b, g_fc_a_.p, N,
                 fc, cfg_.num_classes, false);
  router_bwd(r7_, off_.r7, a7_.p, g_fc_a_.p, dent7, g_fc_b_.p);
  kern::smooth_leaky_bwd(z7_.p, g_fc_b_.p, z7_.n, al, be);
  sk::linear_bwd(h_, r6_.out.p, P + off_.fc7W, g_fc_b_.p, G + off_.fc7W, G + off_.fc7b, g_fc_a_.p,
                 N, fc, fc, false);
  router_bwd(r6_, off_.r6, a6_.p, g_fc_a_.p, dent6, g_fc_b_.p);
  kern::smooth_leaky_bwd(z6_.p, g_fc_b_.p, z6_.n, al, be);
  sk::linear_bwd(h_, p5_.p, P + off_.fc6W, g_fc_b_.p, G + off_.fc6W, G + off_.fc6b, g_p5_.p, N,
                 s_.fc6_in, fc, false);
  kern::maxpool_bwd(g_p5_.p, am5_.p, N * cfg_.ch[4], s_.c[4], s_.p5, g_a_[4].p);
  const float* in[5] = {x_, p1_.p, p2_.p, a_[2].p, a_[3].p};
  for (int i = 4; i >= 0; --i) {
    kern::smooth_leaky_bwd(z_[i].p, g_a_[i].p, z_[i].n, al, be);  // g_a_[i] now dZ
    float* din = (i == 0) ? dx : (i >= 3 ? g_a_[i - 1].p : g_pool_.p);
    conv_bwd(i, in[i], conv_in_ch(cfg_, i), conv_in_hw(s_, i), g_a_[i].p, din);
    if (i == 2) kern::maxpool_bwd(g_pool_.p, am2_.p, N * cfg_.ch[1], s_.c[1], s_.p2, g_a_[1].p);
    if (i == 1) kern::maxpool_bwd(g_pool_.p, am1_.p, N * cfg_.ch[0], s_.c[0], s_.p1, g_a_[0].p);
  }
}

StepStats CudaModel::loss_and_grad(const float* x, const int* labels, int N) {
  forward(x, N);
  const int C = cfg_.num_classes;
  kern::cross_entropy(logits_.p, labels, N, C, loss_rows_.p, dlogits_.p, pred_.p);
  auto lr = loss_rows_.download();
  auto pr = pred_.download();
  std::vector<int> lab(N);
  SK_CUDA(cudaMemcpy(lab.data(), labels, N * sizeof(int), cudaMemcpyDeviceToHost));
  StepStats st;
  double ce = 0;
  int correct = 0;
  for (int n = 0; n < N; ++n) { ce += lr[n]; correct += pr[n] == lab[n]; }
  st.ce = ce / N;
  st.accuracy = (double)correct / N;
  st.ent6 = ent6_;
  st.ent7 = ent7_;
  const double b = cfg_.entropy_bound, w = cfg_.penalty_weight;
  st.loss = st.ce;
  std::vector<float> d6(N, 0.f), d7(N, 0.f);
  st.pen6 = (float)ent6_ > (float)b;
  st.pen7 = (float)ent7_ > (float)b;
  if (st.pen6) {
    st.loss += w * (ent6_ - b) * (ent6_ - b);
    std::fill(d6.begin(), d6.end(), (float)(w * 2 * (ent6_ - b) / N));
  }
  if (st.pen7) {
    st.loss += w * (ent7_ - b) * (ent7_ - b);
    std::fill(d7.begin(), d7.end(), (float)(w * 2 * (ent7_ - b) / N));
  }
  dent6_.upload(d6);
  dent7_.upload(d7);
  backward(dlogits_.p, dent6_.p, dent7_.p, nullptr);
  return st;
}

StepStats CudaModel::train_step_sgd(const float* x, const int* labels, int N, const sk::SgdConfig& sc) {
  StepStats st = loss_and_grad(x, labels, N);
  if (d_mom_.n != tab_.total) {
    d_mom_.resize(tab_.total);
    d_mom_.zero();
  }
  ++step_;
  sk::sgd_step_device(sc, step_, d_params_.p, d_grad_.p, d_mom_.p, nullptr, tab_.total);
  return st;
}

std::vector<int> CudaModel::pool_argmax(int which) const {
  if (which == 1) return am1_.download();
  if (which == 2) return am2_.download();
  if (which == 5) return am5_.download();
  throw std::runtime_error("pool_argmax: which must be 1, 2 or 5");
}
std::vector<int> CudaModel::viol(int router) const {
  return (router == 6 ? r6_ : r7_).viol.download();
}
std::vector<float> CudaModel::tau(int router) const {
  return (router == 6 ? r6_ : r7_).tau.download();
}
int CudaModel::bisect_iters(int router) const {
  return (router == 6 ? r6_ : r7_).iters.download()[0];
}

}  // namespace amm
