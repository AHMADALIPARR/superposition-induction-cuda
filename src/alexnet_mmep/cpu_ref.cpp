// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "alexnet_mmep/cpu_ref.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace amm {

static constexpr double kLn2 = 0.6931471805599453;
static constexpr double kSoftplusThreshold = 20.0;  // F.softplus default
static constexpr double kEntEps = 1e-10;            // (p + 1e-10).log()

template <class T>
T smooth_leaky(T x, T a, T b) {
  const T bx = b * x;
  const T sp = bx > T(kSoftplusThreshold) ? bx : std::log1p(std::exp(bx));
  return a * x + ((T(1) - a) / b) * (sp - T(kLn2));
}

template <class T>
T smooth_leaky_grad(T x, T a, T b) {
  const T bx = b * x;
  T s;
  if (bx > T(kSoftplusThreshold)) {
    s = T(1);
  } else {
    const T z = std::exp(bx);
    s = z / (z + T(1));
  }
  return a + ((T(1) - a) / b) * s * b;
}

// ----------------------------------------------------------------- helpers --
template <class T>
static void softmax_row(const T* l, T inv_tau, int K, T* out) {
  T m = -std::numeric_limits<T>::infinity();
  for (int k = 0; k < K; ++k) m = std::max(m, l[k] * inv_tau);
  T s = 0;
  for (int k = 0; k < K; ++k) { out[k] = std::exp(l[k] * inv_tau - m); s += out[k]; }
  for (int k = 0; k < K; ++k) out[k] /= s;
}

template <class T>
static T entropy_row(const T* p, int K) {
  T h = 0;
  for (int k = 0; k < K; ++k) h += p[k] * std::log(p[k] + T(kEntEps));
  return -h;
}

template <class T>
int constrained_softmax(const T* logits, int rows, int K, T bound, T* alpha, T* tau, int* viol,
                        const int* viol_fz, const double* tau_fz) {
  if (viol_fz) {
    for (int r = 0; r < rows; ++r) {
      viol[r] = viol_fz[r];
      tau[r] = viol[r] ? T(tau_fz[r]) : T(1);
      softmax_row(logits + (size_t)r * K, viol[r] ? T(1) / tau[r] : T(1), K, alpha + (size_t)r * K);
    }
    return 0;
  }
  bool any = false;
  for (int r = 0; r < rows; ++r) {
    softmax_row(logits + (size_t)r * K, T(1), K, alpha + (size_t)r * K);
    viol[r] = entropy_row(alpha + (size_t)r * K, K) > bound;
    tau[r] = T(1);
    any = any || viol[r];
  }
  if (!any) return 0;
  std::vector<T> lo(rows, T(0.01)), hi(rows, T(1)), pbuf(K);
  int it = 0;
  for (int i = 0; i < 32; ++i) {
    ++it;
    T width = 0;
    for (int r = 0; r < rows; ++r) {
      const T mid = (lo[r] + hi[r]) / T(2);
      // logits / tau_mid, then softmax
      T m = -std::numeric_limits<T>::infinity();
      const T* l = logits + (size_t)r * K;
      for (int k = 0; k < K; ++k) m = std::max(m, l[k] / mid);
      T s = 0;
      for (int k = 0; k < K; ++k) { pbuf[k] = std::exp(l[k] / mid - m); s += pbuf[k]; }
      for (int k = 0; k < K; ++k) pbuf[k] /= s;
      const T h = entropy_row(pbuf.data(), K);
      if (h > bound) hi[r] = mid; else lo[r] = mid;
      width = std::max(width, hi[r] - lo[r]);
    }
    if (width < T(1e-6)) break;
  }
  for (int r = 0; r < rows; ++r) {
    if (!viol[r]) continue;
    tau[r] = (lo[r] + hi[r]) / T(2);
    // F.softmax(logits / tau_final)
    const T* l = logits + (size_t)r * K;
    T* a = alpha + (size_t)r * K;
    T m = -std::numeric_limits<T>::infinity();
    for (int k = 0; k < K; ++k) m = std::max(m, l[k] / tau[r]);
    T s = 0;
    for (int k = 0; k < K; ++k) { a[k] = std::exp(l[k] / tau[r] - m); s += a[k]; }
    for (int k = 0; k < K; ++k) a[k] /= s;
  }
  return it;
}

template <class T>
static void linear(const T* X, const T* W, const T* b, T* Y, int M, int K, int N) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      T s = b ? b[j] : T(0);
      const T* x = X + (size_t)i * K;
      const T* w = W + (size_t)j * K;
      for (int k = 0; k < K; ++k) s += x[k] * w[k];
      Y[(size_t)i * N + j] = s;
    }
}

// dW += dY^T X, db += colsum dY, dX (+)= dY W
template <class T>
static void linear_bwd(const T* X, const T* W, const T* dY, T* dW, T* db, T* dX, int M, int K,
                       int N, bool acc_dx) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      const T g = dY[(size_t)i * N + j];
      if (db) db[j] += g;
      T* w = dW + (size_t)j * K;
      const T* x = X + (size_t)i * K;
      for (int k = 0; k < K; ++k) w[k] += g * x[k];
    }
  if (!dX) return;
  if (!acc_dx) std::fill(dX, dX + (size_t)M * K, T(0));
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      const T g = dY[(size_t)i * N + j];
      const T* w = W + (size_t)j * K;
      T* dx = dX + (size_t)i * K;
      for (int k = 0; k < K; ++k) dx[k] += g * w[k];
    }
}

template <class T>
static void conv_fwd(const T* x, int N, int C, int H, const T* W, const T* b, int O,
                     const ConvGeom& g, int Ho, T* z) {
  const int k = g.k;
  for (int n = 0; n < N; ++n)
    for (int o = 0; o < O; ++o)
      for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Ho; ++ox) {
          T s = b[o];
          for (int c = 0; c < C; ++c)
            for (int ky = 0; ky < k; ++ky) {
              const int iy = oy * g.stride - g.pad + ky;
              if (iy < 0 || iy >= H) continue;
              for (int kx = 0; kx < k; ++kx) {
                const int ix = ox * g.stride - g.pad + kx;
                if (ix < 0 || ix >= H) continue;
                s += x[(((size_t)n * C + c) * H + iy) * H + ix] *
                     W[(((size_t)o * C + c) * k + ky) * k + kx];
              }
            }
          z[(((size_t)n * O + o) * Ho + oy) * Ho + ox] = s;
        }
}

template <class T>
static void conv_bwd(const T* x, int N, int C, int H, const T* W, int O, const ConvGeom& g,
                     int Ho, const T* dz, T* dW, T* db, T* dx) {
  const int k = g.k;
  if (dx) std::fill(dx, dx + (size_t)N * C * H * H, T(0));
  for (int n = 0; n < N; ++n)
    for (int o = 0; o < O; ++o)
      for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Ho; ++ox) {
          const T gz = dz[(((size_t)n * O + o) * Ho + oy) * Ho + ox];
          db[o] += gz;
          for (int c = 0; c < C; ++c)
            for (int ky = 0; ky < k; ++ky) {
              const int iy = oy * g.stride - g.pad + ky;
              if (iy < 0 || iy >= H) continue;
              for (int kx = 0; kx < k; ++kx) {
                const int ix = ox * g.stride - g.pad + kx;
                if (ix < 0 || ix >= H) continue;
                const size_t xi = (((size_t)n * C + c) * H + iy) * H + ix;
                const size_t wi = (((size_t)o * C + c) * k + ky) * k + kx;
                dW[wi] += gz * x[xi];
                if (dx) dx[xi] += gz * W[wi];
              }
            }
        }
}

// MaxPool2d(3, 2): first maximum in row-major window order (as PyTorch).
template <class T>
static void pool_fwd(const T* x, int NC, int H, int Ho, T* y, int* am, const int* am_fz) {
  for (int q = 0; q < NC; ++q)
    for (int oy = 0; oy < Ho; ++oy)
      for (int ox = 0; ox < Ho; ++ox) {
        const size_t oi = ((size_t)q * Ho + oy) * Ho + ox;
        const T* plane = x + (size_t)q * H * H;
        int best;
        if (am_fz) {
          best = am_fz[oi];
        } else {
          best = (oy * kPoolS) * H + ox * kPoolS;
          T bv = plane[best];
          for (int ky = 0; ky < kPoolK; ++ky)
            for (int kx = 0; kx < kPoolK; ++kx) {
              const int idx = (oy * kPoolS + ky) * H + ox * kPoolS + kx;
              const T v = plane[idx];
              if (v > bv || std::isnan(v)) { bv = v; best = idx; }
            }
        }
        am[oi] = best;
        y[oi] = plane[best];
      }
}

template <class T>
static void pool_bwd(const T* dy, const int* am, int NC, int H, int Ho, T* dx) {
  std::fill(dx, dx + (size_t)NC * H * H, T(0));
  for (int q = 0; q < NC; ++q)
    for (int i = 0; i < Ho * Ho; ++i)
      dx[(size_t)q * H * H + am[(size_t)q * Ho * Ho + i]] += dy[(size_t)q * Ho * Ho + i];
}

template <class T>
static void router_fwd(const Config& c, const RouterOff& o, const T* p, const T* x, int N,
                       RouterCache<T>& r, const std::vector<int>* vf,
                       const std::vector<double>* tf) {
  const int K = c.num_routes, d = c.fc;
  r.logits.assign((size_t)N * K, T(0));
  r.alpha.assign((size_t)N * K, T(0));
  r.ent.assign(N, T(0));
  r.tau.assign(N, T(1));
  r.viol.assign(N, 0);
  linear(x, p + o.Wr, p + o.br, r.logits.data(), N, d, K);
  const bool frozen = vf && !vf->empty();
  r.bisect_iters = constrained_softmax(r.logits.data(), N, K, T(c.entropy_bound), r.alpha.data(),
                                       r.tau.data(), r.viol.data(),
                                       frozen ? vf->data() : nullptr, frozen ? tf->data() : nullptr);
  for (int n = 0; n < N; ++n) r.ent[n] = entropy_row(&r.alpha[(size_t)n * K], K);
  r.tz.assign((size_t)K * N * d, T(0));
  r.out.assign((size_t)N * d, T(0));
  for (int k = 0; k < K; ++k) {
    T* tz = &r.tz[(size_t)k * N * d];
    linear(x, p + o.W[k], p + o.b[k], tz, N, d, d);
    for (int n = 0; n < N; ++n)
      for (int i = 0; i < d; ++i)
        r.out[(size_t)n * d + i] += r.alpha[(size_t)n * K + k] * tz[(size_t)n * d + i];
  }
}

template <class T>
static void act_fwd(const Config& c, const std::vector<T>& z, std::vector<T>& a) {
  a.resize(z.size());
  const T al = T(c.act_alpha), be = T(c.act_beta);
  for (size_t i = 0; i < z.size(); ++i) a[i] = smooth_leaky(z[i], al, be);
}

template <class T>
void cpu_forward(const Config& c, const sk::ParamTable& t, const T* p, const T* x, int N,
                 CpuState<T>& st, const Freeze* fz) {
  c.validate();
  Offsets o(c, t);
  const Shapes s = shapes(c);
  st.N = N;
  st.s = s;
  st.x.assign(x, x + (size_t)N * c.in_ch * s.in * s.in);
  const T* in = st.x.data();
  int C = c.in_ch, H = s.in;
  auto conv = [&](int i, const T* src, int Cin, int Hin) {
    st.z[i].assign((size_t)N * c.ch[i] * s.c[i] * s.c[i], T(0));
    conv_fwd(src, N, Cin, Hin, p + o.convW[i], p + o.convB[i], c.ch[i], kConv[i], s.c[i],
             st.z[i].data());
    act_fwd(c, st.z[i], st.a[i]);
  };
  auto pool = [&](const std::vector<T>& src, int Cc, int Hin, int Ho, std::vector<T>& y,
                  std::vector<int>& am, const std::vector<int>* f) {
    y.assign((size_t)N * Cc * Ho * Ho, T(0));
    am.assign(y.size(), 0);
    pool_fwd(src.data(), N * Cc, Hin, Ho, y.data(), am.data(),
             (f && !f->empty()) ? f->data() : nullptr);
  };
  conv(0, in, C, H);
  pool(st.a[0], c.ch[0], s.c[0], s.p1, st.p1, st.am1, fz ? &fz->am1 : nullptr);
  conv(1, st.p1.data(), c.ch[0], s.p1);
  pool(st.a[1], c.ch[1], s.c[1], s.p2, st.p2, st.am2, fz ? &fz->am2 : nullptr);
  conv(2, st.p2.data(), c.ch[1], s.p2);
  conv(3, st.a[2].data(), c.ch[2], s.c[2]);
  conv(4, st.a[3].data(), c.ch[3], s.c[3]);
  pool(st.a[4], c.ch[4], s.c[4], s.p5, st.p5, st.am5, fz ? &fz->am5 : nullptr);

  st.z6.assign((size_t)N * c.fc, T(0));
  linear(st.p5.data(), p + o.fc6W, p + o.fc6b, st.z6.data(), N, s.fc6_in, c.fc);
  act_fwd(c, st.z6, st.a6);
  router_fwd(c, o.r6, p, st.a6.data(), N, st.r6, fz ? &fz->viol6 : nullptr, fz ? &fz->tau6 : nullptr);
  st.z7.assign((size_t)N * c.fc, T(0));
  linear(st.r6.out.data(), p + o.fc7W, p + o.fc7b, st.z7.data(), N, c.fc, c.fc);
  act_fwd(c, st.z7, st.a7);
  router_fwd(c, o.r7, p, st.a7.data(), N, st.r7, fz ? &fz->viol7 : nullptr, fz ? &fz->tau7 : nullptr);
  st.logits.assign((size_t)N * c.num_classes, T(0));
  linear(st.r7.out.data(), p + o.fc8W, p + o.fc8b, st.logits.data(), N, c.fc, c.num_classes);
  T e6 = 0, e7 = 0;
  for (int n = 0; n < N; ++n) { e6 += st.r6.ent[n]; e7 += st.r7.ent[n]; }
  st.ent6 = e6 / T(N);
  st.ent7 = e7 / T(N);
}

template <class T>
LossInfo cpu_loss(const Config& c, const CpuState<T>& st, const int* labels,
                  std::vector<T>& dlogits, std::vector<T>& dent6, std::vector<T>& dent7,
                  const Freeze* fz) {
  const int N = st.N, C = c.num_classes;
  LossInfo li;
  dlogits.assign((size_t)N * C, T(0));
  T ce = 0;
  int correct = 0;
  for (int n = 0; n < N; ++n) {
    const T* l = &st.logits[(size_t)n * C];
    T m = l[0];
    int am = 0;
    for (int j = 1; j < C; ++j)
      if (l[j] > m) { m = l[j]; am = j; }
    correct += (am == labels[n]);
    T s = 0;
    for (int j = 0; j < C; ++j) s += std::exp(l[j] - m);
    const T lse = m + std::log(s);
    ce += lse - l[labels[n]];
    for (int j = 0; j < C; ++j)
      dlogits[(size_t)n * C + j] = (std::exp(l[j] - lse) - (j == labels[n] ? T(1) : T(0))) / T(N);
  }
  ce /= T(N);
  T loss = ce;
  const T b = T(c.entropy_bound), w = T(c.penalty_weight);
  dent6.assign(N, T(0));
  dent7.assign(N, T(0));
  li.pen6 = (fz && fz->pen6 >= 0) ? fz->pen6 : (st.ent6 > b);
  li.pen7 = (fz && fz->pen7 >= 0) ? fz->pen7 : (st.ent7 > b);
  if (li.pen6) {
    loss += w * (st.ent6 - b) * (st.ent6 - b);
    for (int n = 0; n < N; ++n) dent6[n] = w * T(2) * (st.ent6 - b) / T(N);
  }
  if (li.pen7) {
    loss += w * (st.ent7 - b) * (st.ent7 - b);
    for (int n = 0; n < N; ++n) dent7[n] = w * T(2) * (st.ent7 - b) / T(N);
  }
  li.ce = (double)ce;
  li.loss = (double)loss;
  li.ent6 = (double)st.ent6;
  li.ent7 = (double)st.ent7;
  li.accuracy = (double)correct / N;
  return li;
}

// Router backward. x is the router input; adds dx into dX (overwritten first).
template <class T>
static void router_bwd(const Config& c, const RouterOff& o, const T* p, const T* x, int N,
                       const RouterCache<T>& r, const T* dout, const T* dent, T* g, T* dX) {
  const int K = c.num_routes, d = c.fc;
  std::fill(dX, dX + (size_t)N * d, T(0));
  std::vector<T> dalpha((size_t)N * K, T(0)), dtz((size_t)N * d);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const T* tz = &r.tz[((size_t)k * N + n) * d];
      const T* go = dout + (size_t)n * d;
      T s = 0;
      for (int i = 0; i < d; ++i) s += go[i] * tz[i];
      const T a = r.alpha[(size_t)n * K + k];
      // entropy = -sum a log(a + eps)
      s += dent[n] * -(std::log(a + T(kEntEps)) + a / (a + T(kEntEps)));
      dalpha[(size_t)n * K + k] = s;
    }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      const T a = r.alpha[(size_t)n * K + k];
      for (int i = 0; i < d; ++i) dtz[(size_t)n * d + i] = a * dout[(size_t)n * d + i];
    }
    linear_bwd(x, p + o.W[k], dtz.data(), g + o.W[k], g + o.b[k], dX, N, d, d, true);
  }
  // alpha = softmax(logits / tau), tau constant (no gradient, as in PyTorch).
  std::vector<T> dl((size_t)N * K);
  for (int n = 0; n < N; ++n) {
    const T* a = &r.alpha[(size_t)n * K];
    const T* da = &dalpha[(size_t)n * K];
    T dot = 0;
    for (int k = 0; k < K; ++k) dot += a[k] * da[k];
    const T inv = T(1) / r.tau[n];
    for (int k = 0; k < K; ++k) dl[(size_t)n * K + k] = a[k] * (da[k] - dot) * inv;
  }
  linear_bwd(x, p + o.Wr, dl.data(), g + o.Wr, g + o.br, dX, N, d, K, true);
}

template <class T>
static void act_bwd(const Config& c, const std::vector<T>& z, std::vector<T>& d) {
  const T al = T(c.act_alpha), be = T(c.act_beta);
  for (size_t i = 0; i < z.size(); ++i) d[i] *= smooth_leaky_grad(z[i], al, be);
}

template <class T>
void cpu_backward(const Config& c, const sk::ParamTable& t, const T* p, const CpuState<T>& st,
                  const T* dlogits, const T* dent6, const T* dent7, T* g, T* dx) {
  Offsets o(c, t);
  const Shapes& s = st.s;
  const int N = st.N, d = c.fc;
  std::fill(g, g + t.total, T(0));
  std::vector<T> dout7((size_t)N * d), da7((size_t)N * d), dout6((size_t)N * d),
      da6((size_t)N * d);
  linear_bwd(st.r7.out.data(), p + o.fc8W, dlogits, g + o.fc8W, g + o.fc8b, dout7.data(), N, d,
             c.num_classes, false);
  router_bwd(c, o.r7, p, st.a7.data(), N, st.r7, dout7.data(), dent7, g, da7.data());
  act_bwd(c, st.z7, da7);
  linear_bwd(st.r6.out.data(), p + o.fc7W, da7.data(), g + o.fc7W, g + o.fc7b, dout6.data(), N,
             d, d, false);
  router_bwd(c, o.r6, p, st.a6.data(), N, st.r6, dout6.data(), dent6, g, da6.data());
  act_bwd(c, st.z6, da6);
  std::vector<T> dp5((size_t)N * s.fc6_in);
  linear_bwd(st.p5.data(), p + o.fc6W, da6.data(), g + o.fc6W, g + o.fc6b, dp5.data(), N,
             s.fc6_in, d, false);

  std::vector<T> da[5];
  for (int i = 0; i < 5; ++i) da[i].assign(st.a[i].size(), T(0));
  pool_bwd(dp5.data(), st.am5.data(), N * c.ch[4], s.c[4], s.p5, da[4].data());
  const int cin[5] = {c.in_ch, c.ch[0], c.ch[1], c.ch[2], c.ch[3]};
  const int hin[5] = {s.in, s.p1, s.p2, s.c[2], s.c[3]};
  const T* xin[5] = {st.x.data(), st.p1.data(), st.p2.data(), st.a[2].data(), st.a[3].data()};
  std::vector<T> dpool;
  for (int i = 4; i >= 0; --i) {
    act_bwd(c, st.z[i], da[i]);  // da[i] now holds dz
    T* dxi = nullptr;
    if (i == 0) {
      dxi = dx;
    } else if (i == 3 || i == 4) {
      dxi = da[i - 1].data();
    } else {  // i == 1, 2: input is a pool output
      dpool.assign((size_t)N * cin[i] * hin[i] * hin[i], T(0));
      dxi = dpool.data();
    }
    conv_bwd(xin[i], N, cin[i], hin[i], p + o.convW[i], c.ch[i], kConv[i], s.c[i], da[i].data(),
             g + o.convW[i], g + o.convB[i], dxi);
    if (i == 2) pool_bwd(dpool.data(), st.am2.data(), N * c.ch[1], s.c[1], s.p2, da[1].data());
    if (i == 1) pool_bwd(dpool.data(), st.am1.data(), N * c.ch[0], s.c[0], s.p1, da[0].data());
  }
}

template <class T>
LossInfo cpu_train_step_sgd(const Config& c, const sk::ParamTable& t, std::vector<T>& p,
                            std::vector<T>& buf, int step, const T* x, const int* labels, int N,
                            const sk::SgdConfig& sc) {
  CpuState<T> st;
  cpu_forward(c, t, p.data(), x, N, st);
  std::vector<T> dl, d6, d7;
  LossInfo li = cpu_loss(c, st, labels, dl, d6, d7);
  std::vector<T> g(t.total);
  cpu_backward(c, t, p.data(), st, dl.data(), d6.data(), d7.data(), g.data(), (T*)nullptr);
  if (buf.size() != t.total) buf.assign(t.total, T(0));
  sk::sgd_step_host(sc, step, p.data(), g.data(), buf.data(), nullptr, t.total);
  return li;
}

#define AMM_INST(T)                                                                          \
  template T smooth_leaky<T>(T, T, T);                                                       \
  template T smooth_leaky_grad<T>(T, T, T);                                                  \
  template int constrained_softmax<T>(const T*, int, int, T, T*, T*, int*, const int*,       \
                                      const double*);                                        \
  template void cpu_forward<T>(const Config&, const sk::ParamTable&, const T*, const T*, int, \
                               CpuState<T>&, const Freeze*);                                 \
  template LossInfo cpu_loss<T>(const Config&, const CpuState<T>&, const int*,               \
                                std::vector<T>&, std::vector<T>&, std::vector<T>&,           \
                                const Freeze*);                                              \
  template void cpu_backward<T>(const Config&, const sk::ParamTable&, const T*,              \
                                const CpuState<T>&, const T*, const T*, const T*, T*, T*);   \
  template LossInfo cpu_train_step_sgd<T>(const Config&, const sk::ParamTable&,              \
                                          std::vector<T>&, std::vector<T>&, int, const T*,   \
                                          const int*, int, const sk::SgdConfig&);
AMM_INST(float)
AMM_INST(double)

}  // namespace amm
