// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
#include "sid/cpu_ref.hpp"

#include <algorithm>
#include <cmath>

namespace sid {

namespace {

// Y[M,N] = X[M,K] W[N,K]^T + b
template <class T>
void lin(const T* X, const T* W, const T* b, T* Y, int M, int K, int N) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      T acc = b ? b[j] : T(0);
      const T* xr = X + (size_t)i * K;
      const T* wr = W + (size_t)j * K;
      for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
      Y[(size_t)i * N + j] = acc;
    }
}

// dW += dY^T X, db += colsum dY, dX (+)= dY W
template <class T>
void lin_bwd(const T* X, const T* W, const T* dY, T* dW, T* db, T* dX, int M, int K, int N,
             bool acc_dx) {
  for (int i = 0; i < M; ++i) {
    const T* g = dY + (size_t)i * N;
    const T* xr = X + (size_t)i * K;
    for (int j = 0; j < N; ++j) {
      if (db) db[j] += g[j];
      T* wr = dW + (size_t)j * K;
      for (int k = 0; k < K; ++k) wr[k] += g[j] * xr[k];
    }
    if (dX) {
      T* dx = dX + (size_t)i * K;
      if (!acc_dx) std::fill(dx, dx + K, T(0));
      for (int j = 0; j < N; ++j) {
        const T* wr = W + (size_t)j * K;
        for (int k = 0; k < K; ++k) dx[k] += g[j] * wr[k];
      }
    }
  }
}

template <class T>
void basis(const Config& c, const T* P, CpuState<T>& st, const Freeze* fz) {
  const int K = c.num_paths, d = c.d_model;
  st.norms.assign(K, 0);
  st.vn.assign((size_t)K * d, 0);
  st.G.assign((size_t)K * K, 0);
  st.mask.assign((size_t)K * K, 0);
  for (int k = 0; k < K; ++k) {
    T n = 0;
    for (int i = 0; i < d; ++i) n += P[(size_t)k * d + i] * P[(size_t)k * d + i];
    n = std::sqrt(n);
    st.norms[k] = n;
    T den = std::max(n, T(c.normalize_eps));  // F.normalize: x / max(||x||, eps)
    for (int i = 0; i < d; ++i) st.vn[(size_t)k * d + i] = P[(size_t)k * d + i] / den;
  }
  st.orth = 0;
  for (int a = 0; a < K; ++a)
    for (int b = 0; b < K; ++b) {
      T g = 0;
      for (int i = 0; i < d; ++i) g += st.vn[(size_t)a * d + i] * st.vn[(size_t)b * d + i];
      st.G[a * K + b] = g;
      if (a != b) {
        st.orth += g * g;
        st.mask[a * K + b] = std::fabs(g) > T(c.interference_threshold) ? 1 : 0;
      }
    }
  if (fz && !fz->mask.empty()) st.mask = fz->mask;
}

// SuperpositionedInductionHeads.forward on x (one refinement iteration).
template <class T>
void heads_forward(const Config& c, const Offsets& o, const T* p, const CpuState<T>& st,
                   IterCache<T>& ic, int B, int S) {
  const int d = c.d_model, K = c.num_paths, H = c.num_heads, dh = c.head_dim, E = c.E();
  const size_t BS = (size_t)B * S;
  const T* x = ic.x_in.data();
  const T* P = p + o.path_basis;
  // alpha = softmax(cls @ path_basis^T / tau)
  ic.alpha.assign((size_t)B * K, 0);
  for (int b = 0; b < B; ++b) {
    const T* cls = x + (size_t)b * S * d;
    T lg[kMaxPaths], mx = -1e300;
    for (int k = 0; k < K; ++k) {
      T s = 0;
      for (int i = 0; i < d; ++i) s += cls[i] * P[(size_t)k * d + i];
      lg[k] = s / T(c.collapse_temperature);
      mx = std::max(mx, lg[k]);
    }
    T z = 0;
    for (int k = 0; k < K; ++k) z += std::exp(lg[k] - mx);
    for (int k = 0; k < K; ++k) ic.alpha[b * K + k] = std::exp(lg[k] - mx) / z;
  }
  ic.Q.resize(BS * E);
  ic.Kp.resize(BS * E);
  ic.V.resize(BS * E);
  lin(x, p + o.Wq, p + o.bq, ic.Q.data(), (int)BS, d, E);
  lin(x, p + o.Wk, p + o.bk, ic.Kp.data(), (int)BS, d, E);
  lin(x, p + o.Wv, p + o.bv, ic.V.data(), (int)BS, d, E);
  ic.Ki.assign((size_t)K * B * H * S * dh, 0);
  ic.A.assign((size_t)K * B * H * S * S, 0);
  ic.O.assign((size_t)K * BS * E, 0);
  ic.h.assign((size_t)K * BS * d, 0);
  const T scale = T(1) / std::sqrt(T(dh));
  std::vector<T> row(S);
  for (int k = 0; k < K; ++k) {
    const T* Wi = p + o.W_induction + (size_t)k * H * dh * dh;
    for (int b = 0; b < B; ++b)
      for (int hh = 0; hh < H; ++hh) {
        T* Ki = &ic.Ki[((((size_t)k * B + b) * H + hh) * S) * dh];
        // einsum("bhsd,hde->bhse")
        for (int s = 0; s < S; ++s)
          for (int e = 0; e < dh; ++e) {
            T acc = 0;
            for (int dd = 0; dd < dh; ++dd)
              acc += ic.Kp[((size_t)b * S + s) * E + hh * dh + dd] * Wi[((size_t)hh * dh + dd) * dh + e];
            Ki[(size_t)s * dh + e] = acc;
          }
        T* A = &ic.A[((((size_t)k * B + b) * H + hh) * S) * S];
        for (int q = 0; q < S; ++q) {
          const T* qv = &ic.Q[((size_t)b * S + q) * E + hh * dh];
          T mx = -1e300;
          for (int s = 0; s < S; ++s) {
            T acc = 0;
            for (int e = 0; e < dh; ++e) acc += qv[e] * Ki[(size_t)s * dh + e];
            row[s] = acc * scale;
            mx = std::max(mx, row[s]);
          }
          T z = 0;
          for (int s = 0; s < S; ++s) z += std::exp(row[s] - mx);
          for (int s = 0; s < S; ++s) A[(size_t)q * S + s] = std::exp(row[s] - mx) / z;
          T* out = &ic.O[(((size_t)k * B + b) * S + q) * E + hh * dh];
          for (int e = 0; e < dh; ++e) {
            T acc = 0;
            for (int s = 0; s < S; ++s) acc += A[(size_t)q * S + s] * ic.V[((size_t)b * S + s) * E + hh * dh + e];
            out[e] = acc;
          }
        }
      }
    lin(&ic.O[(size_t)k * BS * E], p + o.Wo, p + o.bo, &ic.h[(size_t)k * BS * d], (int)BS, E, d);
  }
  // In-place sequential interference: path k sees already-updated paths j < k.
  ic.Y = ic.h;
  const size_t n = BS * d;
  for (int k = 0; k < K; ++k) {
    std::vector<T> inter(n, T(0));
    for (int j = 0; j < K; ++j) {
      if (j == k || !st.mask[k * K + j]) continue;
      const T g = st.G[k * K + j];
      for (size_t i = 0; i < n; ++i) inter[i] += g * ic.Y[(size_t)j * n + i];
    }
    for (size_t i = 0; i < n; ++i) ic.Y[(size_t)k * n + i] += inter[i];
  }
  // Decoherence collapse.
  ic.sp.assign(n, 0);
  for (int b = 0; b < B; ++b)
    for (int k = 0; k < K; ++k) {
      const T a = ic.alpha[b * K + k];
      for (size_t i = 0; i < (size_t)S * d; ++i)
        ic.sp[(size_t)b * S * d + i] += a * ic.Y[(size_t)k * n + (size_t)b * S * d + i];
    }
  ic.t.resize(n);
  lin(ic.sp.data(), p + o.Wc, p + o.bc, ic.t.data(), (int)BS, d, d);
  for (auto& v : ic.t) v = std::tanh(v);
  ic.xhat.resize(n);
  ic.rstd.resize(BS);
  ic.out.resize(n);
  for (size_t r = 0; r < BS; ++r) {
    const T* tr = &ic.t[r * d];
    T mean = 0;
    for (int i = 0; i < d; ++i) mean += tr[i];
    mean /= d;
    T var = 0;
    for (int i = 0; i < d; ++i) var += (tr[i] - mean) * (tr[i] - mean);
    var /= d;  // biased, as nn.LayerNorm
    T rs = T(1) / std::sqrt(var + T(c.ln_eps));
    ic.rstd[r] = rs;
    for (int i = 0; i < d; ++i) {
      T xh = (tr[i] - mean) * rs;
      ic.xhat[r * d + i] = xh;
      ic.out[r * d + i] = xh * p[o.ln_g + i] + p[o.ln_b + i];
    }
  }
}

}  // namespace

template <class T>
void cpu_forward(const Config& c, const sk::ParamTable& tab, const T* p, const T* x, int B, int S,
                 CpuState<T>& st, const Freeze* fz) {
  Offsets o(tab);
  const int d = c.d_model;
  const size_t n = (size_t)B * S * d;
  st.B = B;
  st.S = S;
  st.it.clear();
  st.deltas.clear();
  basis(c, p + o.path_basis, st, fz);
  std::vector<T> cur(x, x + n);
  const int maxit = (fz && fz->iterations > 0) ? fz->iterations : c.max_iterations;
  for (int i = 0; i < maxit; ++i) {
    st.it.emplace_back();
    IterCache<T>& ic = st.it.back();
    ic.x_in = cur;
    heads_forward(c, o, p, st, ic, B, S);
    double num = 0, den = 0;
    for (size_t j = 0; j < n; ++j) {
      double df = (double)ic.out[j] - (double)cur[j];
      num += df * df;
      den += (double)cur[j] * (double)cur[j];
    }
    double delta = std::sqrt(num) / (std::sqrt(den) + 1e-8);
    st.deltas.push_back(delta);
    st.final_delta = delta;
    st.iterations = i + 1;
    if (!(fz && fz->iterations > 0) && delta < c.convergence_threshold) break;
    cur = ic.out;
  }
  st.y.resize(n);
  lin(st.it.back().out.data(), p + o.Wp, p + o.bp, st.y.data(), B * S, d, d);
}

template <class T>
void cpu_backward(const Config& c, const sk::ParamTable& tab, const T* p, const CpuState<T>& st,
                  const T* dy, T orth_weight, T* grad, T* dx) {
  Offsets o(tab);
  const int B = st.B, S = st.S, d = c.d_model, K = c.num_paths, H = c.num_heads, dh = c.head_dim,
            E = c.E();
  const size_t BS = (size_t)B * S, n = BS * d;
  std::fill(grad, grad + tab.total, T(0));
  const T scale = T(1) / std::sqrt(T(dh));
  const T* P = p + o.path_basis;
  T* dP = grad + o.path_basis;
  std::vector<T> dG((size_t)K * K, T(0));

  // output_proj
  std::vector<T> dout(n);
  lin_bwd(st.it.back().out.data(), p + o.Wp, dy, grad + o.Wp, grad + o.bp, dout.data(), (int)BS, d,
          d, false);

  for (int it = st.iterations - 1; it >= 0; --it) {
    const IterCache<T>& ic = st.it[it];
    // LayerNorm + tanh
    std::vector<T> dz(n);
    for (size_t r = 0; r < BS; ++r) {
      T m1 = 0, m2 = 0;
      for (int i = 0; i < d; ++i) {
        T g = dout[r * d + i];
        grad[o.ln_g + i] += g * ic.xhat[r * d + i];
        grad[o.ln_b + i] += g;
        T dxh = g * p[o.ln_g + i];
        m1 += dxh;
        m2 += dxh * ic.xhat[r * d + i];
      }
      m1 /= d;
      m2 /= d;
      for (int i = 0; i < d; ++i) {
        T dxh = dout[r * d + i] * p[o.ln_g + i];
        T dt = ic.rstd[r] * (dxh - m1 - ic.xhat[r * d + i] * m2);
        T tv = ic.t[r * d + i];
        dz[r * d + i] = dt * (T(1) - tv * tv);
      }
    }
    std::vector<T> dsp(n);
    lin_bwd(ic.sp.data(), p + o.Wc, dz.data(), grad + o.Wc, grad + o.bc, dsp.data(), (int)BS, d, d,
            false);
    // superposition: sp = sum_k alpha_k Y_k
    std::vector<T> dalpha((size_t)B * K, T(0));
    std::vector<T> gY((size_t)K * n), dHk((size_t)K * n, T(0));
    for (int k = 0; k < K; ++k)
      for (int b = 0; b < B; ++b) {
        const T a = ic.alpha[b * K + k];
        T acc = 0;
        for (size_t i = 0; i < (size_t)S * d; ++i) {
          size_t li = (size_t)b * S * d + i;
          acc += dsp[li] * ic.Y[(size_t)k * n + li];
          gY[(size_t)k * n + li] = a * dsp[li];
        }
        dalpha[b * K + k] = acc;
      }
    // interference (reverse of the sequential in-place loop)
    for (int k = K - 1; k >= 0; --k) {
      const T* g = &gY[(size_t)k * n];
      for (size_t i = 0; i < n; ++i) dHk[(size_t)k * n + i] += g[i];
      for (int j = 0; j < K; ++j) {
        if (j == k || !st.mask[k * K + j]) continue;
        const T G = st.G[k * K + j];
        T acc = 0;
        if (j < k) {
          for (size_t i = 0; i < n; ++i) {
            gY[(size_t)j * n + i] += G * g[i];
            acc += g[i] * ic.Y[(size_t)j * n + i];
          }
        } else {
          for (size_t i = 0; i < n; ++i) {
            dHk[(size_t)j * n + i] += G * g[i];
            acc += g[i] * ic.h[(size_t)j * n + i];
          }
        }
        dG[k * K + j] += acc;
      }
    }
    // per-path attention
    std::vector<T> dQ(BS * E, T(0)), dKp(BS * E, T(0)), dV(BS * E, T(0)), dO(BS * E);
    std::vector<T> dA(S), dKi((size_t)S * dh);
    for (int k = 0; k < K; ++k) {
      lin_bwd(&ic.O[(size_t)k * BS * E], p + o.Wo, &dHk[(size_t)k * n], grad + o.Wo, grad + o.bo,
              dO.data(), (int)BS, E, d, false);
      const T* Wi = p + o.W_induction + (size_t)k * H * dh * dh;
      T* dWi = grad + o.W_induction + (size_t)k * H * dh * dh;
      for (int b = 0; b < B; ++b)
        for (int hh = 0; hh < H; ++hh) {
          const T* Ki = &ic.Ki[((((size_t)k * B + b) * H + hh) * S) * dh];
          const T* A = &ic.A[((((size_t)k * B + b) * H + hh) * S) * S];
          std::fill(dKi.begin(), dKi.end(), T(0));
          for (int q = 0; q < S; ++q) {
            const T* go = &dO[((size_t)b * S + q) * E + hh * dh];
            T dot = 0;
            for (int s = 0; s < S; ++s) {
              T acc = 0;
              for (int e = 0; e < dh; ++e) acc += go[e] * ic.V[((size_t)b * S + s) * E + hh * dh + e];
              dA[s] = acc;
              dot += acc * A[(size_t)q * S + s];
              for (int e = 0; e < dh; ++e) dV[((size_t)b * S + s) * E + hh * dh + e] += A[(size_t)q * S + s] * go[e];
            }
            for (int s = 0; s < S; ++s) {
              T dsc = A[(size_t)q * S + s] * (dA[s] - dot) * scale;
              for (int e = 0; e < dh; ++e) {
                dQ[((size_t)b * S + q) * E + hh * dh + e] += dsc * Ki[(size_t)s * dh + e];
                dKi[(size_t)s * dh + e] += dsc * ic.Q[((size_t)b * S + q) * E + hh * dh + e];
              }
            }
          }
          // K_induced = Kp_h W_induction[k,h]
          for (int s = 0; s < S; ++s)
            for (int dd = 0; dd < dh; ++dd) {
              T kp = ic.Kp[((size_t)b * S + s) * E + hh * dh + dd];
              T acc = 0;
              for (int e = 0; e < dh; ++e) {
                dWi[((size_t)hh * dh + dd) * dh + e] += kp * dKi[(size_t)s * dh + e];
                acc += dKi[(size_t)s * dh + e] * Wi[((size_t)hh * dh + dd) * dh + e];
              }
              dKp[((size_t)b * S + s) * E + hh * dh + dd] += acc;
            }
        }
    }
    std::vector<T> dxin(n);
    lin_bwd(ic.x_in.data(), p + o.Wq, dQ.data(), grad + o.Wq, grad + o.bq, dxin.data(), (int)BS, d, E, false);
    lin_bwd(ic.x_in.data(), p + o.Wk, dKp.data(), grad + o.Wk, grad + o.bk, dxin.data(), (int)BS, d, E, true);
    lin_bwd(ic.x_in.data(), p + o.Wv, dV.data(), grad + o.Wv, grad + o.bv, dxin.data(), (int)BS, d, E, true);
    // alpha = softmax(cls P^T / tau)
    for (int b = 0; b < B; ++b) {
      T dot = 0;
      for (int k = 0; k < K; ++k) dot += ic.alpha[b * K + k] * dalpha[b * K + k];
      const T* cls = &ic.x_in[(size_t)b * S * d];
      for (int k = 0; k < K; ++k) {
        T dl = ic.alpha[b * K + k] * (dalpha[b * K + k] - dot) / T(c.collapse_temperature);
        for (int i = 0; i < d; ++i) {
          dP[(size_t)k * d + i] += dl * cls[i];
          dxin[(size_t)b * S * d + i] += dl * P[(size_t)k * d + i];
        }
      }
    }
    dout.swap(dxin);  // gradient w.r.t. this iteration's input = previous output
  }
  if (dx) std::copy(dout.begin(), dout.end(), dx);

  // orthogonality loss: sum_{i != j} Gamma_ij^2
  for (int a = 0; a < K; ++a)
    for (int b = 0; b < K; ++b)
      if (a != b) dG[a * K + b] += orth_weight * T(2) * st.G[a * K + b];
  // Gamma = vn vn^T ; vn = P / max(||P||, eps)
  for (int k = 0; k < K; ++k) {
    std::vector<T> dvn(d, T(0));
    for (int j = 0; j < K; ++j) {
      T w = dG[k * K + j] + dG[j * K + k];
      for (int i = 0; i < d; ++i) dvn[i] += w * st.vn[(size_t)j * d + i];
    }
    if (st.norms[k] > T(c.normalize_eps)) {
      T dot = 0;
      for (int i = 0; i < d; ++i) dot += st.vn[(size_t)k * d + i] * dvn[i];
      for (int i = 0; i < d; ++i) dP[(size_t)k * d + i] += (dvn[i] - st.vn[(size_t)k * d + i] * dot) / st.norms[k];
    } else {
      for (int i = 0; i < d; ++i) dP[(size_t)k * d + i] += dvn[i] / T(c.normalize_eps);
    }
  }
  // path_confidence is unused in forward: gradient stays 0.
}

template <class T>
static T loss_and_grad(const Config& c, const sk::ParamTable& tab, const T* p, const T* x,
                       const T* target, int B, int S, std::vector<T>& grad) {
  CpuState<T> st;
  cpu_forward(c, tab, p, x, B, S, st);
  const size_t n = (size_t)B * S * c.d_model;
  std::vector<T> dy(n);
  T mse = 0;
  for (size_t i = 0; i < n; ++i) {
    T df = st.y[i] - target[i];
    mse += df * df;
    dy[i] = T(2) * df / T(n);
  }
  mse /= T(n);
  grad.assign(tab.total, T(0));
  cpu_backward(c, tab, p, st, dy.data(), T(c.orthogonality_reg), grad.data(), (T*)nullptr);
  return mse + T(c.orthogonality_reg) * st.orth;
}

template <class T>
T cpu_train_step_adamw(const Config& c, const sk::ParamTable& tab, T* p, T* m, T* v, int step,
                       const sk::AdamWConfig& oc, const T* x, const T* target, int B, int S) {
  std::vector<T> g;
  T loss = loss_and_grad(c, tab, p, x, target, B, S, g);
  auto mask = sk::grad_mask(tab);
  sk::adamw_step_host(oc, step, p, g.data(), m, v, mask.data(), tab.total);
  return loss;
}

template <class T>
T cpu_train_step_sgd(const Config& c, const sk::ParamTable& tab, T* p, T* buf, int step,
                     const sk::SgdConfig& oc, const T* x, const T* target, int B, int S) {
  std::vector<T> g;
  T loss = loss_and_grad(c, tab, p, x, target, B, S, g);
  auto mask = sk::grad_mask(tab);
  sk::sgd_step_host(oc, step, p, g.data(), buf, mask.data(), tab.total);
  return loss;
}

#define SID_INST(T)                                                                            \
  template void cpu_forward<T>(const Config&, const sk::ParamTable&, const T*, const T*, int, int, \
                               CpuState<T>&, const Freeze*);                                    \
  template void cpu_backward<T>(const Config&, const sk::ParamTable&, const T*,                 \
                                const CpuState<T>&, const T*, T, T*, T*);                       \
  template T cpu_train_step_adamw<T>(const Config&, const sk::ParamTable&, T*, T*, T*, int,     \
                                     const sk::AdamWConfig&, const T*, const T*, int, int);     \
  template T cpu_train_step_sgd<T>(const Config&, const sk::ParamTable&, T*, T*, int,           \
                                   const sk::SgdConfig&, const T*, const T*, int, int);
SID_INST(float)
SID_INST(double)

}  // namespace sid
