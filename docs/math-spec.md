<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright Ahmad Ali Parr -->
# Math spec

Both models written as plain definitions over real numbers, one function per step, with
index ranges and side conditions spelled out. Each definition is meant to map onto a WhyML
`function` or `let` with the stated preconditions as `requires`. Arrays are 0-indexed, and
`sum_{i<n}` means i = 0 .. n-1. Floating point is not modelled. The CUDA and CPU code
compute these definitions in fp32 and fp64.

## Shared pieces

```
softmax(z)_i          = exp(z_i - m) / sum_j exp(z_j - m),   m = max_j z_j
linear(W, b, x)_o     = b_o + sum_i W[o,i] x_i               W : [out, in]
normalize(v)          = v / max(||v||_2, 1e-12)
layernorm(x)_i        = g_i (x_i - mu) / sqrt(var + 1e-5) + beta_i,
                        mu = mean(x), var = mean((x - mu)^2)     (biased variance)
||A||_F               = sqrt(sum of squares of all entries)
```

## Superpositioned induction

Parameters, with d = d_model, K = num_paths, H = num_heads, e = head_dim, E = H e:

```
V      : [K, d]          path_basis
Wind   : [K, H, e, e]    W_induction
Wq, Wk, Wv : [E, d], bq, bk, bv : [E]
Wo : [d, E], bo : [d]
Wc : [d, d], bc : [d]    collapse_proj
g, beta : [d]            ln
Wout : [d, d], bout : [d]
pc : [K]                 path_confidence, never read
```

Constants: T = collapse_temperature, theta = interference_threshold, eps_c = 1e-4,
I = max_iterations. Requires K >= 1, S >= 1, T > 0.

One block, `heads(x)` for x : [B, S, d]:

```
alpha[b]        = softmax_k( sum_i x[b,0,i] V[k,i] / T )                       [B, K]
Q[b,h,s,c]      = linear(Wq, bq, x[b,s])_{h e + c}                             same for Kp, Vv
Kind[k,b,h,s,c] = sum_{c' < e} Kp[b,h,s,c'] Wind[k,h,c',c]
A[k,b,h,s,t]    = softmax_t( sum_c Q[b,h,s,c] Kind[k,b,h,t,c] / sqrt(e) )
O[k,b,s,h e+c]  = sum_t A[k,b,h,s,t] Vv[b,h,t,c]
P0[k,b,s]       = linear(Wo, bo, O[k,b,s])                                     [K, B, S, d]

G[i,j]          = < normalize(V[i]), normalize(V[j]) >                          [K, K]
on(k,j)         = (j != k) and |G[k,j]| > theta
P_{k+1}         = P_k  except  P_{k+1}[k] = P_k[k] + sum_{j : on(k,j)} G[k,j] P_k[j]
                  for k = 0 .. K-1, then P = P_K
```

Each P_{k+1} is P_k with only row k changed, so a row j < k is used after its own update. This
is the in-place loop of the source.

```
sup[b,s]        = sum_k alpha[b,k] P[k,b,s]
y[b,s]          = layernorm( tanh( linear(Wc, bc, sup[b,s]) ) )
orth            = sum_{i != j} G[i,j]^2
```

Refinement and the full model:

```
x_0 = x
for n = 1 .. I:
    y_n = heads(x_{n-1})
    delta_n = ||y_n - x_{n-1}||_F / (||x_{n-1}||_F + 1e-8)
    stop if delta_n < eps_c,  else x_n = y_n
out  = y_N  where N is the pass that stopped, or I
model(x)[b,s] = linear(Wout, bout, out[b,s])
```

The orthogonality term returned is the one from the last pass. It depends only on V, so every
pass gives the same value.

Training loss used by the port, which the source does not define:

```
L(x, target) = mean_{b,s,i} (model(x) - target)^2 + 0.01 * orth
```

Gradients: the mask `on`, the stopping index N and the softmax max shift are treated as
constants. dL/d pc = 0.

## AlexNet-MMEP

### Activation

With a = 0.01 and b = 1:

```
softplus_b(x) = x                    if b x > 20
              = log(1 + exp(b x)) / b otherwise
f(x)          = a x + (1 - a) (softplus_b(x) - log 2 / b)
f'(x)         = a + (1 - a) sigmoid(b x)           (exactly 1 where b x > 20)
```

This is the same as the README form, a x + (1 - a)/b (softplus(b x) - ln 2). f(0) = 0, and
a < f'(x) <= 1.

### Shapes

```
conv(n, k, s, p) = floor((n + 2p - k) / s) + 1
pool(n)          = floor((n - 3) / 2) + 1
n1 = conv(n, 11, 4, 2), q1 = pool(n1)
n2 = conv(q1, 5, 1, 2) = q1, q2 = pool(n2)
n3 = n4 = n5 = q2, q5 = pool(n5)
fc6_in = ch5 * q5 * q5
```

For n = 224: n1 = 55, q1 = 27, q2 = 13, q5 = 6, so fc6_in = 9216. For n = 227: n1 = 56 and the
rest are the same. For every n in [223, 254], q5 = 6.

### Layers

Convolution with zero padding, NCHW, cross-correlation as in PyTorch:

```
conv(W, b, s, p, x)[o, i, j] = b_o + sum_{c, u, v} W[o, c, u, v] x[c, s i + u - p, s j + v - p]
                               (terms outside the image are 0)
maxpool(x)[c, i, j]          = max_{u, v < 3} x[c, 2 i + u, 2 j + v]
```

The backward pass sends each pool gradient to the first maximum in row-major window order.

### constrained_softmax (burt.py version)

For logits z : [N, R], with bound B = 0.2 and H(p) = -sum_k p_k log(p_k + 1e-10):

```
p0[r]   = softmax(z[r])
viol[r] = H(p0[r]) > B
if no row violates: return p0, tau = 1
lo[r] = 0.01, hi[r] = 1 for every row
repeat at most 32 times:
    for every row r:  mid = (lo + hi)/2;  if H(softmax(z[r] / mid)) > B then hi = mid else lo = mid
    if max_r (hi - lo) < 1e-6: stop
tau[r]  = (lo + hi)/2  if viol[r]  else 1
p[r]    = softmax(z[r] / tau[r])
```

Every row is bisected, but only violating rows use the result. tau is a constant for the
gradient, so dp/dz on row r is the softmax Jacobian divided by tau[r].

### Router

For x : [N, D], R routes:

```
alpha       = constrained_softmax(linear(Wr, br, x))                  [N, R]
router(x)   = sum_{k < R} alpha[:, k] * linear(W_k, b_k, x)           summed in k order
ent[n]      = H(alpha[n])
```

### Network and loss

```
x1 = maxpool(f(conv1(x)))         conv1: 11x11, stride 4, pad 2
x2 = maxpool(f(conv2(x1)))        5x5, pad 2
x3 = f(conv3(x2)); x4 = f(conv4(x3))
x5 = maxpool(f(conv5(x4)))        3x3, pad 1 each
h  = flatten(x5)                  NCHW order
u6, e6 = route6(f(fc6(h)))
u7, e7 = route7(f(fc7(u6)))
logits = fc8(u7)
E6 = mean_n e6[n], E7 = mean_n e7[n]

L = mean_n ( -log softmax(logits[n])_{y_n} )
    + sum_{l in {6, 7}} [E_l > 0.2] * 0.1 * (E_l - 0.2)^2
```

The indicator is decided on the forward values and treated as a constant.

### SGD step (PyTorch rule)

With lr = 0.01, m = 0.9, wd = 5e-4 and step t starting at 1:

```
g    = grad + wd * w
buf  = g                 if t = 1
     = m * buf + g       otherwise
w    = w - lr * buf
```

## Gradient checks as properties

The tests check this property for every parameter tensor and the input, with h = 1e-5 on
small configs and in double:

```
| dL/dw_i - (L(w + h e_i) - L(w - h e_i)) / (2h) | <= atol    or    relative error <= 1e-5
```

The AlexNet-MMEP check uses a 5-point stencil, with the discrete decisions (viol, the pool
argmax and the penalty indicator) held fixed. The measured errors are in the README.
