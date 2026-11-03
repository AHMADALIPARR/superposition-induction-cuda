<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright Ahmad Ali Parr -->
# Kernels

All device math is fp32. Matrix products go through cuBLAS (`sk::gemm_rm` and
`sk::gemm_rm_batched` take row-major operands and swap the operands to fit column-major
cuBLAS). Everything else is a hand-written kernel. Elementwise kernels use grid-stride loops
capped at 65535 x 8 blocks of 256 threads. Reductions use warp shuffles
(`include/common/device_reduce.cuh`).

## Shared (`src/common/cuda_util.cu`)

| function | what it does |
| --- | --- |
| `linear_fwd(X, W, b, Y, M, K, N)` | `Y[M,N] = X[M,K] W[N,K]^T + b`, the `nn.Linear` layout |
| `linear_bwd(...)` | `dW += dY^T X`, `db += colsum(dY)`, `dX = dY W` (or `+=`) |
| `col_sum_add`, `col_dot_add` | column reductions used by the bias and LN gradients |
| `softmax_rows_fwd/bwd` | numerically stable row softmax over the last dimension, one warp per row |
| `permute5` | general 5-D transpose for the head and path reshapes |
| `sq_diff_norms` | `sum (a-b)^2` and `sum b^2` accumulated in double, for the refinement delta |
| `adamw_step_device`, `sgd_step_device` | optimizer updates with PyTorch's semantics. A mask skips tensors that have no gradient |

## Superposition (`src/sid/kernels.cu`)

| kernel | grid | notes |
| --- | --- | --- |
| `basis_fwd` | 1 block | row norms, `F.normalize` (eps 1e-12), `Gamma = V V^T`, the `\|Gamma\| > thr` mask, orth loss |
| `basis_bwd` | 1 block | adds `2 w Gamma` off the diagonal, then the normalize backward into `d path_basis` |
| `alpha_fwd/bwd` | 1 block per batch row | CLS logits `x0 P^T / tau` and their softmax |
| `interference_fwd` | 1 thread per (b, s, i) element | walks k = 0..K-1 in order, updating in place and keeping every intermediate `Y_k`, then forms `sp = sum alpha_k Y_k` |
| `interference_bwd` | same | the exact reverse of that walk. Writes `dh`, atomically accumulates `dalpha` and `dGamma` (block-reduced first) |
| `tanh_ln_fwd/bwd` | 1 block per row | tanh then LayerNorm (eps 1e-5, affine), keeping `xhat` and `rstd` |
| `mse_grad` | elementwise | `2 (y - t) / n` for the train step |

One refinement pass on the device:

1. `basis_fwd` (once per forward; the basis does not change between passes) and `alpha_fwd`.
2. `Q, K, V = linear_fwd(x)`, permuted to `[B, H, S, d_h]`.
3. `Ki[b,h,k] = K[b,h] W_ind[k,h]`: one batched GEMM over (b, h, k).
4. Scores `[S, K*S]` per (b, h) in one batched GEMM, scaled by `1/sqrt(d_h)`, then a row
   softmax over each length-S block.
5. `O = A V` (batched), permuted to `[K, B, S, E]`, then `W_O` for all paths at once.
6. `interference_fwd`, then `collapse_proj` and `tanh_ln_fwd`.
7. `sq_diff_norms(y, x)`, which is the single host read per pass for the stopping test.

The backward pass replays the passes in reverse and accumulates the gradients of the shared
weights across passes. The input gradient of pass i becomes the output gradient of pass i-1.

## AlexNet-MMEP (`src/alexnet_mmep/kernels.cu`)

### Convolution without cuDNN

A chunk of `nb` images is unrolled into one "wide" column matrix:

```
col[(c*k + ky)*k + kx][n*Ho*Ho + oy*Ho + ox] = x[n, c, oy*s - p + ky, ox*s - p + kx]   (0 outside)
```

Then:

| step | operation |
| --- | --- |
| forward | `Zw[O, nb*HW] = W[O, C*k*k] col`, then `unwide_bias` gives `Z[n, O, HW] + b` |
| backward, weights | `to_wide(dZ)`, then `dW += dZw col^T` (GEMM with beta 1) and `db += rowsum(dZw)` (`row_sum_add`) |
| backward, input | `dcol = W^T dZw`, then `col2im_wide`, a gather kernel with one thread per input pixel and no atomics |

The chunk size is chosen so `col` and `Zw` together fit in a 256 MiB workspace
(`set_workspace_bytes` changes it). For conv1 at 224 this is about 80 images per GEMM.
Making the GEMM wide over images instead of looping per image keeps the cuBLAS calls large.

### Other kernels

| kernel | notes |
| --- | --- |
| `smooth_leaky_fwd/bwd` | `F.softplus` threshold 20. The backward multiplies the incoming gradient in place by `a + (1-a) sigmoid(b z)` |
| `maxpool_fwd/bwd` | 3x3, stride 2, first maximum wins (NaN propagates). The backward zeroes the input gradient and `atomicAdd`s into the recorded argmax, because overlapping windows can share a winner |
| `constrained_softmax` | one block of 256 threads, with a warp per group of rows. Plain softmax and entropy first. If any row violates, every row bisects in lock-step and the block stops when the widest interval is under 1e-6 (block-wide max), as in the PyTorch loop. Writes alpha, tau, viol, ent and the iteration count. Optional frozen viol/tau for tests |
| `constrained_softmax_bwd` | `dl = alpha (dalpha - <alpha, dalpha>) / tau` |
| `router_combine` | `out = sum_k alpha_k tz_k`, summed in k order like the source's `+=` loop |
| `router_dalpha` | grid (N, K): `<dout, tz_k>` plus the entropy term `-dent (log(a + 1e-10) + a/(a + 1e-10))` |
| `scale_rows` | `dtz_k = alpha_k dout`, which feeds `linear_bwd` of route transform k |
| `cross_entropy` | one block per row: log-sum-exp, per-row loss, `(softmax - onehot)/N`, first argmax |
| `row_sum_add` | bias gradient of a convolution |

Router forward is `linear_fwd` for the logits, `constrained_softmax`, then K `linear_fwd`s into
`tz[K, N, d]` and `router_combine`. The fully connected layers use the shared `linear_fwd/bwd`.

The penalty decision needs the batch-mean entropies on the host. They are read back once per
forward pass, and the per-row entropy gradient is uploaded before the backward pass.

## Determinism

The forward passes are deterministic. The backward passes use atomics in three places:
`interference_bwd` for `dalpha`/`dGamma`, the max-pool backward and the cuBLAS reductions, so
gradients can differ in the last bits from run to run.
