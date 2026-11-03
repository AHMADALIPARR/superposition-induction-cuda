<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright Ahmad Ali Parr -->
# Walkthrough

This document follows the two PyTorch sources line by line and says what the port does at
each step, and why where there was a choice. The sources are in
[source-model.txt](source-model.txt) and [source-alexnet-mmep.txt](source-alexnet-mmep.txt).

## Layout

Each model has the same five pieces:

1. `model.hpp/.cpp`: the config, the parameter table (names, shapes and offsets into one flat
   `float` buffer, in `state_dict` order), initialisation and weight I/O.
2. `cpu_ref.hpp/.cpp`: a plain C++ forward, backward and optimizer step, templated on
   `float` and `double`. This is the ground truth.
3. `kernels.cuh/.cu`: the custom CUDA kernels.
4. `cuda_model.cuh/.cu`: the `CudaModel` class that owns device buffers and chains kernels and
   cuBLAS calls into the forward pass, backward pass and train step.
5. Tests: the CPU backward is checked against finite differences in double, then the CUDA
   forward and backward are checked against the CPU reference.

Every parameter lives in one contiguous device buffer, with gradients in a second buffer of
the same layout, so the optimizers are a single elementwise kernel.

## Superpositioned induction

### Path weights

```python
cls = state[:, 0]
path_logits = cls @ self.path_basis.T / self.config.collapse_temperature
alpha = F.softmax(path_logits, dim=-1)
```

CLS is token 0 of each sequence. The raw (unnormalised) basis is used here; only the
interference uses `F.normalize`. `kern::alpha_fwd` does the K dot products and the softmax in
one block per batch row.

### Per-path attention

`_compute_path_induction` runs the shared `W_Q`, `W_K`, `W_V` (each `nn.Linear` with bias)
and, for path k, multiplies the keys by `W_induction[k]` per head before a standard scaled
softmax attention and `W_O`. Q, K and V do not depend on k, so the port computes them once per
refinement pass and then forms all K induced key sets with one batched GEMM. The scores for all
paths are one `[S, K*S]` GEMM per (batch, head), the row softmax runs over the last dimension
separately for each path, and the values are applied with a second GEMM. The result is
permuted to `[K, B, S, E]` and `W_O` is applied to all paths as one linear layer. This gives
the same numbers as the source's Python loop over k, but with K times fewer kernel launches.

### Interference

```python
for k in range(K):
    interference_k = self._compute_interference(k, path_outputs)
    path_outputs[:, k] = path_outputs[:, k] + interference_k
```

This loop is sequential and in place. Path k's update reads the current `path_outputs`, and
paths j < k have already been updated. The port does the same thing: `kern::interference_fwd`
walks k in order and keeps the intermediate path outputs, which the backward pass needs.
The threshold test is `|gamma_kj| > interference_threshold` on the normalised inner
product, with `F.normalize`'s `max(norm, 1e-12)`. Because the test is discrete, the
on/off mask is computed once per pass on the device and can be overridden (`GpuFreeze`) so
the CUDA and CPU paths can be compared with the same decisions.

The PyTorch code itself cannot be backpropagated once any pair passes the threshold. The
in-place write into `path_outputs[:, k]` changes a tensor that autograd saved for an earlier
view, so `loss.backward()` raises the "modified by an inplace operation" error. The port's
backward is the exact gradient of the forward as written, and it agrees with autograd run
on a functional rewrite that copies instead of writing in place.

### Collapse and orthogonality loss

`superposed = sum_k alpha_k h_k`, then `tanh(collapse_proj(superposed))`, then `nn.LayerNorm`
with eps 1e-5 and the affine weight and bias. The tanh and the LayerNorm share one kernel
(`kern::tanh_ln_fwd`) that keeps `xhat` and `rstd` for the backward pass. The orthogonality loss
`sum_{i != j} Gamma_ij^2` is computed from the same Gamma and returned with the output.

### Refinement

`IterativeSuperpositionRefinement` applies the block again to its own output, at most
`max_iterations` = 5 times, and stops early when
`||output - current||_F / (||current||_F + 1e-8) < 1e-4`. The norms are reduced in double on
the device and read back once per pass, which is the one host sync per pass. The backward pass
runs through every pass that actually ran. The stopping test has no gradient, as in autograd.
`SuperpositionedBURTIMMA` adds `output_proj` at the end.

### What is trained

`path_confidence` is a parameter that the forward pass never reads, so its gradient is
exactly 0. PyTorch would leave `.grad` as `None`, and the optimizers skip it through a mask, so
it is not touched by weight decay either. `W_K.bias` has a true gradient of about 0, since the
softmax is invariant to it up to rounding; the tests check it with an absolute tolerance.

The source has no trainer for this model. The port's `train_step_adamw` / `train_step_sgd`
minimise `mean((y - target)^2) + orthogonality_reg * L_orth` with `orthogonality_reg` = 0.01
from the config. `backward(dy, orth_weight)` is exposed for any other loss.

## AlexNet-MMEP

### Shapes

| layer | 224 input | 227 input |
| --- | --- | --- |
| conv1 11x11 / 4, pad 2 | 96 x 55 x 55 | 96 x 56 x 56 |
| pool 3 / 2 | 27 | 27 |
| conv2 5x5, pad 2, pool | 256 x 27, then 13 | same |
| conv3-5 3x3, pad 1 | 13 | 13 |
| pool | 256 x 6 x 6 = 9216 | 9216 |

`nn.Linear(256 * 6 * 6, 4096)` therefore fits both 224 and 227 inputs. Working through the
floor divisions, any square input from 223 to 254 pixels gives 9216. `Config::fc6_in = -1`
derives the size from the image. Any other value must match, otherwise the config throws
with the derived number, the same way PyTorch would fail with a shape error.

### SmoothLeaky

From `burt_imma/smooth_leaky.py`: `f(x) = a x + (1 - a)/b (softplus(b x) - ln 2)` with
`F.softplus`'s threshold of 20 (linear above `b x > 20`). The derivative is
`a + (1 - a) sigmoid(b x)`. `alpha` and `beta` are registered buffers in the source, not
parameters, so they are stored in the weight file under `activation._alpha` and
`activation._beta` and are not trained.

### constrained_softmax

The bridge imports `constrained_softmax` from `burt_imma/burt.py`, and that is the version
ported here. Two other versions exist in the burt-imma repo, `python/burt_imma/kernels.py`
and `scripts/profile_kernels.py`, and they behave differently. The burt.py version:

1. plain softmax, then the entropy `-sum p log(p + 1e-10)` per row;
2. if no row exceeds the bound, return the plain softmax;
3. otherwise bisect `tau` in `[0.01, 1]` for every row, 32 iterations at most, stopping when
   the widest interval is under 1e-6, which takes 20 iterations;
4. `tau = (lo + hi) / 2`, and only the violating rows take `softmax(logits / tau)`.

`tau` is found by comparisons, so autograd sees it as a constant. The backward pass is the
softmax backward scaled by `1/tau` on violating rows and by 1 on the rest. The CUDA kernel runs
the bisection for all rows in one block, with the same per-row and global stopping rules.
When the bound is out of reach even at `tau = 0.01`, `tau` ends at the lower limit, as in
PyTorch.

### Router and network

`EntropyConstrainedRouter` sums `alpha[:, k] * route_transforms[k](x)` into a zero tensor in
order k = 0..K-1, and returns the per-row entropy of alpha. The model returns the batch means
as `entropy_fc6` / `entropy_fc7`. Each router holds K + 1 linear layers of size 4096, so the
model has 196,661,616 parameters in total and 62,378,344 outside the routers. The source
docstring's "~62M" counts only the latter.

`nn.MaxPool2d(3, 2)` returns the first maximum in the window (a NaN wins). The forward pass
records the argmax and the backward pass scatters into it. Flattening is the NCHW `view`.

### Trainer

`AlexNetMMEPTrainer.train_step` is CrossEntropyLoss (mean), plus for each router
`0.1 * (ent - 0.2)^2` when the batch-mean entropy is above 0.2, then
`torch.optim.SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)` over all parameters. The penalty
gradient reaches each row's entropy as `0.1 * 2 (mean - 0.2) / N`, and through
`d ent / d alpha = -(log(alpha + 1e-10) + alpha / (alpha + 1e-10))` it reaches the router
logits. SGD follows PyTorch's rule: on the first step the momentum buffer is set to the
gradient (with weight decay added), and after that `buf = m buf + g`.

After a bisection the row entropy sits within about 1e-6 of the bound, so whether the
batch mean is "above 0.2" can depend on rounding. The CUDA tests therefore give the CPU
reference the GPU's penalty decisions, along with its pool argmax and bisection results.

## Precision

Device math is fp32. cuBLAS runs in its default (non-TF32) mode unless `set_tf32(true)` is
called or `--tf32` is passed to a bench. The CPU reference runs in double for the gradient
checks.

## Testing

| test | what it checks | needs a GPU |
| --- | --- | --- |
| `weights_io` | save and load of both models, config records included | no |
| `sid_cpu_gradcheck` | CPU backward vs central differences in double, every tensor and dx, with and without active interference | no |
| `amm_cpu_gradcheck` | the 224/227 shape chain, then CPU backward vs 5-point differences in double on a 67x67 config (bisecting rows, the penalty firing, plain softmax) | no |
| `sid_cuda_forward` | CUDA vs CPU forward on small configs and the 768/4/8/64 default | yes |
| `sid_cuda_backward` | CUDA vs CPU gradients of every parameter and dx, and one optimizer step | yes |
| `amm_cuda_forward` | CUDA vs CPU forward at 67x67, 99x99 and the full 224x224 network | yes |
| `amm_cuda_backward` | CUDA vs CPU gradients of every parameter and dx, one trainer SGD step, and a short training run | yes |

The GPU tests exit with code 77 when no device is found, and ctest reports them as skipped.

The CPU references were also checked against the PyTorch source classes outside this repo.
For superposition the forward matched to 5.7e-8. For AlexNet-MMEP the double logits matched
to 1.5e-8 and the worst relative gradient error was 5e-8.

## Build environment note

The toolkit used during development (CUDA 12.9 on glibc 2.41) needed `noexcept(true)` added
to `sinpi`/`cospi` in `crt/math_functions.h` before host code would compile. This is a known
toolkit/glibc mismatch and has nothing to do with this repo. Newer toolkits do not need it.
