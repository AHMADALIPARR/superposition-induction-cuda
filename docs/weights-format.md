<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright Ahmad Ali Parr -->
# Weight format

Both models read and write one small binary format, `SKT1`. Files usually end in `.skt`.
The code is in `src/common/tensor_file.cpp`.

## Layout

All integers are little-endian.

| bytes | field |
| --- | --- |
| 8 | magic `S K T 1 \0 \0 \0 \0` |
| 4 | `u32` record count |

Each record follows:

| bytes | field |
| --- | --- |
| 4 | `u32` name length in bytes (at most 4096) |
| n | name, UTF-8, no terminator |
| 4 | `u32` dtype: 0 = `float32`, 1 = `int32` |
| 4 | `u32` number of dimensions (0 to 8; 0 is a scalar) |
| 8 x ndim | `i64` sizes, outermost first |
| 4 x numel | the data, row-major (C order), the same order as a contiguous PyTorch tensor |

Records can appear in any order. The loaders look tensors up by name, check the dtype and
shape against the config, and refuse the file if either is wrong. Extra records are ignored.

## Names

Parameter names are the PyTorch `state_dict` keys of the source classes, so
`refinement.heads.W_Q.weight` is the tensor `model.state_dict()["refinement.heads.W_Q.weight"]`
of `SuperpositionedBURTIMMA`, and `route6.route_transforms.2.bias` is the same key of
`AlexNetMMEP`. Linear weights are `[out, in]` and conv weights are `[out, in, kh, kw]`, as
in PyTorch, so nothing is transposed on the way in.

Next to the parameters, a file can carry scalar config records. If one is missing, the
loader uses the default given here.

Superposition (`SuperpositionedBURTIMMA`):

| record | dtype | default |
| --- | --- | --- |
| `config.d_model` | i32 | 768 |
| `config.num_paths` | i32 | 4 |
| `config.num_heads` | i32 | 8 |
| `config.head_dim` | i32 | 64 |
| `config.max_iterations` | i32 | 5 |
| `config.interference_threshold` | f32 | 0.1 |
| `config.collapse_temperature` | f32 | 1.0 |
| `config.orthogonality_reg` | f32 | 0.01 |
| `config.convergence_threshold` | f32 | 1e-4 |

`refinement.heads.path_confidence` must be present, because it is in the `state_dict`, but
the forward pass never reads it.

AlexNet-MMEP (`AlexNetMMEP`):

| record | dtype | default |
| --- | --- | --- |
| `config.in_ch` | i32 | 3 |
| `config.image` | i32 | 224 |
| `config.ch1` .. `config.ch5` | i32 | 96, 256, 384, 384, 256 |
| `config.fc` | i32 | 4096 |
| `config.num_classes` | i32 | 1000 |
| `config.num_routes` | i32 | 4 |
| `config.entropy_bound` | f32 | 0.2 |
| `config.penalty_weight` | f32 | 0.1 |
| `activation._alpha` | f32 scalar | 0.01 |
| `activation._beta` | f32 scalar | 1.0 |

`activation._alpha` and `activation._beta` are the SmoothLeaky buffers. They are part of the
PyTorch `state_dict` too, as 0-dimensional tensors, so they export like the parameters.
`config.image` only sets the size that `fc6` expects. Every image size from 223 to 254 gives
the 9216 inputs of the stock layer.

## Exporting a PyTorch state_dict

There is no exporter script in this repo. Writing one takes a few lines in any language that
can write bytes. Load the PyTorch model, call `state_dict()`, and for each entry write its key
as the name, dtype 0, its shape, and its values. Use `float32`, on the CPU, contiguous, as raw
little-endian bytes. Write the count of entries first and the magic before that. Add the
config records above if the model differs from the defaults. A file written by `sid_init` or
`amm_init` is a good reference to compare against byte for byte.

Going the other way, a reader that walks the records and builds a tensor of each shape gives
back a dictionary that `load_state_dict` accepts once the `config.*` keys are dropped.
