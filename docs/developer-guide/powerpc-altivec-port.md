# PowerPC / AltiVec (VMX) port

ncnn ships an architecture backend for 32-bit and 64-bit big-endian PowerPC
using the AltiVec (VMX) vector unit. The backend lives in
`src/layer/powerpc/` and is picked up automatically: every file named
`<layer>_powerpc.cpp` registers `Layer_powerpc` as an alternative
implementation of `Layer` alongside the generic one.

## Build

Two ways to enable it.

### Native / cross build with the provided toolchains

```sh
# 64-bit big-endian, plain AltiVec (POWER3/POWER4/PPC970/PA6T/Cell PPE)
cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/powerpc64-linux-gnu.toolchain.cmake ..

# 32-bit big-endian AltiVec (G4 7400 and newer)
cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/powerpc-linux-gnu.toolchain.cmake ..

# 64-bit big-endian POWER8 with VSX enabled
cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/powerpc64-linux-gnu-vsx.toolchain.cmake ..

# PlayStation 3 Cell PPE (bare-metal ps3-elf, no VSX)
# newlib ships no pthread.h, so the build must be single threaded
cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/powerpc64-ps3-elf.toolchain.cmake \
      -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF ..
```

### CMake options

| option | default | effect |
| --- | --- | --- |
| `NCNN_PPC_ALTIVEC` | `ON` on powerpc/ppc hosts | adds `-maltivec -mabi=altivec` |
| `NCNN_PPC_VSX` | `OFF` | adds `-mvsx` and enables the `#if __VSX__` fast paths |

`NCNN_PPC_VSX` must stay off on Cell PPE: PS3 silicon has no VSX.

## The ISA we target

`powerpc_usability.h` is deliberately written against the *baseline* AltiVec
instruction set (Altivec Technology Programming Environments Manual), because
plain VMX is what every PowerPC core with a vector unit implements:

* **no `vec_mul`, no `vec_div`, no `vec_sqrt`** – these are VSX-only intrinsics.
  Multiply is `vec_madd(x, y, zero)`, division and reciprocal square root use
  Newton-Raphson refinement built on `vec_re` / `vec_rsqrte`.
* **unaligned vector load/store does not fault but silently corrupts**.
  Every float access goes through `__ncnn_altivec_load` /
  `__ncnn_altivec_store`, which use `vec_ld`/`vec_st` only when the address is
  16-byte aligned and fall back to a `memcpy` otherwise.
* `vec_sel(x, y, mask)` selects whole elements, not bits.
* `transpose4x4_ps` composes `vec_perm` (not `vec_mergeh`/`vec_mergel`) so that
  pack4 <-> pack1 ungrouping is correct on big endian.

`powerpc_mathfun.h` provides the vectorized transcendentals
(`exp_ps`, `log_ps`, `pow_ps`, `sin_ps`, `cos_ps`, `tan_ps`, `tanh_ps`,
`erf_ps`, ...) implemented on top of those primitives.

### VSX

When `__VSX__` is available the same helpers switch to
`vec_mul`/`vec_div`/`vec_sqrt` and `vec_xl`/`vec_xst` for natural-alignment
access, and `powerpc64-linux-gnu-vsx.toolchain.cmake` targets POWER8. VSX is
backward compatible with VMX (the 64 VSRs overlay the 32 VMX registers plus the
FPR file) and needs no changes at the layer level: only the six helpers in
`powerpc_usability.h` have `#if __VSX__` branches.

64-bit double precision is not used anywhere: ncnn computes in fp32, and VMX128
(the 128-register Cell extension) is *not* targetable, since no released
toolchain (`ppu-gcc`, `powerpc64-linux-gnu-gcc`, `clang`) accepts `-mvmx128`
and binutils rejects `lvx128`/`stvx128`. `-mcpu=cell` therefore produces the
same code as `-mcpu=powerpc64 -maltivec`.

## Porting recipe

Every layer follows the same shape:

1. `support_packing = true` (plus `support_any_packing` where the semantics
   allow any pack) in the constructor, under `#if __ALTIVEC__`.
2. Vectorize the fp32 path that matters for that layer.
3. Everything else – int8, bf16, dynamic weight, an unsupported kernel or
   padding mode, an unexpected `elempack` – is delegated:

```cpp
if (bottom_blob.elempack != 1)
{
    Mat unpacked;
    convert_packing(bottom_blob, unpacked, 1, opt);
    return Convolution::forward(unpacked, top_blob, opt);
}
return Convolution::forward(bottom_blob, top_blob, opt);
```

and the output is repacked to pack4 when
`out.c % 4 == 0 && opt.use_packing_layout`.

Correctness over coverage: a layer that only delegates is still valuable,
because it keeps the packing contract intact so the rest of the graph can stay
packed.

Two gotchas worth repeating:

* `copy_make_border()` and the generic `Padding` are **not** packing aware.
  Every convolution-like layer carries its own `make_padding_pack4()` helper
  (see `convolution_powerpc.cpp`) and resolves `-233`/`-234`
  (SAME_UPPER/SAME_LOWER) itself.
* Weights are repacked in `create_pipeline()` into a
  `[maxk][inpack][outpack][4 in lane][4 out lane]` layout so the inner loop is
  four `vec_splat` + four `vec_madd` per 4x4 tile. The original `weight_data`
  is kept untouched for the fallback.

## Layer coverage

All 108 compute layers of ncnn have a PowerPC implementation (see
`src/layer/powerpc/`); only the two pure data-source layers, `Input` and
`MemoryData`, are intentionally left to the generic implementation.
Grouped by family:

* **Convolution family** – `Convolution`, `ConvolutionDepthWise`,
  `Deconvolution`, `DeconvolutionDepthWise`, and their 1D/3D variants,
  `InnerProduct`, `Packing`.
* **Elementwise** – `AbsVal`, `ReLU`, `Bias`, `Scale`, `Clip`, `Dropout`,
  `Eltwise`, `Threshold`, `HardSigmoid`, `HardSwish`, `Sigmoid`, `TanH`,
  `ELU`, `SELU`, `Swish`, `Mish`, `GELU`, `Erf`, `SoftPlus`, `BNLL`, `Exp`,
  `Log`, `Celu`, `Power`, `Shrink`, `GLU`, `PReLU`, `BinaryOp`, `UnaryOp`.
* **Normalization** – `BatchNorm`, `Softmax`, `LRN`, `RMSNorm`, `LayerNorm`,
  `GroupNorm`, `InstanceNorm`, `MVN`, `Normalize`.
* **Pooling / reduction** – `Pooling`, `Pooling1D`, `Pooling3D`,
  `Reduction`, `CumulativeSum`, `StatisticsPooling`, `SPP`, `ArgMax`, `Diag`,
  `Fold`, `Unfold`.
* **Layout / metadata** – `Reshape`, `Squeeze`, `ExpandDims`, `Permute`,
  `PixelShuffle`, `Reorg`, `Flip`, `DeepCopy`, `CopyTo`, `Split`, `Noop`,
  `Concat`, `Padding`, `Slice`, `Crop`, `Tile`, `Flatten`, `ShuffleChannel`.
* **Cast / quantization** – `Cast`, `Quantize`, `Dequantize`, `Requantize`.
* **Geometry** – `Interp`, `GridSample`, `DeformableConv2D`, `ROIAlign`,
  `ROIPooling`, `PSROIPooling`.
* **Attention / RNN** – `MatMul`, `Einsum`, `MultiHeadAttention`, `SDPA`,
  `RotaryEmbed`, `Embed`, `RNN`, `LSTM`, `GRU`.
* **Audio** – `Spectrogram`, `InverseSpectrogram`.
* **Detection** – `PriorBox`, `Proposal`, `DetectionOutput`,
  `YoloDetectionOutput`, `YoloV3DetectionOutput`.

`Input` and `MemoryData` intentionally have no PowerPC implementation: they
only produce data from the outside world (a `Mat` handed to `extract`, or a
weight blob), there is no per-element arithmetic to vectorize, and they are
already `support_packing`-free by construction.

## PlayStation 3 / Cell

The PPE of the Cell Broadband Engine *is* a PowerPC64 core with a full AltiVec
unit (32 vector registers), so the whole backend runs on PS3 unchanged with
`toolchains/powerpc64-ps3-elf.toolchain.cmake` (`-mcpu=cell -maltivec`).

Two extra steps are needed to run the resulting ELF:

1. The bare-metal ELF produced by `powerpc64-ps3-elf-gcc` has empty import stub
   metadata. Run the SDK `sprxlinker` on it after linking to generate the
   import table (PSL1GHT does this in its `%.self` rule).
   On hosts older than x86-64-v3, the SDK host tools can be run through Intel
   SDE: `sde64 -hsw -- sprxlinker app.elf`.
2. The SPU (Synergistic Processing Unit) is a *separate* ISA with 128 vector
   registers and dual issue; it is not reachable from the PPE instruction set.
   SPU kernels are built with `spu-gcc`, embedded with `bin2s`, loaded through
   `sysSpuImageImport`, and driven over MFC DMA. MFC transfers must be a
   multiple of 16 bytes and the effective address must be truncated to 32 bits.

## Testing

The whole test suite runs under qemu:

```sh
# 64-bit plain AltiVec
qemu-ppc64 -L /usr/powerpc64-linux-gnu ./tests/test_convolution

# 64-bit VSX
qemu-ppc64 -cpu power8 -L /usr/powerpc64-linux-gnu ./tests/test_convolution

# 32-bit, must select a G4-class core for AltiVec
qemu-ppc -cpu 7400 -L /usr/powerpc-linux-gnu ./tests/test_convolution
```

`test_layer` compares every layer against the naive generic implementation with
a small epsilon, and is run for every packing / bf16 / sgemm / winograd
combination, which is what catches the unaligned-store and packing-contract
mistakes.
