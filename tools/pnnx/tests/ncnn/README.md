# tests/ncnn notes

Tests in this directory run the full ts (torchscript) + pt2 (torch.export) paths
and compare pnnx->ncnn inference against PyTorch.

## transformers version matrix

Some models only exist in a specific `transformers` major version, and a few
legacy models use APIs that were removed in 5.x. There is **no single
transformers version** that can run every test, so a version matrix is required
(no test is permanently skipped; each test really runs under a compatible
version):

| transformers | tests that really run |
|---|---|
| 4.x (e.g. 4.48.3) | the 31+ legacy models (all version-guarded) |
| 5.x (e.g. 5.16.1) | `deepseek_v3`, `qwen2`, `qwen3` (only exist in 5.x) + all 37 top-level transformers tests (the 7 whose APIs changed in 5.x are dual-version adapted: `clip`, `ctrl`, `distilbert`, `layoutlm`, `longformer`, `openai`, `xlnet`) |

All tests guard on `transformers.__version__` and/or `torch.__version__` and are
adapted to run for real under both 4.x and 5.x (no test is permanently skipped;
a version-incompatible test is skipped only in the job where that version cannot
run it).

## Known skips (tests/ncnn full regression)

The skips are **pt2-path conversion gaps only** (the `test()` ts path itself
passes, `EXIT=0`); they are not failures. After the linalg/linear/reshape passes
were added, `test_nn_Linear`, `test_vit_b_32` and
`test_nn_MultiheadAttention` pt2 all pass. See the latest full regression log
(`/tmp/reg8.log`) for the current skip list.

## New ncnn conversion passes

- `convert_aten_baddbmm` (`torch_baddbmm.cpp`): `aten::baddbmm` 3-input variant
  (self, batch1, batch2) -> `MatMul` + `BinaryOp(add)`.
- `convert_Tensor_to` (`convert_Tensor_to.cpp`): eliminate `Tensor.to(i64->f32)`
  by walking back the shape-only chain (unsqueeze/expand/reshape/view/permute/
  transpose/contiguous/squeeze/flatten) to an integer `pnnx.Attribute`, convert
  the data to f32 and reconnect the chain tail. Required because ncnn has no
  Cast support for i64->f32.
- `convert_aten_new_empty` (`convert_aten_new_empty.cpp`): `aten::new_empty` ->
  zeros `pnnx.Attribute` (new_empty is an uninitialized concat buffer, e.g. MLA
  kv concat; following `slice_copy` fully overwrites it).
- `convert_Tensor_slice` step>1 expansion: single-axis strided slice on the last
  dim -> `reshape(-> K*step)` + `Crop(start)` + `reshape(-> K)`. The Crop axis is
  chosen dynamically (dims=3 -> w is axis 2; dims=4 -> w is axis 3) based on
  whether the batch dim is stripped.
- `convert_aten_linalg_vector_norm` (`linalg_vector_norm.cpp`): resolve constant
  p/dim/keepdim that come through `pnnx.Expression` (weight_norm expansion) and
  rewrite to `torch.norm` for the Reduction conversion.
- `ir.cpp` test_inference generation: when the user-provided `inputshape` count
  does not match the graph input count (unused inputs such as CLIP's
  `casual_mask0` were dropped during passes), fall back to the shape-inferred
  shapes so the generated python test inputs stay aligned.
