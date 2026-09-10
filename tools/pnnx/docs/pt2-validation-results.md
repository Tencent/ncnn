# Windows PT2 validation record

## Scope and provenance

These are local execution results for the P0+ working-tree changes based on
`a11a4680` (`ci: run exported program test suites`), not results for that commit
alone. They cover the current static/bare-symbol contract and documented
fail-closed boundaries, not the whole original PT2 roadmap.

| Component | Tested configuration |
|-----------|----------------------|
| Platform | Windows x64, Visual Studio 2026, Release |
| Python producer | Python 3.12.10, Torch 2.13.0+cpu |
| Converter | LibTorch 2.12.1; matching 2.12.1 runtime DLLs |
| Dependencies | NumPy 2.5.2, packaging 26.3, ncnn 1.0.20260526 |
| Converter SHA-256 | `D48B0870BD42291BA9299BE6C695007DAC9BB0046B21DA474A3FE85B780E8926` |

The converter and all native test targets built successfully. Test execution
used single-thread OpenMP/MKL settings and `MKL_ENABLE_INSTRUCTIONS=SSE4_2`.
Python packages were installed in the selected virtual environment; no source
commit or push was performed as part of validation.

## Final selections

All commands used CTest Release configuration, `--output-on-failure` and
`--no-tests=error`.

| Selection | Result | Wall time |
|-----------|--------|-----------|
| `-L '^torchscript$' -R '^test_(F_\|nn_\|Tensor_\|torch_)' -j 8` | 310 passed, 0 failed | 315.73 s |
| `-L '^pt2_operator$' -R '^test_(F_\|nn_\|Tensor_\|torch_)' -j 8` | 310 passed, 0 failed | 446.53 s |
| `-L '^pt2_(frontend\|backend)$' -j 4` | 15 passed, 0 failed | 68.96 s |

The frontend/backend selection comprises 14 frontend registrations and one
native ncnn test. It includes ZIP/JSON/schema/import/effects/IR checks, the
25-case mocked helper suite, real archives, dtype and stride/offset handling,
Python input guards, CLI validation and native inference. The latter supplies
the same deterministic input to eager Torch, generated PNNX Python and actual
`ncnn.Net`, checks dtype and shape before numerical comparison, and verifies
repeat-conversion parameter/weight byte equality.

**A passing PT2 test is not always a converted model.** Existing exporter
limitations and the documented expected rejections are part of the 310-test
selection. In particular, external `fill_`, accumulating `index_put_`, complex
slice-alias updates and data-dependent `masked_select` guards are explicitly
rejected and checked against diagnostic needles. Their TorchScript counterparts
retain numerical comparisons. Crashes and timeouts cannot satisfy these
expected-failure contracts. See [the exact boundaries](pt2-validation.md#intentional-expected-unsupported-cases).

## Not verified by this record

- The complete transformers, torchvision, torchaudio and graph-pass/model suites.
- General mutation functionalization, nested PyTree reconstruction or arbitrary
  dynamic scalar/control-flow semantics.
- ONNX runtime numerical parity (this build has no ONNX frontend/output support).
- Legacy pickle loading (the local compile-time feature probe is unavailable).
- The Linux/macOS or alternate-producer CI jobs. The separately configured
  workflow is not evidence that those jobs have executed successfully.
- Raw ncnn runtime enforcement of the Python-only input guards, or native
  inference for every represented tensor dtype/operator.

The working-tree diff passed `git diff --check`; Python syntax and workflow YAML
parsing checks passed. This is not a sanitizer/fuzzing certification. Rebuild
and repeat the selections for a subsequent source revision, preserving the new
binary digest and CI artifacts rather than carrying these totals forward.