# PT2 validation

The current partial **P0+** implementation includes archive/schema/tensor checks,
fail-closed local-effect normalization, CLI validation, limited generated-Python
runtime guards for bare-symbol input contracts and a focused native ncnn smoke
test. It does not complete consumed-entry DEFLATE, restricted pickle decoding,
full PyTree reconstruction, a Transformers 5 model matrix, the full P2 operator
validation suite or a PT2 re-export round trip; full prior operator support and
release qualification are not claimed.

The [PT2 compatibility contract](../README.md#pt2-compatibility) defines the
supported subset. Source and test definitions describe implemented checks, not
execution results. The [Windows validation record](pt2-validation-results.md)
identifies the tested working tree, converter binary and exact selections;
it is not a cross-platform or full-model-suite qualification.

## Producer and converter are separate

Select the Python producer and linked LibTorch explicitly. Their release numbers
alone do not establish compatibility: archive version **0**, schema **8.20
exactly**, and an `aten` opset equal to the linked
`torch::jit::getMaxOperatorVersion()` are separate requirements. This operator
version is neither a PyTorch release number nor an ONNX opset; no upgrade or
downgrade adapter is implemented. Match the producer's serialized contract to
the reader and linked libraries, rather than assuming a supported release range.

| Selection | What it controls |
|----------|------------------|
| `Python3_EXECUTABLE` | Python exporter, test dependencies and generated-model execution |
| `Torch_INSTALL_DIR`, `Torch_ROOT`, `Torch_DIR` | CMake's LibTorch headers, libraries and operator schemas |
| Windows `PATH` and DLLs beside executables | Native libraries actually loaded by the converter and C++ tests |

Without an explicit Torch selection, CMake may discover Torch through Python.
Use a fresh build directory when changing installations: cached `TORCH_LIBRARY`,
`c10_LIBRARY`, `Torch_DIR`, `Torch_ROOT` and feature probes can otherwise retain
the old installation. Match converter headers, linked libraries and runtime-loaded
libraries; changing Python alone does not reconfigure the converter. Install
producer-side dependencies such as `packaging`, NumPy and compatible optional
model libraries in the selected Python environment.

On Windows, use the linked converter's Torch DLL directory in `PATH` and check
for stale DLLs beside executables. Header/toolchain incompatibilities, missing
dependencies, DLLs and entry-point errors are environment/build failures, not
expected PT2 rejections. Native crashes require diagnosis even if their output
also contains an unsupported-feature diagnostic.

Legacy tensor-dictionary loading is compiled only when
`PNNX_TORCH_HAS_PICKLE_LOAD` succeeds. A producer with `torch.export.save` does not
imply that this converter feature exists. Record the actual feature-probe result
and legacy test inventory separately; neither establishes a successful test run.

### Configured CI matrix, not execution evidence

The [PT2 contract workflow](../../../.github/workflows/pnnx-pt2.yml) configures
Ubuntu, Windows and macOS crossed with Python producer Torch **2.12.1** and
**2.13.0**, while the converter links **LibTorch 2.12.1** in every job. Separate
producer/linked environments and an explicit `Torch_INSTALL_DIR` keep those
roles distinct. Every job selects `pt2_frontend` and the TorchScript/PT2 ReLU,
Linear and Conv2d smoke cases. Linux jobs also install the Python ncnn dependency
and select `pt2_backend`; registration still requires both the producer capability
and `import ncnn` probes, and an empty selection is an error, not a pass. Logs and
the CMake cache are uploaded as artifacts. The workflow does not run the full
`pt2_operator` suite or a Transformers 5 matrix, and its configuration alone is
not evidence of successful jobs or release-wide version compatibility.

## What the current frontend checks

* **Archive:** STORE-only consumed entries, CRC32 on every consumed record
  (including empty records and a stored CRC of zero), ZIP/ZIP64 physical bounds,
  directory counts, and local/central header consistency. Encryption and
  duplicate entry names are rejected. `StoreZipReader::get_file_compression()`
  and `get_file_flags()` let the reader check consumed records before allocating
  their advertised payloads. Unused compressed attachments can remain indexed;
  they are not decompressed, read or CRC-verified. DEFLATE decoding is absent.
* **Tensor contract:** CPU strided raw storage on little-endian hosts only;
  missing `byteorder` means little-endian, while big/unknown markers are rejected.
  Sizes, nonnegative strides and offsets must fit the storage and checked
  arithmetic. Scalar shape `()` has one element; zero-sized dimensions have
  none. Non-zero offsets, zero strides and shared-storage views are accepted
  within bounds and materialized densely.
* **Dtypes:** bool, uint8, signed int8/16/32/64, float16/32/64, bfloat16 and
  complex32/64/128 are representable. Sparse/quantized/float8 payloads and wider
  unsigned dtypes are not implicitly accepted. The generated Python loader
  checks byte counts and reconstructs bfloat16/complex32 from their storage bits,
  including scalar/empty values. It does not call `net.float()` to downcast
  weights. Verify dtype and shape as well as values; ncnn/ONNX support is a
  separate backend concern.
* **Boundary:** flat positional tensor inputs, empty kwargs; a single
  tensor/numeric scalar or a nonempty flat tuple of tensor/scalar output leaves.
  A singleton tensor tuple may be returned as a tensor. Full reconstruction of
  nested/list/dict/namedtuple boundaries is not implemented. Scalar leaf support
  is not a promise of arbitrary symbolic scalar evaluation.
* **Input contracts:** [the importer](../src/load_exported_program.cpp) retains
  dtype, rank, static dimensions, bare symbols, ranges and shared-symbol equality
  in reserved `__pt2_input_*` parameters on `pnnx.Input`. Conversion samples do not
  replace the original contract. [Python generation](../src/ir.cpp) emits explicit
  `ValueError` checks before computation for tensor type, CPU device, strided
  layout, dtype, rank and those dimensions/constraints. They remain active under
  Python `-O`/`-OO`; shared-symbol bindings are local to each call. **Raw ncnn
  models do not enforce these checks.** Derived input expressions and range
  constraints are rejected even with hints. This is not general dynamic
  scalar-expression evaluation or unrestricted dynamic execution.
* **Guard normalization:** [schema/default normalization](../src/exported_program_defaults.cpp)
  accepts only a concrete, proven-true boolean `aten._assert_scalar`, or a
  statically verified `aten._assert_tensor_metadata`. The latter requires an
  already-defined tensor with known dtype, CPU/strided metadata and static
  sizes/strides; hints are not proof. Explicit size, stride, dtype, device and
  layout predicates must match; omitted/`None` predicates add no check. Guarded
  graph inputs must be static float32; known intermediate dtypes are not subject
  to that restriction. The importer cross-checks the operand's dtype/shape before
  removing the guard. An explicit stride predicate on a direct user input is
  retained as a generated-Python stride check, not a native ncnn guard. False,
  unresolved, runtime-dependent or other guard operators are rejected.
* **Factory dtype restoration:** [the importer](../src/load_exported_program.cpp)
  fills an omitted/`None` dtype for exactly `aten.full.default` and
  `aten.full_like.default` from their output tensor metadata. It does not coerce
  `fill_value`, infer dtype from `self`/another tensor or override an explicit
  dtype. Without output metadata it retains the original default; an unsupported
  output metadata dtype is rejected. This preserves the producer's recorded dtype
  rather than relying on the generated Python runtime's default dtype.

### Local-effect normalization

[Whole-program schema/default normalization](../src/exported_program_defaults.cpp)
uses explicit dispatcher overload pairs, not a generic trailing-underscore
rewrite. A value-writing target must be a whole, single-use, unaliased local
tensor from a known allocator. Use counting includes nested references and
graph/signature returns; only the mutation result may escape. External inputs,
parameters, buffers and constants cannot be value-write targets. Views, existing
aliases (even unused ones), unknown roots, reused targets and escaping original
targets fail closed. A missing schema alias annotation alone is not proof of
fresh ownership.

* **Known allocators:** `clone.default`, `empty.memory_format`,
  `zeros.default`/`ones.default`, `new_empty.default`/`new_zeros.default`/
  `new_ones.default`, `add`/`sub`/`mul`/`div` Scalar/Tensor overloads, and the
  functional pointwise/fill counterparts below. `contiguous`, `_unsafe_view`,
  RNG factories, containers and arbitrary pure operators do not establish local
  ownership for this proof.
* **Pointwise/fill whitelist:** default overloads of `relu_`, `relu6_`,
  `hardtanh_`, `hardsigmoid_`, `hardswish_`, `silu_`, `sigmoid_`, `tanh_`,
  `leaky_relu_`, `elu_`, `celu_`; `clamp_` default/Tensor; and `fill_`
  Scalar/Tensor. Target/result metadata must agree when both are present. Proven
  functional results can establish fresh ownership for a subsequent single-use
  operation.
* **Safe arithmetic:** `add_`/`sub_`/`mul_`/`div_` Scalar/Tensor overloads also
  require static CPU/strided target/result metadata, float16/32/64 or bfloat16
  `self`, and unchanged result dtype. `other` must be a concrete real scalar or
  a static same-dtype tensor that broadcasts without changing `self`'s shape;
  scalar `alpha`, where present, must also be concrete and real. Integer targets,
  mixed tensor dtypes, complex or unresolved symbolic scalar operands and missing
  proof are rejected rather than changing in-place casting semantics.
* **Separate constant-detach exception:** a signature-authorized `TensorConstant`
  may pass through exact `aten.lift_fresh_copy.default` to
  `aten.detach_.default`. The lift needs static CPU/strided metadata with matching
  sizes/dtype and `requires_grad=false` for source, copy and constant root. Only
  direct constants, proven copies and non-view `detach` aliases retain this
  permission; the target/result/root must be no-grad and recorded target/result
  metadata must agree. `detach_` becomes `detach`, not a value-writing allocator.
  User-input/parameter/buffer roots and copies, views and arbitrary fresh tensors
  never qualify. Authorized constant-derived tensors still cannot be value-write
  targets.

This is not general functionalization or alias-update support. See the
[intentional expected-unsupported cases](#intentional-expected-unsupported-cases).

### CLI sample contract

[The CLI](../src/main.cpp) validates `inputshape`/`inputshape2` and NumPy
`input`/`input2` counts, dtypes, ranks, static dimensions, ranges and shared
symbols. Omitted shape dtype suffixes retain the exported dtype; explicit
suffixes and NumPy dtypes must match it, not request a cast. `input` conflicts
with `inputshape`, and `input2` with `inputshape2`. NumPy payloads are read and
validated, not executed or used for value specialization; second samples do not
enable dynamic execution. These checks are separate from the implemented
generated-Python runtime contracts.

[The NumPy reader](../src/utils.cpp) accepts bool, uint8, signed int8/16/32/64,
float16/32/64 and complex32/64/128 descriptors, normalizing byte order and
Fortran order when loading data. A producer's NumPy may not expose every accepted
descriptor as a native dtype. Bfloat16, object/structured and wider unsigned
descriptors are unsupported; raw PT2 tensor dtype support is a separate contract.

PT2 explicitly rejects nonempty `customop` and `moduleop`, and any `device` other
than `cpu`, before import/output creation. Diagnostics identify the unsupported
option (`pt2 customop is not supported`, `pt2 moduleop is not supported`, or
`only device=cpu is supported`); empty `customop=`/`moduleop=` remain allowed.

### Resource and trust limits

| Bound | Current hard limit |
|-------|--------------------|
| Aggregate raw tensor loading/materialization | 512 MiB: unique storage bytes plus every dense view, across weights and constants |
| Individual consumed archive record | 512 MiB, also bounded by physical archive/container sizes |
| Individual dense attribute | 512 MiB, including expansion from tiny zero-stride storage |
| ZIP central directory | 256 MiB |
| ZIP entry count | 1,048,576 (1 Mi entries) |
| Cumulative ZIP entry-name bytes | 64 MiB |

The raw reader plans both dictionaries before reading any storage. Shared
storage is counted once, but every materialized view is counted. These limits
can reject otherwise valid large models and are **not a whole-process memory
cap**: JSON objects, ZIP index objects, temporary copies, LibTorch, later IR
passes and backend allocations can add memory beyond these budgets.

The legacy path uses LibTorch `torch::pickle_load`, then checks the resulting
tensor dictionaries and budgets retained/dense copies. Pickle's own allocations
occur before those checks. It accepts **trusted inputs only** and is neither a
restricted pickle implementation nor a safe sandbox. Rejecting raw
`use_pickle` payloads/custom objects does not extend that safety boundary to
legacy archives. Run untrusted-input investigations in an externally isolated,
resource-limited process; these checks are not a security certification.

## Test selection and interpretation

The source of truth is [the test registration](../tests/CMakeLists.txt).
CTest labels are exact tags; anchor label expressions to avoid mixing groups.

| Label | Current meaning |
|-------|-----------------|
| `pt2_frontend` | Native ZIP/JSON/schema/effect/import/IR tests, harness checks, archive/dtype/parity/shape tests, generated-Python guards and CLI contracts; not a backend validation claim |
| `pt2_operator` | PT2 variants registered by `pnnx_add_test` without `FRONTEND`; operator conversion/generated-PNNX comparisons, including diagnostic-based expected failures |
| `pt2_backend` | `test_pnnx_exported_program_ncnn`: native ncnn execution through Python bindings, conditional on both producer capability and a successful Python `import ncnn` probe |
| `pt2` | Umbrella label on frontend, operator and available backend registrations; not proof of all PT2 features |
| `torchscript` | Separate TorchScript regressions; a frontend parity test may also exercise TorchScript internally |

Python PT2 registrations depend on the selected producer's callable
`torch.export.save`, not the linked LibTorch release. Only format-aware scripts
or explicit `PT2_ONLY` registrations get variants through `pnnx_add_test`;
dedicated guards, CLI, archive, dtype and backend tests are registered directly.
Optional dependencies and the pickle feature probe can change the inventory.
Native frontend tests remain registered independently of the producer probe.
A suffix or a green CTest result does not imply conversion of every case:
expected unsupported cases are included.

Inspect configured registrations with `-N` and select groups separately with
anchored labels such as `-L '^pt2_frontend$'`, `-L '^pt2_operator$'`,
`-L '^pt2_backend$'` and `-L '^torchscript$'`. `-L '^pt2$'` selects the combined
PT2 inventory. Use `--output-on-failure --no-tests=error` for validation and
record each exit code. A missing backend registration means the dependency
gate was not satisfied, not that the backend suite passed.

### Focused test definitions

* [Python guards](../tests/test_pnnx_exported_program_guards.py), registered as
  `test_pnnx_exported_program_guards`, exercise valid and invalid contracts at
  converter `optlevel=0/1/2` and Python optimization levels `0/1/2`, including
  shared symbols, specialization, dtype/layout/device/rank checks and explicit
  versus omitted stride predicates. They also check that PT2 contract parameters
  are absent from native ncnn output.
* [CLI contracts](../tests/test_pnnx_exported_program_cli.py), registered as
  `test_pnnx_exported_program_cli`, cover shape/dtype and NumPy samples, second
  samples, unsupported options, malformed archives, early rejection without
  altering outputs, and save-stage failures. They do not run native inference.
* [Importer contracts](../tests/test_load_exported_program.cpp) include static
  metadata guards, known-allocator single-use pointwise/fill normalization and
  omitted factory dtype restoration, with negative cases for mismatched metadata,
  aliases, escaping/reused targets and other writes.
* [Effects contracts](../tests/test_exported_program_effects.cpp), registered as
  `test_pnnx_exported_program_effects` under `pt2_frontend`, cover safe arithmetic
  dtype/broadcast proofs, constant lift/detach provenance, no-grad restrictions,
  normalization idempotence and rejection before IR construction. This is a
  separate native target, not an operator-parity result.
* [Native backend regression](../tests/test_pnnx_exported_program_ncnn.py) compares
  eager PyTorch, generated PNNX Python and native `ncnn.Net` on the same batch-one
  float32 input through Conv2d, ReLU, adaptive pooling, flatten and Linear. It
  checks dtype, shape, finite values, `atol=rtol=1e-4`, required artifacts and
  byte-identical PNNX/ncnn graph and weight artifacts from repeated conversion.
  This focused test is neither native guard coverage nor a full backend/model matrix.

### Intentional expected-unsupported cases

Compared with the original test expectations, these four cases now explicitly
expect **converter rejection** through `unsupported_by_pnnx_pt2`. They are new
expected-unsupported classifications, not restored successful conversions and
not an exhaustive list of all PT2 limitations. Their TorchScript variants retain
the normal conversion/parity expectation.

| PT2 case | Intentional fail-closed limitation | Required diagnostic |
|----------|-----------------------------------|---------------------|
| [Tensor_fill](../tests/test_Tensor_fill.py) | Writes to caller-owned inputs and their views, not single-use local temporaries | `unsupported alias write/mutation of argument self` |
| [Tensor_slice_copy](../tests/test_Tensor_slice_copy.py) | Multiple live slice aliases require alias-update lowering, even though roots were cloned | `unsupported alias write/mutation of argument self` |
| [Tensor_index_put](../tests/test_Tensor_index_put.py) | Accumulating `index_put_` and old-root uses require general functionalization | `index_put_ (torch.ops.aten.index_put_.default): unsupported alias write/mutation of argument self` |
| [torch_masked_select](../tests/test_torch_masked_select.py) | Output sizes/guards depend on tensor values, not bare input-shape symbols | `unsupported guard; only a concrete boolean can be evaluated` |

A green negative test means the intended diagnostic was observed without a
crash or timeout, not that the operator/model converted. Local arithmetic,
constant-detach and factory-dtype fixes therefore do **not** establish full
restoration of prior operator support. The full operator suite still requires
a fresh run against the current sources and inventory.

### Failure phases, timeouts and artifacts

* Identify exactly one expected failure phase: producer export
  (`unsupported_by_torch_export`) or converter import/conversion
  (`unsupported_by_pnnx_pt2`), with a nonempty, specific diagnostic. An arbitrary
  exception or nonzero exit is insufficient. Unexpected success must also fail
  the negative test.
* Converter crashes, signals/Windows exception statuses, launch failures and
  timeouts must fail even if their output contains the expected diagnostic.
  Never hide them with broad exception handling, blanket skips or looser
  comparisons. Keep command, exit status, stdout and stderr for diagnosis.
* The helper's per-converter timeout is `PNNX_TEST_TIMEOUT` (default 300 seconds,
  or an explicit positive finite timeout). C++ CTest cases use 120 seconds;
  Python cases use `PNNX_TEST_CASE_TIMEOUT` (default 1800 seconds). A timeout is
  not an expected unsupported-feature result.
* Keep producer versions, build trees, run logs and generated artifacts
  separate. The helper uses format-specific export/output prefixes, removes
  stale outputs/Python caches, and CTest locks TorchScript/PT2 pairs that share
  auxiliary fixtures. Use a separate directory/prefix for manual cases as well.
  Archive/dtype tests use temporary directories; preserve a focused reproducer
  separately when investigating failures.
* Compare eager and generated-PNNX dtype, shape and values with the test's
  original tolerances. Backend load/inference/numerical failures are separate
  from successful frontend conversion. Broader backend and scalar-output checks
  require their own assertions; do not infer them from tensor-only helper coverage.

## Evidence-record template (not results)

Copy this template into a separate run report; leave unmeasured fields blank.
Publish sanitized logs as shareable CI/PR artifacts, not links to ignored local
build trees. Do not combine counts from different revisions, builds or inventories.

| Field | Record for one run |
|-------|--------------------|
| Source | Revision, working-tree changes, date |
| Environment | OS/architecture, compiler, CMake, build configuration |
| Producer | Python/PyTorch versions, model-library versions and dependencies |
| Converter | Linked Torch version, header/library selection, runtime-loaded libraries, feature-probe results |
| Inventory | Exact registered test names per label, dependency-gated omissions |
| Invocation | Build/test commands and each exit status |
| Outcomes | Per-case export, conversion, generated-Python and native-backend results; exact expected-rejection diagnostics |
| Evidence | Shareable logs, reproducer/artifact identifiers, dtype/shape/tolerances and errors |

Registration alone supplies no pass totals. A full operator/backend result,
cross-platform qualification or producer matrix requires its own complete record.
Keep earlier failing runs separate from subsequent reruns: the validation record
below the linked contract is scoped to the stated binary and test selections,
not every configured model or platform.

## Follow-up, not completed by P0+

* Native ncnn shape/range/shared-symbol guards and general dynamic scalar or
  derived-expression semantics beyond the limited generated-Python checks.
* Restricted legacy pickle decoding with allocation limits before deserialization.
* Bounded DEFLATE support for consumed entries.
* Full PyTree input/output reconstruction, including kwargs and nested containers.
* A full Transformers 5 model/producer matrix, including clearly diagnosed
  unsupported exports, plus broader PyTorch producer-version and platform coverage.
* Completion of the full P2 operator validation suite, general mutation/alias
  functionalization beyond the whitelist, and PT2 re-export round-trip validation.
* Broader backend execution coverage beyond the registered native regression,
  and reproducible full-suite validation with separate
  TorchScript/frontend/operator/backend evidence before release claims.

## Provenance

Inspired-by: [PR #6933](https://github.com/Tencent/ncnn/pull/6933),
[PR #6941](https://github.com/Tencent/ncnn/pull/6941),
[PR #6946](https://github.com/Tencent/ncnn/pull/6946), and
[PR #6953](https://github.com/Tencent/ncnn/pull/6953).
These references acknowledge related PT2 design and implementation discussion,
not feature-completion or validation evidence. Current behavior is defined by
the sources and tests linked above. Repository licensing and existing
copyright/SPDX notices remain unchanged; see [../../../LICENSE.txt](../../../LICENSE.txt).