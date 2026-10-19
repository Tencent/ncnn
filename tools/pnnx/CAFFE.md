# Caffe frontend

Convert a deployment network with `pnnx model.prototxt`. The weights must be
named `model.caffemodel` in the same directory. There is no `caffemodel=` option.
The frontend uses the existing Torch dependency and does not need Caffe installed.
Generated Python follows the usual pnnx `Model().eval()` and `.pnnx.bin` loading
convention. Run Python in the output directory so the generated weight paths
resolve. PyTorch inference, rather than tracing the generated model again, is the
export contract.

## Stable runtime boundary

The Caffe frontend uses existing ncnn layers and Python bindings without runtime
changes or new native layer parameters. The current validation uses upstream
commit `b5721b4bd2c8d0a8b861477f72c515db6b9f6dd0`.

Unsupported parameter or graph structures fail conversion with a layer-specific
diagnostic. Data-dependent legacy limitations that cannot be established from
the graph require explicit warnings and documented input restrictions. A warning,
expected rejection, or successful conversion is not a numerical equivalence pass.

Earlier results using patched runtimes are historical only. The validation
status below records coverage against the unchanged upstream runtime.

## IR organization

The importer reads only the parameter message belonging to each layer type.
Top-level protobuf wrappers are omitted: `softmax_param.axis` becomes `axis`,
captured as `op_0.axis` by a level2 rewrite. Nested names such as
`nms_param.top_k` and `shape.dim` remain qualified. Interp field presence and
the legacy Concat/Slice axis aliases retain their source semantics.

Standard conversions use the existing target-oriented level2 files alongside
ONNX/TNN rules. ReLU (including leaky ReLU), Sigmoid, TanH,
Softmax, ELU, ReLU6, PReLU, Dropout, Flatten, Permute, Reshape, Convolution,
Deconvolution, InnerProduct, Pooling, BatchNorm, BN, Slice,
Tile, Reorg, Embed, ShuffleChannel, Interp and
across-channel LRN now use that path.
Sigmoid and TanH first map directly to `aten::sigmoid` and `aten::tanh` in the
source pass, then reuse the existing generic level2 rules; they no longer have
Caffe-specific level2 registrations.
Broadcast Scale builds Attribute/reshape/arithmetic nodes for static
and dynamic scale inputs, with or without bias. Eltwise preprocessing reduces
variable input counts to left-associated binary operations, preserving SUM
coefficients; the source-stage update places their basic arithmetic expansion
in the corresponding Caffe preprocessing files rather than level2 registrations.
Fixed Exp/Log, Threshold, BNLL, Power 1/2/3 and two-input Eltwise expansions
use source-local `GraphRewriterPass` patterns in the corresponding
`lower_exp_log`, `lower_threshold`, `lower_bnll`, `lower_power` and
`lower_eltwise` source/header pairs.
They run explicitly after the source traversal has validated parameters and
propagated shapes, not through global level2 registration. Variable-arity
Eltwise keeps its ordered procedural expansion; simple type mappings stay direct.
Each arithmetic family keeps its `prepare_*` parameter/shape preparation and
`lower_*` graph rewrite together. Other passes use matching source/header and
entry-point names; `parameter.h` contains only shared `get_*` accessors.
Crop has its own lowering, and LSTM is named explicitly. Rewrite variants
share code only within the same operator family; internal operand metadata is
assigned to explicitly named nodes without inferring ownership from shape.
Concat prepares the same list input convention as TNN before
the existing level2 cat rule. Caffe preprocessing still prepares output shapes
before downstream source operators are processed. Source-specific graph
preparation remains for normalization, reduction and recurrent state handling.
Their tests cover source-specific numerical semantics. Two-input LSTM continuation
requires a diagnostic: the stable native LSTM does not implement that protocol.

Slice prepares a list output and `prim::ListUnpack`, so its level2 rule has fixed
arity without restricting the number of Caffe tops. The existing unpack pass
restores the original output operands and their shape metadata.
Across-channel LRN uses the existing `F.local_response_norm` target; Caffe's
within-channel variant retains its separate spatial normalization preparation.
Power exponents 1, 2 and 3 use source-stage arithmetic expansion. Other exponents retain
the native Power path so negative-base semantics do not depend on BinaryOp's
log/exp approximation.

### Source-stage arithmetic normalization update

Caffe source preprocessing, following the existing ONNX frontend division, produces
`aten::*` operations with `prim::Constant` scalar values and `pnnx.Attribute`
tensor values, instead of directly constructing `pnnx.Expression` strings.
The existing level3 `fuse_expression` pass owns expression formation. Optimized
and final IR may still contain `pnnx.Expression`; this change does not prohibit
expressions or rewrite unrelated frontend implementations.

Spatial and module operations keep their existing `F.*` / `nn.*` mappings.
The special `caffe.Power` path for exponents other than 1, 2 and 3 is retained.
BNLL must preserve its stable softplus formula; Threshold must preserve strict
comparison and floating-point output. Broadcasting, dtype, batch metadata and
both exported execution paths remain part of the numerical contract.

Simple basic-operator mappings belong directly in the source stage; expansion
of Caffe-specific coefficients, broadcasting and stable formulas belongs in
the corresponding responsibility-focused `pass_caffe` files. The preceding
seven level2 aten arithmetic rewrite files are superseded by this organization,
not moved wholesale as registration classes. Meaningful target-semantic
`F.*`, `nn.*` and `Tensor.*` rewrites remain in level2; this is not a migration
of all conversions into the source frontend. Reduction, normalization and
within-channel LRN likewise build standard arithmetic nodes, not Expression strings.

The source-stage arithmetic refactor previously passed its selected suites with
a patched runtime. Those runs are not current baseline-runtime acceptance.
The weighted Eltwise -> Exp -> Log -> average Pooling -> Flatten fixture remains
a shape-propagation regression; Power, BNLL, Threshold, broadcasting, LRN and
Reduction retain independent semantic references. Fresh baseline results must
be recorded separately, including numerical and diagnostic outcomes.

### Existing frontend regression coverage

The shared frontend regression `test_tnn_level2` synthesizes weight-free TNN
models for channel/negative-axis Softmax and Concat followed by Softmax. It
compares generated PyTorch and ncnn outputs at fp16=0/1 (12 comparisons), using
TNN's full-dimensional batch convention. This is TNN coverage, not evidence
that the ONNX frontend was tested.

An independent ONNX-enabled build subsequently passed the existing Softmax,
Conv2d, Linear and BatchNorm2d ONNX-to-PyTorch tests at optlevel=2. It required
an isolated protobuf include-order adjustment and explicit ONNX protobuf
linkage for the official CPU ONNX Runtime release; this is functional regression
evidence, not a claim that this dependency combination builds without adaptation.
The test wrapper retained the legacy Torch ONNX exporter (`dynamo=False`), and
these four tests did not execute native ncnn outputs.

Source preparation is separated by responsibility: embedding index analysis
and weights, interpolation geometry, linear weights, affine parameters, tensor
shapes and elementwise parameters have dedicated `prepare_*` files. The source
entry point calls their operator-specific functions directly. The former
`lower_standard`, `lower_misc` and `lower_legacy` collection files are removed.

After level2, any remaining Caffe operator must be one of the explicitly
supported extensions with Python helpers and ncnn exporters. An unmatched
standard operator fails conversion before required output files are written.

## Compatibility audit

The legacy baseline is ncnn commit
`4366038c23ef44547b940ff5045145fd56392668`, specifically
`tools/caffe/caffe2ncnn.cpp` and `tools/caffe/caffe.proto`. The table inventories
the legacy conversion contract and its implemented lowering/export paths.
The coverage column states the numerical fixtures and supported boundaries;
it does not promise compatibility with unknown fork operators or arbitrary
unverified parameter combinations. SqueezeNet alone covers only a subset, so
the independent operator and extension tests are also required.

Ordinary rows originate in BVLC Caffe. SSD rows refer to the Caffe-SSD dialect.
Other extensions are recorded as fork provenance unknown unless the schema or
converter identifies it; matching a familiar layer name is not proof of origin.
Weights below describe Caffe blobs before ncnn-specific serialization.
The schema's single maintained location is now `src/caffe.proto`; the legacy
converter also consumes this file so standalone pnnx sources include the schema.

Scale's general `axis`/`num_axes` broadcasting is a semantic improvement over
the old converter, which did not serialize those fields. In particular, the
legacy dynamic Scale path (`scale_data_size=-233`) does not load bias weights.
The new dynamic-scale-with-bias tests therefore use an independent tensor
reference, not the old converter as a numerical oracle for that combination.

The stable DetectionOutput implementation retains its background-score decoding
shortcut. Nonnormalized confidences can violate that shortcut's assumptions;
the frontend must warn about this data-dependent boundary rather than claim a
runtime fix. Structural restrictions such as code type and shared-location
requirements must be checked before export.

Pooling's trailing-window correction follows
[BVLC Caffe PoolingLayer](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp):
padding in either direction enables correction in both directions. The
asymmetric-padding regression covers this case with stride larger than kernel.

After Caffe lowering, operator and operand names receive a stable indexed prefix
and an ASCII identifier spelling. This keeps distinct source names such as
`a/b` and `a_b` distinct in generated Python and prevents clashes with Python
keywords or `nn.Module` attributes. Import/lowering diagnostics retain original
names. The dedicated names regression verifies differently weighted modules,
branch operands, multiple inputs and names containing spaces or leading digits.

| Layer / aliases | Legacy parameters and weights to preserve | Intended IR / PyTorch; ncnn export | Coverage / remaining checks |
| --- | --- | --- | --- |
| Input, MemoryData | Ordered shapes; batch, channels, height, width; no weights | pnnx.Input; existing Input exporter | SqueezeNet Input, MemoryData, net-level declarations, shape overrides and multiple-input fixtures pass |
| Convolution, ConvolutionDepthwise, DepthwiseConvolution | OIHW weight, optional bias; group, dilation, rectangular kernel/stride/pad; aliases infer depthwise grouping | nn.Conv2d; existing convolution exporter | Ordinary/grouped/depthwise aliases, rectangular geometry, dilation and bias/no-bias fixtures pass |
| Deconvolution | IOHW weight, group and optional bias; old converter reorders output/input channels per group | nn.ConvTranspose2d; existing exporter | Default parameters, groups 1/2, depthwise, bias/no-bias, rectangular kernel/stride, unequal height/width padding and dilation 1/2 numerical tests pass |
| InnerProduct | Output-by-input weight and bias; axis/transpose need explicit semantic checks (legacy ignores these) | flatten + nn.Linear; existing exporter | Default axis and spatial axes 2/-1, transpose false/true numerical tests pass; spatial-axis fixtures also cover absent bias |
| BatchNorm | Mean, variance, scalar accumulated-statistics factor; epsilon, factor zero | nn.BatchNorm or affine; existing exporter | Factor 0/2 and Conv-BN-Scale fusion numerical tests pass |
| BN | Fork extension; learned scale and shift blobs, distinct from BatchNorm statistics | Affine; Scale or existing arithmetic exporter | Learned channel affine tests pass for rank2/3/4, including negative and zero scales; fork provenance unknown |
| Scale | Learned scale/bias or scale from second input; axis/num_axes broadcasting | Broadcast arithmetic; existing arithmetic exporter | Independent channel scale with default parameters and bias/no-bias, rank2/3 inputs, scalar/spatial broadcast, negative axis, num_axes=-1 and second-input bias/no-bias tests pass |
| ReLU, ReLU6, ELU, PReLU | Negative slope, clip [0,6], alpha; shared/per-channel PReLU slope blob | Existing activation operators/exporters | Inplace ReLU, ReLU6, nondefault ELU alpha and shared/per-channel PReLU tests pass |
| Pooling | MAX/AVE, rectangular geometry, global mode; ceil and edge-window denominator semantics | Pooling operators or exact Caffe helper; existing pooling exporter | SqueezeNet max/global average; odd/even boundary-window MAX/AVE, rectangular windows/strides and large MAX padding pass. Explicit padding plus floor pooling preserves Caffe's trailing-window geometry, including empty MAX windows when pad=0 |
| Concat | Axis and legacy concat_dim, negative axes | torch.cat; existing exporter | Channel and spatial axes, negative last axis and legacy concat_dim tests pass |
| Eltwise | PROD/SUM/MAX and ordered SUM coefficients | Arithmetic / maximum; existing exporters | All three operations with one, two and three inputs pass both exporters at fp16=0/1; weighted SUM includes negative coefficients and left-associated reduction |
| Flatten | No dedicated legacy parameter branch: default same-name ncnn Flatten executes the ordinary flatten case | torch.flatten; existing exporter | Partial axes 1 through 2 pass against independent Torch flatten reference |
| Reshape | Shape dimensions, 0/-1; legacy limited ranks and ignores axis/num_axes | Tensor.reshape; existing exporter | 0/-1, partial axis/num_axes and following Permute pass |
| Permute | Order; legacy handles supported non-batch 3D permutations | Tensor.permute; existing exporter | Reshape-plus-Permute fixture passes; moving batch is rejected, other permutations need coverage |
| Slice | Axis, explicit slice points or equal split | torch.split; existing exporter | Explicit/equal channel split, multiple outputs and Silence pass |
| Softmax | Axis, legacy axis adjustment for omitted batch | F.softmax; existing exporter | SqueezeNet channel softmax and negative last-axis tests pass |
| Dropout | Test-phase behavior; legacy scale_train option changes scaling | Identity or multiply; existing exporter | Both scale_train settings pass at dropout ratios 0, 0.25 and 0.75; SqueezeNet identity also passes |
| Silence | Legacy maps to Noop; intended consumed outputs must remain suppressed | Graph import handling and native sink | Numerical output-count test passes after native sink preservation |
| Split | No dedicated legacy parameter branch; same-name ncnn Split executes fanout | Identity branches with preserved output set | Duplicate terminal-output numerical and output-count tests pass |
| AbsVal, Sigmoid, TanH | No dedicated legacy parameter branch; same-name parameter-free ncnn operators | aten::abs then common fusion, F.sigmoid/F.tanh; existing exporters | FP32/FP16 regression passes after the arithmetic refactor |
| BNLL | Same-name parameter-free legacy operator; stable softplus | Stable aten arithmetic then common fusion; existing exporters | Ordinary inputs and values from -100 to 100 pass after the arithmetic refactor |
| Exp, Log | Legacy same-name fallback preserves defaults but does not serialize base/scale/shift | aten arithmetic preserves Caffe base/scale/shift before common fusion | Natural base, bases 0.5/2, default and nondefault scale/shift pass after refactor; invalid bases are rejected |
| Tile | Legacy fallback is identity (tiles=1); nondefault parameters are not serialized | Tensor.repeat; existing Tile exporter | Identity and channel/spatial repeats, negative axis; repeated batch is diagnosed as unsupported |
| Reduction | Legacy fallback preserves default SUM over all dimensions; other parameters were not serialized | SUM/ASUM/SUMSQ/MEAN via expressions and torch.sum/mean, with tail axis and coeff | Scalar outputs, downstream pointwise/branch operations, negative axis and rank1 input pass; non-singleton batch remains unsupported and missing input shapes are rejected |
| ArgMax | Legacy fallback preserves default flattened top-1 indices; nondefault parameters were not serialized | Existing ArgMax and layout adapters, or supported reduction paths | Native sorting requires logical element count divisible by 4; other counts are rejected with an allocation-padding diagnostic. Axis top-1 values via torch.amax do not have this restriction. Earlier logical-element-fix results are not baseline evidence |
| Power, Threshold | Power/scale/shift; threshold | Powers 1/2/3 use expressions; other powers use a Torch helper and native Power; Threshold uses expressions | Power passes 144 comparisons covering positive/negative integer powers, fractional powers, scale/shift and zero bases; NaN/Inf classifications checked separately. Threshold covers values below, equal to and above zero/nonzero thresholds. Nonfinite input tensors remain unverified |
| Crop | Axis, one or multiple offsets; reference tensor shape | Slicing; Crop exporter | Rank2/3/4 crop, including distinct per-axis offsets, passes; the shape-only reference remains a model input |
| LRN, MVN | LRN region/size/alpha/beta; MVN variance/across_channels/epsilon | Torch expressions; existing exporters | LRN both regions, sizes 1/3/5 and singleton spatial tests pass (144 comparisons, two diagnostics); MVN all rank4 variance/across_channels combinations and rank2/3 across_channels=true with variance on/off pass. Caffe WITHIN_CHANNEL ignores k and uses fixed shift 1, unlike ACROSS_CHANNELS |
| Embed | Input vocabulary size, output size; row-major table and optional bias | Embedding plus bias; Embed exporter | Rank1/rank2, bias/no-bias, shared input and nested Split paths pass: 72 comparisons plus three diagnostics, with documented integer input adaptation |
| LSTM | Recurrent num_output and gate weight/bias blobs | Single-input nn.LSTM with gate reorder and existing exporter | Two-input continuation, exposed states and unsupported batches require diagnostics; do not emit native parameter 4 |
| Interp | PSPNet Caffe; explicit output size, zoom/shrink and nonpositive padding | Crop + F.interpolate align_corners=True; existing Interp exporter | Explicit size, zoom/shrink/crop/singleton tests; legacy correction below |
| Normalize | SSD scale blob, across_spatial, channel_shared, epsilon | Normalize plus scaling; Normalize exporter | All across_spatial/channel_shared boolean combinations pass with nondefault epsilon for rank4; rank2/3 across_spatial=true/channel_shared=true pass with positive and negative shared scales |
| PriorBox | SSD min/max sizes, deduplicated aspect ratios, variances, flip/clip, image/step/offset | Caffe helper; PriorBox exporter | Centered box, multiple sizes/grid order, ratio flip/deduplication, image/step overrides, clipping and default/per-coordinate variances pass; real MobileNet-SSD also tested |
| DetectionOutput | SSD CENTER_SIZE, shared location, background 0 and existing NMS parameters | Caffe helper; existing DetectionOutput exporter | Structural restrictions are diagnosed; warn about legacy background-score assumptions and empty-layout conventions. Previous patched-runtime passes do not establish current coverage |
| Python ProposalLayer | Legacy feat_stride; fixed base, limits, NMS threshold and min_size | Helper and existing Proposal contract where representable | Native NMS is exclusive with legacy tie ordering and empty-output limitations. Warn on data-dependent differences; dynamic Proposal-to-ROI graphs are unsupported, not numerically passed |
| ROIPooling, PSROIPooling, ROIAlign | Pooled size, spatial scale and existing native parameters | Torch helper and existing single-ROI operators with supported static adapters | No roi_batch parameter. Dynamic ROI counts require rejection; clipped/boundary/rounding cases retain legacy limitations and need diagnostics or warnings, not runtime changes |
| Reorg, ShuffleChannel | Stride / group; no weights | Reshape and permutation; existing exporters | Reorg strides 1/2/3 and ShuffleChannel groups 1/2/3/6 pass, including identity and rectangular inputs; fork provenance remains unknown |
| YoloDetectionOutput, Yolov3DetectionOutput | Classes, box count, confidence/NMS, biases and v3 masks/anchors/scale | Caffe helpers; existing detection exporters | Nonempty supported cases require baseline numerical tests; legacy empty-output behavior requires warning and separate diagnostics, not a claimed zero-row runtime fix |

The legacy final fallback emits the original layer type even if it is unknown to
ncnn, and Python layers other than ProposalLayer emit their Python layer name.
This is not unlimited operator support. Known executable same-name cases above
remain in scope. Unsupported types must produce a layer-specific diagnostic in
the new frontend rather than a nominally successful unusable export.

The SSD [NormalizeLayer implementation](https://raw.githubusercontent.com/weiliu89/caffe/ssd/src/caffe/layers/normalize_layer.cpp)
checks that input rank is at least two in both setup and reshape. Its
across-spatial, channel-shared branch divides all non-batch elements by
`sqrt(sum(x*x)+eps)` and multiplies by a scalar scale. The legacy native
Normalize path preserves that formula for rank2/3 inputs as well as rank4;
the frontend therefore supports this specific low-rank combination.
Other low-rank flag combinations remain explicitly rejected, rather than
assuming that native and Caffe channel layouts agree.

### Interp behavior correction

The PSPNet Caffe [Interp layer](https://github.com/hszhao/PSPNet/blob/master/src/caffe/layers/interp_layer.cpp)
defines output geometry using the intervals between corner pixels: shrink gives
`floor((size - 1) / factor) + 1`; zoom gives `(size - 1) * factor + 1`.
Negative `pad_beg` / `pad_end` crop the source before interpolation.
The [interpolation utility](https://github.com/hszhao/PSPNet/blob/master/src/caffe/util/interp.cpp)
maps first and last output pixels to first and last effective input pixels,
equivalent to PyTorch `align_corners=True` (a singleton output samples the first
pixel). The legacy converter instead selected ncnn's default half-pixel sampling,
scaled dimensions directly for zoom and omitted shrink/crop semantics. New
conversion follows the Caffe source above; its interpolation regression uses an
independent Torch reference and deliberately does not reproduce that legacy bug.

## Running the Caffe regression suite

Use a Python environment containing CPU Torch, NumPy and ncnn bindings built
from this checkout. Install a C++ compiler, CMake, protobuf development headers
and libraries, and a compatible `protoc`. Build the retained `caffe2ncnn` using
the usual ncnn tools build. Caffe itself is not required. From the repository
root, configure a separate pnnx build directory:

```sh
cmake -S tools/pnnx -B build-pnnx-caffe \
  -DTorch_INSTALL_DIR=/absolute/path/to/python/site-packages/torch \
  -DPNNX_CAFFE2PNNX=ON \
  -DCAFFE2NCNN_EXECUTABLE=/absolute/path/to/caffe2ncnn \
  -DCAFFE_TEST_MODEL_DIR="$PWD/examples" \
  -DPython3_EXECUTABLE=/absolute/path/to/python3
cmake --build build-pnnx-caffe -j4
ctest --test-dir build-pnnx-caffe -R '^test_caffe_' --output-on-failure -j2
```

For non-system protobuf installations, also set `Protobuf_INCLUDE_DIR`,
`Protobuf_LIBRARY` and `Protobuf_PROTOC_EXECUTABLE`, and make their shared
libraries visible to the loader. Keep any required `PYTHONPATH` in the test
environment. CTest is enabled by the existing pnnx build, without a separate
test option. Add `CAFFE_SSD_MODEL_DIR` at configuration time to include the
external SSD fixture described below; it additionally requires Python OpenCV.
Without that fixture the regular suite has 16 Caffe tests, including SqueezeNet,
but does not validate real SSD. Each numerical script records `results.json` in
its build-tree output directory. Conversion runs use optlevel=2 and both weight
precisions; this suite does not request ARM or Vulkan execution.

## Reproducible SqueezeNet regression

On 2026-09-19, a standalone archive of the working `tools/pnnx` tree, including
new sources, built successfully with Caffe enabled and disabled. The enabled
build passed all six SqueezeNet comparisons. With Caffe and ONNX disabled and
both protobuf discovery mechanisms disabled, the executable had no protobuf
link dependency and rejected `.prototxt optlevel=2` with exit status 255 and an
explicit rebuild instruction. This snapshot includes broadcast Scale but
predates the subsequent Eltwise, Slice, Tile and Reorg level2 migrations; it is
packaging evidence, not validation of those later changes. Archive SHA256:
`61580625a29558a126546fa14426937e6bb914bc447078b83c95fba555c34ad2`.

Build pnnx with Caffe and tests enabled, build the retained `caffe2ncnn`, and install
Python `torch`, `numpy`, and the ncnn Python bindings. No full Caffe runtime is
required. The test uses the repository's `examples/squeezenet_v1.1.prototxt` and
`.caffemodel`, copies them to its output directory, and never overwrites the
existing example ncnn models. The example cites
<https://github.com/nihui/ncnn-assets/tree/master/models> as the model source;
this test reuses checked-in assets and does not download or redistribute another
model. Independent Caffe/fork runtime comparison remains an additional coverage
gap, not a test implicitly counted as passed.

```sh
python3 tools/pnnx/tests/caffe/test_squeezenet.py --pnnx /absolute/path/to/pnnx --caffe2ncnn /absolute/path/to/caffe2ncnn --model-dir examples --output-dir /tmp/pnnx-caffe-squeezenet
```

The CTest name is `test_caffe_squeezenet`; configure `CAFFE2NCNN_EXECUTABLE` and,
for standalone pnnx sources without examples, `CAFFE_TEST_MODEL_DIR`.

The fixed seed-0 synthetic BGR image is already 227x227, matching the final
resize dimensions in `examples/squeezenet.cpp`. Subtract [104,117,123] without
normalization. `input.npy` records the exact NCHW input. The test compares all
1000 probabilities for old converter -> ncnn, new frontend -> ncnn, and new
frontend -> generated PyTorch. It explicitly maps PyTorch [1,1000,1,1] to ncnn
[1000,1,1] from the unified exporter and [1000] from the legacy exporter. Every conversion uses `optlevel=2`; it runs FP32 weights first and then
FP16 weights. FP32 uses atol=1e-5/rtol=1e-4, FP16 uses atol=1e-3/rtol=1e-2.
Nonfinite outputs and shape mismatches fail. Commands and maximum absolute
errors are printed; `results.json` records comparison pass/fail counts with zero
skips. Missing tools, dependencies, or assets fail instead of silently skipping.

Initial execution on 2026-09-15 (CPU, Python Torch 2.12.1 and installed ncnn
bindings) passed all six SqueezeNet comparisons: FP32 new ncnn versus legacy
maximum absolute error 0; PyTorch versus legacy 4.843e-7; FP16 ncnn versus
legacy 1.294e-4. These results cover the checked-in model and fixed input above,
not every parameter combination in the compatibility matrix.

`test_caffe_operators` generates small Caffe models using the canonical schema
and `protoc --encode=caffe.NetParameter`; it needs neither Caffe nor Python
protobuf. References use independent Torch expressions and an explicit Caffe
pooling-window implementation. The cases cover grouped/depthwise convolution,
inplace branches, pooling boundary windows, transposed InnerProduct weights,
BatchNorm zero/nonzero statistics factors, Scale, inputshape overrides,
multi-output Slice, Silence, Eltwise, Power, activations, ShuffleChannel, MVN,
LRN, and Reshape/Permute. Missing files, malformed protobuf text and unknown
operators must fail. Run directly with:

```sh
python3 tools/pnnx/tests/caffe/test_operators.py --pnnx /absolute/path/to/pnnx --protoc /absolute/path/to/protoc --schema tools/pnnx/src/caffe.proto --output-dir /tmp/pnnx-caffe-operators
```

Adding a case does not establish a passed compatibility claim: its generated
`results.json` from an actual run is the evidence.

`test_caffe_detection` uses independent analytical fixtures for PriorBox grid
ordering, multiple sizes, aspect-ratio flip/deduplication, clipping, image/step
overrides and variances. DetectionOutput covers nonzero location offsets,
per-coordinate variances, per-class NMS, top-k/keep-top-k truncation, empty
outputs, confidence/IoU boundaries and nonnormalized confidence scores. Scores
are distinct so unspecified equal-score ordering does not affect the comparison.
PyTorch detection rows include a batch ID; ncnn rows omit it. The test checks
batch IDs and class indices exactly, then compares coordinates and scores.
Previous DetectionOutput numerical results used a runtime correction that has
been withdrawn. Empty-output conventions and nonnormalized confidences must
be assessed separately against the stable runtime.

ROI references retain Caffe pooling-bin and interpolation semantics, including
the [R-FCN endpoint rounding rule](https://raw.githubusercontent.com/daijifeng001/caffe-rfcn/r-fcn/src/caffe/layers/psroi_pooling_layer.cu).
Stable PSROIPooling rounds endpoint-plus-one, which differs at negative
half-integers; ROIAlign boundary sampling and optimized ROIPooling bin rounding
also retain their original behavior. Static representable cases can use the
existing single-ROI layers; no native batch-ROI protocol is available.

Proposal uses the existing exclusive-area NMS and quicksort tie behavior.
These can differ from Caffe's inclusive-area NMS and source-specific tie order.
No `nms_inclusive` or `stable_sort` parameter may be exported. Empty candidates
retain the baseline allocation/error behavior. Tests must separate numerical
comparisons on supported inputs from warnings and rejected configurations.

Dynamic Proposal-to-ROI and ROI-head graphs are not supported by the baseline
single-ROI contract. They must be rejected before required outputs are written;
neither taking only the first ROI nor emitting an ignored `roi_batch` parameter
is acceptable. Previous R=0/1/2 passes relied on withdrawn runtime extensions.

YOLO references retain the legacy confidence formula and class-agnostic NMS.
For empty detections, the stable v2 layer can retain its input and v3 does not
guarantee a shaped zero-row output. The frontend must warn about these
data-dependent limitations. Empty-result warning coverage is not numerical
equivalence coverage. The previous generic/x86 empty-output fixes are withdrawn.

`test_caffe_embed` compares optional-bias embedding lookup with direct table
indexing for rank-one indices and a singleton-batch rank-two input. The exported
input uses int32 storage: callers must explicitly convert Caffe's float-valued
integer indices to integer values before inference. Inputs may feed multiple
Embed layers directly or through Split layers. The complete index subgraph is
validated before type conversion; computed floating-point indices and branches
shared with floating-point operators are rejected. This boundary
adaptation is part of the model's documented input contract.

ArgMax models require an ncnn runtime built with `-DWITH_LAYER_argmax=ON`;
that legacy layer is disabled by default. For Python builds via `setup.py`, pass
`EXTRA_CMAKE_ARGS="-DWITH_LAYER_argmax=ON"`. The ArgMax regression requires this
runtime capability and does not treat a missing layer as a passing or skipped
numerical test.

The baseline native sorting path uses `Mat.total()`, which can include allocation
padding (for example, 18 logical floats occupy 20 sortable entries). The frontend
therefore rejects native-sort configurations whose logical element count is not
divisible by four, with an allocation-padding diagnostic. Numerical fixtures use
aligned counts; unaligned counts have rejection coverage. Axis top-1 value-only
reduction uses `torch.amax` and is not subject to this native-sort restriction.

`test_caffe_export_errors` checks normal conversion, failure of each of the six
required output files using `/dev/full`, and an optional ONNX output that cannot
be opened. Required write/close failures return nonzero and warn that artifacts
may be incomplete. Optional Caffe ONNX failure warns and continues ncnn export;
it does not invalidate the required PyTorch and ncnn targets. Platforms without
`/dev/full` report the write-failure test as skipped, not passed.

The recurrent test uses independent NumPy IFOG equations for single-input LSTM
and repeated calls with zero initial state. Two-input continuation must be
rejected: ignoring masks or exporting a new native parameter is not acceptable.
The former single-node continuation protocol and its patched-runtime results
are withdrawn. A diagnostic test is not an implementation of continuation.

## External MobileNet-SSD regression

The optional real SSD test uses [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
revision `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`: `deploy.prototxt`,
`mobilenet_iter_73000.caffemodel`, and `images/000001.jpg`. Its repository includes
an [MIT license](https://github.com/chuanqi305/MobileNet-SSD/blob/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/LICENSE).
The model and image remain external fixtures; neither is vendored here. Preserve
the upstream license alongside the downloaded archive. Obtain the pinned source
from [GitHub's archive](https://github.com/chuanqi305/MobileNet-SSD/archive/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a.zip)
and extract it to a separate directory.

```sh
python3 tools/pnnx/tests/caffe/test_mobilenet_ssd.py --pnnx /absolute/path/to/pnnx --caffe2ncnn /absolute/path/to/caffe2ncnn --model-dir /absolute/path/to/MobileNet-SSD --output-dir /tmp/pnnx-caffe-ssd
```

This test additionally needs Python OpenCV. Preprocessing follows upstream
`demo.py`: BGR image resized to 300x300, subtract 127.5, multiply by 0.007843.
It requires nonempty detections and compares the complete ordered detection
rows, with exact class/count checks and the standard numerical tolerances. Set
`CAFFE_SSD_MODEL_DIR` to register `test_caffe_mobilenet_ssd` with CTest. Without
the external fixture, this validation remains unperformed, not a passed test.

Earlier six-comparison SSD runs used a patched runtime and remain historical.
Repeat the pinned-fixture test with the restored runtime before reporting
current numerical acceptance; do not reuse the earlier errors as new evidence.

The legacy converter cannot import this network's net-level input declaration.
Only its baseline prototxt is adapted to an equivalent Input layer; pnnx receives
the original upstream prototxt. This adaptation changes no operation or tensor.
Separate synthetic DetectionOutput cases test zero and negative confidence
scores: Caffe's sentinel row and ncnn's empty tensor represent the same empty set.

The initial MobileNet-SSD results likewise do not replace a fresh baseline run.

## Validation status

After updating to upstream `b5721b4bd2c8d0a8b861477f72c515db6b9f6dd0`,
the Caffe and TNN suites were rerun against its unchanged CPU runtime and Python
binding, with ArgMax enabled. Tests cover both FP32 and FP16 model weights.
Expected rejections and warnings remain diagnostic coverage, not evidence of
numerical support for those inputs.

The review fixes add regressions for preserved Crop inputs, empty trailing MAX
pooling windows and downstream Flatten, scalar reductions followed by pointwise
layers and Split/Eltwise, and the output shape returned by generated
DetectionOutput ncnn Python helpers. CI builds the legacy caffe2ncnn executable
and supplies its explicit path for the SqueezeNet comparison.

Local validation uses an isolated Caffe/TNN converter build because the installed
CUDA Torch package lacks its required development toolkit. The TorchScript loader
is stubbed in that validation build; TorchScript conversion and the full CI matrix
are not covered by these results. External MobileNet-SSD, ARM, Vulkan and
optlevel=0/1 were not rerun during this review.
