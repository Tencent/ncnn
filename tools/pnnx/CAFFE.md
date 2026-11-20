# Caffe frontend

Place the deployment network and weights together as `model.prototxt` and
`model.caffemodel`, then run:

```sh
pnnx model.prototxt
```

The weights are loaded from the same basename automatically. The usual pnnx
output options apply; use `fp16=0` for FP32 ncnn weights. Caffe itself is not
required. Generated PyTorch models use `Model().eval()` and load weights from
`.pnnx.bin`; run them in the output directory so relative paths resolve.

The frontend uses existing ncnn layers and Python bindings. Unsupported operators
and parameter combinations produce a layer-specific error. Some detection and
ROI differences depend on input values and produce warnings; see the limits below.

## Conversion flow

`load_caffe.cpp` reads the network and blobs using `src/caffe.proto`, which is
also shared by the legacy `caffe2ncnn` tool. It builds the source graph with
`caffe.*` operators, ordered inputs and outputs, and in-place blob connections.
Only the parameter message for each layer is imported: `softmax_param.axis`
becomes `axis`, while nested names such as `nms_param.top_k` remain qualified.

`pass_caffe.cpp` processes source operators in execution order. The
`pass_caffe/prepare_*` and `lower_*` helpers validate parameters and blobs,
propagate shapes, and normalize Caffe-specific arithmetic or layouts. Basic
arithmetic becomes `aten::*`, `prim::Constant` and `pnnx.Attribute` nodes;
the common level3 expression pass performs expression fusion.

Standard layers use the target-oriented rules in `pass_level2`, alongside the
ONNX and TNN rules. Concat and Slice prepare the existing list-input and
list-output conventions. Extension operators such as PriorBox and DetectionOutput
use generated helpers in `caffe_python.cpp` and exporters in `pass_ncnn`.
Unconverted standard operators are rejected after level2.

Operator and operand names receive unique ASCII identifiers for generated Python.
Import diagnostics retain the source layer names. The output set preserves
terminal Split branches and suppresses blobs consumed by Silence. Crop uses the
reference tensor's static shape and retains that tensor as a model input.

## Supported layers

The table describes implemented parameter handling. Caffe forks may assign
additional semantics to familiar layer names; unlisted combinations need separate
validation. SSD, PSPNet and other extensions follow the schema and the specific
contracts described below.

| Layers | Parameters and conversion |
| --- | --- |
| Input, MemoryData | Net-level or layer-level declarations, ordered shapes and `inputshape` overrides |
| Convolution, ConvolutionDepthwise, DepthwiseConvolution | Optional bias, groups, dilation, rectangular kernel/stride/padding; depthwise aliases infer groups |
| Deconvolution | Grouped IOHW weights, optional bias, rectangular geometry and dilation |
| InnerProduct | Tail flattening, axis, transposed weights and optional bias |
| BatchNorm | Stored mean/variance and accumulated-statistics factor, epsilon |
| BN, Scale | Learned affine weights or a second scale input; axis/num_axes broadcasting and optional bias |
| ReLU, ReLU6, ELU, PReLU | Negative slope, clipping, alpha and shared/per-channel PReLU slopes |
| AbsVal, Sigmoid, TanH, BNLL | Common unary operators and stable softplus arithmetic |
| Pooling | MAX/AVE, global pooling, rectangular geometry and Caffe ceil-window semantics |
| Concat, Eltwise | Negative axes and legacy concat_dim; PROD/SUM/MAX and ordered SUM coefficients |
| Flatten, Reshape, Permute | Partial flattening, 0/-1 reshape dimensions, axis/num_axes and non-batch permutations |
| Slice | Explicit slice points or equal splits, including multiple tops |
| Softmax, Tile | Axis-aware softmax and channel/spatial repetition; repeated batch is unsupported |
| Dropout, Split, Silence | Test-phase dropout including scale_train, fanout and output suppression |
| Exp, Log, Power, Threshold | Base/scale/shift, exponent and strict threshold comparison |
| Reduction | SUM/ASUM/SUMSQ/MEAN over dimensions from axis onward, with coefficient and scalar outputs |
| ArgMax | Flattened native sorting and supported per-axis cases; see native-sort restrictions below |
| Crop | Axis, shared or per-axis offsets and reference shape |
| LRN, MVN, Normalize | Caffe normalization geometry, variance options, epsilon and learned Normalize scale |
| Embed | Row-major table and optional bias, with integer index inputs |
| LSTM | Single-input sequence, IFOG-to-IFGO gate reorder and zero initial state |
| Interp | PSPNet size/zoom/shrink/crop semantics and corner-aligned interpolation |
| PriorBox, DetectionOutput | SSD priors, CENTER_SIZE decoding and supported NMS settings |
| Python ProposalLayer | Proposal anchors, feat_stride and the supported native proposal settings |
| ROIPooling, PSROIPooling, ROIAlign | Static ROI adapters, pooled dimensions and spatial scale |
| Reorg, ShuffleChannel | Stride/group-based reshape and permutation |
| YoloDetectionOutput, Yolov3DetectionOutput | Classes, boxes, confidence/NMS, biases and v3 masks/anchor scales |

## Input and operator limits

- Embed inputs use int32 storage. Convert Caffe's float-valued indices to integer
  values before inference. Index inputs may be shared by Embed layers directly
  or through Split; computed floating-point indices and branches shared with
  floating-point operators are rejected.
- Reduction requires a singleton batch or an unbatched input. Native scalar
  results use a one-element ncnn Mat, while PyTorch results have rank zero.
- LSTM accepts rank-two sequences or rank-three sequences with batch size one.
  Continuation inputs and exposed hidden states are unsupported.
- Permute cannot move the batch dimension. Low-rank Normalize supports the
  across-spatial, channel-shared case; other low-rank flag combinations are
  rejected.
- Native ArgMax requires an ncnn build with `-DWITH_LAYER_argmax=ON`. Its sorting
  path includes allocation padding, so the frontend rejects logical element
  counts not divisible by four. Per-axis indices/top-k require all other
  dimensions to be singleton. Axis top-1 value-only reduction uses `torch.amax`
  and does not have the native-sort alignment restriction.

Caffe [PoolingLayer](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp)
removes a trailing window wholly in padding when padding is configured in either
direction. Without configured padding, MAX pooling retains empty ceil windows
and returns `-FLT_MAX`. Explicit padding and floor pooling preserve this geometry;
AVE divisors include configured padding but exclude extra ceil-tail cells.

The PSPNet [Interp layer](https://github.com/hszhao/PSPNet/blob/master/src/caffe/layers/interp_layer.cpp)
uses `floor((size - 1) / shrink_factor) + 1` and
`(size - 1) * zoom_factor + 1`. Negative `pad_beg`/`pad_end` crop before
interpolation. Sampling follows `align_corners=True`; a singleton output samples
the first effective input pixel.

Power exponents 1, 2 and 3 use arithmetic rewrites. Other exponents use a helper
and native Power so negative-base behavior does not depend on BinaryOp's log/exp
approximation. The [SSD Normalize layer](https://raw.githubusercontent.com/weiliu89/caffe/ssd/src/caffe/layers/normalize_layer.cpp)
uses `sqrt(sum(x*x) + eps)` for the across-spatial, channel-shared case, including
rank-two and rank-three inputs.

### Detection and ROI

DetectionOutput supports CENTER_SIZE decoding, shared locations and background
label zero. ncnn assumes normalized class probabilities: unnormalized background
scores can suppress foreground decoding incorrectly. The frontend warns about
this input restriction. Caffe/PyTorch detection rows include the image ID;
ncnn rows contain only label, score and four coordinates. Empty Caffe detections
use a sentinel row, whereas ncnn returns an empty tensor.

Proposal uses exclusive-area NMS, non-stable tie ordering and fixed `min_size=16`.
These can differ from Caffe's inclusive-area NMS or fork-specific tie order, and
empty candidate sets may fail. Dynamic Proposal-to-ROI and ROI-head graphs are
rejected because ncnn ROI layers consume a single box. Static ROI counts use
explicit adapters.

ROI coordinates have further boundaries: out-of-range ROIAlign samples, clipped
ROIPooling bins and negative-half-integer PSROIPooling endpoints can differ
between Caffe and ncnn. In particular, the
[R-FCN pooling rule](https://raw.githubusercontent.com/daijifeng001/caffe-rfcn/r-fcn/src/caffe/layers/psroi_pooling_layer.cu)
rounds the inclusive endpoint before adding one; ncnn rounds endpoint-plus-one.

YOLO uses the legacy confidence formula and class-agnostic NMS. Empty detections
are unsupported: the v2 layer can retain its input, and v3 does not guarantee
a shaped zero-row output. Conversion warnings do not establish numerical
equivalence for these boundary inputs.

## Build and test

Install a C++ compiler, CMake, Torch, protobuf development files and a compatible
`protoc`. Caffe support is enabled by default; configure with
`-DPNNX_CAFFE2PNNX=OFF` to build without it. The tests need Python Torch, NumPy
and ncnn bindings built from this checkout. Enable ArgMax in the runtime; for
the ncnn Python package, set `EXTRA_CMAKE_ARGS="-DWITH_LAYER_argmax=ON"` when
building it. Build `caffe2ncnn` through the usual ncnn tools configuration.

From the repository root:

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

For non-system protobuf, set `Protobuf_INCLUDE_DIR`, `Protobuf_LIBRARY` and
`Protobuf_PROTOC_EXECUTABLE`, and expose its shared libraries to the loader.
Preserve any required `PYTHONPATH` in the test environment. Standalone pnnx
sources need an explicit `CAFFE_TEST_MODEL_DIR` for the SqueezeNet assets.

The regular suite contains 16 Caffe tests. Numerical tests run at `optlevel=2`
with FP32 and FP16 weights and record `results.json` in their build-tree output
directories. FP32 uses atol=1e-5/rtol=1e-4; FP16 uses atol=1e-3/rtol=1e-2.
Shape mismatches and nonfinite outputs fail unless the test explicitly checks
nonfinite classifications. Diagnostic cases check rejected inputs and warnings
separately. These tests do not request ARM or Vulkan execution.

The operator fixtures use `protoc --encode=caffe.NetParameter`; Caffe and Python
protobuf are unnecessary. References use independent Torch/NumPy expressions
and analytical pooling, detection and recurrent equations. Run individual suites
with their command-line options, for example:

```sh
python3 tools/pnnx/tests/caffe/test_operators.py \
  --pnnx /absolute/path/to/pnnx --protoc /absolute/path/to/protoc \
  --schema tools/pnnx/src/caffe.proto --output-dir /tmp/pnnx-caffe-operators
python3 tools/pnnx/tests/caffe/test_squeezenet.py \
  --pnnx /absolute/path/to/pnnx --caffe2ncnn /absolute/path/to/caffe2ncnn \
  --model-dir examples --output-dir /tmp/pnnx-caffe-squeezenet
```

SqueezeNet reuses the checked-in example assets from
[ncnn-assets](https://github.com/nihui/ncnn-assets/tree/master/models). Its fixed
seed-0 input is a 227x227 BGR image minus [104,117,123], matching the example's
preprocessing. The test compares all 1000 probabilities from legacy caffe2ncnn,
the pnnx ncnn export and generated PyTorch. `input.npy` records the input.

`test_caffe_export_errors` checks failures writing the six required output files
and an optional ONNX output. Required write/close failures return nonzero;
optional Caffe ONNX failure warns and continues ncnn export. The write-failure
test is skipped on platforms without `/dev/full`. Missing ordinary dependencies
or model assets fail their tests rather than being treated as numerical passes.

### External MobileNet-SSD

The optional SSD test uses [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
revision `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`: `deploy.prototxt`,
`mobilenet_iter_73000.caffemodel` and `images/000001.jpg`. Download the
[pinned archive](https://github.com/chuanqi305/MobileNet-SSD/archive/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a.zip)
and preserve its [MIT license](https://github.com/chuanqi305/MobileNet-SSD/blob/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/LICENSE).
The model and image are external fixtures. The test verifies their hashes.

```sh
python3 tools/pnnx/tests/caffe/test_mobilenet_ssd.py \
  --pnnx /absolute/path/to/pnnx --caffe2ncnn /absolute/path/to/caffe2ncnn \
  --model-dir /absolute/path/to/MobileNet-SSD --output-dir /tmp/pnnx-caffe-ssd
```

This test also requires Python OpenCV. It follows upstream preprocessing: resize
BGR to 300x300, subtract 127.5 and multiply by 0.007843. It requires nonempty
detections and compares complete ordered rows, with exact class/count checks.
Only the legacy converter's prototxt replaces the net-level input declaration
with an equivalent Input layer; pnnx receives the original file. Set
`CAFFE_SSD_MODEL_DIR` to include this fixture in CTest. Without it, the regular
suite does not validate the external model.
