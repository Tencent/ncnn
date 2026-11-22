# Post Training Quantization Tools

To support int8 model deployment on mobile devices,we provide the universal post training quantization tools which can convert the float32 model to int8 model.

## User Guide

Example with mobilenet, just need three steps.

### 1. Optimize model

NOTE: **If your model is converted via pnnx, skip this step.**

```shell
./ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 0
```

### 2. Create the calibration table file

#### 2.1 From image

We suggest that using the verification dataset for calibration, which is more than 5000 images.

Some imagenet sample images here https://github.com/nihui/imagenet-sample-images

```shell
find images/ -type f > imagelist.txt
./ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
```

* mean and norm are the values you passed to ```Mat::substract_mean_normalize()```
* shape is the blob shape of your model, [w,h] or [w,h,c]

>
    * if w and h both are given, image will be resized to exactly size.
    * if w and h both are zero or negative, image will not be resized.
    * if only h is zero or negative, image's width will scaled resize to w, keeping aspect ratio.
    * if only w is zero or negative, image's height will scaled resize to h

* pixel is the pixel format of your model, image pixels will be converted to this type before ```Extractor::input()```
* thread is the CPU thread count that could be used for parallel inference
* method is the post training quantization algorithm, kl, aciq, eq and percentile are currently supported

To use percentile calibration, set `method=percentile`. The optional `percentile` argument accepts one value or a list of candidate values in `(0, 1]`; ncnn2table will select the candidate with the lowest quantization error.

Percentile calibration uses the absolute input activation values, including zeros, of Convolution, ConvolutionDepthWise and InnerProduct layers. It does not calibrate the dynamic inputs of Gemm or MultiHeadAttention.

```shell
./ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=percentile
```

or you can specify the value of percentile:

```shell
./ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=percentile percentile=0.9999
```

If your model has multiple input nodes, you can use multiple list files and other parameters

```shell
./ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist-bgr.txt,imagelist-depth.txt mobilenet.table mean=[104,117,123],[128] norm=[0.017,0.017,0.017],[0.0078125] shape=[224,224,3],[224,224,1] pixel=BGR,GRAY thread=8 method=kl
```

#### 2.2 From npy

We suggest that using the validation(development) set for calibration.

Use the same preprocessing as the training set to get the input vectors, in the case of batchsize=1, store each input vector as an npy file, n inputs correspond to n npy files, the actual stored vectors to remove the batch dimension.


test net, shape is in NCHW format, but there's no `N`.
```txt
in0, shape=[512]
in1, shape=[2, 1, 64]
in2, shape=[2, 1, 64]
```

filelist_in0.txt
```txt
0_in0.npy
1_in0.npy
2_in0.npy
...
```

filelist_in1.txt
```txt
0_in1.npy
1_in1.npy
2_in1.npy
...
```

filelist_in2.txt
```txt
0_in2.npy
1_in2.npy
2_in2.npy
...
```

```shell
./ncnn2table test.param test.bin filelist_in0.txt,filelist_in1.txt,filelist_in2.txt test.table shape=[512],[64,1,2],[64,1,2] thread=8 method=kl type=1
```
**Here shape is WHC, because the order of the arguments to `ncnn::Mat`.**

ncnn2table can generate static weight scales without a calibration dataset for RNN,GRU,LSTM,MultiHeadAttention,Gemm and Embed layers

```shell
./ncnn2table rnn.param rnn.bin rnn.table method=kl
```

### 3. Quantize model

```shell
./ncnn2int8 mobilenet-opt.param mobilenet-opt.bin mobilenet-int8.param mobilenet-int8.bin mobilenet.table
```

For Gemm layers, ncnn2int8 uses `<layer_name>_param_0` scales for constant A and `<layer_name>_param_1` for constant B when present in the table. If none of the required Gemm scales are present, it computes them using absmax, preserving compatibility with older tables and conversion without a table. If some required scales are present but others are missing, or the number of values is incorrect, conversion fails without saving the output model.

## Block quantized Gemm and MultiHeadAttention

LLM-oriented `Gemm` and `MultiHeadAttention` block quantization is separate from the post training int8 flow above. The 4-bit and 6-bit modes are weight-only: they store weights as signed int4/int6 blocks and keep activation/output in fp32. The 8-bit CPU mode is dynamic W8A8 per-block: it stores constant weights as signed int8, dynamically quantizes each fp32 activation row and block to signed int8 for every forward, accumulates int8 dot products in int32, applies the activation and weight descales at each block boundary, and produces fp32 output. The 8-bit mode has no W8A32 compatibility path.

The workflow is similar to `ncnn2table` and `ncnn2int8`:

```shell
./ncnnllm2table in.param in.bin model.llm.table method=minmax bits=6 block=64
./ncnnllm2int in.param in.bin out.param out.bin model.llm.table
```

method can be minmax,mseclip,awq,gptq. bits can be 4,6,8. `bits=4` and `bits=6` select weight-only execution, while `bits=8` selects dynamic W8A8 per-block execution on CPU. block can be 32,64,128. thread is the CPU thread count.

awq and gptq need calibration data, same as npy calibration in ncnn2table.

```shell
./ncnnllm2table in.param in.bin calib.list awq.llm.table method=awq bits=4 block=64 type=1 shape=[...]
./ncnnllm2int in.param in.bin awq.param awq.bin awq.llm.table
```

```shell
./ncnnllm2table in.param in.bin calib.list gptq.llm.table method=gptq bits=4 block=128 type=1 shape=[...]
./ncnnllm2int in.param in.bin gptq.param gptq.bin gptq.llm.table
```

The calibration list format follows `ncnn2table`.

```text
gemm_name_param_1 bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_0  bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_1  bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_2  bits=4 block=64 method=mseclip scale0 scale1 ...
mha_name_param_3  bits=4 block=64 method=mseclip scale0 scale1 ...
```

For `MultiHeadAttention`, `_param_0/_param_1/_param_2/_param_3` are q/k/v/out weights.

```text
gemm_name_param_1_input_scale method=awq scale0 scale1 ...
mha_name_param_0_input_scale  method=awq scale0 scale1 ...
```

`method=gptq` uses fixed symmetric block scales and standard GPTQ error compensation. It writes packed qweight files and records them in the table.

```text
gemm_name_param_1 bits=4 block=128 method=gptq qweight=gemm.qweight scale0 scale1 ...
```

The table is text and may be edited before conversion. Missing `Gemm` rows are skipped. `MultiHeadAttention` q/k/v/out rows must exist together. Unused rows are rejected and at least one layer must be quantized.

The generated layer `quantize_term` is `bits * 100 + input_scale * 10 + block_code`, and block_code 0/1/2 means block 32/64/128.

For quick conversion without saving a table, `ncnnllm2int` can still compute scales directly:

```shell
./ncnnllm2int in.param in.bin out.param out.bin method=minmax bits=6 block=64
```

This format is signed symmetric scale-only. Zero point is not used.

## Transformer weight import and quantization

Build with `NCNN_WEIGHT_QUANT=ON` and `NCNN_BUILD_TOOLS=ON`. Use `NCNN_VULKAN=ON` for the Vulkan paths. The tools emit standard ncnn `.param` and `.bin` files using magic `7767517`.

`ncnnllm2table`, `ncnnllm2int`, `ncnndit2table` and `ncnndit2int` take ncnn models as input. GGUF and ComfyUI checkpoints have their own import commands.

### Qwen3-0.6B GGUF Q8_0

```shell
./ncnngguf2ncnn Qwen3-0.6B-Q8_0.gguf benchmark/models/llm/qwen3_0.6b_decoder.ncnn.param qwen3
```

Use the generated embedding, decoder and output projection graphs with the Qwen3 tokenization and KV cache flow. For tied weights, load both the embedding and output projection graphs with `qwen3.embed.ncnn.bin`. The tool reports which binary to use. On CPU, set `Net::opt.use_int8_inference` before loading the model. Enabling it selects dynamic K32 activation quantization. Disabling it uses floating-point activations.

With `Net::opt.use_mapped_model_loading`, the tied graphs can share the file-backed pages containing the quantized codes. The output projection's packed CPU weights or Vulkan weights have their own storage. `Embed` decodes the requested rows on CPU.

For a fixed-token prefill and decode benchmark, build with `NCNN_BUILD_BENCHMARK=ON`.

```shell
./benchncnn_llm --qwen3-q8-0 qwen3 8 1 logits.f32
```

The last arguments select eight threads and dynamic INT8 activations. Use `0` for floating-point activations. The logits file is optional.
The optional `benchmark/qwen3_q8_reference.py` compares those logits with an independent FP32 reference using the same GGUF and token sequence. It uses the Python `gguf`, `numpy` and `torch` packages.

For Vulkan, append the device index. The benchmark passes the CPU embedding output to the Vulkan decoder and runs the output projection on Vulkan. Use `-` as the logits path to skip writing the output file.

```shell
./benchncnn_llm --qwen3-q8-0 qwen3 8 1 logits-vulkan.f32 0
```

For separate output weights, pass the projection binary after the device index. Set the device index to `-1` for CPU inference.

```shell
./benchncnn_llm --qwen3-q8-0 qwen3 8 1 logits.f32 -1 qwen3.proj_out.ncnn.bin
```

### Quantize an ncnn DiT model

`ncnndit2int` converts a floating-point ncnn `.param/.bin` model. It can compute the weight scales directly or read a table from `ncnndit2table`.

```shell
./ncnndit2int in.param in.bin out-int8.param out-int8.bin storage=bf16
./ncnndit2int in.param in.bin out-int6.param out-int6.bin bits=6 block=64 storage=bf16
./ncnndit2int in.param in.bin out-int4.param out-int4.bin bits=4 block=64 storage=bf16

./ncnndit2table in.param in.bin model.dit.table convrot=auto method=mseclip include=layers.txt
./ncnndit2int in.param in.bin out-convrot.param out-convrot.bin model.dit.table storage=bf16
```

Weight-only INT8 is the default and uses one scale per output channel. Weight-only INT6 and INT4 pack signed weights in K groups of 32, 64 (default) or 128. All three support `method=minmax` (default) and `method=mseclip`. FP16 and BF16 weights are loaded in FP32 for conversion.

For ConvRot INT8, set `convrot=auto` or give a regular Hadamard group size such as `convrot=256`. The tool rotates weights along K before calculating their scales. `auto` selects the largest of 256, 64, 16 and 4 that divides each layer's K and skips dimensions without a valid group. An explicit group must divide K for each selected eligible Gemm. A layer listed in `include` must support the requested conversion unless also excluded. The resulting layer uses mode 902 and quantizes activations at inference time.

Optional `include=file` and `exclude=file` take one Gemm layer name per line. `storage` controls tagged floating-point weights in the output. Keep the same storage option when comparing direct and table-driven conversion.

The DiT table records a row for each Gemm. A `keep` row leaves that layer's quantization as is. Quantized rows record the mode, dimensions and the scale values in output-channel order. INT4 and INT6 store divisors in K-group order within each channel. INT8 stores multipliers. A row with no scale values requests recomputation using its settings.

```text
gemm_0 bits=8 block=0 convrot=256 method=mseclip N=3840 K=3840
gemm_1 bits=4 block=64 convrot=0 method=minmax N=3840 K=3840
gemm_2 keep
```

### ComfyUI ConvRot INT8

```shell
./ncnncomfy2int8 in.param in.bin checkpoint.safetensors out.param out.bin layers
```

Use `context_refiner`, `noise_refiner` or `layers` for the corresponding Z-Image-Turbo transformer graph. For a custom graph, pass a mapping file instead with one `gemm_layer comfy_module first_output_row` entry per line. The converter preserves the stored INT8 weights and scales. The runtime rotates the activation along K. For dynamic INT8, the CPU and supported Vulkan devices quantize each rotated row and perform W8A8. The floating-point path reads from compressed INT8 weights.

With the built-in Z-Image-Turbo graphs, use `-` in place of `in.bin` to source the refiner weights directly from the checkpoint. Use the original `in.bin` with custom mappings.

## Quantization formats and backend design

Keep the CNN calibration and requantization flow under `NCNN_INT8`. Its `int8_scale_term` and calibration table describe the existing convolution, depthwise convolution and inner-product inference. Transformer constant-weight formats use `NCNN_WEIGHT_QUANT`. The native block formats store a divisor per block, `900` stores a FP16 multiplier per K32 block, and `901`/`902` store a FP32 multiplier per output channel. The [Gemm parameter reference](../developer-guide/operators.md#gemm) defines the serialized layouts and the rotation parameter.

### LLM formats

| Format | Representation and execution |
| --- | --- |
| GGUF `Q8_0` | One FP16 multiplier and 32 signed INT8 codes per block. The importer preserves both. Gemm supports floating-point activations and dynamic K32 W8A8, and Embed decodes selected rows. [GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h) |
| GGUF `Q4_0`, `Q4_K` and mixed K-quants | `Q4_0` uses two groups of nibbles within a K32 block and a signed FP16 multiplier. `Q4_K` uses eight K32 groups in a K256 superblock with quantized scales and minima. A `Q4_K_M` file can contain other types, including `Q6_K`. Read each tensor's type from the GGUF directory. [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [GGML quantization definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h), [llama.cpp tensor selection](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp) |
| AWQ and GPTQ | AWQ uses activation statistics to protect salient weights. GPTQ uses approximate second-order information to compensate for quantization error. ncnn can calibrate its native symmetric format with `ncnnllm2table`. Imported checkpoints may also require group zero points, a group index or a permutation and checkpoint-specific packing. [AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323), [vLLM format and device support](https://docs.vllm.ai/en/latest/features/quantization/) |
| W8A8, FP8, FP4 and KV cache | SmoothQuant balances activation and weight ranges offline for W8A8. FP8 and FP4 require their own element and scale encodings and matching device kernels. KV-cache formats apply to the activation cache in attention. [SmoothQuant](https://arxiv.org/abs/2211.10438), [TensorRT-LLM precision modes](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html) |

The current GGUF importer targets Qwen3-0.6B `Q8_0`. To add GGUF INT4, extend the converter's tensor-type dispatch and give `Q4_0` and K-quants explicit on-disk layouts. Preserve the published code and scale information, including the minima in `Q4_K`. Decode a scalar reference first, then reuse bounded Gemm tiles and Embed row gather. In an integer kernel for a block with `W = d*q + b`, apply both `d*dot(A,q)` and `b*sum(A)` before accumulating the block. Cover `Q6_K` and the floating tensor types in a mixed `Q4_K_M` fixture. Verify packed bytes, row and axis order, and logits against the same GGUF using llama.cpp before choosing a default for decode or prefill. Extend the same affine tile computation to AWQ or GPTQ only after validating the checkpoint's layout and any activation transform.

### DiT formats

- Use channel-wise INT8 and grouped INT6/INT4 for weight-only inference. Select layers offline and keep activations floating point. A backend unpacks B into a compute tile or directly into registers so the compressed B stays resident. Measure M=1 and the actual large token batches separately.
- Use `901`/`902` for ComfyUI `int8_tensorwise` weights. The importer preserves each layer's rotation and matrix-multiply precision. ConvRot uses normalized regular Hadamard groups along K. The integer kernel computes an INT32 dot and applies the token-row and output-channel multipliers in the epilogue. When integer execution is unavailable or disabled, the floating-point kernel uses the same rotated activation and compressed B. [ComfyUI quantization contract](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md), [Comfy Kitchen INT8 implementation](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py)
- ComfyUI also publishes layout definitions for `convrot_w4a4`, `asym_w4a8_int8`, `w6a8_int8`, and block-scaled floating-point formats. Future importers can reuse the rotation or integer GEMM where the activation quantizer and accumulation boundaries match. Validate relative scales, codebooks, and block layout for each format before exposing a mode. [ComfyUI weight formats](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md#weight-formats)
- Tune mixed precision on early, middle and late denoising steps with the same seed and scheduler. ViDiT-Q targets sensitivity across layers and timesteps. SVDQuant uses a low-rank floating-point component alongside W4A4. These schemes would require new conversion and execution support. Profile the selected DiT and device before adding an FP8/FP4 or low-rank kernel. [Q-Diffusion](https://arxiv.org/abs/2302.04304), [ViDiT-Q](https://arxiv.org/abs/2406.02540), [SVDQuant](https://research.nvidia.com/labs/eai/publication/svdquant/)

### Porting and validation

Use `Gemm`, `Embed`, the standard `.param/.bin` loader and the existing allocator and packing options. Keep container and tensor-name mappings in the conversion tools. Keep runtime prepacking backend-specific. On CPU, reuse the architecture's GEMM packing, tile kernels and epilogue for single- and multi-row inputs. On Vulkan, check buffer ranges, dispatch limits and arithmetic capabilities before selecting portable, integer dot or cooperative matrix kernels. Large aggregate models use multiple allocations and compressed weight shards where the layer supports them.

Compare each backend to a reference computed from the same quantized codes and scales. Report format correctness, numerical accuracy and model quality separately. Record model load, prefill and decode for LLMs, and denoising steps and final images for DiTs. Measure peak RAM/VRAM, upload and compute with the actual backend recorded. Regression coverage includes the old CNN pipeline and native block formats, independent compile flags, scalar and SIMD CPU paths, and Vulkan fallback and accelerated paths.

## use ncnn int8 inference

the ncnn library would use int8 inference automatically, nothing changed in your code

```cpp
ncnn::Net mobilenet;
mobilenet.load_param("mobilenet-int8.param");
mobilenet.load_model("mobilenet-int8.bin");
```

## mixed precision inference

Before quantize your model, comment the layer weight scale line in table file, then the layer will do the float32 inference

```
conv1_param_0 156.639840536
```

```
#conv1_param_0 156.639840536
```
