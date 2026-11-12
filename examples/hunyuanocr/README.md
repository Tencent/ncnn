# HunyuanOCR ncnn example

Runs [Tencent HunyuanOCR](https://github.com/Tencent/HunyuanOCR) on CPU/GPU using ncnn.
HunyuanOCR is a 1B-parameter Vision-Language Model that detects and recognises text in
document images with bounding-box output.

Architecture: ViT visual encoder + XDRoPE text decoder (Qwen2.5-VL-style, 28 layers).

## Requirements

### Export host (needs ~20 GB VRAM or ~40 GB RAM)

```
pip install "huggingface_hub[cli]" torch torchvision transformers pnnx nlohmann-json
```

### Inference host

- ncnn >= 20240102 (with SDPA KV-cache support, `param 7=1`)
- OpenCV >= 3.4  (or build with `-DNCNN_SIMPLEOCV=ON`)
- nlohmann/json >= 3.11

## Step 1 - download model

```bash
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR-hf
```

> The model repository is currently gated. Accept the licence on
> [HuggingFace](https://huggingface.co/tencent/HunyuanOCR) and set
> `HF_TOKEN=<your-token>` before downloading.

## Step 2 - export to ncnn

```bash
cd convert/
python export_pnnx.py  --model ../HunyuanOCR-hf --output ../hunyuan_ocr
python add_kvcache.py  ../hunyuan_ocr/text_decoder.ncnn.param
python export_tokenizer.py --model ../HunyuanOCR-hf --output ../hunyuan_ocr
```

The output directory will contain:

```
hunyuan_ocr/
  model.cfg                # model config consumed by hunyuanocr binary
  vision.ncnn.{param,bin}
  text_embed.ncnn.{param,bin}
  text_decoder.ncnn.{param,bin}   # KV-cache patched by add_kvcache.py
  lm_head.ncnn.{param,bin}
  vocab.txt
  merges.txt
```

## Step 3 - build

```bash
mkdir build && cd build
cmake .. -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build . --parallel
```

## Step 4 - run

```bash
# default prompt: detect text with bounding boxes
./hunyuanocr --model ./hunyuan_ocr --image document.jpg

# custom prompt
./hunyuanocr --model ./hunyuan_ocr --image scan.png --prompt "提取所有文字"

# multi-threaded
./hunyuanocr --model ./hunyuan_ocr --image photo.jpg --threads 8
```

## Output format

The default prompt asks the model to output detected text with coordinates:

```
[{"text": "Hello World", "bbox": [12, 34, 200, 58]}, ...]
```

## Conversion scripts

| Script | Purpose |
|---|---|
| `convert/export_pnnx.py` | pnnx export of all 4 sub-networks |
| `convert/add_kvcache.py` | patch `text_decoder.ncnn.param` for incremental KV cache |
| `convert/export_tokenizer.py` | extract BBPE vocab and merge rules to plain text |

## Notes

- Inference on CPU requires ~6 GB RAM for fp16 weights.
- The vision encoder processes images resized to at most `max_pixels` (≈9.6 M pixels
  by default); very large input is scaled down automatically.
- KV-cache decoding is enabled by default; the first forward pass (prefill) processes
  all input tokens at once, subsequent steps run one token at a time.

## Related

- [HunyuanOCR official repo](https://github.com/Tencent/HunyuanOCR)
- [ncnn_llm reference implementation](https://github.com/LiYulin-s/ncnn_llm)
- [pnnx](https://github.com/pnnx/pnnx)
