# high-performance transformer inference with mha kv cache in ncnn

This document details the implementation and usage of the key-value (kv) cache for the `MultiHeadAttention` and `SDPA` layer in ncnn. This feature significantly accelerates autoregressive inference for Transformer-based models, such as large language models and other encoder-decoder architectures.

## 1. what is kv cache?

### the challenge of autoregressive inference

Transformer models generate output token by token in a process called autoregressive decoding. In each step, the model takes the previously generated tokens as input to predict the next one. A core component of this is the self-attention mechanism, which computes query (q), key (k), and value (v) matrices based on the sequence generated so far.

Without optimization, the model must recompute the k and v matrices for all preceding tokens at every single step. For a sequence of length `N`, the computational cost for the self-attention mechanism is roughly proportional to `N^2`. As the sequence grows, this becomes a significant performance bottleneck.

### the solution: kv cache

**kv cache** is an optimization technique that stores the key and value tensors from previous decoding steps. When generating a new token, we only need to compute the k and v for the *current* token and append them to the cached values. The model then uses the full set of cached k and v tensors for the attention calculation.

### key benefits

- **dramatic speed-up:** It reduces the computational complexity of the self-attention mechanism from O(N^2) per step to approximately O(N). This drastically cuts down inference latency, especially for long sequences.
- **reduced computation:** It eliminates redundant calculations, saving significant computational resources and energy.
- **enables real-time applications:** The performance gain makes it feasible to deploy large Transformer models for interactive and real-time tasks.

## 2. ncnn kv cache implementation

ncnn introduces kv cache support directly into its `MultiHeadAttention` and `SDPA` layer. The implementation is designed to be efficient and flexible, handling both the dynamic cache of self-attention and the static k/v of cross-attention found in encoder-decoder architectures.

### self-attention vs. cross-attention cache logic

The caching strategy is fundamentally different for self-attention and cross-attention layers within a decoder.

#### self-attention (dynamic cache)
- **purpose:** Allows the decoder to attend to previously generated tokens in its own sequence (e.g., the text being generated).
- **cache Logic:** The cache is **dynamic** and grows with each generated token. In step `t`, the k and v for token `t` are computed and appended to the cache from step `t-1`.
- **ncnn implementation:** The `MultiHeadAttention` and `SDPA` layers for self-attention are modified to accept two additional inputs (`cache_k_in`, `cache_v_in`) and produce two corresponding outputs (`cache_k_out`, `cache_v_out`). The `7=1` parameter enables this dynamic caching behavior inside the layer.

#### cross-attention (static k/v)
- **purpose:** Allows the decoder to attend to the output of the encoder (e.g., attending to audio features in speech recognition, or an input sentence in translation).
- **cache Logic:** The k and v matrices are derived from the encoder's output, which is computed only **once** per input sequence. Therefore, the k and v for cross-attention are **static** and do not change during the decoding process. They are "cached" in the sense that they are pre-computed and reused in every decoding step.
- **ncnn implementation:** The `MultiHeadAttention` and `SDPA` layers for cross-attention are also configured with `7=1` and cache I/O blobs. However, the implementation correctly identifies cross-attention (where the query blob is different from the key/value blobs) and reuses the `cache_k_in` and `cache_v_in` directly, without performing concatenation. This allows the static encoder k/v to be passed efficiently through the network.

## 3. ncnn kv cache memory layout

The memory layout of the kv cache is a critical design choice for performance. ncnn uses different layouts for `MultiHeadAttention` and `SDPA` to optimize for their respective calculation patterns.

### `MultiHeadAttention` cache layout (Transposed)

The `MultiHeadAttention` layer uses a **transposed layout** for its cache blobs. The primary reason for this is to **ensure that data for each attention head is contiguous in memory, which significantly boosts gemm performance.**

*   **input blobs (q, k, v):** These typically have a shape where height represents the sequence length.
    *   `ncnn::Mat` dimensions: `(w = embed_dim, h = seq_len)`

*   **cache blobs (`k_cache`, `v_cache`):** These are stored in a **transposed** format.
    *   `ncnn::Mat` dimensions: `(w = seq_len, h = embed_dim)`

**the rationale:**

1.  **slicing by Head:** During the attention calculation, the code slices the `k_cache` and `v_cache` matrices along their height to isolate the data for each head (e.g., using `row_range(head_index * embed_dim_per_head, embed_dim_per_head)`).
2.  **memory contiguity:** Because `ncnn::Mat` uses a row-major memory layout, this slicing operation on the transposed cache blob results in a sub-matrix where all the data for a single head is perfectly contiguous.
3.  **gemm efficiency:** Subsequent matrix multiplication operations (`q * k^T` and `Attention * v`) can then operate on these contiguous memory blocks. This maximizes CPU cache locality and the effectiveness of simd instructions, leading to a substantial increase in computational speed.

If a non-transposed layout were used, the data for each head would be strided in memory, causing frequent cache misses and dramatically slowing down the performance-critical gemm calculations. Therefore, this transposed layout is a deliberate and crucial optimization for computation.

### `SDPA` cache layout (Standard)

The `SDPA` layer uses the **standard ncnn Mat layout**, where the sequence length is represented by the height.

*   **input blobs (q, k, v):** `(w = embed_dim, h = seq_len, c = num_heads)`
*   **cache blobs (`k_cache`, `v_cache`):** `(w = embed_dim, h = seq_len, c = num_heads)`

**the rationale:**

The `SDPA` layer's internal implementation directly concatenates the cache blobs (`past_k`, `past_v`) with the current ones (`cur_k`, `cur_v`) along the height dimension (`seq_len`). This simpler approach avoids the need for a transposed layout while still being highly efficient, as the concatenation logic is handled inside the optimized C++ implementation.

## 4. converting models to support kv cache

To enable kv cache, you must modify the model's `.param` file to add the necessary cache inputs and outputs to all `MultiHeadAttention` and `SDPA` layers in the decoder.

### step 1: export a sequence-length-1 model

First, export your model from its original framework (e.g., PyTorch) using a sequence length of 1 for the decoder. This creates a graph optimized for single-token generation, which is the core of the autoregressive decoding loop.

### step 2: modify the .ncnn.param file

After exporting, a script is needed to edit the generated `.ncnn.param` file to make it cache-aware.

#### A. Adding kv cache to All MultiHeadAttention and SDPA Layers

You must add cache inputs/outputs to **every** `MultiHeadAttention` / `SDPA` layer in the decoder.

- **change `input_count` and `output_count`:** Increase both by 2.
- **add blob names:** Append new, unique blob names for `cache_k_in`, `cache_v_in`, `cache_k_out`, and `cache_v_out`.
- **enable cache behavior:** Add the parameter `7=1`.

Here is a robust Python function that automates this process:
```python
def add_kv_cache_to_ncnn_param(filename):
    """
    Modifies an ncnn.param file to add a kv cache mechanism to all
    MultiHeadAttention and SDPA layers and overwrites the original file.
    This handles both self-attention and cross-attention layers.
    """
    import os

    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_line_index = 1  # line 2, after magic number
    header_parts = lines[header_line_index].strip().split()
    original_layer_count = int(header_parts[0])
    original_blob_count = int(header_parts[1])

    attention_indices = [i for i, line in enumerate(lines) if line.strip().startswith("MultiHeadAttention") or line.strip().startswith("SDPA")]
    attention_count = len(attention_indices)

    if attention_count == 0:
        print("No 'MultiHeadAttention' or 'SDPA' layers found. The file will not be modified.")
        return

    # --- modify MultiHeadAttention and SDPA layers ---
    for i, line_index in enumerate(attention_indices):
        parts = lines[line_index].strip().split()
        layer_type, layer_name, input_count_str, output_count_str = parts[:4]
        input_count, output_count = int(input_count_str), int(output_count_str)

        blob_and_params = parts[4:]
        inputs = blob_and_params[:input_count]
        outputs = blob_and_params[input_count : input_count + output_count]
        params = blob_and_params[input_count + output_count:]

        # add cache I/O blobs and enable cache parameter
        inputs.extend([f"cache_k_in_{i}", f"cache_v_in_{i}"])
        outputs.extend([f"cache_k_out_{i}", f"cache_v_out_{i}"])
        params.append("7=1")

        new_line_parts = [
            f"{layer_type:<24}", f"{layer_name:<24}",
            str(input_count + 2), str(output_count + 2),
            *inputs, *outputs, *params
        ]
        lines[line_index] = " ".join(new_line_parts) + "\n"

    # --- add a single input layer to provide all cache blobs ---
    new_layer_count = original_layer_count + 1
    # each mha needs 2 new *input* blobs and produces 2 new *output* blobs.
    # the total number of unique blobs increases by 4 for each mha.
    new_blob_count = original_blob_count + (attention_count * 4)
    lines[header_line_index] = f"{new_layer_count} {new_blob_count}\n"

    # find where to insert the new input layer (after existing ones)
    insert_pos = header_line_index + 1
    while insert_pos < len(lines) and lines[insert_pos].strip().startswith("Input"):
        insert_pos += 1

    cache_blob_names = [name for i in range(attention_count) for name in (f"cache_k_in_{i}", f"cache_v_in_{i}")]
    input_layer_line = (
        f"{'Input':<24} {'kv_cache_in':<24} 0 {len(cache_blob_names)} "
        f"{' '.join(cache_blob_names)}\n"
    )
    lines.insert(insert_pos, input_layer_line)

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Successfully added kv cache to {attention_count} MultiHeadAttention / SDPA layers.")

# usage:
# add_kv_cache_to_ncnn_param("your_model_decoder.ncnn.param")
```

#### B. Supporting Dynamic Sequence Length in Gemm
Feed-forward networks (`Gemm` layers) that process the output of attention blocks must support dynamic sequence lengths, as the cache grows. To achieve this, change the parameter `7=1` (constant input shape) to `7=0` (dynamic input shape) for the relevant `Gemm` layers.

```python
def update_gemm_params(param_file_path):
    """
    Finds all 'Gemm' layers and changes parameter '7=1' to '7=0'
    to support dynamic input shapes.
    """
    import re
    with open(param_file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith('Gemm'):
            line = re.sub(r'(\b7=)1\b', r'\g<1>0', line)
        new_lines.append(line)

    with open(param_file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated Gemm layers in '{param_file_path}' to support dynamic inputs.")

# usage:
# update_gemm_params("your_model_decoder.ncnn.param")
```

## 5. implementing kv cache inference logic

Your C++ inference code must manage the cache blobs across decoding steps.

### step 1: identify cache blob indices
After loading the network, identify the input and output blob indices for the cache. You can iterate through the mha layers and find the blobs you named in the conversion script.

```cpp
#include "net.h"
#include <vector>
#include <string>

struct kvcache_info
{
    std::vector<int> input_indices;
    std::vector<int> output_indices;
};

void find_mha_kvcache_blobs(const ncnn::Net& net, kvcache_info& info)
{
    for (const ncnn::Layer* layer : net.layers())
    {
        // cache-enabled mha layer has 3 outputs (out, cache_k_out, cache_v_out) instead of 1
        if ((layer->typeindex == ncnn::LayerType::MultiHeadAttention || layer->typeindex == ncnn::LayerType::SDPA) && layer->tops.size() == 3)
        {
            // the script adds cache_k and cache_v as the last two inputs/outputs
            int input_count = layer->bottoms.size();
            int output_count = layer->tops.size();

            info.input_indices.push_back(layer->bottoms[input_count - 2]); // cache_k_in
            info.input_indices.push_back(layer->bottoms[input_count - 1]); // cache_v_in

            info.output_indices.push_back(layer->tops[output_count - 2]);  // cache_k_out, i.e., tops[1]
            info.output_indices.push_back(layer->tops[output_count - 1]);  // cache_v_out, i.e., tops[2]
        }
    }
}
```

### step 2: prefill and decode loop
The inference process is split into two phases: "prefill" for the initial prompt and "decode" for subsequent single-token generation.

- **prefill (`run_decoder_pre`):**
  - input: The entire initial sequence of token IDs
  - the kv cache is empty
  - run the decoder once
  - extract the output logits for the *last* token to predict the next token
  - extract the `out_cache_k` and `out_cache_v` blobs from all mha layers and store them

- **decode (`run_decoder_step`):**
  - input: The single, most recently generated token ID
  - the kv cache blobs from the previous step are fed as input
  - run the decoder
  - extract the output logits to predict the next token
  - extract and store the updated kv cache blobs for the next step

Here is a conceptual C++ implementation:

```cpp
// assume 'decoder_net' is loaded and 'kvcache_info' is populated.

// --- prefill step (processes a sequence of tokens) ---
void run_decoder_pre(const std::vector<int>& tokens, const ncnn::Mat& encoder_states, std::vector<ncnn::Mat>& out_kv_cache)
{
    ncnn::Extractor ex = decoder_net.create_extractor();

    ncnn::Mat input_embeds = prepare_input_embeds(tokens); // your embedding logic
    ex.input("in0", input_embeds); // use your input blob name
    ex.input("encoder_out", encoder_states); // use your encoder output blob name

    out_kv_cache.resize(kvcache_info.output_indices.size());
    for (size_t i = 0; i < kvcache_info.output_indices.size(); i++)
    {
        ex.extract(kvcache_info.output_indices[i], out_kv_cache[i]);
    }

    ncnn::Mat all_logits;
    ex.extract("out0", all_logits); // Use your output blob name
    // ... process logits for the last token ...
}

// --- decode step (processes a single token) ---
void run_decoder_step(int token, const ncnn::Mat& encoder_states, const std::vector<ncnn::Mat>& kv_cache, std::vector<ncnn::Mat>& out_kv_cache)
{
    ncnn::Extractor ex = decoder_net.create_extractor();

    ncnn::Mat input_embeds = prepare_input_embeds({token});
    ex.input("in0", input_embeds);
    ex.input("encoder_out", encoder_states);

    // feed the existing cache
    for (size_t i = 0; i < kvcache_info.input_indices.size(); i++)
    {
        ex.input(kvcache_info.input_indices[i], kv_cache[i]);
    }

    // extract the updated cache
    out_kv_cache.resize(kvcache_info.output_indices.size());
    for (size_t i = 0; i < kvcache_info.output_indices.size(); i++)
    {
        ex.extract(kvcache_info.output_indices[i], out_kv_cache[i]);
    }

    ncnn::Mat logits;
    ex.extract("out0", logits);
    // ... process logits to get the next token ...
}

// --- main inference loop ---
void generate_sequence()
{
    std::vector<int> initial_tokens = { /* SOT and prompt tokens */ };
    ncnn::Mat encoder_states = run_encoder(); // compute encoder output once

    // 1. prefill stage
    std::vector<ncnn::Mat> kv_cache;
    run_decoder_pre(initial_tokens, encoder_states, kv_cache);
    int next_token = get_next_token_from_prefill_logits();

    // 2. autoregressive decoding loop
    while (next_token != EOT_TOKEN && sequence_length < MAX_LENGTH)
    {
        std::vector<ncnn::Mat> next_kv_cache;
        run_decoder_step(next_token, encoder_states, kv_cache, next_kv_cache);
        kv_cache = next_kv_cache; // update cache for the next iteration

        next_token = get_next_token_from_step_logits();
        // append next_token to your generated sequence
    }
}
```
This structured approach allows ncnn to perform highly efficient Transformer inference, correctly handling both dynamic self-attention and static cross-attention caches with an optimized memory layout.
