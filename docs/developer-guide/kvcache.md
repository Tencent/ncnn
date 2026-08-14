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
- **ncnn implementation:** Cache-aware `MultiHeadAttention` can identify its separate query and key/value input form and reuses the cached encoder k/v after the first invocation. `SDPA` does not infer self-attention or cross-attention from tensor shapes; its cache mode treats current k/v as append data. A graph using static encoder k/v with `SDPA` should keep those tensors as ordinary inputs rather than relying on a shape heuristic.

## 3. kv cache storage

Parameter `7=1` selects the cache-aware implementation of `MultiHeadAttention` and `SDPA`. There is one forward implementation per backend; the old concat layer path is not retained. The backend calls `create()` for the first cache and `expand(input_cache, output_cache, new_seqlen)` on later logical expansions. The input handle remains unchanged. Within capacity the output handle shares its allocation and only carries the new logical length; otherwise storage allocates the output and relocates valid history. The contiguous CPU and Vulkan storage implementations retain spare capacity and relocate only when capacity is exhausted.

`Option::kvcache_storage` selects the storage implementation used by a decode session. The application creates one `CPUKVCacheStorage` or `VkKVCacheStorage` for each session/thread and passes the same pointer to every extractor with `Extractor::set_kvcache_storage()`. The pointer is non-owning and must remain unchanged while its cache handles are in use. Setting `Net::opt.kvcache_storage` is also supported when the whole Net is dedicated to one session, but a shared Net must not own shared mutable cache state.

The storage constructor accepts an optional maximum sequence-length hint. For example, `CPUKVCacheStorage(max_context_length)` and `VkKVCacheStorage(vkdev, max_context_length)` reserve that capacity when the first cache is created. The hint belongs to the session storage rather than `Option`, and it is not a hard limit: a cache can still grow beyond it. Without a usable hint, the contiguous storage implementations use the same moderate policy for every cache: small caches initially reserve at least 16 sequence positions, while a larger prefill reserves at most 256 extra positions. Later relocation grows capacity by roughly one half. This avoids doubling a large prefill allocation while still amortizing single-token decode growth; it does not try to infer self-attention or cross-attention.

When no storage is supplied, the layer uses a stateless naive adapter over the current blob allocator. Cache inputs and outputs remain ordinary `Mat` or `VkMat` objects; every growth allocates a new tensor and copies valid history. This preserves the old ncnn_llm-style pattern of extracting cache tensors and feeding them unchanged into a later extractor. Neither `Net` nor `Extractor` creates or owns a default KV cache storage.

The first invocation uses empty cache blobs and later invocations feed the two cache outputs back through the same two input positions. Parameter `7=1`, blob positions, and the ordinary `input()` / `extract()` calling pattern remain compatible.

### managed cache Mat is an opaque handle

A cache `Mat` created by an explicit storage is backend-affine session state, not an ordinary public tensor. Applications must not depend on its `w`, `h`, `c`, `cstep`, `data`, capacity, or element order. The no-storage compatibility path intentionally returns ordinary tensors instead.

Opaque here is an application contract, not a claim that the current layer implementations are layout-independent. Each optimized layer and its concrete storage implementation must agree on a private cache layout. The current CPU and Vulkan implementations use contiguous `Mat` / `VkMat` storage and inspect the handle internally. A future paged-attention implementation may require coordinated storage and backend changes, and should not expose those details to the application.

Generic ncnn execution recognizes a cache by calling `Option::kvcache_storage->owns()`. It preserves that handle without packing, casting, or detaching it. A handle from incompatible storage or a different backend is rejected; there is no layout tag, blob flag, or global data-to-owner map.

Explicit Vulkan storage produces `VkMat` handles, which must remain on Vulkan and use the `VkMat` extractor overload. Managed cache handles are not implicitly converted between CPU and Vulkan storage. The no-storage compatibility path may use ordinary upload/download behavior, but an application seeking the fast Vulkan path should keep an externally managed `VkMat` cache on device.

Vulkan cache expansion may record a relocation before a later allocation in the same layer fails. As with other failed Vulkan forwards, a nonzero return invalidates that `VkCompute` command sequence: discard or reset it without submission. Cache input handles remain valid, and cache outputs from the failed forward must not be used.

The direct-append implementation covers the generic CPU implementation plus the x86, ARM, RISC-V, MIPS, and LoongArch optimized backends, as well as Vulkan. All of them use the same storage lifecycle contract while retaining backend-specific projection, append, packing, and attention kernels. Cache and no-cache execution share the ordinary `forward()` data flow and branch only where cache storage layout or the corresponding attention pipeline differs; there is no separate cache forward path or cache preparation/append helper.

The optimized CPU backends create one QK/QKV pipeline matching parameter `7`. The GEMM transpose setting and `forward()` cache view therefore always describe the same layout.

The generic `SDPA` and `MultiHeadAttention` classes remain scalar reference implementations and do not build cache-specific Gemm or Softmax sublayers. Optimized cache layouts and cache-specific pipelines belong to their corresponding architecture backend.

### storage and handle lifetime

`Option::kvcache_storage` is non-owning. The decode session/thread owns it and every cache handle, and the storage must live across all extractors in that session. End the extractor scope, then pass all remaining handles to `KVCacheStorage::destroy()` before the storage object itself is destroyed.

Cache input follows a consume-and-replace convention. Feed the current handle to one decode step and replace it with that step's extracted handle. Existing code may extract directly into the same `Mat` variable; it must not keep another shallow alias for concurrent or later use.

The executor transfers cache inputs even when light mode is disabled. Light mode therefore does not change cache ownership; it only retains its ordinary meaning for non-cache tensors.

The storage object is allowed to reuse memory while shallow aliases exist because cache handles obey the consume-and-replace convention. A shallow copy is not a cache snapshot. Beam search must use independently created cache state rather than forking a handle by copying `Mat`.

Without explicit storage, cache tensors have ordinary `Mat` / `VkMat` lifetime. A Vulkan caller that keeps a `VkMat` cache across extractors must also provide persistent per-session blob, workspace, and staging Vulkan allocators to every extractor; extractor-local Vulkan allocators cannot own cross-extractor state.

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
#include "kvcache_storage.h"
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
// The real session owns this storage object and keeps it alive across every step.
ncnn::CPUKVCacheStorage kvcache_storage(MAX_LENGTH);

void load_decoder()
{
    decoder_net.load_param("decoder.ncnn.param");
    decoder_net.load_model("decoder.ncnn.bin");
}

// assume 'kvcache_info' is populated after loading decoder_net.

// --- prefill step (processes a sequence of tokens) ---
void run_decoder_pre(const std::vector<int>& tokens, const ncnn::Mat& encoder_states, std::vector<ncnn::Mat>& out_kv_cache)
{
    ncnn::Extractor ex = decoder_net.create_extractor();
    ex.set_kvcache_storage(&kvcache_storage);

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
void run_decoder_step(int token, const ncnn::Mat& encoder_states, std::vector<ncnn::Mat>& kv_cache, std::vector<ncnn::Mat>& out_kv_cache)
{
    ncnn::Extractor ex = decoder_net.create_extractor();
    ex.set_kvcache_storage(&kvcache_storage);

    ncnn::Mat input_embeds = prepare_input_embeds({token});
    ex.input("in0", input_embeds);
    ex.input("encoder_out", encoder_states);

    // feed the existing cache
    for (size_t i = 0; i < kvcache_info.input_indices.size(); i++)
    {
        ex.input(kvcache_info.input_indices[i], kv_cache[i]);
        // The extractor now owns the input handle for this step.
        kv_cache[i].release();
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

    for (size_t i = 0; i < kv_cache.size(); i++)
        kvcache_storage.destroy(kv_cache[i]);
}
```
The storage object must be created before model execution and must not be recreated inside each call. An existing application may omit it; cache outputs then remain ordinary tensors and keep the old allocation/copy behavior. For an explicitly managed Vulkan session, keep cache as `VkMat`, use `VkKVCacheStorage`, set the session's persistent Vulkan allocators and storage on every extractor, submit and wait for commands before storage destruction, and use the `Extractor::extract(VkMat&, VkCompute&)` overload.

This structured approach allows ncnn to perform highly efficient Transformer inference while preserving dynamic self-attention and static cross-attention cache semantics.
