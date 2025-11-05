# Understanding `support_XYZ` Properties in ncnn's `Layer` Class

This document is for developers implementing new layers in `ncnn`. It explains the `support_XYZ` boolean properties in the `ncnn::Layer` base class. Correctly setting these properties declares the capabilities of your layer to the `ncnn` inference engine. This allows the engine to apply specific optimizations, such as enabling SIMD, half-precision floating-point computation, or Vulkan GPU acceleration, to achieve optimal performance and memory efficiency.

## When to Set `support` Properties

A layer can set its `support` properties in two ways:

1.  **Statically in the constructor**: If the layer's capabilities are fixed, the simplest way is to set them in its constructor.
2.  **Dynamically in `create_pipeline`**: If the layer's capabilities depend on parameters loaded from `load_param` or `load_model` (e.g., the data type of weights), you can set these properties dynamically within the `create_pipeline` method.

---

## Property Details

Here is a detailed breakdown of each `support` property and what it means for your layer's implementation.

### `one_blob_only`

*   **Purpose**: Declares that the layer accepts only one input `blob` and produces only one output `blob`.
*   **Requirements if `true`**: You must implement the single-input, single-output version of the `forward` method:
    ```cpp
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    ```
*   **Behavior**: When `true`, `ncnn` calls this overload. If `false` (default), the `std::vector<Mat>` version of `forward` is called.

### `support_inplace`

*   **Purpose**: Declares that the layer supports in-place computation, meaning the input and output can share the same memory. This significantly reduces memory overhead.
*   **Requirements if `true`**: You must implement the `forward_inplace` method. Depending on whether `one_blob_only` is also enabled, implement the corresponding version:
    ```cpp
    // If one_blob_only is true
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    // If one_blob_only is false
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    ```

### `support_vulkan`

*   **Purpose**: Declares that the layer has a Vulkan implementation for GPU-accelerated inference.
*   **Requirements if `true`**:
    *   Implement `forward` / `forward_inplace` methods that accept `VkMat` for input and output.
    *   Implement `upload_model` to transfer weight data to the GPU.
    *   Implement `create_pipeline` and `destroy_pipeline` to manage Vulkan `Pipeline` objects and other GPU resources.

### `support_packing` (for CPU)

*   **Purpose**: Declares that the layer's **CPU implementation** can handle `Mat` data with a "packing" memory layout (i.e., `elempack > 1`). This is crucial for SIMD optimizations (e.g., processing 4 or 8 floats at once with NEON or AVX).
*   **Behavior if `true`**:
    *   When the input `Mat` channel count is a multiple of the SIMD width, the `ncnn` engine ensures that the input `Mat` passed to `forward` / `forward_inplace` is packed (e.g., `elempack=4` or `elempack=8`).
    *   Your implementation must correctly handle `Mat` data where `cstep` and `elempack` are not their default values.
*   **Behavior if `false`**:
    *   The `ncnn` engine guarantees that the input `Mat` passed to your layer will always have `elempack=1`. The engine will automatically insert conversions if the preceding layer produced a packed output.
*   **Output**: Regardless of the property's value, your layer can output a `Mat` with any `elempack`. However, it is highly recommended to output a `Mat` with an adaptive `elempack` to avoid unnecessary conversions in subsequent layers.

### `support_vulkan_packing` (Conceptual for Vulkan)

*   **Purpose**: This is the Vulkan equivalent of `support_packing`. It declares that the layer's **Vulkan implementation** can handle `VkMat` with `elempack=4`.
*   **Behavior if `true`**: When the input `VkMat` has a channel count that is a multiple of 4, the `ncnn` engine will provide a packed `VkMat` (with `elempack=4`) to your Vulkan `forward` methods.
*   **Behavior if `false`**: The engine will ensure the input `VkMat` has `elempack=1`.
*   **Note**: `support_packing` and `support_vulkan_packing` are independent. A layer can support packing on CPU but not on Vulkan, or vice-versa.

### `support_bf16_storage`

*   **Purpose**: Declares that the layer can process `bfloat16` data.
*   **Behavior if `true`**:
    *   The `forward` method may receive an input `Mat` of type `bfloat16` (`elembits() == 16`) or `fp32`.
    *   Inside your `forward` implementation, you must check `opt.use_bf16_storage` and `bottom_blob.elembits()` to determine whether to use a `bfloat16`-optimized code path.
*   **Behavior if `false`**: The `ncnn` engine ensures your layer will **not** receive a `bfloat16` `Mat`.
*   **Output**: Your layer can output either a `bfloat16` or `fp32` `Mat`. When `opt.use_bf16_storage` is active, outputting `bfloat16` is recommended to maintain precision and performance across the network.

### `support_fp16_storage`

*   **Purpose**: Declares that the layer can process `float16` data for half-precision inference.
*   **Behavior if `true`**:
    *   Similar to `support_bf16_storage`, the `forward` method may receive an `fp16` or `fp32` `Mat`.
    *   Your implementation should check `opt.use_fp16_storage` and `bottom_blob.elembits()` to select the correct code path.
*   **Behavior if `false`**: The `ncnn` engine ensures your layer will **not** receive an `fp16` `Mat`.
*   **Output**: Your layer can output either a `fp16` or `fp32` `Mat`. When `opt.use_fp16_storage` is active, outputting an `fp16` `Mat` is recommended.

### `support_int8_storage`

*   **Purpose**: Declares that the layer supports `int8` quantized inference.
*   **Behavior if `true`**:
    *   When `opt.use_int8_inference` is `true`, the `forward` method may receive an `int8` or `fp32` `Mat`.
    *   **Important**: If the input is `fp32`, your `forward` implementation is responsible for dynamically quantizing it to `int8` before performing computations.
*   **Behavior if `false`**: The `ncnn` engine ensures your layer will **not** receive an `int8` `Mat`.
*   **Output**: The output can be `int8` or `fp32`, depending on your layer's design.

---

## Practical Implementation and Priorities

### Handling Multiple Precision Types

A layer can set `support_fp16_storage` and `support_bf16_storage` to `true` simultaneously. The `ncnn` engine prioritizes these formats based on the `Option` flags. As seen in the `convert_layout` function in `src/net.cpp`, if `opt.use_bf16_storage` is true, the engine will prefer converting inputs to `bfloat16`. Otherwise, it falls back to `fp16` if `opt.use_fp16_storage` is true.

The chosen `elempack` also depends on the precision. For instance, with SIMD, the priority might be:
*   FP16: `elempack=8` (if supported), then `elempack=4`, then `1`.
*   BF16: `elempack=4`, then `1`.

Your `forward` implementation should reflect this by checking `elembits()` and `elempack` to dispatch to the correct kernel.

### Code Example: `Clip_arm`

The `Clip_arm` layer provides a great example of these concepts in practice.

1.  **Declaring Support in the Constructor**:
    It declares support for packing and, conditionally, for fp16 and bf16 storage.
    ```cpp
    // From: src/layer/arm/clip_arm.cpp
    Clip_arm::Clip_arm()
    {
    #if __ARM_NEON
        support_packing = true;
    #if NCNN_ARM82
        support_fp16_storage = cpu_support_arm_asimdhp();
    #endif
    #endif // __ARM_NEON

    #if NCNN_BF16
        support_bf16_storage = true;
    #endif
    }
    ```

2.  **Dispatching in `forward_inplace`**:
    The `forward_inplace` method acts as a dispatcher. It first checks the element size (`elembits`) and the corresponding `opt` flag to decide whether to call a specialized low-precision implementation (`fp16s` or `bf16s`). If neither is applicable, it defaults to the standard `fp32` implementation.

    ```cpp
    // From: src/layer/arm/clip_arm.cpp
    int Clip_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
    {
        int elembits = bottom_top_blob.elembits();

    #if NCNN_ARM82
        if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
            return forward_inplace_fp16s(bottom_top_blob, opt);
    #endif

    #if NCNN_BF16
        if (opt.use_bf16_storage && elembits == 16)
            return forward_inplace_bf16s(bottom_top_blob, opt);
    #endif

        // Default fp32 implementation follows...
        int w = bottom_top_blob.w;
        // ...
    }
    ```

### An Incremental Development Workflow

Adopting a gradual approach can simplify the development of a new layer:

1.  **Implement the Core Algorithm**: Start with all `support_XYZ` properties set to `false`. Focus on getting the mathematical logic correct using standard `fp32` data and `elempack=1`.
2.  **Add Packing Support**: Once the core logic is validated, set `support_packing = true`. Modify your code to handle `elempack > 1` and implement SIMD optimizations (e.g., using NEON intrinsics).
3.  **Add Low-Precision Support**: Next, add support for `fp16`, `bf16`, or `int8`. Set the corresponding `support_*_storage` flags to `true` and add branches in your `forward` method to handle these data types based on the `opt` flags.
4.  **Add Vulkan Support**: Finally, if GPU acceleration is desired, set `support_vulkan = true` and implement the Vulkan-specific methods.

This incremental process allows you to tackle one challenge at a time, making it easier to develop a highly optimized and feature-rich layer.
