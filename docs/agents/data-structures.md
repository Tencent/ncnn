# Key Data Structures

## Mat (`src/mat.h`)

The core tensor type. Supports 1D to 4D data with element packing for SIMD.

```cpp
class Mat {
    void* data;          // Raw data pointer
    int* refcount;       // Reference counting (NULL for external data)
    size_t elemsize;     // Bytes per element (4=fp32, 2=fp16, 1=int8 when elempack=1;
                         //   equals scalar_size * elempack when packed, e.g., 16 for pack4 fp32)
    int elempack;        // Packed elements (1=scalar, 4=SSE/NEON, 8=AVX/fp16)
    Allocator* allocator;
    int dims;            // 0=empty, 1=1D, 2=2D, 3=3D, 4=4D
    int w, h, d, c;      // Width, height, depth, channels
    size_t cstep;        // Channel stride (elements per channel)
};
```

Key concepts:
- **Element packing (`elempack`)**: Multiple elements stored together for SIMD. E.g., `elempack=4` means 4 floats packed as one unit (for SSE/NEON 128-bit). `elempack=8` for AVX 256-bit. Channel count `c` is divided by `elempack`.
- **Channel step (`cstep`)**: Aligned stride between channels for SIMD alignment.
- GPU variants: `VkMat` (Vulkan buffer), `VkImageMat` (Vulkan image).

## Net (`src/net.h`)

The inference engine. Loads param (graph) and model (weights), creates `Extractor` for inference.

```cpp
class Net {
    Option opt;                    // Runtime options
    int load_param(const char*);   // Load graph structure (.param)
    int load_model(const char*);   // Load weights (.bin)
    Extractor create_extractor();  // Create inference session
};

class Extractor {
    int input(const char* name, const Mat& in);   // Set input
    int extract(const char* name, Mat& out);       // Get output (runs inference)
};
```

## Layer (`src/layer.h`)

Base class for all operators. Key behavioral flags set in constructor:

```cpp
class Layer {
    bool one_blob_only;     // Single input/output (e.g., ReLU)
    bool support_inplace;   // Can modify input in-place
    bool support_packing;   // Accepts packed Mat (elempack > 1)
    bool support_vulkan;    // Has Vulkan implementation
    bool support_bf16_storage;
    bool support_fp16_storage;
    bool support_int8_storage;
    bool support_any_packing;         // Layer handles any elempack internally (skip auto packing conversion)
    bool support_vulkan_any_packing;  // Same as above, but for Vulkan path

    // CPU forward
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    // Vulkan forward
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

    virtual int load_param(const ParamDict& pd);  // Load params from .param
    virtual int load_model(const ModelBin& mb);    // Load weights from .bin
    virtual int create_pipeline(const Option& opt); // Setup (e.g., create Vulkan pipelines)
    virtual int destroy_pipeline(const Option& opt);
    virtual int upload_model(VkTransfer& cmd, const Option& opt); // Upload weights to GPU
};
```

Forward interface selection table:

| one_blob_only | support_inplace | Required interface |
|---|---|---|
| false | false | `forward(vector<Mat>, vector<Mat>)` |
| false | true | `forward_inplace(vector<Mat>)` (must), `forward(vector<Mat>, vector<Mat>)` (optional) |
| true | false | `forward(Mat, Mat)` |
| true | true | `forward_inplace(Mat)` (must), `forward(Mat, Mat)` (optional) |

## Blob (`src/blob.h`)

A named tensor edge in the computation graph. Each blob has a producer layer and consumer layers.

## ParamDict (`src/paramdict.h`)

Key-value store for layer parameters. Keys are integers (0, 1, 2, ...). Values can be int, float, or arrays thereof. Used in `.param` files as `key=value`.

## Option (`src/option.h`)

Runtime configuration: `num_threads`, `use_vulkan_compute`, `use_fp16_packed`, `use_bf16_storage`, blob/workspace allocators, etc.
