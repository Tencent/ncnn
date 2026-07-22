# WebGPU Native Support for NCNN

This implementation adds WebGPU native support to NCNN, allowing the reuse of existing Vulkan compute shaders with automatic conversion to WebGPU-compatible format.

## Key Features

1. **Push Constant to Uniform Binding Conversion**: Automatically transforms Vulkan push constants to WebGPU uniform bindings
2. **Modified psc Macro**: Updated to use `float(x)==0` for WebGPU compatibility instead of `x==0`
3. **Seamless Integration**: Reuses existing Vulkan shader infrastructure with minimal changes

## Usage

WebGPU support is automatically enabled when compiling to WebAssembly (wasm) target with emscripten and Vulkan support is enabled:

```bash
# Use emscripten toolchain with Vulkan enabled
emcmake cmake .. -DNCNN_VULKAN=ON
```

This will automatically:
- Enable WebGPU when targeting emscripten + vulkan
- Transform all ~300+ compute shaders for WebGPU compatibility  
- Apply the correct psc macro definition

## Shader Transformation Example

**Vulkan (original):**
```glsl
layout (push_constant) uniform parameter
{
    int dims;
    int w;
    int h;
    int c;
    int cstep;
} p;
```

**WebGPU (transformed):**
```glsl
struct parameter
{
    int dims;
    int w;
    int h;
    int c;
    int cstep;
};
layout (binding = 1) uniform parameter_blob { parameter p; };
```

## Implementation Details

The implementation addresses the SPIR-V compilation issues mentioned in the GitHub issue:

1. **Error**: `unknown SPIR-V storage class: 9` - Fixed by converting push constants to uniform bindings
2. **Error**: `unhandled expression for ID 33` - Fixed by changing psc macro to use `float(x)==0`

## Files Modified

- `CMakeLists.txt`: Automatic WebGPU detection for emscripten + vulkan
- `src/gpu.cpp`: Updated psc macro for WebGPU compatibility
- `cmake/ncnn_add_shader.cmake`: Added WebGPU shader preprocessing path
- `cmake/ncnn_generate_webgpu_shader_header.cmake`: New shader transformation logic

## Building

```bash
# Standard build with WebGPU support using emscripten
mkdir build && cd build
emcmake cmake .. -DNCNN_VULKAN=ON -DNCNN_BUILD_TESTS=ON
emmake make -j$(nproc)
```

All 300+ compute shaders will be automatically transformed during the build process when targeting emscripten with vulkan enabled.