# CPU and GPU Operator Dispatch Logic

## CPU Layer Dispatch

The `ncnn_add_layer()` CMake macro (in `cmake/ncnn_add_layer.cmake`) accumulates layer information into CMake variables. Then `src/CMakeLists.txt` calls `configure_file()` to generate the following header files:

1. **Generic implementation**: `src/layer/<name>.cpp` — always compiled
2. **Arch-specific override**: `src/layer/<arch>/<name>_<arch>.cpp` — compiled if the file exists and the target arch matches
3. **Runtime CPU dispatch**: When `NCNN_RUNTIME_CPU=ON`, the build system generates additional variants for sub-ISA levels (e.g., `avx`, `avx2`, `avx512` for x86) and the runtime selects the best one based on detected CPU features

Generated headers (via `configure_file()` in `src/CMakeLists.txt`):
- `layer_declaration.h` — includes all layer headers and `DEFINE_LAYER_CREATOR()` macros
- `layer_registry.h` — a single file containing multiple registry arrays: `layer_registry[]` (generic), `layer_registry_arch[]` (arch-optimized), plus conditional arrays for runtime CPU dispatch (`layer_registry_avx512[]`, `layer_registry_avx[]`, `layer_registry_rvv[]`, etc.) and Vulkan (`layer_registry_vulkan[]`)
- `layer_type_enum.h` — enum of all layer type indices

At runtime, `Net` looks up the layer creator function from the registry. When `NCNN_RUNTIME_CPU=ON`, it uses the arch-optimized registry that was compiled with the appropriate ISA flags.

## Packing and Auto-Conversion

Before calling a layer's `forward()`, the inference engine (`Net`) automatically converts input blobs to the appropriate `elempack` based on the layer's support flags:

- If `support_packing = true`: the engine converts to the widest pack size that divides evenly into the channel count (e.g., `elempack=4` for SSE/NEON, `elempack=8` for AVX).
- If `support_any_packing = true`: the engine **skips** automatic packing conversion entirely and passes blobs as-is. The layer is responsible for handling any `elempack` value internally. This is useful for layers like Concat, Slice, or Reshape that can work with arbitrary packing.
- If `support_vulkan_any_packing = true`: same as `support_any_packing`, but for the Vulkan GPU path. The Vulkan engine skips automatic `VkMat` packing conversion and lets the layer handle it.

## Layer Inheritance Hierarchy

```
Layer (generic C++ in src/layer/relu.cpp)
  └── ReLU_x86 (x86 SIMD in src/layer/x86/relu_x86.cpp)
  └── ReLU_arm (ARM NEON in src/layer/arm/relu_arm.cpp)
  └── ReLU_riscv (RISC-V Vector in src/layer/riscv/relu_riscv.cpp)
  └── ReLU_vulkan (Vulkan GPU in src/layer/vulkan/relu_vulkan.cpp)
```

Each arch-specific class inherits from the generic layer and overrides `forward()` / `forward_inplace()`.

## Vulkan GPU Dispatch

Vulkan layers:
1. Inherit from the generic layer class
2. Set `support_vulkan = true` in constructor
3. Override Vulkan-specific methods: `upload_model()`, `create_pipeline()`, `destroy_pipeline()`, and `forward(VkMat...)` / `forward_inplace(VkMat...)`
4. Use GLSL compute shaders (`.comp` files in `src/layer/vulkan/shader/`)
5. Shaders are compiled to SPIR-V at build time via glslang, then embedded as C arrays

The Vulkan execution flow:
1. `upload_model()` — transfers weights to GPU memory
2. `create_pipeline()` — creates `VkComputePipeline` with specialization constants
3. `forward()` — records commands into `VkCompute` command buffer (dispatches compute shaders)
4. The `VkCompute` command buffer is submitted and executed on the GPU
