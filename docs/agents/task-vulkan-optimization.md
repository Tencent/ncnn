# Task: Add Vulkan GPU Optimization for an Existing Operator

1. **Create the Vulkan layer** — `src/layer/vulkan/<name>_vulkan.h`
   ```cpp
   #ifndef LAYER_NEWOP_VULKAN_H
   #define LAYER_NEWOP_VULKAN_H
   #include "newop.h"
   namespace ncnn {
   class NewOp_vulkan : public NewOp
   {
   public:
       NewOp_vulkan();
       virtual int create_pipeline(const Option& opt);
       virtual int destroy_pipeline(const Option& opt);
       virtual int upload_model(VkTransfer& cmd, const Option& opt);
       virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
   public:
       Pipeline* pipeline_newop;
       Pipeline* pipeline_newop_pack4;
   };
   } // namespace ncnn
   #endif
   ```

2. **Implement the Vulkan layer** — `src/layer/vulkan/<name>_vulkan.cpp`
   - In `create_pipeline()`: create `Pipeline` objects with shader paths and specialization constants
   - In `forward()`: bind descriptors, set push constants, dispatch compute via `cmd.record_pipeline()`

3. **Write GLSL compute shaders** — `src/layer/vulkan/shader/<name>.comp`
   ```glsl
   #version 450
   #if NCNN_fp16_storage
   #extension GL_EXT_shader_16bit_storage: require
   #endif
   layout (binding = 0) readonly buffer bottom_blob { sfp bottom_blob_data[]; };
   layout (binding = 1) writeonly buffer top_blob { sfp top_blob_data[]; };
   layout (push_constant) uniform parameter { int w; int h; int c; /* ... */ } p;
   void main()
   {
       int gx = int(gl_GlobalInvocationID.x);
       // ... compute
   }
   ```

   Create a packing variant: `<name>_pack4.comp`.

4. **No CMakeLists.txt changes needed** — `ncnn_add_layer()` auto-detects `src/layer/vulkan/<name>_vulkan.cpp` and associated `.comp` shaders.

5. **Test with software Vulkan** — CI uses SwiftShader or llvmpipe as software Vulkan implementations. Tests automatically run Vulkan paths when `NCNN_VULKAN=ON`.

## Vulkan Shader Conventions

- Use `sfp` (storage float, may be fp16) and `afp` (arithmetic float) type aliases from ncnn's GLSL extensions
- Support packing variant: `_pack4.comp` (4-element packed). Pack8 shaders have been removed from the project.
- Use push constants for dimensions and parameters
- Use specialization constants for compile-time configuration
- See `docs/developer-guide/glsl-extension.md` for ncnn-specific GLSL extensions
