# ncnn GLSL extension

## rationale
Different GPUs support different features, some support fp16 as buffer storage type, some support fp16 as operand variable, some old GPUs only support fp32

When the GPU supports the `VK_KHR_16bit_storage` extension, in order to minimize the memory bandwidth consumption of the GPU, we will give priority to using fp16 as the storage type. Otherwise, we use `packHalf2x16` and `unpackHalf2x16` in GLSL 4.2 to compress 2 fp32 to uint, reducing read and write bandwidth.

Similarly, when the gpu supports the `VK_KHR_shader_float16_int8` extension, in order to speed up the calculation efficiency, we will give priority to using fp16 as the operation operand, which usually doubles the speed. Otherwise, we use fp32.

To ensure the widest compatibility, the following code for declaring descriptor binding and loading data will be written

```c
#if NCNN_fp16_storage // gpu supports 16bit storage
layout (binding = 0) buffer blob { f16vec4 blob_data[]; };
#elif NCNN_fp16_packed // gpu supports GLSL 4.2
layout (binding = 0) buffer blob { uvec2 blob_data[]; };
#else // gpu only supports fp32
layout (binding = 0) buffer blob { vec4 blob_data[]; };
#endif

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

#if NCNN_fp16_storage && NCNN_fp16_arithmetic // gpu supports 16bit storage and shader float16
    f16vec4 x = blob_data[i];
#elif NCNN_fp16_storage // gpu supports 16bit storage but no shader float16
    vec4 x = vec4(blob_data[i]);
#elif NCNN_fp16_packed && NCNN_fp16_arithmetic // gpu supports GLSL 4.2 and shader float16
    f16vec4 x = f16vec4(unpackFloat2x16(blob_data[i].x), unpackFloat2x16(blob_data[i].y));
#elif NCNN_fp16_packed // gpu supports GLSL 4.2
    vec4 x = vec4(unpackHalf2x16(blob_data[i].x), unpackHalf2x16(blob_data[i].y));
#else // gpu only supports fp32
    vec4 x = blob_data[i];
#endif
}
```

As you can see, just declaring the buffer type and reading a value consumes a lot of lines of code, which is a maintenance nightmare. Therefore, ncnn adds more flexible data types and auxiliary functions to reduce the size of the code and improve readability, and will automatically expand to the most efficient implementation according to the feature level supported by the GPU.

The above code, by using the ncnn glsl extension, can be simplified to

```c
layout (binding = 0) buffer blob { sfpvec4 blob_data[]; };

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

    afpvec4 x = buffer_ld4(blob_data, i);
}
```

The ncnn glsl extension provides the necessary data types for storage, computation, shared memory, and load, store, conversion functions for buffers and images. We also provide some buffer and image copy functions to prevent loss of precision when using fp16 as the intermediate data type, and to avoid unnecessary `unpackHalf2x16` and `packHalf2x16` pair.

# entrypoint for compiling GLSL

The gpu.h header in the ncnn library exposes 3 APIs for compiling glsl code into spir-v binary, they support ncnn glsl extension, these 3 functions accept opt switch to control the expansion form of ncnn glsl extension. The first two accept raw glsl code strings, and the last one is used to create ncnn's built-in shader.

```cpp
namespace ncnn {

// online spirv compilation
NCNN_EXPORT int compile_spirv_module(const char* comp_string, const Option& opt, std::vector<uint32_t>& spirv);
NCNN_EXPORT int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv);
NCNN_EXPORT int compile_spirv_module(int shader_type_index, const Option& opt, std::vector<uint32_t>& spirv);

} // namespace ncnn
```

## compile ncnn extended GLSL code directly

You can write shader code with ncnn glsl extension, compiled to spir-v using ncnn functions. The compiled product is a standard-compliant spir-v binary, which can be directly used to create a pipeline object in the vulkan api

```cpp
static const char my_glsl_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (binding = 0) readonly buffer a_blob { sfpvec4 a_blob_data[]; };
layout (binding = 1) writeonly buffer b_blob { sfpvec4 b_blob_data[]; };

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

    afpvec4 v = buffer_ld4(a_blob_data, i);

    v = v + 123;

    buffer_st4(b_blob_data, i, v);
}
)";

Option opt;
 // you can control the extention behavior
 // even if the gpu supports 16bit storage
opt.use_fp16_storage = false;

std::vector<uint32_t> spirv;
ncnn::compile_spirv_module(my_glsl_data, sizeof(my_glsl_data) - 1, opt, spirv);

// To create pipeline object later
// ncnn::Pipeline pipeline(vkdev);
// pipeline.set_local_size_xyz(64, 1, 1);
// pipeline.create(spirv.data(), spirv.size() * 4, specializations);
```

## ncnn built-in shader

The shader index inside ncnn is exposed in the `layer_shader_type.h` header and can be used if needed

```cpp
#include "layer_shader_type.h"

int shader_type_index = LayerShaderType::convert_ycbcr;

Option opt;

std::vector<uint32_t> spirv;
int retc = compile_spirv_module(shader_type_index, opt, spirv);
```

# data types

## storage type

declare buffer data layout in descriptor binding

```c
layout (binding = 0) buffer top_blob { sfpvec4 top_blob_data[]; };
```

|storage type|fp32|fp16p|fp16s|
|---|---|---|---|
|sfp|float|float|float16_t|
|sfpvec2|vec2|uint|f16vec2|
|sfpvec4|vec4|uvec2|f16vec4|
|sfpvec8|mat2x4|uvec4|f16mat2x4|

## arithmetic type

declare local variable in glsl code

```c
void main()
{
    afpvec4 v = a * b;
}
```

|arithmetic type|fp32|fp16a|
|---|---|---|
|afp|float|float16_t|
|afpvec2|vec2|f16vec2|
|afpvec4|vec4|f16vec4|
|afpvec8|mat2x4|f16mat2x4|

## local type

declare variable in shared local memory

```c
shared lfp tmp_a[8][4][2];
```

|local type|fp32|fp16p / fp16s|fp16s + fp16a|
|---|---|---|---|
|lfp|float|float|float16_t|
|lfpvec4|vec4|uvec2|f16vec4|

## image format and precision hint type

declare image format in descriptor binding

```c
layout (binding = 0) uniform unfp sampler3D bottom_blob_3d;
layout (binding = 1, imfmtc4) writeonly uniform unfp image3D top_blob_3d;
```

|format type|fp32|fp16p|fp16s|
|---|---|---|---|
|imfmt1|r32f|f32f|r16f|
|imfmt4|rgba32f|rgba16f|rgba16f|

|precision hint type|fp32|fp16p|fp16s|
|---|---|---|---|
|unfp|highp|mediump|mediump|

# buffer functions

- load typed value from src[offset]

```c
afp buffer_ld1(sfp src, int offset);
afpvec2 buffer_ld2(sfpvec2 src, int offset);
afpvec4 buffer_ld4(sfpvec4 src, int offset);
afpvec8 buffer_ld8(sfpvec8 src, int offset);
```

- store typed value to dst[offset]

```c
void buffer_st1(sfp dst, int offset, afp v);
void buffer_st2(sfpvec2 dst, int offset, afpvec2 v);
void buffer_st4(sfpvec4 dst, int offset, afpvec4 v);
void buffer_st8(sfpvec8 dst, int offset, afpvec8 v);
```

- copy typed value from src[src_offset] to dst[dst_offset]

```c
void buffer_cp1(sfp dst, int dst_offset, sfp src, int src_offset);
void buffer_cp2(sfpvec2 dst, int dst_offset, sfpvec2 src, int src_offset);
void buffer_cp4(sfpvec4 dst, int dst_offset, sfpvec4 src, int src_offset);
void buffer_cp8(sfpvec4 dst, int dst_offset, sfpvec4 src, int src_offset);
```

- copy and pack value from src[src_offsets[0],src_offsets[1],...] to dst[dst_offset]

```c
void buffer_cp1to4(sfpvec4 dst, int dst_offset, sfp src, ivec4 src_offsets);
void buffer_cp1to8(sfpvec8 dst, int dst_offset, sfp src, ivec4 src_offsets_0, ivec4 src_offsets_1);
void buffer_cp4to8(sfpvec8 dst, int dst_offset, sfpvec4 src, ivec2 src_offsets);
```

- copy and unpack value from src[src_offset] to dst[dst_offsets[0],dst_offsets[1],...]

```c
void buffer_cp4to1(sfp dst, ivec4 dst_offsets, sfpvec4 src, int src_offset);
void buffer_cp8to1(sfp dst, ivec4 dst_offsets_0, ivec4 dst_offsets_1, sfpvec8 src, int src_offset);
void buffer_cp8to4(sfpvec4 dst, ivec2 dst_offsets, sfpvec8 src, int src_offset);
```

# image functions

- load typed value from src at pos

```c
afp image1d_ld1(sampler1D src, float pos);
afp image2d_ld1(sampler2D src, vec2 pos);
afp image3d_ld1(sampler3D src, vec3 pos);
afpvec4 image1d_ld4(sampler1D src, float pos);
afpvec4 image2d_ld4(sampler2D src, vec2 pos);
afpvec4 image3d_ld4(sampler3D src, vec3 pos);
afpvec8 image1d_ld8(sampler1D src, float pos);
afpvec8 image2d_ld8(sampler2D src, vec2 pos);
afpvec8 image3d_ld8(sampler3D src, vec3 pos);
```

- store typed value to dst at pos

```c
void image1d_st1(image1D dst, int pos, afp v);
void image2d_st1(image2D dst, ivec2 pos, afp v);
void image3d_st1(image3D dst, ivec3 pos, afp v);
void image1d_st4(image1D dst, int pos, afpvec4 v);
void image2d_st4(image2D dst, ivec2 pos, afpvec4 v);
void image3d_st4(image3D dst, ivec3 pos, afpvec4 v);
void image1d_st8(image1D dst, int pos, afpvec8 v);
void image2d_st8(image2D dst, ivec2 pos, afpvec8 v);
void image3d_st8(image3D dst, ivec3 pos, afpvec8 v);
```

- copy typed value from src at src_pos to dst at dst_pos

```c
void image1d_cp1(image1D dst, int dst_pos, sampler1D src, float src_pos);
void image2d_cp1(image2D dst, ivec2 dst_pos, sampler2D src, vec2 src_pos);
void image3d_cp1(image3D dst, ivec3 dst_pos, sampler3D src, vec3 src_pos);
void image1d_cp4(image1D dst, int dst_pos, sampler1D src, float src_pos);
void image2d_cp4(image2D dst, ivec2 dst_pos, sampler2D src, vec2 src_pos);
void image3d_cp4(image3D dst, ivec3 dst_pos, sampler3D src, vec3 src_pos);
void image1d_cp8(image1D dst, int dst_pos, sampler1D src, float src_pos);
void image2d_cp8(image2D dst, ivec2 dst_pos, sampler2D src, vec2 src_pos);
void image3d_cp8(image3D dst, ivec3 dst_pos, sampler3D src, vec3 src_pos);
```

Note: Since image is an opaque data structure, no copy and pack/unpack functions are provided. To achieve this operation, you need to load first and then store.

# local data conversion functions

- storage buffer to local memory

```c
lfp sfp2lfp(sfp v);
lfpvec4 sfp2lfpvec4(sfpvec4 v);
```

- local memory to local variable

```c
afp lfp2afp(lfp v);
afpvec4 lfp2afpvec4(lfpvec4 v);
```

Note: The common usage of local memory is to read from global memory first, store it in local memory, and then read local variables from local memory for subsequent use. Therefore, only storage type to local type and local type to arithmetic type conversion functions are provided here.

# misc functions

- prefer specialization constant over push constant

```c
T psc(T x)
```

Declare the same variable in specialization constant AND push constant section, then `psc(x)` will become a compile-time constant when specialization constant given non-zero or be dynamic via push constant otherwise. This is often used for tensor shape specialization. We can usually resolve all shape information and make them be compile-time constants for more aggressive shader optimization.

```c
layout (constant_id = 0) const int size = 0;

layout (push_constant) uniform parameter
{
    int size;
} p;

void main()
{
    const int s = psc(size);
}
```

# platform macros

judge if the current platform is moltenvk, for enabling some platform-specific workaround

```c
#if NCNN_moltenvk
// enable workaround for moltenvk
#endif
```

# option macros

enable glsl extension only if user enable some options

```c
#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif
```

declare descriptor binding for image or buffer

```c
#if NCNN_image_shader
layout (binding = 0) uniform unfp sampler3D bottom_blob_3d;
#else
layout (binding = 0) readonly buffer bottom_blob { sfpvec4 bottom_blob_data[]; };
#endif
```

|macro|defined by option|
|---|---|
|NCNN_fp16_packed|opt.use_fp16_packed|
|NCNN_fp16_storage|opt.use_fp16_storage|
|NCNN_fp16_arithmetic|opt.use_fp16_arithmetic|
|NCNN_int8_packed|opt.use_int8_packed|
|NCNN_int8_storage|opt.use_int8_storage|
|NCNN_int8_arithmetic|opt.use_int8_arithmetic|
|NCNN_image_shader|opt.use_image_storage|
|NCNN_shader_local_memory|opt.use_shader_local_memory|
