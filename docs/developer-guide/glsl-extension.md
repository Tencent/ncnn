# ncnn glsl extension

## rationale

# data types

## storage type

declare buffer data layout in descriptor binding

```glsl
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

```glsl
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

```glsl
shared lfp tmp_a[8][4][2];
```

|local type|fp32|fp16a|
|---|---|---|
|lfp|float|float16_t|
|lfpvec4|vec4|f16vec4|

## image format and precision hint type

declare image format in descriptor binding

```glsl
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

# misc functions

- prefer specialization constant over push constant

```c
T psc(T x)
```

Declare the same variable in specialization constant AND push constant section, then `psc(x)` will become a compile-time constant when specialization constant given non-zero or be dynamic via push constant otherwise. This is often used for tensor shape specialization. We can usually resolve all shape information and make them be compile-time constants for more aggressive shader optimization.

```glsl
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

```glsl
#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif
```

declare descriptor binding for image or buffer

```glsl
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
