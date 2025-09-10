# ncnn GLSL 扩展

## 理由
不同的 GPU 支持不同的功能，有的支持 fp16 作为缓冲存储类型，有的支持 fp16 作为操作数变量，有的老 GPU 只支持 fp32。

当 GPU 支持 `VK_KHR_16bit_storage` 扩展时，为了尽量减少 GPU 的内存带宽消耗，我们会优先使用 fp16 作为存储类型。否则，我们使用 `packHalf2x16` 和 `unpackHalf2x16` 在 GLSL 4.2 中将 2 个 fp32 压缩为 uint，从而减少读写带宽。

同样，当 GPU 支持 `VK_KHR_shader_float16_int8` 扩展时，为了加快计算效率，我们会优先使用 fp16 作为运算操作数，这通常会使速度翻倍。否则，我们使用 fp32。

为了确保最广泛的兼容性，将编写以下用于声明描述符绑定和加载数据的代码

```c
#if NCNN_fp16_storage // GPU支持 16bit storage
layout (binding = 0) buffer blob { f16vec4 blob_data[]; };
#elif NCNN_fp16_packed // GPU支持 GLSL 4.2
layout (binding = 0) buffer blob { uvec2 blob_data[]; };
#else // GPU仅支持 fp32
layout (binding = 0) buffer blob { vec4 blob_data[]; };
#endif

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

#if NCNN_fp16_storage && NCNN_fp16_arithmetic // GPU支持 16bit storage 和 shader float16
    f16vec4 x = blob_data[i];
#elif NCNN_fp16_storage // GPU支持 16bit storage 但不包含 shader float16
    vec4 x = vec4(blob_data[i]);
#elif NCNN_fp16_packed && NCNN_fp16_arithmetic // GPU支持 GLSL 4.2 和 shader float16
    f16vec4 x = f16vec4(unpackFloat2x16(blob_data[i].x), unpackFloat2x16(blob_data[i].y));
#elif NCNN_fp16_packed // GPU支持 GLSL 4.2
    vec4 x = vec4(unpackHalf2x16(blob_data[i].x), unpackHalf2x16(blob_data[i].y));
#else // GPU仅支持 fp32
    vec4 x = blob_data[i];
#endif
}
```

如您所见，仅声明缓冲区类型并读取值会消耗大量代码行，这是项目维护的噩梦。因此，ncnn 增加了更灵活的数据类型和辅助函数，以减小代码的大小并提高可读性，并且会根据 GPU 支持的功能级别自动扩展到最高效的实现。

上面的代码，通过使用 ncnn GLSL 扩展，可以简化为

```c
layout (binding = 0) buffer blob { sfpvec4 blob_data[]; };

void main()
{
    const int i = int(gl_GlobalInvocationID.x);

    afpvec4 x = buffer_ld4(blob_data, i);
}
```

ncnn GLSL 扩展为存储、计算、共享内存以及缓冲区和图像的加载、存储、转换函数提供了必要的数据类型。我们还提供了一些缓冲区和图像复制函数，以防止在使用 fp16 作为中间数据类型时丢失精度，并避免不必要的 `unpackHalf2x16` 和 `packHalf2x16` 配对。

# 编译GLSL的入口点

ncnn库中的 gpu.h 头文件公开了3个用于将 GLSL 代码编译为 Spir-V 二进制的API函数，它们支持 ncnn GLSL 扩展，这3个函数接受 opt switch 来控制 ncnn GLSL 扩展形式。前两个函数接受原始 GLSL 代码字符串作为参数，最后一个函数用于创建 ncnn 的已存在的内置着色器。

```cpp
namespace ncnn {

// 在线 Spir-V 编译器
NCNN_EXPORT int compile_spirv_module(const char* comp_string, const Option& opt, std::vector<uint32_t>& spirv);
NCNN_EXPORT int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv);
NCNN_EXPORT int compile_spirv_module(int shader_type_index, const Option& opt, std::vector<uint32_t>& spirv);

} // namespace ncnn
```

## 直接编译ncnn扩展GLSL代码

您可以使用 ncnn GLSL 扩展编写着色器代码，使用 ncnn 函数编译为 Spir-V。编译后的产品是符合标准的 Spir-V 二进制文件，可以直接用于在 Vulkan API 中创建流水线对象

```cpp
static const char my_glsl_data[] = R"(
#version 450

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
 // 您可以控制Vulkan扩展行为
 // 当GPU支持16位存储的话
opt.use_fp16_storage = false;

std::vector<uint32_t> spirv;
ncnn::compile_spirv_module(my_glsl_data, sizeof(my_glsl_data) - 1, opt, spirv);

// 稍后再创建管道对象
// ncnn::Pipeline pipeline(vkdev);
// pipeline.set_local_size_xyz(64, 1, 1);
// pipeline.create(spirv.data(), spirv.size() * 4, specializations);
```

## ncnn内置着色器

ncnn内部的着色器索引在标头中公开，如果需要可以使用 `layer_shader_type.h`

```cpp
#include "layer_shader_type.h"

int shader_type_index = LayerShaderType::convert_ycbcr;

Option opt;

std::vector<uint32_t> spirv;
int retc = compile_spirv_module(shader_type_index, opt, spirv);
```

# 数据类型

## 存储类型(storage type)

在描述符绑定中声明缓冲区数据布局

```c
layout (binding = 0) buffer top_blob { sfpvec4 top_blob_data[]; };
```

|存储类型|fp32|fp16p|fp16s|
|---|---|---|---|
|sfp|float|uint|float16_t|
|sfpvec2|vec2|uint|f16vec2|
|sfpvec4|vec4|uvec2|f16vec4|

## 算术类型(arithmetic type)

在 GLSL 代码中声明局部变量

```c
void main()
{
    afpvec4 v = a * b;
}
```

|算术类型|fp32|fp16a|
|---|---|---|
|afp|float|float16_t|
|afpvec2|vec2|f16vec2|
|afpvec4|vec4|f16vec4|

## 本地类型(local type)

在共享本地内存中声明变量

```c
shared lfp tmp_a[8][4][2];
```

|local type|fp32|fp16p / fp16s only|fp16s+fp16a|fp16s+fp16u|
|---|---|---|---|---|
|lfp|float|float|float|float16_t|
|lfpvec4|vec4|uvec2|uint64_t|f16vec4|

# 缓冲区函数(buffer functions)

- 从 src[offset] 加载已经确定类型的值

```c
afp buffer_ld1(sfp src, int offset);
afpvec2 buffer_ld2(sfpvec2 src, int offset);
afpvec4 buffer_ld4(sfpvec4 src, int offset);
```

- 将已确定类型的值存储到 dst[偏移量]

```c
void buffer_st1(sfp dst, int offset, afp v);
void buffer_st2(sfpvec2 dst, int offset, afpvec2 v);
void buffer_st4(sfpvec4 dst, int offset, afpvec4 v);
```

- 从已确定类型 src[src_offset] 的值拷贝到 dst[dst_offset]

```c
void buffer_cp1(sfp dst, int dst_offset, sfp src, int src_offset);
void buffer_cp2(sfpvec2 dst, int dst_offset, sfpvec2 src, int src_offset);
void buffer_cp4(sfpvec4 dst, int dst_offset, sfpvec4 src, int src_offset);
```

- 从 src[src_offsets[0],src_offsets[1],...] 的值拷贝并打包到 dst[dst_offset]

```c
void buffer_cp1to4(sfpvec4 dst, int dst_offset, sfp src, ivec4 src_offsets);
```

- 从 src[src_offset] 的值拷贝并解包到 dst[dst_offsets[0],dst_offsets[1],...]

```c
void buffer_cp4to1(sfp dst, ivec4 dst_offsets, sfpvec4 src, int src_offset);
```

# 本地数据转换函数

- 存储缓冲区转换到本地内存

```c
lfp sfp2lfp(sfp v);
lfpvec4 sfp2lfpvec4(sfpvec4 v);
```

- 本地内存转换到局部变量

```c
afp lfp2afp(lfp v);
afpvec4 lfp2afpvec4(lfpvec4 v);
```

注意：本地内存的常见用法是先从全局内存中读取，存储在本地内存中，然后再从本地内存中读取局部变量以供后续使用。因此，此处仅提供存储类型到本地类型和本地类型到算术类型的转换函数。

# 杂项函数

- 更推荐使用专业化常量(specialization constants)，而不是推动常量(push constants)

```c
T psc(T x)
```

在 `专用常量` 和 `推送常量` 部分中声明相同的变量，然后在专用常量给定非零时 `psc(x)` 将成为编译时常量，否则将通过推送常量动态。这通常用于张量形状特化。我们通常可以解析所有形状信息，并使它们成为编译时常量，以实现让着色器得到更积极的优化。

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

# 平台宏定义

判断当前平台是否为 moltenvk，以启用对于某些特定于平台的解决方法

```c
#if NCNN_moltenvk
// 启用moltenvk的解决方法
#endif
```

ncnn 在新版本中添加了额外的宏定义，可能与现在的 glsl 代码冲突或引起混淆。为了实现  ncnn 的跨版本兼容性，可以根据  `ncnn_glsl_version` 宏的版本号在新旧代码之间进行切换 。

```c
#if ncnn_glsl_version >= 1
// 使用自版本 1 起引入的设备宏
#endif
```

ncnn 额外定义了大多数 vulcan 设备相关功能作为宏，我们可以用来区分不同的平台、设备扩展、功能和属性。

### 扩展宏定义

当设备支持某个扩展时，`ncnn_<extension_name>` 被定义为扩展版本

```c
void main()
{
#if ncnn_VK_KHR_16bit_storage
    // 支持 VK_KHR_16bit_storage 设备的代码
#endif

#if ncnn_VK_KHR_sampler_ycbcr_conversion >= 10
    // 支持 VK_KHR_sampler_ycbcr_conversion 且版本 >=10 的代码
#endif
}
```

### 设备特性和属性宏

ncnn 会查询设备特性和属性，然后将它们定义为宏。

宏名称为 `ncnn_<feature_name>` 或 `ncnn_<property_name>`

当设备支持 `shaderInt64` 时，`GL_EXT_shader_explicit_arithmetic_types_int64` 扩展会自动启用，无需显式代码指示。

```c
void main()
{
#if ncnn_robustBufferAccess
    // 支持 robustBufferAccess 特性的设备代码
#endif

#if ncnn_vendorID == 4318
    // 供应商特定代码，4318 是 nvidia 显卡
#endif

#if ncnn_subgroupSize == 32
    // 为 subgroup_size == 32 优化的代码路径
#endif

    // 使用宏定义
    uint size; // 来自先前例程的动态值
    if (size < ncnn_subgroupSize)
    {
#if ncnn_supportedOperations & 4
        // subgroup 支持算术运算
#endif

#if ncnn_subgroup_arithmetic
        // 检查 subgroup 算术运算的简写形式
#endif
    }
}
```

### 验证层宏定义

当启用 vulkan 验证层时，ncnn 会定义一些额外的便捷宏

* `ncnn_enable_validation_layer`
* `NCNN_LOGE`

目前，你必须将 `src/gpu.cpp` 开头的 `ENABLE_VALIDATION_LAYER` 定义修改为 `1` 才能启用这些宏。

`GL_EXT_debug_printf` 扩展会自动启用，无需在代码中显式指定。

```c
void main()
{
    int gx = int(gl_GlobalInvocationID.x);

#if ncnn_enable_validation_layer
    NCNN_LOGE("gx = %d\n", gx);
#endif
}
```

在运行时，`NCNN_LOGE` 将打印出 `gx` 的值

### 选项宏

仅当用户启用某些选项时才启用 GLSL 扩展

`GL_EXT_shader_16bit_storage` 扩展会在设备支持 16 位存储且用户开启了 `opt.use_fp16_storage` 选项时，自动启用，无需显式代码指示。

`GL_EXT_shader_explicit_arithmetic_types_float16` 扩展会在设备支持 16 位算术运算且用户开启了 `opt.use_fp16_arithmetic` 选项时，自动启用，无需显式代码指示。

```c
void main()
{
#if NCNN_fp16_storage
    // 用户启用 fp16 存储选项，且设备支持 fp16 存储
#endif

#if NCNN_fp16_arithmetic
    // 用户启用 fp16 算术选项，且设备支持 fp16 算术运算
#endif
}
```

|宏定义|option中所定义的变量|
|---|---|
|NCNN_fp16_packed|opt.use_fp16_packed|
|NCNN_fp16_storage|opt.use_fp16_storage|
|NCNN_fp16_arithmetic|opt.use_fp16_arithmetic|
|NCNN_int8_packed|opt.use_int8_packed|
|NCNN_int8_storage|opt.use_int8_storage|
|NCNN_int8_arithmetic|opt.use_int8_arithmetic|
|NCNN_shader_local_memory|opt.use_shader_local_memory|
