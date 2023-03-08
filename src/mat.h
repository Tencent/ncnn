// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_MAT_H
#define NCNN_MAT_H

#include <stdlib.h>
#include <string.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif
#if __mips_msa
#include <msa.h>
#endif
#if __loongarch_sx
#include <lsxintrin.h>
#endif
#if __riscv_vector
#include <riscv_vector.h>
#include "cpu.h" // cpu_riscv_vlenb()
#endif

#include "allocator.h"
#include "option.h"
#include "platform.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

#if NCNN_PIXEL
#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
#include <android/bitmap.h>
#include <jni.h>
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API
#endif // NCNN_PIXEL

namespace ncnn {

#if NCNN_VULKAN
class VkMat;
class VkImageMat;
#endif // NCNN_VULKAN

// the three dimension matrix
class NCNN_EXPORT Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // image
    Mat(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // cube
    Mat(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // packed vec
    Mat(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed image
    Mat(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed dim
    Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed cube
    Mat(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external image
    Mat(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external dim
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external cube
    Mat(int w, int h, int d, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external packed vec
    Mat(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed image
    Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed dim
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed cube
    Mat(int w, int h, int d, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // release
    ~Mat();
    // assign
    Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    void fill(int v);
#if __ARM_NEON
    void fill(float32x4_t _v);
    void fill(uint16x4_t _v);
    void fill(int32x4_t _v);
    void fill(int32x4_t _v0, int32x4_t _v1);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    void fill(float16x4_t _v);
    void fill(float16x8_t _v);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __ARM_NEON
#if __SSE2__
#if __AVX__
#if __AVX512F__
    void fill(__m512 _v);
#endif // __AVX512F__
    void fill(__m256 _v, int i = 0);
#endif // __AVX__
    void fill(__m128 _v);
    void fill(__m128i _v);
#endif // __SSE2__
#if __mips_msa
    void fill(v4f32 _v);
#endif // __mips_msa
#if __loongarch_sx
    void fill(__m128 _v);
#endif //__loongarch_sx
#if __riscv_vector
    void fill(vfloat32m1_t _v);
    void fill(vuint16m1_t _v);
    void fill(vint8m1_t _v);
#if __riscv_zfh
    void fill(vfloat16m1_t _v);
#endif // __riscv_zfh
#endif // __riscv_vector
    template<typename T>
    void fill(T v);
    // deep copy
    Mat clone(Allocator* allocator = 0) const;
    // deep copy from other mat, inplace
    void clone_from(const ncnn::Mat& mat, Allocator* allocator = 0);
    // reshape vec
    Mat reshape(int w, Allocator* allocator = 0) const;
    // reshape image
    Mat reshape(int w, int h, Allocator* allocator = 0) const;
    // reshape dim
    Mat reshape(int w, int h, int c, Allocator* allocator = 0) const;
    // reshape cube
    Mat reshape(int w, int h, int d, int c, Allocator* allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate cube
    void create(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed cube
    void create(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate like
    void create_like(const Mat& m, Allocator* allocator = 0);
#if NCNN_VULKAN
    // allocate like
    void create_like(const VkMat& m, Allocator* allocator = 0);
    // allocate like
    void create_like(const VkImageMat& im, Allocator* allocator = 0);
#endif // NCNN_VULKAN
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // bits per element
    int elembits() const;

    // shape only
    Mat shape() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    Mat depth(int z);
    const Mat depth(int z) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T>
    T* row(int y);
    template<typename T>
    const T* row(int y) const;

    // range reference
    Mat channel_range(int c, int channels);
    const Mat channel_range(int c, int channels) const;
    Mat depth_range(int z, int depths);
    const Mat depth_range(int z, int depths) const;
    Mat row_range(int y, int rows);
    const Mat row_range(int y, int rows) const;
    Mat range(int x, int n);
    const Mat range(int x, int n) const;

    // access raw data
    template<typename T>
    operator T*();
    template<typename T>
    operator const T*() const;

    // convenient access float vec element
    float& operator[](size_t i);
    const float& operator[](size_t i) const;

#if NCNN_PIXEL
    enum PixelType
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB = 1,
        PIXEL_BGR = 2,
        PIXEL_GRAY = 3,
        PIXEL_RGBA = 4,
        PIXEL_BGRA = 5,

        PIXEL_RGB2BGR = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2RGBA = PIXEL_RGB | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2BGRA = PIXEL_RGB | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2RGBA = PIXEL_BGR | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2BGRA = PIXEL_BGR | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2RGBA = PIXEL_GRAY | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGRA = PIXEL_GRAY | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGRA = PIXEL_RGBA | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGRA2RGB = PIXEL_BGRA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2BGR = PIXEL_BGRA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2GRAY = PIXEL_BGRA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2RGBA = PIXEL_BGRA | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator = 0);
    // convenient construct from pixel data with stride(bytes-per-row) parameter
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator = 0);
    // convenient construct from pixel data and resize to specific size
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data and resize to specific size with stride(bytes-per-row) parameter
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data roi
    static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);
    // convenient construct from pixel data roi with stride(bytes-per-row) parameter
    static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);
    // convenient construct from pixel data roi and resize to specific size
    static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data roi and resize to specific size with stride(bytes-per-row) parameter
    static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type) const;
    // convenient export to pixel data with stride(bytes-per-row) parameter
    void to_pixels(unsigned char* pixels, int type, int stride) const;
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
    // convenient export to pixel data and resize to specific size with stride(bytes-per-row) parameter
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const;

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
    // convenient construct from android Bitmap
    static Mat from_android_bitmap(JNIEnv* env, jobject bitmap, int type_to, Allocator* allocator = 0);
    // convenient construct from android Bitmap and resize to specific size
    static Mat from_android_bitmap_resize(JNIEnv* env, jobject bitmap, int type_to, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from android Bitmap roi
    static Mat from_android_bitmap_roi(JNIEnv* env, jobject bitmap, int type_to, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);
    // convenient construct from android Bitmap roi and resize to specific size
    static Mat from_android_bitmap_roi_resize(JNIEnv* env, jobject bitmap, int type_to, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);
    // convenient export to android Bitmap and resize to the android Bitmap size
    void to_android_bitmap(JNIEnv* env, jobject bitmap, int type_from) const;
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API
#endif // NCNN_PIXEL

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precision floating point data
    static Mat from_float16(const unsigned short* data, int size);

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-d-h-w-1  c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-d-h-w-8  c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    Allocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int d;
    int c;

    size_t cstep;
};

#if NCNN_VULKAN

// the three dimension matrix, vulkan version
class NCNN_EXPORT VkMat
{
public:
    // empty
    VkMat();
    // vec
    VkMat(int w, size_t elemsize, VkAllocator* allocator);
    // image
    VkMat(int w, int h, size_t elemsize, VkAllocator* allocator);
    // dim
    VkMat(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // cube
    VkMat(int w, int h, int d, int c, size_t elemsize, VkAllocator* allocator);
    // packed vec
    VkMat(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed image
    VkMat(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed dim
    VkMat(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed cube
    VkMat(int w, int h, int d, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // copy
    VkMat(const VkMat& m);
    // external vec
    VkMat(int w, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external image
    VkMat(int w, int h, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external dim
    VkMat(int w, int h, int c, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external cube
    VkMat(int w, int h, int d, int c, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external packed vec
    VkMat(int w, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed image
    VkMat(int w, int h, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed dim
    VkMat(int w, int h, int c, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed cube
    VkMat(int w, int h, int d, int c, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // release
    ~VkMat();
    // assign
    VkMat& operator=(const VkMat& m);
    // allocate vec
    void create(int w, size_t elemsize, VkAllocator* allocator);
    // allocate image
    void create(int w, int h, size_t elemsize, VkAllocator* allocator);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // allocate cube
    void create(int w, int h, int d, int c, size_t elemsize, VkAllocator* allocator);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed cube
    void create(int w, int h, int d, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate like
    void create_like(const Mat& m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkMat& m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkImageMat& im, VkAllocator* allocator);

    // mapped
    Mat mapped() const;
    void* mapped_ptr() const;

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // bits per element
    int elembits() const;

    // shape only
    Mat shape() const;

    // low-level reference
    VkBuffer buffer() const;
    size_t buffer_offset() const;
    size_t buffer_capacity() const;

    // device buffer
    VkBufferMemory* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-d-h-w-1  c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-d-h-w-8  c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    VkAllocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int d;
    int c;

    size_t cstep;
};

class NCNN_EXPORT VkImageMat
{
public:
    // empty
    VkImageMat();
    // vec
    VkImageMat(int w, size_t elemsize, VkAllocator* allocator);
    // image
    VkImageMat(int w, int h, size_t elemsize, VkAllocator* allocator);
    // dim
    VkImageMat(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // cube
    VkImageMat(int w, int h, int d, int c, size_t elemsize, VkAllocator* allocator);
    // packed vec
    VkImageMat(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed image
    VkImageMat(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed dim
    VkImageMat(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed cube
    VkImageMat(int w, int h, int d, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // copy
    VkImageMat(const VkImageMat& m);
    // external vec
    VkImageMat(int w, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external image
    VkImageMat(int w, int h, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external dim
    VkImageMat(int w, int h, int c, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external cube
    VkImageMat(int w, int h, int d, int c, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external packed vec
    VkImageMat(int w, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed image
    VkImageMat(int w, int h, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed dim
    VkImageMat(int w, int h, int c, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed cube
    VkImageMat(int w, int h, int d, int c, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // release
    ~VkImageMat();
    // assign
    VkImageMat& operator=(const VkImageMat& m);
    // allocate vec
    void create(int w, size_t elemsize, VkAllocator* allocator);
    // allocate image
    void create(int w, int h, size_t elemsize, VkAllocator* allocator);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // allocate cube
    void create(int w, int h, int d, int c, size_t elemsize, VkAllocator* allocator);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed cube
    void create(int w, int h, int d, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate like
    void create_like(const Mat& m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkMat& m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkImageMat& im, VkAllocator* allocator);

    // mapped
    Mat mapped() const;
    void* mapped_ptr() const;

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // bits per element
    int elembits() const;

    // shape only
    Mat shape() const;

    // low-level reference
    VkImage image() const;
    VkImageView imageview() const;

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
    // convenient construct from android hardware buffer
    static VkImageMat from_android_hardware_buffer(VkAndroidHardwareBufferImageAllocator* allocator);
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

    // device image
    VkImageMemory* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-d-h-w-1  c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-d-h-w-8  c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    VkAllocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int d;
    int c;
};

// type for vulkan specialization constant and push constant
union vk_specialization_type
{
    int i;
    float f;
    uint32_t u32;
};
union vk_constant_type
{
    int i;
    float f;
};
#endif // NCNN_VULKAN

// misc function
#if NCNN_PIXEL
// convert yuv420sp(nv21) to rgb, the fast approximate version
NCNN_EXPORT void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// convert yuv420sp(nv12) to rgb, the fast approximate version
NCNN_EXPORT void yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// convert yuv420sp(nv21) to rgb with half resize, the faster approximate version
NCNN_EXPORT void yuv420sp2rgb_half(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// image pixel bilinear resize
NCNN_EXPORT void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
NCNN_EXPORT void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
NCNN_EXPORT void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
NCNN_EXPORT void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
// image pixel bilinear resize with stride(bytes-per-row) parameter
NCNN_EXPORT void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
NCNN_EXPORT void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
NCNN_EXPORT void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
NCNN_EXPORT void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
// image pixel bilinear resize, convenient wrapper for yuv420sp(nv21/nv12)
NCNN_EXPORT void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
#endif // NCNN_PIXEL
#if NCNN_PIXEL_ROTATE
// type is the from type, 6 means rotating from 6 to 1
//
//     1        2       3      4         5            6           7          8
//
//   888888  888888      88  88      8888888888  88                  88  8888888888
//   88          88      88  88      88  88      88  88          88  88      88  88
//   8888      8888    8888  8888    88          8888888888  8888888888          88
//   88          88      88  88
//   88          88  888888  888888
//
// ref http://sylvana.net/jpegcrop/exif_orientation.html
// image pixel kanna rotate
NCNN_EXPORT void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
NCNN_EXPORT void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
NCNN_EXPORT void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
NCNN_EXPORT void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
// image pixel kanna rotate with stride(bytes-per-row) parameter
NCNN_EXPORT void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
NCNN_EXPORT void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
NCNN_EXPORT void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
NCNN_EXPORT void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
// image pixel kanna rotate, convenient wrapper for yuv420sp(nv21/nv12)
NCNN_EXPORT void kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
#endif // NCNN_PIXEL_ROTATE
#if NCNN_PIXEL_AFFINE
// resolve affine transform matrix from rotation angle, scale factor and x y offset
NCNN_EXPORT void get_rotation_matrix(float angle, float scale, float dx, float dy, float* tm);
// resolve affine transform matrix from two set of points, num_point must be >= 2
NCNN_EXPORT void get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm);
// resolve the inversion affine transform matrix
NCNN_EXPORT void invert_affine_transform(const float* tm, float* tm_inv);
// image pixel bilinear warpaffine inverse transform, set -233 for transparent border color, the color RGBA is little-endian encoded
NCNN_EXPORT void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
NCNN_EXPORT void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
NCNN_EXPORT void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
NCNN_EXPORT void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
// image pixel bilinear warpaffine inverse transform with stride(bytes-per-row) parameter, set -233 for transparent border color, the color RGBA is little-endian encoded
NCNN_EXPORT void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
NCNN_EXPORT void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
NCNN_EXPORT void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
NCNN_EXPORT void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
// image pixel bilinear warpaffine, convenient wrapper for yuv420sp(nv21/nv12), set -233 for transparent border color, the color YUV_ is little-endian encoded
NCNN_EXPORT void warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
#endif // NCNN_PIXEL_AFFINE
#if NCNN_PIXEL_DRAWING
// draw rectangle, set thickness -1 for filled rectangle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw rectangle with stride(bytes-per-row) parameter, set thickness -1 for filled rectangle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_rectangle_c1(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c2(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c3(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
NCNN_EXPORT void draw_rectangle_c4(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw rectangle, convenient wrapper for yuv420sp(nv21/nv12), set thickness -1 for filled rectangle, the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_rectangle_yuv420sp(unsigned char* yuv420sp, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw circle, set thickness -1 for filled circle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
// draw circle with stride(bytes-per-row) parameter, set thickness -1 for filled circle, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_circle_c1(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c2(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c3(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
NCNN_EXPORT void draw_circle_c4(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
// draw circle, convenient wrapper for yuv420sp(nv21/nv12), set thickness -1 for filled circle, the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_circle_yuv420sp(unsigned char* yuv420sp, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
// draw line, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// draw line with stride(bytes-per-row) parameter, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_line_c1(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c2(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c3(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
NCNN_EXPORT void draw_line_c4(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// draw line, convenient wrapper for yuv420sp(nv21/nv12), the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_line_yuv420sp(unsigned char* yuv420sp, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// resolve text bounding box size
NCNN_EXPORT void get_text_drawing_size(const char* text, int fontpixelsize, int* w, int* h);
// draw ascii printables and newline, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// draw ascii printables and newline with stride(bytes-per-row) parameter, the color RGBA is little-endian encoded
NCNN_EXPORT void draw_text_c1(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c2(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c3(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
NCNN_EXPORT void draw_text_c4(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// draw ascii printables and newline, convenient wrapper for yuv420sp(nv21/nv12), the color YUV_ is little-endian encoded
NCNN_EXPORT void draw_text_yuv420sp(unsigned char* yuv420sp, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
#endif // NCNN_PIXEL_DRAWING

// type conversion
// convert float to half precision floating point
NCNN_EXPORT unsigned short float32_to_float16(float value);
// convert half precision floating point to float
NCNN_EXPORT float float16_to_float32(unsigned short value);
// convert float to brain half
NCNN_EXPORT NCNN_FORCEINLINE unsigned short float32_to_bfloat16(float value)
{
    // 16 : 16
    union
    {
        unsigned int u;
        float f;
    } tmp;
    tmp.f = value;
    return tmp.u >> 16;
}
// convert brain half to float
NCNN_EXPORT NCNN_FORCEINLINE float bfloat16_to_float32(unsigned short value)
{
    // 16 : 16
    union
    {
        unsigned int u;
        float f;
    } tmp;
    tmp.u = value << 16;
    return tmp.f;
}

// mat process
enum BorderType
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_TRANSPARENT = -233,
};
NCNN_EXPORT void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt = Option());
NCNN_EXPORT void copy_make_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, int type, float v, const Option& opt = Option());
NCNN_EXPORT void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt = Option());
NCNN_EXPORT void copy_cut_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, const Option& opt = Option());
NCNN_EXPORT void resize_nearest(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
NCNN_EXPORT void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
NCNN_EXPORT void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
NCNN_EXPORT void convert_packing(const Mat& src, Mat& dst, int elempack, const Option& opt = Option());
NCNN_EXPORT void flatten(const Mat& src, Mat& dst, const Option& opt = Option());
NCNN_EXPORT void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt = Option());
NCNN_EXPORT void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
NCNN_EXPORT void cast_int8_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
NCNN_EXPORT void cast_float32_to_bfloat16(const Mat& src, Mat& dst, const Option& opt = Option());
NCNN_EXPORT void cast_bfloat16_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
NCNN_EXPORT void quantize_to_int8(const Mat& src, Mat& dst, const Mat& scale_data, const Option& opt = Option());
NCNN_EXPORT void dequantize_from_int32(const Mat& src, Mat& dst, const Mat& scale_data, const Mat& bias_data, const Option& opt = Option());
NCNN_EXPORT void requantize_from_int32_to_int8(const Mat& src, Mat& dst, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt = Option());

NCNN_FORCEINLINE Mat::Mat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
}

NCNN_FORCEINLINE Mat::Mat(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(const Mat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), d(m.d), c(m.c), cstep(m.cstep)
{
    addref();
}

NCNN_FORCEINLINE Mat::Mat(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::Mat(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::~Mat()
{
    release();
}

NCNN_FORCEINLINE void Mat::fill(float _v)
{
    int size = (int)total();
    float* ptr = (float*)data;

    int i = 0;
#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
    for (; i + 3 < size; i += 4)
    {
        vst1q_f32(ptr, _c);
        ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *ptr++ = _v;
    }
}

NCNN_FORCEINLINE void Mat::fill(int _v)
{
    int size = (int)total();
    int* ptr = (int*)data;

    int i = 0;
#if __ARM_NEON
    int32x4_t _c = vdupq_n_s32(_v);
    for (; i + 3 < size; i += 4)
    {
        vst1q_s32(ptr, _c);
        ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *ptr++ = _v;
    }
}

#if __ARM_NEON
NCNN_FORCEINLINE void Mat::fill(float32x4_t _v)
{
    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        vst1q_f32(ptr, _v);
        ptr += 4;
    }
}

NCNN_FORCEINLINE void Mat::fill(uint16x4_t _v)
{
    int size = (int)total();
    unsigned short* ptr = (unsigned short*)data;
    for (int i = 0; i < size; i++)
    {
        vst1_u16(ptr, _v);
        ptr += 4;
    }
}

NCNN_FORCEINLINE void Mat::fill(int32x4_t _v)
{
    int size = (int)total();
    int* ptr = (int*)data;
    for (int i = 0; i < size; i++)
    {
        vst1q_s32(ptr, _v);
        ptr += 4;
    }
}

NCNN_FORCEINLINE void Mat::fill(int32x4_t _v0, int32x4_t _v1)
{
    int size = (int)total();
    int* ptr = (int*)data;
    for (int i = 0; i < size; i++)
    {
        vst1q_s32(ptr, _v0);
        vst1q_s32(ptr + 4, _v1);
        ptr += 8;
    }
}
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
NCNN_FORCEINLINE void Mat::fill(float16x4_t _v)
{
    int size = (int)total();
    __fp16* ptr = (__fp16*)data;
    for (int i = 0; i < size; i++)
    {
        vst1_f16(ptr, _v);
        ptr += 4;
    }
}

NCNN_FORCEINLINE void Mat::fill(float16x8_t _v)
{
    int size = (int)total();
    __fp16* ptr = (__fp16*)data;
    for (int i = 0; i < size; i++)
    {
        vst1q_f16(ptr, _v);
        ptr += 8;
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __ARM_NEON

#if __SSE2__
#if __AVX__
#if __AVX512F__
NCNN_FORCEINLINE void Mat::fill(__m512 _v)
{
    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        _mm512_storeu_ps(ptr, _v);
        ptr += 16;
    }
}
#endif // __AVX512F__
NCNN_FORCEINLINE void Mat::fill(__m256 _v, int _i)
{
    // old gcc cannot overload __m128 and __m256 type
    // add a dummy int parameter for different mangled function symbol
    (void)_i;
    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        _mm256_storeu_ps(ptr, _v);
        ptr += 8;
    }
}
#endif // __AVX__
NCNN_FORCEINLINE void Mat::fill(__m128 _v)
{
    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        _mm_storeu_ps(ptr, _v);
        ptr += 4;
    }
}
NCNN_FORCEINLINE void Mat::fill(__m128i _v)
{
    int size = (int)total();
    unsigned short* ptr = (unsigned short*)data;
    for (int i = 0; i < size; i++)
    {
        _mm_store_si128((__m128i*)ptr, _v);
        ptr += 8;
    }
}
#endif // __SSE2__

#if __mips_msa
NCNN_FORCEINLINE void Mat::fill(v4f32 _v)
{
    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        __msa_st_w((v4i32)_v, ptr, 0);
        ptr += 4;
    }
}
#endif // __mips_msa

#if __loongarch_sx
NCNN_FORCEINLINE void Mat::fill(__m128 _v)
{
    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        __lsx_vst(_v, ptr, 0);
        ptr += 4;
    }
}
#endif // __loongarch_sx
#if __riscv_vector
NCNN_FORCEINLINE void Mat::fill(vfloat32m1_t _v)
{
    const int packn = cpu_riscv_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    int size = (int)total();
    float* ptr = (float*)data;
    for (int i = 0; i < size; i++)
    {
        vse32_v_f32m1(ptr, _v, vl);
        ptr += packn;
    }
}

NCNN_FORCEINLINE void Mat::fill(vuint16m1_t _v)
{
    const int packn = cpu_riscv_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int size = (int)total();
    unsigned short* ptr = (unsigned short*)data;
    for (int i = 0; i < size; i++)
    {
        vse16_v_u16m1(ptr, _v, vl);
        ptr += packn;
    }
}

NCNN_FORCEINLINE void Mat::fill(vint8m1_t _v)
{
    const int packn = cpu_riscv_vlenb() / 1;
    const size_t vl = vsetvl_e8m1(packn);

    int size = (int)total();
    signed char* ptr = (signed char*)data;
    for (int i = 0; i < size; i++)
    {
        vse8_v_i8m1(ptr, _v, vl);
        ptr += packn;
    }
}
#if __riscv_zfh
NCNN_FORCEINLINE void Mat::fill(vfloat16m1_t _v)
{
    const int packn = cpu_riscv_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int size = (int)total();
    __fp16* ptr = (__fp16*)data;
    for (int i = 0; i < size; i++)
    {
        vse16_v_f16m1(ptr, _v, vl);
        ptr += packn;
    }
}
#endif // __riscv_zfh
#endif // __riscv_vector

template<typename T>
NCNN_FORCEINLINE void Mat::fill(T _v)
{
    int size = (int)total();
    T* ptr = (T*)data;
    for (int i = 0; i < size; i++)
    {
        ptr[i] = _v;
    }
}

NCNN_FORCEINLINE Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

NCNN_FORCEINLINE void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

NCNN_FORCEINLINE void Mat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

NCNN_FORCEINLINE bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

NCNN_FORCEINLINE size_t Mat::total() const
{
    return cstep * c;
}

NCNN_FORCEINLINE int Mat::elembits() const
{
    return elempack ? static_cast<int>(elemsize * 8) / elempack : 0;
}

NCNN_FORCEINLINE Mat Mat::shape() const
{
    if (dims == 1)
        return Mat(w * elempack, (void*)0);
    if (dims == 2)
        return Mat(w, h * elempack, (void*)0);
    if (dims == 3)
        return Mat(w, h, c * elempack, (void*)0);
    if (dims == 4)
        return Mat(w, h, d, c * elempack, (void*)0);

    return Mat();
}

NCNN_FORCEINLINE Mat Mat::channel(int _c)
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE const Mat Mat::channel(int _c) const
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE Mat Mat::depth(int z)
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE const Mat Mat::depth(int z) const
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

NCNN_FORCEINLINE const float* Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template<typename T>
NCNN_FORCEINLINE T* Mat::row(int y)
{
    return (T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template<typename T>
NCNN_FORCEINLINE const T* Mat::row(int y) const
{
    return (const T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

NCNN_FORCEINLINE Mat Mat::channel_range(int _c, int channels)
{
    Mat m(w, h, d, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims;
    return m;
}

NCNN_FORCEINLINE const Mat Mat::channel_range(int _c, int channels) const
{
    Mat m(w, h, d, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims;
    return m;
}

NCNN_FORCEINLINE Mat Mat::depth_range(int z, int depths)
{
    Mat m(w, h, depths, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
    m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE const Mat Mat::depth_range(int z, int depths) const
{
    Mat m(w, h, depths, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
    m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char*)data + (size_t)w * y * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char*)data + (size_t)w * y * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE Mat Mat::range(int x, int n)
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE const Mat Mat::range(int x, int n) const
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

template<typename T>
NCNN_FORCEINLINE Mat::operator T*()
{
    return (T*)data;
}

template<typename T>
NCNN_FORCEINLINE Mat::operator const T*() const
{
    return (const T*)data;
}

NCNN_FORCEINLINE float& Mat::operator[](size_t i)
{
    return ((float*)data)[i];
}

NCNN_FORCEINLINE const float& Mat::operator[](size_t i) const
{
    return ((const float*)data)[i];
}

#if NCNN_VULKAN

NCNN_FORCEINLINE VkMat::VkMat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _d, int _c, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkMat::VkMat(const VkMat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), d(m.d), c(m.c)
{
    addref();

    cstep = m.cstep;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = w * h;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _c, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _d, int _c, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize(w * h * d * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = w * h;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _c, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE VkMat::VkMat(int _w, int _h, int _d, int _c, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize(w * h * d * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE VkMat::~VkMat()
{
    release();
}

NCNN_FORCEINLINE VkMat& VkMat::operator=(const VkMat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

NCNN_FORCEINLINE Mat VkMat::mapped() const
{
    if (!allocator->mappable)
        return Mat();

    if (dims == 1)
        return Mat(w, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 2)
        return Mat(w, h, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 3)
        return Mat(w, h, c, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 4)
        return Mat(w, h, d, c, mapped_ptr(), elemsize, elempack, 0);

    return Mat();
}

NCNN_FORCEINLINE void* VkMat::mapped_ptr() const
{
    if (!allocator->mappable)
        return 0;

    return (unsigned char*)data->mapped_ptr + data->offset;
}

NCNN_FORCEINLINE void VkMat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

NCNN_FORCEINLINE void VkMat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

NCNN_FORCEINLINE bool VkMat::empty() const
{
    return data == 0 || total() == 0;
}

NCNN_FORCEINLINE size_t VkMat::total() const
{
    return cstep * c;
}

NCNN_FORCEINLINE int VkMat::elembits() const
{
    return elempack ? static_cast<int>(elemsize) * 8 / elempack : 0;
}

NCNN_FORCEINLINE Mat VkMat::shape() const
{
    if (dims == 1)
        return Mat(w * elempack, (void*)0);
    if (dims == 2)
        return Mat(w, h * elempack, (void*)0);
    if (dims == 3)
        return Mat(w, h, c * elempack, (void*)0);
    if (dims == 4)
        return Mat(w, h, d, c * elempack, (void*)0);

    return Mat();
}

NCNN_FORCEINLINE VkBuffer VkMat::buffer() const
{
    return data->buffer;
}

NCNN_FORCEINLINE size_t VkMat::buffer_offset() const
{
    return data->offset;
}

NCNN_FORCEINLINE size_t VkMat::buffer_capacity() const
{
    return data->capacity;
}

NCNN_FORCEINLINE VkImageMat::VkImageMat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _h, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _d, int _c, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _h, _d, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0)
{
    create(_w, _h, _d, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(const VkImageMat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), d(m.d), c(m.c)
{
    addref();
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _c, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _d, int _c, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _c, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
}

NCNN_FORCEINLINE VkImageMat::VkImageMat(int _w, int _h, int _d, int _c, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
}

NCNN_FORCEINLINE VkImageMat::~VkImageMat()
{
    release();
}

NCNN_FORCEINLINE VkImageMat& VkImageMat::operator=(const VkImageMat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    return *this;
}

NCNN_FORCEINLINE Mat VkImageMat::mapped() const
{
    if (!allocator->mappable || !data->mapped_ptr)
        return Mat();

    if (dims == 1)
        return Mat(w, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 2)
        return Mat(w, h, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 3)
        return Mat(w, h, c, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 4)
        return Mat(w, h, d, c, mapped_ptr(), elemsize, elempack, 0);

    return Mat();
}

NCNN_FORCEINLINE void* VkImageMat::mapped_ptr() const
{
    if (!allocator->mappable || !data->mapped_ptr)
        return 0;

    return (unsigned char*)data->mapped_ptr + data->bind_offset;
}

NCNN_FORCEINLINE void VkImageMat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

NCNN_FORCEINLINE void VkImageMat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    refcount = 0;
}

NCNN_FORCEINLINE bool VkImageMat::empty() const
{
    return data == 0 || total() == 0;
}

NCNN_FORCEINLINE size_t VkImageMat::total() const
{
    return w * h * d * c;
}

NCNN_FORCEINLINE int VkImageMat::elembits() const
{
    return elempack ? static_cast<int>(elemsize) * 8 / elempack : 0;
}

NCNN_FORCEINLINE Mat VkImageMat::shape() const
{
    if (dims == 1)
        return Mat(w * elempack, (void*)0);
    if (dims == 2)
        return Mat(w, h * elempack, (void*)0);
    if (dims == 3)
        return Mat(w, h, c * elempack, (void*)0);
    if (dims == 4)
        return Mat(w, h, d, c * elempack, (void*)0);

    return Mat();
}

NCNN_FORCEINLINE VkImage VkImageMat::image() const
{
    return data->image;
}

NCNN_FORCEINLINE VkImageView VkImageMat::imageview() const
{
    return data->imageview;
}

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_MAT_H
