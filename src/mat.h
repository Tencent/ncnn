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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif
#include "platform.h"
#include "allocator.h"
#include "option.h"
#include "gpu.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

#if NCNN_PIXEL
#if __ANDROID_API__ >= 9
#include <jni.h>
#include <android/bitmap.h>
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PIXEL

namespace ncnn {

#if NCNN_VULKAN
class VkMat;
#endif // NCNN_VULKAN

// the three dimension matrix
class Mat
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
    // packed vec
    Mat(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed image
    Mat(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed dim
    Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external image
    Mat(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external dim
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external packed vec
    Mat(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed image
    Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed dim
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // release
    ~Mat();
    // assign
    Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    void fill(int v);
#if __ARM_NEON
    void fill(float32x4_t _v);
#endif // __ARM_NEON
    template <typename T> void fill(T v);
    // deep copy
    Mat clone(Allocator* allocator = 0) const;
    // reshape vec
    Mat reshape(int w, Allocator* allocator = 0) const;
    // reshape image
    Mat reshape(int w, int h, Allocator* allocator = 0) const;
    // reshape dim
    Mat reshape(int w, int h, int c, Allocator* allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate like
    void create_like(const Mat& m, Allocator* allocator = 0);
#if NCNN_VULKAN
    // allocate like
    void create_like(const VkMat& m, Allocator* allocator = 0);
#endif // NCNN_VULKAN
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T> T* row(int y);
    template<typename T> const T* row(int y) const;

    // range reference
    Mat channel_range(int c, int channels);
    const Mat channel_range(int c, int channels) const;
    Mat row_range(int y, int rows);
    const Mat row_range(int y, int rows) const;
    Mat range(int x, int n);
    const Mat range(int x, int n) const;

    // access raw data
    template<typename T> operator T*();
    template<typename T> operator const T*() const;

    // convenient access float vec element
    float& operator[](size_t i);
    const float& operator[](size_t i) const;

#if NCNN_PIXEL
    enum
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB       = 1,
        PIXEL_BGR       = 2,
        PIXEL_GRAY      = 3,
        PIXEL_RGBA      = 4,

        PIXEL_RGB2BGR   = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY  = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2RGBA  = PIXEL_RGB | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB   = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY  = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2RGBA  = PIXEL_BGR | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB  = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR  = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2RGBA = PIXEL_GRAY | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB  = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR  = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator = 0);
    // convenient construct from pixel data with stride(bytes-per-row) parameter
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator = 0);
    // convenient construct from pixel data and resize to specific size
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data and resize to specific size with stride(bytes-per-row) parameter
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator = 0);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type) const;
    // convenient export to pixel data with stride(bytes-per-row) parameter
    void to_pixels(unsigned char* pixels, int type, int stride) const;
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
    // convenient export to pixel data and resize to specific size with stride(bytes-per-row) parameter
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const;

#if __ANDROID_API__ >= 9
    // convenient construct from android Bitmap
    static Mat from_android_bitmap(JNIEnv* env, jobject bitmap, int type_to, Allocator* allocator = 0);
    // convenient construct from android Bitmap and resize to specific size
    static Mat from_android_bitmap_resize(JNIEnv* env, jobject bitmap, int type_to, int target_width, int target_height, Allocator* allocator = 0);
    // convenient export to android Bitmap and resize to the android Bitmap size
    void to_android_bitmap(JNIEnv* env, jobject bitmap, int type_from) const;
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PIXEL

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precisoin floating point data
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
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    Allocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};

#if NCNN_VULKAN

// the three dimension matrix, vulkan version
class VkMat
{
public:
    // empty
    VkMat();
    // vec
    VkMat(int w, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // image
    VkMat(int w, int h, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // dim
    VkMat(int w, int h, int c, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // packed vec
    VkMat(int w, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // packed image
    VkMat(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // packed dim
    VkMat(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // copy
    VkMat(const VkMat& m);
    // external vec
    VkMat(int w, VkBufferMemory* data, size_t offset, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external image
    VkMat(int w, int h, VkBufferMemory* data, size_t offset, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external dim
    VkMat(int w, int h, int c, VkBufferMemory* data, size_t offset, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external packed vec
    VkMat(int w, VkBufferMemory* data, size_t offset, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external packed image
    VkMat(int w, int h, VkBufferMemory* data, size_t offset, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // external packed dim
    VkMat(int w, int h, int c, VkBufferMemory* data, size_t offset, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // release
    ~VkMat();
    // assign
    VkMat& operator=(const VkMat& m);
    // allocate vec
    void create(int w, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate image
    void create(int w, int h, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate like
    void create_like(const Mat& m, VkAllocator* allocator, VkAllocator* staging_allocator);
    // allocate like
    void create_like(const VkMat& m, VkAllocator* allocator, VkAllocator* staging_allocator);

    // staging buffer
    void prepare_staging_buffer();
    void discard_staging_buffer();

    // copy
    void upload(const Mat& m);
    void download(Mat& m) const;

    // mapped
    Mat mapped() const;
    void* mapped_ptr() const;

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    VkMat channel(int c);
    const VkMat channel(int c) const;

    // range reference
    VkMat channel_range(int c, int channels);
    const VkMat channel_range(int c, int channels) const;
    VkMat row_range(int y, int rows);
    const VkMat row_range(int y, int rows) const;
    VkMat range(int x, int n);
    const VkMat range(int x, int n) const;

    // low-level reference
    VkBuffer buffer() const;
    size_t buffer_offset() const;
    VkBuffer staging_buffer() const;
    size_t staging_buffer_offset() const;

    // device buffer
    VkBufferMemory* data;
    // subrange offset
    size_t offset;

    // staging buffer
    VkBufferMemory* staging_data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;
    int* staging_refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    VkAllocator* allocator;
    VkAllocator* staging_allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};

class VkImageMat
{
public:
    // empty
    VkImageMat();
    // image
    VkImageMat(int width, int height, VkFormat format, VkImageAllocator* allocator);
    // copy
    VkImageMat(const VkImageMat& m);
    // external image
    VkImageMat(int width, int height, VkImageMemory* data, VkFormat format, VkImageAllocator* allocator);
    // release
    ~VkImageMat();
    // assign
    VkImageMat& operator=(const VkImageMat& m);
    // allocate image
    void create(int width, int height, VkFormat format, VkImageAllocator* allocator);

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // low-level reference
    VkImage image() const;
    VkImageView imageview() const;

#if __ANDROID_API__ >= 26
    // convenient construct from android hardware buffer
    static VkImageMat from_android_hardware_buffer(AHardwareBuffer* hb, VkAndroidHardwareBufferImageAllocator* allocator);
#endif // __ANDROID_API__ >= 26

    // device image
    VkImageMemory* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // the allocator
    VkImageAllocator* allocator;

    int width;
    int height;
    VkFormat format;
};

// type for vulkan specialization constant and push constant
union vk_specialization_type { int i; float f; uint32_t u32; };
union vk_constant_type { int i; float f; };
#endif // NCNN_VULKAN

// misc function
#if NCNN_PIXEL
// convert yuv420sp(nv21) to rgb, the fast approximate version
void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// image pixel bilinear resize
void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
// image pixel bilinear resize with stride(bytes-per-row) parameter
void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
// image pixel bilinear resize, convenient wrapper for yuv420sp(nv21)
void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
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
void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
// image pixel kanna rotate with stride(bytes-per-row) parameter
void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
// image pixel kanna rotate, convenient wrapper for yuv420sp(nv21)
void kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
#endif // NCNN_PIXEL_ROTATE

// mat process
enum
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
};
void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt = Option());
void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt = Option());
void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
void convert_packing(const Mat& src, Mat& dst, int elempack, const Option& opt = Option());
void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt = Option());
void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
void quantize_float32_to_int8(const Mat& src, Mat& dst, float scale, const Option& opt = Option());
void dequantize_int32_to_float32(Mat& m, float scale, const float* bias, int bias_data_size, const Option& opt = Option());
void requantize_int8_to_int8(const Mat& src, Mat& dst, float scale_in, float scale_out, const float* bias, int bias_data_size, int fusion_relu, const Option& opt = Option());

inline Mat::Mat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline Mat::Mat(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

inline Mat::Mat(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(const Mat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c), cstep(m.cstep)
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Mat::~Mat()
{
    release();
}

inline Mat& Mat::operator=(const Mat& m)
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
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void Mat::fill(float _v)
{
    int size = (int)total();
    float* ptr = (float*)data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
    if (nn > 0)
    {
    asm volatile (
        "0:                             \n"
        "subs       %w0, %w0, #1        \n"
        "st1        {%4.4s}, [%1], #16  \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#else
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

inline void Mat::fill(int _v)
{
    int size = (int)total();
    int* ptr = (int*)data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
    int32x4_t _c = vdupq_n_s32(_v);
#if __aarch64__
    if (nn > 0)
    {
    asm volatile (
        "0:                             \n"
        "subs       %w0, %w0, #1        \n"
        "st1        {%4.4s}, [%1], #16  \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#else
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.s32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

#if __ARM_NEON
inline void Mat::fill(float32x4_t _v)
{
    int size = total();
    float* ptr = (float*)data;
    for (int i=0; i<size; i++)
    {
        vst1q_f32(ptr, _v);
        ptr += 4;
    }
}
#endif // __ARM_NEON

template <typename T>
inline void Mat::fill(T _v)
{
    int size = total();
    T* ptr = (T*)data;
    for (int i=0; i<size; i++)
    {
        ptr[i] = _v;
    }
}

inline Mat Mat::clone(Allocator* allocator) const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, allocator);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * elemsize);
    }

    return m;
}

inline Mat Mat::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * c != _w)
        return Mat();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Mat m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i=0; i<c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

inline Mat Mat::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * c != _w * _h)
        return Mat();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Mat m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i=0; i<c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.cstep = _w * _h;

    return m;
}

inline Mat Mat::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * c != _w * _h * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize(_w * _h * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i=0; i<_c; i++)
            {
                const void* ptr = (unsigned char*)data + i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, _w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.cstep = alignSize(_w * _h * elemsize, 16) / elemsize;

    return m;
}

inline void Mat::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create_like(const Mat& m, Allocator* _allocator)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

#if NCNN_VULKAN
inline void Mat::create_like(const VkMat& m, Allocator* _allocator)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}
#endif // NCNN_VULKAN

inline void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline void Mat::release()
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
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Mat::total() const
{
    return cstep * c;
}

inline Mat Mat::channel(int _c)
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::channel(int _c) const
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + w * y * elemsize);
}

inline const float* Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline T* Mat::row(int y)
{
    return (T*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline const T* Mat::row(int y) const
{
    return (const T*)((unsigned char*)data + w * y * elemsize);
}

inline Mat Mat::channel_range(int _c, int channels)
{
    return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::channel_range(int _c, int channels) const
{
    return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, elempack, allocator);
}

inline Mat Mat::range(int x, int n)
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::range(int x, int n) const
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

template <typename T>
inline Mat::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Mat::operator const T*() const
{
    return (const T*)data;
}

inline float& Mat::operator[](size_t i)
{
    return ((float*)data)[i];
}

inline const float& Mat::operator[](size_t i) const
{
    return ((const float*)data)[i];
}

#if NCNN_VULKAN

inline VkMat::VkMat()
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline VkMat::VkMat(int _w, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator, _staging_allocator);
}

inline VkMat::VkMat(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(0), offset(0), staging_data(0), refcount(0), staging_refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator, _staging_allocator);
}

inline VkMat::VkMat(const VkMat& m)
    : data(m.data), offset(m.offset), staging_data(m.staging_data), refcount(m.refcount), staging_refcount(m.staging_refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), staging_allocator(m.staging_allocator), dims(m.dims), w(m.w), h(m.h), c(m.c)
{
    if (refcount)
        NCNN_XADD(refcount, 1);

    if (staging_refcount)
        NCNN_XADD(staging_refcount, 1);

    cstep = m.cstep;
}

inline VkMat::VkMat(int _w, VkBufferMemory* _data, size_t _offset, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), staging_allocator(_staging_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline VkMat::VkMat(int _w, int _h, VkBufferMemory* _data, size_t _offset, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), staging_allocator(_staging_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline VkMat::VkMat(int _w, int _h, int _c, VkBufferMemory* _data, size_t _offset, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), staging_allocator(_staging_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline VkMat::VkMat(int _w, VkBufferMemory* _data, size_t _offset, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), staging_allocator(_staging_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline VkMat::VkMat(int _w, int _h, VkBufferMemory* _data, size_t _offset, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), staging_allocator(_staging_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline VkMat::VkMat(int _w, int _h, int _c, VkBufferMemory* _data, size_t _offset, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
    : data(_data), offset(_offset), staging_data(0), refcount(0), staging_refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), staging_allocator(_staging_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline VkMat::~VkMat()
{
    release();
}

inline VkMat& VkMat::operator=(const VkMat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    if (m.staging_refcount)
        NCNN_XADD(m.staging_refcount, 1);

    release();

    data = m.data;
    offset = m.offset;
    staging_data = m.staging_data;
    refcount = m.refcount;
    staging_refcount = m.staging_refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;
    staging_allocator = m.staging_allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void VkMat::create(int _w, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator && staging_allocator == _staging_allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;
    staging_allocator = _staging_allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);
        offset = 0;

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkMat::create_like(const Mat& m, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator, _staging_allocator);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator, _staging_allocator);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator, _staging_allocator);
}

inline void VkMat::create_like(const VkMat& m, VkAllocator* _allocator, VkAllocator* _staging_allocator)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator, _staging_allocator);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator, _staging_allocator);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator, _staging_allocator);
}

inline void VkMat::prepare_staging_buffer()
{
    if (allocator->mappable)
        return;

    if (staging_allocator && staging_data)
        return;

    size_t totalsize = alignSize(total() * elemsize, 4);
    staging_data = staging_allocator->fastMalloc(totalsize);

    staging_refcount = (int*)((unsigned char*)staging_data + offsetof(VkBufferMemory, refcount));
    *staging_refcount = 1;
}

inline void VkMat::discard_staging_buffer()
{
    if (allocator->mappable)
        return;

    if (staging_refcount && NCNN_XADD(staging_refcount, -1) == 1)
    {
        if (staging_allocator && staging_data)
        {
            staging_allocator->fastFree(staging_data);
        }
    }

    staging_data = 0;
    staging_refcount = 0;
}

inline void VkMat::upload(const Mat& m)
{
    memcpy(mapped_ptr(), m.data, m.total() * m.elemsize);

    if (allocator->mappable)
    {
        allocator->flush(data);
    }
}

inline void VkMat::download(Mat& m) const
{
    if (allocator->mappable)
    {
        allocator->invalidate(data);
    }

    memcpy(m.data, mapped_ptr(), total() * elemsize);
}

inline Mat VkMat::mapped() const
{
    if (dims == 1)
        return Mat(w, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 2)
        return Mat(w, h, mapped_ptr(), elemsize, elempack, 0);

    if (dims == 3)
        return Mat(w, h, c, mapped_ptr(), elemsize, elempack, 0);

    return Mat();
}

inline void* VkMat::mapped_ptr() const
{
    VkBufferMemory* mappable_data = allocator->mappable ? data : staging_data;
    return (unsigned char*)mappable_data->mapped_ptr + mappable_data->offset + offset;
}

inline void VkMat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);

    if (staging_refcount)
        NCNN_XADD(staging_refcount, 1);
}

inline void VkMat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    if (staging_refcount && NCNN_XADD(staging_refcount, -1) == 1)
    {
        if (staging_allocator && staging_data)
        {
            staging_allocator->fastFree(staging_data);
        }
    }

    data = 0;
    offset = 0;
    staging_data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
    staging_refcount = 0;
}

inline bool VkMat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t VkMat::total() const
{
    return cstep * c;
}

inline VkMat VkMat::channel(int _c)
{
    return VkMat(w, h, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::channel(int _c) const
{
    return VkMat(w, h, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkMat VkMat::channel_range(int _c, int channels)
{
    return VkMat(w, h, channels, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::channel_range(int _c, int channels) const
{
    return VkMat(w, h, channels, data, cstep * _c * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkMat VkMat::row_range(int y, int rows)
{
    return VkMat(w, rows, data, w * y * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::row_range(int y, int rows) const
{
    return VkMat(w, rows, data, w * y * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkMat VkMat::range(int x, int n)
{
    return VkMat(n, data, x * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline const VkMat VkMat::range(int x, int n) const
{
    return VkMat(n, data, x * elemsize, elemsize, elempack, allocator, staging_allocator);
}

inline VkBuffer VkMat::buffer() const
{
    return data->buffer;
}

inline size_t VkMat::buffer_offset() const
{
    return data->offset + offset;
}

inline VkBuffer VkMat::staging_buffer() const
{
    return staging_data->buffer;
}

inline size_t VkMat::staging_buffer_offset() const
{
    return staging_data->offset;
}

inline VkImageMat::VkImageMat()
    : data(0), refcount(0), allocator(0), width(0), height(0), format(VK_FORMAT_UNDEFINED)
{
}

inline VkImageMat::VkImageMat(int _width, int _height, VkFormat _format, VkImageAllocator* _allocator)
    : data(0), refcount(0), allocator(0), width(0), height(0), format(VK_FORMAT_UNDEFINED)
{
    create(_width, _height, _format, _allocator);
}

inline VkImageMat::VkImageMat(const VkImageMat& m)
    : data(m.data), refcount(m.refcount), allocator(m.allocator), width(m.width), height(m.height), format(m.format)
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline VkImageMat::VkImageMat(int _width, int _height, VkImageMemory* _data, VkFormat _format, VkImageAllocator* _allocator)
    : data(_data), refcount(0), allocator(_allocator), width(_width), height(_height), format(_format)
{
}

inline VkImageMat::~VkImageMat()
{
    release();
}

inline VkImageMat& VkImageMat::operator=(const VkImageMat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    allocator = m.allocator;

    width = m.width;
    height = m.height;
    format = m.format;

    return *this;
}

inline void VkImageMat::create(int _width, int _height, VkFormat _format, VkImageAllocator* _allocator)
{
    if (width == _width && height == _height && format == _format && allocator == _allocator)
        return;

    release();

    allocator = _allocator;

    width = _width;
    height = _height;
    format = _format;

    if (total() > 0)
    {
        data = allocator->fastMalloc(width, height, format);

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageMat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline void VkImageMat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    data = 0;

    width = 0;
    height = 0;
    format = VK_FORMAT_UNDEFINED;

    refcount = 0;
}

inline bool VkImageMat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t VkImageMat::total() const
{
    return width * height;
}

inline VkImage VkImageMat::image() const
{
    return data->image;
}

inline VkImageView VkImageMat::imageview() const
{
    return data->imageview;
}

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_MAT_H
