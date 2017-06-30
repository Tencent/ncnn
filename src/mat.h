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

namespace ncnn {

// the three dimension matrix
class Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w);
    // image
    Mat(int w, int h);
    // dim
    Mat(int w, int h, int c);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, float* data);
    // external image
    Mat(int w, int h, float* data);
    // external dim
    Mat(int w, int h, int c, float* data);
    // release
    ~Mat();
    // assign
    Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    // deep copy
    Mat clone() const;
    // reshape vec
    Mat reshape(int w) const;
    // reshape image
    Mat reshape(int w, int h) const;
    // reshape dim
    Mat reshape(int w, int h, int c) const;
    // allocate vec
    void create(int w);
    // allocate image
    void create(int w, int h);
    // allocate dim
    void create(int w, int h, int c);
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
    operator float*();
    operator const float*() const;

    enum
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB       = 1,
        PIXEL_BGR       = (1 << 1),
        PIXEL_GRAY      = (1 << 2),
        PIXEL_RGBA      = (1 << 3),

        PIXEL_RGB2BGR   = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY  = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB   = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY  = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB  = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR  = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB  = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR  = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h);
    // convenient construct from pixel data and resize to specific size
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type);
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height);

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precisoin floating point data
    static Mat from_float16(const unsigned short* data, int size);

    // the dimensionality
    int dims;
    // pointer to the data
    float* data;

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    int w;
    int h;
    int c;

    size_t cstep;
};

// misc function
// image pixel bilinear resize
void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);

// mat process
enum
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
};
void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v);
void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right);

// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
static inline void NCNN_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

inline Mat::Mat()
    : dims(0), data(0), refcount(0), w(0), h(0), c(0), cstep(0)
{
}

inline Mat::Mat(int _w)
    : dims(0), data(0), refcount(0)
{
    create(_w);
}

inline Mat::Mat(int _w, int _h)
    : dims(0), data(0), refcount(0)
{
    create(_w, _h);
}

inline Mat::Mat(int _w, int _h, int _c)
    : dims(0), data(0), refcount(0)
{
    create(_w, _h, _c);
}

inline Mat::Mat(const Mat& m)
    : dims(m.dims), data(m.data), refcount(m.refcount)
{
    if (refcount)
        NCNN_XADD(refcount, 1);

    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;
}

inline Mat::Mat(int _w, float* _data)
    : dims(1), data(_data), refcount(0)
{
    w = _w;
    h = 1;
    c = 1;

    cstep = w;
}

inline Mat::Mat(int _w, int _h, float* _data)
    : dims(2), data(_data), refcount(0)
{
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, float* _data)
    : dims(3), data(_data), refcount(0)
{
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * sizeof(float), 16) >> 2;
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

    dims = m.dims;
    data = m.data;
    refcount = m.refcount;

    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void Mat::fill(float _v)
{
    size_t _total = total();
    for (size_t i = 0; i < _total; i++)
    {
        data[i] = _v;
    }
}

inline Mat Mat::clone() const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w);
    else if (dims == 2)
        m.create(w, h);
    else if (dims == 3)
        m.create(w, h, c);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * sizeof(float));
    }

    return m;
}

inline Mat Mat::reshape(int _w) const
{
    Mat m = *this;

    m.dims = 1;

    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

inline Mat Mat::reshape(int _w, int _h) const
{
    Mat m = *this;

    m.dims = 2;

    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.cstep = _w * _h;

    return m;
}

inline Mat Mat::reshape(int _w, int _h, int _c) const
{
    Mat m = *this;

    m.dims = 3;

    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.cstep = alignSize(_w * _h * sizeof(float), 16) >> 2;

    return m;
}

inline void Mat::create(int _w)
{
    release();

    dims = 1;

    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = total() * sizeof(float);
        data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h)
{
    release();

    dims = 2;

    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = total() * sizeof(float);
        data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c)
{
    release();

    dims = 3;

    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * sizeof(float), 16) >> 2;

    if (total() > 0)
    {
        size_t totalsize = total() * sizeof(float);
        data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

inline void Mat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
        fastFree(data);

    dims = 0;
    data = 0;

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

inline Mat Mat::channel(int c)
{
    return Mat(w, h, data + cstep * c);
}

inline const Mat Mat::channel(int c) const
{
    return Mat(w, h, data + cstep * c);
}

inline float* Mat::row(int y)
{
    return data + w * y;
}

inline const float* Mat::row(int y) const
{
    return data + w * y;
}

inline Mat::operator float*()
{
    return data;
}

inline Mat::operator const float*() const
{
    return data;
}

} // namespace ncnn

#endif // NCNN_MAT_H
