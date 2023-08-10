// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "binaryop_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include <math.h>

namespace ncnn {

BinaryOp_x86::BinaryOp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector_no_broadcast(const float* ptr, const float* ptr1, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _b = _mm512_loadu_ps(ptr1);
        __m512 _outp = op.func_pack16(_p, _b);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        ptr1 += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _b = _mm256_loadu_ps(ptr1);
        __m256 _outp = op.func_pack8(_p, _b);
        _mm256_storeu_ps(outptr, _outp);
        ptr += 8;
        ptr1 += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _b = _mm_loadu_ps(ptr1);
        __m128 _outp = op.func_pack4(_p, _b);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(*ptr, *ptr1);
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_b(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float b = *ptr1;

    int i = 0;
#if __SSE2__
    __m128 _b_128 = (elempack == 4) ? _mm_loadu_ps(ptr1) : _mm_set1_ps(b);
#if __AVX__
    __m256 _b_256 = (elempack == 8) ? _mm256_loadu_ps(ptr1) : _mm256_insertf128_ps(_mm256_castps128_ps256(_b_128), _b_128, 1);
#if __AVX512F__
    __m512 _b_512 = (elempack == 16) ? _mm512_loadu_ps(ptr1) : _mm512_insertf32x8(_mm512_castps256_ps512(_b_256), _b_256, 1);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _outp = op.func_pack16(_p, _b_512);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _outp = op.func_pack8(_p, _b_256);
        _mm256_storeu_ps(outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _outp = op.func_pack4(_p, _b_128);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float a = *ptr;

    int i = 0;
#if __SSE2__
    __m128 _a_128 = (elempack == 4) ? _mm_loadu_ps(ptr) : _mm_set1_ps(a);
#if __AVX__
    __m256 _a_256 = (elempack == 8) ? _mm256_loadu_ps(ptr) : _mm256_insertf128_ps(_mm256_castps128_ps256(_a_128), _a_128, 1);
#if __AVX512F__
    __m512 _a_512 = (elempack == 16) ? _mm512_loadu_ps(ptr) : _mm512_insertf32x8(_mm512_castps256_ps512(_a_256), _a_256, 1);
    for (; i + 15 < size; i += 16)
    {
        __m512 _b = _mm512_loadu_ps(ptr1);
        __m512 _outp = op.func_pack16(_a_512, _b);
        _mm512_storeu_ps(outptr, _outp);
        ptr1 += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _b = _mm256_loadu_ps(ptr1);
        __m256 _outp = op.func_pack8(_a_256, _b);
        _mm256_storeu_ps(outptr, _outp);
        ptr1 += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _b = _mm_loadu_ps(ptr1);
        __m128 _outp = op.func_pack4(_a_128, _b);
        _mm_storeu_ps(outptr, _outp);
        ptr1 += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        int i = 0;
        for (; i < w; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _b = _mm512_set1_ps(*ptr1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 1;
            outptr += 16;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        int i = 0;
#if __AVX512F__
        for (; i + 1 < w; i += 2)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m256 _b0 = _mm256_set1_ps(ptr1[0]);
            __m256 _b1 = _mm256_set1_ps(ptr1[1]);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b0), _b1, 1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 2;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i < w; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _b = _mm256_set1_ps(*ptr1);
            __m256 _outp = op.func_pack8(_p, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            ptr1 += 1;
            outptr += 8;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        int i = 0;
#if __AVX__
#if __AVX512F__
        for (; i + 3 < w; i += 4)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m128 _b2 = _mm_set1_ps(ptr1[2]);
            __m128 _b3 = _mm_set1_ps(ptr1[3]);
            __m256 _b01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _b23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b2), _b3, 1);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b01), _b23, 1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 4;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 1 < w; i += 2)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m256 _b = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _outp = op.func_pack8(_p, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            ptr1 += 2;
            outptr += 8;
        }
#endif // __AVX__
        for (; i < w; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _b = _mm_set1_ps(*ptr1);
            __m128 _outp = op.func_pack4(_p, _b);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _b_avx512 = _mm512_set1_ps(*ptr1);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _outp = op.func_pack16(_p, _b_avx512);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    __m256 _b_avx = _mm256_set1_ps(*ptr1);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _outp = op.func_pack8(_p, _b_avx);
        _mm256_storeu_ps(outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    __m128 _b = _mm_set1_ps(*ptr1);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _outp = op.func_pack4(_p, _b);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        int i = 0;
        __m512 _p = _mm512_loadu_ps(ptr);
        for (; i < w; i++)
        {
            __m512 _b = _mm512_set1_ps(*ptr1);
            __m512 _outp = op.func_pack16(_p, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr1 += 1;
            outptr += 16;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        int i = 0;
        __m256 _p = _mm256_loadu_ps(ptr);
#if __AVX512F__
        __m512 _p_512 = _mm512_insertf32x8(_mm512_castps256_ps512(_p), _p, 1);
        for (; i + 1 < w; i += 2)
        {
            __m256 _b0 = _mm256_set1_ps(ptr1[0]);
            __m256 _b1 = _mm256_set1_ps(ptr1[1]);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b0), _b1, 1);
            __m512 _outp = op.func_pack16(_p_512, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr1 += 2;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i < w; i++)
        {
            __m256 _b = _mm256_set1_ps(*ptr1);
            __m256 _outp = op.func_pack8(_p, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr1 += 1;
            outptr += 8;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        int i = 0;
        __m128 _p = _mm_loadu_ps(ptr);
#if __AVX__
        __m256 _p_256 = _mm256_insertf128_ps(_mm256_castps128_ps256(_p), _p, 1);
#if __AVX512F__
        __m512 _p_512 = _mm512_insertf32x8(_mm512_castps256_ps512(_p_256), _p_256, 1);
        for (; i + 3 < w; i += 4)
        {
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m128 _b2 = _mm_set1_ps(ptr1[2]);
            __m128 _b3 = _mm_set1_ps(ptr1[3]);
            __m256 _b01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _b23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_b2), _b3, 1);
            __m512 _b = _mm512_insertf32x8(_mm512_castps256_ps512(_b01), _b23, 1);
            __m512 _outp = op.func_pack16(_p_512, _b);
            _mm512_storeu_ps(outptr, _outp);
            ptr1 += 4;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 1 < w; i += 2)
        {
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m256 _b = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
            __m256 _outp = op.func_pack8(_p_256, _b);
            _mm256_storeu_ps(outptr, _outp);
            ptr1 += 2;
            outptr += 8;
        }
#endif // __AVX__
        for (; i < w; i++)
        {
            __m128 _b = _mm_set1_ps(*ptr1);
            __m128 _outp = op.func_pack4(_p, _b);
            _mm_storeu_ps(outptr, _outp);
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            // no broadcast
            return binary_op_vector_no_broadcast<Op>(ptr, ptr1, outptr, size);
        }

        if (bw == 1)
        {
            // broadcast single b
            return binary_op_vector_broadcast_b<Op>(ptr, ptr1, outptr, size, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a
            return binary_op_vector_broadcast_a<Op>(ptr, ptr1, outptr, size, elempack);
        }
    }

    if (bp == 1)
    {
        if (aw == bw)
        {
            // broadcast pack1 b
            return binary_op_vector_broadcast_pb<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (bw == 1)
        {
            // broadcast pack1 single b
            return binary_op_vector_broadcast_pb_b<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a and pack1 b
            return binary_op_vector_broadcast_pb_a<Op>(ptr, ptr1, outptr, w, elempack);
        }
    }

    // shall never reach here
}

namespace BinaryOp_x86_functor {

struct binary_op_add
{
    float func(const float& x, const float& y) const
    {
        return x + y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_add_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_sub
{
    float func(const float& x, const float& y) const
    {
        return x - y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_sub_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_mul
{
    float func(const float& x, const float& y) const
    {
        return x * y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_mul_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_mul_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_mul_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_div
{
    float func(const float& x, const float& y) const
    {
        return x / y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_div_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_max
{
    float func(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_max_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_max_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_max_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_min
{
    float func(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_min_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_min_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_min_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_pow
{
    float func(const float& x, const float& y) const
    {
        return (float)powf(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return pow_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return pow256_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return pow512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rsub
{
    float func(const float& x, const float& y) const
    {
        return y - x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(y, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_sub_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rdiv
{
    float func(const float& x, const float& y) const
    {
        return y / x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(y, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_div_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rpow
{
    float func(const float& x, const float& y) const
    {
        return (float)powf(y, x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return pow_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return pow256_ps(y, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return pow512_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_atan2
{
    float func(const float& x, const float& y) const
    {
        return (float)atan2f(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return atan2_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return atan2256_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return atan2512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_ratan2
{
    float func(const float& x, const float& y) const
    {
        return (float)atan2f(y, x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return atan2_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return atan2256_ps(y, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return atan2512_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

} // namespace BinaryOp_x86_functor

static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_x86_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector<binary_op_add>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector<binary_op_sub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector<binary_op_mul>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector<binary_op_div>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector<binary_op_max>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector<binary_op_min>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector<binary_op_pow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector<binary_op_rsub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector<binary_op_rdiv>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector<binary_op_rpow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector<binary_op_atan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector<binary_op_ratan2>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

static void binary_op_scalar(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = c.channel(q);

        binary_op_vector(ptr, &b, outptr, size, 1, 1, 1, op_type);
    }
}

static void binary_op_no_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        const float* ptr1 = b.channel(q);
        float* outptr = c.channel(q);

        binary_op_vector(ptr, ptr1, outptr, size, size, 1, 1, op_type);
    }
}

static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (b.w * b.h * b.d * b.c * b.elempack == 1)
    {
        return binary_op_scalar(a, b[0], c, op_type, opt);
    }

    if (a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c && a.elempack == b.elempack)
    {
        return binary_op_no_broadcast(a, b, c, op_type, opt);
    }

    const int dims = c.dims;

    if (dims == 2)
    {
        const int h = c.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const int y0 = std::min(y, a.h - 1);
            const int y1 = std::min(y, b.h - 1);

            const float* ptr = a.row(y0);
            const float* ptr1 = b.row(y1);
            float* outptr = c.row(y);

            binary_op_vector(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int channels = c.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int q0 = std::min(q, a.c - 1);
            const int q1 = std::min(q, b.c - 1);

            if (b.d * b.h * b.w == 1)
            {
                const float* ptr = a.channel(q0);
                const float* ptr1 = b.channel(q1);
                float* outptr = c.channel(q);

                binary_op_vector(ptr, ptr1, outptr, a.w * a.h * a.d, 1, a.elempack, b.elempack, op_type);
                continue;
            }

            if (b.h * b.w == 1)
            {
                for (int z = 0; z < c.d; z++)
                {
                    const int z0 = std::min(z, a.d - 1);
                    const int z1 = std::min(z, b.d - 1);

                    const float* ptr = a.channel(q0).depth(z0);
                    const float* ptr1 = b.channel(q1).depth(z1);
                    float* outptr = c.channel(q).depth(z);

                    binary_op_vector(ptr, ptr1, outptr, a.w * a.h, 1, a.elempack, b.elempack, op_type);
                }
                continue;
            }

            for (int z = 0; z < c.d; z++)
            {
                const int z0 = std::min(z, a.d - 1);
                const int z1 = std::min(z, b.d - 1);

                for (int y = 0; y < c.h; y++)
                {
                    const int y0 = std::min(y, a.h - 1);
                    const int y1 = std::min(y, b.h - 1);

                    const float* ptr = a.channel(q0).depth(z0).row(y0);
                    const float* ptr1 = b.channel(q1).depth(z1).row(y1);
                    float* outptr = c.channel(q).depth(z).row(y);

                    binary_op_vector(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
                }
            }
        }
    }
}

static void binary_op_scalar_inplace(Mat& a, float b, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        binary_op_vector(ptr, &b, ptr, size, 1, 1, 1, op_type);
    }
}

static int get_reverse_op_type(int op_type)
{
    if (op_type == BinaryOp::Operation_SUB) return BinaryOp::Operation_RSUB;
    if (op_type == BinaryOp::Operation_DIV) return BinaryOp::Operation_RDIV;
    if (op_type == BinaryOp::Operation_POW) return BinaryOp::Operation_RPOW;
    if (op_type == BinaryOp::Operation_ATAN2) return BinaryOp::Operation_RATAN2;
    if (op_type == BinaryOp::Operation_RSUB) return BinaryOp::Operation_SUB;
    if (op_type == BinaryOp::Operation_RDIV) return BinaryOp::Operation_DIV;
    if (op_type == BinaryOp::Operation_RPOW) return BinaryOp::Operation_POW;
    if (op_type == BinaryOp::Operation_RATAN2) return BinaryOp::Operation_ATAN2;
    return op_type;
}

int BinaryOp_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w * A.elempack == B.h * B.elempack)
                A2 = A.reshape(1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 2;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 3;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 4;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c, opt.workspace_allocator);
    }
    if (B.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (B.w * B.elempack == A.h * A.elempack)
                B2 = B.reshape(1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 2;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 3;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 4;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c, opt.workspace_allocator);
    }

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);
    const size_t out_elemsize = std::max(A2.elemsize, B2.elemsize);
    const int out_elempack = std::max(A2.elempack, B2.elempack);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const bool a_pack_is_lower = A2.elempack < B2.elempack;
    const bool a_pack_is_equal = A2.elempack == B2.elempack;
    const bool a_size_is_lower = A2.w * A2.h * A2.d * A2.c * A2.elempack < B2.w * B2.h * B2.d * B2.c * B2.elempack;
    if (a_pack_is_lower || (a_pack_is_equal && a_size_is_lower))
    {
        binary_op_broadcast(B2, A2, top_blob, get_reverse_op_type(op_type), opt);
    }
    else
    {
        binary_op_broadcast(A2, B2, top_blob, op_type, opt);
    }

    return 0;
}

int BinaryOp_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    binary_op_scalar_inplace(bottom_top_blob, b, op_type, opt);

    return 0;
}

} // namespace ncnn
