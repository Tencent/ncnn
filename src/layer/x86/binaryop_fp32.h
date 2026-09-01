// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static bool binaryop_use_fma(int op_type)
{
    return op_type == BinaryOp::Operation_POW
           || op_type == BinaryOp::Operation_RPOW
           || op_type == BinaryOp::Operation_ATAN2
           || op_type == BinaryOp::Operation_RATAN2
           || op_type == BinaryOp::Operation_FMOD
           || op_type == BinaryOp::Operation_RFMOD
           || op_type == BinaryOp::Operation_LOGADDEXP
           || op_type == BinaryOp::Operation_FLOOR_DIVIDE
           || op_type == BinaryOp::Operation_RFLOOR_DIVIDE
           || op_type == BinaryOp::Operation_REMAINDER
           || op_type == BinaryOp::Operation_RREMAINDER;
}

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void binary_op_vector_fma(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void binary_op_vector_fma4(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type);
#endif

namespace binaryop_x86_functor {

#include "binaryop_functor.h"

} // namespace binaryop_x86_functor

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
    __m256 _b_256 = (elempack == 8) ? _mm256_loadu_ps(ptr1) : combine4x2_ps(_b_128, _b_128);
#if __AVX512F__
    __m512 _b_512 = (elempack == 16) ? _mm512_loadu_ps(ptr1) : combine8x2_ps(_b_256, _b_256);
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
    __m256 _a_256 = (elempack == 8) ? _mm256_loadu_ps(ptr) : combine4x2_ps(_a_128, _a_128);
#if __AVX512F__
    __m512 _a_512 = (elempack == 16) ? _mm512_loadu_ps(ptr) : combine8x2_ps(_a_256, _a_256);
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
            __m512 _b = combine8x2_ps(_b0, _b1);
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
            __m512 _b = combine4x4_ps(_b0, _b1, _b2, _b3);
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
            __m256 _b = combine4x2_ps(_b0, _b1);
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
        __m512 _p_512 = combine8x2_ps(_p, _p);
        for (; i + 1 < w; i += 2)
        {
            __m256 _b0 = _mm256_set1_ps(ptr1[0]);
            __m256 _b1 = _mm256_set1_ps(ptr1[1]);
            __m512 _b = combine8x2_ps(_b0, _b1);
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
        __m256 _p_256 = combine4x2_ps(_p, _p);
#if __AVX512F__
        __m512 _p_512 = combine8x2_ps(_p_256, _p_256);
        for (; i + 3 < w; i += 4)
        {
            __m128 _b0 = _mm_set1_ps(ptr1[0]);
            __m128 _b1 = _mm_set1_ps(ptr1[1]);
            __m128 _b2 = _mm_set1_ps(ptr1[2]);
            __m128 _b3 = _mm_set1_ps(ptr1[3]);
            __m512 _b = combine4x4_ps(_b0, _b1, _b2, _b3);
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
            __m256 _b = combine4x2_ps(_b0, _b1);
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

static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (binaryop_use_fma(op_type) && ncnn::cpu_support_x86_fma())
    {
        binary_op_vector_fma(ptr, ptr1, outptr, aw, bw, ap, bp, op_type);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (binaryop_use_fma(op_type) && ncnn::cpu_support_x86_fma4())
    {
        binary_op_vector_fma4(ptr, ptr1, outptr, aw, bw, ap, bp, op_type);
        return;
    }
#endif
    using namespace binaryop_x86_functor;

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
    if (op_type == BinaryOp::Operation_FMOD) return binary_op_vector<binary_op_fmod>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFMOD) return binary_op_vector<binary_op_rfmod>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_LOGADDEXP) return binary_op_vector<binary_op_logaddexp>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FLOOR_DIVIDE) return binary_op_vector<binary_op_floor_divide>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return binary_op_vector<binary_op_rfloor_divide>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_REMAINDER) return binary_op_vector<binary_op_remainder>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RREMAINDER) return binary_op_vector<binary_op_rremainder>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}
