// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Op>
static void binary_op_vector_no_broadcast_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size)
{
    const Op op;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        __m512 _b = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr1));
        __m256i _outp = float2bfloat_avx512(op.func_pack16(_p, _b));
        _mm256_storeu_si256((__m256i*)outptr, _outp);
        ptr += 16;
        ptr1 += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        __m256 _b = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr1));
        __m128i _outp = float2bfloat_avx(op.func_pack8(_p, _b));
        _mm_storeu_si128((__m128i*)outptr, _outp);
        ptr += 8;
        ptr1 += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        __m128 _b = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr1));
        __m128 _outp4 = op.func_pack4(_p, _b);
        _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_outp4, _outp4));
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op.func(bfloat16_to_float32(*ptr++), bfloat16_to_float32(*ptr1++)));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __SSE2__
    __m128i _b1_128i = (elempack == 4) ? _mm_loadl_epi64((const __m128i*)ptr1) : _mm_set1_epi16((short)*ptr1);
    __m128 _b_128 = bfloat2float_sse(_b1_128i);
#if __AVX__
    __m128i _b2_128i = (elempack == 8) ? _mm_loadu_si128((const __m128i*)ptr1) : _mm_set1_epi16((short)*ptr1);
    __m256 _b_256 = bfloat2float_avx(_b2_128i);
#if __AVX512F__
    __m256i _b3_256i = (elempack == 16) ? _mm256_loadu_si256((const __m256i*)ptr1) : _mm256_set1_epi16((short)*ptr1);
    __m512 _b_512 = bfloat2float_avx512(_b3_256i);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        __m256i _outp = float2bfloat_avx512(op.func_pack16(_p, _b_512));
        _mm256_storeu_si256((__m256i*)outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        __m128i _outp = float2bfloat_avx(op.func_pack8(_p, _b_256));
        _mm_storeu_si128((__m128i*)outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        __m128 _outp4 = op.func_pack4(_p, _b_128);
        _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_outp4, _outp4));
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op.func(bfloat16_to_float32(*ptr++), b));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float a = bfloat16_to_float32(*ptr);

    int i = 0;
#if __SSE2__
    __m128i _a1_128i = (elempack == 4) ? _mm_loadl_epi64((const __m128i*)ptr) : _mm_set1_epi16((short)*ptr);
    __m128 _a_128 = bfloat2float_sse(_a1_128i);
#if __AVX__
    __m128i _a2_128i = (elempack == 8) ? _mm_loadu_si128((const __m128i*)ptr) : _mm_set1_epi16((short)*ptr);
    __m256 _a_256 = bfloat2float_avx(_a2_128i);
#if __AVX512F__
    __m256i _a3_256i = (elempack == 16) ? _mm256_loadu_si256((const __m256i*)ptr) : _mm256_set1_epi16((short)*ptr);
    __m512 _a_512 = bfloat2float_avx512(_a3_256i);
    for (; i + 15 < size; i += 16)
    {
        __m512 _b = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr1));
        __m256i _outp = float2bfloat_avx512(op.func_pack16(_a_512, _b));
        _mm256_storeu_si256((__m256i*)outptr, _outp);
        ptr1 += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _b = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr1));
        __m128i _outp = float2bfloat_avx(op.func_pack8(_a_256, _b));
        _mm_storeu_si128((__m128i*)outptr, _outp);
        ptr1 += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _b = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr1));
        __m128 _outp4 = op.func_pack4(_a_128, _b);
        _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_outp4, _outp4));
        ptr1 += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op.func(a, bfloat16_to_float32(*ptr1++)));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        for (int i = 0; i < w; i++)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _b = _mm512_set1_ps(bfloat16_to_float32(*ptr1));
            __m256i _outp = float2bfloat_avx512(op.func_pack16(_p, _b));
            _mm256_storeu_si256((__m256i*)outptr, _outp);
            ptr += 16;
            ptr1 += 1;
            outptr += 16;
        }
        return;
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        for (int i = 0; i < w; i++)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _b = _mm256_set1_ps(bfloat16_to_float32(*ptr1));
            __m128i _outp = float2bfloat_avx(op.func_pack8(_p, _b));
            _mm_storeu_si128((__m128i*)outptr, _outp);
            ptr += 8;
            ptr1 += 1;
            outptr += 8;
        }
        return;
    }
#endif // __AVX__
    if (elempack == 4)
    {
        for (int i = 0; i < w; i++)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _b = _mm_set1_ps(bfloat16_to_float32(*ptr1));
            __m128 _outp4 = op.func_pack4(_p, _b);
            _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_outp4, _outp4));
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
        return;
    }
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;
    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        __m512 _b_avx512 = _mm512_set1_ps(b);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m256i _outp = float2bfloat_avx512(op.func_pack16(_p, _b_avx512));
            _mm256_storeu_si256((__m256i*)outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
    }
#endif // __AVX512F__
    {
        __m256 _b_avx = _mm256_set1_ps(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m128i _outp = float2bfloat_avx(op.func_pack8(_p, _b_avx));
            _mm_storeu_si128((__m128i*)outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
    }
#endif // __AVX__
    {
        __m128 _b_sse = _mm_set1_ps(b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _outp4 = op.func_pack4(_p, _b_sse);
            _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_outp4, _outp4));
            ptr += 4;
            outptr += 4;
        }
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op.func(bfloat16_to_float32(*ptr++), b));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        for (int i = 0; i < w; i++)
        {
            __m512 _b = _mm512_set1_ps(bfloat16_to_float32(*ptr1));
            __m256i _outp = float2bfloat_avx512(op.func_pack16(_p, _b));
            _mm256_storeu_si256((__m256i*)outptr, _outp);
            ptr1 += 1;
            outptr += 16;
        }
        return;
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        for (int i = 0; i < w; i++)
        {
            __m256 _b = _mm256_set1_ps(bfloat16_to_float32(*ptr1));
            __m128i _outp = float2bfloat_avx(op.func_pack8(_p, _b));
            _mm_storeu_si128((__m128i*)outptr, _outp);
            ptr1 += 1;
            outptr += 8;
        }
        return;
    }
#endif // __AVX__
    if (elempack == 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        for (int i = 0; i < w; i++)
        {
            __m128 _b = _mm_set1_ps(bfloat16_to_float32(*ptr1));
            __m128 _outp4 = op.func_pack4(_p, _b);
            _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_outp4, _outp4));
            ptr1 += 1;
            outptr += 4;
        }
        return;
    }
#endif // __SSE2__
}

template<typename Op>
static void binary_op_vector_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
            return binary_op_vector_no_broadcast_bf16s<Op>(ptr, ptr1, outptr, size);
        if (bw == 1)
            return binary_op_vector_broadcast_b_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
        if (aw == 1)
            return binary_op_vector_broadcast_a_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
    }

    if (bp == 1)
    {
        if (aw == bw)
            return binary_op_vector_broadcast_pb_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        if (bw == 1)
            return binary_op_vector_broadcast_pb_b_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        if (aw == 1)
            return binary_op_vector_broadcast_pb_a_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
    }
}
