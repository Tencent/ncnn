// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Op>
static void binary_op_vector_no_broadcast_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size)
{
    const Op op;

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
        __m256 _b = bfloat2float_avx((__m128i)__lsx_vld(ptr1, 0));
        __m256 _outp = op(_p, _b);
        __lsx_vst(float2bfloat_avx(_outp), outptr, 0);
        ptr += 8;
        ptr1 += 8;
        outptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
        __m128 _b = bfloat2float_sse((__m128i)__lsx_vld(ptr1, 0));
        __m128 _outp = op(_p, _b);
        __lsx_vstelm_d(float2bfloat_sse(_outp, _outp), outptr, 0, 0);
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(bfloat16_to_float32(*ptr++), bfloat16_to_float32(*ptr1++)));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __loongarch_sx
    __m128 _b_128 = (elempack == 4) ? bfloat2float_sse((__m128i)__lsx_vld(ptr1, 0)) : (__m128)__lsx_vreplfr2vr_s(b);
#if __loongarch_asx
    __m256 _b_256 = (elempack == 8) ? bfloat2float_avx((__m128i)__lsx_vld(ptr1, 0)) : combine4x2_ps(_b_128, _b_128);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
        __m256 _outp = op(_p, _b_256);
        __lsx_vst(float2bfloat_avx(_outp), outptr, 0);
        ptr += 8;
        outptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
        __m128 _outp = op(_p, _b_128);
        __lsx_vstelm_d(float2bfloat_sse(_outp, _outp), outptr, 0, 0);
        ptr += 4;
        outptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(bfloat16_to_float32(*ptr++), b));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float a = bfloat16_to_float32(*ptr);

    int i = 0;
#if __loongarch_sx
    __m128 _a_128 = (elempack == 4) ? bfloat2float_sse((__m128i)__lsx_vld(ptr, 0)) : (__m128)__lsx_vreplfr2vr_s(a);
#if __loongarch_asx
    __m256 _a_256 = (elempack == 8) ? bfloat2float_avx((__m128i)__lsx_vld(ptr, 0)) : combine4x2_ps(_a_128, _a_128);
    for (; i + 7 < size; i += 8)
    {
        __m256 _b = bfloat2float_avx((__m128i)__lsx_vld(ptr1, 0));
        __m256 _outp = op(_a_256, _b);
        __lsx_vst(float2bfloat_avx(_outp), outptr, 0);
        ptr1 += 8;
        outptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _b = bfloat2float_sse((__m128i)__lsx_vld(ptr1, 0));
        __m128 _outp = op(_a_128, _b);
        __lsx_vstelm_d(float2bfloat_sse(_outp, _outp), outptr, 0, 0);
        ptr1 += 4;
        outptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(a, bfloat16_to_float32(*ptr1++)));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        for (int i = 0; i < w; i++)
        {
            __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
            __m256 _b = __lasx_xvreplfr2vr_s(bfloat16_to_float32(*ptr1));
            __m256 _outp = op(_p, _b);
            __lsx_vst(float2bfloat_avx(_outp), outptr, 0);
            ptr += 8;
            ptr1 += 1;
            outptr += 8;
        }
        return;
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
        for (int i = 0; i < w; i++)
        {
            __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
            __m128 _b = (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(*ptr1));
            __m128 _outp = op(_p, _b);
            __lsx_vstelm_d(float2bfloat_sse(_outp, _outp), outptr, 0, 0);
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
        return;
    }
#endif // __loongarch_sx
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;
    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    {
        __m256 _b_256 = __lasx_xvreplfr2vr_s(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
            __m256 _outp = op(_p, _b_256);
            __lsx_vst(float2bfloat_avx(_outp), outptr, 0);
            ptr += 8;
            outptr += 8;
        }
    }
#endif // __loongarch_asx
    {
        __m128 _b_128 = (__m128)__lsx_vreplfr2vr_s(b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
            __m128 _outp = op(_p, _b_128);
            __lsx_vstelm_d(float2bfloat_sse(_outp, _outp), outptr, 0, 0);
            ptr += 4;
            outptr += 4;
        }
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(bfloat16_to_float32(*ptr++), b));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
        for (int i = 0; i < w; i++)
        {
            __m256 _b = __lasx_xvreplfr2vr_s(bfloat16_to_float32(*ptr1));
            __m256 _outp = op(_p, _b);
            __lsx_vst(float2bfloat_avx(_outp), outptr, 0);
            ptr1 += 1;
            outptr += 8;
        }
        return;
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
        for (int i = 0; i < w; i++)
        {
            __m128 _b = (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(*ptr1));
            __m128 _outp = op(_p, _b);
            __lsx_vstelm_d(float2bfloat_sse(_outp, _outp), outptr, 0, 0);
            ptr1 += 1;
            outptr += 4;
        }
        return;
    }
#endif // __loongarch_sx
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
