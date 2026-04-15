// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void softmax_bf16s_lsx(unsigned short* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __loongarch_sx
#if __loongarch_asx
    __m256 _max_lasx = (__m256)__lasx_xvreplfr2vr_s(-FLT_MAX);
#endif // __loongarch_asx
    __m128 _max = (__m128)__lsx_vreplfr2vr_s(-FLT_MAX);
#endif // __loongarch_sx
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _max_lasx = __lasx_xvfmax_s(_max_lasx, _p);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            _max = __lsx_vfmax_s(_max, _p);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr++));
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        _max_lasx = __lasx_xvfmax_s(_max_lasx, _max_lasx);
    }
    if (elempack == 4)
    {
        {
            __m128 _max0 = (__m128)__lasx_extract_lo128((__m256i)_max_lasx);
            __m128 _max1 = (__m128)__lasx_extract_hi128((__m256i)_max_lasx);
            _max = __lsx_vfmax_s(_max, _max0);
            _max = __lsx_vfmax_s(_max, _max1);
        }

        _max_lasx = combine4x2_ps(_max, _max);
    }
#endif // __loongarch_asx
    if (elempack == 1)
    {
#if __loongarch_asx
        {
            __m128 _max0 = (__m128)__lasx_extract_lo128((__m256i)_max_lasx);
            __m128 _max1 = (__m128)__lasx_extract_hi128((__m256i)_max_lasx);
            _max = __lsx_vfmax_s(_max, _max0);
            _max = __lsx_vfmax_s(_max, _max1);
        }
#endif // __loongarch_asx
        max = std::max(max, __lsx_reduce_fmax_s(_max));

        _max = (__m128)__lsx_vreplfr2vr_s(max);
#if __loongarch_asx
        _max_lasx = combine4x2_ps(_max, _max);
#endif // __loongarch_asx
    }
#endif // __loongarch_sx

    // reduce exp(x - max) and store back to bf16
#if __loongarch_sx
#if __loongarch_asx
    __m256 _sum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
    __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // __loongarch_sx
    float sum = 0.f;
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfsub_s(_p, _max_lasx);
            _p = exp256_ps(_p);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            _sum_lasx = __lasx_xvfadd_s(_sum_lasx, _p);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            _p = __lsx_vfsub_s(_p, _max);
            _p = exp_ps(_p);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max);
            *ptr = float32_to_bfloat16(v);
            sum += v;
            ptr++;
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
        _sum_lasx = __lasx_xvfdiv_s(_one, _sum_lasx);
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
#if __loongarch_asx
        {
            __m128 _sum0 = (__m128)__lasx_extract_lo128((__m256i)_sum_lasx);
            __m128 _sum1 = (__m128)__lasx_extract_hi128((__m256i)_sum_lasx);
            _sum = __lsx_vfadd_s(_sum, _sum0);
            _sum = __lsx_vfadd_s(_sum, _sum1);
        }
#endif // __loongarch_asx

        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        _sum = __lsx_vfdiv_s(_one, _sum);

#if __loongarch_asx
        _sum_lasx = combine4x2_ps(_sum, _sum);
#endif // __loongarch_asx
    }
#endif // __loongarch_sx
    if (elempack == 1)
    {
#if __loongarch_sx
#if __loongarch_asx
        {
            __m128 _sum0 = (__m128)__lasx_extract_lo128((__m256i)_sum_lasx);
            __m128 _sum1 = (__m128)__lasx_extract_hi128((__m256i)_sum_lasx);
            _sum = __lsx_vfadd_s(_sum, _sum0);
            _sum = __lsx_vfadd_s(_sum, _sum1);
        }
#endif // __loongarch_asx
        sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx

        sum = 1.f / sum;

#if __loongarch_sx
        _sum = (__m128)__lsx_vreplfr2vr_s(sum);
#if __loongarch_asx
        _sum_lasx = combine4x2_ps(_sum, _sum);
#endif // __loongarch_asx
#endif // __loongarch_sx
    }

    // div sum (multiply by reciprocal)
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfmul_s(_p, _sum_lasx);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            _p = __lsx_vfmul_s(_p, _sum);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * sum);
            ptr++;
        }
    }
}

#if __loongarch_sx
#if __loongarch_asx
static void softmax_bf16s_pack8_lsx(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            *maxptr = std::max(*maxptr, __lasx_reduce_fmax_s(_p));
            ptr += 8;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            __m256 _max = (__m256)__lasx_xvreplfr2vr_s(*maxptr);
            _p = exp256_ps(__lasx_xvfsub_s(_p, _max));
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            *sumptr += __lasx_reduce_fadd_s(_p);
            ptr += 8;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            _sum = __lasx_xvfdiv_s(_one, _sum);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            _sum = __lsx_vfdiv_s(_one, _sum);
            __lsx_vst(_sum, sumptr, 0);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            __m256 _sum = (__m256)__lasx_xvreplfr2vr_s(*sumptr);
            _p = __lasx_xvfmul_s(_p, _sum);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
            sumptr++;
        }
    }
}
#endif // __loongarch_asx

static void softmax_bf16s_pack4_lsx(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            *maxptr = std::max(*maxptr, __lsx_reduce_fmax_s(_p));
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            __m128 _max = (__m128)__lsx_vreplfr2vr_s(*maxptr);
            _p = exp_ps(__lsx_vfsub_s(_p, _max));
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            *sumptr += __lsx_reduce_fadd_s(_p);
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            _sum = __lasx_xvfdiv_s(_one, _sum);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            _sum = __lsx_vfdiv_s(_one, _sum);
            __lsx_vst(_sum, sumptr, 0);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            __m128 _sum = (__m128)__lsx_vreplfr2vr_s(*sumptr);
            _p = __lsx_vfmul_s(_p, _sum);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __loongarch_sx

static void softmax_bf16s_pack1_lsx(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            __m256 _max = (__m256)__lasx_xvld(maxptr, 0);
            _max = __lasx_xvfmax_s(_max, _p);
            __lasx_xvst((__m256i)_max, maxptr, 0);
            ptr += 8;
            maxptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);
            _max = __lsx_vfmax_s(_max, _p);
            __lsx_vst(_max, maxptr, 0);
            ptr += 4;
            maxptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            __m256 _max = (__m256)__lasx_xvld(maxptr, 0);
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            _p = __lasx_xvfsub_s(_p, _max);
            _p = exp256_ps(_p);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            _sum = __lasx_xvfadd_s(_sum, _p);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            ptr += 8;
            maxptr += 8;
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _p = __lsx_vfsub_s(_p, _max);
            _p = exp_ps(_p);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            __lsx_vst(_sum, sumptr, 0);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - *maxptr);
            *ptr = float32_to_bfloat16(v);
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            _sum = __lasx_xvfdiv_s(_one, _sum);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            _sum = __lsx_vfdiv_s(_one, _sum);
            __lsx_vst(_sum, sumptr, 0);
            sumptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            _p = __lasx_xvfmul_s(_p, _sum);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = bfloat2float_lsx((const unsigned short*)ptr);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _p = __lsx_vfmul_s(_p, _sum);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * *sumptr);
            ptr++;
            sumptr++;
        }
    }
}

static void softmax_bf16s_lsx_dispatch(unsigned short* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // init max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _negmax_lasx = (__m256)__lasx_xvreplfr2vr_s(-FLT_MAX);
        for (; j + 7 < size1; j += 8)
        {
            __lasx_xvst((__m256i)_negmax_lasx, maxptr, 0);
            maxptr += 8;
        }
#endif // __loongarch_asx
        __m128 _negmax = (__m128)__lsx_vreplfr2vr_s(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            __lsx_vst(_negmax, maxptr, 0);
            maxptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // init sum
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256i _zero_lasx = __lasx_xvldi(0);
        for (; j + 7 < size1; j += 8)
        {
            __lasx_xvst(_zero_lasx, sumptr, 0);
            sumptr += 8;
        }
#endif // __loongarch_asx
        __m128i _zero = __lsx_vldi(0);
        for (; j + 3 < size1; j += 4)
        {
            __lsx_vst(_zero, sumptr, 0);
            sumptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        softmax_bf16s_pack8_lsx(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
        softmax_bf16s_pack4_lsx(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __loongarch_sx
    if (elempack == 1)
    {
        softmax_bf16s_pack1_lsx(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}
