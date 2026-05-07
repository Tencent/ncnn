// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void softmax_bf16s_sse_avx512bf16(unsigned short* _ptr, int elemcount, int elempack);
void softmax_bf16s_pack1_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr);
void softmax_bf16s_pack4_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr);
void softmax_bf16s_pack8_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr);
void softmax_bf16s_pack16_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr);
#endif

static void softmax_bf16s_sse(unsigned short* _ptr, int elemcount, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_bf16s_sse_avx512bf16(_ptr, elemcount, elempack);
        return;
    }
#endif

    const int size = elemcount * elempack;

    // reduce max
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _max_avx512 = _mm512_set1_ps(-FLT_MAX);
#endif // __AVX512F__
    __m256 _max_avx = _mm256_set1_ps(-FLT_MAX);
#endif // __AVX__
    __m128 _max = _mm_set1_ps(-FLT_MAX);
#endif // __SSE2__
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            _max_avx512 = _mm512_max_ps(_max_avx512, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            _max_avx = _mm256_max_ps(_max_avx, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _max = _mm_max_ps(_max, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr++));
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 8)
    {
        {
            __m256 _max0 = _mm512_castps512_ps256(_max_avx512);
            __m256 _max1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_max_avx512), 1));
            _max_avx = _mm256_max_ps(_max_avx, _max0);
            _max_avx = _mm256_max_ps(_max_avx, _max1);
        }

        _max_avx512 = combine8x2_ps(_max_avx, _max_avx);
    }
#endif // __AVX512F__
    if (elempack == 4)
    {
#if __AVX512F__
        {
            __m256 _max0 = _mm512_castps512_ps256(_max_avx512);
            __m256 _max1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_max_avx512), 1));
            _max_avx = _mm256_max_ps(_max_avx, _max0);
            _max_avx = _mm256_max_ps(_max_avx, _max1);
        }
#endif // __AVX512F__
        {
            __m128 _max0 = _mm256_castps256_ps128(_max_avx);
            __m128 _max1 = _mm256_extractf128_ps(_max_avx, 1);
            _max = _mm_max_ps(_max, _max0);
            _max = _mm_max_ps(_max, _max1);
        }

        _max_avx = combine4x2_ps(_max, _max);
#if __AVX512F__
        _max_avx512 = combine8x2_ps(_max_avx, _max_avx);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 1)
    {
#if __AVX__
#if __AVX512F__
        max = std::max(max, _mm512_comp_reduce_max_ps(_max_avx512));
#endif // __AVX512F__
        max = std::max(max, _mm256_reduce_max_ps(_max_avx));
#endif // __AVX__
        max = std::max(max, _mm_reduce_max_ps(_max));

        _max = _mm_set1_ps(max);
#if __AVX__
        _max_avx = combine4x2_ps(_max, _max);
#if __AVX512F__
        _max_avx512 = combine8x2_ps(_max_avx, _max_avx);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__

    // reduce exp(x - max) and store back to bf16
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
    __m128 _sum = _mm_set1_ps(0.f);
#endif // __SSE2__
    float sum = 0.f;
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            _p = _mm512_sub_ps(_p, _max_avx512);
            _p = exp512_ps(_p);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            _p = _mm256_sub_ps(_p, _max_avx);
            _p = exp256_ps(_p);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            _sum_avx = _mm256_add_ps(_sum_avx, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _p = _mm_sub_ps(_p, _max);
            _p = exp_ps(_p);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            _sum = _mm_add_ps(_sum, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max);
            *ptr = float32_to_bfloat16(v);
            sum += v;
            ptr++;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        _sum_avx512 = _mm512_rcp_nr_ps(_sum_avx512);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
#if __AVX512F__
        {
            __m256 _sum0 = _mm512_castps512_ps256(_sum_avx512);
            __m256 _sum1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_sum_avx512), 1));
            _sum_avx = _mm256_add_ps(_sum_avx, _sum0);
            _sum_avx = _mm256_add_ps(_sum_avx, _sum1);
        }
#endif // __AVX512F__

        _sum_avx = _mm256_rcp_nr_ps(_sum_avx);

#if __AVX512F__
        _sum_avx512 = combine8x2_ps(_sum_avx, _sum_avx);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 4)
    {
#if __AVX__
#if __AVX512F__
        {
            __m256 _sum0 = _mm512_castps512_ps256(_sum_avx512);
            __m256 _sum1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_sum_avx512), 1));
            _sum_avx = _mm256_add_ps(_sum_avx, _sum0);
            _sum_avx = _mm256_add_ps(_sum_avx, _sum1);
        }
#endif // __AVX512F__
        {
            __m128 _sum0 = _mm256_castps256_ps128(_sum_avx);
            __m128 _sum1 = _mm256_extractf128_ps(_sum_avx, 1);
            _sum = _mm_add_ps(_sum, _sum0);
            _sum = _mm_add_ps(_sum, _sum1);
        }
#endif // __AVX__

        _sum = _mm_rcp_nr_ps(_sum);

#if __AVX__
        _sum_avx = combine4x2_ps(_sum, _sum);
#if __AVX512F__
        _sum_avx512 = combine8x2_ps(_sum_avx, _sum_avx);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__
    if (elempack == 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
        sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
        sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

        sum = 1.f / sum;

#if __SSE2__
        _sum = _mm_set1_ps(sum);
#if __AVX__
        _sum_avx = combine4x2_ps(_sum, _sum);
#if __AVX512F__
        _sum_avx512 = combine8x2_ps(_sum_avx, _sum_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    // div sum
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            _p = _mm512_mul_ps(_p, _sum_avx512);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            _p = _mm256_mul_ps(_p, _sum_avx);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _p = _mm_mul_ps(_p, _sum);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * sum);
            ptr++;
        }
    }
}

#if __SSE2__
#if __AVX__
#if __AVX512F__
static void softmax_bf16s_pack16_sse(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_bf16s_pack16_sse_avx512bf16(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
#endif

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            *maxptr = std::max(*maxptr, _mm512_comp_reduce_max_ps(_p));
            ptr += 16;
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
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _max = _mm512_set1_ps(*maxptr);
            _p = exp512_ps(_mm512_sub_ps(_p, _max));
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            *sumptr += _mm512_comp_reduce_add_ps(_p);
            ptr += 16;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
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
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _sum = _mm512_set1_ps(*sumptr);
            _p = _mm512_mul_ps(_p, _sum);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
            sumptr++;
        }
    }
}
#endif // __AVX512F__

static void softmax_bf16s_pack8_sse(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_bf16s_pack8_sse_avx512bf16(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
#endif

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            *maxptr = std::max(*maxptr, _mm256_reduce_max_ps(_p));
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
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _max = _mm256_set1_ps(*maxptr);
            _p = exp256_ps(_mm256_sub_ps(_p, _max));
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            *sumptr += _mm256_reduce_add_ps(_p);
            ptr += 8;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
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
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _sum = _mm256_set1_ps(*sumptr);
            _p = _mm256_mul_ps(_p, _sum);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
            sumptr++;
        }
    }
}
#endif // __AVX__

static void softmax_bf16s_pack4_sse(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_bf16s_pack4_sse_avx512bf16(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
#endif

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j < size1; j++)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            *maxptr = std::max(*maxptr, _mm_reduce_max_ps(_p));
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
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _max = _mm_set1_ps(*maxptr);
            _p = exp_ps(_mm_sub_ps(_p, _max));
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            *sumptr += _mm_reduce_add_ps(_p);
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
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
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _sum = _mm_set1_ps(*sumptr);
            _p = _mm_mul_ps(_p, _sum);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __SSE2__

static void softmax_bf16s_pack1_sse(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_bf16s_pack1_sse_avx512bf16(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
#endif

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _max = _mm512_loadu_ps(maxptr);
            _max = _mm512_max_ps(_max, _p);
            _mm512_storeu_ps(maxptr, _max);
            ptr += 16;
            maxptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _max = _mm256_loadu_ps(maxptr);
            _max = _mm256_max_ps(_max, _p);
            _mm256_storeu_ps(maxptr, _max);
            ptr += 8;
            maxptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _max = _mm_loadu_ps(maxptr);
            _max = _mm_max_ps(_max, _p);
            _mm_storeu_ps(maxptr, _max);
            ptr += 4;
            maxptr += 4;
        }
#endif // __SSE2__
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _max = _mm512_loadu_ps(maxptr);
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _p = _mm512_sub_ps(_p, _max);
            _p = exp512_ps(_p);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            _sum = _mm512_add_ps(_sum, _p);
            _mm512_storeu_ps(sumptr, _sum);
            ptr += 16;
            maxptr += 16;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _max = _mm256_loadu_ps(maxptr);
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _p = _mm256_sub_ps(_p, _max);
            _p = exp256_ps(_p);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            _sum = _mm256_add_ps(_sum, _p);
            _mm256_storeu_ps(sumptr, _sum);
            ptr += 8;
            maxptr += 8;
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _max = _mm_loadu_ps(maxptr);
            __m128 _sum = _mm_loadu_ps(sumptr);
            _p = _mm_sub_ps(_p, _max);
            _p = exp_ps(_p);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            _sum = _mm_add_ps(_sum, _p);
            _mm_storeu_ps(sumptr, _sum);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __SSE2__
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
            sumptr += 4;
        }
#endif // __SSE2__
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _p = _mm512_mul_ps(_p, _sum);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _p = _mm256_mul_ps(_p, _sum);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _sum = _mm_loadu_ps(sumptr);
            _p = _mm_mul_ps(_p, _sum);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
            sumptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * *sumptr);
            ptr++;
            sumptr++;
        }
    }
}

static void softmax_bf16s_sse_dispatch(unsigned short* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // init max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _negmax_avx512 = _mm512_set1_ps(-FLT_MAX);
        for (; j + 15 < size1; j += 16)
        {
            _mm512_storeu_ps(maxptr, _negmax_avx512);
            maxptr += 16;
        }
#endif // __AVX512F__
        __m256 _negmax_avx = _mm256_set1_ps(-FLT_MAX);
        for (; j + 7 < size1; j += 8)
        {
            _mm256_storeu_ps(maxptr, _negmax_avx);
            maxptr += 8;
        }
#endif // __AVX__
        __m128 _negmax = _mm_set1_ps(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            _mm_storeu_ps(maxptr, _negmax);
            maxptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // init sum
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _zero_avx512 = _mm512_set1_ps(0.f);
        for (; j + 15 < size1; j += 16)
        {
            _mm512_storeu_ps(sumptr, _zero_avx512);
            sumptr += 16;
        }
#endif // __AVX512F__
        __m256 _zero_avx = _mm256_set1_ps(0.f);
        for (; j + 7 < size1; j += 8)
        {
            _mm256_storeu_ps(sumptr, _zero_avx);
            sumptr += 8;
        }
#endif // __AVX__
        __m128 _zero = _mm_set1_ps(0.f);
        for (; j + 3 < size1; j += 4)
        {
            _mm_storeu_ps(sumptr, _zero);
            sumptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        softmax_bf16s_pack16_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        softmax_bf16s_pack8_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __AVX__
    if (elempack == 4)
    {
        softmax_bf16s_pack4_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        softmax_bf16s_pack1_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}
