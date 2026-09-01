// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void dequantize_fma(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void dequantize_fma4(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack);
#endif

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        dequantize_fma(intptr, ptr, scale_data, bias_data, elemcount, elempack);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        dequantize_fma4(intptr, ptr, scale_data, bias_data, elemcount, elempack);
        return;
    }
#endif

    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __SSE2__
    __m128 _scale0 = _mm_set1_ps(scale);
#if __AVX__
    __m256 _scale_avx = _mm256_set1_ps(scale);
#if __AVX512F__
    __m512 _scale_avx512 = _mm512_set1_ps(scale);
#endif // __AVX512F__
#else  // __AVX__
    __m128 _scale1 = _scale0;
#endif // __AVX__
    if (scale_data_size > 1)
    {
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_avx512 = _mm512_loadu_ps((const float*)scale_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX__
            _scale_avx = _mm256_loadu_ps((const float*)scale_data);
#if __AVX512F__
            _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
#endif // __AVX512F__
#else  // __AVX__
            _scale0 = _mm_loadu_ps((const float*)scale_data);
            _scale1 = _mm_loadu_ps((const float*)scale_data + 4);
#endif // __AVX__
        }
        if (elempack == 4)
        {
            _scale0 = _mm_loadu_ps((const float*)scale_data);
#if __AVX__
            _scale_avx = combine4x2_ps(_scale0, _scale0);
#if __AVX512F__
            _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
#endif // __AVX512F__
#else  // __AVX__
            _scale1 = _scale0;
#endif // __AVX__
        }
    }
#endif // __SSE2__

    if (bias_data_size == 0)
    {
        int i = 0;
#if __SSE2__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_mul_ps(_v, _scale_avx512);
            _mm512_storeu_ps(ptr, _v);
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_mul_ps(_v, _scale_avx);
            _mm256_storeu_ps(ptr, _v);
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_mul_ps(_v0, _scale0);
            _v1 = _mm_mul_ps(_v1, _scale1);
            _mm_storeu_ps(ptr, _v0);
            _mm_storeu_ps(ptr + 4, _v1);
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_mul_ps(_v, _scale0);
            _mm_storeu_ps(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __SSE2__
        __m128 _bias0 = _mm_set1_ps(bias);
#if __AVX__
        __m256 _bias_avx = _mm256_set1_ps(bias);
#if __AVX512F__
        __m512 _bias_avx512 = _mm512_set1_ps(bias);
#endif // __AVX512F__
#else  // __AVX__
        __m128 _bias1 = _bias0;
#endif // __AVX__
        if (bias_data_size > 1)
        {
#if __AVX512F__
            if (elempack == 16)
            {
                _bias_avx512 = _mm512_loadu_ps((const float*)bias_data);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
#if __AVX__
                _bias_avx = _mm256_loadu_ps((const float*)bias_data);
#if __AVX512F__
                _bias_avx512 = combine8x2_ps(_bias_avx, _bias_avx);
#endif // __AVX512F__
#else  // __AVX__
                _bias0 = _mm_loadu_ps((const float*)bias_data);
                _bias1 = _mm_loadu_ps((const float*)bias_data + 4);
#endif // __AVX__
            }
            if (elempack == 4)
            {
                _bias0 = _mm_loadu_ps((const float*)bias_data);
#if __AVX__
                _bias_avx = combine4x2_ps(_bias0, _bias0);
#if __AVX512F__
                _bias_avx512 = combine8x2_ps(_bias_avx, _bias_avx);
#endif // __AVX512F__
#else  // __AVX__
                _bias1 = _bias0;
#endif // __AVX__
            }
        }
#endif // __SSE2__

        int i = 0;
#if __SSE2__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_fmadd_ps(_v, _scale_avx512, _bias_avx512);
            _mm512_storeu_ps(ptr, _v);
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_comp_fmadd_ps(_v, _scale_avx, _bias_avx);
            _mm256_storeu_ps(ptr, _v);
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_comp_fmadd_ps(_v0, _scale0, _bias0);
            _v1 = _mm_comp_fmadd_ps(_v1, _scale1, _bias1);
            _mm_storeu_ps(ptr, _v0);
            _mm_storeu_ps(ptr + 4, _v1);
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_comp_fmadd_ps(_v, _scale0, _bias0);
            _mm_storeu_ps(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
    }
}
