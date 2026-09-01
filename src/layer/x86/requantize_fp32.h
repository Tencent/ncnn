// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__
void requantize_fma(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack);
#endif

static void requantize(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__
    if (ncnn::cpu_support_x86_fma())
    {
        requantize_fma(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, elemcount, elempack);
        return;
    }
#endif

    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("requantize %d %d %d   %d %d", scale_in_data_size, bias_data_size, scale_out_data_size, elemcount, elempack);

    float scale_in = scale_in_data[0];
#if __SSE2__
    __m128 _scale_in0 = _mm_set1_ps(scale_in);
#if __AVX__
    __m256 _scale_in_avx = _mm256_set1_ps(scale_in);
#if __AVX512F__
    __m512 _scale_in_avx512 = _mm512_set1_ps(scale_in);
#endif // __AVX512F__
#else  // __AVX__
    __m128 _scale_in1 = _scale_in0;
#endif // __AVX__
    if (scale_in_data_size > 1)
    {
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_in_avx512 = _mm512_loadu_ps((const float*)scale_in_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX__
            _scale_in_avx = _mm256_loadu_ps((const float*)scale_in_data);
#if __AVX512F__
            _scale_in_avx512 = combine8x2_ps(_scale_in_avx, _scale_in_avx);
#endif // __AVX512F__
#else  // __AVX__
            _scale_in0 = _mm_loadu_ps((const float*)scale_in_data);
            _scale_in1 = _mm_loadu_ps((const float*)scale_in_data + 4);
#endif // __AVX__
        }
    }
#endif // __SSE2__

    float scale_out = scale_out_data[0];
#if __SSE2__
    __m128 _scale_out0 = _mm_set1_ps(scale_out);
#if __AVX__
    __m256 _scale_out_avx = _mm256_set1_ps(scale_out);
#if __AVX512F__
    __m512 _scale_out_avx512 = _mm512_set1_ps(scale_out);
#endif // __AVX512F__
#else  // __AVX__
    __m128 _scale_out1 = _scale_out0;
#endif // __AVX__
    if (scale_out_data_size > 1)
    {
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_out_avx512 = _mm512_loadu_ps((const float*)scale_out_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX__
            _scale_out_avx = _mm256_loadu_ps((const float*)scale_out_data);
#if __AVX512F__
            _scale_out_avx512 = combine8x2_ps(_scale_out_avx, _scale_out_avx);
#endif // __AVX512F__
#else  // __AVX__
            _scale_out0 = _mm_loadu_ps((const float*)scale_out_data);
            _scale_out1 = _mm_loadu_ps((const float*)scale_out_data + 4);
#endif // __AVX__
        }
    }
#endif // __SSE2__

    if (bias_data_size == 0)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
        for (; i + 15 < size; i += 16)
        {
#if __AVX512F__
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_mul_ps(_v, _scale_in_avx512);
            _v = activation_avx512(_v, activation_type, activation_params);
            _v = _mm512_mul_ps(_v, _scale_out_avx512);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx512(_v));
#else  // __AVX512F__
            __m256 _v0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            __m256 _v1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(intptr + 8)));
            _v0 = _mm256_mul_ps(_v0, _scale_in_avx);
            _v1 = _mm256_mul_ps(_v1, _scale_in_avx);
            _v0 = activation_avx(_v0, activation_type, activation_params);
            _v1 = activation_avx(_v1, activation_type, activation_params);
            _v0 = _mm256_mul_ps(_v0, _scale_out_avx);
            _v1 = _mm256_mul_ps(_v1, _scale_out_avx);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx(_v0, _v1));
#endif // __AVX512F__
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_mul_ps(_v, _scale_in_avx);
            _v = activation_avx(_v, activation_type, activation_params);
            _v = _mm256_mul_ps(_v, _scale_out_avx);
            *(int64_t*)ptr = float2int8_avx(_v);
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_mul_ps(_v0, _scale_in0);
            _v1 = _mm_mul_ps(_v1, _scale_in1);
            _v0 = activation_sse(_v0, activation_type, activation_params);
            _v1 = activation_sse(_v1, activation_type, activation_params);
            _v0 = _mm_mul_ps(_v0, _scale_out0);
            _v1 = _mm_mul_ps(_v1, _scale_out1);
            *(int64_t*)ptr = float2int8_sse(_v0, _v1);
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_mul_ps(_v, _scale_in0);
            _v = activation_sse(_v, activation_type, activation_params);
            _v = _mm_mul_ps(_v, _scale_out0);
            int32_t v = float2int8_sse(_v);
            ptr[0] = (v >> 0) & 0xff;
            ptr[1] = (v >> 8) & 0xff;
            ptr[2] = (v >> 16) & 0xff;
            ptr[3] = (v >> 24) & 0xff;
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = *intptr * scale_in;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
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
        }
#endif // __SSE2__

        int i = 0;
#if __SSE2__
#if __AVX__
        for (; i + 15 < size; i += 16)
        {
#if __AVX512F__
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_fmadd_ps(_v, _scale_in_avx512, _bias_avx512);
            _v = activation_avx512(_v, activation_type, activation_params);
            _v = _mm512_mul_ps(_v, _scale_out_avx512);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx512(_v));
#else  // __AVX512F__
            __m256 _v0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            __m256 _v1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(intptr + 8)));
            _v0 = _mm256_comp_fmadd_ps(_v0, _scale_in_avx, _bias_avx);
            _v1 = _mm256_comp_fmadd_ps(_v1, _scale_in_avx, _bias_avx);
            _v0 = activation_avx(_v0, activation_type, activation_params);
            _v1 = activation_avx(_v1, activation_type, activation_params);
            _v0 = _mm256_mul_ps(_v0, _scale_out_avx);
            _v1 = _mm256_mul_ps(_v1, _scale_out_avx);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx(_v0, _v1));
#endif // __AVX512F__
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_comp_fmadd_ps(_v, _scale_in_avx, _bias_avx);
            _v = activation_avx(_v, activation_type, activation_params);
            _v = _mm256_mul_ps(_v, _scale_out_avx);
            *(int64_t*)ptr = float2int8_avx(_v);
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_comp_fmadd_ps(_v0, _scale_in0, _bias0);
            _v1 = _mm_comp_fmadd_ps(_v1, _scale_in1, _bias1);
            _v0 = activation_sse(_v0, activation_type, activation_params);
            _v1 = activation_sse(_v1, activation_type, activation_params);
            _v0 = _mm_mul_ps(_v0, _scale_out0);
            _v1 = _mm_mul_ps(_v1, _scale_out1);
            *(int64_t*)ptr = float2int8_sse(_v0, _v1);
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_comp_fmadd_ps(_v, _scale_in0, _bias0);
            _v = activation_sse(_v, activation_type, activation_params);
            _v = _mm_mul_ps(_v, _scale_out0);
            int32_t v = float2int8_sse(_v);
            ptr[0] = (v >> 0) & 0xff;
            ptr[1] = (v >> 8) & 0xff;
            ptr[2] = (v >> 16) & 0xff;
            ptr[3] = (v >> 24) & 0xff;
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = *intptr * scale_in + bias;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
}
