// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void dequantize_forward_bf16s_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Mat& bias_data, int bias_data_size, const Option& opt);
#endif

static void dequantize_bf16(const int* intptr, unsigned short* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

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
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_v));
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_mul_ps(_v, _scale_avx);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_v));
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_mul_ps(_v0, _scale0);
            _v1 = _mm_mul_ps(_v1, _scale1);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_sse(_v0, _v1));
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_mul_ps(_v, _scale0);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_v, _v));
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(*intptr * scale);
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
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_v));
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_comp_fmadd_ps(_v, _scale_avx, _bias_avx);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_v));
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_comp_fmadd_ps(_v0, _scale0, _bias0);
            _v1 = _mm_comp_fmadd_ps(_v1, _scale1, _bias1);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_sse(_v0, _v1));
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_comp_fmadd_ps(_v, _scale0, _bias0);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_v, _v));
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(*intptr * scale + bias);
            intptr++;
            ptr++;
        }
    }
}

static int dequantize_forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Mat& bias_data, int bias_data_size, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        dequantize_forward_bf16s_avx512bf16(bottom_blob, top_blob, scale_data, scale_data_size, bias_data, bias_data_size, opt);
        return 0;
    }
#endif

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const size_t out_elemsize = 2u * elempack;

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            unsigned short* ptr = (unsigned short*)top_blob + i * elempack;

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            dequantize_bf16(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            unsigned short* ptr = top_blob.row<unsigned short>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize_bf16(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            unsigned short* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize_bf16(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}
