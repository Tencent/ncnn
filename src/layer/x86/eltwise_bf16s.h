// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void eltwise_bf16s_avx512bf16(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt);
#endif

static void eltwise_bf16s(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        eltwise_bf16s_avx512bf16(bottom_blobs, top_blob, op_type, coeffs, opt);
        return;
    }
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    if (op_type == 0) // Operation_PROD
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr1));
                _p = _mm512_mul_ps(_p, _p1);
                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                ptr += 16;
                ptr1 += 16;
                outptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr1));
                _p = _mm256_mul_ps(_p, _p1);
                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr1));
                _p = _mm_mul_ps(_p, _p1);
                _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * bfloat16_to_float32(*ptr1));

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob2.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)outptr));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                    _p = _mm512_mul_ps(_p, _p1);
                    _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                    ptr += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)outptr));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                    _p = _mm256_mul_ps(_p, _p1);
                    _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                    ptr += 8;
                    outptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)outptr));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                    _p = _mm_mul_ps(_p, _p1);
                    _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                    ptr += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*outptr) * bfloat16_to_float32(*ptr));

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == 1) // Operation_SUM
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr1));
                    _p = _mm512_add_ps(_p, _p1);
                    _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                    ptr += 16;
                    ptr1 += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr1));
                    _p = _mm256_add_ps(_p, _p1);
                    _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr1));
                    _p = _mm_add_ps(_p, _p1);
                    _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) + bfloat16_to_float32(*ptr1));

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob2.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; i + 15 < size; i += 16)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)outptr));
                        __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                        _p = _mm512_add_ps(_p, _p1);
                        _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                        ptr += 16;
                        outptr += 16;
                    }
#endif // __AVX512F__
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)outptr));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                        _p = _mm256_add_ps(_p, _p1);
                        _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __AVX__
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)outptr));
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                        _p = _mm_add_ps(_p, _p1);
                        _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __SSE2__
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*outptr) + bfloat16_to_float32(*ptr));

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _coeff0_avx512 = _mm512_set1_ps(coeff0);
                __m512 _coeff1_avx512 = _mm512_set1_ps(coeff1);
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr1));
                    _p = _mm512_mul_ps(_p, _coeff0_avx512);
                    _p = _mm512_fmadd_ps(_p1, _coeff1_avx512, _p);
                    _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                    ptr += 16;
                    ptr1 += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                __m256 _coeff0_avx = _mm256_set1_ps(coeff0);
                __m256 _coeff1_avx = _mm256_set1_ps(coeff1);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr1));
                    _p = _mm256_mul_ps(_p, _coeff0_avx);
                    _p = _mm256_comp_fmadd_ps(_p1, _coeff1_avx, _p);
                    _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __AVX__
                __m128 _coeff0 = _mm_set1_ps(coeff0);
                __m128 _coeff1 = _mm_set1_ps(coeff1);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr1));
                    _p = _mm_mul_ps(_p, _coeff0);
                    _p1 = _mm_mul_ps(_p1, _coeff1);
                    _p = _mm_add_ps(_p1, _p);
                    _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * coeff0 + bfloat16_to_float32(*ptr1) * coeff1);

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob2.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

                    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _coeff_avx512 = _mm512_set1_ps(coeff);
                    for (; i + 15 < size; i += 16)
                    {
                        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)outptr));
                        __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                        _p = _mm512_fmadd_ps(_p1, _coeff_avx512, _p);
                        _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                        ptr += 16;
                        outptr += 16;
                    }
#endif // __AVX512F__
                    __m256 _coeff_avx = _mm256_set1_ps(coeff);
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)outptr));
                        __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                        _p = _mm256_comp_fmadd_ps(_p1, _coeff_avx, _p);
                        _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __AVX__
                    __m128 _coeff = _mm_set1_ps(coeff);
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)outptr));
                        _p1 = _mm_mul_ps(_p1, _coeff);
                        _p = _mm_add_ps(_p1, _p);
                        _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __SSE2__
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*outptr) + bfloat16_to_float32(*ptr) * coeff);

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == 2) // Operation_MAX
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr1));
                _p = _mm512_max_ps(_p, _p1);
                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                ptr += 16;
                ptr1 += 16;
                outptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr1));
                _p = _mm256_max_ps(_p, _p1);
                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr1));
                _p = _mm_max_ps(_p, _p1);
                _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob2.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)outptr));
                    __m512 _p1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                    _p = _mm512_max_ps(_p, _p1);
                    _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_p));

                    ptr += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)outptr));
                    __m256 _p1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                    _p = _mm256_max_ps(_p, _p1);
                    _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_p));

                    ptr += 8;
                    outptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)outptr));
                    __m128 _p1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                    _p = _mm_max_ps(_p, _p1);
                    _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_p, _p));

                    ptr += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*outptr)));

                    ptr++;
                    outptr++;
                }
            }
        }
    }
}
