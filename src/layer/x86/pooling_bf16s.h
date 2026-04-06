// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void pooling_global_max_bf16s_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void pooling_global_avg_bf16s_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void pooling_max_bf16s_sse_avx512bf16(const Mat& bottom_blob_bordered, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, const Option& opt);
void pooling_avg_bf16s_sse_avx512bf16(const Mat& bottom_blob_bordered, const Mat& bottom_blob, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_mode, int avgpool_count_include_pad, const Option& opt);
#endif

static void pooling_global_max_bf16s_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pooling_global_max_bf16s_sse_avx512bf16(bottom_blob, top_blob, opt);
        return;
    }
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            __m512 _max = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            ptr += 16;
            for (int i = 1; i < size; i++)
            {
                __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                _max = _mm512_max_ps(_max, _val);
                ptr += 16;
            }

            unsigned short* outptr = top_blob;
            _mm256_storeu_si256((__m256i*)(outptr + q * 16), float2bfloat_avx512(_max));
        }

        return;
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            __m256 _max = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            ptr += 8;
            for (int i = 1; i < size; i++)
            {
                __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                _max = _mm256_max_ps(_max, _val);
                ptr += 8;
            }

            unsigned short* outptr = top_blob;
            _mm_storeu_si128((__m128i*)(outptr + q * 8), float2bfloat_avx(_max));
        }

        return;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            __m128 _max = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            ptr += 4;
            for (int i = 1; i < size; i++)
            {
                __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                _max = _mm_max_ps(_max, _val);
                ptr += 4;
            }

            unsigned short* outptr = top_blob;
            _mm_storel_epi64((__m128i*)(outptr + q * 4), float2bfloat_sse(_max, _max));
        }

        return;
    }
#endif // __SSE2__

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            float max_val = bfloat16_to_float32(ptr[0]);
            for (int i = 1; i < size; i++)
            {
                float val = bfloat16_to_float32(ptr[i]);
                max_val = std::max(max_val, val);
            }

            unsigned short* outptr = top_blob;
            outptr[q] = float32_to_bfloat16(max_val);
        }
    }
}

static void pooling_global_avg_bf16s_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pooling_global_avg_bf16s_sse_avx512bf16(bottom_blob, top_blob, opt);
        return;
    }
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            __m512 _sum = _mm512_set1_ps(0.f);
            for (int i = 0; i < size; i++)
            {
                __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                _sum = _mm512_add_ps(_sum, _val);
                ptr += 16;
            }

            __m512 _inv_size = _mm512_set1_ps(1.f / size);
            __m512 _avg = _mm512_mul_ps(_sum, _inv_size);

            unsigned short* outptr = top_blob;
            _mm256_storeu_si256((__m256i*)(outptr + q * 16), float2bfloat_avx512(_avg));
        }

        return;
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            __m256 _sum = _mm256_set1_ps(0.f);
            for (int i = 0; i < size; i++)
            {
                __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                _sum = _mm256_add_ps(_sum, _val);
                ptr += 8;
            }

            __m256 _inv_size = _mm256_set1_ps(1.f / size);
            __m256 _avg = _mm256_mul_ps(_sum, _inv_size);

            unsigned short* outptr = top_blob;
            _mm_storeu_si128((__m128i*)(outptr + q * 8), float2bfloat_avx(_avg));
        }

        return;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            __m128 _sum = _mm_set1_ps(0.f);
            for (int i = 0; i < size; i++)
            {
                __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                _sum = _mm_add_ps(_sum, _val);
                ptr += 4;
            }

            __m128 _inv_size = _mm_set1_ps(1.f / size);
            __m128 _avg = _mm_mul_ps(_sum, _inv_size);

            unsigned short* outptr = top_blob;
            _mm_storel_epi64((__m128i*)(outptr + q * 4), float2bfloat_sse(_avg, _avg));
        }

        return;
    }
#endif // __SSE2__

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            float sum = 0.f;
            for (int i = 0; i < size; i++)
            {
                sum += bfloat16_to_float32(ptr[i]);
            }

            unsigned short* outptr = top_blob;
            outptr[q] = float32_to_bfloat16(sum / size);
        }
    }
}

static void pooling_max_bf16s_sse(const Mat& bottom_blob_bordered, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pooling_max_bf16s_sse_avx512bf16(bottom_blob_bordered, top_blob, kernel_w, kernel_h, stride_w, stride_h, opt);
        return;
    }
#endif

    int w = bottom_blob_bordered.w;
    int channels = bottom_blob_bordered.c;
    int elempack = bottom_blob_bordered.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 16;

                    __m512 _max = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)sptr));

                    for (int k = 0; k < maxk; k++)
                    {
                        __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + space_ofs[k] * 16)));
                        _max = _mm512_max_ps(_max, _val);
                    }

                    _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_max));
                    outptr += 16;
                }
            }
        }

        return;
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 8;

                    __m256 _max = bfloat2float_avx(_mm_loadu_si128((const __m128i*)sptr));

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + space_ofs[k] * 8)));
                        _max = _mm256_max_ps(_max, _val);
                    }

                    _mm_storeu_si128((__m128i*)(outptr + j * 8), float2bfloat_avx(_max));
                }

                outptr += outw * 8;
            }
        }

        return;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                    __m128 _max = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)sptr));

                    for (int k = 0; k < maxk; k++)
                    {
                        __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + space_ofs[k] * 4)));
                        _max = _mm_max_ps(_max, _val);
                    }

                    _mm_storel_epi64((__m128i*)(outptr + j * 4), float2bfloat_sse(_max, _max));
                }

                outptr += outw * 4;
            }
        }

        return;
    }
#endif // __SSE2__

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w;

                    float max_val = bfloat16_to_float32(sptr[0]);

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = bfloat16_to_float32(sptr[space_ofs[k]]);
                        max_val = std::max(max_val, val);
                    }

                    outptr[j] = float32_to_bfloat16(max_val);
                }

                outptr += outw;
            }
        }
    }
}

static void pooling_avg_bf16s_sse(const Mat& bottom_blob_bordered, const Mat& bottom_blob, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_mode, int avgpool_count_include_pad, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pooling_avg_bf16s_sse_avx512bf16(bottom_blob_bordered, bottom_blob, top_blob, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, pad_mode, avgpool_count_include_pad, opt);
        return;
    }
#endif

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int channels = bottom_blob_bordered.c;
    int elempack = bottom_blob_bordered.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (avgpool_count_include_pad == 0)
    {
        int wtailpad = 0;
        int htailpad = 0;

        if (pad_mode == 0) // full padding
        {
            wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
            htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
        }

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        __m512 _sum = _mm512_set1_ps(0.f);
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(m.row<const unsigned short>(sy) + sx * 16)));
                                _sum = _mm512_add_ps(_sum, _val);
                                area += 1;
                            }
                        }

                        __m512 _inv_area = _mm512_set1_ps(1.f / area);
                        __m512 _avg = _mm512_mul_ps(_sum, _inv_area);
                        _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_avg));
                        outptr += 16;
                    }
                }
            }

            return;
        }
#endif // __AVX512F__

        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        __m256 _sum = _mm256_set1_ps(0.f);
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(m.row<const unsigned short>(sy) + sx * 8)));
                                _sum = _mm256_add_ps(_sum, _val);
                                area += 1;
                            }
                        }

                        __m256 _inv_area = _mm256_set1_ps(1.f / area);
                        __m256 _avg = _mm256_mul_ps(_sum, _inv_area);
                        _mm_storeu_si128((__m128i*)(outptr + j * 8), float2bfloat_avx(_avg));
                    }

                    outptr += outw * 8;
                }
            }

            return;
        }
#endif // __AVX__

        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        __m128 _sum = _mm_set1_ps(0.f);
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(m.row<const unsigned short>(sy) + sx * 4)));
                                _sum = _mm_add_ps(_sum, _val);
                                area += 1;
                            }
                        }

                        __m128 _inv_area = _mm_set1_ps(1.f / area);
                        __m128 _avg = _mm_mul_ps(_sum, _inv_area);
                        _mm_storel_epi64((__m128i*)(outptr + j * 4), float2bfloat_sse(_avg, _avg));
                    }

                    outptr += outw * 4;
                }
            }

            return;
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        float sum = 0.f;
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                sum += bfloat16_to_float32(m.row<const unsigned short>(sy)[sx]);
                                area += 1;
                            }
                        }

                        outptr[j] = float32_to_bfloat16(sum / area);
                    }

                    outptr += outw;
                }
            }
        }
    }
    else // if (avgpool_count_include_pad == 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                __m512 _inv_maxk = _mm512_set1_ps(1.f / maxk);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 16;

                        __m512 _sum = _mm512_set1_ps(0.f);

                        for (int k = 0; k < maxk; k++)
                        {
                            __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + space_ofs[k] * 16)));
                            _sum = _mm512_add_ps(_sum, _val);
                        }

                        __m512 _avg = _mm512_mul_ps(_sum, _inv_maxk);
                        _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_avg));
                        outptr += 16;
                    }
                }
            }

            return;
        }
#endif // __AVX512F__

        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                __m256 _inv_maxk = _mm256_set1_ps(1.f / maxk);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 8;

                        __m256 _sum = _mm256_set1_ps(0.f);

                        for (int k = 0; k < maxk; k++)
                        {
                            __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + space_ofs[k] * 8)));
                            _sum = _mm256_add_ps(_sum, _val);
                        }

                        __m256 _avg = _mm256_mul_ps(_sum, _inv_maxk);
                        _mm_storeu_si128((__m128i*)(outptr + j * 8), float2bfloat_avx(_avg));
                    }

                    outptr += outw * 8;
                }
            }

            return;
        }
#endif // __AVX__

        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                __m128 _inv_maxk = _mm_set1_ps(1.f / maxk);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                        __m128 _sum = _mm_set1_ps(0.f);

                        for (int k = 0; k < maxk; k++)
                        {
                            __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + space_ofs[k] * 4)));
                            _sum = _mm_add_ps(_sum, _val);
                        }

                        __m128 _avg = _mm_mul_ps(_sum, _inv_maxk);
                        _mm_storel_epi64((__m128i*)(outptr + j * 4), float2bfloat_sse(_avg, _avg));
                    }

                    outptr += outw * 4;
                }
            }

            return;
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w;

                        float sum = 0.f;

                        for (int k = 0; k < maxk; k++)
                        {
                            sum += bfloat16_to_float32(sptr[space_ofs[k]]);
                        }

                        outptr[j] = float32_to_bfloat16(sum / maxk);
                    }

                    outptr += outw;
                }
            }
        }
    }
}
