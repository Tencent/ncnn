// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void deconvolutiondepthwise_fp32_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void deconvolutiondepthwise_fp32_fma4(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt);
#endif

static void deconvolutiondepthwise_fp32(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        deconvolutiondepthwise_fp32_fma(bottom_blob, top_blob, weight_data, bias_data, bias_term, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        deconvolutiondepthwise_fp32_fma4(bottom_blob, top_blob, weight_data, bias_data, bias_term, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int maxk = kernel_w * kernel_h;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob.channel(g);
                const float* kptr = (const float*)weight_data + maxk * g * 16;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m512 _sum = _mm512_setzero_ps();

                        if (bias_term)
                        {
                            _sum = _mm512_loadu_ps((const float*)bias_data + g * 16);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const float* sptr = m.row(sy) + sx * 16;

                                int k = y * kernel_w + x;

                                __m512 _val = _mm512_loadu_ps(sptr);
                                __m512 _w = _mm512_loadu_ps(kptr + k * 16);
                                _sum = _mm512_fmadd_ps(_val, _w, _sum);
                            }
                        }

                        _sum = activation_avx512(_sum, activation_type, activation_params);

                        _mm512_storeu_ps(outptr, _sum);
                        outptr += 16;
                    }
                }
            }
        }
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob.channel(g);
                const float* kptr = (const float*)weight_data + maxk * g * 8;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_setzero_ps();

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps((const float*)bias_data + g * 8);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const float* sptr = m.row(sy) + sx * 8;

                                int k = y * kernel_w + x;

                                __m256 _val = _mm256_loadu_ps(sptr);
                                __m256 _w = _mm256_loadu_ps(kptr + k * 8);
                                _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
                            }
                        }

                        _sum = activation_avx(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr, _sum);
                        outptr += 8;
                    }
                }
            }
        }
    }
#endif // __AVX__

    if (elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob.channel(g);
                const float* kptr = (const float*)weight_data + maxk * g * 4;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_setzero_ps();

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data + g * 4);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const float* sptr = m.row(sy) + sx * 4;

                                int k = y * kernel_w + x;

                                __m128 _val = _mm_loadu_ps(sptr);
                                __m128 _w = _mm_loadu_ps(kptr + k * 4);
                                _sum = _mm_comp_fmadd_ps(_val, _w, _sum);
                            }
                        }

                        _sum = activation_sse(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr, _sum);
                        outptr += 4;
                    }
                }
            }
        }
    }
#endif // __SSE2__

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < channels; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[g];
                    }

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        const float* sptr = m.row(sy);

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            float val = sptr[sx];

                            int k = y * kernel_w + x;

                            float w = kptr[k];

                            sum += val * w;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr++;
                }
            }
        }
    }
}

