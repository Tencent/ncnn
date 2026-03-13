// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void deconvolution_transform_kernel_packed(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb

    // transpose kernel (reverse k order for deconvolution)
    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i = 0; i < num_input * num_output; i++)
        {
            for (int k = 0; k < maxk; k++)
            {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

    // clang-format off
    // *INDENT-OFF*
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (num_output >= 16)
    {
        if (num_input >= 16)
            weight_data_tm.create(16 * 16 * maxk, num_input / 16 + (num_input % 16) / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 16 + (num_output % 16) / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 8)
            weight_data_tm.create(16 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 16 + (num_output % 16) / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 4)
            weight_data_tm.create(16 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 16 + (num_output % 16) / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 2)
            weight_data_tm.create(16 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 16 + (num_output % 16) / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else
            weight_data_tm.create(16 * maxk, num_input, num_output / 16 + (num_output % 16) / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
    }
    else
#endif // __AVX512F__
    if (num_output >= 8)
    {
#if __AVX512F__
        if (num_input >= 16)
            weight_data_tm.create(8 * 16 * maxk, num_input / 16 + (num_input % 16) / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else
#endif // __AVX512F__
        if (num_input >= 8)
            weight_data_tm.create(8 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 4)
            weight_data_tm.create(8 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 2)
            weight_data_tm.create(8 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
        else
            weight_data_tm.create(8 * maxk, num_input, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2);
    }
    else
#endif // __AVX__
    if (num_output >= 4)
    {
#if __AVX__
#if __AVX512F__
        if (num_input >= 16)
            weight_data_tm.create(4 * 16 * maxk, num_input / 16 + (num_input % 16) / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else
#endif // __AVX512F__
        if (num_input >= 8)
            weight_data_tm.create(4 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else
#endif // __AVX__
        if (num_input >= 4)
            weight_data_tm.create(4 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 2)
            weight_data_tm.create(4 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else
            weight_data_tm.create(4 * maxk, num_input, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
    }
    else
#endif // __SSE2__
    if (num_output >= 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (num_input >= 16)
            weight_data_tm.create(2 * 16 * maxk, num_input / 16 + (num_input % 16) / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
#endif // __AVX512F__
        if (num_input >= 8)
            weight_data_tm.create(2 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
#endif // __AVX__
        if (num_input >= 4)
            weight_data_tm.create(2 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
#endif // __SSE2__
        if (num_input >= 2)
            weight_data_tm.create(2 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
            weight_data_tm.create(2 * maxk, num_input, num_output / 2 + num_output % 2);
    }
    else
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (num_input >= 16)
            weight_data_tm.create(16 * maxk, num_input / 16 + (num_input % 16) / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output);
        else
#endif // __AVX512F__
        if (num_input >= 8)
            weight_data_tm.create(8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output);
        else
#endif // __AVX__
        if (num_input >= 4)
            weight_data_tm.create(4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output);
        else
#endif // __SSE2__
        if (num_input >= 2)
            weight_data_tm.create(2 * maxk, num_input / 2 + num_input % 2, num_output);
        else
            weight_data_tm.create(maxk, num_input, num_output);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; q + 15 < num_output; q += 16)
    {
        float* g00 = weight_data_tm.channel(q / 16);

        int p = 0;
        for (; p + 15 < num_input; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 1 < num_input; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 16; j++)
                {
                    const float* k00 = weight_data_r2.channel(q + j).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __AVX512F__
    for (; q + 7 < num_output; q += 8)
    {
#if __AVX512F__
        float* g00 = weight_data_tm.channel(q / 16 + (q % 16) / 8);
#else
        float* g00 = weight_data_tm.channel(q / 8);
#endif

        int p = 0;
#if __AVX512F__
        for (; p + 15 < num_input; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
#endif // __AVX512F__
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 1 < num_input; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 8; j++)
                {
                    const float* k00 = weight_data_r2.channel(q + j).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __AVX__
    for (; q + 3 < num_output; q += 4)
    {
#if __AVX512F__
        float* g00 = weight_data_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4);
#elif __AVX__
        float* g00 = weight_data_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = weight_data_tm.channel(q / 4);
#endif

        int p = 0;
#if __AVX__
#if __AVX512F__
        for (; p + 15 < num_input; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
#endif // __AVX512F__
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
#endif // __AVX__
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 1 < num_input; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const float* k00 = weight_data_r2.channel(q + j).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __SSE2__
    for (; q + 1 < num_output; q += 2)
    {
#if __AVX512F__
        float* g00 = weight_data_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __AVX__
        float* g00 = weight_data_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __SSE2__
        float* g00 = weight_data_tm.channel(q / 4 + (q % 4) / 2);
#else
        float* g00 = weight_data_tm.channel(q / 2);
#endif

        int p = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; p + 15 < num_input; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 16; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
#endif // __AVX512F__
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
#endif // __AVX__
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
#endif // __SSE2__
        for (; p + 1 < num_input; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    const float* k00 = weight_data_r2.channel(q + j).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q < num_output; q++)
    {
#if __AVX512F__
        float* g00 = weight_data_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __AVX__
        float* g00 = weight_data_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __SSE2__
        float* g00 = weight_data_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = weight_data_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; p + 15 < num_input; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 16; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
#endif // __AVX512F__
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
#endif // __AVX__
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
#endif // __SSE2__
        for (; p + 1 < num_input; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k00 = weight_data_r2.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void deconvolution_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int out_elempack = top_blob.elempack;

    const int outch = top_blob.c * out_elempack;

    const size_t M = top_blob.cstep * out_elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

    const float* bias_data_ptr = bias_data;

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    nn_outch = outch / 16;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 16;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_loadu_ps(bias_data_ptr + p);
                }

                const float* kptr = weight_data_tm.channel(p / 16);

                int q = 0;
                for (; q + 15 < inch; q += 16)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 16 * 16;

                            if (elempack == 16)
                            {
                                const float* sptr = bottom_blob.channel(q / 16).row(sy) + sx * 16;

                                __m512 _val0 = _mm512_set1_ps(sptr[0]);
                                __m512 _val1 = _mm512_set1_ps(sptr[1]);
                                __m512 _val2 = _mm512_set1_ps(sptr[2]);
                                __m512 _val3 = _mm512_set1_ps(sptr[3]);
                                __m512 _val4 = _mm512_set1_ps(sptr[4]);
                                __m512 _val5 = _mm512_set1_ps(sptr[5]);
                                __m512 _val6 = _mm512_set1_ps(sptr[6]);
                                __m512 _val7 = _mm512_set1_ps(sptr[7]);
                                __m512 _val8 = _mm512_set1_ps(sptr[8]);
                                __m512 _val9 = _mm512_set1_ps(sptr[9]);
                                __m512 _vala = _mm512_set1_ps(sptr[10]);
                                __m512 _valb = _mm512_set1_ps(sptr[11]);
                                __m512 _valc = _mm512_set1_ps(sptr[12]);
                                __m512 _vald = _mm512_set1_ps(sptr[13]);
                                __m512 _vale = _mm512_set1_ps(sptr[14]);
                                __m512 _valf = _mm512_set1_ps(sptr[15]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val8, _mm512_load_ps(kptr0 + 16 * 8), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val9, _mm512_load_ps(kptr0 + 16 * 9), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vala, _mm512_load_ps(kptr0 + 16 * 10), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valb, _mm512_load_ps(kptr0 + 16 * 11), _sum3);
                                _sum0 = _mm512_fmadd_ps(_valc, _mm512_load_ps(kptr0 + 16 * 12), _sum0);
                                _sum1 = _mm512_fmadd_ps(_vald, _mm512_load_ps(kptr0 + 16 * 13), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vale, _mm512_load_ps(kptr0 + 16 * 14), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valf, _mm512_load_ps(kptr0 + 16 * 15), _sum3);
                            }
                            if (elempack == 8)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                const float* sptr1 = bottom_blob.channel(q / 8 + 1).row(sy) + sx * 8;

                                __m512 _val0 = _mm512_set1_ps(sptr0[0]);
                                __m512 _val1 = _mm512_set1_ps(sptr0[1]);
                                __m512 _val2 = _mm512_set1_ps(sptr0[2]);
                                __m512 _val3 = _mm512_set1_ps(sptr0[3]);
                                __m512 _val4 = _mm512_set1_ps(sptr0[4]);
                                __m512 _val5 = _mm512_set1_ps(sptr0[5]);
                                __m512 _val6 = _mm512_set1_ps(sptr0[6]);
                                __m512 _val7 = _mm512_set1_ps(sptr0[7]);
                                __m512 _val8 = _mm512_set1_ps(sptr1[0]);
                                __m512 _val9 = _mm512_set1_ps(sptr1[1]);
                                __m512 _vala = _mm512_set1_ps(sptr1[2]);
                                __m512 _valb = _mm512_set1_ps(sptr1[3]);
                                __m512 _valc = _mm512_set1_ps(sptr1[4]);
                                __m512 _vald = _mm512_set1_ps(sptr1[5]);
                                __m512 _vale = _mm512_set1_ps(sptr1[6]);
                                __m512 _valf = _mm512_set1_ps(sptr1[7]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val8, _mm512_load_ps(kptr0 + 16 * 8), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val9, _mm512_load_ps(kptr0 + 16 * 9), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vala, _mm512_load_ps(kptr0 + 16 * 10), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valb, _mm512_load_ps(kptr0 + 16 * 11), _sum3);
                                _sum0 = _mm512_fmadd_ps(_valc, _mm512_load_ps(kptr0 + 16 * 12), _sum0);
                                _sum1 = _mm512_fmadd_ps(_vald, _mm512_load_ps(kptr0 + 16 * 13), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vale, _mm512_load_ps(kptr0 + 16 * 14), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valf, _mm512_load_ps(kptr0 + 16 * 15), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                const float* sptr2 = bottom_blob.channel(q / 4 + 2).row(sy) + sx * 4;
                                const float* sptr3 = bottom_blob.channel(q / 4 + 3).row(sy) + sx * 4;

                                __m512 _val0 = _mm512_set1_ps(sptr0[0]);
                                __m512 _val1 = _mm512_set1_ps(sptr0[1]);
                                __m512 _val2 = _mm512_set1_ps(sptr0[2]);
                                __m512 _val3 = _mm512_set1_ps(sptr0[3]);
                                __m512 _val4 = _mm512_set1_ps(sptr1[0]);
                                __m512 _val5 = _mm512_set1_ps(sptr1[1]);
                                __m512 _val6 = _mm512_set1_ps(sptr1[2]);
                                __m512 _val7 = _mm512_set1_ps(sptr1[3]);
                                __m512 _val8 = _mm512_set1_ps(sptr2[0]);
                                __m512 _val9 = _mm512_set1_ps(sptr2[1]);
                                __m512 _vala = _mm512_set1_ps(sptr2[2]);
                                __m512 _valb = _mm512_set1_ps(sptr2[3]);
                                __m512 _valc = _mm512_set1_ps(sptr3[0]);
                                __m512 _vald = _mm512_set1_ps(sptr3[1]);
                                __m512 _vale = _mm512_set1_ps(sptr3[2]);
                                __m512 _valf = _mm512_set1_ps(sptr3[3]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val8, _mm512_load_ps(kptr0 + 16 * 8), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val9, _mm512_load_ps(kptr0 + 16 * 9), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vala, _mm512_load_ps(kptr0 + 16 * 10), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valb, _mm512_load_ps(kptr0 + 16 * 11), _sum3);
                                _sum0 = _mm512_fmadd_ps(_valc, _mm512_load_ps(kptr0 + 16 * 12), _sum0);
                                _sum1 = _mm512_fmadd_ps(_vald, _mm512_load_ps(kptr0 + 16 * 13), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vale, _mm512_load_ps(kptr0 + 16 * 14), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valf, _mm512_load_ps(kptr0 + 16 * 15), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m512 _val0 = _mm512_set1_ps(bottom_blob.channel(q).row(sy)[sx]);
                                __m512 _val1 = _mm512_set1_ps(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m512 _val2 = _mm512_set1_ps(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m512 _val3 = _mm512_set1_ps(bottom_blob.channel(q + 3).row(sy)[sx]);
                                __m512 _val4 = _mm512_set1_ps(bottom_blob.channel(q + 4).row(sy)[sx]);
                                __m512 _val5 = _mm512_set1_ps(bottom_blob.channel(q + 5).row(sy)[sx]);
                                __m512 _val6 = _mm512_set1_ps(bottom_blob.channel(q + 6).row(sy)[sx]);
                                __m512 _val7 = _mm512_set1_ps(bottom_blob.channel(q + 7).row(sy)[sx]);
                                __m512 _val8 = _mm512_set1_ps(bottom_blob.channel(q + 8).row(sy)[sx]);
                                __m512 _val9 = _mm512_set1_ps(bottom_blob.channel(q + 9).row(sy)[sx]);
                                __m512 _vala = _mm512_set1_ps(bottom_blob.channel(q + 10).row(sy)[sx]);
                                __m512 _valb = _mm512_set1_ps(bottom_blob.channel(q + 11).row(sy)[sx]);
                                __m512 _valc = _mm512_set1_ps(bottom_blob.channel(q + 12).row(sy)[sx]);
                                __m512 _vald = _mm512_set1_ps(bottom_blob.channel(q + 13).row(sy)[sx]);
                                __m512 _vale = _mm512_set1_ps(bottom_blob.channel(q + 14).row(sy)[sx]);
                                __m512 _valf = _mm512_set1_ps(bottom_blob.channel(q + 15).row(sy)[sx]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val8, _mm512_load_ps(kptr0 + 16 * 8), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val9, _mm512_load_ps(kptr0 + 16 * 9), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vala, _mm512_load_ps(kptr0 + 16 * 10), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valb, _mm512_load_ps(kptr0 + 16 * 11), _sum3);
                                _sum0 = _mm512_fmadd_ps(_valc, _mm512_load_ps(kptr0 + 16 * 12), _sum0);
                                _sum1 = _mm512_fmadd_ps(_vald, _mm512_load_ps(kptr0 + 16 * 13), _sum1);
                                _sum2 = _mm512_fmadd_ps(_vale, _mm512_load_ps(kptr0 + 16 * 14), _sum2);
                                _sum3 = _mm512_fmadd_ps(_valf, _mm512_load_ps(kptr0 + 16 * 15), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 16 * 16;
                }
                for (; q + 7 < inch; q += 8)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 8 * 16;

                            if (elempack == 8)
                            {
                                const float* sptr = bottom_blob.channel(q / 8).row(sy) + sx * 8;

                                __m512 _val0 = _mm512_set1_ps(sptr[0]);
                                __m512 _val1 = _mm512_set1_ps(sptr[1]);
                                __m512 _val2 = _mm512_set1_ps(sptr[2]);
                                __m512 _val3 = _mm512_set1_ps(sptr[3]);
                                __m512 _val4 = _mm512_set1_ps(sptr[4]);
                                __m512 _val5 = _mm512_set1_ps(sptr[5]);
                                __m512 _val6 = _mm512_set1_ps(sptr[6]);
                                __m512 _val7 = _mm512_set1_ps(sptr[7]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;

                                __m512 _val0 = _mm512_set1_ps(sptr0[0]);
                                __m512 _val1 = _mm512_set1_ps(sptr0[1]);
                                __m512 _val2 = _mm512_set1_ps(sptr0[2]);
                                __m512 _val3 = _mm512_set1_ps(sptr0[3]);
                                __m512 _val4 = _mm512_set1_ps(sptr1[0]);
                                __m512 _val5 = _mm512_set1_ps(sptr1[1]);
                                __m512 _val6 = _mm512_set1_ps(sptr1[2]);
                                __m512 _val7 = _mm512_set1_ps(sptr1[3]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m512 _val0 = _mm512_set1_ps(bottom_blob.channel(q).row(sy)[sx]);
                                __m512 _val1 = _mm512_set1_ps(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m512 _val2 = _mm512_set1_ps(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m512 _val3 = _mm512_set1_ps(bottom_blob.channel(q + 3).row(sy)[sx]);
                                __m512 _val4 = _mm512_set1_ps(bottom_blob.channel(q + 4).row(sy)[sx]);
                                __m512 _val5 = _mm512_set1_ps(bottom_blob.channel(q + 5).row(sy)[sx]);
                                __m512 _val6 = _mm512_set1_ps(bottom_blob.channel(q + 6).row(sy)[sx]);
                                __m512 _val7 = _mm512_set1_ps(bottom_blob.channel(q + 7).row(sy)[sx]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                                _sum0 = _mm512_fmadd_ps(_val4, _mm512_load_ps(kptr0 + 16 * 4), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val5, _mm512_load_ps(kptr0 + 16 * 5), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val6, _mm512_load_ps(kptr0 + 16 * 6), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val7, _mm512_load_ps(kptr0 + 16 * 7), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 8 * 16;
                }
                for (; q + 3 < inch; q += 4)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 4 * 16;

                            if (elempack == 4)
                            {
                                const float* sptr = bottom_blob.channel(q / 4).row(sy) + sx * 4;

                                __m512 _val0 = _mm512_set1_ps(sptr[0]);
                                __m512 _val1 = _mm512_set1_ps(sptr[1]);
                                __m512 _val2 = _mm512_set1_ps(sptr[2]);
                                __m512 _val3 = _mm512_set1_ps(sptr[3]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m512 _val0 = _mm512_set1_ps(bottom_blob.channel(q).row(sy)[sx]);
                                __m512 _val1 = _mm512_set1_ps(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m512 _val2 = _mm512_set1_ps(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m512 _val3 = _mm512_set1_ps(bottom_blob.channel(q + 3).row(sy)[sx]);
                                _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                                _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                                _sum2 = _mm512_fmadd_ps(_val2, _mm512_load_ps(kptr0 + 16 * 2), _sum2);
                                _sum3 = _mm512_fmadd_ps(_val3, _mm512_load_ps(kptr0 + 16 * 3), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 4 * 16;
                }
                for (; q + 1 < inch; q += 2)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 2 * 16;

                            const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                            const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;
                            __m512 _val0 = _mm512_set1_ps(sptr0[0]);
                            __m512 _val1 = _mm512_set1_ps(sptr1[0]);
                            _sum0 = _mm512_fmadd_ps(_val0, _mm512_load_ps(kptr0), _sum0);
                            _sum1 = _mm512_fmadd_ps(_val1, _mm512_load_ps(kptr0 + 16), _sum1);
                        }
                    }

                    kptr += maxk * 2 * 16;
                }
                for (; q < inch; q++)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 16;

                            const float* sptr = bottom_blob.channel(q).row(sy) + sx;
                            __m512 _val = _mm512_set1_ps(sptr[0]);
                            _sum0 = _mm512_fmadd_ps(_val, _mm512_load_ps(kptr0), _sum0);
                        }
                    }

                    kptr += maxk * 16;
                }

                _sum0 = _mm512_add_ps(_sum0, _sum1);
                _sum2 = _mm512_add_ps(_sum2, _sum3);
                _sum0 = _mm512_add_ps(_sum0, _sum2);

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);

                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr, _sum0);
                    outptr += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_store_ps(outptr + M, _mm512_extractf32x8_ps(_sum0, 1));
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr + M, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr + M * 2, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr + M * 3, _mm512_extractf32x4_ps(_sum0, 3));
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[16];
                    _mm512_storeu_ps(sum, _sum0);

                    outptr[0] = sum[0];
                    outptr[M] = sum[1];
                    outptr[M * 2] = sum[2];
                    outptr[M * 3] = sum[3];
                    outptr[M * 4] = sum[4];
                    outptr[M * 5] = sum[5];
                    outptr[M * 6] = sum[6];
                    outptr[M * 7] = sum[7];
                    outptr[M * 8] = sum[8];
                    outptr[M * 9] = sum[9];
                    outptr[M * 10] = sum[10];
                    outptr[M * 11] = sum[11];
                    outptr[M * 12] = sum[12];
                    outptr[M * 13] = sum[13];
                    outptr[M * 14] = sum[14];
                    outptr[M * 15] = sum[15];
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 16;
    nn_outch = (outch - remain_outch_start) / 8;
#else // __AVX512F__
    nn_outch = (outch - remain_outch_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX512F__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 8;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_loadu_ps(bias_data_ptr + p);
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
                const float* kptr = weight_data_tm.channel(p / 8);
#endif

                int q = 0;
#if __AVX512F__
                for (; q + 15 < inch; q += 16)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 16 * 8;

                            if (elempack == 16)
                            {
                                const float* sptr = bottom_blob.channel(q / 16).row(sy) + sx * 16;

                                __m256 _val0 = _mm256_broadcast_ss(sptr);
                                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
                                __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
                                __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
                                __m256 _val7 = _mm256_broadcast_ss(sptr + 7);
                                __m256 _val8 = _mm256_broadcast_ss(sptr + 8);
                                __m256 _val9 = _mm256_broadcast_ss(sptr + 9);
                                __m256 _vala = _mm256_broadcast_ss(sptr + 10);
                                __m256 _valb = _mm256_broadcast_ss(sptr + 11);
                                __m256 _valc = _mm256_broadcast_ss(sptr + 12);
                                __m256 _vald = _mm256_broadcast_ss(sptr + 13);
                                __m256 _vale = _mm256_broadcast_ss(sptr + 14);
                                __m256 _valf = _mm256_broadcast_ss(sptr + 15);
                                _sum0 = _mm256_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val8, _mm256_load_ps(kptr0 + 8 * 8), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val9, _mm256_load_ps(kptr0 + 8 * 9), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vala, _mm256_load_ps(kptr0 + 8 * 10), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valb, _mm256_load_ps(kptr0 + 8 * 11), _sum3);
                                _sum0 = _mm256_fmadd_ps(_valc, _mm256_load_ps(kptr0 + 8 * 12), _sum0);
                                _sum1 = _mm256_fmadd_ps(_vald, _mm256_load_ps(kptr0 + 8 * 13), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vale, _mm256_load_ps(kptr0 + 8 * 14), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valf, _mm256_load_ps(kptr0 + 8 * 15), _sum3);
                            }
                            if (elempack == 8)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                const float* sptr1 = bottom_blob.channel(q / 8 + 1).row(sy) + sx * 8;

                                __m256 _val0 = _mm256_broadcast_ss(sptr0);
                                __m256 _val1 = _mm256_broadcast_ss(sptr0 + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr0 + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr0 + 3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr0 + 4);
                                __m256 _val5 = _mm256_broadcast_ss(sptr0 + 5);
                                __m256 _val6 = _mm256_broadcast_ss(sptr0 + 6);
                                __m256 _val7 = _mm256_broadcast_ss(sptr0 + 7);
                                __m256 _val8 = _mm256_broadcast_ss(sptr1);
                                __m256 _val9 = _mm256_broadcast_ss(sptr1 + 1);
                                __m256 _vala = _mm256_broadcast_ss(sptr1 + 2);
                                __m256 _valb = _mm256_broadcast_ss(sptr1 + 3);
                                __m256 _valc = _mm256_broadcast_ss(sptr1 + 4);
                                __m256 _vald = _mm256_broadcast_ss(sptr1 + 5);
                                __m256 _vale = _mm256_broadcast_ss(sptr1 + 6);
                                __m256 _valf = _mm256_broadcast_ss(sptr1 + 7);
                                _sum0 = _mm256_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val8, _mm256_load_ps(kptr0 + 8 * 8), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val9, _mm256_load_ps(kptr0 + 8 * 9), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vala, _mm256_load_ps(kptr0 + 8 * 10), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valb, _mm256_load_ps(kptr0 + 8 * 11), _sum3);
                                _sum0 = _mm256_fmadd_ps(_valc, _mm256_load_ps(kptr0 + 8 * 12), _sum0);
                                _sum1 = _mm256_fmadd_ps(_vald, _mm256_load_ps(kptr0 + 8 * 13), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vale, _mm256_load_ps(kptr0 + 8 * 14), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valf, _mm256_load_ps(kptr0 + 8 * 15), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                const float* sptr2 = bottom_blob.channel(q / 4 + 2).row(sy) + sx * 4;
                                const float* sptr3 = bottom_blob.channel(q / 4 + 3).row(sy) + sx * 4;

                                __m256 _val0 = _mm256_broadcast_ss(sptr0);
                                __m256 _val1 = _mm256_broadcast_ss(sptr0 + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr0 + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr0 + 3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr1);
                                __m256 _val5 = _mm256_broadcast_ss(sptr1 + 1);
                                __m256 _val6 = _mm256_broadcast_ss(sptr1 + 2);
                                __m256 _val7 = _mm256_broadcast_ss(sptr1 + 3);
                                __m256 _val8 = _mm256_broadcast_ss(sptr2);
                                __m256 _val9 = _mm256_broadcast_ss(sptr2 + 1);
                                __m256 _vala = _mm256_broadcast_ss(sptr2 + 2);
                                __m256 _valb = _mm256_broadcast_ss(sptr2 + 3);
                                __m256 _valc = _mm256_broadcast_ss(sptr3);
                                __m256 _vald = _mm256_broadcast_ss(sptr3 + 1);
                                __m256 _vale = _mm256_broadcast_ss(sptr3 + 2);
                                __m256 _valf = _mm256_broadcast_ss(sptr3 + 3);
                                _sum0 = _mm256_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val8, _mm256_load_ps(kptr0 + 8 * 8), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val9, _mm256_load_ps(kptr0 + 8 * 9), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vala, _mm256_load_ps(kptr0 + 8 * 10), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valb, _mm256_load_ps(kptr0 + 8 * 11), _sum3);
                                _sum0 = _mm256_fmadd_ps(_valc, _mm256_load_ps(kptr0 + 8 * 12), _sum0);
                                _sum1 = _mm256_fmadd_ps(_vald, _mm256_load_ps(kptr0 + 8 * 13), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vale, _mm256_load_ps(kptr0 + 8 * 14), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valf, _mm256_load_ps(kptr0 + 8 * 15), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m256 _val0 = _mm256_broadcast_ss(bottom_blob.channel(q).row(sy) + sx);
                                __m256 _val1 = _mm256_broadcast_ss(bottom_blob.channel(q + 1).row(sy) + sx);
                                __m256 _val2 = _mm256_broadcast_ss(bottom_blob.channel(q + 2).row(sy) + sx);
                                __m256 _val3 = _mm256_broadcast_ss(bottom_blob.channel(q + 3).row(sy) + sx);
                                __m256 _val4 = _mm256_broadcast_ss(bottom_blob.channel(q + 4).row(sy) + sx);
                                __m256 _val5 = _mm256_broadcast_ss(bottom_blob.channel(q + 5).row(sy) + sx);
                                __m256 _val6 = _mm256_broadcast_ss(bottom_blob.channel(q + 6).row(sy) + sx);
                                __m256 _val7 = _mm256_broadcast_ss(bottom_blob.channel(q + 7).row(sy) + sx);
                                __m256 _val8 = _mm256_broadcast_ss(bottom_blob.channel(q + 8).row(sy) + sx);
                                __m256 _val9 = _mm256_broadcast_ss(bottom_blob.channel(q + 9).row(sy) + sx);
                                __m256 _vala = _mm256_broadcast_ss(bottom_blob.channel(q + 10).row(sy) + sx);
                                __m256 _valb = _mm256_broadcast_ss(bottom_blob.channel(q + 11).row(sy) + sx);
                                __m256 _valc = _mm256_broadcast_ss(bottom_blob.channel(q + 12).row(sy) + sx);
                                __m256 _vald = _mm256_broadcast_ss(bottom_blob.channel(q + 13).row(sy) + sx);
                                __m256 _vale = _mm256_broadcast_ss(bottom_blob.channel(q + 14).row(sy) + sx);
                                __m256 _valf = _mm256_broadcast_ss(bottom_blob.channel(q + 15).row(sy) + sx);
                                _sum0 = _mm256_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                                _sum0 = _mm256_fmadd_ps(_val8, _mm256_load_ps(kptr0 + 8 * 8), _sum0);
                                _sum1 = _mm256_fmadd_ps(_val9, _mm256_load_ps(kptr0 + 8 * 9), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vala, _mm256_load_ps(kptr0 + 8 * 10), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valb, _mm256_load_ps(kptr0 + 8 * 11), _sum3);
                                _sum0 = _mm256_fmadd_ps(_valc, _mm256_load_ps(kptr0 + 8 * 12), _sum0);
                                _sum1 = _mm256_fmadd_ps(_vald, _mm256_load_ps(kptr0 + 8 * 13), _sum1);
                                _sum2 = _mm256_fmadd_ps(_vale, _mm256_load_ps(kptr0 + 8 * 14), _sum2);
                                _sum3 = _mm256_fmadd_ps(_valf, _mm256_load_ps(kptr0 + 8 * 15), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 16 * 8;
                }
#endif // __AVX512F__
                for (; q + 7 < inch; q += 8)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 8 * 8;

                            if (elempack == 8)
                            {
                                const float* sptr = bottom_blob.channel(q / 8).row(sy) + sx * 8;

                                __m256 _val0 = _mm256_broadcast_ss(sptr);
                                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
                                __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
                                __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
                                __m256 _val7 = _mm256_broadcast_ss(sptr + 7);
                                _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_comp_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;

                                __m256 _val0 = _mm256_broadcast_ss(sptr0);
                                __m256 _val1 = _mm256_broadcast_ss(sptr0 + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr0 + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr0 + 3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr1);
                                __m256 _val5 = _mm256_broadcast_ss(sptr1 + 1);
                                __m256 _val6 = _mm256_broadcast_ss(sptr1 + 2);
                                __m256 _val7 = _mm256_broadcast_ss(sptr1 + 3);
                                _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_comp_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                            }
                            if (elempack == 1)
                            {
                                const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                                const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;
                                const float* sptr2 = bottom_blob.channel(q + 2).row(sy) + sx;
                                const float* sptr3 = bottom_blob.channel(q + 3).row(sy) + sx;
                                const float* sptr4 = bottom_blob.channel(q + 4).row(sy) + sx;
                                const float* sptr5 = bottom_blob.channel(q + 5).row(sy) + sx;
                                const float* sptr6 = bottom_blob.channel(q + 6).row(sy) + sx;
                                const float* sptr7 = bottom_blob.channel(q + 7).row(sy) + sx;
                                __m256 _val0 = _mm256_broadcast_ss(sptr0);
                                __m256 _val1 = _mm256_broadcast_ss(sptr1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr3);
                                __m256 _val4 = _mm256_broadcast_ss(sptr4);
                                __m256 _val5 = _mm256_broadcast_ss(sptr5);
                                __m256 _val6 = _mm256_broadcast_ss(sptr6);
                                __m256 _val7 = _mm256_broadcast_ss(sptr7);
                                _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                                _sum0 = _mm256_comp_fmadd_ps(_val4, _mm256_load_ps(kptr0 + 8 * 4), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val5, _mm256_load_ps(kptr0 + 8 * 5), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val6, _mm256_load_ps(kptr0 + 8 * 6), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val7, _mm256_load_ps(kptr0 + 8 * 7), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 8 * 8;
                }
                for (; q + 3 < inch; q += 4)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 4 * 8;

                            if (elempack == 4)
                            {
                                const float* sptr = bottom_blob.channel(q / 4).row(sy) + sx * 4;

                                __m256 _val0 = _mm256_broadcast_ss(sptr);
                                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
                                _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                            }
                            if (elempack == 1)
                            {
                                const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                                const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;
                                const float* sptr2 = bottom_blob.channel(q + 2).row(sy) + sx;
                                const float* sptr3 = bottom_blob.channel(q + 3).row(sy) + sx;
                                __m256 _val0 = _mm256_broadcast_ss(sptr0);
                                __m256 _val1 = _mm256_broadcast_ss(sptr1);
                                __m256 _val2 = _mm256_broadcast_ss(sptr2);
                                __m256 _val3 = _mm256_broadcast_ss(sptr3);
                                _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_load_ps(kptr0 + 8 * 2), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_load_ps(kptr0 + 8 * 3), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 4 * 8;
                }
                for (; q + 1 < inch; q += 2)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 2 * 8;

                            const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                            const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;
                            __m256 _val0 = _mm256_broadcast_ss(sptr0);
                            __m256 _val1 = _mm256_broadcast_ss(sptr1);
                            _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_load_ps(kptr0), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_load_ps(kptr0 + 8), _sum1);
                        }
                    }

                    kptr += maxk * 2 * 8;
                }
                for (; q < inch; q++)
                {
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

                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 8;

                            const float* sptr = bottom_blob.channel(q).row(sy) + sx;
                            __m256 _val = _mm256_broadcast_ss(sptr);
                            _sum0 = _mm256_comp_fmadd_ps(_val, _mm256_load_ps(kptr0), _sum0);
                        }
                    }

                    kptr += maxk * 8;
                }

                _sum0 = _mm256_add_ps(_sum0, _sum1);
                _sum2 = _mm256_add_ps(_sum2, _sum3);
                _sum0 = _mm256_add_ps(_sum0, _sum2);

                _sum0 = activation_avx(_sum0, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr, _sum0);
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr, _mm256_extractf128_ps(_sum0, 0));
                    _mm_store_ps(outptr + M, _mm256_extractf128_ps(_sum0, 1));
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[8];
                    _mm256_storeu_ps(sum, _sum0);

                    outptr[0] = sum[0];
                    outptr[M] = sum[1];
                    outptr[M * 2] = sum[2];
                    outptr[M * 3] = sum[3];
                    outptr[M * 4] = sum[4];
                    outptr[M * 5] = sum[5];
                    outptr[M * 6] = sum[6];
                    outptr[M * 7] = sum[7];
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
#else // __AVX__
    nn_outch = (outch - remain_outch_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm_loadu_ps(bias_data_ptr + p);
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4);
#elif __AVX__
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
                const float* kptr = weight_data_tm.channel(p / 4);
#endif

                int q = 0;
#if __AVX__
#if __AVX512F__
                for (; q + 15 < inch; q += 16)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 16 * 4;

                            if (elempack == 16)
                            {
                                const float* sptr = bottom_blob.channel(q / 16).row(sy) + sx * 16;
                                __m128 _val0 = _mm_set1_ps(sptr[0]);
                                __m128 _val1 = _mm_set1_ps(sptr[1]);
                                __m128 _val2 = _mm_set1_ps(sptr[2]);
                                __m128 _val3 = _mm_set1_ps(sptr[3]);
                                __m128 _val4 = _mm_set1_ps(sptr[4]);
                                __m128 _val5 = _mm_set1_ps(sptr[5]);
                                __m128 _val6 = _mm_set1_ps(sptr[6]);
                                __m128 _val7 = _mm_set1_ps(sptr[7]);
                                __m128 _val8 = _mm_set1_ps(sptr[8]);
                                __m128 _val9 = _mm_set1_ps(sptr[9]);
                                __m128 _vala = _mm_set1_ps(sptr[10]);
                                __m128 _valb = _mm_set1_ps(sptr[11]);
                                __m128 _valc = _mm_set1_ps(sptr[12]);
                                __m128 _vald = _mm_set1_ps(sptr[13]);
                                __m128 _vale = _mm_set1_ps(sptr[14]);
                                __m128 _valf = _mm_set1_ps(sptr[15]);
                                _sum0 = _mm_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                                _sum0 = _mm_fmadd_ps(_val8, _mm_load_ps(kptr0 + 4 * 8), _sum0);
                                _sum1 = _mm_fmadd_ps(_val9, _mm_load_ps(kptr0 + 4 * 9), _sum1);
                                _sum2 = _mm_fmadd_ps(_vala, _mm_load_ps(kptr0 + 4 * 10), _sum2);
                                _sum3 = _mm_fmadd_ps(_valb, _mm_load_ps(kptr0 + 4 * 11), _sum3);
                                _sum0 = _mm_fmadd_ps(_valc, _mm_load_ps(kptr0 + 4 * 12), _sum0);
                                _sum1 = _mm_fmadd_ps(_vald, _mm_load_ps(kptr0 + 4 * 13), _sum1);
                                _sum2 = _mm_fmadd_ps(_vale, _mm_load_ps(kptr0 + 4 * 14), _sum2);
                                _sum3 = _mm_fmadd_ps(_valf, _mm_load_ps(kptr0 + 4 * 15), _sum3);
                            }
                            if (elempack == 8)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                const float* sptr1 = bottom_blob.channel(q / 8 + 1).row(sy) + sx * 8;
                                __m128 _val0 = _mm_set1_ps(sptr0[0]);
                                __m128 _val1 = _mm_set1_ps(sptr0[1]);
                                __m128 _val2 = _mm_set1_ps(sptr0[2]);
                                __m128 _val3 = _mm_set1_ps(sptr0[3]);
                                __m128 _val4 = _mm_set1_ps(sptr0[4]);
                                __m128 _val5 = _mm_set1_ps(sptr0[5]);
                                __m128 _val6 = _mm_set1_ps(sptr0[6]);
                                __m128 _val7 = _mm_set1_ps(sptr0[7]);
                                __m128 _val8 = _mm_set1_ps(sptr1[0]);
                                __m128 _val9 = _mm_set1_ps(sptr1[1]);
                                __m128 _vala = _mm_set1_ps(sptr1[2]);
                                __m128 _valb = _mm_set1_ps(sptr1[3]);
                                __m128 _valc = _mm_set1_ps(sptr1[4]);
                                __m128 _vald = _mm_set1_ps(sptr1[5]);
                                __m128 _vale = _mm_set1_ps(sptr1[6]);
                                __m128 _valf = _mm_set1_ps(sptr1[7]);
                                _sum0 = _mm_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                                _sum0 = _mm_fmadd_ps(_val8, _mm_load_ps(kptr0 + 4 * 8), _sum0);
                                _sum1 = _mm_fmadd_ps(_val9, _mm_load_ps(kptr0 + 4 * 9), _sum1);
                                _sum2 = _mm_fmadd_ps(_vala, _mm_load_ps(kptr0 + 4 * 10), _sum2);
                                _sum3 = _mm_fmadd_ps(_valb, _mm_load_ps(kptr0 + 4 * 11), _sum3);
                                _sum0 = _mm_fmadd_ps(_valc, _mm_load_ps(kptr0 + 4 * 12), _sum0);
                                _sum1 = _mm_fmadd_ps(_vald, _mm_load_ps(kptr0 + 4 * 13), _sum1);
                                _sum2 = _mm_fmadd_ps(_vale, _mm_load_ps(kptr0 + 4 * 14), _sum2);
                                _sum3 = _mm_fmadd_ps(_valf, _mm_load_ps(kptr0 + 4 * 15), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr_q0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr_q1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                const float* sptr_q2 = bottom_blob.channel(q / 4 + 2).row(sy) + sx * 4;
                                const float* sptr_q3 = bottom_blob.channel(q / 4 + 3).row(sy) + sx * 4;
                                __m128 _val0 = _mm_set1_ps(sptr_q0[0]);
                                __m128 _val1 = _mm_set1_ps(sptr_q0[1]);
                                __m128 _val2 = _mm_set1_ps(sptr_q0[2]);
                                __m128 _val3 = _mm_set1_ps(sptr_q0[3]);
                                __m128 _val4 = _mm_set1_ps(sptr_q1[0]);
                                __m128 _val5 = _mm_set1_ps(sptr_q1[1]);
                                __m128 _val6 = _mm_set1_ps(sptr_q1[2]);
                                __m128 _val7 = _mm_set1_ps(sptr_q1[3]);
                                __m128 _val8 = _mm_set1_ps(sptr_q2[0]);
                                __m128 _val9 = _mm_set1_ps(sptr_q2[1]);
                                __m128 _vala = _mm_set1_ps(sptr_q2[2]);
                                __m128 _valb = _mm_set1_ps(sptr_q2[3]);
                                __m128 _valc = _mm_set1_ps(sptr_q3[0]);
                                __m128 _vald = _mm_set1_ps(sptr_q3[1]);
                                __m128 _vale = _mm_set1_ps(sptr_q3[2]);
                                __m128 _valf = _mm_set1_ps(sptr_q3[3]);
                                _sum0 = _mm_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                                _sum0 = _mm_fmadd_ps(_val8, _mm_load_ps(kptr0 + 4 * 8), _sum0);
                                _sum1 = _mm_fmadd_ps(_val9, _mm_load_ps(kptr0 + 4 * 9), _sum1);
                                _sum2 = _mm_fmadd_ps(_vala, _mm_load_ps(kptr0 + 4 * 10), _sum2);
                                _sum3 = _mm_fmadd_ps(_valb, _mm_load_ps(kptr0 + 4 * 11), _sum3);
                                _sum0 = _mm_fmadd_ps(_valc, _mm_load_ps(kptr0 + 4 * 12), _sum0);
                                _sum1 = _mm_fmadd_ps(_vald, _mm_load_ps(kptr0 + 4 * 13), _sum1);
                                _sum2 = _mm_fmadd_ps(_vale, _mm_load_ps(kptr0 + 4 * 14), _sum2);
                                _sum3 = _mm_fmadd_ps(_valf, _mm_load_ps(kptr0 + 4 * 15), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m128 _val0 = _mm_set1_ps(bottom_blob.channel(q).row(sy)[sx]);
                                __m128 _val1 = _mm_set1_ps(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m128 _val2 = _mm_set1_ps(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m128 _val3 = _mm_set1_ps(bottom_blob.channel(q + 3).row(sy)[sx]);
                                __m128 _val4 = _mm_set1_ps(bottom_blob.channel(q + 4).row(sy)[sx]);
                                __m128 _val5 = _mm_set1_ps(bottom_blob.channel(q + 5).row(sy)[sx]);
                                __m128 _val6 = _mm_set1_ps(bottom_blob.channel(q + 6).row(sy)[sx]);
                                __m128 _val7 = _mm_set1_ps(bottom_blob.channel(q + 7).row(sy)[sx]);
                                __m128 _val8 = _mm_set1_ps(bottom_blob.channel(q + 8).row(sy)[sx]);
                                __m128 _val9 = _mm_set1_ps(bottom_blob.channel(q + 9).row(sy)[sx]);
                                __m128 _vala = _mm_set1_ps(bottom_blob.channel(q + 10).row(sy)[sx]);
                                __m128 _valb = _mm_set1_ps(bottom_blob.channel(q + 11).row(sy)[sx]);
                                __m128 _valc = _mm_set1_ps(bottom_blob.channel(q + 12).row(sy)[sx]);
                                __m128 _vald = _mm_set1_ps(bottom_blob.channel(q + 13).row(sy)[sx]);
                                __m128 _vale = _mm_set1_ps(bottom_blob.channel(q + 14).row(sy)[sx]);
                                __m128 _valf = _mm_set1_ps(bottom_blob.channel(q + 15).row(sy)[sx]);
                                _sum0 = _mm_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                                _sum0 = _mm_fmadd_ps(_val8, _mm_load_ps(kptr0 + 4 * 8), _sum0);
                                _sum1 = _mm_fmadd_ps(_val9, _mm_load_ps(kptr0 + 4 * 9), _sum1);
                                _sum2 = _mm_fmadd_ps(_vala, _mm_load_ps(kptr0 + 4 * 10), _sum2);
                                _sum3 = _mm_fmadd_ps(_valb, _mm_load_ps(kptr0 + 4 * 11), _sum3);
                                _sum0 = _mm_fmadd_ps(_valc, _mm_load_ps(kptr0 + 4 * 12), _sum0);
                                _sum1 = _mm_fmadd_ps(_vald, _mm_load_ps(kptr0 + 4 * 13), _sum1);
                                _sum2 = _mm_fmadd_ps(_vale, _mm_load_ps(kptr0 + 4 * 14), _sum2);
                                _sum3 = _mm_fmadd_ps(_valf, _mm_load_ps(kptr0 + 4 * 15), _sum3);
                            }
                        }
                    }
                    kptr += maxk * 16 * 4;
                }
#endif // __AVX512F__
                for (; q + 7 < inch; q += 8)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 8 * 4;

                            if (elempack == 8)
                            {
                                const float* sptr = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                __m128 _val0 = _mm_set1_ps(sptr[0]);
                                __m128 _val1 = _mm_set1_ps(sptr[1]);
                                __m128 _val2 = _mm_set1_ps(sptr[2]);
                                __m128 _val3 = _mm_set1_ps(sptr[3]);
                                __m128 _val4 = _mm_set1_ps(sptr[4]);
                                __m128 _val5 = _mm_set1_ps(sptr[5]);
                                __m128 _val6 = _mm_set1_ps(sptr[6]);
                                __m128 _val7 = _mm_set1_ps(sptr[7]);
                                _sum0 = _mm_comp_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_comp_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                __m128 _val0 = _mm_set1_ps(sptr0[0]);
                                __m128 _val1 = _mm_set1_ps(sptr0[1]);
                                __m128 _val2 = _mm_set1_ps(sptr0[2]);
                                __m128 _val3 = _mm_set1_ps(sptr0[3]);
                                __m128 _val4 = _mm_set1_ps(sptr1[0]);
                                __m128 _val5 = _mm_set1_ps(sptr1[1]);
                                __m128 _val6 = _mm_set1_ps(sptr1[2]);
                                __m128 _val7 = _mm_set1_ps(sptr1[3]);
                                _sum0 = _mm_comp_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_comp_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m128 _val0 = _mm_set1_ps(bottom_blob.channel(q).row(sy)[sx]);
                                __m128 _val1 = _mm_set1_ps(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m128 _val2 = _mm_set1_ps(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m128 _val3 = _mm_set1_ps(bottom_blob.channel(q + 3).row(sy)[sx]);
                                __m128 _val4 = _mm_set1_ps(bottom_blob.channel(q + 4).row(sy)[sx]);
                                __m128 _val5 = _mm_set1_ps(bottom_blob.channel(q + 5).row(sy)[sx]);
                                __m128 _val6 = _mm_set1_ps(bottom_blob.channel(q + 6).row(sy)[sx]);
                                __m128 _val7 = _mm_set1_ps(bottom_blob.channel(q + 7).row(sy)[sx]);
                                _sum0 = _mm_comp_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                                _sum0 = _mm_comp_fmadd_ps(_val4, _mm_load_ps(kptr0 + 4 * 4), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val5, _mm_load_ps(kptr0 + 4 * 5), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val6, _mm_load_ps(kptr0 + 4 * 6), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val7, _mm_load_ps(kptr0 + 4 * 7), _sum3);
                            }
                        }
                    }
                    kptr += maxk * 8 * 4;
                }
#endif // __AVX__
                for (; q + 3 < inch; q += 4)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 4 * 4;

                            if (elempack == 4)
                            {
                                const float* sptr = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                __m128 _val0 = _mm_load1_ps(sptr);
                                __m128 _val1 = _mm_load1_ps(sptr + 1);
                                __m128 _val2 = _mm_load1_ps(sptr + 2);
                                __m128 _val3 = _mm_load1_ps(sptr + 3);
                                _sum0 = _mm_comp_fmadd_ps(_val0, _mm_load_ps(kptr0), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val2, _mm_load_ps(kptr0 + 8), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val3, _mm_load_ps(kptr0 + 12), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m128 _val0 = _mm_set1_ps(bottom_blob.channel(q).row(sy)[sx]);
                                __m128 _val1 = _mm_set1_ps(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m128 _val2 = _mm_set1_ps(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m128 _val3 = _mm_set1_ps(bottom_blob.channel(q + 3).row(sy)[sx]);
                                _sum0 = _mm_comp_fmadd_ps(_val0, _mm_load_ps(kptr0 + 4 * 0), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4 * 1), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_val2, _mm_load_ps(kptr0 + 4 * 2), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_val3, _mm_load_ps(kptr0 + 4 * 3), _sum3);
                            }
                        }
                    }
                    kptr += maxk * 4 * 4;
                }
                for (; q + 1 < inch; q += 2)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 2 * 4;

                            const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                            const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;
                            __m128 _val0 = _mm_set1_ps(sptr0[0]);
                            __m128 _val1 = _mm_set1_ps(sptr1[0]);
                            _sum0 = _mm_comp_fmadd_ps(_val0, _mm_load_ps(kptr0), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_val1, _mm_load_ps(kptr0 + 4), _sum1);
                        }
                    }
                    kptr += maxk * 2 * 4;
                }
                for (; q < inch; q++)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 4;

                            const float* sptr = bottom_blob.channel(q).row(sy) + sx;
                            __m128 _val = _mm_set1_ps(sptr[0]);
                            _sum0 = _mm_comp_fmadd_ps(_val, _mm_load_ps(kptr0), _sum0);
                        }
                    }
                    kptr += maxk * 4;
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
                _sum0 = _mm_add_ps(_sum0, _sum2);

                _sum0 = activation_sse(_sum0, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr, _sum0);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[4];
                    _mm_storeu_ps(sum, _sum0);

                    outptr[0] = sum[0];
                    outptr[M] = sum[1];
                    outptr[M * 2] = sum[2];
                    outptr[M * 3] = sum[3];
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 4;
    nn_outch = (outch - remain_outch_start) / 2;
#else // __SSE2__
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __SSE2__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum0 = 0.f;
                float sum1 = 0.f;

                if (bias_data_ptr)
                {
                    sum0 = bias_data_ptr[p];
                    sum1 = bias_data_ptr[p + 1];
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __AVX__
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __SSE2__
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
                const float* kptr = weight_data_tm.channel(p / 2);
#endif

                int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum0_avx512 = _mm512_setzero_ps();
                __m512 _sum1_avx512 = _mm512_setzero_ps();
                for (; q + 15 < inch; q += 16)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 16 * 2;

                            if (elempack == 16)
                            {
                                const float* sptr = bottom_blob.channel(q / 16).row(sy) + sx * 16;
                                __m512 _r0 = _mm512_load_ps(sptr);
                                _sum0_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum0_avx512);
                                _sum1_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0 + 16), _sum1_avx512);
                            }
                            if (elempack == 8)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                const float* sptr1 = bottom_blob.channel(q / 8 + 1).row(sy) + sx * 8;
                                __m512 _r0 = combine8x2_ps(_mm256_load_ps(sptr0), _mm256_load_ps(sptr1));
                                _sum0_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum0_avx512);
                                _sum1_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0 + 16), _sum1_avx512);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                const float* sptr2 = bottom_blob.channel(q / 4 + 2).row(sy) + sx * 4;
                                const float* sptr3 = bottom_blob.channel(q / 4 + 3).row(sy) + sx * 4;
                                __m512 _r0 = combine4x4_ps(_mm_load_ps(sptr0), _mm_load_ps(sptr1), _mm_load_ps(sptr2), _mm_load_ps(sptr3));
                                _sum0_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum0_avx512);
                                _sum1_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0 + 16), _sum1_avx512);
                            }
                            if (elempack == 1)
                            {
                                float tmp[16];
                                for (int qi = 0; qi < 16; qi++)
                                {
                                    const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                    tmp[qi] = sptr[0];
                                }
                                __m512 _r0 = _mm512_loadu_ps(tmp);
                                _sum0_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum0_avx512);
                                _sum1_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0 + 16), _sum1_avx512);
                            }
                        }
                    }
                    kptr += maxk * 16 * 2;
                }
                sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
                sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
#endif // __AVX512F__
                __m256 _sum0_avx = _mm256_setzero_ps();
                __m256 _sum1_avx = _mm256_setzero_ps();
                for (; q + 7 < inch; q += 8)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 8 * 2;

                            if (elempack == 8)
                            {
                                const float* sptr = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                __m256 _r0 = _mm256_load_ps(sptr);
                                _sum0_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0), _sum0_avx);
                                _sum1_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0 + 8), _sum1_avx);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                __m256 _r0 = combine4x2_ps(_mm_load_ps(sptr0), _mm_load_ps(sptr1));
                                _sum0_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0), _sum0_avx);
                                _sum1_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0 + 8), _sum1_avx);
                            }
                            if (elempack == 1)
                            {
                                float tmp[8];
                                for (int qi = 0; qi < 8; qi++)
                                {
                                    const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                    tmp[qi] = sptr[0];
                                }
                                __m256 _r0 = _mm256_loadu_ps(tmp);
                                _sum0_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0), _sum0_avx);
                                _sum1_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0 + 8), _sum1_avx);
                            }
                        }
                    }
                    kptr += maxk * 8 * 2;
                }
                sum0 += _mm256_reduce_add_ps(_sum0_avx);
                sum1 += _mm256_reduce_add_ps(_sum1_avx);
#endif // __AVX__
                __m128 _sum0_sse = _mm_setzero_ps();
                __m128 _sum1_sse = _mm_setzero_ps();
                for (; q + 3 < inch; q += 4)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 4 * 2;

                            if (elempack == 4)
                            {
                                const float* sptr = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                __m128 _r0 = _mm_load_ps(sptr);
                                _sum0_sse = _mm_comp_fmadd_ps(_r0, _mm_load_ps(kptr0), _sum0_sse);
                                _sum1_sse = _mm_comp_fmadd_ps(_r0, _mm_load_ps(kptr0 + 4), _sum1_sse);
                            }
                            if (elempack == 1)
                            {
                                float tmp[4];
                                for (int qi = 0; qi < 4; qi++)
                                {
                                    const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                    tmp[qi] = sptr[0];
                                }
                                __m128 _r0 = _mm_loadu_ps(tmp);
                                _sum0_sse = _mm_comp_fmadd_ps(_r0, _mm_load_ps(kptr0), _sum0_sse);
                                _sum1_sse = _mm_comp_fmadd_ps(_r0, _mm_load_ps(kptr0 + 4), _sum1_sse);
                            }
                        }
                    }
                    kptr += maxk * 4 * 2;
                }
                sum0 += _mm_reduce_add_ps(_sum0_sse);
                sum1 += _mm_reduce_add_ps(_sum1_sse);
#endif // __SSE2__
                for (; q + 1 < inch; q += 2)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 2 * 2;

                            for (int qi = 0; qi < 2; qi++)
                            {
                                const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                sum0 += sptr[0] * kptr0[qi];
                                sum1 += sptr[0] * kptr0[2 + qi];
                            }
                        }
                    }
                    kptr += maxk * 2 * 2;
                }
                for (; q < inch; q++)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 2;

                            const float* sptr = bottom_blob.channel(q).row(sy) + sx;
                            sum0 += sptr[0] * kptr0[0];
                            sum1 += sptr[0] * kptr0[1];
                        }
                    }
                    kptr += maxk * 2;
                }

                sum0 = activation_ss(sum0, activation_type, activation_params);
                sum1 = activation_ss(sum1, activation_type, activation_params);

                outptr0[0] = sum0;
                outptr1[0] = sum1;
                outptr0 += 1;
                outptr1 += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __AVX__
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __SSE2__
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
                const float* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

                int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                for (; q + 15 < inch; q += 16)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 16;

                            if (elempack == 16)
                            {
                                const float* sptr = bottom_blob.channel(q / 16).row(sy) + sx * 16;
                                _sum_avx512 = _mm512_fmadd_ps(_mm512_load_ps(sptr), _mm512_load_ps(kptr0), _sum_avx512);
                            }
                            if (elempack == 8)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                const float* sptr1 = bottom_blob.channel(q / 8 + 1).row(sy) + sx * 8;
                                __m512 _r0 = combine8x2_ps(_mm256_load_ps(sptr0), _mm256_load_ps(sptr1));
                                _sum_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum_avx512);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                const float* sptr2 = bottom_blob.channel(q / 4 + 2).row(sy) + sx * 4;
                                const float* sptr3 = bottom_blob.channel(q / 4 + 3).row(sy) + sx * 4;
                                __m512 _r0 = combine4x4_ps(_mm_load_ps(sptr0), _mm_load_ps(sptr1), _mm_load_ps(sptr2), _mm_load_ps(sptr3));
                                _sum_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum_avx512);
                            }
                            if (elempack == 1)
                            {
                                float tmp[16];
                                for (int qi = 0; qi < 16; qi++)
                                {
                                    const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                    tmp[qi] = sptr[0];
                                }
                                __m512 _r0 = _mm512_loadu_ps(tmp);
                                _sum_avx512 = _mm512_fmadd_ps(_r0, _mm512_load_ps(kptr0), _sum_avx512);
                            }
                        }
                    }
                    kptr += maxk * 16;
                }
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                for (; q + 7 < inch; q += 8)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 8;

                            if (elempack == 8)
                            {
                                const float* sptr = bottom_blob.channel(q / 8).row(sy) + sx * 8;
                                _sum_avx = _mm256_comp_fmadd_ps(_mm256_load_ps(sptr), _mm256_load_ps(kptr0), _sum_avx);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;
                                __m256 _r0 = combine4x2_ps(_mm_load_ps(sptr0), _mm_load_ps(sptr1));
                                _sum_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0), _sum_avx);
                            }
                            if (elempack == 1)
                            {
                                float tmp[8];
                                for (int qi = 0; qi < 8; qi++)
                                {
                                    const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                    tmp[qi] = sptr[0];
                                }
                                __m256 _r0 = _mm256_loadu_ps(tmp);
                                _sum_avx = _mm256_comp_fmadd_ps(_r0, _mm256_load_ps(kptr0), _sum_avx);
                            }
                        }
                    }
                    kptr += maxk * 8;
                }
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                __m128 _sum_sse = _mm_setzero_ps();
                for (; q + 3 < inch; q += 4)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 4;

                            if (elempack == 4)
                            {
                                const float* sptr = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                _sum_sse = _mm_comp_fmadd_ps(_mm_load_ps(sptr), _mm_load_ps(kptr0), _sum_sse);
                            }
                            if (elempack == 1)
                            {
                                float tmp[4];
                                for (int qi = 0; qi < 4; qi++)
                                {
                                    const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                    tmp[qi] = sptr[0];
                                }
                                __m128 _r0 = _mm_loadu_ps(tmp);
                                _sum_sse = _mm_comp_fmadd_ps(_r0, _mm_load_ps(kptr0), _sum_sse);
                            }
                        }
                    }
                    kptr += maxk * 4;
                }
                sum += _mm_reduce_add_ps(_sum_sse);
#endif // __SSE2__
                for (; q + 1 < inch; q += 2)
                {
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
                            int k = y * kernel_w + x;
                            const float* kptr0 = kptr + k * 2;

                            for (int qi = 0; qi < 2; qi++)
                            {
                                const float* sptr = bottom_blob.channel(q + qi).row(sy) + sx;
                                sum += sptr[0] * kptr0[qi];
                            }
                        }
                    }
                    kptr += maxk * 2;
                }
                for (; q < inch; q++)
                {
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
                            int k = y * kernel_w + x;

                            const float* sptr = bottom_blob.channel(q).row(sy) + sx;
                            sum += sptr[0] * kptr[k];
                        }
                    }
                    kptr += maxk;
                }

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[0] = sum;
                outptr += 1;
            }
        }
    }
}
