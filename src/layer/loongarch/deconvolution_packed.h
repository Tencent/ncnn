// Copyright 2024 Tencent
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
#if __loongarch_sx
#if __loongarch_asx
    if (num_output >= 8)
    {
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
#endif // __loongarch_asx
    if (num_output >= 4)
    {
#if __loongarch_asx
        if (num_input >= 8)
            weight_data_tm.create(4 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else
#endif // __loongarch_asx
        if (num_input >= 4)
            weight_data_tm.create(4 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 2)
            weight_data_tm.create(4 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else
            weight_data_tm.create(4 * maxk, num_input, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
    }
    else
#endif // __loongarch_sx
    if (num_output >= 2)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (num_input >= 8)
            weight_data_tm.create(2 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
#endif // __loongarch_asx
        if (num_input >= 4)
            weight_data_tm.create(2 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
#endif // __loongarch_sx
        if (num_input >= 2)
            weight_data_tm.create(2 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
            weight_data_tm.create(2 * maxk, num_input, num_output / 2 + num_output % 2);
    }
    else
    {
#if __loongarch_sx
#if __loongarch_asx
        if (num_input >= 8)
            weight_data_tm.create(8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output);
        else
#endif // __loongarch_asx
        if (num_input >= 4)
            weight_data_tm.create(4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output);
        else
#endif // __loongarch_sx
        if (num_input >= 2)
            weight_data_tm.create(2 * maxk, num_input / 2 + num_input % 2, num_output);
        else
            weight_data_tm.create(maxk, num_input, num_output);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; q + 7 < num_output; q += 8)
    {
        float* g00 = weight_data_tm.channel(q / 8);

        int p = 0;
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
#endif // __loongarch_asx
    for (; q + 3 < num_output; q += 4)
    {
#if __loongarch_asx
        float* g00 = weight_data_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = weight_data_tm.channel(q / 4);
#endif

        int p = 0;
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
    for (; q + 1 < num_output; q += 2)
    {
#if __loongarch_sx
#if __loongarch_asx
        float* g00 = weight_data_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        float* g00 = weight_data_tm.channel(q / 4 + (q % 4) / 2);
#endif
#else
        float* g00 = weight_data_tm.channel(q / 2);
#endif

        int p = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        float* g00 = weight_data_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = weight_data_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#endif
#else
        float* g00 = weight_data_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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

    const int M = top_blob.cstep * out_elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

    const float* bias_data_ptr = bias_data;

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __loongarch_sx
#if __loongarch_asx
    nn_outch = outch / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 8;

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
                __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (__m256)__lasx_xvld(bias_data_ptr + p, 0);
                }

                const float* kptr = weight_data_tm.channel(p / 8);

                int q = 0;
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

                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);
                                __m256 _w2 = (__m256)__lasx_xvld(kptr0 + 16, 0);
                                __m256 _w3 = (__m256)__lasx_xvld(kptr0 + 24, 0);
                                __m256 _w4 = (__m256)__lasx_xvld(kptr0 + 32, 0);
                                __m256 _w5 = (__m256)__lasx_xvld(kptr0 + 40, 0);
                                __m256 _w6 = (__m256)__lasx_xvld(kptr0 + 48, 0);
                                __m256 _w7 = (__m256)__lasx_xvld(kptr0 + 56, 0);

                                _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(sptr[0]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(sptr[1]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(sptr[2]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(sptr[3]), _sum3);
                                _sum0 = __lasx_xvfmadd_s(_w4, __lasx_xvreplfr2vr_s(sptr[4]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w5, __lasx_xvreplfr2vr_s(sptr[5]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w6, __lasx_xvreplfr2vr_s(sptr[6]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w7, __lasx_xvreplfr2vr_s(sptr[7]), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;

                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);
                                __m256 _w2 = (__m256)__lasx_xvld(kptr0 + 16, 0);
                                __m256 _w3 = (__m256)__lasx_xvld(kptr0 + 24, 0);
                                __m256 _w4 = (__m256)__lasx_xvld(kptr0 + 32, 0);
                                __m256 _w5 = (__m256)__lasx_xvld(kptr0 + 40, 0);
                                __m256 _w6 = (__m256)__lasx_xvld(kptr0 + 48, 0);
                                __m256 _w7 = (__m256)__lasx_xvld(kptr0 + 56, 0);

                                _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(sptr0[0]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(sptr0[1]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(sptr0[2]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(sptr0[3]), _sum3);
                                _sum0 = __lasx_xvfmadd_s(_w4, __lasx_xvreplfr2vr_s(sptr1[0]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w5, __lasx_xvreplfr2vr_s(sptr1[1]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w6, __lasx_xvreplfr2vr_s(sptr1[2]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w7, __lasx_xvreplfr2vr_s(sptr1[3]), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);
                                __m256 _w2 = (__m256)__lasx_xvld(kptr0 + 16, 0);
                                __m256 _w3 = (__m256)__lasx_xvld(kptr0 + 24, 0);
                                __m256 _w4 = (__m256)__lasx_xvld(kptr0 + 32, 0);
                                __m256 _w5 = (__m256)__lasx_xvld(kptr0 + 40, 0);
                                __m256 _w6 = (__m256)__lasx_xvld(kptr0 + 48, 0);
                                __m256 _w7 = (__m256)__lasx_xvld(kptr0 + 56, 0);

                                _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(bottom_blob.channel(q).row(sy)[sx]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 1).row(sy)[sx]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 2).row(sy)[sx]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 3).row(sy)[sx]), _sum3);
                                _sum0 = __lasx_xvfmadd_s(_w4, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 4).row(sy)[sx]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w5, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 5).row(sy)[sx]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w6, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 6).row(sy)[sx]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w7, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 7).row(sy)[sx]), _sum3);
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

                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);
                                __m256 _w2 = (__m256)__lasx_xvld(kptr0 + 16, 0);
                                __m256 _w3 = (__m256)__lasx_xvld(kptr0 + 24, 0);

                                _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(sptr[0]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(sptr[1]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(sptr[2]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(sptr[3]), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);
                                __m256 _w2 = (__m256)__lasx_xvld(kptr0 + 16, 0);
                                __m256 _w3 = (__m256)__lasx_xvld(kptr0 + 24, 0);

                                _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(bottom_blob.channel(q).row(sy)[sx]), _sum0);
                                _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 1).row(sy)[sx]), _sum1);
                                _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 2).row(sy)[sx]), _sum2);
                                _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(bottom_blob.channel(q + 3).row(sy)[sx]), _sum3);
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

                            __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                            __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);

                            _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(sptr0[0]), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(sptr1[0]), _sum1);
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

                            __m256 _val = __lasx_xvreplfr2vr_s(sptr[0]);
                            __m256 _w = (__m256)__lasx_xvld(kptr0, 0);
                            _sum0 = __lasx_xvfmadd_s(_w, _val, _sum0);
                        }
                    }

                    kptr += maxk * 8;
                }

                _sum0 = __lasx_xvfadd_s(_sum0, _sum1);
                _sum2 = __lasx_xvfadd_s(_sum2, _sum3);
                _sum0 = __lasx_xvfadd_s(_sum0, _sum2);

                _sum0 = activation_lasx(_sum0, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_sum0, outptr, 0);
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    float tmp[8];
                    __lasx_xvst((__m256i)_sum0, tmp, 0);
                    __lsx_vst((__m128i)__lsx_vld(tmp, 0), outptr, 0);
                    __lsx_vst((__m128i)__lsx_vld(tmp + 4, 0), outptr + M, 0);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[8];
                    __lasx_xvst((__m256i)_sum0, sum, 0);

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
#else  // __loongarch_asx
    nn_outch = outch / 4;
#endif // __loongarch_asx
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

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
                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (__m128)__lsx_vld(bias_data_ptr + p, 0);
                }

#if __loongarch_asx
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
                const float* kptr = weight_data_tm.channel(p / 4);
#endif

                int q = 0;
#if __loongarch_asx
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

                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);
                                __m128 _w2 = (__m128)__lsx_vld(kptr0 + 8, 0);
                                __m128 _w3 = (__m128)__lsx_vld(kptr0 + 12, 0);
                                __m128 _w4 = (__m128)__lsx_vld(kptr0 + 16, 0);
                                __m128 _w5 = (__m128)__lsx_vld(kptr0 + 20, 0);
                                __m128 _w6 = (__m128)__lsx_vld(kptr0 + 24, 0);
                                __m128 _w7 = (__m128)__lsx_vld(kptr0 + 28, 0);

                                _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(sptr[0]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(sptr[1]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(sptr[2]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(sptr[3]), _sum3);
                                _sum0 = __lsx_vfmadd_s(_w4, __lsx_vreplfr2vr_s(sptr[4]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w5, __lsx_vreplfr2vr_s(sptr[5]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w6, __lsx_vreplfr2vr_s(sptr[6]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w7, __lsx_vreplfr2vr_s(sptr[7]), _sum3);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;

                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);
                                __m128 _w2 = (__m128)__lsx_vld(kptr0 + 8, 0);
                                __m128 _w3 = (__m128)__lsx_vld(kptr0 + 12, 0);
                                __m128 _w4 = (__m128)__lsx_vld(kptr0 + 16, 0);
                                __m128 _w5 = (__m128)__lsx_vld(kptr0 + 20, 0);
                                __m128 _w6 = (__m128)__lsx_vld(kptr0 + 24, 0);
                                __m128 _w7 = (__m128)__lsx_vld(kptr0 + 28, 0);

                                _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(sptr0[0]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(sptr0[1]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(sptr0[2]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(sptr0[3]), _sum3);
                                _sum0 = __lsx_vfmadd_s(_w4, __lsx_vreplfr2vr_s(sptr1[0]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w5, __lsx_vreplfr2vr_s(sptr1[1]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w6, __lsx_vreplfr2vr_s(sptr1[2]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w7, __lsx_vreplfr2vr_s(sptr1[3]), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);
                                __m128 _w2 = (__m128)__lsx_vld(kptr0 + 8, 0);
                                __m128 _w3 = (__m128)__lsx_vld(kptr0 + 12, 0);
                                __m128 _w4 = (__m128)__lsx_vld(kptr0 + 16, 0);
                                __m128 _w5 = (__m128)__lsx_vld(kptr0 + 20, 0);
                                __m128 _w6 = (__m128)__lsx_vld(kptr0 + 24, 0);
                                __m128 _w7 = (__m128)__lsx_vld(kptr0 + 28, 0);

                                _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(bottom_blob.channel(q).row(sy)[sx]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 1).row(sy)[sx]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 2).row(sy)[sx]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 3).row(sy)[sx]), _sum3);
                                _sum0 = __lsx_vfmadd_s(_w4, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 4).row(sy)[sx]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w5, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 5).row(sy)[sx]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w6, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 6).row(sy)[sx]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w7, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 7).row(sy)[sx]), _sum3);
                            }
                        }
                    }

                    kptr += maxk * 8 * 4;
                }
#endif // __loongarch_asx
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

                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);
                                __m128 _w2 = (__m128)__lsx_vld(kptr0 + 8, 0);
                                __m128 _w3 = (__m128)__lsx_vld(kptr0 + 12, 0);

                                _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(sptr[0]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(sptr[1]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(sptr[2]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(sptr[3]), _sum3);
                            }
                            if (elempack == 1)
                            {
                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);
                                __m128 _w2 = (__m128)__lsx_vld(kptr0 + 8, 0);
                                __m128 _w3 = (__m128)__lsx_vld(kptr0 + 12, 0);

                                _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(bottom_blob.channel(q).row(sy)[sx]), _sum0);
                                _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 1).row(sy)[sx]), _sum1);
                                _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 2).row(sy)[sx]), _sum2);
                                _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(bottom_blob.channel(q + 3).row(sy)[sx]), _sum3);
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

                            __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                            __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);

                            _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(sptr0[0]), _sum0);
                            _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(sptr1[0]), _sum1);
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

                            __m128 _val = __lsx_vreplfr2vr_s(sptr[0]);
                            __m128 _w = (__m128)__lsx_vld(kptr0, 0);
                            _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);
                        }
                    }

                    kptr += maxk * 4;
                }

                _sum0 = __lsx_vfadd_s(_sum0, _sum1);
                _sum2 = __lsx_vfadd_s(_sum2, _sum3);
                _sum0 = __lsx_vfadd_s(_sum0, _sum2);

                _sum0 = activation_lsx(_sum0, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr, 0);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[4];
                    __lsx_vst((__m128i)_sum0, sum, 0);

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
#else  // __loongarch_sx
    nn_outch = (outch - remain_outch_start) / 2;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
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

#if __loongarch_sx
#if __loongarch_asx
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#endif
#else
                const float* kptr = weight_data_tm.channel(p / 2);
#endif

                int q = 0;
#if __loongarch_sx
#if __loongarch_asx
                __m256 _sum0_256 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum1_256 = (__m256)__lasx_xvreplgr2vr_w(0);
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

                                __m256 _r0 = (__m256)__lasx_xvld(sptr, 0);
                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);

                                _sum0_256 = __lasx_xvfmadd_s(_r0, _w0, _sum0_256);
                                _sum1_256 = __lasx_xvfmadd_s(_r0, _w1, _sum1_256);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;

                                __m256 _r0 = __lasx_concat_128_s((__m128)__lsx_vld(sptr0, 0), (__m128)__lsx_vld(sptr1, 0));
                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);

                                _sum0_256 = __lasx_xvfmadd_s(_r0, _w0, _sum0_256);
                                _sum1_256 = __lasx_xvfmadd_s(_r0, _w1, _sum1_256);
                            }
                            if (elempack == 1)
                            {
                                float tmp[8];
                                tmp[0] = bottom_blob.channel(q).row(sy)[sx];
                                tmp[1] = bottom_blob.channel(q + 1).row(sy)[sx];
                                tmp[2] = bottom_blob.channel(q + 2).row(sy)[sx];
                                tmp[3] = bottom_blob.channel(q + 3).row(sy)[sx];
                                tmp[4] = bottom_blob.channel(q + 4).row(sy)[sx];
                                tmp[5] = bottom_blob.channel(q + 5).row(sy)[sx];
                                tmp[6] = bottom_blob.channel(q + 6).row(sy)[sx];
                                tmp[7] = bottom_blob.channel(q + 7).row(sy)[sx];

                                __m256 _r0 = (__m256)__lasx_xvld(tmp, 0);
                                __m256 _w0 = (__m256)__lasx_xvld(kptr0, 0);
                                __m256 _w1 = (__m256)__lasx_xvld(kptr0 + 8, 0);

                                _sum0_256 = __lasx_xvfmadd_s(_r0, _w0, _sum0_256);
                                _sum1_256 = __lasx_xvfmadd_s(_r0, _w1, _sum1_256);
                            }
                        }
                    }

                    kptr += maxk * 8 * 2;
                }
                sum0 += __lasx_reduce_fadd_s(_sum0_256);
                sum1 += __lasx_reduce_fadd_s(_sum1_256);
#endif // __loongarch_asx
                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
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

                                __m128 _r0 = (__m128)__lsx_vld(sptr, 0);
                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);

                                _sum0 = __lsx_vfmadd_s(_r0, _w0, _sum0);
                                _sum1 = __lsx_vfmadd_s(_r0, _w1, _sum1);
                            }
                            if (elempack == 1)
                            {
                                __m128 _val0 = __lsx_vreplfr2vr_s(bottom_blob.channel(q).row(sy)[sx]);
                                __m128 _val1 = __lsx_vreplfr2vr_s(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m128 _val2 = __lsx_vreplfr2vr_s(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m128 _val3 = __lsx_vreplfr2vr_s(bottom_blob.channel(q + 3).row(sy)[sx]);

                                __m128 _val01 = (__m128)__lsx_vilvl_w((__m128i)_val1, (__m128i)_val0);
                                __m128 _val23 = (__m128)__lsx_vilvl_w((__m128i)_val3, (__m128i)_val2);
                                __m128 _r0 = (__m128)__lsx_vilvl_d((__m128i)_val23, (__m128i)_val01);

                                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                                __m128 _w1 = (__m128)__lsx_vld(kptr0 + 4, 0);

                                _sum0 = __lsx_vfmadd_s(_r0, _w0, _sum0);
                                _sum1 = __lsx_vfmadd_s(_r0, _w1, _sum1);
                            }
                        }
                    }

                    kptr += maxk * 4 * 2;
                }
                sum0 += __lsx_reduce_fadd_s(_sum0);
                sum1 += __lsx_reduce_fadd_s(_sum1);
#endif // __loongarch_sx
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

                            const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                            const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;

                            sum0 += sptr0[0] * kptr0[0];
                            sum0 += sptr1[0] * kptr0[1];
                            sum1 += sptr0[0] * kptr0[2];
                            sum1 += sptr1[0] * kptr0[3];
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

                            float val = sptr[0];
                            sum0 += val * kptr0[0];
                            sum1 += val * kptr0[1];
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
        // shadowed variable for less openmp task args
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

#if __loongarch_sx
#if __loongarch_asx
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#endif
#else
                const float* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

                int q = 0;
#if __loongarch_sx
#if __loongarch_asx
                __m256 _sum_256 = (__m256)__lasx_xvreplgr2vr_w(0);
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

                                __m256 _r0 = (__m256)__lasx_xvld(sptr, 0);
                                __m256 _w = (__m256)__lasx_xvld(kptr0, 0);
                                _sum_256 = __lasx_xvfmadd_s(_r0, _w, _sum_256);
                            }
                            if (elempack == 4)
                            {
                                const float* sptr0 = bottom_blob.channel(q / 4).row(sy) + sx * 4;
                                const float* sptr1 = bottom_blob.channel(q / 4 + 1).row(sy) + sx * 4;

                                __m256 _r0 = __lasx_concat_128_s((__m128)__lsx_vld(sptr0, 0), (__m128)__lsx_vld(sptr1, 0));
                                __m256 _w = (__m256)__lasx_xvld(kptr0, 0);
                                _sum_256 = __lasx_xvfmadd_s(_r0, _w, _sum_256);
                            }
                            if (elempack == 1)
                            {
                                float tmp[8];
                                tmp[0] = bottom_blob.channel(q).row(sy)[sx];
                                tmp[1] = bottom_blob.channel(q + 1).row(sy)[sx];
                                tmp[2] = bottom_blob.channel(q + 2).row(sy)[sx];
                                tmp[3] = bottom_blob.channel(q + 3).row(sy)[sx];
                                tmp[4] = bottom_blob.channel(q + 4).row(sy)[sx];
                                tmp[5] = bottom_blob.channel(q + 5).row(sy)[sx];
                                tmp[6] = bottom_blob.channel(q + 6).row(sy)[sx];
                                tmp[7] = bottom_blob.channel(q + 7).row(sy)[sx];

                                __m256 _r0 = (__m256)__lasx_xvld(tmp, 0);
                                __m256 _w = (__m256)__lasx_xvld(kptr0, 0);
                                _sum_256 = __lasx_xvfmadd_s(_r0, _w, _sum_256);
                            }
                        }
                    }

                    kptr += maxk * 8;
                }
                sum += __lasx_reduce_fadd_s(_sum_256);
#endif // __loongarch_asx
                __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);
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

                                __m128 _r0 = (__m128)__lsx_vld(sptr, 0);
                                __m128 _w = (__m128)__lsx_vld(kptr0, 0);
                                _sum = __lsx_vfmadd_s(_r0, _w, _sum);
                            }
                            if (elempack == 1)
                            {
                                __m128 _r0 = __lsx_vreplfr2vr_s(bottom_blob.channel(q).row(sy)[sx]);
                                __m128 _r1 = __lsx_vreplfr2vr_s(bottom_blob.channel(q + 1).row(sy)[sx]);
                                __m128 _r2 = __lsx_vreplfr2vr_s(bottom_blob.channel(q + 2).row(sy)[sx]);
                                __m128 _r3 = __lsx_vreplfr2vr_s(bottom_blob.channel(q + 3).row(sy)[sx]);

                                __m128 _r01 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                                __m128 _r23 = (__m128)__lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);
                                __m128 _r0123 = (__m128)__lsx_vilvl_d((__m128i)_r23, (__m128i)_r01);

                                __m128 _w = (__m128)__lsx_vld(kptr0, 0);
                                _sum = __lsx_vfmadd_s(_r0123, _w, _sum);
                            }
                        }
                    }

                    kptr += maxk * 4;
                }
                sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
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

                            const float* sptr0 = bottom_blob.channel(q).row(sy) + sx;
                            const float* sptr1 = bottom_blob.channel(q + 1).row(sy) + sx;

                            sum += sptr0[0] * kptr0[0];
                            sum += sptr1[0] * kptr0[1];
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
