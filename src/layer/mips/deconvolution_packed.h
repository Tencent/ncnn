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

    Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

    // clang-format off
    // *INDENT-OFF*
#if __mips_msa
    if (num_output >= 4)
    {
        if (num_input >= 4)
            weight_data_tm.create(4 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else if (num_input >= 2)
            weight_data_tm.create(4 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
        else
            weight_data_tm.create(4 * maxk, num_input, num_output / 4 + (num_output % 4) / 2 + num_output % 2);
    }
    else
#endif // __mips_msa
    if (num_output >= 2)
    {
#if __mips_msa
        if (num_input >= 4)
            weight_data_tm.create(2 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
#endif // __mips_msa
        if (num_input >= 2)
            weight_data_tm.create(2 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 2 + num_output % 2);
        else
            weight_data_tm.create(2 * maxk, num_input, num_output / 2 + num_output % 2);
    }
    else
    {
#if __mips_msa
        if (num_input >= 4)
            weight_data_tm.create(4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output);
        else
#endif // __mips_msa
        if (num_input >= 2)
            weight_data_tm.create(2 * maxk, num_input / 2 + num_input % 2, num_output);
        else
            weight_data_tm.create(maxk, num_input, num_output);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
    for (; q + 3 < num_output; q += 4)
    {
        float* g00 = weight_data_tm.channel(q / 4);

        int p = 0;
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
#endif // __mips_msa
    for (; q + 1 < num_output; q += 2)
    {
#if __mips_msa
        float* g00 = weight_data_tm.channel(q / 4 + (q % 4) / 2);
#else
        float* g00 = weight_data_tm.channel(q / 2);
#endif

        int p = 0;
#if __mips_msa
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
#endif // __mips_msa
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
#if __mips_msa
        float* g00 = weight_data_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = weight_data_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa
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
#endif // __mips_msa
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

#if __mips_msa
    nn_outch = outch / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 4;

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
                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w((const float*)bias_data_ptr + p, 0);
                }

                const float* kptr = weight_data_tm.channel(p / 4);

                int q = 0;
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

                                v4f32 _val0 = __msa_fill_w_f32(sptr[0]);
                                v4f32 _val1 = __msa_fill_w_f32(sptr[1]);
                                v4f32 _val2 = __msa_fill_w_f32(sptr[2]);
                                v4f32 _val3 = __msa_fill_w_f32(sptr[3]);
                                _sum0 = __msa_fmadd_w(_sum0, _val0, (v4f32)__msa_ld_w(kptr0, 0));
                                _sum1 = __msa_fmadd_w(_sum1, _val1, (v4f32)__msa_ld_w(kptr0 + 4, 0));
                                _sum2 = __msa_fmadd_w(_sum2, _val2, (v4f32)__msa_ld_w(kptr0 + 4 * 2, 0));
                                _sum3 = __msa_fmadd_w(_sum3, _val3, (v4f32)__msa_ld_w(kptr0 + 4 * 3, 0));
                            }
                            if (elempack == 1)
                            {
                                v4f32 _val0 = __msa_fill_w_f32(bottom_blob.channel(q).row(sy)[sx]);
                                v4f32 _val1 = __msa_fill_w_f32(bottom_blob.channel(q + 1).row(sy)[sx]);
                                v4f32 _val2 = __msa_fill_w_f32(bottom_blob.channel(q + 2).row(sy)[sx]);
                                v4f32 _val3 = __msa_fill_w_f32(bottom_blob.channel(q + 3).row(sy)[sx]);
                                _sum0 = __msa_fmadd_w(_sum0, _val0, (v4f32)__msa_ld_w(kptr0, 0));
                                _sum1 = __msa_fmadd_w(_sum1, _val1, (v4f32)__msa_ld_w(kptr0 + 4, 0));
                                _sum2 = __msa_fmadd_w(_sum2, _val2, (v4f32)__msa_ld_w(kptr0 + 4 * 2, 0));
                                _sum3 = __msa_fmadd_w(_sum3, _val3, (v4f32)__msa_ld_w(kptr0 + 4 * 3, 0));
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
                            v4f32 _val0 = __msa_fill_w_f32(sptr0[0]);
                            v4f32 _val1 = __msa_fill_w_f32(sptr1[0]);
                            _sum0 = __msa_fmadd_w(_sum0, _val0, (v4f32)__msa_ld_w(kptr0, 0));
                            _sum1 = __msa_fmadd_w(_sum1, _val1, (v4f32)__msa_ld_w(kptr0 + 4, 0));
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
                            v4f32 _val = __msa_fill_w_f32(sptr[0]);
                            _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_ld_w(kptr0, 0));
                        }
                    }

                    kptr += maxk * 4;
                }

                _sum0 = __msa_fadd_w(_sum0, _sum1);
                _sum2 = __msa_fadd_w(_sum2, _sum3);
                _sum0 = __msa_fadd_w(_sum0, _sum2);

                _sum0 = activation_msa(_sum0, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_sum0, outptr, 0);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[4];
                    __msa_st_w((v4i32)_sum0, sum, 0);

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
#else  // __mips_msa
    nn_outch = (outch - remain_outch_start) / 2;
#endif // __mips_msa
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
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

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

#if __mips_msa
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
                const float* kptr = weight_data_tm.channel(p / 2);
#endif

                int q = 0;
#if __mips_msa
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

                                sum0 += sptr[0] * kptr0[0];
                                sum0 += sptr[1] * kptr0[1];
                                sum0 += sptr[2] * kptr0[2];
                                sum0 += sptr[3] * kptr0[3];
                                sum1 += sptr[0] * kptr0[4];
                                sum1 += sptr[1] * kptr0[5];
                                sum1 += sptr[2] * kptr0[6];
                                sum1 += sptr[3] * kptr0[7];
                            }
                            if (elempack == 1)
                            {
                                sum0 += bottom_blob.channel(q).row(sy)[sx] * kptr0[0];
                                sum0 += bottom_blob.channel(q + 1).row(sy)[sx] * kptr0[1];
                                sum0 += bottom_blob.channel(q + 2).row(sy)[sx] * kptr0[2];
                                sum0 += bottom_blob.channel(q + 3).row(sy)[sx] * kptr0[3];
                                sum1 += bottom_blob.channel(q).row(sy)[sx] * kptr0[4];
                                sum1 += bottom_blob.channel(q + 1).row(sy)[sx] * kptr0[5];
                                sum1 += bottom_blob.channel(q + 2).row(sy)[sx] * kptr0[6];
                                sum1 += bottom_blob.channel(q + 3).row(sy)[sx] * kptr0[7];
                            }
                        }
                    }

                    kptr += maxk * 4 * 2;
                }
#endif // __mips_msa
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
                            sum0 += sptr[0] * kptr0[0];
                            sum1 += sptr[0] * kptr0[1];
                        }
                    }

                    kptr += maxk * 2;
                }

                sum0 = activation_ss(sum0, activation_type, activation_params);
                sum1 = activation_ss(sum1, activation_type, activation_params);

                if (out_elempack == 1)
                {
                    outptr[0] = sum0;
                    outptr[M] = sum1;
                    outptr += 1;
                }
#if __mips_msa
                if (out_elempack == 4)
                {
                    outptr[0] = sum0;
                    outptr[1] = sum1;
                    outptr += 2;
                }
#endif // __mips_msa
            }
        }
    }
    remain_outch_start += nn_outch * 2;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = remain_outch_start; pp < outch; pp++)
    {
        const int p = pp;

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
                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

#if __mips_msa
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
                const float* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

                int q = 0;
#if __mips_msa
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

                                v4f32 _val = (v4f32)__msa_ld_w(sptr, 0);
                                v4f32 _w = (v4f32)__msa_ld_w(kptr0, 0);
                                v4f32 _s = __msa_fmul_w(_val, _w);
                                sum += __msa_reduce_fadd_w(_s);
                            }
                            if (elempack == 1)
                            {
                                sum += bottom_blob.channel(q).row(sy)[sx] * kptr0[0];
                                sum += bottom_blob.channel(q + 1).row(sy)[sx] * kptr0[1];
                                sum += bottom_blob.channel(q + 2).row(sy)[sx] * kptr0[2];
                                sum += bottom_blob.channel(q + 3).row(sy)[sx] * kptr0[3];
                            }
                        }
                    }

                    kptr += maxk * 4;
                }
#endif // __mips_msa
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

                if (out_elempack == 1)
                {
                    outptr[0] = sum;
                    outptr += 1;
                }
#if __mips_msa
                if (out_elempack == 4)
                {
                    outptr[p % 4] = sum;
                    outptr += 1;
                }
#endif // __mips_msa
            }
        }
    }
}
