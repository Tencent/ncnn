// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void deconvolution_transform_kernel_packed_bf16s(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int out_elempack)
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
    if (out_elempack == 8)
    {
        if (num_input >= 8)
            weight_data_tm.create(8 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
        else if (num_input >= 4)
            weight_data_tm.create(8 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
        else if (num_input >= 2)
            weight_data_tm.create(8 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
        else
            weight_data_tm.create(8 * maxk, num_input, num_output / 8 + (num_output % 8) / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
    }
    else
    if (num_output >= 4)
    {
        if (num_input >= 8)
            weight_data_tm.create(4 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
        else if (num_input >= 4)
            weight_data_tm.create(4 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
        else if (num_input >= 2)
            weight_data_tm.create(4 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
        else
            weight_data_tm.create(4 * maxk, num_input, num_output / 4 + (num_output % 4) / 2 + num_output % 2, (size_t)2u);
    }
    else
#endif // __mips_msa
    if (num_output >= 2)
    {
#if __mips_msa
        if (num_input >= 8)
            weight_data_tm.create(2 * 8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2, (size_t)2u);
        else if (num_input >= 4)
            weight_data_tm.create(2 * 4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output / 2 + num_output % 2, (size_t)2u);
        else
#endif // __mips_msa
        if (num_input >= 2)
            weight_data_tm.create(2 * 2 * maxk, num_input / 2 + num_input % 2, num_output / 2 + num_output % 2, (size_t)2u);
        else
            weight_data_tm.create(2 * maxk, num_input, num_output / 2 + num_output % 2, (size_t)2u);
    }
    else
    {
#if __mips_msa
        if (num_input >= 8)
            weight_data_tm.create(8 * maxk, num_input / 8 + (num_input % 8) / 4 + (num_input % 4) / 2 + num_input % 2, num_output, (size_t)2u);
        else if (num_input >= 4)
            weight_data_tm.create(4 * maxk, num_input / 4 + (num_input % 4) / 2 + num_input % 2, num_output, (size_t)2u);
        else
#endif // __mips_msa
        if (num_input >= 2)
            weight_data_tm.create(2 * maxk, num_input / 2 + num_input % 2, num_output, (size_t)2u);
        else
            weight_data_tm.create(maxk, num_input, num_output, (size_t)2u);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
    for (; out_elempack == 8 && q + 7 < num_output; q += 8)
    {
        unsigned short* g00 = (unsigned short*)weight_data_tm.channel(q / 8);

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
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                    g00[0] = float32_to_bfloat16(k00[k]);
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < num_output; q += 4)
    {
        unsigned short* g00 = (unsigned short*)weight_data_tm.channel(q / 4);

        int p = 0;
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                    g00[0] = float32_to_bfloat16(k00[k]);
                    g00++;
                }
            }
        }
    }
#endif // __mips_msa
    for (; q + 1 < num_output; q += 2)
    {
#if __mips_msa
        unsigned short* g00 = (unsigned short*)weight_data_tm.channel(q / 4 + (q % 4) / 2);
#else
        unsigned short* g00 = (unsigned short*)weight_data_tm.channel(q / 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00++;
                    }
                }
            }
        }
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                        g00[0] = float32_to_bfloat16(k00[k]);
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
                    g00[0] = float32_to_bfloat16(k00[k]);
                    g00++;
                }
            }
        }
    }
    for (; q < num_output; q++)
    {
#if __mips_msa
        unsigned short* g00 = (unsigned short*)weight_data_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        unsigned short* g00 = (unsigned short*)weight_data_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 7 < num_input; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = float32_to_bfloat16(k00[k]);
                    g00++;
                }
            }
        }
        for (; p + 3 < num_input; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = float32_to_bfloat16(k00[k]);
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
                    g00[0] = float32_to_bfloat16(k00[k]);
                    g00++;
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k00 = weight_data_r2.channel(q).row(p);
                g00[0] = float32_to_bfloat16(k00[k]);
                g00++;
            }
        }
    }
}

static void deconvolution_packed_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
    nn_outch = out_elempack == 8 ? outch / 8 : 0;
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

        unsigned short* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);
                v4f32 _sum4 = (v4f32)__msa_fill_w(0);
                v4f32 _sum5 = (v4f32)__msa_fill_w(0);
                v4f32 _sum6 = (v4f32)__msa_fill_w(0);
                v4f32 _sum7 = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w((const float*)bias_data_ptr + p, 0);
                    _sum4 = (v4f32)__msa_ld_w((const float*)bias_data_ptr + p + 4, 0);
                }

                const unsigned short* kptr = (const unsigned short*)weight_data_tm.channel(p / 8);

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
                            const unsigned short* kptr0 = kptr + k * 8 * 8;

                            if (elempack == 8)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 8).row<const unsigned short>(sy) + sx * 8;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                v8i16 _w89_bf16 = __msa_ld_h(kptr0 + 32, 0);
                                v4f32 _w8 = (v4f32)__msa_ilvr_h(_w89_bf16, _zero_bf16);
                                v4f32 _w9 = (v4f32)__msa_ilvl_h(_w89_bf16, _zero_bf16);
                                v8i16 _wab_bf16 = __msa_ld_h(kptr0 + 40, 0);
                                v4f32 _wa = (v4f32)__msa_ilvr_h(_wab_bf16, _zero_bf16);
                                v4f32 _wb = (v4f32)__msa_ilvl_h(_wab_bf16, _zero_bf16);
                                v8i16 _wcd_bf16 = __msa_ld_h(kptr0 + 48, 0);
                                v4f32 _wc = (v4f32)__msa_ilvr_h(_wcd_bf16, _zero_bf16);
                                v4f32 _wd = (v4f32)__msa_ilvl_h(_wcd_bf16, _zero_bf16);
                                v8i16 _wef_bf16 = __msa_ld_h(kptr0 + 56, 0);
                                v4f32 _we = (v4f32)__msa_ilvr_h(_wef_bf16, _zero_bf16);
                                v4f32 _wf = (v4f32)__msa_ilvl_h(_wef_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr[0])), _w0);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(sptr[0])), _w1);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr[1])), _w2);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(sptr[1])), _w3);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr[2])), _w4);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(sptr[2])), _w5);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr[3])), _w6);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(sptr[3])), _w7);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr[4])), _w8);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(sptr[4])), _w9);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr[5])), _wa);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(sptr[5])), _wb);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr[6])), _wc);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(sptr[6])), _wd);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr[7])), _we);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(sptr[7])), _wf);
                            }
                            if (elempack == 4)
                            {
                                const unsigned short* sptr0 = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;
                                const unsigned short* sptr1 = bottom_blob.channel(q / 4 + 1).row<const unsigned short>(sy) + sx * 4;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                v8i16 _w89_bf16 = __msa_ld_h(kptr0 + 32, 0);
                                v4f32 _w8 = (v4f32)__msa_ilvr_h(_w89_bf16, _zero_bf16);
                                v4f32 _w9 = (v4f32)__msa_ilvl_h(_w89_bf16, _zero_bf16);
                                v8i16 _wab_bf16 = __msa_ld_h(kptr0 + 40, 0);
                                v4f32 _wa = (v4f32)__msa_ilvr_h(_wab_bf16, _zero_bf16);
                                v4f32 _wb = (v4f32)__msa_ilvl_h(_wab_bf16, _zero_bf16);
                                v8i16 _wcd_bf16 = __msa_ld_h(kptr0 + 48, 0);
                                v4f32 _wc = (v4f32)__msa_ilvr_h(_wcd_bf16, _zero_bf16);
                                v4f32 _wd = (v4f32)__msa_ilvl_h(_wcd_bf16, _zero_bf16);
                                v8i16 _wef_bf16 = __msa_ld_h(kptr0 + 56, 0);
                                v4f32 _we = (v4f32)__msa_ilvr_h(_wef_bf16, _zero_bf16);
                                v4f32 _wf = (v4f32)__msa_ilvl_h(_wef_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr0[0])), _w0);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(sptr0[0])), _w1);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr0[1])), _w2);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(sptr0[1])), _w3);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr0[2])), _w4);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(sptr0[2])), _w5);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr0[3])), _w6);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(sptr0[3])), _w7);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr1[0])), _w8);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(sptr1[0])), _w9);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr1[1])), _wa);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(sptr1[1])), _wb);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr1[2])), _wc);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(sptr1[2])), _wd);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr1[3])), _we);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(sptr1[3])), _wf);
                            }
                            if (elempack == 1)
                            {
                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                v8i16 _w89_bf16 = __msa_ld_h(kptr0 + 32, 0);
                                v4f32 _w8 = (v4f32)__msa_ilvr_h(_w89_bf16, _zero_bf16);
                                v4f32 _w9 = (v4f32)__msa_ilvl_h(_w89_bf16, _zero_bf16);
                                v8i16 _wab_bf16 = __msa_ld_h(kptr0 + 40, 0);
                                v4f32 _wa = (v4f32)__msa_ilvr_h(_wab_bf16, _zero_bf16);
                                v4f32 _wb = (v4f32)__msa_ilvl_h(_wab_bf16, _zero_bf16);
                                v8i16 _wcd_bf16 = __msa_ld_h(kptr0 + 48, 0);
                                v4f32 _wc = (v4f32)__msa_ilvr_h(_wcd_bf16, _zero_bf16);
                                v4f32 _wd = (v4f32)__msa_ilvl_h(_wcd_bf16, _zero_bf16);
                                v8i16 _wef_bf16 = __msa_ld_h(kptr0 + 56, 0);
                                v4f32 _we = (v4f32)__msa_ilvr_h(_wef_bf16, _zero_bf16);
                                v4f32 _wf = (v4f32)__msa_ilvl_h(_wef_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx])), _w0);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx])), _w1);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx])), _w2);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx])), _w3);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx])), _w4);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx])), _w5);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx])), _w6);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx])), _w7);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 4).row<const unsigned short>(sy)[sx])), _w8);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 4).row<const unsigned short>(sy)[sx])), _w9);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 5).row<const unsigned short>(sy)[sx])), _wa);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 5).row<const unsigned short>(sy)[sx])), _wb);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 6).row<const unsigned short>(sy)[sx])), _wc);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 6).row<const unsigned short>(sy)[sx])), _wd);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 7).row<const unsigned short>(sy)[sx])), _we);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 7).row<const unsigned short>(sy)[sx])), _wf);
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
                            const unsigned short* kptr0 = kptr + k * 4 * 8;

                            if (elempack == 4)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr[0])), _w0);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(sptr[0])), _w1);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr[1])), _w2);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(sptr[1])), _w3);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr[2])), _w4);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(sptr[2])), _w5);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr[3])), _w6);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(sptr[3])), _w7);
                            }
                            if (elempack == 1)
                            {
                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx])), _w0);
                                _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx])), _w1);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx])), _w2);
                                _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx])), _w3);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx])), _w4);
                                _sum6 = __ncnn_msa_fmadd_w(_sum6, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx])), _w5);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx])), _w6);
                                _sum7 = __ncnn_msa_fmadd_w(_sum7, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx])), _w7);
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
                            const unsigned short* kptr0 = kptr + k * 2 * 8;

                            const unsigned short* sptr0 = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            const unsigned short* sptr1 = bottom_blob.channel(q + 1).row<const unsigned short>(sy) + sx;
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                            v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                            v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr0[0])), _w0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, __msa_fill_w_f32(bfloat16_to_float32(sptr0[0])), _w1);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr1[0])), _w2);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, __msa_fill_w_f32(bfloat16_to_float32(sptr1[0])), _w3);
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
                            const unsigned short* kptr0 = kptr + k * 8;

                            const unsigned short* sptr = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(sptr[0]));
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _val, _w1);
                        }
                    }

                    kptr += maxk * 8;
                }

                _sum0 = __msa_fadd_w(_sum0, _sum1);
                _sum2 = __msa_fadd_w(_sum2, _sum3);
                _sum4 = __msa_fadd_w(_sum4, _sum5);
                _sum6 = __msa_fadd_w(_sum6, _sum7);
                _sum0 = __msa_fadd_w(_sum0, _sum2);
                _sum4 = __msa_fadd_w(_sum4, _sum6);

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum4 = activation_msa(_sum4, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __msa_st_w(float2bfloat_msa(_sum0, _sum4), outptr, 0);
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    __msa_storel_d(float2bfloat_msa(_sum0), outptr);
                    __msa_storel_d(float2bfloat_msa(_sum4), outptr + M);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    __msa_st_w((v4i32)_sum0, sum0, 0);
                    __msa_st_w((v4i32)_sum4, sum1, 0);

                    outptr[0] = float32_to_bfloat16(sum0[0]);
                    outptr[M] = float32_to_bfloat16(sum0[1]);
                    outptr[M * 2] = float32_to_bfloat16(sum0[2]);
                    outptr[M * 3] = float32_to_bfloat16(sum0[3]);
                    outptr[M * 4] = float32_to_bfloat16(sum1[0]);
                    outptr[M * 5] = float32_to_bfloat16(sum1[1]);
                    outptr[M * 6] = float32_to_bfloat16(sum1[2]);
                    outptr[M * 7] = float32_to_bfloat16(sum1[3]);
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
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

        unsigned short* outptr = top_blob.channel(p / out_elempack);

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

                const unsigned short* kptr = (const unsigned short*)weight_data_tm.channel(p / 4);

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
                            const unsigned short* kptr0 = kptr + k * 8 * 4;

                            if (elempack == 8)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 8).row<const unsigned short>(sy) + sx * 8;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr[0])), _w0);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr[1])), _w1);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr[2])), _w2);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr[3])), _w3);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr[4])), _w4);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr[5])), _w5);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr[6])), _w6);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr[7])), _w7);
                            }
                            if (elempack == 4)
                            {
                                const unsigned short* sptr0 = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;
                                const unsigned short* sptr1 = bottom_blob.channel(q / 4 + 1).row<const unsigned short>(sy) + sx * 4;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr0[0])), _w0);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr0[1])), _w1);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr0[2])), _w2);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr0[3])), _w3);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(sptr1[0])), _w4);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(sptr1[1])), _w5);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(sptr1[2])), _w6);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(sptr1[3])), _w7);
                            }
                            if (elempack == 1)
                            {
                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                v8i16 _w45_bf16 = __msa_ld_h(kptr0 + 16, 0);
                                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                                v8i16 _w67_bf16 = __msa_ld_h(kptr0 + 24, 0);
                                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx])), _w0);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx])), _w1);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx])), _w2);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx])), _w3);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 4).row<const unsigned short>(sy)[sx])), _w4);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 5).row<const unsigned short>(sy)[sx])), _w5);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 6).row<const unsigned short>(sy)[sx])), _w6);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 7).row<const unsigned short>(sy)[sx])), _w7);
                            }
                        }
                    }

                    kptr += maxk * 8 * 4;
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
                            const unsigned short* kptr0 = kptr + k * 4 * 4;

                            if (elempack == 4)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;

                                v4f32 _val0 = __msa_fill_w_f32(bfloat16_to_float32(sptr[0]));
                                v4f32 _val1 = __msa_fill_w_f32(bfloat16_to_float32(sptr[1]));
                                v4f32 _val2 = __msa_fill_w_f32(bfloat16_to_float32(sptr[2]));
                                v4f32 _val3 = __msa_fill_w_f32(bfloat16_to_float32(sptr[3]));
                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w2);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w3);
                            }
                            if (elempack == 1)
                            {
                                v4f32 _val0 = __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx]));
                                v4f32 _val1 = __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx]));
                                v4f32 _val2 = __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx]));
                                v4f32 _val3 = __msa_fill_w_f32(bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx]));
                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                                v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                                _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                                _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
                                _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w2);
                                _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w3);
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
                            const unsigned short* kptr0 = kptr + k * 2 * 4;

                            const unsigned short* sptr0 = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            const unsigned short* sptr1 = bottom_blob.channel(q + 1).row<const unsigned short>(sy) + sx;
                            v4f32 _val0 = __msa_fill_w_f32(bfloat16_to_float32(sptr0[0]));
                            v4f32 _val1 = __msa_fill_w_f32(bfloat16_to_float32(sptr1[0]));
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
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
                            const unsigned short* kptr0 = kptr + k * 4;

                            const unsigned short* sptr = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(sptr[0]));
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, bfloat2float_msa(kptr0));
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
                    __msa_storel_d(float2bfloat_msa(_sum0), outptr);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[4];
                    __msa_st_w((v4i32)_sum0, sum, 0);

                    outptr[0] = float32_to_bfloat16(sum[0]);
                    outptr[M] = float32_to_bfloat16(sum[1]);
                    outptr[M * 2] = float32_to_bfloat16(sum[2]);
                    outptr[M * 3] = float32_to_bfloat16(sum[3]);
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

        unsigned short* outptr = top_blob.channel(p / out_elempack);

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
                const unsigned short* kptr = (const unsigned short*)weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
                const unsigned short* kptr = (const unsigned short*)weight_data_tm.channel(p / 2);
#endif

                int q = 0;
#if __mips_msa
                v4f32 _sum00 = (v4f32)__msa_fill_w(0);
                v4f32 _sum01 = (v4f32)__msa_fill_w(0);
                v4f32 _sum10 = (v4f32)__msa_fill_w(0);
                v4f32 _sum11 = (v4f32)__msa_fill_w(0);
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
                            const unsigned short* kptr0 = kptr + k * 8 * 2;

                            v4f32 _r0 = (v4f32)__msa_fill_w(0);
                            v4f32 _r1 = (v4f32)__msa_fill_w(0);
                            if (elempack == 8)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 8).row<const unsigned short>(sy) + sx * 8;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _r01_bf16 = __msa_ld_h(sptr, 0);
                                _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                                _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                            }
                            if (elempack == 4)
                            {
                                const unsigned short* sptr0 = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;
                                const unsigned short* sptr1 = bottom_blob.channel(q / 4 + 1).row<const unsigned short>(sy) + sx * 4;

                                _r0 = bfloat2float_msa(sptr0);
                                _r1 = bfloat2float_msa(sptr1);
                            }
                            if (elempack == 1)
                            {
                                unsigned short tmp[8];
                                tmp[0] = bottom_blob.channel(q).row<const unsigned short>(sy)[sx];
                                tmp[1] = bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx];
                                tmp[2] = bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx];
                                tmp[3] = bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx];
                                tmp[4] = bottom_blob.channel(q + 4).row<const unsigned short>(sy)[sx];
                                tmp[5] = bottom_blob.channel(q + 5).row<const unsigned short>(sy)[sx];
                                tmp[6] = bottom_blob.channel(q + 6).row<const unsigned short>(sy)[sx];
                                tmp[7] = bottom_blob.channel(q + 7).row<const unsigned short>(sy)[sx];

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _r01_bf16 = __msa_ld_h(tmp, 0);
                                _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                                _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                            }

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            v8i16 _w23_bf16 = __msa_ld_h(kptr0 + 8, 0);
                            v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                            v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                            _sum00 = __ncnn_msa_fmadd_w(_sum00, _r0, _w0);
                            _sum01 = __ncnn_msa_fmadd_w(_sum01, _r1, _w1);
                            _sum10 = __ncnn_msa_fmadd_w(_sum10, _r0, _w2);
                            _sum11 = __ncnn_msa_fmadd_w(_sum11, _r1, _w3);
                        }
                    }

                    kptr += maxk * 8 * 2;
                }
                _sum00 = __msa_fadd_w(_sum00, _sum01);
                _sum10 = __msa_fadd_w(_sum10, _sum11);
                sum0 += __msa_reduce_fadd_w(_sum00);
                sum1 += __msa_reduce_fadd_w(_sum10);

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
                            const unsigned short* kptr0 = kptr + k * 4 * 2;

                            if (elempack == 4)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;

                                sum0 += bfloat16_to_float32(sptr[0]) * bfloat16_to_float32(kptr0[0]);
                                sum0 += bfloat16_to_float32(sptr[1]) * bfloat16_to_float32(kptr0[1]);
                                sum0 += bfloat16_to_float32(sptr[2]) * bfloat16_to_float32(kptr0[2]);
                                sum0 += bfloat16_to_float32(sptr[3]) * bfloat16_to_float32(kptr0[3]);
                                sum1 += bfloat16_to_float32(sptr[0]) * bfloat16_to_float32(kptr0[4]);
                                sum1 += bfloat16_to_float32(sptr[1]) * bfloat16_to_float32(kptr0[5]);
                                sum1 += bfloat16_to_float32(sptr[2]) * bfloat16_to_float32(kptr0[6]);
                                sum1 += bfloat16_to_float32(sptr[3]) * bfloat16_to_float32(kptr0[7]);
                            }
                            if (elempack == 1)
                            {
                                sum0 += bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[0]);
                                sum0 += bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[1]);
                                sum0 += bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[2]);
                                sum0 += bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[3]);
                                sum1 += bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[4]);
                                sum1 += bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[5]);
                                sum1 += bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[6]);
                                sum1 += bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[7]);
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
                            const unsigned short* kptr0 = kptr + k * 2 * 2;

                            const unsigned short* sptr0 = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            const unsigned short* sptr1 = bottom_blob.channel(q + 1).row<const unsigned short>(sy) + sx;
                            sum0 += bfloat16_to_float32(sptr0[0]) * bfloat16_to_float32(kptr0[0]);
                            sum0 += bfloat16_to_float32(sptr1[0]) * bfloat16_to_float32(kptr0[1]);
                            sum1 += bfloat16_to_float32(sptr0[0]) * bfloat16_to_float32(kptr0[2]);
                            sum1 += bfloat16_to_float32(sptr1[0]) * bfloat16_to_float32(kptr0[3]);
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
                            const unsigned short* kptr0 = kptr + k * 2;

                            const unsigned short* sptr = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            sum0 += bfloat16_to_float32(sptr[0]) * bfloat16_to_float32(kptr0[0]);
                            sum1 += bfloat16_to_float32(sptr[0]) * bfloat16_to_float32(kptr0[1]);
                        }
                    }

                    kptr += maxk * 2;
                }

                sum0 = activation_ss(sum0, activation_type, activation_params);
                sum1 = activation_ss(sum1, activation_type, activation_params);

                if (out_elempack == 1)
                {
                    outptr[0] = float32_to_bfloat16(sum0);
                    outptr[M] = float32_to_bfloat16(sum1);
                    outptr += 1;
                }
#if __mips_msa
                if (out_elempack == 4)
                {
                    outptr[0] = float32_to_bfloat16(sum0);
                    outptr[1] = float32_to_bfloat16(sum1);
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

        unsigned short* outptr = top_blob.channel(p / out_elempack);

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
                const unsigned short* kptr = (const unsigned short*)weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
                const unsigned short* kptr = (const unsigned short*)weight_data_tm.channel(p / 2 + p % 2);
#endif

                int q = 0;
#if __mips_msa
                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
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
                            const unsigned short* kptr0 = kptr + k * 8;

                            v4f32 _r0 = (v4f32)__msa_fill_w(0);
                            v4f32 _r1 = (v4f32)__msa_fill_w(0);
                            if (elempack == 8)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 8).row<const unsigned short>(sy) + sx * 8;

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _r01_bf16 = __msa_ld_h(sptr, 0);
                                _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                                _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                            }
                            if (elempack == 4)
                            {
                                const unsigned short* sptr0 = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;
                                const unsigned short* sptr1 = bottom_blob.channel(q / 4 + 1).row<const unsigned short>(sy) + sx * 4;

                                _r0 = bfloat2float_msa(sptr0);
                                _r1 = bfloat2float_msa(sptr1);
                            }
                            if (elempack == 1)
                            {
                                unsigned short tmp[8];
                                tmp[0] = bottom_blob.channel(q).row<const unsigned short>(sy)[sx];
                                tmp[1] = bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx];
                                tmp[2] = bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx];
                                tmp[3] = bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx];
                                tmp[4] = bottom_blob.channel(q + 4).row<const unsigned short>(sy)[sx];
                                tmp[5] = bottom_blob.channel(q + 5).row<const unsigned short>(sy)[sx];
                                tmp[6] = bottom_blob.channel(q + 6).row<const unsigned short>(sy)[sx];
                                tmp[7] = bottom_blob.channel(q + 7).row<const unsigned short>(sy)[sx];

                                v8i16 _zero_bf16 = __msa_fill_h(0);
                                v8i16 _r01_bf16 = __msa_ld_h(tmp, 0);
                                _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                                _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                            }

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr0, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _r1, _w1);
                        }
                    }

                    kptr += maxk * 8;
                }
                _sum0 = __msa_fadd_w(_sum0, _sum1);
                sum += __msa_reduce_fadd_w(_sum0);

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
                            const unsigned short* kptr0 = kptr + k * 4;

                            if (elempack == 4)
                            {
                                const unsigned short* sptr = bottom_blob.channel(q / 4).row<const unsigned short>(sy) + sx * 4;

                                v4f32 _val = bfloat2float_msa(sptr);
                                v4f32 _w = bfloat2float_msa(kptr0);
                                v4f32 _s = __msa_fmul_w(_val, _w);
                                sum += __msa_reduce_fadd_w(_s);
                            }
                            if (elempack == 1)
                            {
                                sum += bfloat16_to_float32(bottom_blob.channel(q).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[0]);
                                sum += bfloat16_to_float32(bottom_blob.channel(q + 1).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[1]);
                                sum += bfloat16_to_float32(bottom_blob.channel(q + 2).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[2]);
                                sum += bfloat16_to_float32(bottom_blob.channel(q + 3).row<const unsigned short>(sy)[sx]) * bfloat16_to_float32(kptr0[3]);
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
                            const unsigned short* kptr0 = kptr + k * 2;

                            const unsigned short* sptr0 = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            const unsigned short* sptr1 = bottom_blob.channel(q + 1).row<const unsigned short>(sy) + sx;
                            sum += bfloat16_to_float32(sptr0[0]) * bfloat16_to_float32(kptr0[0]);
                            sum += bfloat16_to_float32(sptr1[0]) * bfloat16_to_float32(kptr0[1]);
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

                            const unsigned short* sptr = bottom_blob.channel(q).row<const unsigned short>(sy) + sx;
                            sum += bfloat16_to_float32(sptr[0]) * bfloat16_to_float32(kptr[k]);
                        }
                    }

                    kptr += maxk;
                }

                sum = activation_ss(sum, activation_type, activation_params);

                if (out_elempack == 1)
                {
                    outptr[0] = float32_to_bfloat16(sum);
                    outptr += 1;
                }
#if __mips_msa
                if (out_elempack == 4)
                {
                    outptr[p % 4] = float32_to_bfloat16(sum);
                    outptr += 1;
                }
#endif // __mips_msa
            }
        }
    }
}
