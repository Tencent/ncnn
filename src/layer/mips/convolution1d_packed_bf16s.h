// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution1d_transform_kernel_packed_bf16s(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    // src = kernel(outh, inh, kernel_w)
    // dst = pb-pa-kw-inh/pa-outh/pb

    // clang-format off
    // *INDENT-OFF*
#if __mips_msa
    if (outh >= 8)
    {
        if (inh >= 8)
            kernel_tm.create(8 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else if (inh >= 4)
            kernel_tm.create(8 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else if (inh >= 2)
            kernel_tm.create(8 * 2 * kernel_w, inh / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(8 * kernel_w, inh, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
    }
    else
    if (outh >= 4)
    {
        if (inh >= 8)
            kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
    }
    else
#endif // __mips_msa
    if (outh >= 2)
    {
#if __mips_msa
        if (inh >= 8)
            kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
#endif // __mips_msa
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2, (size_t)2u);
    }
    else
    {
#if __mips_msa
        if (inh >= 8)
            kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh, (size_t)2u);
        else if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh, (size_t)2u);
        else
#endif // __mips_msa
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh, (size_t)2u);
        else
            kernel_tm.create(kernel_w, inh, outh, (size_t)2u);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
    for (; q + 7 < outh; q += 8)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;
        const float* kptr4 = (const float*)kernel + (q + 4) * inh * kernel_w;
        const float* kptr5 = (const float*)kernel + (q + 5) * inh * kernel_w;
        const float* kptr6 = (const float*)kernel + (q + 6) * inh * kernel_w;
        const float* kptr7 = (const float*)kernel + (q + 7) * inh * kernel_w;

        unsigned short* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    g00[2] = float32_to_bfloat16(k2[0]);
                    g00[3] = float32_to_bfloat16(k3[0]);
                    g00[4] = float32_to_bfloat16(k4[0]);
                    g00[5] = float32_to_bfloat16(k5[0]);
                    g00[6] = float32_to_bfloat16(k6[0]);
                    g00[7] = float32_to_bfloat16(k7[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    k4 += kernel_w;
                    k5 += kernel_w;
                    k6 += kernel_w;
                    k7 += kernel_w;
                    g00 += 8;
                }
            }

            kptr0 += kernel_w * 8;
            kptr1 += kernel_w * 8;
            kptr2 += kernel_w * 8;
            kptr3 += kernel_w * 8;
            kptr4 += kernel_w * 8;
            kptr5 += kernel_w * 8;
            kptr6 += kernel_w * 8;
            kptr7 += kernel_w * 8;
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    g00[2] = float32_to_bfloat16(k2[0]);
                    g00[3] = float32_to_bfloat16(k3[0]);
                    g00[4] = float32_to_bfloat16(k4[0]);
                    g00[5] = float32_to_bfloat16(k5[0]);
                    g00[6] = float32_to_bfloat16(k6[0]);
                    g00[7] = float32_to_bfloat16(k7[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    k4 += kernel_w;
                    k5 += kernel_w;
                    k6 += kernel_w;
                    k7 += kernel_w;
                    g00 += 8;
                }
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
            kptr2 += kernel_w * 4;
            kptr3 += kernel_w * 4;
            kptr4 += kernel_w * 4;
            kptr5 += kernel_w * 4;
            kptr6 += kernel_w * 4;
            kptr7 += kernel_w * 4;
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    g00[2] = float32_to_bfloat16(k2[0]);
                    g00[3] = float32_to_bfloat16(k3[0]);
                    g00[4] = float32_to_bfloat16(k4[0]);
                    g00[5] = float32_to_bfloat16(k5[0]);
                    g00[6] = float32_to_bfloat16(k6[0]);
                    g00[7] = float32_to_bfloat16(k7[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    k4 += kernel_w;
                    k5 += kernel_w;
                    k6 += kernel_w;
                    k7 += kernel_w;
                    g00 += 8;
                }
            }

            kptr0 += kernel_w * 2;
            kptr1 += kernel_w * 2;
            kptr2 += kernel_w * 2;
            kptr3 += kernel_w * 2;
            kptr4 += kernel_w * 2;
            kptr5 += kernel_w * 2;
            kptr6 += kernel_w * 2;
            kptr7 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00[1] = float32_to_bfloat16(kptr1[k]);
                g00[2] = float32_to_bfloat16(kptr2[k]);
                g00[3] = float32_to_bfloat16(kptr3[k]);
                g00[4] = float32_to_bfloat16(kptr4[k]);
                g00[5] = float32_to_bfloat16(kptr5[k]);
                g00[6] = float32_to_bfloat16(kptr6[k]);
                g00[7] = float32_to_bfloat16(kptr7[k]);
                g00 += 8;
            }

            kptr0 += kernel_w;
            kptr1 += kernel_w;
            kptr2 += kernel_w;
            kptr3 += kernel_w;
            kptr4 += kernel_w;
            kptr5 += kernel_w;
            kptr6 += kernel_w;
            kptr7 += kernel_w;
        }
    }

    for (; q + 3 < outh; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;

        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        int p = 0;
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    g00[2] = float32_to_bfloat16(k2[0]);
                    g00[3] = float32_to_bfloat16(k3[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }

            kptr0 += kernel_w * 8;
            kptr1 += kernel_w * 8;
            kptr2 += kernel_w * 8;
            kptr3 += kernel_w * 8;
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    g00[2] = float32_to_bfloat16(k2[0]);
                    g00[3] = float32_to_bfloat16(k3[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
            kptr2 += kernel_w * 4;
            kptr3 += kernel_w * 4;
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    g00[2] = float32_to_bfloat16(k2[0]);
                    g00[3] = float32_to_bfloat16(k3[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }

            kptr0 += kernel_w * 2;
            kptr1 += kernel_w * 2;
            kptr2 += kernel_w * 2;
            kptr3 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00[1] = float32_to_bfloat16(kptr1[k]);
                g00[2] = float32_to_bfloat16(kptr2[k]);
                g00[3] = float32_to_bfloat16(kptr3[k]);
                g00 += 4;
            }

            kptr0 += kernel_w;
            kptr1 += kernel_w;
            kptr2 += kernel_w;
            kptr3 += kernel_w;
        }
    }
#endif // __mips_msa
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;

#if __mips_msa
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00[1] = float32_to_bfloat16(k0[kernel_w]);
                g00[2] = float32_to_bfloat16(k0[kernel_w * 2]);
                g00[3] = float32_to_bfloat16(k0[kernel_w * 3]);
                g00[4] = float32_to_bfloat16(k0[kernel_w * 4]);
                g00[5] = float32_to_bfloat16(k0[kernel_w * 5]);
                g00[6] = float32_to_bfloat16(k0[kernel_w * 6]);
                g00[7] = float32_to_bfloat16(k0[kernel_w * 7]);
                g00[8] = float32_to_bfloat16(k1[0]);
                g00[9] = float32_to_bfloat16(k1[kernel_w]);
                g00[10] = float32_to_bfloat16(k1[kernel_w * 2]);
                g00[11] = float32_to_bfloat16(k1[kernel_w * 3]);
                g00[12] = float32_to_bfloat16(k1[kernel_w * 4]);
                g00[13] = float32_to_bfloat16(k1[kernel_w * 5]);
                g00[14] = float32_to_bfloat16(k1[kernel_w * 6]);
                g00[15] = float32_to_bfloat16(k1[kernel_w * 7]);
                g00 += 16;
            }

            kptr0 += kernel_w * 8;
            kptr1 += kernel_w * 8;
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00[1] = float32_to_bfloat16(k0[kernel_w]);
                g00[2] = float32_to_bfloat16(k0[kernel_w * 2]);
                g00[3] = float32_to_bfloat16(k0[kernel_w * 3]);
                g00[4] = float32_to_bfloat16(k1[0]);
                g00[5] = float32_to_bfloat16(k1[kernel_w]);
                g00[6] = float32_to_bfloat16(k1[kernel_w * 2]);
                g00[7] = float32_to_bfloat16(k1[kernel_w * 3]);
                g00 += 8;
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    g00 += 2;
                }
            }

            kptr0 += kernel_w * 2;
            kptr1 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00[1] = float32_to_bfloat16(kptr1[k]);
                g00 += 2;
            }

            kptr0 += kernel_w;
            kptr1 += kernel_w;
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;

#if __mips_msa
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += kernel_w;
                    g00 += 1;
                }
            }

            kptr0 += kernel_w * 8;
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += kernel_w;
                    g00 += 1;
                }
            }

            kptr0 += kernel_w * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += kernel_w;
                    g00 += 1;
                }
            }

            kptr0 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00 += 1;
            }

            kptr0 += kernel_w;
        }
    }
}

static void convolution1d_packed_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int elempack = bottom_blob.elempack;
    const int inh = bottom_blob.h * elempack;
    const int N = bottom_blob.w * elempack;

    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;
    const int outh = top_blob.h * out_elempack;
    const int M = top_blob.w * out_elempack;

    const float* bias_data_ptr = bias_data;

    int nn_outh = 0;
    int remain_outh_start = 0;

#if __mips_msa
    nn_outh = outh / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = pp * 8;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        unsigned short* outptr = top_blob.row<unsigned short>(p / out_elempack);

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
                _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p, 0);
                _sum4 = (v4f32)__msa_ld_w(bias_data_ptr + p + 4, 0);
            }

            const unsigned short* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            if (elempack == 8)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 8) + j * stride_w * 8;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 64);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(r0, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                        v8i16 _w89_bf16 = __msa_ld_h(kptr + 32, 0);
                        v4f32 _w8 = (v4f32)__msa_ilvr_h(_w89_bf16, _zero_bf16);
                        v4f32 _w9 = (v4f32)__msa_ilvl_h(_w89_bf16, _zero_bf16);
                        v8i16 _wab_bf16 = __msa_ld_h(kptr + 40, 0);
                        v4f32 _wa = (v4f32)__msa_ilvr_h(_wab_bf16, _zero_bf16);
                        v4f32 _wb = (v4f32)__msa_ilvl_h(_wab_bf16, _zero_bf16);
                        v8i16 _wcd_bf16 = __msa_ld_h(kptr + 48, 0);
                        v4f32 _wc = (v4f32)__msa_ilvr_h(_wcd_bf16, _zero_bf16);
                        v4f32 _wd = (v4f32)__msa_ilvl_h(_wcd_bf16, _zero_bf16);
                        v8i16 _wef_bf16 = __msa_ld_h(kptr + 56, 0);
                        v4f32 _we = (v4f32)__msa_ilvr_h(_wef_bf16, _zero_bf16);
                        v4f32 _wf = (v4f32)__msa_ilvl_h(_wef_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w8, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _wa, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _wc, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _we, (v4f32)__msa_splati_w((v4i32)_r1, 3));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w9, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _wb, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _wd, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _wf, (v4f32)__msa_splati_w((v4i32)_r1, 3));

                        r0 += dilation_w * 8;
                        kptr += 64;
                    }
                }
            }
            if (elempack == 4)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 4) + j * stride_w * 4;
                    const unsigned short* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(r1 + 16);
                        __builtin_prefetch(kptr + 64);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v4f32 _r0 = bfloat2float_msa(r0);
                        v4f32 _r1 = bfloat2float_msa(r1);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                        v8i16 _w89_bf16 = __msa_ld_h(kptr + 32, 0);
                        v4f32 _w8 = (v4f32)__msa_ilvr_h(_w89_bf16, _zero_bf16);
                        v4f32 _w9 = (v4f32)__msa_ilvl_h(_w89_bf16, _zero_bf16);
                        v8i16 _wab_bf16 = __msa_ld_h(kptr + 40, 0);
                        v4f32 _wa = (v4f32)__msa_ilvr_h(_wab_bf16, _zero_bf16);
                        v4f32 _wb = (v4f32)__msa_ilvl_h(_wab_bf16, _zero_bf16);
                        v8i16 _wcd_bf16 = __msa_ld_h(kptr + 48, 0);
                        v4f32 _wc = (v4f32)__msa_ilvr_h(_wcd_bf16, _zero_bf16);
                        v4f32 _wd = (v4f32)__msa_ilvl_h(_wcd_bf16, _zero_bf16);
                        v8i16 _wef_bf16 = __msa_ld_h(kptr + 56, 0);
                        v4f32 _we = (v4f32)__msa_ilvr_h(_wef_bf16, _zero_bf16);
                        v4f32 _wf = (v4f32)__msa_ilvl_h(_wef_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w8, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _wa, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _wc, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _we, (v4f32)__msa_splati_w((v4i32)_r1, 3));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w9, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _wb, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _wd, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _wf, (v4f32)__msa_splati_w((v4i32)_r1, 3));

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 64;
                    }
                }
            }
            if (elempack == 1)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 64);

                        unsigned short tmpbuf[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(tmpbuf, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);
                        v8i16 _w89_bf16 = __msa_ld_h(kptr + 32, 0);
                        v4f32 _w8 = (v4f32)__msa_ilvr_h(_w89_bf16, _zero_bf16);
                        v4f32 _w9 = (v4f32)__msa_ilvl_h(_w89_bf16, _zero_bf16);
                        v8i16 _wab_bf16 = __msa_ld_h(kptr + 40, 0);
                        v4f32 _wa = (v4f32)__msa_ilvr_h(_wab_bf16, _zero_bf16);
                        v4f32 _wb = (v4f32)__msa_ilvl_h(_wab_bf16, _zero_bf16);
                        v8i16 _wcd_bf16 = __msa_ld_h(kptr + 48, 0);
                        v4f32 _wc = (v4f32)__msa_ilvr_h(_wcd_bf16, _zero_bf16);
                        v4f32 _wd = (v4f32)__msa_ilvl_h(_wcd_bf16, _zero_bf16);
                        v8i16 _wef_bf16 = __msa_ld_h(kptr + 56, 0);
                        v4f32 _we = (v4f32)__msa_ilvr_h(_wef_bf16, _zero_bf16);
                        v4f32 _wf = (v4f32)__msa_ilvl_h(_wef_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w8, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _wa, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _wc, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _we, (v4f32)__msa_splati_w((v4i32)_r1, 3));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w9, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _wb, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _wd, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _wf, (v4f32)__msa_splati_w((v4i32)_r1, 3));

                        r0 += dilation_w;
                        kptr += 64;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 32);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[1])));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[2])));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[3])));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[1])));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[2])));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[3])));

                        r0 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 32);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N])));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N * 2])));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N * 3])));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N])));
                        _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N * 2])));
                        _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N * 3])));

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N])));
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N])));

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _val = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0]));
                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w0);
                        _sum4 = __ncnn_msa_fmadd_w(_sum4, _val, _w1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
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
    remain_outh_start += nn_outh * 8;
    nn_outh = (outh - remain_outh_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 4;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        unsigned short* outptr = top_blob.row<unsigned short>(p / out_elempack);

        for (int j = 0; j < outw; j++)
        {
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);

            if (bias_data_ptr)
            {
                _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p, 0);
            }

            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);

            int q = 0;
            if (elempack == 8)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 8) + j * stride_w * 8;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 32);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(r0, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);

                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w4, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w5, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w6, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w7, (v4f32)__msa_splati_w((v4i32)_r1, 3));

                        r0 += dilation_w * 8;
                        kptr += 32;
                    }
                }
            }
            if (elempack == 4)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 4) + j * stride_w * 4;
                    const unsigned short* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(r1 + 16);
                        __builtin_prefetch(kptr + 32);

                        v4f32 _r0 = bfloat2float_msa(r0);
                        v4f32 _r1 = bfloat2float_msa(r1);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w4, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w5, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w6, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w7, (v4f32)__msa_splati_w((v4i32)_r1, 3));

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 32;
                    }
                }
            }
            if (elempack == 1)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 32);

                        unsigned short tmpbuf[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(tmpbuf, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);

                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                        v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                        v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                        v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                        v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                        v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_splati_w((v4i32)_r0, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_splati_w((v4i32)_r0, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_splati_w((v4i32)_r0, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_splati_w((v4i32)_r0, 3));
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w4, (v4f32)__msa_splati_w((v4i32)_r1, 0));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w5, (v4f32)__msa_splati_w((v4i32)_r1, 1));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w6, (v4f32)__msa_splati_w((v4i32)_r1, 2));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w7, (v4f32)__msa_splati_w((v4i32)_r1, 3));

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 32);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[1])));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[2])));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[3])));

                        r0 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 32);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N])));
                        _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N * 2])));
                        _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N * 3])));

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);

                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0])));
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[N])));

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _val = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[0]));
                        v4f32 _w = bfloat2float_msa(kptr);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
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
    remain_outh_start += nn_outh * 4;
    nn_outh = (outh - remain_outh_start) / 2;
#else // __mips_msa
    nn_outh = (outh - remain_outh_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __mips_msa
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 2;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;

        unsigned short* outptr0 = top_blob.row<unsigned short>(p);
        unsigned short* outptr1 = top_blob.row<unsigned short>(p + 1);

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
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __mips_msa
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            if (elempack == 8)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 8) + j * stride_w * 8;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(r0, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r1, _w1);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w2);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r1, _w3);

                        r0 += dilation_w * 8;
                        kptr += 16;
                    }
                }
            }
            if (elempack == 4)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 4) + j * stride_w * 4;
                    const unsigned short* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(r1 + 16);
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v4f32 _r0 = bfloat2float_msa(r0);
                        v4f32 _r1 = bfloat2float_msa(r1);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r1, _w1);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w2);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r1, _w3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 16;
                    }
                }
            }
            if (elempack == 1)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        unsigned short tmpbuf[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(tmpbuf, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                        v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r1, _w1);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w2);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r1, _w3);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v4f32 _r0 = bfloat2float_msa(r0);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w1);

                        r0 += dilation_w * 4;
                        kptr += 8;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        unsigned short tmpbuf[4] = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v4f32 _r0 = bfloat2float_msa(tmpbuf);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                        _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            sum0 += __msa_reduce_fadd_w(_sum0);
            sum1 += __msa_reduce_fadd_w(_sum1);
#endif // __mips_msa
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        sum0 += bfloat16_to_float32(r0[0]) * bfloat16_to_float32(kptr[0]);
                        sum1 += bfloat16_to_float32(r0[0]) * bfloat16_to_float32(kptr[1]);
                        sum0 += bfloat16_to_float32(r0[N]) * bfloat16_to_float32(kptr[2]);
                        sum1 += bfloat16_to_float32(r0[N]) * bfloat16_to_float32(kptr[3]);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        float val = bfloat16_to_float32(r0[0]);
                        sum0 += val * bfloat16_to_float32(kptr[0]);
                        sum1 += val * bfloat16_to_float32(kptr[1]);

                        r0 += dilation_w;
                        kptr += 2;
                    }
                }
            }

            sum0 = activation_ss(sum0, activation_type, activation_params);
            sum1 = activation_ss(sum1, activation_type, activation_params);

            outptr0[0] = float32_to_bfloat16(sum0);
            outptr1[0] = float32_to_bfloat16(sum1);
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outh_start += nn_outh * 2;
    for (int p = remain_outh_start; p < outh; p++)
    {
        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;

        unsigned short* outptr = top_blob.row<unsigned short>(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
            {
                sum = bias_data_ptr[p];
            }

#if __mips_msa
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __mips_msa
            v4f32 _sum = (v4f32)__msa_fill_w(0);
            if (elempack == 8)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 8) + j * stride_w * 8;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(r0, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w0);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r1, _w1);

                        r0 += dilation_w * 8;
                        kptr += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / 4) + j * stride_w * 4;
                    const unsigned short* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(r1 + 16);
                        __builtin_prefetch(kptr + 16);

                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v4f32 _r0 = bfloat2float_msa(r0);
                        v4f32 _r1 = bfloat2float_msa(r1);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w0);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r1, _w1);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 8;
                    }
                }
            }
            if (elempack == 1)
            {
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        unsigned short tmpbuf[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                        v8i16 _zero_bf16 = __msa_fill_h(0);
                        v8i16 _r01_bf16 = __msa_ld_h(tmpbuf, 0);
                        v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                        v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                        v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                        v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                        v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w0);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r1, _w1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v4f32 _r0 = bfloat2float_msa(r0);
                        v4f32 _w = bfloat2float_msa(kptr);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w);

                        r0 += dilation_w * 4;
                        kptr += 4;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        unsigned short tmpbuf[4] = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        v4f32 _r0 = bfloat2float_msa(tmpbuf);
                        v4f32 _w = bfloat2float_msa(kptr);
                        _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        sum += bfloat16_to_float32(r0[0]) * bfloat16_to_float32(kptr[0]);
                        sum += bfloat16_to_float32(r0[N]) * bfloat16_to_float32(kptr[1]);

                        r0 += dilation_w;
                        kptr += 2;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        float val = bfloat16_to_float32(r0[0]);
                        sum += val * bfloat16_to_float32(kptr[0]);

                        r0 += dilation_w;
                        kptr += 1;
                    }
                }
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[0] = float32_to_bfloat16(sum);
            outptr += 1;
        }
    }
}
