// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution1d_transform_kernel_packed_bf16s(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    // src = kernel(outh, inh, kernel_w)
    // dst = pb-pa-kw-inh/pa-outh/pb

    // clang-format off
    // *INDENT-OFF*
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
    if (outh >= 4)
    {
#if __loongarch_asx
        if (inh >= 8)
            kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else
#endif // __loongarch_asx
        if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
    }
    else
#endif // __loongarch_sx
    if (outh >= 2)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (inh >= 8)
            kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
#endif // __loongarch_asx
        if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
#endif // __loongarch_sx
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2, (size_t)2u);
    }
    else
    {
#if __loongarch_sx
#if __loongarch_asx
        if (inh >= 8)
            kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh, (size_t)2u);
        else
#endif // __loongarch_asx
        if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh, (size_t)2u);
        else
#endif // __loongarch_sx
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh, (size_t)2u);
        else
            kernel_tm.create(kernel_w, inh, outh, (size_t)2u);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
    for (; q + 3 < outh; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;

#if __loongarch_asx
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;

#if __loongarch_sx
#if __loongarch_asx
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#endif
#else
        unsigned short* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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

#if __loongarch_sx
#if __loongarch_asx
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#endif
#else
        unsigned short* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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

#if __loongarch_sx
#if __loongarch_asx
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
            __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);

            if (bias_data_ptr)
            {
                _sum0 = (__m256)__lasx_xvld(bias_data_ptr + p, 0);
            }

            const unsigned short* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            for (; q + 7 < inh; q += 8)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                        __m256 _w2 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 16, 0));
                        __m256 _w3 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 24, 0));
                        __m256 _w4 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 32, 0));
                        __m256 _w5 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 40, 0));
                        __m256 _w6 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 48, 0));
                        __m256 _w7 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 56, 0));

                        _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[1])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[2])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[3])), _sum3);
                        _sum0 = __lasx_xvfmadd_s(_w4, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[4])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w5, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[5])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w6, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[6])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w7, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[7])), _sum3);

                        r0 += dilation_w * 8;
                        kptr += 64;
                    }
                }
                if (elempack == 4)
                {
                    const unsigned short* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                        __m256 _w2 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 16, 0));
                        __m256 _w3 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 24, 0));
                        __m256 _w4 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 32, 0));
                        __m256 _w5 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 40, 0));
                        __m256 _w6 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 48, 0));
                        __m256 _w7 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 56, 0));

                        _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[1])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[2])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[3])), _sum3);
                        _sum0 = __lasx_xvfmadd_s(_w4, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w5, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1[1])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w6, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1[2])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w7, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1[3])), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 64;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                        __m256 _w2 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 16, 0));
                        __m256 _w3 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 24, 0));
                        __m256 _w4 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 32, 0));
                        __m256 _w5 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 40, 0));
                        __m256 _w6 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 48, 0));
                        __m256 _w7 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 56, 0));

                        _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 2])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 3])), _sum3);
                        _sum0 = __lasx_xvfmadd_s(_w4, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 4])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w5, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 5])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w6, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 6])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w7, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 7])), _sum3);

                        r0 += dilation_w;
                        kptr += 64;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 8 || elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        const unsigned short* r0s = r0 + (q % elempack);

                        __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                        __m256 _w2 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 16, 0));
                        __m256 _w3 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 24, 0));

                        _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[1])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[2])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[3])), _sum3);

                        r0 += dilation_w * elempack;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                        __m256 _w2 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 16, 0));
                        __m256 _w3 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 24, 0));

                        _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N])), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 2])), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N * 3])), _sum3);

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
                        __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));

                        _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[N])), _sum1);

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
                        __m256 _val = (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[0]));
                        __m256 _w = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                        _sum0 = __lasx_xvfmadd_s(_w, _val, _sum0);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }

            _sum0 = __lasx_xvfadd_s(_sum0, _sum1);
            _sum2 = __lasx_xvfadd_s(_sum2, _sum3);
            _sum0 = __lasx_xvfadd_s(_sum0, _sum2);

            _sum0 = activation_lasx(_sum0, activation_type, activation_params);

            if (out_elempack == 8)
            {
                __lsx_vst(float2bfloat_lasx(_sum0), outptr, 0);
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                __m128i _bf16 = float2bfloat_lasx(_sum0);
                __lsx_vstelm_d(_bf16, outptr, 0, 0);
                __lsx_vstelm_d(_bf16, outptr + M, 0, 1);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                float sum[8];
                __lasx_xvst(_sum0, sum, 0);

                outptr[0] = float32_to_bfloat16(sum[0]);
                outptr[M] = float32_to_bfloat16(sum[1]);
                outptr[M * 2] = float32_to_bfloat16(sum[2]);
                outptr[M * 3] = float32_to_bfloat16(sum[3]);
                outptr[M * 4] = float32_to_bfloat16(sum[4]);
                outptr[M * 5] = float32_to_bfloat16(sum[5]);
                outptr[M * 6] = float32_to_bfloat16(sum[6]);
                outptr[M * 7] = float32_to_bfloat16(sum[7]);
                outptr += 1;
            }
        }
    }
    remain_outh_start += nn_outh * 8;
    nn_outh = (outh - remain_outh_start) / 4;
#else  // __loongarch_asx
    nn_outh = outh / 4;
#endif // __loongarch_asx
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
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

            if (bias_data_ptr)
            {
                _sum0 = (__m128)__lsx_vld(bias_data_ptr + p, 0);
            }

#if __loongarch_asx
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __loongarch_asx
            for (; q + 7 < inh; q += 8)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        __m128 _w2 = bfloat2float_lsx(kptr + 8);
                        __m128 _w3 = bfloat2float_lsx(kptr + 12);
                        __m128 _w4 = bfloat2float_lsx(kptr + 16);
                        __m128 _w5 = bfloat2float_lsx(kptr + 20);
                        __m128 _w6 = bfloat2float_lsx(kptr + 24);
                        __m128 _w7 = bfloat2float_lsx(kptr + 28);

                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[1])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[2])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[3])), _sum3);
                        _sum0 = __lsx_vfmadd_s(_w4, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[4])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w5, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[5])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w6, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[6])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w7, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[7])), _sum3);

                        r0 += dilation_w * 8;
                        kptr += 32;
                    }
                }
                if (elempack == 4)
                {
                    const unsigned short* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        __m128 _w2 = bfloat2float_lsx(kptr + 8);
                        __m128 _w3 = bfloat2float_lsx(kptr + 12);
                        __m128 _w4 = bfloat2float_lsx(kptr + 16);
                        __m128 _w5 = bfloat2float_lsx(kptr + 20);
                        __m128 _w6 = bfloat2float_lsx(kptr + 24);
                        __m128 _w7 = bfloat2float_lsx(kptr + 28);

                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[1])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[2])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[3])), _sum3);
                        _sum0 = __lsx_vfmadd_s(_w4, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r1[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w5, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r1[1])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w6, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r1[2])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w7, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r1[3])), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        __m128 _w2 = bfloat2float_lsx(kptr + 8);
                        __m128 _w3 = bfloat2float_lsx(kptr + 12);
                        __m128 _w4 = bfloat2float_lsx(kptr + 16);
                        __m128 _w5 = bfloat2float_lsx(kptr + 20);
                        __m128 _w6 = bfloat2float_lsx(kptr + 24);
                        __m128 _w7 = bfloat2float_lsx(kptr + 28);

                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 2])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 3])), _sum3);
                        _sum0 = __lsx_vfmadd_s(_w4, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 4])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w5, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 5])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w6, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 6])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w7, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 7])), _sum3);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
#endif // __loongarch_asx
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 8 || elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        const unsigned short* r0s = r0 + (q % elempack);

                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        __m128 _w2 = bfloat2float_lsx(kptr + 8);
                        __m128 _w3 = bfloat2float_lsx(kptr + 12);

                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[1])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[2])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[3])), _sum3);

                        r0 += dilation_w * elempack;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        __m128 _w2 = bfloat2float_lsx(kptr + 8);
                        __m128 _w3 = bfloat2float_lsx(kptr + 12);

                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N])), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 2])), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N * 3])), _sum3);

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
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);

                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[0])), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[N])), _sum1);

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
                        __m128 _val = (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[0]));
                        __m128 _w = bfloat2float_lsx(kptr);
                        _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }

            _sum0 = __lsx_vfadd_s(_sum0, _sum1);
            _sum2 = __lsx_vfadd_s(_sum2, _sum3);
            _sum0 = __lsx_vfadd_s(_sum0, _sum2);

            _sum0 = activation_lsx(_sum0, activation_type, activation_params);

            if (out_elempack == 4)
            {
                __lsx_vstelm_d(float2bfloat_lsx(_sum0), outptr, 0, 0);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                float sum[4];
                __lsx_vst(_sum0, sum, 0);

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
#else  // __loongarch_sx
    nn_outh = (outh - remain_outh_start) / 2;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(opt.num_threads)
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

#if __loongarch_sx
#if __loongarch_asx
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#endif
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __loongarch_sx
#if __loongarch_asx
            {
                __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(r0, 0));
                            __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                            _sum0 = __lasx_xvfmadd_s(_r0, _w0, _sum0);
                            _sum1 = __lasx_xvfmadd_s(_r0, _w1, _sum1);

                            r0 += dilation_w * 8;
                            kptr += 16;
                        }
                    }
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m128 _r0 = bfloat2float_lsx(r0);
                            __m128 _r1 = bfloat2float_lsx(r1);
                            __m256 _r01 = __lasx_concat_128_s(_r0, _r1);
                            __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                            _sum0 = __lasx_xvfmadd_s(_r01, _w0, _sum0);
                            _sum1 = __lasx_xvfmadd_s(_r01, _w1, _sum1);

                            r0 += dilation_w * 4;
                            r1 += dilation_w * 4;
                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < kernel_w; k++)
                        {
                            unsigned short tmpbuf[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                            __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(tmpbuf, 0));
                            __m256 _w0 = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_lasx((__m128i)__lsx_vld(kptr + 8, 0));
                            _sum0 = __lasx_xvfmadd_s(_r0, _w0, _sum0);
                            _sum1 = __lasx_xvfmadd_s(_r0, _w1, _sum1);

                            r0 += dilation_w;
                            kptr += 16;
                        }
                    }
                }
                __m128 _sum0_128 = __lsx_vfadd_s(__lasx_extract_128_lo_s(_sum0), __lasx_extract_128_hi_s(_sum0));
                __m128 _sum1_128 = __lsx_vfadd_s(__lasx_extract_128_lo_s(_sum1), __lasx_extract_128_hi_s(_sum1));
                sum0 += __lsx_reduce_fadd_s(_sum0_128);
                sum1 += __lsx_reduce_fadd_s(_sum1_128);
            }
#endif // __loongarch_asx
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 8 || elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = bfloat2float_lsx(r0 + (q % elempack));
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        _sum0 = __lsx_vfmadd_s(_r0, _w0, _sum0);
                        _sum1 = __lsx_vfmadd_s(_r0, _w1, _sum1);

                        r0 += dilation_w * elempack;
                        kptr += 8;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        unsigned short tmpbuf[4] = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        __m128 _r0 = bfloat2float_lsx(tmpbuf);
                        __m128 _w0 = bfloat2float_lsx(kptr);
                        __m128 _w1 = bfloat2float_lsx(kptr + 4);
                        _sum0 = __lsx_vfmadd_s(_r0, _w0, _sum0);
                        _sum1 = __lsx_vfmadd_s(_r0, _w1, _sum1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            sum0 += __lsx_reduce_fadd_s(_sum0);
            sum1 += __lsx_reduce_fadd_s(_sum1);
#endif // __loongarch_sx
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

#if __loongarch_sx
#if __loongarch_asx
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#endif
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __loongarch_sx
#if __loongarch_asx
            {
                __m256 _sum = (__m256)__lasx_xvreplgr2vr_w(0);
                for (; q + 7 < inh; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(r0, 0));
                            __m256 _w = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                            _sum = __lasx_xvfmadd_s(_r0, _w, _sum);

                            r0 += dilation_w * 8;
                            kptr += 8;
                        }
                    }
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m128 _r0 = bfloat2float_lsx(r0);
                            __m128 _r1 = bfloat2float_lsx(r1);
                            __m256 _r01 = __lasx_concat_128_s(_r0, _r1);
                            __m256 _w = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                            _sum = __lasx_xvfmadd_s(_r01, _w, _sum);

                            r0 += dilation_w * 4;
                            r1 += dilation_w * 4;
                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < kernel_w; k++)
                        {
                            unsigned short tmpbuf[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                            __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(tmpbuf, 0));
                            __m256 _w = bfloat2float_lasx((__m128i)__lsx_vld(kptr, 0));
                            _sum = __lasx_xvfmadd_s(_r0, _w, _sum);

                            r0 += dilation_w;
                            kptr += 8;
                        }
                    }
                }
                __m128 _ss = __lsx_vfadd_s(__lasx_extract_128_lo_s(_sum), __lasx_extract_128_hi_s(_sum));
                sum += __lsx_reduce_fadd_s(_ss);
            }
#endif // __loongarch_asx
            __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                if (elempack == 8 || elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = bfloat2float_lsx(r0 + (q % elempack));
                        __m128 _w = bfloat2float_lsx(kptr);
                        _sum = __lsx_vfmadd_s(_r0, _w, _sum);

                        r0 += dilation_w * elempack;
                        kptr += 4;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        unsigned short tmpbuf[4] = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        __m128 _r0123 = bfloat2float_lsx(tmpbuf);
                        __m128 _w = bfloat2float_lsx(kptr);
                        _sum = __lsx_vfmadd_s(_r0123, _w, _sum);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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
