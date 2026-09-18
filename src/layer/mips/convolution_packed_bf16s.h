// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CONVOLUTION_PACKED_MIPS_BF16S_H
#define CONVOLUTION_PACKED_MIPS_BF16S_H

static void convolution_transform_kernel_packed_bf16s(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __mips_msa
    if (outch >= 8)
    {
        if (inch >= 8)
            kernel_tm.create(8 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 4)
            kernel_tm.create(8 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 2)
            kernel_tm.create(8 * 2 * maxk, inch / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else
            kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
    }
    else if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(4 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 4)
            kernel_tm.create(4 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 2)
            kernel_tm.create(4 * 2 * maxk, inch / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else
            kernel_tm.create(4 * maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
    }
    else
#endif // __mips_msa
    if (outch >= 2)
    {
#if __mips_msa
        if (inch >= 8)
            kernel_tm.create(2 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 4)
            kernel_tm.create(2 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else
#endif // __mips_msa
        if (inch >= 2)
            kernel_tm.create(2 * 2 * maxk, inch / 2 + inch % 2, outch / 2 + outch % 2, (size_t)2u, 1);
        else
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)2u, 1);
    }
    else
    {
#if __mips_msa
        if (inch >= 8)
            kernel_tm.create(8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else
#endif // __mips_msa
        if (inch >= 2)
            kernel_tm.create(2 * maxk, inch / 2 + inch % 2, outch, (size_t)2u, 1);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)2u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
    for (; q + 7 < outch; q += 8)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;
        const float* kptr4 = (const float*)kernel + (q + 4) * inch * maxk;
        const float* kptr5 = (const float*)kernel + (q + 5) * inch * maxk;
        const float* kptr6 = (const float*)kernel + (q + 6) * inch * maxk;
        const float* kptr7 = (const float*)kernel + (q + 7) * inch * maxk;

        unsigned short* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
            kptr4 += maxk * 8;
            kptr5 += maxk * 8;
            kptr6 += maxk * 8;
            kptr7 += maxk * 8;
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
            kptr2 += maxk * 4;
            kptr3 += maxk * 4;
            kptr4 += maxk * 4;
            kptr5 += maxk * 4;
            kptr6 += maxk * 4;
            kptr7 += maxk * 4;
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
            kptr4 += maxk * 2;
            kptr5 += maxk * 2;
            kptr6 += maxk * 2;
            kptr7 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
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

            kptr0 += maxk;
            kptr1 += maxk;
            kptr2 += maxk;
            kptr3 += maxk;
            kptr4 += maxk;
            kptr5 += maxk;
            kptr6 += maxk;
            kptr7 += maxk;
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;

        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
            kptr2 += maxk * 4;
            kptr3 += maxk * 4;
        }

        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00[1] = float32_to_bfloat16(k1[0]);
                g00[2] = float32_to_bfloat16(k2[0]);
                g00[3] = float32_to_bfloat16(k3[0]);
                g00 += 4;
            }
        }
    }
#endif // __mips_msa
    for (; q + 1 < outch; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;

#if __mips_msa
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00[1] = float32_to_bfloat16(k0[maxk]);
                g00[2] = float32_to_bfloat16(k0[maxk * 2]);
                g00[3] = float32_to_bfloat16(k0[maxk * 3]);
                g00[4] = float32_to_bfloat16(k0[maxk * 4]);
                g00[5] = float32_to_bfloat16(k0[maxk * 5]);
                g00[6] = float32_to_bfloat16(k0[maxk * 6]);
                g00[7] = float32_to_bfloat16(k0[maxk * 7]);
                g00[8] = float32_to_bfloat16(k1[0]);
                g00[9] = float32_to_bfloat16(k1[maxk]);
                g00[10] = float32_to_bfloat16(k1[maxk * 2]);
                g00[11] = float32_to_bfloat16(k1[maxk * 3]);
                g00[12] = float32_to_bfloat16(k1[maxk * 4]);
                g00[13] = float32_to_bfloat16(k1[maxk * 5]);
                g00[14] = float32_to_bfloat16(k1[maxk * 6]);
                g00[15] = float32_to_bfloat16(k1[maxk * 7]);
                g00 += 16;
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00[1] = float32_to_bfloat16(k0[maxk]);
                g00[2] = float32_to_bfloat16(k0[maxk * 2]);
                g00[3] = float32_to_bfloat16(k0[maxk * 3]);
                g00[4] = float32_to_bfloat16(k1[0]);
                g00[5] = float32_to_bfloat16(k1[maxk]);
                g00[6] = float32_to_bfloat16(k1[maxk * 2]);
                g00[7] = float32_to_bfloat16(k1[maxk * 3]);
                g00 += 8;
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    g00[1] = float32_to_bfloat16(k1[0]);
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00[1] = float32_to_bfloat16(k1[0]);
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const float* kptr = (const float*)kernel + q * inch * maxk;

#if __mips_msa
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 8;
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                g00[0] = float32_to_bfloat16(k0[0]);
                g00++;
            }
        }
    }
}

static void convolution_packed_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const size_t N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const size_t M = top_blob.cstep * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data;

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __mips_msa
    nn_outch = outch / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 8;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
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
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p, 0);
                    _sum4 = (v4f32)__msa_ld_w(bias_data_ptr + p + 4, 0);
                }

                const unsigned short* kptr = weight_data_tm.channel(p / 8);

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k];
                            __builtin_prefetch(r0s + 16);
                            __builtin_prefetch(kptr + 64);

                            v4f32 _val0 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[0]));
                            v4f32 _val1 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[1]));
                            v4f32 _val2 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[2]));
                            v4f32 _val3 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[3]));
                            v4f32 _val4 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[4]));
                            v4f32 _val5 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[5]));
                            v4f32 _val6 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[6]));
                            v4f32 _val7 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[7]));

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
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, _val1);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, _val1);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, _val2);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, _val2);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, _val3);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, _val3);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w8, _val4);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w9, _val4);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _wa, _val5);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _wb, _val5);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _wc, _val6);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _wd, _val6);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _we, _val7);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _wf, _val7);

                            kptr += 64;
                        }
                    }
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k];
                            const unsigned short* r1s = r1 + space_ofs[k];
                            __builtin_prefetch(r1s + 16);
                            __builtin_prefetch(kptr + 64);

                            v4f32 _val0 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[0]));
                            v4f32 _val1 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[1]));
                            v4f32 _val2 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[2]));
                            v4f32 _val3 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[3]));
                            v4f32 _val4 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[0]));
                            v4f32 _val5 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[1]));
                            v4f32 _val6 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[2]));
                            v4f32 _val7 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[3]));

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
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, _val1);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, _val1);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, _val2);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, _val2);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, _val3);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, _val3);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w8, _val4);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w9, _val4);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _wa, _val5);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _wb, _val5);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _wc, _val6);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _wd, _val6);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _we, _val7);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _wf, _val7);

                            kptr += 64;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 64);

                            v4f32 _val0 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok]));
                            v4f32 _val1 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N]));
                            v4f32 _val2 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 2]));
                            v4f32 _val3 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 3]));
                            v4f32 _val4 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 4]));
                            v4f32 _val5 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 5]));
                            v4f32 _val6 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 6]));
                            v4f32 _val7 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 7]));

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
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, _val1);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, _val1);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, _val2);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, _val2);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, _val3);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, _val3);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w8, _val4);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w9, _val4);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _wa, _val5);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _wb, _val5);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _wc, _val6);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _wd, _val6);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _we, _val7);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _wf, _val7);

                            kptr += 64;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k];
                            __builtin_prefetch(r0s + 16);
                            __builtin_prefetch(kptr + 32);

                            v4f32 _val0 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[0]));
                            v4f32 _val1 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[1]));
                            v4f32 _val2 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[2]));
                            v4f32 _val3 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[3]));

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
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, _val1);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, _val1);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, _val2);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, _val2);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, _val3);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, _val3);

                            kptr += 32;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 32);

                            v4f32 _val0 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok]));
                            v4f32 _val1 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N]));
                            v4f32 _val2 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 2]));
                            v4f32 _val3 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 3]));

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
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, _val1);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, _val1);
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w4, _val2);
                            _sum6 = __ncnn_msa_fmadd_w(_sum6, _w5, _val2);
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w6, _val3);
                            _sum7 = __ncnn_msa_fmadd_w(_sum7, _w7, _val3);

                            kptr += 32;
                        }
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            v4f32 _val0 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok]));
                            v4f32 _val1 = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N]));

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                            v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                            v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val0);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w2, _val1);
                            _sum5 = __ncnn_msa_fmadd_w(_sum5, _w3, _val1);

                            kptr += 16;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __builtin_prefetch(kptr + 16);

                            v4f32 _val = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[space_ofs[k]]));
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, _val);
                            _sum4 = __ncnn_msa_fmadd_w(_sum4, _w1, _val);

                            kptr += 8;
                        }
                    }
                }

                _sum0 = __msa_fadd_w(_sum0, _sum1);
                _sum2 = __msa_fadd_w(_sum2, _sum3);
                _sum0 = __msa_fadd_w(_sum0, _sum2);
                _sum4 = __msa_fadd_w(_sum4, _sum5);
                _sum6 = __msa_fadd_w(_sum6, _sum7);
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
                    float sum4[4];
                    __msa_st_w((v4i32)_sum0, sum0, 0);
                    __msa_st_w((v4i32)_sum4, sum4, 0);

                    outptr[0] = float32_to_bfloat16(sum0[0]);
                    outptr[M] = float32_to_bfloat16(sum0[1]);
                    outptr[M * 2] = float32_to_bfloat16(sum0[2]);
                    outptr[M * 3] = float32_to_bfloat16(sum0[3]);
                    outptr[M * 4] = float32_to_bfloat16(sum4[0]);
                    outptr[M * 5] = float32_to_bfloat16(sum4[1]);
                    outptr[M * 6] = float32_to_bfloat16(sum4[2]);
                    outptr[M * 7] = float32_to_bfloat16(sum4[3]);
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
        const int p = remain_outch_start + pp * 4;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
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
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p, 0);
                }

                const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k];
                            __builtin_prefetch(r0s + 16);
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

                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[0])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[1])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[2])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[3])));
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w4, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[4])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w5, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[5])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w6, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[6])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w7, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[7])));

                            kptr += 32;
                        }
                    }
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k];
                            const unsigned short* r1s = r1 + space_ofs[k];
                            __builtin_prefetch(r1s + 16);
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

                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[0])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[1])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[2])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[3])));
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w4, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[0])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w5, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[1])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w6, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[2])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w7, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r1s[3])));

                            kptr += 32;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
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

                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 2])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 3])));
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w4, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 4])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w5, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 5])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w6, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 6])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w7, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 7])));

                            kptr += 32;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k];
                            __builtin_prefetch(r0s + 16);
                            __builtin_prefetch(kptr + 32);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                            v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                            v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);

                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[0])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[1])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[2])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0s[3])));

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 32);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                            v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                            v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);

                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N])));
                            _sum2 = __ncnn_msa_fmadd_w(_sum2, _w2, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 2])));
                            _sum3 = __ncnn_msa_fmadd_w(_sum3, _w3, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N * 3])));

                            kptr += 16;
                        }
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);

                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _w0, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok])));
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _w1, (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[sok + N])));

                            kptr += 8;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __builtin_prefetch(kptr + 16);

                            v4f32 _val = (v4f32)__msa_fill_w_f32(bfloat16_to_float32(r0[space_ofs[k]]));
                            v4f32 _w = bfloat2float_msa(kptr);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w);

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
    }
    remain_outch_start += nn_outch * 4;
    nn_outch = (outch - remain_outch_start) / 2;
#else // __mips_msa
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __mips_msa
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        unsigned short* outptr0 = top_blob.channel(p);
        unsigned short* outptr1 = top_blob.channel(p + 1);

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
                const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
                const unsigned short* kptr = weight_data_tm.channel(p / 2);
#endif

                int q = 0;
#if __mips_msa
                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                for (; q + 7 < inch; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(r0 + sok + 16);
                            __builtin_prefetch(kptr + 16);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _r01_bf16 = __msa_ld_h(r0 + sok, 0);
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

                            kptr += 16;
                        }
                    }
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(r1 + sok + 16);
                            __builtin_prefetch(kptr + 16);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v4f32 _r0 = bfloat2float_msa(r0 + sok);
                            v4f32 _r1 = bfloat2float_msa(r1 + sok);
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

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            unsigned short tmpbuf0[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            unsigned short tmpbuf1[4] = {r0[sok + N * 4], r0[sok + N * 5], r0[sok + N * 6], r0[sok + N * 7]};
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v4f32 _r0 = bfloat2float_msa(tmpbuf0);
                            v4f32 _r1 = bfloat2float_msa(tmpbuf1);
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

                            kptr += 16;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(r0 + sok + 16);
                            __builtin_prefetch(kptr + 16);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v4f32 _r0 = bfloat2float_msa(r0 + sok);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w1);

                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            unsigned short tmpbuf[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v4f32 _r0 = bfloat2float_msa(tmpbuf);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum0 = __ncnn_msa_fmadd_w(_sum0, _r0, _w0);
                            _sum1 = __ncnn_msa_fmadd_w(_sum1, _r0, _w1);

                            kptr += 8;
                        }
                    }
                }
                sum0 += __msa_reduce_fadd_w(_sum0);
                sum1 += __msa_reduce_fadd_w(_sum1);
#endif // __mips_msa
                for (; q + 1 < inch; q += 2)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            sum0 += bfloat16_to_float32(r0[sok]) * bfloat16_to_float32(kptr[0]);
                            sum1 += bfloat16_to_float32(r0[sok]) * bfloat16_to_float32(kptr[1]);
                            sum0 += bfloat16_to_float32(r0[sok + N]) * bfloat16_to_float32(kptr[2]);
                            sum1 += bfloat16_to_float32(r0[sok + N]) * bfloat16_to_float32(kptr[3]);

                            kptr += 4;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __builtin_prefetch(kptr + 16);

                            float val = bfloat16_to_float32(r0[space_ofs[k]]);
                            sum0 += val * bfloat16_to_float32(kptr[0]);
                            sum1 += val * bfloat16_to_float32(kptr[1]);

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
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        unsigned short* outptr = top_blob.channel(p);

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
                const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
                const unsigned short* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

                int q = 0;
#if __mips_msa
                v4f32 _sum = (v4f32)__msa_fill_w(0);
                for (; q + 7 < inch; q += 8)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(r0 + sok + 16);
                            __builtin_prefetch(kptr + 16);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v8i16 _r01_bf16 = __msa_ld_h(r0 + sok, 0);
                            v4f32 _r0 = (v4f32)__msa_ilvr_h(_r01_bf16, _zero_bf16);
                            v4f32 _r1 = (v4f32)__msa_ilvl_h(_r01_bf16, _zero_bf16);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w0);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r1, _w1);

                            kptr += 8;
                        }
                    }
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(r1 + sok + 16);
                            __builtin_prefetch(kptr + 16);

                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v4f32 _r0 = bfloat2float_msa(r0 + sok);
                            v4f32 _r1 = bfloat2float_msa(r1 + sok);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w0);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r1, _w1);

                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            unsigned short tmpbuf0[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            unsigned short tmpbuf1[4] = {r0[sok + N * 4], r0[sok + N * 5], r0[sok + N * 6], r0[sok + N * 7]};
                            v8i16 _zero_bf16 = __msa_fill_h(0);
                            v4f32 _r0 = bfloat2float_msa(tmpbuf0);
                            v4f32 _r1 = bfloat2float_msa(tmpbuf1);
                            v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                            v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                            v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w0);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r1, _w1);

                            kptr += 8;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(r0 + sok + 16);
                            __builtin_prefetch(kptr + 16);

                            v4f32 _r0 = bfloat2float_msa(r0 + sok);
                            v4f32 _w = bfloat2float_msa(kptr);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w);

                            kptr += 4;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            unsigned short tmpbuf[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            v4f32 _r0 = bfloat2float_msa(tmpbuf);
                            v4f32 _w = bfloat2float_msa(kptr);
                            _sum = __ncnn_msa_fmadd_w(_sum, _r0, _w);

                            kptr += 4;
                        }
                    }
                }
                sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
                for (; q + 1 < inch; q += 2)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __builtin_prefetch(kptr + 16);

                            sum += bfloat16_to_float32(r0[sok]) * bfloat16_to_float32(kptr[0]);
                            sum += bfloat16_to_float32(r0[sok + N]) * bfloat16_to_float32(kptr[1]);

                            kptr += 2;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __builtin_prefetch(kptr + 16);

                            float val = bfloat16_to_float32(r0[space_ofs[k]]);
                            sum += val * bfloat16_to_float32(kptr[0]);

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
}

#endif // CONVOLUTION_PACKED_MIPS_BF16S_H
