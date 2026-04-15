// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CONVOLUTION_PACKED_LOONGARCH_BF16S_H
#define CONVOLUTION_PACKED_LOONGARCH_BF16S_H

static void convolution_transform_kernel_packed_bf16s(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __loongarch_sx
#if __loongarch_asx
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
    else
#endif // __loongarch_asx
    if (outch >= 4)
    {
        if (inch >= 4)
            kernel_tm.create(4 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else if (inch >= 2)
            kernel_tm.create(4 * 2 * maxk, inch / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)2u, 1);
    }
    else
#endif // __loongarch_sx
    if (outch >= 2)
    {
#if __loongarch_sx
        if (inch >= 4)
            kernel_tm.create(2 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 2 + outch % 2, (size_t)2u, 1);
        else
#endif // __loongarch_sx
        if (inch >= 2)
            kernel_tm.create(2 * 2 * maxk, inch / 2 + inch % 2, outch / 2 + outch % 2, (size_t)2u, 1);
        else
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)2u, 1);
    }
    else
    {
#if __loongarch_sx
        if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch, (size_t)2u, 1);
        else
#endif // __loongarch_sx
        if (inch >= 2)
            kernel_tm.create(2 * maxk, inch / 2 + inch % 2, outch, (size_t)2u, 1);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)2u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
    for (; q + 3 < outch; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;

#if __loongarch_asx
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
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
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00[1] = float32_to_bfloat16(kptr1[k]);
                g00[2] = float32_to_bfloat16(kptr2[k]);
                g00[3] = float32_to_bfloat16(kptr3[k]);
                g00 += 4;
            }

            kptr0 += maxk;
            kptr1 += maxk;
            kptr2 += maxk;
            kptr3 += maxk;
        }
    }
#endif // __loongarch_sx
    for (; q + 1 < outch; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;

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
#endif // __loongarch_sx
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
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00[1] = float32_to_bfloat16(kptr1[k]);
                g00 += 2;
            }

            kptr0 += maxk;
            kptr1 += maxk;
        }
    }
    for (; q < outch; q++)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;

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
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr0 += maxk * 4;
        }
#endif // __loongarch_sx
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[0]);
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr0 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                g00[0] = float32_to_bfloat16(kptr0[k]);
                g00 += 1;
            }

            kptr0 += maxk;
        }
    }
}

static void convolution_packed_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const int M = top_blob.cstep * out_elempack;

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
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        unsigned short* outptr = top_blob.channel(p / out_elempack);

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

                            __m256 _w0 = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 8, 0));
                            __m256 _w2 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 16, 0));
                            __m256 _w3 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 24, 0));
                            __m256 _w4 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 32, 0));
                            __m256 _w5 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 40, 0));
                            __m256 _w6 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 48, 0));
                            __m256 _w7 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 56, 0));

                            _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[0])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[1])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[2])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[3])), _sum3);
                            _sum0 = __lasx_xvfmadd_s(_w4, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[4])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w5, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[5])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w6, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[6])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w7, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[7])), _sum3);

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

                            __m256 _w0 = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 8, 0));
                            __m256 _w2 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 16, 0));
                            __m256 _w3 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 24, 0));
                            __m256 _w4 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 32, 0));
                            __m256 _w5 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 40, 0));
                            __m256 _w6 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 48, 0));
                            __m256 _w7 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 56, 0));

                            _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[0])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[1])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[2])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[3])), _sum3);
                            _sum0 = __lasx_xvfmadd_s(_w4, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1s[0])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w5, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1s[1])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w6, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1s[2])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w7, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r1s[3])), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m256 _w0 = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 8, 0));
                            __m256 _w2 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 16, 0));
                            __m256 _w3 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 24, 0));
                            __m256 _w4 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 32, 0));
                            __m256 _w5 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 40, 0));
                            __m256 _w6 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 48, 0));
                            __m256 _w7 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 56, 0));

                            _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 2])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 3])), _sum3);
                            _sum0 = __lasx_xvfmadd_s(_w4, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 4])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w5, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 5])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w6, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 6])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w7, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 7])), _sum3);

                            kptr += 64;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8 || elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k] + (q % elempack);

                            __m256 _w0 = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 8, 0));
                            __m256 _w2 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 16, 0));
                            __m256 _w3 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 24, 0));

                            _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[0])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[1])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[2])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0s[3])), _sum3);

                            kptr += 32;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m256 _w0 = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 8, 0));
                            __m256 _w2 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 16, 0));
                            __m256 _w3 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 24, 0));

                            _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N])), _sum1);
                            _sum2 = __lasx_xvfmadd_s(_w2, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 2])), _sum2);
                            _sum3 = __lasx_xvfmadd_s(_w3, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 3])), _sum3);

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

                            __m256 _w0 = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            __m256 _w1 = bfloat2float_avx((__m128i)__lsx_vld(kptr + 8, 0));

                            _sum0 = __lasx_xvfmadd_s(_w0, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok])), _sum0);
                            _sum1 = __lasx_xvfmadd_s(_w1, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[sok + N])), _sum1);

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
                            __m256 _val = (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(r0[space_ofs[k]]));
                            __m256 _w = bfloat2float_avx((__m128i)__lsx_vld(kptr, 0));
                            _sum0 = __lasx_xvfmadd_s(_w, _val, _sum0);

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
                    __lsx_vst(float2bfloat_avx(_sum0), outptr, 0);
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    __m128i _bf16 = float2bfloat_avx(_sum0);
                    __lsx_vstelm_d(_bf16, outptr, 0, 0);
                    __lsx_vstelm_d(_bf16, outptr + M, 0, 1);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[8];
                    __lasx_xvst((__m256i)_sum0, sum, 0);

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
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
#else // __loongarch_asx
    nn_outch = outch / 4;
#endif // __loongarch_asx
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
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8 || elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k] + (q % elempack);

                            __m128 _w0 = bfloat2float_sse(kptr);
                            __m128 _w1 = bfloat2float_sse(kptr + 4);
                            __m128 _w2 = bfloat2float_sse(kptr + 8);
                            __m128 _w3 = bfloat2float_sse(kptr + 12);

                            _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[0])), _sum0);
                            _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[1])), _sum1);
                            _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[2])), _sum2);
                            _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0s[3])), _sum3);

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m128 _w0 = bfloat2float_sse(kptr);
                            __m128 _w1 = bfloat2float_sse(kptr + 4);
                            __m128 _w2 = bfloat2float_sse(kptr + 8);
                            __m128 _w3 = bfloat2float_sse(kptr + 12);

                            _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[sok])), _sum0);
                            _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[sok + N])), _sum1);
                            _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 2])), _sum2);
                            _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[sok + N * 3])), _sum3);

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

                            __m128 _w0 = bfloat2float_sse(kptr);
                            __m128 _w1 = bfloat2float_sse(kptr + 4);

                            _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[sok])), _sum0);
                            _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[sok + N])), _sum1);

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
                            __m128 _val = (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(r0[space_ofs[k]]));
                            __m128 _w = bfloat2float_sse(kptr);
                            _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);

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
                    __lsx_vstelm_d(float2bfloat_sse(_sum0), outptr, 0, 0);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[4];
                    __lsx_vst((__m128i)_sum0, sum, 0);

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
#else // __loongarch_sx
    nn_outch = (outch - remain_outch_start) / 2;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(opt.num_threads)
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
                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8 || elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m128 _r0 = bfloat2float_sse(r0 + sok + (q % elempack));
                            __m128 _w0 = bfloat2float_sse(kptr);
                            __m128 _w1 = bfloat2float_sse(kptr + 4);

                            _sum0 = __lsx_vfmadd_s(_r0, _w0, _sum0);
                            _sum1 = __lsx_vfmadd_s(_r0, _w1, _sum1);

                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            unsigned short tmpbuf[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            __m128 _r0 = bfloat2float_sse(tmpbuf);

                            __m128 _w0 = bfloat2float_sse(kptr);
                            __m128 _w1 = bfloat2float_sse(kptr + 4);

                            _sum0 = __lsx_vfmadd_s(_r0, _w0, _sum0);
                            _sum1 = __lsx_vfmadd_s(_r0, _w1, _sum1);

                            kptr += 8;
                        }
                    }
                }
                sum0 += __lsx_reduce_fadd_s(_sum0);
                sum1 += __lsx_reduce_fadd_s(_sum1);
#endif // __loongarch_sx
                for (; q + 1 < inch; q += 2)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

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
        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

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
                __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);
                for (; q + 3 < inch; q += 4)
                {
                    const unsigned short* r0 = bottom_blob.channel(q / elempack).row<const unsigned short>(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8 || elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const unsigned short* r0s = r0 + space_ofs[k] + (q % elempack);

                            __m128 _r0 = bfloat2float_sse(r0s);
                            __m128 _w = bfloat2float_sse(kptr);
                            _sum = __lsx_vfmadd_s(_r0, _w, _sum);

                            kptr += 4;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            unsigned short tmpbuf[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            __m128 _r0123 = bfloat2float_sse(tmpbuf);

                            __m128 _w = bfloat2float_sse(kptr);
                            _sum = __lsx_vfmadd_s(_r0123, _w, _sum);

                            kptr += 4;
                        }
                    }
                }
                sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
                for (; q + 1 < inch; q += 2)
                {
                    const unsigned short* r0 = bottom_blob.channel(q).row<const unsigned short>(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

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

#endif // CONVOLUTION_PACKED_LOONGARCH_BF16S_H
