// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void convolution_transform_kernel_packed_int8_i8mm(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
void convolution_packed_int8_i8mm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
void convolution_transform_kernel_packed_int8_asimddp(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
void convolution_packed_int8_asimddp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif
#endif

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        convolution_transform_kernel_packed_int8_i8mm(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        convolution_transform_kernel_packed_int8_asimddp(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
        return;
    }
#endif
#endif

    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __ARM_NEON
    if (outch >= 8)
    {
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)64u, 64);
        else
            kernel_tm.create(maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)8u, 8);
    }
    else if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)32u, 32);
        else
            kernel_tm.create(maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)4u, 4);
    }
    else
#endif // __ARM_NEON
    if (outch >= 2)
    {
#if __ARM_NEON
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch / 2 + outch % 2, (size_t)16u, 16);
        else
#endif // __ARM_NEON
            kernel_tm.create(maxk, inch, outch / 2 + outch % 2, (size_t)2u, 2);
    }
    else
    {
#if __ARM_NEON
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch, (size_t)8u, 8);
        else
#endif // __ARM_NEON
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __ARM_NEON
    for (; q + 7 < outch; q += 8)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;
        const signed char* kptr4 = (const signed char*)kernel + (q + 4) * inch * maxk;
        const signed char* kptr5 = (const signed char*)kernel + (q + 5) * inch * maxk;
        const signed char* kptr6 = (const signed char*)kernel + (q + 6) * inch * maxk;
        const signed char* kptr7 = (const signed char*)kernel + (q + 7) * inch * maxk;

        signed char* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;
                const signed char* k4 = kptr4 + k;
                const signed char* k5 = kptr5 + k;
                const signed char* k6 = kptr6 + k;
                const signed char* k7 = kptr7 + k;

#if __ARM_FEATURE_MATMUL_INT8
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    g00 += 1;
                    k0 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k1[0];
                    g00 += 1;
                    k1 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k2[0];
                    g00 += 1;
                    k2 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k3[0];
                    g00 += 1;
                    k3 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k4[0];
                    g00 += 1;
                    k4 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k5[0];
                    g00 += 1;
                    k5 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k6[0];
                    g00 += 1;
                    k6 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k7[0];
                    g00 += 1;
                    k7 += maxk;
                }
#elif __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k0[maxk * 2];
                    g00[3] = k0[maxk * 3];
                    g00[4] = k1[0];
                    g00[5] = k1[maxk];
                    g00[6] = k1[maxk * 2];
                    g00[7] = k1[maxk * 3];
                    g00[8] = k2[0];
                    g00[9] = k2[maxk];
                    g00[10] = k2[maxk * 2];
                    g00[11] = k2[maxk * 3];
                    g00[12] = k3[0];
                    g00[13] = k3[maxk];
                    g00[14] = k3[maxk * 2];
                    g00[15] = k3[maxk * 3];
                    g00[16] = k4[0];
                    g00[17] = k4[maxk];
                    g00[18] = k4[maxk * 2];
                    g00[19] = k4[maxk * 3];
                    g00[20] = k5[0];
                    g00[21] = k5[maxk];
                    g00[22] = k5[maxk * 2];
                    g00[23] = k5[maxk * 3];
                    g00[24] = k6[0];
                    g00[25] = k6[maxk];
                    g00[26] = k6[maxk * 2];
                    g00[27] = k6[maxk * 3];
                    g00[28] = k7[0];
                    g00[29] = k7[maxk];
                    g00[30] = k7[maxk * 2];
                    g00[31] = k7[maxk * 3];
                    g00 += 32;
                    k0 += maxk * 4;
                    k1 += maxk * 4;
                    k2 += maxk * 4;
                    k3 += maxk * 4;
                    k4 += maxk * 4;
                    k5 += maxk * 4;
                    k6 += maxk * 4;
                    k7 += maxk * 4;
                }
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];
                    g00[8] = k4[0];
                    g00[9] = k4[maxk];
                    g00[10] = k5[0];
                    g00[11] = k5[maxk];
                    g00[12] = k6[0];
                    g00[13] = k6[maxk];
                    g00[14] = k7[0];
                    g00[15] = k7[maxk];
                    g00 += 16;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                    k4 += maxk * 2;
                    k5 += maxk * 2;
                    k6 += maxk * 2;
                    k7 += maxk * 2;
                }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;
                const signed char* k4 = kptr4 + k;
                const signed char* k5 = kptr5 + k;
                const signed char* k6 = kptr6 + k;
                const signed char* k7 = kptr7 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00[4] = k4[0];
                g00[5] = k5[0];
                g00[6] = k6[0];
                g00[7] = k7[0];
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
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;

        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

#if __ARM_FEATURE_MATMUL_INT8
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    g00 += 1;
                    k0 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k1[0];
                    g00 += 1;
                    k1 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k2[0];
                    g00 += 1;
                    k2 += maxk;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k3[0];
                    g00 += 1;
                    k3 += maxk;
                }
#elif __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k0[maxk * 2];
                    g00[3] = k0[maxk * 3];
                    g00[4] = k1[0];
                    g00[5] = k1[maxk];
                    g00[6] = k1[maxk * 2];
                    g00[7] = k1[maxk * 3];
                    g00[8] = k2[0];
                    g00[9] = k2[maxk];
                    g00[10] = k2[maxk * 2];
                    g00[11] = k2[maxk * 3];
                    g00[12] = k3[0];
                    g00[13] = k3[maxk];
                    g00[14] = k3[maxk * 2];
                    g00[15] = k3[maxk * 3];
                    g00 += 16;
                    k0 += maxk * 4;
                    k1 += maxk * 4;
                    k2 += maxk * 4;
                    k3 += maxk * 4;
                }
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];
                    g00 += 8;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00 += 4;
            }

            kptr0 += maxk;
            kptr1 += maxk;
            kptr2 += maxk;
            kptr3 += maxk;
        }
    }
#endif // __ARM_NEON
    for (; q + 1 < outch; q += 2)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;

#if __ARM_NEON
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __ARM_NEON
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

#if __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k1[0];
                    k1 += maxk;
                    g00 += 1;
                }
#else  // __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k1[0];
                    k1 += maxk;
                    g00 += 1;
                }

                for (int i = 4; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
                for (int i = 4; i < 8; i++)
                {
                    g00[0] = k1[0];
                    k1 += maxk;
                    g00 += 1;
                }
#endif // __ARM_FEATURE_DOTPROD
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
        }
#endif // __ARM_NEON
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }

            kptr0 += maxk;
            kptr1 += maxk;
        }
    }
    for (; q < outch; q++)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;

#if __ARM_NEON
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __ARM_NEON
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 8;
        }
#endif // __ARM_NEON
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;
                g00[0] = k0[0];
                g00++;
            }

            kptr += maxk;
        }
    }
}

static void convolution_packed_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        convolution_packed_int8_i8mm(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        convolution_packed_int8_asimddp(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif
#endif

    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

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

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __ARM_NEON
    nn_outch = (outch - remain_outch_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 8;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            {
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        int8x8_t _r0;
                        int8x8_t _r1;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                            _r1 = vld1_s8(r1s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp0[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            signed char tmp1[8] = {r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7]};
                            _r0 = vld1_s8(tmp0);
                            _r1 = vld1_s8(tmp1);
                        }

                        int8x16_t _w0 = vld1q_s8(kptr);
                        int8x16_t _w1 = vld1q_s8(kptr + 16);
                        int8x16_t _w2 = vld1q_s8(kptr + 32);
                        int8x16_t _w3 = vld1q_s8(kptr + 48);

#if __ARM_FEATURE_MATMUL_INT8
                        int8x16_t _r01 = vcombine_s8(_r0, _r1);
                        _sum0 = vmmlaq_s32(_sum0, _r01, _w0);
                        _sum1 = vmmlaq_s32(_sum1, _r01, _w1);
                        _sum2 = vmmlaq_s32(_sum2, _r01, _w2);
                        _sum3 = vmmlaq_s32(_sum3, _r01, _w3);
#elif __ARM_FEATURE_DOTPROD
                        _sum0 = vdotq_lane_s32(_sum0, _w0, _r0, 0);
                        _sum1 = vdotq_lane_s32(_sum1, _w1, _r0, 0);
                        _sum2 = vdotq_lane_s32(_sum2, _w0, _r1, 0);
                        _sum3 = vdotq_lane_s32(_sum3, _w1, _r1, 0);
                        _sum0 = vdotq_lane_s32(_sum0, _w2, _r0, 1);
                        _sum1 = vdotq_lane_s32(_sum1, _w3, _r0, 1);
                        _sum2 = vdotq_lane_s32(_sum2, _w2, _r1, 1);
                        _sum3 = vdotq_lane_s32(_sum3, _w3, _r1, 1);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                        int16x4_t _rr0 = vreinterpret_s16_s8(_r0);
                        int16x4_t _rr1 = vreinterpret_s16_s8(_r1);

                        int8x8_t _r0ll = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 0));
                        int8x8_t _r1ll = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 0));
                        int8x8_t _r0hl = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 2));
                        int8x8_t _r1hl = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 2));

                        int16x8_t _s0l = vmull_s8(_r0ll, vget_low_s8(_w0));
                        int16x8_t _s1l = vmull_s8(_r0ll, vget_high_s8(_w0));
                        int16x8_t _s2l = vmull_s8(_r1ll, vget_low_s8(_w0));
                        int16x8_t _s3l = vmull_s8(_r1ll, vget_high_s8(_w0));
                        _s0l = vmlal_s8(_s0l, _r0hl, vget_low_s8(_w2));
                        _s1l = vmlal_s8(_s1l, _r0hl, vget_high_s8(_w2));
                        _s2l = vmlal_s8(_s2l, _r1hl, vget_low_s8(_w2));
                        _s3l = vmlal_s8(_s3l, _r1hl, vget_high_s8(_w2));

                        _sum0 = vpadalq_s16(_sum0, _s0l);
                        _sum1 = vpadalq_s16(_sum1, _s1l);
                        _sum2 = vpadalq_s16(_sum2, _s2l);
                        _sum3 = vpadalq_s16(_sum3, _s3l);

                        int8x8_t _r0lh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 1));
                        int8x8_t _r1lh = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 1));
                        int8x8_t _r0hh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 3));
                        int8x8_t _r1hh = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 3));

                        int16x8_t _s0h = vmull_s8(_r0lh, vget_low_s8(_w1));
                        int16x8_t _s1h = vmull_s8(_r0lh, vget_high_s8(_w1));
                        int16x8_t _s2h = vmull_s8(_r1lh, vget_low_s8(_w1));
                        int16x8_t _s3h = vmull_s8(_r1lh, vget_high_s8(_w1));
                        _s0h = vmlal_s8(_s0h, _r0hh, vget_low_s8(_w3));
                        _s1h = vmlal_s8(_s1h, _r0hh, vget_high_s8(_w3));
                        _s2h = vmlal_s8(_s2h, _r1hh, vget_low_s8(_w3));
                        _s3h = vmlal_s8(_s3h, _r1hh, vget_high_s8(_w3));

                        _sum0 = vpadalq_s16(_sum0, _s0h);
                        _sum1 = vpadalq_s16(_sum1, _s1h);
                        _sum2 = vpadalq_s16(_sum2, _s2h);
                        _sum3 = vpadalq_s16(_sum3, _s3h);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD

                        kptr += 64;
                    }
                }
#if __ARM_FEATURE_MATMUL_INT8
                {
                    int32x4_t _tmp0 = vcombine_s32(vget_low_s32(_sum0), vget_low_s32(_sum1));
                    int32x4_t _tmp1 = vcombine_s32(vget_low_s32(_sum2), vget_low_s32(_sum3));
                    int32x4_t _tmp2 = vcombine_s32(vget_high_s32(_sum0), vget_high_s32(_sum1));
                    int32x4_t _tmp3 = vcombine_s32(vget_high_s32(_sum2), vget_high_s32(_sum3));
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                    _sum2 = _tmp2;
                    _sum3 = _tmp3;
                }
#endif
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        int8x8_t _r0 = vdup_n_s8(r0s[0]);
                        int8x8_t _r1 = vdup_n_s8(r1s[0]);
                        int8x8_t _w = vld1_s8(kptr);
                        int16x8_t _s0 = vmull_s8(_r0, _w);
                        int16x8_t _s1 = vmull_s8(_r1, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                        _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                        _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                        kptr += 8;
                    }
                }
            }

            if (out_elempack == 8)
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                outptr += 16;
            }
            if (out_elempack == 4)
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum2);
                vst1q_s32(outptr + M, _sum1);
                vst1q_s32(outptr + M + 4, _sum3);
                outptr += 8;
            }
            if (out_elempack == 1)
            {
                outptr[0] = vgetq_lane_s32(_sum0, 0);
                outptr[1] = vgetq_lane_s32(_sum2, 0);
                outptr[M] = vgetq_lane_s32(_sum0, 1);
                outptr[M + 1] = vgetq_lane_s32(_sum2, 1);
                outptr[M * 2] = vgetq_lane_s32(_sum0, 2);
                outptr[M * 2 + 1] = vgetq_lane_s32(_sum2, 2);
                outptr[M * 3] = vgetq_lane_s32(_sum0, 3);
                outptr[M * 3 + 1] = vgetq_lane_s32(_sum2, 3);
                outptr[M * 4] = vgetq_lane_s32(_sum1, 0);
                outptr[M * 4 + 1] = vgetq_lane_s32(_sum3, 0);
                outptr[M * 5] = vgetq_lane_s32(_sum1, 1);
                outptr[M * 5 + 1] = vgetq_lane_s32(_sum3, 1);
                outptr[M * 6] = vgetq_lane_s32(_sum1, 2);
                outptr[M * 6 + 1] = vgetq_lane_s32(_sum3, 2);
                outptr[M * 7] = vgetq_lane_s32(_sum1, 3);
                outptr[M * 7 + 1] = vgetq_lane_s32(_sum3, 3);
                outptr += 2;
            }
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            {
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        int8x8_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            _r0 = vld1_s8(tmp);
                        }

                        int8x16_t _w0 = vld1q_s8(kptr);
                        int8x16_t _w1 = vld1q_s8(kptr + 16);
                        int8x16_t _w2 = vld1q_s8(kptr + 32);
                        int8x16_t _w3 = vld1q_s8(kptr + 48);

#if __ARM_FEATURE_MATMUL_INT8
                        int8x16_t _r00 = vcombine_s8(_r0, _r0);
                        _sum0 = vdotq_s32(_sum0, _r00, _w0);
                        _sum1 = vdotq_s32(_sum1, _r00, _w1);
                        _sum2 = vdotq_s32(_sum2, _r00, _w2);
                        _sum3 = vdotq_s32(_sum3, _r00, _w3);
#elif __ARM_FEATURE_DOTPROD
                        _sum0 = vdotq_lane_s32(_sum0, _w0, _r0, 0);
                        _sum1 = vdotq_lane_s32(_sum1, _w1, _r0, 0);
                        _sum2 = vdotq_lane_s32(_sum2, _w2, _r0, 1);
                        _sum3 = vdotq_lane_s32(_sum3, _w3, _r0, 1);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                        int16x4_t _rr0 = vreinterpret_s16_s8(_r0);
                        int8x8_t _r0ll = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 0));
                        int8x8_t _r0lh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 1));
                        int8x8_t _r0hl = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 2));
                        int8x8_t _r0hh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 3));

                        int16x8_t _s0l = vmull_s8(_r0ll, vget_low_s8(_w0));
                        int16x8_t _s1l = vmull_s8(_r0ll, vget_high_s8(_w0));
                        int16x8_t _s0h = vmull_s8(_r0lh, vget_low_s8(_w1));
                        int16x8_t _s1h = vmull_s8(_r0lh, vget_high_s8(_w1));
                        _s0l = vmlal_s8(_s0l, _r0hl, vget_low_s8(_w2));
                        _s1l = vmlal_s8(_s1l, _r0hl, vget_high_s8(_w2));
                        _s0h = vmlal_s8(_s0h, _r0hh, vget_low_s8(_w3));
                        _s1h = vmlal_s8(_s1h, _r0hh, vget_high_s8(_w3));

                        _sum0 = vpadalq_s16(_sum0, _s0l);
                        _sum1 = vpadalq_s16(_sum1, _s1l);
                        _sum2 = vpadalq_s16(_sum2, _s0h);
                        _sum3 = vpadalq_s16(_sum3, _s1h);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD

                        kptr += 64;
                    }
                }
#if __ARM_FEATURE_MATMUL_INT8
                {
                    _sum0 = vpaddq_s32(_sum0, _sum1);
                    _sum1 = vpaddq_s32(_sum2, _sum3);
                }
#else
                {
                    _sum0 = vaddq_s32(_sum0, _sum2);
                    _sum1 = vaddq_s32(_sum1, _sum3);
                }
#endif
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        int8x8_t _val = vdup_n_s8(r0s[0]);
                        int8x8_t _w = vld1_s8(kptr);
                        int16x8_t _s0 = vmull_s8(_val, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        kptr += 8;
                    }
                }
            }

            if (out_elempack == 8)
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + M, _sum1);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                outptr[0] = vgetq_lane_s32(_sum0, 0);
                outptr[M] = vgetq_lane_s32(_sum0, 1);
                outptr[M * 2] = vgetq_lane_s32(_sum0, 2);
                outptr[M * 3] = vgetq_lane_s32(_sum0, 3);
                outptr[M * 4] = vgetq_lane_s32(_sum1, 0);
                outptr[M * 5] = vgetq_lane_s32(_sum1, 1);
                outptr[M * 6] = vgetq_lane_s32(_sum1, 2);
                outptr[M * 7] = vgetq_lane_s32(_sum1, 3);
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);

            int q = 0;
            {
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        int8x8_t _r0;
                        int8x8_t _r1;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                            _r1 = vld1_s8(r1s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp0[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            signed char tmp1[8] = {r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7]};
                            _r0 = vld1_s8(tmp0);
                            _r1 = vld1_s8(tmp1);
                        }

                        int8x16_t _w0 = vld1q_s8(kptr);
                        int8x16_t _w1 = vld1q_s8(kptr + 16);

#if __ARM_FEATURE_MATMUL_INT8
                        int8x16_t _r01 = vcombine_s8(_r0, _r1);
                        _sum0 = vmmlaq_s32(_sum0, _r01, _w0);
                        _sum1 = vmmlaq_s32(_sum1, _r01, _w1);
#elif __ARM_FEATURE_DOTPROD
                        _sum0 = vdotq_lane_s32(_sum0, _w0, _r0, 0);
                        _sum1 = vdotq_lane_s32(_sum1, _w0, _r1, 0);
                        _sum0 = vdotq_lane_s32(_sum0, _w1, _r0, 1);
                        _sum1 = vdotq_lane_s32(_sum1, _w1, _r1, 1);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                        int16x4_t _rr0 = vreinterpret_s16_s8(_r0);
                        int16x4_t _rr1 = vreinterpret_s16_s8(_r1);

                        int8x8_t _r0ll = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 0));
                        int8x8_t _r1ll = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 0));
                        int8x8_t _r0lh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 1));
                        int8x8_t _r1lh = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 1));

                        int16x8_t _s0l = vmull_s8(_r0ll, vget_low_s8(_w0));
                        int16x8_t _s1l = vmull_s8(_r1ll, vget_low_s8(_w0));
                        int16x8_t _s0h = vmull_s8(_r0lh, vget_high_s8(_w0));
                        int16x8_t _s1h = vmull_s8(_r1lh, vget_high_s8(_w0));

                        int8x8_t _r0hl = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 2));
                        int8x8_t _r1hl = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 2));
                        int8x8_t _r0hh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 3));
                        int8x8_t _r1hh = vreinterpret_s8_s16(vdup_lane_s16(_rr1, 3));

                        _s0l = vmlal_s8(_s0l, _r0hl, vget_low_s8(_w1));
                        _s1l = vmlal_s8(_s1l, _r1hl, vget_low_s8(_w1));
                        _s0h = vmlal_s8(_s0h, _r0hh, vget_high_s8(_w1));
                        _s1h = vmlal_s8(_s1h, _r1hh, vget_high_s8(_w1));

                        _sum0 = vpadalq_s16(_sum0, _s0l);
                        _sum1 = vpadalq_s16(_sum1, _s1l);
                        _sum0 = vpadalq_s16(_sum0, _s0h);
                        _sum1 = vpadalq_s16(_sum1, _s1h);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD

                        kptr += 32;
                    }
                }
#if __ARM_FEATURE_MATMUL_INT8
                {
                    int32x4_t _tmp0 = vcombine_s32(vget_low_s32(_sum0), vget_low_s32(_sum1));
                    int32x4_t _tmp1 = vcombine_s32(vget_high_s32(_sum0), vget_high_s32(_sum1));
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                }
#endif
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        int8x8_t _r0 = vdup_n_s8(r0s[0]);
                        int8x8_t _r1 = vdup_n_s8(r1s[0]);
                        int8x8_t _r01 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_r0), vreinterpret_s32_s8(_r1)).val[0]);
                        int8x8_t _w = vld1_s8(kptr);
                        int8x8_t _ww = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_w), vreinterpret_s32_s8(_w)).val[0]);
                        int16x8_t _s01 = vmull_s8(_r01, _ww);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));

                        kptr += 4;
                    }
                }
            }

            if (out_elempack == 4)
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                outptr += 8;
            }
            if (out_elempack == 1)
            {
                int32x4x2_t _sum01 = vzipq_s32(_sum0, _sum1);
                vst1_s32(outptr, vget_low_s32(_sum01.val[0]));
                vst1_s32(outptr + M, vget_high_s32(_sum01.val[0]));
                vst1_s32(outptr + M * 2, vget_low_s32(_sum01.val[1]));
                vst1_s32(outptr + M * 3, vget_high_s32(_sum01.val[1]));
                outptr += 2;
            }
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);

            int q = 0;
            {
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        int8x8_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            _r0 = vld1_s8(tmp);
                        }

                        int8x16_t _w0 = vld1q_s8(kptr);
                        int8x16_t _w1 = vld1q_s8(kptr + 16);

#if __ARM_FEATURE_MATMUL_INT8
                        int8x16_t _r00 = vcombine_s8(_r0, _r0);
                        _sum0 = vdotq_s32(_sum0, _r00, _w0);
                        _sum1 = vdotq_s32(_sum1, _r00, _w1);
#elif __ARM_FEATURE_DOTPROD
                        _sum0 = vdotq_lane_s32(_sum0, _w0, _r0, 0);
                        _sum1 = vdotq_lane_s32(_sum1, _w1, _r0, 1);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                        int16x4_t _rr0 = vreinterpret_s16_s8(_r0);
                        int8x8_t _r0ll = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 0));
                        int8x8_t _r0lh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 1));
                        int8x8_t _r0hl = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 2));
                        int8x8_t _r0hh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 3));

                        int16x8_t _sl = vmull_s8(_r0ll, vget_low_s8(_w0));
                        int16x8_t _sh = vmull_s8(_r0lh, vget_high_s8(_w0));
                        _sl = vmlal_s8(_sl, _r0hl, vget_low_s8(_w1));
                        _sh = vmlal_s8(_sh, _r0hh, vget_high_s8(_w1));

                        _sum0 = vpadalq_s16(_sum0, _sl);
                        _sum1 = vpadalq_s16(_sum1, _sh);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD

                        kptr += 32;
                    }
                }
#if __ARM_FEATURE_MATMUL_INT8
                {
                    _sum0 = vpaddq_s32(_sum0, _sum1);
                }
#else
                {
                    _sum0 = vaddq_s32(_sum0, _sum1);
                }
#endif
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        int8x8_t _val = vdup_n_s8(r0s[0]);
                        int8x8_t _w = vld1_s8(kptr);
                        int16x8_t _s0 = vmull_s8(_val, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                        kptr += 4;
                    }
                }
            }

            if (out_elempack == 4)
            {
                vst1q_s32(outptr, _sum0);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                outptr[0] = vgetq_lane_s32(_sum0, 0);
                outptr[M] = vgetq_lane_s32(_sum0, 1);
                outptr[M * 2] = vgetq_lane_s32(_sum0, 2);
                outptr[M * 3] = vgetq_lane_s32(_sum0, 3);
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 4;
    nn_outch = (outch - remain_outch_start) / 2;
#else // __ARM_NEON
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __ARM_NEON
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int sum00 = 0;
            int sum01 = 0;
            int sum10 = 0;
            int sum11 = 0;

#if __ARM_NEON
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __ARM_NEON
            {
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        int8x8_t _r0;
                        int8x8_t _r1;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                            _r1 = vld1_s8(r1s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp0[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            signed char tmp1[8] = {r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7]};
                            _r0 = vld1_s8(tmp0);
                            _r1 = vld1_s8(tmp1);
                        }

                        int8x16_t _w0 = vld1q_s8(kptr);

#if __ARM_FEATURE_DOTPROD
                        int8x16_t _r00 = vcombine_s8(_r0, _r0);
                        int8x16_t _r11 = vcombine_s8(_r1, _r1);
                        _sum01 = vdotq_s32(_sum01, _r00, _w0);
                        _sum23 = vdotq_s32(_sum23, _r11, _w0);
#else  // __ARM_FEATURE_DOTPROD
                        int32x2x2_t _rr0 = vzip_s32(vreinterpret_s32_s8(_r0), vreinterpret_s32_s8(_r0));
                        int32x2x2_t _rr1 = vzip_s32(vreinterpret_s32_s8(_r1), vreinterpret_s32_s8(_r1));
                        int8x8_t _r0l = vreinterpret_s8_s32(_rr0.val[0]);
                        int8x8_t _r0h = vreinterpret_s8_s32(_rr0.val[1]);
                        int8x8_t _r1l = vreinterpret_s8_s32(_rr1.val[0]);
                        int8x8_t _r1h = vreinterpret_s8_s32(_rr1.val[1]);

                        int16x8_t _s01 = vmull_s8(_r0l, vget_low_s8(_w0));
                        int16x8_t _s23 = vmull_s8(_r1l, vget_low_s8(_w0));
                        _s01 = vmlal_s8(_s01, _r0h, vget_high_s8(_w0));
                        _s23 = vmlal_s8(_s23, _r1h, vget_high_s8(_w0));

                        _sum01 = vpadalq_s16(_sum01, _s01);
                        _sum23 = vpadalq_s16(_sum23, _s23);
#endif // __ARM_FEATURE_DOTPROD

                        kptr += 16;
                    }
                }
                int32x2_t _s0 = vpadd_s32(vget_low_s32(_sum01), vget_high_s32(_sum01));
                int32x2_t _s1 = vpadd_s32(vget_low_s32(_sum23), vget_high_s32(_sum23));
                sum00 += vget_lane_s32(_s0, 0);
                sum01 += vget_lane_s32(_s1, 0);
                sum10 += vget_lane_s32(_s0, 1);
                sum11 += vget_lane_s32(_s1, 1);
            }
#endif // __ARM_NEON
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum00 += r0s[0] * kptr[0];
                        sum01 += r1s[0] * kptr[0];
                        sum10 += r0s[0] * kptr[1];
                        sum11 += r1s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr0 += 2;
            outptr1 += 2;
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum0 = 0;
            int sum1 = 0;

#if __ARM_NEON
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __ARM_NEON
            {
                int32x4_t _sum01 = vdupq_n_s32(0);
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        int8x8_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            _r0 = vld1_s8(tmp);
                        }

                        int8x16_t _w0 = vld1q_s8(kptr);

#if __ARM_FEATURE_DOTPROD
                        int8x16_t _r00 = vcombine_s8(_r0, _r0);
                        _sum01 = vdotq_s32(_sum01, _r00, _w0);
#else  // __ARM_FEATURE_DOTPROD
                        int32x2x2_t _rr0 = vzip_s32(vreinterpret_s32_s8(_r0), vreinterpret_s32_s8(_r0));
                        int8x8_t _r0l = vreinterpret_s8_s32(_rr0.val[0]);
                        int8x8_t _r0h = vreinterpret_s8_s32(_rr0.val[1]);

                        int16x8_t _s01 = vmull_s8(_r0l, vget_low_s8(_w0));
                        _s01 = vmlal_s8(_s01, _r0h, vget_high_s8(_w0));

                        _sum01 = vpadalq_s16(_sum01, _s01);
#endif // __ARM_FEATURE_DOTPROD

                        kptr += 16;
                    }
                }
                int32x2_t _s0 = vpadd_s32(vget_low_s32(_sum01), vget_high_s32(_sum01));
                sum0 += vget_lane_s32(_s0, 0);
                sum1 += vget_lane_s32(_s0, 1);
            }
#endif // __ARM_NEON
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r0s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int sum0 = 0;
            int sum1 = 0;

#if __ARM_NEON
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __ARM_NEON
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        int8x8_t _r0;
                        int8x8_t _r1;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                            _r1 = vld1_s8(r1s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp0[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            signed char tmp1[8] = {r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7]};
                            _r0 = vld1_s8(tmp0);
                            _r1 = vld1_s8(tmp1);
                        }

                        int8x8_t _w = vld1_s8(kptr);

                        int16x8_t _s0 = vmull_s8(_r0, _w);
                        int16x8_t _s1 = vmull_s8(_r1, _w);

                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                        _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                        _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                        kptr += 8;
                    }
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);
#if __aarch64__
                sum0 += vaddvq_s32(_sum0);
                sum1 += vaddvq_s32(_sum2);
#else
                int32x2_t _ss0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _ss2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                _ss0 = vpadd_s32(_ss0, _ss2);
                sum0 += vget_lane_s32(_ss0, 0);
                sum1 += vget_lane_s32(_ss0, 1);
#endif
            }
#endif // __ARM_NEON
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r1s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum = 0;

#if __ARM_NEON
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __ARM_NEON
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        int8x8_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            _r0 = vld1_s8(tmp);
                        }

                        int8x8_t _w = vld1_s8(kptr);

                        int16x8_t _s0 = vmull_s8(_r0, _w);

                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        kptr += 8;
                    }
                }
                int32x4_t _sum = vaddq_s32(_sum0, _sum1);
#if __aarch64__
                sum += vaddvq_s32(_sum);
#else
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                _ss = vpadd_s32(_ss, _ss);
                sum += vget_lane_s32(_ss, 0);
#endif
            }
#endif // __ARM_NEON
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum += r0s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
