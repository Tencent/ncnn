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

static void convolution1d_transform_kernel_packed_bf16s(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    // src = kw-inh-outh
    // dst = pb-pa-kw-inh/pa-outh/pb

    // clang-format off
    // *INDENT-OFF*
#if __ARM_NEON
#if __aarch64__
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
#endif // __aarch64__
    if (outh >= 4)
    {
#if __aarch64__
        if (inh >= 8)
            kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else
#endif // __aarch64__
        if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2, (size_t)2u);
    }
    else
#endif // __ARM_NEON
    if (outh >= 2)
    {
#if __ARM_NEON
#if __aarch64__
        if (inh >= 8)
            kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
#endif // __aarch64__
        if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
#endif // __ARM_NEON
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2, (size_t)2u);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2, (size_t)2u);
    }
    else
    {
#if __ARM_NEON
#if __aarch64__
        if (inh >= 8)
            kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh, (size_t)2u);
        else
#endif // __aarch64__
        if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh, (size_t)2u);
        else
#endif // __ARM_NEON
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh, (size_t)2u);
        else
            kernel_tm.create(kernel_w, inh, outh, (size_t)2u);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __ARM_NEON
#if __aarch64__
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
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;
                const float* k2 = kptr2 + p * kernel_w;
                const float* k3 = kptr3 + p * kernel_w;
                const float* k4 = kptr4 + p * kernel_w;
                const float* k5 = kptr5 + p * kernel_w;
                const float* k6 = kptr6 + p * kernel_w;
                const float* k7 = kptr7 + p * kernel_w;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    g00[2] = float32_to_bfloat16(k2[k]);
                    g00[3] = float32_to_bfloat16(k3[k]);
                    g00[4] = float32_to_bfloat16(k4[k]);
                    g00[5] = float32_to_bfloat16(k5[k]);
                    g00[6] = float32_to_bfloat16(k6[k]);
                    g00[7] = float32_to_bfloat16(k7[k]);
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
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;
                const float* k2 = kptr2 + p * kernel_w;
                const float* k3 = kptr3 + p * kernel_w;
                const float* k4 = kptr4 + p * kernel_w;
                const float* k5 = kptr5 + p * kernel_w;
                const float* k6 = kptr6 + p * kernel_w;
                const float* k7 = kptr7 + p * kernel_w;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    g00[2] = float32_to_bfloat16(k2[k]);
                    g00[3] = float32_to_bfloat16(k3[k]);
                    g00[4] = float32_to_bfloat16(k4[k]);
                    g00[5] = float32_to_bfloat16(k5[k]);
                    g00[6] = float32_to_bfloat16(k6[k]);
                    g00[7] = float32_to_bfloat16(k7[k]);
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
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;
                const float* k2 = kptr2 + p * kernel_w;
                const float* k3 = kptr3 + p * kernel_w;
                const float* k4 = kptr4 + p * kernel_w;
                const float* k5 = kptr5 + p * kernel_w;
                const float* k6 = kptr6 + p * kernel_w;
                const float* k7 = kptr7 + p * kernel_w;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    g00[2] = float32_to_bfloat16(k2[k]);
                    g00[3] = float32_to_bfloat16(k3[k]);
                    g00[4] = float32_to_bfloat16(k4[k]);
                    g00[5] = float32_to_bfloat16(k5[k]);
                    g00[6] = float32_to_bfloat16(k6[k]);
                    g00[7] = float32_to_bfloat16(k7[k]);
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
        }
        for (; p < inh; p++)
        {
            const float* k0 = kptr0 + p * kernel_w;
            const float* k1 = kptr1 + p * kernel_w;
            const float* k2 = kptr2 + p * kernel_w;
            const float* k3 = kptr3 + p * kernel_w;
            const float* k4 = kptr4 + p * kernel_w;
            const float* k5 = kptr5 + p * kernel_w;
            const float* k6 = kptr6 + p * kernel_w;
            const float* k7 = kptr7 + p * kernel_w;

            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(k0[k]);
                g00[1] = float32_to_bfloat16(k1[k]);
                g00[2] = float32_to_bfloat16(k2[k]);
                g00[3] = float32_to_bfloat16(k3[k]);
                g00[4] = float32_to_bfloat16(k4[k]);
                g00[5] = float32_to_bfloat16(k5[k]);
                g00[6] = float32_to_bfloat16(k6[k]);
                g00[7] = float32_to_bfloat16(k7[k]);
                g00 += 8;
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outh; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;

#if __aarch64__
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
#if __aarch64__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;
                const float* k2 = kptr2 + p * kernel_w;
                const float* k3 = kptr3 + p * kernel_w;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    g00[2] = float32_to_bfloat16(k2[k]);
                    g00[3] = float32_to_bfloat16(k3[k]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }
        }
#endif // __aarch64__
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;
                const float* k2 = kptr2 + p * kernel_w;
                const float* k3 = kptr3 + p * kernel_w;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    g00[2] = float32_to_bfloat16(k2[k]);
                    g00[3] = float32_to_bfloat16(k3[k]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;
                const float* k2 = kptr2 + p * kernel_w;
                const float* k3 = kptr3 + p * kernel_w;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    g00[2] = float32_to_bfloat16(k2[k]);
                    g00[3] = float32_to_bfloat16(k3[k]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }
        }
        for (; p < inh; p++)
        {
            const float* k0 = kptr0 + p * kernel_w;
            const float* k1 = kptr1 + p * kernel_w;
            const float* k2 = kptr2 + p * kernel_w;
            const float* k3 = kptr3 + p * kernel_w;

            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(k0[k]);
                g00[1] = float32_to_bfloat16(k1[k]);
                g00[2] = float32_to_bfloat16(k2[k]);
                g00[3] = float32_to_bfloat16(k3[k]);
                g00 += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;

#if __aarch64__
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __ARM_NEON
        unsigned short* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __ARM_NEON
#if __aarch64__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w + k;
                const float* k1 = kptr1 + p * kernel_w + k;

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
        }
#endif // __aarch64__
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w + k;
                const float* k1 = kptr1 + p * kernel_w + k;

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
        }
#endif // __ARM_NEON
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w;
                const float* k1 = kptr1 + p * kernel_w;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    g00[1] = float32_to_bfloat16(k1[k]);
                    k0 += kernel_w;
                    k1 += kernel_w;
                    g00 += 2;
                }
            }
        }
        for (; p < inh; p++)
        {
            const float* k0 = kptr0 + p * kernel_w;
            const float* k1 = kptr1 + p * kernel_w;

            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(k0[k]);
                g00[1] = float32_to_bfloat16(k1[k]);
                g00 += 2;
            }
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr = (const float*)kernel + q * inh * kernel_w;

#if __aarch64__
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __ARM_NEON
        unsigned short* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        unsigned short* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __ARM_NEON
#if __aarch64__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + p * kernel_w;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    k0 += kernel_w;
                    g00 += 1;
                }
            }
        }
#endif // __aarch64__
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + p * kernel_w;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    k0 += kernel_w;
                    g00 += 1;
                }
            }
        }
#endif // __ARM_NEON
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + p * kernel_w;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = float32_to_bfloat16(k0[k]);
                    k0 += kernel_w;
                    g00 += 1;
                }
            }
        }
        for (; p < inh; p++)
        {
            const float* k0 = kptr + p * kernel_w;

            for (int k = 0; k < kernel_w; k++)
            {
                g00[0] = float32_to_bfloat16(k0[k]);
                g00++;
            }
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
#if __ARM_NEON
#if __aarch64__
    nn_outh = (outh - remain_outh_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 8;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        unsigned short* outptr = top_blob.row<unsigned short>(p / out_elempack);

        for (int j = 0; j < outw; j++)
        {
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
            float32x4_t _sum4 = vdupq_n_f32(0.f);
            float32x4_t _sum5 = vdupq_n_f32(0.f);
            float32x4_t _sum6 = vdupq_n_f32(0.f);
            float32x4_t _sum7 = vdupq_n_f32(0.f);

            if (bias_data_ptr)
            {
                _sum0 = vld1q_f32(bias_data_ptr + p);
                _sum1 = vld1q_f32(bias_data_ptr + p + 4);
            }

            const unsigned short* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            for (; q + 7 < inh; q += 8)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        _r1 = bfloat2float(vld1_u16(r0 + N));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x8_t _r_u16 = uint16x8_t();
                        _r_u16 = vsetq_lane_u16(r0[0], _r_u16, 0);
                        _r_u16 = vsetq_lane_u16(r0[N], _r_u16, 1);
                        _r_u16 = vsetq_lane_u16(r0[N * 2], _r_u16, 2);
                        _r_u16 = vsetq_lane_u16(r0[N * 3], _r_u16, 3);
                        _r_u16 = vsetq_lane_u16(r0[N * 4], _r_u16, 4);
                        _r_u16 = vsetq_lane_u16(r0[N * 5], _r_u16, 5);
                        _r_u16 = vsetq_lane_u16(r0[N * 6], _r_u16, 6);
                        _r_u16 = vsetq_lane_u16(r0[N * 7], _r_u16, 7);
                        _r0 = bfloat2float(vget_low_u16(_r_u16));
                        _r1 = bfloat2float(vget_high_u16(_r_u16));
                        r0 += dilation_w;
                    }

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);
                    uint16x8_t _w45 = vld1q_u16(kptr + 16);
                    uint16x8_t _w67 = vld1q_u16(kptr + 24);
                    uint16x8_t _w89 = vld1q_u16(kptr + 32);
                    uint16x8_t _wab = vld1q_u16(kptr + 40);
                    uint16x8_t _wcd = vld1q_u16(kptr + 48);
                    uint16x8_t _wef = vld1q_u16(kptr + 56);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w01));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w01));
                    float32x4_t _w2 = bfloat2float(vget_low_u16(_w23));
                    float32x4_t _w3 = bfloat2float(vget_high_u16(_w23));
                    float32x4_t _w4 = bfloat2float(vget_low_u16(_w45));
                    float32x4_t _w5 = bfloat2float(vget_high_u16(_w45));
                    float32x4_t _w6 = bfloat2float(vget_low_u16(_w67));
                    float32x4_t _w7 = bfloat2float(vget_high_u16(_w67));
                    float32x4_t _w8 = bfloat2float(vget_low_u16(_w89));
                    float32x4_t _w9 = bfloat2float(vget_high_u16(_w89));
                    float32x4_t _wa = bfloat2float(vget_low_u16(_wab));
                    float32x4_t _wb = bfloat2float(vget_high_u16(_wab));
                    float32x4_t _wc = bfloat2float(vget_low_u16(_wcd));
                    float32x4_t _wd = bfloat2float(vget_high_u16(_wcd));
                    float32x4_t _we = bfloat2float(vget_low_u16(_wef));
                    float32x4_t _wf = bfloat2float(vget_high_u16(_wef));
                    _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 1);
                    _sum4 = vfmaq_laneq_f32(_sum4, _w4, _r0, 2);
                    _sum5 = vfmaq_laneq_f32(_sum5, _w5, _r0, 2);
                    _sum6 = vfmaq_laneq_f32(_sum6, _w6, _r0, 3);
                    _sum7 = vfmaq_laneq_f32(_sum7, _w7, _r0, 3);
                    _sum0 = vfmaq_laneq_f32(_sum0, _w8, _r1, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _w9, _r1, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _wa, _r1, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _wb, _r1, 1);
                    _sum4 = vfmaq_laneq_f32(_sum4, _wc, _r1, 2);
                    _sum5 = vfmaq_laneq_f32(_sum5, _wd, _r1, 2);
                    _sum6 = vfmaq_laneq_f32(_sum6, _we, _r1, 3);
                    _sum7 = vfmaq_laneq_f32(_sum7, _wf, _r1, 3);

                    kptr += 64;
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x4_t _r_u16 = uint16x4_t();
                        _r_u16 = vset_lane_u16(r0[0], _r_u16, 0);
                        _r_u16 = vset_lane_u16(r0[N], _r_u16, 1);
                        _r_u16 = vset_lane_u16(r0[N * 2], _r_u16, 2);
                        _r_u16 = vset_lane_u16(r0[N * 3], _r_u16, 3);
                        _r0 = bfloat2float(_r_u16);
                        r0 += dilation_w;
                    }

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);
                    uint16x8_t _w45 = vld1q_u16(kptr + 16);
                    uint16x8_t _w67 = vld1q_u16(kptr + 24);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w01));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w01));
                    float32x4_t _w2 = bfloat2float(vget_low_u16(_w23));
                    float32x4_t _w3 = bfloat2float(vget_high_u16(_w23));
                    float32x4_t _w4 = bfloat2float(vget_low_u16(_w45));
                    float32x4_t _w5 = bfloat2float(vget_high_u16(_w45));
                    float32x4_t _w6 = bfloat2float(vget_low_u16(_w67));
                    float32x4_t _w7 = bfloat2float(vget_high_u16(_w67));
                    _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 1);
                    _sum4 = vfmaq_laneq_f32(_sum4, _w4, _r0, 2);
                    _sum5 = vfmaq_laneq_f32(_sum5, _w5, _r0, 2);
                    _sum6 = vfmaq_laneq_f32(_sum6, _w6, _r0, 3);
                    _sum7 = vfmaq_laneq_f32(_sum7, _w7, _r0, 3);

                    kptr += 32;
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = bfloat16_to_float32(r0[0]);
                        val1 = bfloat16_to_float32(r0[N]);
                        r0 += dilation_w;
                    }

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w01));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w01));
                    float32x4_t _w2 = bfloat2float(vget_low_u16(_w23));
                    float32x4_t _w3 = bfloat2float(vget_high_u16(_w23));
                    _sum0 = vfmaq_n_f32(_sum0, _w0, val0);
                    _sum1 = vfmaq_n_f32(_sum1, _w1, val0);
                    _sum2 = vfmaq_n_f32(_sum2, _w2, val1);
                    _sum3 = vfmaq_n_f32(_sum3, _w3, val1);

                    kptr += 16;
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _val;
                    // if (elempack == 1)
                    {
                        _val = bfloat2float(vdup_n_u16(r0[0]));
                        r0 += dilation_w;
                    }

                    uint16x8_t _w = vld1q_u16(kptr);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w));
                    _sum0 = vfmaq_f32(_sum0, _w0, _val);
                    _sum1 = vfmaq_f32(_sum1, _w1, _val);

                    kptr += 8;
                }
            }

            _sum0 = vaddq_f32(_sum0, _sum2);
            _sum1 = vaddq_f32(_sum1, _sum3);
            _sum4 = vaddq_f32(_sum4, _sum6);
            _sum5 = vaddq_f32(_sum5, _sum7);
            _sum0 = vaddq_f32(_sum0, _sum4);
            _sum1 = vaddq_f32(_sum1, _sum5);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);
            _sum1 = activation_ps(_sum1, activation_type, activation_params);

            if (out_elempack == 4)
            {
                vst1_u16(outptr, float2bfloat(_sum0));
                vst1_u16(outptr + M, float2bfloat(_sum1));
                outptr += 4;
            }
            else // if (out_elempack == 1)
            {
                uint16x4_t _sum0_u16 = float2bfloat(_sum0);
                uint16x4_t _sum1_u16 = float2bfloat(_sum1);
                outptr[0] = vget_lane_u16(_sum0_u16, 0);
                outptr[M] = vget_lane_u16(_sum0_u16, 1);
                outptr[M * 2] = vget_lane_u16(_sum0_u16, 2);
                outptr[M * 3] = vget_lane_u16(_sum0_u16, 3);
                outptr[M * 4] = vget_lane_u16(_sum1_u16, 0);
                outptr[M * 5] = vget_lane_u16(_sum1_u16, 1);
                outptr[M * 6] = vget_lane_u16(_sum1_u16, 2);
                outptr[M * 7] = vget_lane_u16(_sum1_u16, 3);
                outptr += 1;
            }
        }
    }
    remain_outh_start += nn_outh * 8;
    nn_outh = (outh - remain_outh_start) / 4;
#else // __aarch64__
    nn_outh = (outh - remain_outh_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __aarch64__
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
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            if (bias_data_ptr)
            {
                _sum0 = vld1q_f32(bias_data_ptr + p);
            }

#if __aarch64__
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __aarch64__
            for (; q + 7 < inh; q += 8)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        _r1 = bfloat2float(vld1_u16(r0 + N));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x8_t _r_u16 = uint16x8_t();
                        _r_u16 = vsetq_lane_u16(r0[0], _r_u16, 0);
                        _r_u16 = vsetq_lane_u16(r0[N], _r_u16, 1);
                        _r_u16 = vsetq_lane_u16(r0[N * 2], _r_u16, 2);
                        _r_u16 = vsetq_lane_u16(r0[N * 3], _r_u16, 3);
                        _r_u16 = vsetq_lane_u16(r0[N * 4], _r_u16, 4);
                        _r_u16 = vsetq_lane_u16(r0[N * 5], _r_u16, 5);
                        _r_u16 = vsetq_lane_u16(r0[N * 6], _r_u16, 6);
                        _r_u16 = vsetq_lane_u16(r0[N * 7], _r_u16, 7);
                        _r0 = bfloat2float(vget_low_u16(_r_u16));
                        _r1 = bfloat2float(vget_high_u16(_r_u16));
                        r0 += dilation_w;
                    }

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);
                    uint16x8_t _w45 = vld1q_u16(kptr + 16);
                    uint16x8_t _w67 = vld1q_u16(kptr + 24);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w01));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w01));
                    float32x4_t _w2 = bfloat2float(vget_low_u16(_w23));
                    float32x4_t _w3 = bfloat2float(vget_high_u16(_w23));
                    float32x4_t _w4 = bfloat2float(vget_low_u16(_w45));
                    float32x4_t _w5 = bfloat2float(vget_high_u16(_w45));
                    float32x4_t _w6 = bfloat2float(vget_low_u16(_w67));
                    float32x4_t _w7 = bfloat2float(vget_high_u16(_w67));
                    _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 3);
                    _sum0 = vfmaq_laneq_f32(_sum0, _w4, _r1, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _w5, _r1, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _w6, _r1, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _w7, _r1, 3);

                    kptr += 32;
                }
            }
#endif // __aarch64__
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x4_t _r_u16 = uint16x4_t();
                        _r_u16 = vset_lane_u16(r0[0], _r_u16, 0);
                        _r_u16 = vset_lane_u16(r0[N], _r_u16, 1);
                        _r_u16 = vset_lane_u16(r0[N * 2], _r_u16, 2);
                        _r_u16 = vset_lane_u16(r0[N * 3], _r_u16, 3);
                        _r0 = bfloat2float(_r_u16);
                        r0 += dilation_w;
                    }

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w01));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w01));
                    float32x4_t _w2 = bfloat2float(vget_low_u16(_w23));
                    float32x4_t _w3 = bfloat2float(vget_high_u16(_w23));
#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _w0, _r0, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _w1, _r0, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _w2, _r0, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _w3, _r0, 3);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r0), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_r0), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_r0), 1);
#endif

                    kptr += 16;
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = bfloat16_to_float32(r0[0]);
                        val1 = bfloat16_to_float32(r0[N]);
                        r0 += dilation_w;
                    }

                    uint16x8_t _w = vld1q_u16(kptr);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w));
#if __aarch64__
                    _sum0 = vfmaq_n_f32(_sum0, _w0, val0);
                    _sum1 = vfmaq_n_f32(_sum1, _w1, val1);
#else
                    _sum0 = vmlaq_n_f32(_sum0, _w0, val0);
                    _sum1 = vmlaq_n_f32(_sum1, _w1, val1);
#endif

                    kptr += 8;
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _val;
                    // if (elempack == 1)
                    {
                        _val = bfloat2float(vdup_n_u16(r0[0]));
                        r0 += dilation_w;
                    }

                    float32x4_t _w = bfloat2float(vld1_u16(kptr));
#if __aarch64__
                    _sum0 = vfmaq_f32(_sum0, _val, _w);
#else
                    _sum0 = vmlaq_f32(_sum0, _val, _w);
#endif

                    kptr += 4;
                }
            }

            _sum0 = vaddq_f32(_sum0, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _sum0 = vaddq_f32(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            if (out_elempack == 4)
            {
                vst1_u16(outptr, float2bfloat(_sum0));
                outptr += 4;
            }
            else // if (out_elempack == 1)
            {
                uint16x4_t _sum0_u16 = float2bfloat(_sum0);
                outptr[0] = vget_lane_u16(_sum0_u16, 0);
                outptr[M] = vget_lane_u16(_sum0_u16, 1);
                outptr[M * 2] = vget_lane_u16(_sum0_u16, 2);
                outptr[M * 3] = vget_lane_u16(_sum0_u16, 3);
                outptr += 1;
            }
        }
    }
    remain_outh_start += nn_outh * 4;
    nn_outh = (outh - remain_outh_start) / 2;
#else // __ARM_NEON
    nn_outh = (outh - remain_outh_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __ARM_NEON
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

#if __aarch64__
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __ARM_NEON
            const unsigned short* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __ARM_NEON
#if __aarch64__
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
            for (; q + 7 < inh; q += 8)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        _r1 = bfloat2float(vld1_u16(r0 + N));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x8_t _r01_u16 = uint16x8_t();
                        _r01_u16 = vsetq_lane_u16(r0[0], _r01_u16, 0);
                        _r01_u16 = vsetq_lane_u16(r0[N], _r01_u16, 1);
                        _r01_u16 = vsetq_lane_u16(r0[N * 2], _r01_u16, 2);
                        _r01_u16 = vsetq_lane_u16(r0[N * 3], _r01_u16, 3);
                        _r01_u16 = vsetq_lane_u16(r0[N * 4], _r01_u16, 4);
                        _r01_u16 = vsetq_lane_u16(r0[N * 5], _r01_u16, 5);
                        _r01_u16 = vsetq_lane_u16(r0[N * 6], _r01_u16, 6);
                        _r01_u16 = vsetq_lane_u16(r0[N * 7], _r01_u16, 7);
                        _r0 = bfloat2float(vget_low_u16(_r01_u16));
                        _r1 = bfloat2float(vget_high_u16(_r01_u16));
                        r0 += dilation_w;
                    }

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w01));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w01));
                    float32x4_t _w2 = bfloat2float(vget_low_u16(_w23));
                    float32x4_t _w3 = bfloat2float(vget_high_u16(_w23));
                    _sum0 = vfmaq_f32(_sum0, _r0, _w0);
                    _sum1 = vfmaq_f32(_sum1, _r1, _w1);
                    _sum2 = vfmaq_f32(_sum2, _r0, _w2);
                    _sum3 = vfmaq_f32(_sum3, _r1, _w3);

                    kptr += 16;
                }
            }
            _sum0 = vaddq_f32(_sum0, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            sum0 += vaddvq_f32(_sum0);
            sum1 += vaddvq_f32(_sum2);
            _sum0 = vdupq_n_f32(0.f);
            _sum1 = vdupq_n_f32(0.f);
#else  // __aarch64__
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
#endif // __aarch64__
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x4_t _r0_u16 = uint16x4_t();
                        _r0_u16 = vset_lane_u16(r0[0], _r0_u16, 0);
                        _r0_u16 = vset_lane_u16(r0[N], _r0_u16, 1);
                        _r0_u16 = vset_lane_u16(r0[N * 2], _r0_u16, 2);
                        _r0_u16 = vset_lane_u16(r0[N * 3], _r0_u16, 3);
                        _r0 = bfloat2float(_r0_u16);
                        r0 += dilation_w;
                    }

                    uint16x8_t _w = vld1q_u16(kptr);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w));
#if __aarch64__
                    _sum0 = vfmaq_f32(_sum0, _r0, _w0);
                    _sum1 = vfmaq_f32(_sum1, _r0, _w1);
#else
                    _sum0 = vmlaq_f32(_sum0, _r0, _w0);
                    _sum1 = vmlaq_f32(_sum1, _r0, _w1);
#endif

                    kptr += 8;
                }
            }
#if __aarch64__
            sum0 += vaddvq_f32(_sum0);
            sum1 += vaddvq_f32(_sum1);
#else
            float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
            float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss = vpadd_f32(_ss0, _ss1);
            sum0 += vget_lane_f32(_ss, 0);
            sum1 += vget_lane_f32(_ss, 1);
#endif
#endif // __ARM_NEON
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = bfloat16_to_float32(r0[0]);
                        val1 = bfloat16_to_float32(r0[N]);
                        r0 += dilation_w;
                    }

                    sum0 += val0 * bfloat16_to_float32(kptr[0]);
                    sum1 += val0 * bfloat16_to_float32(kptr[1]);
                    sum0 += val1 * bfloat16_to_float32(kptr[2]);
                    sum1 += val1 * bfloat16_to_float32(kptr[3]);

                    kptr += 4;
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val;
                    // if (elempack == 1)
                    {
                        val = bfloat16_to_float32(r0[0]);
                        r0 += dilation_w;
                    }

                    sum0 += val * bfloat16_to_float32(kptr[0]);
                    sum1 += val * bfloat16_to_float32(kptr[1]);

                    kptr += 2;
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
        unsigned short* outptr = top_blob.row<unsigned short>(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
            {
                sum = bias_data_ptr[p];
            }

#if __aarch64__
            const unsigned short* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __ARM_NEON
            const unsigned short* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
            const unsigned short* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __ARM_NEON
#if __aarch64__
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            for (; q + 7 < inh; q += 8)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        _r1 = bfloat2float(vld1_u16(r0 + N));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x8_t _r01_u16 = uint16x8_t();
                        _r01_u16 = vsetq_lane_u16(r0[0], _r01_u16, 0);
                        _r01_u16 = vsetq_lane_u16(r0[N], _r01_u16, 1);
                        _r01_u16 = vsetq_lane_u16(r0[N * 2], _r01_u16, 2);
                        _r01_u16 = vsetq_lane_u16(r0[N * 3], _r01_u16, 3);
                        _r01_u16 = vsetq_lane_u16(r0[N * 4], _r01_u16, 4);
                        _r01_u16 = vsetq_lane_u16(r0[N * 5], _r01_u16, 5);
                        _r01_u16 = vsetq_lane_u16(r0[N * 6], _r01_u16, 6);
                        _r01_u16 = vsetq_lane_u16(r0[N * 7], _r01_u16, 7);
                        _r0 = bfloat2float(vget_low_u16(_r01_u16));
                        _r1 = bfloat2float(vget_high_u16(_r01_u16));
                        r0 += dilation_w;
                    }

                    uint16x8_t _w = vld1q_u16(kptr);
                    float32x4_t _w0 = bfloat2float(vget_low_u16(_w));
                    float32x4_t _w1 = bfloat2float(vget_high_u16(_w));
                    _sum0 = vfmaq_f32(_sum0, _r0, _w0);
                    _sum1 = vfmaq_f32(_sum1, _r1, _w1);

                    kptr += 8;
                }
            }
            _sum0 = vaddq_f32(_sum0, _sum1);
            sum += vaddvq_f32(_sum0);
#endif // __aarch64__
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (; q + 3 < inh; q += 4)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        uint16x4_t _r0_u16 = uint16x4_t();
                        _r0_u16 = vset_lane_u16(r0[0], _r0_u16, 0);
                        _r0_u16 = vset_lane_u16(r0[N], _r0_u16, 1);
                        _r0_u16 = vset_lane_u16(r0[N * 2], _r0_u16, 2);
                        _r0_u16 = vset_lane_u16(r0[N * 3], _r0_u16, 3);
                        _r0 = bfloat2float(_r0_u16);
                        r0 += dilation_w;
                    }

                    float32x4_t _w = bfloat2float(vld1_u16(kptr));
#if __aarch64__
                    _sum = vfmaq_f32(_sum, _r0, _w);
#else
                    _sum = vmlaq_f32(_sum, _r0, _w);
#endif

                    kptr += 4;
                }
            }
#if __aarch64__
            sum += vaddvq_f32(_sum);
#else
            float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _ss = vpadd_f32(_ss, _ss);
            sum += vget_lane_f32(_ss, 0);
#endif
#endif // __ARM_NEON
            for (; q + 1 < inh; q += 2)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = bfloat16_to_float32(r0[0]);
                        val1 = bfloat16_to_float32(r0[N]);
                        r0 += dilation_w;
                    }

                    sum += val0 * bfloat16_to_float32(kptr[0]);
                    sum += val1 * bfloat16_to_float32(kptr[1]);

                    kptr += 2;
                }
            }
            for (; q < inh; q++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val;
                    // if (elempack == 1)
                    {
                        val = bfloat16_to_float32(r0[0]);
                        r0 += dilation_w;
                    }

                    sum += val * bfloat16_to_float32(kptr[0]);

                    kptr += 1;
                }
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[0] = float32_to_bfloat16(sum);
            outptr += 1;
        }
    }
}
