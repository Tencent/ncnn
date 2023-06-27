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

static void convolution1d_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
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
            kernel_tm.create(8 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 4)
            kernel_tm.create(8 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(8 * 2 * kernel_w, inh / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(8 * kernel_w, inh, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __aarch64__
    if (outh >= 4)
    {
#if __aarch64__
        if (inh >= 8)
            kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
#endif // __aarch64__
        if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __ARM_NEON
    if (outh >= 2)
    {
#if __ARM_NEON
#if __aarch64__
        if (inh >= 8)
            kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __aarch64__
        if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __ARM_NEON
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2);
    }
    else
    {
#if __ARM_NEON
#if __aarch64__
        if (inh >= 8)
            kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __aarch64__
        if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __ARM_NEON
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh);
        else
            kernel_tm.create(kernel_w, inh, outh);
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

        float* g00 = kernel_tm.channel(q / 8);

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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
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
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00[2] = k2[k];
                g00[3] = k3[k];
                g00[4] = k4[k];
                g00[5] = k5[k];
                g00[6] = k6[k];
                g00[7] = k7[k];
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
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
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
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00[2] = k2[k];
                g00[3] = k3[k];
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
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __ARM_NEON
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 2);
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

                g00[0] = k0[0];
                g00[1] = k0[kernel_w];
                g00[2] = k0[kernel_w * 2];
                g00[3] = k0[kernel_w * 3];
                g00[4] = k0[kernel_w * 4];
                g00[5] = k0[kernel_w * 5];
                g00[6] = k0[kernel_w * 6];
                g00[7] = k0[kernel_w * 7];
                g00[8] = k1[0];
                g00[9] = k1[kernel_w];
                g00[10] = k1[kernel_w * 2];
                g00[11] = k1[kernel_w * 3];
                g00[12] = k1[kernel_w * 4];
                g00[13] = k1[kernel_w * 5];
                g00[14] = k1[kernel_w * 6];
                g00[15] = k1[kernel_w * 7];
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

                g00[0] = k0[0];
                g00[1] = k0[kernel_w];
                g00[2] = k0[kernel_w * 2];
                g00[3] = k0[kernel_w * 3];
                g00[4] = k1[0];
                g00[5] = k1[kernel_w];
                g00[6] = k1[kernel_w * 2];
                g00[7] = k1[kernel_w * 3];
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
                    g00[0] = k0[k];
                    g00[1] = k1[k];
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
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00 += 2;
            }
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr = (const float*)kernel + q * inh * kernel_w;

#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __ARM_NEON
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
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
                    g00[0] = k0[k];
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
                    g00[0] = k0[k];
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
                    g00[0] = k0[k];
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
                g00[0] = k0[k];
                g00++;
            }
        }
    }
}

static void convolution1d_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
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

        float* outptr = top_blob.row(p / out_elempack);

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

            const float* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        _r1 = vld1q_f32(r0 + N);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r1 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        _r1 = vsetq_lane_f32(r0[N * 4], _r1, 0);
                        _r1 = vsetq_lane_f32(r0[N * 5], _r1, 1);
                        _r1 = vsetq_lane_f32(r0[N * 6], _r1, 2);
                        _r1 = vsetq_lane_f32(r0[N * 7], _r1, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 4 * 2);
                    float32x4_t _w3 = vld1q_f32(kptr + 4 * 3);
                    float32x4_t _w4 = vld1q_f32(kptr + 4 * 4);
                    float32x4_t _w5 = vld1q_f32(kptr + 4 * 5);
                    float32x4_t _w6 = vld1q_f32(kptr + 4 * 6);
                    float32x4_t _w7 = vld1q_f32(kptr + 4 * 7);
                    float32x4_t _w8 = vld1q_f32(kptr + 4 * 8);
                    float32x4_t _w9 = vld1q_f32(kptr + 4 * 9);
                    float32x4_t _wa = vld1q_f32(kptr + 4 * 10);
                    float32x4_t _wb = vld1q_f32(kptr + 4 * 11);
                    float32x4_t _wc = vld1q_f32(kptr + 4 * 12);
                    float32x4_t _wd = vld1q_f32(kptr + 4 * 13);
                    float32x4_t _we = vld1q_f32(kptr + 4 * 14);
                    float32x4_t _wf = vld1q_f32(kptr + 4 * 15);
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
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 4 * 2);
                    float32x4_t _w3 = vld1q_f32(kptr + 4 * 3);
                    float32x4_t _w4 = vld1q_f32(kptr + 4 * 4);
                    float32x4_t _w5 = vld1q_f32(kptr + 4 * 5);
                    float32x4_t _w6 = vld1q_f32(kptr + 4 * 6);
                    float32x4_t _w7 = vld1q_f32(kptr + 4 * 7);
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
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = r0[0];
                        val1 = r0[N];
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    _sum0 = vfmaq_n_f32(_sum0, _w0, val0);
                    _sum1 = vfmaq_n_f32(_sum1, _w1, val0);
                    _sum2 = vfmaq_n_f32(_sum2, _w2, val1);
                    _sum3 = vfmaq_n_f32(_sum3, _w3, val1);

                    kptr += 16;
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _val;
                    // if (elempack == 1)
                    {
                        _val = vdupq_n_f32(r0[0]);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
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
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + M, _sum1);
                outptr += 4;
            }
            else // if (out_elempack == 1)
            {
                outptr[0] = vgetq_lane_f32(_sum0, 0);
                outptr[M] = vgetq_lane_f32(_sum0, 1);
                outptr[M * 2] = vgetq_lane_f32(_sum0, 2);
                outptr[M * 3] = vgetq_lane_f32(_sum0, 3);
                outptr[M * 4] = vgetq_lane_f32(_sum1, 0);
                outptr[M * 5] = vgetq_lane_f32(_sum1, 1);
                outptr[M * 6] = vgetq_lane_f32(_sum1, 2);
                outptr[M * 7] = vgetq_lane_f32(_sum1, 3);
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

        float* outptr = top_blob.row(p / out_elempack);

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
            const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const float* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __aarch64__
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        _r1 = vld1q_f32(r0 + N);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r1 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        _r1 = vsetq_lane_f32(r0[N * 4], _r1, 0);
                        _r1 = vsetq_lane_f32(r0[N * 5], _r1, 1);
                        _r1 = vsetq_lane_f32(r0[N * 6], _r1, 2);
                        _r1 = vsetq_lane_f32(r0[N * 7], _r1, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    float32x4_t _w4 = vld1q_f32(kptr + 16);
                    float32x4_t _w5 = vld1q_f32(kptr + 20);
                    float32x4_t _w6 = vld1q_f32(kptr + 24);
                    float32x4_t _w7 = vld1q_f32(kptr + 28);
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
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
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
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = r0[0];
                        val1 = r0[N];
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
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
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _val;
                    // if (elempack == 1)
                    {
                        _val = vdupq_n_f32(r0[0]);
                        r0 += dilation_w;
                    }

                    float32x4_t _w = vld1q_f32(kptr);
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
                vst1q_f32(outptr, _sum0);
                outptr += 4;
            }
            else // if (out_elempack == 1)
            {
                outptr[0] = vgetq_lane_f32(_sum0, 0);
                outptr[M] = vgetq_lane_f32(_sum0, 1);
                outptr[M * 2] = vgetq_lane_f32(_sum0, 2);
                outptr[M * 3] = vgetq_lane_f32(_sum0, 3);
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

        float* outptr0 = top_blob.row(p);
        float* outptr1 = top_blob.row(p + 1);

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
            const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __ARM_NEON
            const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
            const float* kptr = weight_data_tm.channel(p / 2);
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
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        _r1 = vld1q_f32(r0 + N);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r1 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        _r1 = vsetq_lane_f32(r0[N * 4], _r1, 0);
                        _r1 = vsetq_lane_f32(r0[N * 5], _r1, 1);
                        _r1 = vsetq_lane_f32(r0[N * 6], _r1, 2);
                        _r1 = vsetq_lane_f32(r0[N * 7], _r1, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
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
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
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
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = r0[0];
                        val1 = r0[N];
                        r0 += dilation_w;
                    }

                    sum0 += val0 * kptr[0];
                    sum1 += val0 * kptr[1];
                    sum0 += val1 * kptr[2];
                    sum1 += val1 * kptr[3];

                    kptr += 4;
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val;
                    // if (elempack == 1)
                    {
                        val = r0[0];
                        r0 += dilation_w;
                    }

                    sum0 += val * kptr[0];
                    sum1 += val * kptr[1];

                    kptr += 2;
                }
            }

            sum0 = activation_ss(sum0, activation_type, activation_params);
            sum1 = activation_ss(sum1, activation_type, activation_params);

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outh_start += nn_outh * 2;
    for (int p = remain_outh_start; p < outh; p++)
    {
        float* outptr = top_blob.row(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
            {
                sum = bias_data_ptr[p];
            }

#if __aarch64__
            const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __ARM_NEON
            const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
            const float* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __ARM_NEON
#if __aarch64__
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    float32x4_t _r1;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        _r1 = vld1q_f32(r0 + N);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r1 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        _r1 = vsetq_lane_f32(r0[N * 4], _r1, 0);
                        _r1 = vsetq_lane_f32(r0[N * 5], _r1, 1);
                        _r1 = vsetq_lane_f32(r0[N * 6], _r1, 2);
                        _r1 = vsetq_lane_f32(r0[N * 7], _r1, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
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
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                for (int k = 0; k < kernel_w; k++)
                {
                    float32x4_t _r0;
                    if (elempack == 4)
                    {
                        _r0 = vld1q_f32(r0);
                        r0 += dilation_w * 4;
                    }
                    else // if (elempack == 1)
                    {
                        _r0 = float32x4_t();
                        _r0 = vsetq_lane_f32(r0[0], _r0, 0);
                        _r0 = vsetq_lane_f32(r0[N], _r0, 1);
                        _r0 = vsetq_lane_f32(r0[N * 2], _r0, 2);
                        _r0 = vsetq_lane_f32(r0[N * 3], _r0, 3);
                        r0 += dilation_w;
                    }

                    float32x4_t _w = vld1q_f32(kptr);
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
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val0;
                    float val1;
                    // if (elempack == 1)
                    {
                        val0 = r0[0];
                        val1 = r0[N];
                        r0 += dilation_w;
                    }

                    sum += val0 * kptr[0];
                    sum += val1 * kptr[1];

                    kptr += 2;
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val;
                    // if (elempack == 1)
                    {
                        val = r0[0];
                        r0 += dilation_w;
                    }

                    sum += val * kptr[0];

                    kptr += 1;
                }
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
