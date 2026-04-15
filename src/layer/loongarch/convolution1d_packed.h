// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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
    // src = kernel(outh, inh, kernel_w)

    // clang-format off
    // *INDENT-OFF*
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
    if (outh >= 4)
    {
#if __loongarch_asx
        if (inh >= 8)
            kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
#endif // __loongarch_asx
        if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __loongarch_sx
    if (outh >= 2)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (inh >= 8)
            kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __loongarch_asx
        if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __loongarch_sx
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2);
    }
    else
    {
#if __loongarch_sx
#if __loongarch_asx
        if (inh >= 8)
            kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __loongarch_asx
        if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __loongarch_sx
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh);
        else
            kernel_tm.create(kernel_w, inh, outh);
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

        float* g00 = kernel_tm.channel(q / 8);

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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    g00[4] = k4[0];
                    g00[5] = k5[0];
                    g00[6] = k6[0];
                    g00[7] = k7[0];
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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    g00[4] = k4[0];
                    g00[5] = k5[0];
                    g00[6] = k6[0];
                    g00[7] = k7[0];
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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    g00[4] = k4[0];
                    g00[5] = k5[0];
                    g00[6] = k6[0];
                    g00[7] = k7[0];
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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

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
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00 += 4;
            }
        }
    }
#endif // __loongarch_sx
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;

#if __loongarch_sx
#if __loongarch_asx
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#endif
#else
        float* g00 = kernel_tm.channel(q / 2);
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
                    g00[0] = k0[0];
                    g00[1] = k1[0];
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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;

#if __loongarch_sx
#if __loongarch_asx
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#endif
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
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
                    g00[0] = k0[0];
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
                    g00[0] = k0[0];
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
                    g00[0] = k0[0];
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
                const float* k0 = kptr0 + k;

                g00[0] = k0[0];
                g00 += 1;
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

#if __loongarch_sx
#if __loongarch_asx
    nn_outh = outh / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = pp * 8;

        // shadowed variables for thread safety
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.row(p / out_elempack);

        for (int j = 0; j < outw; j++)
        {
            __m256 _sum0 = (__m256)__lasx_xvldi(0);
            __m256 _sum1 = (__m256)__lasx_xvldi(0);
            __m256 _sum2 = (__m256)__lasx_xvldi(0);
            __m256 _sum3 = (__m256)__lasx_xvldi(0);
            __m256 _sum4 = (__m256)__lasx_xvldi(0);
            __m256 _sum5 = (__m256)__lasx_xvldi(0);
            __m256 _sum6 = (__m256)__lasx_xvldi(0);
            __m256 _sum7 = (__m256)__lasx_xvldi(0);

            if (bias_data_ptr)
            {
                _sum0 = (__m256)__lasx_xvld((const float*)bias_data_ptr + p, 0);
            }

            const float* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = (__m256)__lasx_xvld(kptr, 0);
                        __m256 _w1 = (__m256)__lasx_xvld(kptr + 8, 0);
                        __m256 _w2 = (__m256)__lasx_xvld(kptr + 16, 0);
                        __m256 _w3 = (__m256)__lasx_xvld(kptr + 24, 0);
                        __m256 _w4 = (__m256)__lasx_xvld(kptr + 32, 0);
                        __m256 _w5 = (__m256)__lasx_xvld(kptr + 40, 0);
                        __m256 _w6 = (__m256)__lasx_xvld(kptr + 48, 0);
                        __m256 _w7 = (__m256)__lasx_xvld(kptr + 56, 0);

                        _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(r0[1]), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(r0[2]), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(r0[3]), _sum3);
                        _sum4 = __lasx_xvfmadd_s(_w4, __lasx_xvreplfr2vr_s(r0[4]), _sum4);
                        _sum5 = __lasx_xvfmadd_s(_w5, __lasx_xvreplfr2vr_s(r0[5]), _sum5);
                        _sum6 = __lasx_xvfmadd_s(_w6, __lasx_xvreplfr2vr_s(r0[6]), _sum6);
                        _sum7 = __lasx_xvfmadd_s(_w7, __lasx_xvreplfr2vr_s(r0[7]), _sum7);

                        r0 += dilation_w * 8;
                        kptr += 64;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = (__m256)__lasx_xvld(kptr, 0);
                        __m256 _w1 = (__m256)__lasx_xvld(kptr + 8, 0);
                        __m256 _w2 = (__m256)__lasx_xvld(kptr + 16, 0);
                        __m256 _w3 = (__m256)__lasx_xvld(kptr + 24, 0);
                        __m256 _w4 = (__m256)__lasx_xvld(kptr + 32, 0);
                        __m256 _w5 = (__m256)__lasx_xvld(kptr + 40, 0);
                        __m256 _w6 = (__m256)__lasx_xvld(kptr + 48, 0);
                        __m256 _w7 = (__m256)__lasx_xvld(kptr + 56, 0);

                        _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(r0[1]), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(r0[2]), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(r0[3]), _sum3);
                        _sum4 = __lasx_xvfmadd_s(_w4, __lasx_xvreplfr2vr_s(r1[0]), _sum4);
                        _sum5 = __lasx_xvfmadd_s(_w5, __lasx_xvreplfr2vr_s(r1[1]), _sum5);
                        _sum6 = __lasx_xvfmadd_s(_w6, __lasx_xvreplfr2vr_s(r1[2]), _sum6);
                        _sum7 = __lasx_xvfmadd_s(_w7, __lasx_xvreplfr2vr_s(r1[3]), _sum7);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 64;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = (__m256)__lasx_xvld(kptr, 0);
                        __m256 _w1 = (__m256)__lasx_xvld(kptr + 8, 0);
                        __m256 _w2 = (__m256)__lasx_xvld(kptr + 16, 0);
                        __m256 _w3 = (__m256)__lasx_xvld(kptr + 24, 0);
                        __m256 _w4 = (__m256)__lasx_xvld(kptr + 32, 0);
                        __m256 _w5 = (__m256)__lasx_xvld(kptr + 40, 0);
                        __m256 _w6 = (__m256)__lasx_xvld(kptr + 48, 0);
                        __m256 _w7 = (__m256)__lasx_xvld(kptr + 56, 0);

                        _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(r0[N]), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(r0[N * 2]), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(r0[N * 3]), _sum3);
                        _sum4 = __lasx_xvfmadd_s(_w4, __lasx_xvreplfr2vr_s(r0[N * 4]), _sum4);
                        _sum5 = __lasx_xvfmadd_s(_w5, __lasx_xvreplfr2vr_s(r0[N * 5]), _sum5);
                        _sum6 = __lasx_xvfmadd_s(_w6, __lasx_xvreplfr2vr_s(r0[N * 6]), _sum6);
                        _sum7 = __lasx_xvfmadd_s(_w7, __lasx_xvreplfr2vr_s(r0[N * 7]), _sum7);

                        r0 += dilation_w;
                        kptr += 64;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = (__m256)__lasx_xvld(kptr, 0);
                        __m256 _w1 = (__m256)__lasx_xvld(kptr + 8, 0);
                        __m256 _w2 = (__m256)__lasx_xvld(kptr + 16, 0);
                        __m256 _w3 = (__m256)__lasx_xvld(kptr + 24, 0);

                        _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(r0[1]), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(r0[2]), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(r0[3]), _sum3);

                        r0 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = (__m256)__lasx_xvld(kptr, 0);
                        __m256 _w1 = (__m256)__lasx_xvld(kptr + 8, 0);
                        __m256 _w2 = (__m256)__lasx_xvld(kptr + 16, 0);
                        __m256 _w3 = (__m256)__lasx_xvld(kptr + 24, 0);

                        _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(r0[N]), _sum1);
                        _sum2 = __lasx_xvfmadd_s(_w2, __lasx_xvreplfr2vr_s(r0[N * 2]), _sum2);
                        _sum3 = __lasx_xvfmadd_s(_w3, __lasx_xvreplfr2vr_s(r0[N * 3]), _sum3);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = (__m256)__lasx_xvld(kptr, 0);
                        __m256 _w1 = (__m256)__lasx_xvld(kptr + 8, 0);

                        _sum0 = __lasx_xvfmadd_s(_w0, __lasx_xvreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lasx_xvfmadd_s(_w1, __lasx_xvreplfr2vr_s(r0[N]), _sum1);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _val = __lasx_xvreplfr2vr_s(r0[0]);
                        __m256 _w = (__m256)__lasx_xvld(kptr, 0);
                        _sum0 = __lasx_xvfmadd_s(_val, _w, _sum0);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }

            _sum0 = __lasx_xvfadd_s(_sum0, _sum1);
            _sum2 = __lasx_xvfadd_s(_sum2, _sum3);
            _sum4 = __lasx_xvfadd_s(_sum4, _sum5);
            _sum6 = __lasx_xvfadd_s(_sum6, _sum7);
            _sum0 = __lasx_xvfadd_s(_sum0, _sum2);
            _sum4 = __lasx_xvfadd_s(_sum4, _sum6);
            _sum0 = __lasx_xvfadd_s(_sum0, _sum4);

            _sum0 = activation_lasx(_sum0, activation_type, activation_params);

            if (out_elempack == 8)
            {
                __lasx_xvst(_sum0, outptr, 0);
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                __lsx_vst((__m128)__lasx_extract_lo128((__m256i)_sum0), outptr, 0);
                __lsx_vst((__m128)__lasx_extract_hi128((__m256i)_sum0), outptr + M, 0);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                float sum[8];
                __lasx_xvst(_sum0, sum, 0);

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
    remain_outh_start += nn_outh * 8;
    nn_outh = (outh - remain_outh_start) / 4;
#else  // __loongarch_asx
    nn_outh = outh / 4;
#endif // __loongarch_asx
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 4;

        // shadowed variables for thread safety
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.row(p / out_elempack);

        for (int j = 0; j < outw; j++)
        {
            __m128 _sum0 = (__m128)__lsx_vldi(0);
            __m128 _sum1 = (__m128)__lsx_vldi(0);
            __m128 _sum2 = (__m128)__lsx_vldi(0);
            __m128 _sum3 = (__m128)__lsx_vldi(0);

            if (bias_data_ptr)
            {
                _sum0 = (__m128)__lsx_vld((const float*)bias_data_ptr + p, 0);
            }

#if __loongarch_asx
            const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const float* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __loongarch_asx
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                        __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);
                        __m128 _w4 = (__m128)__lsx_vld(kptr + 16, 0);
                        __m128 _w5 = (__m128)__lsx_vld(kptr + 20, 0);
                        __m128 _w6 = (__m128)__lsx_vld(kptr + 24, 0);
                        __m128 _w7 = (__m128)__lsx_vld(kptr + 28, 0);

                        _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(r0[1]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(r0[2]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(r0[3]), _sum3);
                        _sum0 = __lsx_vfmadd_s(_w4, __lsx_vreplfr2vr_s(r0[4]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w5, __lsx_vreplfr2vr_s(r0[5]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w6, __lsx_vreplfr2vr_s(r0[6]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w7, __lsx_vreplfr2vr_s(r0[7]), _sum3);

                        r0 += dilation_w * 8;
                        kptr += 32;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                        __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);
                        __m128 _w4 = (__m128)__lsx_vld(kptr + 16, 0);
                        __m128 _w5 = (__m128)__lsx_vld(kptr + 20, 0);
                        __m128 _w6 = (__m128)__lsx_vld(kptr + 24, 0);
                        __m128 _w7 = (__m128)__lsx_vld(kptr + 28, 0);

                        _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(r0[1]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(r0[2]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(r0[3]), _sum3);
                        _sum0 = __lsx_vfmadd_s(_w4, __lsx_vreplfr2vr_s(r1[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w5, __lsx_vreplfr2vr_s(r1[1]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w6, __lsx_vreplfr2vr_s(r1[2]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w7, __lsx_vreplfr2vr_s(r1[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                        __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);
                        __m128 _w4 = (__m128)__lsx_vld(kptr + 16, 0);
                        __m128 _w5 = (__m128)__lsx_vld(kptr + 20, 0);
                        __m128 _w6 = (__m128)__lsx_vld(kptr + 24, 0);
                        __m128 _w7 = (__m128)__lsx_vld(kptr + 28, 0);

                        _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(r0[N]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(r0[N * 2]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(r0[N * 3]), _sum3);
                        _sum0 = __lsx_vfmadd_s(_w4, __lsx_vreplfr2vr_s(r0[N * 4]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w5, __lsx_vreplfr2vr_s(r0[N * 5]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w6, __lsx_vreplfr2vr_s(r0[N * 6]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w7, __lsx_vreplfr2vr_s(r0[N * 7]), _sum3);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
#endif // __loongarch_asx
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                        __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);

                        _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(r0[1]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(r0[2]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(r0[3]), _sum3);

                        r0 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                        __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);

                        _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(r0[N]), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, __lsx_vreplfr2vr_s(r0[N * 2]), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, __lsx_vreplfr2vr_s(r0[N * 3]), _sum3);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);

                        _sum0 = __lsx_vfmadd_s(_w0, __lsx_vreplfr2vr_s(r0[0]), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, __lsx_vreplfr2vr_s(r0[N]), _sum1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _val = __lsx_vreplfr2vr_s(r0[0]);
                        __m128 _w = (__m128)__lsx_vld(kptr, 0);
                        _sum0 = __lsx_vfmadd_s(_val, _w, _sum0);

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
                __lsx_vst(_sum0, outptr, 0);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                float sum[4];
                __lsx_vst(_sum0, sum, 0);

                outptr[0] = sum[0];
                outptr[M] = sum[1];
                outptr[M * 2] = sum[2];
                outptr[M * 3] = sum[3];
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

        // shadowed variables for thread safety
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
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            sum0 += r0[i] * kptr[i];
                            sum1 += r0[i] * kptr[8 + i];
                        }

                        r0 += dilation_w * 8;
                        kptr += 16;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            sum0 += r0[i] * kptr[i];
                            sum1 += r0[i] * kptr[8 + i];
                        }
                        for (int i = 0; i < 4; i++)
                        {
                            sum0 += r1[i] * kptr[4 + i];
                            sum1 += r1[i] * kptr[12 + i];
                        }

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            float val = r0[N * i];
                            sum0 += val * kptr[i];
                            sum1 += val * kptr[8 + i];
                        }

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
#endif // __loongarch_asx
            __m128 _sum0l = (__m128)__lsx_vldi(0);
            __m128 _sum1l = (__m128)__lsx_vldi(0);
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        _sum0l = __lsx_vfmadd_s(_r0, _w0, _sum0l);
                        _sum1l = __lsx_vfmadd_s(_r0, _w1, _sum1l);

                        r0 += dilation_w * 4;
                        kptr += 8;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0;
                        float tmp[4] = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        _r0 = (__m128)__lsx_vld(tmp, 0);
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        _sum0l = __lsx_vfmadd_s(_r0, _w0, _sum0l);
                        _sum1l = __lsx_vfmadd_s(_r0, _w1, _sum1l);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            sum0 += __lsx_reduce_fadd_s(_sum0l);
            sum1 += __lsx_reduce_fadd_s(_sum1l);
#endif // __loongarch_sx
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        sum0 += r0[0] * kptr[0];
                        sum1 += r0[0] * kptr[1];
                        sum0 += r0[N] * kptr[2];
                        sum1 += r0[N] * kptr[3];

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        float val = r0[0];
                        sum0 += val * kptr[0];
                        sum1 += val * kptr[1];

                        r0 += dilation_w;
                        kptr += 2;
                    }
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outh_start; p < outh; p++)
    {
        // shadowed variables for thread safety
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;

        float* outptr = top_blob.row(p);

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
            {
                __m256 _sum = (__m256)__lasx_xvldi(0);
                for (; q + 7 < inh; q += 8)
                {
                    const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                            __m256 _w = (__m256)__lasx_xvld(kptr, 0);
                            _sum = __lasx_xvfmadd_s(_r0, _w, _sum);

                            r0 += dilation_w * 8;
                            kptr += 8;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                            __m128 _r1 = (__m128)__lsx_vld(r1, 0);
                            __m256 _r01 = combine4x2_ps(_r0, _r1);
                            __m256 _w = (__m256)__lasx_xvld(kptr, 0);
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
                            float tmp[8] = {r0[0], r0[N], r0[N * 2], r0[N * 3], r0[N * 4], r0[N * 5], r0[N * 6], r0[N * 7]};
                            __m256 _r0 = (__m256)__lasx_xvld(tmp, 0);
                            __m256 _w = (__m256)__lasx_xvld(kptr, 0);
                            _sum = __lasx_xvfmadd_s(_r0, _w, _sum);

                            r0 += dilation_w;
                            kptr += 8;
                        }
                    }
                }
                __m128 _slo = (__m128)__lasx_extract_lo128((__m256i)_sum);
                __m128 _shi = (__m128)__lasx_extract_hi128((__m256i)_sum);
                __m128 _ss = __lsx_vfadd_s(_slo, _shi);
                sum += __lsx_reduce_fadd_s(_ss);
            }
#endif // __loongarch_asx
            __m128 _sum = (__m128)__lsx_vldi(0);
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _w = (__m128)__lsx_vld(kptr, 0);
                        _sum = __lsx_vfmadd_s(_r0, _w, _sum);

                        r0 += dilation_w * 4;
                        kptr += 4;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        float tmp[4] = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        __m128 _r0 = (__m128)__lsx_vld(tmp, 0);
                        __m128 _w = (__m128)__lsx_vld(kptr, 0);
                        _sum = __lsx_vfmadd_s(_r0, _w, _sum);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        sum += r0[0] * kptr[0];
                        sum += r0[N] * kptr[1];

                        r0 += dilation_w;
                        kptr += 2;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        float val = r0[0];
                        sum += val * kptr[0];

                        r0 += dilation_w;
                        kptr += 1;
                    }
                }
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
