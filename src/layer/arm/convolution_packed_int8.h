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

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
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
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k2[0];
                    k2 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k3[0];
                    k3 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k4[0];
                    k4 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k5[0];
                    k5 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k6[0];
                    k6 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k7[0];
                    k7 += maxk;
                    g00 += 1;
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
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k2[0];
                    k2 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k3[0];
                    k3 += maxk;
                    g00 += 1;
                }
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
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            {
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
                int32x4_t _sum45 = vdupq_n_s32(0);
                int32x4_t _sum67 = vdupq_n_s32(0);
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

                        int8x8_t _w0 = vld1_s8(kptr);
                        int8x8_t _w1 = vld1_s8(kptr + 8);
                        int8x8_t _w2 = vld1_s8(kptr + 16);
                        int8x8_t _w3 = vld1_s8(kptr + 24);
                        int8x8_t _w4 = vld1_s8(kptr + 32);
                        int8x8_t _w5 = vld1_s8(kptr + 40);
                        int8x8_t _w6 = vld1_s8(kptr + 48);
                        int8x8_t _w7 = vld1_s8(kptr + 56);

                        int16x8_t _wv0 = vmull_s8(_r0, _w0);
                        int16x8_t _wv1 = vmull_s8(_r0, _w1);
                        int16x8_t _wv2 = vmull_s8(_r0, _w2);
                        int16x8_t _wv3 = vmull_s8(_r0, _w3);
                        int16x8_t _wv4 = vmull_s8(_r0, _w4);
                        int16x8_t _wv5 = vmull_s8(_r0, _w5);
                        int16x8_t _wv6 = vmull_s8(_r0, _w6);
                        int16x8_t _wv7 = vmull_s8(_r0, _w7);

                        int16x4_t _wv00 = vpadd_s16(vget_low_s16(_wv0), vget_high_s16(_wv0));
                        int16x4_t _wv11 = vpadd_s16(vget_low_s16(_wv1), vget_high_s16(_wv1));
                        int16x4_t _wv22 = vpadd_s16(vget_low_s16(_wv2), vget_high_s16(_wv2));
                        int16x4_t _wv33 = vpadd_s16(vget_low_s16(_wv3), vget_high_s16(_wv3));
                        int16x4_t _wv44 = vpadd_s16(vget_low_s16(_wv4), vget_high_s16(_wv4));
                        int16x4_t _wv55 = vpadd_s16(vget_low_s16(_wv5), vget_high_s16(_wv5));
                        int16x4_t _wv66 = vpadd_s16(vget_low_s16(_wv6), vget_high_s16(_wv6));
                        int16x4_t _wv77 = vpadd_s16(vget_low_s16(_wv7), vget_high_s16(_wv7));

                        _sum01 = vpadalq_s16(_sum01, vcombine_s16(_wv00, _wv11));
                        _sum23 = vpadalq_s16(_sum23, vcombine_s16(_wv22, _wv33));
                        _sum45 = vpadalq_s16(_sum45, vcombine_s16(_wv44, _wv55));
                        _sum67 = vpadalq_s16(_sum67, vcombine_s16(_wv66, _wv77));

                        kptr += 64;
                    }
                }
                int32x4_t _s0 = vcombine_s32(vpadd_s32(vget_low_s32(_sum01), vget_high_s32(_sum01)), vpadd_s32(vget_low_s32(_sum23), vget_high_s32(_sum23)));
                int32x4_t _s1 = vcombine_s32(vpadd_s32(vget_low_s32(_sum45), vget_high_s32(_sum45)), vpadd_s32(vget_low_s32(_sum67), vget_high_s32(_sum67)));
                _sum0 = vaddq_s32(_sum0, _s0);
                _sum1 = vaddq_s32(_sum1, _s1);
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
                int sum[8];
                vst1q_s32(sum, _sum0);
                vst1q_s32(sum + 4, _sum1);

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
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int32x4_t _sum0 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);

            int q = 0;
            {
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
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

                        int8x8_t _w0 = vld1_s8(kptr);
                        int8x8_t _w1 = vld1_s8(kptr + 8);
                        int8x8_t _w2 = vld1_s8(kptr + 16);
                        int8x8_t _w3 = vld1_s8(kptr + 24);

                        int16x8_t _wv0 = vmull_s8(_r0, _w0);
                        int16x8_t _wv1 = vmull_s8(_r0, _w1);
                        int16x8_t _wv2 = vmull_s8(_r0, _w2);
                        int16x8_t _wv3 = vmull_s8(_r0, _w3);

                        int16x4_t _wv00 = vpadd_s16(vget_low_s16(_wv0), vget_high_s16(_wv0));
                        int16x4_t _wv11 = vpadd_s16(vget_low_s16(_wv1), vget_high_s16(_wv1));
                        int16x4_t _wv22 = vpadd_s16(vget_low_s16(_wv2), vget_high_s16(_wv2));
                        int16x4_t _wv33 = vpadd_s16(vget_low_s16(_wv3), vget_high_s16(_wv3));

                        _sum01 = vpadalq_s16(_sum01, vcombine_s16(_wv00, _wv11));
                        _sum23 = vpadalq_s16(_sum23, vcombine_s16(_wv22, _wv33));

                        kptr += 32;
                    }
                }
                int32x4_t _s0 = vcombine_s32(vpadd_s32(vget_low_s32(_sum01), vget_high_s32(_sum01)), vpadd_s32(vget_low_s32(_sum23), vget_high_s32(_sum23)));
                _sum0 = vaddq_s32(_sum0, _s0);
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
                int sum[4];
                vst1q_s32(sum, _sum0);

                outptr[0] = sum[0];
                outptr[M] = sum[1];
                outptr[M * 2] = sum[2];
                outptr[M * 3] = sum[3];
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

                        int8x8_t _w0 = vld1_s8(kptr);
                        int8x8_t _w1 = vld1_s8(kptr + 8);

                        int16x8_t _wv0 = vmull_s8(_r0, _w0);
                        int16x8_t _wv1 = vmull_s8(_r0, _w1);

                        int16x4_t _wv00 = vpadd_s16(vget_low_s16(_wv0), vget_high_s16(_wv0));
                        int16x4_t _wv11 = vpadd_s16(vget_low_s16(_wv1), vget_high_s16(_wv1));

                        _sum01 = vpadalq_s16(_sum01, vcombine_s16(_wv00, _wv11));

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

                        int16x8_t _s8 = vmull_s8(_r0, _w);

                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

                        kptr += 8;
                    }
                }
                int32x4_t _sum = vaddq_s32(_sum0, _sum1);
#if __aarch64__
                sum += vaddvq_s32(_sum); // dot
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
