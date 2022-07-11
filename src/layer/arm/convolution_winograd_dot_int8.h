// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_winograd_dot_int8_neon(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 2u, 1, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __ARM_NEON
#if __aarch64__
    if (tiles >= 8)
        bottom_blob_tm2.create(inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, batch, 16u, 8, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(inch, tiles / 4 + tiles % 4, batch, 8u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(inch, tiles, batch, 2u, 1, opt.workspace_allocator);
#else
    if (tiles >= 4)
        bottom_blob_tm2.create(inch, tiles / 4 + tiles % 4, batch, 8u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(inch, tiles, batch, 2u, 1, opt.workspace_allocator);
#endif
#else  // __ARM_NEON
    if (tiles >= 2)
        bottom_blob_tm2.create(inch, tiles / 2 + tiles % 2, batch, 4u, 2, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(inch, tiles, batch, 2u, 1, opt.workspace_allocator);
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
#if __ARM_NEON
#if __aarch64__
        for (; i + 7 < tiles; i += 8)
        {
            short* tmpptr = tm2.row<short>(i / 8);
            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
            for (; q < inch; q++)
            {
                int16x8_t _r0 = vld1q_s16(r0);
                vst1q_s16(tmpptr, _r0);
                r0 += bottom_blob_tm.cstep;
                tmpptr += 8;
            }
        }
#endif
        for (; i + 3 < tiles; i += 4)
        {
#if __aarch64__
            short* tmpptr = tm2.row<short>(i / 8 + (i % 8) / 4);
#else
            short* tmpptr = tm2.row<short>(i / 4);
#endif
            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
            for (; q < inch; q++)
            {
                int16x4_t _r0 = vld1_s16(r0);
                vst1_s16(tmpptr, _r0);
                r0 += bottom_blob_tm.cstep;
                tmpptr += 4;
            }
        }
#else // __ARM_NEON
        for (; i + 1 < tiles; i += 2)
        {
            short* tmpptr = tm2.row<short>(i / 2);
            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
#if __ARM_FEATURE_SIMD32
            for (; q + 1 < inch; q += 2)
            {
                tmpptr[0] = r0[0];
                tmpptr[2] = r0[1];
                r0 += bottom_blob_tm.cstep;
                tmpptr[1] = r0[0];
                tmpptr[3] = r0[1];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 4;
            }
#endif // __ARM_FEATURE_SIMD32
            for (; q < inch; q++)
            {
                tmpptr[0] = r0[0];
                tmpptr[1] = r0[1];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 2;
            }
        }
#endif // __ARM_NEON
        for (; i < tiles; i++)
        {
#if __ARM_NEON
#if __aarch64__
            short* tmpptr = tm2.row<short>(i / 8 + (i % 8) / 4 + i % 4);
#else
            short* tmpptr = tm2.row<short>(i / 4 + i % 4);
#endif
#else
            short* tmpptr = tm2.row<short>(i / 2 + i % 2);
#endif
            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
            for (; q < inch; q++)
            {
                tmpptr[0] = r0[0];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 1;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 4u, 1, opt.workspace_allocator);

#if __ARM_NEON
    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);
        int* output2_tm = top_blob_tm.channel(p + 2);
        int* output3_tm = top_blob_tm.channel(p + 3);
        int* output4_tm = top_blob_tm.channel(p + 4);
        int* output5_tm = top_blob_tm.channel(p + 5);
        int* output6_tm = top_blob_tm.channel(p + 6);
        int* output7_tm = top_blob_tm.channel(p + 7);

        const Mat kernel0_tm = kernel_tm.channel(p / 8);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
#if __aarch64__
            for (; i + 7 < tiles; i += 8)
            {
                const short* r0 = bb2.row<const short>(i / 8);
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum20 = vdupq_n_s32(0);
                int32x4_t _sum30 = vdupq_n_s32(0);
                int32x4_t _sum40 = vdupq_n_s32(0);
                int32x4_t _sum50 = vdupq_n_s32(0);
                int32x4_t _sum60 = vdupq_n_s32(0);
                int32x4_t _sum70 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum21 = vdupq_n_s32(0);
                int32x4_t _sum31 = vdupq_n_s32(0);
                int32x4_t _sum41 = vdupq_n_s32(0);
                int32x4_t _sum51 = vdupq_n_s32(0);
                int32x4_t _sum61 = vdupq_n_s32(0);
                int32x4_t _sum71 = vdupq_n_s32(0);

                int j = 0;
                for (; j < inch; j++)
                {
                    int16x8_t _val0 = vld1q_s16(r0);
                    int16x8_t _w0 = vld1q_s16(k0);

                    _sum00 = vmlal_lane_s16(_sum00, vget_low_s16(_val0), vget_low_s16(_w0), 0);
                    _sum10 = vmlal_lane_s16(_sum10, vget_low_s16(_val0), vget_low_s16(_w0), 1);
                    _sum20 = vmlal_lane_s16(_sum20, vget_low_s16(_val0), vget_low_s16(_w0), 2);
                    _sum30 = vmlal_lane_s16(_sum30, vget_low_s16(_val0), vget_low_s16(_w0), 3);
                    _sum40 = vmlal_lane_s16(_sum40, vget_low_s16(_val0), vget_high_s16(_w0), 0);
                    _sum50 = vmlal_lane_s16(_sum50, vget_low_s16(_val0), vget_high_s16(_w0), 1);
                    _sum60 = vmlal_lane_s16(_sum60, vget_low_s16(_val0), vget_high_s16(_w0), 2);
                    _sum70 = vmlal_lane_s16(_sum70, vget_low_s16(_val0), vget_high_s16(_w0), 3);

                    _sum01 = vmlal_lane_s16(_sum01, vget_high_s16(_val0), vget_low_s16(_w0), 0);
                    _sum11 = vmlal_lane_s16(_sum11, vget_high_s16(_val0), vget_low_s16(_w0), 1);
                    _sum21 = vmlal_lane_s16(_sum21, vget_high_s16(_val0), vget_low_s16(_w0), 2);
                    _sum31 = vmlal_lane_s16(_sum31, vget_high_s16(_val0), vget_low_s16(_w0), 3);
                    _sum41 = vmlal_lane_s16(_sum41, vget_high_s16(_val0), vget_high_s16(_w0), 0);
                    _sum51 = vmlal_lane_s16(_sum51, vget_high_s16(_val0), vget_high_s16(_w0), 1);
                    _sum61 = vmlal_lane_s16(_sum61, vget_high_s16(_val0), vget_high_s16(_w0), 2);
                    _sum71 = vmlal_lane_s16(_sum71, vget_high_s16(_val0), vget_high_s16(_w0), 3);

                    r0 += 8;
                    k0 += 8;
                }

                vst1q_s32(output0_tm, _sum00);
                vst1q_s32(output0_tm + 4, _sum01);
                vst1q_s32(output1_tm, _sum10);
                vst1q_s32(output1_tm + 4, _sum11);
                vst1q_s32(output2_tm, _sum20);
                vst1q_s32(output2_tm + 4, _sum21);
                vst1q_s32(output3_tm, _sum30);
                vst1q_s32(output3_tm + 4, _sum31);
                vst1q_s32(output4_tm, _sum40);
                vst1q_s32(output4_tm + 4, _sum41);
                vst1q_s32(output5_tm, _sum50);
                vst1q_s32(output5_tm + 4, _sum51);
                vst1q_s32(output6_tm, _sum60);
                vst1q_s32(output6_tm + 4, _sum61);
                vst1q_s32(output7_tm, _sum70);
                vst1q_s32(output7_tm + 4, _sum71);

                output0_tm += 8;
                output1_tm += 8;
                output2_tm += 8;
                output3_tm += 8;
                output4_tm += 8;
                output5_tm += 8;
                output6_tm += 8;
                output7_tm += 8;
            }
#endif // __aarch64__
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4);
#else
                const short* r0 = bb2.row<const short>(i / 4);
#endif
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < inch; j += 2)
                {
                    int16x8_t _val01 = vld1q_s16(r0);
                    int16x8_t _w0 = vld1q_s16(k0);
                    int16x8_t _w1 = vld1q_s16(k0 + 8);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val01), vget_low_s16(_w0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_val01), vget_low_s16(_w0), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val01), vget_low_s16(_w0), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_val01), vget_low_s16(_w0), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_val01), vget_high_s16(_w0), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_val01), vget_high_s16(_w0), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_val01), vget_high_s16(_w0), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_val01), vget_high_s16(_w0), 3);

                    _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_val01), vget_low_s16(_w1), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val01), vget_low_s16(_w1), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_val01), vget_low_s16(_w1), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val01), vget_low_s16(_w1), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_val01), vget_high_s16(_w1), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_val01), vget_high_s16(_w1), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_val01), vget_high_s16(_w1), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_val01), vget_high_s16(_w1), 3);

                    r0 += 8;
                    k0 += 16;
                }
                for (; j < inch; j++)
                {
                    int16x4_t _val0 = vld1_s16(r0);
                    int16x8_t _w0 = vld1q_s16(k0);

                    _sum0 = vmlal_lane_s16(_sum0, _val0, vget_low_s16(_w0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, _val0, vget_low_s16(_w0), 1);
                    _sum2 = vmlal_lane_s16(_sum2, _val0, vget_low_s16(_w0), 2);
                    _sum3 = vmlal_lane_s16(_sum3, _val0, vget_low_s16(_w0), 3);
                    _sum4 = vmlal_lane_s16(_sum4, _val0, vget_high_s16(_w0), 0);
                    _sum5 = vmlal_lane_s16(_sum5, _val0, vget_high_s16(_w0), 1);
                    _sum6 = vmlal_lane_s16(_sum6, _val0, vget_high_s16(_w0), 2);
                    _sum7 = vmlal_lane_s16(_sum7, _val0, vget_high_s16(_w0), 3);

                    r0 += 4;
                    k0 += 8;
                }

                vst1q_s32(output0_tm, _sum0);
                vst1q_s32(output1_tm, _sum1);
                vst1q_s32(output2_tm, _sum2);
                vst1q_s32(output3_tm, _sum3);
                vst1q_s32(output4_tm, _sum4);
                vst1q_s32(output5_tm, _sum5);
                vst1q_s32(output6_tm, _sum6);
                vst1q_s32(output7_tm, _sum7);

                output0_tm += 4;
                output1_tm += 4;
                output2_tm += 4;
                output3_tm += 4;
                output4_tm += 4;
                output5_tm += 4;
                output6_tm += 4;
                output7_tm += 4;
            }
            for (; i < tiles; i++)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4 + i % 4);
#else
                const short* r0 = bb2.row<const short>(i / 4 + i % 4);
#endif
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 3 < inch; j += 4)
                {
                    int16x4_t _val0123 = vld1_s16(r0);
                    int16x8_t _w0 = vld1q_s16(k0);
                    int16x8_t _w1 = vld1q_s16(k0 + 8);
                    int16x8_t _w2 = vld1q_s16(k0 + 16);
                    int16x8_t _w3 = vld1q_s16(k0 + 24);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), _val0123, 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), _val0123, 0);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), _val0123, 1);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), _val0123, 1);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), _val0123, 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), _val0123, 2);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), _val0123, 3);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), _val0123, 3);

                    r0 += 4;
                    k0 += 32;
                }
                for (; j < inch; j++)
                {
                    int16x4_t _val0 = vld1_dup_s16(r0);
                    int16x8_t _w0 = vld1q_s16(k0);

                    _sum0 = vmlal_s16(_sum0, _val0, vget_low_s16(_w0));
                    _sum1 = vmlal_s16(_sum1, _val0, vget_high_s16(_w0));

                    r0 += 1;
                    k0 += 8;
                }

                output0_tm[0] = vgetq_lane_s32(_sum0, 0);
                output1_tm[0] = vgetq_lane_s32(_sum0, 1);
                output2_tm[0] = vgetq_lane_s32(_sum0, 2);
                output3_tm[0] = vgetq_lane_s32(_sum0, 3);
                output4_tm[0] = vgetq_lane_s32(_sum1, 0);
                output5_tm[0] = vgetq_lane_s32(_sum1, 1);
                output6_tm[0] = vgetq_lane_s32(_sum1, 2);
                output7_tm[0] = vgetq_lane_s32(_sum1, 3);
                output0_tm += 1;
                output1_tm += 1;
                output2_tm += 1;
                output3_tm += 1;
                output4_tm += 1;
                output5_tm += 1;
                output6_tm += 1;
                output7_tm += 1;
            }
        }
    }

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);
        int* output2_tm = top_blob_tm.channel(p + 2);
        int* output3_tm = top_blob_tm.channel(p + 3);

        const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
#if __aarch64__
            for (; i + 7 < tiles; i += 8)
            {
                const short* r0 = bb2.row<const short>(i / 8);
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum20 = vdupq_n_s32(0);
                int32x4_t _sum30 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum21 = vdupq_n_s32(0);
                int32x4_t _sum31 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < inch; j += 2)
                {
                    int16x8_t _val01 = vld1q_s16(r0);
                    int16x8_t _val23 = vld1q_s16(r0 + 8);
                    int16x8_t _w01 = vld1q_s16(k0);

                    _sum00 = vmlal_lane_s16(_sum00, vget_low_s16(_val01), vget_low_s16(_w01), 0);
                    _sum10 = vmlal_lane_s16(_sum10, vget_low_s16(_val01), vget_low_s16(_w01), 1);
                    _sum20 = vmlal_lane_s16(_sum20, vget_low_s16(_val01), vget_low_s16(_w01), 2);
                    _sum30 = vmlal_lane_s16(_sum30, vget_low_s16(_val01), vget_low_s16(_w01), 3);
                    _sum01 = vmlal_lane_s16(_sum01, vget_high_s16(_val01), vget_low_s16(_w01), 0);
                    _sum11 = vmlal_lane_s16(_sum11, vget_high_s16(_val01), vget_low_s16(_w01), 1);
                    _sum21 = vmlal_lane_s16(_sum21, vget_high_s16(_val01), vget_low_s16(_w01), 2);
                    _sum31 = vmlal_lane_s16(_sum31, vget_high_s16(_val01), vget_low_s16(_w01), 3);

                    _sum00 = vmlal_lane_s16(_sum00, vget_low_s16(_val23), vget_high_s16(_w01), 0);
                    _sum10 = vmlal_lane_s16(_sum10, vget_low_s16(_val23), vget_high_s16(_w01), 1);
                    _sum20 = vmlal_lane_s16(_sum20, vget_low_s16(_val23), vget_high_s16(_w01), 2);
                    _sum30 = vmlal_lane_s16(_sum30, vget_low_s16(_val23), vget_high_s16(_w01), 3);
                    _sum01 = vmlal_lane_s16(_sum01, vget_high_s16(_val23), vget_high_s16(_w01), 0);
                    _sum11 = vmlal_lane_s16(_sum11, vget_high_s16(_val23), vget_high_s16(_w01), 1);
                    _sum21 = vmlal_lane_s16(_sum21, vget_high_s16(_val23), vget_high_s16(_w01), 2);
                    _sum31 = vmlal_lane_s16(_sum31, vget_high_s16(_val23), vget_high_s16(_w01), 3);

                    r0 += 16;
                    k0 += 8;
                }
                for (; j < inch; j++)
                {
                    int16x8_t _val0 = vld1q_s16(r0);
                    int16x4_t _w0 = vld1_s16(k0);

                    _sum00 = vmlal_lane_s16(_sum00, vget_low_s16(_val0), _w0, 0);
                    _sum10 = vmlal_lane_s16(_sum10, vget_low_s16(_val0), _w0, 1);
                    _sum20 = vmlal_lane_s16(_sum20, vget_low_s16(_val0), _w0, 2);
                    _sum30 = vmlal_lane_s16(_sum30, vget_low_s16(_val0), _w0, 3);
                    _sum01 = vmlal_lane_s16(_sum01, vget_high_s16(_val0), _w0, 0);
                    _sum11 = vmlal_lane_s16(_sum11, vget_high_s16(_val0), _w0, 1);
                    _sum21 = vmlal_lane_s16(_sum21, vget_high_s16(_val0), _w0, 2);
                    _sum31 = vmlal_lane_s16(_sum31, vget_high_s16(_val0), _w0, 3);

                    r0 += 8;
                    k0 += 4;
                }

                vst1q_s32(output0_tm, _sum00);
                vst1q_s32(output0_tm + 4, _sum01);
                vst1q_s32(output1_tm, _sum10);
                vst1q_s32(output1_tm + 4, _sum11);
                vst1q_s32(output2_tm, _sum20);
                vst1q_s32(output2_tm + 4, _sum21);
                vst1q_s32(output3_tm, _sum30);
                vst1q_s32(output3_tm + 4, _sum31);

                output0_tm += 8;
                output1_tm += 8;
                output2_tm += 8;
                output3_tm += 8;
            }
#endif // __aarch64__
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4);
#else
                const short* r0 = bb2.row<const short>(i / 4);
#endif
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < inch; j += 2)
                {
                    int16x8_t _val01 = vld1q_s16(r0);
                    int16x8_t _w01 = vld1q_s16(k0);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val01), vget_low_s16(_w01), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_val01), vget_low_s16(_w01), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val01), vget_low_s16(_w01), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_val01), vget_low_s16(_w01), 3);
                    _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_val01), vget_high_s16(_w01), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val01), vget_high_s16(_w01), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_val01), vget_high_s16(_w01), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val01), vget_high_s16(_w01), 3);

                    r0 += 8;
                    k0 += 8;
                }
                for (; j < inch; j++)
                {
                    int16x4_t _val0 = vld1_s16(r0);
                    int16x4_t _w0 = vld1_s16(k0);

                    _sum0 = vmlal_lane_s16(_sum0, _val0, _w0, 0);
                    _sum1 = vmlal_lane_s16(_sum1, _val0, _w0, 1);
                    _sum2 = vmlal_lane_s16(_sum2, _val0, _w0, 2);
                    _sum3 = vmlal_lane_s16(_sum3, _val0, _w0, 3);

                    r0 += 4;
                    k0 += 4;
                }

                vst1q_s32(output0_tm, _sum0);
                vst1q_s32(output1_tm, _sum1);
                vst1q_s32(output2_tm, _sum2);
                vst1q_s32(output3_tm, _sum3);

                output0_tm += 4;
                output1_tm += 4;
                output2_tm += 4;
                output3_tm += 4;
            }
            for (; i < tiles; i++)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4 + i % 4);
#else
                const short* r0 = bb2.row<const short>(i / 4 + i % 4);
#endif
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 3 < inch; j += 4)
                {
                    int16x4_t _val0123 = vld1_s16(r0);
                    int16x8_t _w01 = vld1q_s16(k0);
                    int16x8_t _w23 = vld1q_s16(k0 + 8);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w01), _val0123, 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w01), _val0123, 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w23), _val0123, 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w23), _val0123, 3);

                    r0 += 4;
                    k0 += 16;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);
                _sum0 = vaddq_s32(_sum0, _sum2);
                for (; j < inch; j++)
                {
                    int16x4_t _val0 = vld1_dup_s16(r0);
                    int16x4_t _w0 = vld1_s16(k0);

                    _sum0 = vmlal_s16(_sum0, _val0, _w0);

                    r0 += 1;
                    k0 += 4;
                }

                output0_tm[0] = vgetq_lane_s32(_sum0, 0);
                output1_tm[0] = vgetq_lane_s32(_sum0, 1);
                output2_tm[0] = vgetq_lane_s32(_sum0, 2);
                output3_tm[0] = vgetq_lane_s32(_sum0, 3);
                output0_tm += 1;
                output1_tm += 1;
                output2_tm += 1;
                output3_tm += 1;
            }
        }
    }

    remain_outch_start += nn_outch << 2;
#else // __ARM_NEON
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);

        const Mat kernel0_tm = kernel_tm.channel(p / 2);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum00 = 0;
                int sum10 = 0;
                int sum01 = 0;
                int sum11 = 0;

                int j = 0;
#if __ARM_FEATURE_SIMD32
                for (; j + 1 < inch; j += 2)
                {
                    // fomit-frame-pointer implied in optimized flag spare one register
                    // let us stay away from error: ‘asm’ operand has impossible constraints   --- nihui
#if __OPTIMIZE__
                    asm volatile(
                        "ldr    r2, [%0], #4    \n" // int16x2_t _val02 = *((int16x2_t*)r0); r0 += 2;
                        "ldr    r3, [%0], #4    \n" // int16x2_t _val13 = *((int16x2_t*)r0); r0 += 2;
                        "ldr    r4, [%1], #4    \n" // int16x2_t _w02 = *((int16x2_t*)k0); k0 += 2;
                        "ldr    r5, [%1], #4    \n" // int16x2_t _w13 = *((int16x2_t*)k0); k0 += 2;
                        "smlad  %2, r2, r4, %2  \n" // sum00 = __smlad(_val02, _w02, sum00);
                        "smlad  %3, r3, r4, %3  \n" // sum01 = __smlad(_val13, _w02, sum01);
                        "smlad  %4, r2, r5, %4  \n" // sum10 = __smlad(_val02, _w13, sum10);
                        "smlad  %5, r3, r5, %5  \n" // sum11 = __smlad(_val13, _w13, sum11);
                        : "=r"(r0),
                        "=r"(k0),
                        "=r"(sum00),
                        "=r"(sum01),
                        "=r"(sum10),
                        "=r"(sum11)
                        : "0"(r0),
                        "1"(k0),
                        "2"(sum00),
                        "3"(sum01),
                        "4"(sum10),
                        "5"(sum11)
                        : "memory", "r2", "r3", "r4", "r5");
#else
                    int _val02 = *((int*)r0);
                    int _val13 = *((int*)(r0 + 2));
                    int _w02 = *((int*)k0);
                    int _w13 = *((int*)(k0 + 2));
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum00)
                                 : "0"(sum00), "r"(_val02), "r"(_w02)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum01)
                                 : "0"(sum01), "r"(_val13), "r"(_w02)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum10)
                                 : "0"(sum10), "r"(_val02), "r"(_w13)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum11)
                                 : "0"(sum11), "r"(_val13), "r"(_w13)
                                 :);
                    r0 += 4;
                    k0 += 4;
#endif
                }
#endif // __ARM_FEATURE_SIMD32
                for (; j < inch; j++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];

                    signed short w0 = k0[0];
                    signed short w1 = k0[1];

                    sum00 += val0 * w0;
                    sum10 += val0 * w1;
                    sum01 += val1 * w0;
                    sum11 += val1 * w1;

                    r0 += 2;
                    k0 += 2;
                }

                output0_tm[0] = sum00;
                output1_tm[0] = sum10;
                output0_tm[1] = sum01;
                output1_tm[1] = sum11;
                output0_tm += 2;
                output1_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum0 = 0;
                int sum1 = 0;

                int j = 0;
#if __ARM_FEATURE_SIMD32
                for (; j + 1 < inch; j += 2)
                {
                    asm volatile(
                        "ldr    r2, [%0], #4    \n" // int16x2_t _val01 = *((int16x2_t*)r0); r0 += 2;
                        "ldr    r3, [%1], #4    \n" // int16x2_t _w02 = *((int16x2_t*)k0); k0 += 2;
                        "ldr    r4, [%1], #4    \n" // int16x2_t _w13 = *((int16x2_t*)k0); k0 += 2;
                        "smlad  %2, r2, r3, %2  \n" // sum00 = __smlad(_val01, _w02, sum00);
                        "smlad  %3, r2, r4, %3  \n" // sum01 = __smlad(_val01, _w02, sum01);
                        : "=r"(r0),
                        "=r"(k0),
                        "=r"(sum0),
                        "=r"(sum1)
                        : "0"(r0),
                        "1"(k0),
                        "2"(sum0),
                        "3"(sum1)
                        : "memory", "r2", "r3", "r4");
                }
#endif // __ARM_FEATURE_SIMD32
                for (; j < inch; j++)
                {
                    signed short val = r0[0];

                    sum0 += val * k0[0];
                    sum1 += val * k0[1];

                    r0 += 1;
                    k0 += 2;
                }

                output0_tm[0] = sum0;
                output1_tm[0] = sum1;
                output0_tm += 1;
                output1_tm += 1;
            }
        }
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* output0_tm = top_blob_tm.channel(p);

#if __ARM_NEON
        const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
        const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#endif

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
#if __ARM_NEON
#if __aarch64__
            for (; i + 7 < tiles; i += 8)
            {
                const short* r0 = bb2.row<const short>(i / 8);
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 3 < inch; j += 4)
                {
                    int16x8_t _val01 = vld1q_s16(r0);
                    int16x8_t _val23 = vld1q_s16(r0 + 8);
                    int16x8_t _val45 = vld1q_s16(r0 + 16);
                    int16x8_t _val67 = vld1q_s16(r0 + 24);
                    int16x4_t _w0123 = vld1_s16(k0);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val01), _w0123, 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val01), _w0123, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val23), _w0123, 1);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val23), _w0123, 1);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val45), _w0123, 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val45), _w0123, 2);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val67), _w0123, 3);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val67), _w0123, 3);

                    k0 += 4;
                    r0 += 32;
                }
                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);
                for (; j < inch; j++)
                {
                    int16x8_t _val0 = vld1q_s16(r0);
                    int16x4_t _w0 = vld1_dup_s16(k0);

                    _sum0 = vmlal_s16(_sum0, _w0, vget_low_s16(_val0));
                    _sum1 = vmlal_s16(_sum1, _w0, vget_high_s16(_val0));

                    k0 += 1;
                    r0 += 8;
                }

                vst1q_s32(output0_tm, _sum0);
                vst1q_s32(output0_tm + 4, _sum1);
                output0_tm += 8;
            }
#endif // __aarch64__
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4);
#else
                const short* r0 = bb2.row<const short>(i / 4);
#endif
                const short* k0 = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 3 < inch; j += 4)
                {
                    int16x8_t _val01 = vld1q_s16(r0);
                    int16x8_t _val23 = vld1q_s16(r0 + 8);
                    int16x4_t _w0123 = vld1_s16(k0);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val01), _w0123, 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val01), _w0123, 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val23), _w0123, 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val23), _w0123, 3);

                    k0 += 4;
                    r0 += 16;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);
                _sum0 = vaddq_s32(_sum0, _sum2);
                for (; j < inch; j++)
                {
                    int16x4_t _val0 = vld1_s16(r0);
                    int16x4_t _w0 = vld1_dup_s16(k0);

                    _sum0 = vmlal_s16(_sum0, _val0, _w0);

                    k0 += 1;
                    r0 += 4;
                }

                vst1q_s32(output0_tm, _sum0);
                output0_tm += 4;
            }
#else
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum0 = 0;
                int sum1 = 0;

                int j = 0;
#if __ARM_FEATURE_SIMD32
                for (; j + 1 < inch; j += 2)
                {
                    asm volatile(
                        "ldr    r2, [%0], #4    \n" // int16x2_t _val02 = *((int16x2_t*)r0); r0 += 2;
                        "ldr    r3, [%0], #4    \n" // int16x2_t _val13 = *((int16x2_t*)r0); r0 += 2;
                        "ldr    r4, [%1], #4    \n" // int16x2_t _w01 = *((int16x2_t*)k0); k0 += 2;
                        "smlad  %2, r2, r4, %2  \n" // sum00 = __smlad(_val02, _w01, sum00);
                        "smlad  %3, r3, r4, %3  \n" // sum01 = __smlad(_val13, _w01, sum01);
                        : "=r"(r0),
                        "=r"(k0),
                        "=r"(sum0),
                        "=r"(sum1)
                        : "0"(r0),
                        "1"(k0),
                        "2"(sum0),
                        "3"(sum1)
                        : "memory", "r2", "r3", "r4");
                }
#endif // __ARM_FEATURE_SIMD32
                for (; j < inch; j++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];
                    signed short w = k0[0];

                    sum0 += val0 * w;
                    sum1 += val1 * w;

                    k0 += 1;
                    r0 += 2;
                }

                output0_tm[0] = sum0;
                output0_tm[1] = sum1;
                output0_tm += 2;
            }
#endif
            for (; i < tiles; i++)
            {
#if __ARM_NEON
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4 + i % 4);
#else
                const short* r0 = bb2.row<const short>(i / 4 + i % 4);
#endif
#else
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
#endif
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum = 0;

                int j = 0;
#if __ARM_NEON
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                for (; j + 7 < inch; j += 8)
                {
                    int16x8_t _val = vld1q_s16(r0);
                    int16x8_t _w = vld1q_s16(k0);

                    _sum0 = vmlal_s16(_sum0, vget_low_s16(_val), vget_low_s16(_w));
                    _sum1 = vmlal_s16(_sum1, vget_high_s16(_val), vget_high_s16(_w));

                    k0 += 8;
                    r0 += 8;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                for (; j + 3 < inch; j += 4)
                {
                    int16x4_t _val = vld1_s16(r0);
                    int16x4_t _w = vld1_s16(k0);

                    _sum0 = vmlal_s16(_sum0, _val, _w);

                    k0 += 4;
                    r0 += 4;
                }
#if __aarch64__
                sum = vaddvq_s32(_sum0);
#else
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _ss = vpadd_s32(_ss, _ss);

                sum = vget_lane_s32(_ss, 0);
#endif
#endif // __ARM_NEON
#if __ARM_FEATURE_SIMD32
                for (; j + 1 < inch; j += 2)
                {
                    asm volatile(
                        "ldr    r2, [%0], #4    \n" // int16x2_t _val = *((int16x2_t*)r0); r0 += 2;
                        "ldr    r3, [%1], #4    \n" // int16x2_t _w = *((int16x2_t*)k0); k0 += 2;
                        "smlad  %2, r2, r3, %2  \n" // sum = __smlad(_val, _w, sum);
                        : "=r"(r0),
                        "=r"(k0),
                        "=r"(sum)
                        : "0"(r0),
                        "1"(k0),
                        "2"(sum)
                        : "memory", "r2", "r3");
                }
#endif // __ARM_FEATURE_SIMD32
                for (; j < inch; j++)
                {
                    signed short val = r0[0];
                    signed short w = k0[0];

                    sum += val * w;

                    k0 += 1;
                    r0 += 1;
                }

                output0_tm[0] = sum;
                output0_tm++;
            }
        }
    }
}
