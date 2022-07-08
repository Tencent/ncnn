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

static void convolution_winograd_dot_pack4_fp16sa_neon(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 8u, 4, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, batch, 8u, 4, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 8u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 8u, 4, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 7 < tiles; i += 8)
        {
            __fp16* tm2p = tm2.row<__fp16>(i / 8);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x8
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3");

                r0 += bottom_blob_tm.cstep * 4;
            }
        }
        for (; i + 3 < tiles; i += 4)
        {
            __fp16* tm2p = tm2.row<__fp16>(i / 8 + (i % 8) / 4);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x4
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]   \n"
                    "ld4    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0] \n"
                    "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3");

                r0 += bottom_blob_tm.cstep * 4;
            }
        }
        for (; i < tiles; i++)
        {
            __fp16* tm2p = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + i % 4);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #64]    \n"
                    "ld1    {v0.4h}, [%0]           \n"
                    "st1    {v0.4h}, [%1], #8       \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0");

                r0 += bottom_blob_tm.cstep * 4;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 8u, 4, opt.workspace_allocator);

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        __fp16* output0_tm = top_blob_tm.channel(p);
        __fp16* output1_tm = top_blob_tm.channel(p + 1);

        const Mat kernel01_tm = kernel_tm.channel(pp);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8);

                const __fp16* kptr = kernel01_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v25.16b, v25.16b, v25.16b   \n"
                    "eor    v26.16b, v26.16b, v26.16b   \n"
                    "eor    v27.16b, v27.16b, v27.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r01 r23 r45 r67

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%4], #64 \n" // k0123

                    "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v0.h[3]      \n"
                    "fmla   v28.8h, v4.8h, v0.h[4]      \n"
                    "fmla   v29.8h, v4.8h, v0.h[5]      \n"
                    "fmla   v30.8h, v4.8h, v0.h[6]      \n"
                    "fmla   v31.8h, v4.8h, v0.h[7]      \n"

                    "fmla   v24.8h, v5.8h, v1.h[0]      \n"
                    "fmla   v25.8h, v5.8h, v1.h[1]      \n"
                    "fmla   v26.8h, v5.8h, v1.h[2]      \n"
                    "fmla   v27.8h, v5.8h, v1.h[3]      \n"
                    "fmla   v28.8h, v5.8h, v1.h[4]      \n"
                    "fmla   v29.8h, v5.8h, v1.h[5]      \n"
                    "fmla   v30.8h, v5.8h, v1.h[6]      \n"
                    "fmla   v31.8h, v5.8h, v1.h[7]      \n"

                    "fmla   v24.8h, v6.8h, v2.h[0]      \n"
                    "fmla   v25.8h, v6.8h, v2.h[1]      \n"
                    "fmla   v26.8h, v6.8h, v2.h[2]      \n"
                    "fmla   v27.8h, v6.8h, v2.h[3]      \n"
                    "fmla   v28.8h, v6.8h, v2.h[4]      \n"
                    "fmla   v29.8h, v6.8h, v2.h[5]      \n"
                    "fmla   v30.8h, v6.8h, v2.h[6]      \n"
                    "fmla   v31.8h, v6.8h, v2.h[7]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v24.8h, v7.8h, v3.h[0]      \n"
                    "fmla   v25.8h, v7.8h, v3.h[1]      \n"
                    "fmla   v26.8h, v7.8h, v3.h[2]      \n"
                    "fmla   v27.8h, v7.8h, v3.h[3]      \n"
                    "fmla   v28.8h, v7.8h, v3.h[4]      \n"
                    "fmla   v29.8h, v7.8h, v3.h[5]      \n"
                    "fmla   v30.8h, v7.8h, v3.h[6]      \n"
                    "fmla   v31.8h, v7.8h, v3.h[7]      \n"

                    "bne    0b                          \n"

                    "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"
                    "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%1], #32 \n"

                    "ext    v24.16b, v24.16b, v24.16b, #8 \n"
                    "ext    v25.16b, v25.16b, v25.16b, #8 \n"
                    "ext    v26.16b, v26.16b, v26.16b, #8 \n"
                    "ext    v27.16b, v27.16b, v27.16b, #8 \n"
                    "ext    v28.16b, v28.16b, v28.16b, #8 \n"
                    "ext    v29.16b, v29.16b, v29.16b, #8 \n"
                    "ext    v30.16b, v30.16b, v30.16b, #8 \n"
                    "ext    v31.16b, v31.16b, v31.16b, #8 \n"

                    "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"
                    "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%2], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(kptr)        // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(kptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 3 < tiles; i += 4)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4);

                const __fp16* kptr = kernel01_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v25.16b, v25.16b, v25.16b   \n"
                    "eor    v26.16b, v26.16b, v26.16b   \n"
                    "eor    v27.16b, v27.16b, v27.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r01 r23 r45 r67

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%4], #64 \n" // k0123

                    "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                    "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                    "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                    "fmla   v27.8h, v4.8h, v0.h[3]      \n"

                    "fmla   v24.8h, v5.8h, v1.h[0]      \n"
                    "fmla   v25.8h, v5.8h, v1.h[1]      \n"
                    "fmla   v26.8h, v5.8h, v1.h[2]      \n"
                    "fmla   v27.8h, v5.8h, v1.h[3]      \n"

                    "fmla   v24.8h, v6.8h, v2.h[0]      \n"
                    "fmla   v25.8h, v6.8h, v2.h[1]      \n"
                    "fmla   v26.8h, v6.8h, v2.h[2]      \n"
                    "fmla   v27.8h, v6.8h, v2.h[3]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v24.8h, v7.8h, v3.h[0]      \n"
                    "fmla   v25.8h, v7.8h, v3.h[1]      \n"
                    "fmla   v26.8h, v7.8h, v3.h[2]      \n"
                    "fmla   v27.8h, v7.8h, v3.h[3]      \n"

                    "bne    0b                          \n"

                    "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"

                    "ext    v24.16b, v24.16b, v24.16b, #8 \n"
                    "ext    v25.16b, v25.16b, v25.16b, #8 \n"
                    "ext    v26.16b, v26.16b, v26.16b, #8 \n"
                    "ext    v27.16b, v27.16b, v27.16b, #8 \n"

                    "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(kptr)        // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(kptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27");
            }
            for (; i < tiles; i++)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + i % 4);

                const __fp16* kptr = kernel01_tm.row<const __fp16>(r);

                float16x8_t _sum0 = vdupq_n_f16(0.f);

                for (int q = 0; q < inch; q++)
                {
                    float16x4_t _r0 = vld1_f16(r0);

                    float16x8_t _k0 = vld1q_f16(kptr);
                    float16x8_t _k1 = vld1q_f16(kptr + 8);
                    float16x8_t _k2 = vld1q_f16(kptr + 16);
                    float16x8_t _k3 = vld1q_f16(kptr + 24);

                    _sum0 = vfmaq_lane_f16(_sum0, _k0, _r0, 0);
                    _sum0 = vfmaq_lane_f16(_sum0, _k1, _r0, 1);
                    _sum0 = vfmaq_lane_f16(_sum0, _k2, _r0, 2);
                    _sum0 = vfmaq_lane_f16(_sum0, _k3, _r0, 3);

                    kptr += 32;
                    r0 += 4;
                }

                vst1_f16(output0_tm, vget_low_f16(_sum0));
                vst1_f16(output1_tm, vget_high_f16(_sum0));

                output0_tm += 4;
                output1_tm += 4;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        __fp16* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8);

                const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v25.16b, v25.16b, v25.16b   \n"
                    "eor    v26.16b, v26.16b, v26.16b   \n"
                    "eor    v27.16b, v27.16b, v27.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r01 r23 r45 r67

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3], #32 \n" // k0123

                    "fmla   v24.4h, v4.4h, v0.h[0]      \n"
                    "fmla   v25.4h, v4.4h, v0.h[1]      \n"
                    "fmla   v26.4h, v4.4h, v0.h[2]      \n"
                    "fmla   v27.4h, v4.4h, v0.h[3]      \n"
                    "fmla   v28.4h, v4.4h, v0.h[4]      \n"
                    "fmla   v29.4h, v4.4h, v0.h[5]      \n"
                    "fmla   v30.4h, v4.4h, v0.h[6]      \n"
                    "fmla   v31.4h, v4.4h, v0.h[7]      \n"

                    "fmla   v24.4h, v5.4h, v1.h[0]      \n"
                    "fmla   v25.4h, v5.4h, v1.h[1]      \n"
                    "fmla   v26.4h, v5.4h, v1.h[2]      \n"
                    "fmla   v27.4h, v5.4h, v1.h[3]      \n"
                    "fmla   v28.4h, v5.4h, v1.h[4]      \n"
                    "fmla   v29.4h, v5.4h, v1.h[5]      \n"
                    "fmla   v30.4h, v5.4h, v1.h[6]      \n"
                    "fmla   v31.4h, v5.4h, v1.h[7]      \n"

                    "fmla   v24.4h, v6.4h, v2.h[0]      \n"
                    "fmla   v25.4h, v6.4h, v2.h[1]      \n"
                    "fmla   v26.4h, v6.4h, v2.h[2]      \n"
                    "fmla   v27.4h, v6.4h, v2.h[3]      \n"
                    "fmla   v28.4h, v6.4h, v2.h[4]      \n"
                    "fmla   v29.4h, v6.4h, v2.h[5]      \n"
                    "fmla   v30.4h, v6.4h, v2.h[6]      \n"
                    "fmla   v31.4h, v6.4h, v2.h[7]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v24.4h, v7.4h, v3.h[0]      \n"
                    "fmla   v25.4h, v7.4h, v3.h[1]      \n"
                    "fmla   v26.4h, v7.4h, v3.h[2]      \n"
                    "fmla   v27.4h, v7.4h, v3.h[3]      \n"
                    "fmla   v28.4h, v7.4h, v3.h[4]      \n"
                    "fmla   v29.4h, v7.4h, v3.h[5]      \n"
                    "fmla   v30.4h, v7.4h, v3.h[6]      \n"
                    "fmla   v31.4h, v7.4h, v3.h[7]      \n"

                    "bne    0b                          \n"

                    "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"
                    "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%1], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(kptr)        // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(kptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 3 < tiles; i += 4)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4);

                const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v25.16b, v25.16b, v25.16b   \n"
                    "eor    v26.16b, v26.16b, v26.16b   \n"
                    "eor    v27.16b, v27.16b, v27.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r01 r23 r45 r67

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3], #32 \n" // k0123

                    "fmla   v24.4h, v4.4h, v0.h[0]      \n"
                    "fmla   v25.4h, v4.4h, v0.h[1]      \n"
                    "fmla   v26.4h, v4.4h, v0.h[2]      \n"
                    "fmla   v27.4h, v4.4h, v0.h[3]      \n"

                    "fmla   v24.4h, v5.4h, v1.h[0]      \n"
                    "fmla   v25.4h, v5.4h, v1.h[1]      \n"
                    "fmla   v26.4h, v5.4h, v1.h[2]      \n"
                    "fmla   v27.4h, v5.4h, v1.h[3]      \n"

                    "fmla   v24.4h, v6.4h, v2.h[0]      \n"
                    "fmla   v25.4h, v6.4h, v2.h[1]      \n"
                    "fmla   v26.4h, v6.4h, v2.h[2]      \n"
                    "fmla   v27.4h, v6.4h, v2.h[3]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v24.4h, v7.4h, v3.h[0]      \n"
                    "fmla   v25.4h, v7.4h, v3.h[1]      \n"
                    "fmla   v26.4h, v7.4h, v3.h[2]      \n"
                    "fmla   v27.4h, v7.4h, v3.h[3]      \n"

                    "bne    0b                          \n"

                    "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(kptr)        // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(kptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27");
            }
            for (; i < tiles; i++)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + i % 4);

                const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                float16x4_t _sum0 = vdup_n_f16(0.f);

                for (int q = 0; q < inch; q++)
                {
                    float16x4_t _r0 = vld1_f16(r0);

                    float16x4_t _k0 = vld1_f16(kptr);
                    float16x4_t _k1 = vld1_f16(kptr + 4);
                    float16x4_t _k2 = vld1_f16(kptr + 8);
                    float16x4_t _k3 = vld1_f16(kptr + 12);

                    _sum0 = vfma_lane_f16(_sum0, _k0, _r0, 0);
                    _sum0 = vfma_lane_f16(_sum0, _k1, _r0, 1);
                    _sum0 = vfma_lane_f16(_sum0, _k2, _r0, 2);
                    _sum0 = vfma_lane_f16(_sum0, _k3, _r0, 3);

                    kptr += 16;
                    r0 += 4;
                }

                vst1_f16(output0_tm, _sum0);

                output0_tm += 4;
            }
        }
    }
}
