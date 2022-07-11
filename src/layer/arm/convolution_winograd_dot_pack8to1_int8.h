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

static void convolution_winograd_dot_pack8to1_int8_neon(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 16u, 8, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __aarch64__
    if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, batch, 16u, 8, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 16u, 8, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 8, opt.workspace_allocator);
#else
    if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 16u, 8, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 8, opt.workspace_allocator);
#endif // __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
#if __aarch64__
        for (; i + 7 < tiles; i += 8)
        {
            short* tm2p = tm2.row<short>(i / 8);

            const short* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
                // transpose 8x8
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0] \n"
                    "sub    %0, %0, #64             \n"

                    "uzp1   v16.8h, v0.8h, v4.8h    \n"
                    "uzp2   v20.8h, v0.8h, v4.8h    \n"
                    "uzp1   v17.8h, v1.8h, v5.8h    \n"
                    "uzp2   v21.8h, v1.8h, v5.8h    \n"
                    "uzp1   v18.8h, v2.8h, v6.8h    \n"
                    "uzp2   v22.8h, v2.8h, v6.8h    \n"
                    "uzp1   v19.8h, v3.8h, v7.8h    \n"
                    "uzp2   v23.8h, v3.8h, v7.8h    \n"

                    "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"
                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");

                r0 += bottom_blob_tm.cstep * 8;
            }
        }
#endif
        for (; i + 3 < tiles; i += 4)
        {
#if __aarch64__
            short* tm2p = tm2.row<short>(i / 8 + (i % 8) / 4);
#else
            short* tm2p = tm2.row<short>(i / 4);
#endif

            const short* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
                // transpose 8x4
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3");
#else
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vswp       d1, d2              \n"
                    "vswp       d5, d6              \n"
                    "vswp       q1, q2              \n"
                    "vst4.s16   {d0-d3}, [%1 :64]!  \n"
                    "vst4.s16   {d4-d7}, [%1 :64]!  \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
                r0 += bottom_blob_tm.cstep * 8;
            }
        }
        for (; i < tiles; i++)
        {
#if __aarch64__
            short* tm2p = tm2.row<short>(i / 8 + (i % 8) / 4 + i % 4);
#else
            short* tm2p = tm2.row<short>(i / 4 + i % 4);
#endif

            const short* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]   \n"
                    "ld1    {v0.8h}, [%0]           \n"
                    "st1    {v0.8h}, [%1], #16      \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0");
#else
                asm volatile(
                    "pld        [%0, #128]          \n"
                    "vld1.s16   {d0-d1}, [%0 :64]   \n"
                    "vst1.s16   {d0-d1}, [%1 :64]!  \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "q0");
#endif // __aarch64__
                r0 += bottom_blob_tm.cstep * 8;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 4u, 1, opt.workspace_allocator);

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 3;

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

        const Mat kernel01_tm = kernel_tm.channel(p / 8);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
#if __aarch64__
            for (; i + 7 < tiles; i += 8)
            {
                const short* r0 = bb2.row<const short>(i / 8);
                const short* kptr = kernel01_tm.row<const short>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"
                    "eor    v21.16b, v21.16b, v21.16b   \n"
                    "eor    v22.16b, v22.16b, v22.16b   \n"
                    "eor    v23.16b, v23.16b, v23.16b   \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v25.16b, v25.16b, v25.16b   \n"
                    "eor    v26.16b, v26.16b, v26.16b   \n"
                    "eor    v27.16b, v27.16b, v27.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%9, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%9], #64 \n"

                    "prfm   pldl1keep, [%10, #512]      \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%10], #64 \n"

                    "smlal  v16.4s, v8.4h, v0.h[0]      \n"
                    "smlal2 v17.4s, v8.8h, v0.h[0]      \n"
                    "smlal  v18.4s, v8.4h, v0.h[1]      \n"
                    "smlal2 v19.4s, v8.8h, v0.h[1]      \n"
                    "smlal  v20.4s, v8.4h, v0.h[2]      \n"
                    "smlal2 v21.4s, v8.8h, v0.h[2]      \n"
                    "smlal  v22.4s, v8.4h, v0.h[3]      \n"
                    "smlal2 v23.4s, v8.8h, v0.h[3]      \n"
                    "smlal  v24.4s, v8.4h, v0.h[4]      \n"
                    "smlal2 v25.4s, v8.8h, v0.h[4]      \n"
                    "smlal  v26.4s, v8.4h, v0.h[5]      \n"
                    "smlal2 v27.4s, v8.8h, v0.h[5]      \n"
                    "smlal  v28.4s, v8.4h, v0.h[6]      \n"
                    "smlal2 v29.4s, v8.8h, v0.h[6]      \n"
                    "smlal  v30.4s, v8.4h, v0.h[7]      \n"
                    "smlal2 v31.4s, v8.8h, v0.h[7]      \n"

                    "smlal  v16.4s, v9.4h, v1.h[0]      \n"
                    "smlal2 v17.4s, v9.8h, v1.h[0]      \n"
                    "smlal  v18.4s, v9.4h, v1.h[1]      \n"
                    "smlal2 v19.4s, v9.8h, v1.h[1]      \n"
                    "smlal  v20.4s, v9.4h, v1.h[2]      \n"
                    "smlal2 v21.4s, v9.8h, v1.h[2]      \n"
                    "smlal  v22.4s, v9.4h, v1.h[3]      \n"
                    "smlal2 v23.4s, v9.8h, v1.h[3]      \n"
                    "smlal  v24.4s, v9.4h, v1.h[4]      \n"
                    "smlal2 v25.4s, v9.8h, v1.h[4]      \n"
                    "smlal  v26.4s, v9.4h, v1.h[5]      \n"
                    "smlal2 v27.4s, v9.8h, v1.h[5]      \n"
                    "smlal  v28.4s, v9.4h, v1.h[6]      \n"
                    "smlal2 v29.4s, v9.8h, v1.h[6]      \n"
                    "smlal  v30.4s, v9.4h, v1.h[7]      \n"
                    "smlal2 v31.4s, v9.8h, v1.h[7]      \n"

                    "prfm   pldl1keep, [%9, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%9], #64 \n"

                    "smlal  v16.4s, v10.4h, v2.h[0]     \n"
                    "smlal2 v17.4s, v10.8h, v2.h[0]     \n"
                    "smlal  v18.4s, v10.4h, v2.h[1]     \n"
                    "smlal2 v19.4s, v10.8h, v2.h[1]     \n"
                    "smlal  v20.4s, v10.4h, v2.h[2]     \n"
                    "smlal2 v21.4s, v10.8h, v2.h[2]     \n"
                    "smlal  v22.4s, v10.4h, v2.h[3]     \n"
                    "smlal2 v23.4s, v10.8h, v2.h[3]     \n"
                    "smlal  v24.4s, v10.4h, v2.h[4]     \n"
                    "smlal2 v25.4s, v10.8h, v2.h[4]     \n"
                    "smlal  v26.4s, v10.4h, v2.h[5]     \n"
                    "smlal2 v27.4s, v10.8h, v2.h[5]     \n"
                    "smlal  v28.4s, v10.4h, v2.h[6]     \n"
                    "smlal2 v29.4s, v10.8h, v2.h[6]     \n"
                    "smlal  v30.4s, v10.4h, v2.h[7]     \n"
                    "smlal2 v31.4s, v10.8h, v2.h[7]     \n"

                    "prfm   pldl1keep, [%10, #512]      \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%10], #64 \n"

                    "smlal  v16.4s, v11.4h, v3.h[0]     \n"
                    "smlal2 v17.4s, v11.8h, v3.h[0]     \n"
                    "smlal  v18.4s, v11.4h, v3.h[1]     \n"
                    "smlal2 v19.4s, v11.8h, v3.h[1]     \n"
                    "smlal  v20.4s, v11.4h, v3.h[2]     \n"
                    "smlal2 v21.4s, v11.8h, v3.h[2]     \n"
                    "smlal  v22.4s, v11.4h, v3.h[3]     \n"
                    "smlal2 v23.4s, v11.8h, v3.h[3]     \n"
                    "smlal  v24.4s, v11.4h, v3.h[4]     \n"
                    "smlal2 v25.4s, v11.8h, v3.h[4]     \n"
                    "smlal  v26.4s, v11.4h, v3.h[5]     \n"
                    "smlal2 v27.4s, v11.8h, v3.h[5]     \n"
                    "smlal  v28.4s, v11.4h, v3.h[6]     \n"
                    "smlal2 v29.4s, v11.8h, v3.h[6]     \n"
                    "smlal  v30.4s, v11.4h, v3.h[7]     \n"
                    "smlal2 v31.4s, v11.8h, v3.h[7]     \n"

                    "smlal  v16.4s, v12.4h, v4.h[0]     \n"
                    "smlal2 v17.4s, v12.8h, v4.h[0]     \n"
                    "smlal  v18.4s, v12.4h, v4.h[1]     \n"
                    "smlal2 v19.4s, v12.8h, v4.h[1]     \n"
                    "smlal  v20.4s, v12.4h, v4.h[2]     \n"
                    "smlal2 v21.4s, v12.8h, v4.h[2]     \n"
                    "smlal  v22.4s, v12.4h, v4.h[3]     \n"
                    "smlal2 v23.4s, v12.8h, v4.h[3]     \n"
                    "smlal  v24.4s, v12.4h, v4.h[4]     \n"
                    "smlal2 v25.4s, v12.8h, v4.h[4]     \n"
                    "smlal  v26.4s, v12.4h, v4.h[5]     \n"
                    "smlal2 v27.4s, v12.8h, v4.h[5]     \n"
                    "smlal  v28.4s, v12.4h, v4.h[6]     \n"
                    "smlal2 v29.4s, v12.8h, v4.h[6]     \n"
                    "smlal  v30.4s, v12.4h, v4.h[7]     \n"
                    "smlal2 v31.4s, v12.8h, v4.h[7]     \n"

                    "smlal  v16.4s, v13.4h, v5.h[0]     \n"
                    "smlal2 v17.4s, v13.8h, v5.h[0]     \n"
                    "smlal  v18.4s, v13.4h, v5.h[1]     \n"
                    "smlal2 v19.4s, v13.8h, v5.h[1]     \n"
                    "smlal  v20.4s, v13.4h, v5.h[2]     \n"
                    "smlal2 v21.4s, v13.8h, v5.h[2]     \n"
                    "smlal  v22.4s, v13.4h, v5.h[3]     \n"
                    "smlal2 v23.4s, v13.8h, v5.h[3]     \n"
                    "smlal  v24.4s, v13.4h, v5.h[4]     \n"
                    "smlal2 v25.4s, v13.8h, v5.h[4]     \n"
                    "smlal  v26.4s, v13.4h, v5.h[5]     \n"
                    "smlal2 v27.4s, v13.8h, v5.h[5]     \n"
                    "smlal  v28.4s, v13.4h, v5.h[6]     \n"
                    "smlal2 v29.4s, v13.8h, v5.h[6]     \n"
                    "smlal  v30.4s, v13.4h, v5.h[7]     \n"
                    "smlal2 v31.4s, v13.8h, v5.h[7]     \n"

                    "smlal  v16.4s, v14.4h, v6.h[0]     \n"
                    "smlal2 v17.4s, v14.8h, v6.h[0]     \n"
                    "smlal  v18.4s, v14.4h, v6.h[1]     \n"
                    "smlal2 v19.4s, v14.8h, v6.h[1]     \n"
                    "smlal  v20.4s, v14.4h, v6.h[2]     \n"
                    "smlal2 v21.4s, v14.8h, v6.h[2]     \n"
                    "smlal  v22.4s, v14.4h, v6.h[3]     \n"
                    "smlal2 v23.4s, v14.8h, v6.h[3]     \n"
                    "smlal  v24.4s, v14.4h, v6.h[4]     \n"
                    "smlal2 v25.4s, v14.8h, v6.h[4]     \n"
                    "smlal  v26.4s, v14.4h, v6.h[5]     \n"
                    "smlal2 v27.4s, v14.8h, v6.h[5]     \n"
                    "smlal  v28.4s, v14.4h, v6.h[6]     \n"
                    "smlal2 v29.4s, v14.8h, v6.h[6]     \n"
                    "smlal  v30.4s, v14.4h, v6.h[7]     \n"
                    "smlal2 v31.4s, v14.8h, v6.h[7]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "smlal  v16.4s, v15.4h, v7.h[0]     \n"
                    "smlal2 v17.4s, v15.8h, v7.h[0]     \n"
                    "smlal  v18.4s, v15.4h, v7.h[1]     \n"
                    "smlal2 v19.4s, v15.8h, v7.h[1]     \n"
                    "smlal  v20.4s, v15.4h, v7.h[2]     \n"
                    "smlal2 v21.4s, v15.8h, v7.h[2]     \n"
                    "smlal  v22.4s, v15.4h, v7.h[3]     \n"
                    "smlal2 v23.4s, v15.8h, v7.h[3]     \n"
                    "smlal  v24.4s, v15.4h, v7.h[4]     \n"
                    "smlal2 v25.4s, v15.8h, v7.h[4]     \n"
                    "smlal  v26.4s, v15.4h, v7.h[5]     \n"
                    "smlal2 v27.4s, v15.8h, v7.h[5]     \n"
                    "smlal  v28.4s, v15.4h, v7.h[6]     \n"
                    "smlal2 v29.4s, v15.8h, v7.h[6]     \n"
                    "smlal  v30.4s, v15.4h, v7.h[7]     \n"
                    "smlal2 v31.4s, v15.8h, v7.h[7]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s}, [%1], #32 \n"
                    "st1    {v18.4s, v19.4s}, [%2], #32 \n"
                    "st1    {v20.4s, v21.4s}, [%3], #32 \n"
                    "st1    {v22.4s, v23.4s}, [%4], #32 \n"
                    "st1    {v24.4s, v25.4s}, [%5], #32 \n"
                    "st1    {v26.4s, v27.4s}, [%6], #32 \n"
                    "st1    {v28.4s, v29.4s}, [%7], #32 \n"
                    "st1    {v30.4s, v31.4s}, [%8], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(output2_tm), // %3
                    "=r"(output3_tm), // %4
                    "=r"(output4_tm), // %5
                    "=r"(output5_tm), // %6
                    "=r"(output6_tm), // %7
                    "=r"(output7_tm), // %8
                    "=r"(r0),         // %9
                    "=r"(kptr)        // %10
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(output2_tm),
                    "4"(output3_tm),
                    "5"(output4_tm),
                    "6"(output5_tm),
                    "7"(output6_tm),
                    "8"(output7_tm),
                    "9"(r0),
                    "10"(kptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#endif
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4);
#else
                const short* r0 = bb2.row<const short>(i / 4);
#endif
                const short* k0 = kernel01_tm.row<const short>(r);

                int nn = inch; // inch always > 0

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int16x8_t _val0 = vld1q_s16(r0);
                    int16x8_t _val1 = vld1q_s16(r0 + 8);
                    int16x8_t _val2 = vld1q_s16(r0 + 16);
                    int16x8_t _val3 = vld1q_s16(r0 + 24);

                    int16x8_t _w0 = vld1q_s16(k0);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val0), vget_low_s16(_w0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_val0), vget_low_s16(_w0), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val0), vget_low_s16(_w0), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_val0), vget_low_s16(_w0), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_val0), vget_high_s16(_w0), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_val0), vget_high_s16(_w0), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_val0), vget_high_s16(_w0), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_val0), vget_high_s16(_w0), 3);

                    int16x8_t _w1 = vld1q_s16(k0 + 8);

                    _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_val0), vget_low_s16(_w1), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val0), vget_low_s16(_w1), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_val0), vget_low_s16(_w1), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val0), vget_low_s16(_w1), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_val0), vget_high_s16(_w1), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_val0), vget_high_s16(_w1), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_val0), vget_high_s16(_w1), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_val0), vget_high_s16(_w1), 3);

                    int16x8_t _w2 = vld1q_s16(k0 + 16);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val1), vget_low_s16(_w2), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_val1), vget_low_s16(_w2), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val1), vget_low_s16(_w2), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_val1), vget_low_s16(_w2), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_val1), vget_high_s16(_w2), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_val1), vget_high_s16(_w2), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_val1), vget_high_s16(_w2), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_val1), vget_high_s16(_w2), 3);

                    int16x8_t _w3 = vld1q_s16(k0 + 24);

                    _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_val1), vget_low_s16(_w3), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val1), vget_low_s16(_w3), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_val1), vget_low_s16(_w3), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val1), vget_low_s16(_w3), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_val1), vget_high_s16(_w3), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_val1), vget_high_s16(_w3), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_val1), vget_high_s16(_w3), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_val1), vget_high_s16(_w3), 3);

                    int16x8_t _w4 = vld1q_s16(k0 + 32);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val2), vget_low_s16(_w4), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_val2), vget_low_s16(_w4), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val2), vget_low_s16(_w4), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_val2), vget_low_s16(_w4), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_val2), vget_high_s16(_w4), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_val2), vget_high_s16(_w4), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_val2), vget_high_s16(_w4), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_val2), vget_high_s16(_w4), 3);

                    int16x8_t _w5 = vld1q_s16(k0 + 40);

                    _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_val2), vget_low_s16(_w5), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val2), vget_low_s16(_w5), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_val2), vget_low_s16(_w5), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val2), vget_low_s16(_w5), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_val2), vget_high_s16(_w5), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_val2), vget_high_s16(_w5), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_val2), vget_high_s16(_w5), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_val2), vget_high_s16(_w5), 3);

                    int16x8_t _w6 = vld1q_s16(k0 + 48);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_val3), vget_low_s16(_w6), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_val3), vget_low_s16(_w6), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_val3), vget_low_s16(_w6), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_val3), vget_low_s16(_w6), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_val3), vget_high_s16(_w6), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_val3), vget_high_s16(_w6), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_val3), vget_high_s16(_w6), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_val3), vget_high_s16(_w6), 3);

                    int16x8_t _w7 = vld1q_s16(k0 + 56);

                    _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_val3), vget_low_s16(_w7), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_val3), vget_low_s16(_w7), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_val3), vget_low_s16(_w7), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_val3), vget_low_s16(_w7), 3);
                    _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_val3), vget_high_s16(_w7), 0);
                    _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_val3), vget_high_s16(_w7), 1);
                    _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_val3), vget_high_s16(_w7), 2);
                    _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_val3), vget_high_s16(_w7), 3);

                    r0 += 32;
                    k0 += 64;
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
                const short* k0 = kernel01_tm.row<const short>(r);

                int nn = inch; // inch always > 0

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int16x8_t _val0 = vld1q_s16(r0);

                    int16x8_t _w0 = vld1q_s16(k0);
                    int16x8_t _w1 = vld1q_s16(k0 + 8);
                    int16x8_t _w2 = vld1q_s16(k0 + 16);
                    int16x8_t _w3 = vld1q_s16(k0 + 24);
                    int16x8_t _w4 = vld1q_s16(k0 + 32);
                    int16x8_t _w5 = vld1q_s16(k0 + 40);
                    int16x8_t _w6 = vld1q_s16(k0 + 48);
                    int16x8_t _w7 = vld1q_s16(k0 + 56);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val0), 0);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val0), 1);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val0), 1);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_low_s16(_val0), 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_low_s16(_val0), 2);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_low_s16(_val0), 3);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_low_s16(_val0), 3);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w4), vget_high_s16(_val0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w4), vget_high_s16(_val0), 0);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w5), vget_high_s16(_val0), 1);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w5), vget_high_s16(_val0), 1);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w6), vget_high_s16(_val0), 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w6), vget_high_s16(_val0), 2);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w7), vget_high_s16(_val0), 3);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w7), vget_high_s16(_val0), 3);

                    r0 += 8;
                    k0 += 64;
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

    remain_outch_start += nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p / 8 + p % 8);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
#if __aarch64__
            for (; i + 7 < tiles; i += 8)
            {
                const short* r0 = bb2.row<const short>(i / 8);

                const short* kptr = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                for (int q = 0; q < inch; q++)
                {
                    int16x8_t _r0 = vld1q_s16(r0);
                    int16x8_t _r1 = vld1q_s16(r0 + 8);
                    int16x8_t _r2 = vld1q_s16(r0 + 16);
                    int16x8_t _r3 = vld1q_s16(r0 + 24);
                    int16x8_t _r4 = vld1q_s16(r0 + 32);
                    int16x8_t _r5 = vld1q_s16(r0 + 40);
                    int16x8_t _r6 = vld1q_s16(r0 + 48);
                    int16x8_t _r7 = vld1q_s16(r0 + 56);

                    int16x8_t _k0 = vld1q_s16(kptr);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r0), vget_low_s16(_k0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r0), vget_low_s16(_k0), 0);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r1), vget_low_s16(_k0), 1);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_r1), vget_low_s16(_k0), 1);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2), vget_low_s16(_k0), 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r2), vget_low_s16(_k0), 2);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r3), vget_low_s16(_k0), 3);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_r3), vget_low_s16(_k0), 3);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r4), vget_high_s16(_k0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r4), vget_high_s16(_k0), 0);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r5), vget_high_s16(_k0), 1);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_r5), vget_high_s16(_k0), 1);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r6), vget_high_s16(_k0), 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r6), vget_high_s16(_k0), 2);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r7), vget_high_s16(_k0), 3);
                    _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_r7), vget_high_s16(_k0), 3);

                    kptr += 8;
                    r0 += 64;
                }

                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);

                vst1q_s32(output0_tm, _sum0);
                vst1q_s32(output0_tm + 4, _sum1);

                output0_tm += 8;
            }
#endif
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4);
#else
                const short* r0 = bb2.row<const short>(i / 4);
#endif
                const short* kptr = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                for (int q = 0; q < inch; q++)
                {
                    int16x8_t _r0 = vld1q_s16(r0);
                    int16x8_t _r1 = vld1q_s16(r0 + 8);
                    int16x8_t _r2 = vld1q_s16(r0 + 16);
                    int16x8_t _r3 = vld1q_s16(r0 + 24);

                    int16x8_t _k0 = vld1q_s16(kptr);

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r0), vget_low_s16(_k0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r0), vget_low_s16(_k0), 1);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1), vget_low_s16(_k0), 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r1), vget_low_s16(_k0), 3);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2), vget_high_s16(_k0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r2), vget_high_s16(_k0), 1);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r3), vget_high_s16(_k0), 2);
                    _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_r3), vget_high_s16(_k0), 3);

                    kptr += 8;
                    r0 += 32;
                }

                int32x4_t _sum01 = vaddq_s32(_sum0, _sum1);

                vst1q_s32(output0_tm, _sum01);

                output0_tm += 4;
            }
            for (; i < tiles; i++)
            {
#if __aarch64__
                const short* r0 = bb2.row<const short>(i / 8 + (i % 8) / 4 + i % 4);
#else
                const short* r0 = bb2.row<const short>(i / 4 + i % 4);
#endif
                const short* kptr = kernel0_tm.row<const short>(r);

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                for (int q = 0; q < inch; q++)
                {
                    int16x8_t _r0 = vld1q_s16(r0);

                    int16x8_t _k0 = vld1q_s16(kptr);

                    _sum0 = vmlal_s16(_sum0, vget_low_s16(_r0), vget_low_s16(_k0));
                    _sum1 = vmlal_s16(_sum1, vget_high_s16(_r0), vget_high_s16(_k0));

                    kptr += 8;
                    r0 += 8;
                }

                int32x4_t _sum = vaddq_s32(_sum0, _sum1);
#if __aarch64__
                int sum = vaddvq_s32(_sum); // dot
#else
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                _ss = vpadd_s32(_ss, _ss);
                int sum = vget_lane_s32(_ss, 0);
#endif

                output0_tm[0] = sum;

                output0_tm++;
            }
        }
    }
}
