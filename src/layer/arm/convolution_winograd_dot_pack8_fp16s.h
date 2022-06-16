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

static void convolution_winograd_dot_pack8_fp16sa_neon(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 16u, 8, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 12)
        bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, batch, 16u, 8, opt.workspace_allocator);
    else if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 8, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 8, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 16u, 8, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 8, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 11 < tiles; i += 12)
        {
            __fp16* tm2p = tm2.row<__fp16>(i / 12);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
                // transpose 12x8
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64 \n"
                    "ld4    {v16.8h, v17.8h, v18.8h, v19.8h}, [%0] \n"

                    "sub    %0, %0, #128            \n"

                    "uzp1   v20.8h, v0.8h, v4.8h    \n" // 0
                    "uzp1   v21.8h, v16.8h, v1.8h   \n" // 1
                    "uzp1   v22.8h, v5.8h, v17.8h   \n" // 2
                    "uzp1   v23.8h, v2.8h, v6.8h    \n" // 3
                    "uzp1   v24.8h, v18.8h, v3.8h   \n" // 4
                    "uzp1   v25.8h, v7.8h, v19.8h   \n" // 5
                    "uzp2   v26.8h, v0.8h, v4.8h    \n" // 6
                    "uzp2   v27.8h, v16.8h, v1.8h   \n" // 7
                    "uzp2   v28.8h, v5.8h, v17.8h   \n" // 8
                    "uzp2   v29.8h, v2.8h, v6.8h    \n" // 9
                    "uzp2   v30.8h, v18.8h, v3.8h   \n" // 10
                    "uzp2   v31.8h, v7.8h, v19.8h   \n" // 11

                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%1], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

                r0 += bottom_blob_tm.cstep * 8;
            }
        }
        for (; i + 7 < tiles; i += 8)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 12 + (i % 12) / 8);

            const __fp16* r0 = bottom_blob_tm;

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
                    : "=r"(r0),    // %0
                    "=r"(tmpptr) // %1
                    : "0"(r0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");

                r0 += bottom_blob_tm.cstep * 8;
            }
        }
        for (; i + 3 < tiles; i += 4)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(r0),    // %0
                    "=r"(tmpptr) // %1
                    : "0"(r0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3");

                r0 += bottom_blob_tm.cstep * 8;
            }
        }
        for (; i + 1 < tiles; i += 2)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]   \n"
                    "ld1    {v0.8h, v1.8h}, [%0]    \n"
                    "st1    {v0.8h, v1.8h}, [%1], #32 \n"
                    : "=r"(r0),    // %0
                    "=r"(tmpptr) // %1
                    : "0"(r0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1");

                r0 += bottom_blob_tm.cstep * 8;
            }
        }
        for (; i < tiles; i++)
        {
            __fp16* tmpptr = tm2.row<__fp16>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

            const __fp16* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 8;

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]   \n"
                    "ld1    {v0.8h}, [%0]           \n"
                    "st1    {v0.8h}, [%1], #16      \n"
                    : "=r"(r0),    // %0
                    "=r"(tmpptr) // %1
                    : "0"(r0),
                    "1"(tmpptr)
                    : "memory", "v0");

                r0 += bottom_blob_tm.cstep * 8;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 16u, 8, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 12);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
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

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r0123

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w0123

                    "fmla   v20.8h, v12.8h, v0.h[0]     \n"
                    "fmla   v21.8h, v12.8h, v0.h[1]     \n"
                    "fmla   v22.8h, v12.8h, v0.h[2]     \n"
                    "fmla   v23.8h, v12.8h, v0.h[3]     \n"
                    "fmla   v24.8h, v12.8h, v0.h[4]     \n"
                    "fmla   v25.8h, v12.8h, v0.h[5]     \n"
                    "fmla   v26.8h, v12.8h, v0.h[6]     \n"
                    "fmla   v27.8h, v12.8h, v0.h[7]     \n"
                    "fmla   v28.8h, v12.8h, v1.h[0]     \n"
                    "fmla   v29.8h, v12.8h, v1.h[1]     \n"
                    "fmla   v30.8h, v12.8h, v1.h[2]     \n"
                    "fmla   v31.8h, v12.8h, v1.h[3]     \n"

                    "fmla   v20.8h, v13.8h, v1.h[4]     \n"
                    "fmla   v21.8h, v13.8h, v1.h[5]     \n"
                    "fmla   v22.8h, v13.8h, v1.h[6]     \n"
                    "fmla   v23.8h, v13.8h, v1.h[7]     \n"
                    "fmla   v24.8h, v13.8h, v2.h[0]     \n"
                    "fmla   v25.8h, v13.8h, v2.h[1]     \n"
                    "fmla   v26.8h, v13.8h, v2.h[2]     \n"
                    "fmla   v27.8h, v13.8h, v2.h[3]     \n"
                    "fmla   v28.8h, v13.8h, v2.h[4]     \n"
                    "fmla   v29.8h, v13.8h, v2.h[5]     \n"
                    "fmla   v30.8h, v13.8h, v2.h[6]     \n"
                    "fmla   v31.8h, v13.8h, v2.h[7]     \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r4567

                    "fmla   v20.8h, v14.8h, v3.h[0]     \n"
                    "fmla   v21.8h, v14.8h, v3.h[1]     \n"
                    "fmla   v22.8h, v14.8h, v3.h[2]     \n"
                    "fmla   v23.8h, v14.8h, v3.h[3]     \n"
                    "fmla   v24.8h, v14.8h, v3.h[4]     \n"
                    "fmla   v25.8h, v14.8h, v3.h[5]     \n"
                    "fmla   v26.8h, v14.8h, v3.h[6]     \n"
                    "fmla   v27.8h, v14.8h, v3.h[7]     \n"
                    "fmla   v28.8h, v14.8h, v4.h[0]     \n"
                    "fmla   v29.8h, v14.8h, v4.h[1]     \n"
                    "fmla   v30.8h, v14.8h, v4.h[2]     \n"
                    "fmla   v31.8h, v14.8h, v4.h[3]     \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%3], #64 \n" // w4567

                    "fmla   v20.8h, v15.8h, v4.h[4]     \n"
                    "fmla   v21.8h, v15.8h, v4.h[5]     \n"
                    "fmla   v22.8h, v15.8h, v4.h[6]     \n"
                    "fmla   v23.8h, v15.8h, v4.h[7]     \n"
                    "fmla   v24.8h, v15.8h, v5.h[0]     \n"
                    "fmla   v25.8h, v15.8h, v5.h[1]     \n"
                    "fmla   v26.8h, v15.8h, v5.h[2]     \n"
                    "fmla   v27.8h, v15.8h, v5.h[3]     \n"
                    "fmla   v28.8h, v15.8h, v5.h[4]     \n"
                    "fmla   v29.8h, v15.8h, v5.h[5]     \n"
                    "fmla   v30.8h, v15.8h, v5.h[6]     \n"
                    "fmla   v31.8h, v15.8h, v5.h[7]     \n"

                    "fmla   v20.8h, v16.8h, v6.h[0]     \n"
                    "fmla   v21.8h, v16.8h, v6.h[1]     \n"
                    "fmla   v22.8h, v16.8h, v6.h[2]     \n"
                    "fmla   v23.8h, v16.8h, v6.h[3]     \n"
                    "fmla   v24.8h, v16.8h, v6.h[4]     \n"
                    "fmla   v25.8h, v16.8h, v6.h[5]     \n"
                    "fmla   v26.8h, v16.8h, v6.h[6]     \n"
                    "fmla   v27.8h, v16.8h, v6.h[7]     \n"
                    "fmla   v28.8h, v16.8h, v7.h[0]     \n"
                    "fmla   v29.8h, v16.8h, v7.h[1]     \n"
                    "fmla   v30.8h, v16.8h, v7.h[2]     \n"
                    "fmla   v31.8h, v16.8h, v7.h[3]     \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%2], #64 \n" // r891011

                    "fmla   v20.8h, v17.8h, v7.h[4]     \n"
                    "fmla   v21.8h, v17.8h, v7.h[5]     \n"
                    "fmla   v22.8h, v17.8h, v7.h[6]     \n"
                    "fmla   v23.8h, v17.8h, v7.h[7]     \n"
                    "fmla   v24.8h, v17.8h, v8.h[0]     \n"
                    "fmla   v25.8h, v17.8h, v8.h[1]     \n"
                    "fmla   v26.8h, v17.8h, v8.h[2]     \n"
                    "fmla   v27.8h, v17.8h, v8.h[3]     \n"
                    "fmla   v28.8h, v17.8h, v8.h[4]     \n"
                    "fmla   v29.8h, v17.8h, v8.h[5]     \n"
                    "fmla   v30.8h, v17.8h, v8.h[6]     \n"
                    "fmla   v31.8h, v17.8h, v8.h[7]     \n"

                    "fmla   v20.8h, v18.8h, v9.h[0]     \n"
                    "fmla   v21.8h, v18.8h, v9.h[1]     \n"
                    "fmla   v22.8h, v18.8h, v9.h[2]     \n"
                    "fmla   v23.8h, v18.8h, v9.h[3]     \n"
                    "fmla   v24.8h, v18.8h, v9.h[4]     \n"
                    "fmla   v25.8h, v18.8h, v9.h[5]     \n"
                    "fmla   v26.8h, v18.8h, v9.h[6]     \n"
                    "fmla   v27.8h, v18.8h, v9.h[7]     \n"
                    "fmla   v28.8h, v18.8h, v10.h[0]    \n"
                    "fmla   v29.8h, v18.8h, v10.h[1]    \n"
                    "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                    "fmla   v31.8h, v18.8h, v10.h[3]    \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v20.8h, v19.8h, v10.h[4]    \n"
                    "fmla   v21.8h, v19.8h, v10.h[5]    \n"
                    "fmla   v22.8h, v19.8h, v10.h[6]    \n"
                    "fmla   v23.8h, v19.8h, v10.h[7]    \n"
                    "fmla   v24.8h, v19.8h, v11.h[0]    \n"
                    "fmla   v25.8h, v19.8h, v11.h[1]    \n"
                    "fmla   v26.8h, v19.8h, v11.h[2]    \n"
                    "fmla   v27.8h, v19.8h, v11.h[3]    \n"
                    "fmla   v28.8h, v19.8h, v11.h[4]    \n"
                    "fmla   v29.8h, v19.8h, v11.h[5]    \n"
                    "fmla   v30.8h, v19.8h, v11.h[6]    \n"
                    "fmla   v31.8h, v19.8h, v11.h[7]    \n"

                    "bne    0b                          \n"

                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%1], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < tiles; i += 8)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 12 + (i % 12) / 8);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

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

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r0123

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                    "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                    "fmla   v17.8h, v8.8h, v0.h[1]      \n"
                    "fmla   v18.8h, v8.8h, v0.h[2]      \n"
                    "fmla   v19.8h, v8.8h, v0.h[3]      \n"
                    "fmla   v20.8h, v8.8h, v0.h[4]      \n"
                    "fmla   v21.8h, v8.8h, v0.h[5]      \n"
                    "fmla   v22.8h, v8.8h, v0.h[6]      \n"
                    "fmla   v23.8h, v8.8h, v0.h[7]      \n"

                    "fmla   v16.8h, v9.8h, v1.h[0]      \n"
                    "fmla   v17.8h, v9.8h, v1.h[1]      \n"
                    "fmla   v18.8h, v9.8h, v1.h[2]      \n"
                    "fmla   v19.8h, v9.8h, v1.h[3]      \n"
                    "fmla   v20.8h, v9.8h, v1.h[4]      \n"
                    "fmla   v21.8h, v9.8h, v1.h[5]      \n"
                    "fmla   v22.8h, v9.8h, v1.h[6]      \n"
                    "fmla   v23.8h, v9.8h, v1.h[7]      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r4567

                    "fmla   v16.8h, v10.8h, v2.h[0]     \n"
                    "fmla   v17.8h, v10.8h, v2.h[1]     \n"
                    "fmla   v18.8h, v10.8h, v2.h[2]     \n"
                    "fmla   v19.8h, v10.8h, v2.h[3]     \n"
                    "fmla   v20.8h, v10.8h, v2.h[4]     \n"
                    "fmla   v21.8h, v10.8h, v2.h[5]     \n"
                    "fmla   v22.8h, v10.8h, v2.h[6]     \n"
                    "fmla   v23.8h, v10.8h, v2.h[7]     \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                    "fmla   v16.8h, v11.8h, v3.h[0]     \n"
                    "fmla   v17.8h, v11.8h, v3.h[1]     \n"
                    "fmla   v18.8h, v11.8h, v3.h[2]     \n"
                    "fmla   v19.8h, v11.8h, v3.h[3]     \n"
                    "fmla   v20.8h, v11.8h, v3.h[4]     \n"
                    "fmla   v21.8h, v11.8h, v3.h[5]     \n"
                    "fmla   v22.8h, v11.8h, v3.h[6]     \n"
                    "fmla   v23.8h, v11.8h, v3.h[7]     \n"

                    "fmla   v16.8h, v12.8h, v4.h[0]     \n"
                    "fmla   v17.8h, v12.8h, v4.h[1]     \n"
                    "fmla   v18.8h, v12.8h, v4.h[2]     \n"
                    "fmla   v19.8h, v12.8h, v4.h[3]     \n"
                    "fmla   v20.8h, v12.8h, v4.h[4]     \n"
                    "fmla   v21.8h, v12.8h, v4.h[5]     \n"
                    "fmla   v22.8h, v12.8h, v4.h[6]     \n"
                    "fmla   v23.8h, v12.8h, v4.h[7]     \n"

                    "fmla   v16.8h, v13.8h, v5.h[0]     \n"
                    "fmla   v17.8h, v13.8h, v5.h[1]     \n"
                    "fmla   v18.8h, v13.8h, v5.h[2]     \n"
                    "fmla   v19.8h, v13.8h, v5.h[3]     \n"
                    "fmla   v20.8h, v13.8h, v5.h[4]     \n"
                    "fmla   v21.8h, v13.8h, v5.h[5]     \n"
                    "fmla   v22.8h, v13.8h, v5.h[6]     \n"
                    "fmla   v23.8h, v13.8h, v5.h[7]     \n"

                    "fmla   v16.8h, v14.8h, v6.h[0]     \n"
                    "fmla   v17.8h, v14.8h, v6.h[1]     \n"
                    "fmla   v18.8h, v14.8h, v6.h[2]     \n"
                    "fmla   v19.8h, v14.8h, v6.h[3]     \n"
                    "fmla   v20.8h, v14.8h, v6.h[4]     \n"
                    "fmla   v21.8h, v14.8h, v6.h[5]     \n"
                    "fmla   v22.8h, v14.8h, v6.h[6]     \n"
                    "fmla   v23.8h, v14.8h, v6.h[7]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.8h, v15.8h, v7.h[0]     \n"
                    "fmla   v17.8h, v15.8h, v7.h[1]     \n"
                    "fmla   v18.8h, v15.8h, v7.h[2]     \n"
                    "fmla   v19.8h, v15.8h, v7.h[3]     \n"
                    "fmla   v20.8h, v15.8h, v7.h[4]     \n"
                    "fmla   v21.8h, v15.8h, v7.h[5]     \n"
                    "fmla   v22.8h, v15.8h, v7.h[6]     \n"
                    "fmla   v23.8h, v15.8h, v7.h[7]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"

                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
            }
            for (; i + 3 < tiles; i += 4)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r0123

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                    "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                    "fmla   v17.8h, v8.8h, v1.h[0]      \n"
                    "fmla   v18.8h, v8.8h, v2.h[0]      \n"
                    "fmla   v19.8h, v8.8h, v3.h[0]      \n"

                    "fmla   v16.8h, v9.8h, v0.h[1]      \n"
                    "fmla   v17.8h, v9.8h, v1.h[1]      \n"
                    "fmla   v18.8h, v9.8h, v2.h[1]      \n"
                    "fmla   v19.8h, v9.8h, v3.h[1]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                    "fmla   v16.8h, v10.8h, v0.h[2]     \n"
                    "fmla   v17.8h, v10.8h, v1.h[2]     \n"
                    "fmla   v18.8h, v10.8h, v2.h[2]     \n"
                    "fmla   v19.8h, v10.8h, v3.h[2]     \n"

                    "fmla   v16.8h, v11.8h, v0.h[3]     \n"
                    "fmla   v17.8h, v11.8h, v1.h[3]     \n"
                    "fmla   v18.8h, v11.8h, v2.h[3]     \n"
                    "fmla   v19.8h, v11.8h, v3.h[3]     \n"

                    "fmla   v16.8h, v12.8h, v0.h[4]     \n"
                    "fmla   v17.8h, v12.8h, v1.h[4]     \n"
                    "fmla   v18.8h, v12.8h, v2.h[4]     \n"
                    "fmla   v19.8h, v12.8h, v3.h[4]     \n"

                    "fmla   v16.8h, v13.8h, v0.h[5]     \n"
                    "fmla   v17.8h, v13.8h, v1.h[5]     \n"
                    "fmla   v18.8h, v13.8h, v2.h[5]     \n"
                    "fmla   v19.8h, v13.8h, v3.h[5]     \n"

                    "fmla   v16.8h, v14.8h, v0.h[6]     \n"
                    "fmla   v17.8h, v14.8h, v1.h[6]     \n"
                    "fmla   v18.8h, v14.8h, v2.h[6]     \n"
                    "fmla   v19.8h, v14.8h, v3.h[6]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.8h, v15.8h, v0.h[7]     \n"
                    "fmla   v17.8h, v15.8h, v1.h[7]     \n"
                    "fmla   v18.8h, v15.8h, v2.h[7]     \n"
                    "fmla   v19.8h, v15.8h, v3.h[7]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
            }
            for (; i + 1 < tiles; i += 2)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.8h, v1.8h}, [%2], #32   \n" // r01

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                    "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                    "fmla   v17.8h, v8.8h, v1.h[0]      \n"

                    "fmla   v16.8h, v9.8h, v0.h[1]      \n"
                    "fmla   v17.8h, v9.8h, v1.h[1]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                    "fmla   v16.8h, v10.8h, v0.h[2]     \n"
                    "fmla   v17.8h, v10.8h, v1.h[2]     \n"

                    "fmla   v16.8h, v11.8h, v0.h[3]     \n"
                    "fmla   v17.8h, v11.8h, v1.h[3]     \n"

                    "fmla   v16.8h, v12.8h, v0.h[4]     \n"
                    "fmla   v17.8h, v12.8h, v1.h[4]     \n"

                    "fmla   v16.8h, v13.8h, v0.h[5]     \n"
                    "fmla   v17.8h, v13.8h, v1.h[5]     \n"

                    "fmla   v16.8h, v14.8h, v0.h[6]     \n"
                    "fmla   v17.8h, v14.8h, v1.h[6]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.8h, v15.8h, v0.h[7]     \n"
                    "fmla   v17.8h, v15.8h, v1.h[7]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.8h, v17.8h}, [%1], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
            }
            for (; i < tiles; i++)
            {
                const __fp16* r0 = bb2.row<const __fp16>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v0.8h}, [%2], #16          \n" // r0

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                    "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                    "fmla   v16.8h, v9.8h, v0.h[1]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                    "fmla   v16.8h, v10.8h, v0.h[2]     \n"
                    "fmla   v16.8h, v11.8h, v0.h[3]     \n"

                    "fmla   v16.8h, v12.8h, v0.h[4]     \n"
                    "fmla   v16.8h, v13.8h, v0.h[5]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.8h, v14.8h, v0.h[6]     \n"
                    "fmla   v16.8h, v15.8h, v0.h[7]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.8h}, [%1], #16         \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
            }
        }
    }
}
