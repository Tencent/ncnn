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

static void convolution_winograd_dot_pack4_neon(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 16u, 4, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __aarch64__
    if (tiles >= 12)
        bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 4, opt.workspace_allocator);
#else
    if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 4, opt.workspace_allocator);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
#if __aarch64__
        for (; i + 11 < tiles; i += 12)
        {
            float* tm2p = tm2.row(i / 12);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x12
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v8.4s}, [%1], #16          \n"
                    "sub    %0, %0, #128                \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "st1    {v9.4s}, [%1], #16          \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v10.4s}, [%1], #16         \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    "st1    {v11.4s}, [%1], #16         \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
                r0 += bottom_blob_tm.cstep * 4;
            }
        }
#endif
        for (; i + 7 < tiles; i += 8)
        {
#if __aarch64__
            float* tm2p = tm2.row(i / 12 + (i % 12) / 8);
#else
            float* tm2p = tm2.row(i / 8);
#endif

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x8
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    "st1    {v4.4s}, [%1], #16          \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v5.4s}, [%1], #16          \n"
                    "sub    %0, %0, #64                 \n"
                    "st1    {v2.4s}, [%1], #16          \n"
                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v3.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%1], #16          \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0!, {d0-d7}        \n"
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d16-d23}       \n"

                    // transpose 8x4
                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vtrn.32    q8, q9              \n"
                    "vtrn.32    q10, q11            \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vswp       d17, d20            \n"
                    "vswp       d19, d22            \n"
                    "vswp       q1, q8              \n"
                    "vswp       q3, q10             \n"

                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "vst1.f32   {d16-d19}, [%1 :128]! \n"
                    "sub        %0, %0, #64         \n"
                    "vst1.f32   {d4-d7}, [%1 :128]! \n"
                    "vst1.f32   {d20-d23}, [%1 :128]! \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif
                r0 += bottom_blob_tm.cstep * 4;
            }
        }
        for (; i + 3 < tiles; i += 4)
        {
#if __aarch64__
            float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
            float* tm2p = tm2.row(i / 8 + (i % 8) / 4);
#endif

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x4
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "st4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1", "v2", "v3");
#else
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vtrn.32    q0, q1              \n"
                    "vtrn.32    q2, q3              \n"
                    "vswp       d1, d4              \n"
                    "vswp       d3, d6              \n"
                    "vstm       %1!, {d0-d7}        \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
                r0 += bottom_blob_tm.cstep * 4;
            }
        }
        for (; i + 1 < tiles; i += 2)
        {
#if __aarch64__
            float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
            float* tm2p = tm2.row(i / 8 + (i % 8) / 4 + (i % 4) / 2);
#endif

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x2
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%0]        \n"
                    "st2    {v0.4s, v1.4s}, [%1], #32   \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0", "v1");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]  \n"
                    "vst2.f32   {d0-d3}, [%1 :128]! \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "q0", "q1");
#endif // __aarch64__
                r0 += bottom_blob_tm.cstep * 4;
            }
        }
        for (; i < tiles; i++)
        {
#if __aarch64__
            float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
            float* tm2p = tm2.row(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
#endif

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v0.4s}, [%0]               \n"
                    "st1    {v0.4s}, [%1], #16          \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "v0");
#else
                asm volatile(
                    "pld        [%0, #128]          \n"
                    "vld1.f32   {d0-d1}, [%0 :128]  \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"
                    : "=r"(r0),  // %0
                    "=r"(tm2p) // %1
                    : "0"(r0),
                    "1"(tm2p)
                    : "memory", "q0");
#endif // __aarch64__
                r0 += bottom_blob_tm.cstep * 4;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 16u, 4, opt.workspace_allocator);

    int remain_outch_start = 0;

#if __aarch64__
    int nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        float* output0_tm = top_blob_tm.channel(p);
        float* output1_tm = top_blob_tm.channel(p + 1);

        const Mat kernel01_tm = kernel_tm.channel(pp);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                const float* r0 = bb2.row(i / 12);

                const float* k01 = kernel01_tm.row(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"
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

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64   \n" // w0011_01

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"

                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"

                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64   \n" // w2233_01

                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"

                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"

                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                    "bne    0b                          \n"

                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(k01)         // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(k01)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < tiles; i += 8)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8);

                const float* k01 = kernel01_tm.row(r);

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

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v8.4s, v1.s[3]      \n"

                    "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v9.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v9.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v9.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v9.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v1.s[3]      \n"

                    "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v2.s[3]     \n"
                    "fmla   v20.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v21.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v22.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v3.s[3]     \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v24.4s, v11.4s, v2.s[0]     \n"
                    "fmla   v25.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v26.4s, v11.4s, v2.s[2]     \n"
                    "fmla   v27.4s, v11.4s, v2.s[3]     \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r4 r5 r6 r7

                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"

                    "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v4.s[1]     \n"
                    "fmla   v18.4s, v12.4s, v4.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v4.s[3]     \n"
                    "fmla   v20.4s, v12.4s, v5.s[0]     \n"
                    "fmla   v21.4s, v12.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v12.4s, v5.s[2]     \n"
                    "fmla   v23.4s, v12.4s, v5.s[3]     \n"

                    "fmla   v24.4s, v13.4s, v4.s[0]     \n"
                    "fmla   v25.4s, v13.4s, v4.s[1]     \n"
                    "fmla   v26.4s, v13.4s, v4.s[2]     \n"
                    "fmla   v27.4s, v13.4s, v4.s[3]     \n"
                    "fmla   v28.4s, v13.4s, v5.s[0]     \n"
                    "fmla   v29.4s, v13.4s, v5.s[1]     \n"
                    "fmla   v30.4s, v13.4s, v5.s[2]     \n"
                    "fmla   v31.4s, v13.4s, v5.s[3]     \n"

                    "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                    "fmla   v17.4s, v14.4s, v6.s[1]     \n"
                    "fmla   v18.4s, v14.4s, v6.s[2]     \n"
                    "fmla   v19.4s, v14.4s, v6.s[3]     \n"
                    "fmla   v20.4s, v14.4s, v7.s[0]     \n"
                    "fmla   v21.4s, v14.4s, v7.s[1]     \n"
                    "fmla   v22.4s, v14.4s, v7.s[2]     \n"
                    "fmla   v23.4s, v14.4s, v7.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v24.4s, v15.4s, v6.s[0]     \n"
                    "fmla   v25.4s, v15.4s, v6.s[1]     \n"
                    "fmla   v26.4s, v15.4s, v6.s[2]     \n"
                    "fmla   v27.4s, v15.4s, v6.s[3]     \n"
                    "fmla   v28.4s, v15.4s, v7.s[0]     \n"
                    "fmla   v29.4s, v15.4s, v7.s[1]     \n"
                    "fmla   v30.4s, v15.4s, v7.s[2]     \n"
                    "fmla   v31.4s, v15.4s, v7.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(k01)         // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(k01)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

                const float* k01 = kernel01_tm.row(r);

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

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v9.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v9.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v9.4s, v0.s[3]      \n"

                    "fmla   v16.4s, v10.4s, v1.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v1.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v1.s[3]     \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v20.4s, v11.4s, v1.s[0]     \n"
                    "fmla   v21.4s, v11.4s, v1.s[1]     \n"
                    "fmla   v22.4s, v11.4s, v1.s[2]     \n"
                    "fmla   v23.4s, v11.4s, v1.s[3]     \n"

                    "fmla   v16.4s, v12.4s, v2.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v2.s[1]     \n"
                    "fmla   v18.4s, v12.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v2.s[3]     \n"
                    "fmla   v20.4s, v13.4s, v2.s[0]     \n"
                    "fmla   v21.4s, v13.4s, v2.s[1]     \n"
                    "fmla   v22.4s, v13.4s, v2.s[2]     \n"
                    "fmla   v23.4s, v13.4s, v2.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v14.4s, v3.s[0]     \n"
                    "fmla   v17.4s, v14.4s, v3.s[1]     \n"
                    "fmla   v18.4s, v14.4s, v3.s[2]     \n"
                    "fmla   v19.4s, v14.4s, v3.s[3]     \n"
                    "fmla   v20.4s, v15.4s, v3.s[0]     \n"
                    "fmla   v21.4s, v15.4s, v3.s[1]     \n"
                    "fmla   v22.4s, v15.4s, v3.s[2]     \n"
                    "fmla   v23.4s, v15.4s, v3.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(k01)         // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(k01)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
            }
            for (; i + 1 < tiles; i += 2)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                const float* k01 = kernel01_tm.row(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%3], #32   \n" // r0 r1

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v9.4s, v0.s[0]     \n"
                    "fmla   v19.4s, v9.4s, v0.s[1]     \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v16.4s, v10.4s, v0.s[2]      \n"
                    "fmla   v17.4s, v10.4s, v0.s[3]      \n"
                    "fmla   v18.4s, v11.4s, v0.s[2]     \n"
                    "fmla   v19.4s, v11.4s, v0.s[3]     \n"

                    "fmla   v16.4s, v12.4s, v1.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v1.s[1]     \n"
                    "fmla   v18.4s, v13.4s, v1.s[0]     \n"
                    "fmla   v19.4s, v13.4s, v1.s[1]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v14.4s, v1.s[2]     \n"
                    "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                    "fmla   v18.4s, v15.4s, v1.s[2]     \n"
                    "fmla   v19.4s, v15.4s, v1.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s}, [%1], #32 \n"
                    "st1    {v18.4s, v19.4s}, [%2], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(k01)         // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(k01)
                    : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

                const float* k01 = kernel01_tm.row(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v0.4s}, [%3], #16          \n" // r0

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v9.4s, v0.s[0]      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v18.4s, v10.4s, v0.s[1]     \n"
                    "fmla   v19.4s, v11.4s, v0.s[1]     \n"

                    "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v13.4s, v0.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v18.4s, v14.4s, v0.s[3]     \n"
                    "fmla   v19.4s, v15.4s, v0.s[3]     \n"

                    "bne    0b                          \n"

                    "fadd   v16.4s, v16.4s, v18.4s      \n"
                    "fadd   v17.4s, v17.4s, v19.4s      \n"

                    "st1    {v16.4s}, [%1], #16         \n"
                    "st1    {v17.4s}, [%2], #16         \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(output1_tm), // %2
                    "=r"(r0),         // %3
                    "=r"(k01)         // %4
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(output1_tm),
                    "3"(r0),
                    "4"(k01)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
            }
        }
    }
#endif // __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* output0_tm = top_blob_tm.channel(p);

#if __aarch64__
        const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#else
        const Mat kernel0_tm = kernel_tm.channel(p);
#endif

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
#if __aarch64__
            for (; i + 11 < tiles; i += 12)
            {
                const float* r0 = bb2.row(i / 12);

                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                asm volatile(
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // w0123_0

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                    "fmla   v8.4s, v5.4s, v3.s[0]       \n"
                    "fmla   v9.4s, v5.4s, v3.s[1]       \n"
                    "fmla   v10.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v5.4s, v3.s[3]      \n"
                    "fmla   v12.4s, v5.4s, v20.s[0]     \n"
                    "fmla   v13.4s, v5.4s, v20.s[1]     \n"
                    "fmla   v14.4s, v5.4s, v20.s[2]     \n"
                    "fmla   v15.4s, v5.4s, v20.s[3]     \n"
                    "fmla   v16.4s, v5.4s, v21.s[0]     \n"
                    "fmla   v17.4s, v5.4s, v21.s[1]     \n"
                    "fmla   v18.4s, v5.4s, v21.s[2]     \n"
                    "fmla   v19.4s, v5.4s, v21.s[3]     \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"

                    "fmla   v8.4s, v6.4s, v22.s[0]      \n"
                    "fmla   v9.4s, v6.4s, v22.s[1]      \n"
                    "fmla   v10.4s, v6.4s, v22.s[2]     \n"
                    "fmla   v11.4s, v6.4s, v22.s[3]     \n"
                    "fmla   v12.4s, v6.4s, v23.s[0]     \n"
                    "fmla   v13.4s, v6.4s, v23.s[1]     \n"
                    "fmla   v14.4s, v6.4s, v23.s[2]     \n"
                    "fmla   v15.4s, v6.4s, v23.s[3]     \n"
                    "fmla   v16.4s, v6.4s, v24.s[0]     \n"
                    "fmla   v17.4s, v6.4s, v24.s[1]     \n"
                    "fmla   v18.4s, v6.4s, v24.s[2]     \n"
                    "fmla   v19.4s, v6.4s, v24.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v8.4s, v7.4s, v25.s[0]      \n"
                    "fmla   v9.4s, v7.4s, v25.s[1]      \n"
                    "fmla   v10.4s, v7.4s, v25.s[2]     \n"
                    "fmla   v11.4s, v7.4s, v25.s[3]     \n"
                    "fmla   v12.4s, v7.4s, v26.s[0]     \n"
                    "fmla   v13.4s, v7.4s, v26.s[1]     \n"
                    "fmla   v14.4s, v7.4s, v26.s[2]     \n"
                    "fmla   v15.4s, v7.4s, v26.s[3]     \n"
                    "fmla   v16.4s, v7.4s, v27.s[0]     \n"
                    "fmla   v17.4s, v7.4s, v27.s[1]     \n"
                    "fmla   v18.4s, v7.4s, v27.s[2]     \n"
                    "fmla   v19.4s, v7.4s, v27.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
            }
#endif
            for (; i + 7 < tiles; i += 8)
            {
#if __aarch64__
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
#else
                const float* r0 = bb2.row(i / 8);
#endif

                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

#if __aarch64__
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
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v8.4s, v1.s[3]      \n"

                    "fmla   v16.4s, v9.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v9.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v9.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v9.4s, v2.s[3]      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r4 r5 r6 r7

                    "fmla   v20.4s, v9.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v9.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v9.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v9.4s, v3.s[3]      \n"

                    "fmla   v16.4s, v10.4s, v4.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v4.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v4.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v4.s[3]     \n"
                    "fmla   v20.4s, v10.4s, v5.s[0]     \n"
                    "fmla   v21.4s, v10.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v10.4s, v5.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v5.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v11.4s, v6.s[0]     \n"
                    "fmla   v17.4s, v11.4s, v6.s[1]     \n"
                    "fmla   v18.4s, v11.4s, v6.s[2]     \n"
                    "fmla   v19.4s, v11.4s, v6.s[3]     \n"
                    "fmla   v20.4s, v11.4s, v7.s[0]     \n"
                    "fmla   v21.4s, v11.4s, v7.s[1]     \n"
                    "fmla   v22.4s, v11.4s, v7.s[2]     \n"
                    "fmla   v23.4s, v11.4s, v7.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                asm volatile(
                    "veor       q8, q8          \n"
                    "veor       q9, q9          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"
                    "veor       q12, q12        \n"
                    "veor       q13, q13        \n"
                    "veor       q14, q14        \n"
                    "veor       q15, q15        \n"

                    "0:                         \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q4, d0[1]   \n"
                    "vmla.f32   q10, q4, d1[0]  \n"
                    "vmla.f32   q11, q4, d1[1]  \n"
                    "vmla.f32   q12, q4, d2[0]  \n"
                    "vmla.f32   q13, q4, d2[1]  \n"
                    "vmla.f32   q14, q4, d3[0]  \n"
                    "vmla.f32   q15, q4, d3[1]  \n"

                    "vmla.f32   q8, q5, d4[0]   \n"
                    "vmla.f32   q9, q5, d4[1]   \n"
                    "vmla.f32   q10, q5, d5[0]  \n"
                    "vmla.f32   q11, q5, d5[1]  \n"
                    "vmla.f32   q12, q5, d6[0]  \n"
                    "vmla.f32   q13, q5, d6[1]  \n"
                    "vmla.f32   q14, q5, d7[0]  \n"
                    "vmla.f32   q15, q5, d7[1]  \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "vmla.f32   q8, q6, d0[0]   \n"
                    "vmla.f32   q9, q6, d0[1]   \n"
                    "vmla.f32   q10, q6, d1[0]  \n"
                    "vmla.f32   q11, q6, d1[1]  \n"
                    "vmla.f32   q12, q6, d2[0]  \n"
                    "vmla.f32   q13, q6, d2[1]  \n"
                    "vmla.f32   q14, q6, d3[0]  \n"
                    "vmla.f32   q15, q6, d3[1]  \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q7, d4[0]   \n"
                    "vmla.f32   q9, q7, d4[1]   \n"
                    "vmla.f32   q10, q7, d5[0]  \n"
                    "vmla.f32   q11, q7, d5[1]  \n"
                    "vmla.f32   q12, q7, d6[0]  \n"
                    "vmla.f32   q13, q7, d6[1]  \n"
                    "vmla.f32   q14, q7, d7[0]  \n"
                    "vmla.f32   q15, q7, d7[1]  \n"

                    "bne        0b              \n"

                    "vstm       %1!, {d16-d23}  \n"
                    "vstm       %1!, {d24-d31}  \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
            }
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
                const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#endif

                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

#if __aarch64__
                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"

                    "fmla   v16.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v17.4s, v9.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v9.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v9.4s, v1.s[3]      \n"

                    "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v2.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v17.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v18.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v19.4s, v11.4s, v3.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
#else
                asm volatile(
                    "veor       q8, q8          \n"
                    "veor       q9, q9          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"

                    "0:                         \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q4, d0[1]   \n"
                    "vmla.f32   q10, q4, d1[0]  \n"
                    "vmla.f32   q11, q4, d1[1]  \n"

                    "vmla.f32   q8, q5, d2[0]   \n"
                    "vmla.f32   q9, q5, d2[1]   \n"
                    "vmla.f32   q10, q5, d3[0]  \n"
                    "vmla.f32   q11, q5, d3[1]  \n"

                    "vmla.f32   q8, q6, d4[0]   \n"
                    "vmla.f32   q9, q6, d4[1]   \n"
                    "vmla.f32   q10, q6, d5[0]  \n"
                    "vmla.f32   q11, q6, d5[1]  \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q7, d6[0]   \n"
                    "vmla.f32   q9, q7, d6[1]   \n"
                    "vmla.f32   q10, q7, d7[0]  \n"
                    "vmla.f32   q11, q7, d7[1]  \n"

                    "bne        0b              \n"

                    "vstm       %1!, {d16-d23}  \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
            }
            for (; i + 1 < tiles; i += 2)
            {
#if __aarch64__
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
                const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + (i % 4) / 2);
#endif

                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

#if __aarch64__
                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%2], #32   \n" // r0 r1

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"

                    "fmla   v18.4s, v9.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v9.4s, v0.s[3]      \n"

                    "fmla   v16.4s, v10.4s, v1.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[1]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v18.4s, v11.4s, v1.s[2]     \n"
                    "fmla   v19.4s, v11.4s, v1.s[3]     \n"

                    "bne    0b                          \n"

                    "fadd   v16.4s, v16.4s, v18.4s      \n"
                    "fadd   v17.4s, v17.4s, v19.4s      \n"

                    "st1    {v16.4s, v17.4s}, [%1], #32 \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
#else
                asm volatile(
                    "veor       q8, q8          \n"
                    "veor       q9, q9          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"

                    "0:                         \n"

                    "pld        [%2, #256]      \n"
                    "vld1.f32   {d0-d3}, [%2 :128]! \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q4, d0[1]   \n"

                    "vmla.f32   q10, q5, d1[0]  \n"
                    "vmla.f32   q11, q5, d1[1]  \n"

                    "vmla.f32   q8, q6, d2[0]   \n"
                    "vmla.f32   q9, q6, d2[1]   \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q10, q7, d3[0]  \n"
                    "vmla.f32   q11, q7, d3[1]  \n"

                    "bne        0b              \n"

                    "vadd.f32   q8, q8, q10     \n"
                    "vadd.f32   q9, q9, q11     \n"

                    "vst1.f32   {d16-d19}, [%1 :128]! \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
            }
            for (; i < tiles; i++)
            {
#if __aarch64__
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
                const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
#endif

                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

#if __aarch64__
                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v0.4s}, [%2], #16          \n" // r0

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v9.4s, v0.s[1]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v18.4s, v10.4s, v0.s[2]     \n"
                    "fmla   v19.4s, v11.4s, v0.s[3]     \n"

                    "bne    0b                          \n"

                    "fadd   v16.4s, v16.4s, v18.4s      \n"
                    "fadd   v17.4s, v17.4s, v19.4s      \n"
                    "fadd   v16.4s, v16.4s, v17.4s      \n"

                    "st1    {v16.4s}, [%1], #16         \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
#else
                asm volatile(
                    "veor       q8, q8          \n"
                    "veor       q9, q9          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"

                    "0:                         \n"

                    "pld        [%2, #128]      \n"
                    "vld1.f32   {d0-d1}, [%2 :128]! \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q5, d0[1]   \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q10, q6, d1[0]  \n"
                    "vmla.f32   q11, q7, d1[1]  \n"

                    "bne        0b              \n"

                    "vadd.f32   q8, q8, q10     \n"
                    "vadd.f32   q9, q9, q11     \n"
                    "vadd.f32   q8, q8, q9      \n"

                    "vst1.f32   {d16-d17}, [%1 :128]! \n"

                    : "=r"(nn),         // %0
                    "=r"(output0_tm), // %1
                    "=r"(r0),         // %2
                    "=r"(k0)          // %3
                    : "0"(nn),
                    "1"(output0_tm),
                    "2"(r0),
                    "3"(k0)
                    : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
            }
        }
    }
}
