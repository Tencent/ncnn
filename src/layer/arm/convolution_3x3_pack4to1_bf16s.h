// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd64_pack4to1_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 64, inch, 16u, elempack, opt.workspace_allocator);
        conv3x3s1_winograd64_transform_input_pack4_bf16s_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = h_tm / 8 * w_tm / 8;

        // permute
        //         bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;
#if __aarch64__
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + tiles % 12 % 4, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, 4u * elempack, elempack, opt.workspace_allocator);
#else
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, 4u * elempack, elempack, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
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
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0] \n"
                        "sub    %0, %0, #128                \n"
                        "st1    {v0.4s}, [%1], #16          \n"
                        "st1    {v4.4s}, [%1], #16          \n"
                        "st1    {v16.4s}, [%1], #16         \n"
                        "st1    {v1.4s}, [%1], #16          \n"
                        "st1    {v5.4s}, [%1], #16          \n"
                        "st1    {v17.4s}, [%1], #16         \n"
                        "st1    {v2.4s}, [%1], #16          \n"
                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v18.4s}, [%1], #16         \n"
                        "st1    {v3.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%1], #16          \n"
                        "st1    {v19.4s}, [%1], #16         \n"
                        : "=r"(r0),  // %0
                        "=r"(tm2p) // %1
                        : "0"(r0),
                        "1"(tm2p)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19");
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
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                        "sub    %0, %0, #64                 \n"
                        "st1    {v0.4s}, [%1], #16          \n"
                        "st1    {v4.4s}, [%1], #16          \n"
                        "st1    {v1.4s}, [%1], #16          \n"
                        "st1    {v5.4s}, [%1], #16          \n"
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
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d0-d3}, [%0 :128]! \n"
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d4-d7}, [%0 :128]! \n"
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d16-d19}, [%0 :128]! \n"
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d20-d23}, [%0 :128] \n"
                        "sub        %0, %0, #96         \n"
                        "vswp       d1, d4              \n"
                        "vswp       d3, d6              \n"
                        "vswp       d17, d20            \n"
                        "vswp       d19, d22            \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
                        "vst1.f32   {d16-d17}, [%1 :128]! \n"
                        "vst1.f32   {d4-d5}, [%1 :128]! \n"
                        "vst1.f32   {d20-d21}, [%1 :128]! \n"
                        "vst1.f32   {d2-d3}, [%1 :128]! \n"
                        "vst1.f32   {d18-d19}, [%1 :128]! \n"
                        "vst1.f32   {d6-d7}, [%1 :128]! \n"
                        "vst1.f32   {d22-d23}, [%1 :128]! \n"
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
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                        : "=r"(r0),  // %0
                        "=r"(tm2p) // %1
                        : "0"(r0),
                        "1"(tm2p)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d0-d3}, [%0 :128]! \n"
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d4-d7}, [%0 :128]  \n"
                        "sub        %0, %0, #32         \n"
                        "vswp       d1, d4              \n"
                        "vswp       d3, d6              \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
                        "vst1.f32   {d4-d5}, [%1 :128]! \n"
                        "vst1.f32   {d2-d3}, [%1 :128]! \n"
                        "vst1.f32   {d6-d7}, [%1 :128]! \n"
                        : "=r"(r0),  // %0
                        "=r"(tm2p) // %1
                        : "0"(r0),
                        "1"(tm2p)
                        : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
                    r0 += bottom_blob_tm.cstep * 4;
                }
            }
            for (; i < tiles; i++)
            {
#if __aarch64__
                float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
#else
                float* tm2p = tm2.row(i / 8 + (i % 8) / 4 + i % 4);
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

        top_blob_tm.create(tiles, 64, outch, 4u, 1, opt.workspace_allocator);

        int nn_outch = 0;
        int remain_outch_start = 0;

#if __aarch64__
        nn_outch = outch >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 8;

            float* output0_tm = top_blob_tm.channel(p);
            float* output1_tm = top_blob_tm.channel(p + 1);
            float* output2_tm = top_blob_tm.channel(p + 2);
            float* output3_tm = top_blob_tm.channel(p + 3);
            float* output4_tm = top_blob_tm.channel(p + 4);
            float* output5_tm = top_blob_tm.channel(p + 5);
            float* output6_tm = top_blob_tm.channel(p + 6);
            float* output7_tm = top_blob_tm.channel(p + 7);

            const Mat kernel01_tm = kernel_tm.channel(p / 8);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);

                    const float* kptr = kernel01_tm.row(r);

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

                        "0:                             \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v11.4s, v0.4s, v4.s[1]  \n"
                        "fmla   v14.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v17.4s, v0.4s, v4.s[3]  \n"
                        "fmla   v20.4s, v0.4s, v5.s[0]  \n"
                        "fmla   v23.4s, v0.4s, v5.s[1]  \n"
                        "fmla   v26.4s, v0.4s, v5.s[2]  \n"
                        "fmla   v29.4s, v0.4s, v5.s[3]  \n"

                        "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                        "fmla   v12.4s, v1.4s, v4.s[1]  \n"
                        "fmla   v15.4s, v1.4s, v4.s[2]  \n"
                        "fmla   v18.4s, v1.4s, v4.s[3]  \n"
                        "fmla   v21.4s, v1.4s, v5.s[0]  \n"
                        "fmla   v24.4s, v1.4s, v5.s[1]  \n"
                        "fmla   v27.4s, v1.4s, v5.s[2]  \n"
                        "fmla   v30.4s, v1.4s, v5.s[3]  \n"

                        "fmla   v10.4s, v2.4s, v4.s[0]  \n"
                        "fmla   v13.4s, v2.4s, v4.s[1]  \n"
                        "fmla   v16.4s, v2.4s, v4.s[2]  \n"
                        "fmla   v19.4s, v2.4s, v4.s[3]  \n"
                        "fmla   v22.4s, v2.4s, v5.s[0]  \n"
                        "fmla   v25.4s, v2.4s, v5.s[1]  \n"
                        "fmla   v28.4s, v2.4s, v5.s[2]  \n"
                        "fmla   v31.4s, v2.4s, v5.s[3]  \n"

                        "fmla   v8.4s, v3.4s, v6.s[0]   \n"
                        "fmla   v11.4s, v3.4s, v6.s[1]  \n"
                        "fmla   v14.4s, v3.4s, v6.s[2]  \n"
                        "fmla   v17.4s, v3.4s, v6.s[3]  \n"
                        "fmla   v20.4s, v3.4s, v7.s[0]  \n"
                        "fmla   v23.4s, v3.4s, v7.s[1]  \n"
                        "fmla   v26.4s, v3.4s, v7.s[2]  \n"
                        "fmla   v29.4s, v3.4s, v7.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                        "fmla   v9.4s, v0.4s, v6.s[0]   \n"
                        "fmla   v12.4s, v0.4s, v6.s[1]  \n"
                        "fmla   v15.4s, v0.4s, v6.s[2]  \n"
                        "fmla   v18.4s, v0.4s, v6.s[3]  \n"
                        "fmla   v21.4s, v0.4s, v7.s[0]  \n"
                        "fmla   v24.4s, v0.4s, v7.s[1]  \n"
                        "fmla   v27.4s, v0.4s, v7.s[2]  \n"
                        "fmla   v30.4s, v0.4s, v7.s[3]  \n"

                        "fmla   v10.4s, v1.4s, v6.s[0]  \n"
                        "fmla   v13.4s, v1.4s, v6.s[1]  \n"
                        "fmla   v16.4s, v1.4s, v6.s[2]  \n"
                        "fmla   v19.4s, v1.4s, v6.s[3]  \n"
                        "fmla   v22.4s, v1.4s, v7.s[0]  \n"
                        "fmla   v25.4s, v1.4s, v7.s[1]  \n"
                        "fmla   v28.4s, v1.4s, v7.s[2]  \n"
                        "fmla   v31.4s, v1.4s, v7.s[3]  \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                        "fmla   v8.4s, v2.4s, v4.s[0]   \n"
                        "fmla   v11.4s, v2.4s, v4.s[1]  \n"
                        "fmla   v14.4s, v2.4s, v4.s[2]  \n"
                        "fmla   v17.4s, v2.4s, v4.s[3]  \n"
                        "fmla   v20.4s, v2.4s, v5.s[0]  \n"
                        "fmla   v23.4s, v2.4s, v5.s[1]  \n"
                        "fmla   v26.4s, v2.4s, v5.s[2]  \n"
                        "fmla   v29.4s, v2.4s, v5.s[3]  \n"

                        "fmla   v9.4s, v3.4s, v4.s[0]   \n"
                        "fmla   v12.4s, v3.4s, v4.s[1]  \n"
                        "fmla   v15.4s, v3.4s, v4.s[2]  \n"
                        "fmla   v18.4s, v3.4s, v4.s[3]  \n"
                        "fmla   v21.4s, v3.4s, v5.s[0]  \n"
                        "fmla   v24.4s, v3.4s, v5.s[1]  \n"
                        "fmla   v27.4s, v3.4s, v5.s[2]  \n"
                        "fmla   v30.4s, v3.4s, v5.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                        "fmla   v10.4s, v0.4s, v4.s[0]  \n"
                        "fmla   v13.4s, v0.4s, v4.s[1]  \n"
                        "fmla   v16.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v19.4s, v0.4s, v4.s[3]  \n"
                        "fmla   v22.4s, v0.4s, v5.s[0]  \n"
                        "fmla   v25.4s, v0.4s, v5.s[1]  \n"
                        "fmla   v28.4s, v0.4s, v5.s[2]  \n"
                        "fmla   v31.4s, v0.4s, v5.s[3]  \n"

                        "fmla   v8.4s, v1.4s, v6.s[0]   \n"
                        "fmla   v11.4s, v1.4s, v6.s[1]  \n"
                        "fmla   v14.4s, v1.4s, v6.s[2]  \n"
                        "fmla   v17.4s, v1.4s, v6.s[3]  \n"
                        "fmla   v20.4s, v1.4s, v7.s[0]  \n"
                        "fmla   v23.4s, v1.4s, v7.s[1]  \n"
                        "fmla   v26.4s, v1.4s, v7.s[2]  \n"
                        "fmla   v29.4s, v1.4s, v7.s[3]  \n"

                        "fmla   v9.4s, v2.4s, v6.s[0]   \n"
                        "fmla   v12.4s, v2.4s, v6.s[1]  \n"
                        "fmla   v15.4s, v2.4s, v6.s[2]  \n"
                        "fmla   v18.4s, v2.4s, v6.s[3]  \n"
                        "fmla   v21.4s, v2.4s, v7.s[0]  \n"
                        "fmla   v24.4s, v2.4s, v7.s[1]  \n"
                        "fmla   v27.4s, v2.4s, v7.s[2]  \n"
                        "fmla   v30.4s, v2.4s, v7.s[3]  \n"

                        "fmla   v10.4s, v3.4s, v6.s[0]  \n"
                        "fmla   v13.4s, v3.4s, v6.s[1]  \n"
                        "fmla   v16.4s, v3.4s, v6.s[2]  \n"
                        "fmla   v19.4s, v3.4s, v6.s[3]  \n"
                        "fmla   v22.4s, v3.4s, v7.s[0]  \n"
                        "fmla   v25.4s, v3.4s, v7.s[1]  \n"
                        "fmla   v28.4s, v3.4s, v7.s[2]  \n"
                        "fmla   v31.4s, v3.4s, v7.s[3]  \n"

                        "bne    0b                      \n"

                        "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"
                        "st1    {v11.4s, v12.4s, v13.4s}, [%2], #48 \n"
                        "st1    {v14.4s, v15.4s, v16.4s}, [%3], #48 \n"
                        "st1    {v17.4s, v18.4s, v19.4s}, [%4], #48 \n"
                        "st1    {v20.4s, v21.4s, v22.4s}, [%5], #48 \n"
                        "st1    {v23.4s, v24.4s, v25.4s}, [%6], #48 \n"
                        "st1    {v26.4s, v27.4s, v28.4s}, [%7], #48 \n"
                        "st1    {v29.4s, v30.4s, v31.4s}, [%8], #48 \n"

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
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);

                    const float* kptr = kernel01_tm.row(r);

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

                        "0:                             \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v16.4s, v0.4s, v4.s[0]  \n"
                        "fmla   v18.4s, v0.4s, v4.s[1]  \n"
                        "fmla   v20.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v22.4s, v0.4s, v4.s[3]  \n"
                        "fmla   v24.4s, v0.4s, v5.s[0]  \n"
                        "fmla   v26.4s, v0.4s, v5.s[1]  \n"
                        "fmla   v28.4s, v0.4s, v5.s[2]  \n"
                        "fmla   v30.4s, v0.4s, v5.s[3]  \n"
                        "fmla   v17.4s, v1.4s, v4.s[0]  \n"
                        "fmla   v19.4s, v1.4s, v4.s[1]  \n"
                        "fmla   v21.4s, v1.4s, v4.s[2]  \n"
                        "fmla   v23.4s, v1.4s, v4.s[3]  \n"
                        "fmla   v25.4s, v1.4s, v5.s[0]  \n"
                        "fmla   v27.4s, v1.4s, v5.s[1]  \n"
                        "fmla   v29.4s, v1.4s, v5.s[2]  \n"
                        "fmla   v31.4s, v1.4s, v5.s[3]  \n"

                        "fmla   v16.4s, v2.4s, v6.s[0]  \n"
                        "fmla   v18.4s, v2.4s, v6.s[1]  \n"
                        "fmla   v20.4s, v2.4s, v6.s[2]  \n"
                        "fmla   v22.4s, v2.4s, v6.s[3]  \n"
                        "fmla   v24.4s, v2.4s, v7.s[0]  \n"
                        "fmla   v26.4s, v2.4s, v7.s[1]  \n"
                        "fmla   v28.4s, v2.4s, v7.s[2]  \n"
                        "fmla   v30.4s, v2.4s, v7.s[3]  \n"
                        "fmla   v17.4s, v3.4s, v6.s[0]  \n"
                        "fmla   v19.4s, v3.4s, v6.s[1]  \n"
                        "fmla   v21.4s, v3.4s, v6.s[2]  \n"
                        "fmla   v23.4s, v3.4s, v6.s[3]  \n"
                        "fmla   v25.4s, v3.4s, v7.s[0]  \n"
                        "fmla   v27.4s, v3.4s, v7.s[1]  \n"
                        "fmla   v29.4s, v3.4s, v7.s[2]  \n"
                        "fmla   v31.4s, v3.4s, v7.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%9], #64 \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                        "fmla   v16.4s, v12.4s, v8.s[0] \n"
                        "fmla   v18.4s, v12.4s, v8.s[1] \n"
                        "fmla   v20.4s, v12.4s, v8.s[2] \n"
                        "fmla   v22.4s, v12.4s, v8.s[3] \n"
                        "fmla   v24.4s, v12.4s, v9.s[0] \n"
                        "fmla   v26.4s, v12.4s, v9.s[1] \n"
                        "fmla   v28.4s, v12.4s, v9.s[2] \n"
                        "fmla   v30.4s, v12.4s, v9.s[3] \n"
                        "fmla   v17.4s, v13.4s, v8.s[0] \n"
                        "fmla   v19.4s, v13.4s, v8.s[1] \n"
                        "fmla   v21.4s, v13.4s, v8.s[2] \n"
                        "fmla   v23.4s, v13.4s, v8.s[3] \n"
                        "fmla   v25.4s, v13.4s, v9.s[0] \n"
                        "fmla   v27.4s, v13.4s, v9.s[1] \n"
                        "fmla   v29.4s, v13.4s, v9.s[2] \n"
                        "fmla   v31.4s, v13.4s, v9.s[3] \n"

                        "fmla   v16.4s, v14.4s, v10.s[0] \n"
                        "fmla   v18.4s, v14.4s, v10.s[1] \n"
                        "fmla   v20.4s, v14.4s, v10.s[2] \n"
                        "fmla   v22.4s, v14.4s, v10.s[3] \n"
                        "fmla   v24.4s, v14.4s, v11.s[0] \n"
                        "fmla   v26.4s, v14.4s, v11.s[1] \n"
                        "fmla   v28.4s, v14.4s, v11.s[2] \n"
                        "fmla   v30.4s, v14.4s, v11.s[3] \n"
                        "fmla   v17.4s, v15.4s, v10.s[0] \n"
                        "fmla   v19.4s, v15.4s, v10.s[1] \n"
                        "fmla   v21.4s, v15.4s, v10.s[2] \n"
                        "fmla   v23.4s, v15.4s, v10.s[3] \n"
                        "fmla   v25.4s, v15.4s, v11.s[0] \n"
                        "fmla   v27.4s, v15.4s, v11.s[1] \n"
                        "fmla   v29.4s, v15.4s, v11.s[2] \n"
                        "fmla   v31.4s, v15.4s, v11.s[3] \n"

                        "bne    0b                      \n"

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
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

                    const float* kptr = kernel01_tm.row(r);

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

                        "0:                             \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v16.4s, v0.4s, v4.s[0]  \n"
                        "fmla   v17.4s, v0.4s, v4.s[1]  \n"
                        "fmla   v18.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v19.4s, v0.4s, v4.s[3]  \n"
                        "fmla   v20.4s, v0.4s, v5.s[0]  \n"
                        "fmla   v21.4s, v0.4s, v5.s[1]  \n"
                        "fmla   v22.4s, v0.4s, v5.s[2]  \n"
                        "fmla   v23.4s, v0.4s, v5.s[3]  \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                        "fmla   v16.4s, v1.4s, v6.s[0]  \n"
                        "fmla   v17.4s, v1.4s, v6.s[1]  \n"
                        "fmla   v18.4s, v1.4s, v6.s[2]  \n"
                        "fmla   v19.4s, v1.4s, v6.s[3]  \n"
                        "fmla   v20.4s, v1.4s, v7.s[0]  \n"
                        "fmla   v21.4s, v1.4s, v7.s[1]  \n"
                        "fmla   v22.4s, v1.4s, v7.s[2]  \n"
                        "fmla   v23.4s, v1.4s, v7.s[3]  \n"

                        "fmla   v16.4s, v2.4s, v8.s[0]  \n"
                        "fmla   v17.4s, v2.4s, v8.s[1]  \n"
                        "fmla   v18.4s, v2.4s, v8.s[2]  \n"
                        "fmla   v19.4s, v2.4s, v8.s[3]  \n"
                        "fmla   v20.4s, v2.4s, v9.s[0]  \n"
                        "fmla   v21.4s, v2.4s, v9.s[1]  \n"
                        "fmla   v22.4s, v2.4s, v9.s[2]  \n"
                        "fmla   v23.4s, v2.4s, v9.s[3]  \n"

                        "fmla   v16.4s, v3.4s, v10.s[0] \n"
                        "fmla   v17.4s, v3.4s, v10.s[1] \n"
                        "fmla   v18.4s, v3.4s, v10.s[2] \n"
                        "fmla   v19.4s, v3.4s, v10.s[3] \n"
                        "fmla   v20.4s, v3.4s, v11.s[0] \n"
                        "fmla   v21.4s, v3.4s, v11.s[1] \n"
                        "fmla   v22.4s, v3.4s, v11.s[2] \n"
                        "fmla   v23.4s, v3.4s, v11.s[3] \n"

                        "bne    0b                      \n"

                        "st1    {v16.4s}, [%1], #16     \n"
                        "st1    {v17.4s}, [%2], #16     \n"
                        "st1    {v18.4s}, [%3], #16     \n"
                        "st1    {v19.4s}, [%4], #16     \n"
                        "st1    {v20.4s}, [%5], #16     \n"
                        "st1    {v21.4s}, [%6], #16     \n"
                        "st1    {v22.4s}, [%7], #16     \n"
                        "st1    {v23.4s}, [%8], #16     \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%9, #128]   \n"
                        "ld1    {v0.4s}, [%9], #16      \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v16.4s, v4.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v5.4s, v0.s[0]  \n"
                        "fmla   v18.4s, v6.4s, v0.s[1]  \n"
                        "fmla   v19.4s, v7.4s, v0.s[1]  \n"

                        "prfm   pldl1keep, [%10, #512]  \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                        "fmla   v16.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v17.4s, v9.4s, v0.s[2]  \n"
                        "fmla   v18.4s, v10.4s, v0.s[3] \n"
                        "fmla   v19.4s, v11.4s, v0.s[3] \n"

                        "bne    0b                      \n"

                        "fadd   v16.4s, v16.4s, v18.4s  \n"
                        "fadd   v17.4s, v17.4s, v19.4s  \n"

                        "st1    {v16.s}[0], [%1], #4    \n"
                        "st1    {v16.s}[1], [%2], #4    \n"
                        "st1    {v16.s}[2], [%3], #4    \n"
                        "st1    {v16.s}[3], [%4], #4    \n"
                        "st1    {v17.s}[0], [%5], #4    \n"
                        "st1    {v17.s}[1], [%6], #4    \n"
                        "st1    {v17.s}[2], [%7], #4    \n"
                        "st1    {v17.s}[3], [%8], #4    \n"

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
                        : "cc", "memory", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
                }
            }
        }

        remain_outch_start += nn_outch << 3;
        nn_outch = (outch - remain_outch_start) >> 2;
#else  // __aarch64__
        nn_outch = outch >> 2;
#endif // __aarch64__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = remain_outch_start + pp * 4;

            float* output0_tm = top_blob_tm.channel(p);
            float* output1_tm = top_blob_tm.channel(p + 1);
            float* output2_tm = top_blob_tm.channel(p + 2);
            float* output3_tm = top_blob_tm.channel(p + 3);

#if __aarch64__
            const Mat kernel01_tm = kernel_tm.channel(p / 8 + (p % 8) / 4);
#else
            const Mat kernel01_tm = kernel_tm.channel(p / 4);
#endif

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __aarch64__
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);

                    const float* kptr = kernel01_tm.row(r);

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

                        "0:                             \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                        "prfm   pldl1keep, [%6, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v11.4s, v0.4s, v4.s[1]  \n"
                        "fmla   v14.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v17.4s, v0.4s, v4.s[3]  \n"
                        "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                        "fmla   v12.4s, v1.4s, v4.s[1]  \n"
                        "fmla   v15.4s, v1.4s, v4.s[2]  \n"
                        "fmla   v18.4s, v1.4s, v4.s[3]  \n"
                        "fmla   v10.4s, v2.4s, v4.s[0]  \n"
                        "fmla   v13.4s, v2.4s, v4.s[1]  \n"
                        "fmla   v16.4s, v2.4s, v4.s[2]  \n"
                        "fmla   v19.4s, v2.4s, v4.s[3]  \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%5], #64 \n"

                        "fmla   v8.4s, v3.4s, v5.s[0]   \n"
                        "fmla   v11.4s, v3.4s, v5.s[1]  \n"
                        "fmla   v14.4s, v3.4s, v5.s[2]  \n"
                        "fmla   v17.4s, v3.4s, v5.s[3]  \n"
                        "fmla   v9.4s, v20.4s, v5.s[0]  \n"
                        "fmla   v12.4s, v20.4s, v5.s[1] \n"
                        "fmla   v15.4s, v20.4s, v5.s[2] \n"
                        "fmla   v18.4s, v20.4s, v5.s[3] \n"
                        "fmla   v10.4s, v21.4s, v5.s[0] \n"
                        "fmla   v13.4s, v21.4s, v5.s[1] \n"
                        "fmla   v16.4s, v21.4s, v5.s[2] \n"
                        "fmla   v19.4s, v21.4s, v5.s[3] \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%5], #64 \n"

                        "fmla   v8.4s, v22.4s, v6.s[0]  \n"
                        "fmla   v11.4s, v22.4s, v6.s[1] \n"
                        "fmla   v14.4s, v22.4s, v6.s[2] \n"
                        "fmla   v17.4s, v22.4s, v6.s[3] \n"
                        "fmla   v9.4s, v23.4s, v6.s[0]  \n"
                        "fmla   v12.4s, v23.4s, v6.s[1] \n"
                        "fmla   v15.4s, v23.4s, v6.s[2] \n"
                        "fmla   v18.4s, v23.4s, v6.s[3] \n"
                        "fmla   v10.4s, v24.4s, v6.s[0] \n"
                        "fmla   v13.4s, v24.4s, v6.s[1] \n"
                        "fmla   v16.4s, v24.4s, v6.s[2] \n"
                        "fmla   v19.4s, v24.4s, v6.s[3] \n"

                        "fmla   v8.4s, v25.4s, v7.s[0]  \n"
                        "fmla   v11.4s, v25.4s, v7.s[1] \n"
                        "fmla   v14.4s, v25.4s, v7.s[2] \n"
                        "fmla   v17.4s, v25.4s, v7.s[3] \n"
                        "fmla   v9.4s, v26.4s, v7.s[0]  \n"
                        "fmla   v12.4s, v26.4s, v7.s[1] \n"
                        "fmla   v15.4s, v26.4s, v7.s[2] \n"
                        "fmla   v18.4s, v26.4s, v7.s[3] \n"
                        "fmla   v10.4s, v27.4s, v7.s[0] \n"
                        "fmla   v13.4s, v27.4s, v7.s[1] \n"
                        "fmla   v16.4s, v27.4s, v7.s[2] \n"
                        "fmla   v19.4s, v27.4s, v7.s[3] \n"

                        "bne    0b                      \n"

                        "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"
                        "st1    {v11.4s, v12.4s, v13.4s}, [%2], #48 \n"
                        "st1    {v14.4s, v15.4s, v16.4s}, [%3], #48 \n"
                        "st1    {v17.4s, v18.4s, v19.4s}, [%4], #48 \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
                }
#endif // __aarch64__
                for (; i + 7 < tiles; i += 8)
                {
#if __aarch64__
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
#else
                    const float* r0 = bb2.row(i / 8);
#endif

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"
                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"
                        "eor    v12.16b, v12.16b, v12.16b   \n"
                        "eor    v13.16b, v13.16b, v13.16b   \n"
                        "eor    v14.16b, v14.16b, v14.16b   \n"
                        "eor    v15.16b, v15.16b, v15.16b   \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                        "prfm   pldl1keep, [%6, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v10.4s, v0.4s, v4.s[1]  \n"
                        "fmla   v12.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v14.4s, v0.4s, v4.s[3]  \n"
                        "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                        "fmla   v11.4s, v1.4s, v4.s[1]  \n"
                        "fmla   v13.4s, v1.4s, v4.s[2]  \n"
                        "fmla   v15.4s, v1.4s, v4.s[3]  \n"

                        "fmla   v8.4s, v2.4s, v5.s[0]   \n"
                        "fmla   v10.4s, v2.4s, v5.s[1]  \n"
                        "fmla   v12.4s, v2.4s, v5.s[2]  \n"
                        "fmla   v14.4s, v2.4s, v5.s[3]  \n"
                        "fmla   v9.4s, v3.4s, v5.s[0]   \n"
                        "fmla   v11.4s, v3.4s, v5.s[1]  \n"
                        "fmla   v13.4s, v3.4s, v5.s[2]  \n"
                        "fmla   v15.4s, v3.4s, v5.s[3]  \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%5], #64 \n"

                        "fmla   v8.4s, v16.4s, v6.s[0]  \n"
                        "fmla   v10.4s, v16.4s, v6.s[1] \n"
                        "fmla   v12.4s, v16.4s, v6.s[2] \n"
                        "fmla   v14.4s, v16.4s, v6.s[3] \n"
                        "fmla   v9.4s, v17.4s, v6.s[0]  \n"
                        "fmla   v11.4s, v17.4s, v6.s[1] \n"
                        "fmla   v13.4s, v17.4s, v6.s[2] \n"
                        "fmla   v15.4s, v17.4s, v6.s[3] \n"

                        "fmla   v8.4s, v18.4s, v7.s[0]  \n"
                        "fmla   v10.4s, v18.4s, v7.s[1] \n"
                        "fmla   v12.4s, v18.4s, v7.s[2] \n"
                        "fmla   v14.4s, v18.4s, v7.s[3] \n"
                        "fmla   v9.4s, v19.4s, v7.s[0]  \n"
                        "fmla   v11.4s, v19.4s, v7.s[1] \n"
                        "fmla   v13.4s, v19.4s, v7.s[2] \n"
                        "fmla   v15.4s, v19.4s, v7.s[3] \n"

                        "bne    0b                      \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32 \n"
                        "st1    {v10.4s, v11.4s}, [%2], #32 \n"
                        "st1    {v12.4s, v13.4s}, [%3], #32 \n"
                        "st1    {v14.4s, v15.4s}, [%4], #32 \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
#else  // __aarch64__
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

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"

                        "pld        [%6, #512]      \n"
                        "vldm       %6!, {d8-d15}   \n"

                        "vmla.f32   q8, q0, d8[0]   \n"
                        "vmla.f32   q10, q0, d8[1]  \n"
                        "vmla.f32   q12, q0, d9[0]  \n"
                        "vmla.f32   q14, q0, d9[1]  \n"
                        "vmla.f32   q9, q1, d8[0]   \n"
                        "vmla.f32   q11, q1, d8[1]  \n"
                        "vmla.f32   q13, q1, d9[0]  \n"
                        "vmla.f32   q15, q1, d9[1]  \n"

                        "vmla.f32   q8, q2, d10[0]  \n"
                        "vmla.f32   q10, q2, d10[1] \n"
                        "vmla.f32   q12, q2, d11[0] \n"
                        "vmla.f32   q14, q2, d11[1] \n"
                        "vmla.f32   q9, q3, d10[0]  \n"
                        "vmla.f32   q11, q3, d10[1] \n"
                        "vmla.f32   q13, q3, d11[0] \n"
                        "vmla.f32   q15, q3, d11[1] \n"

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"

                        "vmla.f32   q8, q0, d12[0]  \n"
                        "vmla.f32   q10, q0, d12[1] \n"
                        "vmla.f32   q12, q0, d13[0] \n"
                        "vmla.f32   q14, q0, d13[1] \n"
                        "vmla.f32   q9, q1, d12[0]  \n"
                        "vmla.f32   q11, q1, d12[1] \n"
                        "vmla.f32   q13, q1, d13[0] \n"
                        "vmla.f32   q15, q1, d13[1] \n"

                        "subs       %0, %0, #1      \n"

                        "vmla.f32   q8, q2, d14[0]  \n"
                        "vmla.f32   q10, q2, d14[1] \n"
                        "vmla.f32   q12, q2, d15[0] \n"
                        "vmla.f32   q14, q2, d15[1] \n"
                        "vmla.f32   q9, q3, d14[0]  \n"
                        "vmla.f32   q11, q3, d14[1] \n"
                        "vmla.f32   q13, q3, d15[0] \n"
                        "vmla.f32   q15, q3, d15[1] \n"

                        "bne        0b              \n"

                        "vst1.f32   {d16-d19}, [%1]! \n"
                        "vst1.f32   {d20-d23}, [%2]! \n"
                        "vst1.f32   {d24-d27}, [%3]! \n"
                        "vst1.f32   {d28-d31}, [%4]! \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; i + 3 < tiles; i += 4)
                {
#if __aarch64__
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#endif

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"
                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                        "prfm   pldl1keep, [%6, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v9.4s, v0.4s, v4.s[1]   \n"
                        "fmla   v10.4s, v0.4s, v4.s[2]  \n"
                        "fmla   v11.4s, v0.4s, v4.s[3]  \n"

                        "fmla   v8.4s, v1.4s, v5.s[0]   \n"
                        "fmla   v9.4s, v1.4s, v5.s[1]   \n"
                        "fmla   v10.4s, v1.4s, v5.s[2]  \n"
                        "fmla   v11.4s, v1.4s, v5.s[3]  \n"

                        "fmla   v8.4s, v2.4s, v6.s[0]   \n"
                        "fmla   v9.4s, v2.4s, v6.s[1]   \n"
                        "fmla   v10.4s, v2.4s, v6.s[2]  \n"
                        "fmla   v11.4s, v2.4s, v6.s[3]  \n"

                        "fmla   v8.4s, v3.4s, v7.s[0]   \n"
                        "fmla   v9.4s, v3.4s, v7.s[1]   \n"
                        "fmla   v10.4s, v3.4s, v7.s[2]  \n"
                        "fmla   v11.4s, v3.4s, v7.s[3]  \n"

                        "bne    0b                      \n"

                        "st1    {v8.4s}, [%1], #16      \n"
                        "st1    {v9.4s}, [%2], #16      \n"
                        "st1    {v10.4s}, [%3], #16     \n"
                        "st1    {v11.4s}, [%4], #16     \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
                    asm volatile(
                        "veor       q8, q8          \n"
                        "veor       q9, q9          \n"
                        "veor       q10, q10        \n"
                        "veor       q11, q11        \n"

                        "0:                         \n"

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"

                        "pld        [%6, #512]      \n"
                        "vldm       %6!, {d8-d15}   \n"

                        "vmla.f32   q8, q0, d8[0]   \n"
                        "vmla.f32   q9, q0, d8[1]   \n"
                        "vmla.f32   q10, q0, d9[0]  \n"
                        "vmla.f32   q11, q0, d9[1]  \n"

                        "vmla.f32   q8, q1, d10[0]  \n"
                        "vmla.f32   q9, q1, d10[1]  \n"
                        "vmla.f32   q10, q1, d11[0] \n"
                        "vmla.f32   q11, q1, d11[1] \n"

                        "subs       %0, %0, #1      \n"

                        "vmla.f32   q8, q2, d12[0]  \n"
                        "vmla.f32   q9, q2, d12[1]  \n"
                        "vmla.f32   q10, q2, d13[0] \n"
                        "vmla.f32   q11, q2, d13[1] \n"

                        "vmla.f32   q8, q3, d14[0]  \n"
                        "vmla.f32   q9, q3, d14[1]  \n"
                        "vmla.f32   q10, q3, d15[0] \n"
                        "vmla.f32   q11, q3, d15[1] \n"

                        "bne        0b              \n"

                        "vst1.f32   {d16-d17}, [%1]! \n"
                        "vst1.f32   {d18-d19}, [%2]! \n"
                        "vst1.f32   {d20-d21}, [%3]! \n"
                        "vst1.f32   {d22-d23}, [%4]! \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
                }
                for (; i < tiles; i++)
                {
#if __aarch64__
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
#else
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#endif

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b  \n"
                        "eor    v9.16b, v9.16b, v9.16b  \n"
                        "eor    v10.16b, v10.16b, v10.16b \n"
                        "eor    v11.16b, v11.16b, v11.16b \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "prfm   pldl1keep, [%6, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[1]   \n"
                        "fmla   v10.4s, v6.4s, v0.s[2]  \n"
                        "fmla   v11.4s, v7.4s, v0.s[3]  \n"

                        "bne    0b                      \n"

                        "fadd   v8.4s, v8.4s, v9.4s     \n"
                        "fadd   v10.4s, v10.4s, v11.4s  \n"
                        "fadd   v8.4s, v8.4s, v10.4s    \n"

                        "st1    {v8.s}[0], [%1], #4     \n"
                        "st1    {v8.s}[1], [%2], #4     \n"
                        "st1    {v8.s}[2], [%3], #4     \n"
                        "st1    {v8.s}[3], [%4], #4     \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
                    asm volatile(
                        "veor       q8, q8          \n"
                        "veor       q9, q9          \n"
                        "veor       q10, q10        \n"
                        "veor       q11, q11        \n"

                        "0:                         \n"

                        "pld        [%5, #128]      \n"
                        "vld1.f32   {d0-d1}, [%5]!  \n"

                        "pld        [%6, #512]      \n"
                        "vldm       %6!, {d8-d15}   \n"

                        "subs       %0, %0, #1      \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q5, d0[1]   \n"
                        "vmla.f32   q10, q6, d1[0]  \n"
                        "vmla.f32   q11, q7, d1[1]  \n"

                        "bne        0b              \n"

                        "vadd.f32   q8, q8, q9      \n"
                        "vadd.f32   q10, q10, q11   \n"
                        "vadd.f32   q8, q8, q10     \n"

                        "vst1.f32   {d16[0]}, [%1]! \n"
                        "vst1.f32   {d16[1]}, [%2]! \n"
                        "vst1.f32   {d17[0]}, [%3]! \n"
                        "vst1.f32   {d17[1]}, [%4]! \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(output1_tm), // %2
                        "=r"(output2_tm), // %3
                        "=r"(output3_tm), // %4
                        "=r"(r0),         // %5
                        "=r"(kptr)        // %6
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(output2_tm),
                        "4"(output3_tm),
                        "5"(r0),
                        "6"(kptr)
                        : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
                }
            }
        }

        remain_outch_start += nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

#if __aarch64__
            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const Mat kernel0_tm = kernel_tm.channel(p / 4 + p % 4);
#endif

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __aarch64__
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b  \n"
                        "eor    v9.16b, v9.16b, v9.16b  \n"
                        "eor    v10.16b, v10.16b, v10.16b \n"
                        "eor    v5.16b, v5.16b, v5.16b  \n"
                        "eor    v6.16b, v6.16b, v6.16b  \n"
                        "eor    v7.16b, v7.16b, v7.16b  \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "prfm   pldl1keep, [%3, #128]   \n"
                        "ld1    {v4.4s}, [%3], #16      \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                        "fmla   v10.4s, v2.4s, v4.s[0]  \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n"

                        "fmla   v5.4s, v3.4s, v4.s[1]   \n"
                        "fmla   v6.4s, v12.4s, v4.s[1]  \n"
                        "fmla   v7.4s, v13.4s, v4.s[1]  \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64 \n"

                        "fmla   v8.4s, v14.4s, v4.s[2]  \n"
                        "fmla   v9.4s, v15.4s, v4.s[2]  \n"
                        "fmla   v10.4s, v16.4s, v4.s[2] \n"

                        "fmla   v5.4s, v17.4s, v4.s[3]  \n"
                        "fmla   v6.4s, v18.4s, v4.s[3]  \n"
                        "fmla   v7.4s, v19.4s, v4.s[3]  \n"

                        "bne    0b                      \n"

                        "fadd   v8.4s, v8.4s, v5.4s     \n"
                        "fadd   v9.4s, v9.4s, v6.4s     \n"
                        "fadd   v10.4s, v10.4s, v7.4s   \n"

                        "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
                }
#endif
                for (; i + 7 < tiles; i += 8)
                {
#if __aarch64__
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
#else
                    const float* r0 = bb2.row(i / 8);
#endif

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b  \n"
                        "eor    v9.16b, v9.16b, v9.16b  \n"
                        "eor    v10.16b, v10.16b, v10.16b \n"
                        "eor    v11.16b, v11.16b, v11.16b \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "prfm   pldl1keep, [%3, #128]   \n"
                        "ld1    {v4.4s}, [%3], #16      \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                        "fmla   v10.4s, v2.4s, v4.s[1]  \n"
                        "fmla   v11.4s, v3.4s, v4.s[1]  \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n"

                        "fmla   v8.4s, v12.4s, v4.s[2]  \n"
                        "fmla   v9.4s, v13.4s, v4.s[2]  \n"
                        "fmla   v10.4s, v14.4s, v4.s[3] \n"
                        "fmla   v11.4s, v15.4s, v4.s[3] \n"

                        "bne    0b                      \n"

                        "fadd   v8.4s, v8.4s, v10.4s    \n"
                        "fadd   v9.4s, v9.4s, v11.4s    \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32 \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
#else  // __aarch64__
                    asm volatile(
                        "veor       q8, q8          \n"
                        "veor       q9, q9          \n"
                        "veor       q10, q10        \n"
                        "veor       q11, q11        \n"

                        "0:                         \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d0-d7}    \n"

                        "pld        [%3, #128]      \n"
                        "vld1.f32   {d8-d9}, [%3]!  \n"

                        "vmla.f32   q8, q0, d8[0]   \n"
                        "vmla.f32   q9, q1, d8[0]   \n"
                        "vmla.f32   q10, q2, d8[1]  \n"
                        "vmla.f32   q11, q3, d8[1]  \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d24-d31}  \n"

                        "subs       %0, %0, #1      \n"

                        "vmla.f32   q8, q12, d9[0]  \n"
                        "vmla.f32   q9, q13, d9[0]  \n"
                        "vmla.f32   q10, q14, d9[1] \n"
                        "vmla.f32   q11, q15, d9[1] \n"

                        "bne        0b              \n"

                        "vadd.f32   q8, q8, q10     \n"
                        "vadd.f32   q9, q9, q11     \n"

                        "vst1.f32   {d16-d19}, [%1]! \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; i + 3 < tiles; i += 4)
                {
#if __aarch64__
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#endif

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b  \n"
                        "eor    v9.16b, v9.16b, v9.16b  \n"
                        "eor    v10.16b, v10.16b, v10.16b \n"
                        "eor    v11.16b, v11.16b, v11.16b \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "prfm   pldl1keep, [%3, #128]   \n"
                        "ld1    {v4.4s}, [%3], #16      \n"

                        "subs   %w0, %w0, #1            \n"

                        "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                        "fmla   v9.4s, v1.4s, v4.s[1]   \n"
                        "fmla   v10.4s, v2.4s, v4.s[2]  \n"
                        "fmla   v11.4s, v3.4s, v4.s[3]  \n"

                        "bne    0b                      \n"

                        "fadd   v8.4s, v8.4s, v9.4s     \n"
                        "fadd   v10.4s, v10.4s, v11.4s  \n"
                        "fadd   v8.4s, v8.4s, v10.4s    \n"

                        "st1    {v8.4s}, [%1], #16      \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11");
#else  // __aarch64__
                    asm volatile(
                        "veor       q8, q8          \n"
                        "veor       q9, q9          \n"
                        "veor       q10, q10        \n"
                        "veor       q11, q11        \n"

                        "0:                         \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d0-d7}    \n"

                        "pld        [%3, #128]      \n"
                        "vld1.f32   {d8-d9}, [%3]!  \n"

                        "subs       %0, %0, #1      \n"

                        "vmla.f32   q8, q0, d8[0]   \n"
                        "vmla.f32   q9, q1, d8[1]   \n"
                        "vmla.f32   q10, q2, d9[0]  \n"
                        "vmla.f32   q11, q3, d9[1]  \n"

                        "bne        0b              \n"

                        "vadd.f32   q8, q8, q9      \n"
                        "vadd.f32   q10, q10, q11   \n"
                        "vadd.f32   q8, q8, q10     \n"

                        "vst1.f32   {d16-d17}, [%1]! \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11");
#endif // __aarch64__
                }
                for (; i < tiles; i++)
                {
#if __aarch64__
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
#else
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#endif

                    const float* kptr = kernel0_tm.row(r);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);

                    for (int q = 0; q < inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32(r0);

                        float32x4_t _k0 = vld1q_f32(kptr);

                        _sum0 = vmlaq_f32(_sum0, _r0, _k0);

                        kptr += 4;
                        r0 += 4;
                    }

#if __aarch64__
                    float sum0 = vaddvq_f32(_sum0);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss2 = vpadd_f32(_ss, _ss);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif

                    output0_tm[0] = sum0;

                    output0_tm++;
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 2u, 1, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd64_transform_output_bf16s_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
