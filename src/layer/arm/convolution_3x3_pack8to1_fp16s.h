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

static void conv3x3s1_winograd64_transform_kernel_pack8to1_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch, const Option& opt)
{
    // winograd63 transform kernel
    Mat kernel_tm;
    kernel_tm.create(8 * 8, inch, outch);

    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i = 0; i < 8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j = 0; j < 8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++)
                {
                    kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 64-inch-outch
    // dst = 8a-inch/8a-64-outch;
    kernel_tm_pack8to1.create(8 * inch / 8, 64, outch / 8 + outch % 8, (size_t)2u * 8, 8);

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        const Mat k0 = kernel_tm.channel(p);
        const Mat k1 = kernel_tm.channel(p + 1);
        const Mat k2 = kernel_tm.channel(p + 2);
        const Mat k3 = kernel_tm.channel(p + 3);
        const Mat k4 = kernel_tm.channel(p + 4);
        const Mat k5 = kernel_tm.channel(p + 5);
        const Mat k6 = kernel_tm.channel(p + 6);
        const Mat k7 = kernel_tm.channel(p + 7);

        Mat g0 = kernel_tm_pack8to1.channel(p / 8);

        for (int k = 0; k < 64; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = (__fp16)k0.row(q + i)[k];
                    g00[1] = (__fp16)k1.row(q + i)[k];
                    g00[2] = (__fp16)k2.row(q + i)[k];
                    g00[3] = (__fp16)k3.row(q + i)[k];
                    g00[4] = (__fp16)k4.row(q + i)[k];
                    g00[5] = (__fp16)k5.row(q + i)[k];
                    g00[6] = (__fp16)k6.row(q + i)[k];
                    g00[7] = (__fp16)k7.row(q + i)[k];

                    g00 += 8;
                }
            }
        }
    }
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

        Mat g0 = kernel_tm_pack8to1.channel(p / 8 + p % 8);

        for (int k = 0; k < 64; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = (__fp16)k0.row(q + i)[k];

                    g00 += 1;
                }
            }
        }
    }
}

static void conv3x3s1_winograd64_pack8to1_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd64_transform_input_pack8_fp16sa_neon(bottom_blob_bordered, bottom_blob_tm, opt);
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
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 64, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 64, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, 2u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                __fp16* tm2p = tm2.row<__fp16>(i / 8);

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
                        : "=r"(r0),  // %0
                        "=r"(tm2p) // %1
                        : "0"(r0),
                        "1"(tm2p)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");

                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                __fp16* tm2p = tm2.row<__fp16>(i / 8 + (i % 8) / 4);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x4
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]   \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                        "st4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                        : "=r"(r0),  // %0
                        "=r"(tm2p) // %1
                        : "0"(r0),
                        "1"(tm2p)
                        : "memory", "v0", "v1", "v2", "v3");

                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i < tiles; i++)
            {
                __fp16* tm2p = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + i % 4);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]   \n"
                        "ld1    {v0.8h}, [%0]           \n"
                        "st1    {v0.8h}, [%1], #16      \n"
                        : "=r"(r0),  // %0
                        "=r"(tm2p) // %1
                        : "0"(r0),
                        "1"(tm2p)
                        : "memory", "v0");

                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 64, outch, 2u, 1, opt.workspace_allocator);

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 8;

            __fp16* output0_tm = top_blob_tm.channel(p);
            __fp16* output1_tm = top_blob_tm.channel(p + 1);
            __fp16* output2_tm = top_blob_tm.channel(p + 2);
            __fp16* output3_tm = top_blob_tm.channel(p + 3);
            __fp16* output4_tm = top_blob_tm.channel(p + 4);
            __fp16* output5_tm = top_blob_tm.channel(p + 5);
            __fp16* output6_tm = top_blob_tm.channel(p + 6);
            __fp16* output7_tm = top_blob_tm.channel(p + 7);

            const Mat kernel01_tm = kernel_tm.channel(p / 8);

            for (int r = 0; r < 64; r++)
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

                        "prfm   pldl1keep, [%9, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%9], #64 \n"

                        "prfm   pldl1keep, [%10, #512]      \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%10], #64 \n"

                        "fmla   v24.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v25.8h, v16.8h, v0.h[1]     \n"
                        "fmla   v26.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v27.8h, v16.8h, v0.h[3]     \n"
                        "fmla   v28.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v16.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v16.8h, v0.h[7]     \n"

                        "fmla   v24.8h, v17.8h, v1.h[0]     \n"
                        "fmla   v25.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v26.8h, v17.8h, v1.h[2]     \n"
                        "fmla   v27.8h, v17.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%9, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%9], #64 \n"

                        "fmla   v24.8h, v18.8h, v2.h[0]     \n"
                        "fmla   v25.8h, v18.8h, v2.h[1]     \n"
                        "fmla   v26.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v27.8h, v18.8h, v2.h[3]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v18.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%10, #512]      \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%10], #64 \n"

                        "fmla   v24.8h, v19.8h, v3.h[0]     \n"
                        "fmla   v25.8h, v19.8h, v3.h[1]     \n"
                        "fmla   v26.8h, v19.8h, v3.h[2]     \n"
                        "fmla   v27.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v28.8h, v19.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[7]     \n"

                        "fmla   v24.8h, v20.8h, v4.h[0]     \n"
                        "fmla   v25.8h, v20.8h, v4.h[1]     \n"
                        "fmla   v26.8h, v20.8h, v4.h[2]     \n"
                        "fmla   v27.8h, v20.8h, v4.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v20.8h, v4.h[7]     \n"

                        "fmla   v24.8h, v21.8h, v5.h[0]     \n"
                        "fmla   v25.8h, v21.8h, v5.h[1]     \n"
                        "fmla   v26.8h, v21.8h, v5.h[2]     \n"
                        "fmla   v27.8h, v21.8h, v5.h[3]     \n"
                        "fmla   v28.8h, v21.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[7]     \n"

                        "fmla   v24.8h, v22.8h, v6.h[0]     \n"
                        "fmla   v25.8h, v22.8h, v6.h[1]     \n"
                        "fmla   v26.8h, v22.8h, v6.h[2]     \n"
                        "fmla   v27.8h, v22.8h, v6.h[3]     \n"
                        "fmla   v28.8h, v22.8h, v6.h[4]     \n"
                        "fmla   v29.8h, v22.8h, v6.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[7]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v24.8h, v23.8h, v7.h[0]     \n"
                        "fmla   v25.8h, v23.8h, v7.h[1]     \n"
                        "fmla   v26.8h, v23.8h, v7.h[2]     \n"
                        "fmla   v27.8h, v23.8h, v7.h[3]     \n"
                        "fmla   v28.8h, v23.8h, v7.h[4]     \n"
                        "fmla   v29.8h, v23.8h, v7.h[5]     \n"
                        "fmla   v30.8h, v23.8h, v7.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "bne    0b                          \n"

                        "st1    {v24.8h}, [%1], #16         \n"
                        "st1    {v25.8h}, [%2], #16         \n"
                        "st1    {v26.8h}, [%3], #16         \n"
                        "st1    {v27.8h}, [%4], #16         \n"
                        "st1    {v28.8h}, [%5], #16         \n"
                        "st1    {v29.8h}, [%6], #16         \n"
                        "st1    {v30.8h}, [%7], #16         \n"
                        "st1    {v31.8h}, [%8], #16         \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
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
                        "eor    v28.16b, v28.16b, v28.16b   \n"
                        "eor    v29.16b, v29.16b, v29.16b   \n"
                        "eor    v30.16b, v30.16b, v30.16b   \n"
                        "eor    v31.16b, v31.16b, v31.16b   \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%9], #32 \n"

                        "prfm   pldl1keep, [%10, #512]      \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%10], #64 \n"

                        "fmla   v24.4h, v16.4h, v0.h[0]     \n"
                        "fmla   v25.4h, v16.4h, v0.h[1]     \n"
                        "fmla   v26.4h, v16.4h, v0.h[2]     \n"
                        "fmla   v27.4h, v16.4h, v0.h[3]     \n"
                        "fmla   v28.4h, v16.4h, v0.h[4]     \n"
                        "fmla   v29.4h, v16.4h, v0.h[5]     \n"
                        "fmla   v30.4h, v16.4h, v0.h[6]     \n"
                        "fmla   v31.4h, v16.4h, v0.h[7]     \n"

                        "fmla   v24.4h, v17.4h, v1.h[0]     \n"
                        "fmla   v25.4h, v17.4h, v1.h[1]     \n"
                        "fmla   v26.4h, v17.4h, v1.h[2]     \n"
                        "fmla   v27.4h, v17.4h, v1.h[3]     \n"
                        "fmla   v28.4h, v17.4h, v1.h[4]     \n"
                        "fmla   v29.4h, v17.4h, v1.h[5]     \n"
                        "fmla   v30.4h, v17.4h, v1.h[6]     \n"
                        "fmla   v31.4h, v17.4h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%9], #32 \n"

                        "fmla   v24.4h, v18.4h, v2.h[0]     \n"
                        "fmla   v25.4h, v18.4h, v2.h[1]     \n"
                        "fmla   v26.4h, v18.4h, v2.h[2]     \n"
                        "fmla   v27.4h, v18.4h, v2.h[3]     \n"
                        "fmla   v28.4h, v18.4h, v2.h[4]     \n"
                        "fmla   v29.4h, v18.4h, v2.h[5]     \n"
                        "fmla   v30.4h, v18.4h, v2.h[6]     \n"
                        "fmla   v31.4h, v18.4h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%10, #512]      \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%10], #64 \n"

                        "fmla   v24.4h, v19.4h, v3.h[0]     \n"
                        "fmla   v25.4h, v19.4h, v3.h[1]     \n"
                        "fmla   v26.4h, v19.4h, v3.h[2]     \n"
                        "fmla   v27.4h, v19.4h, v3.h[3]     \n"
                        "fmla   v28.4h, v19.4h, v3.h[4]     \n"
                        "fmla   v29.4h, v19.4h, v3.h[5]     \n"
                        "fmla   v30.4h, v19.4h, v3.h[6]     \n"
                        "fmla   v31.4h, v19.4h, v3.h[7]     \n"

                        "fmla   v24.4h, v20.4h, v4.h[0]     \n"
                        "fmla   v25.4h, v20.4h, v4.h[1]     \n"
                        "fmla   v26.4h, v20.4h, v4.h[2]     \n"
                        "fmla   v27.4h, v20.4h, v4.h[3]     \n"
                        "fmla   v28.4h, v20.4h, v4.h[4]     \n"
                        "fmla   v29.4h, v20.4h, v4.h[5]     \n"
                        "fmla   v30.4h, v20.4h, v4.h[6]     \n"
                        "fmla   v31.4h, v20.4h, v4.h[7]     \n"

                        "fmla   v24.4h, v21.4h, v5.h[0]     \n"
                        "fmla   v25.4h, v21.4h, v5.h[1]     \n"
                        "fmla   v26.4h, v21.4h, v5.h[2]     \n"
                        "fmla   v27.4h, v21.4h, v5.h[3]     \n"
                        "fmla   v28.4h, v21.4h, v5.h[4]     \n"
                        "fmla   v29.4h, v21.4h, v5.h[5]     \n"
                        "fmla   v30.4h, v21.4h, v5.h[6]     \n"
                        "fmla   v31.4h, v21.4h, v5.h[7]     \n"

                        "fmla   v24.4h, v22.4h, v6.h[0]     \n"
                        "fmla   v25.4h, v22.4h, v6.h[1]     \n"
                        "fmla   v26.4h, v22.4h, v6.h[2]     \n"
                        "fmla   v27.4h, v22.4h, v6.h[3]     \n"
                        "fmla   v28.4h, v22.4h, v6.h[4]     \n"
                        "fmla   v29.4h, v22.4h, v6.h[5]     \n"
                        "fmla   v30.4h, v22.4h, v6.h[6]     \n"
                        "fmla   v31.4h, v22.4h, v6.h[7]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v24.4h, v23.4h, v7.h[0]     \n"
                        "fmla   v25.4h, v23.4h, v7.h[1]     \n"
                        "fmla   v26.4h, v23.4h, v7.h[2]     \n"
                        "fmla   v27.4h, v23.4h, v7.h[3]     \n"
                        "fmla   v28.4h, v23.4h, v7.h[4]     \n"
                        "fmla   v29.4h, v23.4h, v7.h[5]     \n"
                        "fmla   v30.4h, v23.4h, v7.h[6]     \n"
                        "fmla   v31.4h, v23.4h, v7.h[7]     \n"

                        "bne    0b                          \n"

                        "st1    {v24.4h}, [%1], #8          \n"
                        "st1    {v25.4h}, [%2], #8          \n"
                        "st1    {v26.4h}, [%3], #8          \n"
                        "st1    {v27.4h}, [%4], #8          \n"
                        "st1    {v28.4h}, [%5], #8          \n"
                        "st1    {v29.4h}, [%6], #8          \n"
                        "st1    {v30.4h}, [%7], #8          \n"
                        "st1    {v31.4h}, [%8], #8          \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                for (; i < tiles; i++)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + i % 4);

                    const __fp16* kptr = kernel01_tm.row<const __fp16>(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "eor    v30.16b, v30.16b, v30.16b   \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%9, #128]       \n"
                        "ld1    {v0.8h}, [%9], #16          \n"

                        "prfm   pldl1keep, [%10, #512]      \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%10], #64 \n"

                        "fmla   v30.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%10, #512]      \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%10], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v30.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"

                        "bne    0b                          \n"

                        "st1    {v30.h}[0], [%1], #2        \n"
                        "st1    {v30.h}[1], [%2], #2        \n"
                        "st1    {v30.h}[2], [%3], #2        \n"
                        "st1    {v30.h}[3], [%4], #2        \n"
                        "st1    {v30.h}[4], [%5], #2        \n"
                        "st1    {v30.h}[5], [%6], #2        \n"
                        "st1    {v30.h}[6], [%7], #2        \n"
                        "st1    {v30.h}[7], [%8], #2        \n"

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
                        : "cc", "memory", "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v30");
                }
            }
        }

        remain_outch_start += nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            __fp16* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p / 8 + p % 8);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8);

                    const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "eor    v30.16b, v30.16b, v30.16b   \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%2], #64 \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.8h}, [%3], #16          \n"

                        "fmla   v30.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%2], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v30.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"

                        "bne    0b                          \n"

                        "st1    {v30.8h}, [%1], #16         \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v30");
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4);

                    const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "eor    v30.16b, v30.16b, v30.16b   \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%2], #32 \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.8h}, [%3], #16          \n"

                        "fmla   v30.4h, v16.4h, v0.h[0]     \n"
                        "fmla   v30.4h, v17.4h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n"

                        "fmla   v30.4h, v18.4h, v0.h[2]     \n"
                        "fmla   v30.4h, v19.4h, v0.h[3]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v30.4h, v20.4h, v0.h[4]     \n"
                        "fmla   v30.4h, v21.4h, v0.h[5]     \n"
                        "fmla   v30.4h, v22.4h, v0.h[6]     \n"
                        "fmla   v30.4h, v23.4h, v0.h[7]     \n"

                        "bne    0b                          \n"

                        "st1    {v30.4h}, [%1], #8          \n"

                        : "=r"(nn),         // %0
                        "=r"(output0_tm), // %1
                        "=r"(r0),         // %2
                        "=r"(kptr)        // %3
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(kptr)
                        : "cc", "memory", "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v30");
                }
                for (; i < tiles; i++)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + i % 4);

                    const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                    float16x8_t _sum0 = vdupq_n_f16((__fp16)0.f);

                    for (int q = 0; q < inch; q++)
                    {
                        float16x8_t _r0 = vld1q_f16(r0);

                        float16x8_t _k0 = vld1q_f16(kptr);

                        _sum0 = vfmaq_f16(_sum0, _r0, _k0);

                        kptr += 8;
                        r0 += 8;
                    }

                    __fp16 sum0 = vaddvq_f32(vcvt_f32_f16(vadd_f16(vget_low_f16(_sum0), vget_high_f16(_sum0))));

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
        conv3x3s1_winograd64_transform_output_fp16sa_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
