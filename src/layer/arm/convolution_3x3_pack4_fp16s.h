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

static void conv3x3s1_winograd64_transform_kernel_pack4_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
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
    // dst = 4b-4a-inch/4a-64-outch/4b;
    kernel_tm_pack4.create(2 * inch / 4, 64, (outch / 4) / 2 + (outch / 4) % 2, (size_t)2u * 16, 16);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);
        const Mat k4 = kernel_tm.channel(q + 4);
        const Mat k5 = kernel_tm.channel(q + 5);
        const Mat k6 = kernel_tm.channel(q + 6);
        const Mat k7 = kernel_tm.channel(q + 7);

        Mat g0 = kernel_tm_pack4.channel(q / 8);

        for (int k = 0; k < 64; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p + 1);
                const float* k02 = k0.row(p + 2);
                const float* k03 = k0.row(p + 3);

                const float* k10 = k1.row(p);
                const float* k11 = k1.row(p + 1);
                const float* k12 = k1.row(p + 2);
                const float* k13 = k1.row(p + 3);

                const float* k20 = k2.row(p);
                const float* k21 = k2.row(p + 1);
                const float* k22 = k2.row(p + 2);
                const float* k23 = k2.row(p + 3);

                const float* k30 = k3.row(p);
                const float* k31 = k3.row(p + 1);
                const float* k32 = k3.row(p + 2);
                const float* k33 = k3.row(p + 3);

                const float* k40 = k4.row(p);
                const float* k41 = k4.row(p + 1);
                const float* k42 = k4.row(p + 2);
                const float* k43 = k4.row(p + 3);

                const float* k50 = k5.row(p);
                const float* k51 = k5.row(p + 1);
                const float* k52 = k5.row(p + 2);
                const float* k53 = k5.row(p + 3);

                const float* k60 = k6.row(p);
                const float* k61 = k6.row(p + 1);
                const float* k62 = k6.row(p + 2);
                const float* k63 = k6.row(p + 3);

                const float* k70 = k7.row(p);
                const float* k71 = k7.row(p + 1);
                const float* k72 = k7.row(p + 2);
                const float* k73 = k7.row(p + 3);

                g00[0] = (__fp16)k00[k];
                g00[1] = (__fp16)k10[k];
                g00[2] = (__fp16)k20[k];
                g00[3] = (__fp16)k30[k];

                g00[4] = (__fp16)k40[k];
                g00[5] = (__fp16)k50[k];
                g00[6] = (__fp16)k60[k];
                g00[7] = (__fp16)k70[k];

                g00[8] = (__fp16)k01[k];
                g00[9] = (__fp16)k11[k];
                g00[10] = (__fp16)k21[k];
                g00[11] = (__fp16)k31[k];

                g00[12] = (__fp16)k41[k];
                g00[13] = (__fp16)k51[k];
                g00[14] = (__fp16)k61[k];
                g00[15] = (__fp16)k71[k];

                g00[16] = (__fp16)k02[k];
                g00[17] = (__fp16)k12[k];
                g00[18] = (__fp16)k22[k];
                g00[19] = (__fp16)k32[k];

                g00[20] = (__fp16)k42[k];
                g00[21] = (__fp16)k52[k];
                g00[22] = (__fp16)k62[k];
                g00[23] = (__fp16)k72[k];

                g00[24] = (__fp16)k03[k];
                g00[25] = (__fp16)k13[k];
                g00[26] = (__fp16)k23[k];
                g00[27] = (__fp16)k33[k];

                g00[28] = (__fp16)k43[k];
                g00[29] = (__fp16)k53[k];
                g00[30] = (__fp16)k63[k];
                g00[31] = (__fp16)k73[k];

                g00 += 32;
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);

        Mat g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 64; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p + 1);
                const float* k02 = k0.row(p + 2);
                const float* k03 = k0.row(p + 3);

                const float* k10 = k1.row(p);
                const float* k11 = k1.row(p + 1);
                const float* k12 = k1.row(p + 2);
                const float* k13 = k1.row(p + 3);

                const float* k20 = k2.row(p);
                const float* k21 = k2.row(p + 1);
                const float* k22 = k2.row(p + 2);
                const float* k23 = k2.row(p + 3);

                const float* k30 = k3.row(p);
                const float* k31 = k3.row(p + 1);
                const float* k32 = k3.row(p + 2);
                const float* k33 = k3.row(p + 3);

                g00[0] = (__fp16)k00[k];
                g00[1] = (__fp16)k10[k];
                g00[2] = (__fp16)k20[k];
                g00[3] = (__fp16)k30[k];

                g00[4] = (__fp16)k01[k];
                g00[5] = (__fp16)k11[k];
                g00[6] = (__fp16)k21[k];
                g00[7] = (__fp16)k31[k];

                g00[8] = (__fp16)k02[k];
                g00[9] = (__fp16)k12[k];
                g00[10] = (__fp16)k22[k];
                g00[11] = (__fp16)k32[k];

                g00[12] = (__fp16)k03[k];
                g00[13] = (__fp16)k13[k];
                g00[14] = (__fp16)k23[k];
                g00[15] = (__fp16)k33[k];

                g00 += 16;
            }
        }
    }
}

static void conv3x3s1_winograd64_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        conv3x3s1_winograd64_transform_input_pack4_fp16sa_neon(bottom_blob_bordered, bottom_blob_tm, opt);
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

        top_blob_tm.create(tiles, 64, outch, 2u * elempack, elempack, opt.workspace_allocator);

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
        top_blob_bordered.create(outw, outh, outch, 2u * 4, 4, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd64_transform_output_pack4_fp16sa_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x4_t _bias0 = bias ? vld1_f16(bias + p * 4) : vdup_n_f16((__fp16)0.f);
        out0.fill(_bias0);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0.row<__fp16>(0);

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            // 16 * 9
            float16x8_t _k00_01 = vld1q_f16(kptr);
            float16x8_t _k00_23 = vld1q_f16(kptr + 8);
            float16x8_t _k01_01 = vld1q_f16(kptr + 16);
            float16x8_t _k01_23 = vld1q_f16(kptr + 24);
            float16x8_t _k02_01 = vld1q_f16(kptr + 32);
            float16x8_t _k02_23 = vld1q_f16(kptr + 40);
            float16x8_t _k10_01 = vld1q_f16(kptr + 48);
            float16x8_t _k10_23 = vld1q_f16(kptr + 56);
            float16x8_t _k11_01 = vld1q_f16(kptr + 64);
            float16x8_t _k11_23 = vld1q_f16(kptr + 72);
            float16x8_t _k12_01 = vld1q_f16(kptr + 80);
            float16x8_t _k12_23 = vld1q_f16(kptr + 88);
            float16x8_t _k20_01 = vld1q_f16(kptr + 96);
            float16x8_t _k20_23 = vld1q_f16(kptr + 104);
            float16x8_t _k21_01 = vld1q_f16(kptr + 112);
            float16x8_t _k21_23 = vld1q_f16(kptr + 120);
            float16x8_t _k22_01 = vld1q_f16(kptr + 128);
            float16x8_t _k22_23 = vld1q_f16(kptr + 136);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%1] \n" // r00 r01 r02 r03 r04 r05

                        "ext    v6.16b, %8.16b, %8.16b, #8  \n"
                        "fmla   v10.4h, %8.4h, v0.h[0]      \n"
                        "fmla   v11.4h, %8.4h, v0.h[4]      \n"
                        "fmla   v12.4h, %8.4h, v1.h[0]      \n"
                        "fmla   v13.4h, %8.4h, v1.h[4]      \n"
                        "fmla   v10.4h, v6.4h, v0.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[5]      \n"
                        "ext    v7.16b, %9.16b, %9.16b, #8  \n"
                        "fmla   v10.4h, %9.4h, v0.h[2]      \n"
                        "fmla   v11.4h, %9.4h, v0.h[6]      \n"
                        "fmla   v12.4h, %9.4h, v1.h[2]      \n"
                        "fmla   v13.4h, %9.4h, v1.h[6]      \n"
                        "fmla   v10.4h, v7.4h, v0.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[7]      \n"

                        "ext    v8.16b, %10.16b, %10.16b, #8 \n"
                        "fmla   v10.4h, %10.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %10.4h, v1.h[0]     \n"
                        "fmla   v12.4h, %10.4h, v1.h[4]     \n"
                        "fmla   v13.4h, %10.4h, v2.h[0]     \n"
                        "fmla   v10.4h, v8.4h, v0.h[5]      \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v12.4h, v8.4h, v1.h[5]      \n"
                        "fmla   v13.4h, v8.4h, v2.h[1]      \n"
                        "ext    v9.16b, %11.16b, %11.16b, #8 \n"
                        "fmla   v10.4h, %11.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %11.4h, v1.h[2]     \n"
                        "fmla   v12.4h, %11.4h, v1.h[6]     \n"
                        "fmla   v13.4h, %11.4h, v2.h[2]     \n"
                        "fmla   v10.4h, v9.4h, v0.h[7]      \n"
                        "fmla   v11.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v12.4h, v9.4h, v1.h[7]      \n"
                        "fmla   v13.4h, v9.4h, v2.h[3]      \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.8h, v4.8h, v5.8h}, [%2] \n" // r10 r11 r12 r13 r14 r15

                        "ext    v6.16b, %12.16b, %12.16b, #8 \n"
                        "fmla   v10.4h, %12.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %12.4h, v1.h[4]     \n"
                        "fmla   v12.4h, %12.4h, v2.h[0]     \n"
                        "fmla   v13.4h, %12.4h, v2.h[4]     \n"
                        "fmla   v10.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v1.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v2.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v2.h[5]      \n"
                        "ext    v7.16b, %13.16b, %13.16b, #8 \n"
                        "fmla   v10.4h, %13.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %13.4h, v1.h[6]     \n"
                        "fmla   v12.4h, %13.4h, v2.h[2]     \n"
                        "fmla   v13.4h, %13.4h, v2.h[6]     \n"
                        "fmla   v10.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v1.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v2.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v2.h[7]      \n"

                        "ext    v8.16b, %14.16b, %14.16b, #8 \n"
                        "fmla   v10.4h, %14.4h, v3.h[0]     \n"
                        "fmla   v11.4h, %14.4h, v3.h[4]     \n"
                        "fmla   v12.4h, %14.4h, v4.h[0]     \n"
                        "fmla   v13.4h, %14.4h, v4.h[4]     \n"
                        "fmla   v10.4h, v8.4h, v3.h[1]      \n"
                        "fmla   v11.4h, v8.4h, v3.h[5]      \n"
                        "fmla   v12.4h, v8.4h, v4.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v4.h[5]      \n"
                        "ext    v9.16b, %15.16b, %15.16b, #8 \n"
                        "fmla   v10.4h, %15.4h, v3.h[2]     \n"
                        "fmla   v11.4h, %15.4h, v3.h[6]     \n"
                        "fmla   v12.4h, %15.4h, v4.h[2]     \n"
                        "fmla   v13.4h, %15.4h, v4.h[6]     \n"
                        "fmla   v10.4h, v9.4h, v3.h[3]      \n"
                        "fmla   v11.4h, v9.4h, v3.h[7]      \n"
                        "fmla   v12.4h, v9.4h, v4.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v4.h[7]      \n"

                        "ext    v6.16b, %16.16b, %16.16b, #8 \n"
                        "fmla   v10.4h, %16.4h, v3.h[4]     \n"
                        "fmla   v11.4h, %16.4h, v4.h[0]     \n"
                        "fmla   v12.4h, %16.4h, v4.h[4]     \n"
                        "fmla   v13.4h, %16.4h, v5.h[0]     \n"
                        "fmla   v10.4h, v6.4h, v3.h[5]      \n"
                        "fmla   v11.4h, v6.4h, v4.h[1]      \n"
                        "fmla   v12.4h, v6.4h, v4.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v5.h[1]      \n"
                        "ext    v7.16b, %17.16b, %17.16b, #8 \n"
                        "fmla   v10.4h, %17.4h, v3.h[6]     \n"
                        "fmla   v11.4h, %17.4h, v4.h[2]     \n"
                        "fmla   v12.4h, %17.4h, v4.h[6]     \n"
                        "fmla   v13.4h, %17.4h, v5.h[2]     \n"
                        "fmla   v10.4h, v7.4h, v3.h[7]      \n"
                        "fmla   v11.4h, v7.4h, v4.h[3]      \n"
                        "fmla   v12.4h, v7.4h, v4.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v5.h[3]      \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%3] \n" // r20 r21 r22 r23 r24 r25

                        "ext    v8.16b, %18.16b, %18.16b, #8 \n"
                        "fmla   v10.4h, %18.4h, v4.h[0]     \n"
                        "fmla   v11.4h, %18.4h, v4.h[4]     \n"
                        "fmla   v12.4h, %18.4h, v5.h[0]     \n"
                        "fmla   v13.4h, %18.4h, v5.h[4]     \n"
                        "fmla   v10.4h, v8.4h, v4.h[1]      \n"
                        "fmla   v11.4h, v8.4h, v4.h[5]      \n"
                        "fmla   v12.4h, v8.4h, v5.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v5.h[5]      \n"
                        "ext    v9.16b, %19.16b, %19.16b, #8 \n"
                        "fmla   v10.4h, %19.4h, v4.h[2]     \n"
                        "fmla   v11.4h, %19.4h, v4.h[6]     \n"
                        "fmla   v12.4h, %19.4h, v5.h[2]     \n"
                        "fmla   v13.4h, %19.4h, v5.h[6]     \n"
                        "fmla   v10.4h, v9.4h, v4.h[3]      \n"
                        "fmla   v11.4h, v9.4h, v4.h[7]      \n"
                        "fmla   v12.4h, v9.4h, v5.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v5.h[7]      \n"

                        "ext    v6.16b, %20.16b, %20.16b, #8 \n"
                        "fmla   v10.4h, %20.4h, v0.h[0]     \n"
                        "fmla   v11.4h, %20.4h, v0.h[4]     \n"
                        "fmla   v12.4h, %20.4h, v1.h[0]     \n"
                        "fmla   v13.4h, %20.4h, v1.h[4]     \n"
                        "fmla   v10.4h, v6.4h, v0.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[5]      \n"
                        "ext    v7.16b, %21.16b, %21.16b, #8 \n"
                        "fmla   v10.4h, %21.4h, v0.h[2]     \n"
                        "fmla   v11.4h, %21.4h, v0.h[6]     \n"
                        "fmla   v12.4h, %21.4h, v1.h[2]     \n"
                        "fmla   v13.4h, %21.4h, v1.h[6]     \n"
                        "fmla   v10.4h, v7.4h, v0.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[7]      \n"

                        "ext    v8.16b, %22.16b, %22.16b, #8 \n"
                        "fmla   v10.4h, %22.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %22.4h, v1.h[0]     \n"
                        "fmla   v12.4h, %22.4h, v1.h[4]     \n"
                        "fmla   v13.4h, %22.4h, v2.h[0]     \n"
                        "fmla   v10.4h, v8.4h, v0.h[5]      \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v12.4h, v8.4h, v1.h[5]      \n"
                        "fmla   v13.4h, v8.4h, v2.h[1]      \n"
                        "ext    v9.16b, %23.16b, %23.16b, #8 \n"
                        "fmla   v10.4h, %23.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %23.4h, v1.h[2]     \n"
                        "fmla   v12.4h, %23.4h, v1.h[6]     \n"
                        "fmla   v13.4h, %23.4h, v2.h[2]     \n"
                        "fmla   v10.4h, v9.4h, v0.h[7]      \n"
                        "fmla   v11.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v12.4h, v9.4h, v1.h[7]      \n"
                        "fmla   v13.4h, v9.4h, v2.h[3]      \n"

                        "ext    v6.16b, %24.16b, %24.16b, #8 \n"
                        "fmla   v10.4h, %24.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %24.4h, v1.h[4]     \n"
                        "fmla   v12.4h, %24.4h, v2.h[0]     \n"
                        "fmla   v13.4h, %24.4h, v2.h[4]     \n"

                        "add    %1, %1, #32                 \n"

                        "fmla   v10.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v1.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v2.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v2.h[5]      \n"
                        "ext    v7.16b, %25.16b, %25.16b, #8 \n"
                        "fmla   v10.4h, %25.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %25.4h, v1.h[6]     \n"
                        "fmla   v12.4h, %25.4h, v2.h[2]     \n"
                        "fmla   v13.4h, %25.4h, v2.h[6]     \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v10.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v1.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v2.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v2.h[7]      \n"

                        "add    %3, %3, #32                 \n"

                        "st1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00_01), // %8
                        "w"(_k00_23), // %9
                        "w"(_k01_01), // %10
                        "w"(_k01_23), // %11
                        "w"(_k02_01), // %12
                        "w"(_k02_23), // %13
                        "w"(_k10_01), // %14
                        "w"(_k10_23), // %15
                        "w"(_k11_01), // %16
                        "w"(_k11_23), // %17
                        "w"(_k12_01), // %18
                        "w"(_k12_23), // %19
                        "w"(_k20_01), // %20
                        "w"(_k20_23), // %21
                        "w"(_k21_01), // %22
                        "w"(_k21_23), // %23
                        "w"(_k22_01), // %24
                        "w"(_k22_23)  // %25
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%1]        \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v12.4h, v13.4h}, [%0]      \n" // sum0 sum1

                        "ext    v4.16b, %8.16b, %8.16b, #8  \n"
                        "fmul   v10.4h, %8.4h, v0.h[0]      \n"
                        "fmul   v11.4h, %8.4h, v0.h[4]      \n"
                        "fmla   v12.4h, v4.4h, v0.h[1]      \n"
                        "fmla   v13.4h, v4.4h, v0.h[5]      \n"
                        "ext    v5.16b, %9.16b, %9.16b, #8  \n"
                        "fmla   v10.4h, %9.4h, v0.h[2]      \n"
                        "fmla   v11.4h, %9.4h, v0.h[6]      \n"
                        "fmla   v12.4h, v5.4h, v0.h[3]      \n"
                        "fmla   v13.4h, v5.4h, v0.h[7]      \n"

                        "ext    v6.16b, %10.16b, %10.16b, #8 \n"
                        "fmla   v10.4h, %10.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %10.4h, v1.h[0]     \n"
                        "fmla   v12.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[1]      \n"
                        "ext    v7.16b, %11.16b, %11.16b, #8 \n"
                        "fmla   v10.4h, %11.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %11.4h, v1.h[2]     \n"
                        "fmla   v12.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[3]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v2.8h, v3.8h}, [%2]        \n" // r10 r11 r12 r13

                        "ext    v8.16b, %12.16b, %12.16b, #8 \n"
                        "fmla   v10.4h, %12.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %12.4h, v1.h[4]     \n"
                        "fmla   v12.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v1.h[5]      \n"
                        "ext    v9.16b, %13.16b, %13.16b, #8 \n"
                        "fmla   v10.4h, %13.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %13.4h, v1.h[6]     \n"
                        "fmla   v12.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v1.h[7]      \n"

                        "ext    v4.16b, %14.16b, %14.16b, #8 \n"
                        "fmla   v10.4h, %14.4h, v2.h[0]     \n"
                        "fmla   v11.4h, %14.4h, v2.h[4]     \n"
                        "fmla   v12.4h, v4.4h, v2.h[1]      \n"
                        "fmla   v13.4h, v4.4h, v2.h[5]      \n"
                        "ext    v5.16b, %15.16b, %15.16b, #8 \n"
                        "fmla   v10.4h, %15.4h, v2.h[2]     \n"
                        "fmla   v11.4h, %15.4h, v2.h[6]     \n"
                        "fmla   v12.4h, v5.4h, v2.h[3]      \n"
                        "fmla   v13.4h, v5.4h, v2.h[7]      \n"

                        "ext    v6.16b, %16.16b, %16.16b, #8 \n"
                        "fmla   v10.4h, %16.4h, v2.h[4]     \n"
                        "fmla   v11.4h, %16.4h, v3.h[0]     \n"
                        "fmla   v12.4h, v6.4h, v2.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v3.h[1]      \n"
                        "ext    v7.16b, %17.16b, %17.16b, #8 \n"
                        "fmla   v10.4h, %17.4h, v2.h[6]     \n"
                        "fmla   v11.4h, %17.4h, v3.h[2]     \n"
                        "fmla   v12.4h, v7.4h, v2.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v3.h[3]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%3]        \n" // r20 r21 r22 r23

                        "ext    v8.16b, %18.16b, %18.16b, #8 \n"
                        "fmla   v10.4h, %18.4h, v3.h[0]     \n"
                        "fmla   v11.4h, %18.4h, v3.h[4]     \n"
                        "fmla   v12.4h, v8.4h, v3.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v3.h[5]      \n"
                        "ext    v9.16b, %19.16b, %19.16b, #8 \n"
                        "fmla   v10.4h, %19.4h, v3.h[2]     \n"
                        "fmla   v11.4h, %19.4h, v3.h[6]     \n"
                        "fmla   v12.4h, v9.4h, v3.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v3.h[7]      \n"

                        "ext    v4.16b, %20.16b, %20.16b, #8 \n"
                        "fmla   v10.4h, %20.4h, v0.h[0]     \n"
                        "fmla   v11.4h, %20.4h, v0.h[4]     \n"
                        "fmla   v12.4h, v4.4h, v0.h[1]      \n"
                        "fmla   v13.4h, v4.4h, v0.h[5]      \n"
                        "ext    v5.16b, %21.16b, %21.16b, #8 \n"
                        "fmla   v10.4h, %21.4h, v0.h[2]     \n"
                        "fmla   v11.4h, %21.4h, v0.h[6]     \n"
                        "fmla   v12.4h, v5.4h, v0.h[3]      \n"
                        "fmla   v13.4h, v5.4h, v0.h[7]      \n"

                        "ext    v6.16b, %22.16b, %22.16b, #8 \n"
                        "fmla   v10.4h, %22.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %22.4h, v1.h[0]     \n"
                        "fmla   v12.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[1]      \n"
                        "ext    v7.16b, %23.16b, %23.16b, #8 \n"
                        "fmla   v10.4h, %23.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %23.4h, v1.h[2]     \n"
                        "fmla   v12.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[3]      \n"

                        "ext    v8.16b, %24.16b, %24.16b, #8 \n"
                        "fmla   v10.4h, %24.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %24.4h, v1.h[4]     \n"
                        "fmla   v12.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v1.h[5]      \n"
                        "ext    v9.16b, %25.16b, %25.16b, #8 \n"
                        "fmla   v10.4h, %25.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %25.4h, v1.h[6]     \n"
                        "fmla   v12.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v1.h[7]      \n"

                        "add    %1, %1, #16                 \n"

                        "fadd   v10.4h, v10.4h, v12.4h      \n"

                        "add    %2, %2, #16                 \n"

                        "fadd   v11.4h, v11.4h, v13.4h      \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v10.4h, v11.4h}, [%0], #16 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00_01), // %8
                        "w"(_k00_23), // %9
                        "w"(_k01_01), // %10
                        "w"(_k01_23), // %11
                        "w"(_k02_01), // %12
                        "w"(_k02_23), // %13
                        "w"(_k10_01), // %14
                        "w"(_k10_23), // %15
                        "w"(_k11_01), // %16
                        "w"(_k11_23), // %17
                        "w"(_k12_01), // %18
                        "w"(_k12_23), // %19
                        "w"(_k20_01), // %20
                        "w"(_k20_23), // %21
                        "w"(_k21_01), // %22
                        "w"(_k21_23), // %23
                        "w"(_k22_01), // %24
                        "w"(_k22_23)  // %25
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%1] \n" // r00 r01 r02

                        "prfm   pldl1keep, [%0, #64]        \n"
                        "ld1    {v13.4h}, [%0]              \n" // sum0

                        "ext    v6.16b, %8.16b, %8.16b, #8  \n"
                        "fmul   v10.4h, %8.4h, v0.h[0]      \n"
                        "fmul   v11.4h, v6.4h, v0.h[1]      \n"
                        "ext    v7.16b, %9.16b, %9.16b, #8  \n"
                        "fmul   v12.4h, %9.4h, v0.h[2]      \n"
                        "fmla   v13.4h, v7.4h, v0.h[3]      \n"

                        "ext    v8.16b, %10.16b, %10.16b, #8 \n"
                        "fmla   v10.4h, %10.4h, v1.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "ext    v9.16b, %11.16b, %11.16b, #8 \n"
                        "fmla   v12.4h, %11.4h, v1.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v1.h[3]      \n"

                        "prfm   pldl1keep, [%2, #192]       \n"
                        "ld1    {v3.4h, v4.4h, v5.4h}, [%2] \n" // r10 r11 r12

                        "ext    v6.16b, %12.16b, %12.16b, #8 \n"
                        "fmla   v10.4h, %12.4h, v2.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v2.h[1]      \n"
                        "ext    v7.16b, %13.16b, %13.16b, #8 \n"
                        "fmla   v12.4h, %13.4h, v2.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v2.h[3]      \n"

                        "ext    v8.16b, %14.16b, %14.16b, #8 \n"
                        "fmla   v10.4h, %14.4h, v3.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v3.h[1]      \n"
                        "ext    v9.16b, %15.16b, %15.16b, #8 \n"
                        "fmla   v12.4h, %15.4h, v3.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v3.h[3]      \n"

                        "ext    v6.16b, %16.16b, %16.16b, #8 \n"
                        "fmla   v10.4h, %16.4h, v4.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v4.h[1]      \n"
                        "ext    v7.16b, %17.16b, %17.16b, #8 \n"
                        "fmla   v12.4h, %17.4h, v4.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v4.h[3]      \n"

                        "prfm   pldl1keep, [%3, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%3] \n" // r20 r21 r22

                        "ext    v8.16b, %18.16b, %18.16b, #8 \n"
                        "fmla   v10.4h, %18.4h, v5.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v5.h[1]      \n"
                        "ext    v9.16b, %19.16b, %19.16b, #8 \n"
                        "fmla   v12.4h, %19.4h, v5.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v5.h[3]      \n"

                        "ext    v6.16b, %20.16b, %20.16b, #8 \n"
                        "fmla   v10.4h, %20.4h, v0.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v0.h[1]      \n"
                        "ext    v7.16b, %21.16b, %21.16b, #8 \n"
                        "fmla   v12.4h, %21.4h, v0.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v0.h[3]      \n"

                        "ext    v8.16b, %22.16b, %22.16b, #8 \n"
                        "fmla   v10.4h, %22.4h, v1.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "ext    v9.16b, %23.16b, %23.16b, #8 \n"
                        "fmla   v12.4h, %23.4h, v1.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v1.h[3]      \n"

                        "ext    v6.16b, %24.16b, %24.16b, #8 \n"
                        "fmla   v10.4h, %24.4h, v2.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v2.h[1]      \n"
                        "ext    v7.16b, %25.16b, %25.16b, #8 \n"
                        "fmla   v12.4h, %25.4h, v2.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v2.h[3]      \n"

                        "fadd   v10.4h, v10.4h, v11.4h      \n"

                        "add    %1, %1, #8                  \n"

                        "fadd   v12.4h, v12.4h, v13.4h      \n"

                        "add    %2, %2, #8                  \n"

                        "fadd   v10.4h, v10.4h, v12.4h      \n"

                        "add    %3, %3, #8                  \n"

                        "st1    {v10.4h}, [%0], #8          \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00_01), // %8
                        "w"(_k00_23), // %9
                        "w"(_k01_01), // %10
                        "w"(_k01_23), // %11
                        "w"(_k02_01), // %12
                        "w"(_k02_23), // %13
                        "w"(_k10_01), // %14
                        "w"(_k10_23), // %15
                        "w"(_k11_01), // %16
                        "w"(_k11_23), // %17
                        "w"(_k12_01), // %18
                        "w"(_k12_23), // %19
                        "w"(_k20_01), // %20
                        "w"(_k20_23), // %21
                        "w"(_k21_01), // %22
                        "w"(_k21_23), // %23
                        "w"(_k22_01), // %24
                        "w"(_k22_23)  // %25
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }

                r0 += 8;
                r1 += 8;
                r2 += 8;
            }
        }
    }
}
