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

static void conv3x3s1_winograd42_transform_kernel_pack8to4_int8_neon(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch)
{
    // winograd42 transform kernel
    Mat kernel_tm(6 * 6, inch, outch, 2u);

    const short ktm[6][3] = {
        {6, 0, 0},
        {-4, -4, -4},
        {-4, 4, -4},
        {1, 2, 4},
        {1, -2, 4},
        {0, 0, 6}
    };

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const signed char* kernel0 = (const signed char*)kernel + p * inch * 9 + q * 9;
            short* kernel_tm0 = kernel_tm.channel(p).row<short>(q);

            // transform kernel
            const signed char* k0 = kernel0;
            const signed char* k1 = kernel0 + 3;
            const signed char* k2 = kernel0 + 6;

            // h
            short tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                short* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 36-inch-outch
    // dst = 4b-8a-inch/8a-36-outch/4b
    kernel_tm_pack8.create(inch / 8, 36, outch / 8 + (outch % 8) / 4, (size_t)2u * 64, 64);

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

        Mat kernel_tm = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = kernel_tm.row<short>(k);

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const short* k00 = k0.row<const short>(p + i);
                    const short* k10 = k1.row<const short>(p + i);
                    const short* k20 = k2.row<const short>(p + i);
                    const short* k30 = k3.row<const short>(p + i);
                    const short* k40 = k4.row<const short>(p + i);
                    const short* k50 = k5.row<const short>(p + i);
                    const short* k60 = k6.row<const short>(p + i);
                    const short* k70 = k7.row<const short>(p + i);

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];
                    g00[4] = k40[k];
                    g00[5] = k50[k];
                    g00[6] = k60[k];
                    g00[7] = k70[k];

                    g00 += 8;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);

        Mat kernel_tm = kernel_tm_pack8.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = kernel_tm.row<short>(k);

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const short* k00 = k0.row<const short>(p + i);
                    const short* k10 = k1.row<const short>(p + i);
                    const short* k20 = k2.row<const short>(p + i);
                    const short* k30 = k3.row<const short>(p + i);

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];

                    g00 += 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_pack8to4_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    //     size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 4n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = w_tm / 6 * h_tm / 6;

        bottom_blob_tm.create(tiles, 36, inch, 2u * elempack, elempack, opt.workspace_allocator);

        // const float itm[4][4] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =  4 * r00 - 5 * r02 + r04
        // 1 = -4 * (r01 + r02) + r04 + r03
        // 2 =  4 * (r01 - r02) + r04 - r03
        // 3 = -2 * (r01 - r03) + r04 - r02
        // 4 =  2 * (r01 - r03) + r04 - r02
        // 5 =  4 * r01 - 5 * r03 + r05

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            short tmp[6][6][8];

            // tile
            for (int i = 0; i < h_tm / 6; i++)
            {
                for (int j = 0; j < w_tm / 6; j++)
                {
                    const signed char* r0 = img0.row<const signed char>(i * 4) + (j * 4) * 8;

                    for (int m = 0; m < 6; m++)
                    {
                        int8x8_t _r00 = vld1_s8(r0);
                        int8x8_t _r01 = vld1_s8(r0 + 8);
                        int8x8_t _r02 = vld1_s8(r0 + 16);
                        int8x8_t _r03 = vld1_s8(r0 + 24);
                        int8x8_t _r04 = vld1_s8(r0 + 32);
                        int8x8_t _r05 = vld1_s8(r0 + 40);

                        int8x8_t _v4s8 = vdup_n_s8(4);
                        int8x8_t _v5s8 = vdup_n_s8(5);
                        int16x8_t _v2 = vdupq_n_s16(2);
                        int16x8_t _v4 = vdupq_n_s16(4);

                        //                         int16x8_t _tmp0m = vfmsq_n_f16(vfmaq_n_f16(_r04, _r00, 4.f), _r02, 5.f);
                        int16x8_t _tmp0m = vsubq_s16(vaddw_s8(vmull_s8(_r00, _v4s8), _r04), vmull_s8(_r02, _v5s8));

                        //                         int16x8_t _tmp1m = vfmsq_n_f16(vaddq_f16(_r04, _r03), vaddq_f16(_r01, _r02), 4.f);
                        int16x8_t _tmp1m = vmlsq_s16(vaddl_s8(_r04, _r03), vaddl_s8(_r01, _r02), _v4);

                        //                         int16x8_t _tmp2m = vfmaq_n_f16(vsubq_f16(_r04, _r03), vsubq_f16(_r01, _r02), 4.f);
                        int16x8_t _tmp2m = vmlaq_s16(vsubl_s8(_r04, _r03), vsubl_s8(_r01, _r02), _v4);

                        //                         int16x8_t _tmp3m = vfmsq_n_f16(vsubq_f16(_r04, _r02), vsubq_f16(_r01, _r03), 2.f);
                        int16x8_t _tmp3m = vmlsq_s16(vsubl_s8(_r04, _r02), vsubl_s8(_r01, _r03), _v2);

                        //                         int16x8_t _tmp4m = vfmaq_n_f16(vsubq_f16(_r04, _r02), vsubq_f16(_r01, _r03), 2.f);
                        int16x8_t _tmp4m = vmlaq_s16(vsubl_s8(_r04, _r02), vsubl_s8(_r01, _r03), _v2);

                        //                         int16x8_t _tmp5m = vfmsq_n_f16(vfmaq_n_f16(_r05, _r01, 4.f), _r03, 5.f);
                        int16x8_t _tmp5m = vsubq_s16(vaddw_s8(vmull_s8(_r01, _v4s8), _r05), vmull_s8(_r03, _v5s8));

                        vst1q_s16(tmp[0][m], _tmp0m);
                        vst1q_s16(tmp[1][m], _tmp1m);
                        vst1q_s16(tmp[2][m], _tmp2m);
                        vst1q_s16(tmp[3][m], _tmp3m);
                        vst1q_s16(tmp[4][m], _tmp4m);
                        vst1q_s16(tmp[5][m], _tmp5m);

                        r0 += w * 8;
                    }

                    short* r0_tm_0 = (short*)img0_tm + (i * w_tm / 6 + j) * 8;
                    short* r0_tm_1 = r0_tm_0 + tiles * 8;
                    short* r0_tm_2 = r0_tm_0 + tiles * 16;
                    short* r0_tm_3 = r0_tm_0 + tiles * 24;
                    short* r0_tm_4 = r0_tm_0 + tiles * 32;
                    short* r0_tm_5 = r0_tm_0 + tiles * 40;

                    for (int m = 0; m < 6; m++)
                    {
                        int16x8_t _tmp00 = vld1q_s16(tmp[m][0]);
                        int16x8_t _tmp01 = vld1q_s16(tmp[m][1]);
                        int16x8_t _tmp02 = vld1q_s16(tmp[m][2]);
                        int16x8_t _tmp03 = vld1q_s16(tmp[m][3]);
                        int16x8_t _tmp04 = vld1q_s16(tmp[m][4]);
                        int16x8_t _tmp05 = vld1q_s16(tmp[m][5]);

                        int16x8_t _v2 = vdupq_n_s16(2);
                        int16x8_t _v4 = vdupq_n_s16(4);
                        int16x8_t _v5 = vdupq_n_s16(5);

                        int16x8_t _r0tm0 = vmlsq_s16(vmlaq_s16(_tmp04, _tmp00, _v4), _tmp02, _v5);
                        int16x8_t _r0tm1 = vmlsq_s16(vaddq_s16(_tmp04, _tmp03), vaddq_s16(_tmp01, _tmp02), _v4);
                        int16x8_t _r0tm2 = vmlaq_s16(vsubq_s16(_tmp04, _tmp03), vsubq_s16(_tmp01, _tmp02), _v4);
                        int16x8_t _r0tm3 = vmlsq_s16(vsubq_s16(_tmp04, _tmp02), vsubq_s16(_tmp01, _tmp03), _v2);
                        int16x8_t _r0tm4 = vmlaq_s16(vsubq_s16(_tmp04, _tmp02), vsubq_s16(_tmp01, _tmp03), _v2);
                        int16x8_t _r0tm5 = vmlsq_s16(vmlaq_s16(_tmp05, _tmp01, _v4), _tmp03, _v5);

                        vst1q_s16(r0_tm_0, _r0tm0);
                        vst1q_s16(r0_tm_1, _r0tm1);
                        vst1q_s16(r0_tm_2, _r0tm2);
                        vst1q_s16(r0_tm_3, _r0tm3);
                        vst1q_s16(r0_tm_4, _r0tm4);
                        vst1q_s16(r0_tm_5, _r0tm5);

                        r0_tm_0 += tiles * 48;
                        r0_tm_1 += tiles * 48;
                        r0_tm_2 += tiles * 48;
                        r0_tm_3 += tiles * 48;
                        r0_tm_4 += tiles * 48;
                        r0_tm_5 += tiles * 48;
                    }
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = h_tm / 6 * w_tm / 6;

        // permute
        //         bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;
#if __aarch64__
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);
#else
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
#if __aarch64__
            for (; i + 11 < tiles; i += 12)
            {
                short* tm2p = tm2.row<short>(i / 12);

                const short* r0 = bottom_blob_tm;

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
                short* tmpptr = tm2.row<short>(i / 12 + (i % 12) / 8);

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
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");

                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
#endif // __aarch64__
            for (; i + 3 < tiles; i += 4)
            {
#if __aarch64__
                short* tmpptr = tm2.row<short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
                short* tmpptr = tm2.row<short>(i / 4);
#endif

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]   \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                        "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d0-d7}         \n"
                        "vstm       %1!, {d0-d7}        \n"
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "q0", "q1", "q2", "q3");
#endif
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
#if __aarch64__
                short* tmpptr = tm2.row<short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
                short* tmpptr = tm2.row<short>(i / 4 + (i % 4) / 2);
#endif

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]   \n"
                        "ld1    {v0.8h, v1.8h}, [%0]    \n"
                        "st1    {v0.8h, v1.8h}, [%1], #32 \n"
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1");
#else
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.s16   {d0-d3}, [%0 :128]  \n"
                        "vst1.s16   {d0-d3}, [%1 :128]! \n"
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "q0", "q1");
#endif
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i < tiles; i++)
            {
#if __aarch64__
                short* tmpptr = tm2.row<short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
                short* tmpptr = tm2.row<short>(i / 4 + (i % 4) / 2 + i % 2);
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
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "v0");
#else
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.s16   {d0-d1}, [%0 :128]  \n"
                        "vst1.s16   {d0-d1}, [%1 :128]! \n"
                        : "=r"(r0),    // %0
                        "=r"(tmpptr) // %1
                        : "0"(r0),
                        "1"(tmpptr)
                        : "memory", "q0");
#endif
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 36, outch, 4u * 4, 4, opt.workspace_allocator);

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 2;

            int* output0_tm = top_blob_tm.channel(p);
            int* output1_tm = top_blob_tm.channel(p + 1);

            const Mat kernel0_tm = kernel_tm.channel(p / 2);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __aarch64__
                for (; i + 11 < tiles; i += 12)
                {
                    const short* r0 = bb2.row<const short>(i / 12);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "ld1    {v0.8h, v1.8h}, [%3], #32   \n" // r01

                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"

                        "ld1    {v4.8h, v5.8h}, [%4], #32   \n" // w01

                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "eor    v12.16b, v12.16b, v12.16b   \n"
                        "eor    v13.16b, v13.16b, v13.16b   \n"

                        "prfm   pldl1keep, [%4, #256]       \n"

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

                        "smlal  v8.4s, v4.4h, v0.h[0]       \n"
                        "smlal2 v20.4s, v4.8h, v0.h[0]      \n"
                        "smlal  v9.4s, v4.4h, v0.h[1]       \n"
                        "smlal2 v21.4s, v4.8h, v0.h[1]      \n"
                        "smlal  v10.4s, v4.4h, v0.h[2]      \n"
                        "smlal2 v22.4s, v4.8h, v0.h[2]      \n"
                        "smlal  v11.4s, v4.4h, v0.h[3]      \n"
                        "smlal2 v23.4s, v4.8h, v0.h[3]      \n"
                        "smlal  v12.4s, v4.4h, v0.h[4]      \n"
                        "smlal2 v24.4s, v4.8h, v0.h[4]      \n"
                        "smlal  v13.4s, v4.4h, v0.h[5]      \n"
                        "smlal2 v25.4s, v4.8h, v0.h[5]      \n"
                        "smlal  v14.4s, v4.4h, v0.h[6]      \n"
                        "smlal2 v26.4s, v4.8h, v0.h[6]      \n"
                        "smlal  v15.4s, v4.4h, v0.h[7]      \n"
                        "smlal2 v27.4s, v4.8h, v0.h[7]      \n"

                        "ld1    {v2.8h, v3.8h}, [%3], #32   \n" // r23

                        "smlal  v16.4s, v4.4h, v1.h[0]      \n"
                        "smlal2 v28.4s, v4.8h, v1.h[0]      \n"
                        "smlal  v17.4s, v4.4h, v1.h[1]      \n"
                        "smlal2 v29.4s, v4.8h, v1.h[1]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal  v18.4s, v4.4h, v1.h[2]      \n"
                        "smlal2 v30.4s, v4.8h, v1.h[2]      \n"
                        "smlal  v19.4s, v4.4h, v1.h[3]      \n"
                        "smlal2 v31.4s, v4.8h, v1.h[3]      \n"

                        "ld1    {v6.8h, v7.8h}, [%4], #32   \n" // w23

                        "smlal  v8.4s, v5.4h, v1.h[4]       \n"
                        "smlal2 v20.4s, v5.8h, v1.h[4]      \n"
                        "smlal  v9.4s, v5.4h, v1.h[5]       \n"
                        "smlal2 v21.4s, v5.8h, v1.h[5]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"

                        "smlal  v10.4s, v5.4h, v1.h[6]      \n"
                        "smlal2 v22.4s, v5.8h, v1.h[6]      \n"
                        "smlal  v11.4s, v5.4h, v1.h[7]      \n"
                        "smlal2 v23.4s, v5.8h, v1.h[7]      \n"
                        "smlal  v12.4s, v5.4h, v2.h[0]      \n"
                        "smlal2 v24.4s, v5.8h, v2.h[0]      \n"
                        "smlal  v13.4s, v5.4h, v2.h[1]      \n"
                        "smlal2 v25.4s, v5.8h, v2.h[1]      \n"
                        "smlal  v14.4s, v5.4h, v2.h[2]      \n"
                        "smlal2 v26.4s, v5.8h, v2.h[2]      \n"
                        "smlal  v15.4s, v5.4h, v2.h[3]      \n"
                        "smlal2 v27.4s, v5.8h, v2.h[3]      \n"
                        "smlal  v16.4s, v5.4h, v2.h[4]      \n"
                        "smlal2 v28.4s, v5.8h, v2.h[4]      \n"
                        "smlal  v17.4s, v5.4h, v2.h[5]      \n"
                        "smlal2 v29.4s, v5.8h, v2.h[5]      \n"
                        "smlal  v18.4s, v5.4h, v2.h[6]      \n"
                        "smlal2 v30.4s, v5.8h, v2.h[6]      \n"
                        "smlal  v19.4s, v5.4h, v2.h[7]      \n"
                        "smlal2 v31.4s, v5.8h, v2.h[7]      \n"

                        "ld1    {v0.8h, v1.8h}, [%3], #32   \n" // r45

                        "smlal  v8.4s, v6.4h, v3.h[0]       \n"
                        "smlal2 v20.4s, v6.8h, v3.h[0]      \n"
                        "smlal  v9.4s, v6.4h, v3.h[1]       \n"
                        "smlal2 v21.4s, v6.8h, v3.h[1]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal  v10.4s, v6.4h, v3.h[2]      \n"
                        "smlal2 v22.4s, v6.8h, v3.h[2]      \n"
                        "smlal  v11.4s, v6.4h, v3.h[3]      \n"
                        "smlal2 v23.4s, v6.8h, v3.h[3]      \n"
                        "smlal  v12.4s, v6.4h, v3.h[4]      \n"
                        "smlal2 v24.4s, v6.8h, v3.h[4]      \n"
                        "smlal  v13.4s, v6.4h, v3.h[5]      \n"
                        "smlal2 v25.4s, v6.8h, v3.h[5]      \n"
                        "smlal  v14.4s, v6.4h, v3.h[6]      \n"
                        "smlal2 v26.4s, v6.8h, v3.h[6]      \n"
                        "smlal  v15.4s, v6.4h, v3.h[7]      \n"
                        "smlal2 v27.4s, v6.8h, v3.h[7]      \n"

                        "smlal  v16.4s, v6.4h, v0.h[0]      \n"
                        "smlal2 v28.4s, v6.8h, v0.h[0]      \n"
                        "smlal  v17.4s, v6.4h, v0.h[1]      \n"
                        "smlal2 v29.4s, v6.8h, v0.h[1]      \n"
                        "smlal  v18.4s, v6.4h, v0.h[2]      \n"
                        "smlal2 v30.4s, v6.8h, v0.h[2]      \n"
                        "smlal  v19.4s, v6.4h, v0.h[3]      \n"
                        "smlal2 v31.4s, v6.8h, v0.h[3]      \n"

                        "ld1    {v4.8h, v5.8h}, [%4], #32   \n" // w45

                        "smlal  v8.4s, v7.4h, v0.h[4]       \n"
                        "smlal2 v20.4s, v7.8h, v0.h[4]      \n"
                        "smlal  v9.4s, v7.4h, v0.h[5]       \n"
                        "smlal2 v21.4s, v7.8h, v0.h[5]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"

                        "smlal  v10.4s, v7.4h, v0.h[6]      \n"
                        "smlal2 v22.4s, v7.8h, v0.h[6]      \n"
                        "smlal  v11.4s, v7.4h, v0.h[7]      \n"
                        "smlal2 v23.4s, v7.8h, v0.h[7]      \n"

                        "ld1    {v2.8h, v3.8h}, [%3], #32   \n" // r67

                        "smlal  v12.4s, v7.4h, v1.h[0]      \n"
                        "smlal2 v24.4s, v7.8h, v1.h[0]      \n"
                        "smlal  v13.4s, v7.4h, v1.h[1]      \n"
                        "smlal2 v25.4s, v7.8h, v1.h[1]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal  v14.4s, v7.4h, v1.h[2]      \n"
                        "smlal2 v26.4s, v7.8h, v1.h[2]      \n"
                        "smlal  v15.4s, v7.4h, v1.h[3]      \n"
                        "smlal2 v27.4s, v7.8h, v1.h[3]      \n"
                        "smlal  v16.4s, v7.4h, v1.h[4]      \n"
                        "smlal2 v28.4s, v7.8h, v1.h[4]      \n"
                        "smlal  v17.4s, v7.4h, v1.h[5]      \n"
                        "smlal2 v29.4s, v7.8h, v1.h[5]      \n"
                        "smlal  v18.4s, v7.4h, v1.h[6]      \n"
                        "smlal2 v30.4s, v7.8h, v1.h[6]      \n"
                        "smlal  v19.4s, v7.4h, v1.h[7]      \n"
                        "smlal2 v31.4s, v7.8h, v1.h[7]      \n"

                        "smlal  v8.4s, v4.4h, v2.h[0]       \n"
                        "smlal2 v20.4s, v4.8h, v2.h[0]      \n"
                        "smlal  v9.4s, v4.4h, v2.h[1]       \n"
                        "smlal2 v21.4s, v4.8h, v2.h[1]      \n"
                        "smlal  v10.4s, v4.4h, v2.h[2]      \n"
                        "smlal2 v22.4s, v4.8h, v2.h[2]      \n"
                        "smlal  v11.4s, v4.4h, v2.h[3]      \n"
                        "smlal2 v23.4s, v4.8h, v2.h[3]      \n"
                        "smlal  v12.4s, v4.4h, v2.h[4]      \n"
                        "smlal2 v24.4s, v4.8h, v2.h[4]      \n"
                        "smlal  v13.4s, v4.4h, v2.h[5]      \n"
                        "smlal2 v25.4s, v4.8h, v2.h[5]      \n"
                        "smlal  v14.4s, v4.4h, v2.h[6]      \n"
                        "smlal2 v26.4s, v4.8h, v2.h[6]      \n"
                        "smlal  v15.4s, v4.4h, v2.h[7]      \n"
                        "smlal2 v27.4s, v4.8h, v2.h[7]      \n"

                        "ld1    {v0.8h, v1.8h}, [%3], #32   \n" // r89

                        "smlal  v16.4s, v4.4h, v3.h[0]      \n"
                        "smlal2 v28.4s, v4.8h, v3.h[0]      \n"
                        "smlal  v17.4s, v4.4h, v3.h[1]      \n"
                        "smlal2 v29.4s, v4.8h, v3.h[1]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal  v18.4s, v4.4h, v3.h[2]      \n"
                        "smlal2 v30.4s, v4.8h, v3.h[2]      \n"
                        "smlal  v19.4s, v4.4h, v3.h[3]      \n"
                        "smlal2 v31.4s, v4.8h, v3.h[3]      \n"

                        "ld1    {v6.8h, v7.8h}, [%4], #32   \n" // w67

                        "smlal  v8.4s, v5.4h, v3.h[4]       \n"
                        "smlal2 v20.4s, v5.8h, v3.h[4]      \n"
                        "smlal  v9.4s, v5.4h, v3.h[5]       \n"
                        "smlal2 v21.4s, v5.8h, v3.h[5]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"

                        "smlal  v10.4s, v5.4h, v3.h[6]      \n"
                        "smlal2 v22.4s, v5.8h, v3.h[6]      \n"
                        "smlal  v11.4s, v5.4h, v3.h[7]      \n"
                        "smlal2 v23.4s, v5.8h, v3.h[7]      \n"

                        "smlal  v12.4s, v5.4h, v0.h[0]      \n"
                        "smlal2 v24.4s, v5.8h, v0.h[0]      \n"
                        "smlal  v13.4s, v5.4h, v0.h[1]      \n"
                        "smlal2 v25.4s, v5.8h, v0.h[1]      \n"
                        "smlal  v14.4s, v5.4h, v0.h[2]      \n"
                        "smlal2 v26.4s, v5.8h, v0.h[2]      \n"
                        "smlal  v15.4s, v5.4h, v0.h[3]      \n"
                        "smlal2 v27.4s, v5.8h, v0.h[3]      \n"
                        "smlal  v16.4s, v5.4h, v0.h[4]      \n"
                        "smlal2 v28.4s, v5.8h, v0.h[4]      \n"
                        "smlal  v17.4s, v5.4h, v0.h[5]      \n"
                        "smlal2 v29.4s, v5.8h, v0.h[5]      \n"
                        "smlal  v18.4s, v5.4h, v0.h[6]      \n"
                        "smlal2 v30.4s, v5.8h, v0.h[6]      \n"
                        "smlal  v19.4s, v5.4h, v0.h[7]      \n"
                        "smlal2 v31.4s, v5.8h, v0.h[7]      \n"

                        "ld1    {v2.8h, v3.8h}, [%3], #32   \n" // r1011

                        "smlal  v8.4s, v6.4h, v1.h[0]       \n"
                        "smlal2 v20.4s, v6.8h, v1.h[0]      \n"
                        "smlal  v9.4s, v6.4h, v1.h[1]       \n"
                        "smlal2 v21.4s, v6.8h, v1.h[1]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal  v10.4s, v6.4h, v1.h[2]      \n"
                        "smlal2 v22.4s, v6.8h, v1.h[2]      \n"
                        "smlal  v11.4s, v6.4h, v1.h[3]      \n"
                        "smlal2 v23.4s, v6.8h, v1.h[3]      \n"
                        "smlal  v12.4s, v6.4h, v1.h[4]      \n"
                        "smlal2 v24.4s, v6.8h, v1.h[4]      \n"
                        "smlal  v13.4s, v6.4h, v1.h[5]      \n"
                        "smlal2 v25.4s, v6.8h, v1.h[5]      \n"
                        "smlal  v14.4s, v6.4h, v1.h[6]      \n"
                        "smlal2 v26.4s, v6.8h, v1.h[6]      \n"
                        "smlal  v15.4s, v6.4h, v1.h[7]      \n"
                        "smlal2 v27.4s, v6.8h, v1.h[7]      \n"
                        "smlal  v16.4s, v6.4h, v2.h[0]      \n"
                        "smlal2 v28.4s, v6.8h, v2.h[0]      \n"
                        "smlal  v17.4s, v6.4h, v2.h[1]      \n"
                        "smlal2 v29.4s, v6.8h, v2.h[1]      \n"
                        "smlal  v18.4s, v6.4h, v2.h[2]      \n"
                        "smlal2 v30.4s, v6.8h, v2.h[2]      \n"
                        "smlal  v19.4s, v6.4h, v2.h[3]      \n"
                        "smlal2 v31.4s, v6.8h, v2.h[3]      \n"

                        "ld1    {v4.8h, v5.8h}, [%4], #32   \n" // w01

                        "smlal  v8.4s, v7.4h, v2.h[4]       \n"
                        "smlal2 v20.4s, v7.8h, v2.h[4]      \n"
                        "smlal  v9.4s, v7.4h, v2.h[5]       \n"
                        "smlal2 v21.4s, v7.8h, v2.h[5]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"

                        "smlal  v10.4s, v7.4h, v2.h[6]      \n"
                        "smlal2 v22.4s, v7.8h, v2.h[6]      \n"
                        "smlal  v11.4s, v7.4h, v2.h[7]      \n"
                        "smlal2 v23.4s, v7.8h, v2.h[7]      \n"

                        "ld1    {v0.8h, v1.8h}, [%3], #32   \n" // r01

                        "smlal  v12.4s, v7.4h, v3.h[0]      \n"
                        "smlal2 v24.4s, v7.8h, v3.h[0]      \n"
                        "smlal  v13.4s, v7.4h, v3.h[1]      \n"
                        "smlal2 v25.4s, v7.8h, v3.h[1]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal  v14.4s, v7.4h, v3.h[2]      \n"
                        "smlal2 v26.4s, v7.8h, v3.h[2]      \n"
                        "smlal  v15.4s, v7.4h, v3.h[3]      \n"
                        "smlal2 v27.4s, v7.8h, v3.h[3]      \n"
                        "smlal  v16.4s, v7.4h, v3.h[4]      \n"
                        "smlal2 v28.4s, v7.8h, v3.h[4]      \n"
                        "smlal  v17.4s, v7.4h, v3.h[5]      \n"
                        "smlal2 v29.4s, v7.8h, v3.h[5]      \n"

                        "subs   %w0, %w0, #1                \n"

                        "smlal  v18.4s, v7.4h, v3.h[6]      \n"
                        "smlal2 v30.4s, v7.8h, v3.h[6]      \n"
                        "smlal  v19.4s, v7.4h, v3.h[7]      \n"
                        "smlal2 v31.4s, v7.8h, v3.h[7]      \n"

                        "bne    0b                          \n"

                        "sub    %3, %3, #32                 \n"
                        "sub    %4, %4, #32                 \n"

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
                        "=r"(k0)          // %4
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(r0),
                        "4"(k0)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    int32x4_t _sum2 = vdupq_n_s32(0);
                    int32x4_t _sum3 = vdupq_n_s32(0);
                    int32x4_t _sum4 = vdupq_n_s32(0);
                    int32x4_t _sum5 = vdupq_n_s32(0);
                    int32x4_t _sum6 = vdupq_n_s32(0);
                    int32x4_t _sum7 = vdupq_n_s32(0);
                    int32x4_t _sum8 = vdupq_n_s32(0);
                    int32x4_t _sum9 = vdupq_n_s32(0);
                    int32x4_t _suma = vdupq_n_s32(0);
                    int32x4_t _sumb = vdupq_n_s32(0);
                    int32x4_t _sumc = vdupq_n_s32(0);
                    int32x4_t _sumd = vdupq_n_s32(0);
                    int32x4_t _sume = vdupq_n_s32(0);
                    int32x4_t _sumf = vdupq_n_s32(0);

                    for (int j = 0; j < nn; j++)
                    {
                        int16x8_t _val0 = vld1q_s16(r0);
                        int16x8_t _val1 = vld1q_s16(r0 + 8);
                        int16x8_t _val2 = vld1q_s16(r0 + 16);
                        int16x8_t _val3 = vld1q_s16(r0 + 24);
                        int16x8_t _val4 = vld1q_s16(r0 + 32);
                        int16x8_t _val5 = vld1q_s16(r0 + 40);
                        int16x8_t _val6 = vld1q_s16(r0 + 48);
                        int16x8_t _val7 = vld1q_s16(r0 + 56);

                        int16x8_t _w0 = vld1q_s16(k0);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val0), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w0), vget_low_s16(_val0), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w0), vget_low_s16(_val0), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w0), vget_low_s16(_val0), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w0), vget_low_s16(_val0), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w0), vget_low_s16(_val0), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w0), vget_low_s16(_val0), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w0), vget_high_s16(_val0), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w0), vget_high_s16(_val0), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w0), vget_high_s16(_val0), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w0), vget_high_s16(_val0), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w0), vget_high_s16(_val0), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w0), vget_high_s16(_val0), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w0), vget_high_s16(_val0), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w0), vget_high_s16(_val0), 3);

                        int16x8_t _w1 = vld1q_s16(k0 + 8);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val1), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val1), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w1), vget_low_s16(_val1), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w1), vget_low_s16(_val1), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w1), vget_low_s16(_val1), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w1), vget_low_s16(_val1), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w1), vget_low_s16(_val1), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w1), vget_low_s16(_val1), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w1), vget_high_s16(_val1), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w1), vget_high_s16(_val1), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w1), vget_high_s16(_val1), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w1), vget_high_s16(_val1), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w1), vget_high_s16(_val1), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w1), vget_high_s16(_val1), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w1), vget_high_s16(_val1), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w1), vget_high_s16(_val1), 3);

                        int16x8_t _w2 = vld1q_s16(k0 + 16);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_low_s16(_val2), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_low_s16(_val2), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w2), vget_low_s16(_val2), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w2), vget_low_s16(_val2), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w2), vget_low_s16(_val2), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w2), vget_low_s16(_val2), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w2), vget_low_s16(_val2), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w2), vget_low_s16(_val2), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w2), vget_high_s16(_val2), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w2), vget_high_s16(_val2), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w2), vget_high_s16(_val2), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w2), vget_high_s16(_val2), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w2), vget_high_s16(_val2), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w2), vget_high_s16(_val2), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w2), vget_high_s16(_val2), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w2), vget_high_s16(_val2), 3);

                        int16x8_t _w3 = vld1q_s16(k0 + 24);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_low_s16(_val3), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_low_s16(_val3), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w3), vget_low_s16(_val3), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w3), vget_low_s16(_val3), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w3), vget_low_s16(_val3), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w3), vget_low_s16(_val3), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w3), vget_low_s16(_val3), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w3), vget_low_s16(_val3), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w3), vget_high_s16(_val3), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w3), vget_high_s16(_val3), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w3), vget_high_s16(_val3), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w3), vget_high_s16(_val3), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w3), vget_high_s16(_val3), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w3), vget_high_s16(_val3), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w3), vget_high_s16(_val3), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w3), vget_high_s16(_val3), 3);

                        int16x8_t _w4 = vld1q_s16(k0 + 32);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w4), vget_low_s16(_val4), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w4), vget_low_s16(_val4), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w4), vget_low_s16(_val4), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w4), vget_low_s16(_val4), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w4), vget_low_s16(_val4), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w4), vget_low_s16(_val4), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w4), vget_low_s16(_val4), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w4), vget_low_s16(_val4), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w4), vget_high_s16(_val4), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w4), vget_high_s16(_val4), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w4), vget_high_s16(_val4), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w4), vget_high_s16(_val4), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w4), vget_high_s16(_val4), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w4), vget_high_s16(_val4), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w4), vget_high_s16(_val4), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w4), vget_high_s16(_val4), 3);

                        int16x8_t _w5 = vld1q_s16(k0 + 40);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w5), vget_low_s16(_val5), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w5), vget_low_s16(_val5), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w5), vget_low_s16(_val5), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w5), vget_low_s16(_val5), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w5), vget_low_s16(_val5), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w5), vget_low_s16(_val5), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w5), vget_low_s16(_val5), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w5), vget_low_s16(_val5), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w5), vget_high_s16(_val5), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w5), vget_high_s16(_val5), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w5), vget_high_s16(_val5), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w5), vget_high_s16(_val5), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w5), vget_high_s16(_val5), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w5), vget_high_s16(_val5), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w5), vget_high_s16(_val5), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w5), vget_high_s16(_val5), 3);

                        int16x8_t _w6 = vld1q_s16(k0 + 48);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w6), vget_low_s16(_val6), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w6), vget_low_s16(_val6), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w6), vget_low_s16(_val6), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w6), vget_low_s16(_val6), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w6), vget_low_s16(_val6), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w6), vget_low_s16(_val6), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w6), vget_low_s16(_val6), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w6), vget_low_s16(_val6), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w6), vget_high_s16(_val6), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w6), vget_high_s16(_val6), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w6), vget_high_s16(_val6), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w6), vget_high_s16(_val6), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w6), vget_high_s16(_val6), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w6), vget_high_s16(_val6), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w6), vget_high_s16(_val6), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w6), vget_high_s16(_val6), 3);

                        int16x8_t _w7 = vld1q_s16(k0 + 56);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w7), vget_low_s16(_val7), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w7), vget_low_s16(_val7), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w7), vget_low_s16(_val7), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w7), vget_low_s16(_val7), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w7), vget_low_s16(_val7), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w7), vget_low_s16(_val7), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w7), vget_low_s16(_val7), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w7), vget_low_s16(_val7), 3);
                        _sum8 = vmlal_lane_s16(_sum8, vget_low_s16(_w7), vget_high_s16(_val7), 0);
                        _sum9 = vmlal_lane_s16(_sum9, vget_high_s16(_w7), vget_high_s16(_val7), 0);
                        _suma = vmlal_lane_s16(_suma, vget_low_s16(_w7), vget_high_s16(_val7), 1);
                        _sumb = vmlal_lane_s16(_sumb, vget_high_s16(_w7), vget_high_s16(_val7), 1);
                        _sumc = vmlal_lane_s16(_sumc, vget_low_s16(_w7), vget_high_s16(_val7), 2);
                        _sumd = vmlal_lane_s16(_sumd, vget_high_s16(_w7), vget_high_s16(_val7), 2);
                        _sume = vmlal_lane_s16(_sume, vget_low_s16(_w7), vget_high_s16(_val7), 3);
                        _sumf = vmlal_lane_s16(_sumf, vget_high_s16(_w7), vget_high_s16(_val7), 3);

                        r0 += 64;
                        k0 += 64;
                    }

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output1_tm, _sum1);
                    vst1q_s32(output0_tm + 4, _sum2);
                    vst1q_s32(output1_tm + 4, _sum3);
                    vst1q_s32(output0_tm + 8, _sum4);
                    vst1q_s32(output1_tm + 8, _sum5);
                    vst1q_s32(output0_tm + 12, _sum6);
                    vst1q_s32(output1_tm + 12, _sum7);
                    vst1q_s32(output0_tm + 16, _sum8);
                    vst1q_s32(output1_tm + 16, _sum9);
                    vst1q_s32(output0_tm + 20, _suma);
                    vst1q_s32(output1_tm + 20, _sumb);
                    vst1q_s32(output0_tm + 24, _sumc);
                    vst1q_s32(output1_tm + 24, _sumd);
                    vst1q_s32(output0_tm + 28, _sume);
                    vst1q_s32(output1_tm + 28, _sumf);
                    output0_tm += 32;
                    output1_tm += 32;
                }
#endif // __aarch64__
                for (; i + 3 < tiles; i += 4)
                {
#if __aarch64__
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
                    const short* r0 = bb2.row<const short>(i / 4);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
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

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val0), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w0), vget_low_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w0), vget_low_s16(_val1), 0);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w0), vget_low_s16(_val2), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w0), vget_low_s16(_val2), 0);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w0), vget_low_s16(_val3), 0);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w0), vget_low_s16(_val3), 0);

                        int16x8_t _w1 = vld1q_s16(k0 + 8);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val0), 1);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w1), vget_low_s16(_val1), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w1), vget_low_s16(_val1), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w1), vget_low_s16(_val2), 1);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w1), vget_low_s16(_val2), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w1), vget_low_s16(_val3), 1);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w1), vget_low_s16(_val3), 1);

                        int16x8_t _w2 = vld1q_s16(k0 + 16);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_low_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_low_s16(_val0), 2);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w2), vget_low_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w2), vget_low_s16(_val1), 2);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w2), vget_low_s16(_val2), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w2), vget_low_s16(_val2), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w2), vget_low_s16(_val3), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w2), vget_low_s16(_val3), 2);

                        int16x8_t _w3 = vld1q_s16(k0 + 24);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_low_s16(_val0), 3);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_low_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w3), vget_low_s16(_val1), 3);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w3), vget_low_s16(_val1), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w3), vget_low_s16(_val2), 3);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w3), vget_low_s16(_val2), 3);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w3), vget_low_s16(_val3), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w3), vget_low_s16(_val3), 3);

                        int16x8_t _w4 = vld1q_s16(k0 + 32);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w4), vget_high_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w4), vget_high_s16(_val0), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w4), vget_high_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w4), vget_high_s16(_val1), 0);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w4), vget_high_s16(_val2), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w4), vget_high_s16(_val2), 0);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w4), vget_high_s16(_val3), 0);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w4), vget_high_s16(_val3), 0);

                        int16x8_t _w5 = vld1q_s16(k0 + 40);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w5), vget_high_s16(_val0), 1);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w5), vget_high_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w5), vget_high_s16(_val1), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w5), vget_high_s16(_val1), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w5), vget_high_s16(_val2), 1);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w5), vget_high_s16(_val2), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w5), vget_high_s16(_val3), 1);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w5), vget_high_s16(_val3), 1);

                        int16x8_t _w6 = vld1q_s16(k0 + 48);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w6), vget_high_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w6), vget_high_s16(_val0), 2);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w6), vget_high_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w6), vget_high_s16(_val1), 2);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w6), vget_high_s16(_val2), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w6), vget_high_s16(_val2), 2);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w6), vget_high_s16(_val3), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w6), vget_high_s16(_val3), 2);

                        int16x8_t _w7 = vld1q_s16(k0 + 56);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w7), vget_high_s16(_val0), 3);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w7), vget_high_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w7), vget_high_s16(_val1), 3);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w7), vget_high_s16(_val1), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w7), vget_high_s16(_val2), 3);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w7), vget_high_s16(_val2), 3);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w7), vget_high_s16(_val3), 3);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w7), vget_high_s16(_val3), 3);

                        r0 += 32;
                        k0 += 64;
                    }

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output1_tm, _sum1);
                    vst1q_s32(output0_tm + 4, _sum2);
                    vst1q_s32(output1_tm + 4, _sum3);
                    vst1q_s32(output0_tm + 8, _sum4);
                    vst1q_s32(output1_tm + 8, _sum5);
                    vst1q_s32(output0_tm + 12, _sum6);
                    vst1q_s32(output1_tm + 12, _sum7);
                    output0_tm += 16;
                    output1_tm += 16;
#else
                    asm volatile(
                        "veor       q8, q8              \n"
                        "veor       q9, q9              \n"
                        "veor       q10, q10            \n"
                        "veor       q11, q11            \n"
                        "veor       q12, q12            \n"
                        "veor       q13, q13            \n"
                        "veor       q14, q14            \n"
                        "veor       q15, q15            \n"

                        "0:                             \n"

                        "pld        [%3, #256]          \n"
                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d0-d7}        \n"

                        "pld        [%4, #256]          \n"
                        "vld1.s16   {d8-d11}, [%4 :128]! \n"

                        "vmlal.s16  q8, d8, d0[0]       \n"
                        "vmlal.s16  q12, d9, d0[0]      \n"
                        "vmlal.s16  q9, d8, d2[0]       \n"
                        "vmlal.s16  q13, d9, d2[0]      \n"
                        "vmlal.s16  q10, d8, d4[0]      \n"
                        "vmlal.s16  q14, d9, d4[0]      \n"
                        "vmlal.s16  q11, d8, d6[0]      \n"
                        "vmlal.s16  q15, d9, d6[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.s16   {d8-d9}, [%4 :128]! \n"

                        "vmlal.s16  q8, d10, d0[1]      \n"
                        "vmlal.s16  q12, d11, d0[1]     \n"
                        "vmlal.s16  q9, d10, d2[1]      \n"
                        "vmlal.s16  q13, d11, d2[1]     \n"
                        "vmlal.s16  q10, d10, d4[1]     \n"
                        "vmlal.s16  q14, d11, d4[1]     \n"
                        "vmlal.s16  q11, d10, d6[1]     \n"
                        "vmlal.s16  q15, d11, d6[1]     \n"

                        "pld        [%4, #128]          \n"
                        "vld1.s16   {d10-d11}, [%4 :128]! \n"

                        "vmlal.s16  q8, d8, d0[2]       \n"
                        "vmlal.s16  q12, d9, d0[2]      \n"
                        "vmlal.s16  q9, d8, d2[2]       \n"
                        "vmlal.s16  q13, d9, d2[2]      \n"
                        "vmlal.s16  q10, d8, d4[2]      \n"
                        "vmlal.s16  q14, d9, d4[2]      \n"
                        "vmlal.s16  q11, d8, d6[2]      \n"
                        "vmlal.s16  q15, d9, d6[2]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.s16   {d8-d9}, [%4 :128]! \n"

                        "vmlal.s16  q8, d10, d0[3]      \n"
                        "vmlal.s16  q12, d11, d0[3]     \n"
                        "vmlal.s16  q9, d10, d2[3]      \n"
                        "vmlal.s16  q13, d11, d2[3]     \n"
                        "vmlal.s16  q10, d10, d4[3]     \n"
                        "vmlal.s16  q14, d11, d4[3]     \n"
                        "vmlal.s16  q11, d10, d6[3]     \n"
                        "vmlal.s16  q15, d11, d6[3]     \n"

                        "pld        [%4, #128]          \n"
                        "vld1.s16   {d10-d11}, [%4 :128]! \n"

                        "vmlal.s16  q8, d8, d1[0]       \n"
                        "vmlal.s16  q12, d9, d1[0]      \n"
                        "vmlal.s16  q9, d8, d3[0]       \n"
                        "vmlal.s16  q13, d9, d3[0]      \n"
                        "vmlal.s16  q10, d8, d5[0]      \n"
                        "vmlal.s16  q14, d9, d5[0]      \n"
                        "vmlal.s16  q11, d8, d7[0]      \n"
                        "vmlal.s16  q15, d9, d7[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.s16   {d8-d9}, [%4 :128]! \n"

                        "vmlal.s16  q8, d10, d1[1]      \n"
                        "vmlal.s16  q12, d11, d1[1]     \n"
                        "vmlal.s16  q9, d10, d3[1]      \n"
                        "vmlal.s16  q13, d11, d3[1]     \n"
                        "vmlal.s16  q10, d10, d5[1]     \n"
                        "vmlal.s16  q14, d11, d5[1]     \n"
                        "vmlal.s16  q11, d10, d7[1]     \n"
                        "vmlal.s16  q15, d11, d7[1]     \n"

                        "pld        [%4, #128]          \n"
                        "vld1.s16   {d10-d11}, [%4 :128]! \n"

                        "vmlal.s16  q8, d8, d1[2]       \n"
                        "vmlal.s16  q12, d9, d1[2]      \n"
                        "vmlal.s16  q9, d8, d3[2]       \n"
                        "vmlal.s16  q13, d9, d3[2]      \n"
                        "vmlal.s16  q10, d8, d5[2]      \n"
                        "vmlal.s16  q14, d9, d5[2]      \n"
                        "vmlal.s16  q11, d8, d7[2]      \n"
                        "vmlal.s16  q15, d9, d7[2]      \n"

                        "subs       %0, %0, #1          \n"

                        "vmlal.s16  q8, d10, d1[3]      \n"
                        "vmlal.s16  q12, d11, d1[3]     \n"
                        "vmlal.s16  q9, d10, d3[3]      \n"
                        "vmlal.s16  q13, d11, d3[3]     \n"
                        "vmlal.s16  q10, d10, d5[3]     \n"
                        "vmlal.s16  q14, d11, d5[3]     \n"
                        "vmlal.s16  q11, d10, d7[3]     \n"
                        "vmlal.s16  q15, d11, d7[3]     \n"

                        "bne        0b                  \n"

                        "vstm       %1!, {d16-d23}      \n"
                        "vstm       %2!, {d24-d31}      \n"

                        : "=r"(nn),
                        "=r"(output0_tm),
                        "=r"(output1_tm),
                        "=r"(r0),
                        "=r"(k0)
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(output1_tm),
                        "3"(r0),
                        "4"(k0)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
                }
                for (; i + 1 < tiles; i += 2)
                {
#if __aarch64__
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    int32x4_t _sum2 = vdupq_n_s32(0);
                    int32x4_t _sum3 = vdupq_n_s32(0);

                    for (int j = 0; j < nn; j++)
                    {
                        int16x8_t _val0 = vld1q_s16(r0);
                        int16x8_t _val1 = vld1q_s16(r0 + 8);

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
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w0), vget_low_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w0), vget_low_s16(_val1), 0);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val0), 1);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w1), vget_low_s16(_val1), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w1), vget_low_s16(_val1), 1);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_low_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_low_s16(_val0), 2);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w2), vget_low_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w2), vget_low_s16(_val1), 2);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_low_s16(_val0), 3);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_low_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w3), vget_low_s16(_val1), 3);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w3), vget_low_s16(_val1), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w4), vget_high_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w4), vget_high_s16(_val0), 0);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w4), vget_high_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w4), vget_high_s16(_val1), 0);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w5), vget_high_s16(_val0), 1);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w5), vget_high_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w5), vget_high_s16(_val1), 1);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w5), vget_high_s16(_val1), 1);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w6), vget_high_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w6), vget_high_s16(_val0), 2);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w6), vget_high_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w6), vget_high_s16(_val1), 2);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w7), vget_high_s16(_val0), 3);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w7), vget_high_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w7), vget_high_s16(_val1), 3);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w7), vget_high_s16(_val1), 3);

                        r0 += 16;
                        k0 += 64;
                    }

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output1_tm, _sum1);
                    vst1q_s32(output0_tm + 4, _sum2);
                    vst1q_s32(output1_tm + 4, _sum3);
                    output0_tm += 8;
                    output1_tm += 8;
                }
                for (; i < tiles; i++)
                {
#if __aarch64__
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2 + i % 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

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

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output1_tm, _sum1);
                    output0_tm += 4;
                    output1_tm += 4;
                }
            }
        }

        remain_outch_start += nn_outch << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            int* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __aarch64__
                for (; i + 11 < tiles; i += 12)
                {
                    const short* r0 = bb2.row<const short>(i / 12);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    asm volatile(
                        "ld1    {v0.8h, v1.8h}, [%2], #32   \n" // r01

                        "eor    v8.16b, v8.16b, v8.16b      \n"
                        "eor    v9.16b, v9.16b, v9.16b      \n"

                        "ld1    {v4.8h, v5.8h}, [%3], #32   \n" // w01

                        "eor    v10.16b, v10.16b, v10.16b   \n"
                        "eor    v11.16b, v11.16b, v11.16b   \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "eor    v12.16b, v12.16b, v12.16b   \n"
                        "eor    v13.16b, v13.16b, v13.16b   \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "eor    v14.16b, v14.16b, v14.16b   \n"
                        "eor    v15.16b, v15.16b, v15.16b   \n"
                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "eor    v17.16b, v17.16b, v17.16b   \n"
                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"

                        "0:                                 \n"

                        "smlal  v8.4s, v4.4h, v0.h[0]       \n"
                        "smlal  v9.4s, v4.4h, v0.h[1]       \n"
                        "smlal  v10.4s, v4.4h, v0.h[2]      \n"
                        "smlal  v11.4s, v4.4h, v0.h[3]      \n"
                        "smlal  v12.4s, v4.4h, v0.h[4]      \n"
                        "smlal  v13.4s, v4.4h, v0.h[5]      \n"
                        "smlal  v14.4s, v4.4h, v0.h[6]      \n"
                        "smlal  v15.4s, v4.4h, v0.h[7]      \n"

                        "ld1    {v2.8h, v3.8h}, [%2], #32   \n" // r23

                        "smlal  v16.4s, v4.4h, v1.h[0]      \n"
                        "smlal  v17.4s, v4.4h, v1.h[1]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "smlal  v18.4s, v4.4h, v1.h[2]      \n"
                        "smlal  v19.4s, v4.4h, v1.h[3]      \n"

                        "smlal2 v8.4s, v4.8h, v1.h[4]       \n"
                        "smlal2 v9.4s, v4.8h, v1.h[5]       \n"
                        "smlal2 v10.4s, v4.8h, v1.h[6]      \n"
                        "smlal2 v11.4s, v4.8h, v1.h[7]      \n"
                        "smlal2 v12.4s, v4.8h, v2.h[0]      \n"
                        "smlal2 v13.4s, v4.8h, v2.h[1]      \n"
                        "smlal2 v14.4s, v4.8h, v2.h[2]      \n"
                        "smlal2 v15.4s, v4.8h, v2.h[3]      \n"
                        "smlal2 v16.4s, v4.8h, v2.h[4]      \n"
                        "smlal2 v17.4s, v4.8h, v2.h[5]      \n"
                        "smlal2 v18.4s, v4.8h, v2.h[6]      \n"
                        "smlal2 v19.4s, v4.8h, v2.h[7]      \n"

                        "ld1    {v0.8h, v1.8h}, [%2], #32   \n" // r45

                        "smlal  v8.4s, v5.4h, v3.h[0]       \n"
                        "smlal  v9.4s, v5.4h, v3.h[1]       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "smlal  v10.4s, v5.4h, v3.h[2]      \n"
                        "smlal  v11.4s, v5.4h, v3.h[3]      \n"
                        "smlal  v12.4s, v5.4h, v3.h[4]      \n"
                        "smlal  v13.4s, v5.4h, v3.h[5]      \n"
                        "smlal  v14.4s, v5.4h, v3.h[6]      \n"
                        "smlal  v15.4s, v5.4h, v3.h[7]      \n"
                        "smlal  v16.4s, v5.4h, v0.h[0]      \n"
                        "smlal  v17.4s, v5.4h, v0.h[1]      \n"
                        "smlal  v18.4s, v5.4h, v0.h[2]      \n"
                        "smlal  v19.4s, v5.4h, v0.h[3]      \n"

                        "ld1    {v6.8h, v7.8h}, [%3], #32   \n" // w23

                        "smlal2 v8.4s, v5.8h, v0.h[4]       \n"
                        "smlal2 v9.4s, v5.8h, v0.h[5]       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal2 v10.4s, v5.8h, v0.h[6]      \n"
                        "smlal2 v11.4s, v5.8h, v0.h[7]      \n"

                        "ld1    {v2.8h, v3.8h}, [%2], #32   \n" // r67

                        "smlal2 v12.4s, v5.8h, v1.h[0]      \n"
                        "smlal2 v13.4s, v5.8h, v1.h[1]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "smlal2 v14.4s, v5.8h, v1.h[2]      \n"
                        "smlal2 v15.4s, v5.8h, v1.h[3]      \n"
                        "smlal2 v16.4s, v5.8h, v1.h[4]      \n"
                        "smlal2 v17.4s, v5.8h, v1.h[5]      \n"
                        "smlal2 v18.4s, v5.8h, v1.h[6]      \n"
                        "smlal2 v19.4s, v5.8h, v1.h[7]      \n"

                        "smlal  v8.4s, v6.4h, v2.h[0]       \n"
                        "smlal  v9.4s, v6.4h, v2.h[1]       \n"
                        "smlal  v10.4s, v6.4h, v2.h[2]      \n"
                        "smlal  v11.4s, v6.4h, v2.h[3]      \n"
                        "smlal  v12.4s, v6.4h, v2.h[4]      \n"
                        "smlal  v13.4s, v6.4h, v2.h[5]      \n"
                        "smlal  v14.4s, v6.4h, v2.h[6]      \n"
                        "smlal  v15.4s, v6.4h, v2.h[7]      \n"

                        "ld1    {v0.8h, v1.8h}, [%2], #32   \n" // r89

                        "smlal  v16.4s, v6.4h, v3.h[0]      \n"
                        "smlal  v17.4s, v6.4h, v3.h[1]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "smlal  v18.4s, v6.4h, v3.h[2]      \n"
                        "smlal  v19.4s, v6.4h, v3.h[3]      \n"

                        "smlal2 v8.4s, v6.8h, v3.h[4]       \n"
                        "smlal2 v9.4s, v6.8h, v3.h[5]       \n"
                        "smlal2 v10.4s, v6.8h, v3.h[6]      \n"
                        "smlal2 v11.4s, v6.8h, v3.h[7]      \n"
                        "smlal2 v12.4s, v6.8h, v0.h[0]      \n"
                        "smlal2 v13.4s, v6.8h, v0.h[1]      \n"
                        "smlal2 v14.4s, v6.8h, v0.h[2]      \n"
                        "smlal2 v15.4s, v6.8h, v0.h[3]      \n"
                        "smlal2 v16.4s, v6.8h, v0.h[4]      \n"
                        "smlal2 v17.4s, v6.8h, v0.h[5]      \n"
                        "smlal2 v18.4s, v6.8h, v0.h[6]      \n"
                        "smlal2 v19.4s, v6.8h, v0.h[7]      \n"

                        "ld1    {v2.8h, v3.8h}, [%2], #32   \n" // r1011

                        "smlal  v8.4s, v7.4h, v1.h[0]       \n"
                        "smlal  v9.4s, v7.4h, v1.h[1]       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "smlal  v10.4s, v7.4h, v1.h[2]      \n"
                        "smlal  v11.4s, v7.4h, v1.h[3]      \n"
                        "smlal  v12.4s, v7.4h, v1.h[4]      \n"
                        "smlal  v13.4s, v7.4h, v1.h[5]      \n"
                        "smlal  v14.4s, v7.4h, v1.h[6]      \n"
                        "smlal  v15.4s, v7.4h, v1.h[7]      \n"
                        "smlal  v16.4s, v7.4h, v2.h[0]      \n"
                        "smlal  v17.4s, v7.4h, v2.h[1]      \n"
                        "smlal  v18.4s, v7.4h, v2.h[2]      \n"
                        "smlal  v19.4s, v7.4h, v2.h[3]      \n"

                        "ld1    {v4.8h, v5.8h}, [%3], #32   \n" // w01

                        "smlal2 v8.4s, v7.8h, v2.h[4]       \n"
                        "smlal2 v9.4s, v7.8h, v2.h[5]       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"

                        "smlal2 v10.4s, v7.8h, v2.h[6]      \n"
                        "smlal2 v11.4s, v7.8h, v2.h[7]      \n"

                        "ld1    {v0.8h, v1.8h}, [%2], #32   \n" // r01

                        "smlal2 v12.4s, v7.8h, v3.h[0]      \n"
                        "smlal2 v13.4s, v7.8h, v3.h[1]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"

                        "smlal2 v14.4s, v7.8h, v3.h[2]      \n"
                        "smlal2 v15.4s, v7.8h, v3.h[3]      \n"
                        "smlal2 v16.4s, v7.8h, v3.h[4]      \n"
                        "smlal2 v17.4s, v7.8h, v3.h[5]      \n"

                        "subs   %w0, %w0, #1                \n"

                        "smlal2 v18.4s, v7.8h, v3.h[6]      \n"
                        "smlal2 v19.4s, v7.8h, v3.h[7]      \n"

                        "bne    0b                          \n"

                        "sub    %2, %2, #32                 \n"
                        "sub    %3, %3, #32                 \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8);
                    const short* k0 = kernel0_tm.row<const short>(r);

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
                        int16x8_t _val4 = vld1q_s16(r0 + 32);
                        int16x8_t _val5 = vld1q_s16(r0 + 40);
                        int16x8_t _val6 = vld1q_s16(r0 + 48);
                        int16x8_t _val7 = vld1q_s16(r0 + 56);

                        int16x8_t _w0 = vld1q_s16(k0);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_w0), vget_low_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w0), vget_low_s16(_val0), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_w0), vget_low_s16(_val0), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w0), vget_high_s16(_val0), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_w0), vget_high_s16(_val0), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w0), vget_high_s16(_val0), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_w0), vget_high_s16(_val0), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_w0), vget_low_s16(_val1), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val1), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_w0), vget_low_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w0), vget_low_s16(_val1), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_w0), vget_high_s16(_val1), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w0), vget_high_s16(_val1), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_w0), vget_high_s16(_val1), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w0), vget_high_s16(_val1), 3);

                        int16x8_t _w1 = vld1q_s16(k0 + 8);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val2), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_w1), vget_low_s16(_val2), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w1), vget_low_s16(_val2), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_w1), vget_low_s16(_val2), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w1), vget_high_s16(_val2), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_w1), vget_high_s16(_val2), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w1), vget_high_s16(_val2), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_w1), vget_high_s16(_val2), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_w1), vget_low_s16(_val3), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val3), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_w1), vget_low_s16(_val3), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w1), vget_low_s16(_val3), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_w1), vget_high_s16(_val3), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w1), vget_high_s16(_val3), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_w1), vget_high_s16(_val3), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w1), vget_high_s16(_val3), 3);

                        int16x8_t _w2 = vld1q_s16(k0 + 16);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_low_s16(_val4), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_w2), vget_low_s16(_val4), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w2), vget_low_s16(_val4), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_w2), vget_low_s16(_val4), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w2), vget_high_s16(_val4), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_w2), vget_high_s16(_val4), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w2), vget_high_s16(_val4), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_w2), vget_high_s16(_val4), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_w2), vget_low_s16(_val5), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_low_s16(_val5), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_w2), vget_low_s16(_val5), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w2), vget_low_s16(_val5), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_w2), vget_high_s16(_val5), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w2), vget_high_s16(_val5), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_w2), vget_high_s16(_val5), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w2), vget_high_s16(_val5), 3);

                        int16x8_t _w3 = vld1q_s16(k0 + 24);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_low_s16(_val6), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_w3), vget_low_s16(_val6), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w3), vget_low_s16(_val6), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_w3), vget_low_s16(_val6), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w3), vget_high_s16(_val6), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_w3), vget_high_s16(_val6), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w3), vget_high_s16(_val6), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_low_s16(_w3), vget_high_s16(_val6), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_high_s16(_w3), vget_low_s16(_val7), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_low_s16(_val7), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_high_s16(_w3), vget_low_s16(_val7), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w3), vget_low_s16(_val7), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_high_s16(_w3), vget_high_s16(_val7), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w3), vget_high_s16(_val7), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_high_s16(_w3), vget_high_s16(_val7), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w3), vget_high_s16(_val7), 3);

                        r0 += 64;
                        k0 += 32;
                    }

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output0_tm + 4, _sum1);
                    vst1q_s32(output0_tm + 8, _sum2);
                    vst1q_s32(output0_tm + 12, _sum3);
                    vst1q_s32(output0_tm + 16, _sum4);
                    vst1q_s32(output0_tm + 20, _sum5);
                    vst1q_s32(output0_tm + 24, _sum6);
                    vst1q_s32(output0_tm + 28, _sum7);
                    output0_tm += 32;
                }
#endif // __aarch64__
                for (; i + 3 < tiles; i += 4)
                {
#if __aarch64__
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
                    const short* r0 = bb2.row<const short>(i / 4);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

#if __aarch64__
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

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w0), vget_low_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w0), vget_low_s16(_val1), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w0), vget_low_s16(_val2), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w0), vget_low_s16(_val2), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w0), vget_low_s16(_val3), 0);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w0), vget_low_s16(_val3), 1);

                        int16x8_t _w1 = vld1q_s16(k0 + 8);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w1), vget_low_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w1), vget_low_s16(_val1), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w1), vget_low_s16(_val2), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w1), vget_low_s16(_val2), 3);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w1), vget_low_s16(_val3), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w1), vget_low_s16(_val3), 3);

                        int16x8_t _w2 = vld1q_s16(k0 + 16);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_high_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_high_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w2), vget_high_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w2), vget_high_s16(_val1), 1);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w2), vget_high_s16(_val2), 0);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w2), vget_high_s16(_val2), 1);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w2), vget_high_s16(_val3), 0);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w2), vget_high_s16(_val3), 1);

                        int16x8_t _w3 = vld1q_s16(k0 + 24);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_high_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_high_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w3), vget_high_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w3), vget_high_s16(_val1), 3);
                        _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_w3), vget_high_s16(_val2), 2);
                        _sum5 = vmlal_lane_s16(_sum5, vget_high_s16(_w3), vget_high_s16(_val2), 3);
                        _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_w3), vget_high_s16(_val3), 2);
                        _sum7 = vmlal_lane_s16(_sum7, vget_high_s16(_w3), vget_high_s16(_val3), 3);

                        r0 += 32;
                        k0 += 32;
                    }

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum2 = vaddq_s32(_sum2, _sum3);
                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum6 = vaddq_s32(_sum6, _sum7);

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output0_tm + 4, _sum2);
                    vst1q_s32(output0_tm + 8, _sum4);
                    vst1q_s32(output0_tm + 12, _sum6);
                    output0_tm += 16;
#else
                    asm volatile(
                        "veor       q8, q8              \n"
                        "veor       q9, q9              \n"
                        "veor       q10, q10            \n"
                        "veor       q11, q11            \n"
                        "veor       q12, q12            \n"
                        "veor       q13, q13            \n"
                        "veor       q14, q14            \n"
                        "veor       q15, q15            \n"

                        "0:                             \n"

                        "pld        [%2, #256]          \n"
                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d0-d7}        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.s16   {d8-d11}, [%3 :128]! \n"

                        "vmlal.s16  q8, d8, d0[0]       \n"
                        "vmlal.s16  q12, d9, d0[1]      \n"
                        "vmlal.s16  q9, d8, d2[0]       \n"
                        "vmlal.s16  q13, d9, d2[1]      \n"
                        "vmlal.s16  q10, d8, d4[0]      \n"
                        "vmlal.s16  q14, d9, d4[1]      \n"
                        "vmlal.s16  q11, d8, d6[0]      \n"
                        "vmlal.s16  q15, d9, d6[1]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.s16   {d8-d9}, [%3 :128]! \n"

                        "vmlal.s16  q8, d10, d0[2]      \n"
                        "vmlal.s16  q12, d11, d0[3]     \n"
                        "vmlal.s16  q9, d10, d2[2]      \n"
                        "vmlal.s16  q13, d11, d2[3]     \n"
                        "vmlal.s16  q10, d10, d4[2]     \n"
                        "vmlal.s16  q14, d11, d4[3]     \n"
                        "vmlal.s16  q11, d10, d6[2]     \n"
                        "vmlal.s16  q15, d11, d6[3]     \n"

                        "pld        [%3, #128]          \n"
                        "vld1.s16   {d10-d11}, [%3 :128]! \n"

                        "vmlal.s16  q8, d8, d1[0]       \n"
                        "vmlal.s16  q12, d9, d1[1]      \n"
                        "vmlal.s16  q9, d8, d3[0]       \n"
                        "vmlal.s16  q13, d9, d3[1]      \n"
                        "vmlal.s16  q10, d8, d5[0]      \n"
                        "vmlal.s16  q14, d9, d5[1]      \n"
                        "vmlal.s16  q11, d8, d7[0]      \n"
                        "vmlal.s16  q15, d9, d7[1]      \n"

                        "subs       %0, %0, #1          \n"

                        "vmlal.s16  q8, d10, d1[2]      \n"
                        "vmlal.s16  q12, d11, d1[3]     \n"
                        "vmlal.s16  q9, d10, d3[2]      \n"
                        "vmlal.s16  q13, d11, d3[3]     \n"
                        "vmlal.s16  q10, d10, d5[2]     \n"
                        "vmlal.s16  q14, d11, d5[3]     \n"
                        "vmlal.s16  q11, d10, d7[2]     \n"
                        "vmlal.s16  q15, d11, d7[3]     \n"

                        "bne        0b                  \n"

                        "vadd.s32   q8, q8, q12         \n"
                        "vadd.s32   q9, q9, q13         \n"
                        "vadd.s32   q10, q10, q14       \n"
                        "vadd.s32   q11, q11, q15       \n"

                        "vstm       %1!, {d16-d23}      \n"

                        : "=r"(nn),
                        "=r"(output0_tm),
                        "=r"(r0),
                        "=r"(k0)
                        : "0"(nn),
                        "1"(output0_tm),
                        "2"(r0),
                        "3"(k0)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
                }
                for (; i + 1 < tiles; i += 2)
                {
#if __aarch64__
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    int32x4_t _sum2 = vdupq_n_s32(0);
                    int32x4_t _sum3 = vdupq_n_s32(0);

                    for (int j = 0; j < nn; j++)
                    {
                        int16x8_t _val0 = vld1q_s16(r0);
                        int16x8_t _val1 = vld1q_s16(r0 + 8);

                        int16x8_t _w0 = vld1q_s16(k0);
                        int16x8_t _w1 = vld1q_s16(k0 + 8);
                        int16x8_t _w2 = vld1q_s16(k0 + 16);
                        int16x8_t _w3 = vld1q_s16(k0 + 24);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w0), vget_low_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w0), vget_low_s16(_val1), 1);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w1), vget_low_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w1), vget_low_s16(_val1), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_high_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_high_s16(_val0), 1);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w2), vget_high_s16(_val1), 0);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w2), vget_high_s16(_val1), 1);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_high_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_high_s16(_val0), 3);
                        _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_w3), vget_high_s16(_val1), 2);
                        _sum3 = vmlal_lane_s16(_sum3, vget_high_s16(_w3), vget_high_s16(_val1), 3);

                        r0 += 16;
                        k0 += 32;
                    }

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum2 = vaddq_s32(_sum2, _sum3);

                    vst1q_s32(output0_tm, _sum0);
                    vst1q_s32(output0_tm + 4, _sum2);
                    output0_tm += 8;
                }
                for (; i < tiles; i++)
                {
#if __aarch64__
                    const short* r0 = bb2.row<const short>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2 + i % 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

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

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0), vget_low_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0), vget_low_s16(_val0), 1);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1), vget_low_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1), vget_low_s16(_val0), 3);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w2), vget_high_s16(_val0), 0);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w2), vget_high_s16(_val0), 1);

                        _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w3), vget_high_s16(_val0), 2);
                        _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w3), vget_high_s16(_val0), 3);

                        r0 += 8;
                        k0 += 32;
                    }

                    _sum0 = vaddq_s32(_sum0, _sum1);

                    vst1q_s32(output0_tm, _sum0);
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
        top_blob_bordered.create(outw, outh, outch, 4u * 4, 4, opt.workspace_allocator);
    }
    {
        // const float otm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 = r00 + (r01 + r02) + (r03 + r04)
        // 1 =       (r01 - r02) + (r03 - r04) * 2
        // 2 =       (r01 + r02) + (r03 + r04) * 4
        // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;
        const int tiles = w_tm / 6 * h_tm / 6;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            int tmp[4][6][4];

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    // top_blob_tm.create(tiles, 36, outch, elemsize, elempack);

                    const int* output0_tm_0 = (const int*)out0_tm + (i * w_tm / 6 + j) * 4;
                    const int* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const int* output0_tm_2 = output0_tm_0 + tiles * 8;
                    const int* output0_tm_3 = output0_tm_0 + tiles * 12;
                    const int* output0_tm_4 = output0_tm_0 + tiles * 16;
                    const int* output0_tm_5 = output0_tm_0 + tiles * 20;

                    int* output0 = out0.row<int>(i * 4) + (j * 4) * 4;

                    // TODO neon optimize
                    for (int m = 0; m < 5; m++)
                    {
                        int32x4_t _out0tm0 = vld1q_s32(output0_tm_0);
                        int32x4_t _out0tm1 = vld1q_s32(output0_tm_1);
                        int32x4_t _out0tm2 = vld1q_s32(output0_tm_2);
                        int32x4_t _out0tm3 = vld1q_s32(output0_tm_3);
                        int32x4_t _out0tm4 = vld1q_s32(output0_tm_4);
                        int32x4_t _out0tm5 = vld1q_s32(output0_tm_5);

                        int32x4_t _tmp02a = vaddq_s32(_out0tm1, _out0tm2);
                        int32x4_t _tmp13a = vsubq_s32(_out0tm1, _out0tm2);

                        int32x4_t _tmp02b = vaddq_s32(_out0tm3, _out0tm4);
                        int32x4_t _tmp13b = vsubq_s32(_out0tm3, _out0tm4);

                        int32x4_t _v2 = vdupq_n_s32(2);
                        int32x4_t _v4 = vdupq_n_s32(4);
                        int32x4_t _v8 = vdupq_n_s32(8);

                        int32x4_t _tmp0m = vaddq_s32(vaddq_s32(_out0tm0, _tmp02a), _tmp02b);
                        int32x4_t _tmp1m = vmlaq_s32(_tmp13a, _tmp13b, _v2);
                        int32x4_t _tmp2m = vmlaq_s32(_tmp02a, _tmp02b, _v4);
                        int32x4_t _tmp3m = vmlaq_s32(vmlaq_s32(_tmp13a, _out0tm5, _v4), _tmp13b, _v8);

                        vst1q_s32(tmp[0][m], _tmp0m);
                        vst1q_s32(tmp[1][m], _tmp1m);
                        vst1q_s32(tmp[2][m], _tmp2m);
                        vst1q_s32(tmp[3][m], _tmp3m);

                        output0_tm_0 += tiles * 24;
                        output0_tm_1 += tiles * 24;
                        output0_tm_2 += tiles * 24;
                        output0_tm_3 += tiles * 24;
                        output0_tm_4 += tiles * 24;
                        output0_tm_5 += tiles * 24;
                    }
                    for (int m = 5; m < 6; m++)
                    {
                        int32x4_t _out0tm0 = vld1q_s32(output0_tm_0);
                        int32x4_t _out0tm1 = vld1q_s32(output0_tm_1);
                        int32x4_t _out0tm2 = vld1q_s32(output0_tm_2);
                        int32x4_t _out0tm3 = vld1q_s32(output0_tm_3);
                        int32x4_t _out0tm4 = vld1q_s32(output0_tm_4);
                        int32x4_t _out0tm5 = vld1q_s32(output0_tm_5);

                        int32x4_t _tmp02a = vaddq_s32(_out0tm1, _out0tm2);
                        int32x4_t _tmp13a = vsubq_s32(_out0tm1, _out0tm2);

                        int32x4_t _tmp02b = vaddq_s32(_out0tm3, _out0tm4);
                        int32x4_t _tmp13b = vsubq_s32(_out0tm3, _out0tm4);

                        int32x4_t _v2 = vdupq_n_s32(2);
                        int32x4_t _v4 = vdupq_n_s32(4);
                        int32x4_t _v8 = vdupq_n_s32(8);

                        int32x4_t _tmp0m = vaddq_s32(vaddq_s32(_out0tm0, _tmp02a), _tmp02b);
                        int32x4_t _tmp1m = vmlaq_s32(_tmp13a, _tmp13b, _v2);
                        int32x4_t _tmp2m = vmlaq_s32(_tmp02a, _tmp02b, _v4);
                        int32x4_t _tmp3m = vmlaq_s32(vmlaq_s32(_tmp13a, _out0tm5, _v4), _tmp13b, _v8);

                        _tmp0m = vmulq_s32(_tmp0m, _v4);
                        _tmp1m = vmulq_s32(_tmp1m, _v4);
                        _tmp2m = vmulq_s32(_tmp2m, _v4);
                        _tmp3m = vmulq_s32(_tmp3m, _v4);

                        vst1q_s32(tmp[0][m], _tmp0m);
                        vst1q_s32(tmp[1][m], _tmp1m);
                        vst1q_s32(tmp[2][m], _tmp2m);
                        vst1q_s32(tmp[3][m], _tmp3m);

                        output0_tm_0 += tiles * 24;
                        output0_tm_1 += tiles * 24;
                        output0_tm_2 += tiles * 24;
                        output0_tm_3 += tiles * 24;
                        output0_tm_4 += tiles * 24;
                        output0_tm_5 += tiles * 24;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        int32x4_t _tmp00 = vld1q_s32(tmp[m][0]);
                        int32x4_t _tmp01 = vld1q_s32(tmp[m][1]);
                        int32x4_t _tmp02 = vld1q_s32(tmp[m][2]);
                        int32x4_t _tmp03 = vld1q_s32(tmp[m][3]);
                        int32x4_t _tmp04 = vld1q_s32(tmp[m][4]);
                        int32x4_t _tmp05 = vld1q_s32(tmp[m][5]);

                        int32x4_t _tmp02a = vaddq_s32(_tmp01, _tmp02);
                        int32x4_t _tmp13a = vsubq_s32(_tmp01, _tmp02);

                        int32x4_t _tmp02b = vaddq_s32(_tmp03, _tmp04);
                        int32x4_t _tmp13b = vsubq_s32(_tmp03, _tmp04);

                        int32x4_t _v2 = vdupq_n_s32(2);
                        int32x4_t _v4 = vdupq_n_s32(4);
                        int32x4_t _v8 = vdupq_n_s32(8);

                        int32x4_t _out00 = vaddq_s32(vaddq_s32(_tmp00, _tmp02a), _tmp02b);
                        int32x4_t _out01 = vmlaq_s32(_tmp13a, _tmp13b, _v2);
                        int32x4_t _out02 = vmlaq_s32(_tmp02a, _tmp02b, _v4);
                        int32x4_t _out03 = vmlaq_s32(vaddq_s32(_tmp05, _tmp13a), _tmp13b, _v8);

                        // TODO use integer trick for division by 576
                        float32x4_t _v576 = vdupq_n_f32(1.0 / 576);
                        _out00 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out00), _v576));
                        _out01 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out01), _v576));
                        _out02 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out02), _v576));
                        _out03 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out03), _v576));

                        vst1q_s32(output0, _out00);
                        vst1q_s32(output0 + 4, _out01);
                        vst1q_s32(output0 + 8, _out02);
                        vst1q_s32(output0 + 12, _out03);

                        output0 += outw * 4;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
