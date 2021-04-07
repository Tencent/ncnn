// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd42_transform_kernel_pack8to1_int8_neon(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch)
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
    // dst = 8a-inch/8a-36-outch
    kernel_tm_pack8to1.create(8 * inch / 8, 36, outch / 8 + outch % 8, (size_t)2u * 8, 8);

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

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0.row<const short>(q + i)[k];
                    g00[1] = k1.row<const short>(q + i)[k];
                    g00[2] = k2.row<const short>(q + i)[k];
                    g00[3] = k3.row<const short>(q + i)[k];
                    g00[4] = k4.row<const short>(q + i)[k];
                    g00[5] = k5.row<const short>(q + i)[k];
                    g00[6] = k6.row<const short>(q + i)[k];
                    g00[7] = k7.row<const short>(q + i)[k];

                    g00 += 8;
                }
            }
        }
    }
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

        Mat g0 = kernel_tm_pack8to1.channel(p / 8 + p % 8);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0.row<const short>(q + i)[k];

                    g00 += 1;
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_pack8to1_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
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
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);
#else
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);
#endif // __aarch64__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
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

        top_blob_tm.create(tiles, 36, outch, 4u, 1, opt.workspace_allocator);

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

            for (int r = 0; r < 36; r++)
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

            for (int r = 0; r < 36; r++)
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
        top_blob_bordered.create(outw, outh, outch, 4u, 1, opt.workspace_allocator);
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

            int tmp[4][6];

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    //                     top_blob_tm.create(tiles, 36, outch, 4u, 1, opt.workspace_allocator);

                    const int* output0_tm_0 = (const int*)out0_tm + (i * w_tm / 6 + j) * 1;
                    const int* output0_tm_1 = output0_tm_0 + tiles * 1;
                    const int* output0_tm_2 = output0_tm_0 + tiles * 2;
                    const int* output0_tm_3 = output0_tm_0 + tiles * 3;
                    const int* output0_tm_4 = output0_tm_0 + tiles * 4;
                    const int* output0_tm_5 = output0_tm_0 + tiles * 5;

                    int* output0 = out0.row<int>(i * 4) + j * 4;

                    // 0 = r00 + (r01 + r02) + (r03 + r04)
                    // 1 =       (r01 - r02) + (r03 - r04) * 2
                    // 2 =       (r01 + r02) + (r03 + r04) * 4
                    // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

                    // TODO neon optimize
                    for (int m = 0; m < 5; m++)
                    {
                        int tmp02a = output0_tm_1[0] + output0_tm_2[0];
                        int tmp13a = output0_tm_1[0] - output0_tm_2[0];

                        int tmp02b = output0_tm_3[0] + output0_tm_4[0];
                        int tmp13b = output0_tm_3[0] - output0_tm_4[0];

                        tmp[0][m] = output0_tm_0[0] + tmp02a + tmp02b;
                        tmp[1][m] = tmp13a + tmp13b * 2;
                        tmp[2][m] = tmp02a + tmp02b * 4;
                        tmp[3][m] = output0_tm_5[0] * 4 + tmp13a + tmp13b * 8;

                        output0_tm_0 += tiles * 6;
                        output0_tm_1 += tiles * 6;
                        output0_tm_2 += tiles * 6;
                        output0_tm_3 += tiles * 6;
                        output0_tm_4 += tiles * 6;
                        output0_tm_5 += tiles * 6;
                    }
                    for (int m = 5; m < 6; m++)
                    {
                        int tmp02a = output0_tm_1[0] + output0_tm_2[0];
                        int tmp13a = output0_tm_1[0] - output0_tm_2[0];

                        int tmp02b = output0_tm_3[0] + output0_tm_4[0];
                        int tmp13b = output0_tm_3[0] - output0_tm_4[0];

                        tmp[0][m] = (output0_tm_0[0] + tmp02a + tmp02b) * 4;
                        tmp[1][m] = (tmp13a + tmp13b * 2) * 4;
                        tmp[2][m] = (tmp02a + tmp02b * 4) * 4;
                        tmp[3][m] = (output0_tm_5[0] * 4 + tmp13a + tmp13b * 8) * 4;

                        output0_tm_0 += tiles * 6;
                        output0_tm_1 += tiles * 6;
                        output0_tm_2 += tiles * 6;
                        output0_tm_3 += tiles * 6;
                        output0_tm_4 += tiles * 6;
                        output0_tm_5 += tiles * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        const int* tmp0 = tmp[m];

                        int tmp02a = tmp0[1] + tmp0[2];
                        int tmp13a = tmp0[1] - tmp0[2];

                        int tmp02b = tmp0[3] + tmp0[4];
                        int tmp13b = tmp0[3] - tmp0[4];

                        output0[0] = (tmp0[0] + tmp02a + tmp02b) / 576;
                        output0[1] = (tmp13a + tmp13b * 2) / 576;
                        output0[2] = (tmp02a + tmp02b * 4) / 576;
                        output0[3] = (tmp0[5] + tmp13a + tmp13b * 8) / 576;

                        output0 += outw;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
