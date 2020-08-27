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

static void conv3x3s1_winograd64_transform_kernel_pack8to4_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8to4, int inch, int outch)
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

    #pragma omp parallel for
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
    // dst = 4b-8a-inch/8a-64-outch/4b
    kernel_tm_pack8to4.create(2 * inch / 8, 64, outch / 8 + (outch % 8) / 4, (size_t)2u * 32, 32);

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

        Mat g0 = kernel_tm_pack8to4.channel(p / 8);

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
    for (; p + 3 < outch; p += 4)
    {
        const Mat k0 = kernel_tm.channel(p);
        const Mat k1 = kernel_tm.channel(p + 1);
        const Mat k2 = kernel_tm.channel(p + 2);
        const Mat k3 = kernel_tm.channel(p + 3);

        Mat g0 = kernel_tm_pack8to4.channel(p / 8 + (p % 8) / 4);

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

                    g00 += 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd64_pack8to4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

    const __fp16* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm / 8 * h_tm / 8;

        //         bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        bottom_blob_tm.create(tiles, 64, inch, 2u * elempack, elempack, opt.workspace_allocator);

        //         const float itm[8][8] = {
        //             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
        //
        //             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
        //
        //             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        //         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            __fp16 tmp[8][8][8];

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const __fp16* r0 = img0.row<const __fp16>(i * 6) + (j * 6) * 8;

                    for (int m = 0; m < 8; m++)
                    {
                        float16x8_t _r00 = vld1q_f16(r0);
                        float16x8_t _r01 = vld1q_f16(r0 + 8);
                        float16x8_t _r02 = vld1q_f16(r0 + 16);
                        float16x8_t _r03 = vld1q_f16(r0 + 24);
                        float16x8_t _r04 = vld1q_f16(r0 + 32);
                        float16x8_t _r05 = vld1q_f16(r0 + 40);
                        float16x8_t _r06 = vld1q_f16(r0 + 48);
                        float16x8_t _r07 = vld1q_f16(r0 + 56);

                        float16x8_t _tmp0m = vfmaq_n_f16(vsubq_f16(_r00, _r06), vsubq_f16(_r04, _r02), 5.25f);
                        float16x8_t _tmp7m = vfmaq_n_f16(vsubq_f16(_r07, _r01), vsubq_f16(_r03, _r05), 5.25f);
                        vst1q_f16(tmp[0][m], _tmp0m);
                        vst1q_f16(tmp[7][m], _tmp7m);

                        //                         tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        //                         tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float16x8_t _tmp12a = vfmsq_n_f16(vaddq_f16(_r02, _r06), _r04, 4.25f);
                        float16x8_t _tmp12b = vfmsq_n_f16(vaddq_f16(_r01, _r05), _r03, 4.25f);

                        //                         float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        //                         float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        float16x8_t _tmp1m = vaddq_f16(_tmp12a, _tmp12b);
                        float16x8_t _tmp2m = vsubq_f16(_tmp12a, _tmp12b);
                        vst1q_f16(tmp[1][m], _tmp1m);
                        vst1q_f16(tmp[2][m], _tmp2m);

                        //                         tmp[1][m] = tmp12a + tmp12b;
                        //                         tmp[2][m] = tmp12a - tmp12b;

                        float16x8_t _tmp34a = vfmsq_n_f16(vfmaq_n_f16(_r06, _r02, 0.25f), _r04, 1.25f);
                        float16x8_t _tmp34b = vfmaq_n_f16(vfmsq_n_f16(vmulq_n_f16(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

                        //                         float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        //                         float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        float16x8_t _tmp3m = vaddq_f16(_tmp34a, _tmp34b);
                        float16x8_t _tmp4m = vsubq_f16(_tmp34a, _tmp34b);
                        vst1q_f16(tmp[3][m], _tmp3m);
                        vst1q_f16(tmp[4][m], _tmp4m);

                        //                         tmp[3][m] = tmp34a + tmp34b;
                        //                         tmp[4][m] = tmp34a - tmp34b;

                        float16x8_t _tmp56a = vfmaq_n_f16(_r06, vfmsq_n_f16(_r02, _r04, 1.25f), 4.f);
                        float16x8_t _tmp56b = vfmaq_n_f16(vfmsq_n_f16(vmulq_n_f16(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

                        //                         float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        //                         float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        float16x8_t _tmp5m = vaddq_f16(_tmp56a, _tmp56b);
                        float16x8_t _tmp6m = vsubq_f16(_tmp56a, _tmp56b);
                        vst1q_f16(tmp[5][m], _tmp5m);
                        vst1q_f16(tmp[6][m], _tmp6m);

                        //                         tmp[5][m] = tmp56a + tmp56b;
                        //                         tmp[6][m] = tmp56a - tmp56b;

                        r0 += w * 8;
                    }

                    __fp16* r0_tm_0 = (__fp16*)img0_tm + (i * w_tm / 8 + j) * 8;
                    __fp16* r0_tm_1 = r0_tm_0 + tiles * 8;
                    __fp16* r0_tm_2 = r0_tm_0 + tiles * 16;
                    __fp16* r0_tm_3 = r0_tm_0 + tiles * 24;
                    __fp16* r0_tm_4 = r0_tm_0 + tiles * 32;
                    __fp16* r0_tm_5 = r0_tm_0 + tiles * 40;
                    __fp16* r0_tm_6 = r0_tm_0 + tiles * 48;
                    __fp16* r0_tm_7 = r0_tm_0 + tiles * 56;

                    for (int m = 0; m < 8; m++)
                    {
                        float16x8_t _tmp00 = vld1q_f16(tmp[m][0]);
                        float16x8_t _tmp01 = vld1q_f16(tmp[m][1]);
                        float16x8_t _tmp02 = vld1q_f16(tmp[m][2]);
                        float16x8_t _tmp03 = vld1q_f16(tmp[m][3]);
                        float16x8_t _tmp04 = vld1q_f16(tmp[m][4]);
                        float16x8_t _tmp05 = vld1q_f16(tmp[m][5]);
                        float16x8_t _tmp06 = vld1q_f16(tmp[m][6]);
                        float16x8_t _tmp07 = vld1q_f16(tmp[m][7]);

                        float16x8_t _r0tm0 = vfmaq_n_f16(vsubq_f16(_tmp00, _tmp06), vsubq_f16(_tmp04, _tmp02), 5.25f);
                        float16x8_t _r0tm7 = vfmaq_n_f16(vsubq_f16(_tmp07, _tmp01), vsubq_f16(_tmp03, _tmp05), 5.25f);

                        //                         r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        //                         r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float16x8_t _tmp12a = vfmsq_n_f16(vaddq_f16(_tmp02, _tmp06), _tmp04, 4.25f);
                        float16x8_t _tmp12b = vfmsq_n_f16(vaddq_f16(_tmp01, _tmp05), _tmp03, 4.25f);

                        //                         float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        //                         float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25);

                        float16x8_t _r0tm1 = vaddq_f16(_tmp12a, _tmp12b);
                        float16x8_t _r0tm2 = vsubq_f16(_tmp12a, _tmp12b);

                        //                         r0_tm[1] = tmp12a + tmp12b;
                        //                         r0_tm[2] = tmp12a - tmp12b;

                        float16x8_t _tmp34a = vfmsq_n_f16(vfmaq_n_f16(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                        float16x8_t _tmp34b = vfmaq_n_f16(vfmsq_n_f16(vmulq_n_f16(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

                        //                         float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        //                         float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        float16x8_t _r0tm3 = vaddq_f16(_tmp34a, _tmp34b);
                        float16x8_t _r0tm4 = vsubq_f16(_tmp34a, _tmp34b);

                        //                         r0_tm[3] = tmp34a + tmp34b;
                        //                         r0_tm[4] = tmp34a - tmp34b;

                        float16x8_t _tmp56a = vfmaq_n_f16(_tmp06, vfmsq_n_f16(_tmp02, _tmp04, 1.25f), 4.f);
                        float16x8_t _tmp56b = vfmaq_n_f16(vfmsq_n_f16(vmulq_n_f16(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

                        //                         float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        //                         float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        float16x8_t _r0tm5 = vaddq_f16(_tmp56a, _tmp56b);
                        float16x8_t _r0tm6 = vsubq_f16(_tmp56a, _tmp56b);

                        //                         r0_tm[5] = tmp56a + tmp56b;
                        //                         r0_tm[6] = tmp56a - tmp56b;

                        vst1q_f16(r0_tm_0, _r0tm0);
                        vst1q_f16(r0_tm_1, _r0tm1);
                        vst1q_f16(r0_tm_2, _r0tm2);
                        vst1q_f16(r0_tm_3, _r0tm3);
                        vst1q_f16(r0_tm_4, _r0tm4);
                        vst1q_f16(r0_tm_5, _r0tm5);
                        vst1q_f16(r0_tm_6, _r0tm6);
                        vst1q_f16(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 64;
                        r0_tm_1 += tiles * 64;
                        r0_tm_2 += tiles * 64;
                        r0_tm_3 += tiles * 64;
                        r0_tm_4 += tiles * 64;
                        r0_tm_5 += tiles * 64;
                        r0_tm_6 += tiles * 64;
                        r0_tm_7 += tiles * 64;
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

        top_blob_tm.create(tiles, 64, outch, 2u * 4, 4, opt.workspace_allocator);

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 2;

            __fp16* output0_tm = top_blob_tm.channel(p);
            __fp16* output1_tm = top_blob_tm.channel(p + 1);

            const Mat kernel01_tm = kernel_tm.channel(p / 2);

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

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"

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

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v24.8h, v18.8h, v2.h[0]     \n"
                        "fmla   v25.8h, v18.8h, v2.h[1]     \n"
                        "fmla   v26.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v27.8h, v18.8h, v2.h[3]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v18.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3], #64 \n"

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

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"

                        "fmla   v24.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v25.8h, v16.8h, v0.h[1]     \n"
                        "fmla   v26.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v27.8h, v16.8h, v0.h[3]     \n"
                        "fmla   v24.8h, v17.8h, v0.h[4]     \n"
                        "fmla   v25.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v26.8h, v17.8h, v0.h[6]     \n"
                        "fmla   v27.8h, v17.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v24.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v25.8h, v18.8h, v1.h[1]     \n"
                        "fmla   v26.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v27.8h, v18.8h, v1.h[3]     \n"
                        "fmla   v24.8h, v19.8h, v1.h[4]     \n"
                        "fmla   v25.8h, v19.8h, v1.h[5]     \n"
                        "fmla   v26.8h, v19.8h, v1.h[6]     \n"
                        "fmla   v27.8h, v19.8h, v1.h[7]     \n"

                        "fmla   v24.8h, v20.8h, v2.h[0]     \n"
                        "fmla   v25.8h, v20.8h, v2.h[1]     \n"
                        "fmla   v26.8h, v20.8h, v2.h[2]     \n"
                        "fmla   v27.8h, v20.8h, v2.h[3]     \n"
                        "fmla   v24.8h, v21.8h, v2.h[4]     \n"
                        "fmla   v25.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v26.8h, v21.8h, v2.h[6]     \n"
                        "fmla   v27.8h, v21.8h, v2.h[7]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v24.8h, v22.8h, v3.h[0]     \n"
                        "fmla   v25.8h, v22.8h, v3.h[1]     \n"
                        "fmla   v26.8h, v22.8h, v3.h[2]     \n"
                        "fmla   v27.8h, v22.8h, v3.h[3]     \n"
                        "fmla   v24.8h, v23.8h, v3.h[4]     \n"
                        "fmla   v25.8h, v23.8h, v3.h[5]     \n"
                        "fmla   v26.8h, v23.8h, v3.h[6]     \n"
                        "fmla   v27.8h, v23.8h, v3.h[7]     \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                for (; i < tiles; i++)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + i % 4);

                    const __fp16* kptr = kernel01_tm.row<const __fp16>(r);

                    float16x8_t _sum0 = vdupq_n_f16((__fp16)0.f);

                    for (int q = 0; q < inch; q++)
                    {
                        float16x8_t _r0 = vld1q_f16(r0);

                        float16x8_t _k0 = vld1q_f16(kptr);
                        float16x8_t _k1 = vld1q_f16(kptr + 8);
                        float16x8_t _k2 = vld1q_f16(kptr + 16);
                        float16x8_t _k3 = vld1q_f16(kptr + 24);
                        float16x8_t _k4 = vld1q_f16(kptr + 32);
                        float16x8_t _k5 = vld1q_f16(kptr + 40);
                        float16x8_t _k6 = vld1q_f16(kptr + 48);
                        float16x8_t _k7 = vld1q_f16(kptr + 56);

                        _sum0 = vfmaq_laneq_f16(_sum0, _k0, _r0, 0);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k1, _r0, 1);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k2, _r0, 2);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k3, _r0, 3);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k4, _r0, 4);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k5, _r0, 5);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k6, _r0, 6);
                        _sum0 = vfmaq_laneq_f16(_sum0, _k7, _r0, 7);

                        kptr += 64;
                        r0 += 8;
                    }

                    vst1_f16(output0_tm, vget_low_f16(_sum0));
                    vst1_f16(output1_tm, vget_high_f16(_sum0));

                    output0_tm += 4;
                    output1_tm += 4;
                }
            }
        }

        remain_outch_start += nn_outch << 1;

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

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%3], #32 \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"

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

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%3], #32 \n"

                        "fmla   v24.4h, v18.4h, v2.h[0]     \n"
                        "fmla   v25.4h, v18.4h, v2.h[1]     \n"
                        "fmla   v26.4h, v18.4h, v2.h[2]     \n"
                        "fmla   v27.4h, v18.4h, v2.h[3]     \n"
                        "fmla   v28.4h, v18.4h, v2.h[4]     \n"
                        "fmla   v29.4h, v18.4h, v2.h[5]     \n"
                        "fmla   v30.4h, v18.4h, v2.h[6]     \n"
                        "fmla   v31.4h, v18.4h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
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

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%3], #32 \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"

                        "fmla   v24.4h, v16.4h, v0.h[0]     \n"
                        "fmla   v25.4h, v16.4h, v0.h[1]     \n"
                        "fmla   v26.4h, v16.4h, v0.h[2]     \n"
                        "fmla   v27.4h, v16.4h, v0.h[3]     \n"
                        "fmla   v24.4h, v17.4h, v0.h[4]     \n"
                        "fmla   v25.4h, v17.4h, v0.h[5]     \n"
                        "fmla   v26.4h, v17.4h, v0.h[6]     \n"
                        "fmla   v27.4h, v17.4h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%3], #32 \n"

                        "fmla   v24.4h, v18.4h, v1.h[0]     \n"
                        "fmla   v25.4h, v18.4h, v1.h[1]     \n"
                        "fmla   v26.4h, v18.4h, v1.h[2]     \n"
                        "fmla   v27.4h, v18.4h, v1.h[3]     \n"
                        "fmla   v24.4h, v19.4h, v1.h[4]     \n"
                        "fmla   v25.4h, v19.4h, v1.h[5]     \n"
                        "fmla   v26.4h, v19.4h, v1.h[6]     \n"
                        "fmla   v27.4h, v19.4h, v1.h[7]     \n"

                        "fmla   v24.4h, v20.4h, v2.h[0]     \n"
                        "fmla   v25.4h, v20.4h, v2.h[1]     \n"
                        "fmla   v26.4h, v20.4h, v2.h[2]     \n"
                        "fmla   v27.4h, v20.4h, v2.h[3]     \n"
                        "fmla   v24.4h, v21.4h, v2.h[4]     \n"
                        "fmla   v25.4h, v21.4h, v2.h[5]     \n"
                        "fmla   v26.4h, v21.4h, v2.h[6]     \n"
                        "fmla   v27.4h, v21.4h, v2.h[7]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "fmla   v24.4h, v22.4h, v3.h[0]     \n"
                        "fmla   v25.4h, v22.4h, v3.h[1]     \n"
                        "fmla   v26.4h, v22.4h, v3.h[2]     \n"
                        "fmla   v27.4h, v22.4h, v3.h[3]     \n"
                        "fmla   v24.4h, v23.4h, v3.h[4]     \n"
                        "fmla   v25.4h, v23.4h, v3.h[5]     \n"
                        "fmla   v26.4h, v23.4h, v3.h[6]     \n"
                        "fmla   v27.4h, v23.4h, v3.h[7]     \n"

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
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
                }
                for (; i < tiles; i++)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + i % 4);

                    const __fp16* kptr = kernel0_tm.row<const __fp16>(r);

                    float16x4_t _sum0 = vdup_n_f16((__fp16)0.f);

                    for (int q = 0; q < inch; q++)
                    {
                        float16x8_t _r0 = vld1q_f16(r0);

                        float16x4_t _k0 = vld1_f16(kptr);
                        float16x4_t _k1 = vld1_f16(kptr + 4);
                        float16x4_t _k2 = vld1_f16(kptr + 8);
                        float16x4_t _k3 = vld1_f16(kptr + 12);
                        float16x4_t _k4 = vld1_f16(kptr + 16);
                        float16x4_t _k5 = vld1_f16(kptr + 20);
                        float16x4_t _k6 = vld1_f16(kptr + 24);
                        float16x4_t _k7 = vld1_f16(kptr + 28);

                        _sum0 = vfma_laneq_f16(_sum0, _k0, _r0, 0);
                        _sum0 = vfma_laneq_f16(_sum0, _k1, _r0, 1);
                        _sum0 = vfma_laneq_f16(_sum0, _k2, _r0, 2);
                        _sum0 = vfma_laneq_f16(_sum0, _k3, _r0, 3);
                        _sum0 = vfma_laneq_f16(_sum0, _k4, _r0, 4);
                        _sum0 = vfma_laneq_f16(_sum0, _k5, _r0, 5);
                        _sum0 = vfma_laneq_f16(_sum0, _k6, _r0, 6);
                        _sum0 = vfma_laneq_f16(_sum0, _k7, _r0, 7);

                        kptr += 32;
                        r0 += 8;
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
        //         const float otm[6][8] = {
        //             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
        //         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            //             const float bias0 = bias ? bias[p] : 0.f;
            float16x4_t _bias0 = bias ? vld1_f16((const __fp16*)bias + p * 4) : vdup_n_f16(0.f);

            __fp16 tmp[6][8][4];

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    //                     top_blob_tm.create(tiles, 64, outch, elemsize, elempack);

                    const __fp16* output0_tm_0 = (const __fp16*)out0_tm + (i * w_tm / 8 + j) * 4;
                    const __fp16* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const __fp16* output0_tm_2 = output0_tm_0 + tiles * 8;
                    const __fp16* output0_tm_3 = output0_tm_0 + tiles * 12;
                    const __fp16* output0_tm_4 = output0_tm_0 + tiles * 16;
                    const __fp16* output0_tm_5 = output0_tm_0 + tiles * 20;
                    const __fp16* output0_tm_6 = output0_tm_0 + tiles * 24;
                    const __fp16* output0_tm_7 = output0_tm_0 + tiles * 28;

                    __fp16* output0 = out0.row<__fp16>(i * 6) + (j * 6) * 4;

                    // TODO neon optimize
                    for (int m = 0; m < 8; m++)
                    {
                        float16x4_t _out0tm0 = vld1_f16(output0_tm_0);
                        float16x4_t _out0tm1 = vld1_f16(output0_tm_1);
                        float16x4_t _out0tm2 = vld1_f16(output0_tm_2);
                        float16x4_t _out0tm3 = vld1_f16(output0_tm_3);
                        float16x4_t _out0tm4 = vld1_f16(output0_tm_4);
                        float16x4_t _out0tm5 = vld1_f16(output0_tm_5);
                        float16x4_t _out0tm6 = vld1_f16(output0_tm_6);
                        float16x4_t _out0tm7 = vld1_f16(output0_tm_7);

                        float16x4_t _tmp024a = vadd_f16(_out0tm1, _out0tm2);
                        float16x4_t _tmp135a = vsub_f16(_out0tm1, _out0tm2);

                        //                         float tmp024a = output0_tm[1] + output0_tm[2];
                        //                         float tmp135a = output0_tm[1] - output0_tm[2];

                        float16x4_t _tmp024b = vadd_f16(_out0tm3, _out0tm4);
                        float16x4_t _tmp135b = vsub_f16(_out0tm3, _out0tm4);

                        //                         float tmp024b = output0_tm[3] + output0_tm[4];
                        //                         float tmp135b = output0_tm[3] - output0_tm[4];

                        float16x4_t _tmp024c = vadd_f16(_out0tm5, _out0tm6);
                        float16x4_t _tmp135c = vsub_f16(_out0tm5, _out0tm6);

                        //                         float tmp024c = output0_tm[5] + output0_tm[6];
                        //                         float tmp135c = output0_tm[5] - output0_tm[6];

                        float16x4_t _tmp0m = vadd_f16(vadd_f16(_out0tm0, _tmp024a), vfma_n_f16(_tmp024b, _tmp024c, 32.f));
                        float16x4_t _tmp2m = vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
                        float16x4_t _tmp4m = vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
                        vst1_f16(tmp[0][m], _tmp0m);
                        vst1_f16(tmp[2][m], _tmp2m);
                        vst1_f16(tmp[4][m], _tmp4m);

                        //                         tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
                        //                         tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        //                         tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        float16x4_t _tmp1m = vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
                        float16x4_t _tmp3m = vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
                        float16x4_t _tmp5m = vadd_f16(vadd_f16(_out0tm7, _tmp135a), vfma_n_f16(_tmp135c, _tmp135b, 32.f));
                        vst1_f16(tmp[1][m], _tmp1m);
                        vst1_f16(tmp[3][m], _tmp3m);
                        vst1_f16(tmp[5][m], _tmp5m);

                        //                         tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        //                         tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        //                         tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += tiles * 32;
                        output0_tm_1 += tiles * 32;
                        output0_tm_2 += tiles * 32;
                        output0_tm_3 += tiles * 32;
                        output0_tm_4 += tiles * 32;
                        output0_tm_5 += tiles * 32;
                        output0_tm_6 += tiles * 32;
                        output0_tm_7 += tiles * 32;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        float16x4_t _tmp00 = vld1_f16(tmp[m][0]);
                        float16x4_t _tmp01 = vld1_f16(tmp[m][1]);
                        float16x4_t _tmp02 = vld1_f16(tmp[m][2]);
                        float16x4_t _tmp03 = vld1_f16(tmp[m][3]);
                        float16x4_t _tmp04 = vld1_f16(tmp[m][4]);
                        float16x4_t _tmp05 = vld1_f16(tmp[m][5]);
                        float16x4_t _tmp06 = vld1_f16(tmp[m][6]);
                        float16x4_t _tmp07 = vld1_f16(tmp[m][7]);

                        float16x4_t _tmp024a = vadd_f16(_tmp01, _tmp02);
                        float16x4_t _tmp135a = vsub_f16(_tmp01, _tmp02);

                        //                         float tmp024a = tmp0[1] + tmp0[2];
                        //                         float tmp135a = tmp0[1] - tmp0[2];

                        float16x4_t _tmp024b = vadd_f16(_tmp03, _tmp04);
                        float16x4_t _tmp135b = vsub_f16(_tmp03, _tmp04);

                        //                         float tmp024b = tmp0[3] + tmp0[4];
                        //                         float tmp135b = tmp0[3] - tmp0[4];

                        float16x4_t _tmp024c = vadd_f16(_tmp05, _tmp06);
                        float16x4_t _tmp135c = vsub_f16(_tmp05, _tmp06);

                        //                         float tmp024c = tmp0[5] + tmp0[6];
                        //                         float tmp135c = tmp0[5] - tmp0[6];

                        float16x4_t _out00 = vadd_f16(_bias0, vadd_f16(vadd_f16(_tmp00, _tmp024a), vfma_n_f16(_tmp024b, _tmp024c, 32.f)));
                        float16x4_t _out02 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
                        float16x4_t _out04 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));
                        vst1_f16(output0, _out00);
                        vst1_f16(output0 + 8, _out02);
                        vst1_f16(output0 + 16, _out04);

                        //                         output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        //                         output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        //                         output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        float16x4_t _out01 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
                        float16x4_t _out03 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
                        float16x4_t _out05 = vadd_f16(_bias0, vadd_f16(vadd_f16(_tmp07, _tmp135a), vfma_n_f16(_tmp135c, _tmp135b, 32.f)));
                        vst1_f16(output0 + 4, _out01);
                        vst1_f16(output0 + 12, _out03);
                        vst1_f16(output0 + 20, _out05);

                        //                         output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        //                         output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        //                         output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

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
