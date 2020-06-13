// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd64_transform_kernel_pack4to1_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
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
    // dst = 4a-inch/4a-64-outch;
#if __aarch64__
    kernel_tm_pack4.create(8 * inch / 4, 64, outch / 8 + (outch % 8) / 4 + outch % 4, (size_t)4u * 4, 4);
#else
    kernel_tm_pack4.create(4 * inch / 4, 64, outch / 4 + outch % 4, (size_t)4u * 4, 4);
#endif

    int p = 0;
#if __aarch64__
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

        Mat g0 = kernel_tm_pack4.channel(p / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 3 < inch; q += 4)
            {
                const float* k00 = k0.row(q);
                const float* k01 = k0.row(q + 1);
                const float* k02 = k0.row(q + 2);
                const float* k03 = k0.row(q + 3);

                const float* k10 = k1.row(q);
                const float* k11 = k1.row(q + 1);
                const float* k12 = k1.row(q + 2);
                const float* k13 = k1.row(q + 3);

                const float* k20 = k2.row(q);
                const float* k21 = k2.row(q + 1);
                const float* k22 = k2.row(q + 2);
                const float* k23 = k2.row(q + 3);

                const float* k30 = k3.row(q);
                const float* k31 = k3.row(q + 1);
                const float* k32 = k3.row(q + 2);
                const float* k33 = k3.row(q + 3);

                const float* k40 = k4.row(q);
                const float* k41 = k4.row(q + 1);
                const float* k42 = k4.row(q + 2);
                const float* k43 = k4.row(q + 3);

                const float* k50 = k5.row(q);
                const float* k51 = k5.row(q + 1);
                const float* k52 = k5.row(q + 2);
                const float* k53 = k5.row(q + 3);

                const float* k60 = k6.row(q);
                const float* k61 = k6.row(q + 1);
                const float* k62 = k6.row(q + 2);
                const float* k63 = k6.row(q + 3);

                const float* k70 = k7.row(q);
                const float* k71 = k7.row(q + 1);
                const float* k72 = k7.row(q + 2);
                const float* k73 = k7.row(q + 3);

                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00[8] = k01[k];
                g00[9] = k11[k];
                g00[10] = k21[k];
                g00[11] = k31[k];
                g00[12] = k41[k];
                g00[13] = k51[k];
                g00[14] = k61[k];
                g00[15] = k71[k];

                g00[16] = k02[k];
                g00[17] = k12[k];
                g00[18] = k22[k];
                g00[19] = k32[k];
                g00[20] = k42[k];
                g00[21] = k52[k];
                g00[22] = k62[k];
                g00[23] = k72[k];

                g00[24] = k03[k];
                g00[25] = k13[k];
                g00[26] = k23[k];
                g00[27] = k33[k];
                g00[28] = k43[k];
                g00[29] = k53[k];
                g00[30] = k63[k];
                g00[31] = k73[k];

                g00 += 32;
            }
        }
    }
#endif // __aarch64__
    for (; p + 3 < outch; p += 4)
    {
        const Mat k0 = kernel_tm.channel(p);
        const Mat k1 = kernel_tm.channel(p + 1);
        const Mat k2 = kernel_tm.channel(p + 2);
        const Mat k3 = kernel_tm.channel(p + 3);

#if __aarch64__
        Mat g0 = kernel_tm_pack4.channel(p / 8 + (p % 8) / 4);
#else
        Mat g0 = kernel_tm_pack4.channel(p / 4);
#endif

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 3 < inch; q += 4)
            {
                const float* k00 = k0.row(q);
                const float* k01 = k0.row(q + 1);
                const float* k02 = k0.row(q + 2);
                const float* k03 = k0.row(q + 3);

                const float* k10 = k1.row(q);
                const float* k11 = k1.row(q + 1);
                const float* k12 = k1.row(q + 2);
                const float* k13 = k1.row(q + 3);

                const float* k20 = k2.row(q);
                const float* k21 = k2.row(q + 1);
                const float* k22 = k2.row(q + 2);
                const float* k23 = k2.row(q + 3);

                const float* k30 = k3.row(q);
                const float* k31 = k3.row(q + 1);
                const float* k32 = k3.row(q + 2);
                const float* k33 = k3.row(q + 3);

                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k01[k];
                g00[5] = k11[k];
                g00[6] = k21[k];
                g00[7] = k31[k];

                g00[8] = k02[k];
                g00[9] = k12[k];
                g00[10] = k22[k];
                g00[11] = k32[k];

                g00[12] = k03[k];
                g00[13] = k13[k];
                g00[14] = k23[k];
                g00[15] = k33[k];

                g00 += 16;
            }
        }
    }
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

#if __aarch64__
        Mat g0 = kernel_tm_pack4.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
        Mat g0 = kernel_tm_pack4.channel(p / 4 + p % 4);
#endif

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 3 < inch; q += 4)
            {
                const float* k00 = k0.row(q);
                const float* k01 = k0.row(q + 1);
                const float* k02 = k0.row(q + 2);
                const float* k03 = k0.row(q + 3);

                g00[0] = k00[k];
                g00[1] = k01[k];
                g00[2] = k02[k];
                g00[3] = k03[k];

                g00 += 4;
            }
        }
    }
}

static void conv3x3s1_winograd64_pack4to1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm / 8 * h_tm / 8;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);

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

            float tmp[8][8][4];

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 4;

                    for (int m = 0; m < 8; m++)
                    {
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r01 = vld1q_f32(r0 + 4);
                        float32x4_t _r02 = vld1q_f32(r0 + 8);
                        float32x4_t _r03 = vld1q_f32(r0 + 12);
                        float32x4_t _r04 = vld1q_f32(r0 + 16);
                        float32x4_t _r05 = vld1q_f32(r0 + 20);
                        float32x4_t _r06 = vld1q_f32(r0 + 24);
                        float32x4_t _r07 = vld1q_f32(r0 + 28);

                        float32x4_t _tmp0m = vmlaq_n_f32(vsubq_f32(_r00, _r06), vsubq_f32(_r04, _r02), 5.25f);
                        float32x4_t _tmp7m = vmlaq_n_f32(vsubq_f32(_r07, _r01), vsubq_f32(_r03, _r05), 5.25f);
                        vst1q_f32(tmp[0][m], _tmp0m);
                        vst1q_f32(tmp[7][m], _tmp7m);

                        //                         tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        //                         tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_r02, _r06), _r04, 4.25f);
                        float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_r01, _r05), _r03, 4.25f);

                        //                         float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        //                         float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        float32x4_t _tmp1m = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _tmp2m = vsubq_f32(_tmp12a, _tmp12b);
                        vst1q_f32(tmp[1][m], _tmp1m);
                        vst1q_f32(tmp[2][m], _tmp2m);

                        //                         tmp[1][m] = tmp12a + tmp12b;
                        //                         tmp[2][m] = tmp12a - tmp12b;

                        float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_r06, _r02, 0.25f), _r04, 1.25f);
                        float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

                        //                         float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        //                         float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        float32x4_t _tmp3m = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _tmp4m = vsubq_f32(_tmp34a, _tmp34b);
                        vst1q_f32(tmp[3][m], _tmp3m);
                        vst1q_f32(tmp[4][m], _tmp4m);

                        //                         tmp[3][m] = tmp34a + tmp34b;
                        //                         tmp[4][m] = tmp34a - tmp34b;

                        float32x4_t _tmp56a = vmlaq_n_f32(_r06, vmlsq_n_f32(_r02, _r04, 1.25f), 4.f);
                        float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

                        //                         float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        //                         float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        float32x4_t _tmp5m = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _tmp6m = vsubq_f32(_tmp56a, _tmp56b);
                        vst1q_f32(tmp[5][m], _tmp5m);
                        vst1q_f32(tmp[6][m], _tmp6m);

                        //                         tmp[5][m] = tmp56a + tmp56b;
                        //                         tmp[6][m] = tmp56a - tmp56b;

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 8 + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 8;
                    float* r0_tm_3 = r0_tm_0 + tiles * 12;
                    float* r0_tm_4 = r0_tm_0 + tiles * 16;
                    float* r0_tm_5 = r0_tm_0 + tiles * 20;
                    float* r0_tm_6 = r0_tm_0 + tiles * 24;
                    float* r0_tm_7 = r0_tm_0 + tiles * 28;

                    for (int m = 0; m < 8; m++)
                    {
                        float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
                        float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
                        float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
                        float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
                        float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
                        float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);
                        float32x4_t _tmp06 = vld1q_f32(tmp[m][6]);
                        float32x4_t _tmp07 = vld1q_f32(tmp[m][7]);

                        float32x4_t _r0tm0 = vmlaq_n_f32(vsubq_f32(_tmp00, _tmp06), vsubq_f32(_tmp04, _tmp02), 5.25f);
                        float32x4_t _r0tm7 = vmlaq_n_f32(vsubq_f32(_tmp07, _tmp01), vsubq_f32(_tmp03, _tmp05), 5.25f);

                        //                         r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        //                         r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_tmp02, _tmp06), _tmp04, 4.25f);
                        float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_tmp01, _tmp05), _tmp03, 4.25f);

                        //                         float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        //                         float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25);

                        float32x4_t _r0tm1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _r0tm2 = vsubq_f32(_tmp12a, _tmp12b);

                        //                         r0_tm[1] = tmp12a + tmp12b;
                        //                         r0_tm[2] = tmp12a - tmp12b;

                        float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                        float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

                        //                         float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        //                         float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        float32x4_t _r0tm3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _r0tm4 = vsubq_f32(_tmp34a, _tmp34b);

                        //                         r0_tm[3] = tmp34a + tmp34b;
                        //                         r0_tm[4] = tmp34a - tmp34b;

                        float32x4_t _tmp56a = vmlaq_n_f32(_tmp06, vmlsq_n_f32(_tmp02, _tmp04, 1.25f), 4.f);
                        float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

                        //                         float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        //                         float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        float32x4_t _r0tm5 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _r0tm6 = vsubq_f32(_tmp56a, _tmp56b);

                        //                         r0_tm[5] = tmp56a + tmp56b;
                        //                         r0_tm[6] = tmp56a - tmp56b;

                        vst1q_f32(r0_tm_0, _r0tm0);
                        vst1q_f32(r0_tm_1, _r0tm1);
                        vst1q_f32(r0_tm_2, _r0tm2);
                        vst1q_f32(r0_tm_3, _r0tm3);
                        vst1q_f32(r0_tm_4, _r0tm4);
                        vst1q_f32(r0_tm_5, _r0tm5);
                        vst1q_f32(r0_tm_6, _r0tm6);
                        vst1q_f32(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 32;
                        r0_tm_1 += tiles * 32;
                        r0_tm_2 += tiles * 32;
                        r0_tm_3 += tiles * 32;
                        r0_tm_4 += tiles * 32;
                        r0_tm_5 += tiles * 32;
                        r0_tm_6 += tiles * 32;
                        r0_tm_7 += tiles * 32;
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
#if __aarch64__
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + tiles % 12 % 4, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);
#else
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);
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
        top_blob_bordered.create(outw, outh, outch, 4u, 1, opt.workspace_allocator);
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

            const float bias0 = bias ? bias[p] : 0.f;
            // float32x2_t _bias0 = vdup_n_f32(bias0);

            float tmp[6][8];

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    //                     top_blob_tm.create(tiles, 64, outch, 4u, 1, opt.workspace_allocator);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 8 + j) * 1;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 1;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 5;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 6;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 7;

                    // TODO neon optimize
                    for (int m = 0; m < 8; m++)
                    {
                        float tmp024a = output0_tm_1[0] + output0_tm_2[0];
                        float tmp135a = output0_tm_1[0] - output0_tm_2[0];

                        float tmp024b = output0_tm_3[0] + output0_tm_4[0];
                        float tmp135b = output0_tm_3[0] - output0_tm_4[0];

                        float tmp024c = output0_tm_5[0] + output0_tm_6[0];
                        float tmp135c = output0_tm_5[0] - output0_tm_6[0];

                        tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm_7[0] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += tiles * 8;
                        output0_tm_1 += tiles * 8;
                        output0_tm_2 += tiles * 8;
                        output0_tm_3 += tiles * 8;
                        output0_tm_4 += tiles * 8;
                        output0_tm_5 += tiles * 8;
                        output0_tm_6 += tiles * 8;
                        output0_tm_7 += tiles * 8;
                    }

                    float* output0 = out0.row(i * 6) + j * 6;

                    for (int m = 0; m < 6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

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

static void conv3x3s1_pack4to1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        out0.fill(bias0);
        out1.fill(bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            float32x4_t _k00_0 = vld1q_f32(k0);
            float32x4_t _k01_0 = vld1q_f32(k0 + 4);
            float32x4_t _k02_0 = vld1q_f32(k0 + 8);
            float32x4_t _k10_0 = vld1q_f32(k0 + 12);
            float32x4_t _k11_0 = vld1q_f32(k0 + 16);
            float32x4_t _k12_0 = vld1q_f32(k0 + 20);
            float32x4_t _k20_0 = vld1q_f32(k0 + 24);
            float32x4_t _k21_0 = vld1q_f32(k0 + 28);
            float32x4_t _k22_0 = vld1q_f32(k0 + 32);

            float32x4_t _k00_1 = vld1q_f32(k1);
            float32x4_t _k01_1 = vld1q_f32(k1 + 4);
            float32x4_t _k02_1 = vld1q_f32(k1 + 8);
            float32x4_t _k10_1 = vld1q_f32(k1 + 12);
            float32x4_t _k11_1 = vld1q_f32(k1 + 16);
            float32x4_t _k12_1 = vld1q_f32(k1 + 20);
            float32x4_t _k20_1 = vld1q_f32(k1 + 24);
            float32x4_t _k21_1 = vld1q_f32(k1 + 28);
            float32x4_t _k22_1 = vld1q_f32(k1 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r00 r01 r02 r03

                        "fmul   v16.4s, %10.4s, v0.4s       \n"
                        "fmul   v17.4s, %19.4s, v0.4s       \n"
                        "fmul   v18.4s, %10.4s, v1.4s       \n"
                        "fmul   v19.4s, %19.4s, v1.4s       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%2]        \n" // r04 r05

                        "fmul   v6.4s, %10.4s, v2.4s        \n"
                        "fmul   v7.4s, %19.4s, v2.4s        \n"
                        "fmul   v8.4s, %10.4s, v3.4s        \n"
                        "fmul   v9.4s, %19.4s, v3.4s        \n"

                        "fmla   v16.4s, %11.4s, v1.4s       \n"
                        "fmla   v17.4s, %20.4s, v1.4s       \n"
                        "fmla   v18.4s, %11.4s, v2.4s       \n"
                        "fmla   v19.4s, %20.4s, v2.4s       \n"
                        "fmla   v6.4s, %11.4s, v3.4s        \n"
                        "fmla   v7.4s, %20.4s, v3.4s        \n"
                        "fmla   v8.4s, %11.4s, v4.4s        \n"
                        "fmla   v9.4s, %20.4s, v4.4s        \n"

                        "fmla   v16.4s, %12.4s, v2.4s       \n"
                        "fmla   v17.4s, %21.4s, v2.4s       \n"
                        "fmla   v18.4s, %12.4s, v3.4s       \n"
                        "fmla   v19.4s, %21.4s, v3.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r10 r11 r12 r12

                        "fmla   v6.4s, %12.4s, v4.4s        \n"
                        "fmla   v7.4s, %21.4s, v4.4s        \n"
                        "fmla   v8.4s, %12.4s, v5.4s        \n"
                        "fmla   v9.4s, %21.4s, v5.4s        \n"

                        "fmla   v16.4s, %13.4s, v0.4s       \n"
                        "fmla   v17.4s, %22.4s, v0.4s       \n"
                        "fmla   v18.4s, %13.4s, v1.4s       \n"
                        "fmla   v19.4s, %22.4s, v1.4s       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%3]        \n" // r14 r15

                        "fmla   v6.4s, %13.4s, v2.4s        \n"
                        "fmla   v7.4s, %22.4s, v2.4s        \n"
                        "fmla   v8.4s, %13.4s, v3.4s        \n"
                        "fmla   v9.4s, %22.4s, v3.4s        \n"

                        "fmla   v16.4s, %14.4s, v1.4s       \n"
                        "fmla   v17.4s, %23.4s, v1.4s       \n"
                        "fmla   v18.4s, %14.4s, v2.4s       \n"
                        "fmla   v19.4s, %23.4s, v2.4s       \n"
                        "fmla   v6.4s, %14.4s, v3.4s        \n"
                        "fmla   v7.4s, %23.4s, v3.4s        \n"
                        "fmla   v8.4s, %14.4s, v4.4s        \n"
                        "fmla   v9.4s, %23.4s, v4.4s        \n"

                        "fmla   v16.4s, %15.4s, v2.4s       \n"
                        "fmla   v17.4s, %24.4s, v2.4s       \n"
                        "fmla   v18.4s, %15.4s, v3.4s       \n"
                        "fmla   v19.4s, %24.4s, v3.4s       \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%4], #64 \n" // r20 r21 r22 r22

                        "fmla   v6.4s, %15.4s, v4.4s        \n"
                        "fmla   v7.4s, %24.4s, v4.4s        \n"
                        "fmla   v8.4s, %15.4s, v5.4s        \n"
                        "fmla   v9.4s, %24.4s, v5.4s        \n"

                        "fmla   v16.4s, %16.4s, v0.4s       \n"
                        "fmla   v17.4s, %25.4s, v0.4s       \n"
                        "fmla   v18.4s, %16.4s, v1.4s       \n"
                        "fmla   v19.4s, %25.4s, v1.4s       \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%4]        \n" // r24 r25

                        "fmla   v6.4s, %16.4s, v2.4s        \n"
                        "fmla   v7.4s, %25.4s, v2.4s        \n"
                        "fmla   v8.4s, %16.4s, v3.4s        \n"
                        "fmla   v9.4s, %25.4s, v3.4s        \n"

                        "fmla   v16.4s, %17.4s, v1.4s       \n"
                        "fmla   v17.4s, %26.4s, v1.4s       \n"
                        "fmla   v18.4s, %17.4s, v2.4s       \n"
                        "fmla   v19.4s, %26.4s, v2.4s       \n"
                        "fmla   v6.4s, %17.4s, v3.4s        \n"
                        "fmla   v7.4s, %26.4s, v3.4s        \n"
                        "fmla   v8.4s, %17.4s, v4.4s        \n"
                        "fmla   v9.4s, %26.4s, v4.4s        \n"

                        "fmla   v16.4s, %18.4s, v2.4s       \n"
                        "fmla   v17.4s, %27.4s, v2.4s       \n"
                        "fmla   v18.4s, %18.4s, v3.4s       \n"
                        "fmla   v19.4s, %27.4s, v3.4s       \n"
                        "fmla   v6.4s, %18.4s, v4.4s        \n"
                        "fmla   v7.4s, %27.4s, v4.4s        \n"
                        "fmla   v8.4s, %18.4s, v5.4s        \n"
                        "fmla   v9.4s, %27.4s, v5.4s        \n"

                        "ld1    {v0.4s}, [%0]               \n" // sum00 sum01 sum02 sum03
                        "ld1    {v1.4s}, [%1]               \n" // sum10 sum11 sum12 sum13

                        "faddp  v16.4s, v16.4s, v16.4s      \n"
                        "faddp  v17.4s, v17.4s, v17.4s      \n"
                        "faddp  v18.4s, v18.4s, v18.4s      \n"
                        "faddp  v19.4s, v19.4s, v19.4s      \n"
                        "faddp  v6.4s, v6.4s, v6.4s         \n"
                        "faddp  v7.4s, v7.4s, v7.4s         \n"
                        "faddp  v8.4s, v8.4s, v8.4s         \n"
                        "faddp  v9.4s, v9.4s, v9.4s         \n"

                        "faddp  v16.2s, v16.2s, v18.2s      \n"
                        "faddp  v17.2s, v17.2s, v19.2s      \n"
                        "faddp  v6.2s, v6.2s, v8.2s         \n"
                        "faddp  v7.2s, v7.2s, v9.2s         \n"

                        "trn1   v16.2d, v16.2d, v6.2d       \n"
                        "trn1   v17.2d, v17.2d, v7.2d       \n"

                        "fadd   v0.4s, v0.4s, v16.4s        \n"
                        "fadd   v1.4s, v1.4s, v17.4s        \n"

                        "st1    {v0.4s}, [%0], #16          \n"
                        "st1    {v1.4s}, [%1], #16          \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_0), // %10
                        "w"(_k01_0), // %11
                        "w"(_k02_0), // %12
                        "w"(_k10_0), // %13
                        "w"(_k11_0), // %14
                        "w"(_k12_0), // %15
                        "w"(_k20_0), // %16
                        "w"(_k21_0), // %17
                        "w"(_k22_0), // %18
                        "w"(_k00_1), // %19
                        "w"(_k01_1), // %20
                        "w"(_k02_1), // %21
                        "w"(_k10_1), // %22
                        "w"(_k11_1), // %23
                        "w"(_k12_1), // %24
                        "w"(_k20_1), // %25
                        "w"(_k21_1), // %26
                        "w"(_k22_1)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2] \n" // r00 r01 r02 r03

                        "fmul   v16.4s, %10.4s, v0.4s       \n"
                        "fmul   v17.4s, %19.4s, v0.4s       \n"
                        "fmul   v18.4s, %10.4s, v1.4s       \n"
                        "fmul   v19.4s, %19.4s, v1.4s       \n"
                        "fmla   v16.4s, %11.4s, v1.4s       \n"
                        "fmla   v17.4s, %20.4s, v1.4s       \n"
                        "fmla   v18.4s, %11.4s, v2.4s       \n"
                        "fmla   v19.4s, %20.4s, v2.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3] \n" // r10 r11 r12 r12

                        "fmla   v16.4s, %12.4s, v2.4s       \n"
                        "fmla   v17.4s, %21.4s, v2.4s       \n"
                        "fmla   v18.4s, %12.4s, v3.4s       \n"
                        "fmla   v19.4s, %21.4s, v3.4s       \n"

                        "fmla   v16.4s, %13.4s, v4.4s       \n"
                        "fmla   v17.4s, %22.4s, v4.4s       \n"
                        "fmla   v18.4s, %13.4s, v5.4s       \n"
                        "fmla   v19.4s, %22.4s, v5.4s       \n"
                        "fmla   v16.4s, %14.4s, v5.4s       \n"
                        "fmla   v17.4s, %23.4s, v5.4s       \n"
                        "fmla   v18.4s, %14.4s, v6.4s       \n"
                        "fmla   v19.4s, %23.4s, v6.4s       \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%4] \n" // r20 r21 r22 r22

                        "fmla   v16.4s, %15.4s, v6.4s       \n"
                        "fmla   v17.4s, %24.4s, v6.4s       \n"
                        "fmla   v18.4s, %15.4s, v7.4s       \n"
                        "fmla   v19.4s, %24.4s, v7.4s       \n"

                        "fmla   v16.4s, %16.4s, v0.4s       \n"
                        "fmla   v17.4s, %25.4s, v0.4s       \n"
                        "fmla   v18.4s, %16.4s, v1.4s       \n"
                        "fmla   v19.4s, %25.4s, v1.4s       \n"
                        "fmla   v16.4s, %17.4s, v1.4s       \n"
                        "fmla   v17.4s, %26.4s, v1.4s       \n"
                        "fmla   v18.4s, %17.4s, v2.4s       \n"
                        "fmla   v19.4s, %26.4s, v2.4s       \n"
                        "fmla   v16.4s, %18.4s, v2.4s       \n"
                        "fmla   v17.4s, %27.4s, v2.4s       \n"
                        "fmla   v18.4s, %18.4s, v3.4s       \n"
                        "fmla   v19.4s, %27.4s, v3.4s       \n"

                        "ld1    {v4.2s}, [%0]               \n" // sum00 sum01
                        "ld1    {v5.2s}, [%1]               \n" // sum10 sum11

                        "faddp  v16.4s, v16.4s, v16.4s      \n"
                        "faddp  v17.4s, v17.4s, v17.4s      \n"
                        "faddp  v18.4s, v18.4s, v18.4s      \n"
                        "faddp  v19.4s, v19.4s, v19.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "faddp  v16.2s, v16.2s, v18.2s      \n"
                        "faddp  v17.2s, v17.2s, v19.2s      \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v4.2s, v4.2s, v16.2s        \n"
                        "fadd   v5.2s, v5.2s, v17.2s        \n"

                        "add    %4, %4, #32                 \n"

                        "st1    {v4.2s}, [%0], #8           \n"
                        "st1    {v5.2s}, [%1], #8           \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_0), // %10
                        "w"(_k01_0), // %11
                        "w"(_k02_0), // %12
                        "w"(_k10_0), // %13
                        "w"(_k11_0), // %14
                        "w"(_k12_0), // %15
                        "w"(_k20_0), // %16
                        "w"(_k21_0), // %17
                        "w"(_k22_0), // %18
                        "w"(_k00_1), // %19
                        "w"(_k01_1), // %20
                        "w"(_k02_1), // %21
                        "w"(_k10_1), // %22
                        "w"(_k11_1), // %23
                        "w"(_k12_1), // %24
                        "w"(_k20_1), // %25
                        "w"(_k21_1), // %26
                        "w"(_k22_1)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%2] \n" // r00 r01 r02

                        "fmul   v16.4s, %10.4s, v0.4s       \n"
                        "fmul   v17.4s, %19.4s, v0.4s       \n"
                        "fmul   v18.4s, %11.4s, v1.4s       \n"
                        "fmul   v19.4s, %20.4s, v1.4s       \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v3.4s, v4.4s, v5.4s}, [%3] \n" // r10 r11 r12

                        "fmla   v16.4s, %12.4s, v2.4s       \n"
                        "fmla   v17.4s, %21.4s, v2.4s       \n"

                        "fmla   v18.4s, %13.4s, v3.4s       \n"
                        "fmla   v19.4s, %22.4s, v3.4s       \n"
                        "fmla   v16.4s, %14.4s, v4.4s       \n"
                        "fmla   v17.4s, %23.4s, v4.4s       \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%4] \n" // r20 r21 r22

                        "fmla   v18.4s, %15.4s, v5.4s       \n"
                        "fmla   v19.4s, %24.4s, v5.4s       \n"

                        "fmla   v16.4s, %16.4s, v0.4s       \n"
                        "fmla   v17.4s, %25.4s, v0.4s       \n"
                        "fmla   v18.4s, %17.4s, v1.4s       \n"
                        "fmla   v19.4s, %26.4s, v1.4s       \n"
                        "fmla   v16.4s, %18.4s, v2.4s       \n"
                        "fmla   v17.4s, %27.4s, v2.4s       \n"

                        "ld1    {v3.s}[0], [%0]             \n" // sum00
                        "ld1    {v4.s}[0], [%1]             \n" // sum10

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "add    %2, %2, #16                 \n"

                        "faddp  v16.4s, v16.4s, v16.4s      \n"
                        "faddp  v17.4s, v17.4s, v17.4s      \n"

                        "add    %3, %3, #16                 \n"

                        "faddp  v16.2s, v16.2s, v16.2s      \n"
                        "faddp  v17.2s, v17.2s, v17.2s      \n"

                        "add    %4, %4, #16                 \n"

                        "fadd   v3.2s, v3.2s, v16.2s        \n"
                        "fadd   v4.2s, v4.2s, v17.2s        \n"

                        "st1    {v3.s}[0], [%0], #4         \n"
                        "st1    {v4.s}[0], [%1], #4         \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_0), // %10
                        "w"(_k01_0), // %11
                        "w"(_k02_0), // %12
                        "w"(_k10_0), // %13
                        "w"(_k11_0), // %14
                        "w"(_k12_0), // %15
                        "w"(_k20_0), // %16
                        "w"(_k21_0), // %17
                        "w"(_k22_0), // %18
                        "w"(_k00_1), // %19
                        "w"(_k01_1), // %20
                        "w"(_k02_1), // %21
                        "w"(_k10_1), // %22
                        "w"(_k11_1), // %23
                        "w"(_k12_1), // %24
                        "w"(_k20_1), // %25
                        "w"(_k21_1), // %26
                        "w"(_k22_1)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19");
                }

                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
            }

            k0 += 9 * 4;
            k1 += 9 * 4;
        }
    }
#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;
        out0.fill(bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k01 = vld1q_f32(k0 + 4);
            float32x4_t _k02 = vld1q_f32(k0 + 8);
            float32x4_t _k10 = vld1q_f32(k0 + 12);
            float32x4_t _k11 = vld1q_f32(k0 + 16);
            float32x4_t _k12 = vld1q_f32(k0 + 20);
            float32x4_t _k20 = vld1q_f32(k0 + 24);
            float32x4_t _k21 = vld1q_f32(k0 + 28);
            float32x4_t _k22 = vld1q_f32(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

#if __aarch64__
                for (; j + 7 < outw; j += 8)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n" // r04 r05 r06 r07

                        "fmul   v16.4s, %8.4s, v0.4s        \n"
                        "fmul   v17.4s, %8.4s, v1.4s        \n"
                        "fmul   v18.4s, %8.4s, v2.4s        \n"
                        "fmul   v19.4s, %8.4s, v3.4s        \n"
                        "fmul   v20.4s, %8.4s, v4.4s        \n"
                        "fmul   v21.4s, %8.4s, v5.4s        \n"
                        "fmul   v22.4s, %8.4s, v6.4s        \n"
                        "fmul   v23.4s, %8.4s, v7.4s        \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%1]        \n" // r08 r09

                        "fmla   v16.4s, %9.4s, v1.4s        \n"
                        "fmla   v17.4s, %9.4s, v2.4s        \n"
                        "fmla   v18.4s, %9.4s, v3.4s        \n"
                        "fmla   v19.4s, %9.4s, v4.4s        \n"
                        "fmla   v20.4s, %9.4s, v5.4s        \n"
                        "fmla   v21.4s, %9.4s, v6.4s        \n"
                        "fmla   v22.4s, %9.4s, v7.4s        \n"
                        "fmla   v23.4s, %9.4s, v8.4s        \n"

                        "fmla   v16.4s, %10.4s, v2.4s       \n"
                        "fmla   v17.4s, %10.4s, v3.4s       \n"
                        "fmla   v18.4s, %10.4s, v4.4s       \n"
                        "fmla   v19.4s, %10.4s, v5.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v20.4s, %10.4s, v6.4s       \n"
                        "fmla   v21.4s, %10.4s, v7.4s       \n"
                        "fmla   v22.4s, %10.4s, v8.4s       \n"
                        "fmla   v23.4s, %10.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r14 r15 r16 r17

                        "fmla   v16.4s, %11.4s, v0.4s       \n"
                        "fmla   v17.4s, %11.4s, v1.4s       \n"
                        "fmla   v18.4s, %11.4s, v2.4s       \n"
                        "fmla   v19.4s, %11.4s, v3.4s       \n"
                        "fmla   v20.4s, %11.4s, v4.4s       \n"
                        "fmla   v21.4s, %11.4s, v5.4s       \n"
                        "fmla   v22.4s, %11.4s, v6.4s       \n"
                        "fmla   v23.4s, %11.4s, v7.4s       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r18 r19

                        "fmla   v16.4s, %12.4s, v1.4s       \n"
                        "fmla   v17.4s, %12.4s, v2.4s       \n"
                        "fmla   v18.4s, %12.4s, v3.4s       \n"
                        "fmla   v19.4s, %12.4s, v4.4s       \n"
                        "fmla   v20.4s, %12.4s, v5.4s       \n"
                        "fmla   v21.4s, %12.4s, v6.4s       \n"
                        "fmla   v22.4s, %12.4s, v7.4s       \n"
                        "fmla   v23.4s, %12.4s, v8.4s       \n"

                        "fmla   v16.4s, %13.4s, v2.4s       \n"
                        "fmla   v17.4s, %13.4s, v3.4s       \n"
                        "fmla   v18.4s, %13.4s, v4.4s       \n"
                        "fmla   v19.4s, %13.4s, v5.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v20.4s, %13.4s, v6.4s       \n"
                        "fmla   v21.4s, %13.4s, v7.4s       \n"
                        "fmla   v22.4s, %13.4s, v8.4s       \n"
                        "fmla   v23.4s, %13.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r24 r25 r26 r27

                        "fmla   v16.4s, %14.4s, v0.4s       \n"
                        "fmla   v17.4s, %14.4s, v1.4s       \n"
                        "fmla   v18.4s, %14.4s, v2.4s       \n"
                        "fmla   v19.4s, %14.4s, v3.4s       \n"
                        "fmla   v20.4s, %14.4s, v4.4s       \n"
                        "fmla   v21.4s, %14.4s, v5.4s       \n"
                        "fmla   v22.4s, %14.4s, v6.4s       \n"
                        "fmla   v23.4s, %14.4s, v7.4s       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r28 r29

                        "fmla   v16.4s, %15.4s, v1.4s       \n"
                        "fmla   v17.4s, %15.4s, v2.4s       \n"
                        "fmla   v18.4s, %15.4s, v3.4s       \n"
                        "fmla   v19.4s, %15.4s, v4.4s       \n"
                        "fmla   v20.4s, %15.4s, v5.4s       \n"
                        "fmla   v21.4s, %15.4s, v6.4s       \n"
                        "fmla   v22.4s, %15.4s, v7.4s       \n"
                        "fmla   v23.4s, %15.4s, v8.4s       \n"

                        "fmla   v16.4s, %16.4s, v2.4s       \n"
                        "fmla   v17.4s, %16.4s, v3.4s       \n"
                        "fmla   v18.4s, %16.4s, v4.4s       \n"
                        "fmla   v19.4s, %16.4s, v5.4s       \n"
                        "fmla   v20.4s, %16.4s, v6.4s       \n"
                        "fmla   v21.4s, %16.4s, v7.4s       \n"
                        "fmla   v22.4s, %16.4s, v8.4s       \n"
                        "fmla   v23.4s, %16.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%0]        \n" // sum0 sum1 sum2 sum3 sum4 sum5 sum6 sum7

                        "faddp  v16.4s, v16.4s, v17.4s      \n"
                        "faddp  v18.4s, v18.4s, v19.4s      \n"
                        "faddp  v20.4s, v20.4s, v21.4s      \n"
                        "faddp  v22.4s, v22.4s, v23.4s      \n"

                        "faddp  v16.4s, v16.4s, v18.4s      \n"
                        "faddp  v20.4s, v20.4s, v22.4s      \n"

                        "fadd   v0.4s, v0.4s, v16.4s        \n"
                        "fadd   v1.4s, v1.4s, v20.4s        \n"

                        "st1    {v0.4s, v1.4s}, [%0], #32   \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                }
#endif // __aarch64__
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%1]        \n" // r04 r05

                        "fmul   v16.4s, %8.4s, v0.4s        \n"
                        "fmul   v17.4s, %8.4s, v1.4s        \n"
                        "fmul   v18.4s, %8.4s, v2.4s        \n"
                        "fmul   v19.4s, %8.4s, v3.4s        \n"

                        "fmla   v16.4s, %9.4s, v1.4s        \n"
                        "fmla   v17.4s, %9.4s, v2.4s        \n"
                        "fmla   v18.4s, %9.4s, v3.4s        \n"
                        "fmla   v19.4s, %9.4s, v8.4s        \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v16.4s, %10.4s, v2.4s       \n"
                        "fmla   v17.4s, %10.4s, v3.4s       \n"
                        "fmla   v18.4s, %10.4s, v8.4s       \n"
                        "fmla   v19.4s, %10.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r14 r15

                        "fmla   v16.4s, %11.4s, v4.4s       \n"
                        "fmla   v17.4s, %11.4s, v5.4s       \n"
                        "fmla   v18.4s, %11.4s, v6.4s       \n"
                        "fmla   v19.4s, %11.4s, v7.4s       \n"

                        "fmla   v16.4s, %12.4s, v5.4s       \n"
                        "fmla   v17.4s, %12.4s, v6.4s       \n"
                        "fmla   v18.4s, %12.4s, v7.4s       \n"
                        "fmla   v19.4s, %12.4s, v8.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v16.4s, %13.4s, v6.4s       \n"
                        "fmla   v17.4s, %13.4s, v7.4s       \n"
                        "fmla   v18.4s, %13.4s, v8.4s       \n"
                        "fmla   v19.4s, %13.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r24 r25

                        "fmla   v16.4s, %14.4s, v0.4s       \n"
                        "fmla   v17.4s, %14.4s, v1.4s       \n"
                        "fmla   v18.4s, %14.4s, v2.4s       \n"
                        "fmla   v19.4s, %14.4s, v3.4s       \n"

                        "fmla   v16.4s, %15.4s, v1.4s       \n"
                        "fmla   v17.4s, %15.4s, v2.4s       \n"
                        "fmla   v18.4s, %15.4s, v3.4s       \n"
                        "fmla   v19.4s, %15.4s, v8.4s       \n"

                        "fmla   v16.4s, %16.4s, v2.4s       \n"
                        "fmla   v17.4s, %16.4s, v3.4s       \n"
                        "fmla   v18.4s, %16.4s, v8.4s       \n"
                        "fmla   v19.4s, %16.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v0.4s}, [%0]               \n" // sum0 sum1 sum2 sum3

                        "faddp  v16.4s, v16.4s, v17.4s      \n"
                        "faddp  v18.4s, v18.4s, v19.4s      \n"

                        "faddp  v16.4s, v16.4s, v18.4s      \n"

                        "fadd   v0.4s, v0.4s, v16.4s        \n"

                        "st1    {v0.4s}, [%0], #16          \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r00 r01

                        "vmul.f32   q3, %q8, q0     \n"

                        "pld        [%1, #128]      \n"
                        "vld1.f32   {d4-d5}, [%1 :128]! \n" // r02

                        "vmul.f32   q4, %q8, q1     \n"
                        "vmla.f32   q3, %q9, q1     \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r03 r04

                        "vmul.f32   q5, %q8, q2     \n"
                        "vmla.f32   q4, %q9, q2     \n"
                        "vmla.f32   q3, %q10, q2    \n"

                        "vmul.f32   q6, %q8, q0     \n"
                        "vmla.f32   q5, %q9, q0     \n"
                        "vmla.f32   q4, %q10, q0    \n"

                        "pld        [%1, #128]      \n"
                        "vld1.f32   {d4-d5}, [%1 :128] \n" // r05

                        "vmla.f32   q6, %q9, q1     \n"
                        "vmla.f32   q5, %q10, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q6, %q10, q2    \n"

                        "vmla.f32   q3, %q11, q0    \n"

                        "pld        [%2, #128]      \n"
                        "vld1.f32   {d4-d5}, [%2 :128]! \n" // r12

                        "vmla.f32   q4, %q11, q1    \n"
                        "vmla.f32   q3, %q12, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r13 r14

                        "vmla.f32   q5, %q11, q2    \n"
                        "vmla.f32   q4, %q12, q2    \n"
                        "vmla.f32   q3, %q13, q2    \n"

                        "vmla.f32   q6, %q11, q0    \n"
                        "vmla.f32   q5, %q12, q0    \n"
                        "vmla.f32   q4, %q13, q0    \n"

                        "pld        [%2, #128]      \n"
                        "vld1.f32   {d4-d5}, [%2 :128] \n" // r15

                        "vmla.f32   q6, %q12, q1    \n"
                        "vmla.f32   q5, %q13, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q6, %q13, q2    \n"

                        "vmla.f32   q3, %q14, q0    \n"

                        "pld        [%3, #128]      \n"
                        "vld1.f32   {d4-d5}, [%3 :128]! \n" // r22

                        "vmla.f32   q4, %q14, q1    \n"
                        "vmla.f32   q3, %q15, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r23 r24

                        "vmla.f32   q5, %q14, q2    \n"
                        "vmla.f32   q4, %q15, q2    \n"
                        "vmla.f32   q3, %q16, q2    \n"

                        "vmla.f32   q6, %q14, q0    \n"
                        "vmla.f32   q5, %q15, q0    \n"
                        "vmla.f32   q4, %q16, q0    \n"

                        "pld        [%3, #128]      \n"
                        "vld1.f32   {d4-d5}, [%3 :128] \n" // r25

                        "vmla.f32   q6, %q15, q1    \n"
                        "vmla.f32   q5, %q16, q1    \n"

                        "vld1.f32   {d0-d1}, [%0]   \n" // sum0 sum1 sum2 sum3

                        "vmla.f32   q6, %q16, q2    \n"

                        "vadd.f32   d6, d6, d7      \n"
                        "vadd.f32   d8, d8, d9      \n"
                        "vadd.f32   d10, d10, d11   \n"
                        "vadd.f32   d12, d12, d13   \n"

                        "sub        %1, %1, #16     \n"

                        "vpadd.f32  d6, d6, d8      \n"
                        "vpadd.f32  d7, d10, d12    \n"

                        "sub        %2, %2, #16     \n"

                        "vadd.f32   q0, q0, q3      \n"

                        "sub        %3, %3, #16     \n"

                        "vst1.f32   {d0-d1}, [%0]!  \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1] \n" // r00 r01 r02 r03

                        "fmul   v16.4s, %8.4s, v0.4s        \n"
                        "fmul   v17.4s, %8.4s, v1.4s        \n"
                        "fmul   v18.4s, %9.4s, v1.4s        \n"
                        "fmul   v19.4s, %9.4s, v2.4s        \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2] \n" // r10 r11 r12 r13

                        "fmla   v16.4s, %10.4s, v2.4s       \n"
                        "fmla   v17.4s, %10.4s, v3.4s       \n"

                        "fmla   v18.4s, %11.4s, v4.4s       \n"
                        "fmla   v19.4s, %11.4s, v5.4s       \n"
                        "fmla   v16.4s, %12.4s, v5.4s       \n"
                        "fmla   v17.4s, %12.4s, v6.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3] \n" // r20 r21 r22 r23

                        "fmla   v18.4s, %13.4s, v6.4s       \n"
                        "fmla   v19.4s, %13.4s, v7.4s       \n"

                        "fmla   v16.4s, %14.4s, v0.4s       \n"
                        "fmla   v17.4s, %14.4s, v1.4s       \n"
                        "fmla   v18.4s, %15.4s, v1.4s       \n"
                        "fmla   v19.4s, %15.4s, v2.4s       \n"
                        "fmla   v16.4s, %16.4s, v2.4s       \n"
                        "fmla   v17.4s, %16.4s, v3.4s       \n"

                        "ld1    {v0.2s}, [%0]               \n" // sum0 sum1

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "add    %1, %1, #32                 \n"

                        "faddp  v16.4s, v16.4s, v17.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "faddp  v16.4s, v16.4s, v16.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v0.2s, v0.2s, v16.2s        \n"

                        "st1    {v0.2s}, [%0], #8           \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r00 r01

                        "vmul.f32   q5, %q8, q0     \n"
                        "vmul.f32   q6, %q8, q1     \n"
                        "vmul.f32   q2, %q9, q1     \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128] \n" // r02 r03

                        "vmul.f32   q3, %q9, q0     \n"
                        "vmla.f32   q5, %q10, q0    \n"
                        "vmla.f32   q6, %q10, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q2, %q11, q0    \n"
                        "vmla.f32   q3, %q11, q1    \n"
                        "vmla.f32   q5, %q12, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128] \n" // r12 r13

                        "vmla.f32   q6, %q12, q0    \n"
                        "vmla.f32   q2, %q13, q0    \n"
                        "vmla.f32   q3, %q13, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q5, %q14, q0    \n"
                        "vmla.f32   q6, %q14, q1    \n"
                        "vmla.f32   q2, %q15, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128] \n" // r22 r23

                        "vmla.f32   q3, %q15, q0    \n"
                        "vmla.f32   q5, %q16, q0    \n"
                        "vmla.f32   q6, %q16, q1    \n"

                        "vld1.f32   {d8}, [%0]      \n" // sum0 sum1

                        "vadd.f32   q5, q5, q2      \n"
                        "vadd.f32   q6, q6, q3      \n"

                        "vadd.f32   d10, d10, d11   \n"
                        "vadd.f32   d12, d12, d13   \n"

                        "vpadd.f32  d10, d10, d12   \n"

                        "vadd.f32   d8, d8, d10     \n"

                        "vst1.f32   {d8}, [%0]!     \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%1] \n" // r00 r01 r02

                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "ld1    {v16.s}[0], [%0]            \n" // sum0

                        "fmul   v17.4s, %8.4s, v0.4s        \n"
                        "fmul   v18.4s, %9.4s, v1.4s        \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.4s, v4.4s, v5.4s}, [%2] \n" // r10 r11 r12

                        "fmla   v16.4s, %10.4s, v2.4s       \n"

                        "fmla   v17.4s, %11.4s, v3.4s       \n"
                        "fmla   v18.4s, %12.4s, v4.4s       \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%3] \n" // r20 r21 r22

                        "fmla   v16.4s, %13.4s, v5.4s       \n"

                        "fmla   v17.4s, %14.4s, v0.4s       \n"
                        "fmla   v18.4s, %15.4s, v1.4s       \n"
                        "fmla   v16.4s, %16.4s, v2.4s       \n"

                        "fadd   v17.4s, v17.4s, v18.4s      \n"
                        "fadd   v16.4s, v16.4s, v17.4s      \n"

                        "add    %1, %1, #16                 \n"

                        "faddp  v16.4s, v16.4s, v16.4s      \n"

                        "add    %2, %2, #16                 \n"

                        "faddp  v16.2s, v16.2s, v16.2s      \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v16.s}[0], [%0], #4        \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #384]      \n"
                        "vldm       %1, {d0-d5}     \n" // r00 r01 r02

                        "veor       q3, q3          \n"
                        "vld1.f32   {d6[0]}, [%0]   \n" // sum0

                        "vmul.f32   q4, %q8, q0     \n"
                        "vmul.f32   q5, %q9, q1     \n"
                        "vmla.f32   q3, %q10, q2    \n"

                        "pld        [%2, #384]      \n"
                        "vldm       %2, {d0-d5}     \n" // r10 r11 r12

                        "vmla.f32   q4, %q11, q0    \n"
                        "vmla.f32   q5, %q12, q1    \n"
                        "vmla.f32   q3, %q13, q2    \n"

                        "pld        [%3, #384]      \n"
                        "vldm       %3, {d0-d5}     \n" // r20 r21 r22

                        "vmla.f32   q4, %q14, q0    \n"
                        "vmla.f32   q5, %q15, q1    \n"
                        "vmla.f32   q3, %q16, q2    \n"

                        "vadd.f32   q4, q4, q5      \n"
                        "vadd.f32   q3, q3, q4      \n"

                        "add        %1, %1, #16     \n"

                        "vadd.f32   d6, d6, d7      \n"

                        "add        %2, %2, #16     \n"

                        "vpadd.f32  d6, d6, d6      \n"

                        "add        %3, %3, #16     \n"

                        "vst1.f32   {d6[0]}, [%0]!  \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5");
#endif // __aarch64__
                }

                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
            }

            k0 += 9 * 4;
        }
    }
}
