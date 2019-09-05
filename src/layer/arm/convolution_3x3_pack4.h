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

static void conv3x3s1_winograd64_transform_kernel_pack4_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
{
    // winograd23 transform kernel
    Mat kernel_tm;
    kernel_tm.create(8*8, inch, outch);

    const float ktm[8][3] = {
        {   1.0f,     0.0f,     0.0f},
        {-2.0f/9,  -2.0f/9,  -2.0f/9},
        {-2.0f/9,   2.0f/9,  -2.0f/9},
        {1.0f/90,  1.0f/45,  2.0f/45},
        {1.0f/90, -1.0f/45,  2.0f/45},
        {1.0f/45,  1.0f/90, 1.0f/180},
        {1.0f/45, -1.0f/90, 1.0f/180},
        {   0.0f,     0.0f,     1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p<outch; p++)
    {
        for (int q = 0; q<inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p*inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i=0; i<8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j=0; j<8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i=0; i<8; i++)
                {
                    kernel_tm0[j*8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 64-inch-outch
    // dst = 4b-4a-inch/4a-64-outch/4b;
    kernel_tm_pack4.create(inch/4, 64, outch/4, (size_t)4u*16, 16);

    for (int q=0; q+3<outch; q+=4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q+1);
        const Mat k2 = kernel_tm.channel(q+2);
        const Mat k3 = kernel_tm.channel(q+3);

        Mat g0 = kernel_tm_pack4.channel(q/4);

        for (int k=0; k<64; k++)
        {
            float* g00 = g0.row(k);

            for (int p=0; p+3<inch; p+=4)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p+1);
                const float* k02 = k0.row(p+2);
                const float* k03 = k0.row(p+3);

                const float* k10 = k1.row(p);
                const float* k11 = k1.row(p+1);
                const float* k12 = k1.row(p+2);
                const float* k13 = k1.row(p+3);

                const float* k20 = k2.row(p);
                const float* k21 = k2.row(p+1);
                const float* k22 = k2.row(p+2);
                const float* k23 = k2.row(p+3);

                const float* k30 = k3.row(p);
                const float* k31 = k3.row(p+1);
                const float* k32 = k3.row(p+2);
                const float* k33 = k3.row(p+3);

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
}

static void conv3x3s1_winograd64_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

        const int tiles = w_tm/8 * h_tm/8;

        bottom_blob_tm.create(8*8, tiles, inch, elemsize, elempack);

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
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8][4];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 4;
                    float* r0_tm = img0_tm.row(i * w_tm/8 + j);

                    for (int m=0; m<8; m++)
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

                    for (int m=0; m<8; m++)
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
                        vst1q_f32(r0_tm, _r0tm0);
                        vst1q_f32(r0_tm + 28, _r0tm7);

//                         r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
//                         r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_tmp02, _tmp06), _tmp04, 4.25f);
                        float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_tmp01, _tmp05), _tmp03, 4.25f);

//                         float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
//                         float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25);

                        float32x4_t _r0tm1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _r0tm2 = vsubq_f32(_tmp12a, _tmp12b);
                        vst1q_f32(r0_tm + 4, _r0tm1);
                        vst1q_f32(r0_tm + 8, _r0tm2);

//                         r0_tm[1] = tmp12a + tmp12b;
//                         r0_tm[2] = tmp12a - tmp12b;

                        float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                        float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

//                         float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
//                         float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        float32x4_t _r0tm3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _r0tm4 = vsubq_f32(_tmp34a, _tmp34b);
                        vst1q_f32(r0_tm + 12, _r0tm3);
                        vst1q_f32(r0_tm + 16, _r0tm4);

//                         r0_tm[3] = tmp34a + tmp34b;
//                         r0_tm[4] = tmp34a - tmp34b;

                        float32x4_t _tmp56a = vmlaq_n_f32(_tmp06, vmlsq_n_f32(_tmp02, _tmp04, 1.25f), 4.f);
                        float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

//                         float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
//                         float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        float32x4_t _r0tm5 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _r0tm6 = vsubq_f32(_tmp56a, _tmp56b);
                        vst1q_f32(r0_tm + 20, _r0tm5);
                        vst1q_f32(r0_tm + 24, _r0tm6);

//                         r0_tm[5] = tmp56a + tmp56b;
//                         r0_tm[6] = tmp56a - tmp56b;

                        r0_tm += 8*4;
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

        const int tiles = h_tm/8 * w_tm/8;

        // permute
//         bottom_blob_tm.create(8*8, tiles, inch, elemsize, elempack);
        Mat bottom_blob_tm2(8 * inch, tiles/8 + (tiles%8)/4 + (tiles%4)/2 + tiles%2, 64, elemsize, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r=0; r<64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i=0;
            for (; i+7<tiles; i+=8)
            {
                float* tm2p = tm2.row(i/8);

                const float* r0 = bottom_blob_tm.channel(0).row(i) + r * 4;
                const float* r1 = bottom_blob_tm.channel(0).row(i+1) + r * 4;
                const float* r2 = bottom_blob_tm.channel(0).row(i+2) + r * 4;
                const float* r3 = bottom_blob_tm.channel(0).row(i+3) + r * 4;
                const float* r4 = bottom_blob_tm.channel(0).row(i+4) + r * 4;
                const float* r5 = bottom_blob_tm.channel(0).row(i+5) + r * 4;
                const float* r6 = bottom_blob_tm.channel(0).row(i+6) + r * 4;
                const float* r7 = bottom_blob_tm.channel(0).row(i+7) + r * 4;

                for (int q=0; q<inch; q++)
                {
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);
                    float32x4_t _r3 = vld1q_f32(r3);
                    float32x4_t _r4 = vld1q_f32(r4);
                    float32x4_t _r5 = vld1q_f32(r5);
                    float32x4_t _r6 = vld1q_f32(r6);
                    float32x4_t _r7 = vld1q_f32(r7);
                    vst1q_f32(tm2p, _r0);
                    vst1q_f32(tm2p + 4, _r1);
                    vst1q_f32(tm2p + 8, _r2);
                    vst1q_f32(tm2p + 12, _r3);
                    vst1q_f32(tm2p + 16, _r4);
                    vst1q_f32(tm2p + 20, _r5);
                    vst1q_f32(tm2p + 24, _r6);
                    vst1q_f32(tm2p + 28, _r7);
//                     tm2p[0] = r0[0];

                    r0 += bottom_blob_tm.cstep * 4;
                    r1 += bottom_blob_tm.cstep * 4;
                    r2 += bottom_blob_tm.cstep * 4;
                    r3 += bottom_blob_tm.cstep * 4;
                    r4 += bottom_blob_tm.cstep * 4;
                    r5 += bottom_blob_tm.cstep * 4;
                    r6 += bottom_blob_tm.cstep * 4;
                    r7 += bottom_blob_tm.cstep * 4;
                    tm2p += 32;
                }
            }
            for (; i+3<tiles; i+=4)
            {
                float* tm2p = tm2.row(i/8 + (i%8)/4);

                const float* r0 = bottom_blob_tm.channel(0).row(i) + r * 4;
                const float* r1 = bottom_blob_tm.channel(0).row(i+1) + r * 4;
                const float* r2 = bottom_blob_tm.channel(0).row(i+2) + r * 4;
                const float* r3 = bottom_blob_tm.channel(0).row(i+3) + r * 4;

                for (int q=0; q<inch; q++)
                {
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);
                    float32x4_t _r3 = vld1q_f32(r3);
                    vst1q_f32(tm2p, _r0);
                    vst1q_f32(tm2p + 4, _r1);
                    vst1q_f32(tm2p + 8, _r2);
                    vst1q_f32(tm2p + 12, _r3);
//                     tm2p[0] = r0[0];

                    r0 += bottom_blob_tm.cstep * 4;
                    r1 += bottom_blob_tm.cstep * 4;
                    r2 += bottom_blob_tm.cstep * 4;
                    r3 += bottom_blob_tm.cstep * 4;
                    tm2p += 16;
                }
            }
            for (; i+1<tiles; i+=2)
            {
                float* tm2p = tm2.row(i/8 + (i%8)/4 + (i%4)/2);

                const float* r0 = bottom_blob_tm.channel(0).row(i) + r * 4;
                const float* r1 = bottom_blob_tm.channel(0).row(i+1) + r * 4;

                for (int q=0; q<inch; q++)
                {
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    vst1q_f32(tm2p, _r0);
                    vst1q_f32(tm2p + 4, _r1);
//                     tm2p[0] = r0[0];

                    r0 += bottom_blob_tm.cstep * 4;
                    r1 += bottom_blob_tm.cstep * 4;
                    tm2p += 8;
                }
            }
            for (; i<tiles; i++)
            {
                float* tm2p = tm2.row(i/8 + (i%8)/4 + (i%4)/2 + i%2);

                const float* r0 = bottom_blob_tm.channel(0).row(i) + r * 4;

                for (int q=0; q<inch; q++)
                {
                    float32x4_t _r0 = vld1q_f32(r0);
                    vst1q_f32(tm2p, _r0);
//                     tm2p[0] = r0[0];

                    r0 += bottom_blob_tm.cstep * 4;
                    tm2p += 4;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(8*8, tiles, outch, elemsize, elempack);

        int nn_outch = 0;
        int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
        nn_outch = outch >> 1;
        remain_outch_start = nn_outch << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 2;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p+1);

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    float* output0_tm0 = out0_tm.row(i);
                    float* output0_tm1 = out0_tm.row(i+1);
                    float* output0_tm2 = out0_tm.row(i+2);
                    float* output0_tm3 = out0_tm.row(i+3);
                    float* output0_tm4 = out0_tm.row(i+4);
                    float* output0_tm5 = out0_tm.row(i+5);
                    float* output0_tm6 = out0_tm.row(i+6);
                    float* output0_tm7 = out0_tm.row(i+7);

                    float* output1_tm0 = out1_tm.row(i);
                    float* output1_tm1 = out1_tm.row(i+1);
                    float* output1_tm2 = out1_tm.row(i+2);
                    float* output1_tm3 = out1_tm.row(i+3);
                    float* output1_tm4 = out1_tm.row(i+4);
                    float* output1_tm5 = out1_tm.row(i+5);
                    float* output1_tm6 = out1_tm.row(i+6);
                    float* output1_tm7 = out1_tm.row(i+7);

                    const float* r0 = bb2.row(i/8);

                    const float* k0 = kernel0_tm.row(r);
                    const float* k1 = kernel1_tm.row(r);

                    float32x4_t _sum0_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum2_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum3_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum4_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum5_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum6_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum7_0 = vdupq_n_f32(0.f);

                    float32x4_t _sum0_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum1_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum3_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum4_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum5_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum6_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum7_1 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );
                        float32x4_t _r1 = vld1q_f32( r0 + 4 );
                        float32x4_t _r2 = vld1q_f32( r0 + 8 );
                        float32x4_t _r3 = vld1q_f32( r0 + 12 );
                        float32x4_t _r4 = vld1q_f32( r0 + 16 );
                        float32x4_t _r5 = vld1q_f32( r0 + 20 );
                        float32x4_t _r6 = vld1q_f32( r0 + 24 );
                        float32x4_t _r7 = vld1q_f32( r0 + 28 );

                        float32x4_t _w0_0 = vld1q_f32( k0 );
                        float32x4_t _w1_0 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2_0 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3_0 = vld1q_f32( k0 + 12 );

                        float32x4_t _w0_1 = vld1q_f32( k1 );
                        float32x4_t _w1_1 = vld1q_f32( k1 + 4 );
                        float32x4_t _w2_1 = vld1q_f32( k1 + 8 );
                        float32x4_t _w3_1 = vld1q_f32( k1 + 12 );

                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w0_0, _r1, 0);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w1_0, _r1, 1);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w2_0, _r1, 2);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w3_0, _r1, 3);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w0_0, _r2, 0);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w1_0, _r2, 1);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w2_0, _r2, 2);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w3_0, _r2, 3);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w0_0, _r3, 0);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w1_0, _r3, 1);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w2_0, _r3, 2);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w3_0, _r3, 3);
                        _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w0_0, _r4, 0);
                        _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w1_0, _r4, 1);
                        _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w2_0, _r4, 2);
                        _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w3_0, _r4, 3);
                        _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w0_0, _r5, 0);
                        _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w1_0, _r5, 1);
                        _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w2_0, _r5, 2);
                        _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w3_0, _r5, 3);
                        _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w0_0, _r6, 0);
                        _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w1_0, _r6, 1);
                        _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w2_0, _r6, 2);
                        _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w3_0, _r6, 3);
                        _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w0_0, _r7, 0);
                        _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w1_0, _r7, 1);
                        _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w2_0, _r7, 2);
                        _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w3_0, _r7, 3);

                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w0_1, _r1, 0);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w1_1, _r1, 1);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w2_1, _r1, 2);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w3_1, _r1, 3);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w0_1, _r2, 0);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w1_1, _r2, 1);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w2_1, _r2, 2);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w3_1, _r2, 3);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w0_1, _r3, 0);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w1_1, _r3, 1);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w2_1, _r3, 2);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w3_1, _r3, 3);
                        _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w0_1, _r4, 0);
                        _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w1_1, _r4, 1);
                        _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w2_1, _r4, 2);
                        _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w3_1, _r4, 3);
                        _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w0_1, _r5, 0);
                        _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w1_1, _r5, 1);
                        _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w2_1, _r5, 2);
                        _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w3_1, _r5, 3);
                        _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w0_1, _r6, 0);
                        _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w1_1, _r6, 1);
                        _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w2_1, _r6, 2);
                        _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w3_1, _r6, 3);
                        _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w0_1, _r7, 0);
                        _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w1_1, _r7, 1);
                        _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w2_1, _r7, 2);
                        _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w3_1, _r7, 3);

//                         sum0 += r0[0] * k0[0];

                        r0 += 32;
                        k0 += 16;
                        k1 += 16;
                    }

                    vst1q_f32(output0_tm0 + r * 4, _sum0_0);
                    vst1q_f32(output0_tm1 + r * 4, _sum1_0);
                    vst1q_f32(output0_tm2 + r * 4, _sum2_0);
                    vst1q_f32(output0_tm3 + r * 4, _sum3_0);
                    vst1q_f32(output0_tm4 + r * 4, _sum4_0);
                    vst1q_f32(output0_tm5 + r * 4, _sum5_0);
                    vst1q_f32(output0_tm6 + r * 4, _sum6_0);
                    vst1q_f32(output0_tm7 + r * 4, _sum7_0);

                    vst1q_f32(output1_tm0 + r * 4, _sum0_1);
                    vst1q_f32(output1_tm1 + r * 4, _sum1_1);
                    vst1q_f32(output1_tm2 + r * 4, _sum2_1);
                    vst1q_f32(output1_tm3 + r * 4, _sum3_1);
                    vst1q_f32(output1_tm4 + r * 4, _sum4_1);
                    vst1q_f32(output1_tm5 + r * 4, _sum5_1);
                    vst1q_f32(output1_tm6 + r * 4, _sum6_1);
                    vst1q_f32(output1_tm7 + r * 4, _sum7_1);
                }
                for (; i+3<tiles; i+=4)
                {
                    float* output0_tm0 = out0_tm.row(i);
                    float* output0_tm1 = out0_tm.row(i+1);
                    float* output0_tm2 = out0_tm.row(i+2);
                    float* output0_tm3 = out0_tm.row(i+3);

                    float* output1_tm0 = out1_tm.row(i);
                    float* output1_tm1 = out1_tm.row(i+1);
                    float* output1_tm2 = out1_tm.row(i+2);
                    float* output1_tm3 = out1_tm.row(i+3);

                    const float* r0 = bb2.row(i/8 + (i%8)/4);

                    const float* k0 = kernel0_tm.row(r);
                    const float* k1 = kernel1_tm.row(r);

                    float32x4_t _sum0_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum2_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum3_0 = vdupq_n_f32(0.f);

                    float32x4_t _sum0_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum1_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum3_1 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );
                        float32x4_t _r1 = vld1q_f32( r0 + 4 );
                        float32x4_t _r2 = vld1q_f32( r0 + 8 );
                        float32x4_t _r3 = vld1q_f32( r0 + 12 );

                        float32x4_t _w0_0 = vld1q_f32( k0 );
                        float32x4_t _w1_0 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2_0 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3_0 = vld1q_f32( k0 + 12 );

                        float32x4_t _w0_1 = vld1q_f32( k1 );
                        float32x4_t _w1_1 = vld1q_f32( k1 + 4 );
                        float32x4_t _w2_1 = vld1q_f32( k1 + 8 );
                        float32x4_t _w3_1 = vld1q_f32( k1 + 12 );

                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w0_0, _r1, 0);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w1_0, _r1, 1);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w2_0, _r1, 2);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w3_0, _r1, 3);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w0_0, _r2, 0);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w1_0, _r2, 1);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w2_0, _r2, 2);
                        _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w3_0, _r2, 3);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w0_0, _r3, 0);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w1_0, _r3, 1);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w2_0, _r3, 2);
                        _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w3_0, _r3, 3);

                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w0_1, _r1, 0);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w1_1, _r1, 1);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w2_1, _r1, 2);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w3_1, _r1, 3);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w0_1, _r2, 0);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w1_1, _r2, 1);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w2_1, _r2, 2);
                        _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w3_1, _r2, 3);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w0_1, _r3, 0);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w1_1, _r3, 1);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w2_1, _r3, 2);
                        _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w3_1, _r3, 3);

//                         sum0 += r0[0] * k0[0];

                        r0 += 16;
                        k0 += 16;
                        k1 += 16;
                    }

                    vst1q_f32(output0_tm0 + r * 4, _sum0_0);
                    vst1q_f32(output0_tm1 + r * 4, _sum1_0);
                    vst1q_f32(output0_tm2 + r * 4, _sum2_0);
                    vst1q_f32(output0_tm3 + r * 4, _sum3_0);

                    vst1q_f32(output1_tm0 + r * 4, _sum0_1);
                    vst1q_f32(output1_tm1 + r * 4, _sum1_1);
                    vst1q_f32(output1_tm2 + r * 4, _sum2_1);
                    vst1q_f32(output1_tm3 + r * 4, _sum3_1);
                }
                for (; i+1<tiles; i+=2)
                {
                    float* output0_tm0 = out0_tm.row(i);
                    float* output0_tm1 = out0_tm.row(i+1);

                    float* output1_tm0 = out1_tm.row(i);
                    float* output1_tm1 = out1_tm.row(i+1);

                    const float* r0 = bb2.row(i/8 + (i%8)/4 + (i%4)/2);

                    const float* k0 = kernel0_tm.row(r);
                    const float* k1 = kernel1_tm.row(r);

                    float32x4_t _sum0_0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1_0 = vdupq_n_f32(0.f);

                    float32x4_t _sum0_1 = vdupq_n_f32(0.f);
                    float32x4_t _sum1_1 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );
                        float32x4_t _r1 = vld1q_f32( r0 + 4 );

                        float32x4_t _w0_0 = vld1q_f32( k0 );
                        float32x4_t _w1_0 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2_0 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3_0 = vld1q_f32( k0 + 12 );

                        float32x4_t _w0_1 = vld1q_f32( k1 );
                        float32x4_t _w1_1 = vld1q_f32( k1 + 4 );
                        float32x4_t _w2_1 = vld1q_f32( k1 + 8 );
                        float32x4_t _w3_1 = vld1q_f32( k1 + 12 );

                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w0_0, _r1, 0);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w1_0, _r1, 1);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w2_0, _r1, 2);
                        _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w3_0, _r1, 3);

                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w0_1, _r1, 0);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w1_1, _r1, 1);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w2_1, _r1, 2);
                        _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w3_1, _r1, 3);

//                         sum0 += r0[0] * k0[0];

                        r0 += 8;
                        k0 += 16;
                        k1 += 16;
                    }

                    vst1q_f32(output0_tm0 + r * 4, _sum0_0);
                    vst1q_f32(output0_tm1 + r * 4, _sum1_0);

                    vst1q_f32(output1_tm0 + r * 4, _sum0_1);
                    vst1q_f32(output1_tm1 + r * 4, _sum1_1);
                }
                for (; i<tiles; i++)
                {
                    float* output0_tm = out0_tm.row(i);

                    float* output1_tm = out1_tm.row(i);

                    const float* r0 = bb2.row(i/8 + (i%8)/4 + (i%4)/2 + i%2);

                    const float* k0 = kernel0_tm.row(r);
                    const float* k1 = kernel1_tm.row(r);

                    float32x4_t _sum0_0 = vdupq_n_f32(0.f);

                    float32x4_t _sum0_1 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );

                        float32x4_t _w0_0 = vld1q_f32( k0 );
                        float32x4_t _w1_0 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2_0 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3_0 = vld1q_f32( k0 + 12 );

                        float32x4_t _w0_1 = vld1q_f32( k1 );
                        float32x4_t _w1_1 = vld1q_f32( k1 + 4 );
                        float32x4_t _w2_1 = vld1q_f32( k1 + 8 );
                        float32x4_t _w3_1 = vld1q_f32( k1 + 12 );

                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                        _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);

                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                        _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);

//                         sum0 += r0[0] * k0[0];

                        r0 += 4;
                        k0 += 16;
                        k1 += 16;
                    }

                    vst1q_f32(output0_tm + r * 4, _sum0_0);

                    vst1q_f32(output1_tm + r * 4, _sum0_1);
                }

            }
        }
#endif // __ARM_NEON && __aarch64__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    float* output0_tm0 = out0_tm.row(i);
                    float* output0_tm1 = out0_tm.row(i+1);
                    float* output0_tm2 = out0_tm.row(i+2);
                    float* output0_tm3 = out0_tm.row(i+3);
                    float* output0_tm4 = out0_tm.row(i+4);
                    float* output0_tm5 = out0_tm.row(i+5);
                    float* output0_tm6 = out0_tm.row(i+6);
                    float* output0_tm7 = out0_tm.row(i+7);

                    const float* r0 = bb2.row(i/8);

                    const float* k0 = kernel0_tm.row(r);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vdupq_n_f32(0.f);
                    float32x4_t _sum4 = vdupq_n_f32(0.f);
                    float32x4_t _sum5 = vdupq_n_f32(0.f);
                    float32x4_t _sum6 = vdupq_n_f32(0.f);
                    float32x4_t _sum7 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );
                        float32x4_t _r1 = vld1q_f32( r0 + 4 );
                        float32x4_t _r2 = vld1q_f32( r0 + 8 );
                        float32x4_t _r3 = vld1q_f32( r0 + 12 );
                        float32x4_t _r4 = vld1q_f32( r0 + 16 );
                        float32x4_t _r5 = vld1q_f32( r0 + 20 );
                        float32x4_t _r6 = vld1q_f32( r0 + 24 );
                        float32x4_t _r7 = vld1q_f32( r0 + 28 );

                        float32x4_t _w0 = vld1q_f32( k0 );
                        float32x4_t _w1 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3 = vld1q_f32( k0 + 12 );

#if __aarch64__
                        _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w0, _r1, 0);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w1, _r1, 1);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w2, _r1, 2);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w3, _r1, 3);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w0, _r2, 0);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w1, _r2, 1);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w2, _r2, 2);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w3, _r2, 3);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w0, _r3, 0);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w1, _r3, 1);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w2, _r3, 2);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w3, _r3, 3);
                        _sum4 = vmlaq_laneq_f32(_sum4, _w0, _r4, 0);
                        _sum4 = vmlaq_laneq_f32(_sum4, _w1, _r4, 1);
                        _sum4 = vmlaq_laneq_f32(_sum4, _w2, _r4, 2);
                        _sum4 = vmlaq_laneq_f32(_sum4, _w3, _r4, 3);
                        _sum5 = vmlaq_laneq_f32(_sum5, _w0, _r5, 0);
                        _sum5 = vmlaq_laneq_f32(_sum5, _w1, _r5, 1);
                        _sum5 = vmlaq_laneq_f32(_sum5, _w2, _r5, 2);
                        _sum5 = vmlaq_laneq_f32(_sum5, _w3, _r5, 3);
                        _sum6 = vmlaq_laneq_f32(_sum6, _w0, _r6, 0);
                        _sum6 = vmlaq_laneq_f32(_sum6, _w1, _r6, 1);
                        _sum6 = vmlaq_laneq_f32(_sum6, _w2, _r6, 2);
                        _sum6 = vmlaq_laneq_f32(_sum6, _w3, _r6, 3);
                        _sum7 = vmlaq_laneq_f32(_sum7, _w0, _r7, 0);
                        _sum7 = vmlaq_laneq_f32(_sum7, _w1, _r7, 1);
                        _sum7 = vmlaq_laneq_f32(_sum7, _w2, _r7, 2);
                        _sum7 = vmlaq_laneq_f32(_sum7, _w3, _r7, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                        _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
                        _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_r1), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r1), 1);
                        _sum1 = vmlaq_lane_f32(_sum1, _w2, vget_high_f32(_r1), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w3, vget_high_f32(_r1), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_low_f32(_r2), 0);
                        _sum2 = vmlaq_lane_f32(_sum2, _w1, vget_low_f32(_r2), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_r2), 0);
                        _sum2 = vmlaq_lane_f32(_sum2, _w3, vget_high_f32(_r2), 1);
                        _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_low_f32(_r3), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _w1, vget_low_f32(_r3), 1);
                        _sum3 = vmlaq_lane_f32(_sum3, _w2, vget_high_f32(_r3), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_r3), 1);
                        _sum4 = vmlaq_lane_f32(_sum4, _w0, vget_low_f32(_r4), 0);
                        _sum4 = vmlaq_lane_f32(_sum4, _w1, vget_low_f32(_r4), 1);
                        _sum4 = vmlaq_lane_f32(_sum4, _w2, vget_high_f32(_r4), 0);
                        _sum4 = vmlaq_lane_f32(_sum4, _w3, vget_high_f32(_r4), 1);
                        _sum5 = vmlaq_lane_f32(_sum5, _w0, vget_low_f32(_r5), 0);
                        _sum5 = vmlaq_lane_f32(_sum5, _w1, vget_low_f32(_r5), 1);
                        _sum5 = vmlaq_lane_f32(_sum5, _w2, vget_high_f32(_r5), 0);
                        _sum5 = vmlaq_lane_f32(_sum5, _w3, vget_high_f32(_r5), 1);
                        _sum6 = vmlaq_lane_f32(_sum6, _w0, vget_low_f32(_r6), 0);
                        _sum6 = vmlaq_lane_f32(_sum6, _w1, vget_low_f32(_r6), 1);
                        _sum6 = vmlaq_lane_f32(_sum6, _w2, vget_high_f32(_r6), 0);
                        _sum6 = vmlaq_lane_f32(_sum6, _w3, vget_high_f32(_r6), 1);
                        _sum7 = vmlaq_lane_f32(_sum7, _w0, vget_low_f32(_r7), 0);
                        _sum7 = vmlaq_lane_f32(_sum7, _w1, vget_low_f32(_r7), 1);
                        _sum7 = vmlaq_lane_f32(_sum7, _w2, vget_high_f32(_r7), 0);
                        _sum7 = vmlaq_lane_f32(_sum7, _w3, vget_high_f32(_r7), 1);
#endif
//                         sum0 += r0[0] * k0[0];

                        r0 += 32;
                        k0 += 16;
                    }

                    vst1q_f32(output0_tm0 + r * 4, _sum0);
                    vst1q_f32(output0_tm1 + r * 4, _sum1);
                    vst1q_f32(output0_tm2 + r * 4, _sum2);
                    vst1q_f32(output0_tm3 + r * 4, _sum3);
                    vst1q_f32(output0_tm4 + r * 4, _sum4);
                    vst1q_f32(output0_tm5 + r * 4, _sum5);
                    vst1q_f32(output0_tm6 + r * 4, _sum6);
                    vst1q_f32(output0_tm7 + r * 4, _sum7);
                }
                for (; i+3<tiles; i+=4)
                {
                    float* output0_tm0 = out0_tm.row(i);
                    float* output0_tm1 = out0_tm.row(i+1);
                    float* output0_tm2 = out0_tm.row(i+2);
                    float* output0_tm3 = out0_tm.row(i+3);

                    const float* r0 = bb2.row(i/8 + (i%8)/4);

                    const float* k0 = kernel0_tm.row(r);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );
                        float32x4_t _r1 = vld1q_f32( r0 + 4 );
                        float32x4_t _r2 = vld1q_f32( r0 + 8 );
                        float32x4_t _r3 = vld1q_f32( r0 + 12 );

                        float32x4_t _w0 = vld1q_f32( k0 );
                        float32x4_t _w1 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3 = vld1q_f32( k0 + 12 );

#if __aarch64__
                        _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w0, _r1, 0);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w1, _r1, 1);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w2, _r1, 2);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w3, _r1, 3);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w0, _r2, 0);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w1, _r2, 1);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w2, _r2, 2);
                        _sum2 = vmlaq_laneq_f32(_sum2, _w3, _r2, 3);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w0, _r3, 0);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w1, _r3, 1);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w2, _r3, 2);
                        _sum3 = vmlaq_laneq_f32(_sum3, _w3, _r3, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                        _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
                        _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_r1), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r1), 1);
                        _sum1 = vmlaq_lane_f32(_sum1, _w2, vget_high_f32(_r1), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w3, vget_high_f32(_r1), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_low_f32(_r2), 0);
                        _sum2 = vmlaq_lane_f32(_sum2, _w1, vget_low_f32(_r2), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_r2), 0);
                        _sum2 = vmlaq_lane_f32(_sum2, _w3, vget_high_f32(_r2), 1);
                        _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_low_f32(_r3), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _w1, vget_low_f32(_r3), 1);
                        _sum3 = vmlaq_lane_f32(_sum3, _w2, vget_high_f32(_r3), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_r3), 1);
#endif
//                         sum0 += r0[0] * k0[0];

                        r0 += 16;
                        k0 += 16;
                    }

                    vst1q_f32(output0_tm0 + r * 4, _sum0);
                    vst1q_f32(output0_tm1 + r * 4, _sum1);
                    vst1q_f32(output0_tm2 + r * 4, _sum2);
                    vst1q_f32(output0_tm3 + r * 4, _sum3);
                }
                for (; i+1<tiles; i+=2)
                {
                    float* output0_tm0 = out0_tm.row(i);
                    float* output0_tm1 = out0_tm.row(i+1);

                    const float* r0 = bb2.row(i/8 + (i%8)/4 + (i%4)/2);

                    const float* k0 = kernel0_tm.row(r);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );
                        float32x4_t _r1 = vld1q_f32( r0 + 4 );

                        float32x4_t _w0 = vld1q_f32( k0 );
                        float32x4_t _w1 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3 = vld1q_f32( k0 + 12 );

#if __aarch64__
                        _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w0, _r1, 0);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w1, _r1, 1);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w2, _r1, 2);
                        _sum1 = vmlaq_laneq_f32(_sum1, _w3, _r1, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                        _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
                        _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_r1), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r1), 1);
                        _sum1 = vmlaq_lane_f32(_sum1, _w2, vget_high_f32(_r1), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w3, vget_high_f32(_r1), 1);
#endif
//                         sum0 += r0[0] * k0[0];

                        r0 += 8;
                        k0 += 16;
                    }

                    vst1q_f32(output0_tm0 + r * 4, _sum0);
                    vst1q_f32(output0_tm1 + r * 4, _sum1);
                }
                for (; i<tiles; i++)
                {
                    float* output0_tm = out0_tm.row(i);

                    const float* r0 = bb2.row(i/8 + (i%8)/4 + (i%4)/2 + i%2);

                    const float* k0 = kernel0_tm.row(r);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q<inch; q++)
                    {
                        float32x4_t _r0 = vld1q_f32( r0 );

                        float32x4_t _w0 = vld1q_f32( k0 );
                        float32x4_t _w1 = vld1q_f32( k0 + 4 );
                        float32x4_t _w2 = vld1q_f32( k0 + 8 );
                        float32x4_t _w3 = vld1q_f32( k0 + 12 );

#if __aarch64__
                        _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                        _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                        _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                        _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
#endif
//                         sum0 += r0[0] * k0[0];

                        r0 += 4;
                        k0 += 16;
                    }

                    vst1q_f32(output0_tm + r * 4, _sum0);
                }

            }
        }

    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, elemsize, elempack);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

//             const float bias0 = bias ? bias[p] : 0.f;
            float32x4_t _bias0 = bias ? vld1q_f32( (const float*)bias + p * 4) : vdupq_n_f32(0.f);

            float tmp[6][8][4];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
                    const float* output0_tm = out0_tm.row(i * w_tm/8 + j);
                    float* output0 = out0.row(i * 6) + (j * 6) * 4;

                    // TODO neon optimize
                    for (int m=0; m<8; m++)
                    {
                        float32x4_t _out0tm0 = vld1q_f32(output0_tm);
                        float32x4_t _out0tm1 = vld1q_f32(output0_tm + 4);
                        float32x4_t _out0tm2 = vld1q_f32(output0_tm + 8);
                        float32x4_t _out0tm3 = vld1q_f32(output0_tm + 12);
                        float32x4_t _out0tm4 = vld1q_f32(output0_tm + 16);
                        float32x4_t _out0tm5 = vld1q_f32(output0_tm + 20);
                        float32x4_t _out0tm6 = vld1q_f32(output0_tm + 24);
                        float32x4_t _out0tm7 = vld1q_f32(output0_tm + 28);

                        float32x4_t _tmp024a = vaddq_f32(_out0tm1, _out0tm2);
                        float32x4_t _tmp135a = vsubq_f32(_out0tm1, _out0tm2);

//                         float tmp024a = output0_tm[1] + output0_tm[2];
//                         float tmp135a = output0_tm[1] - output0_tm[2];

                        float32x4_t _tmp024b = vaddq_f32(_out0tm3, _out0tm4);
                        float32x4_t _tmp135b = vsubq_f32(_out0tm3, _out0tm4);

//                         float tmp024b = output0_tm[3] + output0_tm[4];
//                         float tmp135b = output0_tm[3] - output0_tm[4];

                        float32x4_t _tmp024c = vaddq_f32(_out0tm5, _out0tm6);
                        float32x4_t _tmp135c = vsubq_f32(_out0tm5, _out0tm6);

//                         float tmp024c = output0_tm[5] + output0_tm[6];
//                         float tmp135c = output0_tm[5] - output0_tm[6];

                        float32x4_t _tmp0m = vaddq_f32(vaddq_f32(_out0tm0, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f));
                        float32x4_t _tmp2m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
                        float32x4_t _tmp4m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
                        vst1q_f32(tmp[0][m], _tmp0m);
                        vst1q_f32(tmp[2][m], _tmp2m);
                        vst1q_f32(tmp[4][m], _tmp4m);

//                         tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
//                         tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
//                         tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        float32x4_t _tmp1m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
                        float32x4_t _tmp3m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
                        float32x4_t _tmp5m = vaddq_f32(vaddq_f32(_out0tm7, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f));
                        vst1q_f32(tmp[1][m], _tmp1m);
                        vst1q_f32(tmp[3][m], _tmp3m);
                        vst1q_f32(tmp[5][m], _tmp5m);

//                         tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
//                         tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
//                         tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm += 8*4;
                    }

                    for (int m=0; m<6; m++)
                    {
                        float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
                        float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
                        float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
                        float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
                        float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
                        float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);
                        float32x4_t _tmp06 = vld1q_f32(tmp[m][6]);
                        float32x4_t _tmp07 = vld1q_f32(tmp[m][7]);

                        float32x4_t _tmp024a = vaddq_f32(_tmp01, _tmp02);
                        float32x4_t _tmp135a = vsubq_f32(_tmp01, _tmp02);

//                         float tmp024a = tmp0[1] + tmp0[2];
//                         float tmp135a = tmp0[1] - tmp0[2];

                        float32x4_t _tmp024b = vaddq_f32(_tmp03, _tmp04);
                        float32x4_t _tmp135b = vsubq_f32(_tmp03, _tmp04);

//                         float tmp024b = tmp0[3] + tmp0[4];
//                         float tmp135b = tmp0[3] - tmp0[4];

                        float32x4_t _tmp024c = vaddq_f32(_tmp05, _tmp06);
                        float32x4_t _tmp135c = vsubq_f32(_tmp05, _tmp06);

//                         float tmp024c = tmp0[5] + tmp0[6];
//                         float tmp135c = tmp0[5] - tmp0[6];

                        float32x4_t _out00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp00, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f)));
                        float32x4_t _out02 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
                        float32x4_t _out04 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));
                        vst1q_f32(output0, _out00);
                        vst1q_f32(output0 + 8, _out02);
                        vst1q_f32(output0 + 16, _out04);

//                         output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
//                         output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
//                         output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        float32x4_t _out01 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
                        float32x4_t _out03 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
                        float32x4_t _out05 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp07, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f)));
                        vst1q_f32(output0 + 4, _out01);
                        vst1q_f32(output0 + 12, _out03);
                        vst1q_f32(output0 + 20, _out05);

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
