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

static void conv3x3s1_winograd64_transform_input_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 6;
    const int h_tiles = (h - 2) / 6;
    const int tiles = w_tiles * h_tiles;

    // const float itm[8][8] = {
    //     {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
    //
    //     {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
    //     {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
    //     {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
    //     {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
    //
    //     {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
    // };

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
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        __fp16 tmp[8][8][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const __fp16* r0 = img0.row<const __fp16>(i * 6) + (j * 6) * 4;

                for (int m = 0; m < 8; m++)
                {
                    float16x4_t _r00 = vld1_f16(r0);
                    float16x4_t _r01 = vld1_f16(r0 + 4);
                    float16x4_t _r02 = vld1_f16(r0 + 8);
                    float16x4_t _r03 = vld1_f16(r0 + 12);
                    float16x4_t _r04 = vld1_f16(r0 + 16);
                    float16x4_t _r05 = vld1_f16(r0 + 20);
                    float16x4_t _r06 = vld1_f16(r0 + 24);
                    float16x4_t _r07 = vld1_f16(r0 + 28);

                    float16x4_t _tmp0m = vfma_n_f16(vsub_f16(_r00, _r06), vsub_f16(_r04, _r02), 5.25f);
                    float16x4_t _tmp7m = vfma_n_f16(vsub_f16(_r07, _r01), vsub_f16(_r03, _r05), 5.25f);
                    vst1_f16(tmp[0][m], _tmp0m);
                    vst1_f16(tmp[7][m], _tmp7m);

                    float16x4_t _tmp12a = vfms_n_f16(vadd_f16(_r02, _r06), _r04, 4.25f);
                    float16x4_t _tmp12b = vfms_n_f16(vadd_f16(_r01, _r05), _r03, 4.25f);

                    float16x4_t _tmp1m = vadd_f16(_tmp12a, _tmp12b);
                    float16x4_t _tmp2m = vsub_f16(_tmp12a, _tmp12b);
                    vst1_f16(tmp[1][m], _tmp1m);
                    vst1_f16(tmp[2][m], _tmp2m);

                    float16x4_t _tmp34a = vfms_n_f16(vfma_n_f16(_r06, _r02, 0.25f), _r04, 1.25f);
                    float16x4_t _tmp34b = vfma_n_f16(vfms_n_f16(vmul_n_f16(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

                    float16x4_t _tmp3m = vadd_f16(_tmp34a, _tmp34b);
                    float16x4_t _tmp4m = vsub_f16(_tmp34a, _tmp34b);
                    vst1_f16(tmp[3][m], _tmp3m);
                    vst1_f16(tmp[4][m], _tmp4m);

                    float16x4_t _tmp56a = vfma_n_f16(_r06, vfms_n_f16(_r02, _r04, 1.25f), 4.f);
                    float16x4_t _tmp56b = vfma_n_f16(vfms_n_f16(vmul_n_f16(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

                    float16x4_t _tmp5m = vadd_f16(_tmp56a, _tmp56b);
                    float16x4_t _tmp6m = vsub_f16(_tmp56a, _tmp56b);
                    vst1_f16(tmp[5][m], _tmp5m);
                    vst1_f16(tmp[6][m], _tmp6m);

                    r0 += w * 4;
                }

                __fp16* r0_tm_0 = (__fp16*)img0_tm + (i * w_tiles + j) * 4;
                __fp16* r0_tm_1 = r0_tm_0 + tiles * 4;
                __fp16* r0_tm_2 = r0_tm_0 + tiles * 8;
                __fp16* r0_tm_3 = r0_tm_0 + tiles * 12;
                __fp16* r0_tm_4 = r0_tm_0 + tiles * 16;
                __fp16* r0_tm_5 = r0_tm_0 + tiles * 20;
                __fp16* r0_tm_6 = r0_tm_0 + tiles * 24;
                __fp16* r0_tm_7 = r0_tm_0 + tiles * 28;

                for (int m = 0; m < 8; m++)
                {
                    float16x4_t _tmp00 = vld1_f16(tmp[m][0]);
                    float16x4_t _tmp01 = vld1_f16(tmp[m][1]);
                    float16x4_t _tmp02 = vld1_f16(tmp[m][2]);
                    float16x4_t _tmp03 = vld1_f16(tmp[m][3]);
                    float16x4_t _tmp04 = vld1_f16(tmp[m][4]);
                    float16x4_t _tmp05 = vld1_f16(tmp[m][5]);
                    float16x4_t _tmp06 = vld1_f16(tmp[m][6]);
                    float16x4_t _tmp07 = vld1_f16(tmp[m][7]);

                    float16x4_t _r0tm0 = vfma_n_f16(vsub_f16(_tmp00, _tmp06), vsub_f16(_tmp04, _tmp02), 5.25f);
                    float16x4_t _r0tm7 = vfma_n_f16(vsub_f16(_tmp07, _tmp01), vsub_f16(_tmp03, _tmp05), 5.25f);

                    float16x4_t _tmp12a = vfms_n_f16(vadd_f16(_tmp02, _tmp06), _tmp04, 4.25f);
                    float16x4_t _tmp12b = vfms_n_f16(vadd_f16(_tmp01, _tmp05), _tmp03, 4.25f);

                    float16x4_t _r0tm1 = vadd_f16(_tmp12a, _tmp12b);
                    float16x4_t _r0tm2 = vsub_f16(_tmp12a, _tmp12b);

                    float16x4_t _tmp34a = vfms_n_f16(vfma_n_f16(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                    float16x4_t _tmp34b = vfma_n_f16(vfms_n_f16(vmul_n_f16(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

                    float16x4_t _r0tm3 = vadd_f16(_tmp34a, _tmp34b);
                    float16x4_t _r0tm4 = vsub_f16(_tmp34a, _tmp34b);

                    float16x4_t _tmp56a = vfma_n_f16(_tmp06, vfms_n_f16(_tmp02, _tmp04, 1.25f), 4.f);
                    float16x4_t _tmp56b = vfma_n_f16(vfms_n_f16(vmul_n_f16(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

                    float16x4_t _r0tm5 = vadd_f16(_tmp56a, _tmp56b);
                    float16x4_t _r0tm6 = vsub_f16(_tmp56a, _tmp56b);

                    vst1_f16(r0_tm_0, _r0tm0);
                    vst1_f16(r0_tm_1, _r0tm1);
                    vst1_f16(r0_tm_2, _r0tm2);
                    vst1_f16(r0_tm_3, _r0tm3);
                    vst1_f16(r0_tm_4, _r0tm4);
                    vst1_f16(r0_tm_5, _r0tm5);
                    vst1_f16(r0_tm_6, _r0tm6);
                    vst1_f16(r0_tm_7, _r0tm7);

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

static void conv3x3s1_winograd64_transform_output_pack4_fp16sa_neon(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 6;
    const int h_tiles = outh / 6;
    const int tiles = w_tiles * h_tiles;

    const __fp16* biasptr = bias;

    // const float otm[6][8] = {
    //     {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    // };

    // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
    // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
    // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
    // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
    // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
    // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        float16x4_t _bias0 = biasptr ? vld1_f16(biasptr + p * 4) : vdup_n_f16(0.f);

        __fp16 tmp[6][8][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const __fp16* output0_tm_0 = (const __fp16*)out0_tm + (i * w_tiles + j) * 4;
                const __fp16* output0_tm_1 = output0_tm_0 + tiles * 4;
                const __fp16* output0_tm_2 = output0_tm_0 + tiles * 8;
                const __fp16* output0_tm_3 = output0_tm_0 + tiles * 12;
                const __fp16* output0_tm_4 = output0_tm_0 + tiles * 16;
                const __fp16* output0_tm_5 = output0_tm_0 + tiles * 20;
                const __fp16* output0_tm_6 = output0_tm_0 + tiles * 24;
                const __fp16* output0_tm_7 = output0_tm_0 + tiles * 28;

                __fp16* output0 = out0.row<__fp16>(i * 6) + (j * 6) * 4;

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

                    float16x4_t _tmp024b = vadd_f16(_out0tm3, _out0tm4);
                    float16x4_t _tmp135b = vsub_f16(_out0tm3, _out0tm4);

                    float16x4_t _tmp024c = vadd_f16(_out0tm5, _out0tm6);
                    float16x4_t _tmp135c = vsub_f16(_out0tm5, _out0tm6);

                    float16x4_t _tmp0m = vadd_f16(vadd_f16(_out0tm0, _tmp024a), vfma_n_f16(_tmp024b, _tmp024c, 32.f));
                    float16x4_t _tmp2m = vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
                    float16x4_t _tmp4m = vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
                    vst1_f16(tmp[0][m], _tmp0m);
                    vst1_f16(tmp[2][m], _tmp2m);
                    vst1_f16(tmp[4][m], _tmp4m);

                    float16x4_t _tmp1m = vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
                    float16x4_t _tmp3m = vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
                    float16x4_t _tmp5m = vadd_f16(vadd_f16(_out0tm7, _tmp135a), vfma_n_f16(_tmp135c, _tmp135b, 32.f));
                    vst1_f16(tmp[1][m], _tmp1m);
                    vst1_f16(tmp[3][m], _tmp3m);
                    vst1_f16(tmp[5][m], _tmp5m);

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

                    float16x4_t _tmp024b = vadd_f16(_tmp03, _tmp04);
                    float16x4_t _tmp135b = vsub_f16(_tmp03, _tmp04);

                    float16x4_t _tmp024c = vadd_f16(_tmp05, _tmp06);
                    float16x4_t _tmp135c = vsub_f16(_tmp05, _tmp06);

                    float16x4_t _out00 = vadd_f16(_bias0, vadd_f16(vadd_f16(_tmp00, _tmp024a), vfma_n_f16(_tmp024b, _tmp024c, 32.f)));
                    float16x4_t _out02 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
                    float16x4_t _out04 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));
                    vst1_f16(output0, _out00);
                    vst1_f16(output0 + 8, _out02);
                    vst1_f16(output0 + 16, _out04);

                    float16x4_t _out01 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
                    float16x4_t _out03 = vadd_f16(_bias0, vfma_n_f16(vfma_n_f16(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
                    float16x4_t _out05 = vadd_f16(_bias0, vadd_f16(vadd_f16(_tmp07, _tmp135a), vfma_n_f16(_tmp135c, _tmp135b, 32.f)));
                    vst1_f16(output0 + 4, _out01);
                    vst1_f16(output0 + 12, _out03);
                    vst1_f16(output0 + 20, _out05);

                    output0 += outw * 4;
                }
            }
        }
    }
}
