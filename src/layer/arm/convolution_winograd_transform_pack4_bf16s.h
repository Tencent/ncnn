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

static void conv3x3s1_winograd64_transform_input_pack4_bf16s_neon(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        float tmp[8][8][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const unsigned short* r0 = img0.row<const unsigned short>(i * 6) + (j * 6) * 4;

                for (int m = 0; m < 8; m++)
                {
                    float32x4_t _r00 = vcvt_f32_bf16(vld1_u16(r0));
                    float32x4_t _r01 = vcvt_f32_bf16(vld1_u16(r0 + 4));
                    float32x4_t _r02 = vcvt_f32_bf16(vld1_u16(r0 + 8));
                    float32x4_t _r03 = vcvt_f32_bf16(vld1_u16(r0 + 12));
                    float32x4_t _r04 = vcvt_f32_bf16(vld1_u16(r0 + 16));
                    float32x4_t _r05 = vcvt_f32_bf16(vld1_u16(r0 + 20));
                    float32x4_t _r06 = vcvt_f32_bf16(vld1_u16(r0 + 24));
                    float32x4_t _r07 = vcvt_f32_bf16(vld1_u16(r0 + 28));

                    float32x4_t _tmp0m = vmlaq_n_f32(vsubq_f32(_r00, _r06), vsubq_f32(_r04, _r02), 5.25f);
                    float32x4_t _tmp7m = vmlaq_n_f32(vsubq_f32(_r07, _r01), vsubq_f32(_r03, _r05), 5.25f);
                    vst1q_f32(tmp[0][m], _tmp0m);
                    vst1q_f32(tmp[7][m], _tmp7m);

                    float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_r02, _r06), _r04, 4.25f);
                    float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_r01, _r05), _r03, 4.25f);

                    float32x4_t _tmp1m = vaddq_f32(_tmp12a, _tmp12b);
                    float32x4_t _tmp2m = vsubq_f32(_tmp12a, _tmp12b);
                    vst1q_f32(tmp[1][m], _tmp1m);
                    vst1q_f32(tmp[2][m], _tmp2m);

                    float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_r06, _r02, 0.25f), _r04, 1.25f);
                    float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

                    float32x4_t _tmp3m = vaddq_f32(_tmp34a, _tmp34b);
                    float32x4_t _tmp4m = vsubq_f32(_tmp34a, _tmp34b);
                    vst1q_f32(tmp[3][m], _tmp3m);
                    vst1q_f32(tmp[4][m], _tmp4m);

                    float32x4_t _tmp56a = vmlaq_n_f32(_r06, vmlsq_n_f32(_r02, _r04, 1.25f), 4.f);
                    float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

                    float32x4_t _tmp5m = vaddq_f32(_tmp56a, _tmp56b);
                    float32x4_t _tmp6m = vsubq_f32(_tmp56a, _tmp56b);
                    vst1q_f32(tmp[5][m], _tmp5m);
                    vst1q_f32(tmp[6][m], _tmp6m);

                    r0 += w * 4;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 4;
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

                    float32x4_t _tmp12a = vmlsq_n_f32(vaddq_f32(_tmp02, _tmp06), _tmp04, 4.25f);
                    float32x4_t _tmp12b = vmlsq_n_f32(vaddq_f32(_tmp01, _tmp05), _tmp03, 4.25f);

                    float32x4_t _r0tm1 = vaddq_f32(_tmp12a, _tmp12b);
                    float32x4_t _r0tm2 = vsubq_f32(_tmp12a, _tmp12b);

                    float32x4_t _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                    float32x4_t _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

                    float32x4_t _r0tm3 = vaddq_f32(_tmp34a, _tmp34b);
                    float32x4_t _r0tm4 = vsubq_f32(_tmp34a, _tmp34b);

                    float32x4_t _tmp56a = vmlaq_n_f32(_tmp06, vmlsq_n_f32(_tmp02, _tmp04, 1.25f), 4.f);
                    float32x4_t _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

                    float32x4_t _r0tm5 = vaddq_f32(_tmp56a, _tmp56b);
                    float32x4_t _r0tm6 = vsubq_f32(_tmp56a, _tmp56b);

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

static void conv3x3s1_winograd64_transform_output_pack4_bf16s_neon(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 6;
    const int h_tiles = outh / 6;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

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

        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + p * 4) : vdupq_n_f32(0.f);

        float tmp[6][8][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 4;
                const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                const float* output0_tm_2 = output0_tm_0 + tiles * 8;
                const float* output0_tm_3 = output0_tm_0 + tiles * 12;
                const float* output0_tm_4 = output0_tm_0 + tiles * 16;
                const float* output0_tm_5 = output0_tm_0 + tiles * 20;
                const float* output0_tm_6 = output0_tm_0 + tiles * 24;
                const float* output0_tm_7 = output0_tm_0 + tiles * 28;

                unsigned short* output0 = out0.row<unsigned short>(i * 6) + (j * 6) * 4;

                for (int m = 0; m < 8; m++)
                {
                    float32x4_t _out0tm0 = vld1q_f32(output0_tm_0);
                    float32x4_t _out0tm1 = vld1q_f32(output0_tm_1);
                    float32x4_t _out0tm2 = vld1q_f32(output0_tm_2);
                    float32x4_t _out0tm3 = vld1q_f32(output0_tm_3);
                    float32x4_t _out0tm4 = vld1q_f32(output0_tm_4);
                    float32x4_t _out0tm5 = vld1q_f32(output0_tm_5);
                    float32x4_t _out0tm6 = vld1q_f32(output0_tm_6);
                    float32x4_t _out0tm7 = vld1q_f32(output0_tm_7);

                    float32x4_t _tmp024a = vaddq_f32(_out0tm1, _out0tm2);
                    float32x4_t _tmp135a = vsubq_f32(_out0tm1, _out0tm2);

                    float32x4_t _tmp024b = vaddq_f32(_out0tm3, _out0tm4);
                    float32x4_t _tmp135b = vsubq_f32(_out0tm3, _out0tm4);

                    float32x4_t _tmp024c = vaddq_f32(_out0tm5, _out0tm6);
                    float32x4_t _tmp135c = vsubq_f32(_out0tm5, _out0tm6);

                    float32x4_t _tmp0m = vaddq_f32(vaddq_f32(_out0tm0, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f));
                    float32x4_t _tmp2m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
                    float32x4_t _tmp4m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
                    vst1q_f32(tmp[0][m], _tmp0m);
                    vst1q_f32(tmp[2][m], _tmp2m);
                    vst1q_f32(tmp[4][m], _tmp4m);

                    float32x4_t _tmp1m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
                    float32x4_t _tmp3m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
                    float32x4_t _tmp5m = vaddq_f32(vaddq_f32(_out0tm7, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f));
                    vst1q_f32(tmp[1][m], _tmp1m);
                    vst1q_f32(tmp[3][m], _tmp3m);
                    vst1q_f32(tmp[5][m], _tmp5m);

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

                    float32x4_t _tmp024b = vaddq_f32(_tmp03, _tmp04);
                    float32x4_t _tmp135b = vsubq_f32(_tmp03, _tmp04);

                    float32x4_t _tmp024c = vaddq_f32(_tmp05, _tmp06);
                    float32x4_t _tmp135c = vsubq_f32(_tmp05, _tmp06);

                    float32x4_t _out00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp00, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f)));
                    float32x4_t _out02 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
                    float32x4_t _out04 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));
                    vst1_u16(output0, vcvt_bf16_f32(_out00));
                    vst1_u16(output0 + 8, vcvt_bf16_f32(_out02));
                    vst1_u16(output0 + 16, vcvt_bf16_f32(_out04));

                    float32x4_t _out01 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
                    float32x4_t _out03 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
                    float32x4_t _out05 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp07, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f)));
                    vst1_u16(output0 + 4, vcvt_bf16_f32(_out01));
                    vst1_u16(output0 + 12, vcvt_bf16_f32(_out03));
                    vst1_u16(output0 + 20, vcvt_bf16_f32(_out05));

                    output0 += outw * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_transform_input_pack4_bf16s_neon(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    // const float itm[6][6] = {
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
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        float tmp[6][6][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const unsigned short* r0 = img0.row<const unsigned short>(i * 4) + (j * 4) * 4;

                for (int m = 0; m < 6; m++)
                {
                    float32x4_t _r00 = vcvt_f32_bf16(vld1_u16(r0));
                    float32x4_t _r01 = vcvt_f32_bf16(vld1_u16(r0 + 4));
                    float32x4_t _r02 = vcvt_f32_bf16(vld1_u16(r0 + 8));
                    float32x4_t _r03 = vcvt_f32_bf16(vld1_u16(r0 + 12));
                    float32x4_t _r04 = vcvt_f32_bf16(vld1_u16(r0 + 16));
                    float32x4_t _r05 = vcvt_f32_bf16(vld1_u16(r0 + 20));

                    float32x4_t _tmp0m = vmlsq_n_f32(vmlaq_n_f32(_r04, _r00, 4.f), _r02, 5.f);
                    float32x4_t _tmp1m = vmlsq_n_f32(vaddq_f32(_r04, _r03), vaddq_f32(_r01, _r02), 4.f);
                    float32x4_t _tmp2m = vmlaq_n_f32(vsubq_f32(_r04, _r03), vsubq_f32(_r01, _r02), 4.f);
                    float32x4_t _tmp3m = vmlsq_n_f32(vsubq_f32(_r04, _r02), vsubq_f32(_r01, _r03), 2.f);
                    float32x4_t _tmp4m = vmlaq_n_f32(vsubq_f32(_r04, _r02), vsubq_f32(_r01, _r03), 2.f);
                    float32x4_t _tmp5m = vmlsq_n_f32(vmlaq_n_f32(_r05, _r01, 4.f), _r03, 5.f);

                    vst1q_f32(tmp[0][m], _tmp0m);
                    vst1q_f32(tmp[1][m], _tmp1m);
                    vst1q_f32(tmp[2][m], _tmp2m);
                    vst1q_f32(tmp[3][m], _tmp3m);
                    vst1q_f32(tmp[4][m], _tmp4m);
                    vst1q_f32(tmp[5][m], _tmp5m);

                    r0 += w * 4;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 4;
                float* r0_tm_1 = r0_tm_0 + tiles * 4;
                float* r0_tm_2 = r0_tm_0 + tiles * 8;
                float* r0_tm_3 = r0_tm_0 + tiles * 12;
                float* r0_tm_4 = r0_tm_0 + tiles * 16;
                float* r0_tm_5 = r0_tm_0 + tiles * 20;

                for (int m = 0; m < 6; m++)
                {
                    float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
                    float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
                    float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
                    float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
                    float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
                    float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);

                    float32x4_t _r0tm0 = vmlsq_n_f32(vmlaq_n_f32(_tmp04, _tmp00, 4.f), _tmp02, 5.f);
                    float32x4_t _r0tm1 = vmlsq_n_f32(vaddq_f32(_tmp04, _tmp03), vaddq_f32(_tmp01, _tmp02), 4.f);
                    float32x4_t _r0tm2 = vmlaq_n_f32(vsubq_f32(_tmp04, _tmp03), vsubq_f32(_tmp01, _tmp02), 4.f);
                    float32x4_t _r0tm3 = vmlsq_n_f32(vsubq_f32(_tmp04, _tmp02), vsubq_f32(_tmp01, _tmp03), 2.f);
                    float32x4_t _r0tm4 = vmlaq_n_f32(vsubq_f32(_tmp04, _tmp02), vsubq_f32(_tmp01, _tmp03), 2.f);
                    float32x4_t _r0tm5 = vmlsq_n_f32(vmlaq_n_f32(_tmp05, _tmp01, 4.f), _tmp03, 5.f);

                    vst1q_f32(r0_tm_0, _r0tm0);
                    vst1q_f32(r0_tm_1, _r0tm1);
                    vst1q_f32(r0_tm_2, _r0tm2);
                    vst1q_f32(r0_tm_3, _r0tm3);
                    vst1q_f32(r0_tm_4, _r0tm4);
                    vst1q_f32(r0_tm_5, _r0tm5);

                    r0_tm_0 += tiles * 24;
                    r0_tm_1 += tiles * 24;
                    r0_tm_2 += tiles * 24;
                    r0_tm_3 += tiles * 24;
                    r0_tm_4 += tiles * 24;
                    r0_tm_5 += tiles * 24;
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_transform_output_pack4_bf16s_neon(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + p * 4) : vdupq_n_f32(0.f);

        float tmp[4][6][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 4;
                const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                const float* output0_tm_2 = output0_tm_0 + tiles * 8;
                const float* output0_tm_3 = output0_tm_0 + tiles * 12;
                const float* output0_tm_4 = output0_tm_0 + tiles * 16;
                const float* output0_tm_5 = output0_tm_0 + tiles * 20;

                unsigned short* output0 = out0.row<unsigned short>(i * 4) + (j * 4) * 4;

                for (int m = 0; m < 6; m++)
                {
                    float32x4_t _out0tm0 = vld1q_f32(output0_tm_0);
                    float32x4_t _out0tm1 = vld1q_f32(output0_tm_1);
                    float32x4_t _out0tm2 = vld1q_f32(output0_tm_2);
                    float32x4_t _out0tm3 = vld1q_f32(output0_tm_3);
                    float32x4_t _out0tm4 = vld1q_f32(output0_tm_4);
                    float32x4_t _out0tm5 = vld1q_f32(output0_tm_5);

                    float32x4_t _tmp02a = vaddq_f32(_out0tm1, _out0tm2);
                    float32x4_t _tmp13a = vsubq_f32(_out0tm1, _out0tm2);

                    float32x4_t _tmp02b = vaddq_f32(_out0tm3, _out0tm4);
                    float32x4_t _tmp13b = vsubq_f32(_out0tm3, _out0tm4);

                    float32x4_t _tmp0m = vaddq_f32(vaddq_f32(_out0tm0, _tmp02a), _tmp02b);
                    float32x4_t _tmp1m = vmlaq_n_f32(_tmp13a, _tmp13b, 2.f);
                    float32x4_t _tmp2m = vmlaq_n_f32(_tmp02a, _tmp02b, 4.f);
                    float32x4_t _tmp3m = vmlaq_n_f32(vaddq_f32(_out0tm5, _tmp13a), _tmp13b, 8.f);

                    vst1q_f32(tmp[0][m], _tmp0m);
                    vst1q_f32(tmp[1][m], _tmp1m);
                    vst1q_f32(tmp[2][m], _tmp2m);
                    vst1q_f32(tmp[3][m], _tmp3m);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }

                for (int m = 0; m < 4; m++)
                {
                    float32x4_t _tmp00 = vld1q_f32(tmp[m][0]);
                    float32x4_t _tmp01 = vld1q_f32(tmp[m][1]);
                    float32x4_t _tmp02 = vld1q_f32(tmp[m][2]);
                    float32x4_t _tmp03 = vld1q_f32(tmp[m][3]);
                    float32x4_t _tmp04 = vld1q_f32(tmp[m][4]);
                    float32x4_t _tmp05 = vld1q_f32(tmp[m][5]);

                    float32x4_t _tmp02a = vaddq_f32(_tmp01, _tmp02);
                    float32x4_t _tmp13a = vsubq_f32(_tmp01, _tmp02);

                    float32x4_t _tmp02b = vaddq_f32(_tmp03, _tmp04);
                    float32x4_t _tmp13b = vsubq_f32(_tmp03, _tmp04);

                    float32x4_t _out00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp00, _tmp02a), _tmp02b));
                    float32x4_t _out01 = vaddq_f32(_bias0, vmlaq_n_f32(_tmp13a, _tmp13b, 2.f));
                    float32x4_t _out02 = vaddq_f32(_bias0, vmlaq_n_f32(_tmp02a, _tmp02b, 4.f));
                    float32x4_t _out03 = vaddq_f32(_bias0, vmlaq_n_f32(vaddq_f32(_tmp05, _tmp13a), _tmp13b, 8.f));

                    vst1_u16(output0, vcvt_bf16_f32(_out00));
                    vst1_u16(output0 + 4, vcvt_bf16_f32(_out01));
                    vst1_u16(output0 + 8, vcvt_bf16_f32(_out02));
                    vst1_u16(output0 + 12, vcvt_bf16_f32(_out03));

                    output0 += outw * 4;
                }
            }
        }
    }
}
