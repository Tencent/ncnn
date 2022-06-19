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

static void conv3x3s1_winograd63_transform_input_pack4_msa(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        v4f32 _v5_25 = __msa_fill_w_f32(5.25f);
        v4f32 _vm4_25 = __msa_fill_w_f32(-4.25f);
        v4f32 _vm1_25 = __msa_fill_w_f32(-1.25f);
        v4f32 _v0_25 = __msa_fill_w_f32(0.25f);
        v4f32 _vm2_5 = __msa_fill_w_f32(-2.5f);
        v4f32 _v0_5 = __msa_fill_w_f32(0.5f);
        v4f32 _v2 = __msa_fill_w_f32(2.f);
        v4f32 _v4 = __msa_fill_w_f32(4.f);

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 6) + (j * 6) * 4;

                for (int m = 0; m < 8; m++)
                {
                    v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                    v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                    v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                    v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);
                    v4f32 _r05 = (v4f32)__msa_ld_w(r0 + 4 * 5, 0);
                    v4f32 _r06 = (v4f32)__msa_ld_w(r0 + 4 * 6, 0);
                    v4f32 _r07 = (v4f32)__msa_ld_w(r0 + 4 * 7, 0);

                    v4f32 _tmp0m = __msa_fmadd_w(__msa_fsub_w(_r00, _r06), _v5_25, __msa_fsub_w(_r04, _r02));
                    v4f32 _tmp7m = __msa_fmadd_w(__msa_fsub_w(_r07, _r01), _v5_25, __msa_fsub_w(_r03, _r05));
                    __msa_st_w((v4i32)_tmp0m, tmp[0][m], 0);
                    __msa_st_w((v4i32)_tmp7m, tmp[7][m], 0);

                    v4f32 _tmp12a = __msa_fmadd_w(__msa_fadd_w(_r02, _r06), _vm4_25, _r04);
                    v4f32 _tmp12b = __msa_fmadd_w(__msa_fadd_w(_r01, _r05), _vm4_25, _r03);

                    v4f32 _tmp1m = __msa_fadd_w(_tmp12a, _tmp12b);
                    v4f32 _tmp2m = __msa_fsub_w(_tmp12a, _tmp12b);
                    __msa_st_w((v4i32)_tmp1m, tmp[1][m], 0);
                    __msa_st_w((v4i32)_tmp2m, tmp[2][m], 0);

                    v4f32 _tmp34a = __msa_fmadd_w(__msa_fmadd_w(_r06, _v0_25, _r02), _vm1_25, _r04);
                    v4f32 _tmp34b = __msa_fmadd_w(__msa_fmadd_w(__msa_fmul_w(_r01, _v0_5), _vm2_5, _r03), _v2, _r05);

                    v4f32 _tmp3m = __msa_fadd_w(_tmp34a, _tmp34b);
                    v4f32 _tmp4m = __msa_fsub_w(_tmp34a, _tmp34b);
                    __msa_st_w((v4i32)_tmp3m, tmp[3][m], 0);
                    __msa_st_w((v4i32)_tmp4m, tmp[4][m], 0);

                    v4f32 _tmp56a = __msa_fmadd_w(_r06, _v4, __msa_fmadd_w(_r02, _vm1_25, _r04));
                    v4f32 _tmp56b = __msa_fmadd_w(__msa_fmadd_w(__msa_fmul_w(_r01, _v2), _vm2_5, _r03), _v0_5, _r05);

                    v4f32 _tmp5m = __msa_fadd_w(_tmp56a, _tmp56b);
                    v4f32 _tmp6m = __msa_fsub_w(_tmp56a, _tmp56b);
                    __msa_st_w((v4i32)_tmp5m, tmp[5][m], 0);
                    __msa_st_w((v4i32)_tmp6m, tmp[6][m], 0);

                    r0 += w * 4;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 4;
                float* r0_tm_1 = r0_tm_0 + tiles * 4;
                float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * 4 * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * 4 * 5;
                float* r0_tm_6 = r0_tm_0 + tiles * 4 * 6;
                float* r0_tm_7 = r0_tm_0 + tiles * 4 * 7;

                for (int m = 0; m < 8; m++)
                {
                    v4f32 _tmp00 = (v4f32)__msa_ld_w(tmp[m][0], 0);
                    v4f32 _tmp01 = (v4f32)__msa_ld_w(tmp[m][1], 0);
                    v4f32 _tmp02 = (v4f32)__msa_ld_w(tmp[m][2], 0);
                    v4f32 _tmp03 = (v4f32)__msa_ld_w(tmp[m][3], 0);
                    v4f32 _tmp04 = (v4f32)__msa_ld_w(tmp[m][4], 0);
                    v4f32 _tmp05 = (v4f32)__msa_ld_w(tmp[m][5], 0);
                    v4f32 _tmp06 = (v4f32)__msa_ld_w(tmp[m][6], 0);
                    v4f32 _tmp07 = (v4f32)__msa_ld_w(tmp[m][7], 0);

                    v4f32 _r0tm0 = __msa_fmadd_w(__msa_fsub_w(_tmp00, _tmp06), _v5_25, __msa_fsub_w(_tmp04, _tmp02));
                    v4f32 _r0tm7 = __msa_fmadd_w(__msa_fsub_w(_tmp07, _tmp01), _v5_25, __msa_fsub_w(_tmp03, _tmp05));

                    v4f32 _tmp12a = __msa_fmadd_w(__msa_fadd_w(_tmp02, _tmp06), _vm4_25, _tmp04);
                    v4f32 _tmp12b = __msa_fmadd_w(__msa_fadd_w(_tmp01, _tmp05), _vm4_25, _tmp03);

                    v4f32 _r0tm1 = __msa_fadd_w(_tmp12a, _tmp12b);
                    v4f32 _r0tm2 = __msa_fsub_w(_tmp12a, _tmp12b);

                    v4f32 _tmp34a = __msa_fmadd_w(__msa_fmadd_w(_tmp06, _v0_25, _tmp02), _vm1_25, _tmp04);
                    v4f32 _tmp34b = __msa_fmadd_w(__msa_fmadd_w(__msa_fmul_w(_tmp01, _v0_5), _vm2_5, _tmp03), _v2, _tmp05);

                    v4f32 _r0tm3 = __msa_fadd_w(_tmp34a, _tmp34b);
                    v4f32 _r0tm4 = __msa_fsub_w(_tmp34a, _tmp34b);

                    v4f32 _tmp56a = __msa_fmadd_w(_tmp06, _v4, __msa_fmadd_w(_tmp02, _vm1_25, _tmp04));
                    v4f32 _tmp56b = __msa_fmadd_w(__msa_fmadd_w(__msa_fmul_w(_tmp01, _v2), _vm2_5, _tmp03), _v0_5, _tmp05);

                    v4f32 _r0tm5 = __msa_fadd_w(_tmp56a, _tmp56b);
                    v4f32 _r0tm6 = __msa_fsub_w(_tmp56a, _tmp56b);

                    __msa_st_w((v4i32)_r0tm0, r0_tm_0, 0);
                    __msa_st_w((v4i32)_r0tm1, r0_tm_1, 0);
                    __msa_st_w((v4i32)_r0tm2, r0_tm_2, 0);
                    __msa_st_w((v4i32)_r0tm3, r0_tm_3, 0);
                    __msa_st_w((v4i32)_r0tm4, r0_tm_4, 0);
                    __msa_st_w((v4i32)_r0tm5, r0_tm_5, 0);
                    __msa_st_w((v4i32)_r0tm6, r0_tm_6, 0);
                    __msa_st_w((v4i32)_r0tm7, r0_tm_7, 0);

                    r0_tm_0 += tiles * 4 * 8;
                    r0_tm_1 += tiles * 4 * 8;
                    r0_tm_2 += tiles * 4 * 8;
                    r0_tm_3 += tiles * 4 * 8;
                    r0_tm_4 += tiles * 4 * 8;
                    r0_tm_5 += tiles * 4 * 8;
                    r0_tm_6 += tiles * 4 * 8;
                    r0_tm_7 += tiles * 4 * 8;
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_output_pack4_msa(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        v4f32 _bias0 = biasptr ? (v4f32)__msa_ld_w(biasptr + p * 4, 0) : (v4f32)__msa_fill_w(0);

        float tmp[6][8][4];

        v4f32 _v32 = __msa_fill_w_f32(32.f);
        v4f32 _v16 = __msa_fill_w_f32(16.f);
        v4f32 _v8 = __msa_fill_w_f32(8.f);
        v4f32 _v4 = __msa_fill_w_f32(4.f);
        v4f32 _v2 = __msa_fill_w_f32(2.f);

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 4;
                const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * 4 * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * 4 * 5;
                const float* output0_tm_6 = output0_tm_0 + tiles * 4 * 6;
                const float* output0_tm_7 = output0_tm_0 + tiles * 4 * 7;

                float* output0 = out0.row<float>(i * 6) + (j * 6) * 4;

                for (int m = 0; m < 8; m++)
                {
                    v4f32 _out0tm0 = (v4f32)__msa_ld_w(output0_tm_0, 0);
                    v4f32 _out0tm1 = (v4f32)__msa_ld_w(output0_tm_1, 0);
                    v4f32 _out0tm2 = (v4f32)__msa_ld_w(output0_tm_2, 0);
                    v4f32 _out0tm3 = (v4f32)__msa_ld_w(output0_tm_3, 0);
                    v4f32 _out0tm4 = (v4f32)__msa_ld_w(output0_tm_4, 0);
                    v4f32 _out0tm5 = (v4f32)__msa_ld_w(output0_tm_5, 0);
                    v4f32 _out0tm6 = (v4f32)__msa_ld_w(output0_tm_6, 0);
                    v4f32 _out0tm7 = (v4f32)__msa_ld_w(output0_tm_7, 0);

                    v4f32 _tmp024a = __msa_fadd_w(_out0tm1, _out0tm2);
                    v4f32 _tmp135a = __msa_fsub_w(_out0tm1, _out0tm2);

                    v4f32 _tmp024b = __msa_fadd_w(_out0tm3, _out0tm4);
                    v4f32 _tmp135b = __msa_fsub_w(_out0tm3, _out0tm4);

                    v4f32 _tmp024c = __msa_fadd_w(_out0tm5, _out0tm6);
                    v4f32 _tmp135c = __msa_fsub_w(_out0tm5, _out0tm6);

                    v4f32 _tmp0m = __msa_fadd_w(__msa_fadd_w(_out0tm0, _tmp024a), __msa_fmadd_w(_tmp024b, _v32, _tmp024c));
                    v4f32 _tmp2m = __msa_fmadd_w(__msa_fmadd_w(_tmp024a, _v4, _tmp024b), _v8, _tmp024c);
                    v4f32 _tmp4m = __msa_fmadd_w(__msa_fmadd_w(_tmp024a, _v16, _tmp024b), _v2, _tmp024c);
                    __msa_st_w((v4i32)_tmp0m, tmp[0][m], 0);
                    __msa_st_w((v4i32)_tmp2m, tmp[2][m], 0);
                    __msa_st_w((v4i32)_tmp4m, tmp[4][m], 0);

                    v4f32 _tmp1m = __msa_fmadd_w(__msa_fmadd_w(_tmp135a, _v2, _tmp135b), _v16, _tmp135c);
                    v4f32 _tmp3m = __msa_fmadd_w(__msa_fmadd_w(_tmp135a, _v8, _tmp135b), _v4, _tmp135c);
                    v4f32 _tmp5m = __msa_fadd_w(__msa_fadd_w(_out0tm7, _tmp135a), __msa_fmadd_w(_tmp135c, _v32, _tmp135b));
                    __msa_st_w((v4i32)_tmp1m, tmp[1][m], 0);
                    __msa_st_w((v4i32)_tmp3m, tmp[3][m], 0);
                    __msa_st_w((v4i32)_tmp5m, tmp[5][m], 0);

                    output0_tm_0 += tiles * 4 * 8;
                    output0_tm_1 += tiles * 4 * 8;
                    output0_tm_2 += tiles * 4 * 8;
                    output0_tm_3 += tiles * 4 * 8;
                    output0_tm_4 += tiles * 4 * 8;
                    output0_tm_5 += tiles * 4 * 8;
                    output0_tm_6 += tiles * 4 * 8;
                    output0_tm_7 += tiles * 4 * 8;
                }

                for (int m = 0; m < 6; m++)
                {
                    v4f32 _tmp00 = (v4f32)__msa_ld_w(tmp[m][0], 0);
                    v4f32 _tmp01 = (v4f32)__msa_ld_w(tmp[m][1], 0);
                    v4f32 _tmp02 = (v4f32)__msa_ld_w(tmp[m][2], 0);
                    v4f32 _tmp03 = (v4f32)__msa_ld_w(tmp[m][3], 0);
                    v4f32 _tmp04 = (v4f32)__msa_ld_w(tmp[m][4], 0);
                    v4f32 _tmp05 = (v4f32)__msa_ld_w(tmp[m][5], 0);
                    v4f32 _tmp06 = (v4f32)__msa_ld_w(tmp[m][6], 0);
                    v4f32 _tmp07 = (v4f32)__msa_ld_w(tmp[m][7], 0);

                    v4f32 _tmp024a = __msa_fadd_w(_tmp01, _tmp02);
                    v4f32 _tmp135a = __msa_fsub_w(_tmp01, _tmp02);

                    v4f32 _tmp024b = __msa_fadd_w(_tmp03, _tmp04);
                    v4f32 _tmp135b = __msa_fsub_w(_tmp03, _tmp04);

                    v4f32 _tmp024c = __msa_fadd_w(_tmp05, _tmp06);
                    v4f32 _tmp135c = __msa_fsub_w(_tmp05, _tmp06);

                    v4f32 _out00 = __msa_fadd_w(_bias0, __msa_fadd_w(__msa_fadd_w(_tmp00, _tmp024a), __msa_fmadd_w(_tmp024b, _v32, _tmp024c)));
                    v4f32 _out02 = __msa_fadd_w(_bias0, __msa_fmadd_w(__msa_fmadd_w(_tmp024a, _v4, _tmp024b), _v8, _tmp024c));
                    v4f32 _out04 = __msa_fadd_w(_bias0, __msa_fmadd_w(__msa_fmadd_w(_tmp024a, _v16, _tmp024b), _v2, _tmp024c));
                    __msa_st_w((v4i32)_out00, output0, 0);
                    __msa_st_w((v4i32)_out02, output0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_out04, output0 + 4 * 4, 0);

                    v4f32 _out01 = __msa_fadd_w(_bias0, __msa_fmadd_w(__msa_fmadd_w(_tmp135a, _v2, _tmp135b), _v16, _tmp135c));
                    v4f32 _out03 = __msa_fadd_w(_bias0, __msa_fmadd_w(__msa_fmadd_w(_tmp135a, _v8, _tmp135b), _v4, _tmp135c));
                    v4f32 _out05 = __msa_fadd_w(_bias0, __msa_fadd_w(__msa_fadd_w(_tmp07, _tmp135a), __msa_fmadd_w(_tmp135c, _v32, _tmp135b)));
                    __msa_st_w((v4i32)_out01, output0 + 4, 0);
                    __msa_st_w((v4i32)_out03, output0 + 4 * 3, 0);
                    __msa_st_w((v4i32)_out05, output0 + 4 * 5, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_input_pack4_msa(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        v4f32 _vm5 = __msa_fill_w_f32(-5.f);
        v4f32 _vm4 = __msa_fill_w_f32(-4.f);
        v4f32 _v4 = __msa_fill_w_f32(4.f);
        v4f32 _vm2 = __msa_fill_w_f32(-2.f);
        v4f32 _v2 = __msa_fill_w_f32(2.f);

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 4) + (j * 4) * 4;

                for (int m = 0; m < 6; m++)
                {
                    v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                    v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                    v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                    v4f32 _r04 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);
                    v4f32 _r05 = (v4f32)__msa_ld_w(r0 + 4 * 5, 0);

                    v4f32 _tmp0m = __msa_fmadd_w(__msa_fmadd_w(_r04, _v4, _r00), _vm5, _r02);
                    v4f32 _tmp1m = __msa_fmadd_w(__msa_fadd_w(_r04, _r03), _vm4, __msa_fadd_w(_r01, _r02));
                    v4f32 _tmp2m = __msa_fmadd_w(__msa_fsub_w(_r04, _r03), _v4, __msa_fsub_w(_r01, _r02));
                    v4f32 _tmp3m = __msa_fmadd_w(__msa_fsub_w(_r04, _r02), _vm2, __msa_fsub_w(_r01, _r03));
                    v4f32 _tmp4m = __msa_fmadd_w(__msa_fsub_w(_r04, _r02), _v2, __msa_fsub_w(_r01, _r03));
                    v4f32 _tmp5m = __msa_fmadd_w(__msa_fmadd_w(_r05, _v4, _r01), _vm5, _r03);

                    __msa_st_w((v4i32)_tmp0m, tmp[0][m], 0);
                    __msa_st_w((v4i32)_tmp1m, tmp[1][m], 0);
                    __msa_st_w((v4i32)_tmp2m, tmp[2][m], 0);
                    __msa_st_w((v4i32)_tmp3m, tmp[3][m], 0);
                    __msa_st_w((v4i32)_tmp4m, tmp[4][m], 0);
                    __msa_st_w((v4i32)_tmp5m, tmp[5][m], 0);

                    r0 += w * 4;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 4;
                float* r0_tm_1 = r0_tm_0 + tiles * 4;
                float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * 4 * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * 4 * 5;

                for (int m = 0; m < 6; m++)
                {
                    v4f32 _tmp00 = (v4f32)__msa_ld_w(tmp[m][0], 0);
                    v4f32 _tmp01 = (v4f32)__msa_ld_w(tmp[m][1], 0);
                    v4f32 _tmp02 = (v4f32)__msa_ld_w(tmp[m][2], 0);
                    v4f32 _tmp03 = (v4f32)__msa_ld_w(tmp[m][3], 0);
                    v4f32 _tmp04 = (v4f32)__msa_ld_w(tmp[m][4], 0);
                    v4f32 _tmp05 = (v4f32)__msa_ld_w(tmp[m][5], 0);

                    v4f32 _r0tm0 = __msa_fmadd_w(__msa_fmadd_w(_tmp04, _v4, _tmp00), _vm5, _tmp02);
                    v4f32 _r0tm1 = __msa_fmadd_w(__msa_fadd_w(_tmp04, _tmp03), _vm4, __msa_fadd_w(_tmp01, _tmp02));
                    v4f32 _r0tm2 = __msa_fmadd_w(__msa_fsub_w(_tmp04, _tmp03), _v4, __msa_fsub_w(_tmp01, _tmp02));
                    v4f32 _r0tm3 = __msa_fmadd_w(__msa_fsub_w(_tmp04, _tmp02), _vm2, __msa_fsub_w(_tmp01, _tmp03));
                    v4f32 _r0tm4 = __msa_fmadd_w(__msa_fsub_w(_tmp04, _tmp02), _v2, __msa_fsub_w(_tmp01, _tmp03));
                    v4f32 _r0tm5 = __msa_fmadd_w(__msa_fmadd_w(_tmp05, _v4, _tmp01), _vm5, _tmp03);

                    __msa_st_w((v4i32)_r0tm0, r0_tm_0, 0);
                    __msa_st_w((v4i32)_r0tm1, r0_tm_1, 0);
                    __msa_st_w((v4i32)_r0tm2, r0_tm_2, 0);
                    __msa_st_w((v4i32)_r0tm3, r0_tm_3, 0);
                    __msa_st_w((v4i32)_r0tm4, r0_tm_4, 0);
                    __msa_st_w((v4i32)_r0tm5, r0_tm_5, 0);

                    r0_tm_0 += tiles * 4 * 6;
                    r0_tm_1 += tiles * 4 * 6;
                    r0_tm_2 += tiles * 4 * 6;
                    r0_tm_3 += tiles * 4 * 6;
                    r0_tm_4 += tiles * 4 * 6;
                    r0_tm_5 += tiles * 4 * 6;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_pack4_msa(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        v4f32 _bias0 = biasptr ? (v4f32)__msa_ld_w(biasptr + p * 4, 0) : (v4f32)__msa_fill_w(0);

        float tmp[4][6][4];

        v4f32 _v2 = __msa_fill_w_f32(2.f);
        v4f32 _v4 = __msa_fill_w_f32(4.f);
        v4f32 _v8 = __msa_fill_w_f32(8.f);

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 4;
                const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * 4 * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * 4 * 5;

                float* output0 = out0.row<float>(i * 4) + (j * 4) * 4;

                for (int m = 0; m < 6; m++)
                {
                    v4f32 _out0tm0 = (v4f32)__msa_ld_w(output0_tm_0, 0);
                    v4f32 _out0tm1 = (v4f32)__msa_ld_w(output0_tm_1, 0);
                    v4f32 _out0tm2 = (v4f32)__msa_ld_w(output0_tm_2, 0);
                    v4f32 _out0tm3 = (v4f32)__msa_ld_w(output0_tm_3, 0);
                    v4f32 _out0tm4 = (v4f32)__msa_ld_w(output0_tm_4, 0);
                    v4f32 _out0tm5 = (v4f32)__msa_ld_w(output0_tm_5, 0);

                    v4f32 _tmp02a = __msa_fadd_w(_out0tm1, _out0tm2);
                    v4f32 _tmp13a = __msa_fsub_w(_out0tm1, _out0tm2);

                    v4f32 _tmp02b = __msa_fadd_w(_out0tm3, _out0tm4);
                    v4f32 _tmp13b = __msa_fsub_w(_out0tm3, _out0tm4);

                    v4f32 _tmp0m = __msa_fadd_w(__msa_fadd_w(_out0tm0, _tmp02a), _tmp02b);
                    v4f32 _tmp1m = __msa_fmadd_w(_tmp13a, _v2, _tmp13b);
                    v4f32 _tmp2m = __msa_fmadd_w(_tmp02a, _v4, _tmp02b);
                    v4f32 _tmp3m = __msa_fmadd_w(__msa_fadd_w(_out0tm5, _tmp13a), _v8, _tmp13b);

                    __msa_st_w((v4i32)_tmp0m, tmp[0][m], 0);
                    __msa_st_w((v4i32)_tmp1m, tmp[1][m], 0);
                    __msa_st_w((v4i32)_tmp2m, tmp[2][m], 0);
                    __msa_st_w((v4i32)_tmp3m, tmp[3][m], 0);

                    output0_tm_0 += tiles * 4 * 6;
                    output0_tm_1 += tiles * 4 * 6;
                    output0_tm_2 += tiles * 4 * 6;
                    output0_tm_3 += tiles * 4 * 6;
                    output0_tm_4 += tiles * 4 * 6;
                    output0_tm_5 += tiles * 4 * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    v4f32 _tmp00 = (v4f32)__msa_ld_w(tmp[m][0], 0);
                    v4f32 _tmp01 = (v4f32)__msa_ld_w(tmp[m][1], 0);
                    v4f32 _tmp02 = (v4f32)__msa_ld_w(tmp[m][2], 0);
                    v4f32 _tmp03 = (v4f32)__msa_ld_w(tmp[m][3], 0);
                    v4f32 _tmp04 = (v4f32)__msa_ld_w(tmp[m][4], 0);
                    v4f32 _tmp05 = (v4f32)__msa_ld_w(tmp[m][5], 0);

                    v4f32 _tmp02a = __msa_fadd_w(_tmp01, _tmp02);
                    v4f32 _tmp13a = __msa_fsub_w(_tmp01, _tmp02);

                    v4f32 _tmp02b = __msa_fadd_w(_tmp03, _tmp04);
                    v4f32 _tmp13b = __msa_fsub_w(_tmp03, _tmp04);

                    v4f32 _out00 = __msa_fadd_w(_bias0, __msa_fadd_w(__msa_fadd_w(_tmp00, _tmp02a), _tmp02b));
                    v4f32 _out01 = __msa_fadd_w(_bias0, __msa_fmadd_w(_tmp13a, _v2, _tmp13b));
                    v4f32 _out02 = __msa_fadd_w(_bias0, __msa_fmadd_w(_tmp02a, _v4, _tmp02b));
                    v4f32 _out03 = __msa_fadd_w(_bias0, __msa_fmadd_w(__msa_fadd_w(_tmp05, _tmp13a), _v8, _tmp13b));

                    __msa_st_w((v4i32)_out00, output0, 0);
                    __msa_st_w((v4i32)_out01, output0 + 4, 0);
                    __msa_st_w((v4i32)_out02, output0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_out03, output0 + 4 * 3, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_pack4_msa(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 2;
    const int h_tiles = (h - 2) / 2;
    const int tiles = w_tiles * h_tiles;

    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    // 0 = r00 - r02
    // 1 = r01 + r02
    // 2 = r02 - r01
    // 3 = r03 - r01

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        float tmp[4][4][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 2) + (j * 2) * 4;

                for (int m = 0; m < 4; m++)
                {
                    v4f32 _r00 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r01 = (v4f32)__msa_ld_w(r0 + 4, 0);
                    v4f32 _r02 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                    v4f32 _r03 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);

                    v4f32 _tmp0m = __msa_fsub_w(_r00, _r02);
                    v4f32 _tmp1m = __msa_fadd_w(_r01, _r02);
                    v4f32 _tmp2m = __msa_fsub_w(_r02, _r01);
                    v4f32 _tmp3m = __msa_fsub_w(_r03, _r01);

                    __msa_st_w((v4i32)_tmp0m, tmp[0][m], 0);
                    __msa_st_w((v4i32)_tmp1m, tmp[1][m], 0);
                    __msa_st_w((v4i32)_tmp2m, tmp[2][m], 0);
                    __msa_st_w((v4i32)_tmp3m, tmp[3][m], 0);

                    r0 += w * 4;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 4;
                float* r0_tm_1 = r0_tm_0 + tiles * 4;
                float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;

                for (int m = 0; m < 4; m++)
                {
                    v4f32 _tmp00 = (v4f32)__msa_ld_w(tmp[m][0], 0);
                    v4f32 _tmp01 = (v4f32)__msa_ld_w(tmp[m][1], 0);
                    v4f32 _tmp02 = (v4f32)__msa_ld_w(tmp[m][2], 0);
                    v4f32 _tmp03 = (v4f32)__msa_ld_w(tmp[m][3], 0);

                    v4f32 _r0tm0 = __msa_fsub_w(_tmp00, _tmp02);
                    v4f32 _r0tm1 = __msa_fadd_w(_tmp01, _tmp02);
                    v4f32 _r0tm2 = __msa_fsub_w(_tmp02, _tmp01);
                    v4f32 _r0tm3 = __msa_fsub_w(_tmp03, _tmp01);

                    __msa_st_w((v4i32)_r0tm0, r0_tm_0, 0);
                    __msa_st_w((v4i32)_r0tm1, r0_tm_1, 0);
                    __msa_st_w((v4i32)_r0tm2, r0_tm_2, 0);
                    __msa_st_w((v4i32)_r0tm3, r0_tm_3, 0);

                    r0_tm_0 += tiles * 4 * 4;
                    r0_tm_1 += tiles * 4 * 4;
                    r0_tm_2 += tiles * 4 * 4;
                    r0_tm_3 += tiles * 4 * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_pack4_msa(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        v4f32 _bias0 = biasptr ? (v4f32)__msa_ld_w(biasptr + p * 4, 0) : (v4f32)__msa_fill_w(0);

        float tmp[2][4][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 4;
                const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;

                float* output0 = out0.row<float>(i * 2) + (j * 2) * 4;

                for (int m = 0; m < 4; m++)
                {
                    v4f32 _out0tm0 = (v4f32)__msa_ld_w(output0_tm_0, 0);
                    v4f32 _out0tm1 = (v4f32)__msa_ld_w(output0_tm_1, 0);
                    v4f32 _out0tm2 = (v4f32)__msa_ld_w(output0_tm_2, 0);
                    v4f32 _out0tm3 = (v4f32)__msa_ld_w(output0_tm_3, 0);

                    v4f32 _tmp0m = __msa_fadd_w(__msa_fadd_w(_out0tm0, _out0tm1), _out0tm2);
                    v4f32 _tmp1m = __msa_fadd_w(__msa_fsub_w(_out0tm1, _out0tm2), _out0tm3);

                    __msa_st_w((v4i32)_tmp0m, tmp[0][m], 0);
                    __msa_st_w((v4i32)_tmp1m, tmp[1][m], 0);

                    output0_tm_0 += tiles * 4 * 4;
                    output0_tm_1 += tiles * 4 * 4;
                    output0_tm_2 += tiles * 4 * 4;
                    output0_tm_3 += tiles * 4 * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    v4f32 _tmp00 = (v4f32)__msa_ld_w(tmp[m][0], 0);
                    v4f32 _tmp01 = (v4f32)__msa_ld_w(tmp[m][1], 0);
                    v4f32 _tmp02 = (v4f32)__msa_ld_w(tmp[m][2], 0);
                    v4f32 _tmp03 = (v4f32)__msa_ld_w(tmp[m][3], 0);

                    v4f32 _out00 = __msa_fadd_w(_bias0, __msa_fadd_w(__msa_fadd_w(_tmp00, _tmp01), _tmp02));
                    v4f32 _out01 = __msa_fadd_w(_bias0, __msa_fadd_w(__msa_fsub_w(_tmp01, _tmp02), _tmp03));

                    __msa_st_w((v4i32)_out00, output0, 0);
                    __msa_st_w((v4i32)_out01, output0 + 4, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}
