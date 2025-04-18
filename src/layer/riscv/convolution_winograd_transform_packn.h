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

static void conv3x3s1_winograd63_transform_input_packn_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

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

        // NOTE c99 variable length array
        float tmp[8][8][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 6) + (j * 6) * packn;

                for (int m = 0; m < 8; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(r0 + packn * 5, vl);
                    vfloat32m1_t _r06 = __riscv_vle32_v_f32m1(r0 + packn * 6, vl);
                    vfloat32m1_t _r07 = __riscv_vle32_v_f32m1(r0 + packn * 7, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_r00, _r06, vl), 5.25f, __riscv_vfsub_vv_f32m1(_r04, _r02, vl), vl);
                    vfloat32m1_t _tmp7m = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_r07, _r01, vl), 5.25f, __riscv_vfsub_vv_f32m1(_r03, _r05, vl), vl);
                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[7][m], _tmp7m, vl);

                    vfloat32m1_t _tmp12a = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r02, _r06, vl), -4.25f, _r04, vl);
                    vfloat32m1_t _tmp12b = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r01, _r05, vl), -4.25f, _r03, vl);

                    vfloat32m1_t _tmp1m = __riscv_vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);

                    vfloat32m1_t _tmp34a = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r06, 0.25f, _r02, vl), -1.25f, _r04, vl);
                    vfloat32m1_t _tmp34b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, 0.5f, vl), -2.5f, _r03, vl), 2.f, _r05, vl);

                    vfloat32m1_t _tmp3m = __riscv_vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][m], _tmp4m, vl);

                    vfloat32m1_t _tmp56a = __riscv_vfmacc_vf_f32m1(_r06, 4.f, __riscv_vfmacc_vf_f32m1(_r02, -1.25f, _r04, vl), vl);
                    vfloat32m1_t _tmp56b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, 2.f, vl), -2.5f, _r03, vl), 0.5f, _r05, vl);

                    vfloat32m1_t _tmp5m = __riscv_vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                    vfloat32m1_t _tmp6m = __riscv_vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);
                    __riscv_vse32_v_f32m1(tmp[5][m], _tmp5m, vl);
                    __riscv_vse32_v_f32m1(tmp[6][m], _tmp6m, vl);

                    r0 += w * packn;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * packn;
                float* r0_tm_1 = r0_tm_0 + tiles * packn;
                float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * packn * 5;
                float* r0_tm_6 = r0_tm_0 + tiles * packn * 6;
                float* r0_tm_7 = r0_tm_0 + tiles * packn * 7;

                for (int m = 0; m < 8; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);
                    vfloat32m1_t _tmp06 = __riscv_vle32_v_f32m1(tmp[m][6], vl);
                    vfloat32m1_t _tmp07 = __riscv_vle32_v_f32m1(tmp[m][7], vl);

                    vfloat32m1_t _r0tm0 = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_tmp00, _tmp06, vl), 5.25f, __riscv_vfsub_vv_f32m1(_tmp04, _tmp02, vl), vl);
                    vfloat32m1_t _r0tm7 = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_tmp07, _tmp01, vl), 5.25f, __riscv_vfsub_vv_f32m1(_tmp03, _tmp05, vl), vl);

                    vfloat32m1_t _tmp12a = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_tmp02, _tmp06, vl), -4.25f, _tmp04, vl);
                    vfloat32m1_t _tmp12b = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_tmp01, _tmp05, vl), -4.25f, _tmp03, vl);

                    vfloat32m1_t _r0tm1 = __riscv_vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                    vfloat32m1_t _r0tm2 = __riscv_vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                    vfloat32m1_t _tmp34a = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp06, 0.25f, _tmp02, vl), -1.25f, _tmp04, vl);
                    vfloat32m1_t _tmp34b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp01, 0.5f, vl), -2.5f, _tmp03, vl), 2.f, _tmp05, vl);

                    vfloat32m1_t _r0tm3 = __riscv_vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                    vfloat32m1_t _r0tm4 = __riscv_vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                    vfloat32m1_t _tmp56a = __riscv_vfmacc_vf_f32m1(_tmp06, 4.f, __riscv_vfmacc_vf_f32m1(_tmp02, -1.25f, _tmp04, vl), vl);
                    vfloat32m1_t _tmp56b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp01, 2.f, vl), -2.5f, _tmp03, vl), 0.5f, _tmp05, vl);

                    vfloat32m1_t _r0tm5 = __riscv_vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                    vfloat32m1_t _r0tm6 = __riscv_vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                    __riscv_vse32_v_f32m1(r0_tm_0, _r0tm0, vl);
                    __riscv_vse32_v_f32m1(r0_tm_1, _r0tm1, vl);
                    __riscv_vse32_v_f32m1(r0_tm_2, _r0tm2, vl);
                    __riscv_vse32_v_f32m1(r0_tm_3, _r0tm3, vl);
                    __riscv_vse32_v_f32m1(r0_tm_4, _r0tm4, vl);
                    __riscv_vse32_v_f32m1(r0_tm_5, _r0tm5, vl);
                    __riscv_vse32_v_f32m1(r0_tm_6, _r0tm6, vl);
                    __riscv_vse32_v_f32m1(r0_tm_7, _r0tm7, vl);

                    r0_tm_0 += tiles * packn * 8;
                    r0_tm_1 += tiles * packn * 8;
                    r0_tm_2 += tiles * packn * 8;
                    r0_tm_3 += tiles * packn * 8;
                    r0_tm_4 += tiles * packn * 8;
                    r0_tm_5 += tiles * packn * 8;
                    r0_tm_6 += tiles * packn * 8;
                    r0_tm_7 += tiles * packn * 8;
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_output_packn_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

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

        vfloat32m1_t _bias0 = biasptr ? __riscv_vle32_v_f32m1(biasptr + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

        // NOTE c99 variable length array
        float tmp[6][8][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * packn;
                const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * packn * 5;
                const float* output0_tm_6 = output0_tm_0 + tiles * packn * 6;
                const float* output0_tm_7 = output0_tm_0 + tiles * packn * 7;

                float* output0 = out0.row(i * 6) + (j * 6) * packn;

                for (int m = 0; m < 8; m++)
                {
                    vfloat32m1_t _out0tm0 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _out0tm1 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _out0tm2 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _out0tm3 = __riscv_vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _out0tm4 = __riscv_vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _out0tm5 = __riscv_vle32_v_f32m1(output0_tm_5, vl);
                    vfloat32m1_t _out0tm6 = __riscv_vle32_v_f32m1(output0_tm_6, vl);
                    vfloat32m1_t _out0tm7 = __riscv_vle32_v_f32m1(output0_tm_7, vl);

                    vfloat32m1_t _tmp024a = __riscv_vfadd_vv_f32m1(_out0tm1, _out0tm2, vl);
                    vfloat32m1_t _tmp135a = __riscv_vfsub_vv_f32m1(_out0tm1, _out0tm2, vl);

                    vfloat32m1_t _tmp024b = __riscv_vfadd_vv_f32m1(_out0tm3, _out0tm4, vl);
                    vfloat32m1_t _tmp135b = __riscv_vfsub_vv_f32m1(_out0tm3, _out0tm4, vl);

                    vfloat32m1_t _tmp024c = __riscv_vfadd_vv_f32m1(_out0tm5, _out0tm6, vl);
                    vfloat32m1_t _tmp135c = __riscv_vfsub_vv_f32m1(_out0tm5, _out0tm6, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm0, _tmp024a, vl), __riscv_vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                    vfloat32m1_t _tmp2m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);
                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][m], _tmp4m, vl);

                    vfloat32m1_t _tmp1m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm7, _tmp135a, vl), __riscv_vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                    output0_tm_0 += tiles * packn * 8;
                    output0_tm_1 += tiles * packn * 8;
                    output0_tm_2 += tiles * packn * 8;
                    output0_tm_3 += tiles * packn * 8;
                    output0_tm_4 += tiles * packn * 8;
                    output0_tm_5 += tiles * packn * 8;
                    output0_tm_6 += tiles * packn * 8;
                    output0_tm_7 += tiles * packn * 8;
                }

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);
                    vfloat32m1_t _tmp06 = __riscv_vle32_v_f32m1(tmp[m][6], vl);
                    vfloat32m1_t _tmp07 = __riscv_vle32_v_f32m1(tmp[m][7], vl);

                    vfloat32m1_t _tmp024a = __riscv_vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp135a = __riscv_vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                    vfloat32m1_t _tmp024b = __riscv_vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp135b = __riscv_vfsub_vv_f32m1(_tmp03, _tmp04, vl);

                    vfloat32m1_t _tmp024c = __riscv_vfadd_vv_f32m1(_tmp05, _tmp06, vl);
                    vfloat32m1_t _tmp135c = __riscv_vfsub_vv_f32m1(_tmp05, _tmp06, vl);

                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp00, _tmp024a, vl), __riscv_vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl), vl);
                    vfloat32m1_t _out02 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl), vl);
                    vfloat32m1_t _out04 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl), vl);
                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 4, _out04, vl);

                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl), vl);
                    vfloat32m1_t _out03 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl), vl);
                    vfloat32m1_t _out05 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp07, _tmp135a, vl), __riscv_vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl), vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 3, _out03, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 5, _out05, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_input_packn_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    const float sq2 = 1.41421356237;
    const float sq2_d2 = 1.41421356237 / 2;

    // const float itm[6][6] = {
    //     {1.0f,  0.0f,  -2.5f,  0.0f,  1.0f, 0.0f},
    //     {0.0f, -sq2,   -2.0f,  sq2/2, 1.0f, 0.0f},
    //     {0.0f,  sq2,   -2.0f, -sq2/2, 1.0f, 0.0f},
    //     {0.0f, -sq2/2, -0.5f,  sq2,   1.0f, 0.0f},
    //     {0.0f,  sq2/2, -0.5f, -sq2,   1.0f, 0.0f},
    //     {0.0f,  1.0f,   0.0f,  -2.5f, 0.0f, 1.0f}
    // };

    // 0 =  r00 - 2.5f * r02 + r04
    // 1 = -(sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 2 =  (sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 3 = -(sq2_d2 * r01 - sq2 * r03) + (r04 - 0.5f * r02)
    // 4 =  (sq2_d2 * r01 - sq2 * r03) + (r04 - 0.5f * r02)
    // 5 =  r01 - 2.5f * r03 + r05

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        // NOTE c99 variable length array
        float tmp[6][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(r0 + packn * 5, vl);

                    vfloat32m1_t _tmp01a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2, vl), -sq2_d2, _r03, vl);
                    vfloat32m1_t _tmp01b = __riscv_vfmacc_vf_f32m1(_r04, -2.f, _r02, vl);
                    vfloat32m1_t _tmp23a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2_d2, vl), -sq2, _r03, vl);
                    vfloat32m1_t _tmp23b = __riscv_vfmacc_vf_f32m1(_r04, -0.5f, _r02, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r00, _r04, vl), -2.5f, _r02, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfsub_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfadd_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfsub_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfadd_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r01, _r05, vl), -2.5f, _r03, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                    r0 += w * packn;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * packn;
                float* r0_tm_1 = r0_tm_0 + tiles * packn;
                float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * packn * 5;

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);

                    vfloat32m1_t _tmp01a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2, vl), -sq2_d2, _r03, vl);
                    vfloat32m1_t _tmp01b = __riscv_vfmacc_vf_f32m1(_r04, -2.f, _r02, vl);
                    vfloat32m1_t _tmp23a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2_d2, vl), -sq2, _r03, vl);
                    vfloat32m1_t _tmp23b = __riscv_vfmacc_vf_f32m1(_r04, -0.5f, _r02, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r00, _r04, vl), -2.5f, _r02, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfsub_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfadd_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfsub_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfadd_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r01, _r05, vl), -2.5f, _r03, vl);

                    __riscv_vse32_v_f32m1(r0_tm_0, _tmp0m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_1, _tmp1m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_2, _tmp2m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_3, _tmp3m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_4, _tmp4m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_5, _tmp5m, vl);

                    r0_tm_0 += tiles * packn * 6;
                    r0_tm_1 += tiles * packn * 6;
                    r0_tm_2 += tiles * packn * 6;
                    r0_tm_3 += tiles * packn * 6;
                    r0_tm_4 += tiles * packn * 6;
                    r0_tm_5 += tiles * packn * 6;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_packn_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

    const float sq2 = 1.41421356237;
    const float sq2_m2 = 1.41421356237 * 2;
    const float sq2_d2 = 1.41421356237 / 2;
    const float sq2_d4 = 1.41421356237 / 4;

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,   1.0f,  1.0f,  1.0f,   0.0f},
    //     {0.0f, sq2/2, -sq2/2, sq2,   -sq2,   0.0f},
    //     {0.0f, 0.5f,   0.5f,  2.0f,  2.0f,   0.0f},
    //     {0.0f, sq2/4, -sq2/4, sq2*2, -sq2*2, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) * sq2_d2 + (r03 - r04) * sq2
    // 2 =       (r01 + r02) * 0.5f + (r03 + r04) * 2
    // 3 = r05 + (r01 - r02) * sq2_d4 + (r03 - r04) * sq2_m2

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        vfloat32m1_t _bias0 = biasptr ? __riscv_vle32_v_f32m1(biasptr + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

        // NOTE variable length array
        float tmp[4][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * packn;
                const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * packn * 5;

                float* output0 = out0.row(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(output0_tm_5, vl);

                    vfloat32m1_t _tmp02a = __riscv_vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp02b = __riscv_vfadd_vv_f32m1(_r03, _r04, vl);
                    vfloat32m1_t _tmp13a = __riscv_vfsub_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp13b = __riscv_vfsub_vv_f32m1(_r03, _r04, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_r00, _tmp02a, vl), _tmp02b, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp02a, 0.5f, vl), 2.f, _tmp02b, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r05, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);

                    vfloat32m1_t _tmp02a = __riscv_vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp02b = __riscv_vfadd_vv_f32m1(_r03, _r04, vl);
                    vfloat32m1_t _tmp13a = __riscv_vfsub_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp13b = __riscv_vfsub_vv_f32m1(_r03, _r04, vl);

                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_r00, _tmp02a, vl), _tmp02b, vl), vl);
                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl), vl);
                    vfloat32m1_t _out02 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp02a, 0.5f, vl), 2.f, _tmp02b, vl), vl);
                    vfloat32m1_t _out03 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r05, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl), vl);

                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 3, _out03, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_packn_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

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

        // NOTE c99 variable length array
        float tmp[4][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 2) + (j * 2) * packn;

                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfsub_vv_f32m1(_r00, _r02, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfsub_vv_f32m1(_r02, _r01, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfsub_vv_f32m1(_r03, _r01, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);

                    r0 += w * packn;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * packn;
                float* r0_tm_1 = r0_tm_0 + tiles * packn;
                float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;

                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);

                    vfloat32m1_t _r0tm0 = __riscv_vfsub_vv_f32m1(_tmp00, _tmp02, vl);
                    vfloat32m1_t _r0tm1 = __riscv_vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _r0tm2 = __riscv_vfsub_vv_f32m1(_tmp02, _tmp01, vl);
                    vfloat32m1_t _r0tm3 = __riscv_vfsub_vv_f32m1(_tmp03, _tmp01, vl);

                    __riscv_vse32_v_f32m1(r0_tm_0, _r0tm0, vl);
                    __riscv_vse32_v_f32m1(r0_tm_1, _r0tm1, vl);
                    __riscv_vse32_v_f32m1(r0_tm_2, _r0tm2, vl);
                    __riscv_vse32_v_f32m1(r0_tm_3, _r0tm3, vl);

                    r0_tm_0 += tiles * packn * 4;
                    r0_tm_1 += tiles * packn * 4;
                    r0_tm_2 += tiles * packn * 4;
                    r0_tm_3 += tiles * packn * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_packn_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

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

        vfloat32m1_t _bias0 = biasptr ? __riscv_vle32_v_f32m1(biasptr + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

        // NOTE variable length array
        float tmp[2][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * packn;
                const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;

                float* output0 = out0.row(i * 2) + (j * 2) * packn;

                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _out0tm0 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _out0tm1 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _out0tm2 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _out0tm3 = __riscv_vle32_v_f32m1(output0_tm_3, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm0, _out0tm1, vl), _out0tm2, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(_out0tm1, _out0tm2, vl), _out0tm3, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);

                    output0_tm_0 += tiles * packn * 4;
                    output0_tm_1 += tiles * packn * 4;
                    output0_tm_2 += tiles * packn * 4;
                    output0_tm_3 += tiles * packn * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);

                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp00, _tmp01, vl), _tmp02, vl), vl);
                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(_tmp01, _tmp02, vl), _tmp03, vl), vl);

                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}
