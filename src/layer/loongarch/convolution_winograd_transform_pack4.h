// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void conv3x3s1_winograd63_transform_input_pack4_lsx(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        __m128 _v5_25 = __lsx_vreplfr2vr_s(5.25f);
        __m128 _vm4_25 = __lsx_vreplfr2vr_s(-4.25f);
        __m128 _vm1_25 = __lsx_vreplfr2vr_s(-1.25f);
        __m128 _v0_25 = __lsx_vreplfr2vr_s(0.25f);
        __m128 _vm2_5 = __lsx_vreplfr2vr_s(-2.5f);
        __m128 _v0_5 = __lsx_vreplfr2vr_s(0.5f);
        __m128 _v2 = __lsx_vreplfr2vr_s(2.f);
        __m128 _v4 = __lsx_vreplfr2vr_s(4.f);

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 6) + (j * 6) * 4;

                for (int m = 0; m < 8; m++)
                {
                    __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                    __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                    __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                    __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);
                    __m128 _r04 = (__m128)__lsx_vld(r0 + 4 * 4, 0);
                    __m128 _r05 = (__m128)__lsx_vld(r0 + 4 * 5, 0);
                    __m128 _r06 = (__m128)__lsx_vld(r0 + 4 * 6, 0);
                    __m128 _r07 = (__m128)__lsx_vld(r0 + 4 * 7, 0);

                    __m128 _tmp0m = __lsx_vfmadd_s(__lsx_vfsub_s(_r04, _r02), _v5_25, __lsx_vfsub_s(_r00, _r06));
                    __m128 _tmp7m = __lsx_vfmadd_s(__lsx_vfsub_s(_r03, _r05), _v5_25, __lsx_vfsub_s(_r07, _r01));
                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp7m, tmp[7][m], 0);

                    __m128 _tmp12a = __lsx_vfmadd_s(_r04, _vm4_25, __lsx_vfadd_s(_r02, _r06));
                    __m128 _tmp12b = __lsx_vfmadd_s(_r03, _vm4_25, __lsx_vfadd_s(_r01, _r05));

                    __m128 _tmp1m = __lsx_vfadd_s(_tmp12a, _tmp12b);
                    __m128 _tmp2m = __lsx_vfsub_s(_tmp12a, _tmp12b);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);

                    __m128 _tmp34a = __lsx_vfmadd_s(_r04, _vm1_25, __lsx_vfmadd_s(_r02, _v0_25, _r06));
                    __m128 _tmp34b = __lsx_vfmadd_s(_r05, _v2, __lsx_vfmadd_s(_r03, _vm2_5, __lsx_vfmul_s(_r01, _v0_5)));

                    __m128 _tmp3m = __lsx_vfadd_s(_tmp34a, _tmp34b);
                    __m128 _tmp4m = __lsx_vfsub_s(_tmp34a, _tmp34b);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);
                    __lsx_vst(_tmp4m, tmp[4][m], 0);

                    __m128 _tmp56a = __lsx_vfmadd_s(__lsx_vfmadd_s(_r04, _vm1_25, _r02), _v4, _r06);
                    __m128 _tmp56b = __lsx_vfmadd_s(_r05, _v0_5, __lsx_vfmadd_s(_r03, _vm2_5, __lsx_vfmul_s(_r01, _v2)));

                    __m128 _tmp5m = __lsx_vfadd_s(_tmp56a, _tmp56b);
                    __m128 _tmp6m = __lsx_vfsub_s(_tmp56a, _tmp56b);
                    __lsx_vst(_tmp5m, tmp[5][m], 0);
                    __lsx_vst(_tmp6m, tmp[6][m], 0);

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
                    __m128 _tmp00 = (__m128)__lsx_vld(tmp[m][0], 0);
                    __m128 _tmp01 = (__m128)__lsx_vld(tmp[m][1], 0);
                    __m128 _tmp02 = (__m128)__lsx_vld(tmp[m][2], 0);
                    __m128 _tmp03 = (__m128)__lsx_vld(tmp[m][3], 0);
                    __m128 _tmp04 = (__m128)__lsx_vld(tmp[m][4], 0);
                    __m128 _tmp05 = (__m128)__lsx_vld(tmp[m][5], 0);
                    __m128 _tmp06 = (__m128)__lsx_vld(tmp[m][6], 0);
                    __m128 _tmp07 = (__m128)__lsx_vld(tmp[m][7], 0);

                    __m128 _r0tm0 = __lsx_vfmadd_s(__lsx_vfsub_s(_tmp04, _tmp02), _v5_25, __lsx_vfsub_s(_tmp00, _tmp06));
                    __m128 _r0tm7 = __lsx_vfmadd_s(__lsx_vfsub_s(_tmp03, _tmp05), _v5_25, __lsx_vfsub_s(_tmp07, _tmp01));

                    __m128 _tmp12a = __lsx_vfmadd_s(_tmp04, _vm4_25, __lsx_vfadd_s(_tmp02, _tmp06));
                    __m128 _tmp12b = __lsx_vfmadd_s(_tmp03, _vm4_25, __lsx_vfadd_s(_tmp01, _tmp05));

                    __m128 _r0tm1 = __lsx_vfadd_s(_tmp12a, _tmp12b);
                    __m128 _r0tm2 = __lsx_vfsub_s(_tmp12a, _tmp12b);

                    __m128 _tmp34a = __lsx_vfmadd_s(_tmp04, _vm1_25, __lsx_vfmadd_s(_tmp02, _v0_25, _tmp06));
                    __m128 _tmp34b = __lsx_vfmadd_s(_tmp05, _v2, __lsx_vfmadd_s(_tmp03, _vm2_5, __lsx_vfmul_s(_tmp01, _v0_5)));

                    __m128 _r0tm3 = __lsx_vfadd_s(_tmp34a, _tmp34b);
                    __m128 _r0tm4 = __lsx_vfsub_s(_tmp34a, _tmp34b);

                    __m128 _tmp56a = __lsx_vfmadd_s(__lsx_vfmadd_s(_tmp04, _vm1_25, _tmp02), _v4, _tmp06);
                    __m128 _tmp56b = __lsx_vfmadd_s(_tmp05, _v0_5, __lsx_vfmadd_s(_tmp03, _vm2_5, __lsx_vfmul_s(_tmp01, _v2)));

                    __m128 _r0tm5 = __lsx_vfadd_s(_tmp56a, _tmp56b);
                    __m128 _r0tm6 = __lsx_vfsub_s(_tmp56a, _tmp56b);

                    __lsx_vst(_r0tm0, r0_tm_0, 0);
                    __lsx_vst(_r0tm1, r0_tm_1, 0);
                    __lsx_vst(_r0tm2, r0_tm_2, 0);
                    __lsx_vst(_r0tm3, r0_tm_3, 0);
                    __lsx_vst(_r0tm4, r0_tm_4, 0);
                    __lsx_vst(_r0tm5, r0_tm_5, 0);
                    __lsx_vst(_r0tm6, r0_tm_6, 0);
                    __lsx_vst(_r0tm7, r0_tm_7, 0);

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

static void conv3x3s1_winograd63_transform_output_pack4_lsx(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        __m128 _bias0 = biasptr ? (__m128)__lsx_vld(biasptr + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);

        float tmp[6][8][4];

        __m128 _v32 = __lsx_vreplfr2vr_s(32.f);
        __m128 _v16 = __lsx_vreplfr2vr_s(16.f);
        __m128 _v8 = __lsx_vreplfr2vr_s(8.f);
        __m128 _v4 = __lsx_vreplfr2vr_s(4.f);
        __m128 _v2 = __lsx_vreplfr2vr_s(2.f);

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
                    __m128 _out0tm0 = (__m128)__lsx_vld(output0_tm_0, 0);
                    __m128 _out0tm1 = (__m128)__lsx_vld(output0_tm_1, 0);
                    __m128 _out0tm2 = (__m128)__lsx_vld(output0_tm_2, 0);
                    __m128 _out0tm3 = (__m128)__lsx_vld(output0_tm_3, 0);
                    __m128 _out0tm4 = (__m128)__lsx_vld(output0_tm_4, 0);
                    __m128 _out0tm5 = (__m128)__lsx_vld(output0_tm_5, 0);
                    __m128 _out0tm6 = (__m128)__lsx_vld(output0_tm_6, 0);
                    __m128 _out0tm7 = (__m128)__lsx_vld(output0_tm_7, 0);

                    __m128 _tmp024a = __lsx_vfadd_s(_out0tm1, _out0tm2);
                    __m128 _tmp135a = __lsx_vfsub_s(_out0tm1, _out0tm2);

                    __m128 _tmp024b = __lsx_vfadd_s(_out0tm3, _out0tm4);
                    __m128 _tmp135b = __lsx_vfsub_s(_out0tm3, _out0tm4);

                    __m128 _tmp024c = __lsx_vfadd_s(_out0tm5, _out0tm6);
                    __m128 _tmp135c = __lsx_vfsub_s(_out0tm5, _out0tm6);

                    __m128 _tmp0m = __lsx_vfadd_s(__lsx_vfadd_s(_out0tm0, _tmp024a), __lsx_vfmadd_s(_tmp024c, _v32, _tmp024b));
                    __m128 _tmp2m = __lsx_vfmadd_s(_tmp024c, _v8, __lsx_vfmadd_s(_tmp024b, _v4, _tmp024a));
                    __m128 _tmp4m = __lsx_vfmadd_s(_tmp024c, _v2, __lsx_vfmadd_s(_tmp024b, _v16, _tmp024a));
                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp4m, tmp[4][m], 0);

                    __m128 _tmp1m = __lsx_vfmadd_s(_tmp135c, _v16, __lsx_vfmadd_s(_tmp135b, _v2, _tmp135a));
                    __m128 _tmp3m = __lsx_vfmadd_s(_tmp135c, _v4, __lsx_vfmadd_s(_tmp135b, _v8, _tmp135a));
                    __m128 _tmp5m = __lsx_vfadd_s(__lsx_vfadd_s(_out0tm7, _tmp135a), __lsx_vfmadd_s(_tmp135b, _v32, _tmp135c));
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);
                    __lsx_vst(_tmp5m, tmp[5][m], 0);

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
                    __m128 _tmp00 = (__m128)__lsx_vld(tmp[m][0], 0);
                    __m128 _tmp01 = (__m128)__lsx_vld(tmp[m][1], 0);
                    __m128 _tmp02 = (__m128)__lsx_vld(tmp[m][2], 0);
                    __m128 _tmp03 = (__m128)__lsx_vld(tmp[m][3], 0);
                    __m128 _tmp04 = (__m128)__lsx_vld(tmp[m][4], 0);
                    __m128 _tmp05 = (__m128)__lsx_vld(tmp[m][5], 0);
                    __m128 _tmp06 = (__m128)__lsx_vld(tmp[m][6], 0);
                    __m128 _tmp07 = (__m128)__lsx_vld(tmp[m][7], 0);

                    __m128 _tmp024a = __lsx_vfadd_s(_tmp01, _tmp02);
                    __m128 _tmp135a = __lsx_vfsub_s(_tmp01, _tmp02);

                    __m128 _tmp024b = __lsx_vfadd_s(_tmp03, _tmp04);
                    __m128 _tmp135b = __lsx_vfsub_s(_tmp03, _tmp04);

                    __m128 _tmp024c = __lsx_vfadd_s(_tmp05, _tmp06);
                    __m128 _tmp135c = __lsx_vfsub_s(_tmp05, _tmp06);

                    __m128 _out00 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_tmp00, _tmp024a), __lsx_vfmadd_s(_tmp024c, _v32, _tmp024b)));
                    __m128 _out02 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp024c, _v8, __lsx_vfmadd_s(_tmp024b, _v4, _tmp024a)));
                    __m128 _out04 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp024c, _v2, __lsx_vfmadd_s(_tmp024b, _v16, _tmp024a)));
                    __lsx_vst(_out00, output0, 0);
                    __lsx_vst(_out02, output0 + 4 * 2, 0);
                    __lsx_vst(_out04, output0 + 4 * 4, 0);

                    __m128 _out01 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp135c, _v16, __lsx_vfmadd_s(_tmp135b, _v2, _tmp135a)));
                    __m128 _out03 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp135c, _v4, __lsx_vfmadd_s(_tmp135b, _v8, _tmp135a)));
                    __m128 _out05 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_tmp07, _tmp135a), __lsx_vfmadd_s(_tmp135b, _v32, _tmp135c)));
                    __lsx_vst(_out01, output0 + 4, 0);
                    __lsx_vst(_out03, output0 + 4 * 3, 0);
                    __lsx_vst(_out05, output0 + 4 * 5, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_input_pack4_lsx(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        __m128 _vm5 = __lsx_vreplfr2vr_s(-5.f);
        __m128 _vm4 = __lsx_vreplfr2vr_s(-4.f);
        __m128 _v4 = __lsx_vreplfr2vr_s(4.f);
        __m128 _vm2 = __lsx_vreplfr2vr_s(-2.f);
        __m128 _v2 = __lsx_vreplfr2vr_s(2.f);

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 4) + (j * 4) * 4;

                for (int m = 0; m < 6; m++)
                {
                    __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                    __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                    __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                    __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);
                    __m128 _r04 = (__m128)__lsx_vld(r0 + 4 * 4, 0);
                    __m128 _r05 = (__m128)__lsx_vld(r0 + 4 * 5, 0);

                    __m128 _tmp0m = __lsx_vfmadd_s(_r02, _vm5, __lsx_vfmadd_s(_r00, _v4, _r04));
                    __m128 _tmp1m = __lsx_vfmadd_s(__lsx_vfadd_s(_r01, _r02), _vm4, __lsx_vfadd_s(_r04, _r03));
                    __m128 _tmp2m = __lsx_vfmadd_s(__lsx_vfsub_s(_r01, _r02), _v4, __lsx_vfsub_s(_r04, _r03));
                    __m128 _tmp3m = __lsx_vfmadd_s(__lsx_vfsub_s(_r01, _r03), _vm2, __lsx_vfsub_s(_r04, _r02));
                    __m128 _tmp4m = __lsx_vfmadd_s(__lsx_vfsub_s(_r01, _r03), _v2, __lsx_vfsub_s(_r04, _r02));
                    __m128 _tmp5m = __lsx_vfmadd_s(_r03, _vm5, __lsx_vfmadd_s(_r01, _v4, _r05));

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);
                    __lsx_vst(_tmp4m, tmp[4][m], 0);
                    __lsx_vst(_tmp5m, tmp[5][m], 0);

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
                    __m128 _tmp00 = (__m128)__lsx_vld(tmp[m][0], 0);
                    __m128 _tmp01 = (__m128)__lsx_vld(tmp[m][1], 0);
                    __m128 _tmp02 = (__m128)__lsx_vld(tmp[m][2], 0);
                    __m128 _tmp03 = (__m128)__lsx_vld(tmp[m][3], 0);
                    __m128 _tmp04 = (__m128)__lsx_vld(tmp[m][4], 0);
                    __m128 _tmp05 = (__m128)__lsx_vld(tmp[m][5], 0);

                    __m128 _r0tm0 = __lsx_vfmadd_s(_tmp02, _vm5, __lsx_vfmadd_s(_tmp00, _v4, _tmp04));
                    __m128 _r0tm1 = __lsx_vfmadd_s(__lsx_vfadd_s(_tmp01, _tmp02), _vm4, __lsx_vfadd_s(_tmp04, _tmp03));
                    __m128 _r0tm2 = __lsx_vfmadd_s(__lsx_vfsub_s(_tmp01, _tmp02), _v4, __lsx_vfsub_s(_tmp04, _tmp03));
                    __m128 _r0tm3 = __lsx_vfmadd_s(__lsx_vfsub_s(_tmp01, _tmp03), _vm2, __lsx_vfsub_s(_tmp04, _tmp02));
                    __m128 _r0tm4 = __lsx_vfmadd_s(__lsx_vfsub_s(_tmp01, _tmp03), _v2, __lsx_vfsub_s(_tmp04, _tmp02));
                    __m128 _r0tm5 = __lsx_vfmadd_s(_tmp03, _vm5, __lsx_vfmadd_s(_tmp01, _v4, _tmp05));

                    __lsx_vst(_r0tm0, r0_tm_0, 0);
                    __lsx_vst(_r0tm1, r0_tm_1, 0);
                    __lsx_vst(_r0tm2, r0_tm_2, 0);
                    __lsx_vst(_r0tm3, r0_tm_3, 0);
                    __lsx_vst(_r0tm4, r0_tm_4, 0);
                    __lsx_vst(_r0tm5, r0_tm_5, 0);

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

static void conv3x3s1_winograd43_transform_output_pack4_lsx(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        __m128 _bias0 = biasptr ? (__m128)__lsx_vld(biasptr + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);

        float tmp[4][6][4];

        __m128 _v2 = __lsx_vreplfr2vr_s(2.f);
        __m128 _v4 = __lsx_vreplfr2vr_s(4.f);
        __m128 _v8 = __lsx_vreplfr2vr_s(8.f);

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
                    __m128 _out0tm0 = (__m128)__lsx_vld(output0_tm_0, 0);
                    __m128 _out0tm1 = (__m128)__lsx_vld(output0_tm_1, 0);
                    __m128 _out0tm2 = (__m128)__lsx_vld(output0_tm_2, 0);
                    __m128 _out0tm3 = (__m128)__lsx_vld(output0_tm_3, 0);
                    __m128 _out0tm4 = (__m128)__lsx_vld(output0_tm_4, 0);
                    __m128 _out0tm5 = (__m128)__lsx_vld(output0_tm_5, 0);

                    __m128 _tmp02a = __lsx_vfadd_s(_out0tm1, _out0tm2);
                    __m128 _tmp13a = __lsx_vfsub_s(_out0tm1, _out0tm2);

                    __m128 _tmp02b = __lsx_vfadd_s(_out0tm3, _out0tm4);
                    __m128 _tmp13b = __lsx_vfsub_s(_out0tm3, _out0tm4);

                    __m128 _tmp0m = __lsx_vfadd_s(__lsx_vfadd_s(_out0tm0, _tmp02a), _tmp02b);
                    __m128 _tmp1m = __lsx_vfmadd_s(_tmp13b, _v2, _tmp13a);
                    __m128 _tmp2m = __lsx_vfmadd_s(_tmp02b, _v4, _tmp02a);
                    __m128 _tmp3m = __lsx_vfmadd_s(_tmp13b, _v8, __lsx_vfadd_s(_out0tm5, _tmp13a));

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);

                    output0_tm_0 += tiles * 4 * 6;
                    output0_tm_1 += tiles * 4 * 6;
                    output0_tm_2 += tiles * 4 * 6;
                    output0_tm_3 += tiles * 4 * 6;
                    output0_tm_4 += tiles * 4 * 6;
                    output0_tm_5 += tiles * 4 * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    __m128 _tmp00 = (__m128)__lsx_vld(tmp[m][0], 0);
                    __m128 _tmp01 = (__m128)__lsx_vld(tmp[m][1], 0);
                    __m128 _tmp02 = (__m128)__lsx_vld(tmp[m][2], 0);
                    __m128 _tmp03 = (__m128)__lsx_vld(tmp[m][3], 0);
                    __m128 _tmp04 = (__m128)__lsx_vld(tmp[m][4], 0);
                    __m128 _tmp05 = (__m128)__lsx_vld(tmp[m][5], 0);

                    __m128 _tmp02a = __lsx_vfadd_s(_tmp01, _tmp02);
                    __m128 _tmp13a = __lsx_vfsub_s(_tmp01, _tmp02);

                    __m128 _tmp02b = __lsx_vfadd_s(_tmp03, _tmp04);
                    __m128 _tmp13b = __lsx_vfsub_s(_tmp03, _tmp04);

                    __m128 _out00 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_tmp00, _tmp02a), _tmp02b));
                    __m128 _out01 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp13b, _v2, _tmp13a));
                    __m128 _out02 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp02b, _v4, _tmp02a));
                    __m128 _out03 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_tmp13b, _v8, __lsx_vfadd_s(_tmp05, _tmp13a)));

                    __lsx_vst(_out00, output0, 0);
                    __lsx_vst(_out01, output0 + 4, 0);
                    __lsx_vst(_out02, output0 + 4 * 2, 0);
                    __lsx_vst(_out03, output0 + 4 * 3, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_pack4_lsx(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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
                    __m128 _r00 = (__m128)__lsx_vld(r0, 0);
                    __m128 _r01 = (__m128)__lsx_vld(r0 + 4, 0);
                    __m128 _r02 = (__m128)__lsx_vld(r0 + 4 * 2, 0);
                    __m128 _r03 = (__m128)__lsx_vld(r0 + 4 * 3, 0);

                    __m128 _tmp0m = __lsx_vfsub_s(_r00, _r02);
                    __m128 _tmp1m = __lsx_vfadd_s(_r01, _r02);
                    __m128 _tmp2m = __lsx_vfsub_s(_r02, _r01);
                    __m128 _tmp3m = __lsx_vfsub_s(_r03, _r01);

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);

                    r0 += w * 4;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 4;
                float* r0_tm_1 = r0_tm_0 + tiles * 4;
                float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;

                for (int m = 0; m < 4; m++)
                {
                    __m128 _tmp00 = (__m128)__lsx_vld(tmp[m][0], 0);
                    __m128 _tmp01 = (__m128)__lsx_vld(tmp[m][1], 0);
                    __m128 _tmp02 = (__m128)__lsx_vld(tmp[m][2], 0);
                    __m128 _tmp03 = (__m128)__lsx_vld(tmp[m][3], 0);

                    __m128 _r0tm0 = __lsx_vfsub_s(_tmp00, _tmp02);
                    __m128 _r0tm1 = __lsx_vfadd_s(_tmp01, _tmp02);
                    __m128 _r0tm2 = __lsx_vfsub_s(_tmp02, _tmp01);
                    __m128 _r0tm3 = __lsx_vfsub_s(_tmp03, _tmp01);

                    __lsx_vst(_r0tm0, r0_tm_0, 0);
                    __lsx_vst(_r0tm1, r0_tm_1, 0);
                    __lsx_vst(_r0tm2, r0_tm_2, 0);
                    __lsx_vst(_r0tm3, r0_tm_3, 0);

                    r0_tm_0 += tiles * 4 * 4;
                    r0_tm_1 += tiles * 4 * 4;
                    r0_tm_2 += tiles * 4 * 4;
                    r0_tm_3 += tiles * 4 * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_pack4_lsx(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        __m128 _bias0 = biasptr ? (__m128)__lsx_vld(biasptr + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);

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
                    __m128 _out0tm0 = (__m128)__lsx_vld(output0_tm_0, 0);
                    __m128 _out0tm1 = (__m128)__lsx_vld(output0_tm_1, 0);
                    __m128 _out0tm2 = (__m128)__lsx_vld(output0_tm_2, 0);
                    __m128 _out0tm3 = (__m128)__lsx_vld(output0_tm_3, 0);

                    __m128 _tmp0m = __lsx_vfadd_s(__lsx_vfadd_s(_out0tm0, _out0tm1), _out0tm2);
                    __m128 _tmp1m = __lsx_vfadd_s(__lsx_vfsub_s(_out0tm1, _out0tm2), _out0tm3);

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);

                    output0_tm_0 += tiles * 4 * 4;
                    output0_tm_1 += tiles * 4 * 4;
                    output0_tm_2 += tiles * 4 * 4;
                    output0_tm_3 += tiles * 4 * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    __m128 _tmp00 = (__m128)__lsx_vld(tmp[m][0], 0);
                    __m128 _tmp01 = (__m128)__lsx_vld(tmp[m][1], 0);
                    __m128 _tmp02 = (__m128)__lsx_vld(tmp[m][2], 0);
                    __m128 _tmp03 = (__m128)__lsx_vld(tmp[m][3], 0);

                    __m128 _out00 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_tmp00, _tmp01), _tmp02));
                    __m128 _out01 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfsub_s(_tmp01, _tmp02), _tmp03));

                    __lsx_vst(_out00, output0, 0);
                    __lsx_vst(_out01, output0 + 4, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}
