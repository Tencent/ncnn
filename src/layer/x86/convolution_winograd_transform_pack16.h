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

static void conv3x3s1_winograd63_transform_input_pack16_avx512(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[8][8][16];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 6) + (j * 6) * 16;

                for (int m = 0; m < 8; m++)
                {
                    __m512 _r00 = _mm512_load_ps(r0);
                    __m512 _r01 = _mm512_load_ps(r0 + 16);
                    __m512 _r02 = _mm512_load_ps(r0 + 16 * 2);
                    __m512 _r03 = _mm512_load_ps(r0 + 16 * 3);
                    __m512 _r04 = _mm512_load_ps(r0 + 16 * 4);
                    __m512 _r05 = _mm512_load_ps(r0 + 16 * 5);
                    __m512 _r06 = _mm512_load_ps(r0 + 16 * 6);
                    __m512 _r07 = _mm512_load_ps(r0 + 16 * 7);

                    __m512 _tmp0m = _mm512_fmadd_ps(_mm512_set1_ps(5.25f), _mm512_sub_ps(_r04, _r02), _mm512_sub_ps(_r00, _r06));
                    __m512 _tmp7m = _mm512_fmadd_ps(_mm512_set1_ps(5.25f), _mm512_sub_ps(_r03, _r05), _mm512_sub_ps(_r07, _r01));
                    _mm512_store_ps(tmp[0][m], _tmp0m);
                    _mm512_store_ps(tmp[7][m], _tmp7m);

                    __m512 _tmp12a = _mm512_fmadd_ps(_mm512_set1_ps(-4.25f), _r04, _mm512_add_ps(_r02, _r06));
                    __m512 _tmp12b = _mm512_fmadd_ps(_mm512_set1_ps(-4.25f), _r03, _mm512_add_ps(_r01, _r05));

                    __m512 _tmp1m = _mm512_add_ps(_tmp12a, _tmp12b);
                    __m512 _tmp2m = _mm512_sub_ps(_tmp12a, _tmp12b);
                    _mm512_store_ps(tmp[1][m], _tmp1m);
                    _mm512_store_ps(tmp[2][m], _tmp2m);

                    __m512 _tmp34a = _mm512_fmadd_ps(_mm512_set1_ps(-1.25f), _r04, _mm512_fmadd_ps(_mm512_set1_ps(0.25f), _r02, _r06));
                    __m512 _tmp34b = _mm512_fmadd_ps(_mm512_set1_ps(2.f), _r05, _mm512_fmadd_ps(_mm512_set1_ps(-2.5f), _r03, _mm512_mul_ps(_r01, _mm512_set1_ps(0.5f))));

                    __m512 _tmp3m = _mm512_add_ps(_tmp34a, _tmp34b);
                    __m512 _tmp4m = _mm512_sub_ps(_tmp34a, _tmp34b);
                    _mm512_store_ps(tmp[3][m], _tmp3m);
                    _mm512_store_ps(tmp[4][m], _tmp4m);

                    __m512 _tmp56a = _mm512_fmadd_ps(_mm512_set1_ps(4.f), _mm512_fmadd_ps(_mm512_set1_ps(-1.25f), _r04, _r02), _r06);
                    __m512 _tmp56b = _mm512_fmadd_ps(_mm512_set1_ps(0.5f), _r05, _mm512_fmadd_ps(_mm512_set1_ps(-2.5f), _r03, _mm512_mul_ps(_r01, _mm512_set1_ps(2.f))));

                    __m512 _tmp5m = _mm512_add_ps(_tmp56a, _tmp56b);
                    __m512 _tmp6m = _mm512_sub_ps(_tmp56a, _tmp56b);
                    _mm512_store_ps(tmp[5][m], _tmp5m);
                    _mm512_store_ps(tmp[6][m], _tmp6m);

                    r0 += w * 16;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 16;
                float* r0_tm_1 = r0_tm_0 + tiles * 16;
                float* r0_tm_2 = r0_tm_0 + tiles * 16 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 16 * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * 16 * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * 16 * 5;
                float* r0_tm_6 = r0_tm_0 + tiles * 16 * 6;
                float* r0_tm_7 = r0_tm_0 + tiles * 16 * 7;

                for (int m = 0; m < 8; m++)
                {
                    __m512 _tmp00 = _mm512_load_ps(tmp[m][0]);
                    __m512 _tmp01 = _mm512_load_ps(tmp[m][1]);
                    __m512 _tmp02 = _mm512_load_ps(tmp[m][2]);
                    __m512 _tmp03 = _mm512_load_ps(tmp[m][3]);
                    __m512 _tmp04 = _mm512_load_ps(tmp[m][4]);
                    __m512 _tmp05 = _mm512_load_ps(tmp[m][5]);
                    __m512 _tmp06 = _mm512_load_ps(tmp[m][6]);
                    __m512 _tmp07 = _mm512_load_ps(tmp[m][7]);

                    __m512 _r0tm0 = _mm512_fmadd_ps(_mm512_set1_ps(5.25f), _mm512_sub_ps(_tmp04, _tmp02), _mm512_sub_ps(_tmp00, _tmp06));
                    __m512 _r0tm7 = _mm512_fmadd_ps(_mm512_set1_ps(5.25f), _mm512_sub_ps(_tmp03, _tmp05), _mm512_sub_ps(_tmp07, _tmp01));

                    __m512 _tmp12a = _mm512_fmadd_ps(_mm512_set1_ps(-4.25f), _tmp04, _mm512_add_ps(_tmp02, _tmp06));
                    __m512 _tmp12b = _mm512_fmadd_ps(_mm512_set1_ps(-4.25f), _tmp03, _mm512_add_ps(_tmp01, _tmp05));

                    __m512 _r0tm1 = _mm512_add_ps(_tmp12a, _tmp12b);
                    __m512 _r0tm2 = _mm512_sub_ps(_tmp12a, _tmp12b);

                    __m512 _tmp34a = _mm512_fmadd_ps(_mm512_set1_ps(-1.25f), _tmp04, _mm512_fmadd_ps(_mm512_set1_ps(0.25f), _tmp02, _tmp06));
                    __m512 _tmp34b = _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp05, _mm512_fmadd_ps(_mm512_set1_ps(-2.5f), _tmp03, _mm512_mul_ps(_tmp01, _mm512_set1_ps(0.5f))));

                    __m512 _r0tm3 = _mm512_add_ps(_tmp34a, _tmp34b);
                    __m512 _r0tm4 = _mm512_sub_ps(_tmp34a, _tmp34b);

                    __m512 _tmp56a = _mm512_fmadd_ps(_mm512_set1_ps(4.f), _mm512_fmadd_ps(_mm512_set1_ps(-1.25f), _tmp04, _tmp02), _tmp06);
                    __m512 _tmp56b = _mm512_fmadd_ps(_mm512_set1_ps(0.5f), _tmp05, _mm512_fmadd_ps(_mm512_set1_ps(-2.5f), _tmp03, _mm512_mul_ps(_tmp01, _mm512_set1_ps(2.f))));

                    __m512 _r0tm5 = _mm512_add_ps(_tmp56a, _tmp56b);
                    __m512 _r0tm6 = _mm512_sub_ps(_tmp56a, _tmp56b);

                    _mm512_store_ps(r0_tm_0, _r0tm0);
                    _mm512_store_ps(r0_tm_1, _r0tm1);
                    _mm512_store_ps(r0_tm_2, _r0tm2);
                    _mm512_store_ps(r0_tm_3, _r0tm3);
                    _mm512_store_ps(r0_tm_4, _r0tm4);
                    _mm512_store_ps(r0_tm_5, _r0tm5);
                    _mm512_store_ps(r0_tm_6, _r0tm6);
                    _mm512_store_ps(r0_tm_7, _r0tm7);

                    r0_tm_0 += tiles * 128;
                    r0_tm_1 += tiles * 128;
                    r0_tm_2 += tiles * 128;
                    r0_tm_3 += tiles * 128;
                    r0_tm_4 += tiles * 128;
                    r0_tm_5 += tiles * 128;
                    r0_tm_6 += tiles * 128;
                    r0_tm_7 += tiles * 128;
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_output_pack16_avx512(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + p * 16) : _mm512_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[6][8][16];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 16;
                const float* output0_tm_1 = output0_tm_0 + tiles * 16;
                const float* output0_tm_2 = output0_tm_0 + tiles * 16 * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 16 * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * 16 * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * 16 * 5;
                const float* output0_tm_6 = output0_tm_0 + tiles * 16 * 6;
                const float* output0_tm_7 = output0_tm_0 + tiles * 16 * 7;

                float* output0 = out0.row(i * 6) + (j * 6) * 16;

                for (int m = 0; m < 8; m++)
                {
                    __m512 _out0tm0 = _mm512_load_ps(output0_tm_0);
                    __m512 _out0tm1 = _mm512_load_ps(output0_tm_1);
                    __m512 _out0tm2 = _mm512_load_ps(output0_tm_2);
                    __m512 _out0tm3 = _mm512_load_ps(output0_tm_3);
                    __m512 _out0tm4 = _mm512_load_ps(output0_tm_4);
                    __m512 _out0tm5 = _mm512_load_ps(output0_tm_5);
                    __m512 _out0tm6 = _mm512_load_ps(output0_tm_6);
                    __m512 _out0tm7 = _mm512_load_ps(output0_tm_7);

                    __m512 _tmp024a = _mm512_add_ps(_out0tm1, _out0tm2);
                    __m512 _tmp135a = _mm512_sub_ps(_out0tm1, _out0tm2);

                    __m512 _tmp024b = _mm512_add_ps(_out0tm3, _out0tm4);
                    __m512 _tmp135b = _mm512_sub_ps(_out0tm3, _out0tm4);

                    __m512 _tmp024c = _mm512_add_ps(_out0tm5, _out0tm6);
                    __m512 _tmp135c = _mm512_sub_ps(_out0tm5, _out0tm6);

                    __m512 _tmp0m = _mm512_add_ps(_mm512_add_ps(_out0tm0, _tmp024a), _mm512_fmadd_ps(_mm512_set1_ps(32.f), _tmp024c, _tmp024b));
                    __m512 _tmp2m = _mm512_fmadd_ps(_mm512_set1_ps(8.f), _tmp024c, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp024b, _tmp024a));
                    __m512 _tmp4m = _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp024c, _mm512_fmadd_ps(_mm512_set1_ps(16.f), _tmp024b, _tmp024a));
                    _mm512_store_ps(tmp[0][m], _tmp0m);
                    _mm512_store_ps(tmp[2][m], _tmp2m);
                    _mm512_store_ps(tmp[4][m], _tmp4m);

                    __m512 _tmp1m = _mm512_fmadd_ps(_mm512_set1_ps(16.f), _tmp135c, _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp135b, _tmp135a));
                    __m512 _tmp3m = _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp135c, _mm512_fmadd_ps(_mm512_set1_ps(8.f), _tmp135b, _tmp135a));
                    __m512 _tmp5m = _mm512_add_ps(_mm512_add_ps(_out0tm7, _tmp135a), _mm512_fmadd_ps(_mm512_set1_ps(32.f), _tmp135b, _tmp135c));
                    _mm512_store_ps(tmp[1][m], _tmp1m);
                    _mm512_store_ps(tmp[3][m], _tmp3m);
                    _mm512_store_ps(tmp[5][m], _tmp5m);

                    output0_tm_0 += tiles * 128;
                    output0_tm_1 += tiles * 128;
                    output0_tm_2 += tiles * 128;
                    output0_tm_3 += tiles * 128;
                    output0_tm_4 += tiles * 128;
                    output0_tm_5 += tiles * 128;
                    output0_tm_6 += tiles * 128;
                    output0_tm_7 += tiles * 128;
                }

                for (int m = 0; m < 6; m++)
                {
                    __m512 _tmp00 = _mm512_load_ps(tmp[m][0]);
                    __m512 _tmp01 = _mm512_load_ps(tmp[m][1]);
                    __m512 _tmp02 = _mm512_load_ps(tmp[m][2]);
                    __m512 _tmp03 = _mm512_load_ps(tmp[m][3]);
                    __m512 _tmp04 = _mm512_load_ps(tmp[m][4]);
                    __m512 _tmp05 = _mm512_load_ps(tmp[m][5]);
                    __m512 _tmp06 = _mm512_load_ps(tmp[m][6]);
                    __m512 _tmp07 = _mm512_load_ps(tmp[m][7]);

                    __m512 _tmp024a = _mm512_add_ps(_tmp01, _tmp02);
                    __m512 _tmp135a = _mm512_sub_ps(_tmp01, _tmp02);

                    __m512 _tmp024b = _mm512_add_ps(_tmp03, _tmp04);
                    __m512 _tmp135b = _mm512_sub_ps(_tmp03, _tmp04);

                    __m512 _tmp024c = _mm512_add_ps(_tmp05, _tmp06);
                    __m512 _tmp135c = _mm512_sub_ps(_tmp05, _tmp06);

                    __m512 _out00 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_tmp00, _tmp024a), _mm512_fmadd_ps(_mm512_set1_ps(32.f), _tmp024c, _tmp024b)));
                    __m512 _out02 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(8.f), _tmp024c, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp024b, _tmp024a)));
                    __m512 _out04 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp024c, _mm512_fmadd_ps(_mm512_set1_ps(16.f), _tmp024b, _tmp024a)));
                    _mm512_store_ps(output0, _out00);
                    _mm512_store_ps(output0 + 32, _out02);
                    _mm512_store_ps(output0 + 64, _out04);

                    __m512 _out01 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(16.f), _tmp135c, _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp135b, _tmp135a)));
                    __m512 _out03 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp135c, _mm512_fmadd_ps(_mm512_set1_ps(8.f), _tmp135b, _tmp135a)));
                    __m512 _out05 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_tmp07, _tmp135a), _mm512_fmadd_ps(_mm512_set1_ps(32.f), _tmp135b, _tmp135c)));
                    _mm512_store_ps(output0 + 16, _out01);
                    _mm512_store_ps(output0 + 48, _out03);
                    _mm512_store_ps(output0 + 80, _out05);

                    output0 += outw * 16;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_input_pack16_avx512(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

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
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[6][6][16];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 4) + (j * 4) * 16;

                for (int m = 0; m < 6; m++)
                {
                    __m512 _r00 = _mm512_load_ps(r0);
                    __m512 _r01 = _mm512_load_ps(r0 + 16);
                    __m512 _r02 = _mm512_load_ps(r0 + 16 * 2);
                    __m512 _r03 = _mm512_load_ps(r0 + 16 * 3);
                    __m512 _r04 = _mm512_load_ps(r0 + 16 * 4);
                    __m512 _r05 = _mm512_load_ps(r0 + 16 * 5);

                    __m512 _tmp0m = _mm512_fmadd_ps(_mm512_set1_ps(-5.f), _r02, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _r00, _r04));
                    __m512 _tmp1m = _mm512_fmadd_ps(_mm512_set1_ps(-4.f), _mm512_add_ps(_r01, _r02), _mm512_add_ps(_r04, _r03));
                    __m512 _tmp2m = _mm512_fmadd_ps(_mm512_set1_ps(4.f), _mm512_sub_ps(_r01, _r02), _mm512_sub_ps(_r04, _r03));
                    __m512 _tmp3m = _mm512_fmadd_ps(_mm512_set1_ps(-2.f), _mm512_sub_ps(_r01, _r03), _mm512_sub_ps(_r04, _r02));
                    __m512 _tmp4m = _mm512_fmadd_ps(_mm512_set1_ps(2.f), _mm512_sub_ps(_r01, _r03), _mm512_sub_ps(_r04, _r02));
                    __m512 _tmp5m = _mm512_fmadd_ps(_mm512_set1_ps(-5.f), _r03, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _r01, _r05));

                    _mm512_store_ps(tmp[0][m], _tmp0m);
                    _mm512_store_ps(tmp[1][m], _tmp1m);
                    _mm512_store_ps(tmp[2][m], _tmp2m);
                    _mm512_store_ps(tmp[3][m], _tmp3m);
                    _mm512_store_ps(tmp[4][m], _tmp4m);
                    _mm512_store_ps(tmp[5][m], _tmp5m);

                    r0 += w * 16;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 16;
                float* r0_tm_1 = r0_tm_0 + tiles * 16;
                float* r0_tm_2 = r0_tm_0 + tiles * 16 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 16 * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * 16 * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * 16 * 5;

                for (int m = 0; m < 6; m++)
                {
                    __m512 _tmp00 = _mm512_load_ps(tmp[m][0]);
                    __m512 _tmp01 = _mm512_load_ps(tmp[m][1]);
                    __m512 _tmp02 = _mm512_load_ps(tmp[m][2]);
                    __m512 _tmp03 = _mm512_load_ps(tmp[m][3]);
                    __m512 _tmp04 = _mm512_load_ps(tmp[m][4]);
                    __m512 _tmp05 = _mm512_load_ps(tmp[m][5]);

                    __m512 _r0tm0 = _mm512_fmadd_ps(_mm512_set1_ps(-5.f), _tmp02, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp00, _tmp04));
                    __m512 _r0tm1 = _mm512_fmadd_ps(_mm512_set1_ps(-4.f), _mm512_add_ps(_tmp01, _tmp02), _mm512_add_ps(_tmp04, _tmp03));
                    __m512 _r0tm2 = _mm512_fmadd_ps(_mm512_set1_ps(4.f), _mm512_sub_ps(_tmp01, _tmp02), _mm512_sub_ps(_tmp04, _tmp03));
                    __m512 _r0tm3 = _mm512_fmadd_ps(_mm512_set1_ps(-2.f), _mm512_sub_ps(_tmp01, _tmp03), _mm512_sub_ps(_tmp04, _tmp02));
                    __m512 _r0tm4 = _mm512_fmadd_ps(_mm512_set1_ps(2.f), _mm512_sub_ps(_tmp01, _tmp03), _mm512_sub_ps(_tmp04, _tmp02));
                    __m512 _r0tm5 = _mm512_fmadd_ps(_mm512_set1_ps(-5.f), _tmp03, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp01, _tmp05));

                    _mm512_store_ps(r0_tm_0, _r0tm0);
                    _mm512_store_ps(r0_tm_1, _r0tm1);
                    _mm512_store_ps(r0_tm_2, _r0tm2);
                    _mm512_store_ps(r0_tm_3, _r0tm3);
                    _mm512_store_ps(r0_tm_4, _r0tm4);
                    _mm512_store_ps(r0_tm_5, _r0tm5);

                    r0_tm_0 += tiles * 96;
                    r0_tm_1 += tiles * 96;
                    r0_tm_2 += tiles * 96;
                    r0_tm_3 += tiles * 96;
                    r0_tm_4 += tiles * 96;
                    r0_tm_5 += tiles * 96;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_pack16_avx512(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + p * 16) : _mm512_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[4][6][16];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 16;
                const float* output0_tm_1 = output0_tm_0 + tiles * 16;
                const float* output0_tm_2 = output0_tm_0 + tiles * 16 * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 16 * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * 16 * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * 16 * 5;

                float* output0 = out0.row(i * 4) + (j * 4) * 16;

                for (int m = 0; m < 6; m++)
                {
                    __m512 _out0tm0 = _mm512_load_ps(output0_tm_0);
                    __m512 _out0tm1 = _mm512_load_ps(output0_tm_1);
                    __m512 _out0tm2 = _mm512_load_ps(output0_tm_2);
                    __m512 _out0tm3 = _mm512_load_ps(output0_tm_3);
                    __m512 _out0tm4 = _mm512_load_ps(output0_tm_4);
                    __m512 _out0tm5 = _mm512_load_ps(output0_tm_5);

                    __m512 _tmp02a = _mm512_add_ps(_out0tm1, _out0tm2);
                    __m512 _tmp13a = _mm512_sub_ps(_out0tm1, _out0tm2);

                    __m512 _tmp02b = _mm512_add_ps(_out0tm3, _out0tm4);
                    __m512 _tmp13b = _mm512_sub_ps(_out0tm3, _out0tm4);

                    __m512 _tmp0m = _mm512_add_ps(_mm512_add_ps(_out0tm0, _tmp02a), _tmp02b);
                    __m512 _tmp1m = _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp13b, _tmp13a);
                    __m512 _tmp2m = _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp02b, _tmp02a);
                    __m512 _tmp3m = _mm512_fmadd_ps(_mm512_set1_ps(8.f), _tmp13b, _mm512_add_ps(_out0tm5, _tmp13a));

                    _mm512_store_ps(tmp[0][m], _tmp0m);
                    _mm512_store_ps(tmp[1][m], _tmp1m);
                    _mm512_store_ps(tmp[2][m], _tmp2m);
                    _mm512_store_ps(tmp[3][m], _tmp3m);

                    output0_tm_0 += tiles * 96;
                    output0_tm_1 += tiles * 96;
                    output0_tm_2 += tiles * 96;
                    output0_tm_3 += tiles * 96;
                    output0_tm_4 += tiles * 96;
                    output0_tm_5 += tiles * 96;
                }

                for (int m = 0; m < 4; m++)
                {
                    __m512 _tmp00 = _mm512_load_ps(tmp[m][0]);
                    __m512 _tmp01 = _mm512_load_ps(tmp[m][1]);
                    __m512 _tmp02 = _mm512_load_ps(tmp[m][2]);
                    __m512 _tmp03 = _mm512_load_ps(tmp[m][3]);
                    __m512 _tmp04 = _mm512_load_ps(tmp[m][4]);
                    __m512 _tmp05 = _mm512_load_ps(tmp[m][5]);

                    __m512 _tmp02a = _mm512_add_ps(_tmp01, _tmp02);
                    __m512 _tmp13a = _mm512_sub_ps(_tmp01, _tmp02);

                    __m512 _tmp02b = _mm512_add_ps(_tmp03, _tmp04);
                    __m512 _tmp13b = _mm512_sub_ps(_tmp03, _tmp04);

                    __m512 _out00 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_tmp00, _tmp02a), _tmp02b));
                    __m512 _out01 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(2.f), _tmp13b, _tmp13a));
                    __m512 _out02 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(4.f), _tmp02b, _tmp02a));
                    __m512 _out03 = _mm512_add_ps(_bias0, _mm512_fmadd_ps(_mm512_set1_ps(8.f), _tmp13b, _mm512_add_ps(_tmp05, _tmp13a)));

                    _mm512_store_ps(output0, _out00);
                    _mm512_store_ps(output0 + 16, _out01);
                    _mm512_store_ps(output0 + 16 * 2, _out02);
                    _mm512_store_ps(output0 + 16 * 3, _out03);

                    output0 += outw * 16;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_pack16_avx512(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[4][4][16];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 2) + (j * 2) * 16;

                for (int m = 0; m < 4; m++)
                {
                    __m512 _r00 = _mm512_load_ps(r0);
                    __m512 _r01 = _mm512_load_ps(r0 + 16);
                    __m512 _r02 = _mm512_load_ps(r0 + 16 * 2);
                    __m512 _r03 = _mm512_load_ps(r0 + 16 * 3);

                    __m512 _tmp0m = _mm512_sub_ps(_r00, _r02);
                    __m512 _tmp1m = _mm512_add_ps(_r01, _r02);
                    __m512 _tmp2m = _mm512_sub_ps(_r02, _r01);
                    __m512 _tmp3m = _mm512_sub_ps(_r03, _r01);

                    _mm512_store_ps(tmp[0][m], _tmp0m);
                    _mm512_store_ps(tmp[1][m], _tmp1m);
                    _mm512_store_ps(tmp[2][m], _tmp2m);
                    _mm512_store_ps(tmp[3][m], _tmp3m);

                    r0 += w * 16;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * 16;
                float* r0_tm_1 = r0_tm_0 + tiles * 16;
                float* r0_tm_2 = r0_tm_0 + tiles * 16 * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 16 * 3;

                for (int m = 0; m < 4; m++)
                {
                    __m512 _tmp00 = _mm512_load_ps(tmp[m][0]);
                    __m512 _tmp01 = _mm512_load_ps(tmp[m][1]);
                    __m512 _tmp02 = _mm512_load_ps(tmp[m][2]);
                    __m512 _tmp03 = _mm512_load_ps(tmp[m][3]);

                    __m512 _r0tm0 = _mm512_sub_ps(_tmp00, _tmp02);
                    __m512 _r0tm1 = _mm512_add_ps(_tmp01, _tmp02);
                    __m512 _r0tm2 = _mm512_sub_ps(_tmp02, _tmp01);
                    __m512 _r0tm3 = _mm512_sub_ps(_tmp03, _tmp01);

                    _mm512_store_ps(r0_tm_0, _r0tm0);
                    _mm512_store_ps(r0_tm_1, _r0tm1);
                    _mm512_store_ps(r0_tm_2, _r0tm2);
                    _mm512_store_ps(r0_tm_3, _r0tm3);

                    r0_tm_0 += tiles * 16 * 4;
                    r0_tm_1 += tiles * 16 * 4;
                    r0_tm_2 += tiles * 16 * 4;
                    r0_tm_3 += tiles * 16 * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_pack16_avx512(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        __m512 _bias0 = biasptr ? _mm512_loadu_ps(biasptr + p * 16) : _mm512_setzero_ps();

#ifdef _MSC_VER
        __declspec(align(64))
#else
        __attribute__((aligned(64)))
#endif
        float tmp[2][4][16];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * 16;
                const float* output0_tm_1 = output0_tm_0 + tiles * 16;
                const float* output0_tm_2 = output0_tm_0 + tiles * 16 * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 16 * 3;

                float* output0 = out0.row(i * 2) + (j * 2) * 16;

                for (int m = 0; m < 4; m++)
                {
                    __m512 _out0tm0 = _mm512_load_ps(output0_tm_0);
                    __m512 _out0tm1 = _mm512_load_ps(output0_tm_1);
                    __m512 _out0tm2 = _mm512_load_ps(output0_tm_2);
                    __m512 _out0tm3 = _mm512_load_ps(output0_tm_3);

                    __m512 _tmp0m = _mm512_add_ps(_mm512_add_ps(_out0tm0, _out0tm1), _out0tm2);
                    __m512 _tmp1m = _mm512_add_ps(_mm512_sub_ps(_out0tm1, _out0tm2), _out0tm3);

                    _mm512_store_ps(tmp[0][m], _tmp0m);
                    _mm512_store_ps(tmp[1][m], _tmp1m);

                    output0_tm_0 += tiles * 16 * 4;
                    output0_tm_1 += tiles * 16 * 4;
                    output0_tm_2 += tiles * 16 * 4;
                    output0_tm_3 += tiles * 16 * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    __m512 _tmp00 = _mm512_load_ps(tmp[m][0]);
                    __m512 _tmp01 = _mm512_load_ps(tmp[m][1]);
                    __m512 _tmp02 = _mm512_load_ps(tmp[m][2]);
                    __m512 _tmp03 = _mm512_load_ps(tmp[m][3]);

                    __m512 _out00 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_add_ps(_tmp00, _tmp01), _tmp02));
                    __m512 _out01 = _mm512_add_ps(_bias0, _mm512_add_ps(_mm512_sub_ps(_tmp01, _tmp02), _tmp03));

                    _mm512_store_ps(output0, _out00);
                    _mm512_store_ps(output0 + 16, _out01);

                    output0 += outw * 16;
                }
            }
        }
    }
}
