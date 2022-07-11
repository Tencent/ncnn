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

static void conv3x3s1_winograd43_transform_input_msa(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        float tmp[6][6];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 4) + (j * 4);

                for (int m = 0; m < 6; m++)
                {
                    float r00 = r0[0];
                    float r01 = r0[1];
                    float r02 = r0[2];
                    float r03 = r0[3];
                    float r04 = r0[4];
                    float r05 = r0[5];

                    float tmp0m = 4 * r00 - 5 * r02 + r04;
                    float tmp1m = -4 * (r01 + r02) + r04 + r03;
                    float tmp2m = 4 * (r01 - r02) + r04 - r03;
                    float tmp3m = -2 * (r01 - r03) + r04 - r02;
                    float tmp4m = 2 * (r01 - r03) + r04 - r02;
                    float tmp5m = 4 * r01 - 5 * r03 + r05;

                    tmp[0][m] = tmp0m;
                    tmp[1][m] = tmp1m;
                    tmp[2][m] = tmp2m;
                    tmp[3][m] = tmp3m;
                    tmp[4][m] = tmp4m;
                    tmp[5][m] = tmp5m;

                    r0 += w;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j);
                float* r0_tm_1 = r0_tm_0 + tiles;
                float* r0_tm_2 = r0_tm_0 + tiles * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * 5;

                for (int m = 0; m < 6; m++)
                {
                    float tmp00 = tmp[m][0];
                    float tmp01 = tmp[m][1];
                    float tmp02 = tmp[m][2];
                    float tmp03 = tmp[m][3];
                    float tmp04 = tmp[m][4];
                    float tmp05 = tmp[m][5];

                    float r0tm0 = 4 * tmp00 - 5 * tmp02 + tmp04;
                    float r0tm1 = -4 * (tmp01 + tmp02) + tmp04 + tmp03;
                    float r0tm2 = 4 * (tmp01 - tmp02) + tmp04 - tmp03;
                    float r0tm3 = -2 * (tmp01 - tmp03) + tmp04 - tmp02;
                    float r0tm4 = 2 * (tmp01 - tmp03) + tmp04 - tmp02;
                    float r0tm5 = 4 * tmp01 - 5 * tmp03 + tmp05;

                    r0_tm_0[0] = r0tm0;
                    r0_tm_1[0] = r0tm1;
                    r0_tm_2[0] = r0tm2;
                    r0_tm_3[0] = r0tm3;
                    r0_tm_4[0] = r0tm4;
                    r0_tm_5[0] = r0tm5;

                    r0_tm_0 += tiles * 6;
                    r0_tm_1 += tiles * 6;
                    r0_tm_2 += tiles * 6;
                    r0_tm_3 += tiles * 6;
                    r0_tm_4 += tiles * 6;
                    r0_tm_5 += tiles * 6;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_msa(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        float bias0 = biasptr ? biasptr[p] : 0.f;

        float tmp[4][6];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j);
                const float* output0_tm_1 = output0_tm_0 + tiles;
                const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * 5;

                float* output0 = out0.row(i * 4) + (j * 4);

                for (int m = 0; m < 6; m++)
                {
                    float out0tm0 = output0_tm_0[0];
                    float out0tm1 = output0_tm_1[0];
                    float out0tm2 = output0_tm_2[0];
                    float out0tm3 = output0_tm_3[0];
                    float out0tm4 = output0_tm_4[0];
                    float out0tm5 = output0_tm_5[0];

                    float tmp02a = out0tm1 + out0tm2;
                    float tmp13a = out0tm1 - out0tm2;

                    float tmp02b = out0tm3 + out0tm4;
                    float tmp13b = out0tm3 - out0tm4;

                    float tmp0m = out0tm0 + tmp02a + tmp02b;
                    float tmp1m = tmp13a + tmp13b * 2;
                    float tmp2m = tmp02a + tmp02b * 4;
                    float tmp3m = out0tm5 + tmp13a + tmp13b * 8;

                    tmp[0][m] = tmp0m;
                    tmp[1][m] = tmp1m;
                    tmp[2][m] = tmp2m;
                    tmp[3][m] = tmp3m;

                    output0_tm_0 += tiles * 6;
                    output0_tm_1 += tiles * 6;
                    output0_tm_2 += tiles * 6;
                    output0_tm_3 += tiles * 6;
                    output0_tm_4 += tiles * 6;
                    output0_tm_5 += tiles * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    float tmp00 = tmp[m][0];
                    float tmp01 = tmp[m][1];
                    float tmp02 = tmp[m][2];
                    float tmp03 = tmp[m][3];
                    float tmp04 = tmp[m][4];
                    float tmp05 = tmp[m][5];

                    float tmp02a = tmp01 + tmp02;
                    float tmp13a = tmp01 - tmp02;

                    float tmp02b = tmp03 + tmp04;
                    float tmp13b = tmp03 - tmp04;

                    float out00 = bias0 + tmp00 + tmp02a + tmp02b;
                    float out01 = bias0 + tmp13a + tmp13b * 2;
                    float out02 = bias0 + tmp02a + tmp02b * 4;
                    float out03 = bias0 + tmp05 + tmp13a + tmp13b * 8;

                    output0[0] = out00;
                    output0[1] = out01;
                    output0[2] = out02;
                    output0[3] = out03;

                    output0 += outw;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_msa(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        float tmp[4][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 2) + (j * 2);

                for (int m = 0; m < 4; m++)
                {
                    float r00 = r0[0];
                    float r01 = r0[1];
                    float r02 = r0[2];
                    float r03 = r0[3];

                    float tmp0m = r00 - r02;
                    float tmp1m = r01 + r02;
                    float tmp2m = r02 - r01;
                    float tmp3m = r03 - r01;

                    tmp[0][m] = tmp0m;
                    tmp[1][m] = tmp1m;
                    tmp[2][m] = tmp2m;
                    tmp[3][m] = tmp3m;

                    r0 += w;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j);
                float* r0_tm_1 = r0_tm_0 + tiles;
                float* r0_tm_2 = r0_tm_0 + tiles * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * 3;

                for (int m = 0; m < 4; m++)
                {
                    float tmp00 = tmp[m][0];
                    float tmp01 = tmp[m][1];
                    float tmp02 = tmp[m][2];
                    float tmp03 = tmp[m][3];

                    float r0tm0 = tmp00 - tmp02;
                    float r0tm1 = tmp01 + tmp02;
                    float r0tm2 = tmp02 - tmp01;
                    float r0tm3 = tmp03 - tmp01;

                    r0_tm_0[0] = r0tm0;
                    r0_tm_1[0] = r0tm1;
                    r0_tm_2[0] = r0tm2;
                    r0_tm_3[0] = r0tm3;

                    r0_tm_0 += tiles * 4;
                    r0_tm_1 += tiles * 4;
                    r0_tm_2 += tiles * 4;
                    r0_tm_3 += tiles * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_msa(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        float bias0 = biasptr ? biasptr[p] : 0.f;

        float tmp[2][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j);
                const float* output0_tm_1 = output0_tm_0 + tiles;
                const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 3;

                float* output0 = out0.row(i * 2) + (j * 2);

                for (int m = 0; m < 4; m++)
                {
                    float out0tm0 = output0_tm_0[0];
                    float out0tm1 = output0_tm_1[0];
                    float out0tm2 = output0_tm_2[0];
                    float out0tm3 = output0_tm_3[0];

                    float tmp0m = out0tm0 + out0tm1 + out0tm2;
                    float tmp1m = out0tm1 - out0tm2 + out0tm3;

                    tmp[0][m] = tmp0m;
                    tmp[1][m] = tmp1m;

                    output0_tm_0 += tiles * 4;
                    output0_tm_1 += tiles * 4;
                    output0_tm_2 += tiles * 4;
                    output0_tm_3 += tiles * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    float tmp00 = tmp[m][0];
                    float tmp01 = tmp[m][1];
                    float tmp02 = tmp[m][2];
                    float tmp03 = tmp[m][3];

                    float out00 = bias0 + tmp00 + tmp01 + tmp02;
                    float out01 = bias0 + tmp01 - tmp02 + tmp03;

                    output0[0] = out00;
                    output0[1] = out01;

                    output0 += outw;
                }
            }
        }
    }
}
