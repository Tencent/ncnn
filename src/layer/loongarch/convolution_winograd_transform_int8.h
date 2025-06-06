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

static void conv3x3s1_winograd43_transform_input_int8_lsx(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        short tmp[6][6];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const signed char* r0 = img0.row<const signed char>(i * 4) + (j * 4);

                for (int m = 0; m < 6; m++)
                {
                    signed char r00 = r0[0];
                    signed char r01 = r0[1];
                    signed char r02 = r0[2];
                    signed char r03 = r0[3];
                    signed char r04 = r0[4];
                    signed char r05 = r0[5];

                    short tmp0m = 4 * r00 - 5 * r02 + r04;
                    short tmp1m = -4 * (r01 + r02) + r04 + r03;
                    short tmp2m = 4 * (r01 - r02) + r04 - r03;
                    short tmp3m = -2 * (r01 - r03) + r04 - r02;
                    short tmp4m = 2 * (r01 - r03) + r04 - r02;
                    short tmp5m = 4 * r01 - 5 * r03 + r05;

                    tmp[0][m] = tmp0m;
                    tmp[1][m] = tmp1m;
                    tmp[2][m] = tmp2m;
                    tmp[3][m] = tmp3m;
                    tmp[4][m] = tmp4m;
                    tmp[5][m] = tmp5m;

                    r0 += w;
                }

                short* r0_tm_0 = (short*)img0_tm + (i * w_tiles + j);
                short* r0_tm_1 = r0_tm_0 + tiles;
                short* r0_tm_2 = r0_tm_0 + tiles * 2;
                short* r0_tm_3 = r0_tm_0 + tiles * 3;
                short* r0_tm_4 = r0_tm_0 + tiles * 4;
                short* r0_tm_5 = r0_tm_0 + tiles * 5;

                for (int m = 0; m < 6; m++)
                {
                    short tmp00 = tmp[m][0];
                    short tmp01 = tmp[m][1];
                    short tmp02 = tmp[m][2];
                    short tmp03 = tmp[m][3];
                    short tmp04 = tmp[m][4];
                    short tmp05 = tmp[m][5];

                    short r0tm0 = 4 * tmp00 - 5 * tmp02 + tmp04;
                    short r0tm1 = -4 * (tmp01 + tmp02) + tmp04 + tmp03;
                    short r0tm2 = 4 * (tmp01 - tmp02) + tmp04 - tmp03;
                    short r0tm3 = -2 * (tmp01 - tmp03) + tmp04 - tmp02;
                    short r0tm4 = 2 * (tmp01 - tmp03) + tmp04 - tmp02;
                    short r0tm5 = 4 * tmp01 - 5 * tmp03 + tmp05;

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

static void conv3x3s1_winograd43_transform_output_int8_lsx(const Mat& top_blob_tm, Mat& top_blob, const Option& opt)
{
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

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

        int tmp[4][6];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int* output0_tm_0 = (const int*)out0_tm + (i * w_tiles + j) * 1;
                const int* output0_tm_1 = output0_tm_0 + tiles * 1;
                const int* output0_tm_2 = output0_tm_0 + tiles * 2;
                const int* output0_tm_3 = output0_tm_0 + tiles * 3;
                const int* output0_tm_4 = output0_tm_0 + tiles * 4;
                const int* output0_tm_5 = output0_tm_0 + tiles * 5;

                int* output0 = out0.row<int>(i * 4) + j * 4;

                for (int m = 0; m < 5; m++)
                {
                    int tmp02a = output0_tm_1[0] + output0_tm_2[0];
                    int tmp13a = output0_tm_1[0] - output0_tm_2[0];

                    int tmp02b = output0_tm_3[0] + output0_tm_4[0];
                    int tmp13b = output0_tm_3[0] - output0_tm_4[0];

                    tmp[0][m] = output0_tm_0[0] + tmp02a + tmp02b;
                    tmp[1][m] = tmp13a + tmp13b * 2;
                    tmp[2][m] = tmp02a + tmp02b * 4;
                    tmp[3][m] = output0_tm_5[0] * 4 + tmp13a + tmp13b * 8;

                    output0_tm_0 += tiles * 6;
                    output0_tm_1 += tiles * 6;
                    output0_tm_2 += tiles * 6;
                    output0_tm_3 += tiles * 6;
                    output0_tm_4 += tiles * 6;
                    output0_tm_5 += tiles * 6;
                }
                for (int m = 5; m < 6; m++)
                {
                    int tmp02a = output0_tm_1[0] + output0_tm_2[0];
                    int tmp13a = output0_tm_1[0] - output0_tm_2[0];

                    int tmp02b = output0_tm_3[0] + output0_tm_4[0];
                    int tmp13b = output0_tm_3[0] - output0_tm_4[0];

                    tmp[0][m] = (output0_tm_0[0] + tmp02a + tmp02b) * 4;
                    tmp[1][m] = (tmp13a + tmp13b * 2) * 4;
                    tmp[2][m] = (tmp02a + tmp02b * 4) * 4;
                    tmp[3][m] = (output0_tm_5[0] * 4 + tmp13a + tmp13b * 8) * 4;

                    output0_tm_0 += tiles * 6;
                    output0_tm_1 += tiles * 6;
                    output0_tm_2 += tiles * 6;
                    output0_tm_3 += tiles * 6;
                    output0_tm_4 += tiles * 6;
                    output0_tm_5 += tiles * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    const int* tmp0 = tmp[m];

                    int tmp02a = tmp0[1] + tmp0[2];
                    int tmp13a = tmp0[1] - tmp0[2];

                    int tmp02b = tmp0[3] + tmp0[4];
                    int tmp13b = tmp0[3] - tmp0[4];

                    output0[0] = (tmp0[0] + tmp02a + tmp02b) / 576;
                    output0[1] = (tmp13a + tmp13b * 2) / 576;
                    output0[2] = (tmp02a + tmp02b * 4) / 576;
                    output0[3] = (tmp0[5] + tmp13a + tmp13b * 8) / 576;

                    output0 += outw;
                }
            }
        }
    }
}
