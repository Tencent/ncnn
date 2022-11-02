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

static void conv3x3s1_winograd43_transform_output_pack4_int8_lsx(const Mat& top_blob_tm, Mat& top_blob, const Option& opt)
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

        int tmp[4][6][4];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int* output0_tm_0 = (const int*)out0_tm + (i * w_tiles + j) * 4;
                const int* output0_tm_1 = output0_tm_0 + tiles * 4;
                const int* output0_tm_2 = output0_tm_0 + tiles * 8;
                const int* output0_tm_3 = output0_tm_0 + tiles * 12;
                const int* output0_tm_4 = output0_tm_0 + tiles * 16;
                const int* output0_tm_5 = output0_tm_0 + tiles * 20;

                int* output0 = out0.row<int>(i * 4) + (j * 4) * 4;

                for (int m = 0; m < 5; m++)
                {
                    __m128i _out0tm0 = __lsx_vld(output0_tm_0, 0);
                    __m128i _out0tm1 = __lsx_vld(output0_tm_1, 0);
                    __m128i _out0tm2 = __lsx_vld(output0_tm_2, 0);
                    __m128i _out0tm3 = __lsx_vld(output0_tm_3, 0);
                    __m128i _out0tm4 = __lsx_vld(output0_tm_4, 0);
                    __m128i _out0tm5 = __lsx_vld(output0_tm_5, 0);

                    __m128i _tmp02a = __lsx_vadd_w(_out0tm1, _out0tm2);
                    __m128i _tmp13a = __lsx_vsub_w(_out0tm1, _out0tm2);

                    __m128i _tmp02b = __lsx_vadd_w(_out0tm3, _out0tm4);
                    __m128i _tmp13b = __lsx_vsub_w(_out0tm3, _out0tm4);

                    __m128i _tmp0m = __lsx_vadd_w(__lsx_vadd_w(_out0tm0, _tmp02a), _tmp02b);
                    __m128i _tmp1m = __lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 1));
                    __m128i _tmp2m = __lsx_vadd_w(_tmp02a, __lsx_vslli_w(_tmp02b, 2));
                    __m128i _tmp3m = __lsx_vadd_w(__lsx_vadd_w(_tmp13a, __lsx_vslli_w(_out0tm5, 2)), __lsx_vslli_w(_tmp13b, 3));

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }
                for (int m = 5; m < 6; m++)
                {
                    __m128i _out0tm0 = __lsx_vld(output0_tm_0, 0);
                    __m128i _out0tm1 = __lsx_vld(output0_tm_1, 0);
                    __m128i _out0tm2 = __lsx_vld(output0_tm_2, 0);
                    __m128i _out0tm3 = __lsx_vld(output0_tm_3, 0);
                    __m128i _out0tm4 = __lsx_vld(output0_tm_4, 0);
                    __m128i _out0tm5 = __lsx_vld(output0_tm_5, 0);

                    __m128i _tmp02a = __lsx_vadd_w(_out0tm1, _out0tm2);
                    __m128i _tmp13a = __lsx_vsub_w(_out0tm1, _out0tm2);

                    __m128i _tmp02b = __lsx_vadd_w(_out0tm3, _out0tm4);
                    __m128i _tmp13b = __lsx_vsub_w(_out0tm3, _out0tm4);

                    __m128i _tmp0m = __lsx_vadd_w(__lsx_vadd_w(_out0tm0, _tmp02a), _tmp02b);
                    __m128i _tmp1m = __lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 1));
                    __m128i _tmp2m = __lsx_vadd_w(_tmp02a, __lsx_vslli_w(_tmp02b, 2));
                    __m128i _tmp3m = __lsx_vadd_w(__lsx_vadd_w(_tmp13a, __lsx_vslli_w(_out0tm5, 2)), __lsx_vslli_w(_tmp13b, 3));

                    _tmp0m = __lsx_vslli_w(_tmp0m, 2);
                    _tmp1m = __lsx_vslli_w(_tmp1m, 2);
                    _tmp2m = __lsx_vslli_w(_tmp2m, 2);
                    _tmp3m = __lsx_vslli_w(_tmp3m, 2);

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }

                for (int m = 0; m < 4; m++)
                {
                    __m128i _tmp00 = __lsx_vld(tmp[m][0], 0);
                    __m128i _tmp01 = __lsx_vld(tmp[m][1], 0);
                    __m128i _tmp02 = __lsx_vld(tmp[m][2], 0);
                    __m128i _tmp03 = __lsx_vld(tmp[m][3], 0);
                    __m128i _tmp04 = __lsx_vld(tmp[m][4], 0);
                    __m128i _tmp05 = __lsx_vld(tmp[m][5], 0);

                    __m128i _tmp02a = __lsx_vadd_w(_tmp01, _tmp02);
                    __m128i _tmp13a = __lsx_vsub_w(_tmp01, _tmp02);

                    __m128i _tmp02b = __lsx_vadd_w(_tmp03, _tmp04);
                    __m128i _tmp13b = __lsx_vsub_w(_tmp03, _tmp04);

                    __m128i _out00 = __lsx_vadd_w(__lsx_vadd_w(_tmp00, _tmp02a), _tmp02b);
                    __m128i _out01 = __lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 1));
                    __m128i _out02 = __lsx_vadd_w(_tmp02a, __lsx_vslli_w(_tmp02b, 2));
                    __m128i _out03 = __lsx_vadd_w(__lsx_vadd_w(_tmp05, _tmp13a), __lsx_vslli_w(_tmp13b, 3));

                    // TODO use integer trick for division by 576
                    __m128 _v576 = __lsx_vreplfr2vr_s(1.0 / 576);
                    _out00 = __lsx_vftint_w_s(__lsx_vfmul_s(__lsx_vffint_s_w(_out00), _v576));
                    _out01 = __lsx_vftint_w_s(__lsx_vfmul_s(__lsx_vffint_s_w(_out01), _v576));
                    _out02 = __lsx_vftint_w_s(__lsx_vfmul_s(__lsx_vffint_s_w(_out02), _v576));
                    _out03 = __lsx_vftint_w_s(__lsx_vfmul_s(__lsx_vffint_s_w(_out03), _v576));

                    __lsx_vst(_out00, output0, 0);
                    __lsx_vst(_out01, output0 + 4, 0);
                    __lsx_vst(_out02, output0 + 8, 0);
                    __lsx_vst(_out03, output0 + 12, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}
