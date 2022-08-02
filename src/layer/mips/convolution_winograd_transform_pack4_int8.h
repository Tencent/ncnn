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

static void conv3x3s1_winograd43_transform_output_pack4_int8_msa(const Mat& top_blob_tm, Mat& top_blob, const Option& opt)
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
                    v4i32 _out0tm0 = __msa_ld_w(output0_tm_0, 0);
                    v4i32 _out0tm1 = __msa_ld_w(output0_tm_1, 0);
                    v4i32 _out0tm2 = __msa_ld_w(output0_tm_2, 0);
                    v4i32 _out0tm3 = __msa_ld_w(output0_tm_3, 0);
                    v4i32 _out0tm4 = __msa_ld_w(output0_tm_4, 0);
                    v4i32 _out0tm5 = __msa_ld_w(output0_tm_5, 0);

                    v4i32 _tmp02a = __msa_addv_w(_out0tm1, _out0tm2);
                    v4i32 _tmp13a = __msa_subv_w(_out0tm1, _out0tm2);

                    v4i32 _tmp02b = __msa_addv_w(_out0tm3, _out0tm4);
                    v4i32 _tmp13b = __msa_subv_w(_out0tm3, _out0tm4);

                    v4i32 _tmp0m = __msa_addv_w(__msa_addv_w(_out0tm0, _tmp02a), _tmp02b);
                    v4i32 _tmp1m = __msa_addv_w(_tmp13a, __msa_slli_w(_tmp13b, 1));
                    v4i32 _tmp2m = __msa_addv_w(_tmp02a, __msa_slli_w(_tmp02b, 2));
                    v4i32 _tmp3m = __msa_addv_w(__msa_addv_w(_tmp13a, __msa_slli_w(_out0tm5, 2)), __msa_slli_w(_tmp13b, 3));

                    __msa_st_w(_tmp0m, tmp[0][m], 0);
                    __msa_st_w(_tmp1m, tmp[1][m], 0);
                    __msa_st_w(_tmp2m, tmp[2][m], 0);
                    __msa_st_w(_tmp3m, tmp[3][m], 0);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }
                for (int m = 5; m < 6; m++)
                {
                    v4i32 _out0tm0 = __msa_ld_w(output0_tm_0, 0);
                    v4i32 _out0tm1 = __msa_ld_w(output0_tm_1, 0);
                    v4i32 _out0tm2 = __msa_ld_w(output0_tm_2, 0);
                    v4i32 _out0tm3 = __msa_ld_w(output0_tm_3, 0);
                    v4i32 _out0tm4 = __msa_ld_w(output0_tm_4, 0);
                    v4i32 _out0tm5 = __msa_ld_w(output0_tm_5, 0);

                    v4i32 _tmp02a = __msa_addv_w(_out0tm1, _out0tm2);
                    v4i32 _tmp13a = __msa_subv_w(_out0tm1, _out0tm2);

                    v4i32 _tmp02b = __msa_addv_w(_out0tm3, _out0tm4);
                    v4i32 _tmp13b = __msa_subv_w(_out0tm3, _out0tm4);

                    v4i32 _tmp0m = __msa_addv_w(__msa_addv_w(_out0tm0, _tmp02a), _tmp02b);
                    v4i32 _tmp1m = __msa_addv_w(_tmp13a, __msa_slli_w(_tmp13b, 1));
                    v4i32 _tmp2m = __msa_addv_w(_tmp02a, __msa_slli_w(_tmp02b, 2));
                    v4i32 _tmp3m = __msa_addv_w(__msa_addv_w(_tmp13a, __msa_slli_w(_out0tm5, 2)), __msa_slli_w(_tmp13b, 3));

                    _tmp0m = __msa_slli_w(_tmp0m, 2);
                    _tmp1m = __msa_slli_w(_tmp1m, 2);
                    _tmp2m = __msa_slli_w(_tmp2m, 2);
                    _tmp3m = __msa_slli_w(_tmp3m, 2);

                    __msa_st_w(_tmp0m, tmp[0][m], 0);
                    __msa_st_w(_tmp1m, tmp[1][m], 0);
                    __msa_st_w(_tmp2m, tmp[2][m], 0);
                    __msa_st_w(_tmp3m, tmp[3][m], 0);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }

                for (int m = 0; m < 4; m++)
                {
                    v4i32 _tmp00 = __msa_ld_w(tmp[m][0], 0);
                    v4i32 _tmp01 = __msa_ld_w(tmp[m][1], 0);
                    v4i32 _tmp02 = __msa_ld_w(tmp[m][2], 0);
                    v4i32 _tmp03 = __msa_ld_w(tmp[m][3], 0);
                    v4i32 _tmp04 = __msa_ld_w(tmp[m][4], 0);
                    v4i32 _tmp05 = __msa_ld_w(tmp[m][5], 0);

                    v4i32 _tmp02a = __msa_addv_w(_tmp01, _tmp02);
                    v4i32 _tmp13a = __msa_subv_w(_tmp01, _tmp02);

                    v4i32 _tmp02b = __msa_addv_w(_tmp03, _tmp04);
                    v4i32 _tmp13b = __msa_subv_w(_tmp03, _tmp04);

                    v4i32 _out00 = __msa_addv_w(__msa_addv_w(_tmp00, _tmp02a), _tmp02b);
                    v4i32 _out01 = __msa_addv_w(_tmp13a, __msa_slli_w(_tmp13b, 1));
                    v4i32 _out02 = __msa_addv_w(_tmp02a, __msa_slli_w(_tmp02b, 2));
                    v4i32 _out03 = __msa_addv_w(__msa_addv_w(_tmp05, _tmp13a), __msa_slli_w(_tmp13b, 3));

                    // TODO use integer trick for division by 576
                    v4f32 _v576 = __msa_fill_w_f32(1.0 / 576);
                    _out00 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out00), _v576));
                    _out01 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out01), _v576));
                    _out02 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out02), _v576));
                    _out03 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out03), _v576));

                    __msa_st_w(_out00, output0, 0);
                    __msa_st_w(_out01, output0 + 4, 0);
                    __msa_st_w(_out02, output0 + 8, 0);
                    __msa_st_w(_out03, output0 + 12, 0);

                    output0 += outw * 4;
                }
            }
        }
    }
}
