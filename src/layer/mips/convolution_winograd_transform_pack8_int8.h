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

static void conv3x3s1_winograd43_transform_input_pack8_int8_msa(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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

        short tmp[6][6][8];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const signed char* r0 = img0.row<const signed char>(i * 4) + (j * 4) * 8;

                for (int m = 0; m < 6; m++)
                {
                    v16i8 _r00_01 = __msa_ld_b(r0, 0);
                    v16i8 _r02_03 = __msa_ld_b(r0 + 16, 0);
                    v16i8 _r04_05 = __msa_ld_b(r0 + 32, 0);
                    v16i8 _extr0001 = __msa_clti_s_b(_r00_01, 0);
                    v16i8 _extr0203 = __msa_clti_s_b(_r02_03, 0);
                    v16i8 _extr0405 = __msa_clti_s_b(_r04_05, 0);
                    v8i16 _r00 = (v8i16)__msa_ilvr_b(_extr0001, _r00_01);
                    v8i16 _r01 = (v8i16)__msa_ilvl_b(_extr0001, _r00_01);
                    v8i16 _r02 = (v8i16)__msa_ilvr_b(_extr0203, _r02_03);
                    v8i16 _r03 = (v8i16)__msa_ilvl_b(_extr0203, _r02_03);
                    v8i16 _r04 = (v8i16)__msa_ilvr_b(_extr0405, _r04_05);
                    v8i16 _r05 = (v8i16)__msa_ilvl_b(_extr0405, _r04_05);

                    v8i16 _v5 = __msa_fill_h(5);

                    v8i16 _tmp0m = __msa_subv_h(__msa_addv_h(__msa_slli_h(_r00, 2), _r04), __msa_mulv_h(_r02, _v5));
                    v8i16 _tmp1m = __msa_subv_h(__msa_addv_h(_r04, _r03), __msa_slli_h(__msa_addv_h(_r01, _r02), 2));
                    v8i16 _tmp2m = __msa_addv_h(__msa_subv_h(_r04, _r03), __msa_slli_h(__msa_subv_h(_r01, _r02), 2));
                    v8i16 _tmp3m = __msa_subv_h(__msa_subv_h(_r04, _r02), __msa_slli_h(__msa_subv_h(_r01, _r03), 1));
                    v8i16 _tmp4m = __msa_addv_h(__msa_subv_h(_r04, _r02), __msa_slli_h(__msa_subv_h(_r01, _r03), 1));
                    v8i16 _tmp5m = __msa_subv_h(__msa_addv_h(__msa_slli_h(_r01, 2), _r05), __msa_mulv_h(_r03, _v5));

                    __msa_st_h(_tmp0m, tmp[0][m], 0);
                    __msa_st_h(_tmp1m, tmp[1][m], 0);
                    __msa_st_h(_tmp2m, tmp[2][m], 0);
                    __msa_st_h(_tmp3m, tmp[3][m], 0);
                    __msa_st_h(_tmp4m, tmp[4][m], 0);
                    __msa_st_h(_tmp5m, tmp[5][m], 0);

                    r0 += w * 8;
                }

                short* r0_tm_0 = (short*)img0_tm + (i * w_tiles + j) * 8;
                short* r0_tm_1 = r0_tm_0 + tiles * 8;
                short* r0_tm_2 = r0_tm_0 + tiles * 16;
                short* r0_tm_3 = r0_tm_0 + tiles * 24;
                short* r0_tm_4 = r0_tm_0 + tiles * 32;
                short* r0_tm_5 = r0_tm_0 + tiles * 40;

                for (int m = 0; m < 6; m++)
                {
                    v8i16 _tmp00 = __msa_ld_h(tmp[m][0], 0);
                    v8i16 _tmp01 = __msa_ld_h(tmp[m][1], 0);
                    v8i16 _tmp02 = __msa_ld_h(tmp[m][2], 0);
                    v8i16 _tmp03 = __msa_ld_h(tmp[m][3], 0);
                    v8i16 _tmp04 = __msa_ld_h(tmp[m][4], 0);
                    v8i16 _tmp05 = __msa_ld_h(tmp[m][5], 0);

                    v8i16 _v5 = __msa_fill_h(5);

                    v8i16 _r0tm0 = __msa_subv_h(__msa_addv_h(__msa_slli_h(_tmp00, 2), _tmp04), __msa_mulv_h(_tmp02, _v5));
                    v8i16 _r0tm1 = __msa_subv_h(__msa_addv_h(_tmp04, _tmp03), __msa_slli_h(__msa_addv_h(_tmp01, _tmp02), 2));
                    v8i16 _r0tm2 = __msa_addv_h(__msa_subv_h(_tmp04, _tmp03), __msa_slli_h(__msa_subv_h(_tmp01, _tmp02), 2));
                    v8i16 _r0tm3 = __msa_subv_h(__msa_subv_h(_tmp04, _tmp02), __msa_slli_h(__msa_subv_h(_tmp01, _tmp03), 1));
                    v8i16 _r0tm4 = __msa_addv_h(__msa_subv_h(_tmp04, _tmp02), __msa_slli_h(__msa_subv_h(_tmp01, _tmp03), 1));
                    v8i16 _r0tm5 = __msa_subv_h(__msa_addv_h(__msa_slli_h(_tmp01, 2), _tmp05), __msa_mulv_h(_tmp03, _v5));

                    __msa_st_h(_r0tm0, r0_tm_0, 0);
                    __msa_st_h(_r0tm1, r0_tm_1, 0);
                    __msa_st_h(_r0tm2, r0_tm_2, 0);
                    __msa_st_h(_r0tm3, r0_tm_3, 0);
                    __msa_st_h(_r0tm4, r0_tm_4, 0);
                    __msa_st_h(_r0tm5, r0_tm_5, 0);

                    r0_tm_0 += tiles * 48;
                    r0_tm_1 += tiles * 48;
                    r0_tm_2 += tiles * 48;
                    r0_tm_3 += tiles * 48;
                    r0_tm_4 += tiles * 48;
                    r0_tm_5 += tiles * 48;
                }
            }
        }
    }
}
