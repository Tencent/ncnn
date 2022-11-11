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

static void conv3x3s1_winograd43_transform_input_pack8_int8_lsx(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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
                    __m128i _r00_01 = __lsx_vld(r0, 0);
                    __m128i _r02_03 = __lsx_vld(r0 + 16, 0);
                    __m128i _r04_05 = __lsx_vld(r0 + 32, 0);
                    __m128i _extr0001 = __lsx_vslti_b(_r00_01, 0);
                    __m128i _extr0203 = __lsx_vslti_b(_r02_03, 0);
                    __m128i _extr0405 = __lsx_vslti_b(_r04_05, 0);
                    __m128i _r00 = __lsx_vilvl_b(_extr0001, _r00_01);
                    __m128i _r01 = __lsx_vilvh_b(_extr0001, _r00_01);
                    __m128i _r02 = __lsx_vilvl_b(_extr0203, _r02_03);
                    __m128i _r03 = __lsx_vilvh_b(_extr0203, _r02_03);
                    __m128i _r04 = __lsx_vilvl_b(_extr0405, _r04_05);
                    __m128i _r05 = __lsx_vilvh_b(_extr0405, _r04_05);

                    __m128i _v5 = __lsx_vreplgr2vr_h(5);

                    __m128i _tmp0m = __lsx_vsub_h(__lsx_vadd_h(__lsx_vslli_h(_r00, 2), _r04), __lsx_vmul_h(_r02, _v5));
                    __m128i _tmp1m = __lsx_vsub_h(__lsx_vadd_h(_r04, _r03), __lsx_vslli_h(__lsx_vadd_h(_r01, _r02), 2));
                    __m128i _tmp2m = __lsx_vadd_h(__lsx_vsub_h(_r04, _r03), __lsx_vslli_h(__lsx_vsub_h(_r01, _r02), 2));
                    __m128i _tmp3m = __lsx_vsub_h(__lsx_vsub_h(_r04, _r02), __lsx_vslli_h(__lsx_vsub_h(_r01, _r03), 1));
                    __m128i _tmp4m = __lsx_vadd_h(__lsx_vsub_h(_r04, _r02), __lsx_vslli_h(__lsx_vsub_h(_r01, _r03), 1));
                    __m128i _tmp5m = __lsx_vsub_h(__lsx_vadd_h(__lsx_vslli_h(_r01, 2), _r05), __lsx_vmul_h(_r03, _v5));

                    __lsx_vst(_tmp0m, tmp[0][m], 0);
                    __lsx_vst(_tmp1m, tmp[1][m], 0);
                    __lsx_vst(_tmp2m, tmp[2][m], 0);
                    __lsx_vst(_tmp3m, tmp[3][m], 0);
                    __lsx_vst(_tmp4m, tmp[4][m], 0);
                    __lsx_vst(_tmp5m, tmp[5][m], 0);

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
                    __m128i _tmp00 = __lsx_vld(tmp[m][0], 0);
                    __m128i _tmp01 = __lsx_vld(tmp[m][1], 0);
                    __m128i _tmp02 = __lsx_vld(tmp[m][2], 0);
                    __m128i _tmp03 = __lsx_vld(tmp[m][3], 0);
                    __m128i _tmp04 = __lsx_vld(tmp[m][4], 0);
                    __m128i _tmp05 = __lsx_vld(tmp[m][5], 0);

                    __m128i _v5 = __lsx_vreplgr2vr_h(5);

                    __m128i _r0tm0 = __lsx_vsub_h(__lsx_vadd_h(__lsx_vslli_h(_tmp00, 2), _tmp04), __lsx_vmul_h(_tmp02, _v5));
                    __m128i _r0tm1 = __lsx_vsub_h(__lsx_vadd_h(_tmp04, _tmp03), __lsx_vslli_h(__lsx_vadd_h(_tmp01, _tmp02), 2));
                    __m128i _r0tm2 = __lsx_vadd_h(__lsx_vsub_h(_tmp04, _tmp03), __lsx_vslli_h(__lsx_vsub_h(_tmp01, _tmp02), 2));
                    __m128i _r0tm3 = __lsx_vsub_h(__lsx_vsub_h(_tmp04, _tmp02), __lsx_vslli_h(__lsx_vsub_h(_tmp01, _tmp03), 1));
                    __m128i _r0tm4 = __lsx_vadd_h(__lsx_vsub_h(_tmp04, _tmp02), __lsx_vslli_h(__lsx_vsub_h(_tmp01, _tmp03), 1));
                    __m128i _r0tm5 = __lsx_vsub_h(__lsx_vadd_h(__lsx_vslli_h(_tmp01, 2), _tmp05), __lsx_vmul_h(_tmp03, _v5));

                    __lsx_vst(_r0tm0, r0_tm_0, 0);
                    __lsx_vst(_r0tm1, r0_tm_1, 0);
                    __lsx_vst(_r0tm2, r0_tm_2, 0);
                    __lsx_vst(_r0tm3, r0_tm_3, 0);
                    __lsx_vst(_r0tm4, r0_tm_4, 0);
                    __lsx_vst(_r0tm5, r0_tm_5, 0);

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
