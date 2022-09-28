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

static void conv3x3s1_winograd43_transform_input_pack8_int8_neon(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
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
                    int8x8_t _r00 = vld1_s8(r0);
                    int8x8_t _r01 = vld1_s8(r0 + 8);
                    int8x8_t _r02 = vld1_s8(r0 + 16);
                    int8x8_t _r03 = vld1_s8(r0 + 24);
                    int8x8_t _r04 = vld1_s8(r0 + 32);
                    int8x8_t _r05 = vld1_s8(r0 + 40);

                    int8x8_t _v4s8 = vdup_n_s8(4);
                    int8x8_t _v5s8 = vdup_n_s8(5);
                    int16x8_t _v2 = vdupq_n_s16(2);
                    int16x8_t _v4 = vdupq_n_s16(4);

                    int16x8_t _tmp0m = vsubq_s16(vaddw_s8(vmull_s8(_r00, _v4s8), _r04), vmull_s8(_r02, _v5s8));
                    int16x8_t _tmp1m = vmlsq_s16(vaddl_s8(_r04, _r03), vaddl_s8(_r01, _r02), _v4);
                    int16x8_t _tmp2m = vmlaq_s16(vsubl_s8(_r04, _r03), vsubl_s8(_r01, _r02), _v4);
                    int16x8_t _tmp3m = vmlsq_s16(vsubl_s8(_r04, _r02), vsubl_s8(_r01, _r03), _v2);
                    int16x8_t _tmp4m = vmlaq_s16(vsubl_s8(_r04, _r02), vsubl_s8(_r01, _r03), _v2);
                    int16x8_t _tmp5m = vsubq_s16(vaddw_s8(vmull_s8(_r01, _v4s8), _r05), vmull_s8(_r03, _v5s8));

                    vst1q_s16(tmp[0][m], _tmp0m);
                    vst1q_s16(tmp[1][m], _tmp1m);
                    vst1q_s16(tmp[2][m], _tmp2m);
                    vst1q_s16(tmp[3][m], _tmp3m);
                    vst1q_s16(tmp[4][m], _tmp4m);
                    vst1q_s16(tmp[5][m], _tmp5m);

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
                    int16x8_t _tmp00 = vld1q_s16(tmp[m][0]);
                    int16x8_t _tmp01 = vld1q_s16(tmp[m][1]);
                    int16x8_t _tmp02 = vld1q_s16(tmp[m][2]);
                    int16x8_t _tmp03 = vld1q_s16(tmp[m][3]);
                    int16x8_t _tmp04 = vld1q_s16(tmp[m][4]);
                    int16x8_t _tmp05 = vld1q_s16(tmp[m][5]);

                    int16x8_t _v2 = vdupq_n_s16(2);
                    int16x8_t _v4 = vdupq_n_s16(4);
                    int16x8_t _v5 = vdupq_n_s16(5);

                    int16x8_t _r0tm0 = vmlsq_s16(vmlaq_s16(_tmp04, _tmp00, _v4), _tmp02, _v5);
                    int16x8_t _r0tm1 = vmlsq_s16(vaddq_s16(_tmp04, _tmp03), vaddq_s16(_tmp01, _tmp02), _v4);
                    int16x8_t _r0tm2 = vmlaq_s16(vsubq_s16(_tmp04, _tmp03), vsubq_s16(_tmp01, _tmp02), _v4);
                    int16x8_t _r0tm3 = vmlsq_s16(vsubq_s16(_tmp04, _tmp02), vsubq_s16(_tmp01, _tmp03), _v2);
                    int16x8_t _r0tm4 = vmlaq_s16(vsubq_s16(_tmp04, _tmp02), vsubq_s16(_tmp01, _tmp03), _v2);
                    int16x8_t _r0tm5 = vmlsq_s16(vmlaq_s16(_tmp05, _tmp01, _v4), _tmp03, _v5);

                    vst1q_s16(r0_tm_0, _r0tm0);
                    vst1q_s16(r0_tm_1, _r0tm1);
                    vst1q_s16(r0_tm_2, _r0tm2);
                    vst1q_s16(r0_tm_3, _r0tm3);
                    vst1q_s16(r0_tm_4, _r0tm4);
                    vst1q_s16(r0_tm_5, _r0tm5);

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
