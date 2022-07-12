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

static void conv3x3s1_winograd43_transform_output_pack4_int8_neon(const Mat& top_blob_tm, Mat& top_blob, const Option& opt)
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
                    int32x4_t _out0tm0 = vld1q_s32(output0_tm_0);
                    int32x4_t _out0tm1 = vld1q_s32(output0_tm_1);
                    int32x4_t _out0tm2 = vld1q_s32(output0_tm_2);
                    int32x4_t _out0tm3 = vld1q_s32(output0_tm_3);
                    int32x4_t _out0tm4 = vld1q_s32(output0_tm_4);
                    int32x4_t _out0tm5 = vld1q_s32(output0_tm_5);

                    int32x4_t _tmp02a = vaddq_s32(_out0tm1, _out0tm2);
                    int32x4_t _tmp13a = vsubq_s32(_out0tm1, _out0tm2);

                    int32x4_t _tmp02b = vaddq_s32(_out0tm3, _out0tm4);
                    int32x4_t _tmp13b = vsubq_s32(_out0tm3, _out0tm4);

                    int32x4_t _v2 = vdupq_n_s32(2);
                    int32x4_t _v4 = vdupq_n_s32(4);
                    int32x4_t _v8 = vdupq_n_s32(8);

                    int32x4_t _tmp0m = vaddq_s32(vaddq_s32(_out0tm0, _tmp02a), _tmp02b);
                    int32x4_t _tmp1m = vmlaq_s32(_tmp13a, _tmp13b, _v2);
                    int32x4_t _tmp2m = vmlaq_s32(_tmp02a, _tmp02b, _v4);
                    int32x4_t _tmp3m = vmlaq_s32(vmlaq_s32(_tmp13a, _out0tm5, _v4), _tmp13b, _v8);

                    vst1q_s32(tmp[0][m], _tmp0m);
                    vst1q_s32(tmp[1][m], _tmp1m);
                    vst1q_s32(tmp[2][m], _tmp2m);
                    vst1q_s32(tmp[3][m], _tmp3m);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }
                for (int m = 5; m < 6; m++)
                {
                    int32x4_t _out0tm0 = vld1q_s32(output0_tm_0);
                    int32x4_t _out0tm1 = vld1q_s32(output0_tm_1);
                    int32x4_t _out0tm2 = vld1q_s32(output0_tm_2);
                    int32x4_t _out0tm3 = vld1q_s32(output0_tm_3);
                    int32x4_t _out0tm4 = vld1q_s32(output0_tm_4);
                    int32x4_t _out0tm5 = vld1q_s32(output0_tm_5);

                    int32x4_t _tmp02a = vaddq_s32(_out0tm1, _out0tm2);
                    int32x4_t _tmp13a = vsubq_s32(_out0tm1, _out0tm2);

                    int32x4_t _tmp02b = vaddq_s32(_out0tm3, _out0tm4);
                    int32x4_t _tmp13b = vsubq_s32(_out0tm3, _out0tm4);

                    int32x4_t _v2 = vdupq_n_s32(2);
                    int32x4_t _v4 = vdupq_n_s32(4);
                    int32x4_t _v8 = vdupq_n_s32(8);

                    int32x4_t _tmp0m = vaddq_s32(vaddq_s32(_out0tm0, _tmp02a), _tmp02b);
                    int32x4_t _tmp1m = vmlaq_s32(_tmp13a, _tmp13b, _v2);
                    int32x4_t _tmp2m = vmlaq_s32(_tmp02a, _tmp02b, _v4);
                    int32x4_t _tmp3m = vmlaq_s32(vmlaq_s32(_tmp13a, _out0tm5, _v4), _tmp13b, _v8);

                    _tmp0m = vmulq_s32(_tmp0m, _v4);
                    _tmp1m = vmulq_s32(_tmp1m, _v4);
                    _tmp2m = vmulq_s32(_tmp2m, _v4);
                    _tmp3m = vmulq_s32(_tmp3m, _v4);

                    vst1q_s32(tmp[0][m], _tmp0m);
                    vst1q_s32(tmp[1][m], _tmp1m);
                    vst1q_s32(tmp[2][m], _tmp2m);
                    vst1q_s32(tmp[3][m], _tmp3m);

                    output0_tm_0 += tiles * 24;
                    output0_tm_1 += tiles * 24;
                    output0_tm_2 += tiles * 24;
                    output0_tm_3 += tiles * 24;
                    output0_tm_4 += tiles * 24;
                    output0_tm_5 += tiles * 24;
                }

                for (int m = 0; m < 4; m++)
                {
                    int32x4_t _tmp00 = vld1q_s32(tmp[m][0]);
                    int32x4_t _tmp01 = vld1q_s32(tmp[m][1]);
                    int32x4_t _tmp02 = vld1q_s32(tmp[m][2]);
                    int32x4_t _tmp03 = vld1q_s32(tmp[m][3]);
                    int32x4_t _tmp04 = vld1q_s32(tmp[m][4]);
                    int32x4_t _tmp05 = vld1q_s32(tmp[m][5]);

                    int32x4_t _tmp02a = vaddq_s32(_tmp01, _tmp02);
                    int32x4_t _tmp13a = vsubq_s32(_tmp01, _tmp02);

                    int32x4_t _tmp02b = vaddq_s32(_tmp03, _tmp04);
                    int32x4_t _tmp13b = vsubq_s32(_tmp03, _tmp04);

                    int32x4_t _v2 = vdupq_n_s32(2);
                    int32x4_t _v4 = vdupq_n_s32(4);
                    int32x4_t _v8 = vdupq_n_s32(8);

                    int32x4_t _out00 = vaddq_s32(vaddq_s32(_tmp00, _tmp02a), _tmp02b);
                    int32x4_t _out01 = vmlaq_s32(_tmp13a, _tmp13b, _v2);
                    int32x4_t _out02 = vmlaq_s32(_tmp02a, _tmp02b, _v4);
                    int32x4_t _out03 = vmlaq_s32(vaddq_s32(_tmp05, _tmp13a), _tmp13b, _v8);

                    // TODO use integer trick for division by 576
                    float32x4_t _v576 = vdupq_n_f32(1.0 / 576);
                    _out00 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out00), _v576));
                    _out01 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out01), _v576));
                    _out02 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out02), _v576));
                    _out03 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(_out03), _v576));

                    vst1q_s32(output0, _out00);
                    vst1q_s32(output0 + 4, _out01);
                    vst1q_s32(output0 + 8, _out02);
                    vst1q_s32(output0 + 12, _out03);

                    output0 += outw * 4;
                }
            }
        }
    }
}
