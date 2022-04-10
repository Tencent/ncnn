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

static void conv3x3s1_winograd64_transform_output_bf16s_neon(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
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

        const float bias0 = biasptr ? biasptr[p] : 0.f;

        float tmp[6][8];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j);
                const float* output0_tm_1 = output0_tm_0 + tiles * 1;
                const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * 5;
                const float* output0_tm_6 = output0_tm_0 + tiles * 6;
                const float* output0_tm_7 = output0_tm_0 + tiles * 7;

                // TODO neon optimize
                for (int m = 0; m < 8; m++)
                {
                    float tmp024a = output0_tm_1[0] + output0_tm_2[0];
                    float tmp135a = output0_tm_1[0] - output0_tm_2[0];

                    float tmp024b = output0_tm_3[0] + output0_tm_4[0];
                    float tmp135b = output0_tm_3[0] - output0_tm_4[0];

                    float tmp024c = output0_tm_5[0] + output0_tm_6[0];
                    float tmp135c = output0_tm_5[0] - output0_tm_6[0];

                    tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                    tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                    tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                    tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                    tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                    tmp[5][m] = output0_tm_7[0] + tmp135a + tmp135b * 32 + tmp135c;

                    output0_tm_0 += tiles * 8;
                    output0_tm_1 += tiles * 8;
                    output0_tm_2 += tiles * 8;
                    output0_tm_3 += tiles * 8;
                    output0_tm_4 += tiles * 8;
                    output0_tm_5 += tiles * 8;
                    output0_tm_6 += tiles * 8;
                    output0_tm_7 += tiles * 8;
                }

                unsigned short* output0 = out0.row<unsigned short>(i * 6) + j * 6;

                for (int m = 0; m < 6; m++)
                {
                    const float* tmp0 = tmp[m];

                    float tmp024a = tmp0[1] + tmp0[2];
                    float tmp135a = tmp0[1] - tmp0[2];

                    float tmp024b = tmp0[3] + tmp0[4];
                    float tmp135b = tmp0[3] - tmp0[4];

                    float tmp024c = tmp0[5] + tmp0[6];
                    float tmp135c = tmp0[5] - tmp0[6];

                    output0[0] = float32_to_bfloat16(bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32);
                    output0[2] = float32_to_bfloat16(bias0 + tmp024a + tmp024b * 4 + tmp024c * 8);
                    output0[4] = float32_to_bfloat16(bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c);

                    output0[1] = float32_to_bfloat16(bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16);
                    output0[3] = float32_to_bfloat16(bias0 + tmp135a + tmp135b * 8 + tmp135c * 4);
                    output0[5] = float32_to_bfloat16(bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c);

                    output0 += outw;
                }
            }
        }
    }
}
