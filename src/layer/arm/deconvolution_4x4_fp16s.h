// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void deconv4x4s2_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outch = top_blob.c;

    const __fp16* kernel = _kernel;
    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const __fp16 bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            const __fp16* img0 = bottom_blob.channel(q);

            const __fp16* kernel0 = kernel + p * inch * 16 + q * 16;

            const __fp16* r0 = img0;

            const __fp16* k0 = kernel0;
            const __fp16* k1 = kernel0 + 4;
            const __fp16* k2 = kernel0 + 8;
            const __fp16* k3 = kernel0 + 12;

            float16x4_t _k0 = vld1_f16(k0);
            float16x4_t _k1 = vld1_f16(k1);
            float16x4_t _k2 = vld1_f16(k2);
            float16x4_t _k3 = vld1_f16(k3);

            for (int i = 0; i < h; i++)
            {
                __fp16* outptr = out.row<__fp16>(i * 2);

                __fp16* outptr0 = outptr;
                __fp16* outptr1 = outptr0 + outw;
                __fp16* outptr2 = outptr1 + outw;
                __fp16* outptr3 = outptr2 + outw;

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _v = vld1_f16(r0);

                    // row 0
                    float16x4x2_t _out0 = vld2_f16(outptr0);
                    // 0,2,4,6
                    _out0.val[0] = vfma_lane_f16(_out0.val[0], _v, _k0, 0);
                    // 1,3,5,7
                    _out0.val[1] = vfma_lane_f16(_out0.val[1], _v, _k0, 1);
                    vst2_f16(outptr0, _out0);

                    _out0 = vld2_f16(outptr0 + 2);
                    // 2,4,6,8
                    _out0.val[0] = vfma_lane_f16(_out0.val[0], _v, _k0, 2);
                    // 3,5,7,9
                    _out0.val[1] = vfma_lane_f16(_out0.val[1], _v, _k0, 3);
                    vst2_f16(outptr0 + 2, _out0);

                    // row 1
                    float16x4x2_t _out1 = vld2_f16(outptr1);
                    // 0,2,4,6
                    _out1.val[0] = vfma_lane_f16(_out1.val[0], _v, _k1, 0);
                    // 1,3,5,7
                    _out1.val[1] = vfma_lane_f16(_out1.val[1], _v, _k1, 1);
                    vst2_f16(outptr1, _out1);

                    _out1 = vld2_f16(outptr1 + 2);
                    // 2,4,6,8
                    _out1.val[0] = vfma_lane_f16(_out1.val[0], _v, _k1, 2);
                    // 3,5,7,9
                    _out1.val[1] = vfma_lane_f16(_out1.val[1], _v, _k1, 3);
                    vst2_f16(outptr1 + 2, _out1);

                    // row 2
                    float16x4x2_t _out2 = vld2_f16(outptr2);
                    _out2.val[0] = vfma_lane_f16(_out2.val[0], _v, _k2, 0);
                    _out2.val[1] = vfma_lane_f16(_out2.val[1], _v, _k2, 1);
                    vst2_f16(outptr2, _out2);

                    _out2 = vld2_f16(outptr2 + 2);
                    _out2.val[0] = vfma_lane_f16(_out2.val[0], _v, _k2, 2);
                    _out2.val[1] = vfma_lane_f16(_out2.val[1], _v, _k2, 3);
                    vst2_f16(outptr2 + 2, _out2);

                    // row 3
                    float16x4x2_t _out3 = vld2_f16(outptr3);
                    _out3.val[0] = vfma_lane_f16(_out3.val[0], _v, _k3, 0);
                    _out3.val[1] = vfma_lane_f16(_out3.val[1], _v, _k3, 1);
                    vst2_f16(outptr3, _out3);

                    _out3 = vld2_f16(outptr3 + 2);
                    _out3.val[0] = vfma_lane_f16(_out3.val[0], _v, _k3, 2);
                    _out3.val[1] = vfma_lane_f16(_out3.val[1], _v, _k3, 3);
                    vst2_f16(outptr3 + 2, _out3);

                    r0 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }
                for (; j < w; j++)
                {
                    __fp16 val = r0[0];

                    outptr0[0] += val * k0[0];
                    outptr0[1] += val * k0[1];
                    outptr0[2] += val * k0[2];
                    outptr0[3] += val * k0[3];

                    outptr1[0] += val * k1[0];
                    outptr1[1] += val * k1[1];
                    outptr1[2] += val * k1[2];
                    outptr1[3] += val * k1[3];

                    outptr2[0] += val * k2[0];
                    outptr2[1] += val * k2[1];
                    outptr2[2] += val * k2[2];
                    outptr2[3] += val * k2[3];

                    outptr3[0] += val * k3[0];
                    outptr3[1] += val * k3[1];
                    outptr3[2] += val * k3[2];
                    outptr3[3] += val * k3[3];

                    r0++;
                    outptr0 += 2;
                    outptr1 += 2;
                    outptr2 += 2;
                    outptr3 += 2;
                }
            }
        }
    }
}
