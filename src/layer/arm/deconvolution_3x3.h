// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static void deconv3x3s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q=0; q<inch; q++)
        {
            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*9 + q*9;

            const float* r0 = img0;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if __ARM_NEON
            float32x4_t _k0 = vld1q_f32(k0);
            float32x4_t _k1 = vld1q_f32(k1);
            float32x4_t _k2 = vld1q_f32(k2);
#endif // __ARM_NEON

            for (int i = 0; i < h; i++)
            {
                float* outptr = out.data + out.w * i;

                float* outptr0 = outptr;
                float* outptr1 = outptr + outw;
                float* outptr2 = outptr + outw*2;

                int j = 0;

#if __ARM_NEON
                for (; j+3 < w; j+=4)
                {
                    float32x4_t _v = vld1q_f32(r0);

#if 0 // bad compiler generate slow instructions :(
                    // 0
                    float32x4_t _out00 = vld1q_f32(outptr0 + 0);
                    _out00 = vmlaq_lane_f32(_out00, _v, vget_low_f32(_k0), 0);

                    float32x4_t _out01 = vmulq_lane_f32(_v, vget_low_f32(_k0), 1);

                    // ext
                    float32x4_t _zero_out01 = vdupq_n_f32(0.f);
                    _zero_out01 = vextq_f32(_zero_out01, _out01, 3);
                    _out00 = vaddq_f32(_out00, _zero_out01);

                    //
                    float32x2_t _out00low = vget_low_f32(_out00);
                    float32x2_t _out00high = vget_high_f32(_out00);

                    _out00high = vmla_lane_f32(_out00high, vget_low_f32(_v), vget_high_f32(_k0), 0);

                    _out00 = vcombine_f32(_out00low, _out00high);

                    vst1q_f32(outptr0 + 0, _out00);

                    //
                    float32x2_t _out02high = vld1_f32(outptr0 + 4);

                    float32x2_t _out01_zero = vext_f32(vget_high_f32(_out01), vget_low_f32(_zero_out01), 1);
                    _out02high = vadd_f32(_out02high, _out01_zero);

                    _out02high = vmla_lane_f32(_out02high, vget_high_f32(_v), vget_high_f32(_k0), 0);

                    vst1_f32(outptr0 + 4, _out02high);

                    // 1
                    float32x4_t _out10 = vld1q_f32(outptr1 + 0);
                    _out10 = vmlaq_lane_f32(_out10, _v, vget_low_f32(_k1), 0);

                    float32x4_t _out11 = vmulq_lane_f32(_v, vget_low_f32(_k1), 1);

                    // ext
                    float32x4_t _zero_out11 = vdupq_n_f32(0.f);
                    _zero_out11 = vextq_f32(_zero_out11, _out11, 3);
                    _out10 = vaddq_f32(_out10, _zero_out11);

                    //
                    float32x2_t _out10low = vget_low_f32(_out10);
                    float32x2_t _out10high = vget_high_f32(_out10);

                    _out10high = vmla_lane_f32(_out10high, vget_low_f32(_v), vget_high_f32(_k1), 0);

                    _out10 = vcombine_f32(_out10low, _out10high);

                    vst1q_f32(outptr1 + 0, _out10);

                    //
                    float32x2_t _out12high = vld1_f32(outptr1 + 4);

                    float32x2_t _out11_zero = vext_f32(vget_high_f32(_out11), vget_low_f32(_zero_out11), 1);
                    _out12high = vadd_f32(_out12high, _out11_zero);

                    _out12high = vmla_lane_f32(_out12high, vget_high_f32(_v), vget_high_f32(_k1), 0);

                    vst1_f32(outptr1 + 4, _out12high);

                    // 2
                    float32x4_t _out20 = vld1q_f32(outptr2 + 0);
                    _out20 = vmlaq_lane_f32(_out20, _v, vget_low_f32(_k2), 0);

                    float32x4_t _out21 = vmulq_lane_f32(_v, vget_low_f32(_k2), 1);

                    // ext
                    float32x4_t _zero_out21 = vdupq_n_f32(0.f);
                    _zero_out21 = vextq_f32(_zero_out21, _out21, 3);
                    _out20 = vaddq_f32(_out20, _zero_out21);

                    //
                    float32x2_t _out20low = vget_low_f32(_out20);
                    float32x2_t _out20high = vget_high_f32(_out20);

                    _out20high = vmla_lane_f32(_out20high, vget_low_f32(_v), vget_high_f32(_k2), 0);

                    _out20 = vcombine_f32(_out20low, _out20high);

                    vst1q_f32(outptr2 + 0, _out20);

                    //
                    float32x2_t _out22high = vld1_f32(outptr2 + 4);

                    float32x2_t _out21_zero = vext_f32(vget_high_f32(_out21), vget_low_f32(_zero_out21), 1);
                    _out22high = vadd_f32(_out22high, _out21_zero);

                    _out22high = vmla_lane_f32(_out22high, vget_high_f32(_v), vget_high_f32(_k2), 0);

                    vst1_f32(outptr2 + 4, _out22high);
#else
                    //
                    float32x4_t _out00 = vld1q_f32(outptr0 + 0);
                    _out00 = vmlaq_lane_f32(_out00, _v, vget_low_f32(_k0), 0);
                    vst1q_f32(outptr0 + 0, _out00);

                    float32x4_t _out01 = vld1q_f32(outptr0 + 1);
                    _out01 = vmlaq_lane_f32(_out01, _v, vget_low_f32(_k0), 1);
                    vst1q_f32(outptr0 + 1, _out01);

                    float32x4_t _out02 = vld1q_f32(outptr0 + 2);
                    _out02 = vmlaq_lane_f32(_out02, _v, vget_high_f32(_k0), 0);
                    vst1q_f32(outptr0 + 2, _out02);

                    //
                    float32x4_t _out10 = vld1q_f32(outptr1 + 0);
                    _out10 = vmlaq_lane_f32(_out10, _v, vget_low_f32(_k1), 0);
                    vst1q_f32(outptr1 + 0, _out10);

                    float32x4_t _out11 = vld1q_f32(outptr1 + 1);
                    _out11 = vmlaq_lane_f32(_out11, _v, vget_low_f32(_k1), 1);
                    vst1q_f32(outptr1 + 1, _out11);

                    float32x4_t _out12 = vld1q_f32(outptr1 + 2);
                    _out12 = vmlaq_lane_f32(_out12, _v, vget_high_f32(_k1), 0);
                    vst1q_f32(outptr1 + 2, _out12);

                    //
                    float32x4_t _out20 = vld1q_f32(outptr2 + 0);
                    _out20 = vmlaq_lane_f32(_out20, _v, vget_low_f32(_k2), 0);
                    vst1q_f32(outptr2 + 0, _out20);

                    float32x4_t _out21 = vld1q_f32(outptr2 + 1);
                    _out21 = vmlaq_lane_f32(_out21, _v, vget_low_f32(_k2), 1);
                    vst1q_f32(outptr2 + 1, _out21);

                    float32x4_t _out22 = vld1q_f32(outptr2 + 2);
                    _out22 = vmlaq_lane_f32(_out22, _v, vget_high_f32(_k2), 0);
                    vst1q_f32(outptr2 + 2, _out22);
#endif

                    r0 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                }
#endif // __ARM_NEON

                for (; j < w; j++)
                {
                    float val = r0[0];

                    outptr0[0] += val * k0[0];
                    outptr0[1] += val * k0[1];
                    outptr0[2] += val * k0[2];

                    outptr1[0] += val * k1[0];
                    outptr1[1] += val * k1[1];
                    outptr1[2] += val * k1[2];

                    outptr2[0] += val * k2[0];
                    outptr2[1] += val * k2[1];
                    outptr2[2] += val * k2[2];

                    r0++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                }
            }
        }
    }
}
