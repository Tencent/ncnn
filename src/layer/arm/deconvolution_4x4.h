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

static void deconv4x4s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch * 16 + q * 16;

            const float* r0 = img0;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 4;
            const float* k2 = kernel0 + 8;
            const float* k3 = kernel0 + 12;

#if __ARM_NEON
            float32x4_t _k0 = vld1q_f32(k0);
            float32x4_t _k1 = vld1q_f32(k1);
            float32x4_t _k2 = vld1q_f32(k2);
            float32x4_t _k3 = vld1q_f32(k3);
#endif // __ARM_NEON

            for (int i = 0; i < h; i++)
            {
                float* outptr = out.row(i);

                float* outptr0 = outptr;
                float* outptr1 = outptr0 + outw;
                float* outptr2 = outptr1 + outw;
                float* outptr3 = outptr2 + outw;

                int j = 0;

#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vld1q_f32(r0);

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

                    float32x4_t _out03 = vld1q_f32(outptr0 + 3);
                    _out03 = vmlaq_lane_f32(_out03, _v, vget_high_f32(_k0), 1);
                    vst1q_f32(outptr0 + 3, _out03);

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

                    float32x4_t _out13 = vld1q_f32(outptr1 + 3);
                    _out13 = vmlaq_lane_f32(_out13, _v, vget_high_f32(_k1), 1);
                    vst1q_f32(outptr1 + 3, _out13);

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

                    float32x4_t _out23 = vld1q_f32(outptr2 + 3);
                    _out23 = vmlaq_lane_f32(_out23, _v, vget_high_f32(_k2), 1);
                    vst1q_f32(outptr2 + 3, _out23);

                    //
                    float32x4_t _out30 = vld1q_f32(outptr3 + 0);
                    _out30 = vmlaq_lane_f32(_out30, _v, vget_low_f32(_k3), 0);
                    vst1q_f32(outptr3 + 0, _out30);

                    float32x4_t _out31 = vld1q_f32(outptr3 + 1);
                    _out31 = vmlaq_lane_f32(_out31, _v, vget_low_f32(_k3), 1);
                    vst1q_f32(outptr3 + 1, _out31);

                    float32x4_t _out32 = vld1q_f32(outptr3 + 2);
                    _out32 = vmlaq_lane_f32(_out32, _v, vget_high_f32(_k3), 0);
                    vst1q_f32(outptr3 + 2, _out32);

                    float32x4_t _out33 = vld1q_f32(outptr3 + 3);
                    _out33 = vmlaq_lane_f32(_out33, _v, vget_high_f32(_k3), 1);
                    vst1q_f32(outptr3 + 3, _out33);

                    r0 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }

#endif // __ARM_NEON

                for (; j < w; j++)
                {
                    float val = r0[0];

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
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }
            }
        }
    }
}

static void deconv4x4s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch * 16 + q * 16;

            const float* r0 = img0;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 4;
            const float* k2 = kernel0 + 8;
            const float* k3 = kernel0 + 12;

#if __ARM_NEON
            float32x4_t _k0 = vld1q_f32(k0);
            float32x4_t _k1 = vld1q_f32(k1);
            float32x4_t _k2 = vld1q_f32(k2);
            float32x4_t _k3 = vld1q_f32(k3);
#endif // __ARM_NEON

            for (int i = 0; i < h; i++)
            {
                float* outptr = out.row(i * 2);

                float* outptr0 = outptr;
                float* outptr1 = outptr0 + outw;
                float* outptr2 = outptr1 + outw;
                float* outptr3 = outptr2 + outw;

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vld1q_f32(r0);

                    // row 0
                    float32x4x2_t _out0 = vld2q_f32(outptr0);
                    // 0,2,4,6
                    _out0.val[0] = vmlaq_lane_f32(_out0.val[0], _v, vget_low_f32(_k0), 0);
                    // 1,3,5,7
                    _out0.val[1] = vmlaq_lane_f32(_out0.val[1], _v, vget_low_f32(_k0), 1);
                    vst2q_f32(outptr0, _out0);

                    _out0 = vld2q_f32(outptr0 + 2);
                    // 2,4,6,8
                    _out0.val[0] = vmlaq_lane_f32(_out0.val[0], _v, vget_high_f32(_k0), 0);
                    // 3,5,7,9
                    _out0.val[1] = vmlaq_lane_f32(_out0.val[1], _v, vget_high_f32(_k0), 1);
                    vst2q_f32(outptr0 + 2, _out0);

                    // row 1
                    float32x4x2_t _out1 = vld2q_f32(outptr1);
                    // 0,2,4,6
                    _out1.val[0] = vmlaq_lane_f32(_out1.val[0], _v, vget_low_f32(_k1), 0);
                    // 1,3,5,7
                    _out1.val[1] = vmlaq_lane_f32(_out1.val[1], _v, vget_low_f32(_k1), 1);
                    vst2q_f32(outptr1, _out1);

                    _out1 = vld2q_f32(outptr1 + 2);
                    // 2,4,6,8
                    _out1.val[0] = vmlaq_lane_f32(_out1.val[0], _v, vget_high_f32(_k1), 0);
                    // 3,5,7,9
                    _out1.val[1] = vmlaq_lane_f32(_out1.val[1], _v, vget_high_f32(_k1), 1);
                    vst2q_f32(outptr1 + 2, _out1);

                    // row 2
                    float32x4x2_t _out2 = vld2q_f32(outptr2);
                    _out2.val[0] = vmlaq_lane_f32(_out2.val[0], _v, vget_low_f32(_k2), 0);
                    _out2.val[1] = vmlaq_lane_f32(_out2.val[1], _v, vget_low_f32(_k2), 1);
                    vst2q_f32(outptr2, _out2);

                    _out2 = vld2q_f32(outptr2 + 2);
                    _out2.val[0] = vmlaq_lane_f32(_out2.val[0], _v, vget_high_f32(_k2), 0);
                    _out2.val[1] = vmlaq_lane_f32(_out2.val[1], _v, vget_high_f32(_k2), 1);
                    vst2q_f32(outptr2 + 2, _out2);

                    // row 3
                    float32x4x2_t _out3 = vld2q_f32(outptr3);
                    _out3.val[0] = vmlaq_lane_f32(_out3.val[0], _v, vget_low_f32(_k3), 0);
                    _out3.val[1] = vmlaq_lane_f32(_out3.val[1], _v, vget_low_f32(_k3), 1);
                    vst2q_f32(outptr3, _out3);

                    _out3 = vld2q_f32(outptr3 + 2);
                    _out3.val[0] = vmlaq_lane_f32(_out3.val[0], _v, vget_high_f32(_k3), 0);
                    _out3.val[1] = vmlaq_lane_f32(_out3.val[1], _v, vget_high_f32(_k3), 1);
                    vst2q_f32(outptr3 + 2, _out3);

                    r0 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }

#endif // __ARM_NEON

                for (; j < w; j++)
                {
                    float val = r0[0];

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
