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

static void convdw3x3s1_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const __fp16* kernel = _kernel;
    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const __fp16 bias0 = bias ? bias[g] : 0.f;

        const __fp16* kernel0 = kernel + g * 9;

        __fp16* outptr0 = out;
        __fp16* outptr1 = outptr0 + outw;

        const __fp16* img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0;
        const __fp16* r1 = img0 + w;
        const __fp16* r2 = img0 + w * 2;
        const __fp16* r3 = img0 + w * 3;

        float16x4_t _k012x = vld1_f16(kernel0);
        float16x4_t _k345x = vld1_f16(kernel0 + 3);
        float16x4_t _k678x = vld1_f16(kernel0 + 6);

        _k012x = vset_lane_f16(0.f, _k012x, 3);
        _k345x = vset_lane_f16(0.f, _k345x, 3);
        _k678x = vset_lane_f16(0.f, _k678x, 3);

        float16x8_t _bias0 = vdupq_n_f16(bias0);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 7 < outw; j += 8)
            {
                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r20 = vld1q_f16(r2);
                float16x8_t _r30 = vld1q_f16(r3);

                float16x8_t _r0n = vld1q_f16(r0 + 8);
                float16x8_t _r1n = vld1q_f16(r1 + 8);
                float16x8_t _r2n = vld1q_f16(r2 + 8);
                float16x8_t _r3n = vld1q_f16(r3 + 8);

                float16x8_t _r01 = vextq_f16(_r00, _r0n, 1);
                float16x8_t _r11 = vextq_f16(_r10, _r1n, 1);
                float16x8_t _r21 = vextq_f16(_r20, _r2n, 1);
                float16x8_t _r31 = vextq_f16(_r30, _r3n, 1);

                float16x8_t _r02 = vextq_f16(_r00, _r0n, 2);
                float16x8_t _r12 = vextq_f16(_r10, _r1n, 2);
                float16x8_t _r22 = vextq_f16(_r20, _r2n, 2);
                float16x8_t _r32 = vextq_f16(_r30, _r3n, 2);

                float16x8_t _sum0 = _bias0;
                float16x8_t _sum1 = _bias0;

                _sum0 = vfmaq_lane_f16(_sum0, _r00, _k012x, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _r01, _k012x, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _r02, _k012x, 2);
                _sum1 = vfmaq_lane_f16(_sum1, _r10, _k012x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _r11, _k012x, 1);
                _sum1 = vfmaq_lane_f16(_sum1, _r12, _k012x, 2);

                _sum0 = vfmaq_lane_f16(_sum0, _r10, _k345x, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _r11, _k345x, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _r12, _k345x, 2);
                _sum1 = vfmaq_lane_f16(_sum1, _r20, _k345x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _r21, _k345x, 1);
                _sum1 = vfmaq_lane_f16(_sum1, _r22, _k345x, 2);

                _sum0 = vfmaq_lane_f16(_sum0, _r20, _k678x, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _r21, _k678x, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _r22, _k678x, 2);
                _sum1 = vfmaq_lane_f16(_sum1, _r30, _k678x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _r31, _k678x, 1);
                _sum1 = vfmaq_lane_f16(_sum1, _r32, _k678x, 2);

                vst1q_f16(outptr0, _sum0);
                vst1q_f16(outptr1, _sum1);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr0 += 8;
                outptr1 += 8;
            }
            for (; j + 3 < outw; j += 4)
            {
                float16x4_t _r00 = vld1_f16(r0);
                float16x4_t _r10 = vld1_f16(r1);
                float16x4_t _r20 = vld1_f16(r2);
                float16x4_t _r30 = vld1_f16(r3);

                float16x4_t _r0n = vld1_f16(r0 + 4);
                float16x4_t _r1n = vld1_f16(r1 + 4);
                float16x4_t _r2n = vld1_f16(r2 + 4);
                float16x4_t _r3n = vld1_f16(r3 + 4);

                float16x4_t _r01 = vext_f16(_r00, _r0n, 1);
                float16x4_t _r11 = vext_f16(_r10, _r1n, 1);
                float16x4_t _r21 = vext_f16(_r20, _r2n, 1);
                float16x4_t _r31 = vext_f16(_r30, _r3n, 1);

                float16x4_t _r02 = vext_f16(_r00, _r0n, 2);
                float16x4_t _r12 = vext_f16(_r10, _r1n, 2);
                float16x4_t _r22 = vext_f16(_r20, _r2n, 2);
                float16x4_t _r32 = vext_f16(_r30, _r3n, 2);

                float16x4_t _sum0 = vget_low_f16(_bias0);
                float16x4_t _sum1 = vget_low_f16(_bias0);

                _sum0 = vfma_lane_f16(_sum0, _r00, _k012x, 0);
                _sum0 = vfma_lane_f16(_sum0, _r01, _k012x, 1);
                _sum0 = vfma_lane_f16(_sum0, _r02, _k012x, 2);
                _sum1 = vfma_lane_f16(_sum1, _r10, _k012x, 0);
                _sum1 = vfma_lane_f16(_sum1, _r11, _k012x, 1);
                _sum1 = vfma_lane_f16(_sum1, _r12, _k012x, 2);

                _sum0 = vfma_lane_f16(_sum0, _r10, _k345x, 0);
                _sum0 = vfma_lane_f16(_sum0, _r11, _k345x, 1);
                _sum0 = vfma_lane_f16(_sum0, _r12, _k345x, 2);
                _sum1 = vfma_lane_f16(_sum1, _r20, _k345x, 0);
                _sum1 = vfma_lane_f16(_sum1, _r21, _k345x, 1);
                _sum1 = vfma_lane_f16(_sum1, _r22, _k345x, 2);

                _sum0 = vfma_lane_f16(_sum0, _r20, _k678x, 0);
                _sum0 = vfma_lane_f16(_sum0, _r21, _k678x, 1);
                _sum0 = vfma_lane_f16(_sum0, _r22, _k678x, 2);
                _sum1 = vfma_lane_f16(_sum1, _r30, _k678x, 0);
                _sum1 = vfma_lane_f16(_sum1, _r31, _k678x, 1);
                _sum1 = vfma_lane_f16(_sum1, _r32, _k678x, 2);

                vst1_f16(outptr0, _sum0);
                vst1_f16(outptr1, _sum1);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                outptr0 += 4;
                outptr1 += 4;
            }
            for (; j < outw; j++)
            {
                float16x4_t _r0 = vld1_f16(r0);
                float16x4_t _r1 = vld1_f16(r1);
                float16x4_t _r2 = vld1_f16(r2);
                float16x4_t _r3 = vld1_f16(r3);

                float16x4_t _sum0 = vmul_f16(_r0, _k012x);
                _sum0 = vfma_f16(_sum0, _r1, _k345x);
                _sum0 = vfma_f16(_sum0, _r2, _k678x);

                float16x4_t _sum1 = vmul_f16(_r1, _k012x);
                _sum1 = vfma_f16(_sum1, _r2, _k345x);
                _sum1 = vfma_f16(_sum1, _r3, _k678x);

                _sum0 = vset_lane_f16(bias0, _sum0, 3);
                _sum1 = vset_lane_f16(bias0, _sum1, 3);

                *outptr0 = (__fp16)vaddvq_f32(vcvt_f32_f16(_sum0));
                *outptr1 = (__fp16)vaddvq_f32(vcvt_f32_f16(_sum1));

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr1 += outw;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 7 < outw; j += 8)
            {
                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r20 = vld1q_f16(r2);

                float16x8_t _r0n = vld1q_f16(r0 + 8);
                float16x8_t _r1n = vld1q_f16(r1 + 8);
                float16x8_t _r2n = vld1q_f16(r2 + 8);

                float16x8_t _r01 = vextq_f16(_r00, _r0n, 1);
                float16x8_t _r11 = vextq_f16(_r10, _r1n, 1);
                float16x8_t _r21 = vextq_f16(_r20, _r2n, 1);

                float16x8_t _r02 = vextq_f16(_r00, _r0n, 2);
                float16x8_t _r12 = vextq_f16(_r10, _r1n, 2);
                float16x8_t _r22 = vextq_f16(_r20, _r2n, 2);

                float16x8_t _sum0 = _bias0;

                _sum0 = vfmaq_lane_f16(_sum0, _r00, _k012x, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _r01, _k012x, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _r02, _k012x, 2);

                _sum0 = vfmaq_lane_f16(_sum0, _r10, _k345x, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _r11, _k345x, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _r12, _k345x, 2);

                _sum0 = vfmaq_lane_f16(_sum0, _r20, _k678x, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _r21, _k678x, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _r22, _k678x, 2);

                vst1q_f16(outptr0, _sum0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }
            for (; j + 3 < outw; j += 4)
            {
                float16x4_t _r00 = vld1_f16(r0);
                float16x4_t _r10 = vld1_f16(r1);
                float16x4_t _r20 = vld1_f16(r2);

                float16x4_t _r0n = vld1_f16(r0 + 4);
                float16x4_t _r1n = vld1_f16(r1 + 4);
                float16x4_t _r2n = vld1_f16(r2 + 4);

                float16x4_t _r01 = vext_f16(_r00, _r0n, 1);
                float16x4_t _r11 = vext_f16(_r10, _r1n, 1);
                float16x4_t _r21 = vext_f16(_r20, _r2n, 1);

                float16x4_t _r02 = vext_f16(_r00, _r0n, 2);
                float16x4_t _r12 = vext_f16(_r10, _r1n, 2);
                float16x4_t _r22 = vext_f16(_r20, _r2n, 2);

                float16x4_t _sum0 = vget_low_f16(_bias0);

                _sum0 = vfma_lane_f16(_sum0, _r00, _k012x, 0);
                _sum0 = vfma_lane_f16(_sum0, _r01, _k012x, 1);
                _sum0 = vfma_lane_f16(_sum0, _r02, _k012x, 2);

                _sum0 = vfma_lane_f16(_sum0, _r10, _k345x, 0);
                _sum0 = vfma_lane_f16(_sum0, _r11, _k345x, 1);
                _sum0 = vfma_lane_f16(_sum0, _r12, _k345x, 2);

                _sum0 = vfma_lane_f16(_sum0, _r20, _k678x, 0);
                _sum0 = vfma_lane_f16(_sum0, _r21, _k678x, 1);
                _sum0 = vfma_lane_f16(_sum0, _r22, _k678x, 2);

                vst1_f16(outptr0, _sum0);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                outptr0 += 4;
            }
            for (; j < outw; j++)
            {
                float16x4_t _r0 = vld1_f16(r0);
                float16x4_t _r1 = vld1_f16(r1);
                float16x4_t _r2 = vld1_f16(r2);

                float16x4_t _sum = vmul_f16(_r0, _k012x);
                _sum = vfma_f16(_sum, _r1, _k345x);
                _sum = vfma_f16(_sum, _r2, _k678x);

                _sum = vset_lane_f16(bias0, _sum, 3);

                *outptr0 = (__fp16)vaddvq_f32(vcvt_f32_f16(_sum));

                r0++;
                r1++;
                r2++;
                outptr0++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static void convdw3x3s2_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2 * outw + w;

    const __fp16* kernel = _kernel;
    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const __fp16 bias0 = bias ? bias[g] : 0.f;

        const __fp16* kernel0 = kernel + g * 9;

        __fp16* outptr = out;

        const __fp16* img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0;
        const __fp16* r1 = img0 + w;
        const __fp16* r2 = img0 + w * 2;

        float16x4_t _k012x = vld1_f16(kernel0);
        float16x4_t _k345x = vld1_f16(kernel0 + 3);
        float16x4_t _k678x = vld1_f16(kernel0 + 6);

        _k012x = vset_lane_f16(0.f, _k012x, 3);
        _k345x = vset_lane_f16(0.f, _k345x, 3);
        _k678x = vset_lane_f16(0.f, _k678x, 3);

        float16x8_t _bias0 = vdupq_n_f16(bias0);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 7 < outw; j += 8)
            {
                float16x8x2_t _r00 = vld2q_f16(r0);
                float16x8x2_t _r10 = vld2q_f16(r1);
                float16x8x2_t _r20 = vld2q_f16(r2);

                float16x8x2_t _r0n = vld2q_f16(r0 + 16);
                float16x8x2_t _r1n = vld2q_f16(r1 + 16);
                float16x8x2_t _r2n = vld2q_f16(r2 + 16);

                float16x8_t _r02 = vextq_f16(_r00.val[0], _r0n.val[0], 1);
                float16x8_t _r12 = vextq_f16(_r10.val[0], _r1n.val[0], 1);
                float16x8_t _r22 = vextq_f16(_r20.val[0], _r2n.val[0], 1);

                float16x8_t _sum = _bias0;

                _sum = vfmaq_lane_f16(_sum, _r00.val[0], _k012x, 0);
                _sum = vfmaq_lane_f16(_sum, _r00.val[1], _k012x, 1);
                _sum = vfmaq_lane_f16(_sum, _r02, _k012x, 2);

                _sum = vfmaq_lane_f16(_sum, _r10.val[0], _k345x, 0);
                _sum = vfmaq_lane_f16(_sum, _r10.val[1], _k345x, 1);
                _sum = vfmaq_lane_f16(_sum, _r12, _k345x, 2);

                _sum = vfmaq_lane_f16(_sum, _r20.val[0], _k678x, 0);
                _sum = vfmaq_lane_f16(_sum, _r20.val[1], _k678x, 1);
                _sum = vfmaq_lane_f16(_sum, _r22, _k678x, 2);

                vst1q_f16(outptr, _sum);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }
            for (; j + 3 < outw; j += 4)
            {
                float16x4x2_t _r00 = vld2_f16(r0);
                float16x4x2_t _r10 = vld2_f16(r1);
                float16x4x2_t _r20 = vld2_f16(r2);

                float16x4x2_t _r0n = vld2_f16(r0 + 8);
                float16x4x2_t _r1n = vld2_f16(r1 + 8);
                float16x4x2_t _r2n = vld2_f16(r2 + 8);

                float16x4_t _r02 = vext_f16(_r00.val[0], _r0n.val[0], 1);
                float16x4_t _r12 = vext_f16(_r10.val[0], _r1n.val[0], 1);
                float16x4_t _r22 = vext_f16(_r20.val[0], _r2n.val[0], 1);

                float16x4_t _sum = vget_low_f16(_bias0);

                _sum = vfma_lane_f16(_sum, _r00.val[0], _k012x, 0);
                _sum = vfma_lane_f16(_sum, _r00.val[1], _k012x, 1);
                _sum = vfma_lane_f16(_sum, _r02, _k012x, 2);

                _sum = vfma_lane_f16(_sum, _r10.val[0], _k345x, 0);
                _sum = vfma_lane_f16(_sum, _r10.val[1], _k345x, 1);
                _sum = vfma_lane_f16(_sum, _r12, _k345x, 2);

                _sum = vfma_lane_f16(_sum, _r20.val[0], _k678x, 0);
                _sum = vfma_lane_f16(_sum, _r20.val[1], _k678x, 1);
                _sum = vfma_lane_f16(_sum, _r22, _k678x, 2);

                vst1_f16(outptr, _sum);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }
            for (; j < outw; j++)
            {
                float16x4_t _r0 = vld1_f16(r0);
                float16x4_t _r1 = vld1_f16(r1);
                float16x4_t _r2 = vld1_f16(r2);

                float16x4_t _sum = vmul_f16(_r0, _k012x);
                _sum = vfma_f16(_sum, _r1, _k345x);
                _sum = vfma_f16(_sum, _r2, _k678x);

                _sum = vset_lane_f16(bias0, _sum, 3);

                *outptr = (__fp16)vaddvq_f32(vcvt_f32_f16(_sum));

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
