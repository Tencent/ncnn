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

static void conv3x3s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);

        const float* k0 = kernel + p*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr0n = outptr0 + outw;
            float* outptr1n = outptr1 + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

#if __ARM_NEON
            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0+3);
            float32x4_t _k06 = vld1q_f32(k0+6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1+3);
            float32x4_t _k16 = vld1q_f32(k1+6);
#endif // __ARM_NEON

            int i = 0;

            for (; i+1 < outh; i+=2)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);
                    float32x4_t _sum1 = vld1q_f32(outptr1);
                    float32x4_t _sum0n = vld1q_f32(outptr0n);
                    float32x4_t _sum1n = vld1q_f32(outptr1n);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r30n = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r30n, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r30n, 2);

                    _sum0 = vfmaq_laneq_f32(_sum0, _r00, _k00, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r01, _k00, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r02, _k00, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r10, _k03, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r11, _k03, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r12, _k03, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r20, _k06, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r21, _k06, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r22, _k06, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k10, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r01, _k10, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k10, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r10, _k13, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k13, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r12, _k13, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k16, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r21, _k16, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k16, 2);

                    _sum0n = vfmaq_laneq_f32(_sum0n, _r10, _k00, 0);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r11, _k00, 1);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r12, _k00, 2);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r20, _k03, 0);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r21, _k03, 1);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r22, _k03, 2);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r30, _k06, 0);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r31, _k06, 1);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r32, _k06, 2);

                    _sum1n = vfmaq_laneq_f32(_sum1n, _r10, _k10, 0);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r11, _k10, 1);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r12, _k10, 2);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r20, _k13, 0);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r21, _k13, 1);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r22, _k13, 2);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r30, _k16, 0);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r31, _k16, 1);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r32, _k16, 2);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr1, _sum1);
                    vst1q_f32(outptr0n, _sum0n);
                    vst1q_f32(outptr1n, _sum1n);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr0n += 4;
                    outptr1n += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(

                    "pld        [%5, #192]          \n"
                    "vld1.f32   {d16-d18}, [%5 :64] \n"// r0
                    "add        %5, #16             \n"

                    "pld        [%8, #192]          \n"
                    "vld1.f32   {d28-d30}, [%8]     \n"// r3
                    "add        %8, #16             \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q14, q15, #2   \n"

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d12-d13}, [%1 :64] \n"// _sum0

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d14-d15}, [%2 :64] \n"// _sum1

                    "vmla.f32   q6, q8, %e18[0]     \n"
                    "vmla.f32   q7, q8, %e21[0]     \n"

                    "pld        [%3, #128]          \n"
                    "vld1.f32   {d24-d25}, [%3]     \n"// _sum0n

                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d26-d27}, [%4]     \n"// _sum1n

                    "vmla.f32   q12, q14, %e20[0]   \n"
                    "vmla.f32   q13, q14, %e23[0]   \n"

                    "vext.32    q8, q8, q9, #2      \n"
                    "vext.32    q9, q14, q15, #1    \n"

                    "vmla.f32   q6, q10, %e18[1]    \n"
                    "vmla.f32   q7, q10, %e21[1]    \n"
                    "vmla.f32   q12, q11, %f20[0]   \n"
                    "vmla.f32   q13, q11, %f23[0]   \n"

                    "pld        [%6, #192]          \n"
                    "vld1.f32   {d28-d30}, [%6]     \n"// r1
                    "add        %6, #16             \n"

                    "vmla.f32   q6, q8, %f18[0]     \n"
                    "vmla.f32   q7, q8, %f21[0]     \n"
                    "vmla.f32   q12, q9, %e20[1]    \n"
                    "vmla.f32   q13, q9, %e23[1]    \n"

                    "vext.32    q10, q14, q15, #1   \n"

                    "vmla.f32   q6, q14, %e19[0]    \n"
                    "vmla.f32   q7, q14, %e22[0]    \n"
                    "vmla.f32   q12, q14, %e18[0]   \n"
                    "vmla.f32   q13, q14, %e21[0]   \n"

                    "vext.32    q11, q14, q15, #2   \n"

                    "vmla.f32   q6, q10, %e19[1]    \n"
                    "vmla.f32   q7, q10, %e22[1]    \n"
                    "vmla.f32   q12, q10, %e18[1]   \n"
                    "vmla.f32   q13, q10, %e21[1]   \n"

                    "pld        [%7, #192]          \n"
                    "vld1.f32   {d16-d18}, [%7 :64] \n"// r2
                    "add        %7, #16             \n"

                    "vmla.f32   q6, q11, %f19[0]    \n"
                    "vmla.f32   q7, q11, %f22[0]    \n"
                    "vmla.f32   q12, q11, %f18[0]   \n"
                    "vmla.f32   q13, q11, %f21[0]   \n"

                    "vext.32    q10, q8, q9, #1     \n"

                    "vmla.f32   q6, q8, %e20[0]     \n"
                    "vmla.f32   q7, q8, %e23[0]     \n"
                    "vmla.f32   q12, q8, %e19[0]    \n"
                    "vmla.f32   q13, q8, %e22[0]    \n"

                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q6, q10, %e20[1]    \n"
                    "vmla.f32   q7, q10, %e23[1]    \n"
                    "vmla.f32   q12, q10, %e19[1]   \n"
                    "vmla.f32   q13, q10, %e22[1]   \n"

                    "pld        [%5, #192]          \n"
                    "vld1.f32   {d16-d18}, [%5 :64] \n"// r0
                    "add        %5, #16             \n"

                    "vmla.f32   q6, q11, %f20[0]    \n"
                    "vmla.f32   q7, q11, %f23[0]    \n"
                    "vmla.f32   q12, q11, %f19[0]   \n"
                    "vmla.f32   q13, q11, %f22[0]   \n"

                    "pld        [%8, #192]          \n"
                    "vld1.f32   {d28-d30}, [%8]     \n"// r3
                    "add        %8, #16             \n"

                    "vext.32    q10, q8, q9, #1     \n"

                    "vst1.f32   {d12-d13}, [%1 : 64]!\n"
                    "vst1.f32   {d14-d15}, [%2 : 64]!\n"

                    "vext.32    q11, q14, q15, #2   \n"

                    "vst1.f32   {d24-d25}, [%3]!    \n"
                    "vst1.f32   {d26-d27}, [%4]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %5, #16             \n"
                    "sub        %8, #16             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr0n),   // %3
                      "=r"(outptr1n),   // %4
                      "=r"(r0),         // %5
                      "=r"(r1),         // %6
                      "=r"(r2),         // %7
                      "=r"(r3)          // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr0n),
                      "4"(outptr1n),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "w"(_k00),      // %18
                      "w"(_k03),      // %19
                      "w"(_k06),      // %20
                      "w"(_k10),      // %21
                      "w"(_k13),      // %22
                      "w"(_k16)       // %23
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    float32x4_t _sum0n = vmulq_f32(_r10, _k00);
                    float32x4_t _sum1n = vmulq_f32(_r10, _k10);
                    _sum0n = vmlaq_f32(_sum0n, _r20, _k03);
                    _sum1n = vmlaq_f32(_sum1n, _r20, _k13);
                    _sum0n = vmlaq_f32(_sum0n, _r30, _k06);
                    _sum1n = vmlaq_f32(_sum1n, _r30, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
                    _sum0n = vsetq_lane_f32(*outptr0n, _sum0n, 3);
                    _sum1n = vsetq_lane_f32(*outptr1n, _sum1n, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
                    *outptr0n = vaddvq_f32(_sum0n);
                    *outptr1n = vaddvq_f32(_sum1n);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss0n = vadd_f32(vget_low_f32(_sum0n), vget_high_f32(_sum0n));
                    float32x2_t _ss1n = vadd_f32(vget_low_f32(_sum1n), vget_high_f32(_sum1n));

                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);
                    float32x2_t _ss01n = vpadd_f32(_ss0n, _ss1n);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
                    *outptr0n = vget_lane_f32(_ss01n, 0);
                    *outptr1n = vget_lane_f32(_ss01n, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum0n = 0.f;
                    float sum1 = 0.f;
                    float sum1n = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    sum0n += r1[0] * k0[0];
                    sum0n += r1[1] * k0[1];
                    sum0n += r1[2] * k0[2];
                    sum0n += r2[0] * k0[3];
                    sum0n += r2[1] * k0[4];
                    sum0n += r2[2] * k0[5];
                    sum0n += r3[0] * k0[6];
                    sum0n += r3[1] * k0[7];
                    sum0n += r3[2] * k0[8];

                    sum1n += r1[0] * k1[0];
                    sum1n += r1[1] * k1[1];
                    sum1n += r1[2] * k1[2];
                    sum1n += r2[0] * k1[3];
                    sum1n += r2[1] * k1[4];
                    sum1n += r2[2] * k1[5];
                    sum1n += r3[0] * k1[6];
                    sum1n += r3[1] * k1[7];
                    sum1n += r3[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr0n += sum0n;
                    *outptr1n += sum1n;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr0++;
                    outptr1++;
                    outptr0n++;
                    outptr1n++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr1 += outw;
                outptr0n += outw;
                outptr1n += outw;
            }

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);
                    float32x4_t _sum1 = vld1q_f32(outptr1);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    _sum0 = vfmaq_laneq_f32(_sum0, _r00, _k00, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r01, _k00, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r02, _k00, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r10, _k03, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r11, _k03, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r12, _k03, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r20, _k06, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r21, _k06, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r22, _k06, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k10, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r01, _k10, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k10, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r10, _k13, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k13, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r12, _k13, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k16, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r21, _k16, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k16, 2);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr1, _sum1);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d16-d18}, [%3]     \n"// r0
                    "add        %3, #16             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d12-d13}, [%1]     \n"// _sum0

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d14-d15}, [%2]     \n"// _sum1

                    "vmul.f32   q14, q8, %e12[0]    \n"
                    "vmul.f32   q15, q8, %e15[0]    \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q6, q10, %e12[1]    \n"
                    "vmla.f32   q7, q10, %e15[1]    \n"

                    "pld        [%4, #192]          \n"
                    "vld1.f32   {d16-d18}, [%4]     \n"// r1
                    "add        %4, #16             \n"

                    "vmla.f32   q14, q11, %f12[0]   \n"
                    "vmla.f32   q15, q11, %f15[0]   \n"

                    "vmla.f32   q6, q8, %e13[0]     \n"
                    "vmla.f32   q7, q8, %e16[0]     \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q14, q10, %e13[1]   \n"
                    "vmla.f32   q15, q10, %e16[1]   \n"

                    "pld        [%5, #192]          \n"
                    "vld1.f32   {d16-d18}, [%5]     \n"// r2
                    "add        %5, #16             \n"

                    "vmla.f32   q6, q11, %f13[0]    \n"
                    "vmla.f32   q7, q11, %f16[0]    \n"

                    "vmla.f32   q14, q8, %e14[0]    \n"
                    "vmla.f32   q15, q8, %e17[0]    \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q6, q10, %e14[1]    \n"
                    "vmla.f32   q7, q10, %e17[1]    \n"

                    "vmla.f32   q14, q11, %f14[0]   \n"
                    "vmla.f32   q15, q11, %f17[0]   \n"

                    "vadd.f32   q6, q6, q14         \n"
                    "vadd.f32   q7, q7, q15         \n"

                    "vst1.f32   {d12-d13}, [%1]!    \n"

                    "vst1.f32   {d14-d15}, [%2]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2)          // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "w"(_k00),      // %12
                      "w"(_k03),      // %13
                      "w"(_k06),      // %14
                      "w"(_k10),      // %15
                      "w"(_k13),      // %16
                      "w"(_k16)       // %17
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                    outptr1++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* kernel0 = kernel + p*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k3456 = vld1q_f32(kernel0+3);
            float32x4_t _k6789 = vld1q_f32(kernel0+6);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

            int i = 0;

            for (; i+1 < outh; i+=2)
            {

#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum1 = vld1q_f32(outptr);
                    float32x4_t _sum3 = vld1q_f32(outptr2);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r30n = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r30n, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r30n, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
                    float32x4_t _sum2 = vmulq_laneq_f32(_r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);

                    _sum3 = vfmaq_laneq_f32(_sum3, _r10, _k0123, 0);
                    float32x4_t _sum4 = vmulq_laneq_f32(_r11, _k0123, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r12, _k0123, 2);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r20, _k3456, 0);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r21, _k3456, 1);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r22, _k3456, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r30, _k6789, 0);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r31, _k6789, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r32, _k6789, 2);

                    _sum1 = vaddq_f32(_sum1, _sum2);
                    _sum3 = vaddq_f32(_sum3, _sum4);

                    vst1q_f32(outptr, _sum1);
                    vst1q_f32(outptr2, _sum3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 4;
                    outptr2 += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d18-d20}, [%3 :64] \n"// r0
                    "add        %3, #16             \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d14-d15}, [%1 :64] \n"// _sum

                    "vmla.f32   q7, q9, %e14[0]     \n"
                    "vmul.f32   q6, q11, %e14[1]    \n"
                    "vmul.f32   q13, q12, %f14[0]   \n"

                    "pld        [%4, #192]          \n"
                    "vld1.f32   {d18-d20}, [%4]     \n"// r1
                    "add        %4, #16             \n"

                    "vmla.f32   q7, q9, %e15[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q6, q11, %e15[1]    \n"
                    "vmla.f32   q13, q12, %f15[0]   \n"

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d16-d17}, [%2]     \n"// _sum2

                    "vmla.f32   q8, q9, %e14[0]     \n"
                    "vmul.f32   q14, q11, %e14[1]   \n"
                    "vmul.f32   q15, q12, %f14[0]   \n"

                    "pld        [%5, #192]          \n"
                    "vld1.f32   {d18-d20}, [%5 :64] \n"// r2
                    "add        %5, #16             \n"

                    "vmla.f32   q7, q9, %e16[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q6, q11, %e16[1]    \n"
                    "vmla.f32   q13, q12, %f16[0]   \n"

                    "vmla.f32   q8, q9, %e15[0]     \n"
                    "vmla.f32   q14, q11, %e15[1]   \n"
                    "vmla.f32   q15, q12, %f15[0]   \n"

                    "pld        [%6, #192]          \n"
                    "vld1.f32   {d18-d20}, [%6]     \n"// r3
                    "add        %6, #16             \n"

                    "vmla.f32   q8, q9, %e16[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q14, q11, %e16[1]   \n"
                    "vmla.f32   q15, q12, %f16[0]   \n"

                    "vadd.f32   q7, q7, q6          \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d18-d20}, [%3 :64] \n"// r0

                    "vadd.f32   q8, q8, q14         \n"
                    "vadd.f32   q7, q7, q13         \n"
                    "vadd.f32   q8, q8, q15         \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "add        %3, #16             \n"

                    "vst1.f32   {d14-d15}, [%1]!    \n"
                    "vst1.f32   {d16-d17}, [%2]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %3, #16             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(outptr2),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2),         // %5
                      "=r"(r3)          // %6
                    : "0"(nn),
                      "1"(outptr),
                      "2"(outptr2),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "6"(r3),
                      "w"(_k0123),      // %14
                      "w"(_k3456),      // %15
                      "w"(_k6789)       // %16
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    float32x4_t _sum2 = vmulq_f32(_r10, _k0123);
                    _sum2 = vmlaq_f32(_sum2, _r20, _k3456);
                    _sum2 = vmlaq_f32(_sum2, _r30, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);
                    _sum2 = vsetq_lane_f32(*outptr2, _sum2, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
                    *outptr2 = vaddvq_f32(_sum2);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));

                    float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

                    *outptr = vget_lane_f32(_sss2, 0);
                    *outptr2 = vget_lane_f32(_sss2, 1);
#endif // __aarch64__
#else
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {

#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum1 = vld1q_f32(outptr);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
                    float32x4_t _sum2 = vmulq_laneq_f32(_r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);

                    _sum1 = vaddq_f32(_sum1, _sum2);

                    vst1q_f32(outptr, _sum1);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #192]          \n"
                    "vld1.f32   {d16-d18}, [%2]     \n"// r0
                    "add        %2, #16             \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d14-d15}, [%1]     \n"// _sum

                    "vmla.f32   q7, q8, %e10[0]     \n"
                    "vmul.f32   q13, q10, %e10[1]   \n"
                    "vmul.f32   q14, q11, %f10[0]   \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d16-d18}, [%3]     \n"// r1
                    "add        %3, #16             \n"

                    "vmla.f32   q7, q8, %e11[0]     \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q13, q10, %e11[1]   \n"
                    "vmla.f32   q14, q11, %f11[0]   \n"

                    "pld        [%4, #192]          \n"
                    "vld1.f32   {d16-d18}, [%4]     \n"// r2
                    "add        %4, #16             \n"

                    "vmla.f32   q7, q8, %e12[0]     \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q13, q10, %e12[1]   \n"
                    "vmla.f32   q14, q11, %f12[0]   \n"

                    "pld        [%2, #192]          \n"
                    "vld1.f32   {d16-d18}, [%2]     \n"// r0
                    "add        %2, #16             \n"

                    "vadd.f32   q7, q7, q13         \n"
                    "vadd.f32   q7, q7, q14         \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vst1.f32   {d14-d15}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %2, #16             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2)          // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),      // %10
                      "w"(_k3456),      // %11
                      "w"(_k6789)       // %12
                    : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif
                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }

}

static void conv3x3s1_winograd64_transform_kernel_neon(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*8, inch, outch);

    const float ktm[8][3] = {
        {   1.0f,     0.0f,     0.0f},
        {-2.0f/9,  -2.0f/9,  -2.0f/9},
        {-2.0f/9,   2.0f/9,  -2.0f/9},
        {1.0f/90,  1.0f/45,  2.0f/45},
        {1.0f/90, -1.0f/45,  2.0f/45},
        {1.0f/45,  1.0f/90, 1.0f/180},
        {1.0f/45, -1.0f/90, 1.0f/180},
        {   0.0f,     0.0f,     1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p<outch; p++)
    {
        for (int q = 0; q<inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p*inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i=0; i<8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j=0; j<8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i=0; i<8; i++)
                {
                    kernel_tm0[j*8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // optimized layout for winograd4
    // interleave weights
    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    Mat kernel_tm2(8*8 * inch * 4, 1, nn_outch + (outch % 4 + 3) / 4);

    #pragma omp parallel for
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        float* ktm2 = kernel_tm2.channel(pp);

        const Mat kernel0_tm = kernel_tm.channel(p);
        const Mat kernel1_tm = kernel_tm.channel(p+1);
        const Mat kernel2_tm = kernel_tm.channel(p+2);
        const Mat kernel3_tm = kernel_tm.channel(p+3);

        int q=0;

#if __ARM_NEON && __aarch64__
        for (; q+3<inch; q+=4)
        {
            const float* k00 = kernel0_tm.row(q);
            const float* k01 = kernel0_tm.row(q+1);
            const float* k02 = kernel0_tm.row(q+2);
            const float* k03 = kernel0_tm.row(q+3);
            const float* k10 = kernel1_tm.row(q);
            const float* k11 = kernel1_tm.row(q+1);
            const float* k12 = kernel1_tm.row(q+2);
            const float* k13 = kernel1_tm.row(q+3);
            const float* k20 = kernel2_tm.row(q);
            const float* k21 = kernel2_tm.row(q+1);
            const float* k22 = kernel2_tm.row(q+2);
            const float* k23 = kernel2_tm.row(q+3);
            const float* k30 = kernel3_tm.row(q);
            const float* k31 = kernel3_tm.row(q+1);
            const float* k32 = kernel3_tm.row(q+2);
            const float* k33 = kernel3_tm.row(q+3);

            for (int r=0; r<16; r++)
            {
            // split into two asm blocks for gcc reject over 30 oprands :(
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "ld1    {v2.4s}, [%3], #16  \n"
                "ld1    {v3.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                "ld1    {v0.4s}, [%5], #16  \n"
                "ld1    {v1.4s}, [%6], #16  \n"
                "ld1    {v2.4s}, [%7], #16  \n"
                "ld1    {v3.4s}, [%8], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k01),    // %2
                  "=r"(k02),    // %3
                  "=r"(k03),    // %4
                  "=r"(k10),    // %5
                  "=r"(k11),    // %6
                  "=r"(k12),    // %7
                  "=r"(k13)     // %8
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k01),
                  "3"(k02),
                  "4"(k03),
                  "5"(k10),
                  "6"(k11),
                  "7"(k12),
                  "8"(k13)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "ld1    {v2.4s}, [%3], #16  \n"
                "ld1    {v3.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                "ld1    {v0.4s}, [%5], #16  \n"
                "ld1    {v1.4s}, [%6], #16  \n"
                "ld1    {v2.4s}, [%7], #16  \n"
                "ld1    {v3.4s}, [%8], #16  \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k20),    // %1
                  "=r"(k21),    // %2
                  "=r"(k22),    // %3
                  "=r"(k23),    // %4
                  "=r"(k30),    // %5
                  "=r"(k31),    // %6
                  "=r"(k32),    // %7
                  "=r"(k33)     // %8
                : "0"(ktm2),
                  "1"(k20),
                  "2"(k21),
                  "3"(k22),
                  "4"(k23),
                  "5"(k30),
                  "6"(k31),
                  "7"(k32),
                  "8"(k33)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }
        }
#endif // __ARM_NEON && __aarch64__

        for (; q+1<inch; q+=2)
        {
            const float* k00 = kernel0_tm.row(q);
            const float* k01 = kernel0_tm.row(q+1);
            const float* k10 = kernel1_tm.row(q);
            const float* k11 = kernel1_tm.row(q+1);
            const float* k20 = kernel2_tm.row(q);
            const float* k21 = kernel2_tm.row(q+1);
            const float* k30 = kernel3_tm.row(q);
            const float* k31 = kernel3_tm.row(q+1);

            for (int r=0; r<16; r++)
            {
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%3], #16  \n"
                "ld1    {v1.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%5], #16  \n"
                "ld1    {v1.4s}, [%6], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%7], #16  \n"
                "ld1    {v1.4s}, [%8], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k01),    // %2
                  "=r"(k10),    // %3
                  "=r"(k11),    // %4
                  "=r"(k20),    // %5
                  "=r"(k21),    // %6
                  "=r"(k30),    // %7
                  "=r"(k31)     // %8
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k01),
                  "3"(k10),
                  "4"(k11),
                  "5"(k20),
                  "6"(k21),
                  "7"(k30),
                  "8"(k31)
                : "cc", "memory", "v0", "v1"
            );
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%3 :128]! \n"
                "vld1.f32   {d2-d3}, [%4 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%5 :128]! \n"
                "vld1.f32   {d2-d3}, [%6 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%7 :128]! \n"
                "vld1.f32   {d2-d3}, [%8 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k01),    // %2
                  "=r"(k10),    // %3
                  "=r"(k11),    // %4
                  "=r"(k20),    // %5
                  "=r"(k21),    // %6
                  "=r"(k30),    // %7
                  "=r"(k31)     // %8
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k01),
                  "3"(k10),
                  "4"(k11),
                  "5"(k20),
                  "6"(k21),
                  "7"(k30),
                  "8"(k31)
                : "cc", "memory", "q0", "q1"
            );
#endif // __aarch64__
#else
                for (int m=0; m<4; m++)
                {
                    ktm2[0 +m] = k00[m];
                    ktm2[4 +m] = k01[m];
                    ktm2[8 +m] = k10[m];
                    ktm2[12+m] = k11[m];
                    ktm2[16+m] = k20[m];
                    ktm2[20+m] = k21[m];
                    ktm2[24+m] = k30[m];
                    ktm2[28+m] = k31[m];
                }

                k00 += 4;
                k01 += 4;
                k10 += 4;
                k11 += 4;
                k20 += 4;
                k21 += 4;
                k30 += 4;
                k31 += 4;
                ktm2 += 32;
#endif // __ARM_NEON
            }
        }

        for (; q<inch; q++)
        {
            const float* k00 = kernel0_tm.row(q);
            const float* k10 = kernel1_tm.row(q);
            const float* k20 = kernel2_tm.row(q);
            const float* k30 = kernel3_tm.row(q);

            for (int r=0; r<16; r++)
            {
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "ld1    {v1.4s}, [%2], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                "ld1    {v0.4s}, [%3], #16  \n"
                "ld1    {v1.4s}, [%4], #16  \n"
                "st1    {v0.4s, v1.4s}, [%0], #32  \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k10),    // %2
                  "=r"(k20),    // %3
                  "=r"(k30)     // %4
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k10),
                  "3"(k20),
                  "4"(k30)
                : "cc", "memory", "v0", "v1"
            );
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                "vld1.f32   {d0-d1}, [%3 :128]! \n"
                "vld1.f32   {d2-d3}, [%4 :128]! \n"
                "vst1.f32   {d0-d3}, [%0 :128]! \n"

                : "=r"(ktm2),   // %0
                  "=r"(k00),    // %1
                  "=r"(k10),    // %2
                  "=r"(k20),    // %3
                  "=r"(k30)     // %4
                : "0"(ktm2),
                  "1"(k00),
                  "2"(k10),
                  "3"(k20),
                  "4"(k30)
                : "cc", "memory", "q0", "q1"
            );
#endif // __aarch64__
#else
                for (int m=0; m<4; m++)
                {
                    ktm2[0 +m] = k00[m];
                    ktm2[4 +m] = k10[m];
                    ktm2[8 +m] = k20[m];
                    ktm2[12+m] = k30[m];
                }

                k00 += 4;
                k10 += 4;
                k20 += 4;
                k30 += 4;
                ktm2 += 16;
#endif // __ARM_NEON
            }
        }
    }

    #pragma omp parallel for
    for (int p = remain_outch_start; p<outch; p++)
    {
        float* ktm2 = (float*)kernel_tm2.channel(nn_outch) + 8*8 * inch * (p-remain_outch_start);

        const Mat kernel0_tm = kernel_tm.channel(p);

        int q = 0;

        for (; q<inch; q++)
        {
            const float* k00 = kernel0_tm.row(q);

            for (int r=0; r<16; r++)
            {
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%1], #16  \n"
                "st1    {v0.4s}, [%0], #16  \n"
                : "=r"(ktm2),   // %0
                  "=r"(k00)     // %1
                : "0"(ktm2),
                  "1"(k00)
                : "cc", "memory", "v0"
            );
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vst1.f32   {d0-d1}, [%0 :128]! \n"
                : "=r"(ktm2),   // %0
                  "=r"(k00)     // %1
                : "0"(ktm2),
                  "1"(k00)
                : "cc", "memory", "q0"
            );
#endif // __aarch64__
#else
                for (int m=0; m<4; m++)
                {
                    ktm2[m] = k00[m];
                }

                k00 += 4;
                ktm2 += 4;
#endif // __ARM_NEON
            }
        }
    }

    kernel_tm = kernel_tm2;
}

static void conv3x3s1_winograd64_transform_kernel_neon5(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*8, inch, outch);

    const float ktm[8][3] = {
        {   1.0f,     0.0f,     0.0f},
        {-2.0f/9,  -2.0f/9,  -2.0f/9},
        {-2.0f/9,   2.0f/9,  -2.0f/9},
        {1.0f/90,  1.0f/45,  2.0f/45},
        {1.0f/90, -1.0f/45,  2.0f/45},
        {1.0f/45,  1.0f/90, 1.0f/180},
        {1.0f/45, -1.0f/90, 1.0f/180},
        {   0.0f,     0.0f,     1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p<outch; p++)
    {
        for (int q = 0; q<inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p*inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i=0; i<8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j=0; j<8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i=0; i<8; i++)
                {
                    kernel_tm0[j*8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }


    // optimized layout for winograd5
    // interleave weights
//     Mat kernel_tm2(8*8, inch, outch);
//     Mat kernel_tm2(inch, 64, outch);
#if __ARM_NEON && __aarch64__
    Mat kernel_tm2(8*4*(inch/4) + 8*(inch%4), 64, outch/8 + (outch%8)/4 + outch%4);
#else
    Mat kernel_tm2(4*4*(inch/4) + 4*(inch%4), 64, outch/4 + outch%4);
#endif

    int p=0;
#if __aarch64__
    for (; p+7<outch; p+=8)
    {
        const Mat kernel0_tm = kernel_tm.channel(p);
        const Mat kernel1_tm = kernel_tm.channel(p+1);
        const Mat kernel2_tm = kernel_tm.channel(p+2);
        const Mat kernel3_tm = kernel_tm.channel(p+3);
        const Mat kernel4_tm = kernel_tm.channel(p+4);
        const Mat kernel5_tm = kernel_tm.channel(p+5);
        const Mat kernel6_tm = kernel_tm.channel(p+6);
        const Mat kernel7_tm = kernel_tm.channel(p+7);

        Mat ktm2 = kernel_tm2.channel(p/8);

        for (int r=0; r<64; r++)
        {
            float* ktm2p = ktm2.row(r);

            for (int q=0; q<inch; q++)
            {
                const float* ktm0_0 = kernel0_tm.row(q);
                const float* ktm1_0 = kernel1_tm.row(q);
                const float* ktm2_0 = kernel2_tm.row(q);
                const float* ktm3_0 = kernel3_tm.row(q);
                const float* ktm4_0 = kernel4_tm.row(q);
                const float* ktm5_0 = kernel5_tm.row(q);
                const float* ktm6_0 = kernel6_tm.row(q);
                const float* ktm7_0 = kernel7_tm.row(q);

                ktm2p[0] = ktm0_0[r];
                ktm2p[1] = ktm1_0[r];
                ktm2p[2] = ktm2_0[r];
                ktm2p[3] = ktm3_0[r];
                ktm2p[4] = ktm4_0[r];
                ktm2p[5] = ktm5_0[r];
                ktm2p[6] = ktm6_0[r];
                ktm2p[7] = ktm7_0[r];

                ktm2p += 8;
            }
        }
    }
#endif // __aarch64__
    for (; p+3<outch; p+=4)
    {
        const Mat kernel0_tm = kernel_tm.channel(p);
        const Mat kernel1_tm = kernel_tm.channel(p+1);
        const Mat kernel2_tm = kernel_tm.channel(p+2);
        const Mat kernel3_tm = kernel_tm.channel(p+3);

#if __ARM_NEON && __aarch64__
        Mat ktm2 = kernel_tm2.channel(p/8+(p%8)/4);
#else
        Mat ktm2 = kernel_tm2.channel(p/4);
#endif

        for (int r=0; r<64; r++)
        {
            float* ktm2p = ktm2.row(r);

            for (int q=0; q<inch; q++)
            {
                const float* ktm0_0 = kernel0_tm.row(q);
                const float* ktm1_0 = kernel1_tm.row(q);
                const float* ktm2_0 = kernel2_tm.row(q);
                const float* ktm3_0 = kernel3_tm.row(q);

                ktm2p[0] = ktm0_0[r];
                ktm2p[1] = ktm1_0[r];
                ktm2p[2] = ktm2_0[r];
                ktm2p[3] = ktm3_0[r];

                ktm2p += 4;
            }
        }
    }
    for (; p<outch; p++)
    {
        const Mat kernel0_tm = kernel_tm.channel(p);

#if __ARM_NEON && __aarch64__
        Mat ktm2 = kernel_tm2.channel(p/8+(p%8)/4+p%4);
#else
        Mat ktm2 = kernel_tm2.channel(p/4+p%4);
#endif

        for (int r=0; r<64; r++)
        {
            float* ktm2p = ktm2.row(r);

            for (int q=0; q<inch; q++)
            {
                const float* ktm0_0 = kernel0_tm.row(q);

                ktm2p[0] = ktm0_0[r];

                ktm2p += 1;
            }
        }
    }

    kernel_tm = kernel_tm2;
}

#if 0//TODO remove old code sometime later
static void conv3x3s1_winograd64_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        bottom_blob_tm.create(8*8, w_tm/8 * h_tm/8, inch);

//         const float itm[8][8] = {
//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
//
//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
//
//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
//
//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
//
//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
//         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
                    const float* r0 = img0.row(i * 6) + j * 6;
                    float* r0_tm = img0_tm.row(i * w_tm/8 + j);

                    // TODO neon optimize
                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25 + tmp0[5]);

                        r0_tm[1] = tmp12a + tmp12b;
                        r0_tm[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        r0_tm[3] = tmp34a + tmp34b;
                        r0_tm[4] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        r0_tm[5] = tmp56a + tmp56b;
                        r0_tm[6] = tmp56a - tmp56b;

                        r0_tm += 8;
                    }
                }
            }
        }

    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(8*8, w_tm/8 * h_tm/8, outch);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);
            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p+1);
            const Mat kernel2_tm = kernel_tm.channel(p+2);
            const Mat kernel3_tm = kernel_tm.channel(p+3);

            out0_tm.fill(0.f);
            out1_tm.fill(0.f);
            out2_tm.fill(0.f);
            out3_tm.fill(0.f);

            int q = 0;
            for (; q+3<inch; q+=4)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);
                const float* r2 = bottom_blob_tm.channel(q+2);
                const float* r3 = bottom_blob_tm.channel(q+3);

                const float* k00 = kernel0_tm.row(q);
                const float* k10 = kernel1_tm.row(q);
                const float* k20 = kernel2_tm.row(q);
                const float* k30 = kernel3_tm.row(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    for (int m=0; m+7<64; m+=8)
                    {
                        float32x4_t _output0_tm = vld1q_f32(output0_tm);
                        float32x4_t _output1_tm = vld1q_f32(output1_tm);
                        float32x4_t _output2_tm = vld1q_f32(output2_tm);
                        float32x4_t _output3_tm = vld1q_f32(output3_tm);

                        float32x4_t _r0 = vld1q_f32(r0);
                        float32x4_t _r1 = vld1q_f32(r1);
                        float32x4_t _r2 = vld1q_f32(r2);
                        float32x4_t _r3 = vld1q_f32(r3);

                        float32x4_t _k00 = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k01 = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k02 = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k03 = vld1q_f32(k00);
                        k00 += 64;

                        k00 -= 64*4;

                        _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                        _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                        _output0_tm = vmlaq_f32(_output0_tm, _r2, _k02);
                        _output0_tm = vmlaq_f32(_output0_tm, _r3, _k03);

                        float32x4_t _k10 = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k11 = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k12 = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k13 = vld1q_f32(k10);
                        k10 += 64;

                        k10 -= 64*4;

                        _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                        _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                        _output1_tm = vmlaq_f32(_output1_tm, _r2, _k12);
                        _output1_tm = vmlaq_f32(_output1_tm, _r3, _k13);

                        float32x4_t _k20 = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k21 = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k22 = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k23 = vld1q_f32(k20);
                        k20 += 64;

                        k20 -= 64*4;

                        _output2_tm = vmlaq_f32(_output2_tm, _r0, _k20);
                        _output2_tm = vmlaq_f32(_output2_tm, _r1, _k21);
                        _output2_tm = vmlaq_f32(_output2_tm, _r2, _k22);
                        _output2_tm = vmlaq_f32(_output2_tm, _r3, _k23);

                        float32x4_t _k30 = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k31 = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k32 = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k33 = vld1q_f32(k30);
                        k30 += 64;

                        k30 -= 64*4;

                        _output3_tm = vmlaq_f32(_output3_tm, _r0, _k30);
                        _output3_tm = vmlaq_f32(_output3_tm, _r1, _k31);
                        _output3_tm = vmlaq_f32(_output3_tm, _r2, _k32);
                        _output3_tm = vmlaq_f32(_output3_tm, _r3, _k33);

                        vst1q_f32(output0_tm, _output0_tm);
                        vst1q_f32(output1_tm, _output1_tm);
                        vst1q_f32(output2_tm, _output2_tm);
                        vst1q_f32(output3_tm, _output3_tm);

                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k00 += 4;
                        k10 += 4;
                        k20 += 4;
                        k30 += 4;

                        float32x4_t _output0_tmn = vld1q_f32(output0_tm);
                        float32x4_t _output1_tmn = vld1q_f32(output1_tm);
                        float32x4_t _output2_tmn = vld1q_f32(output2_tm);
                        float32x4_t _output3_tmn = vld1q_f32(output3_tm);

                        float32x4_t _r0n = vld1q_f32(r0);
                        float32x4_t _r1n = vld1q_f32(r1);
                        float32x4_t _r2n = vld1q_f32(r2);
                        float32x4_t _r3n = vld1q_f32(r3);

                        float32x4_t _k00n = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k01n = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k02n = vld1q_f32(k00);
                        k00 += 64;
                        float32x4_t _k03n = vld1q_f32(k00);
                        k00 += 64;

                        k00 -= 64*4;

                        _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r2n, _k02n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r3n, _k03n);

                        float32x4_t _k10n = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k11n = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k12n = vld1q_f32(k10);
                        k10 += 64;
                        float32x4_t _k13n = vld1q_f32(k10);
                        k10 += 64;

                        k10 -= 64*4;

                        _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                        _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);
                        _output1_tmn = vmlaq_f32(_output1_tmn, _r2n, _k12n);
                        _output1_tmn = vmlaq_f32(_output1_tmn, _r3n, _k13n);

                        float32x4_t _k20n = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k21n = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k22n = vld1q_f32(k20);
                        k20 += 64;
                        float32x4_t _k23n = vld1q_f32(k20);
                        k20 += 64;

                        k20 -= 64*4;

                        _output2_tmn = vmlaq_f32(_output2_tmn, _r0n, _k20n);
                        _output2_tmn = vmlaq_f32(_output2_tmn, _r1n, _k21n);
                        _output2_tmn = vmlaq_f32(_output2_tmn, _r2n, _k22n);
                        _output2_tmn = vmlaq_f32(_output2_tmn, _r3n, _k23n);

                        float32x4_t _k30n = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k31n = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k32n = vld1q_f32(k30);
                        k30 += 64;
                        float32x4_t _k33n = vld1q_f32(k30);
                        k30 += 64;

                        k30 -= 64*4;

                        _output3_tmn = vmlaq_f32(_output3_tmn, _r0n, _k30n);
                        _output3_tmn = vmlaq_f32(_output3_tmn, _r1n, _k31n);
                        _output3_tmn = vmlaq_f32(_output3_tmn, _r2n, _k32n);
                        _output3_tmn = vmlaq_f32(_output3_tmn, _r3n, _k33n);

                        vst1q_f32(output0_tm, _output0_tmn);
                        vst1q_f32(output1_tm, _output1_tmn);
                        vst1q_f32(output2_tm, _output2_tmn);
                        vst1q_f32(output3_tm, _output3_tmn);

                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k00 += 4;
                        k10 += 4;
                        k20 += 4;
                        k30 += 4;
                    }
#else // __aarch64__
                    asm volatile(
                        "mov        r4, #8              \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d16-d19}, [%0 :128]\n"//q8 q9 = _output0_tm

                        "0:                             \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d0-d3}, [%4 :128]! \n"//q0 q1 = _r0

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k00
                        "add        %8, %8, #256        \n"

                        "vmla.f32   q8, q0, q10         \n"
                        "vmla.f32   q9, q1, q11         \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d24-d27}, [%1 :128]\n"//q12 q13 = _output1_tm

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k10
                        "add        %9, %9, #256        \n"

                        "vmla.f32   q12, q0, q14        \n"
                        "vmla.f32   q13, q1, q15        \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"//q2 q3 = _r1

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k01
                        "add        %8, %8, #256        \n"

                        "vmla.f32   q8, q2, q10         \n"
                        "vmla.f32   q9, q3, q11         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k11
                        "add        %9, %9, #256        \n"

                        "vmla.f32   q12, q2, q14        \n"
                        "vmla.f32   q13, q3, q15        \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d8-d11}, [%6 :128]!\n"//q4 q5 = _r2

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k02
                        "add        %8, %8, #256        \n"

                        "vmla.f32   q8, q4, q10         \n"
                        "vmla.f32   q9, q5, q11         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k12
                        "add        %9, %9, #256        \n"

                        "vmla.f32   q12, q4, q14        \n"
                        "vmla.f32   q13, q5, q15        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d12-d15}, [%7 :128]!\n"//q6 q7 = _r3

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]\n"//q10 q11 = _k03
                        "sub        %8, %8, #736        \n"

                        "vmla.f32   q8, q6, q10         \n"
                        "vmla.f32   q9, q7, q11         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d28-d31}, [%9 :128]\n"//q14 q15 = _k13
                        "sub        %9, %9, #736        \n"

                        "vmla.f32   q12, q6, q14        \n"
                        "vmla.f32   q13, q7, q15        \n"

                        "vst1.f32   {d16-d19}, [%0 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]\n"//q8 q9 = _output2_tm

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k20
                        "add        %10, %10, #256      \n"

                        "vmla.f32   q8, q0, q10         \n"
                        "vmla.f32   q9, q1, q11         \n"

                        "vst1.f32   {d24-d27}, [%1 :128]!\n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d24-d27}, [%3 :128]\n"//q12 q13 = _output3_tm

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k30
                        "add        %11, %11, #256      \n"

                        "vmla.f32   q12, q0, q14        \n"
                        "vmla.f32   q13, q1, q15        \n"

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k21
                        "add        %10, %10, #256      \n"

                        "vmla.f32   q8, q2, q10         \n"
                        "vmla.f32   q9, q3, q11         \n"

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k31
                        "add        %11, %11, #256      \n"

                        "vmla.f32   q12, q2, q14        \n"
                        "vmla.f32   q13, q3, q15        \n"

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k22
                        "add        %10, %10, #256      \n"

                        "vmla.f32   q8, q4, q10         \n"
                        "vmla.f32   q9, q5, q11         \n"

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k32
                        "add        %11, %11, #256      \n"

                        "vmla.f32   q12, q4, q14        \n"
                        "vmla.f32   q13, q5, q15        \n"

                        "pld        [%10, #256]         \n"
                        "vld1.f32   {d20-d23}, [%10 :128]\n"//q10 q11 = _k23
                        "sub        %10, %10, #736      \n"

                        "vmla.f32   q8, q6, q10         \n"
                        "vmla.f32   q9, q7, q11         \n"

                        "pld        [%11, #256]         \n"
                        "vld1.f32   {d28-d31}, [%11 :128]\n"//q14 q15 = _k33
                        "sub        %11, %11, #736      \n"

                        "vmla.f32   q12, q6, q14        \n"
                        "vmla.f32   q13, q7, q15        \n"

                        "vst1.f32   {d16-d19}, [%2 :128]!\n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d16-d19}, [%0 :128]\n"//q8 q9 = _output0_tm

                        "subs       r4, r4, #1          \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!\n"

                        "bne        0b                  \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(r3),         // %7
                          "=r"(k00),        // %8
                          "=r"(k10),        // %9
                          "=r"(k20),        // %10
                          "=r"(k30)         // %11
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(r3),
                          "8"(k00),
                          "9"(k10),
                          "10"(k20),
                          "11"(k30)
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__

                    k00 -= 64;
                    k10 -= 64;
                    k20 -= 64;
                    k30 -= 64;
#else
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k00[m];
                        k00 += 64;
                        output0_tm[m] += r1[m] * k00[m];
                        k00 += 64;
                        output0_tm[m] += r2[m] * k00[m];
                        k00 += 64;
                        output0_tm[m] += r3[m] * k00[m];
                        k00 += 64;

                        k00 -= 64 * 4;

                        output1_tm[m] += r0[m] * k10[m];
                        k10 += 64;
                        output1_tm[m] += r1[m] * k10[m];
                        k10 += 64;
                        output1_tm[m] += r2[m] * k10[m];
                        k10 += 64;
                        output1_tm[m] += r3[m] * k10[m];
                        k10 += 64;

                        k10 -= 64 * 4;

                        output2_tm[m] += r0[m] * k20[m];
                        k20 += 64;
                        output2_tm[m] += r1[m] * k20[m];
                        k20 += 64;
                        output2_tm[m] += r2[m] * k20[m];
                        k20 += 64;
                        output2_tm[m] += r3[m] * k20[m];
                        k20 += 64;

                        k20 -= 64 * 4;

                        output3_tm[m] += r0[m] * k30[m];
                        k30 += 64;
                        output3_tm[m] += r1[m] * k30[m];
                        k30 += 64;
                        output3_tm[m] += r2[m] * k30[m];
                        k30 += 64;
                        output3_tm[m] += r3[m] * k30[m];
                        k30 += 64;

                        k30 -= 64 * 4;
                    }

                    r0 += 64;
                    r1 += 64;
                    r2 += 64;
                    r3 += 64;
                    output0_tm += 64;
                    output1_tm += 64;
                    output2_tm += 64;
                    output3_tm += 64;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k0 = kernel0_tm.row(q);
                const float* k1 = kernel1_tm.row(q);
                const float* k2 = kernel2_tm.row(q);
                const float* k3 = kernel3_tm.row(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
                    // TODO neon optimize
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                        output1_tm[m] += r0[m] * k1[m];
                        output2_tm[m] += r0[m] * k2[m];
                        output3_tm[m] += r0[m] * k3[m];
                    }

                    r0 += 64;
                    output0_tm += 64;
                    output1_tm += 64;
                    output2_tm += 64;
                    output3_tm += 64;
                }

            }
        }

        #pragma omp parallel for
        for (int p=remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            out0_tm.fill(0.f);

            int q = 0;
            for (; q+3<inch; q+=4)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);
                const float* r2 = bottom_blob_tm.channel(q+2);
                const float* r3 = bottom_blob_tm.channel(q+3);

                const float* k0 = kernel0_tm.row(q);
                const float* k1 = kernel0_tm.row(q+1);
                const float* k2 = kernel0_tm.row(q+2);
                const float* k3 = kernel0_tm.row(q+3);

                float* output0_tm = out0_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    for (int m=0; m+7<64; m+=8)
                    {
                        float32x4_t _output0_tm = vld1q_f32(output0_tm);

                        float32x4_t _r0 = vld1q_f32(r0);
                        float32x4_t _r1 = vld1q_f32(r1);
                        float32x4_t _r2 = vld1q_f32(r2);
                        float32x4_t _r3 = vld1q_f32(r3);

                        float32x4_t _k0 = vld1q_f32(k0);
                        float32x4_t _k1 = vld1q_f32(k1);
                        float32x4_t _k2 = vld1q_f32(k2);
                        float32x4_t _k3 = vld1q_f32(k3);

                        _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                        _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                        _output0_tm = vmlaq_f32(_output0_tm, _r2, _k2);
                        _output0_tm = vmlaq_f32(_output0_tm, _r3, _k3);

                        vst1q_f32(output0_tm, _output0_tm);

                        output0_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k0 += 4;
                        k1 += 4;
                        k2 += 4;
                        k3 += 4;

                        float32x4_t _output0_tmn = vld1q_f32(output0_tm);

                        float32x4_t _r0n = vld1q_f32(r0);
                        float32x4_t _r1n = vld1q_f32(r1);
                        float32x4_t _r2n = vld1q_f32(r2);
                        float32x4_t _r3n = vld1q_f32(r3);

                        float32x4_t _k0n = vld1q_f32(k0);
                        float32x4_t _k1n = vld1q_f32(k1);
                        float32x4_t _k2n = vld1q_f32(k2);
                        float32x4_t _k3n = vld1q_f32(k3);

                        _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r2n, _k2n);
                        _output0_tmn = vmlaq_f32(_output0_tmn, _r3n, _k3n);

                        vst1q_f32(output0_tm, _output0_tmn);

                        output0_tm += 4;

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;

                        k0 += 4;
                        k1 += 4;
                        k2 += 4;
                        k3 += 4;
                    }
#else
                    asm volatile(
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "mov        r4, %0              \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128]!\n"//q12 q13 = output0_tm
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q12, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q13, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q12, q8, q10        \n"

                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128]!\n"//q14 q15 = output0_tm
                        "vmla.f32   q13, q9, q11        \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "vst1.f32   {d24-d27}, [r4 :128]!\n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d16-d19}, [%2 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7 :128]! \n"

                        "vmla.f32   q14, q0, q2         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d16-d19}, [%4 :128]!\n"
                        "vmla.f32   q15, q1, q3         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d20-d23}, [%8 :128]!\n"

                        "vmla.f32   q14, q8, q10        \n"
                        "vmla.f32   q15, q9, q11        \n"

                        "vst1.f32   {d28-d31}, [r4 :128]!\n"

                        : "=r"(output0_tm), // %0
                          "=r"(r0),         // %1
                          "=r"(r1),         // %2
                          "=r"(r2),         // %3
                          "=r"(r3),         // %4
                          "=r"(k0),         // %5
                          "=r"(k1),         // %6
                          "=r"(k2),         // %7
                          "=r"(k3)          // %8
                        : "0"(output0_tm),
                          "1"(r0),
                          "2"(r1),
                          "3"(r2),
                          "4"(r3),
                          "5"(k0),
                          "6"(k1),
                          "7"(k2),
                          "8"(k3)
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__

                    k0 -= 64;
                    k1 -= 64;
                    k2 -= 64;
                    k3 -= 64;
#else
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                        output0_tm[m] += r1[m] * k1[m];
                        output0_tm[m] += r2[m] * k2[m];
                        output0_tm[m] += r3[m] * k3[m];
                    }

                    r0 += 64;
                    r1 += 64;
                    r2 += 64;
                    r3 += 64;
                    output0_tm += 64;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k0 = kernel0_tm.row(q);

                float* output0_tm = out0_tm;

                // tile
                for (int i=0; i<h_tm/8 * w_tm/8; i++)
                {
                    // TODO neon optimize
                    for (int m=0; m<64; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                    }

                    r0 += 64;
                    output0_tm += 64;
                }

            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch);
    {
//         const float otm[6][8] = {
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;

        #pragma omp parallel for
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
                    const float* output0_tm = out0_tm.row(i * w_tm/8 + j);
                    float* output0 = out0.row(i * 6) + j * 6;

                    // TODO neon optimize
                    for (int m=0; m<8; m++)
                    {
                        float tmp024a = output0_tm[1] + output0_tm[2];
                        float tmp135a = output0_tm[1] - output0_tm[2];

                        float tmp024b = output0_tm[3] + output0_tm[4];
                        float tmp135b = output0_tm[3] - output0_tm[4];

                        float tmp024c = output0_tm[5] + output0_tm[6];
                        float tmp135c = output0_tm[5] - output0_tm[6];

                        tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm += 8;
                    }

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w);
}

static void conv3x3s1_winograd64_neon2(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        bottom_blob_tm.create(2*8, 4 * w_tm/8 * h_tm/8, inch);
        const int tiles = w_tm/8 * h_tm/8;

//         const float itm[8][8] = {
//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
//
//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
//
//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
//
//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
//
//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
//         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
                    const float* r0 = img0.row(i * 6) + j * 6;
                    float* r0_tm01 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm23 = img0_tm.row(tiles + i * w_tm/8 + j);
                    float* r0_tm45 = img0_tm.row(tiles * 2 + i * w_tm/8 + j);
                    float* r0_tm67 = img0_tm.row(tiles * 3 + i * w_tm/8 + j);

                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    float* r0_tms[4] = { r0_tm01, r0_tm23, r0_tm45, r0_tm67 };

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float* r0_tm = r0_tms[m/2] + (m%2) * 8;

                        r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25 + tmp0[5]);

                        r0_tm[1] = tmp12a + tmp12b;
                        r0_tm[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        r0_tm[3] = tmp34a + tmp34b;
                        r0_tm[4] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        r0_tm[5] = tmp56a + tmp56b;
                        r0_tm[6] = tmp56a - tmp56b;
                    }
                }
            }
        }

    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(2*8, 4 * w_tm/8 * h_tm/8, outch);

        const int tiles = h_tm/8 * w_tm/8;

        #pragma omp parallel for
        for (int p = 0; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            out0_tm.fill(0.f);

            int q = 0;
            for (; q+1<inch; q+=2)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);

                const float* k0 = kernel0_tm.row(q);
                const float* k1 = kernel0_tm.row(q+1);

                float* output0_tm = out0_tm;

                for (int r=0; r<4; r++)
                {
#if __ARM_NEON
#if __aarch64__
                float32x4_t _k0 = vld1q_f32(k0);
                float32x4_t _k0n = vld1q_f32(k0+4);
                float32x4_t _k0nn = vld1q_f32(k0+8);
                float32x4_t _k0nnn = vld1q_f32(k0+12);
                float32x4_t _k1 = vld1q_f32(k1);
                float32x4_t _k1n = vld1q_f32(k1+4);
                float32x4_t _k1nn = vld1q_f32(k1+8);
                float32x4_t _k1nnn = vld1q_f32(k1+12);
#else
                float32x4_t _k0;
                float32x4_t _k0n;
                float32x4_t _k0nn;
                float32x4_t _k0nnn;
                float32x4_t _k1;
                float32x4_t _k1n;
                float32x4_t _k1nn;
                float32x4_t _k1nnn;

                asm volatile(
                    "pld        [%0, #512]              \n"
                    "vld1.f32   {%e2-%f2}, [%0 :128]!   \n"
                    "pld        [%1, #512]              \n"
                    "vld1.f32   {%e4-%f4}, [%1 :128]!   \n"

                    "vld1.f32   {%e3-%f3}, [%0 :128]!   \n"
                    "vld1.f32   {%e5-%f5}, [%1 :128]!   \n"

                    "vld1.f32   {%e6-%f6}, [%0 :128]!   \n"
                    "vld1.f32   {%e8-%f8}, [%1 :128]!   \n"

                    "vld1.f32   {%e7-%f7}, [%0 :128]!   \n"
                    "vld1.f32   {%e9-%f9}, [%1 :128]!   \n"
                    : "=r"(k0),     // %0
                      "=r"(k1),     // %1
                      "=w"(_k0),    // %2
                      "=w"(_k0n),   // %3
                      "=w"(_k1),    // %4
                      "=w"(_k1n),   // %5
                      "=w"(_k0nn),  // %6
                      "=w"(_k0nnn), // %7
                      "=w"(_k1nn),  // %8
                      "=w"(_k1nnn)  // %9
                    : "0"(k0),
                      "1"(k1)
                    : "cc", "memory"
                );
#endif // __aarch64__
#endif // __ARM_NEON

                // tile
#if __ARM_NEON
                int nn = tiles >> 2;
                int remain = tiles & 3;
#else
                int remain = tiles;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0nnn);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1nnn);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0nnn);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1nnn);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0nnn);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1nnn);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0nnn);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1nnn);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "mov        r4, %1                  \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "0:                                 \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d20-d23}, [%1 :128]!   \n"// q10 q11 = _output0_tm

                        "vmla.f32   q10, q12, %q12          \n"
                        "vmla.f32   q11, q13, %q13          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q14          \n"
                        "vmla.f32   q11, q15, %q15          \n"

                        "vst1.f32   {d16-d19}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d20-d23}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d20-d23}, [%1 :128]!   \n"// q10 q11 = _output0_tm

                        "vmla.f32   q10, q12, %q12          \n"
                        "vmla.f32   q11, q13, %q13          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q14          \n"
                        "vmla.f32   q11, q15, %q15          \n"

                        "vst1.f32   {d16-d19}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d20-d23}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d20-d23}, [%1 :128]!   \n"// q10 q11 = _output0_tm

                        "vmla.f32   q10, q12, %q12          \n"
                        "vmla.f32   q11, q13, %q13          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q14          \n"
                        "vmla.f32   q11, q15, %q15          \n"

                        "vst1.f32   {d16-d19}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d20-d23}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d20-d23}, [%1 :128]!   \n"// q10 q11 = _output0_tm

                        "vmla.f32   q10, q12, %q12          \n"
                        "vmla.f32   q11, q13, %q13          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q14          \n"
                        "vmla.f32   q11, q15, %q15          \n"

                        "vst1.f32   {d16-d19}, [r4 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "subs       %0, #1                  \n"

                        "vst1.f32   {d20-d23}, [r4 :128]!   \n"

                        "bne        0b                      \n"

                        "sub        %1, #32                 \n"
                        "sub        %2, #64                 \n"
                        : "=r"(nn),         // %0
                          "=r"(output0_tm), // %1
                          "=r"(r0),         // %2
                          "=r"(r1)          // %3
                        : "0"(nn),
                          "1"(output0_tm),
                          "2"(r0),
                          "3"(r1),
                          "w"(_k0),         // %8
                          "w"(_k0n),        // %9
                          "w"(_k1),         // %10
                          "w"(_k1n),        // %11
                          "w"(_k0nn),       // %12
                          "w"(_k0nnn),      // %13
                          "w"(_k1nn),       // %14
                          "w"(_k1nnn)       // %15
                        : "cc", "memory", "r4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0nnn);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k1nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k1nnn);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;
#else
                    asm volatile(
                        "mov        r4, %0                  \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d24-d27}, [%1 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d16-d19}, [%0 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q6            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d28-d31}, [%2 :128]!   \n"// q14 q15 = _r1
                        "vmla.f32   q9, q13, %q7            \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d24-d27}, [%1 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q8, q14, %q8            \n"

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d20-d23}, [%0 :128]    \n"// q10 q11 = _output0_tm
                        "vmla.f32   q9, q15, %q9            \n"

                        "vmla.f32   q10, q12, %q10          \n"
                        "vmla.f32   q11, q13, %q11          \n"

                        "vst1.f32   {d16-d19}, [r4 :128]    \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d28-d31}, [%2 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q10, q14, %q12          \n"
                        "vmla.f32   q11, q15, %q13          \n"

                        "vst1.f32   {d20-d23}, [%0 :128]!   \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0),         // %1
                          "=r"(r1)          // %2
                        : "0"(output0_tm),
                          "1"(r0),
                          "2"(r1),
                          "w"(_k0),         // %6
                          "w"(_k0n),        // %7
                          "w"(_k1),         // %8
                          "w"(_k1n),        // %9
                          "w"(_k0nn),       // %10
                          "w"(_k0nnn),      // %11
                          "w"(_k1nn),       // %12
                          "w"(_k1nnn)       // %13
                        : "cc", "memory", "r4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<16; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                        output0_tm[m] += r1[m] * k1[m];
                    }

                    r0 += 16;
                    r1 += 16;
                    output0_tm += 16;
#endif // __ARM_NEON
                }

#if __ARM_NEON
#if __aarch64__
                k0 += 16;
                k1 += 16;
#endif // __aarch64__
#else
                k0 += 16;
                k1 += 16;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k0 = kernel0_tm.row(q);

                float* output0_tm = out0_tm;

                for (int r=0; r<4; r++)
                {
#if __ARM_NEON
#if __aarch64__
                float32x4_t _k0 = vld1q_f32(k0);
                float32x4_t _k0n = vld1q_f32(k0+4);
                float32x4_t _k0nn = vld1q_f32(k0+8);
                float32x4_t _k0nnn = vld1q_f32(k0+12);
#else
                float32x4_t _k0;
                float32x4_t _k0n;
                float32x4_t _k0nn;
                float32x4_t _k0nnn;

                asm volatile(
                    "pld        [%0, #512]              \n"
                    "vld1.f32   {%e1-%f1}, [%0 :128]!   \n"
                    "vld1.f32   {%e2-%f2}, [%0 :128]!   \n"
                    "vld1.f32   {%e3-%f3}, [%0 :128]!   \n"
                    "vld1.f32   {%e4-%f4}, [%0 :128]!   \n"
                    : "=r"(k0),     // %0
                      "=w"(_k0),    // %1
                      "=w"(_k0n),   // %2
                      "=w"(_k0nn),  // %3
                      "=w"(_k0nnn)  // %4
                    : "0"(k0)
                    : "cc", "memory"
                );
#endif // __aarch64__
#endif // __ARM_NEON

                // tile
                for (int i=0; i<tiles; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k0nn);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k0nnn);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;
#else
                    asm volatile(
                        "mov        r4, %0                  \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d24-d27}, [%1 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d16-d19}, [%0 :128]!   \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q4            \n"
                        "vmla.f32   q9, q13, %q5            \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d24-d27}, [%1 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d20-d23}, [%0 :128]    \n"// q10 q11 = _output0_tm

                        "vmla.f32   q10, q12, %q6           \n"

                        "vst1.f32   {d16-d19}, [r4 :128]    \n"

                        "vmla.f32   q11, q13, %q7           \n"

                        "vst1.f32   {d20-d23}, [%0 :128]!   \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0)          // %1
                        : "0"(output0_tm),
                          "1"(r0),
                          "w"(_k0),         // %4
                          "w"(_k0n),        // %5
                          "w"(_k0nn),       // %6
                          "w"(_k0nnn)       // %7
                        : "cc", "memory", "r4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<16; m++)
                    {
                        output0_tm[m] += r0[m] * k0[m];
                    }

                    r0 += 16;
                    output0_tm += 16;
#endif // __ARM_NEON
                }

#if __ARM_NEON
#if __aarch64__
                k0 += 16;
#endif // __aarch64__
#else
                k0 += 16;
#endif // __ARM_NEON
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch);
    {
//         const float otm[6][8] = {
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        #pragma omp parallel for
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
                    const float* output0_tm01 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm23 = out0_tm.row(tiles + i * w_tm/8 + j);
                    const float* output0_tm45 = out0_tm.row(tiles * 2 + i * w_tm/8 + j);
                    const float* output0_tm67 = out0_tm.row(tiles * 3 + i * w_tm/8 + j);
                    float* output0 = out0.row(i * 6) + j * 6;

                    const float* output0_tms[4] = { output0_tm01, output0_tm23, output0_tm45, output0_tm67 };

                    for (int m=0; m<8; m++)
                    {
                        const float* output0_tm = output0_tms[m/2] + (m%2) * 8;

                        float tmp024a = output0_tm[1] + output0_tm[2];
                        float tmp135a = output0_tm[1] - output0_tm[2];

                        float tmp024b = output0_tm[3] + output0_tm[4];
                        float tmp135b = output0_tm[3] - output0_tm[4];

                        float tmp024c = output0_tm[5] + output0_tm[6];
                        float tmp135c = output0_tm[5] - output0_tm[6];

                        tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;
                    }

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w);
}

static void conv3x3s1_winograd64_neon3(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        bottom_blob_tm.create(8, 8 * w_tm/8 * h_tm/8, inch);
        const int tiles = w_tm/8 * h_tm/8;

//         const float itm[8][8] = {
//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
//
//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
//
//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
//
//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
//
//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
//         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
                    const float* r0 = img0.row(i * 6) + j * 6;
                    float* r0_tm0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm1 = img0_tm.row(i * w_tm/8 + j + tiles);
                    float* r0_tm2 = img0_tm.row(i * w_tm/8 + j + tiles * 2);
                    float* r0_tm3 = img0_tm.row(i * w_tm/8 + j + tiles * 3);
                    float* r0_tm4 = img0_tm.row(i * w_tm/8 + j + tiles * 4);
                    float* r0_tm5 = img0_tm.row(i * w_tm/8 + j + tiles * 5);
                    float* r0_tm6 = img0_tm.row(i * w_tm/8 + j + tiles * 6);
                    float* r0_tm7 = img0_tm.row(i * w_tm/8 + j + tiles * 7);

                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    float* r0_tms[8] = { r0_tm0, r0_tm1, r0_tm2, r0_tm3, r0_tm4, r0_tm5, r0_tm6, r0_tm7 };

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float* r0_tm = r0_tms[m];

                        r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25 + tmp0[5]);

                        r0_tm[1] = tmp12a + tmp12b;
                        r0_tm[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        r0_tm[3] = tmp34a + tmp34b;
                        r0_tm[4] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        r0_tm[5] = tmp56a + tmp56b;
                        r0_tm[6] = tmp56a - tmp56b;
                    }
                }
            }
        }

    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(8, 8 * w_tm/8 * h_tm/8, outch);

        const int tiles = h_tm/8 * w_tm/8;

        int nn_outch = outch >> 1;
        int remain_outch_start = nn_outch << 1;

        #pragma omp parallel for
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 2;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p+1);

            out0_tm.fill(0.f);
            out1_tm.fill(0.f);

            int q = 0;
            for (; q+1<inch; q+=2)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);

                const float* k00 = kernel0_tm.row(q);
                const float* k01 = kernel0_tm.row(q+1);
                const float* k10 = kernel1_tm.row(q);
                const float* k11 = kernel1_tm.row(q+1);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;

                for (int r=0; r<8; r++)
                {
#if __ARM_NEON
#if __aarch64__
                float32x4_t _k00 = vld1q_f32(k00);
                float32x4_t _k00n = vld1q_f32(k00+4);
                float32x4_t _k01 = vld1q_f32(k01);
                float32x4_t _k01n = vld1q_f32(k01+4);
                float32x4_t _k10 = vld1q_f32(k10);
                float32x4_t _k10n = vld1q_f32(k10+4);
                float32x4_t _k11 = vld1q_f32(k11);
                float32x4_t _k11n = vld1q_f32(k11+4);
#else
                float32x4_t _k00;
                float32x4_t _k00n;
                float32x4_t _k01;
                float32x4_t _k01n;
                float32x4_t _k10;
                float32x4_t _k10n;
                float32x4_t _k11;
                float32x4_t _k11n;

                asm volatile(
                    "pld        [%0, #256]              \n"
                    "vld1.f32   {%e4-%f4}, [%0 :128]!   \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {%e6-%f6}, [%1 :128]!   \n"
                    "pld        [%2, #256]              \n"
                    "vld1.f32   {%e8-%f8}, [%2 :128]!   \n"
                    "pld        [%3, #256]              \n"
                    "vld1.f32   {%e10-%f10}, [%3 :128]! \n"

                    "vld1.f32   {%e5-%f5}, [%0 :128]!   \n"
                    "vld1.f32   {%e7-%f7}, [%1 :128]!   \n"
                    "vld1.f32   {%e9-%f9}, [%2 :128]!   \n"
                    "vld1.f32   {%e11-%f11}, [%3 :128]! \n"
                    : "=r"(k00),    // %0
                      "=r"(k01),    // %1
                      "=r"(k10),    // %2
                      "=r"(k11),    // %3
                      "=w"(_k00),   // %4
                      "=w"(_k00n),  // %5
                      "=w"(_k01),   // %6
                      "=w"(_k01n),  // %7
                      "=w"(_k10),   // %8
                      "=w"(_k10n),  // %9
                      "=w"(_k11),   // %10
                      "=w"(_k11n)   // %11
                    : "0"(k00),
                      "1"(k01),
                      "2"(k10),
                      "3"(k11)
                    : "cc", "memory"
                );
#endif // __aarch64__
#endif // __ARM_NEON

                // tile
#if __ARM_NEON
                int nn = tiles >> 2;
                int remain = tiles & 3;
#else
                int remain = tiles;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _output1_tm = vld1q_f32(output1_tm);
                    float32x4_t _output1_tmn = vld1q_f32(output1_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                    _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _output1_tm = vld1q_f32(output1_tm);
                    _output1_tmn = vld1q_f32(output1_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                    _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _output1_tm = vld1q_f32(output1_tm);
                    _output1_tmn = vld1q_f32(output1_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                    _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _output1_tm = vld1q_f32(output1_tm);
                    _output1_tmn = vld1q_f32(output1_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                    _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q10           \n"
                        "vmla.f32   q9, q13, %q11           \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q12           \n"
                        "vmla.f32   q9, q15, %q13           \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q14          \n"
                        "vmla.f32   q11, q13, %q15          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q16          \n"
                        "vmla.f32   q11, q15, %q17          \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q10           \n"
                        "vmla.f32   q9, q13, %q11           \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q12           \n"
                        "vmla.f32   q9, q15, %q13           \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q14          \n"
                        "vmla.f32   q11, q13, %q15          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q16          \n"
                        "vmla.f32   q11, q15, %q17          \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q10           \n"
                        "vmla.f32   q9, q13, %q11           \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q12           \n"
                        "vmla.f32   q9, q15, %q13           \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q14          \n"
                        "vmla.f32   q11, q13, %q15          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "vmla.f32   q10, q14, %q16          \n"
                        "vmla.f32   q11, q15, %q17          \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q10           \n"
                        "vmla.f32   q9, q13, %q11           \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q12           \n"
                        "vmla.f32   q9, q15, %q13           \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q14          \n"
                        "vmla.f32   q11, q13, %q15          \n"

                        "vmla.f32   q10, q14, %q16          \n"
                        "vmla.f32   q11, q15, %q17          \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"
                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "subs       %0, #1                  \n"

                        "bne        0b                      \n"

                        : "=r"(nn),         // %0
                          "=r"(output0_tm), // %1
                          "=r"(output1_tm), // %2
                          "=r"(r0),         // %3
                          "=r"(r1)          // %4
                        : "0"(nn),
                          "1"(output0_tm),
                          "2"(output1_tm),
                          "3"(r0),
                          "4"(r1),
                          "w"(_k00),         // %10
                          "w"(_k00n),        // %11
                          "w"(_k01),         // %12
                          "w"(_k01n),        // %13
                          "w"(_k10),         // %14
                          "w"(_k10n),        // %15
                          "w"(_k11),         // %16
                          "w"(_k11n)         // %17
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _output1_tm = vld1q_f32(output1_tm);
                    float32x4_t _output1_tmn = vld1q_f32(output1_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);
                    _output1_tm = vmlaq_f32(_output1_tm, _r1, _k11);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r1n, _k11n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;
#else
                    asm volatile(
                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d16-d19}, [%0 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d20-d23}, [%1 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q12          \n"
                        "vmla.f32   q11, q13, %q13          \n"

                        "vmla.f32   q10, q14, %q14          \n"
                        "vmla.f32   q11, q15, %q15          \n"

                        "vst1.f32   {d16-d19}, [%0 :128]!   \n"
                        "vst1.f32   {d20-d23}, [%1 :128]!   \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(r0),         // %2
                          "=r"(r1)          // %3
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(r0),
                          "3"(r1),
                          "w"(_k00),         // %8
                          "w"(_k00n),        // %9
                          "w"(_k01),         // %10
                          "w"(_k01n),        // %11
                          "w"(_k10),         // %12
                          "w"(_k10n),        // %13
                          "w"(_k11),         // %14
                          "w"(_k11n)         // %15
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<8; m++)
                    {
                        output0_tm[m] += r0[m] * k00[m];
                        output0_tm[m] += r1[m] * k01[m];
                        output1_tm[m] += r0[m] * k10[m];
                        output1_tm[m] += r1[m] * k11[m];
                    }

                    r0 += 8;
                    r1 += 8;
                    output0_tm += 8;
                    output1_tm += 8;
#endif // __ARM_NEON
                }

#if __ARM_NEON
#if __aarch64__
                k00 += 8;
                k01 += 8;
                k10 += 8;
                k11 += 8;
#endif // __aarch64__
#else
                k00 += 8;
                k01 += 8;
                k10 += 8;
                k11 += 8;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k00 = kernel0_tm.row(q);
                const float* k10 = kernel1_tm.row(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;

                for (int r=0; r<8; r++)
                {
#if __ARM_NEON
#if __aarch64__
                float32x4_t _k00 = vld1q_f32(k00);
                float32x4_t _k00n = vld1q_f32(k00+4);
                float32x4_t _k10 = vld1q_f32(k10);
                float32x4_t _k10n = vld1q_f32(k10+4);
#else
                float32x4_t _k00;
                float32x4_t _k00n;
                float32x4_t _k10;
                float32x4_t _k10n;

                asm volatile(
                    "pld        [%0, #256]              \n"
                    "vld1.f32   {%e2-%f2}, [%0 :128]!   \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {%e4-%f4}, [%1 :128]!   \n"
                    "vld1.f32   {%e3-%f3}, [%0 :128]!   \n"
                    "vld1.f32   {%e5-%f5}, [%1 :128]!   \n"
                    : "=r"(k00),    // %0
                      "=r"(k10),    // %1
                      "=w"(_k00),   // %2
                      "=w"(_k00n),  // %3
                      "=w"(_k10),   // %4
                      "=w"(_k10n)   // %5
                    : "0"(k00),
                      "1"(k10)
                    : "cc", "memory"
                );
#endif // __aarch64__
#endif // __ARM_NEON

                // tile
#if __ARM_NEON
                int nn = tiles >> 2;
                int remain = tiles & 3;
#else
                int remain = tiles;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _output1_tm = vld1q_f32(output1_tm);
                    float32x4_t _output1_tmn = vld1q_f32(output1_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _output1_tm = vld1q_f32(output1_tm);
                    _output1_tmn = vld1q_f32(output1_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _output1_tm = vld1q_f32(output1_tm);
                    _output1_tmn = vld1q_f32(output1_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _output1_tm = vld1q_f32(output1_tm);
                    _output1_tmn = vld1q_f32(output1_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q10          \n"
                        "vmla.f32   q11, q13, %q11          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q10          \n"
                        "vmla.f32   q11, q13, %q11          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q10          \n"
                        "vmla.f32   q11, q13, %q11          \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q10          \n"
                        "vmla.f32   q11, q13, %q11          \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"
                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "subs       %0, #1                  \n"

                        "bne        0b                      \n"

                        : "=r"(nn),         // %0
                          "=r"(output0_tm), // %1
                          "=r"(output1_tm), // %2
                          "=r"(r0)          // %3
                        : "0"(nn),
                          "1"(output0_tm),
                          "2"(output1_tm),
                          "3"(r0),
                          "w"(_k00),        // %8
                          "w"(_k00n),       // %9
                          "w"(_k10),        // %10
                          "w"(_k10n)        // %11
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _output1_tm = vld1q_f32(output1_tm);
                    float32x4_t _output1_tmn = vld1q_f32(output1_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);

                    _output1_tm = vmlaq_f32(_output1_tm, _r0, _k10);
                    _output1_tmn = vmlaq_f32(_output1_tmn, _r0n, _k10n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    vst1q_f32(output1_tm, _output1_tm);
                    vst1q_f32(output1_tm+4, _output1_tmn);

                    output0_tm += 8;
                    output1_tm += 8;
#else
                    asm volatile(
                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d16-d19}, [%0 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q6            \n"
                        "vmla.f32   q9, q13, %q7            \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d20-d23}, [%1 :128]    \n"// q10 q11 = _output1_tm

                        "vmla.f32   q10, q12, %q8           \n"
                        "vmla.f32   q11, q13, %q9           \n"

                        "vst1.f32   {d16-d19}, [%0 :128]!   \n"
                        "vst1.f32   {d20-d23}, [%1 :128]!   \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(r0)          // %2
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(r0),
                          "w"(_k00),        // %6
                          "w"(_k00n),       // %7
                          "w"(_k10),        // %8
                          "w"(_k10n)        // %9
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<8; m++)
                    {
                        output0_tm[m] += r0[m] * k00[m];
                        output1_tm[m] += r0[m] * k10[m];
                    }

                    r0 += 8;
                    output0_tm += 8;
                    output1_tm += 8;
#endif // __ARM_NEON
                }

#if __ARM_NEON
#if __aarch64__
                k00 += 8;
                k10 += 8;
#endif // __aarch64__
#else
                k00 += 8;
                k10 += 8;
#endif // __ARM_NEON
                }
            }
        }

        #pragma omp parallel for
        for (int p = remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            out0_tm.fill(0.f);

            int q = 0;
            for (; q+1<inch; q+=2)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);

                const float* k00 = kernel0_tm.row(q);
                const float* k01 = kernel0_tm.row(q+1);

                float* output0_tm = out0_tm;

                for (int r=0; r<8; r++)
                {
#if __ARM_NEON
#if __aarch64__
                float32x4_t _k00 = vld1q_f32(k00);
                float32x4_t _k00n = vld1q_f32(k00+4);
                float32x4_t _k01 = vld1q_f32(k01);
                float32x4_t _k01n = vld1q_f32(k01+4);
#else
                float32x4_t _k00;
                float32x4_t _k00n;
                float32x4_t _k01;
                float32x4_t _k01n;

                asm volatile(
                    "pld        [%0, #256]              \n"
                    "vld1.f32   {%e2-%f2}, [%0 :128]!   \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {%e4-%f4}, [%1 :128]!   \n"

                    "vld1.f32   {%e3-%f3}, [%0 :128]!   \n"
                    "vld1.f32   {%e5-%f5}, [%1 :128]!   \n"
                    : "=r"(k00),    // %0
                      "=r"(k01),    // %1
                      "=w"(_k00),   // %2
                      "=w"(_k00n),  // %3
                      "=w"(_k01),   // %4
                      "=w"(_k01n)   // %5
                    : "0"(k00),
                      "1"(k01)
                    : "cc", "memory"
                );
#endif // __aarch64__
#endif // __ARM_NEON

                // tile
#if __ARM_NEON
                int nn = tiles >> 2;
                int remain = tiles & 3;
#else
                int remain = tiles;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;

                    _output0_tm = vld1q_f32(output0_tm);
                    _output0_tmn = vld1q_f32(output0_tm+4);

                    _r0 = vld1q_f32(r0);
                    _r0n = vld1q_f32(r0+4);
                    _r1 = vld1q_f32(r1);
                    _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d24-d27}, [%2 :128]!   \n"// q12 q13 = _r0

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q8            \n"
                        "vmla.f32   q9, q13, %q9            \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d28-d31}, [%3 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q10           \n"
                        "vmla.f32   q9, q15, %q11           \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "subs       %0, #1                  \n"

                        "bne        0b                      \n"

                        : "=r"(nn),         // %0
                          "=r"(output0_tm), // %1
                          "=r"(r0),         // %2
                          "=r"(r1)          // %3
                        : "0"(nn),
                          "1"(output0_tm),
                          "2"(r0),
                          "3"(r1),
                          "w"(_k00),        // %8
                          "w"(_k00n),       // %9
                          "w"(_k01),        // %10
                          "w"(_k01n)        // %11
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r1n = vld1q_f32(r1+4);

                    r0 += 8;
                    r1 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);
                    _output0_tm = vmlaq_f32(_output0_tm, _r1, _k01);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r1n, _k01n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;
#else
                    asm volatile(
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d24-d27}, [%1 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d16-d19}, [%0 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q6            \n"
                        "vmla.f32   q9, q13, %q7            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d28-d31}, [%2 :128]!   \n"// q14 q15 = _r1

                        "vmla.f32   q8, q14, %q8            \n"
                        "vmla.f32   q9, q15, %q9            \n"

                        "vst1.f32   {d16-d19}, [%0 :128]!   \n"

                        : "=r"(output0_tm), // %0
                          "=r"(r0),         // %1
                          "=r"(r1)          // %2
                        : "0"(output0_tm),
                          "1"(r0),
                          "2"(r1),
                          "w"(_k00),        // %6
                          "w"(_k00n),       // %7
                          "w"(_k01),        // %8
                          "w"(_k01n)        // %9
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<8; m++)
                    {
                        output0_tm[m] += r0[m] * k00[m];
                        output0_tm[m] += r1[m] * k01[m];
                    }

                    r0 += 8;
                    r1 += 8;
                    output0_tm += 8;
#endif // __ARM_NEON
                }

#if __ARM_NEON
#if __aarch64__
                k00 += 8;
                k01 += 8;
#endif // __aarch64__
#else
                k00 += 8;
                k01 += 8;
#endif // __ARM_NEON
                }
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                const float* k00 = kernel0_tm.row(q);

                float* output0_tm = out0_tm;

                for (int r=0; r<8; r++)
                {
#if __ARM_NEON
#if __aarch64__
                float32x4_t _k00 = vld1q_f32(k00);
                float32x4_t _k00n = vld1q_f32(k00+4);
#else
                float32x4_t _k00;
                float32x4_t _k00n;

                asm volatile(
                    "pld        [%0, #256]              \n"
                    "vld1.f32   {%e1-%f1}, [%0 :128]!   \n"
                    "vld1.f32   {%e2-%f2}, [%0 :128]!   \n"
                    : "=r"(k00),    // %0
                      "=w"(_k00),   // %1
                      "=w"(_k00n)   // %2
                    : "0"(k00)
                    : "cc", "memory"
                );
#endif // __aarch64__
#endif // __ARM_NEON

                // tile
                for (int i=0; i<tiles; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _output0_tm = vld1q_f32(output0_tm);
                    float32x4_t _output0_tmn = vld1q_f32(output0_tm+4);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);

                    r0 += 8;

                    _output0_tm = vmlaq_f32(_output0_tm, _r0, _k00);
                    _output0_tmn = vmlaq_f32(_output0_tmn, _r0n, _k00n);

                    vst1q_f32(output0_tm, _output0_tm);
                    vst1q_f32(output0_tm+4, _output0_tmn);

                    output0_tm += 8;
#else
                    asm volatile(
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d24-d27}, [%1 :128]!   \n"// q12 q13 = _r0

                        "pld        [%0, #256]              \n"
                        "vld1.f32   {d16-d19}, [%0 :128]    \n"// q8 q9 = _output0_tm

                        "vmla.f32   q8, q12, %q4            \n"
                        "vmla.f32   q9, q13, %q5            \n"

                        "vst1.f32   {d16-d19}, [%0 :128]!   \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0)          // %1
                        : "0"(output0_tm),
                          "1"(r0),
                          "w"(_k00),        // %4
                          "w"(_k00n)        // %5
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<8; m++)
                    {
                        output0_tm[m] += r0[m] * k00[m];
                    }

                    r0 += 8;
                    output0_tm += 8;
#endif // __ARM_NEON
                }

#if __ARM_NEON
#if __aarch64__
                k00 += 8;
#endif // __aarch64__
#else
                k00 += 8;
#endif // __ARM_NEON
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch);
    {
//         const float otm[6][8] = {
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        #pragma omp parallel for
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
                    const float* output0_tm0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm1 = out0_tm.row(i * w_tm/8 + j + tiles);
                    const float* output0_tm2 = out0_tm.row(i * w_tm/8 + j + tiles * 2);
                    const float* output0_tm3 = out0_tm.row(i * w_tm/8 + j + tiles * 3);
                    const float* output0_tm4 = out0_tm.row(i * w_tm/8 + j + tiles * 4);
                    const float* output0_tm5 = out0_tm.row(i * w_tm/8 + j + tiles * 5);
                    const float* output0_tm6 = out0_tm.row(i * w_tm/8 + j + tiles * 6);
                    const float* output0_tm7 = out0_tm.row(i * w_tm/8 + j + tiles * 7);
                    float* output0 = out0.row(i * 6) + j * 6;

                    const float* output0_tms[8] = { output0_tm0, output0_tm1, output0_tm2, output0_tm3, output0_tm4, output0_tm5, output0_tm6, output0_tm7 };

                    for (int m=0; m<8; m++)
                    {
                        const float* output0_tm = output0_tms[m];

                        float tmp024a = output0_tm[1] + output0_tm[2];
                        float tmp135a = output0_tm[1] - output0_tm[2];

                        float tmp024b = output0_tm[3] + output0_tm[4];
                        float tmp135b = output0_tm[3] - output0_tm[4];

                        float tmp024c = output0_tm[5] + output0_tm[6];
                        float tmp135c = output0_tm[5] - output0_tm[6];

                        tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;
                    }

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w);
}
#endif

static void conv3x3s1_winograd64_neon4(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt.workspace_allocator, opt.num_threads);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        bottom_blob_tm.create(4, 16 * w_tm/8 * h_tm/8, inch, 4u, opt.workspace_allocator);
        const int tiles = w_tm/8 * h_tm/8;

//         const float itm[8][8] = {
//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
//
//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
//
//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
//
//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
//
//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
//         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

#if __ARM_NEON
        const float coeff[8] = {
            0.25f, 0.5f, -1.25f,   2.f,
            -2.5f,  4.f,  4.25f, 5.25f
        };
        float32x4_t _coeff0 = vld1q_f32(coeff);
        float32x4_t _coeff1 = vld1q_f32(coeff+4);
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
#if __ARM_NEON
                    const float* r0 = img0.row(i * 6) + j * 6;
                    const float* r1 = r0 + w;
                    const float* r2 = r0 + w*2;
                    const float* r3 = r0 + w*3;

#if __aarch64__
                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _r0_0123 = vld1q_f32(r0);
                        float32x4_t _r0_4567 = vld1q_f32(r0+4);
                        float32x4_t _r1_0123 = vld1q_f32(r1);
                        float32x4_t _r1_4567 = vld1q_f32(r1+4);
                        float32x4_t _r2_0123 = vld1q_f32(r2);
                        float32x4_t _r2_4567 = vld1q_f32(r2+4);
                        float32x4_t _r3_0123 = vld1q_f32(r3);
                        float32x4_t _r3_4567 = vld1q_f32(r3+4);

                        float32x4x2_t _r01_00221133 = vtrnq_f32(_r0_0123, _r1_0123);
                        float32x4x2_t _r01_44665577 = vtrnq_f32(_r0_4567, _r1_4567);
                        float32x4x2_t _r23_00221133 = vtrnq_f32(_r2_0123, _r3_0123);
                        float32x4x2_t _r23_44665577 = vtrnq_f32(_r2_4567, _r3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _r_00 = vcombine_f32(vget_low_f32(_r01_00221133.val[0]), vget_low_f32(_r23_00221133.val[0]));
                        float32x4_t _r_11 = vcombine_f32(vget_low_f32(_r01_00221133.val[1]), vget_low_f32(_r23_00221133.val[1]));
                        float32x4_t _r_22 = vcombine_f32(vget_high_f32(_r01_00221133.val[0]), vget_high_f32(_r23_00221133.val[0]));
                        float32x4_t _r_33 = vcombine_f32(vget_high_f32(_r01_00221133.val[1]), vget_high_f32(_r23_00221133.val[1]));
                        float32x4_t _r_44 = vcombine_f32(vget_low_f32(_r01_44665577.val[0]), vget_low_f32(_r23_44665577.val[0]));
                        float32x4_t _r_55 = vcombine_f32(vget_low_f32(_r01_44665577.val[1]), vget_low_f32(_r23_44665577.val[1]));
                        float32x4_t _r_66 = vcombine_f32(vget_high_f32(_r01_44665577.val[0]), vget_high_f32(_r23_44665577.val[0]));
                        float32x4_t _r_77 = vcombine_f32(vget_high_f32(_r01_44665577.val[1]), vget_high_f32(_r23_44665577.val[1]));

                        float32x4_t _r_0_m_6 = vsubq_f32(_r_00, _r_66);
                        float32x4_t _r_7_m_1 = vsubq_f32(_r_77, _r_11);

                        float32x4_t _r_4_m_2 = vsubq_f32(_r_44, _r_22);
                        float32x4_t _r_3_m_5 = vsubq_f32(_r_33, _r_55);

                        float32x4_t _tmp0 = vmlaq_lane_f32(_r_0_m_6, _r_4_m_2, vget_high_f32(_coeff1), 1);
                        float32x4_t _tmp7 = vmlaq_lane_f32(_r_7_m_1, _r_3_m_5, vget_high_f32(_coeff1), 1);

                        vst1q_f32(&tmp[0][m], _tmp0);
                        vst1q_f32(&tmp[7][m], _tmp7);

                        float32x4_t _r_2_a_6 = vaddq_f32(_r_22, _r_66);
                        float32x4_t _r_1_a_5 = vaddq_f32(_r_11, _r_55);

                        float32x4_t _tmp12a = vmlsq_lane_f32(_r_2_a_6, _r_44, vget_high_f32(_coeff1), 0);
                        float32x4_t _tmp12b = vmlsq_lane_f32(_r_1_a_5, _r_33, vget_high_f32(_coeff1), 0);

                        float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);

                        vst1q_f32(&tmp[1][m], _tmp1);
                        vst1q_f32(&tmp[2][m], _tmp2);

                        float32x4_t _r_4_x_c = vmulq_lane_f32(_r_44, vget_high_f32(_coeff0), 0);
                        float32x4_t _r_3_x_c = vmulq_lane_f32(_r_33, vget_low_f32(_coeff1), 0);

                        float32x4_t _tmp34a = vaddq_f32(_r_66, _r_4_x_c);
                        _tmp34a = vmlaq_lane_f32(_tmp34a, _r_22, vget_low_f32(_coeff0), 0);

                        float32x4_t _tmp34b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_low_f32(_coeff0), 1);
                        _tmp34b = vmlaq_lane_f32(_tmp34b, _r_55, vget_high_f32(_coeff0), 1);

                        float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);

                        vst1q_f32(&tmp[3][m], _tmp3);
                        vst1q_f32(&tmp[4][m], _tmp4);

                        // reuse r04 * 1.25
                        // reuse r03 * 2.5
                        float32x4_t _r_2_a_4c = vaddq_f32(_r_22, _r_4_x_c);
                        float32x4_t _tmp56a = vmlaq_lane_f32(_r_66, _r_2_a_4c, vget_low_f32(_coeff1), 1);
                        float32x4_t _tmp56b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_high_f32(_coeff0), 1);
                        _tmp56b = vmlaq_lane_f32(_tmp56b, _r_55, vget_low_f32(_coeff0), 1);

                        float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);

                        vst1q_f32(&tmp[5][m], _tmp5);
                        vst1q_f32(&tmp[6][m], _tmp6);

                        r0 += w*4;
                        r1 += w*4;
                        r2 += w*4;
                        r3 += w*4;
                    }

                    const float* t0 = tmp[0];
                    const float* t1 = tmp[1];
                    const float* t2 = tmp[2];
                    const float* t3 = tmp[3];

                    float* r0_tm0_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm0_4 = img0_tm.row(i * w_tm/8 + j + tiles);
                    float* r0_tm1_0 = img0_tm.row(i * w_tm/8 + j + tiles*2);
                    float* r0_tm1_4 = img0_tm.row(i * w_tm/8 + j + tiles*3);
                    float* r0_tm2_0 = img0_tm.row(i * w_tm/8 + j + tiles*4);
                    float* r0_tm2_4 = img0_tm.row(i * w_tm/8 + j + tiles*5);
                    float* r0_tm3_0 = img0_tm.row(i * w_tm/8 + j + tiles*6);
                    float* r0_tm3_4 = img0_tm.row(i * w_tm/8 + j + tiles*7);

                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0+4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1+4);
                        float32x4_t _t2_0123 = vld1q_f32(t2);
                        float32x4_t _t2_4567 = vld1q_f32(t2+4);
                        float32x4_t _t3_0123 = vld1q_f32(t3);
                        float32x4_t _t3_4567 = vld1q_f32(t3+4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                        float32x4x2_t _t23_00221133 = vtrnq_f32(_t2_0123, _t3_0123);
                        float32x4x2_t _t23_44665577 = vtrnq_f32(_t2_4567, _t3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _t_00 = vcombine_f32(vget_low_f32(_t01_00221133.val[0]), vget_low_f32(_t23_00221133.val[0]));
                        float32x4_t _t_11 = vcombine_f32(vget_low_f32(_t01_00221133.val[1]), vget_low_f32(_t23_00221133.val[1]));
                        float32x4_t _t_22 = vcombine_f32(vget_high_f32(_t01_00221133.val[0]), vget_high_f32(_t23_00221133.val[0]));
                        float32x4_t _t_33 = vcombine_f32(vget_high_f32(_t01_00221133.val[1]), vget_high_f32(_t23_00221133.val[1]));
                        float32x4_t _t_44 = vcombine_f32(vget_low_f32(_t01_44665577.val[0]), vget_low_f32(_t23_44665577.val[0]));
                        float32x4_t _t_55 = vcombine_f32(vget_low_f32(_t01_44665577.val[1]), vget_low_f32(_t23_44665577.val[1]));
                        float32x4_t _t_66 = vcombine_f32(vget_high_f32(_t01_44665577.val[0]), vget_high_f32(_t23_44665577.val[0]));
                        float32x4_t _t_77 = vcombine_f32(vget_high_f32(_t01_44665577.val[1]), vget_high_f32(_t23_44665577.val[1]));

                        float32x4_t _t_0_m_6 = vsubq_f32(_t_00, _t_66);
                        float32x4_t _t_7_m_1 = vsubq_f32(_t_77, _t_11);

                        float32x4_t _t_4_m_2 = vsubq_f32(_t_44, _t_22);
                        float32x4_t _t_3_m_5 = vsubq_f32(_t_33, _t_55);

                        float32x4_t _r0_tm_0_0 = vmlaq_lane_f32(_t_0_m_6, _t_4_m_2, vget_high_f32(_coeff1), 1);
                        float32x4_t _r0_tm_4_3 = vmlaq_lane_f32(_t_7_m_1, _t_3_m_5, vget_high_f32(_coeff1), 1);

                        r0_tm0_0[0] = vgetq_lane_f32(_r0_tm_0_0, 0);
                        r0_tm1_0[0] = vgetq_lane_f32(_r0_tm_0_0, 1);
                        r0_tm2_0[0] = vgetq_lane_f32(_r0_tm_0_0, 2);
                        r0_tm3_0[0] = vgetq_lane_f32(_r0_tm_0_0, 3);

                        r0_tm0_4[3] = vgetq_lane_f32(_r0_tm_4_3, 0);
                        r0_tm1_4[3] = vgetq_lane_f32(_r0_tm_4_3, 1);
                        r0_tm2_4[3] = vgetq_lane_f32(_r0_tm_4_3, 2);
                        r0_tm3_4[3] = vgetq_lane_f32(_r0_tm_4_3, 3);

                        float32x4_t _t_2_m_6 = vaddq_f32(_t_22, _t_66);
                        float32x4_t _t_1_m_5 = vaddq_f32(_t_11, _t_55);

                        float32x4_t _tmp12a = vmlsq_lane_f32(_t_2_m_6, _t_44, vget_high_f32(_coeff1), 0);
                        float32x4_t _tmp12b = vmlsq_lane_f32(_t_1_m_5, _t_33, vget_high_f32(_coeff1), 0);

                        float32x4_t _r0_tm_0_1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _r0_tm_0_2 = vsubq_f32(_tmp12a, _tmp12b);

                        r0_tm0_0[1] = vgetq_lane_f32(_r0_tm_0_1, 0);
                        r0_tm1_0[1] = vgetq_lane_f32(_r0_tm_0_1, 1);
                        r0_tm2_0[1] = vgetq_lane_f32(_r0_tm_0_1, 2);
                        r0_tm3_0[1] = vgetq_lane_f32(_r0_tm_0_1, 3);

                        r0_tm0_0[2] = vgetq_lane_f32(_r0_tm_0_2, 0);
                        r0_tm1_0[2] = vgetq_lane_f32(_r0_tm_0_2, 1);
                        r0_tm2_0[2] = vgetq_lane_f32(_r0_tm_0_2, 2);
                        r0_tm3_0[2] = vgetq_lane_f32(_r0_tm_0_2, 3);

                        float32x4_t _t_4_x_c = vmulq_lane_f32(_t_44, vget_high_f32(_coeff0), 0);
                        float32x4_t _t_3_x_c = vmulq_lane_f32(_t_33, vget_low_f32(_coeff1), 0);

                        float32x4_t _tmp34a = vaddq_f32(_t_66, _t_4_x_c);
                        _tmp34a = vmlaq_lane_f32(_tmp34a, _t_22, vget_low_f32(_coeff0), 0);

                        float32x4_t _tmp34b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_low_f32(_coeff0), 1);
                        _tmp34b = vmlaq_lane_f32(_tmp34b, _t_55, vget_high_f32(_coeff0), 1);

                        float32x4_t _r0_tm_0_3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _r0_tm_4_0 = vsubq_f32(_tmp34a, _tmp34b);

                        r0_tm0_0[3] = vgetq_lane_f32(_r0_tm_0_3, 0);
                        r0_tm1_0[3] = vgetq_lane_f32(_r0_tm_0_3, 1);
                        r0_tm2_0[3] = vgetq_lane_f32(_r0_tm_0_3, 2);
                        r0_tm3_0[3] = vgetq_lane_f32(_r0_tm_0_3, 3);

                        r0_tm0_4[0] = vgetq_lane_f32(_r0_tm_4_0, 0);
                        r0_tm1_4[0] = vgetq_lane_f32(_r0_tm_4_0, 1);
                        r0_tm2_4[0] = vgetq_lane_f32(_r0_tm_4_0, 2);
                        r0_tm3_4[0] = vgetq_lane_f32(_r0_tm_4_0, 3);

                        float32x4_t _t_2_a_4c = vaddq_f32(_t_22, _t_4_x_c);
                        float32x4_t _tmp56a = vmlaq_lane_f32(_t_66, _t_2_a_4c, vget_low_f32(_coeff1), 1);
                        float32x4_t _tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                        _tmp56b = vmlaq_lane_f32(_tmp56b, _t_55, vget_low_f32(_coeff0), 1);

                        float32x4_t _r0_tm_4_1 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _r0_tm_4_2 = vsubq_f32(_tmp56a, _tmp56b);

                        r0_tm0_4[1] = vgetq_lane_f32(_r0_tm_4_1, 0);
                        r0_tm1_4[1] = vgetq_lane_f32(_r0_tm_4_1, 1);
                        r0_tm2_4[1] = vgetq_lane_f32(_r0_tm_4_1, 2);
                        r0_tm3_4[1] = vgetq_lane_f32(_r0_tm_4_1, 3);

                        r0_tm0_4[2] = vgetq_lane_f32(_r0_tm_4_2, 0);
                        r0_tm1_4[2] = vgetq_lane_f32(_r0_tm_4_2, 1);
                        r0_tm2_4[2] = vgetq_lane_f32(_r0_tm_4_2, 2);
                        r0_tm3_4[2] = vgetq_lane_f32(_r0_tm_4_2, 3);

                        t0 += 8*4;
                        t1 += 8*4;
                        t2 += 8*4;
                        t3 += 8*4;

                        r0_tm0_0 += img0_tm.w*tiles*2*4;
                        r0_tm0_4 += img0_tm.w*tiles*2*4;
                        r0_tm1_0 += img0_tm.w*tiles*2*4;
                        r0_tm1_4 += img0_tm.w*tiles*2*4;
                        r0_tm2_0 += img0_tm.w*tiles*2*4;
                        r0_tm2_4 += img0_tm.w*tiles*2*4;
                        r0_tm3_0 += img0_tm.w*tiles*2*4;
                        r0_tm3_4 += img0_tm.w*tiles*2*4;
                    }
#else // __aarch64__
                    float* t0 = tmp[0];
                    float* t1 = tmp[1];
                    float* t2 = tmp[2];
                    float* t3 = tmp[3];
                    float* t4 = tmp[4];
                    float* t5 = tmp[5];
                    float* t6 = tmp[6];
                    float* t7 = tmp[7];

                    int stepw = w*4*4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%8], %26    \n"
                        "vld1.f32   {d20-d23}, [%9], %26    \n"
                        "vld1.f32   {d24-d27}, [%10], %26   \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11], %26   \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

                        "vmov       q3, q7              \n"// use q7

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

                        "vmla.f32   q4, q6, %e25[1]     \n"
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

                        "vadd.f32   q8, q2, q3          \n"
                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
                        "vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
                        "vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

                        "vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

                        // loop1
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

                        "vmov       q3, q7              \n"// use q7

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

                        "vmla.f32   q4, q6, %e25[1]     \n"
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

                        "vadd.f32   q8, q2, q3          \n"
                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
                        "vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
                        "vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

                        "vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

                        : "=r"(t0),     // %0
                          "=r"(t1),     // %1
                          "=r"(t2),     // %2
                          "=r"(t3),     // %3
                          "=r"(t4),     // %4
                          "=r"(t5),     // %5
                          "=r"(t6),     // %6
                          "=r"(t7),     // %7
                          "=r"(r0),     // %8
                          "=r"(r1),     // %9
                          "=r"(r2),     // %10
                          "=r"(r3)      // %11
                        : "0"(t0),
                          "1"(t1),
                          "2"(t2),
                          "3"(t3),
                          "4"(t4),
                          "5"(t5),
                          "6"(t6),
                          "7"(t7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(r3),
                          "w"(_coeff0), // %24
                          "w"(_coeff1), // %25
                          "r"(stepw)        // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmp[0];
                    t1 = tmp[1];
                    t2 = tmp[2];
                    t3 = tmp[3];

                    float* r0_tm0_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm0_4 = img0_tm.row(i * w_tm/8 + j + tiles);
                    float* r0_tm1_0 = img0_tm.row(i * w_tm/8 + j + tiles*2);
                    float* r0_tm1_4 = img0_tm.row(i * w_tm/8 + j + tiles*3);
                    float* r0_tm2_0 = img0_tm.row(i * w_tm/8 + j + tiles*4);
                    float* r0_tm2_4 = img0_tm.row(i * w_tm/8 + j + tiles*5);
                    float* r0_tm3_0 = img0_tm.row(i * w_tm/8 + j + tiles*6);
                    float* r0_tm3_4 = img0_tm.row(i * w_tm/8 + j + tiles*7);

                    int step = img0_tm.w*tiles*2*4*4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "add        %8, %8, #128        \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "add        %9, %9, #128        \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"
                        "add        %10, %10, #128      \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"
                        "add        %11, %11, #128      \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4[0]}, [%0]!      \n"
                        "vst1.f32   {d4[1]}, [%2]!      \n"

                        "vmov       q3, q7              \n"// use q7

                        "vst1.f32   {d5[0]}, [%4]!      \n"
                        "vst1.f32   {d5[1]}, [%6]!      \n"

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16[0]}, [%0]!     \n"
                        "vst1.f32   {d16[1]}, [%2]!     \n"

                        "vmla.f32   q4, q6, %e25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4]!     \n"
                        "vst1.f32   {d17[1]}, [%6]!     \n"

                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18[0]}, [%0]!     \n"
                        "vst1.f32   {d18[1]}, [%2]!     \n"

                        "vadd.f32   q8, q2, q3          \n"

                        "vst1.f32   {d19[0]}, [%4]!     \n"
                        "vst1.f32   {d19[1]}, [%6]!     \n"

                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16[0]}, [%0], %26 \n"
                        "vst1.f32   {d16[1]}, [%2], %26 \n"

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4], %26 \n"
                        "vst1.f32   {d17[1]}, [%6], %26 \n"

                        "vtrn.32    q9, q2              \n"
                        "vtrn.32    q3, q6              \n"

                        "sub        %0, %0, #12         \n"
                        "sub        %2, %2, #12         \n"
                        "sub        %4, %4, #12         \n"
                        "sub        %6, %6, #12         \n"

                        "vswp       d19, d6             \n"
                        "vswp       d5, d12             \n"

                        "vst1.f32   {d18-d19}, [%1], %26 \n"
                        "vst1.f32   {d4-d5}, [%3], %26  \n"
                        "vst1.f32   {d6-d7}, [%5], %26  \n"
                        "vst1.f32   {d12-d13}, [%7], %26 \n"

                        // loop1
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4[0]}, [%0]!      \n"
                        "vst1.f32   {d4[1]}, [%2]!      \n"

                        "vmov       q3, q7              \n"// use q7

                        "vst1.f32   {d5[0]}, [%4]!      \n"
                        "vst1.f32   {d5[1]}, [%6]!      \n"

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16[0]}, [%0]!     \n"
                        "vst1.f32   {d16[1]}, [%2]!     \n"

                        "vmla.f32   q4, q6, %e25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4]!     \n"
                        "vst1.f32   {d17[1]}, [%6]!     \n"

                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18[0]}, [%0]!     \n"
                        "vst1.f32   {d18[1]}, [%2]!     \n"

                        "vadd.f32   q8, q2, q3          \n"

                        "vst1.f32   {d19[0]}, [%4]!     \n"
                        "vst1.f32   {d19[1]}, [%6]!     \n"

                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16[0]}, [%0]      \n"
                        "vst1.f32   {d16[1]}, [%2]      \n"

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d17[0]}, [%4]      \n"
                        "vst1.f32   {d17[1]}, [%6]      \n"

                        "vtrn.32    q9, q2              \n"
                        "vtrn.32    q3, q6              \n"

                        "vswp       d19, d6             \n"
                        "vswp       d5, d12             \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "vst1.f32   {d4-d5}, [%3]       \n"
                        "vst1.f32   {d6-d7}, [%5]       \n"
                        "vst1.f32   {d12-d13}, [%7]     \n"

                        : "=r"(r0_tm0_0),     // %0
                          "=r"(r0_tm0_4),     // %1
                          "=r"(r0_tm1_0),     // %2
                          "=r"(r0_tm1_4),     // %3
                          "=r"(r0_tm2_0),     // %4
                          "=r"(r0_tm2_4),     // %5
                          "=r"(r0_tm3_0),     // %6
                          "=r"(r0_tm3_4),     // %7
                          "=r"(t0),     // %8
                          "=r"(t1),     // %9
                          "=r"(t2),     // %10
                          "=r"(t3)      // %11
                        : "0"(r0_tm0_0),
                          "1"(r0_tm0_4),
                          "2"(r0_tm1_0),
                          "3"(r0_tm1_4),
                          "4"(r0_tm2_0),
                          "5"(r0_tm2_4),
                          "6"(r0_tm3_0),
                          "7"(r0_tm3_4),
                          "8"(t0),
                          "9"(t1),
                          "10"(t2),
                          "11"(t3),
                          "w"(_coeff0), // %24
                          "w"(_coeff1), // %25
                          "r"(step)        // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    const float* r0 = img0.row(i * 6) + j * 6;

                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    float* r0_tm_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm_4 = img0_tm.row(i * w_tm/8 + j + tiles);

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        r0_tm_0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
                        r0_tm_4[3] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25f + tmp0[5]);

                        r0_tm_0[1] = tmp12a + tmp12b;
                        r0_tm_0[2] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                        float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);

                        r0_tm_0[3] = tmp34a + tmp34b;
                        r0_tm_4[0] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                        float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);

                        r0_tm_4[1] = tmp56a + tmp56b;
                        r0_tm_4[2] = tmp56a - tmp56b;

                        r0_tm_0 += img0_tm.w * tiles * 2;
                        r0_tm_4 += img0_tm.w * tiles * 2;
                    }
#endif // __ARM_NEON
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(4, 16 * w_tm/8 * h_tm/8, outch, 4u, opt.workspace_allocator);

        const int tiles = h_tm/8 * w_tm/8;

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);

            const float* ktm = kernel_tm.channel(pp);

            out0_tm.fill(0.f);
            out1_tm.fill(0.f);
            out2_tm.fill(0.f);
            out3_tm.fill(0.f);

            int q = 0;

#if __ARM_NEON && __aarch64__
            for (; q+3<inch; q+=4)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);
                const float* r2 = bottom_blob_tm.channel(q+2);
                const float* r3 = bottom_blob_tm.channel(q+3);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

                asm volatile(
                    "mov    w0, #16                     \n"// w0 = r = 16
                    "0:                                 \n"

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%8], #64     \n"// v0  v1  v2  v3  = _k00 _k01 _k02 _k03

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%8], #64     \n"// v4  v5  v6  v7  = _k10 _k11 _k12 _k13

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"// v8  v9  v10 v11 = _k20 _k21 _k22 _k23

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"// v12 v13 v14 v15 = _k30 _k31 _k32 _k33

                    // tile loop
                    "lsr    w1, %w18, #2                \n"// w1 = nn = tiles >> 2
                    "cmp    w1, #0                      \n"
                    "beq    2f                          \n"

                    //BEGIN tile loop
                    "prfm   pldl1keep, [%4, #128]       \n"//
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "1:                                 \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v20.4s}, [%0]              \n"
                    "add    x4, %0, #16                 \n"// x4 = %0 next

                    "fmla   v20.4s, v16.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.4s}, [%1]              \n"
                    "add    x5, %1, #16                 \n"// x5 = %1 next

                    "fmla   v21.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v22.4s}, [%2]              \n"
                    "add    x6, %2, #16                 \n"// x6 = %2 next

                    "fmla   v22.4s, v16.4s, v8.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v23.4s}, [%3]              \n"
                    "add    x7, %3, #16                 \n"// x7 = %3 next

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v23.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [x4, #128]       \n"
                    "ld1    {v24.4s}, [x4]              \n"

                    "fmla   v20.4s, v17.4s, v1.4s       \n"
                    "fmla   v21.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v22.4s, v17.4s, v9.4s       \n"
                    "fmla   v23.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [x5, #128]       \n"
                    "ld1    {v25.4s}, [x5]              \n"

                    "fmla   v20.4s, v18.4s, v2.4s       \n"
                    "fmla   v21.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v22.4s, v18.4s, v10.4s      \n"
                    "fmla   v23.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [x6, #128]       \n"
                    "ld1    {v26.4s}, [x6]              \n"

                    "fmla   v20.4s, v19.4s, v3.4s       \n"
                    "fmla   v21.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v22.4s, v19.4s, v11.4s      \n"
                    "fmla   v23.4s, v19.4s, v15.4s      \n"

                    ///////

                    "prfm   pldl1keep, [x7, #128]       \n"
                    "ld1    {v27.4s}, [x7]              \n"

                    "st1    {v20.4s}, [%0]              \n"
                    "add    %0, %0, #32                 \n"

                    "fmla   v24.4s, v16.4s, v0.4s       \n"
                    "fmla   v25.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v26.4s, v16.4s, v8.4s       \n"
                    "fmla   v27.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v20.4s}, [%0]              \n"

                    "st1    {v21.4s}, [%1]              \n"
                    "add    %1, %1, #32                 \n"

                    "fmla   v24.4s, v17.4s, v1.4s       \n"
                    "fmla   v25.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v26.4s, v17.4s, v9.4s       \n"
                    "fmla   v27.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.4s}, [%1]              \n"

                    "st1    {v22.4s}, [%2]              \n"
                    "add    %2, %2, #32                 \n"

                    "fmla   v24.4s, v18.4s, v2.4s       \n"
                    "fmla   v25.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v26.4s, v18.4s, v10.4s      \n"
                    "fmla   v27.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v22.4s}, [%2]              \n"

                    "st1    {v23.4s}, [%3]              \n"
                    "add    %3, %3, #32                 \n"

                    "fmla   v24.4s, v19.4s, v3.4s       \n"
                    "fmla   v25.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v26.4s, v19.4s, v11.4s      \n"
                    "fmla   v27.4s, v19.4s, v15.4s      \n"

                    ///////

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v23.4s}, [%3]              \n"

                    "st1    {v24.4s}, [x4]              \n"
                    "add    x4, x4, #32                 \n"

                    "fmla   v20.4s, v16.4s, v0.4s       \n"
                    "fmla   v21.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v22.4s, v16.4s, v8.4s       \n"
                    "fmla   v23.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [x4, #128]       \n"
                    "ld1    {v24.4s}, [x4]              \n"

                    "st1    {v25.4s}, [x5]              \n"
                    "add    x5, x5, #32                 \n"

                    "fmla   v20.4s, v17.4s, v1.4s       \n"
                    "fmla   v21.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v22.4s, v17.4s, v9.4s       \n"
                    "fmla   v23.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [x5, #128]       \n"
                    "ld1    {v25.4s}, [x5]              \n"

                    "st1    {v26.4s}, [x6]              \n"
                    "add    x6, x6, #32                 \n"

                    "fmla   v20.4s, v18.4s, v2.4s       \n"
                    "fmla   v21.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v22.4s, v18.4s, v10.4s      \n"
                    "fmla   v23.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [x6, #128]       \n"
                    "ld1    {v26.4s}, [x6]              \n"

                    "st1    {v27.4s}, [x7]              \n"
                    "add    x7, x7, #32                 \n"

                    "fmla   v20.4s, v19.4s, v3.4s       \n"
                    "fmla   v21.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v22.4s, v19.4s, v11.4s      \n"
                    "fmla   v23.4s, v19.4s, v15.4s      \n"

                    ///////

                    "prfm   pldl1keep, [x7, #128]       \n"
                    "ld1    {v27.4s}, [x7]              \n"

                    "st1    {v20.4s}, [%0]              \n"

                    "fmla   v24.4s, v16.4s, v0.4s       \n"
                    "fmla   v25.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v26.4s, v16.4s, v8.4s       \n"
                    "fmla   v27.4s, v16.4s, v12.4s      \n"

                    "st1    {v21.4s}, [%1]              \n"

                    "fmla   v24.4s, v17.4s, v1.4s       \n"
                    "fmla   v25.4s, v17.4s, v5.4s       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v26.4s, v17.4s, v9.4s       \n"
                    "fmla   v27.4s, v17.4s, v13.4s      \n"

                    "st1    {v22.4s}, [%2]              \n"

                    "fmla   v24.4s, v18.4s, v2.4s       \n"
                    "fmla   v25.4s, v18.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v26.4s, v18.4s, v10.4s      \n"
                    "fmla   v27.4s, v18.4s, v14.4s      \n"

                    "st1    {v23.4s}, [%3]              \n"

                    "fmla   v24.4s, v19.4s, v3.4s       \n"
                    "fmla   v25.4s, v19.4s, v7.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "fmla   v26.4s, v19.4s, v11.4s      \n"
                    "fmla   v27.4s, v19.4s, v15.4s      \n"

                    "st1    {v24.4s}, [x4], #16         \n"
                    "mov    %0, x4                      \n"

                    "st1    {v25.4s}, [x5], #16         \n"
                    "mov    %1, x5                      \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v26.4s}, [x6], #16         \n"
                    "mov    %2, x6                      \n"

                    "st1    {v27.4s}, [x7], #16         \n"
                    "mov    %3, x7                      \n"

                    "bne    1b                          \n"
                    "sub    %4, %4, #16                 \n"
                    //END tile loop

                    "2:                                 \n"

                    // remain loop
                    "and    w1, %w18, #3                \n"// w1 = remain = tiles & 3;
                    "cmp    w1, #0                      \n"
                    "beq    4f                          \n"

                    //BEGIN remain loop
                    "3:                                 \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v20.4s}, [%0]              \n"

                    "fmla   v20.4s, v16.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.4s}, [%1]              \n"

                    "fmla   v21.4s, v16.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v22.4s}, [%2]              \n"

                    "fmla   v22.4s, v16.4s, v8.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v23.4s}, [%3]              \n"

                    "fmla   v23.4s, v16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v17.4s}, [%5], #16         \n"

                    "fmla   v20.4s, v17.4s, v1.4s       \n"
                    "fmla   v21.4s, v17.4s, v5.4s       \n"

                    "fmla   v22.4s, v17.4s, v9.4s       \n"
                    "fmla   v23.4s, v17.4s, v13.4s      \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v18.4s}, [%6], #16         \n"

                    "fmla   v20.4s, v18.4s, v2.4s       \n"
                    "fmla   v21.4s, v18.4s, v6.4s       \n"

                    "fmla   v22.4s, v18.4s, v10.4s      \n"
                    "fmla   v23.4s, v18.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v19.4s}, [%7], #16         \n"

                    "fmla   v20.4s, v19.4s, v3.4s       \n"
                    "fmla   v21.4s, v19.4s, v7.4s       \n"
                    "fmla   v22.4s, v19.4s, v11.4s      \n"
                    "fmla   v23.4s, v19.4s, v15.4s      \n"

                    "st1    {v20.4s}, [%0], #16         \n"
                    "st1    {v21.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v22.4s}, [%2], #16         \n"
                    "st1    {v23.4s}, [%3], #16         \n"

                    "bne    3b                          \n"
                    //END remain loop

                    "4:                                 \n"

                    "subs   w0, w0, #1                  \n"
                    "bne    0b                          \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(r2),         // %6
                      "=r"(r3),         // %7
                      "=r"(ktm)         // %8
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(r1),
                      "6"(r2),
                      "7"(r3),
                      "8"(ktm),
                      "r"(tiles)        // %18
                    : "cc", "memory", "x0", "x1", "x4", "x5", "x6", "x7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
                );
            }
#endif // __ARM_NEON && __aarch64__

            for (; q+1<inch; q+=2)
            {
                const float* r0 = bottom_blob_tm.channel(q);
                const float* r1 = bottom_blob_tm.channel(q+1);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "mov    w0, #16                     \n"// w0 = r = 16
                    "0:                                 \n"

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%6], #32   \n"// v0 v1 = _k00 _k01

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v2.4s, v3.4s}, [%6], #32   \n"// v2 v3 = _k10 _k11

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v4.4s, v5.4s}, [%6], #32   \n"// v4 v5 = _k20 _k21

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%6], #32   \n"// v6 v7 = _k30 _k31

                    // tile loop
                    "lsr    w1, %w14, #2                \n"// w1 = nn = tiles >> 2
                    "cmp    w1, #0                      \n"
                    "beq    2f                          \n"

                    //BEGIN tile loop
                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "1:                                 \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    ////

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    ////

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    ////

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "bne    1b                          \n"
                    "sub    %4, %4, #16                 \n"
                    //END tile loop

                    "2:                                 \n"

                    // remain loop
                    "and    w1, %w14, #3                \n"// w1 = remain = tiles & 3;
                    "cmp    w1, #0                      \n"
                    "beq    4f                          \n"

                    //BEGIN remain loop
                    "3:                                 \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v20.4s}, [%4], #16         \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v16.4s}, [%0]              \n"

                    "fmla   v16.4s, v20.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v17.4s}, [%1]              \n"

                    "fmla   v17.4s, v20.4s, v2.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v18.4s}, [%2]              \n"

                    "fmla   v18.4s, v20.4s, v4.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v19.4s}, [%3]              \n"

                    "fmla   v19.4s, v20.4s, v6.4s       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.4s}, [%5], #16         \n"

                    "fmla   v16.4s, v21.4s, v1.4s       \n"
                    "fmla   v17.4s, v21.4s, v3.4s       \n"
                    "fmla   v18.4s, v21.4s, v5.4s       \n"
                    "fmla   v19.4s, v21.4s, v7.4s       \n"

                    "st1    {v16.4s}, [%0], #16         \n"
                    "st1    {v17.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v18.4s}, [%2], #16         \n"
                    "st1    {v19.4s}, [%3], #16         \n"

                    "bne    3b                          \n"
                    //END remain loop

                    "4:                                 \n"

                    "subs   w0, w0, #1                  \n"
                    "bne    0b                          \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(ktm)         // %6
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(r1),
                      "6"(ktm),
                      "r"(tiles)        // %14
                    : "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21"
                );
#else
                asm volatile(
                    "mov        r0, #16                 \n"// r0 = r = 16
                    "0:                                 \n"

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d0-d3}, [%6 :128]!     \n"// q0 q1 = _k00 _k01

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d4-d7}, [%6 :128]!     \n"// q2 q3 = _k10 _k11

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d8-d11}, [%6 :128]!    \n"// q4 q5 = _k20 _k21

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d12-d15}, [%6 :128]!   \n"// q6 q7 = _k30 _k31

                    // tile loop
                    "lsr        r1, %14, #2             \n"// r1 = nn = tiles >> 2
                    "cmp        r1, #0                  \n"
                    "beq        2f                      \n"

                    //BEGIN tile loop
                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "1:                                 \n"

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    ////

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    ////

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    ////

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    "subs       r1, #1                  \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "bne        1b                      \n"
                    "sub        %4, %4, #16             \n"
                    //END tile loop

                    "2:                                 \n"

                    // remain loop
                    "and        r1, %14, #3             \n"// r1 = remain = tiles & 3;
                    "cmp        r1, #0                  \n"
                    "beq        4f                      \n"

                    //BEGIN remain loop
                    "3:                                 \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q2             \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q4            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q6            \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5 :128]!   \n"// q13 = _r1

                    "vmla.f32   q8, q13, q1             \n"
                    "vmla.f32   q9, q13, q3             \n"
                    "vmla.f32   q10, q13, q5            \n"
                    "vmla.f32   q11, q13, q7            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    "subs       r1, #1                  \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "bne        3b                      \n"
                    //END remain loop

                    "4:                                 \n"

                    "subs       r0, #1                  \n"
                    "bne        0b                      \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(ktm)         // %6
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(r1),
                      "6"(ktm),
                      "r"(tiles)        // %14
                    : "cc", "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
#endif // __aarch64__
#else
                for (int r=0; r<16; r++)
                {
                    for (int t=0; t<tiles; t++)
                    {
                        for (int m=0; m<4; m++)
                        {
                            output0_tm[m] += r0[m] * ktm[0 +m];
                            output0_tm[m] += r1[m] * ktm[4 +m];
                            output1_tm[m] += r0[m] * ktm[8 +m];
                            output1_tm[m] += r1[m] * ktm[12+m];
                            output2_tm[m] += r0[m] * ktm[16+m];
                            output2_tm[m] += r1[m] * ktm[20+m];
                            output3_tm[m] += r0[m] * ktm[24+m];
                            output3_tm[m] += r1[m] * ktm[28+m];
                        }

                        r0 += 4;
                        r1 += 4;
                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
                    }

                    ktm += 32;
                }
#endif // __ARM_NEON
            }

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                float* output0_tm = out0_tm;
                float* output1_tm = out1_tm;
                float* output2_tm = out2_tm;
                float* output3_tm = out3_tm;

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "mov    w0, #16                     \n"// w0 = r = 16
                    "0:                                 \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%5], #32   \n"// v0 v1 = _k00 _k10

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v2.4s, v3.4s}, [%5], #32   \n"// v2 v3 = _k20 _k30

                    // tile loop
                    "mov    w1, %w12                    \n"// w1 = tiles
                    "cmp    w1, #0                      \n"
                    "beq    2f                          \n"

                    //BEGIN tile loop
                    "1:                                 \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4s}, [%4], #16         \n"

                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v17.4s}, [%0]              \n"

                    "fmla   v17.4s, v16.4s, v0.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v18.4s}, [%1]              \n"

                    "fmla   v18.4s, v16.4s, v1.4s       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v19.4s}, [%2]              \n"

                    "fmla   v19.4s, v16.4s, v2.4s       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v20.4s}, [%3]              \n"

                    "fmla   v20.4s, v16.4s, v3.4s       \n"

                    "st1    {v17.4s}, [%0], #16         \n"
                    "st1    {v18.4s}, [%1], #16         \n"

                    "subs   w1, w1, #1                  \n"

                    "st1    {v19.4s}, [%2], #16         \n"
                    "st1    {v20.4s}, [%3], #16         \n"

                    "bne    1b                          \n"
                    //END tile loop

                    "2:                                 \n"

                    "subs   w0, w0, #1                  \n"
                    "bne    0b                          \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(ktm)         // %5
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(ktm),
                      "r"(tiles)        // %12
                    : "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "mov        r0, #16                 \n"// r0 = r = 16
                    "0:                                 \n"

                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d0-d3}, [%5 :128]!     \n"// q0 q1 = _k00 _k10

                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d4-d7}, [%5 :128]!     \n"// q2 q3 = _k20 _k30

                    // tile loop
                    "mov        r1, %12                 \n"// r1 = tiles
                    "cmp        r1, #0                  \n"
                    "beq        2f                      \n"

                    //BEGIN tile loop
                    "1:                                 \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4 :128]!   \n"// q12 = _r0

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                    "vmla.f32   q8, q12, q0             \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1 :128]    \n"// q9 = _output1_tm

                    "vmla.f32   q9, q12, q1             \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2 :128]    \n"// q10 = _output2_tm

                    "vmla.f32   q10, q12, q2            \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3 :128]    \n"// q11 = _output3_tm

                    "vmla.f32   q11, q12, q3            \n"

                    "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%1 :128]!   \n"

                    "subs       r1, #1                  \n"

                    "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                    "bne        1b                      \n"
                    //END tile loop

                    "2:                                 \n"

                    "subs       r0, #1                  \n"
                    "bne        0b                      \n"

                    : "=r"(output0_tm), // %0
                      "=r"(output1_tm), // %1
                      "=r"(output2_tm), // %2
                      "=r"(output3_tm), // %3
                      "=r"(r0),         // %4
                      "=r"(ktm)         // %5
                    : "0"(output0_tm),
                      "1"(output1_tm),
                      "2"(output2_tm),
                      "3"(output3_tm),
                      "4"(r0),
                      "5"(ktm),
                      "r"(tiles)        // %12
                    : "cc", "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
#endif // __aarch64__
#else
                for (int r=0; r<16; r++)
                {
                    for (int t=0; t<tiles; t++)
                    {
                        for (int m=0; m<4; m++)
                        {
                            output0_tm[m] += r0[m] * ktm[0 +m];
                            output1_tm[m] += r0[m] * ktm[4 +m];
                            output2_tm[m] += r0[m] * ktm[8 +m];
                            output3_tm[m] += r0[m] * ktm[12+m];
                        }

                        r0 += 4;
                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
                    }

                    ktm += 16;
                }
#endif // __ARM_NEON
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);

            const float* ktm = (const float*)kernel_tm.channel(nn_outch) + 8*8 * inch * (p-remain_outch_start);

            out0_tm.fill(0.f);

            int q = 0;

            for (; q<inch; q++)
            {
                const float* r0 = bottom_blob_tm.channel(q);

                float* output0_tm = out0_tm;

                for (int r=0; r<16; r++)
                {
#if __ARM_NEON
                float32x4_t _k00 = vld1q_f32(ktm); ktm += 4;
#endif // __ARM_NEON

                // tile
                for (int i=0; i<tiles; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #128]   \n"
                        "ld1    {v17.4s}, [%1], #16     \n"

                        "prfm   pldl1keep, [%0, #128]   \n"
                        "ld1    {v16.4s}, [%0]          \n"

                        "fmla   v16.4s, v17.4s, %4.4s   \n"

                        "st1    {v16.4s}, [%0], #16     \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0)          // %1
                        : "0"(output0_tm),
                          "1"(r0),
                          "w"(_k00)         // %4
                        : "cc", "memory", "v16", "v17"
                    );
#else
                    asm volatile(
                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d18-d19}, [%1 :128]!   \n"// q9 = _r0

                        "pld        [%0, #128]              \n"
                        "vld1.f32   {d16-d17}, [%0 :128]    \n"// q8 = _output0_tm

                        "vmla.f32   q8, q9, %q4             \n"

                        "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                        : "=r"(output0_tm), // %0
                          "=r"(r0)          // %1
                        : "0"(output0_tm),
                          "1"(r0),
                          "w"(_k00)         // %4
                        : "cc", "memory", "q8", "q9"
                    );
#endif // __aarch64__
#else
                    for (int m=0; m<4; m++)
                    {
                        output0_tm[m] += r0[m] * ktm[m];
                    }

                    r0 += 4;
                    output0_tm += 4;
#endif // __ARM_NEON
                }

#if !__ARM_NEON
                ktm += 4;
#endif // __ARM_NEON
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    {
//         const float otm[6][8] = {
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

#if __ARM_NEON
        const float coeff[4] = { 4.f, 8.f, 16.f, 32.f };
        float32x4_t _coeff = vld1q_f32(coeff);
#endif // __ARM_NEON

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;
#if __ARM_NEON
            float32x2_t _bias0 = vdup_n_f32(bias0);
#endif // __ARM_NEON

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
#if __ARM_NEON
                    const float* output0_tm0_0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm0_4 = out0_tm.row(i * w_tm/8 + j + tiles);
                    const float* output0_tm1_0 = out0_tm.row(i * w_tm/8 + j + tiles*2);
                    const float* output0_tm1_4 = out0_tm.row(i * w_tm/8 + j + tiles*3);
                    const float* output0_tm2_0 = out0_tm.row(i * w_tm/8 + j + tiles*4);
                    const float* output0_tm2_4 = out0_tm.row(i * w_tm/8 + j + tiles*5);
                    const float* output0_tm3_0 = out0_tm.row(i * w_tm/8 + j + tiles*6);
                    const float* output0_tm3_4 = out0_tm.row(i * w_tm/8 + j + tiles*7);

#if __aarch64__
                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _output0_tm0_0123 = vld1q_f32(output0_tm0_0);
                        float32x4_t _output0_tm0_4567 = vld1q_f32(output0_tm0_4);
                        float32x4_t _output0_tm1_0123 = vld1q_f32(output0_tm1_0);
                        float32x4_t _output0_tm1_4567 = vld1q_f32(output0_tm1_4);
                        float32x4_t _output0_tm2_0123 = vld1q_f32(output0_tm2_0);
                        float32x4_t _output0_tm2_4567 = vld1q_f32(output0_tm2_4);
                        float32x4_t _output0_tm3_0123 = vld1q_f32(output0_tm3_0);
                        float32x4_t _output0_tm3_4567 = vld1q_f32(output0_tm3_4);

                        float32x4x2_t _output0_tm01_00221133 = vtrnq_f32(_output0_tm0_0123, _output0_tm1_0123);
                        float32x4x2_t _output0_tm01_44665577 = vtrnq_f32(_output0_tm0_4567, _output0_tm1_4567);
                        float32x4x2_t _output0_tm23_00221133 = vtrnq_f32(_output0_tm2_0123, _output0_tm3_0123);
                        float32x4x2_t _output0_tm23_44665577 = vtrnq_f32(_output0_tm2_4567, _output0_tm3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _output0_tm_00 = vcombine_f32(vget_low_f32(_output0_tm01_00221133.val[0]), vget_low_f32(_output0_tm23_00221133.val[0]));
                        float32x4_t _output0_tm_11 = vcombine_f32(vget_low_f32(_output0_tm01_00221133.val[1]), vget_low_f32(_output0_tm23_00221133.val[1]));
                        float32x4_t _output0_tm_22 = vcombine_f32(vget_high_f32(_output0_tm01_00221133.val[0]), vget_high_f32(_output0_tm23_00221133.val[0]));
                        float32x4_t _output0_tm_33 = vcombine_f32(vget_high_f32(_output0_tm01_00221133.val[1]), vget_high_f32(_output0_tm23_00221133.val[1]));
                        float32x4_t _output0_tm_44 = vcombine_f32(vget_low_f32(_output0_tm01_44665577.val[0]), vget_low_f32(_output0_tm23_44665577.val[0]));
                        float32x4_t _output0_tm_55 = vcombine_f32(vget_low_f32(_output0_tm01_44665577.val[1]), vget_low_f32(_output0_tm23_44665577.val[1]));
                        float32x4_t _output0_tm_66 = vcombine_f32(vget_high_f32(_output0_tm01_44665577.val[0]), vget_high_f32(_output0_tm23_44665577.val[0]));
                        float32x4_t _output0_tm_77 = vcombine_f32(vget_high_f32(_output0_tm01_44665577.val[1]), vget_high_f32(_output0_tm23_44665577.val[1]));

                        float32x4_t _tmp024a = vaddq_f32(_output0_tm_11, _output0_tm_22);
                        float32x4_t _tmp135a = vsubq_f32(_output0_tm_11, _output0_tm_22);

                        float32x4_t _tmp024b = vaddq_f32(_output0_tm_33, _output0_tm_44);
                        float32x4_t _tmp135b = vsubq_f32(_output0_tm_33, _output0_tm_44);

                        float32x4_t _tmp024c = vaddq_f32(_output0_tm_55, _output0_tm_66);
                        float32x4_t _tmp135c = vsubq_f32(_output0_tm_55, _output0_tm_66);

                        float32x4_t _tmp0 = vaddq_f32(_output0_tm_00, _tmp024a);
                        _tmp0 = vmlaq_lane_f32(_tmp0, _tmp024c, vget_high_f32(_coeff), 1);
                        _tmp0 = vaddq_f32(_tmp0, _tmp024b);

                        float32x4_t _tmp2 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _tmp2 = vmlaq_lane_f32(_tmp2, _tmp024c, vget_low_f32(_coeff), 1);

                        float32x4_t _tmp4 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);

                        vst1q_f32(&tmp[0][m], _tmp0);
                        vst1q_f32(&tmp[2][m], _tmp2);
                        vst1q_f32(&tmp[4][m], _tmp4);

                        float32x4_t _tmp1 = vmlaq_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);

                        float32x4_t _tmp3 = vmlaq_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        _tmp3 = vmlaq_lane_f32(_tmp3, _tmp135c, vget_low_f32(_coeff), 0);

                        float32x4_t _tmp5 = vaddq_f32(_output0_tm_77, _tmp135a);
                        _tmp5 = vmlaq_lane_f32(_tmp5, _tmp135b, vget_high_f32(_coeff), 1);
                        _tmp5 = vaddq_f32(_tmp5, _tmp135c);

                        vst1q_f32(&tmp[1][m], _tmp1);
                        vst1q_f32(&tmp[3][m], _tmp3);
                        vst1q_f32(&tmp[5][m], _tmp5);

                        output0_tm0_0 += out0_tm.w * tiles * 2*4;
                        output0_tm0_4 += out0_tm.w * tiles * 2*4;
                        output0_tm1_0 += out0_tm.w * tiles * 2*4;
                        output0_tm1_4 += out0_tm.w * tiles * 2*4;
                        output0_tm2_0 += out0_tm.w * tiles * 2*4;
                        output0_tm2_4 += out0_tm.w * tiles * 2*4;
                        output0_tm3_0 += out0_tm.w * tiles * 2*4;
                        output0_tm3_4 += out0_tm.w * tiles * 2*4;
                    }

                    const float* t0 = tmp[0];
                    const float* t1 = tmp[1];

                    float* output0 = out0.row(i * 6) + j * 6;
                    float* output1 = output0 + outw;

                    for (int m=0; m+1<6; m+=2)
                    {
                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0+4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1+4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);

                        float32x2_t _t_00 = vget_low_f32(_t01_00221133.val[0]);
                        float32x2_t _t_11 = vget_low_f32(_t01_00221133.val[1]);
                        float32x2_t _t_22 = vget_high_f32(_t01_00221133.val[0]);
                        float32x2_t _t_33 = vget_high_f32(_t01_00221133.val[1]);
                        float32x2_t _t_44 = vget_low_f32(_t01_44665577.val[0]);
                        float32x2_t _t_55 = vget_low_f32(_t01_44665577.val[1]);
                        float32x2_t _t_66 = vget_high_f32(_t01_44665577.val[0]);
                        float32x2_t _t_77 = vget_high_f32(_t01_44665577.val[1]);

                        float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);

                        float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);

                        float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);

                        float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        _output_0 = vadd_f32(_output_0, _tmp024b);
                        _output_0 = vadd_f32(_output_0, _bias0);

                        float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                        _output_2 = vadd_f32(_output_2, _bias0);

                        float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _bias0);

                        output0[0] = vget_lane_f32(_output_0, 0);
                        output1[0] = vget_lane_f32(_output_0, 1);
                        output0[2] = vget_lane_f32(_output_2, 0);
                        output1[2] = vget_lane_f32(_output_2, 1);
                        output0[4] = vget_lane_f32(_output_4, 0);
                        output1[4] = vget_lane_f32(_output_4, 1);

                        float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        _output_1 = vadd_f32(_output_1, _bias0);

                        float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                        _output_3 = vadd_f32(_output_3, _bias0);

                        float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        _output_5 = vadd_f32(_output_5, _tmp135c);
                        _output_5 = vadd_f32(_output_5, _bias0);

                        output0[1] = vget_lane_f32(_output_1, 0);
                        output1[1] = vget_lane_f32(_output_1, 1);
                        output0[3] = vget_lane_f32(_output_3, 0);
                        output1[3] = vget_lane_f32(_output_3, 1);
                        output0[5] = vget_lane_f32(_output_5, 0);
                        output1[5] = vget_lane_f32(_output_5, 1);

                        t0 += 8*2;
                        t1 += 8*2;
                        output0 += outw*2;
                        output1 += outw*2;
                    }
#else // __aarch64__
                    float* t0 = tmp[0];
                    float* t1 = tmp[1];

                    int step = out0_tm.w * tiles * 2*4 *4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d17}, [%2], %21 \n"
                        "vld1.f32   {d18-d19}, [%3], %21 \n"
                        "vld1.f32   {d20-d21}, [%4], %21 \n"
                        "vld1.f32   {d22-d23}, [%5], %21 \n"
                        "vld1.f32   {d24-d25}, [%6], %21 \n"
                        "vld1.f32   {d26-d27}, [%7], %21 \n"
                        "vld1.f32   {d28-d29}, [%8], %21 \n"
                        "vld1.f32   {d30-d31}, [%9], %21 \n"

                        "vtrn.32    q8, q10             \n"
                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vadd.f32   q2, q10, q12        \n"
                        "vsub.f32   q3, q10, q12        \n"

                        "vadd.f32   q4, q14, q9         \n"
                        "vsub.f32   q5, q14, q9         \n"

                        "vadd.f32   q6, q11, q13        \n"
                        "vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

                        "vmov       q9, q3              \n"
                        "vadd.f32   q8, q8, q2          \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vmov       q12, q2             \n"
                        "vmov       q10, q2             \n"
                        "vmov       q11, q3             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q8, q8, q4          \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q15, q15, q7        \n"

                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"
                        "sub        %0, %0, #112        \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"
                        "sub        %1, %1, #112        \n"

                        // loop1
                        "vld1.f32   {d16-d17}, [%2]     \n"
                        "vld1.f32   {d18-d19}, [%3]     \n"
                        "vld1.f32   {d20-d21}, [%4]     \n"
                        "vld1.f32   {d22-d23}, [%5]     \n"
                        "vld1.f32   {d24-d25}, [%6]     \n"
                        "vld1.f32   {d26-d27}, [%7]     \n"
                        "vld1.f32   {d28-d29}, [%8]     \n"
                        "vld1.f32   {d30-d31}, [%9]     \n"

                        "vtrn.32    q8, q10             \n"
                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vadd.f32   q2, q10, q12        \n"
                        "vsub.f32   q3, q10, q12        \n"

                        "vadd.f32   q4, q14, q9         \n"
                        "vsub.f32   q5, q14, q9         \n"

                        "vadd.f32   q6, q11, q13        \n"
                        "vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

                        "vmov       q9, q3              \n"
                        "vadd.f32   q8, q8, q2          \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vmov       q12, q2             \n"
                        "vmov       q10, q2             \n"
                        "vmov       q11, q3             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q8, q8, q4          \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q15, q15, q7        \n"

                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"

                        : "=r"(t0),             // %0
                          "=r"(t1),             // %1
                          "=r"(output0_tm0_0),  // %2
                          "=r"(output0_tm0_4),  // %3
                          "=r"(output0_tm1_0),  // %4
                          "=r"(output0_tm1_4),  // %5
                          "=r"(output0_tm2_0),  // %6
                          "=r"(output0_tm2_4),  // %7
                          "=r"(output0_tm3_0),  // %8
                          "=r"(output0_tm3_4)   // %9
                        : "0"(t0),
                          "1"(t1),
                          "2"(output0_tm0_0),
                          "3"(output0_tm0_4),
                          "4"(output0_tm1_0),
                          "5"(output0_tm1_4),
                          "6"(output0_tm2_0),
                          "7"(output0_tm2_4),
                          "8"(output0_tm3_0),
                          "9"(output0_tm3_4),
                          "w"(_coeff),          // %20
                          "r"(step)             // %21
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmp[0];
                    t1 = tmp[1];

                    float* output0 = out0.row(i * 6) + j * 6;
                    float* output1 = output0 + outw;

                    int stepw = outw*2 * 4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        // loop1
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        // loop2
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        : "=r"(output0),    // %0
                          "=r"(output1),    // %1
                          "=r"(t0),         // %2
                          "=r"(t1)          // %3
                        : "0"(output0),
                          "1"(output1),
                          "2"(t0),
                          "3"(t1),
                          "w"(_coeff),      // %8
                          "w"(_bias0),      // %9
                          "r"(stepw)        // %10
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    const float* output0_tm_0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm_4 = out0_tm.row(i * w_tm/8 + j + tiles);

                    for (int m=0; m<8; m++)
                    {
                        float tmp024a = output0_tm_0[1] + output0_tm_0[2];
                        float tmp135a = output0_tm_0[1] - output0_tm_0[2];

                        float tmp024b = output0_tm_0[3] + output0_tm_4[0];
                        float tmp135b = output0_tm_0[3] - output0_tm_4[0];

                        float tmp024c = output0_tm_4[1] + output0_tm_4[2];
                        float tmp135c = output0_tm_4[1] - output0_tm_4[2];

                        tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm_4[3] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += out0_tm.w * tiles * 2;
                        output0_tm_4 += out0_tm.w * tiles * 2;
                    }

                    float* output0 = out0.row(i * 6) + j * 6;

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
#endif // __ARM_NEON
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt.blob_allocator, opt.num_threads);
}

static void conv3x3s1_winograd64_neon5(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt.workspace_allocator, opt.num_threads);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;
        bottom_blob_tm.create(1, 64 * tiles, inch, 4u, opt.workspace_allocator);
//         bottom_blob_tm.create(inch, tiles, 64);

//         const float itm[8][8] = {
//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
//
//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
//
//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
//
//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
//
//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
//         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

#if __ARM_NEON
        const float coeff[8] = {
            0.25f, 0.5f, -1.25f,   2.f,
            -2.5f,  4.f,  4.25f, 5.25f
        };
        float32x4_t _coeff0 = vld1q_f32(coeff);
        float32x4_t _coeff1 = vld1q_f32(coeff+4);
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q<inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8];

            // tile
            for (int i=0; i<h_tm/8; i++)
            {
                for (int j=0; j<w_tm/8; j++)
                {
#if __ARM_NEON
                    const float* r0 = img0.row(i * 6) + j * 6;
                    const float* r1 = r0 + w;
                    const float* r2 = r0 + w*2;
                    const float* r3 = r0 + w*3;

#if __aarch64__
                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _r0_0123 = vld1q_f32(r0);
                        float32x4_t _r0_4567 = vld1q_f32(r0+4);
                        float32x4_t _r1_0123 = vld1q_f32(r1);
                        float32x4_t _r1_4567 = vld1q_f32(r1+4);
                        float32x4_t _r2_0123 = vld1q_f32(r2);
                        float32x4_t _r2_4567 = vld1q_f32(r2+4);
                        float32x4_t _r3_0123 = vld1q_f32(r3);
                        float32x4_t _r3_4567 = vld1q_f32(r3+4);

                        float32x4x2_t _r01_00221133 = vtrnq_f32(_r0_0123, _r1_0123);
                        float32x4x2_t _r01_44665577 = vtrnq_f32(_r0_4567, _r1_4567);
                        float32x4x2_t _r23_00221133 = vtrnq_f32(_r2_0123, _r3_0123);
                        float32x4x2_t _r23_44665577 = vtrnq_f32(_r2_4567, _r3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _r_00 = vcombine_f32(vget_low_f32(_r01_00221133.val[0]), vget_low_f32(_r23_00221133.val[0]));
                        float32x4_t _r_11 = vcombine_f32(vget_low_f32(_r01_00221133.val[1]), vget_low_f32(_r23_00221133.val[1]));
                        float32x4_t _r_22 = vcombine_f32(vget_high_f32(_r01_00221133.val[0]), vget_high_f32(_r23_00221133.val[0]));
                        float32x4_t _r_33 = vcombine_f32(vget_high_f32(_r01_00221133.val[1]), vget_high_f32(_r23_00221133.val[1]));
                        float32x4_t _r_44 = vcombine_f32(vget_low_f32(_r01_44665577.val[0]), vget_low_f32(_r23_44665577.val[0]));
                        float32x4_t _r_55 = vcombine_f32(vget_low_f32(_r01_44665577.val[1]), vget_low_f32(_r23_44665577.val[1]));
                        float32x4_t _r_66 = vcombine_f32(vget_high_f32(_r01_44665577.val[0]), vget_high_f32(_r23_44665577.val[0]));
                        float32x4_t _r_77 = vcombine_f32(vget_high_f32(_r01_44665577.val[1]), vget_high_f32(_r23_44665577.val[1]));

                        float32x4_t _r_0_m_6 = vsubq_f32(_r_00, _r_66);
                        float32x4_t _r_7_m_1 = vsubq_f32(_r_77, _r_11);

                        float32x4_t _r_4_m_2 = vsubq_f32(_r_44, _r_22);
                        float32x4_t _r_3_m_5 = vsubq_f32(_r_33, _r_55);

                        float32x4_t _tmp0 = vmlaq_lane_f32(_r_0_m_6, _r_4_m_2, vget_high_f32(_coeff1), 1);
                        float32x4_t _tmp7 = vmlaq_lane_f32(_r_7_m_1, _r_3_m_5, vget_high_f32(_coeff1), 1);

                        vst1q_f32(&tmp[0][m], _tmp0);
                        vst1q_f32(&tmp[7][m], _tmp7);

                        float32x4_t _r_2_a_6 = vaddq_f32(_r_22, _r_66);
                        float32x4_t _r_1_a_5 = vaddq_f32(_r_11, _r_55);

                        float32x4_t _tmp12a = vmlsq_lane_f32(_r_2_a_6, _r_44, vget_high_f32(_coeff1), 0);
                        float32x4_t _tmp12b = vmlsq_lane_f32(_r_1_a_5, _r_33, vget_high_f32(_coeff1), 0);

                        float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);

                        vst1q_f32(&tmp[1][m], _tmp1);
                        vst1q_f32(&tmp[2][m], _tmp2);

                        float32x4_t _r_4_x_c = vmulq_lane_f32(_r_44, vget_high_f32(_coeff0), 0);
                        float32x4_t _r_3_x_c = vmulq_lane_f32(_r_33, vget_low_f32(_coeff1), 0);

                        float32x4_t _tmp34a = vaddq_f32(_r_66, _r_4_x_c);
                        _tmp34a = vmlaq_lane_f32(_tmp34a, _r_22, vget_low_f32(_coeff0), 0);

                        float32x4_t _tmp34b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_low_f32(_coeff0), 1);
                        _tmp34b = vmlaq_lane_f32(_tmp34b, _r_55, vget_high_f32(_coeff0), 1);

                        float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);

                        vst1q_f32(&tmp[3][m], _tmp3);
                        vst1q_f32(&tmp[4][m], _tmp4);

                        // reuse r04 * 1.25
                        // reuse r03 * 2.5
                        float32x4_t _r_2_a_4c = vaddq_f32(_r_22, _r_4_x_c);
                        float32x4_t _tmp56a = vmlaq_lane_f32(_r_66, _r_2_a_4c, vget_low_f32(_coeff1), 1);
                        float32x4_t _tmp56b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_high_f32(_coeff0), 1);
                        _tmp56b = vmlaq_lane_f32(_tmp56b, _r_55, vget_low_f32(_coeff0), 1);

                        float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);

                        vst1q_f32(&tmp[5][m], _tmp5);
                        vst1q_f32(&tmp[6][m], _tmp6);

                        r0 += w*4;
                        r1 += w*4;
                        r2 += w*4;
                        r3 += w*4;
                    }

                    const float* t0 = tmp[0];
                    const float* t1 = tmp[1];
                    const float* t2 = tmp[2];
                    const float* t3 = tmp[3];

                    float* r0_tm0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm1 = img0_tm.row(i * w_tm/8 + j + tiles*8);
                    float* r0_tm2 = img0_tm.row(i * w_tm/8 + j + tiles*16);
                    float* r0_tm3 = img0_tm.row(i * w_tm/8 + j + tiles*24);

                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0+4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1+4);
                        float32x4_t _t2_0123 = vld1q_f32(t2);
                        float32x4_t _t2_4567 = vld1q_f32(t2+4);
                        float32x4_t _t3_0123 = vld1q_f32(t3);
                        float32x4_t _t3_4567 = vld1q_f32(t3+4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                        float32x4x2_t _t23_00221133 = vtrnq_f32(_t2_0123, _t3_0123);
                        float32x4x2_t _t23_44665577 = vtrnq_f32(_t2_4567, _t3_4567);

                        // no vswp intrinsic  :(
                        float32x4_t _t_00 = vcombine_f32(vget_low_f32(_t01_00221133.val[0]), vget_low_f32(_t23_00221133.val[0]));
                        float32x4_t _t_11 = vcombine_f32(vget_low_f32(_t01_00221133.val[1]), vget_low_f32(_t23_00221133.val[1]));
                        float32x4_t _t_22 = vcombine_f32(vget_high_f32(_t01_00221133.val[0]), vget_high_f32(_t23_00221133.val[0]));
                        float32x4_t _t_33 = vcombine_f32(vget_high_f32(_t01_00221133.val[1]), vget_high_f32(_t23_00221133.val[1]));
                        float32x4_t _t_44 = vcombine_f32(vget_low_f32(_t01_44665577.val[0]), vget_low_f32(_t23_44665577.val[0]));
                        float32x4_t _t_55 = vcombine_f32(vget_low_f32(_t01_44665577.val[1]), vget_low_f32(_t23_44665577.val[1]));
                        float32x4_t _t_66 = vcombine_f32(vget_high_f32(_t01_44665577.val[0]), vget_high_f32(_t23_44665577.val[0]));
                        float32x4_t _t_77 = vcombine_f32(vget_high_f32(_t01_44665577.val[1]), vget_high_f32(_t23_44665577.val[1]));

                        float32x4_t _t_0_m_6 = vsubq_f32(_t_00, _t_66);
                        float32x4_t _t_7_m_1 = vsubq_f32(_t_77, _t_11);

                        float32x4_t _t_4_m_2 = vsubq_f32(_t_44, _t_22);
                        float32x4_t _t_3_m_5 = vsubq_f32(_t_33, _t_55);

                        float32x4_t _r0_tm_0_0 = vmlaq_lane_f32(_t_0_m_6, _t_4_m_2, vget_high_f32(_coeff1), 1);
                        float32x4_t _r0_tm_4_3 = vmlaq_lane_f32(_t_7_m_1, _t_3_m_5, vget_high_f32(_coeff1), 1);

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_0, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_0, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_0, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_0, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        float32x4_t _t_2_m_6 = vaddq_f32(_t_22, _t_66);
                        float32x4_t _t_1_m_5 = vaddq_f32(_t_11, _t_55);

                        float32x4_t _tmp12a = vmlsq_lane_f32(_t_2_m_6, _t_44, vget_high_f32(_coeff1), 0);
                        float32x4_t _tmp12b = vmlsq_lane_f32(_t_1_m_5, _t_33, vget_high_f32(_coeff1), 0);

                        float32x4_t _r0_tm_0_1 = vaddq_f32(_tmp12a, _tmp12b);
                        float32x4_t _r0_tm_0_2 = vsubq_f32(_tmp12a, _tmp12b);

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_1, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_1, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_1, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_1, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_2, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_2, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_2, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_2, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        float32x4_t _t_4_x_c = vmulq_lane_f32(_t_44, vget_high_f32(_coeff0), 0);
                        float32x4_t _t_3_x_c = vmulq_lane_f32(_t_33, vget_low_f32(_coeff1), 0);

                        float32x4_t _tmp34a = vaddq_f32(_t_66, _t_4_x_c);
                        _tmp34a = vmlaq_lane_f32(_tmp34a, _t_22, vget_low_f32(_coeff0), 0);

                        float32x4_t _tmp34b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_low_f32(_coeff0), 1);
                        _tmp34b = vmlaq_lane_f32(_tmp34b, _t_55, vget_high_f32(_coeff0), 1);

                        float32x4_t _r0_tm_0_3 = vaddq_f32(_tmp34a, _tmp34b);
                        float32x4_t _r0_tm_4_0 = vsubq_f32(_tmp34a, _tmp34b);

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_3, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_3, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_3, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_3, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_0, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_0, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_0, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_0, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        float32x4_t _t_2_a_4c = vaddq_f32(_t_22, _t_4_x_c);
                        float32x4_t _tmp56a = vmlaq_lane_f32(_t_66, _t_2_a_4c, vget_low_f32(_coeff1), 1);
                        float32x4_t _tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                        _tmp56b = vmlaq_lane_f32(_tmp56b, _t_55, vget_low_f32(_coeff0), 1);

                        float32x4_t _r0_tm_4_1 = vaddq_f32(_tmp56a, _tmp56b);
                        float32x4_t _r0_tm_4_2 = vsubq_f32(_tmp56a, _tmp56b);

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_1, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_1, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_1, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_1, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_2, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_2, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_2, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_2, 3);

                        r0_tm0 += img0_tm.w*tiles;
                        r0_tm1 += img0_tm.w*tiles;
                        r0_tm2 += img0_tm.w*tiles;
                        r0_tm3 += img0_tm.w*tiles;

                        r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_3, 0);
                        r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_3, 1);
                        r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_3, 2);
                        r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_3, 3);

                        t0 += 8*4;
                        t1 += 8*4;
                        t2 += 8*4;
                        t3 += 8*4;

                        r0_tm0 += img0_tm.w*tiles*25;
                        r0_tm1 += img0_tm.w*tiles*25;
                        r0_tm2 += img0_tm.w*tiles*25;
                        r0_tm3 += img0_tm.w*tiles*25;
                    }
#else // __aarch64__
                    float* t0 = tmp[0];
                    float* t1 = tmp[1];
                    float* t2 = tmp[2];
                    float* t3 = tmp[3];
                    float* t4 = tmp[4];
                    float* t5 = tmp[5];
                    float* t6 = tmp[6];
                    float* t7 = tmp[7];

                    int stepw = w*4*4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%8], %26    \n"
                        "vld1.f32   {d20-d23}, [%9], %26    \n"
                        "vld1.f32   {d24-d27}, [%10], %26   \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11], %26   \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

                        "vmov       q3, q7              \n"// use q7

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

                        "vmla.f32   q4, q6, %e25[1]     \n"
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

                        "vadd.f32   q8, q2, q3          \n"
                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
                        "vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
                        "vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

                        "vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

                        // loop1
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

                        "vmov       q3, q7              \n"// use q7

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

                        "vmla.f32   q4, q6, %e25[1]     \n"
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

                        "vadd.f32   q8, q2, q3          \n"
                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vadd.f32   q2, q4, q5          \n"
                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
                        "vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
                        "vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

                        "vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

                        : "=r"(t0),     // %0
                          "=r"(t1),     // %1
                          "=r"(t2),     // %2
                          "=r"(t3),     // %3
                          "=r"(t4),     // %4
                          "=r"(t5),     // %5
                          "=r"(t6),     // %6
                          "=r"(t7),     // %7
                          "=r"(r0),     // %8
                          "=r"(r1),     // %9
                          "=r"(r2),     // %10
                          "=r"(r3)      // %11
                        : "0"(t0),
                          "1"(t1),
                          "2"(t2),
                          "3"(t3),
                          "4"(t4),
                          "5"(t5),
                          "6"(t6),
                          "7"(t7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(r3),
                          "w"(_coeff0), // %24
                          "w"(_coeff1), // %25
                          "r"(stepw)        // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmp[0];
                    t1 = tmp[1];
                    t2 = tmp[2];
                    t3 = tmp[3];

                    float* r0_tm0_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm1_0 = img0_tm.row(i * w_tm/8 + j + tiles*8);
                    float* r0_tm2_0 = img0_tm.row(i * w_tm/8 + j + tiles*16);
                    float* r0_tm3_0 = img0_tm.row(i * w_tm/8 + j + tiles*24);
                    float* r0_tm0_4 = img0_tm.row(i * w_tm/8 + j + tiles*32);
                    float* r0_tm1_4 = img0_tm.row(i * w_tm/8 + j + tiles*40);
                    float* r0_tm2_4 = img0_tm.row(i * w_tm/8 + j + tiles*48);
                    float* r0_tm3_4 = img0_tm.row(i * w_tm/8 + j + tiles*56);

                    int step = img0_tm.w*tiles*4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "add        %8, %8, #128        \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "add        %9, %9, #128        \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"
                        "add        %10, %10, #128      \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"
                        "add        %11, %11, #128      \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4[0]}, [%0], %26  \n"
                        "vst1.f32   {d4[1]}, [%1], %26  \n"

                        "vmov       q3, q7              \n"// use q7

                        "vst1.f32   {d5[0]}, [%2], %26  \n"
                        "vst1.f32   {d5[1]}, [%3], %26  \n"

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16[0]}, [%0], %26 \n"
                        "vst1.f32   {d16[1]}, [%1], %26 \n"

                        "vmla.f32   q4, q6, %e25[1]     \n"

                        "vst1.f32   {d17[0]}, [%2], %26 \n"
                        "vst1.f32   {d17[1]}, [%3], %26 \n"

                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18[0]}, [%0], %26 \n"
                        "vst1.f32   {d18[1]}, [%1], %26 \n"

                        "vadd.f32   q8, q2, q3          \n"

                        "vst1.f32   {d19[0]}, [%2], %26 \n"
                        "vst1.f32   {d19[1]}, [%3], %26 \n"

                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vst1.f32   {d16[0]}, [%0], %26 \n"
                        "vst1.f32   {d16[1]}, [%1], %26 \n"
                        "vst1.f32   {d17[0]}, [%2], %26 \n"
                        "vst1.f32   {d17[1]}, [%3], %26 \n"

                        "vadd.f32   q2, q4, q5          \n"

                        "vst1.f32   {d18[0]}, [%0], %26 \n"
                        "vst1.f32   {d18[1]}, [%1], %26 \n"
                        "vst1.f32   {d19[0]}, [%2], %26 \n"
                        "vst1.f32   {d19[1]}, [%3], %26 \n"

                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d4[0]}, [%0], %26  \n"
                        "vst1.f32   {d4[1]}, [%1], %26  \n"
                        "vst1.f32   {d5[0]}, [%2], %26  \n"
                        "vst1.f32   {d5[1]}, [%3], %26  \n"

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d6[0]}, [%0], %26  \n"
                        "vst1.f32   {d6[1]}, [%1], %26  \n"
                        "vst1.f32   {d7[0]}, [%2], %26  \n"
                        "vst1.f32   {d7[1]}, [%3], %26  \n"

                        "vst1.f32   {d12[0]}, [%0]      \n"
                        "vst1.f32   {d12[1]}, [%1]      \n"
                        "vst1.f32   {d13[0]}, [%2]      \n"
                        "vst1.f32   {d13[1]}, [%3]      \n"

                        // loop1
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "vld1.f32   {d24-d27}, [%10]    \n"

                        "vtrn.32    q8, q10             \n"

                        "vld1.f32   {d28-d31}, [%11]    \n"

                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

                        "vsub.f32   q2, q8, q13         \n"
                        "vsub.f32   q3, q9, q12         \n"

                        "vadd.f32   q4, q12, q13        \n"
                        "vadd.f32   q5, q10, q11        \n"

                        "vmla.f32   q2, q3, %f25[1]     \n"

                        "vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
                        "vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

                        "vmls.f32   q4, q9, %f25[0]     \n"
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        "vst1.f32   {d4[0]}, [%4], %26  \n"
                        "vst1.f32   {d4[1]}, [%5], %26  \n"

                        "vmov       q3, q7              \n"// use q7

                        "vst1.f32   {d5[0]}, [%6], %26  \n"
                        "vst1.f32   {d5[1]}, [%7], %26  \n"

                        "vadd.f32   q2, q13, q6         \n"// use q6
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        "vadd.f32   q8, q4, q5          \n"
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n"// use q7

                        "vadd.f32   q6, q12, q6         \n"// use q6
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n"

                        "vmla.f32   q2, q12, %e24[0]    \n"
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        "vst1.f32   {d16[0]}, [%4], %26 \n"
                        "vst1.f32   {d16[1]}, [%5], %26 \n"

                        "vmla.f32   q4, q6, %e25[1]     \n"

                        "vst1.f32   {d17[0]}, [%6], %26 \n"
                        "vst1.f32   {d17[1]}, [%7], %26 \n"

                        "vmla.f32   q5, q11, %e24[1]    \n"

                        "vst1.f32   {d18[0]}, [%4], %26 \n"
                        "vst1.f32   {d18[1]}, [%5], %26 \n"

                        "vadd.f32   q8, q2, q3          \n"

                        "vst1.f32   {d19[0]}, [%6], %26 \n"
                        "vst1.f32   {d19[1]}, [%7], %26 \n"

                        "vsub.f32   q9, q2, q3          \n"

                        "vsub.f32   q6, q15, q10        \n"
                        "vsub.f32   q7, q14, q11        \n"

                        "vst1.f32   {d16[0]}, [%4], %26 \n"
                        "vst1.f32   {d16[1]}, [%5], %26 \n"
                        "vst1.f32   {d17[0]}, [%6], %26 \n"
                        "vst1.f32   {d17[1]}, [%7], %26 \n"

                        "vadd.f32   q2, q4, q5          \n"

                        "vst1.f32   {d18[0]}, [%4], %26 \n"
                        "vst1.f32   {d18[1]}, [%5], %26 \n"
                        "vst1.f32   {d19[0]}, [%6], %26 \n"
                        "vst1.f32   {d19[1]}, [%7], %26 \n"

                        "vsub.f32   q3, q4, q5          \n"

                        "vst1.f32   {d4[0]}, [%4], %26  \n"
                        "vst1.f32   {d4[1]}, [%5], %26  \n"
                        "vst1.f32   {d5[0]}, [%6], %26  \n"
                        "vst1.f32   {d5[1]}, [%7], %26  \n"

                        "vmla.f32   q6, q7, %f25[1]     \n"

                        "vst1.f32   {d6[0]}, [%4], %26  \n"
                        "vst1.f32   {d6[1]}, [%5], %26  \n"
                        "vst1.f32   {d7[0]}, [%6], %26  \n"
                        "vst1.f32   {d7[1]}, [%7], %26  \n"

                        "vst1.f32   {d12[0]}, [%4]      \n"
                        "vst1.f32   {d12[1]}, [%5]      \n"
                        "vst1.f32   {d13[0]}, [%6]      \n"
                        "vst1.f32   {d13[1]}, [%7]      \n"

                        : "=r"(r0_tm0_0),     // %0
                          "=r"(r0_tm1_0),     // %1
                          "=r"(r0_tm2_0),     // %2
                          "=r"(r0_tm3_0),     // %3
                          "=r"(r0_tm0_4),     // %4
                          "=r"(r0_tm1_4),     // %5
                          "=r"(r0_tm2_4),     // %6
                          "=r"(r0_tm3_4),     // %7
                          "=r"(t0),     // %8
                          "=r"(t1),     // %9
                          "=r"(t2),     // %10
                          "=r"(t3)      // %11
                        : "0"(r0_tm0_0),
                          "1"(r0_tm1_0),
                          "2"(r0_tm2_0),
                          "3"(r0_tm3_0),
                          "4"(r0_tm0_4),
                          "5"(r0_tm1_4),
                          "6"(r0_tm2_4),
                          "7"(r0_tm3_4),
                          "8"(t0),
                          "9"(t1),
                          "10"(t2),
                          "11"(t3),
                          "w"(_coeff0), // %24
                          "w"(_coeff1), // %25
                          "r"(step)        // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    const float* r0 = img0.row(i * 6) + j * 6;

                    for (int m=0; m<8; m++)
                    {
                        tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                        tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                        float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
                        float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);

                        tmp[1][m] = tmp12a + tmp12b;
                        tmp[2][m] = tmp12a - tmp12b;

                        float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                        tmp[3][m] = tmp34a + tmp34b;
                        tmp[4][m] = tmp34a - tmp34b;

                        float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                        tmp[5][m] = tmp56a + tmp56b;
                        tmp[6][m] = tmp56a - tmp56b;

                        r0 += w;
                    }

                    float* r0_tm_0 = img0_tm.row(i * w_tm/8 + j);
                    float* r0_tm_1 = img0_tm.row(i * w_tm/8 + j + tiles);
                    float* r0_tm_2 = img0_tm.row(i * w_tm/8 + j + tiles*2);
                    float* r0_tm_3 = img0_tm.row(i * w_tm/8 + j + tiles*3);
                    float* r0_tm_4 = img0_tm.row(i * w_tm/8 + j + tiles*4);
                    float* r0_tm_5 = img0_tm.row(i * w_tm/8 + j + tiles*5);
                    float* r0_tm_6 = img0_tm.row(i * w_tm/8 + j + tiles*6);
                    float* r0_tm_7 = img0_tm.row(i * w_tm/8 + j + tiles*7);

                    for (int m=0; m<8; m++)
                    {
                        const float* tmp0 = tmp[m];

                        r0_tm_0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
                        r0_tm_7[0] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

                        float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                        float tmp12b = (tmp0[1] - tmp0[3] * 4.25f + tmp0[5]);

                        r0_tm_1[0] = tmp12a + tmp12b;
                        r0_tm_2[0] = tmp12a - tmp12b;

                        float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                        float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);

                        r0_tm_3[0] = tmp34a + tmp34b;
                        r0_tm_4[0] = tmp34a - tmp34b;

                        float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                        float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);

                        r0_tm_5[0] = tmp56a + tmp56b;
                        r0_tm_6[0] = tmp56a - tmp56b;

                        r0_tm_0 += img0_tm.w * tiles * 8;
                        r0_tm_1 += img0_tm.w * tiles * 8;
                        r0_tm_2 += img0_tm.w * tiles * 8;
                        r0_tm_3 += img0_tm.w * tiles * 8;
                        r0_tm_4 += img0_tm.w * tiles * 8;
                        r0_tm_5 += img0_tm.w * tiles * 8;
                        r0_tm_6 += img0_tm.w * tiles * 8;
                        r0_tm_7 += img0_tm.w * tiles * 8;
                    }
#endif // __ARM_NEON
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        // permute
        // bottom_blob_tm.create(1, 64 * tiles, inch);
//         Mat bottom_blob_tm2(inch, tiles, 64);
        Mat bottom_blob_tm2(8*inch, tiles/8 + (tiles%8)/4 + tiles%4, 64, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r=0; r<64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i=0;
            for (; i+7<tiles; i+=8)
            {
                float* tm2p = tm2.row(i/8);

                const float* r0 = bottom_blob_tm;

                r0 += r*tiles + i;

                for (int q=0; q<inch; q++)
                {
#if __ARM_NEON
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r0n = vld1q_f32(r0+4);
                    vst1q_f32(tm2p, _r0);
                    vst1q_f32(tm2p+4, _r0n);
#else
                    tm2p[0] = r0[0];
                    tm2p[1] = r0[1];
                    tm2p[2] = r0[2];
                    tm2p[3] = r0[3];
                    tm2p[4] = r0[4];
                    tm2p[5] = r0[5];
                    tm2p[6] = r0[6];
                    tm2p[7] = r0[7];
#endif // __ARM_NEON

                    r0 += bottom_blob_tm.cstep;
                    tm2p += 8;
                }
            }
            for (; i+3<tiles; i+=4)
            {
                float* tm2p = tm2.row(i/8+(i%8)/4);

                const float* r0 = bottom_blob_tm;

                r0 += r*tiles + i;

                for (int q=0; q<inch; q++)
                {
#if __ARM_NEON
                    float32x4_t _r0 = vld1q_f32(r0);
                    vst1q_f32(tm2p, _r0);
#else
                    tm2p[0] = r0[0];
                    tm2p[1] = r0[1];
                    tm2p[2] = r0[2];
                    tm2p[3] = r0[3];
#endif // __ARM_NEON

                    r0 += bottom_blob_tm.cstep;
                    tm2p += 4;
                }
            }
            for (; i<tiles; i++)
            {
                float* tm2p = tm2.row(i/8+(i%8)/4+i%4);

                const float* r0 = bottom_blob_tm;

                r0 += r*tiles + i;

                for (int q=0; q<inch; q++)
                {
                    tm2p[0] = r0[0];

                    r0 += bottom_blob_tm.cstep;
                    tm2p += 1;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(1, 64 * tiles, outch);

        int nn_outch = 0;
        int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
        nn_outch = outch >> 3;
        remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 8;

            const Mat kernel_tm0 = kernel_tm.channel(p/8);

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);
            Mat out4_tm = top_blob_tm.channel(p+4);
            Mat out5_tm = top_blob_tm.channel(p+5);
            Mat out6_tm = top_blob_tm.channel(p+6);
            Mat out7_tm = top_blob_tm.channel(p+7);

            float* output0_tm = out0_tm;
            float* output1_tm = out1_tm;
            float* output2_tm = out2_tm;
            float* output3_tm = out3_tm;
            float* output4_tm = out4_tm;
            float* output5_tm = out5_tm;
            float* output6_tm = out6_tm;
            float* output7_tm = out7_tm;

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                // tile
                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    const float* bb2p0 = bb2.row(i/8);

                    const float* ktm0 = kernel_tm0.row(r);

                    asm volatile(
                        "eor    v16.16b, v16.16b, v16.16b  \n"
                        "eor    v17.16b, v17.16b, v17.16b  \n"
                        "eor    v18.16b, v18.16b, v18.16b  \n"
                        "eor    v19.16b, v19.16b, v19.16b  \n"
                        "eor    v20.16b, v20.16b, v20.16b  \n"
                        "eor    v21.16b, v21.16b, v21.16b  \n"
                        "eor    v22.16b, v22.16b, v22.16b  \n"
                        "eor    v23.16b, v23.16b, v23.16b  \n"
                        "eor    v24.16b, v24.16b, v24.16b  \n"
                        "eor    v25.16b, v25.16b, v25.16b  \n"
                        "eor    v26.16b, v26.16b, v26.16b  \n"
                        "eor    v27.16b, v27.16b, v27.16b  \n"
                        "eor    v28.16b, v28.16b, v28.16b  \n"
                        "eor    v29.16b, v29.16b, v29.16b  \n"
                        "eor    v30.16b, v30.16b, v30.16b  \n"
                        "eor    v31.16b, v31.16b, v31.16b  \n"

                        // inch loop
                        "lsr    w4, %w20, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%8, #512]   \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]  \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

                        "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]  \n"
                        "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                        "fmla   v16.4s, v10.4s, v2.s[0] \n"
                        "fmla   v17.4s, v11.4s, v2.s[0] \n"
                        "fmla   v18.4s, v10.4s, v2.s[1] \n"
                        "fmla   v19.4s, v11.4s, v2.s[1] \n"
                        "fmla   v20.4s, v10.4s, v2.s[2] \n"
                        "fmla   v21.4s, v11.4s, v2.s[2] \n"
                        "fmla   v22.4s, v10.4s, v2.s[3] \n"
                        "fmla   v23.4s, v11.4s, v2.s[3] \n"

                        "prfm   pldl1keep, [%8, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"

                        "fmla   v24.4s, v10.4s, v3.s[0] \n"
                        "fmla   v25.4s, v11.4s, v3.s[0] \n"
                        "fmla   v26.4s, v10.4s, v3.s[1] \n"
                        "fmla   v27.4s, v11.4s, v3.s[1] \n"
                        "fmla   v28.4s, v10.4s, v3.s[2] \n"
                        "fmla   v29.4s, v11.4s, v3.s[2] \n"
                        "fmla   v30.4s, v10.4s, v3.s[3] \n"
                        "fmla   v31.4s, v11.4s, v3.s[3] \n"

                        "fmla   v16.4s, v12.4s, v4.s[0] \n"
                        "fmla   v17.4s, v13.4s, v4.s[0] \n"
                        "fmla   v18.4s, v12.4s, v4.s[1] \n"
                        "fmla   v19.4s, v13.4s, v4.s[1] \n"
                        "fmla   v20.4s, v12.4s, v4.s[2] \n"
                        "fmla   v21.4s, v13.4s, v4.s[2] \n"
                        "fmla   v22.4s, v12.4s, v4.s[3] \n"
                        "fmla   v23.4s, v13.4s, v4.s[3] \n"

                        "fmla   v24.4s, v12.4s, v5.s[0] \n"
                        "fmla   v25.4s, v13.4s, v5.s[0] \n"
                        "fmla   v26.4s, v12.4s, v5.s[1] \n"
                        "fmla   v27.4s, v13.4s, v5.s[1] \n"
                        "fmla   v28.4s, v12.4s, v5.s[2] \n"
                        "fmla   v29.4s, v13.4s, v5.s[2] \n"
                        "fmla   v30.4s, v12.4s, v5.s[3] \n"
                        "fmla   v31.4s, v13.4s, v5.s[3] \n"

                        "fmla   v16.4s, v14.4s, v6.s[0] \n"
                        "fmla   v17.4s, v15.4s, v6.s[0] \n"
                        "fmla   v18.4s, v14.4s, v6.s[1] \n"
                        "fmla   v19.4s, v15.4s, v6.s[1] \n"
                        "fmla   v20.4s, v14.4s, v6.s[2] \n"
                        "fmla   v21.4s, v15.4s, v6.s[2] \n"
                        "fmla   v22.4s, v14.4s, v6.s[3] \n"
                        "fmla   v23.4s, v15.4s, v6.s[3] \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v24.4s, v14.4s, v7.s[0] \n"
                        "fmla   v25.4s, v15.4s, v7.s[0] \n"
                        "fmla   v26.4s, v14.4s, v7.s[1] \n"
                        "fmla   v27.4s, v15.4s, v7.s[1] \n"
                        "fmla   v28.4s, v14.4s, v7.s[2] \n"
                        "fmla   v29.4s, v15.4s, v7.s[2] \n"
                        "fmla   v30.4s, v14.4s, v7.s[3] \n"
                        "fmla   v31.4s, v15.4s, v7.s[3] \n"

                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w20, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%8, #256]   \n"
                        "ld1    {v8.4s, v9.4s}, [%8], #32   \n"

                        "prfm   pldl1keep, [%9, #256]   \n"
                        "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                        "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                        "fmla   v22.4s, v8.4s, v0.s[3]  \n"
                        "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                        "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                        "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                        "fmla   v30.4s, v8.4s, v1.s[3]  \n"
                        "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v16.4s, v17.4s}, [%0], #32 \n"
                        "st1    {v18.4s, v19.4s}, [%1], #32 \n"
                        "st1    {v20.4s, v21.4s}, [%2], #32 \n"
                        "st1    {v22.4s, v23.4s}, [%3], #32 \n"
                        "st1    {v24.4s, v25.4s}, [%4], #32 \n"
                        "st1    {v26.4s, v27.4s}, [%5], #32 \n"
                        "st1    {v28.4s, v29.4s}, [%6], #32 \n"
                        "st1    {v30.4s, v31.4s}, [%7], #32 \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(output4_tm), // %4
                          "=r"(output5_tm), // %5
                          "=r"(output6_tm), // %6
                          "=r"(output7_tm), // %7
                          "=r"(bb2p0),      // %8
                          "=r"(ktm0)        // %9
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(output4_tm),
                          "5"(output5_tm),
                          "6"(output6_tm),
                          "7"(output7_tm),
                          "8"(bb2p0),
                          "9"(ktm0),
                          "r"(inch)         // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                    );
                }
                for (; i+3<tiles; i+=4)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4);

                    const float* ktm0 = kernel_tm0.row(r);

                    asm volatile(
                        "eor    v16.16b, v16.16b, v16.16b  \n"
                        "eor    v17.16b, v17.16b, v17.16b  \n"
                        "eor    v18.16b, v18.16b, v18.16b  \n"
                        "eor    v19.16b, v19.16b, v19.16b  \n"
                        "eor    v20.16b, v20.16b, v20.16b  \n"
                        "eor    v21.16b, v21.16b, v21.16b  \n"
                        "eor    v22.16b, v22.16b, v22.16b  \n"
                        "eor    v23.16b, v23.16b, v23.16b  \n"

                        // inch loop
                        "lsr    w4, %w20, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%8, #512]   \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64 \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v19.4s, v8.4s, v0.s[3]  \n"
                        "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                        "prfm   pldl1keep, [%9, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

                        "fmla   v16.4s, v9.4s, v2.s[0]  \n"
                        "fmla   v17.4s, v9.4s, v2.s[1]  \n"
                        "fmla   v18.4s, v9.4s, v2.s[2]  \n"
                        "fmla   v19.4s, v9.4s, v2.s[3]  \n"
                        "fmla   v20.4s, v9.4s, v3.s[0]  \n"
                        "fmla   v21.4s, v9.4s, v3.s[1]  \n"
                        "fmla   v22.4s, v9.4s, v3.s[2]  \n"
                        "fmla   v23.4s, v9.4s, v3.s[3]  \n"

                        "fmla   v16.4s, v10.4s, v4.s[0] \n"
                        "fmla   v17.4s, v10.4s, v4.s[1] \n"
                        "fmla   v18.4s, v10.4s, v4.s[2] \n"
                        "fmla   v19.4s, v10.4s, v4.s[3] \n"
                        "fmla   v20.4s, v10.4s, v5.s[0] \n"
                        "fmla   v21.4s, v10.4s, v5.s[1] \n"
                        "fmla   v22.4s, v10.4s, v5.s[2] \n"
                        "fmla   v23.4s, v10.4s, v5.s[3] \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v16.4s, v11.4s, v6.s[0] \n"
                        "fmla   v17.4s, v11.4s, v6.s[1] \n"
                        "fmla   v18.4s, v11.4s, v6.s[2] \n"
                        "fmla   v19.4s, v11.4s, v6.s[3] \n"
                        "fmla   v20.4s, v11.4s, v7.s[0] \n"
                        "fmla   v21.4s, v11.4s, v7.s[1] \n"
                        "fmla   v22.4s, v11.4s, v7.s[2] \n"
                        "fmla   v23.4s, v11.4s, v7.s[3] \n"

                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w20, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%8, #128]   \n"
                        "ld1    {v8.4s}, [%8], #16      \n"

                        "prfm   pldl1keep, [%9, #256]   \n"
                        "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                        "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                        "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                        "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                        "fmla   v19.4s, v8.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"

                        "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                        "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                        "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                        "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v16.4s}, [%0], #16     \n"
                        "st1    {v17.4s}, [%1], #16     \n"
                        "st1    {v18.4s}, [%2], #16     \n"
                        "st1    {v19.4s}, [%3], #16     \n"
                        "st1    {v20.4s}, [%4], #16     \n"
                        "st1    {v21.4s}, [%5], #16     \n"
                        "st1    {v22.4s}, [%6], #16     \n"
                        "st1    {v23.4s}, [%7], #16     \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(output4_tm), // %4
                          "=r"(output5_tm), // %5
                          "=r"(output6_tm), // %6
                          "=r"(output7_tm), // %7
                          "=r"(bb2p0),      // %8
                          "=r"(ktm0)        // %9
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(output4_tm),
                          "5"(output5_tm),
                          "6"(output6_tm),
                          "7"(output7_tm),
                          "8"(bb2p0),
                          "9"(ktm0),
                          "r"(inch)         // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    );
                }
                for (; i<tiles; i++)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4+i%4);

                    const float* ktm0 = kernel_tm0.row(r);

                    float32x4_t _sum0123 = vdupq_n_f32(0.f);
                    float32x4_t _sum4567 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q+3<inch; q+=4)
                    {
//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                        float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                        bb2p0 += 4;

//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                        float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
                        float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
                        float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
                        ktm0 += 16;

                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm1, _bb2p0, 0);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 1);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm3, _bb2p0, 1);

//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                        float32x4_t _ktm4 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm5 = vld1q_f32(ktm0 + 4);
                        float32x4_t _ktm6 = vld1q_f32(ktm0 + 8);
                        float32x4_t _ktm7 = vld1q_f32(ktm0 + 12);
                        ktm0 += 16;

                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm4, _bb2p0, 2);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm5, _bb2p0, 2);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm6, _bb2p0, 3);
                        _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm7, _bb2p0, 3);
                    }

                    for (; q<inch; q++)
                    {
                        float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
                        float32x4_t _ktm0123 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm4567 = vld1q_f32(ktm0 + 4);

                        _sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0123);
                        _sum4567 = vmlaq_f32(_sum4567, _bb2p0, _ktm4567);

                        bb2p0 += 1;
                        ktm0 += 8;
                    }

                    float sum0 = vgetq_lane_f32(_sum0123, 0);
                    float sum1 = vgetq_lane_f32(_sum0123, 1);
                    float sum2 = vgetq_lane_f32(_sum0123, 2);
                    float sum3 = vgetq_lane_f32(_sum0123, 3);
                    float sum4 = vgetq_lane_f32(_sum4567, 0);
                    float sum5 = vgetq_lane_f32(_sum4567, 1);
                    float sum6 = vgetq_lane_f32(_sum4567, 2);
                    float sum7 = vgetq_lane_f32(_sum4567, 3);

                    output0_tm[0] = sum0;
                    output1_tm[0] = sum1;
                    output2_tm[0] = sum2;
                    output3_tm[0] = sum3;
                    output4_tm[0] = sum4;
                    output5_tm[0] = sum5;
                    output6_tm[0] = sum6;
                    output7_tm[0] = sum7;

                    output0_tm += 1;
                    output1_tm += 1;
                    output2_tm += 1;
                    output3_tm += 1;
                    output4_tm += 1;
                    output5_tm += 1;
                    output6_tm += 1;
                    output7_tm += 1;
                }
            }
        }
#endif // __aarch64__

        nn_outch = (outch - remain_outch_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = remain_outch_start + pp * 4;

#if __ARM_NEON && __aarch64__
            const Mat kernel_tm0 = kernel_tm.channel(p/8+(p%8)/4);
#else
            const Mat kernel_tm0 = kernel_tm.channel(p/4);
#endif

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);

            float* output0_tm = out0_tm;
            float* output1_tm = out1_tm;
            float* output2_tm = out2_tm;
            float* output3_tm = out3_tm;

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                // tile
                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    const float* bb2p0 = bb2.row(i/8);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"
                        "eor    v9.16b, v9.16b, v9.16b     \n"
                        "eor    v10.16b, v10.16b, v10.16b  \n"
                        "eor    v11.16b, v11.16b, v11.16b  \n"
                        "eor    v12.16b, v12.16b, v12.16b  \n"
                        "eor    v13.16b, v13.16b, v13.16b  \n"
                        "eor    v14.16b, v14.16b, v14.16b  \n"
                        "eor    v15.16b, v15.16b, v15.16b  \n"

                        // inch loop
                        "lsr    w4, %w12, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]  \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v8.4s, v6.4s, v1.s[0]   \n"
                        "fmla   v9.4s, v7.4s, v1.s[0]   \n"
                        "fmla   v10.4s, v6.4s, v1.s[1]  \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]  \n"
                        "fmla   v12.4s, v6.4s, v1.s[2]  \n"
                        "fmla   v13.4s, v7.4s, v1.s[2]  \n"
                        "fmla   v14.4s, v6.4s, v1.s[3]  \n"
                        "fmla   v15.4s, v7.4s, v1.s[3]  \n"

                        "fmla   v8.4s, v16.4s, v2.s[0]  \n"
                        "fmla   v9.4s, v17.4s, v2.s[0]  \n"
                        "fmla   v10.4s, v16.4s, v2.s[1] \n"
                        "fmla   v11.4s, v17.4s, v2.s[1] \n"
                        "fmla   v12.4s, v16.4s, v2.s[2] \n"
                        "fmla   v13.4s, v17.4s, v2.s[2] \n"
                        "fmla   v14.4s, v16.4s, v2.s[3] \n"
                        "fmla   v15.4s, v17.4s, v2.s[3] \n"

                        "fmla   v8.4s, v18.4s, v3.s[0]  \n"
                        "fmla   v9.4s, v19.4s, v3.s[0]  \n"
                        "fmla   v10.4s, v18.4s, v3.s[1] \n"
                        "fmla   v11.4s, v19.4s, v3.s[1] \n"
                        "fmla   v12.4s, v18.4s, v3.s[2] \n"
                        "fmla   v13.4s, v19.4s, v3.s[2] \n"
                        "fmla   v14.4s, v18.4s, v3.s[3] \n"
                        "fmla   v15.4s, v19.4s, v3.s[3] \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w12, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%4, #256]   \n"
                        "ld1    {v4.4s, v5.4s}, [%4], #32      \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                        "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                        "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                        "fmla   v14.4s, v4.4s, v0.s[3]  \n"
                        "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s, v9.4s}, [%0], #32       \n"
                        "st1    {v10.4s, v11.4s}, [%1], #32     \n"
                        "st1    {v12.4s, v13.4s}, [%2], #32     \n"
                        "st1    {v14.4s, v15.4s}, [%3], #32     \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8      \n"
                        "veor       q9, q9, q9      \n"
                        "veor       q10, q10, q10   \n"
                        "veor       q11, q11, q11   \n"
                        "veor       q12, q12, q12   \n"
                        "veor       q13, q13, q13   \n"
                        "veor       q14, q14, q14   \n"
                        "veor       q15, q15, q15   \n"

                        // inch loop
                        "lsr        r4, %12, #2     \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0          \n"
                        "beq        1f              \n"

                        "0:                         \n"

                        "pld        [%4, #512]      \n"
                        "vldm       %4!, {d8-d15}   \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"
//                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
//                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q5, d0[0]   \n"
                        "vmla.f32   q10, q4, d0[1]  \n"
                        "vmla.f32   q11, q5, d0[1]  \n"
                        "vmla.f32   q12, q4, d1[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q4, d1[1]  \n"
                        "vmla.f32   q15, q5, d1[1]  \n"

                        "vmla.f32   q8, q6, d2[0]   \n"
                        "vmla.f32   q9, q7, d2[0]   \n"
                        "vmla.f32   q10, q6, d2[1]  \n"
                        "vmla.f32   q11, q7, d2[1]  \n"
                        "vmla.f32   q12, q6, d3[0]  \n"
                        "vmla.f32   q13, q7, d3[0]  \n"
                        "vmla.f32   q14, q6, d3[1]  \n"
                        "vmla.f32   q15, q7, d3[1]  \n"

                        "pld        [%4, #512]      \n"
                        "vldm       %4!, {d8-d15}   \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "vmla.f32   q8, q4, d4[0]   \n"
                        "vmla.f32   q9, q5, d4[0]   \n"
                        "vmla.f32   q10, q4, d4[1]  \n"
                        "vmla.f32   q11, q5, d4[1]  \n"
                        "vmla.f32   q12, q4, d5[0]  \n"
                        "vmla.f32   q13, q5, d5[0]  \n"
                        "vmla.f32   q14, q4, d5[1]  \n"
                        "vmla.f32   q15, q5, d5[1]  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q8, q6, d6[0]   \n"
                        "vmla.f32   q9, q7, d6[0]   \n"
                        "vmla.f32   q10, q6, d6[1]  \n"
                        "vmla.f32   q11, q7, d6[1]  \n"
                        "vmla.f32   q12, q6, d7[0]  \n"
                        "vmla.f32   q13, q7, d7[0]  \n"
                        "vmla.f32   q14, q6, d7[1]  \n"
                        "vmla.f32   q15, q7, d7[1]  \n"

                        "bne        0b              \n"

                        "1:                         \n"

                        // remain loop
                        "and        r4, %12, #3     \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0          \n"
                        "beq        3f              \n"

                        "2:                         \n"

                        "pld        [%4, #256]      \n"
                        "vld1.f32   {d8-d11}, [%4 :128]! \n"

                        "pld        [%5, #128]      \n"
                        "vld1.f32   {d0-d1}, [%5 :128]!  \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q5, d0[0]   \n"
                        "vmla.f32   q10, q4, d0[1]  \n"
                        "vmla.f32   q11, q5, d0[1]  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q12, q4, d1[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q4, d1[1]  \n"
                        "vmla.f32   q15, q5, d1[1]  \n"

                        "bne        2b              \n"

                        "3:                         \n"

                        "vst1.f32   {d16-d19}, [%0]! \n"
                        "vst1.f32   {d20-d23}, [%1]! \n"
                        "vst1.f32   {d24-d27}, [%2]! \n"
                        "vst1.f32   {d28-d31}, [%3]! \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    float sum0_0 = 0.f;
                    float sum0_1 = 0.f;
                    float sum0_2 = 0.f;
                    float sum0_3 = 0.f;
                    float sum0_4 = 0.f;
                    float sum0_5 = 0.f;
                    float sum0_6 = 0.f;
                    float sum0_7 = 0.f;

                    float sum1_0 = 0.f;
                    float sum1_1 = 0.f;
                    float sum1_2 = 0.f;
                    float sum1_3 = 0.f;
                    float sum1_4 = 0.f;
                    float sum1_5 = 0.f;
                    float sum1_6 = 0.f;
                    float sum1_7 = 0.f;

                    float sum2_0 = 0.f;
                    float sum2_1 = 0.f;
                    float sum2_2 = 0.f;
                    float sum2_3 = 0.f;
                    float sum2_4 = 0.f;
                    float sum2_5 = 0.f;
                    float sum2_6 = 0.f;
                    float sum2_7 = 0.f;

                    float sum3_0 = 0.f;
                    float sum3_1 = 0.f;
                    float sum3_2 = 0.f;
                    float sum3_3 = 0.f;
                    float sum3_4 = 0.f;
                    float sum3_5 = 0.f;
                    float sum3_6 = 0.f;
                    float sum3_7 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0_0 += bb2p0[0] * ktm0[0];
                        sum0_1 += bb2p0[1] * ktm0[0];
                        sum0_2 += bb2p0[2] * ktm0[0];
                        sum0_3 += bb2p0[3] * ktm0[0];
                        sum0_4 += bb2p0[4] * ktm0[0];
                        sum0_5 += bb2p0[5] * ktm0[0];
                        sum0_6 += bb2p0[6] * ktm0[0];
                        sum0_7 += bb2p0[7] * ktm0[0];

                        sum1_0 += bb2p0[0] * ktm0[1];
                        sum1_1 += bb2p0[1] * ktm0[1];
                        sum1_2 += bb2p0[2] * ktm0[1];
                        sum1_3 += bb2p0[3] * ktm0[1];
                        sum1_4 += bb2p0[4] * ktm0[1];
                        sum1_5 += bb2p0[5] * ktm0[1];
                        sum1_6 += bb2p0[6] * ktm0[1];
                        sum1_7 += bb2p0[7] * ktm0[1];

                        sum2_0 += bb2p0[0] * ktm0[2];
                        sum2_1 += bb2p0[1] * ktm0[2];
                        sum2_2 += bb2p0[2] * ktm0[2];
                        sum2_3 += bb2p0[3] * ktm0[2];
                        sum2_4 += bb2p0[4] * ktm0[2];
                        sum2_5 += bb2p0[5] * ktm0[2];
                        sum2_6 += bb2p0[6] * ktm0[2];
                        sum2_7 += bb2p0[7] * ktm0[2];

                        sum3_0 += bb2p0[0] * ktm0[3];
                        sum3_1 += bb2p0[1] * ktm0[3];
                        sum3_2 += bb2p0[2] * ktm0[3];
                        sum3_3 += bb2p0[3] * ktm0[3];
                        sum3_4 += bb2p0[4] * ktm0[3];
                        sum3_5 += bb2p0[5] * ktm0[3];
                        sum3_6 += bb2p0[6] * ktm0[3];
                        sum3_7 += bb2p0[7] * ktm0[3];

                        bb2p0 += 8;
                        ktm0 += 4;
                    }

                    output0_tm[0] = sum0_0;
                    output0_tm[1] = sum0_1;
                    output0_tm[2] = sum0_2;
                    output0_tm[3] = sum0_3;
                    output0_tm[4] = sum0_4;
                    output0_tm[5] = sum0_5;
                    output0_tm[6] = sum0_6;
                    output0_tm[7] = sum0_7;

                    output1_tm[0] = sum1_0;
                    output1_tm[1] = sum1_1;
                    output1_tm[2] = sum1_2;
                    output1_tm[3] = sum1_3;
                    output1_tm[4] = sum1_4;
                    output1_tm[5] = sum1_5;
                    output1_tm[6] = sum1_6;
                    output1_tm[7] = sum1_7;

                    output2_tm[0] = sum2_0;
                    output2_tm[1] = sum2_1;
                    output2_tm[2] = sum2_2;
                    output2_tm[3] = sum2_3;
                    output2_tm[4] = sum2_4;
                    output2_tm[5] = sum2_5;
                    output2_tm[6] = sum2_6;
                    output2_tm[7] = sum2_7;

                    output3_tm[0] = sum3_0;
                    output3_tm[1] = sum3_1;
                    output3_tm[2] = sum3_2;
                    output3_tm[3] = sum3_3;
                    output3_tm[4] = sum3_4;
                    output3_tm[5] = sum3_5;
                    output3_tm[6] = sum3_6;
                    output3_tm[7] = sum3_7;

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
#endif // __ARM_NEON
                }
                for (; i+3<tiles; i+=4)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"
                        "eor    v9.16b, v9.16b, v9.16b     \n"
                        "eor    v10.16b, v10.16b, v10.16b  \n"
                        "eor    v11.16b, v11.16b, v11.16b  \n"

                        // inch loop
                        "lsr    w4, %w12, #2            \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                        "prfm   pldl1keep, [%5, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                        "fmla   v8.4s, v5.4s, v1.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v1.s[1]   \n"
                        "fmla   v10.4s, v5.4s, v1.s[2]  \n"
                        "fmla   v11.4s, v5.4s, v1.s[3]  \n"

                        "fmla   v8.4s, v6.4s, v2.s[0]   \n"
                        "fmla   v9.4s, v6.4s, v2.s[1]   \n"
                        "fmla   v10.4s, v6.4s, v2.s[2]  \n"
                        "fmla   v11.4s, v6.4s, v2.s[3]  \n"

                        "fmla   v8.4s, v7.4s, v3.s[0]   \n"
                        "fmla   v9.4s, v7.4s, v3.s[1]   \n"
                        "fmla   v10.4s, v7.4s, v3.s[2]  \n"
                        "fmla   v11.4s, v7.4s, v3.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w12, #3            \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%4, #128]   \n"
                        "ld1    {v4.4s}, [%4], #16      \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                        "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                        "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s}, [%0], #16      \n"
                        "st1    {v9.4s}, [%1], #16      \n"
                        "st1    {v10.4s}, [%2], #16     \n"
                        "st1    {v11.4s}, [%3], #16     \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8      \n"
                        "veor       q9, q9, q9      \n"
                        "veor       q10, q10, q10   \n"
                        "veor       q11, q11, q11   \n"

                        // inch loop
                        "lsr        r4, %12, #2     \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0          \n"
                        "beq        1f              \n"

                        "0:                         \n"

                        "pld        [%4, #512]      \n"
                        "vldm       %4!, {d8-d15}   \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "pld        [%5, #512]      \n"
                        "vldm       %5!, {d0-d7}    \n"
//                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
//                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q4, d0[1]   \n"
                        "vmla.f32   q10, q4, d1[0]  \n"
                        "vmla.f32   q11, q4, d1[1]  \n"

                        "vmla.f32   q8, q5, d2[0]   \n"
                        "vmla.f32   q9, q5, d2[1]   \n"
                        "vmla.f32   q10, q5, d3[0]  \n"
                        "vmla.f32   q11, q5, d3[1]  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q8, q6, d4[0]   \n"
                        "vmla.f32   q9, q6, d4[1]   \n"
                        "vmla.f32   q10, q6, d5[0]  \n"
                        "vmla.f32   q11, q6, d5[1]  \n"

                        "vmla.f32   q8, q7, d6[0]   \n"
                        "vmla.f32   q9, q7, d6[1]   \n"
                        "vmla.f32   q10, q7, d7[0]  \n"
                        "vmla.f32   q11, q7, d7[1]  \n"

                        "bne        0b              \n"

                        "1:                         \n"

                        // remain loop
                        "and        r4, %12, #3     \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0          \n"
                        "beq        3f              \n"

                        "2:                         \n"

                        "pld        [%4, #128]      \n"
                        "vld1.f32   {d8-d9}, [%4 :128]!  \n"

                        "pld        [%5, #128]      \n"
                        "vld1.f32   {d0-d1}, [%5 :128]!  \n"

                        "subs       r4, r4, #1      \n"

                        "vmla.f32   q8, q4, d0[0]   \n"
                        "vmla.f32   q9, q4, d0[1]   \n"
                        "vmla.f32   q10, q4, d1[0]  \n"
                        "vmla.f32   q11, q4, d1[1]  \n"

                        "bne        2b              \n"

                        "3:                         \n"

                        "vst1.f32   {d16-d17}, [%0]! \n"
                        "vst1.f32   {d18-d19}, [%1]! \n"
                        "vst1.f32   {d20-d21}, [%2]! \n"
                        "vst1.f32   {d22-d23}, [%3]! \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(bb2p0),      // %4
                          "=r"(ktm0)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(bb2p0),
                          "5"(ktm0),
                          "r"(inch)         // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                    );
#endif // __aarch64__
#else
                    float sum0_0 = 0.f;
                    float sum0_1 = 0.f;
                    float sum0_2 = 0.f;
                    float sum0_3 = 0.f;

                    float sum1_0 = 0.f;
                    float sum1_1 = 0.f;
                    float sum1_2 = 0.f;
                    float sum1_3 = 0.f;

                    float sum2_0 = 0.f;
                    float sum2_1 = 0.f;
                    float sum2_2 = 0.f;
                    float sum2_3 = 0.f;

                    float sum3_0 = 0.f;
                    float sum3_1 = 0.f;
                    float sum3_2 = 0.f;
                    float sum3_3 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0_0 += bb2p0[0] * ktm0[0];
                        sum0_1 += bb2p0[1] * ktm0[0];
                        sum0_2 += bb2p0[2] * ktm0[0];
                        sum0_3 += bb2p0[3] * ktm0[0];

                        sum1_0 += bb2p0[0] * ktm0[1];
                        sum1_1 += bb2p0[1] * ktm0[1];
                        sum1_2 += bb2p0[2] * ktm0[1];
                        sum1_3 += bb2p0[3] * ktm0[1];

                        sum2_0 += bb2p0[0] * ktm0[2];
                        sum2_1 += bb2p0[1] * ktm0[2];
                        sum2_2 += bb2p0[2] * ktm0[2];
                        sum2_3 += bb2p0[3] * ktm0[2];

                        sum3_0 += bb2p0[0] * ktm0[3];
                        sum3_1 += bb2p0[1] * ktm0[3];
                        sum3_2 += bb2p0[2] * ktm0[3];
                        sum3_3 += bb2p0[3] * ktm0[3];

                        bb2p0 += 4;
                        ktm0 += 4;
                    }

                    output0_tm[0] = sum0_0;
                    output0_tm[1] = sum0_1;
                    output0_tm[2] = sum0_2;
                    output0_tm[3] = sum0_3;

                    output1_tm[0] = sum1_0;
                    output1_tm[1] = sum1_1;
                    output1_tm[2] = sum1_2;
                    output1_tm[3] = sum1_3;

                    output2_tm[0] = sum2_0;
                    output2_tm[1] = sum2_1;
                    output2_tm[2] = sum2_2;
                    output2_tm[3] = sum2_3;

                    output3_tm[0] = sum3_0;
                    output3_tm[1] = sum3_1;
                    output3_tm[2] = sum3_2;
                    output3_tm[3] = sum3_3;

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
#endif // __ARM_NEON
                }
                for (; i<tiles; i++)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4+i%4);

                    const float* ktm0 = kernel_tm0.row(r);

#if __ARM_NEON
                    float32x4_t _sum0123 = vdupq_n_f32(0.f);

                    int q=0;
                    for (; q+3<inch; q+=4)
                    {
//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                        float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                        bb2p0 += 4;

//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                        float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
                        float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
                        float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
                        float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
                        ktm0 += 16;

#if __aarch64__
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm1, _bb2p0, 1);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 2);
                        _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm3, _bb2p0, 3);
#else
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm0, vget_low_f32(_bb2p0), 0);
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm1, vget_low_f32(_bb2p0), 1);
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm2, vget_high_f32(_bb2p0), 0);
                        _sum0123 = vmlaq_lane_f32(_sum0123, _ktm3, vget_high_f32(_bb2p0), 1);
#endif // __aarch64__
                    }

                    for (; q<inch; q++)
                    {
                        float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
                        float32x4_t _ktm0 = vld1q_f32(ktm0);

                        _sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0);

                        bb2p0 += 1;
                        ktm0 += 4;
                    }

                    float sum0 = vgetq_lane_f32(_sum0123, 0);
                    float sum1 = vgetq_lane_f32(_sum0123, 1);
                    float sum2 = vgetq_lane_f32(_sum0123, 2);
                    float sum3 = vgetq_lane_f32(_sum0123, 3);
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];
                        sum1 += bb2p0[0] * ktm0[1];
                        sum2 += bb2p0[0] * ktm0[2];
                        sum3 += bb2p0[0] * ktm0[3];

                        bb2p0 += 1;
                        ktm0 += 4;
                    }
#endif // __ARM_NEON

                    output0_tm[0] = sum0;
                    output1_tm[0] = sum1;
                    output2_tm[0] = sum2;
                    output3_tm[0] = sum3;

                    output0_tm += 1;
                    output1_tm += 1;
                    output2_tm += 1;
                    output3_tm += 1;
                }
            }
        }

        remain_outch_start += nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=remain_outch_start; p<outch; p++)
        {
#if __ARM_NEON && __aarch64__
            const Mat kernel_tm0 = kernel_tm.channel(p/8+(p%8)/4+p%4);
#else
            const Mat kernel_tm0 = kernel_tm.channel(p/4+p%4);
#endif

            Mat out0_tm = top_blob_tm.channel(p);

            float* output0_tm = out0_tm;

            for (int r=0; r<64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                // tile
                int i=0;
                for (; i+7<tiles; i+=8)
                {
                    const float* bb2p0 = bb2.row(i/8);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"
                        "eor    v9.16b, v9.16b, v9.16b     \n"

                        // inch loop
                        "lsr    w4, %w6, #2             \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

                        "prfm   pldl1keep, [%2, #128]   \n"
                        "ld1    {v0.4s}, [%2], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                        "fmla   v8.4s, v6.4s, v0.s[1]   \n"
                        "fmla   v9.4s, v7.4s, v0.s[1]   \n"

                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

                        "fmla   v8.4s, v12.4s, v0.s[2]  \n"
                        "fmla   v9.4s, v13.4s, v0.s[2]  \n"
                        "fmla   v8.4s, v14.4s, v0.s[3]  \n"
                        "fmla   v9.4s, v15.4s, v0.s[3]  \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w6, #3             \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%1, #256]   \n"
                        "ld1    {v4.4s, v5.4s}, [%1], #32      \n"

                        "prfm   pldl1keep, [%2, #32]    \n"
                        "ld1r   {v0.4s}, [%2], #4       \n"

                        "fmla   v8.4s, v4.4s, v0.4s     \n"
                        "fmla   v9.4s, v5.4s, v0.4s     \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s, v9.4s}, [%0], #32       \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8          \n"
                        "veor       q9, q9, q9          \n"

                        // inch loop
                        "lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0              \n"
                        "beq        1f                  \n"

                        "0:                             \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d8-d15}       \n"
//                         "vld1.f32   {d8-d11}, [%1 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%1 :128]! \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%2 :128]! \n"

                        "vmla.f32   q8, q4, d0[0]       \n"
                        "vmla.f32   q9, q5, d0[0]       \n"
                        "vmla.f32   q8, q6, d0[1]       \n"
                        "vmla.f32   q9, q7, d0[1]       \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d24-d31}      \n"
//                         "vld1.f32   {d24-d27}, [%1 :128]! \n"
//                         "vld1.f32   {d28-d31}, [%1 :128]! \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q12, d1[0]      \n"
                        "vmla.f32   q9, q13, d1[0]      \n"
                        "vmla.f32   q8, q14, d1[1]      \n"
                        "vmla.f32   q9, q15, d1[1]      \n"

                        "bne        0b                  \n"

                        "1:                             \n"

                        // remain loop
                        "and        r4, %6, #3          \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0              \n"
                        "beq        3f                  \n"

                        "2:                             \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d8-d11}, [%1 :128]! \n"

                        "pld        [%2, #32]           \n"
                        "vld1.f32   {d0[],d1[]}, [%2]!  \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q4, q0          \n"
                        "vmla.f32   q9, q5, q0          \n"

                        "bne        2b                  \n"

                        "3:                             \n"

                        "vst1.f32   {d16-d19}, [%0]!    \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;
                    float sum4 = 0.f;
                    float sum5 = 0.f;
                    float sum6 = 0.f;
                    float sum7 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];
                        sum1 += bb2p0[1] * ktm0[0];
                        sum2 += bb2p0[2] * ktm0[0];
                        sum3 += bb2p0[3] * ktm0[0];
                        sum4 += bb2p0[4] * ktm0[0];
                        sum5 += bb2p0[5] * ktm0[0];
                        sum6 += bb2p0[6] * ktm0[0];
                        sum7 += bb2p0[7] * ktm0[0];

                        bb2p0 += 8;
                        ktm0 += 1;
                    }

                    output0_tm[0] = sum0;
                    output0_tm[1] = sum1;
                    output0_tm[2] = sum2;
                    output0_tm[3] = sum3;
                    output0_tm[4] = sum4;
                    output0_tm[5] = sum5;
                    output0_tm[6] = sum6;
                    output0_tm[7] = sum7;

                    output0_tm += 8;
#endif // __ARM_NEON
                }
                for (; i+3<tiles; i+=4)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4);

                    const float* ktm0 = kernel_tm0.row(r);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "eor    v8.16b, v8.16b, v8.16b     \n"

                        // inch loop
                        "lsr    w4, %w6, #2             \n"// w4 = nn = inch >> 2
                        "cmp    w4, #0                  \n"
                        "beq    1f                      \n"

                        "0:                             \n"

                        "prfm   pldl1keep, [%4, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                        "prfm   pldl1keep, [%5, #128]   \n"
                        "ld1    {v0.4s}, [%5], #16      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                        "fmla   v8.4s, v5.4s, v0.s[1]   \n"
                        "fmla   v8.4s, v6.4s, v0.s[2]   \n"
                        "fmla   v8.4s, v7.4s, v0.s[3]   \n"

                        "subs   w4, w4, #1              \n"
                        "bne    0b                      \n"

                        "1:                             \n"

                        // remain loop
                        "and    w4, %w6, #3             \n"// w4 = remain = tiles & 3;
                        "cmp    w4, #0                  \n"
                        "beq    3f                      \n"

                        "2:                             \n"

                        "prfm   pldl1keep, [%4, #128]   \n"
                        "ld1    {v4.4s}, [%4], #16      \n"

                        "prfm   pldl1keep, [%5, #32]    \n"
                        "ld1r   {v0.4s}, [%5], #4       \n"

                        "fmla   v8.4s, v4.4s, v0.4s     \n"

                        "subs   w4, w4, #1              \n"
                        "bne    2b                      \n"

                        "3:                             \n"

                        "st1    {v8.4s}, [%0], #16      \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8"
                    );
#else // __aarch64__
                    asm volatile(
                        "veor       q8, q8, q8          \n"

                        // inch loop
                        "lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
                        "cmp        r4, #0              \n"
                        "beq        1f                  \n"

                        "0:                             \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d8-d15}       \n"
//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d0-d1}, [%5 :128]! \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q4, d0[0]       \n"
                        "vmla.f32   q8, q5, d0[1]       \n"
                        "vmla.f32   q8, q6, d1[0]       \n"
                        "vmla.f32   q8, q7, d1[1]       \n"

                        "bne        0b                  \n"

                        "1:                             \n"

                        // remain loop
                        "and        r4, %6, #3          \n"// r4 = remain = tiles & 3;
                        "cmp        r4, #0              \n"
                        "beq        3f                  \n"

                        "2:                             \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d8-d9}, [%4]!      \n"

                        "pld        [%5, #32]           \n"
                        "vld1.f32   {d0[],d1[]}, [%5]!  \n"

                        "subs       r4, r4, #1          \n"

                        "vmla.f32   q8, q4, q0          \n"

                        "bne        2b                  \n"

                        "3:                             \n"

                        "vst1.f32   {d16-d17}, [%0]!    \n"

                        : "=r"(output0_tm), // %0
                          "=r"(bb2p0),      // %1
                          "=r"(ktm0)        // %2
                        : "0"(output0_tm),
                          "1"(bb2p0),
                          "2"(ktm0),
                          "r"(inch)         // %6
                        : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8"
                    );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;

                    for (int q=0; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];
                        sum1 += bb2p0[1] * ktm0[0];
                        sum2 += bb2p0[2] * ktm0[0];
                        sum3 += bb2p0[3] * ktm0[0];

                        bb2p0 += 4;
                        ktm0 += 1;
                    }

                    output0_tm[0] = sum0;
                    output0_tm[1] = sum1;
                    output0_tm[2] = sum2;
                    output0_tm[3] = sum3;

                    output0_tm += 4;
#endif // __ARM_NEON
                }
                for (; i<tiles; i++)
                {
                    const float* bb2p0 = bb2.row(i/8+(i%8)/4+i%4);

                    const float* ktm0 = kernel_tm0.row(r);

                    int q=0;
#if __ARM_NEON
                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    for (; q+3<inch; q+=4)
                    {
//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                        float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                        bb2p0 += 4;

                        float32x4_t _ktm0 = vld1q_f32(ktm0);
                        ktm0 += 4;

                        _sum0 = vmlaq_f32(_sum0, _bb2p0, _ktm0);
                    }

#if __aarch64__
                    float sum0 = vaddvq_f32(_sum0);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float sum0 = vget_lane_f32(vpadd_f32(_ss0, _ss0), 0);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
#endif
                    for (; q<inch; q++)
                    {
                        sum0 += bb2p0[0] * ktm0[0];

                        bb2p0 += 1;
                        ktm0 += 1;
                    }

                    output0_tm[0] = sum0;

                    output0_tm += 1;
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    {
//         const float otm[6][8] = {
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

#if __ARM_NEON
        const float coeff[4] = { 4.f, 8.f, 16.f, 32.f };
        float32x4_t _coeff = vld1q_f32(coeff);
#endif // __ARM_NEON

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm/8 * h_tm/8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p<outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;
#if __ARM_NEON
            float32x2_t _bias0 = vdup_n_f32(bias0);
#endif // __ARM_NEON

            float tmp[6][8];

            // tile
            for (int i=0; i<outh/6; i++)
            {
                for (int j=0; j<outw/6; j++)
                {
#if __ARM_NEON
#if __aarch64__
                    const float* output0_tm0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm1 = out0_tm.row(i * w_tm/8 + j + tiles*8);
                    const float* output0_tm2 = out0_tm.row(i * w_tm/8 + j + tiles*16);
                    const float* output0_tm3 = out0_tm.row(i * w_tm/8 + j + tiles*24);

                    for (int m=0; m+3<8; m+=4)
                    {
                        float32x4_t _output0_tm_00;
                        float32x4_t _output0_tm_11;
                        float32x4_t _output0_tm_22;
                        float32x4_t _output0_tm_33;
                        float32x4_t _output0_tm_44;
                        float32x4_t _output0_tm_55;
                        float32x4_t _output0_tm_66;
                        float32x4_t _output0_tm_77;

                        _output0_tm_00 = vsetq_lane_f32(output0_tm0[0], _output0_tm_00, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_00 = vsetq_lane_f32(output0_tm1[0], _output0_tm_00, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_00 = vsetq_lane_f32(output0_tm2[0], _output0_tm_00, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_00 = vsetq_lane_f32(output0_tm3[0], _output0_tm_00, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_11 = vsetq_lane_f32(output0_tm0[0], _output0_tm_11, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_11 = vsetq_lane_f32(output0_tm1[0], _output0_tm_11, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_11 = vsetq_lane_f32(output0_tm2[0], _output0_tm_11, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_11 = vsetq_lane_f32(output0_tm3[0], _output0_tm_11, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_22 = vsetq_lane_f32(output0_tm0[0], _output0_tm_22, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_22 = vsetq_lane_f32(output0_tm1[0], _output0_tm_22, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_22 = vsetq_lane_f32(output0_tm2[0], _output0_tm_22, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_22 = vsetq_lane_f32(output0_tm3[0], _output0_tm_22, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_33 = vsetq_lane_f32(output0_tm0[0], _output0_tm_33, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_33 = vsetq_lane_f32(output0_tm1[0], _output0_tm_33, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_33 = vsetq_lane_f32(output0_tm2[0], _output0_tm_33, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_33 = vsetq_lane_f32(output0_tm3[0], _output0_tm_33, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_44 = vsetq_lane_f32(output0_tm0[0], _output0_tm_44, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_44 = vsetq_lane_f32(output0_tm1[0], _output0_tm_44, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_44 = vsetq_lane_f32(output0_tm2[0], _output0_tm_44, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_44 = vsetq_lane_f32(output0_tm3[0], _output0_tm_44, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_55 = vsetq_lane_f32(output0_tm0[0], _output0_tm_55, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_55 = vsetq_lane_f32(output0_tm1[0], _output0_tm_55, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_55 = vsetq_lane_f32(output0_tm2[0], _output0_tm_55, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_55 = vsetq_lane_f32(output0_tm3[0], _output0_tm_55, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_66 = vsetq_lane_f32(output0_tm0[0], _output0_tm_66, 0);
                        output0_tm0 += out0_tm.w * tiles;
                        _output0_tm_66 = vsetq_lane_f32(output0_tm1[0], _output0_tm_66, 1);
                        output0_tm1 += out0_tm.w * tiles;
                        _output0_tm_66 = vsetq_lane_f32(output0_tm2[0], _output0_tm_66, 2);
                        output0_tm2 += out0_tm.w * tiles;
                        _output0_tm_66 = vsetq_lane_f32(output0_tm3[0], _output0_tm_66, 3);
                        output0_tm3 += out0_tm.w * tiles;

                        _output0_tm_77 = vsetq_lane_f32(output0_tm0[0], _output0_tm_77, 0);
                        _output0_tm_77 = vsetq_lane_f32(output0_tm1[0], _output0_tm_77, 1);
                        _output0_tm_77 = vsetq_lane_f32(output0_tm2[0], _output0_tm_77, 2);
                        _output0_tm_77 = vsetq_lane_f32(output0_tm3[0], _output0_tm_77, 3);

                        float32x4_t _tmp024a = vaddq_f32(_output0_tm_11, _output0_tm_22);
                        float32x4_t _tmp135a = vsubq_f32(_output0_tm_11, _output0_tm_22);

                        float32x4_t _tmp024b = vaddq_f32(_output0_tm_33, _output0_tm_44);
                        float32x4_t _tmp135b = vsubq_f32(_output0_tm_33, _output0_tm_44);

                        float32x4_t _tmp024c = vaddq_f32(_output0_tm_55, _output0_tm_66);
                        float32x4_t _tmp135c = vsubq_f32(_output0_tm_55, _output0_tm_66);

                        float32x4_t _tmp0 = vaddq_f32(_output0_tm_00, _tmp024a);
                        _tmp0 = vmlaq_lane_f32(_tmp0, _tmp024c, vget_high_f32(_coeff), 1);
                        _tmp0 = vaddq_f32(_tmp0, _tmp024b);

                        float32x4_t _tmp2 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _tmp2 = vmlaq_lane_f32(_tmp2, _tmp024c, vget_low_f32(_coeff), 1);

                        float32x4_t _tmp4 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);

                        vst1q_f32(&tmp[0][m], _tmp0);
                        vst1q_f32(&tmp[2][m], _tmp2);
                        vst1q_f32(&tmp[4][m], _tmp4);

                        float32x4_t _tmp1 = vmlaq_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);

                        float32x4_t _tmp3 = vmlaq_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        _tmp3 = vmlaq_lane_f32(_tmp3, _tmp135c, vget_low_f32(_coeff), 0);

                        float32x4_t _tmp5 = vaddq_f32(_output0_tm_77, _tmp135a);
                        _tmp5 = vmlaq_lane_f32(_tmp5, _tmp135b, vget_high_f32(_coeff), 1);
                        _tmp5 = vaddq_f32(_tmp5, _tmp135c);

                        vst1q_f32(&tmp[1][m], _tmp1);
                        vst1q_f32(&tmp[3][m], _tmp3);
                        vst1q_f32(&tmp[5][m], _tmp5);

                        output0_tm0 += out0_tm.w*tiles*25;
                        output0_tm1 += out0_tm.w*tiles*25;
                        output0_tm2 += out0_tm.w*tiles*25;
                        output0_tm3 += out0_tm.w*tiles*25;
                    }

                    const float* t0 = tmp[0];
                    const float* t1 = tmp[1];

                    float* output0 = out0.row(i * 6) + j * 6;
                    float* output1 = output0 + outw;

                    for (int m=0; m+1<6; m+=2)
                    {
                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0+4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1+4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);

                        float32x2_t _t_00 = vget_low_f32(_t01_00221133.val[0]);
                        float32x2_t _t_11 = vget_low_f32(_t01_00221133.val[1]);
                        float32x2_t _t_22 = vget_high_f32(_t01_00221133.val[0]);
                        float32x2_t _t_33 = vget_high_f32(_t01_00221133.val[1]);
                        float32x2_t _t_44 = vget_low_f32(_t01_44665577.val[0]);
                        float32x2_t _t_55 = vget_low_f32(_t01_44665577.val[1]);
                        float32x2_t _t_66 = vget_high_f32(_t01_44665577.val[0]);
                        float32x2_t _t_77 = vget_high_f32(_t01_44665577.val[1]);

                        float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);

                        float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);

                        float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);

                        float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        _output_0 = vadd_f32(_output_0, _tmp024b);
                        _output_0 = vadd_f32(_output_0, _bias0);

                        float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                        _output_2 = vadd_f32(_output_2, _bias0);

                        float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _bias0);

                        output0[0] = vget_lane_f32(_output_0, 0);
                        output1[0] = vget_lane_f32(_output_0, 1);
                        output0[2] = vget_lane_f32(_output_2, 0);
                        output1[2] = vget_lane_f32(_output_2, 1);
                        output0[4] = vget_lane_f32(_output_4, 0);
                        output1[4] = vget_lane_f32(_output_4, 1);

                        float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        _output_1 = vadd_f32(_output_1, _bias0);

                        float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                        _output_3 = vadd_f32(_output_3, _bias0);

                        float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        _output_5 = vadd_f32(_output_5, _tmp135c);
                        _output_5 = vadd_f32(_output_5, _bias0);

                        output0[1] = vget_lane_f32(_output_1, 0);
                        output1[1] = vget_lane_f32(_output_1, 1);
                        output0[3] = vget_lane_f32(_output_3, 0);
                        output1[3] = vget_lane_f32(_output_3, 1);
                        output0[5] = vget_lane_f32(_output_5, 0);
                        output1[5] = vget_lane_f32(_output_5, 1);

                        t0 += 8*2;
                        t1 += 8*2;
                        output0 += outw*2;
                        output1 += outw*2;
                    }
#else // __aarch64__
                    const float* output0_tm0_0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm1_0 = out0_tm.row(i * w_tm/8 + j + tiles*8);
                    const float* output0_tm2_0 = out0_tm.row(i * w_tm/8 + j + tiles*16);
                    const float* output0_tm3_0 = out0_tm.row(i * w_tm/8 + j + tiles*24);
                    const float* output0_tm0_4 = out0_tm.row(i * w_tm/8 + j + tiles*32);
                    const float* output0_tm1_4 = out0_tm.row(i * w_tm/8 + j + tiles*40);
                    const float* output0_tm2_4 = out0_tm.row(i * w_tm/8 + j + tiles*48);
                    const float* output0_tm3_4 = out0_tm.row(i * w_tm/8 + j + tiles*56);

                    float* t0 = tmp[0];
                    float* t1 = tmp[1];

//                     int step = out0_tm.w * tiles * 2*4 *4;
                    int step = out0_tm.w * tiles *4;

                    asm volatile(

                        // loop0
//                         "vld1.f32   {d16-d17}, [%2], %21 \n"
//                         "vld1.f32   {d18-d19}, [%3], %21 \n"
//                         "vld1.f32   {d20-d21}, [%4], %21 \n"
//                         "vld1.f32   {d22-d23}, [%5], %21 \n"
//                         "vld1.f32   {d24-d25}, [%6], %21 \n"
//                         "vld1.f32   {d26-d27}, [%7], %21 \n"
//                         "vld1.f32   {d28-d29}, [%8], %21 \n"
//                         "vld1.f32   {d30-d31}, [%9], %21 \n"

//                         "vtrn.32    q8, q10             \n"
//                         "vtrn.32    q9, q11             \n"
//                         "vtrn.32    q12, q14            \n"
//                         "vtrn.32    q13, q15            \n"

//                         "vswp       d17, d24            \n"
//                         "vswp       d19, d26            \n"
//                         "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
//                         "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        "vld1.f32   {d16[0]}, [%2], %21 \n"
                        "vld1.f32   {d16[1]}, [%3], %21 \n"
                        "vld1.f32   {d17[0]}, [%4], %21 \n"
                        "vld1.f32   {d17[1]}, [%5], %21 \n"

                        "vld1.f32   {d20[0]}, [%2], %21 \n"
                        "vld1.f32   {d20[1]}, [%3], %21 \n"
                        "vld1.f32   {d21[0]}, [%4], %21 \n"
                        "vld1.f32   {d21[1]}, [%5], %21 \n"

                        "vld1.f32   {d24[0]}, [%2], %21 \n"
                        "vld1.f32   {d24[1]}, [%3], %21 \n"
                        "vld1.f32   {d25[0]}, [%4], %21 \n"
                        "vld1.f32   {d25[1]}, [%5], %21 \n"

                        "vadd.f32   q2, q10, q12        \n"
                        "vsub.f32   q3, q10, q12        \n"

                        "vld1.f32   {d28[0]}, [%2], %21 \n"
                        "vld1.f32   {d28[1]}, [%3], %21 \n"
                        "vld1.f32   {d29[0]}, [%4], %21 \n"
                        "vld1.f32   {d29[1]}, [%5], %21 \n"

                        "vld1.f32   {d18[0]}, [%2], %21 \n"
                        "vld1.f32   {d18[1]}, [%3], %21 \n"
                        "vld1.f32   {d19[0]}, [%4], %21 \n"
                        "vld1.f32   {d19[1]}, [%5], %21 \n"

                        "vadd.f32   q4, q14, q9         \n"
                        "vsub.f32   q5, q14, q9         \n"

                        "vld1.f32   {d22[0]}, [%2], %21 \n"
                        "vld1.f32   {d22[1]}, [%3], %21 \n"
                        "vld1.f32   {d23[0]}, [%4], %21 \n"
                        "vld1.f32   {d23[1]}, [%5], %21 \n"

                        "vld1.f32   {d26[0]}, [%2], %21 \n"
                        "vld1.f32   {d26[1]}, [%3], %21 \n"
                        "vld1.f32   {d27[0]}, [%4], %21 \n"
                        "vld1.f32   {d27[1]}, [%5], %21 \n"

                        "vadd.f32   q6, q11, q13        \n"
                        "vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

                        "vld1.f32   {d30[0]}, [%2]      \n"
                        "vld1.f32   {d30[1]}, [%3]      \n"
                        "vld1.f32   {d31[0]}, [%4]      \n"
                        "vld1.f32   {d31[1]}, [%5]      \n"

                        "vmov       q9, q3              \n"
                        "vadd.f32   q8, q8, q2          \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vmov       q12, q2             \n"
                        "vmov       q10, q2             \n"
                        "vmov       q11, q3             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q8, q8, q4          \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q15, q15, q7        \n"

                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"
                        "sub        %0, %0, #112        \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"
                        "sub        %1, %1, #112        \n"

                        // loop1
//                         "vld1.f32   {d16-d17}, [%2]     \n"
//                         "vld1.f32   {d18-d19}, [%3]     \n"
//                         "vld1.f32   {d20-d21}, [%4]     \n"
//                         "vld1.f32   {d22-d23}, [%5]     \n"
//                         "vld1.f32   {d24-d25}, [%6]     \n"
//                         "vld1.f32   {d26-d27}, [%7]     \n"
//                         "vld1.f32   {d28-d29}, [%8]     \n"
//                         "vld1.f32   {d30-d31}, [%9]     \n"

//                         "vtrn.32    q8, q10             \n"
//                         "vtrn.32    q9, q11             \n"
//                         "vtrn.32    q12, q14            \n"
//                         "vtrn.32    q13, q15            \n"

//                         "vswp       d17, d24            \n"
//                         "vswp       d19, d26            \n"
//                         "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
//                         "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        "vld1.f32   {d16[0]}, [%6], %21 \n"
                        "vld1.f32   {d16[1]}, [%7], %21 \n"
                        "vld1.f32   {d17[0]}, [%8], %21 \n"
                        "vld1.f32   {d17[1]}, [%9], %21 \n"

                        "vld1.f32   {d20[0]}, [%6], %21 \n"
                        "vld1.f32   {d20[1]}, [%7], %21 \n"
                        "vld1.f32   {d21[0]}, [%8], %21 \n"
                        "vld1.f32   {d21[1]}, [%9], %21 \n"

                        "vld1.f32   {d24[0]}, [%6], %21 \n"
                        "vld1.f32   {d24[1]}, [%7], %21 \n"
                        "vld1.f32   {d25[0]}, [%8], %21 \n"
                        "vld1.f32   {d25[1]}, [%9], %21 \n"

                        "vadd.f32   q2, q10, q12        \n"
                        "vsub.f32   q3, q10, q12        \n"

                        "vld1.f32   {d28[0]}, [%6], %21 \n"
                        "vld1.f32   {d28[1]}, [%7], %21 \n"
                        "vld1.f32   {d29[0]}, [%8], %21 \n"
                        "vld1.f32   {d29[1]}, [%9], %21 \n"

                        "vld1.f32   {d18[0]}, [%6], %21 \n"
                        "vld1.f32   {d18[1]}, [%7], %21 \n"
                        "vld1.f32   {d19[0]}, [%8], %21 \n"
                        "vld1.f32   {d19[1]}, [%9], %21 \n"

                        "vadd.f32   q4, q14, q9         \n"
                        "vsub.f32   q5, q14, q9         \n"

                        "vld1.f32   {d22[0]}, [%6], %21 \n"
                        "vld1.f32   {d22[1]}, [%7], %21 \n"
                        "vld1.f32   {d23[0]}, [%8], %21 \n"
                        "vld1.f32   {d23[1]}, [%9], %21 \n"

                        "vld1.f32   {d26[0]}, [%6], %21 \n"
                        "vld1.f32   {d26[1]}, [%7], %21 \n"
                        "vld1.f32   {d27[0]}, [%8], %21 \n"
                        "vld1.f32   {d27[1]}, [%9], %21 \n"

                        "vadd.f32   q6, q11, q13        \n"
                        "vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

                        "vld1.f32   {d30[0]}, [%6]      \n"
                        "vld1.f32   {d30[1]}, [%7]      \n"
                        "vld1.f32   {d31[0]}, [%8]      \n"
                        "vld1.f32   {d31[1]}, [%9]      \n"

                        "vmov       q9, q3              \n"
                        "vadd.f32   q8, q8, q2          \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vmov       q12, q2             \n"
                        "vmov       q10, q2             \n"
                        "vmov       q11, q3             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q8, q8, q4          \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q15, q15, q7        \n"

                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"

                        : "=r"(t0),             // %0
                          "=r"(t1),             // %1
                          "=r"(output0_tm0_0),  // %2
                          "=r"(output0_tm1_0),  // %3
                          "=r"(output0_tm2_0),  // %4
                          "=r"(output0_tm3_0),  // %5
                          "=r"(output0_tm0_4),  // %6
                          "=r"(output0_tm1_4),  // %7
                          "=r"(output0_tm2_4),  // %8
                          "=r"(output0_tm3_4)   // %9
                        : "0"(t0),
                          "1"(t1),
                          "2"(output0_tm0_0),
                          "3"(output0_tm1_0),
                          "4"(output0_tm2_0),
                          "5"(output0_tm3_0),
                          "6"(output0_tm0_4),
                          "7"(output0_tm1_4),
                          "8"(output0_tm2_4),
                          "9"(output0_tm3_4),
                          "w"(_coeff),          // %20
                          "r"(step)             // %21
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmp[0];
                    t1 = tmp[1];

                    float* output0 = out0.row(i * 6) + j * 6;
                    float* output1 = output0 + outw;

                    int stepw = outw*2 * 4;

                    asm volatile(

                        // loop0
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        // loop1
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        // loop2
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        "vld1.f32   {d20-d23}, [%3]     \n"

                        "add        %2, %2, #64         \n"
                        "add        %3, %3, #64         \n"

                        "vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
                        "vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

                        "vadd.f32   d4, d20, d17        \n"
                        "vsub.f32   d5, d20, d17        \n"

                        "vadd.f32   d6, d21, d18        \n"
                        "vsub.f32   d7, d21, d18        \n"

                        "vadd.f32   d8, d22, d19        \n"
                        "vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

                        "vmov       d20, d5             \n"
                        "vmov       d18, d4             \n"

                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vmov       d17, d4             \n"
                        "vmov       d21, d5             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d22, d23, d5        \n"

                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"

                        "vadd.f32   d16, d16, d6        \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d22, d22, d9        \n"

                        "vadd.f32   d16, d16, %P9       \n"// _bias0
                        "vadd.f32   d20, d20, %P9       \n"// _bias0
                        "vadd.f32   d17, d17, %P9       \n"// _bias0
                        "vadd.f32   d21, d21, %P9       \n"// _bias0
                        "vadd.f32   d18, d18, %P9       \n"// _bias0
                        "vadd.f32   d22, d22, %P9       \n"// _bias0

                        "vtrn.f32   q8, q10             \n"
                        "vtrn.f32   d18, d22            \n"

                        "vst1.f32   {d16-d18}, [%0], %10 \n"
                        "vst1.f32   {d20-d22}, [%1], %10 \n"

                        : "=r"(output0),    // %0
                          "=r"(output1),    // %1
                          "=r"(t0),         // %2
                          "=r"(t1)          // %3
                        : "0"(output0),
                          "1"(output1),
                          "2"(t0),
                          "3"(t1),
                          "w"(_coeff),      // %8
                          "w"(_bias0),      // %9
                          "r"(stepw)        // %10
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else
                    const float* output0_tm_0 = out0_tm.row(i * w_tm/8 + j);
                    const float* output0_tm_1 = out0_tm.row(i * w_tm/8 + j + tiles);
                    const float* output0_tm_2 = out0_tm.row(i * w_tm/8 + j + tiles*2);
                    const float* output0_tm_3 = out0_tm.row(i * w_tm/8 + j + tiles*3);
                    const float* output0_tm_4 = out0_tm.row(i * w_tm/8 + j + tiles*4);
                    const float* output0_tm_5 = out0_tm.row(i * w_tm/8 + j + tiles*5);
                    const float* output0_tm_6 = out0_tm.row(i * w_tm/8 + j + tiles*6);
                    const float* output0_tm_7 = out0_tm.row(i * w_tm/8 + j + tiles*7);

                    for (int m=0; m<8; m++)
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

                        output0_tm_0 += out0_tm.w * tiles * 8;
                        output0_tm_1 += out0_tm.w * tiles * 8;
                        output0_tm_2 += out0_tm.w * tiles * 8;
                        output0_tm_3 += out0_tm.w * tiles * 8;
                        output0_tm_4 += out0_tm.w * tiles * 8;
                        output0_tm_5 += out0_tm.w * tiles * 8;
                        output0_tm_6 += out0_tm.w * tiles * 8;
                        output0_tm_7 += out0_tm.w * tiles * 8;
                    }

                    float* output0 = out0.row(i * 6) + j * 6;

                    for (int m=0; m<6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
#endif // __ARM_NEON
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt.blob_allocator, opt.num_threads);
}

static void conv3x3s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);

        const float* k0 = kernel + p*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

#if __ARM_NEON
            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0+3);
            float32x4_t _k06 = vld1q_f32(k0+6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1+3);
            float32x4_t _k16 = vld1q_f32(k1+6);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%3], #32   \n"// v8 v9 = r0

                    "0:                                 \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v6.4s}, [%1]               \n"// v6 = _sum0

                    "fmul   v12.4s, v8.4s, %12.s[0]     \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v7.4s}, [%2]               \n"// v7 = _sum1

                    "fmul   v13.4s, v8.4s, %15.s[0]     \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld2    {v10.4s, v11.4s}, [%3]      \n"// v10

                    "fmla   v6.4s, v9.4s, %12.s[1]      \n"

                    "ext    v14.16b, v8.16b, v10.16b, #4\n"

                    "fmla   v7.4s, v9.4s, %15.s[1]      \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%4], #32   \n"// r1

                    "fmla   v12.4s, v14.4s, %12.s[2]    \n"
                    "fmla   v13.4s, v14.4s, %15.s[2]    \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld2    {v10.4s, v11.4s}, [%4]      \n"

                    "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                    "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                    "ext    v14.16b, v8.16b, v10.16b, #4\n"

                    "fmla   v12.4s, v9.4s, %13.s[1]     \n"
                    "fmla   v13.4s, v9.4s, %16.s[1]     \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%5], #32   \n"// r2

                    "fmla   v6.4s, v14.4s, %13.s[2]     \n"
                    "fmla   v7.4s, v14.4s, %16.s[2]     \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld2    {v10.4s, v11.4s}, [%5]      \n"

                    "fmla   v12.4s, v8.4s, %14.s[0]     \n"
                    "fmla   v13.4s, v8.4s, %17.s[0]     \n"

                    "ext    v14.16b, v8.16b, v10.16b, #4\n"

                    "fmla   v6.4s, v9.4s, %14.s[1]      \n"
                    "fmla   v7.4s, v9.4s, %17.s[1]      \n"

                    "fmla   v12.4s, v14.4s, %14.s[2]    \n"
                    "fmla   v13.4s, v14.4s, %17.s[2]    \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld2    {v8.4s, v9.4s}, [%3], #32   \n"// v8 v9 = r0

                    "fadd   v6.4s, v6.4s, v12.4s        \n"
                    "fadd   v7.4s, v7.4s, v13.4s        \n"

                    "subs   %w0, %w0, #1                \n"

                    "st1    {v6.4s}, [%1], #16          \n"
                    "st1    {v7.4s}, [%2], #16          \n"

                    "bne    0b                          \n"
                    "sub    %3, %3, #32                 \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2)          // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "w"(_k00),      // %12
                      "w"(_k03),      // %13
                      "w"(_k06),      // %14
                      "w"(_k10),      // %15
                      "w"(_k13),      // %16
                      "w"(_k16)       // %17
                    : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d16-d19}, [%3]!    \n"// q8 q9 = r0

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d12-d13}, [%1]     \n"// q6 = _sum0

                    "vmul.f32   q12, q8, %e12[0]    \n"

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d14-d15}, [%2]     \n"// q7 = _sum1

                    "vmul.f32   q13, q8, %e15[0]    \n"

                    "pld        [%3, #128]          \n"
                    "vld2.f32   {d20-d21}, [%3]     \n"// q10

                    "vmla.f32   q6, q9, %e12[1]     \n"

                    "vext.32    q11, q8, q10, #1    \n"

                    "vmla.f32   q7, q9, %e15[1]     \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d16-d19}, [%4]!    \n"// r1

                    "vmla.f32   q12, q11, %f12[0]   \n"
                    "vmla.f32   q13, q11, %f15[0]   \n"

                    "pld        [%4, #128]          \n"
                    "vld2.f32   {d20-d21}, [%4]     \n"

                    "vmla.f32   q6, q8, %e13[0]     \n"
                    "vmla.f32   q7, q8, %e16[0]     \n"

                    "vext.32    q11, q8, q10, #1    \n"

                    "vmla.f32   q12, q9, %e13[1]    \n"
                    "vmla.f32   q13, q9, %e16[1]    \n"

                    "pld        [%5, #256]          \n"
                    "vld2.f32   {d16-d19}, [%5]!    \n"// r2

                    "vmla.f32   q6, q11, %f13[0]    \n"
                    "vmla.f32   q7, q11, %f16[0]    \n"

                    "pld        [%5, #128]          \n"
                    "vld2.f32   {d20-d21}, [%5]     \n"

                    "vmla.f32   q12, q8, %e14[0]    \n"
                    "vmla.f32   q13, q8, %e17[0]    \n"

                    "vext.32    q11, q8, q10, #1    \n"

                    "vmla.f32   q6, q9, %e14[1]     \n"
                    "vmla.f32   q7, q9, %e17[1]     \n"

                    "vmla.f32   q12, q11, %f14[0]   \n"
                    "vmla.f32   q13, q11, %f17[0]   \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d16-d19}, [%3]!    \n"// q8 q9 = r0

                    "vadd.f32   q6, q6, q12         \n"
                    "vadd.f32   q7, q7, q13         \n"

                    "subs       %0, #1              \n"

                    "vst1.f32   {d12-d13}, [%1]!    \n"
                    "vst1.f32   {d14-d15}, [%2]!    \n"

                    "bne        0b                  \n"
                    "sub        %3, #32             \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2)          // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "w"(_k00),      // %12
                      "w"(_k03),      // %13
                      "w"(_k06),      // %14
                      "w"(_k10),      // %15
                      "w"(_k13),      // %16
                      "w"(_k16)       // %17
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* kernel0 = kernel + p*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                    "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                    "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%3]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                    "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%4]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                    "fadd       v0.4s, v0.4s, v10.4s           \n"
                    "fadd       v0.4s, v0.4s, v11.4s           \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s}, [%1], #16             \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]       \n"

                    "vmla.f32   q0, q2, %e10[0]     \n"
                    "vmul.f32   q10, q3, %e10[1]    \n"

                    "pld        [%2, #128]          \n"
                    "vld2.f32   {d16-d17}, [%2]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmul.f32   q11, q1, %f10[0]    \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"

                    "vmla.f32   q0, q2, %e11[0]     \n"
                    "vmla.f32   q10, q3, %e11[1]    \n"

                    "pld        [%3, #128]          \n"
                    "vld2.f32   {d16-d17}, [%3]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f11[0]    \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"

                    "vmla.f32   q0, q2, %e12[0]     \n"
                    "vmla.f32   q10, q3, %e12[1]    \n"

                    "pld        [%4, #128]          \n"
                    "vld2.f32   {d16-d17}, [%4]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f12[0]    \n"

                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vadd.f32   q0, q0, q10         \n"
                    "vadd.f32   q0, q0, q11         \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}

static void conv3x3s2_transform_kernel_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*9, inch, outch/8 + outch%8);

    const float* kernel = _kernel;

    int p=0;
    for (; p+7<outch; p+=8)
    {
        const float* k0 = kernel + (p+0)*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;
        const float* k2 = kernel + (p+2)*inch*9;
        const float* k3 = kernel + (p+3)*inch*9;
        const float* k4 = kernel + (p+4)*inch*9;
        const float* k5 = kernel + (p+5)*inch*9;
        const float* k6 = kernel + (p+6)*inch*9;
        const float* k7 = kernel + (p+7)*inch*9;

        float* ktmp = kernel_tm.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p<outch; p++)
    {
        const float* k0 = kernel + (p+0)*inch*9;

        float* ktmp = kernel_tm.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

static void conv3x3s2_packed_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

//     const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p+0);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);
        Mat out4 = top_blob.channel(p+4);
        Mat out5 = top_blob.channel(p+5);
        Mat out6 = top_blob.channel(p+6);
        Mat out7 = top_blob.channel(p+7);

        const float bias0 = bias ? bias[p+0] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;
        const float bias2 = bias ? bias[p+2] : 0.f;
        const float bias3 = bias ? bias[p+3] : 0.f;
        const float bias4 = bias ? bias[p+4] : 0.f;
        const float bias5 = bias ? bias[p+5] : 0.f;
        const float bias6 = bias ? bias[p+6] : 0.f;
        const float bias7 = bias ? bias[p+7] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);
        out4.fill(bias4);
        out5.fill(bias5);
        out6.fill(bias6);
        out7.fill(bias7);

        const float* ktmp = _kernel.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;
            float* outptr6 = out6;
            float* outptr7 = out7;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                                 \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v8.4s}, [%1]               \n"
                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v9.4s}, [%2]               \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v10.4s}, [%3]              \n"
                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v11.4s}, [%4]              \n"

                    ///
                    "prfm   pldl1keep, [%9, #256]       \n"
                    "ld2    {v4.4s, v5.4s}, [%9], #32   \n"// v4=00 v5=01

                    "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v12.4s}, [%5]              \n"
                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v13.4s}, [%6]              \n"

                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v14.4s}, [%7]              \n"
                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v15.4s}, [%8]              \n"

                    "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"

                    "prfm   pldl1keep, [%9, #256]       \n"
                    "ld2    {v6.4s, v7.4s}, [%9]        \n"// v6

                    "fmla   v8.4s, v5.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v5.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v5.4s, v2.s[3]      \n"

                    "ext    v6.16b, v4.16b, v6.16b, #4  \n"// v6=02

                    "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v5.4s, v3.s[3]      \n"

                    ///
                    "prfm   pldl1keep, [%10, #256]      \n"
                    "ld2    {v4.4s, v5.4s}, [%10], #32  \n"// v4=10 v5=11

                    "fmla   v8.4s, v6.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v0.s[3]      \n"

                    "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v1.s[3]      \n"

                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"

                    "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%10, #256]      \n"
                    "ld2    {v6.4s, v7.4s}, [%10]       \n"// v6

                    "fmla   v8.4s, v5.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v5.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v5.4s, v0.s[3]      \n"

                    "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                    "ext    v6.16b, v4.16b, v6.16b, #4  \n"// v6=12

                    "fmla   v12.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v5.4s, v1.s[3]      \n"

                    ///
                    "prfm   pldl1keep, [%11, #256]      \n"
                    "ld2    {v4.4s, v5.4s}, [%11], #32  \n"// v4=20 v5=21

                    "fmla   v8.4s, v6.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v2.s[3]      \n"

                    "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v6.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v3.s[3]      \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"

                    "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"

                    "prfm   pldl1keep, [%11, #256]      \n"
                    "ld2    {v6.4s, v7.4s}, [%11]       \n"// v6

                    "fmla   v8.4s, v5.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v5.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v5.4s, v2.s[3]      \n"

                    "ext    v6.16b, v4.16b, v6.16b, #4  \n"// v6=22

                    "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                    "fmla   v12.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v5.4s, v3.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v0.s[3]      \n"

                    "fmla   v12.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v1.s[1]      \n"

                    "st1    {v8.4s}, [%1], #16          \n"
                    "st1    {v9.4s}, [%2], #16          \n"

                    "fmla   v14.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v1.s[3]      \n"

                    "st1    {v10.4s}, [%3], #16         \n"
                    "st1    {v11.4s}, [%4], #16         \n"

                    "sub    %12, %12, #288              \n"

                    "st1    {v12.4s}, [%5], #16         \n"
                    "st1    {v13.4s}, [%6], #16         \n"

                    "subs   %w0, %w0, #1                \n"

                    "st1    {v14.4s}, [%7], #16         \n"
                    "st1    {v15.4s}, [%8], #16         \n"

                    "bne    0b                          \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(outptr4),    // %5
                      "=r"(outptr5),    // %6
                      "=r"(outptr6),    // %7
                      "=r"(outptr7),    // %8
                      "=r"(r0),         // %9
                      "=r"(r1),         // %10
                      "=r"(r2),         // %11
                      "=r"(ktmp)        // %12
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(outptr4),
                      "6"(outptr5),
                      "7"(outptr6),
                      "8"(outptr7),
                      "9"(r0),
                      "10"(r1),
                      "11"(r2),
                      "12"(ktmp)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else // __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d16-d17}, [%1]     \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d18-d19}, [%2]     \n"

                    "pld        [%3, #128]          \n"
                    "vld1.f32   {d20-d21}, [%3]     \n"
                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d22-d23}, [%4]     \n"

                    ///
                    "pld        [%9, #256]          \n"
                    "vld2.f32   {d8-d11}, [%9]!     \n"// q4=00 q5=01

                    "vld1.f32   {d0-d3}, [%12 :128]! \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q9, q4, d0[1]       \n"

                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d24-d25}, [%5]     \n"
                    "pld        [%6, #128]          \n"
                    "vld1.f32   {d26-d27}, [%6]     \n"

                    "vmla.f32   q10, q4, d1[0]      \n"
                    "vmla.f32   q11, q4, d1[1]      \n"

                    "pld        [%7, #128]          \n"
                    "vld1.f32   {d28-d29}, [%7]     \n"
                    "pld        [%8, #128]          \n"
                    "vld1.f32   {d30-d31}, [%8]     \n"

                    "vld1.f32   {d4-d7}, [%12 :128]! \n"

                    "vmla.f32   q12, q4, d2[0]      \n"
                    "vmla.f32   q13, q4, d2[1]      \n"
                    "vmla.f32   q14, q4, d3[0]      \n"
                    "vmla.f32   q15, q4, d3[1]      \n"

                    "pld        [%9, #128]          \n"
                    "vld2.f32   {d12-d13}, [%9]     \n"// q6

                    "vmla.f32   q8, q5, d4[0]       \n"
                    "vmla.f32   q9, q5, d4[1]       \n"
                    "vmla.f32   q10, q5, d5[0]      \n"
                    "vmla.f32   q11, q5, d5[1]      \n"

                    "vext.f32   q6, q4, q6, #1      \n"// q6=02

                    "vld1.f32   {d0-d3}, [%12 :128]! \n"

                    "vmla.f32   q12, q5, d6[0]      \n"
                    "vmla.f32   q13, q5, d6[1]      \n"
                    "vmla.f32   q14, q5, d7[0]      \n"
                    "vmla.f32   q15, q5, d7[1]      \n"

                    ///
                    "pld        [%10, #256]         \n"
                    "vld2.f32   {d8-d11}, [%10]!    \n"// q4=10 q5=11

                    "vmla.f32   q8, q6, d0[0]       \n"
                    "vmla.f32   q9, q6, d0[1]       \n"
                    "vmla.f32   q10, q6, d1[0]      \n"
                    "vmla.f32   q11, q6, d1[1]      \n"

                    "vld1.f32   {d4-d7}, [%12 :128]! \n"

                    "vmla.f32   q12, q6, d2[0]      \n"
                    "vmla.f32   q13, q6, d2[1]      \n"
                    "vmla.f32   q14, q6, d3[0]      \n"
                    "vmla.f32   q15, q6, d3[1]      \n"

                    "vmla.f32   q8, q4, d4[0]       \n"
                    "vmla.f32   q9, q4, d4[1]       \n"
                    "vmla.f32   q10, q4, d5[0]      \n"
                    "vmla.f32   q11, q4, d5[1]      \n"

                    "vld1.f32   {d0-d3}, [%12 :128]! \n"

                    "vmla.f32   q12, q4, d6[0]      \n"
                    "vmla.f32   q13, q4, d6[1]      \n"
                    "vmla.f32   q14, q4, d7[0]      \n"
                    "vmla.f32   q15, q4, d7[1]      \n"

                    "pld        [%10, #128]         \n"
                    "vld2.f32   {d12-d13}, [%10]    \n"// q6

                    "vmla.f32   q8, q5, d0[0]       \n"
                    "vmla.f32   q9, q5, d0[1]       \n"
                    "vmla.f32   q10, q5, d1[0]      \n"
                    "vmla.f32   q11, q5, d1[1]      \n"

                    "vld1.f32   {d4-d7}, [%12 :128]! \n"

                    "vext.f32   q6, q4, q6, #1      \n"// q6=12

                    "vmla.f32   q12, q5, d2[0]      \n"
                    "vmla.f32   q13, q5, d2[1]      \n"
                    "vmla.f32   q14, q5, d3[0]      \n"
                    "vmla.f32   q15, q5, d3[1]      \n"

                    ///
                    "pld        [%11, #256]         \n"
                    "vld2.f32   {d8-d11}, [%11]!    \n"// q4=20 q5=21

                    "vmla.f32   q8, q6, d4[0]       \n"
                    "vmla.f32   q9, q6, d4[1]       \n"
                    "vmla.f32   q10, q6, d5[0]      \n"
                    "vmla.f32   q11, q6, d5[1]      \n"

                    "vld1.f32   {d0-d3}, [%12 :128]! \n"

                    "vmla.f32   q12, q6, d6[0]      \n"
                    "vmla.f32   q13, q6, d6[1]      \n"
                    "vmla.f32   q14, q6, d7[0]      \n"
                    "vmla.f32   q15, q6, d7[1]      \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q9, q4, d0[1]       \n"
                    "vmla.f32   q10, q4, d1[0]      \n"
                    "vmla.f32   q11, q4, d1[1]      \n"

                    "vld1.f32   {d4-d7}, [%12 :128]! \n"

                    "vmla.f32   q12, q4, d2[0]      \n"
                    "vmla.f32   q13, q4, d2[1]      \n"
                    "vmla.f32   q14, q4, d3[0]      \n"
                    "vmla.f32   q15, q4, d3[1]      \n"

                    "pld        [%11, #128]         \n"
                    "vld2.f32   {d12-d13}, [%11]    \n"// q6

                    "vmla.f32   q8, q5, d4[0]       \n"
                    "vmla.f32   q9, q5, d4[1]       \n"
                    "vmla.f32   q10, q5, d5[0]      \n"
                    "vmla.f32   q11, q5, d5[1]      \n"

                    "vext.f32   q6, q4, q6, #1      \n"// q6=22

                    "vld1.f32   {d0-d3}, [%12 :128]! \n"

                    "vmla.f32   q12, q5, d6[0]      \n"
                    "vmla.f32   q13, q5, d6[1]      \n"
                    "vmla.f32   q14, q5, d7[0]      \n"
                    "vmla.f32   q15, q5, d7[1]      \n"

                    "vmla.f32   q8, q6, d0[0]       \n"
                    "vmla.f32   q9, q6, d0[1]       \n"
                    "vmla.f32   q10, q6, d1[0]      \n"
                    "vmla.f32   q11, q6, d1[1]      \n"

                    "vmla.f32   q12, q6, d2[0]      \n"
                    "vmla.f32   q13, q6, d2[1]      \n"

                    "vst1.f32   {d16-d17}, [%1]!    \n"
                    "vst1.f32   {d18-d19}, [%2]!    \n"

                    "vmla.f32   q14, q6, d3[0]      \n"
                    "vmla.f32   q15, q6, d3[1]      \n"

                    "vst1.f32   {d20-d21}, [%3]!    \n"
                    "vst1.f32   {d22-d23}, [%4]!    \n"

                    "sub        %12, %12, #288      \n"

                    "vst1.f32   {d24-d25}, [%5]!    \n"
                    "vst1.f32   {d26-d27}, [%6]!    \n"

                    "subs       %0, #1              \n"

                    "vst1.f32   {d28-d29}, [%7]!    \n"
                    "vst1.f32   {d30-d31}, [%8]!    \n"

                    "bne        0b                  \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(outptr4),    // %5
                      "=r"(outptr5),    // %6
                      "=r"(outptr6),    // %7
                      "=r"(outptr7),    // %8
                      "=r"(r0),         // %9
                      "=r"(r1),         // %10
                      "=r"(r2),         // %11
                      "=r"(ktmp)        // %12
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(outptr4),
                      "6"(outptr5),
                      "7"(outptr6),
                      "8"(outptr7),
                      "9"(r0),
                      "10"(r1),
                      "11"(r2),
                      "12"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "prfm   pldl1keep, [%8, #128]   \n"
                        "ld1    {v0.4s}, [%8]           \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "ld1    {v8.s}[0], [%0]         \n"
                        "ld1    {v8.s}[1], [%1]         \n"
                        "ld1    {v8.s}[2], [%2]         \n"
                        "ld1    {v8.s}[3], [%3]         \n"

                        "fmul   v14.4s, v10.4s, v0.s[0] \n"
                        "fmul   v15.4s, v11.4s, v0.s[0] \n"

                        "ld1    {v9.s}[0], [%4]         \n"
                        "ld1    {v9.s}[1], [%5]         \n"
                        "ld1    {v9.s}[2], [%6]         \n"
                        "ld1    {v9.s}[3], [%7]         \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v0.s[1]  \n"
                        "fmla   v9.4s, v13.4s, v0.s[1]  \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "fmla   v14.4s, v10.4s, v0.s[2] \n"
                        "fmla   v15.4s, v11.4s, v0.s[2] \n"

                        "prfm   pldl1keep, [%9, #128]   \n"
                        "ld1    {v1.4s}, [%9]           \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v1.s[0]  \n"
                        "fmla   v9.4s, v13.4s, v1.s[0]  \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "fmla   v14.4s, v10.4s, v1.s[1] \n"
                        "fmla   v15.4s, v11.4s, v1.s[1] \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v1.s[2]  \n"
                        "fmla   v9.4s, v13.4s, v1.s[2]  \n"

                        "prfm   pldl1keep, [%10, #128]  \n"
                        "ld1    {v0.4s}, [%10]          \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "fmla   v14.4s, v10.4s, v0.s[0] \n"
                        "fmla   v15.4s, v11.4s, v0.s[0] \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v0.s[1]  \n"
                        "fmla   v9.4s, v13.4s, v0.s[1]  \n"

                        "fmla   v14.4s, v10.4s, v0.s[2] \n"
                        "fmla   v15.4s, v11.4s, v0.s[2] \n"

                        "fadd   v8.4s, v8.4s, v14.4s    \n"
                        "fadd   v9.4s, v9.4s, v15.4s    \n"

                        "sub    %11, %11, #288          \n"

                        "st1    {v8.s}[0], [%0], #4     \n"
                        "st1    {v8.s}[1], [%1], #4     \n"
                        "st1    {v8.s}[2], [%2], #4     \n"
                        "st1    {v8.s}[3], [%3], #4     \n"

                        "st1    {v9.s}[0], [%4], #4     \n"
                        "st1    {v9.s}[1], [%5], #4     \n"
                        "st1    {v9.s}[2], [%6], #4     \n"
                        "st1    {v9.s}[3], [%7], #4     \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(outptr4),    // %4
                          "=r"(outptr5),    // %5
                          "=r"(outptr6),    // %6
                          "=r"(outptr7),    // %7
                          "=r"(r0),         // %8
                          "=r"(r1),         // %9
                          "=r"(r2),         // %10
                          "=r"(ktmp)        // %11
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(outptr4),
                          "5"(outptr5),
                          "6"(outptr6),
                          "7"(outptr7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(ktmp)
                        : "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    );
#else // __aarch64__
                    asm volatile(
                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "pld        [%8, #128]      \n"
                        "vld1.f32   {d0-d1}, [%8]   \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vld1.f32   {d16[0]}, [%0]  \n"
                        "vld1.f32   {d16[1]}, [%1]  \n"
                        "vld1.f32   {d17[0]}, [%2]  \n"
                        "vld1.f32   {d17[1]}, [%3]  \n"

                        "vmul.f32   q14, q10, d0[0] \n"
                        "vmul.f32   q15, q11, d0[0] \n"

                        "vld1.f32   {d18[0]}, [%4]  \n"
                        "vld1.f32   {d18[1]}, [%5]  \n"
                        "vld1.f32   {d19[0]}, [%6]  \n"
                        "vld1.f32   {d19[1]}, [%7]  \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d0[1]  \n"
                        "vmla.f32   q9, q13, d0[1]  \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vmla.f32   q14, q10, d1[0] \n"
                        "vmla.f32   q15, q11, d1[0] \n"

                        "pld        [%9, #128]      \n"
                        "vld1.f32   {d2-d3}, [%9]   \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d2[0]  \n"
                        "vmla.f32   q9, q13, d2[0]  \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vmla.f32   q14, q10, d2[1] \n"
                        "vmla.f32   q15, q11, d2[1] \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d3[0]  \n"
                        "vmla.f32   q9, q13, d3[0]  \n"

                        "pld        [%10, #128]     \n"
                        "vld1.f32   {d0-d1}, [%10]  \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vmla.f32   q14, q10, d0[0] \n"
                        "vmla.f32   q15, q11, d0[0] \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d0[1]  \n"
                        "vmla.f32   q9, q13, d0[1]  \n"

                        "vmla.f32   q14, q10, d1[0] \n"
                        "vmla.f32   q15, q11, d1[0] \n"

                        "vadd.f32   q8, q8, q14     \n"
                        "vadd.f32   q9, q9, q15     \n"

                        "sub        %11, %11, #288  \n"

                        "vst1.f32   {d16[0]}, [%0]! \n"
                        "vst1.f32   {d16[1]}, [%1]! \n"
                        "vst1.f32   {d17[0]}, [%2]! \n"
                        "vst1.f32   {d17[1]}, [%3]! \n"

                        "vst1.f32   {d18[0]}, [%4]! \n"
                        "vst1.f32   {d18[1]}, [%5]! \n"
                        "vst1.f32   {d19[0]}, [%6]! \n"
                        "vst1.f32   {d19[1]}, [%7]! \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(outptr4),    // %4
                          "=r"(outptr5),    // %5
                          "=r"(outptr6),    // %6
                          "=r"(outptr7),    // %7
                          "=r"(r0),         // %8
                          "=r"(r1),         // %9
                          "=r"(r2),         // %10
                          "=r"(ktmp)        // %11
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(outptr4),
                          "5"(outptr5),
                          "6"(outptr6),
                          "7"(outptr7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(ktmp)
                        : "memory", "q0", "q1", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else // __ARM_NEON
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;
                    float sum4 = 0.f;
                    float sum5 = 0.f;
                    float sum6 = 0.f;
                    float sum7 = 0.f;

                    sum0 += r0[0] * ktmp[0];
                    sum1 += r0[0] * ktmp[1];
                    sum2 += r0[0] * ktmp[2];
                    sum3 += r0[0] * ktmp[3];
                    sum4 += r0[0] * ktmp[4];
                    sum5 += r0[0] * ktmp[5];
                    sum6 += r0[0] * ktmp[6];
                    sum7 += r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[1] * ktmp[0];
                    sum1 += r0[1] * ktmp[1];
                    sum2 += r0[1] * ktmp[2];
                    sum3 += r0[1] * ktmp[3];
                    sum4 += r0[1] * ktmp[4];
                    sum5 += r0[1] * ktmp[5];
                    sum6 += r0[1] * ktmp[6];
                    sum7 += r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[2] * ktmp[0];
                    sum1 += r0[2] * ktmp[1];
                    sum2 += r0[2] * ktmp[2];
                    sum3 += r0[2] * ktmp[3];
                    sum4 += r0[2] * ktmp[4];
                    sum5 += r0[2] * ktmp[5];
                    sum6 += r0[2] * ktmp[6];
                    sum7 += r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[0] * ktmp[0];
                    sum1 += r1[0] * ktmp[1];
                    sum2 += r1[0] * ktmp[2];
                    sum3 += r1[0] * ktmp[3];
                    sum4 += r1[0] * ktmp[4];
                    sum5 += r1[0] * ktmp[5];
                    sum6 += r1[0] * ktmp[6];
                    sum7 += r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[1] * ktmp[0];
                    sum1 += r1[1] * ktmp[1];
                    sum2 += r1[1] * ktmp[2];
                    sum3 += r1[1] * ktmp[3];
                    sum4 += r1[1] * ktmp[4];
                    sum5 += r1[1] * ktmp[5];
                    sum6 += r1[1] * ktmp[6];
                    sum7 += r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[2] * ktmp[0];
                    sum1 += r1[2] * ktmp[1];
                    sum2 += r1[2] * ktmp[2];
                    sum3 += r1[2] * ktmp[3];
                    sum4 += r1[2] * ktmp[4];
                    sum5 += r1[2] * ktmp[5];
                    sum6 += r1[2] * ktmp[6];
                    sum7 += r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[0] * ktmp[0];
                    sum1 += r2[0] * ktmp[1];
                    sum2 += r2[0] * ktmp[2];
                    sum3 += r2[0] * ktmp[3];
                    sum4 += r2[0] * ktmp[4];
                    sum5 += r2[0] * ktmp[5];
                    sum6 += r2[0] * ktmp[6];
                    sum7 += r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[1] * ktmp[0];
                    sum1 += r2[1] * ktmp[1];
                    sum2 += r2[1] * ktmp[2];
                    sum3 += r2[1] * ktmp[3];
                    sum4 += r2[1] * ktmp[4];
                    sum5 += r2[1] * ktmp[5];
                    sum6 += r2[1] * ktmp[6];
                    sum7 += r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[2] * ktmp[0];
                    sum1 += r2[2] * ktmp[1];
                    sum2 += r2[2] * ktmp[2];
                    sum3 += r2[2] * ktmp[3];
                    sum4 += r2[2] * ktmp[4];
                    sum5 += r2[2] * ktmp[5];
                    sum6 += r2[2] * ktmp[6];
                    sum7 += r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8*9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* ktmp = _kernel.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;

            const float* k0 = ktmp;
            const float* k1 = ktmp + 3;
            const float* k2 = ktmp + 6;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                    "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                    "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%3]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                    "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%4]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                    "fadd       v0.4s, v0.4s, v10.4s           \n"
                    "fadd       v0.4s, v0.4s, v11.4s           \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s}, [%1], #16             \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]       \n"

                    "vmla.f32   q0, q2, %e10[0]     \n"
                    "vmul.f32   q10, q3, %e10[1]    \n"

                    "pld        [%2, #128]          \n"
                    "vld2.f32   {d16-d17}, [%2]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmul.f32   q11, q1, %f10[0]    \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"

                    "vmla.f32   q0, q2, %e11[0]     \n"
                    "vmla.f32   q10, q3, %e11[1]    \n"

                    "pld        [%3, #128]          \n"
                    "vld2.f32   {d16-d17}, [%3]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f11[0]    \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"

                    "vmla.f32   q0, q2, %e12[0]     \n"
                    "vmla.f32   q10, q3, %e12[1]    \n"

                    "pld        [%4, #128]          \n"
                    "vld2.f32   {d16-d17}, [%4]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f12[0]    \n"

                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vadd.f32   q0, q0, q10         \n"
                    "vadd.f32   q0, q0, q11         \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * ktmp[0];
                    sum += r0[1] * ktmp[1];
                    sum += r0[2] * ktmp[2];
                    sum += r1[0] * ktmp[3];
                    sum += r1[1] * ktmp[4];
                    sum += r1[2] * ktmp[5];
                    sum += r2[0] * ktmp[6];
                    sum += r2[1] * ktmp[7];
                    sum += r2[2] * ktmp[8];

                    *outptr += sum;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
