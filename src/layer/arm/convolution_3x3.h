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

static void conv3x3s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
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

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k3456 = vld1q_f32(kernel0+3);
            float32x4_t _k6789 = vld1q_f32(kernel0+6);
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
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vld1q_f32(outptr2);
                    float32x4_t _sum4 = vdupq_n_f32(0.f);

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
                    _sum2 = vfmaq_laneq_f32(_sum2, _r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);

                    _sum3 = vfmaq_laneq_f32(_sum3, _r10, _k0123, 0);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r11, _k0123, 1);
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
                    "veor       q6, q6              \n"
                    "veor       q15, q15            \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d18-d20}, [%3 :64] \n"// r0
                    "add        %3, #16             \n"

                    "veor       q13, q13            \n"
                    "veor       q14, q14            \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d14-d15}, [%1 :64] \n"// _sum

                    "vmla.f32   q7, q9, %e14[0]     \n"
                    "vmla.f32   q6, q11, %e14[1]    \n"
                    "vmla.f32   q13, q12, %f14[0]   \n"

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
                    "vmla.f32   q14, q11, %e14[1]   \n"
                    "vmla.f32   q15, q12, %f14[0]   \n"

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
                    "veor       q6, q6              \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d18-d20}, [%3 :64] \n"// r0

                    "vadd.f32   q8, q8, q14         \n"
                    "veor       q14, q14            \n"
                    "vadd.f32   q7, q7, q13         \n"
                    "veor       q13, q13            \n"
                    "vadd.f32   q8, q8, q15         \n"
                    "veor       q15, q15            \n"

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
                    float32x4_t _sum2 = vdupq_n_f32(0.f);

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
                    _sum2 = vfmaq_laneq_f32(_sum2, _r01, _k0123, 1);
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

                    "veor       q13, q13            \n"
                    "veor       q14, q14            \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "0:                             \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d14-d15}, [%1]     \n"// _sum

                    "vmla.f32   q7, q8, %e10[0]     \n"
                    "vmla.f32   q13, q10, %e10[1]   \n"
                    "vmla.f32   q14, q11, %f10[0]   \n"

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
                    "veor       q13, q13            \n"
                    "vadd.f32   q7, q7, q14         \n"
                    "veor       q14, q14            \n"

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
            const float* kernel0 = kernel.data + p*inch * 9 + q * 9;
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
}

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
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        top_blob_tm.create(8*8, w_tm/8 * h_tm/8, outch);

        #pragma omp parallel for
        for (int p = 0; p<outch; p++)
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
        int h_tm = outh / 6 * 8;

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

static void conv3x3s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
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
                for (; nn>0; nn--)
                {
                    float32x4_t _outp = vld1q_f32(outptr);

                    float32x4x2_t _r0 = vld2q_f32(r0);
                    float32x4x2_t _r0n = vld2q_f32(r0+8);

                    float32x4_t _r00 = _r0.val[0];// 0 2 4 6
                    float32x4_t _r01 = _r0.val[1];// 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0n.val[0], 1);// 2 4 6 8

                    _outp = vfmaq_laneq_f32(_outp, _r00, _k0123, 0);
                    _outp = vfmaq_laneq_f32(_outp, _r01, _k0123, 1);
                    _outp = vfmaq_laneq_f32(_outp, _r02, _k0123, 2);

                    float32x4x2_t _r1 = vld2q_f32(r1);
                    float32x4x2_t _r1n = vld2q_f32(r1+8);

                    float32x4_t _r10 = _r1.val[0];
                    float32x4_t _r11 = _r1.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1n.val[0], 1);

                    _outp = vfmaq_laneq_f32(_outp, _r10, _k3456, 0);
                    _outp = vfmaq_laneq_f32(_outp, _r11, _k3456, 1);
                    _outp = vfmaq_laneq_f32(_outp, _r12, _k3456, 2);

                    float32x4x2_t _r2 = vld2q_f32(r2);
                    float32x4x2_t _r2n = vld2q_f32(r2+8);

                    float32x4_t _r20 = _r2.val[0];
                    float32x4_t _r21 = _r2.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2n.val[0], 1);

                    _outp = vfmaq_laneq_f32(_outp, _r20, _k6789, 0);
                    _outp = vfmaq_laneq_f32(_outp, _r21, _k6789, 1);
                    _outp = vfmaq_laneq_f32(_outp, _r22, _k6789, 2);

                    vst1q_f32(outptr, _outp);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "veor       q10, q10            \n"
                    "veor       q11, q11            \n"

                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]       \n"

                    "vmla.f32   q0, q2, %e10[0]     \n"
                    "vmla.f32   q10, q3, %e10[1]    \n"

                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d16-d19}, [%2]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f10[0]    \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"

                    "vmla.f32   q0, q2, %e11[0]     \n"
                    "vmla.f32   q10, q3, %e11[1]    \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d16-d19}, [%3]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f11[0]    \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"

                    "vmla.f32   q0, q2, %e12[0]     \n"
                    "vmla.f32   q10, q3, %e12[1]    \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d16-d19}, [%4]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f12[0]    \n"

                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vadd.f32   q0, q0, q10         \n"
                    "veor       q10, q10            \n"
                    "vadd.f32   q0, q0, q11         \n"
                    "veor       q11, q11            \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),
                      "=r"(r2)
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
