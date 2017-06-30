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

static void conv4x4s4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
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
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*16  + q*16;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0+4);
            float32x4_t _k891011 = vld1q_f32(kernel0+8);
            float32x4_t _k12131415 = vld1q_f32(kernel0+12);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 4;
            const float* k2 = kernel0 + 8;
            const float* k3 = kernel0 + 12;
#endif // __ARM_NEON

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);

                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);

                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k0123);
                    float32x4_t _sum1 = vmulq_f32(_r01, _k0123);
                    float32x4_t _sum2 = vmulq_f32(_r02, _k0123);
                    float32x4_t _sum3 = vmulq_f32(_r03, _k0123);

                    _sum0 = vfmaq_f32(_sum0, _r10, _k4567);
                    _sum1 = vfmaq_f32(_sum1, _r11, _k4567);
                    _sum2 = vfmaq_f32(_sum2, _r12, _k4567);
                    _sum3 = vfmaq_f32(_sum3, _r13, _k4567);

                    _sum0 = vfmaq_f32(_sum0, _r20, _k891011);
                    _sum1 = vfmaq_f32(_sum1, _r21, _k891011);
                    _sum2 = vfmaq_f32(_sum2, _r22, _k891011);
                    _sum3 = vfmaq_f32(_sum3, _r23, _k891011);

                    _sum0 = vfmaq_f32(_sum0, _r30, _k12131415);
                    _sum1 = vfmaq_f32(_sum1, _r31, _k12131415);
                    _sum2 = vfmaq_f32(_sum2, _r32, _k12131415);
                    _sum3 = vfmaq_f32(_sum3, _r33, _k12131415);

                    float32x4_t _s01 = vpaddq_f32(_sum0, _sum1);
                    float32x4_t _s23 = vpaddq_f32(_sum2, _sum3);
                    float32x4_t _sum = vpaddq_f32(_s01, _s23);

                    float32x4_t _outp = vld1q_f32(outptr);

                    _outp = vaddq_f32(_outp, _sum);

                    vst1q_f32(outptr, _sum);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(

                    "pld        [%1, #128]          \n"

                    "0:                             \n"

                    "pld        [%2, #512]          \n"
                    "pld        [%3, #512]          \n"

                    "vld1.f32   {d14-d15}, [%1]     \n"// q7 = outptr

                    "vld1.f32   {d16-d17}, [%2]!    \n"// q8  = r0
                    "vld1.f32   {d18-d19}, [%3]!    \n"// q9  = r1

                    "pld        [%4, #512]          \n"
                    "pld        [%5, #512]          \n"

                    "vmul.f32   q12, q8, %q12       \n"
                    "vmul.f32   q13, q9, %q13       \n"

                    "vld1.f32   {d20-d21}, [%4]!    \n"// q10 = r2
                    "vld1.f32   {d22-d23}, [%5]!    \n"// q11 = r3

                    "vmla.f32   q12, q10, %q14      \n"
                    "vmla.f32   q13, q11, %q15      \n"

                    "vadd.f32   q5, q12, q13        \n"

                    "vld1.f32   {d16-d17}, [%2]!    \n"// q8  = r0
                    "vld1.f32   {d18-d19}, [%3]!    \n"// q9  = r1

                    "vmul.f32   q12, q8, %q12       \n"
                    "vmul.f32   q13, q9, %q13       \n"

                    "vld1.f32   {d20-d21}, [%4]!    \n"// q10 = r2
                    "vld1.f32   {d22-d23}, [%5]!    \n"// q11 = r3

                    "vmla.f32   q12, q10, %q14      \n"
                    "vmla.f32   q13, q11, %q15      \n"

                    "vadd.f32   q6, q12, q13        \n"

                    "vld1.f32   {d16-d17}, [%2]!    \n"// q8  = r0
                    "vld1.f32   {d18-d19}, [%3]!    \n"// q9  = r1

                    "vmul.f32   q12, q8, %q12       \n"
                    "vmul.f32   q13, q9, %q13       \n"

                    "vld1.f32   {d20-d21}, [%4]!    \n"// q10 = r2
                    "vld1.f32   {d22-d23}, [%5]!    \n"// q11 = r3

                    "vmla.f32   q12, q10, %q14      \n"
                    "vmla.f32   q13, q11, %q15      \n"

                    "vadd.f32   q14, q12, q13       \n"

                    "vld1.f32   {d16-d17}, [%2]!    \n"// q8  = r0
                    "vld1.f32   {d18-d19}, [%3]!    \n"// q9  = r1

                    "vmul.f32   q12, q8, %q12       \n"
                    "vmul.f32   q13, q9, %q13       \n"

                    "vld1.f32   {d20-d21}, [%4]!    \n"// q10 = r2
                    "vld1.f32   {d22-d23}, [%5]!    \n"// q11 = r3

                    "vmla.f32   q12, q10, %q14      \n"
                    "vmla.f32   q13, q11, %q15      \n"

                    "vadd.f32   q15, q12, q13       \n"

                    "vadd.f32   d10, d10, d11       \n"
                    "vadd.f32   d28, d28, d29       \n"
                    "vadd.f32   d11, d12, d13       \n"
                    "vadd.f32   d29, d30, d31       \n"

                    "vpadd.f32  d10, d10, d11       \n"
                    "vpadd.f32  d11, d28, d29       \n"

                    "vadd.f32   q7, q7, q5          \n"

                    "vst1.f32   {d14-d15}, [%1]!    \n"

                    "pld        [%1, #128]          \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3)          // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k0123),      // %12
                      "w"(_k4567),      // %13
                      "w"(_k891011),    // %14
                      "w"(_k12131415)   // %15
                    : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);
                    float32x4_t _r3 = vld1q_f32(r3);

                    float32x4_t _sum = vmulq_f32(_r0, _k0123);
                    _sum = vmlaq_f32(_sum, _r1, _k4567);
                    _sum = vmlaq_f32(_sum, _r2, _k891011);
                    _sum = vmlaq_f32(_sum, _r3, _k12131415);

                    *outptr += vaddvq_f32(_sum);
#else
                    float sum = 0.f;

                    asm volatile(
                        "vld1.f32   {d16-d17}, [%0]!    \n"// q8  = r0
                        "vld1.f32   {d18-d19}, [%1]!    \n"// q9  = r1

                        "vmul.f32   q12, q8, %q9        \n"
                        "vmul.f32   q13, q9, %q10       \n"

                        "vld1.f32   {d20-d21}, [%2]!    \n"// q10 = r2
                        "vld1.f32   {d22-d23}, [%3]!    \n"// q11 = r3

                        "vmla.f32   q12, q10, %q11      \n"
                        "vmla.f32   q13, q11, %q12      \n"

                        "vadd.f32   q5, q12, q13        \n"
                        "vadd.f32   d10, d10, d11       \n"
                        "vpadd.f32  d10, d10, d10       \n"
                        "vmov.f32   %4, d10[0]          \n"
                        : "=r"(r0),         // %0
                          "=r"(r1),         // %1
                          "=r"(r2),         // %2
                          "=r"(r3),         // %3
                          "=r"(sum)         // %4
                        : "0"(r0),
                          "1"(r1),
                          "2"(r2),
                          "3"(r3),
                          "w"(_k0123),      // %9
                          "w"(_k4567),      // %10
                          "w"(_k891011),    // %11
                          "w"(_k12131415)   // %12
                        : "cc", "memory", "q5", "q6", "q8", "q9", "q10", "q11", "q12", "q13"
                    );

                    *outptr += sum;
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];

                    *outptr += sum;
#endif // __ARM_NEON
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr++;
                }

                r0 += w * 3;
                r1 += w * 3;
                r2 += w * 3;
                r3 += w * 3;
            }

        }
    }

}

