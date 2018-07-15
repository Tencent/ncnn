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

static void conv2x2s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+1<inch; q+=2)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);

            const float* kernel0 = kernel + p*inch*4  + q*4;
            const float* kernel1 = kernel0 + 4;

            const float* r00 = img0;
            const float* r01 = img0 + w;

            const float* r10 = img1;
            const float* r11 = img1 + w;

#if __ARM_NEON
            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
#endif // __ARM_NEON

            for (int i = 0; i < outh; i++)
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
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1], #16             \n"
                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v2.4s}, [%2], #16             \n"
                    "prfm       pldl1keep, [%3, #128]          \n"
                    "ld1        {v12.4s}, [%3], #16            \n"
                    "prfm       pldl1keep, [%4, #128]          \n"
                    "ld1        {v14.4s}, [%4], #16            \n"

                    "0:                                        \n"
                    "prfm       pldl1keep, [%5, #128]          \n"
                    "ld1        {v9.4s}, [%5]                  \n"

                    "fmul       v8.4s, v0.4s, %12.s[0]         \n"
                    "fmla       v9.4s, v2.4s, %12.s[2]         \n"
                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v1.4s}, [%1], #16             \n"

                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v3.4s}, [%2], #16             \n"

                    "ext        v10.16b, v0.16b, v1.16b, #4    \n"
                    "ext        v11.16b, v2.16b, v3.16b, #4    \n"

                    "fmla       v8.4s, v12.4s, %13.s[0]        \n"
                    "fmla       v9.4s, v14.4s, %13.s[2]        \n"

                    "prfm       pldl1keep, [%3, #128]          \n"
                    "ld1        {v13.4s}, [%3], #16            \n"

                    "prfm       pldl1keep, [%4, #128]          \n"
                    "ld1        {v15.4s}, [%4], #16            \n"

                    "fmla       v8.4s, v10.4s, %12.s[1]        \n"
                    "fmla       v9.4s, v11.4s, %12.s[3]        \n"

                    "ext        v10.16b, v12.16b, v13.16b, #4  \n"
                    "ext        v11.16b, v14.16b, v15.16b, #4  \n"

                    "fmla       v8.4s, v10.4s, %13.s[1]        \n"
                    "fmla       v9.4s, v11.4s, %13.s[3]        \n"

                    "orr        v0.16b, v1.16b, v1.16b         \n"
                    "orr        v2.16b, v3.16b, v3.16b         \n"

                    "fadd       v8.4s, v8.4s, v9.4s            \n"

                    "orr        v12.16b, v13.16b, v13.16b      \n"
                    "orr        v14.16b, v15.16b, v15.16b      \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v8.4s}, [%5], #16             \n"
                    "bne        0b                             \n"
                    "sub        %1, %1, #16                    \n"
                    "sub        %2, %2, #16                    \n"
                    "sub        %3, %3, #16                    \n"
                    "sub        %4, %4, #16                    \n"
                    : "=r"(nn),     // %0
                      "=r"(r00),    // %1
                      "=r"(r01),    // %2
                      "=r"(r10),    // %3
                      "=r"(r11),    // %4
                      "=r"(outptr)  // %5
                    : "0"(nn),
                      "1"(r00),
                      "2"(r01),
                      "3"(r10),
                      "4"(r11),
                      "5"(outptr),
                      "w"(_k0),     // %12
                      "w"(_k1)      // %13
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]!      \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d4-d5}, [%2]!      \n"

                    "pld        [%3, #128]          \n"
                    "vld1.f32   {d24-d25}, [%3]!    \n"
                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d28-d29}, [%4]!    \n"

                    "0:                             \n"
                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d18-d19}, [%5]     \n"// q9 = sum

                    "vmul.f32   q8, q0, %e12[0]     \n"
                    "vmla.f32   q9, q2, %f12[0]     \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d2-d3}, [%1]!      \n"

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d6-d7}, [%2]!      \n"

                    "vext.f32   q10, q0, q1, #1     \n"
                    "vext.f32   q11, q2, q3, #1     \n"

                    "vmla.f32   q8, q12, %e13[0]    \n"
                    "vmla.f32   q9, q14, %f13[0]    \n"

                    "pld        [%3, #128]          \n"
                    "vld1.f32   {d26-d27}, [%3]!    \n"

                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d30-d31}, [%4]!    \n"

                    "vmla.f32   q8, q10, %e12[1]    \n"
                    "vmla.f32   q9, q11, %f12[1]    \n"

                    "vext.f32   q10, q12, q13, #1   \n"
                    "vext.f32   q11, q14, q15, #1   \n"

                    "vmla.f32   q8, q10, %e13[1]    \n"
                    "vmla.f32   q9, q11, %f13[1]    \n"

                    "vorr       q0, q1, q1          \n"
                    "vorr       q2, q3, q3          \n"

                    "vadd.f32   q8, q8, q9          \n"

                    "vorr       q12, q13, q13       \n"
                    "vorr       q14, q15, q15       \n"

                    "subs       %0, #1              \n"

                    "vst1.f32   {d16-d17}, [%5]!    \n"

                    "bne        0b                  \n"
                    "sub        %1, #16             \n"
                    "sub        %2, #16             \n"
                    "sub        %3, #16             \n"
                    "sub        %4, #16             \n"
                    : "=r"(nn),     // %0
                      "=r"(r00),    // %1
                      "=r"(r01),    // %2
                      "=r"(r10),    // %3
                      "=r"(r11),    // %4
                      "=r"(outptr)  // %5
                    : "0"(nn),
                      "1"(r00),
                      "2"(r01),
                      "3"(r10),
                      "4"(r11),
                      "5"(outptr),
                      "w"(_k0),     // %12
                      "w"(_k1)      // %13
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON

                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x2_t _r00 = vld1_f32(r00);
                    float32x2_t _r01 = vld1_f32(r01);
                    float32x4_t _r00r1 = vcombine_f32(_r00, _r01);
                    float32x4_t _s0s1 = vmulq_f32(_r00r1, _k0);

                    float32x2_t _r10 = vld1_f32(r10);
                    float32x2_t _r11 = vld1_f32(r11);
                    float32x4_t _r10r1 = vcombine_f32(_r10, _r11);
                    _s0s1 = vmlaq_f32(_s0s1, _r10r1, _k1);

                    float32x2_t _s = vadd_f32(vget_low_f32(_s0s1), vget_high_f32(_s0s1));
                    _s = vpadd_f32(_s, _s);
                    *outptr += vget_lane_f32(_s, 0);
#else
                    float sum = 0.f;

                    sum += r00[0] * kernel0[0];
                    sum += r00[1] * kernel0[1];
                    sum += r01[0] * kernel0[2];
                    sum += r01[1] * kernel0[3];

                    sum += r10[0] * kernel1[0];
                    sum += r10[1] * kernel1[1];
                    sum += r11[0] * kernel1[2];
                    sum += r11[1] * kernel1[3];

                    *outptr += sum;
#endif // __ARM_NEON

                    r00 += 1;
                    r01 += 1;
                    r10 += 1;
                    r11 += 1;
                    outptr++;
                }

                r00 += 1;
                r01 += 1;
                r10 += 1;
                r11 += 1;
            }
        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*4  + q*4;

            const float* r0 = img0;
            const float* r1 = img0 + w;

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(kernel0[0]);
            float32x4_t _k1 = vdupq_n_f32(kernel0[1]);
            float32x4_t _k2 = vdupq_n_f32(kernel0[2]);
            float32x4_t _k3 = vdupq_n_f32(kernel0[3]);
#endif // __ARM_NEON

            for (int i = 0; i < outh; i++)
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
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1], #16             \n"
                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v2.4s}, [%2], #16             \n"

                    "0:                                        \n"
                    "prfm       pldl1keep, [%3, #128]          \n"
                    "ld1        {v9.4s}, [%3]                  \n"

                    "fmul       v8.4s, v0.4s, %8.4s            \n"
                    "fmla       v9.4s, v2.4s, %10.4s           \n"

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v1.4s}, [%1], #16             \n"
                    "ext        v10.16b, v0.16b, v1.16b, #4    \n"

                    "fmla       v8.4s, v10.4s, %9.4s           \n"

                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v3.4s}, [%2], #16             \n"
                    "ext        v11.16b, v2.16b, v3.16b, #4    \n"

                    "fmla       v9.4s, v11.4s, %11.4s          \n"

                    "orr        v0.16b, v1.16b, v1.16b         \n"
                    "fadd       v8.4s, v8.4s, v9.4s            \n"
                    "orr        v2.16b, v3.16b, v3.16b         \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v8.4s}, [%3], #16             \n"
                    "bne        0b                             \n"
                    "sub        %1, %1, #16                    \n"
                    "sub        %2, %2, #16                    \n"
                    : "=r"(nn),     // %0
                      "=r"(r0),     // %1
                      "=r"(r1),     // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(r0),
                      "2"(r1),
                      "3"(outptr),
                      "w"(_k0),     // %8
                      "w"(_k1),     // %9
                      "w"(_k2),     // %10
                      "w"(_k3)      // %11
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1]!      \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d4-d5}, [%2]!      \n"

                    "0:                             \n"
                    "pld        [%3, #128]          \n"
                    "vld1.f32   {d18-d19}, [%3]     \n"// q9 = sum

                    "vmul.f32   q8, q0, %q8         \n"
                    "vmla.f32   q9, q2, %q10        \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d2-d3}, [%1]!      \n"
                    "vext.f32   q10, q0, q1, #1     \n"

                    "vmla.f32   q8, q10, %q9        \n"

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d6-d7}, [%2]!      \n"
                    "vext.f32   q11, q2, q3, #1     \n"

                    "vmla.f32   q9, q11, %q11       \n"

                    "vorr       q0, q1, q1          \n"
                    "vadd.f32   q8, q8, q9          \n"
                    "vorr       q2, q3, q3          \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d16-d17}, [%3]!    \n"
                    "bne        0b                  \n"
                    "sub        %1, #16             \n"
                    "sub        %2, #16             \n"
                    : "=r"(nn),     // %0
                      "=r"(r0),     // %1
                      "=r"(r1),     // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(r0),
                      "2"(r1),
                      "3"(outptr),
                      "w"(_k0),     // %8
                      "w"(_k1),     // %9
                      "w"(_k2),     // %10
                      "w"(_k3)      // %11
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0123 = vld1q_f32(kernel0);
#endif

                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x2_t _r0 = vld1_f32(r0);
                    float32x2_t _r1 = vld1_f32(r1);
                    float32x4_t _r0r1 = vcombine_f32(_r0, _r1);
                    float32x4_t _s0s1 = vmulq_f32(_r0r1, _k0123);
                    float32x2_t _s = vadd_f32(vget_low_f32(_s0s1), vget_high_f32(_s0s1));
                    _s = vpadd_f32(_s, _s);
                    *outptr += vget_lane_f32(_s, 0);
#else
                    float sum = 0.f;
                    sum += r0[0] * kernel0[0];
                    sum += r0[1] * kernel0[1];
                    sum += r1[0] * kernel0[2];
                    sum += r1[1] * kernel0[3];
                    *outptr += sum;
#endif

                    r0 += 1;
                    r1 += 1;
                    outptr++;
                }

                r0 += 1;
                r1 += 1;

            }

        }
    }

}
