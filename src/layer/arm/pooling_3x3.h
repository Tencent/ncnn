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

static void pooling3x3s2_max_neon(const Mat& bottom_blob, Mat& top_blob)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    #pragma omp parallel for
    for (int q=0; q<inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;

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
            float32x4x2_t _r0 = vld2q_f32(r0);
            float32x4x2_t _r1 = vld2q_f32(r1);
            float32x4x2_t _r2 = vld2q_f32(r2);
            for (; nn>0; nn--)
            {
                float32x4x2_t _r0n = vld2q_f32(r0+8);
                float32x4x2_t _r1n = vld2q_f32(r1+8);
                float32x4x2_t _r2n = vld2q_f32(r2+8);

                float32x4_t _max0 = vmaxq_f32(_r0.val[0], _r0.val[1]);
                float32x4_t _max1 = vmaxq_f32(_r1.val[0], _r1.val[1]);
                float32x4_t _max2 = vmaxq_f32(_r2.val[0], _r2.val[1]);

                float32x4_t _r02 = vextq_f32(_r0.val[0], _r0n.val[0], 1);
                float32x4_t _r12 = vextq_f32(_r1.val[0], _r1n.val[0], 1);
                float32x4_t _r22 = vextq_f32(_r2.val[0], _r2n.val[0], 1);

                _max0 = vmaxq_f32(_max0, _r02);
                _max1 = vmaxq_f32(_max1, _r12);
                _max2 = vmaxq_f32(_max2, _r22);

                float32x4_t _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);

                vst1q_f32(outptr, _max);

                _r0 = _r0n;
                _r1 = _r1n;
                _r2 = _r2n;

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "pld        [%1, #256]          \n"
                "vld2.f32   {d0-d3}, [%1]!      \n"// q0 = 0 2 4 6  q1 = 1 3 5 7
                "pld        [%2, #256]          \n"
                "vld2.f32   {d4-d7}, [%2]!      \n"
                "pld        [%3, #256]          \n"
                "vld2.f32   {d8-d11}, [%3]!     \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld2.f32   {d12-d15}, [%1]!    \n"// q6 = 8 10 12 14  q7 = 9 11 13 15

                "vmax.f32   q12, q0, q1         \n"
                "vmax.f32   q13, q2, q3         \n"

                "pld        [%2, #256]          \n"
                "vld2.f32   {d16-d19}, [%2]!    \n"

                "vmax.f32   q14, q4, q5         \n"
                "vext.32    q0, q0, q6, #1      \n"

                "pld        [%3, #256]          \n"
                "vld2.f32   {d20-d23}, [%3]!    \n"

                "vext.32    q2, q2, q8, #1      \n"

                "vmax.f32   q12, q12, q0        \n"
                "vext.32    q4, q4, q10, #1     \n"

                "vmax.f32   q13, q13, q2        \n"
                "vmax.f32   q14, q14, q4        \n"
                "vmax.f32   q12, q12, q13       \n"

                "vorr       q0, q6, q6          \n"
                "vorr       q1, q7, q7          \n"
                "vmax.f32   q12, q12, q14       \n"

                "vorr       q2, q8, q8          \n"
                "vorr       q3, q9, q9          \n"
                "vorr       q4, q10, q10        \n"
                "vorr       q5, q11, q11        \n"

                "subs       %0, #1              \n"
                "vst1.f32   {d24-d25}, [%4]!    \n"
                "bne        0b                  \n"
                "sub        %1, #32             \n"
                "sub        %2, #32             \n"
                "sub        %3, #32             \n"
                : "=r"(nn),     // %0
                  "=r"(r0),     // %1
                  "=r"(r1),     // %2
                  "=r"(r2),     // %3
                  "=r"(outptr)  // %4
                : "0"(nn),
                  "1"(r0),
                  "2"(r1),
                  "3"(r2),
                  "4"(outptr)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float max0 = std::max(std::max(r0[0], r0[1]), r0[2]);
                float max1 = std::max(std::max(r1[0], r1[1]), r1[2]);
                float max2 = std::max(std::max(r2[0], r2[1]), r2[2]);

                *outptr = std::max(std::max(max0, max1), max2);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;//1 + w;
            r1 += tailstep;//1 + w;
            r2 += tailstep;//1 + w;
        }
    }
}
