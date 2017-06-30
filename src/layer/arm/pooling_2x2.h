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

static void pooling2x2s2_max_neon(const Mat& bottom_blob, Mat& top_blob)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    #pragma omp parallel for
    for (int q=0; q<inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0;
        const float* r1 = img0 + w;

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
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r11 = vld1q_f32(r1 + 4);

                float32x4_t _max0 = vmaxq_f32(_r00, _r10);
                float32x4_t _max1 = vmaxq_f32(_r01, _r11);

                float32x4_t _max = vpmaxq_f32(_max0, _max1);

                vst1q_f32(outptr, _max);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "pld        [%2, #256]          \n"
                "vld1.f32   {d0-d3}, [%1]!      \n"
                "vld1.f32   {d4-d7}, [%2]!      \n"
                "vmax.f32   q0, q0, q2          \n"
                "vmax.f32   q1, q1, q3          \n"
                "vpmax.f32  d4, d0, d1          \n"
                "vpmax.f32  d5, d2, d3          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d4-d5}, [%3]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(r0),     // %1
                  "=r"(r1),     // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(r0),
                  "2"(r1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float max0 = std::max(r0[0], r0[1]);
                float max1 = std::max(r1[0], r1[1]);

                *outptr = std::max(max0, max1);

                r0 += 2;
                r1 += 2;
                outptr++;
            }

            r0 += w;
            r1 += w;
        }
    }
}
