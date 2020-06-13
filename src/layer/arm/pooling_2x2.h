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

static void pooling2x2s2_max_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
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
            if (nn > 0)
            {
                asm volatile(
                    "0:                                   \n"
                    "prfm       pldl1keep, [%1, #256]     \n"
                    "prfm       pldl1keep, [%2, #256]     \n"
                    "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                    "fmax       v0.4s, v0.4s, v2.4s       \n"
                    "fmax       v1.4s, v1.4s, v3.4s       \n"
                    "fmaxp      v2.4s, v0.4s, v1.4s       \n"
                    "subs       %w0, %w0, #1              \n"
                    "st1        {v2.4s}, [%3], #16        \n"
                    "bne        0b                        \n"
                    : "=r"(nn),    // %0
                    "=r"(r0),    // %1
                    "=r"(r1),    // %2
                    "=r"(outptr) // %3
                    : "0"(nn),
                    "1"(r0),
                    "2"(r1),
                    "3"(outptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3");
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
                    : "=r"(nn),    // %0
                    "=r"(r0),    // %1
                    "=r"(r1),    // %2
                    "=r"(outptr) // %3
                    : "0"(nn),
                    "1"(r0),
                    "2"(r1),
                    "3"(outptr)
                    : "cc", "memory", "q0", "q1", "q2", "q3");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                float max0 = std::max(r0[0], r0[1]);
                float max1 = std::max(r1[0], r1[1]);

                *outptr = std::max(max0, max1);

                r0 += 2;
                r1 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
