// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void pooling2x2s2_max_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q);

        const unsigned short* r0 = img0.row<const unsigned short>(0);
        const unsigned short* r1 = img0.row<const unsigned short>(1);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]   \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"

                    "shll   v0.4s, v0.4h, #16       \n"
                    "shll   v1.4s, v1.4h, #16       \n"
                    "shll   v2.4s, v2.4h, #16       \n"
                    "shll   v3.4s, v3.4h, #16       \n"

                    "fmax   v0.4s, v0.4s, v1.4s     \n"
                    "fmax   v2.4s, v2.4s, v3.4s     \n"

                    "prfm   pldl1keep, [%1, #256]   \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32 \n"

                    "shll   v4.4s, v4.4h, #16       \n"
                    "shll   v5.4s, v5.4h, #16       \n"
                    "shll   v6.4s, v6.4h, #16       \n"
                    "shll   v7.4s, v7.4h, #16       \n"

                    "fmax   v4.4s, v4.4s, v5.4s     \n"
                    "fmax   v6.4s, v6.4s, v7.4s     \n"

                    "prfm   pldl1keep, [%2, #256]   \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%2], #32 \n"

                    "shll   v16.4s, v16.4h, #16     \n"
                    "shll   v17.4s, v17.4h, #16     \n"
                    "shll   v18.4s, v18.4h, #16     \n"
                    "shll   v19.4s, v19.4h, #16     \n"

                    "fmax   v16.4s, v16.4s, v17.4s  \n"
                    "fmax   v18.4s, v18.4s, v19.4s  \n"

                    "prfm   pldl1keep, [%2, #256]   \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n"

                    "shll   v20.4s, v20.4h, #16     \n"
                    "shll   v21.4s, v21.4h, #16     \n"
                    "shll   v22.4s, v22.4h, #16     \n"
                    "shll   v23.4s, v23.4h, #16     \n"

                    "fmax   v20.4s, v20.4s, v21.4s  \n"
                    "fmax   v22.4s, v22.4s, v23.4s  \n"

                    "fmax   v0.4s, v0.4s, v16.4s    \n"
                    "fmax   v1.4s, v2.4s, v18.4s    \n"
                    "fmax   v2.4s, v4.4s, v20.4s    \n"
                    "fmax   v3.4s, v6.4s, v22.4s    \n"

                    "shrn   v0.4h, v0.4s, #16       \n"
                    "shrn   v1.4h, v1.4s, #16       \n"
                    "shrn   v2.4h, v2.4s, #16       \n"
                    "shrn   v3.4h, v3.4s, #16       \n"

                    "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0], #32 \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1)      // %2
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else  // __aarch64__
                asm volatile(
                    "pld        [%1, #256]      \n"
                    "vld1.u16   {d4-d7}, [%1]!  \n"

                    "vshll.u16  q0, d4, #16     \n"
                    "vshll.u16  q1, d5, #16     \n"
                    "vshll.u16  q2, d6, #16     \n"
                    "vshll.u16  q3, d7, #16     \n"

                    "vmax.f32   q0, q0, q1      \n"
                    "vmax.f32   q2, q2, q3      \n"

                    "pld        [%1, #256]      \n"
                    "vld1.u16   {d12-d15}, [%1]! \n"

                    "vshll.u16  q4, d12, #16    \n"
                    "vshll.u16  q5, d13, #16    \n"
                    "vshll.u16  q6, d14, #16    \n"
                    "vshll.u16  q7, d15, #16    \n"

                    "vmax.f32   q4, q4, q5      \n"
                    "vmax.f32   q6, q6, q7      \n"

                    "pld        [%2, #256]      \n"
                    "vld1.u16   {d20-d23}, [%2]! \n"

                    "vshll.u16  q8, d20, #16    \n"
                    "vshll.u16  q9, d21, #16    \n"
                    "vshll.u16  q10, d22, #16   \n"
                    "vshll.u16  q11, d23, #16   \n"

                    "vmax.f32   q8, q8, q9      \n"
                    "vmax.f32   q10, q10, q11   \n"

                    "pld        [%2, #256]      \n"
                    "vld1.u16   {d28-d31}, [%2]! \n"

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"
                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmax.f32   q12, q12, q13   \n"
                    "vmax.f32   q14, q14, q15   \n"

                    "vmax.f32   q0, q0, q8      \n"
                    "vmax.f32   q1, q2, q10     \n"
                    "vmax.f32   q2, q4, q12     \n"
                    "vmax.f32   q3, q6, q14     \n"

                    "vshrn.u32  d0, q0, #16     \n"
                    "vshrn.u32  d1, q1, #16     \n"
                    "vshrn.u32  d2, q2, #16     \n"
                    "vshrn.u32  d3, q3, #16     \n"

                    "vst1.u16   {d0-d3}, [%0]!  \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1)      // %2
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
            }
            for (; j < outw; j++)
            {
                float32x4_t _r00 = bfloat2float(vld1_u16(r0));
                float32x4_t _r01 = bfloat2float(vld1_u16(r0 + 4));
                float32x4_t _r10 = bfloat2float(vld1_u16(r1));
                float32x4_t _r11 = bfloat2float(vld1_u16(r1 + 4));

                float32x4_t _max0 = vmaxq_f32(_r00, _r01);
                float32x4_t _max1 = vmaxq_f32(_r10, _r11);
                float32x4_t _max = vmaxq_f32(_max0, _max1);

                vst1_u16(outptr, float2bfloat(_max));

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
