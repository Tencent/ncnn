// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void pooling3x3s2_max_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
        float* outptr = top_blob.channel(q);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"

                    "fmax   v16.4s, v0.4s, v1.4s    \n"
                    "fmax   v17.4s, v2.4s, v3.4s    \n"

                    "prfm   pldl1keep, [%1, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                    "fmax   v18.4s, v4.4s, v5.4s    \n"
                    "fmax   v19.4s, v6.4s, v7.4s    \n"

                    "ld1    {v8.4s}, [%1]           \n"

                    "fmax   v20.4s, v16.4s, v2.4s   \n"
                    "fmax   v21.4s, v17.4s, v4.4s   \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                    "fmax   v22.4s, v18.4s, v6.4s   \n"
                    "fmax   v23.4s, v19.4s, v8.4s   \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"

                    "fmax   v16.4s, v0.4s, v1.4s    \n"
                    "fmax   v17.4s, v2.4s, v3.4s    \n"

                    "fmax   v18.4s, v4.4s, v5.4s    \n"
                    "fmax   v19.4s, v6.4s, v7.4s    \n"

                    "ld1    {v8.4s}, [%2]           \n"

                    "fmax   v24.4s, v16.4s, v2.4s   \n"
                    "fmax   v25.4s, v17.4s, v4.4s   \n"

                    "prfm   pldl1keep, [%3, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "fmax   v26.4s, v18.4s, v6.4s   \n"
                    "fmax   v27.4s, v19.4s, v8.4s   \n"

                    "prfm   pldl1keep, [%3, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n"

                    "fmax   v16.4s, v0.4s, v1.4s    \n"
                    "fmax   v17.4s, v2.4s, v3.4s    \n"

                    "fmax   v18.4s, v4.4s, v5.4s    \n"
                    "fmax   v19.4s, v6.4s, v7.4s    \n"

                    "ld1    {v8.4s}, [%3]           \n"

                    "fmax   v28.4s, v16.4s, v2.4s   \n"
                    "fmax   v29.4s, v17.4s, v4.4s   \n"
                    "fmax   v30.4s, v18.4s, v6.4s   \n"
                    "fmax   v31.4s, v19.4s, v8.4s   \n"

                    "fmax   v20.4s, v20.4s, v24.4s  \n"
                    "fmax   v21.4s, v21.4s, v25.4s  \n"
                    "fmax   v22.4s, v22.4s, v26.4s  \n"
                    "fmax   v23.4s, v23.4s, v27.4s  \n"

                    "fmax   v20.4s, v20.4s, v28.4s  \n"
                    "fmax   v21.4s, v21.4s, v29.4s  \n"
                    "fmax   v22.4s, v22.4s, v30.4s  \n"
                    "fmax   v23.4s, v23.4s, v31.4s  \n"

                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2)      // %3
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
                asm volatile(
                    "pld        [%1, #512]      \n"
                    "vldm       %1!, {d0-d7}    \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d8-d15}   \n"

                    "vmax.f32   q0, q0, q4      \n"
                    "vmax.f32   q1, q1, q5      \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d16-d23}  \n"

                    "vmax.f32   q2, q2, q6      \n"
                    "vmax.f32   q3, q3, q7      \n"

                    "vmax.f32   q0, q0, q8      \n"
                    "vmax.f32   q1, q1, q9      \n"

                    "pld        [%1, #512]      \n"
                    "vldm       %1!, {d8-d15}   \n"

                    "vmax.f32   q2, q2, q10     \n"
                    "vmax.f32   q3, q3, q11     \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d16-d23}  \n"

                    "vmax.f32   q4, q4, q8      \n"
                    "vmax.f32   q5, q5, q9      \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d24-d31}  \n"

                    "vmax.f32   q6, q6, q10     \n"
                    "vmax.f32   q7, q7, q11     \n"

                    "vmax.f32   q4, q4, q12     \n"
                    "vmax.f32   q5, q5, q13     \n"

                    "vld1.f32   {d24-d25}, [%1 :128] \n"
                    "vld1.f32   {d26-d27}, [%2 :128] \n"

                    "vmax.f32   q6, q6, q14     \n"
                    "vmax.f32   q7, q7, q15     \n"

                    "vld1.f32   {d28-d29}, [%3 :128] \n"

                    "vmax.f32   q8, q12, q13    \n"
                    "vmax.f32   q8, q8, q14     \n"

                    "vmax.f32   q12, q0, q1     \n"
                    "vmax.f32   q13, q2, q3     \n"
                    "vmax.f32   q14, q4, q5     \n"
                    "vmax.f32   q15, q6, q7     \n"

                    "vmax.f32   q12, q12, q2    \n"
                    "vmax.f32   q13, q13, q4    \n"
                    "vmax.f32   q14, q14, q6    \n"
                    "vmax.f32   q15, q15, q8    \n"

                    "vstm       %0!, {d24-d31}  \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2)      // %3
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
            }
            for (; j + 1 < outw; j += 2)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"

                    "fmax   v16.4s, v0.4s, v4.4s    \n"
                    "fmax   v17.4s, v1.4s, v5.4s    \n"

                    "prfm   pldl1keep, [%3, #512]   \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"

                    "fmax   v18.4s, v2.4s, v6.4s    \n"
                    "fmax   v19.4s, v3.4s, v7.4s    \n"

                    "ld1    {v0.4s}, [%1]           \n"

                    "fmax   v16.4s, v16.4s, v20.4s  \n"
                    "fmax   v17.4s, v17.4s, v21.4s  \n"

                    "ld1    {v1.4s}, [%2]           \n"

                    "fmax   v18.4s, v18.4s, v22.4s  \n"
                    "fmax   v19.4s, v19.4s, v23.4s  \n"

                    "ld1    {v2.4s}, [%3]           \n"

                    "fmax   v3.4s, v0.4s, v1.4s     \n"

                    "fmax   v20.4s, v16.4s, v17.4s  \n"
                    "fmax   v21.4s, v18.4s, v19.4s  \n"

                    "fmax   v3.4s, v3.4s, v2.4s     \n"

                    "fmax   v20.4s, v20.4s, v18.4s  \n"
                    "fmax   v21.4s, v21.4s, v3.4s   \n"

                    "st1    {v20.4s, v21.4s}, [%0], #32 \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2)      // %3
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else  // __aarch64__
                asm volatile(
                    "pld        [%1, #512]      \n"
                    "vldm       %1!, {d0-d7}    \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d8-d15}   \n"

                    "vmax.f32   q12, q0, q4     \n"
                    "vmax.f32   q13, q1, q5     \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d16-d23}  \n"

                    "vmax.f32   q14, q2, q6     \n"
                    "vmax.f32   q15, q3, q7     \n"

                    "vld1.f32   {d0-d1}, [%1 :128] \n"

                    "vmax.f32   q12, q12, q8    \n"
                    "vmax.f32   q13, q13, q9    \n"

                    "vld1.f32   {d2-d3}, [%2 :128] \n"

                    "vmax.f32   q14, q14, q10   \n"
                    "vmax.f32   q15, q15, q11   \n"

                    "vld1.f32   {d4-d5}, [%3 :128] \n"

                    "vmax.f32   q3, q0, q1      \n"

                    "vmax.f32   q4, q12, q13    \n"
                    "vmax.f32   q5, q14, q15    \n"

                    "vmax.f32   q3, q3, q2      \n"

                    "vmax.f32   q4, q4, q14     \n"
                    "vmax.f32   q5, q5, q3      \n"

                    "vst1.f32   {d8-d11}, [%0 :128]! \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2)      // %3
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
            }
            for (; j < outw; j++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r02 = vld1q_f32(r0 + 8);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r12 = vld1q_f32(r1 + 8);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r22 = vld1q_f32(r2 + 8);

                float32x4_t _max0 = vmaxq_f32(vmaxq_f32(_r00, _r01), _r02);
                float32x4_t _max1 = vmaxq_f32(vmaxq_f32(_r10, _r11), _r12);
                float32x4_t _max2 = vmaxq_f32(vmaxq_f32(_r20, _r21), _r22);

                float32x4_t _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);

                vst1q_f32(outptr, _max);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
