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

static void pooling2x2s2_max_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"

                    "fmax   v0.4s, v0.4s, v1.4s     \n"
                    "fmax   v2.4s, v2.4s, v3.4s     \n"

                    "prfm   pldl1keep, [%1, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                    "fmax   v4.4s, v4.4s, v5.4s     \n"
                    "fmax   v6.4s, v6.4s, v7.4s     \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64 \n"

                    "fmax   v16.4s, v16.4s, v17.4s  \n"
                    "fmax   v18.4s, v18.4s, v19.4s  \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                    "fmax   v20.4s, v20.4s, v21.4s  \n"
                    "fmax   v22.4s, v22.4s, v23.4s  \n"

                    "fmax   v0.4s, v0.4s, v16.4s    \n"
                    "fmax   v1.4s, v2.4s, v18.4s    \n"
                    "fmax   v2.4s, v4.4s, v20.4s    \n"
                    "fmax   v3.4s, v6.4s, v22.4s    \n"

                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1)      // %2
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else  // __aarch64__
                asm volatile(
                    "pld        [%1, #512]      \n"
                    "vldm       %1!, {d0-d7}    \n"

                    "vmax.f32   q0, q0, q1      \n"
                    "vmax.f32   q2, q2, q3      \n"

                    "pld        [%1, #512]      \n"
                    "vldm       %1!, {d8-d15}   \n"

                    "vmax.f32   q4, q4, q5      \n"
                    "vmax.f32   q6, q6, q7      \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d16-d23}  \n"

                    "vmax.f32   q8, q8, q9      \n"
                    "vmax.f32   q10, q10, q11   \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d24-d31}  \n"

                    "vmax.f32   q12, q12, q13   \n"
                    "vmax.f32   q14, q14, q15   \n"

                    "vmax.f32   q0, q0, q8      \n"
                    "vmax.f32   q1, q2, q10     \n"
                    "vmax.f32   q2, q4, q12     \n"
                    "vmax.f32   q3, q6, q14     \n"

                    "vstm       %0!, {d0-d7}    \n"

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
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);

                float32x4_t _max0 = vmaxq_f32(_r00, _r01);
                float32x4_t _max1 = vmaxq_f32(_r10, _r11);
                float32x4_t _max = vmaxq_f32(_max0, _max1);

                vst1q_f32(outptr, _max);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
