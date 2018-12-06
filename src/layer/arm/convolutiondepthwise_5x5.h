// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convdw5x5s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g*25;

        float* outptr = out;
        float* outptr2 = outptr + outw;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;
        const float* r3 = img0 + w*3;
        const float* r4 = img0 + w*4;
        const float* r5 = img0 + w*5;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 5;
        const float* k2 = kernel0 + 10;
        const float* k3 = kernel0 + 15;
        const float* k4 = kernel0 + 20;

#if __ARM_NEON
        float32x4_t _k0123 = vld1q_f32(kernel0);
        float32x4_t _k4567 = vld1q_f32(kernel0+4);
        float32x4_t _k891011 = vld1q_f32(kernel0+8);
        float32x4_t _k12131415 = vld1q_f32(kernel0+12);
        float32x4_t _k16171819 = vld1q_f32(kernel0+16);
        float32x4_t _k20212223 = vld1q_f32(kernel0+20);
        float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);

        float32x4_t _bias0 = vdupq_n_f32(bias0);
#endif // __ARM_NEON

        int i = 0;

        for (; i+1 < outh; i+=2)
        {
#if __ARM_NEON
#if __aarch64__
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int nn = outw >> 2;
            int remain = outw & 3;
#endif // __aarch64__
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                // r1
                "prfm   pldl1keep, [%4, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%4]  \n"// v16 v17 v18 = r10 r14 r18

                "mov    v8.16b, %25.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %25.16b                 \n"// v9 = _bias0

                "0:                                     \n"

                "mov    v10.16b, %25.16b                \n"// v10 = _bias0
                "mov    v11.16b, %25.16b                \n"// v11 = _bias0

                "fmla   v8.4s, v16.4s, %19.s[1]         \n"
                "fmla   v10.4s, v16.4s, %18.s[0]        \n"

                "ext    v19.16b, v16.16b, v17.16b, #4   \n"// r11

                "fmla   v9.4s, v17.4s, %19.s[1]         \n"
                "fmla   v11.4s, v17.4s, %18.s[0]        \n"

                "ext    v20.16b, v17.16b, v18.16b, #4   \n"// r15

                "fmla   v8.4s, v17.4s, %20.s[1]         \n"
                "fmla   v10.4s, v17.4s, %19.s[0]        \n"

                "ext    v21.16b, v16.16b, v17.16b, #8   \n"// r12

                "fmla   v9.4s, v18.4s, %20.s[1]         \n"
                "fmla   v11.4s, v18.4s, %19.s[0]        \n"

                "ext    v22.16b, v17.16b, v18.16b, #8   \n"// r16

                "fmla   v8.4s, v19.4s, %19.s[2]         \n"
                "fmla   v10.4s, v19.4s, %18.s[1]        \n"

                "ext    v19.16b, v16.16b, v17.16b, #12  \n"// r13

                "fmla   v9.4s, v20.4s, %19.s[2]         \n"
                "fmla   v11.4s, v20.4s, %18.s[1]        \n"

                "ext    v20.16b, v17.16b, v18.16b, #12  \n"// r17

                "fmla   v8.4s, v21.4s, %19.s[3]         \n"
                "fmla   v10.4s, v21.4s, %18.s[2]        \n"

                "add    %4, %4, #32                     \n"

                "fmla   v9.4s, v22.4s, %19.s[3]         \n"
                "fmla   v11.4s, v22.4s, %18.s[2]        \n"

                // r2
                "prfm   pldl1keep, [%5, #384]           \n"
                "ld1    {v12.4s, v13.4s, v14.4s}, [%5]  \n"// v12 v13 v14 = r20 r24 r28

                "fmla   v8.4s, v19.4s, %20.s[0]         \n"
                "fmla   v10.4s, v19.4s, %18.s[3]        \n"
                "fmla   v9.4s, v20.4s, %20.s[0]         \n"
                "fmla   v11.4s, v20.4s, %18.s[3]        \n"

                "add    %5, %5, #32                     \n"

                "fmla   v8.4s, v12.4s, %20.s[2]         \n"
                "fmla   v10.4s, v12.4s, %19.s[1]        \n"

                "ext    v21.16b, v12.16b, v13.16b, #4   \n"// r21

                "fmla   v9.4s, v13.4s, %20.s[2]         \n"
                "fmla   v11.4s, v13.4s, %19.s[1]        \n"

                "ext    v22.16b, v13.16b, v14.16b, #4   \n"// r25

                "fmla   v8.4s, v13.4s, %21.s[2]         \n"
                "fmla   v10.4s, v13.4s, %20.s[1]        \n"

                "ext    v19.16b, v12.16b, v13.16b, #8   \n"// r22

                "fmla   v9.4s, v14.4s, %21.s[2]         \n"
                "fmla   v11.4s, v14.4s, %20.s[1]        \n"

                "ext    v20.16b, v13.16b, v14.16b, #8   \n"// r26

                "fmla   v8.4s, v21.4s, %20.s[3]         \n"
                "fmla   v10.4s, v21.4s, %19.s[2]        \n"

                "ext    v21.16b, v12.16b, v13.16b, #12  \n"// r23

                "fmla   v9.4s, v22.4s, %20.s[3]         \n"
                "fmla   v11.4s, v22.4s, %19.s[2]        \n"

                "ext    v22.16b, v13.16b, v14.16b, #12  \n"// r27

                "fmla   v8.4s, v19.4s, %21.s[0]         \n"
                "fmla   v10.4s, v19.4s, %19.s[3]        \n"
                "fmla   v9.4s, v20.4s, %21.s[0]         \n"
                "fmla   v11.4s, v20.4s, %19.s[3]        \n"

                // r3
                "prfm   pldl1keep, [%6, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n"// v16 v17 v18 = r30 r34 r38

                "fmla   v8.4s, v21.4s, %21.s[1]         \n"
                "fmla   v10.4s, v21.4s, %20.s[0]        \n"
                "fmla   v9.4s, v22.4s, %21.s[1]         \n"
                "fmla   v11.4s, v22.4s, %20.s[0]        \n"

                "add    %6, %6, #32                     \n"

                "fmla   v8.4s, v16.4s, %21.s[3]         \n"
                "fmla   v10.4s, v16.4s, %20.s[2]        \n"

                "ext    v19.16b, v16.16b, v17.16b, #4   \n"// r31

                "fmla   v9.4s, v17.4s, %21.s[3]         \n"
                "fmla   v11.4s, v17.4s, %20.s[2]        \n"

                "ext    v20.16b, v17.16b, v18.16b, #4   \n"// r35

                "fmla   v8.4s, v17.4s, %22.s[3]         \n"
                "fmla   v10.4s, v17.4s, %21.s[2]        \n"

                "ext    v21.16b, v16.16b, v17.16b, #8   \n"// r32

                "fmla   v9.4s, v18.4s, %22.s[3]         \n"
                "fmla   v11.4s, v18.4s, %21.s[2]        \n"

                "ext    v22.16b, v17.16b, v18.16b, #8   \n"// r36

                "fmla   v8.4s, v19.4s, %22.s[0]         \n"
                "fmla   v10.4s, v19.4s, %20.s[3]        \n"

                "ext    v19.16b, v16.16b, v17.16b, #12  \n"// r33

                "fmla   v9.4s, v20.4s, %22.s[0]         \n"
                "fmla   v11.4s, v20.4s, %20.s[3]        \n"

                "ext    v20.16b, v17.16b, v18.16b, #12  \n"// r37

                "fmla   v8.4s, v21.4s, %22.s[1]         \n"
                "fmla   v10.4s, v21.4s, %21.s[0]        \n"
                "fmla   v9.4s, v22.4s, %22.s[1]         \n"
                "fmla   v11.4s, v22.4s, %21.s[0]        \n"

                // r4
                "prfm   pldl1keep, [%7, #384]           \n"
                "ld1    {v12.4s, v13.4s, v14.4s}, [%7]  \n"// v12 v13 v14 = r40 r44 r48

                "fmla   v8.4s, v19.4s, %22.s[2]         \n"
                "fmla   v10.4s, v19.4s, %21.s[1]        \n"

                "add    %7, %7, #32                     \n"

                "fmla   v9.4s, v20.4s, %22.s[2]         \n"
                "fmla   v11.4s, v20.4s, %21.s[1]        \n"

                "ext    v21.16b, v12.16b, v13.16b, #4   \n"// r41

                "fmla   v8.4s, v12.4s, %23.s[0]         \n"
                "fmla   v10.4s, v12.4s, %21.s[3]        \n"

                "ext    v22.16b, v13.16b, v14.16b, #4   \n"// r45

                "fmla   v9.4s, v13.4s, %23.s[0]         \n"
                "fmla   v11.4s, v13.4s, %21.s[3]        \n"

                "ext    v19.16b, v12.16b, v13.16b, #8   \n"// r42

                "fmla   v8.4s, v13.4s, %24.s[0]         \n"
                "fmla   v10.4s, v13.4s, %22.s[3]        \n"

                "ext    v20.16b, v13.16b, v14.16b, #8   \n"// r46

                "fmla   v9.4s, v14.4s, %24.s[0]         \n"
                "fmla   v11.4s, v14.4s, %22.s[3]        \n"

                // r0 and r5
                "prfm   pldl1keep, [%3, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%3]  \n"// v16 v17 v18 = r00 r04 r08

                "fmla   v8.4s, v21.4s, %23.s[1]         \n"
                "fmla   v10.4s, v21.4s, %22.s[0]        \n"

                "ext    v21.16b, v12.16b, v13.16b, #12  \n"// r43

                "fmla   v9.4s, v22.4s, %23.s[1]         \n"
                "fmla   v11.4s, v22.4s, %22.s[0]        \n"

                "ext    v22.16b, v13.16b, v14.16b, #12  \n"// r47

                "fmla   v8.4s, v19.4s, %23.s[2]         \n"
                "fmla   v10.4s, v19.4s, %22.s[1]        \n"

                "prfm   pldl1keep, [%8, #384]           \n"
                "ld1    {v12.4s, v13.4s, v14.4s}, [%8]  \n"// v12 v13 v14 = r50 r54 r58

                "fmla   v9.4s, v20.4s, %23.s[2]         \n"
                "fmla   v11.4s, v20.4s, %22.s[1]        \n"

                "ext    v19.16b, v16.16b, v17.16b, #4   \n"// r01

                "fmla   v8.4s, v21.4s, %23.s[3]         \n"
                "fmla   v10.4s, v21.4s, %22.s[2]        \n"

                "ext    v23.16b, v12.16b, v13.16b, #4   \n"// r51

                "fmla   v9.4s, v22.4s, %23.s[3]         \n"
                "fmla   v11.4s, v22.4s, %22.s[2]        \n"

                "ext    v20.16b, v17.16b, v18.16b, #4   \n"// r05

                "fmla   v8.4s, v16.4s, %18.s[0]         \n"
                "fmla   v10.4s, v12.4s, %23.s[0]        \n"

                "ext    v24.16b, v13.16b, v14.16b, #4   \n"// r55

                "fmla   v9.4s, v17.4s, %18.s[0]         \n"
                "fmla   v11.4s, v13.4s, %23.s[0]        \n"

                "ext    v21.16b, v16.16b, v17.16b, #8   \n"// r02

                "fmla   v8.4s, v17.4s, %19.s[0]         \n"
                "fmla   v10.4s, v13.4s, %24.s[0]        \n"

                "ext    v25.16b, v12.16b, v13.16b, #8   \n"// r52

                "fmla   v9.4s, v18.4s, %19.s[0]         \n"
                "fmla   v11.4s, v14.4s, %24.s[0]        \n"

                "ext    v22.16b, v17.16b, v18.16b, #8   \n"// r06

                "fmla   v8.4s, v19.4s, %18.s[1]         \n"
                "fmla   v10.4s, v23.4s, %23.s[1]        \n"

                "ext    v26.16b, v13.16b, v14.16b, #8   \n"// r56

                "fmla   v9.4s, v20.4s, %18.s[1]         \n"
                "fmla   v11.4s, v24.4s, %23.s[1]        \n"

                "ext    v19.16b, v16.16b, v17.16b, #12  \n"// r03

                "fmla   v8.4s, v21.4s, %18.s[2]         \n"
                "fmla   v10.4s, v25.4s, %23.s[2]        \n"

                "ext    v23.16b, v12.16b, v13.16b, #12  \n"// r53

                "fmla   v9.4s, v22.4s, %18.s[2]         \n"
                "fmla   v11.4s, v26.4s, %23.s[2]        \n"

                "ext    v20.16b, v17.16b, v18.16b, #12  \n"// r07

                "fmla   v8.4s, v19.4s, %18.s[3]         \n"
                "fmla   v10.4s, v23.4s, %23.s[3]        \n"

                "ext    v24.16b, v13.16b, v14.16b, #12  \n"// r57

                "fmla   v9.4s, v20.4s, %18.s[3]         \n"

                "add    %3, %3, #32                     \n"

                "fmla   v11.4s, v24.4s, %23.s[3]        \n"

                "add    %8, %8, #32                     \n"

                // r1
                "prfm   pldl1keep, [%4, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%4]  \n"// v16 v17 v18 = r10 r14 r18

                "subs   %w0, %w0, #1                    \n"

                "st1    {v8.4s, v9.4s}, [%1], #32       \n"

                "mov    v8.16b, %25.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %25.16b                 \n"// v9 = _bias0

                "st1    {v10.4s, v11.4s}, [%2], #32     \n"

                "bne    0b                              \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(outptr2),    // %2
                  "=r"(r0),         // %3
                  "=r"(r1),         // %4
                  "=r"(r2),         // %5
                  "=r"(r3),         // %6
                  "=r"(r4),         // %7
                  "=r"(r5)          // %8
                : "0"(nn),
                  "1"(outptr),
                  "2"(outptr2),
                  "3"(r0),
                  "4"(r1),
                  "5"(r2),
                  "6"(r3),
                  "7"(r4),
                  "8"(r5),
                  "w"(_k0123),      // %18
                  "w"(_k4567),      // %19
                  "w"(_k891011),    // %20
                  "w"(_k12131415),  // %21
                  "w"(_k16171819),  // %22
                  "w"(_k20212223),  // %23
                  "w"(_k24242424),  // %24
                  "w"(_bias0)       // %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26"
            );
            }

            if (remain >= 4)
            {
                remain -= 4;
                asm volatile(
                    // r1
                    "prfm   pldl1keep, [%3, #256]           \n"
                    "ld1    {v12.4s, v13.4s}, [%3]          \n"// v12 v13 = r10 r14

                    "mov    v8.16b, %23.16b                 \n"// v8 = _bias0
                    "mov    v9.16b, %23.16b                 \n"// v9 = _bias0

                    "fmul   v10.4s, v12.4s, %17.s[1]        \n"
                    "fmul   v11.4s, v12.4s, %16.s[0]        \n"

                    "ext    v21.16b, v12.16b, v13.16b, #4   \n"// r11

                    "fmla   v8.4s, v13.4s, %18.s[1]         \n"
                    "fmla   v9.4s, v13.4s, %17.s[0]         \n"

                    "ext    v22.16b, v12.16b, v13.16b, #8   \n"// r12

                    "fmla   v10.4s, v21.4s, %17.s[2]        \n"
                    "fmla   v11.4s, v21.4s, %16.s[1]        \n"

                    "ext    v23.16b, v12.16b, v13.16b, #12  \n"// r13

                    "fmla   v8.4s, v22.4s, %17.s[3]         \n"
                    "fmla   v9.4s, v22.4s, %16.s[2]         \n"

                    // r2
                    "prfm   pldl1keep, [%4, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%4]          \n"// v16 v17 = r20 r24

                    "fmla   v10.4s, v23.4s, %18.s[0]        \n"
                    "fmla   v11.4s, v23.4s, %16.s[3]        \n"

                    "add    %4, %4, #16                     \n"

                    "fmla   v8.4s, v16.4s, %18.s[2]         \n"
                    "fmla   v9.4s, v16.4s, %17.s[1]         \n"

                    "ext    v18.16b, v16.16b, v17.16b, #4   \n"// r21

                    "fmla   v10.4s, v17.4s, %19.s[2]        \n"
                    "fmla   v11.4s, v17.4s, %18.s[1]        \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"// r22

                    "fmla   v8.4s, v18.4s, %18.s[3]         \n"
                    "fmla   v9.4s, v18.4s, %17.s[2]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #12  \n"// r23

                    "fmla   v10.4s, v19.4s, %19.s[0]        \n"
                    "fmla   v11.4s, v19.4s, %17.s[3]        \n"

                    // r3
                    "prfm   pldl1keep, [%5, #256]           \n"
                    "ld1    {v12.4s, v13.4s}, [%5]          \n"// v12 v13 = r30 r34

                    "fmla   v8.4s, v20.4s, %19.s[1]         \n"
                    "fmla   v9.4s, v20.4s, %18.s[0]         \n"

                    "add    %5, %5, #16                     \n"

                    "fmla   v10.4s, v12.4s, %19.s[3]        \n"
                    "fmla   v11.4s, v12.4s, %18.s[2]        \n"

                    "ext    v21.16b, v12.16b, v13.16b, #4   \n"// r31

                    "fmla   v8.4s, v13.4s, %20.s[3]         \n"
                    "fmla   v9.4s, v13.4s, %19.s[2]         \n"

                    "ext    v22.16b, v12.16b, v13.16b, #8   \n"// r32

                    "fmla   v10.4s, v21.4s, %20.s[0]        \n"
                    "fmla   v11.4s, v21.4s, %18.s[3]        \n"

                    "ext    v23.16b, v12.16b, v13.16b, #12  \n"// r33

                    "fmla   v8.4s, v22.4s, %20.s[1]         \n"
                    "fmla   v9.4s, v22.4s, %19.s[0]         \n"

                    // r4
                    "prfm   pldl1keep, [%6, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%6]          \n"// v16 v17 = r40 r44

                    "fmla   v10.4s, v23.4s, %20.s[2]        \n"
                    "fmla   v11.4s, v23.4s, %19.s[1]        \n"

                    "add    %6, %6, #16                     \n"

                    "fmla   v8.4s, v16.4s, %21.s[0]         \n"
                    "fmla   v9.4s, v16.4s, %19.s[3]         \n"

                    "ext    v18.16b, v16.16b, v17.16b, #4   \n"// r41

                    "fmla   v10.4s, v17.4s, %22.s[0]        \n"
                    "fmla   v11.4s, v17.4s, %20.s[3]        \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"// r42

                    "fmla   v8.4s, v18.4s, %21.s[1]         \n"
                    "fmla   v9.4s, v18.4s, %20.s[0]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #12  \n"// r43

                    "fmla   v10.4s, v19.4s, %21.s[2]        \n"
                    "fmla   v11.4s, v19.4s, %20.s[1]        \n"

                    // r0
                    "prfm   pldl1keep, [%2, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%2]          \n"// v16 v17 = r00 r04

                    "fmla   v8.4s, v20.4s, %21.s[3]         \n"
                    "fmla   v9.4s, v20.4s, %20.s[2]         \n"

                    // r5
                    "prfm   pldl1keep, [%7, #256]           \n"
                    "ld1    {v12.4s, v13.4s}, [%7]          \n"// v12 v13 = r50 r54

                    "fmla   v10.4s, v16.4s, %16.s[0]        \n"
                    "fmla   v11.4s, v12.4s, %21.s[0]        \n"

                    "ext    v18.16b, v16.16b, v17.16b, #4   \n"// r01

                    "fmla   v8.4s, v17.4s, %17.s[0]         \n"

                    "ext    v21.16b, v12.16b, v13.16b, #4   \n"// r51

                    "fmla   v9.4s, v13.4s, %22.s[0]         \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"// r02

                    "fmla   v10.4s, v18.4s, %16.s[1]        \n"

                    "ext    v22.16b, v12.16b, v13.16b, #8   \n"// r52

                    "fmla   v11.4s, v21.4s, %21.s[1]        \n"

                    "ext    v20.16b, v16.16b, v17.16b, #12  \n"// r03

                    "fmla   v8.4s, v19.4s, %16.s[2]         \n"

                    "ext    v23.16b, v12.16b, v13.16b, #12  \n"// r53

                    "fmla   v9.4s, v22.4s, %21.s[2]         \n"

                    "add    %3, %3, #16                     \n"

                    "fmla   v10.4s, v20.4s, %16.s[3]        \n"
                    "fmla   v11.4s, v23.4s, %21.s[3]        \n"

                    "add    %2, %2, #16                     \n"

                    "fadd   v8.4s, v8.4s, v10.4s            \n"
                    "fadd   v9.4s, v9.4s, v11.4s            \n"

                    "add    %7, %7, #16                     \n"

                    "st1    {v8.4s}, [%0], #16              \n"
                    "st1    {v9.4s}, [%1], #16              \n"

                    : "=r"(outptr),     // %0
                      "=r"(outptr2),    // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5)          // %7
                    : "0"(outptr),
                      "1"(outptr2),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "w"(_k0123),      // %16
                      "w"(_k4567),      // %17
                      "w"(_k891011),    // %18
                      "w"(_k12131415),  // %19
                      "w"(_k16171819),  // %20
                      "w"(_k20212223),  // %21
                      "w"(_k24242424),  // %22
                      "w"(_bias0)       // %23
                    : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                // r1
                "pld        [%4, #256]          \n"
                "vld1.f32   {d28-d31}, [%4]     \n"// q14 q15 = r10 r14

                "vmov       q8, %q25            \n"// q8 = _bias0

                "0:                             \n"

                "vmov       q9, %q25            \n"// q9 = _bias0

                "vmla.f32   q8, q14, %e19[1]    \n"
                "vmla.f32   q9, q14, %e18[0]    \n"

                "vext.32    q12, q14, q15, #1   \n"// r11

                "vmla.f32   q8, q15, %e20[1]    \n"
                "vmla.f32   q9, q15, %e19[0]    \n"

                "vext.32    q13, q14, q15, #2   \n"// r12

                "vmla.f32   q8, q12, %f19[0]    \n"
                "vmla.f32   q9, q12, %e18[1]    \n"

                "vext.32    q12, q14, q15, #3   \n"// r13

                "vmla.f32   q8, q13, %f19[1]    \n"
                "vmla.f32   q9, q13, %f18[0]    \n"

                // r2
                "pld        [%5, #256]          \n"
                "vld1.f32   {d20-d23}, [%5]     \n"// q10 q11 = r20 r24

                "vmla.f32   q8, q12, %e20[0]    \n"
                "vmla.f32   q9, q12, %f18[1]    \n"

                "add        %5, #16             \n"

                "vmla.f32   q8, q10, %f20[0]    \n"
                "vmla.f32   q9, q10, %e19[1]    \n"

                "vext.32    q12, q10, q11, #1   \n"// r21

                "vmla.f32   q8, q11, %f21[0]    \n"
                "vmla.f32   q9, q11, %e20[1]    \n"

                "vext.32    q13, q10, q11, #2   \n"// r22

                "vmla.f32   q8, q12, %f20[1]    \n"
                "vmla.f32   q9, q12, %f19[0]    \n"

                "vext.32    q12, q10, q11, #3   \n"// r23

                "vmla.f32   q8, q13, %e21[0]    \n"
                "vmla.f32   q9, q13, %f19[1]    \n"

                // r3
                "pld        [%6, #256]          \n"
                "vld1.f32   {d28-d31}, [%6]     \n"// q14 q15 = r30 r34

                "vmla.f32   q8, q12, %e21[1]    \n"
                "vmla.f32   q9, q12, %e20[0]    \n"

                "add        %6, #16             \n"

                "vmla.f32   q8, q14, %f21[1]    \n"
                "vmla.f32   q9, q14, %f20[0]    \n"

                "vext.32    q12, q14, q15, #1   \n"// r31

                "vmla.f32   q8, q15, %f22[1]    \n"
                "vmla.f32   q9, q15, %f21[0]    \n"

                "vext.32    q13, q14, q15, #2   \n"// r32

                "vmla.f32   q8, q12, %e22[0]    \n"
                "vmla.f32   q9, q12, %f20[1]    \n"

                "vext.32    q12, q14, q15, #3   \n"// r33

                "vmla.f32   q8, q13, %e22[1]    \n"
                "vmla.f32   q9, q13, %e21[0]    \n"

                // r4
                "pld        [%7, #256]          \n"
                "vld1.f32   {d20-d23}, [%7]     \n"// q10 q11 = r40 r44

                "vmla.f32   q8, q12, %f22[0]    \n"
                "vmla.f32   q9, q12, %e21[1]    \n"

                "add        %7, #16             \n"

                "vmla.f32   q8, q10, %e23[0]    \n"
                "vmla.f32   q9, q10, %f21[1]    \n"

                "vext.32    q12, q10, q11, #1   \n"// r41

                "vmla.f32   q8, q11, %e24[0]    \n"
                "vmla.f32   q9, q11, %f22[1]    \n"

                "vext.32    q13, q10, q11, #2   \n"// r42

                "vmla.f32   q8, q12, %e23[1]    \n"
                "vmla.f32   q9, q12, %e22[0]    \n"

                "vext.32    q12, q10, q11, #3   \n"// r43

                "vmla.f32   q8, q13, %f23[0]    \n"
                "vmla.f32   q9, q13, %e22[1]    \n"

                // r0 and r5
                "pld        [%3, #256]          \n"
                "vld1.f32   {d20-d23}, [%3]     \n"// q10 q11 = r00 r04

                "vmla.f32   q8, q12, %f23[1]    \n"
                "vmla.f32   q9, q12, %f22[0]    \n"

                // r5
                "pld        [%8, #256]          \n"
                "vld1.f32   {d28-d31}, [%8]     \n"// q14 q15 = r50 r54

                "vmla.f32   q8, q10, %e18[0]    \n"
                "vmla.f32   q9, q14, %e23[0]    \n"

                "vext.32    q12, q10, q11, #1   \n"// r01

                "vmla.f32   q8, q11, %e19[0]    \n"
                "vmla.f32   q9, q15, %e24[0]    \n"

                "vext.32    q13, q14, q15, #1   \n"// r51

                "vmla.f32   q8, q12, %e18[1]    \n"

                "vext.32    q12, q10, q11, #2   \n"// r02

                "vmla.f32   q9, q13, %e23[1]    \n"

                "vext.32    q13, q14, q15, #2   \n"// r52

                "vmla.f32   q8, q12, %f18[0]    \n"

                "vext.32    q12, q10, q11, #3   \n"// r03

                "vmla.f32   q9, q13, %f23[0]    \n"

                "vext.32    q13, q14, q15, #3   \n"// r33

                "vmla.f32   q8, q12, %f18[1]    \n"

                "add        %3, #16             \n"

                "vmla.f32   q9, q13, %f23[1]    \n"

                "add        %4, #16             \n"

                // r1
                "pld        [%4, #256]          \n"
                "vld1.f32   {d28-d31}, [%4]     \n"// q14 q15 = r10 r14

                "add        %8, #16             \n"

                "vst1.f32   {d16-d17}, [%1]!    \n"

                "vmov       q8, %q25            \n"// q8 = _bias0

                "subs       %0, #1              \n"

                "vst1.f32   {d18-d19}, [%2]!    \n"

                "bne        0b                  \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(outptr2),    // %2
                  "=r"(r0),         // %3
                  "=r"(r1),         // %4
                  "=r"(r2),         // %5
                  "=r"(r3),         // %6
                  "=r"(r4),         // %7
                  "=r"(r5)          // %8
                : "0"(nn),
                  "1"(outptr),
                  "2"(outptr2),
                  "3"(r0),
                  "4"(r1),
                  "5"(r2),
                  "6"(r3),
                  "7"(r4),
                  "8"(r5),
                  "w"(_k0123),      // %18
                  "w"(_k4567),      // %19
                  "w"(_k891011),    // %20
                  "w"(_k12131415),  // %21
                  "w"(_k16171819),  // %22
                  "w"(_k20212223),  // %23
                  "w"(_k24242424),  // %24
                  "w"(_bias0)       // %25
                : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float sum = bias0;
                float sum2 = bias0;
#if __ARM_NEON
                // TODO neon assembly optimize
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _k1 = vld1q_f32(k1);
                float32x4_t _sum = vmulq_f32(_r1, _k1);
                float32x4_t _sum2 = vmulq_f32(_r1, _k0123);

                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _k2 = vld1q_f32(k2);
                _sum = vmlaq_f32(_sum, _r2, _k2);
                _sum2 = vmlaq_f32(_sum2, _r2, _k1);

                float32x4_t _r3 = vld1q_f32(r3);
                float32x4_t _k3 = vld1q_f32(k3);
                _sum = vmlaq_f32(_sum, _r3, _k3);
                _sum2 = vmlaq_f32(_sum2, _r3, _k2);

                float32x4_t _r4 = vld1q_f32(r4);
                _sum = vmlaq_f32(_sum, _r4, _k20212223);
                _sum2 = vmlaq_f32(_sum2, _r4, _k3);

                float32x4_t _r0 = vld1q_f32(r0);
                _sum = vmlaq_f32(_sum, _r0, _k0123);
                float32x4_t _r5 = vld1q_f32(r5);
                _sum2 = vmlaq_f32(_sum2, _r5, _k20212223);

                float32x4_t _k_t4;
                _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                float32x4_t _r_t4;

                _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                sum += r4[4] * k4[4];

                _r_t4 = vextq_f32(_r_t4, _r_t4, 1);
                _r_t4 = vsetq_lane_f32(r4[4], _r_t4, 3);
                _sum2 = vmlaq_f32(_sum2, _r_t4, _k_t4);

                sum2 += r5[4] * k4[4];

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
                float32x2_t _ss_ss2 = vpadd_f32(_ss, _ss2);

                sum += vget_lane_f32(_ss_ss2, 0);
                sum2 += vget_lane_f32(_ss_ss2, 1);
#else
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];

                sum2 += r1[0] * k0[0];
                sum2 += r1[1] * k0[1];
                sum2 += r1[2] * k0[2];
                sum2 += r1[3] * k0[3];
                sum2 += r1[4] * k0[4];

                sum2 += r2[0] * k1[0];
                sum2 += r2[1] * k1[1];
                sum2 += r2[2] * k1[2];
                sum2 += r2[3] * k1[3];
                sum2 += r2[4] * k1[4];

                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];
                sum2 += r3[3] * k2[3];
                sum2 += r3[4] * k2[4];

                sum2 += r4[0] * k3[0];
                sum2 += r4[1] * k3[1];
                sum2 += r4[2] * k3[2];
                sum2 += r4[3] * k3[3];
                sum2 += r4[4] * k3[4];

                sum2 += r5[0] * k4[0];
                sum2 += r5[1] * k4[1];
                sum2 += r5[2] * k4[2];
                sum2 += r5[3] * k4[3];
                sum2 += r5[4] * k4[4];
#endif // __ARM_NEON
                *outptr = sum;
                *outptr2 = sum2;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                outptr++;
                outptr2++;
            }

            r0 += 4 + w;
            r1 += 4 + w;
            r2 += 4 + w;
            r3 += 4 + w;
            r4 += 4 + w;
            r5 += 4 + w;

            outptr += outw;
            outptr2 += outw;
        }

        for (; i < outh; i++)
        {
#if __ARM_NEON
#if __aarch64__
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int nn = outw >> 2;
            int remain = outw & 3;
#endif // __aarch64__
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                // v10 v11
                // r0
                "prfm   pldl1keep, [%2, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%2]  \n"// v16 v17 v18 = r00 r04 r08

                "mov    v8.16b, %21.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %21.16b                 \n"// v9 = _bias0

                "0:                                     \n"

                "fmul   v10.4s, v16.4s, %14.s[0]         \n"

                "ext    v19.16b, v16.16b, v17.16b, #4   \n"// r01

                "fmul   v11.4s, v17.4s, %14.s[0]         \n"

                "ext    v20.16b, v17.16b, v18.16b, #4   \n"// r05

                "fmla   v8.4s, v17.4s, %15.s[0]         \n"

                "ext    v21.16b, v16.16b, v17.16b, #8   \n"// r02

                "fmla   v9.4s, v18.4s, %15.s[0]         \n"

                "ext    v22.16b, v17.16b, v18.16b, #8   \n"// r06

                "fmla   v10.4s, v19.4s, %14.s[1]         \n"

                "ext    v19.16b, v16.16b, v17.16b, #12  \n"// r03

                "fmla   v11.4s, v20.4s, %14.s[1]         \n"

                "ext    v20.16b, v17.16b, v18.16b, #12  \n"// r07

                "fmla   v8.4s, v21.4s, %14.s[2]         \n"
                "fmla   v9.4s, v22.4s, %14.s[2]         \n"

                // r1
                "prfm   pldl1keep, [%3, #384]           \n"
                "ld1    {v12.4s, v13.4s, v14.4s}, [%3]  \n"// v12 v13 v14 = r10 r14 r18

                "fmla   v10.4s, v19.4s, %14.s[3]         \n"
                "fmla   v11.4s, v20.4s, %14.s[3]         \n"

                "fmla   v8.4s, v12.4s, %15.s[1]         \n"

                "ext    v19.16b, v12.16b, v13.16b, #4   \n"// r11

                "fmla   v9.4s, v13.4s, %15.s[1]         \n"

                "ext    v20.16b, v13.16b, v14.16b, #4   \n"// r15

                "fmla   v10.4s, v13.4s, %16.s[1]         \n"

                "ext    v21.16b, v12.16b, v13.16b, #8   \n"// r12

                "fmla   v11.4s, v14.4s, %16.s[1]         \n"

                "ext    v22.16b, v13.16b, v14.16b, #8   \n"// r16

                "fmla   v8.4s, v19.4s, %15.s[2]         \n"

                "ext    v19.16b, v12.16b, v13.16b, #12  \n"// r13

                "fmla   v9.4s, v20.4s, %15.s[2]         \n"

                "ext    v20.16b, v13.16b, v14.16b, #12  \n"// r17

                "fmla   v10.4s, v21.4s, %15.s[3]         \n"
                "fmla   v11.4s, v22.4s, %15.s[3]         \n"

                // r2
                "prfm   pldl1keep, [%4, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%4]  \n"// v16 v17 v18 = r20 r24 r28

                "fmla   v8.4s, v19.4s, %16.s[0]         \n"
                "fmla   v9.4s, v20.4s, %16.s[0]         \n"

                "fmla   v10.4s, v16.4s, %16.s[2]         \n"

                "ext    v19.16b, v16.16b, v17.16b, #4   \n"// r21

                "fmla   v11.4s, v17.4s, %16.s[2]         \n"

                "ext    v20.16b, v17.16b, v18.16b, #4   \n"// r25

                "fmla   v8.4s, v17.4s, %17.s[2]         \n"

                "ext    v21.16b, v16.16b, v17.16b, #8   \n"// r22

                "fmla   v9.4s, v18.4s, %17.s[2]         \n"

                "ext    v22.16b, v17.16b, v18.16b, #8   \n"// r26

                "fmla   v10.4s, v19.4s, %16.s[3]         \n"

                "ext    v19.16b, v16.16b, v17.16b, #12  \n"// r23

                "fmla   v11.4s, v20.4s, %16.s[3]         \n"

                "ext    v20.16b, v17.16b, v18.16b, #12  \n"// r27

                "fmla   v8.4s, v21.4s, %17.s[0]         \n"
                "fmla   v9.4s, v22.4s, %17.s[0]         \n"

                // r3
                "prfm   pldl1keep, [%5, #384]           \n"
                "ld1    {v12.4s, v13.4s, v14.4s}, [%5]  \n"// v12 v13 v14 = r30 r34 r38

                "fmla   v10.4s, v19.4s, %17.s[1]         \n"
                "fmla   v11.4s, v20.4s, %17.s[1]         \n"

                "fmla   v8.4s, v12.4s, %17.s[3]         \n"

                "ext    v19.16b, v12.16b, v13.16b, #4   \n"// r11

                "fmla   v9.4s, v13.4s, %17.s[3]         \n"

                "ext    v20.16b, v13.16b, v14.16b, #4   \n"// r15

                "fmla   v10.4s, v13.4s, %18.s[3]         \n"

                "ext    v21.16b, v12.16b, v13.16b, #8   \n"// r12

                "fmla   v11.4s, v14.4s, %18.s[3]         \n"

                "ext    v22.16b, v13.16b, v14.16b, #8   \n"// r16

                "fmla   v8.4s, v19.4s, %18.s[0]         \n"

                "ext    v19.16b, v12.16b, v13.16b, #12  \n"// r13

                "fmla   v9.4s, v20.4s, %18.s[0]         \n"

                "ext    v20.16b, v13.16b, v14.16b, #12  \n"// r17

                "fmla   v10.4s, v21.4s, %18.s[1]         \n"
                "fmla   v11.4s, v22.4s, %18.s[1]         \n"

                // r4
                "prfm   pldl1keep, [%6, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n"// v16 v17 v18 = r40 r44 r48

                "fmla   v8.4s, v19.4s, %18.s[2]         \n"
                "fmla   v9.4s, v20.4s, %18.s[2]         \n"

                "fmla   v10.4s, v16.4s, %19.s[0]         \n"

                "ext    v19.16b, v16.16b, v17.16b, #4   \n"// r41

                "fmla   v11.4s, v17.4s, %19.s[0]         \n"

                "ext    v20.16b, v17.16b, v18.16b, #4   \n"// r45

                "fmla   v8.4s, v17.4s, %20.s[0]         \n"

                "ext    v21.16b, v16.16b, v17.16b, #8   \n"// r42

                "fmla   v9.4s, v18.4s, %20.s[0]         \n"

                "ext    v22.16b, v17.16b, v18.16b, #8   \n"// r46

                "fmla   v10.4s, v19.4s, %19.s[1]         \n"

                "ext    v19.16b, v16.16b, v17.16b, #12  \n"// r43

                "fmla   v11.4s, v20.4s, %19.s[1]         \n"

                "ext    v20.16b, v17.16b, v18.16b, #12  \n"// r47

                "fmla   v8.4s, v21.4s, %19.s[2]         \n"

                "add    %2, %2, #32                     \n"

                "fmla   v9.4s, v22.4s, %19.s[2]         \n"

                "add    %3, %3, #32                     \n"

                "fmla   v10.4s, v19.4s, %19.s[3]         \n"

                "add    %4, %4, #32                     \n"

                "fmla   v11.4s, v20.4s, %19.s[3]         \n"

                // r0
                "prfm   pldl1keep, [%2, #384]           \n"
                "ld1    {v16.4s, v17.4s, v18.4s}, [%2]  \n"// v16 v17 v18 = r00 r04 r08

                "add    %5, %5, #32                     \n"

                "fadd   v10.4s, v8.4s, v10.4s           \n"

                "add    %6, %6, #32                     \n"

                "fadd   v11.4s, v9.4s, v11.4s           \n"

                "mov    v8.16b, %21.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %21.16b                 \n"// v9 = _bias0

                "subs   %w0, %w0, #1                    \n"

                "st1    {v10.4s, v11.4s}, [%1], #32     \n"

                "bne    0b                              \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2),         // %4
                  "=r"(r3),         // %5
                  "=r"(r4)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "6"(r4),
                  "w"(_k0123),      // %14
                  "w"(_k4567),      // %15
                  "w"(_k891011),    // %16
                  "w"(_k12131415),  // %17
                  "w"(_k16171819),  // %18
                  "w"(_k20212223),  // %19
                  "w"(_k24242424),  // %20
                  "w"(_bias0)       // %21
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22"
            );
            }

            if (remain >= 4)
            {
                remain -= 4;
                asm volatile(
                    // r0
                    "prfm   pldl1keep, [%1, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%1]          \n"// v16 v17 = r00 r04

                    "mov    v8.16b, %19.16b                 \n"// v8 = _bias0

                    "add    %1, %1, #16                     \n"

                    "fmul   v9.4s, v16.4s, %12.s[0]         \n"

                    "ext    v18.16b, v16.16b, v17.16b, #4   \n"// r01

                    "fmla   v8.4s, v17.4s, %13.s[0]         \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"// r02

                    "fmla   v9.4s, v18.4s, %12.s[1]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #12  \n"// r03

                    "fmla   v8.4s, v19.4s, %12.s[2]         \n"

                    // r1
                    "prfm   pldl1keep, [%2, #256]           \n"
                    "ld1    {v10.4s, v11.4s}, [%2]          \n"// v10 v11 = r10 r14

                    "fmla   v9.4s, v20.4s, %12.s[3]         \n"

                    "add    %2, %2, #16                     \n"

                    "fmla   v8.4s, v10.4s, %13.s[1]         \n"

                    "ext    v12.16b, v10.16b, v11.16b, #4   \n"// r11

                    "fmla   v9.4s, v11.4s, %14.s[1]         \n"

                    "ext    v13.16b, v10.16b, v11.16b, #8   \n"// r12

                    "fmla   v8.4s, v12.4s, %13.s[2]         \n"

                    "ext    v14.16b, v10.16b, v11.16b, #12  \n"// r13

                    "fmla   v9.4s, v13.4s, %13.s[3]         \n"

                    // r2
                    "prfm   pldl1keep, [%3, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%3]          \n"// v16 v17 = r20 r24

                    "fmla   v8.4s, v14.4s, %14.s[0]         \n"

                    "add    %3, %3, #16                     \n"

                    "fmla   v9.4s, v16.4s, %14.s[2]         \n"

                    "ext    v18.16b, v16.16b, v17.16b, #4   \n"// r21

                    "fmla   v8.4s, v17.4s, %15.s[2]         \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"// r22

                    "fmla   v9.4s, v18.4s, %14.s[3]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #12  \n"// r23

                    "fmla   v8.4s, v19.4s, %15.s[0]         \n"

                    // r3
                    "prfm   pldl1keep, [%4, #256]           \n"
                    "ld1    {v10.4s, v11.4s}, [%4]          \n"// v10 v11 = r30 r34

                    "fmla   v9.4s, v20.4s, %15.s[1]         \n"

                    "add    %4, %4, #16                     \n"

                    "fmla   v8.4s, v10.4s, %15.s[3]         \n"

                    "ext    v12.16b, v10.16b, v11.16b, #4   \n"// r31

                    "fmla   v9.4s, v11.4s, %16.s[3]         \n"

                    "ext    v13.16b, v10.16b, v11.16b, #8   \n"// r32

                    "fmla   v8.4s, v12.4s, %16.s[0]         \n"

                    "ext    v14.16b, v10.16b, v11.16b, #12  \n"// r33

                    "fmla   v9.4s, v13.4s, %16.s[1]         \n"

                    // r4
                    "prfm   pldl1keep, [%5, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%5]          \n"// v16 v17 = r40 r44

                    "fmla   v8.4s, v14.4s, %16.s[2]         \n"

                    "add    %5, %5, #16                     \n"

                    "fmla   v9.4s, v16.4s, %17.s[0]         \n"

                    "ext    v18.16b, v16.16b, v17.16b, #4   \n"// r41

                    "fmla   v8.4s, v17.4s, %18.s[0]         \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"// r42

                    "fmla   v9.4s, v18.4s, %17.s[1]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #12  \n"// r43

                    "fmla   v8.4s, v19.4s, %17.s[2]         \n"

                    "fmla   v9.4s, v20.4s, %17.s[3]         \n"

                    "fadd   v8.4s, v8.4s, v9.4s             \n"

                    "st1    {v8.4s}, [%0], #16              \n"

                    : "=r"(outptr),     // %0
                      "=r"(r0),         // %1
                      "=r"(r1),         // %2
                      "=r"(r2),         // %3
                      "=r"(r3),         // %4
                      "=r"(r4)          // %5
                    : "0"(outptr),
                      "1"(r0),
                      "2"(r1),
                      "3"(r2),
                      "4"(r3),
                      "5"(r4),
                      "w"(_k0123),      // %12
                      "w"(_k4567),      // %13
                      "w"(_k891011),    // %14
                      "w"(_k12131415),  // %15
                      "w"(_k16171819),  // %16
                      "w"(_k20212223),  // %17
                      "w"(_k24242424),  // %18
                      "w"(_bias0)       // %19
                    : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20"
                );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                // r0
                "pld        [%2, #256]          \n"
                "vld1.f32   {d20-d23}, [%2]     \n"// q10 q11 = r00 r04

                "vmov       q8, %q21            \n"// q8 = _bias0

                "0:                             \n"

                "vmul.f32   q9, q10, %e14[0]    \n"

                "vext.32    q12, q10, q11, #1   \n"// r01

                "vmla.f32   q8, q11, %e15[0]    \n"

                "vext.32    q13, q10, q11, #2   \n"// r02

                "vmla.f32   q9, q12, %e14[1]    \n"

                "vext.32    q12, q10, q11, #3   \n"// r03

                "vmla.f32   q8, q13, %f14[0]    \n"

                // r1
                "pld        [%3, #256]          \n"
                "vld1.f32   {d28-d31}, [%3]     \n"// q14 q15 = r10 r14

                "vmla.f32   q9, q12, %f14[1]    \n"

                "add        %3, #16             \n"

                "vmla.f32   q8, q14, %e15[1]    \n"

                "vext.32    q12, q14, q15, #1   \n"// r11

                "vmla.f32   q9, q15, %e16[1]    \n"

                "vext.32    q13, q14, q15, #2   \n"// r12

                "vmla.f32   q8, q12, %f15[0]    \n"

                "vext.32    q12, q14, q15, #3   \n"// r13

                "vmla.f32   q9, q13, %f15[1]    \n"

                // r2
                "pld        [%4, #256]          \n"
                "vld1.f32   {d20-d23}, [%4]     \n"// q10 q11 = r20 r24

                "vmla.f32   q8, q12, %e16[0]    \n"

                "add        %4, #16             \n"

                "vmla.f32   q9, q10, %f16[0]    \n"

                "vext.32    q12, q10, q11, #1   \n"// r21

                "vmla.f32   q8, q11, %f17[0]    \n"

                "vext.32    q13, q10, q11, #2   \n"// r22

                "vmla.f32   q9, q12, %f16[1]    \n"

                "vext.32    q12, q10, q11, #3   \n"// r23

                "vmla.f32   q8, q13, %e17[0]    \n"

                // r3
                "pld        [%5, #256]          \n"
                "vld1.f32   {d28-d31}, [%5]     \n"// q14 q15 = r30 r34

                "vmla.f32   q9, q12, %e17[1]    \n"

                "add        %5, #16             \n"

                "vmla.f32   q8, q14, %f17[1]    \n"

                "vext.32    q12, q14, q15, #1   \n"// r31

                "vmla.f32   q9, q15, %f18[1]    \n"

                "vext.32    q13, q14, q15, #2   \n"// r32

                "vmla.f32   q8, q12, %e18[0]    \n"

                "vext.32    q12, q14, q15, #3   \n"// r33

                "vmla.f32   q9, q13, %e18[1]    \n"

                // r4
                "pld        [%6, #256]          \n"
                "vld1.f32   {d20-d23}, [%6]     \n"// q10 q11 = r40 r44

                "vmla.f32   q8, q12, %f18[0]    \n"

                "add        %6, #16             \n"

                "vmla.f32   q9, q10, %e19[0]    \n"

                "vext.32    q12, q10, q11, #1   \n"// r41

                "vmla.f32   q8, q11, %e20[0]    \n"

                "vext.32    q13, q10, q11, #2   \n"// r42

                "vmla.f32   q9, q12, %e19[1]    \n"

                "vext.32    q12, q10, q11, #3   \n"// r43

                "vmla.f32   q8, q13, %f19[0]    \n"

                "add        %2, #16             \n"

                "vmla.f32   q9, q12, %f19[1]    \n"

                // r0
                "pld        [%2, #256]          \n"
                "vld1.f32   {d20-d23}, [%2]     \n"// q10 q11 = r00 r04

                "vadd.f32   q9, q9, q8          \n"

                "vmov       q8, %q21            \n"// q8 = _bias0

                "subs       %0, #1              \n"

                "vst1.f32   {d18-d19}, [%1]!    \n"

                "bne        0b                  \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2),         // %4
                  "=r"(r3),         // %5
                  "=r"(r4)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "6"(r4),
                  "w"(_k0123),      // %14
                  "w"(_k4567),      // %15
                  "w"(_k891011),    // %16
                  "w"(_k12131415),  // %17
                  "w"(_k16171819),  // %18
                  "w"(_k20212223),  // %19
                  "w"(_k24242424),  // %20
                  "w"(_bias0)       // %21
                : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
#if __ARM_NEON
#if __aarch64__
                // TODO neon assembly optimize
                float sum = bias0;

                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _sum = vmulq_f32(_r0, _k0123);

                float32x4_t _r1 = vld1q_f32(r1);
                _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                float32x4_t _r2 = vld1q_f32(r2);
                _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                float32x4_t _r3 = vld1q_f32(r3);
                _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                float32x4_t _r4 = vld1q_f32(r4);
                _sum = vmlaq_f32(_sum, _r4, _k20212223);

                float32x4_t _k_t4;
                _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                float32x4_t _r_t4;

                _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                sum += r4[4] * k4[4];

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                sum += vget_lane_f32(_ss, 0);

                *outptr = sum;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                outptr++;
#else
                // TODO neon assembly optimize
                asm volatile(
                    "veor       q14, q14            \n"
                    "vext.32    q14, %q19, q14, #3  \n"// q14 = bias0 0 0 0

                    "vld1.f32   {d16-d17}, [%1]     \n"// q8 = r00 r01 r02 r03

                    "vld1.f32   {d18-d19}, [%2]     \n"// q9 = r10 r11 r12 r13(X)
                    "add        r4, %1, #16         \n"
                    "vld1.f32   {d19[1]}, [r4]      \n"
                    "vext.32    q9, q9, q9, #3      \n"// q9 = r04 r10 r11 r12

                    "vmla.f32   q14, q8, %q12       \n"

                    "add        r4, %2, #12         \n"
                    "vld1.f32   {d20}, [r4]         \n"// d20 = r13 r14
                    "vld1.f32   {d21}, [%3]         \n"// d21 = r20 r21

                    "vmla.f32   q14, q9, %q13       \n"

                    "add        r4, %3, #8          \n"
                    "vld1.f32   {d22-d23}, [r4]     \n"// q11 = r22 r23 r24 X
                    "vld1.f32   {d23[1]}, [%4]      \n"// q11 = r22 r23 r24 r30

                    "vmla.f32   q14, q10, %q14      \n"

                    "add        r4, %4, #4          \n"
                    "vld1.f32   {d24-d25}, [r4]     \n"// q12 = r31 r32 r33 r34

                    "vmla.f32   q14, q11, %q15      \n"

                    "vld1.f32   {d26-d27}, [%5]     \n"// q13 = r40 r41 r42 r43

                    "vmla.f32   q14, q12, %q16      \n"

                    "veor       d30, d30            \n"
                    "add        r4, %5, #16         \n"
                    "vld1.f32   {d30[0]}, [r4]      \n"// d30 = r44 0

                    "vmla.f32   q14, q13, %q17      \n"

                    "vmla.f32   d28, d30, %e18      \n"

                    "add        %1, #4              \n"

                    // h-sum
                    "vadd.f32   d28, d28, d29       \n"

                    "add        %2, #4              \n"
                    "add        %3, #4              \n"

                    "vpadd.f32  d28, d28, d28       \n"

                    "add        %4, #4              \n"
                    "add        %5, #4              \n"

                    "vst1.f32   {d28[0]}, [%0]!     \n"

                    : "=r"(outptr),     // %0
                      "=r"(r0),         // %1
                      "=r"(r1),         // %2
                      "=r"(r2),         // %3
                      "=r"(r3),         // %4
                      "=r"(r4)          // %5
                    : "0"(outptr),
                      "1"(r0),
                      "2"(r1),
                      "3"(r2),
                      "4"(r3),
                      "5"(r4),
                      "w"(_k0123),      // %12
                      "w"(_k4567),      // %13
                      "w"(_k891011),    // %14
                      "w"(_k12131415),  // %15
                      "w"(_k16171819),  // %16
                      "w"(_k20212223),  // %17
                      "w"(_k24242424),  // %18
                      "w"(_bias0)       // %19
                    : "cc", "memory", "r4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif // __aarch64__
#else
                float sum = bias0;

                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];

                *outptr = sum;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                outptr++;
#endif
            }

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;

        }
    }

}

static void convdw5x5s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g*25;

        float* outptr = out;
        float* outptr2 = outptr + outw;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;
        const float* r3 = img0 + w*3;
        const float* r4 = img0 + w*4;
        const float* r5 = img0 + w*5;
        const float* r6 = img0 + w*6;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 5;
        const float* k2 = kernel0 + 10;
        const float* k3 = kernel0 + 15;
        const float* k4 = kernel0 + 20;

#if __ARM_NEON
        float32x4_t _k0123 = vld1q_f32(kernel0);
        float32x4_t _k4567 = vld1q_f32(kernel0+4);
        float32x4_t _k891011 = vld1q_f32(kernel0+8);
        float32x4_t _k12131415 = vld1q_f32(kernel0+12);
        float32x4_t _k16171819 = vld1q_f32(kernel0+16);
        float32x4_t _k20212223 = vld1q_f32(kernel0+20);
        float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);

        float32x4_t _bias0 = vdupq_n_f32(bias0);
#endif // __ARM_NEON

        int i = 0;

        // NOTE unroll outh 2 results somewhat speed drop :| (about -4%)
        // so we do not implement it here

        for (; i < outh; i++)
        {
#if __ARM_NEON
#if __aarch64__
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int nn = outw >> 2;
            int remain = outw & 3;
#endif // __aarch64__
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                // r0
                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%2], #32     \n"// v16 v17 = r00 r01

                "mov    v8.16b, %21.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %21.16b                 \n"// v9 = _bias0

                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%2], #32     \n"// v18 v19 = r08 r09

                "0:                                     \n"

                "fmul   v10.4s, v16.4s, %14.s[0]        \n"

                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%2]          \n"// v20 v21 = r016 r017

                "fmul   v11.4s, v18.4s, %14.s[0]        \n"

                "ext    v22.16b, v16.16b, v18.16b, #4   \n"// v22 = r02

                "fmla   v8.4s, v17.4s, %14.s[1]         \n"

                "ext    v25.16b, v18.16b, v20.16b, #4   \n"// v25 = r010

                "fmla   v9.4s, v19.4s, %14.s[1]         \n"

                "ext    v23.16b, v17.16b, v19.16b, #4   \n"// v23 = r03

                "fmla   v10.4s, v22.4s, %14.s[2]        \n"

                "ext    v26.16b, v19.16b, v21.16b, #4   \n"// v26 = r011

                "fmla   v11.4s, v25.4s, %14.s[2]        \n"

                "ext    v24.16b, v16.16b, v18.16b, #8   \n"// v24 = r04

                "fmla   v8.4s, v23.4s, %14.s[3]         \n"

                "ext    v27.16b, v18.16b, v20.16b, #8   \n"// v27 = r012

                "fmla   v9.4s, v26.4s, %14.s[3]         \n"

                // r1
                "prfm   pldl1keep, [%3, #256]           \n"
                "ld2    {v12.4s, v13.4s}, [%3], #32     \n"// v12 v13 = r10 r11

                "fmla   v10.4s, v24.4s, %15.s[0]        \n"

                "prfm   pldl1keep, [%3, #256]           \n"
                "ld2    {v14.4s, v15.4s}, [%3], #32     \n"// v14 v15 = r18 r19

                "fmla   v11.4s, v27.4s, %15.s[0]        \n"

                "fmla   v8.4s, v12.4s, %15.s[1]         \n"

                "prfm   pldl1keep, [%3, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%3]          \n"// v20 v21 = r116 r117

                "fmla   v9.4s, v14.4s, %15.s[1]         \n"

                "ext    v22.16b, v12.16b, v14.16b, #4   \n"// v22 = r12

                "fmla   v10.4s, v13.4s, %15.s[2]        \n"

                "ext    v25.16b, v14.16b, v20.16b, #4   \n"// v25 = r110

                "fmla   v11.4s, v15.4s, %15.s[2]        \n"

                "ext    v23.16b, v13.16b, v15.16b, #4   \n"// v23 = r13

                "fmla   v8.4s, v22.4s, %15.s[3]         \n"

                "ext    v26.16b, v15.16b, v21.16b, #4   \n"// v26 = r111

                "fmla   v9.4s, v25.4s, %15.s[3]         \n"

                "ext    v24.16b, v12.16b, v14.16b, #8   \n"// v24 = r14

                "fmla   v10.4s, v23.4s, %16.s[0]        \n"

                "ext    v27.16b, v14.16b, v20.16b, #8   \n"// v27 = r112

                "fmla   v11.4s, v26.4s, %16.s[0]        \n"

                // r2
                "prfm   pldl1keep, [%4, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%4], #32     \n"// v16 v17 = r20 r21

                "fmla   v8.4s, v24.4s, %16.s[1]         \n"

                "prfm   pldl1keep, [%4, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%4], #32     \n"// v18 v19 = r28 r29

                "fmla   v9.4s, v27.4s, %16.s[1]         \n"

                "fmla   v10.4s, v16.4s, %16.s[2]        \n"

                "prfm   pldl1keep, [%4, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%4]          \n"// v20 v21 = r216 r217

                "fmla   v11.4s, v18.4s, %16.s[2]        \n"

                "ext    v22.16b, v16.16b, v18.16b, #4   \n"// v22 = r22

                "fmla   v8.4s, v17.4s, %16.s[3]         \n"

                "ext    v25.16b, v18.16b, v20.16b, #4   \n"// v25 = r210

                "fmla   v9.4s, v19.4s, %16.s[3]         \n"

                "ext    v23.16b, v17.16b, v19.16b, #4   \n"// v23 = r23

                "fmla   v10.4s, v22.4s, %17.s[0]        \n"

                "ext    v26.16b, v19.16b, v21.16b, #4   \n"// v26 = r211

                "fmla   v11.4s, v25.4s, %17.s[0]        \n"

                "ext    v24.16b, v16.16b, v18.16b, #8   \n"// v24 = r24

                "fmla   v8.4s, v23.4s, %17.s[1]         \n"

                "ext    v27.16b, v18.16b, v20.16b, #8   \n"// v27 = r212

                "fmla   v9.4s, v26.4s, %17.s[1]         \n"

                // r3
                "prfm   pldl1keep, [%5, #256]           \n"
                "ld2    {v12.4s, v13.4s}, [%5], #32     \n"// v12 v13 = r30 r31

                "fmla   v10.4s, v24.4s, %17.s[2]        \n"

                "prfm   pldl1keep, [%5, #256]           \n"
                "ld2    {v14.4s, v15.4s}, [%5], #32     \n"// v14 v15 = r38 r39

                "fmla   v11.4s, v27.4s, %17.s[2]        \n"

                "fmla   v8.4s, v12.4s, %17.s[3]         \n"

                "prfm   pldl1keep, [%5, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%5]          \n"// v20 v21 = r316 r317

                "fmla   v9.4s, v14.4s, %17.s[3]         \n"

                "ext    v22.16b, v12.16b, v14.16b, #4   \n"// v22 = r32

                "fmla   v10.4s, v13.4s, %18.s[0]        \n"

                "ext    v25.16b, v14.16b, v20.16b, #4   \n"// v25 = r310

                "fmla   v11.4s, v15.4s, %18.s[0]        \n"

                "ext    v23.16b, v13.16b, v15.16b, #4   \n"// v23 = r33

                "fmla   v8.4s, v22.4s, %18.s[1]         \n"

                "ext    v26.16b, v15.16b, v21.16b, #4   \n"// v26 = r311

                "fmla   v9.4s, v25.4s, %18.s[1]         \n"

                "ext    v24.16b, v12.16b, v14.16b, #8   \n"// v24 = r34

                "fmla   v10.4s, v23.4s, %18.s[2]        \n"

                "ext    v27.16b, v14.16b, v20.16b, #8   \n"// v27 = r312

                "fmla   v11.4s, v26.4s, %18.s[2]        \n"

                // r4
                "prfm   pldl1keep, [%6, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%6], #32     \n"// v16 v17 = r40 r41

                "fmla   v8.4s, v24.4s, %18.s[3]         \n"

                "prfm   pldl1keep, [%6, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%6], #32     \n"// v18 v19 = r48 r49

                "fmla   v9.4s, v27.4s, %18.s[3]         \n"

                "fmla   v10.4s, v16.4s, %19.s[0]        \n"

                "prfm   pldl1keep, [%6, #256]           \n"
                "ld2    {v20.4s, v21.4s}, [%6]          \n"// v20 v21 = r416 r417

                "fmla   v11.4s, v18.4s, %19.s[0]        \n"

                "ext    v22.16b, v16.16b, v18.16b, #4   \n"// v22 = r42

                "fmla   v8.4s, v17.4s, %19.s[1]         \n"

                "ext    v25.16b, v18.16b, v20.16b, #4   \n"// v25 = r410

                "fmla   v9.4s, v19.4s, %19.s[1]         \n"

                "ext    v23.16b, v17.16b, v19.16b, #4   \n"// v23 = r43

                "fmla   v10.4s, v22.4s, %19.s[2]        \n"

                "ext    v26.16b, v19.16b, v21.16b, #4   \n"// v26 = r411

                "fmla   v11.4s, v25.4s, %19.s[2]        \n"

                "ext    v24.16b, v16.16b, v18.16b, #8   \n"// v24 = r44

                "fmla   v8.4s, v23.4s, %19.s[3]         \n"

                "ext    v27.16b, v18.16b, v20.16b, #8   \n"// v27 = r412

                "fmla   v9.4s, v26.4s, %19.s[3]         \n"
                "fmla   v10.4s, v24.4s, %20.s[0]        \n"

                // r0
                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v16.4s, v17.4s}, [%2], #32     \n"// v16 v17 = r00 r01

                "fmla   v11.4s, v27.4s, %20.s[0]        \n"

                "prfm   pldl1keep, [%2, #256]           \n"
                "ld2    {v18.4s, v19.4s}, [%2], #32     \n"// v18 v19 = r08 r09

                "fadd   v10.4s, v8.4s, v10.4s           \n"
                "fadd   v11.4s, v9.4s, v11.4s           \n"

                "subs   %w0, %w0, #1                    \n"

                "mov    v8.16b, %21.16b                 \n"// v8 = _bias0
                "mov    v9.16b, %21.16b                 \n"// v9 = _bias0

                "st1    {v10.4s, v11.4s}, [%1], #32     \n"

                "bne    0b                              \n"
                "sub    %2, %2, #64                     \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2),         // %4
                  "=r"(r3),         // %5
                  "=r"(r4)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "6"(r4),
                  "w"(_k0123),      // %14
                  "w"(_k4567),      // %15
                  "w"(_k891011),    // %16
                  "w"(_k12131415),  // %17
                  "w"(_k16171819),  // %18
                  "w"(_k20212223),  // %19
                  "w"(_k24242424),  // %20
                  "w"(_bias0)       // %21
                : "cc", "memory", "v8", "v9", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                // r0
                "pld        [%2, #256]          \n"
                "vld2.f32   {d20-d23}, [%2]!    \n"// q10 q11 = r00 r01

                "vmov       q8, %q21            \n"

                "pld        [%2, #128]          \n"
                "vld2.f32   {d24-d25}, [%2]     \n"// q12 = r08 x x

                "0:                             \n"

                "vmul.f32   q9, q10, %e14[0]    \n"

                "vmov       d26, d25            \n"// q13 = r09 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r02

                "vmla.f32   q8, q11, %e14[1]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r03

                "vmla.f32   q9, q14, %f14[0]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r04

                "vmla.f32   q8, q15, %f14[1]    \n"

                // r1
                "pld        [%3, #256]          \n"
                "vld2.f32   {d20-d23}, [%3]!    \n"// q10 q11 = r10 r11

                "vmla.f32   q9, q14, %e15[0]    \n"

                "pld        [%3, #128]          \n"
                "vld2.f32   {d24-d25}, [%3]     \n"// q12 = r18 x x

                "vmla.f32   q8, q10, %e15[1]    \n"

                "vmov       d26, d25            \n"// q13 = r19 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r12

                "vmla.f32   q9, q11, %f15[0]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r13

                "vmla.f32   q8, q14, %f15[1]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r14

                "vmla.f32   q9, q15, %e16[0]    \n"

                // r2
                "pld        [%4, #256]          \n"
                "vld2.f32   {d20-d23}, [%4]!    \n"// q10 q11 = r20 r21

                "vmla.f32   q8, q14, %e16[1]    \n"

                "pld        [%4, #128]          \n"
                "vld2.f32   {d24-d25}, [%4]     \n"// q12 = r28 x x

                "vmla.f32   q9, q10, %f16[0]    \n"

                "vmov       d26, d25            \n"// q13 = r29 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r22

                "vmla.f32   q8, q11, %f16[1]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r23

                "vmla.f32   q9, q14, %e17[0]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r24

                "vmla.f32   q8, q15, %e17[1]    \n"

                // r3
                "pld        [%5, #256]          \n"
                "vld2.f32   {d20-d23}, [%5]!    \n"// q10 q11 = r30 r31

                "vmla.f32   q9, q14, %f17[0]    \n"

                "pld        [%5, #128]          \n"
                "vld2.f32   {d24-d25}, [%5]     \n"// q12 = r38 x x

                "vmla.f32   q8, q10, %f17[1]    \n"

                "vmov       d26, d25            \n"// q13 = r39 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r32

                "vmla.f32   q9, q11, %e18[0]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r33

                "vmla.f32   q8, q14, %e18[1]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r34

                "vmla.f32   q9, q15, %f18[0]    \n"

                // r4
                "pld        [%6, #256]          \n"
                "vld2.f32   {d20-d23}, [%6]!    \n"// q10 q11 = r40 r41

                "vmla.f32   q8, q14, %f18[1]    \n"

                "pld        [%6, #128]          \n"
                "vld2.f32   {d24-d25}, [%6]     \n"// q12 = r48 x x

                "vmla.f32   q9, q10, %e19[0]    \n"

                "vmov       d26, d25            \n"// q13 = r49 x x

                "vext.32    q14, q10, q12, #1   \n"// q14 = r42

                "vmla.f32   q8, q11, %e19[1]    \n"

                "vext.32    q15, q11, q13, #1   \n"// q15 = r43

                "vmla.f32   q9, q14, %f19[0]    \n"

                "vext.32    q14, q10, q12, #2   \n"// q14 = r44

                "vmla.f32   q8, q15, %f19[1]    \n"

                // r0
                "pld        [%2, #256]          \n"
                "vld2.f32   {d20-d23}, [%2]!    \n"// q10 q11 = r00 r01

                "vmla.f32   q9, q14, %e20[0]    \n"

                "pld        [%2, #128]          \n"
                "vld2.f32   {d24-d25}, [%2]     \n"// q12 = r08 x x

                "vadd.f32   q9, q8, q9          \n"

                "vmov       q8, %q21            \n"

                "subs       %0, #1              \n"

                "vst1.f32   {d18-d19}, [%1]!    \n"

                "bne        0b                  \n"
                "sub        %2, #32             \n"

                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2),         // %4
                  "=r"(r3),         // %5
                  "=r"(r4)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "6"(r4),
                  "w"(_k0123),      // %14
                  "w"(_k4567),      // %15
                  "w"(_k891011),    // %16
                  "w"(_k12131415),  // %17
                  "w"(_k16171819),  // %18
                  "w"(_k20212223),  // %19
                  "w"(_k24242424),  // %20
                  "w"(_bias0)       // %21
                : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float sum = bias0;
#if __ARM_NEON
                // TODO neon assembly optimize
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _sum = vmulq_f32(_r0, _k0123);

                float32x4_t _r1 = vld1q_f32(r1);
                _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                float32x4_t _r2 = vld1q_f32(r2);
                _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                float32x4_t _r3 = vld1q_f32(r3);
                _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                float32x4_t _r4 = vld1q_f32(r4);
                _sum = vmlaq_f32(_sum, _r4, _k20212223);

                sum += r0[4] * k0[4];
                sum += r1[4] * k1[4];
                sum += r2[4] * k2[4];
                sum += r3[4] * k3[4];
                sum += r4[4] * k4[4];

                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                sum += vget_lane_f32(_ss, 0);
#else
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r0[3] * k0[3];
                sum += r0[4] * k0[4];

                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r1[3] * k1[3];
                sum += r1[4] * k1[4];

                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum += r2[3] * k2[3];
                sum += r2[4] * k2[4];

                sum += r3[0] * k3[0];
                sum += r3[1] * k3[1];
                sum += r3[2] * k3[2];
                sum += r3[3] * k3[3];
                sum += r3[4] * k3[4];

                sum += r4[0] * k4[0];
                sum += r4[1] * k4[1];
                sum += r4[2] * k4[2];
                sum += r4[3] * k4[3];
                sum += r4[4] * k4[4];
#endif
                *outptr = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }

}
