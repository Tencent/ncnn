// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convdw3x3s1_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + g * 8) : vdupq_n_f16((__fp16)0.f);

        const __fp16* k0 = kernel.row<const __fp16>(g);

        __fp16* outptr0 = out.row<__fp16>(0);
        __fp16* outptr1 = out.row<__fp16>(1);

        const Mat img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);
        const __fp16* r2 = img0.row<const __fp16>(2);
        const __fp16* r3 = img0.row<const __fp16>(3);

        float16x8_t _k00 = vld1q_f16(k0);
        float16x8_t _k01 = vld1q_f16(k0 + 8);
        float16x8_t _k02 = vld1q_f16(k0 + 16);
        float16x8_t _k10 = vld1q_f16(k0 + 24);
        float16x8_t _k11 = vld1q_f16(k0 + 32);
        float16x8_t _k12 = vld1q_f16(k0 + 40);
        float16x8_t _k20 = vld1q_f16(k0 + 48);
        float16x8_t _k21 = vld1q_f16(k0 + 56);
        float16x8_t _k22 = vld1q_f16(k0 + 64);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // r10 r11 r12 r13

                    "mov    v24.16b, %21.16b            \n" // sum00
                    "mov    v25.16b, %21.16b            \n" // sum01
                    "mov    v26.16b, %21.16b            \n" // sum02
                    "mov    v27.16b, %21.16b            \n" // sum03

                    "fmla   v24.8h, %15.8h, v12.8h      \n"
                    "fmla   v25.8h, %15.8h, v13.8h      \n"

                    "mov    v28.16b, %21.16b            \n" // sum10
                    "mov    v29.16b, %21.16b            \n" // sum11
                    "mov    v30.16b, %21.16b            \n" // sum12
                    "mov    v31.16b, %21.16b            \n" // sum13

                    "fmla   v26.8h, %15.8h, v14.8h      \n"
                    "fmla   v27.8h, %15.8h, v15.8h      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%3]      \n" // r14 r15

                    "fmla   v28.8h, %12.8h, v12.8h      \n"
                    "fmla   v29.8h, %12.8h, v13.8h      \n"
                    "fmla   v30.8h, %12.8h, v14.8h      \n"
                    "fmla   v31.8h, %12.8h, v15.8h      \n"

                    "fmla   v24.8h, %16.8h, v13.8h      \n"
                    "fmla   v25.8h, %16.8h, v14.8h      \n"
                    "fmla   v26.8h, %16.8h, v15.8h      \n"
                    "fmla   v27.8h, %16.8h, v16.8h      \n"

                    "fmla   v28.8h, %13.8h, v13.8h      \n"
                    "fmla   v29.8h, %13.8h, v14.8h      \n"
                    "fmla   v30.8h, %13.8h, v15.8h      \n"
                    "fmla   v31.8h, %13.8h, v16.8h      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%4], #64 \n" // r20 r21 r22 r23

                    "fmla   v24.8h, %17.8h, v14.8h      \n"
                    "fmla   v25.8h, %17.8h, v15.8h      \n"
                    "fmla   v26.8h, %17.8h, v16.8h      \n"
                    "fmla   v27.8h, %17.8h, v17.8h      \n"

                    "fmla   v28.8h, %14.8h, v14.8h      \n"
                    "fmla   v29.8h, %14.8h, v15.8h      \n"
                    "fmla   v30.8h, %14.8h, v16.8h      \n"
                    "fmla   v31.8h, %14.8h, v17.8h      \n"

                    "fmla   v24.8h, %18.8h, v18.8h      \n"
                    "fmla   v25.8h, %18.8h, v19.8h      \n"
                    "fmla   v26.8h, %18.8h, v20.8h      \n"
                    "fmla   v27.8h, %18.8h, v21.8h      \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%4]      \n" // r24 r25

                    "fmla   v28.8h, %15.8h, v18.8h      \n"
                    "fmla   v29.8h, %15.8h, v19.8h      \n"
                    "fmla   v30.8h, %15.8h, v20.8h      \n"
                    "fmla   v31.8h, %15.8h, v21.8h      \n"

                    "fmla   v24.8h, %19.8h, v19.8h      \n"
                    "fmla   v25.8h, %19.8h, v20.8h      \n"
                    "fmla   v26.8h, %19.8h, v21.8h      \n"
                    "fmla   v27.8h, %19.8h, v22.8h      \n"

                    "fmla   v28.8h, %16.8h, v19.8h      \n"
                    "fmla   v29.8h, %16.8h, v20.8h      \n"
                    "fmla   v30.8h, %16.8h, v21.8h      \n"
                    "fmla   v31.8h, %16.8h, v22.8h      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%2], #64 \n" // r00 r01 r02 r03

                    "fmla   v24.8h, %20.8h, v20.8h      \n"
                    "fmla   v25.8h, %20.8h, v21.8h      \n"
                    "fmla   v26.8h, %20.8h, v22.8h      \n"
                    "fmla   v27.8h, %20.8h, v23.8h      \n"

                    "fmla   v28.8h, %17.8h, v20.8h      \n"
                    "fmla   v29.8h, %17.8h, v21.8h      \n"
                    "fmla   v30.8h, %17.8h, v22.8h      \n"
                    "fmla   v31.8h, %17.8h, v23.8h      \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%5], #64 \n" // r30 r31 r32 r33

                    "fmla   v24.8h, %12.8h, v12.8h      \n"
                    "fmla   v25.8h, %12.8h, v13.8h      \n"
                    "fmla   v26.8h, %12.8h, v14.8h      \n"
                    "fmla   v27.8h, %12.8h, v15.8h      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%2]      \n" // r04 r05

                    "fmla   v28.8h, %18.8h, v18.8h      \n"
                    "fmla   v29.8h, %18.8h, v19.8h      \n"
                    "fmla   v30.8h, %18.8h, v20.8h      \n"
                    "fmla   v31.8h, %18.8h, v21.8h      \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%5]      \n" // r34 r35

                    "fmla   v24.8h, %13.8h, v13.8h      \n"
                    "fmla   v25.8h, %13.8h, v14.8h      \n"
                    "fmla   v26.8h, %13.8h, v15.8h      \n"
                    "fmla   v27.8h, %13.8h, v16.8h      \n"

                    "fmla   v28.8h, %19.8h, v19.8h      \n"
                    "fmla   v29.8h, %19.8h, v20.8h      \n"
                    "fmla   v30.8h, %19.8h, v21.8h      \n"
                    "fmla   v31.8h, %19.8h, v22.8h      \n"

                    "fmla   v24.8h, %14.8h, v14.8h      \n"
                    "fmla   v25.8h, %14.8h, v15.8h      \n"
                    "fmla   v26.8h, %14.8h, v16.8h      \n"
                    "fmla   v27.8h, %14.8h, v17.8h      \n"

                    "fmla   v28.8h, %20.8h, v20.8h      \n"
                    "fmla   v29.8h, %20.8h, v21.8h      \n"
                    "fmla   v30.8h, %20.8h, v22.8h      \n"
                    "fmla   v31.8h, %20.8h, v23.8h      \n"

                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3)       // %5
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "w"(_k00),  // %12
                    "w"(_k01),  // %13
                    "w"(_k02),  // %14
                    "w"(_k10),  // %15
                    "w"(_k11),  // %16
                    "w"(_k12),  // %17
                    "w"(_k20),  // %18
                    "w"(_k21),  // %19
                    "w"(_k22),  // %20
                    "w"(_bias0) // %21
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; j + 1 < outw; j += 2)
            {
                asm volatile(
                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%3] \n" // r10 r11 r12 r13

                    "mov    v28.16b, %21.16b            \n" // sum00
                    "mov    v29.16b, %21.16b            \n" // sum01
                    "mov    v30.16b, %21.16b            \n" // sum10
                    "mov    v31.16b, %21.16b            \n" // sum11

                    "fmla   v28.8h, %15.8h, v16.8h      \n"
                    "fmla   v30.8h, %12.8h, v16.8h      \n"
                    "fmla   v29.8h, %15.8h, v17.8h      \n"
                    "fmla   v31.8h, %12.8h, v17.8h      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n" // r20 r21 r22 r23

                    "fmla   v28.8h, %16.8h, v17.8h      \n"
                    "fmla   v30.8h, %13.8h, v17.8h      \n"
                    "fmla   v29.8h, %16.8h, v18.8h      \n"
                    "fmla   v31.8h, %13.8h, v18.8h      \n"

                    "fmla   v28.8h, %17.8h, v18.8h      \n"
                    "fmla   v30.8h, %14.8h, v18.8h      \n"
                    "fmla   v29.8h, %17.8h, v19.8h      \n"
                    "fmla   v31.8h, %14.8h, v19.8h      \n"

                    "fmla   v28.8h, %18.8h, v20.8h      \n"
                    "fmla   v30.8h, %15.8h, v20.8h      \n"
                    "fmla   v29.8h, %18.8h, v21.8h      \n"
                    "fmla   v31.8h, %15.8h, v21.8h      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%2] \n" // r00 r01 r02 r03

                    "fmla   v28.8h, %19.8h, v21.8h      \n"
                    "fmla   v30.8h, %16.8h, v21.8h      \n"
                    "fmla   v29.8h, %19.8h, v22.8h      \n"
                    "fmla   v31.8h, %16.8h, v22.8h      \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%5] \n" // r30 r31 r32 r33

                    "fmla   v28.8h, %20.8h, v22.8h      \n"
                    "fmla   v30.8h, %17.8h, v22.8h      \n"
                    "fmla   v29.8h, %20.8h, v23.8h      \n"
                    "fmla   v31.8h, %17.8h, v23.8h      \n"

                    "fmla   v28.8h, %12.8h, v12.8h      \n"
                    "fmla   v30.8h, %18.8h, v24.8h      \n"
                    "fmla   v29.8h, %12.8h, v13.8h      \n"
                    "fmla   v31.8h, %18.8h, v25.8h      \n"
                    "fmla   v28.8h, %13.8h, v13.8h      \n"
                    "fmla   v30.8h, %19.8h, v25.8h      \n"
                    "fmla   v29.8h, %13.8h, v14.8h      \n"
                    "fmla   v31.8h, %19.8h, v26.8h      \n"
                    "fmla   v28.8h, %14.8h, v14.8h      \n"
                    "fmla   v30.8h, %20.8h, v26.8h      \n"
                    "fmla   v29.8h, %14.8h, v15.8h      \n"
                    "fmla   v31.8h, %20.8h, v27.8h      \n"

                    "add    %2, %2, #32                 \n"
                    "add    %3, %3, #32                 \n"
                    "add    %4, %4, #32                 \n"
                    "add    %5, %5, #32                 \n"

                    "st1    {v28.8h, v29.8h}, [%0], #32 \n"
                    "st1    {v30.8h, v31.8h}, [%1], #32 \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3)       // %5
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "w"(_k00),  // %12
                    "w"(_k01),  // %13
                    "w"(_k02),  // %14
                    "w"(_k10),  // %15
                    "w"(_k11),  // %16
                    "w"(_k12),  // %17
                    "w"(_k20),  // %18
                    "w"(_k21),  // %19
                    "w"(_k22),  // %20
                    "w"(_bias0) // %21
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; j < outw; j++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%3, #384]       \n"
                    "ld1    {v15.8h, v16.8h, v17.8h}, [%3] \n" // r10 r11 r12

                    "mov    v28.16b, %21.16b            \n" // sum00
                    "mov    v30.16b, %21.16b            \n" // sum10

                    "fmul   v29.8h, %15.8h, v15.8h      \n"
                    "fmul   v31.8h, %12.8h, v15.8h      \n"

                    "prfm   pldl1keep, [%4, #384]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h}, [%4] \n" // r20 r21 r22

                    "fmla   v28.8h, %16.8h, v16.8h      \n"
                    "fmla   v30.8h, %13.8h, v16.8h      \n"

                    "fmla   v29.8h, %17.8h, v17.8h      \n"
                    "fmla   v31.8h, %14.8h, v17.8h      \n"

                    "prfm   pldl1keep, [%2, #384]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h}, [%2] \n" // r00 r01 r02

                    "fmla   v28.8h, %18.8h, v18.8h      \n"
                    "fmla   v30.8h, %15.8h, v18.8h      \n"

                    "fmla   v29.8h, %19.8h, v19.8h      \n"
                    "fmla   v31.8h, %16.8h, v19.8h      \n"

                    "prfm   pldl1keep, [%5, #384]       \n"
                    "ld1    {v21.8h, v22.8h, v23.8h}, [%5] \n" // r30 r31 r32

                    "fmla   v28.8h, %20.8h, v20.8h      \n"
                    "fmla   v30.8h, %17.8h, v20.8h      \n"

                    "fmla   v29.8h, %12.8h, v12.8h      \n"
                    "fmla   v31.8h, %18.8h, v21.8h      \n"
                    "fmla   v28.8h, %13.8h, v13.8h      \n"
                    "fmla   v30.8h, %19.8h, v22.8h      \n"
                    "fmla   v29.8h, %14.8h, v14.8h      \n"
                    "fmla   v31.8h, %20.8h, v23.8h      \n"

                    "add    %2, %2, #16                 \n"
                    "add    %3, %3, #16                 \n"

                    "fadd   v28.8h, v28.8h, v29.8h      \n"
                    "fadd   v30.8h, v30.8h, v31.8h      \n"

                    "add    %4, %4, #16                 \n"
                    "add    %5, %5, #16                 \n"

                    "st1    {v28.8h}, [%0], #16         \n"
                    "st1    {v30.8h}, [%1], #16         \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3)       // %5
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "w"(_k00),  // %12
                    "w"(_k01),  // %13
                    "w"(_k02),  // %14
                    "w"(_k10),  // %15
                    "w"(_k11),  // %16
                    "w"(_k12),  // %17
                    "w"(_k20),  // %18
                    "w"(_k21),  // %19
                    "w"(_k22),  // %20
                    "w"(_bias0) // %21
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
            }

            r0 += 2 * 8 + w * 8;
            r1 += 2 * 8 + w * 8;
            r2 += 2 * 8 + w * 8;
            r3 += 2 * 8 + w * 8;

            outptr0 += outw * 8;
            outptr1 += outw * 8;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n" // r00 r01 r02 r03

                    "mov    v28.16b, %17.16b            \n" // sum00
                    "mov    v29.16b, %17.16b            \n" // sum01
                    "mov    v30.16b, %17.16b            \n" // sum02
                    "mov    v31.16b, %17.16b            \n" // sum03

                    "fmla   v28.8h, %8.8h, v12.8h       \n"
                    "fmla   v29.8h, %8.8h, v13.8h       \n"
                    "fmla   v30.8h, %8.8h, v14.8h       \n"
                    "fmla   v31.8h, %8.8h, v15.8h       \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%1]      \n" // r04 r05

                    "fmla   v28.8h, %9.8h, v13.8h       \n"
                    "fmla   v29.8h, %9.8h, v14.8h       \n"
                    "fmla   v30.8h, %9.8h, v15.8h       \n"
                    "fmla   v31.8h, %9.8h, v16.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%2], #64 \n" // r10 r11 r12 r13

                    "fmla   v28.8h, %10.8h, v14.8h      \n"
                    "fmla   v29.8h, %10.8h, v15.8h      \n"
                    "fmla   v30.8h, %10.8h, v16.8h      \n"
                    "fmla   v31.8h, %10.8h, v17.8h      \n"

                    "fmla   v28.8h, %11.8h, v18.8h      \n"
                    "fmla   v29.8h, %11.8h, v19.8h      \n"
                    "fmla   v30.8h, %11.8h, v20.8h      \n"
                    "fmla   v31.8h, %11.8h, v21.8h      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%2]      \n" // r14 r15

                    "fmla   v28.8h, %12.8h, v19.8h      \n"
                    "fmla   v29.8h, %12.8h, v20.8h      \n"
                    "fmla   v30.8h, %12.8h, v21.8h      \n"
                    "fmla   v31.8h, %12.8h, v22.8h      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // r20 r21 r22 r23

                    "fmla   v28.8h, %13.8h, v20.8h      \n"
                    "fmla   v29.8h, %13.8h, v21.8h      \n"
                    "fmla   v30.8h, %13.8h, v22.8h      \n"
                    "fmla   v31.8h, %13.8h, v23.8h      \n"

                    "fmla   v28.8h, %14.8h, v12.8h      \n"
                    "fmla   v29.8h, %14.8h, v13.8h      \n"
                    "fmla   v30.8h, %14.8h, v14.8h      \n"
                    "fmla   v31.8h, %14.8h, v15.8h      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%3]      \n" // r24 r25

                    "fmla   v28.8h, %15.8h, v13.8h      \n"
                    "fmla   v29.8h, %15.8h, v14.8h      \n"
                    "fmla   v30.8h, %15.8h, v15.8h      \n"
                    "fmla   v31.8h, %15.8h, v16.8h      \n"

                    "fmla   v28.8h, %16.8h, v14.8h      \n"
                    "fmla   v29.8h, %16.8h, v15.8h      \n"
                    "fmla   v30.8h, %16.8h, v16.8h      \n"
                    "fmla   v31.8h, %16.8h, v17.8h      \n"

                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2)       // %3
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k00),  // %8
                    "w"(_k01),  // %9
                    "w"(_k02),  // %10
                    "w"(_k10),  // %11
                    "w"(_k11),  // %12
                    "w"(_k12),  // %13
                    "w"(_k20),  // %14
                    "w"(_k21),  // %15
                    "w"(_k22),  // %16
                    "w"(_bias0) // %17
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
            }
            for (; j + 1 < outw; j += 2)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1] \n" // r00 r01 r02 r03

                    "mov    v28.16b, %17.16b            \n" // sum00
                    "mov    v29.16b, %17.16b            \n" // sum01

                    "fmul   v30.8h, %8.8h, v12.8h       \n"
                    "fmul   v31.8h, %8.8h, v13.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%2] \n" // r10 r11 r12 r13

                    "fmla   v28.8h, %9.8h, v13.8h       \n"
                    "fmla   v29.8h, %9.8h, v14.8h       \n"
                    "fmla   v30.8h, %10.8h, v14.8h      \n"
                    "fmla   v31.8h, %10.8h, v15.8h      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%3] \n" // r20 r21 r22 r23

                    "fmla   v28.8h, %11.8h, v16.8h      \n"
                    "fmla   v29.8h, %11.8h, v17.8h      \n"
                    "fmla   v30.8h, %12.8h, v17.8h      \n"
                    "fmla   v31.8h, %12.8h, v18.8h      \n"
                    "fmla   v28.8h, %13.8h, v18.8h      \n"
                    "fmla   v29.8h, %13.8h, v19.8h      \n"

                    "fmla   v30.8h, %14.8h, v20.8h      \n"
                    "fmla   v31.8h, %14.8h, v21.8h      \n"
                    "fmla   v28.8h, %15.8h, v21.8h      \n"
                    "fmla   v29.8h, %15.8h, v22.8h      \n"
                    "fmla   v30.8h, %16.8h, v22.8h      \n"
                    "fmla   v31.8h, %16.8h, v23.8h      \n"

                    "add    %1, %1, #32                 \n"

                    "fadd   v28.8h, v28.8h, v30.8h      \n"
                    "fadd   v29.8h, v29.8h, v31.8h      \n"

                    "add    %2, %2, #32                 \n"
                    "add    %3, %3, #32                 \n"

                    "st1    {v28.8h, v29.8h}, [%0], #32 \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2)       // %3
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k00),  // %8
                    "w"(_k01),  // %9
                    "w"(_k02),  // %10
                    "w"(_k10),  // %11
                    "w"(_k11),  // %12
                    "w"(_k12),  // %13
                    "w"(_k20),  // %14
                    "w"(_k21),  // %15
                    "w"(_k22),  // %16
                    "w"(_bias0) // %17
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
            }
            for (; j < outw; j++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #384]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h}, [%1] \n" // r00 r01 r02

                    "mov    v28.16b, %17.16b            \n" // sum00

                    "fmul   v29.8h, %8.8h, v12.8h       \n"

                    "prfm   pldl1keep, [%2, #384]       \n"
                    "ld1    {v15.8h, v16.8h, v17.8h}, [%2] \n" // r10 r11 r12

                    "fmul   v30.8h, %9.8h, v13.8h       \n"
                    "fmla   v28.8h, %10.8h, v14.8h      \n"

                    "prfm   pldl1keep, [%3, #384]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h}, [%3] \n" // r20 r21 r22

                    "fmla   v29.8h, %11.8h, v15.8h      \n"
                    "fmla   v30.8h, %12.8h, v16.8h      \n"
                    "fmla   v28.8h, %13.8h, v17.8h      \n"

                    "fmla   v29.8h, %14.8h, v18.8h      \n"
                    "fmla   v30.8h, %15.8h, v19.8h      \n"
                    "fmla   v28.8h, %16.8h, v20.8h      \n"

                    "add    %1, %1, #16                 \n"

                    "fadd   v29.8h, v29.8h, v30.8h      \n"
                    "fadd   v28.8h, v28.8h, v29.8h      \n"

                    "add    %2, %2, #16                 \n"
                    "add    %3, %3, #16                 \n"

                    "st1    {v28.8h}, [%0], #16         \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2)       // %3
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k00),  // %8
                    "w"(_k01),  // %9
                    "w"(_k02),  // %10
                    "w"(_k10),  // %11
                    "w"(_k11),  // %12
                    "w"(_k12),  // %13
                    "w"(_k20),  // %14
                    "w"(_k21),  // %15
                    "w"(_k22),  // %16
                    "w"(_bias0) // %17
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v28", "v29", "v30");
            }

            r0 += 2 * 8;
            r1 += 2 * 8;
            r2 += 2 * 8;
        }
    }
}

static void convdw3x3s2_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + g * 8) : vdupq_n_f16((__fp16)0.f);

        const __fp16* k0 = kernel.row<const __fp16>(g);

        __fp16* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);
        const __fp16* r2 = img0.row<const __fp16>(2);

        float16x8_t _k00 = vld1q_f16(k0);
        float16x8_t _k01 = vld1q_f16(k0 + 8);
        float16x8_t _k02 = vld1q_f16(k0 + 16);
        float16x8_t _k10 = vld1q_f16(k0 + 24);
        float16x8_t _k11 = vld1q_f16(k0 + 32);
        float16x8_t _k12 = vld1q_f16(k0 + 40);
        float16x8_t _k20 = vld1q_f16(k0 + 48);
        float16x8_t _k21 = vld1q_f16(k0 + 56);
        float16x8_t _k22 = vld1q_f16(k0 + 64);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n" // r00 r01 r02 r03

                    "mov    v28.16b, %17.16b            \n" // sum00
                    "mov    v29.16b, %17.16b            \n" // sum01
                    "mov    v30.16b, %17.16b            \n" // sum02
                    "mov    v31.16b, %17.16b            \n" // sum03

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n" // r04 r05 r06 r07

                    "fmla   v28.8h, %8.8h, v0.8h        \n"
                    "fmla   v29.8h, %8.8h, v2.8h        \n"
                    "fmla   v30.8h, %8.8h, v4.8h        \n"
                    "fmla   v31.8h, %8.8h, v6.8h        \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v8.8h}, [%1]               \n" // r08

                    "fmla   v28.8h, %9.8h, v1.8h        \n"
                    "fmla   v29.8h, %9.8h, v3.8h        \n"
                    "fmla   v30.8h, %9.8h, v5.8h        \n"
                    "fmla   v31.8h, %9.8h, v7.8h        \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%2], #64 \n" // r10 r11 r12 r13

                    "fmla   v28.8h, %10.8h, v2.8h       \n"
                    "fmla   v29.8h, %10.8h, v4.8h       \n"
                    "fmla   v30.8h, %10.8h, v6.8h       \n"
                    "fmla   v31.8h, %10.8h, v8.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%2], #64 \n" // r14 r15 r16 r17

                    "fmla   v28.8h, %11.8h, v16.8h      \n"
                    "fmla   v29.8h, %11.8h, v18.8h      \n"
                    "fmla   v30.8h, %11.8h, v20.8h      \n"
                    "fmla   v31.8h, %11.8h, v22.8h      \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v24.8h}, [%2]              \n" // r18

                    "fmla   v28.8h, %12.8h, v17.8h      \n"
                    "fmla   v29.8h, %12.8h, v19.8h      \n"
                    "fmla   v30.8h, %12.8h, v21.8h      \n"
                    "fmla   v31.8h, %12.8h, v23.8h      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r20 r21 r22 r23

                    "fmla   v28.8h, %13.8h, v18.8h      \n"
                    "fmla   v29.8h, %13.8h, v20.8h      \n"
                    "fmla   v30.8h, %13.8h, v22.8h      \n"
                    "fmla   v31.8h, %13.8h, v24.8h      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3], #64 \n" // r24 r25 r26 r27

                    "fmla   v28.8h, %14.8h, v0.8h       \n"
                    "fmla   v29.8h, %14.8h, v2.8h       \n"
                    "fmla   v30.8h, %14.8h, v4.8h       \n"
                    "fmla   v31.8h, %14.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v8.8h}, [%3]               \n" // r28

                    "fmla   v28.8h, %15.8h, v1.8h       \n"
                    "fmla   v29.8h, %15.8h, v3.8h       \n"
                    "fmla   v30.8h, %15.8h, v5.8h       \n"
                    "fmla   v31.8h, %15.8h, v7.8h       \n"

                    "fmla   v28.8h, %16.8h, v2.8h       \n"
                    "fmla   v29.8h, %16.8h, v4.8h       \n"
                    "fmla   v30.8h, %16.8h, v6.8h       \n"
                    "fmla   v31.8h, %16.8h, v8.8h       \n"

                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2)       // %3
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k00),  // %8
                    "w"(_k01),  // %9
                    "w"(_k02),  // %10
                    "w"(_k10),  // %11
                    "w"(_k11),  // %12
                    "w"(_k12),  // %13
                    "w"(_k20),  // %14
                    "w"(_k21),  // %15
                    "w"(_k22),  // %16
                    "w"(_bias0) // %17
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v28", "v29", "v30", "v31");
            }
            for (; j + 1 < outw; j += 2)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n" // r00 r01 r02 r03

                    "mov    v28.16b, %17.16b            \n" // sum00
                    "mov    v29.16b, %17.16b            \n" // sum01

                    "fmul   v30.8h, %8.8h, v12.8h       \n"
                    "fmul   v31.8h, %8.8h, v14.8h       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v16.8h}, [%1]              \n" // r04

                    "fmla   v28.8h, %9.8h, v13.8h       \n"
                    "fmla   v29.8h, %9.8h, v15.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%2], #64 \n" // r10 r11 r12 r13

                    "fmla   v30.8h, %10.8h, v14.8h      \n"
                    "fmla   v31.8h, %10.8h, v16.8h      \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v21.8h}, [%1]              \n" // r14

                    "fmla   v28.8h, %11.8h, v17.8h      \n"
                    "fmla   v29.8h, %11.8h, v19.8h      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v22.8h, v23.8h, v24.8h, v25.8h}, [%3], #64 \n" // r20 r21 r22 r23

                    "fmla   v30.8h, %12.8h, v18.8h      \n"
                    "fmla   v31.8h, %12.8h, v20.8h      \n"

                    "fmla   v28.8h, %13.8h, v19.8h      \n"
                    "fmla   v29.8h, %13.8h, v21.8h      \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v26.8h}, [%1]              \n" // r24

                    "fmla   v30.8h, %14.8h, v22.8h      \n"
                    "fmla   v31.8h, %14.8h, v24.8h      \n"

                    "fmla   v28.8h, %15.8h, v23.8h      \n"
                    "fmla   v29.8h, %15.8h, v25.8h      \n"
                    "fmla   v30.8h, %16.8h, v24.8h      \n"
                    "fmla   v31.8h, %16.8h, v26.8h      \n"

                    "fadd   v28.8h, v28.8h, v30.8h      \n"
                    "fadd   v29.8h, v29.8h, v31.8h      \n"

                    "st1    {v28.8h, v29.8h}, [%0], #32 \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2)       // %3
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k00),  // %8
                    "w"(_k01),  // %9
                    "w"(_k02),  // %10
                    "w"(_k10),  // %11
                    "w"(_k11),  // %12
                    "w"(_k12),  // %13
                    "w"(_k20),  // %14
                    "w"(_k21),  // %15
                    "w"(_k22),  // %16
                    "w"(_bias0) // %17
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v28", "v29", "v30", "v31");
            }
            for (; j < outw; j++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #384]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h}, [%1] \n" // r00 r01 r02

                    "mov    v28.16b, %17.16b            \n" // sum00

                    "fmul   v29.8h, %8.8h, v12.8h       \n"

                    "prfm   pldl1keep, [%2, #384]       \n"
                    "ld1    {v15.8h, v16.8h, v17.8h}, [%2] \n" // r10 r11 r12

                    "fmul   v30.8h, %9.8h, v13.8h       \n"
                    "fmla   v28.8h, %10.8h, v14.8h      \n"

                    "prfm   pldl1keep, [%3, #384]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h}, [%3] \n" // r20 r21 r22

                    "fmla   v29.8h, %11.8h, v15.8h      \n"
                    "fmla   v30.8h, %12.8h, v16.8h      \n"
                    "fmla   v28.8h, %13.8h, v17.8h      \n"

                    "fmla   v29.8h, %14.8h, v18.8h      \n"
                    "fmla   v30.8h, %15.8h, v19.8h      \n"
                    "fmla   v28.8h, %16.8h, v20.8h      \n"

                    "add    %1, %1, #32                 \n"

                    "fadd   v29.8h, v29.8h, v30.8h      \n"
                    "fadd   v28.8h, v28.8h, v29.8h      \n"

                    "add    %2, %2, #32                 \n"
                    "add    %3, %3, #32                 \n"

                    "st1    {v28.8h}, [%0], #16         \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2)       // %3
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k00),  // %8
                    "w"(_k01),  // %9
                    "w"(_k02),  // %10
                    "w"(_k10),  // %11
                    "w"(_k11),  // %12
                    "w"(_k12),  // %13
                    "w"(_k20),  // %14
                    "w"(_k21),  // %15
                    "w"(_k22),  // %16
                    "w"(_bias0) // %17
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v28", "v29", "v30");
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
