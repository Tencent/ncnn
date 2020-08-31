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

static void convdw3x3s1_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        const unsigned short* k0 = kernel.row<const unsigned short>(g);

        unsigned short* outptr0 = out.row<unsigned short>(0);
        unsigned short* outptr1 = out.row<unsigned short>(1);

        const Mat img0 = bottom_blob.channel(g);

        const unsigned short* r0 = img0.row<const unsigned short>(0);
        const unsigned short* r1 = img0.row<const unsigned short>(1);
        const unsigned short* r2 = img0.row<const unsigned short>(2);
        const unsigned short* r3 = img0.row<const unsigned short>(3);

        float32x4_t _k00 = vcvt_f32_bf16(vld1_u16(k0));
        float32x4_t _k01 = vcvt_f32_bf16(vld1_u16(k0 + 4));
        float32x4_t _k02 = vcvt_f32_bf16(vld1_u16(k0 + 8));
        float32x4_t _k10 = vcvt_f32_bf16(vld1_u16(k0 + 12));
        float32x4_t _k11 = vcvt_f32_bf16(vld1_u16(k0 + 16));
        float32x4_t _k12 = vcvt_f32_bf16(vld1_u16(k0 + 20));
        float32x4_t _k20 = vcvt_f32_bf16(vld1_u16(k0 + 24));
        float32x4_t _k21 = vcvt_f32_bf16(vld1_u16(k0 + 28));
        float32x4_t _k22 = vcvt_f32_bf16(vld1_u16(k0 + 32));

        int i = 0;

#if __aarch64__
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%3], #32 \n" // r10 r11 r12 r13

                    "mov    v16.16b, %21.16b            \n" // sum00
                    "mov    v17.16b, %21.16b            \n" // sum01

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v28.4h, v29.4h}, [%3]      \n" // r14 r15

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "mov    v18.16b, %21.16b            \n" // sum02
                    "mov    v19.16b, %21.16b            \n" // sum03

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "mov    v20.16b, %21.16b            \n" // sum10

                    "fmla   v16.4s, %15.4s, v10.4s      \n"
                    "fmla   v17.4s, %15.4s, v11.4s      \n"

                    "mov    v21.16b, %21.16b            \n" // sum11

                    "fmla   v18.4s, %15.4s, v12.4s      \n"
                    "fmla   v19.4s, %15.4s, v13.4s      \n"

                    "mov    v22.16b, %21.16b            \n" // sum12

                    "fmla   v20.4s, %12.4s, v10.4s      \n"
                    "fmla   v21.4s, %12.4s, v11.4s      \n"

                    "mov    v23.16b, %21.16b            \n" // sum13

                    "fmla   v22.4s, %12.4s, v12.4s      \n"
                    "fmla   v23.4s, %12.4s, v13.4s      \n"

                    "shll   v28.4s, v28.4h, #16         \n"

                    "fmla   v16.4s, %16.4s, v11.4s      \n"
                    "fmla   v17.4s, %16.4s, v12.4s      \n"

                    "shll   v29.4s, v29.4h, #16         \n"

                    "fmla   v18.4s, %16.4s, v13.4s      \n"
                    "fmla   v19.4s, %16.4s, v28.4s      \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%4], #32 \n" // r20 r21 r22 r23

                    "fmla   v20.4s, %13.4s, v11.4s      \n"
                    "fmla   v21.4s, %13.4s, v12.4s      \n"
                    "fmla   v22.4s, %13.4s, v13.4s      \n"
                    "fmla   v23.4s, %13.4s, v28.4s      \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v14.4h, v15.4h}, [%4]      \n" // r24 r25

                    "fmla   v16.4s, %17.4s, v12.4s      \n"
                    "fmla   v17.4s, %17.4s, v13.4s      \n"

                    "shll   v24.4s, v24.4h, #16         \n"

                    "fmla   v18.4s, %17.4s, v28.4s      \n"
                    "fmla   v19.4s, %17.4s, v29.4s      \n"

                    "shll   v25.4s, v25.4h, #16         \n"

                    "fmla   v20.4s, %14.4s, v12.4s      \n"
                    "fmla   v21.4s, %14.4s, v13.4s      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%2], #32 \n" // r00 r01 r02 r03

                    "fmla   v22.4s, %14.4s, v28.4s      \n"
                    "fmla   v23.4s, %14.4s, v29.4s      \n"

                    "shll   v26.4s, v26.4h, #16         \n"

                    "fmla   v16.4s, %18.4s, v24.4s      \n"
                    "fmla   v17.4s, %18.4s, v25.4s      \n"

                    "shll   v27.4s, v27.4h, #16         \n"

                    "fmla   v18.4s, %18.4s, v26.4s      \n"
                    "fmla   v19.4s, %18.4s, v27.4s      \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%5], #32 \n" // r30 r31 r32 r33

                    "fmla   v20.4s, %15.4s, v24.4s      \n"
                    "fmla   v21.4s, %15.4s, v25.4s      \n"

                    "shll   v14.4s, v14.4h, #16         \n"

                    "fmla   v22.4s, %15.4s, v26.4s      \n"
                    "fmla   v23.4s, %15.4s, v27.4s      \n"

                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v16.4s, %19.4s, v25.4s      \n"
                    "fmla   v17.4s, %19.4s, v26.4s      \n"

                    "fmla   v18.4s, %19.4s, v27.4s      \n"
                    "fmla   v19.4s, %19.4s, v14.4s      \n"

                    "fmla   v20.4s, %16.4s, v25.4s      \n"
                    "fmla   v21.4s, %16.4s, v26.4s      \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v24.4h, v25.4h}, [%2]      \n" // r04 r05

                    "fmla   v22.4s, %16.4s, v27.4s      \n"
                    "fmla   v23.4s, %16.4s, v14.4s      \n"

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "fmla   v16.4s, %20.4s, v26.4s      \n"
                    "fmla   v17.4s, %20.4s, v27.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"

                    "fmla   v18.4s, %20.4s, v14.4s      \n"
                    "fmla   v19.4s, %20.4s, v15.4s      \n"

                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v20.4s, %17.4s, v26.4s      \n"
                    "fmla   v21.4s, %17.4s, v27.4s      \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v26.4h, v27.4h}, [%5]      \n" // r34 r35

                    "fmla   v22.4s, %17.4s, v14.4s      \n"
                    "fmla   v23.4s, %17.4s, v15.4s      \n"

                    "shll   v28.4s, v28.4h, #16         \n"

                    "fmla   v16.4s, %12.4s, v10.4s      \n"
                    "fmla   v17.4s, %12.4s, v11.4s      \n"

                    "shll   v29.4s, v29.4h, #16         \n"

                    "fmla   v18.4s, %12.4s, v12.4s      \n"
                    "fmla   v19.4s, %12.4s, v13.4s      \n"

                    "shll   v30.4s, v30.4h, #16         \n"

                    "fmla   v20.4s, %18.4s, v28.4s      \n"
                    "fmla   v21.4s, %18.4s, v29.4s      \n"

                    "shll   v31.4s, v31.4h, #16         \n"

                    "fmla   v22.4s, %18.4s, v30.4s      \n"
                    "fmla   v23.4s, %18.4s, v31.4s      \n"

                    "shll   v24.4s, v24.4h, #16         \n"

                    "fmla   v16.4s, %13.4s, v11.4s      \n"
                    "fmla   v17.4s, %13.4s, v12.4s      \n"
                    "fmla   v18.4s, %13.4s, v13.4s      \n"
                    "fmla   v19.4s, %13.4s, v24.4s      \n"

                    "shll   v26.4s, v26.4h, #16         \n"

                    "fmla   v20.4s, %19.4s, v29.4s      \n"
                    "fmla   v21.4s, %19.4s, v30.4s      \n"
                    "fmla   v22.4s, %19.4s, v31.4s      \n"
                    "fmla   v23.4s, %19.4s, v26.4s      \n"

                    "shll   v25.4s, v25.4h, #16         \n"

                    "fmla   v16.4s, %14.4s, v12.4s      \n"
                    "fmla   v17.4s, %14.4s, v13.4s      \n"
                    "fmla   v18.4s, %14.4s, v24.4s      \n"
                    "fmla   v19.4s, %14.4s, v25.4s      \n"

                    "shll   v27.4s, v27.4h, #16         \n"

                    "fmla   v20.4s, %20.4s, v30.4s      \n"
                    "fmla   v21.4s, %20.4s, v31.4s      \n"
                    "fmla   v22.4s, %20.4s, v26.4s      \n"
                    "fmla   v23.4s, %20.4s, v27.4s      \n"

                    "shrn   v16.4h, v16.4s, #16         \n"
                    "shrn   v17.4h, v17.4s, #16         \n"
                    "shrn   v18.4h, v18.4s, #16         \n"
                    "shrn   v19.4h, v19.4s, #16         \n"
                    "shrn   v20.4h, v20.4s, #16         \n"
                    "shrn   v21.4h, v21.4s, #16         \n"
                    "shrn   v22.4h, v22.4s, #16         \n"
                    "shrn   v23.4h, v23.4s, #16         \n"

                    "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%0], #32 \n"
                    "st1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%1], #32 \n"

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
                    : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; j + 1 < outw; j += 2)
            {
                asm volatile(
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%3] \n" // r10 r11 r12 r13

                    "mov    v16.16b, %21.16b            \n" // sum00
                    "mov    v17.16b, %21.16b            \n" // sum01

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "mov    v18.16b, %21.16b            \n" // sum10
                    "mov    v19.16b, %21.16b            \n" // sum11

                    "fmla   v16.4s, %15.4s, v10.4s      \n"
                    "fmla   v17.4s, %15.4s, v11.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"

                    "fmla   v18.4s, %12.4s, v10.4s      \n"
                    "fmla   v19.4s, %12.4s, v11.4s      \n"

                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v16.4s, %16.4s, v11.4s      \n"
                    "fmla   v17.4s, %16.4s, v12.4s      \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%4] \n" // r20 r21 r22 r23

                    "fmla   v18.4s, %13.4s, v11.4s      \n"
                    "fmla   v19.4s, %13.4s, v12.4s      \n"

                    "shll   v20.4s, v20.4h, #16         \n"

                    "fmla   v16.4s, %17.4s, v12.4s      \n"
                    "fmla   v17.4s, %17.4s, v13.4s      \n"

                    "shll   v21.4s, v21.4h, #16         \n"

                    "fmla   v18.4s, %14.4s, v12.4s      \n"
                    "fmla   v19.4s, %14.4s, v13.4s      \n"

                    "shll   v22.4s, v22.4h, #16         \n"

                    "fmla   v16.4s, %18.4s, v20.4s      \n"
                    "fmla   v17.4s, %18.4s, v21.4s      \n"

                    "shll   v23.4s, v23.4h, #16         \n"

                    "fmla   v18.4s, %15.4s, v20.4s      \n"
                    "fmla   v19.4s, %15.4s, v21.4s      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%2] \n" // r00 r01 r02 r03

                    "fmla   v16.4s, %19.4s, v21.4s      \n"
                    "fmla   v17.4s, %19.4s, v22.4s      \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%5] \n" // r30 r31 r32 r33

                    "fmla   v18.4s, %16.4s, v21.4s      \n"
                    "fmla   v19.4s, %16.4s, v22.4s      \n"

                    "shll   v10.4s, v10.4h, #16         \n"

                    "fmla   v16.4s, %20.4s, v22.4s      \n"
                    "fmla   v17.4s, %20.4s, v23.4s      \n"

                    "shll   v24.4s, v24.4h, #16         \n"

                    "fmla   v18.4s, %17.4s, v22.4s      \n"
                    "fmla   v19.4s, %17.4s, v23.4s      \n"

                    "shll   v11.4s, v11.4h, #16         \n"
                    "shll   v25.4s, v25.4h, #16         \n"

                    "fmla   v16.4s, %12.4s, v10.4s      \n"
                    "fmla   v17.4s, %12.4s, v11.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"

                    "fmla   v18.4s, %18.4s, v24.4s      \n"
                    "fmla   v19.4s, %18.4s, v25.4s      \n"

                    "shll   v26.4s, v26.4h, #16         \n"

                    "fmla   v16.4s, %13.4s, v11.4s      \n"
                    "fmla   v17.4s, %13.4s, v12.4s      \n"

                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v18.4s, %19.4s, v25.4s      \n"
                    "fmla   v19.4s, %19.4s, v26.4s      \n"

                    "shll   v27.4s, v27.4h, #16         \n"

                    "fmla   v16.4s, %14.4s, v12.4s      \n"
                    "fmla   v17.4s, %14.4s, v13.4s      \n"

                    "add    %3, %3, #16                 \n"

                    "fmla   v18.4s, %20.4s, v26.4s      \n"
                    "fmla   v19.4s, %20.4s, v27.4s      \n"

                    "add    %4, %4, #16                 \n"

                    "shrn   v16.4h, v16.4s, #16         \n"
                    "shrn   v17.4h, v17.4s, #16         \n"

                    "add    %2, %2, #16                 \n"

                    "shrn   v18.4h, v18.4s, #16         \n"
                    "shrn   v19.4h, v19.4s, #16         \n"

                    "add    %5, %5, #16                 \n"

                    "st1    {v16.4h, v17.4h}, [%0], #16 \n"
                    "st1    {v18.4h, v19.4h}, [%1], #16 \n"

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
                    : "memory", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
            }
            for (; j < outw; j++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%3, #192]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h}, [%3] \n" // r10 r11 r12

                    "mov    v18.16b, %21.16b            \n" // sum0
                    "mov    v19.16b, %21.16b            \n" // sum1

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "fmul   v16.4s, %15.4s, v10.4s      \n"
                    "fmul   v17.4s, %12.4s, v10.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"

                    "fmla   v18.4s, %16.4s, v11.4s      \n"
                    "fmla   v19.4s, %13.4s, v11.4s      \n"

                    "prfm   pldl1keep, [%4, #192]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h}, [%4] \n" // r20 r21 r22

                    "fmla   v16.4s, %17.4s, v12.4s      \n"
                    "fmla   v17.4s, %14.4s, v12.4s      \n"

                    "shll   v20.4s, v20.4h, #16         \n"
                    "shll   v21.4s, v21.4h, #16         \n"

                    "fmla   v18.4s, %18.4s, v20.4s      \n"
                    "fmla   v19.4s, %15.4s, v20.4s      \n"

                    "prfm   pldl1keep, [%2, #192]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h}, [%2] \n" // r00 r01 r02

                    "shll   v22.4s, v22.4h, #16         \n"

                    "prfm   pldl1keep, [%5, #192]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h}, [%5] \n" // r30 r31 r32

                    "fmla   v16.4s, %19.4s, v21.4s      \n"
                    "fmla   v17.4s, %16.4s, v21.4s      \n"

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v24.4s, v24.4h, #16         \n"

                    "fmla   v18.4s, %20.4s, v22.4s      \n"
                    "fmla   v19.4s, %17.4s, v22.4s      \n"

                    "shll   v11.4s, v11.4h, #16         \n"
                    "shll   v25.4s, v25.4h, #16         \n"

                    "fmla   v16.4s, %12.4s, v10.4s      \n"
                    "fmla   v17.4s, %18.4s, v24.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v26.4s, v26.4h, #16         \n"

                    "fmla   v18.4s, %13.4s, v11.4s      \n"
                    "fmla   v19.4s, %19.4s, v25.4s      \n"

                    "add    %3, %3, #8                  \n"

                    "fmla   v16.4s, %14.4s, v12.4s      \n"
                    "fmla   v17.4s, %20.4s, v26.4s      \n"

                    "add    %4, %4, #8                  \n"

                    "fadd   v18.4s, v18.4s, v16.4s      \n"
                    "fadd   v19.4s, v19.4s, v17.4s      \n"

                    "add    %2, %2, #8                  \n"

                    "shrn   v18.4h, v18.4s, #16         \n"
                    "shrn   v19.4h, v19.4s, #16         \n"

                    "add    %5, %5, #8                  \n"

                    "st1    {v18.4h}, [%0], #8          \n"
                    "st1    {v19.4h}, [%1], #8          \n"

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
                    : "memory", "v10", "v11", "v12", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26");
            }

            r0 += 2 * 4 + w * 4;
            r1 += 2 * 4 + w * 4;
            r2 += 2 * 4 + w * 4;
            r3 += 2 * 4 + w * 4;

            outptr0 += outw * 4;
            outptr1 += outw * 4;
        }
#endif // __aarch64__
        for (; i < outh; i++)
        {
            int j = 0;

            for (; j + 3 < outw; j += 4)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%1], #32 \n" // r00 r01 r02 r03

                    "mov    v16.16b, %17.16b            \n" // sum00
                    "mov    v17.16b, %17.16b            \n" // sum01
                    "mov    v18.16b, %17.16b            \n" // sum02
                    "mov    v19.16b, %17.16b            \n" // sum03

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "fmla   v16.4s, %8.4s, v10.4s       \n"
                    "fmla   v17.4s, %8.4s, v11.4s       \n"

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v18.4s, %8.4s, v12.4s       \n"
                    "fmla   v19.4s, %8.4s, v13.4s       \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v14.4h, v15.4h}, [%1]      \n" // r04 r05

                    "fmla   v16.4s, %9.4s, v11.4s       \n"
                    "fmla   v17.4s, %9.4s, v12.4s       \n"

                    "shll   v14.4s, v14.4h, #16         \n"

                    "fmla   v18.4s, %9.4s, v13.4s       \n"
                    "fmla   v19.4s, %9.4s, v14.4s       \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n" // r10 r11 r12 r13

                    "fmla   v16.4s, %10.4s, v12.4s      \n"
                    "fmla   v17.4s, %10.4s, v13.4s      \n"

                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v18.4s, %10.4s, v14.4s      \n"
                    "fmla   v19.4s, %10.4s, v15.4s      \n"

                    "shll   v20.4s, v20.4h, #16         \n"
                    "shll   v21.4s, v21.4h, #16         \n"

                    "fmla   v16.4s, %11.4s, v20.4s      \n"
                    "fmla   v17.4s, %11.4s, v21.4s      \n"

                    "shll   v22.4s, v22.4h, #16         \n"
                    "shll   v23.4s, v23.4h, #16         \n"

                    "fmla   v18.4s, %11.4s, v22.4s      \n"
                    "fmla   v19.4s, %11.4s, v23.4s      \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v14.4h, v15.4h}, [%2]      \n" // r14 r15

                    "fmla   v16.4s, %12.4s, v21.4s      \n"
                    "fmla   v17.4s, %12.4s, v22.4s      \n"

                    "shll   v14.4s, v14.4h, #16         \n"

                    "fmla   v18.4s, %12.4s, v23.4s      \n"
                    "fmla   v19.4s, %12.4s, v14.4s      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%3], #32 \n" // r20 r21 r22 r23

                    "fmla   v16.4s, %13.4s, v22.4s      \n"
                    "fmla   v17.4s, %13.4s, v23.4s      \n"

                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v18.4s, %13.4s, v14.4s      \n"
                    "fmla   v19.4s, %13.4s, v15.4s      \n"

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "fmla   v16.4s, %14.4s, v10.4s      \n"
                    "fmla   v17.4s, %14.4s, v11.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v18.4s, %14.4s, v12.4s      \n"
                    "fmla   v19.4s, %14.4s, v13.4s      \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v14.4h, v15.4h}, [%3]      \n" // r24 r25

                    "fmla   v16.4s, %15.4s, v11.4s      \n"
                    "fmla   v17.4s, %15.4s, v12.4s      \n"

                    "shll   v14.4s, v14.4h, #16         \n"

                    "fmla   v18.4s, %15.4s, v13.4s      \n"
                    "fmla   v19.4s, %15.4s, v14.4s      \n"

                    "fmla   v16.4s, %16.4s, v12.4s      \n"
                    "fmla   v17.4s, %16.4s, v13.4s      \n"

                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v18.4s, %16.4s, v14.4s      \n"
                    "fmla   v19.4s, %16.4s, v15.4s      \n"

                    "shrn   v16.4h, v16.4s, #16         \n"
                    "shrn   v17.4h, v17.4s, #16         \n"
                    "shrn   v18.4h, v18.4s, #16         \n"
                    "shrn   v19.4h, v19.4s, #16         \n"

                    "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%0], #32 \n"

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
                    : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                asm volatile(
                    "pld        [%1, #128]      \n"
                    "vld1.u16   {d30-d31}, [%1 :64]! \n" // r00 r01

                    "vmov       q10, %q17       \n" // sum00
                    "vmov       q11, %q17       \n" // sum01

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q10, %q8, q14   \n"
                    "vmla.f32   q11, %q8, q15   \n"
                    "vmla.f32   q10, %q9, q15   \n"

                    "pld        [%1, #128]      \n"
                    "vld1.u16   {d30-d31}, [%1 :64]! \n" // r02 r03

                    "vmov       q12, %q17       \n" // sum02
                    "vmov       q13, %q17       \n" // sum03

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q12, %q8, q14   \n"
                    "vmla.f32   q11, %q9, q14   \n"
                    "vmla.f32   q13, %q8, q15   \n"
                    "vmla.f32   q10, %q10, q14  \n"
                    "vmla.f32   q12, %q9, q15   \n"
                    "vmla.f32   q11, %q10, q15  \n"

                    //                     "pld        [%1, #128]      \n"
                    "vld1.u16   {d30-d31}, [%1 :64] \n" // r04 r05

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q13, %q9, q14   \n"
                    "vmla.f32   q12, %q10, q14  \n"
                    "vmla.f32   q13, %q10, q15  \n"

                    "pld        [%2, #128]      \n"
                    "vld1.u16   {d30-d31}, [%2 :64]! \n" // r10 r11

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q10, %q11, q14  \n"
                    "vmla.f32   q11, %q11, q15  \n"
                    "vmla.f32   q10, %q12, q15  \n"

                    "pld        [%2, #128]      \n"
                    "vld1.u16   {d30-d31}, [%2 :64]! \n" // r12 r13

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q12, %q11, q14  \n"
                    "vmla.f32   q11, %q12, q14  \n"
                    "vmla.f32   q13, %q11, q15  \n"
                    "vmla.f32   q10, %q13, q14  \n"
                    "vmla.f32   q12, %q12, q15  \n"
                    "vmla.f32   q11, %q13, q15  \n"

                    //                     "pld        [%2, #128]      \n"
                    "vld1.u16   {d30-d31}, [%2 :64] \n" // r14 r15

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q13, %q12, q14  \n"
                    "vmla.f32   q12, %q13, q14  \n"
                    "vmla.f32   q13, %q13, q15  \n"

                    "pld        [%3, #128]      \n"
                    "vld1.u16   {d30-d31}, [%3 :64]! \n" // r20 r21

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q10, %q14, q14  \n"
                    "vmla.f32   q11, %q14, q15  \n"
                    "vmla.f32   q10, %q15, q15  \n"

                    "pld        [%3, #128]      \n"
                    "vld1.u16   {d30-d31}, [%3 :64]! \n" // r22 r23

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q12, %q14, q14  \n"
                    "vmla.f32   q11, %q15, q14  \n"
                    "vmla.f32   q13, %q14, q15  \n"
                    "vmla.f32   q10, %q16, q14  \n"
                    "vmla.f32   q12, %q15, q15  \n"
                    "vmla.f32   q11, %q16, q15  \n"

                    //                     "pld        [%3, #128]      \n"
                    "vld1.u16   {d30-d31}, [%3 :64] \n" // r24 r25

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q13, %q15, q14  \n"
                    "vmla.f32   q12, %q16, q14  \n"
                    "vmla.f32   q13, %q16, q15  \n"

                    "vshrn.u32  d20, q10, #16   \n"
                    "vshrn.u32  d21, q11, #16   \n"
                    "vshrn.u32  d22, q12, #16   \n"
                    "vshrn.u32  d23, q13, #16   \n"

                    "vst1.u16   {d20-d23}, [%0 :64]! \n"

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
                    : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
            }
            for (; j + 1 < outw; j += 2)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%1] \n" // r00 r01 r02 r03

                    "mov    v18.16b, %17.16b            \n" // sum00
                    "mov    v19.16b, %17.16b            \n" // sum01

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmul   v16.4s, %8.4s, v12.4s       \n"
                    "fmul   v17.4s, %8.4s, v13.4s       \n"

                    "shll   v14.4s, v14.4h, #16         \n"
                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v18.4s, %9.4s, v13.4s       \n"
                    "fmla   v19.4s, %9.4s, v14.4s       \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2] \n" // r10 r11 r12 r13

                    "fmla   v16.4s, %10.4s, v14.4s      \n"
                    "fmla   v17.4s, %10.4s, v15.4s      \n"

                    "shll   v20.4s, v20.4h, #16         \n"
                    "shll   v21.4s, v21.4h, #16         \n"

                    "fmla   v18.4s, %11.4s, v20.4s      \n"
                    "fmla   v19.4s, %11.4s, v21.4s      \n"

                    "shll   v22.4s, v22.4h, #16         \n"
                    "shll   v23.4s, v23.4h, #16         \n"

                    "fmla   v16.4s, %12.4s, v21.4s      \n"
                    "fmla   v17.4s, %12.4s, v22.4s      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%3] \n" // r20 r21 r22 r23

                    "fmla   v18.4s, %13.4s, v22.4s      \n"
                    "fmla   v19.4s, %13.4s, v23.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v16.4s, %14.4s, v12.4s      \n"
                    "fmla   v17.4s, %14.4s, v13.4s      \n"

                    "shll   v14.4s, v14.4h, #16         \n"
                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v18.4s, %15.4s, v13.4s      \n"
                    "fmla   v19.4s, %15.4s, v14.4s      \n"

                    "add    %1, %1, #16                 \n"

                    "fmla   v16.4s, %16.4s, v14.4s      \n"
                    "fmla   v17.4s, %16.4s, v15.4s      \n"

                    "add    %2, %2, #16                 \n"

                    "fadd   v18.4s, v18.4s, v16.4s      \n"
                    "fadd   v19.4s, v19.4s, v17.4s      \n"

                    "add    %3, %3, #16                 \n"

                    "shrn   v18.4h, v18.4s, #16         \n"
                    "shrn   v19.4h, v19.4s, #16         \n"

                    "st1    {v18.4h, v19.4h}, [%0], #16 \n"

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
                    : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                asm volatile(
                    "pld        [%1, #256]      \n"
                    "vld1.u16   {d28-d31}, [%1 :64] \n" // r00 r01 r02 r03

                    "vmov       q10, %q17       \n" // sum00
                    "vmov       q11, %q17       \n" // sum01

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"

                    "vmla.f32   q10, %q8, q12   \n"
                    "vmla.f32   q11, %q8, q13   \n"

                    "vshll.u16  q14, d30, #16   \n"

                    "vmla.f32   q10, %q9, q13   \n"
                    "vmla.f32   q11, %q9, q14   \n"

                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q10, %q10, q14  \n"
                    "vmla.f32   q11, %q10, q15  \n"

                    "pld        [%2, #256]      \n"
                    "vld1.u16   {d28-d31}, [%2 :64] \n" // r10 r11 r12 r13

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"

                    "vmla.f32   q10, %q11, q12  \n"
                    "vmla.f32   q11, %q11, q13  \n"

                    "vshll.u16  q14, d30, #16   \n"

                    "vmla.f32   q10, %q12, q13  \n"
                    "vmla.f32   q11, %q12, q14  \n"

                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q10, %q13, q14  \n"
                    "vmla.f32   q11, %q13, q15  \n"

                    "pld        [%3, #256]      \n"
                    "vld1.u16   {d28-d31}, [%3 :64] \n" // r20 r21 r22 r23

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"

                    "vmla.f32   q10, %q14, q12  \n"
                    "vmla.f32   q11, %q14, q13  \n"

                    "vshll.u16  q14, d30, #16   \n"

                    "vmla.f32   q10, %q15, q13  \n"
                    "vmla.f32   q11, %q15, q14  \n"

                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q10, %q16, q14  \n"
                    "vmla.f32   q11, %q16, q15  \n"

                    "add        %1, %1, #16     \n"
                    "add        %2, %2, #16     \n"

                    "vshrn.u32  d20, q10, #16   \n"
                    "vshrn.u32  d21, q11, #16   \n"

                    "add        %3, %3, #16     \n"

                    "vst1.u16   {d20-d21}, [%0 :64]! \n"

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
                    : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
            }
            for (; j < outw; j++)
            {
                float32x4_t _sum0 = _bias0;

                float32x4_t _r00 = vcvt_f32_bf16(vld1_u16(r0));
                float32x4_t _r01 = vcvt_f32_bf16(vld1_u16(r0 + 4));
                float32x4_t _r02 = vcvt_f32_bf16(vld1_u16(r0 + 8));
                float32x4_t _r10 = vcvt_f32_bf16(vld1_u16(r1));
                float32x4_t _r11 = vcvt_f32_bf16(vld1_u16(r1 + 4));
                float32x4_t _r12 = vcvt_f32_bf16(vld1_u16(r1 + 8));
                float32x4_t _r20 = vcvt_f32_bf16(vld1_u16(r2));
                float32x4_t _r21 = vcvt_f32_bf16(vld1_u16(r2 + 4));
                float32x4_t _r22 = vcvt_f32_bf16(vld1_u16(r2 + 8));

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);

                vst1_u16(outptr0, vcvt_bf16_f32(_sum0));

                r0 += 4;
                r1 += 4;
                r2 += 4;
                outptr0 += 4;
            }

            r0 += 2 * 4;
            r1 += 2 * 4;
            r2 += 2 * 4;
        }
    }
}

static void convdw3x3s2_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 4;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

        const unsigned short* k0 = kernel.row<const unsigned short>(g);

        unsigned short* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const unsigned short* r0 = img0.row<const unsigned short>(0);
        const unsigned short* r1 = img0.row<const unsigned short>(1);
        const unsigned short* r2 = img0.row<const unsigned short>(2);

        float32x4_t _k00 = vcvt_f32_bf16(vld1_u16(k0));
        float32x4_t _k01 = vcvt_f32_bf16(vld1_u16(k0 + 4));
        float32x4_t _k02 = vcvt_f32_bf16(vld1_u16(k0 + 8));
        float32x4_t _k10 = vcvt_f32_bf16(vld1_u16(k0 + 12));
        float32x4_t _k11 = vcvt_f32_bf16(vld1_u16(k0 + 16));
        float32x4_t _k12 = vcvt_f32_bf16(vld1_u16(k0 + 20));
        float32x4_t _k20 = vcvt_f32_bf16(vld1_u16(k0 + 24));
        float32x4_t _k21 = vcvt_f32_bf16(vld1_u16(k0 + 28));
        float32x4_t _k22 = vcvt_f32_bf16(vld1_u16(k0 + 32));

        int i = 0;

        for (; i < outh; i++)
        {
            int j = 0;

#if __aarch64__
            for (; j + 3 < outw; j += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%1], #32 \n" // r00 r01 r02 r03

                    "mov    v28.16b, %17.16b            \n" // sum00
                    "mov    v29.16b, %17.16b            \n" // sum01
                    "mov    v30.16b, %17.16b            \n" // sum02
                    "mov    v31.16b, %17.16b            \n" // sum03

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v14.4h, v15.4h, v16.4h, v17.4h}, [%1], #32 \n" // r04 r05 r06 r07

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"
                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "prfm   pldl1keep, [%1, #64]        \n"
                    "ld1    {v18.4h}, [%1]              \n" // r08

                    "shll   v14.4s, v14.4h, #16         \n"
                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v28.4s, %8.4s, v10.4s       \n"
                    "fmla   v29.4s, %8.4s, v12.4s       \n"

                    "shll   v16.4s, v16.4h, #16         \n"

                    "fmla   v30.4s, %8.4s, v14.4s       \n"
                    "fmla   v31.4s, %8.4s, v16.4s       \n"

                    "shll   v17.4s, v17.4h, #16         \n"

                    "fmla   v28.4s, %9.4s, v11.4s       \n"
                    "fmla   v29.4s, %9.4s, v13.4s       \n"
                    "fmla   v30.4s, %9.4s, v15.4s       \n"
                    "fmla   v31.4s, %9.4s, v17.4s       \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n" // r10 r11 r12 r13

                    "fmla   v28.4s, %10.4s, v12.4s      \n"
                    "fmla   v29.4s, %10.4s, v14.4s      \n"

                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmla   v30.4s, %10.4s, v16.4s      \n"
                    "fmla   v31.4s, %10.4s, v18.4s      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n" // r14 r15 r16 r17

                    "shll   v20.4s, v20.4h, #16         \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "shll   v23.4s, v23.4h, #16         \n"

                    "prfm   pldl1keep, [%2, #64]        \n"
                    "ld1    {v19.4h}, [%2]              \n" // r18

                    "shll   v24.4s, v24.4h, #16         \n"
                    "shll   v25.4s, v25.4h, #16         \n"

                    "fmla   v28.4s, %11.4s, v20.4s      \n"
                    "fmla   v29.4s, %11.4s, v22.4s      \n"

                    "shll   v26.4s, v26.4h, #16         \n"

                    "fmla   v30.4s, %11.4s, v24.4s      \n"
                    "fmla   v31.4s, %11.4s, v26.4s      \n"

                    "shll   v27.4s, v27.4h, #16         \n"

                    "fmla   v28.4s, %12.4s, v21.4s      \n"
                    "fmla   v29.4s, %12.4s, v23.4s      \n"
                    "fmla   v30.4s, %12.4s, v25.4s      \n"
                    "fmla   v31.4s, %12.4s, v27.4s      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%3], #32 \n" // r20 r21 r22 r23

                    "fmla   v28.4s, %13.4s, v22.4s      \n"
                    "fmla   v29.4s, %13.4s, v24.4s      \n"

                    "shll   v19.4s, v19.4h, #16         \n"

                    "fmla   v30.4s, %13.4s, v26.4s      \n"
                    "fmla   v31.4s, %13.4s, v19.4s      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v14.4h, v15.4h, v16.4h, v17.4h}, [%3], #32 \n" // r24 r25 r26 r27

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"
                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "prfm   pldl1keep, [%3, #64]        \n"
                    "ld1    {v18.4h}, [%3]              \n" // r28

                    "shll   v14.4s, v14.4h, #16         \n"
                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v28.4s, %14.4s, v10.4s      \n"
                    "fmla   v29.4s, %14.4s, v12.4s      \n"

                    "shll   v16.4s, v16.4h, #16         \n"

                    "fmla   v30.4s, %14.4s, v14.4s      \n"
                    "fmla   v31.4s, %14.4s, v16.4s      \n"

                    "shll   v17.4s, v17.4h, #16         \n"

                    "fmla   v28.4s, %15.4s, v11.4s      \n"
                    "fmla   v29.4s, %15.4s, v13.4s      \n"
                    "fmla   v30.4s, %15.4s, v15.4s      \n"
                    "fmla   v31.4s, %15.4s, v17.4s      \n"

                    "fmla   v28.4s, %16.4s, v12.4s      \n"
                    "fmla   v29.4s, %16.4s, v14.4s      \n"

                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmla   v30.4s, %16.4s, v16.4s      \n"
                    "fmla   v31.4s, %16.4s, v18.4s      \n"

                    "shrn   v28.4h, v28.4s, #16         \n"
                    "shrn   v29.4h, v29.4s, #16         \n"
                    "shrn   v30.4h, v30.4s, #16         \n"
                    "shrn   v31.4h, v31.4s, #16         \n"

                    "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%0], #32 \n"

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
                    : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#endif // __aarch64__
            for (; j + 1 < outw; j += 2)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%1], #32 \n" // r00 r01 r02 r03

                    "mov    v22.16b, %17.16b            \n" // sum00
                    "mov    v23.16b, %17.16b            \n" // sum01

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "fmul   v20.4s, %8.4s, v10.4s       \n"

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmul   v21.4s, %8.4s, v12.4s       \n"

                    "prfm   pldl1keep, [%1, #64]        \n"
                    "ld1    {v14.4h}, [%1]              \n" // r04

                    "fmla   v22.4s, %9.4s, v11.4s       \n"
                    "fmla   v23.4s, %9.4s, v13.4s       \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%2], #32 \n" // r10 r11 r12 r13

                    "shll   v14.4s, v14.4h, #16         \n"

                    "fmla   v20.4s, %10.4s, v12.4s      \n"
                    "fmla   v21.4s, %10.4s, v14.4s      \n"

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"

                    "fmla   v22.4s, %11.4s, v16.4s      \n"

                    "shll   v18.4s, v18.4h, #16         \n"
                    "shll   v19.4s, v19.4h, #16         \n"

                    "fmla   v23.4s, %11.4s, v18.4s      \n"

                    "prfm   pldl1keep, [%2, #64]        \n"
                    "ld1    {v15.4h}, [%2]              \n" // r14

                    "fmla   v20.4s, %12.4s, v17.4s      \n"
                    "fmla   v21.4s, %12.4s, v19.4s      \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%3], #32 \n" // r20 r21 r22 r23

                    "shll   v15.4s, v15.4h, #16         \n"

                    "fmla   v22.4s, %13.4s, v18.4s      \n"
                    "fmla   v23.4s, %13.4s, v15.4s      \n"

                    "shll   v10.4s, v10.4h, #16         \n"
                    "shll   v11.4s, v11.4h, #16         \n"

                    "fmla   v20.4s, %14.4s, v10.4s      \n"

                    "shll   v12.4s, v12.4h, #16         \n"
                    "shll   v13.4s, v13.4h, #16         \n"

                    "fmla   v21.4s, %14.4s, v12.4s      \n"

                    "prfm   pldl1keep, [%3, #64]        \n"
                    "ld1    {v14.4h}, [%3]              \n" // r24

                    "fmla   v22.4s, %15.4s, v11.4s      \n"
                    "fmla   v23.4s, %15.4s, v13.4s      \n"

                    "shll   v14.4s, v14.4h, #16         \n"

                    "fmla   v20.4s, %16.4s, v12.4s      \n"
                    "fmla   v21.4s, %16.4s, v14.4s      \n"

                    "fadd   v22.4s, v20.4s, v22.4s      \n"
                    "fadd   v23.4s, v21.4s, v23.4s      \n"

                    "shrn   v22.4h, v22.4s, #16         \n"
                    "shrn   v23.4h, v23.4s, #16         \n"

                    "st1    {v22.4h, v23.4h}, [%0], #16 \n"

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
                    : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                asm volatile(
                    "pld        [%1, #256]      \n"
                    "vld1.u16   {d28-d31}, [%1 :64]! \n" // r00 r01 r02 r03

                    "vmov       q10, %q17       \n" // sum00
                    "vmov       q11, %q17       \n" // sum01

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"

                    "vmla.f32   q10, %q8, q12   \n"

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q11, %q8, q14   \n"

                    "vld1.u16   {d25}, [%1]     \n" // r04

                    "vmla.f32   q10, %q9, q13   \n"
                    "vmla.f32   q11, %q9, q15   \n"

                    "vshll.u16  q12, d25, #16   \n"

                    "vmla.f32   q10, %q10, q14  \n"

                    "pld        [%2, #256]      \n"
                    "vld1.u16   {d28-d31}, [%2 :64]! \n" // r10 r11 r12 r13

                    "vmla.f32   q11, %q10, q12  \n"

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"

                    "vmla.f32   q10, %q11, q12  \n"

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q11, %q11, q14  \n"

                    "vld1.u16   {d25}, [%2]     \n" // r14

                    "vmla.f32   q10, %q12, q13  \n"
                    "vmla.f32   q11, %q12, q15  \n"

                    "vshll.u16  q12, d25, #16   \n"

                    "vmla.f32   q10, %q13, q14  \n"

                    "pld        [%3, #256]      \n"
                    "vld1.u16   {d28-d31}, [%3 :64]! \n" // r20 r21 r22 r23

                    "vmla.f32   q11, %q13, q12  \n"

                    "vshll.u16  q12, d28, #16   \n"
                    "vshll.u16  q13, d29, #16   \n"

                    "vmla.f32   q10, %q14, q12  \n"

                    "vshll.u16  q14, d30, #16   \n"
                    "vshll.u16  q15, d31, #16   \n"

                    "vmla.f32   q11, %q14, q14  \n"

                    "vld1.u16   {d25}, [%3]     \n" // r24

                    "vmla.f32   q10, %q15, q13  \n"
                    "vmla.f32   q11, %q15, q15  \n"

                    "vshll.u16  q12, d25, #16   \n"

                    "vmla.f32   q10, %q16, q14  \n"
                    "vmla.f32   q11, %q16, q12  \n"

                    "vshrn.u32  d20, q10, #16   \n"
                    "vshrn.u32  d21, q11, #16   \n"

                    "vst1.u16   {d20-d21}, [%0 :64]! \n"

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
                    : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
            }
            for (; j < outw; j++)
            {
                float32x4_t _sum0 = _bias0;

                float32x4_t _r00 = vcvt_f32_bf16(vld1_u16(r0));
                float32x4_t _r01 = vcvt_f32_bf16(vld1_u16(r0 + 4));
                float32x4_t _r02 = vcvt_f32_bf16(vld1_u16(r0 + 8));
                float32x4_t _r10 = vcvt_f32_bf16(vld1_u16(r1));
                float32x4_t _r11 = vcvt_f32_bf16(vld1_u16(r1 + 4));
                float32x4_t _r12 = vcvt_f32_bf16(vld1_u16(r1 + 8));
                float32x4_t _r20 = vcvt_f32_bf16(vld1_u16(r2));
                float32x4_t _r21 = vcvt_f32_bf16(vld1_u16(r2 + 4));
                float32x4_t _r22 = vcvt_f32_bf16(vld1_u16(r2 + 8));

                _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                _sum0 = vmlaq_f32(_sum0, _k22, _r22);

                vst1_u16(outptr0, vcvt_bf16_f32(_sum0));

                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
                outptr0 += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
