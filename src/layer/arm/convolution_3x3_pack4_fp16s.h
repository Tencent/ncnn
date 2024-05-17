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

static void conv3x3s1_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x4_t _bias0 = bias ? vld1_f16(bias + p * 4) : vdup_n_f16((__fp16)0.f);
        out0.fill(_bias0);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0.row<__fp16>(0);

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            // 16 * 9
            float16x8_t _k00_01 = vld1q_f16(kptr);
            float16x8_t _k00_23 = vld1q_f16(kptr + 8);
            float16x8_t _k01_01 = vld1q_f16(kptr + 16);
            float16x8_t _k01_23 = vld1q_f16(kptr + 24);
            float16x8_t _k02_01 = vld1q_f16(kptr + 32);
            float16x8_t _k02_23 = vld1q_f16(kptr + 40);
            float16x8_t _k10_01 = vld1q_f16(kptr + 48);
            float16x8_t _k10_23 = vld1q_f16(kptr + 56);
            float16x8_t _k11_01 = vld1q_f16(kptr + 64);
            float16x8_t _k11_23 = vld1q_f16(kptr + 72);
            float16x8_t _k12_01 = vld1q_f16(kptr + 80);
            float16x8_t _k12_23 = vld1q_f16(kptr + 88);
            float16x8_t _k20_01 = vld1q_f16(kptr + 96);
            float16x8_t _k20_23 = vld1q_f16(kptr + 104);
            float16x8_t _k21_01 = vld1q_f16(kptr + 112);
            float16x8_t _k21_23 = vld1q_f16(kptr + 120);
            float16x8_t _k22_01 = vld1q_f16(kptr + 128);
            float16x8_t _k22_23 = vld1q_f16(kptr + 136);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%1] \n" // r00 r01 r02 r03 r04 r05

                        "ext    v6.16b, %8.16b, %8.16b, #8  \n"
                        "fmla   v10.4h, %8.4h, v0.h[0]      \n"
                        "fmla   v11.4h, %8.4h, v0.h[4]      \n"
                        "fmla   v12.4h, %8.4h, v1.h[0]      \n"
                        "fmla   v13.4h, %8.4h, v1.h[4]      \n"
                        "fmla   v10.4h, v6.4h, v0.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[5]      \n"
                        "ext    v7.16b, %9.16b, %9.16b, #8  \n"
                        "fmla   v10.4h, %9.4h, v0.h[2]      \n"
                        "fmla   v11.4h, %9.4h, v0.h[6]      \n"
                        "fmla   v12.4h, %9.4h, v1.h[2]      \n"
                        "fmla   v13.4h, %9.4h, v1.h[6]      \n"
                        "fmla   v10.4h, v7.4h, v0.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[7]      \n"

                        "ext    v8.16b, %10.16b, %10.16b, #8 \n"
                        "fmla   v10.4h, %10.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %10.4h, v1.h[0]     \n"
                        "fmla   v12.4h, %10.4h, v1.h[4]     \n"
                        "fmla   v13.4h, %10.4h, v2.h[0]     \n"
                        "fmla   v10.4h, v8.4h, v0.h[5]      \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v12.4h, v8.4h, v1.h[5]      \n"
                        "fmla   v13.4h, v8.4h, v2.h[1]      \n"
                        "ext    v9.16b, %11.16b, %11.16b, #8 \n"
                        "fmla   v10.4h, %11.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %11.4h, v1.h[2]     \n"
                        "fmla   v12.4h, %11.4h, v1.h[6]     \n"
                        "fmla   v13.4h, %11.4h, v2.h[2]     \n"
                        "fmla   v10.4h, v9.4h, v0.h[7]      \n"
                        "fmla   v11.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v12.4h, v9.4h, v1.h[7]      \n"
                        "fmla   v13.4h, v9.4h, v2.h[3]      \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.8h, v4.8h, v5.8h}, [%2] \n" // r10 r11 r12 r13 r14 r15

                        "ext    v6.16b, %12.16b, %12.16b, #8 \n"
                        "fmla   v10.4h, %12.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %12.4h, v1.h[4]     \n"
                        "fmla   v12.4h, %12.4h, v2.h[0]     \n"
                        "fmla   v13.4h, %12.4h, v2.h[4]     \n"
                        "fmla   v10.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v1.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v2.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v2.h[5]      \n"
                        "ext    v7.16b, %13.16b, %13.16b, #8 \n"
                        "fmla   v10.4h, %13.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %13.4h, v1.h[6]     \n"
                        "fmla   v12.4h, %13.4h, v2.h[2]     \n"
                        "fmla   v13.4h, %13.4h, v2.h[6]     \n"
                        "fmla   v10.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v1.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v2.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v2.h[7]      \n"

                        "ext    v8.16b, %14.16b, %14.16b, #8 \n"
                        "fmla   v10.4h, %14.4h, v3.h[0]     \n"
                        "fmla   v11.4h, %14.4h, v3.h[4]     \n"
                        "fmla   v12.4h, %14.4h, v4.h[0]     \n"
                        "fmla   v13.4h, %14.4h, v4.h[4]     \n"
                        "fmla   v10.4h, v8.4h, v3.h[1]      \n"
                        "fmla   v11.4h, v8.4h, v3.h[5]      \n"
                        "fmla   v12.4h, v8.4h, v4.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v4.h[5]      \n"
                        "ext    v9.16b, %15.16b, %15.16b, #8 \n"
                        "fmla   v10.4h, %15.4h, v3.h[2]     \n"
                        "fmla   v11.4h, %15.4h, v3.h[6]     \n"
                        "fmla   v12.4h, %15.4h, v4.h[2]     \n"
                        "fmla   v13.4h, %15.4h, v4.h[6]     \n"
                        "fmla   v10.4h, v9.4h, v3.h[3]      \n"
                        "fmla   v11.4h, v9.4h, v3.h[7]      \n"
                        "fmla   v12.4h, v9.4h, v4.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v4.h[7]      \n"

                        "ext    v6.16b, %16.16b, %16.16b, #8 \n"
                        "fmla   v10.4h, %16.4h, v3.h[4]     \n"
                        "fmla   v11.4h, %16.4h, v4.h[0]     \n"
                        "fmla   v12.4h, %16.4h, v4.h[4]     \n"
                        "fmla   v13.4h, %16.4h, v5.h[0]     \n"
                        "fmla   v10.4h, v6.4h, v3.h[5]      \n"
                        "fmla   v11.4h, v6.4h, v4.h[1]      \n"
                        "fmla   v12.4h, v6.4h, v4.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v5.h[1]      \n"
                        "ext    v7.16b, %17.16b, %17.16b, #8 \n"
                        "fmla   v10.4h, %17.4h, v3.h[6]     \n"
                        "fmla   v11.4h, %17.4h, v4.h[2]     \n"
                        "fmla   v12.4h, %17.4h, v4.h[6]     \n"
                        "fmla   v13.4h, %17.4h, v5.h[2]     \n"
                        "fmla   v10.4h, v7.4h, v3.h[7]      \n"
                        "fmla   v11.4h, v7.4h, v4.h[3]      \n"
                        "fmla   v12.4h, v7.4h, v4.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v5.h[3]      \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%3] \n" // r20 r21 r22 r23 r24 r25

                        "ext    v8.16b, %18.16b, %18.16b, #8 \n"
                        "fmla   v10.4h, %18.4h, v4.h[0]     \n"
                        "fmla   v11.4h, %18.4h, v4.h[4]     \n"
                        "fmla   v12.4h, %18.4h, v5.h[0]     \n"
                        "fmla   v13.4h, %18.4h, v5.h[4]     \n"
                        "fmla   v10.4h, v8.4h, v4.h[1]      \n"
                        "fmla   v11.4h, v8.4h, v4.h[5]      \n"
                        "fmla   v12.4h, v8.4h, v5.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v5.h[5]      \n"
                        "ext    v9.16b, %19.16b, %19.16b, #8 \n"
                        "fmla   v10.4h, %19.4h, v4.h[2]     \n"
                        "fmla   v11.4h, %19.4h, v4.h[6]     \n"
                        "fmla   v12.4h, %19.4h, v5.h[2]     \n"
                        "fmla   v13.4h, %19.4h, v5.h[6]     \n"
                        "fmla   v10.4h, v9.4h, v4.h[3]      \n"
                        "fmla   v11.4h, v9.4h, v4.h[7]      \n"
                        "fmla   v12.4h, v9.4h, v5.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v5.h[7]      \n"

                        "ext    v6.16b, %20.16b, %20.16b, #8 \n"
                        "fmla   v10.4h, %20.4h, v0.h[0]     \n"
                        "fmla   v11.4h, %20.4h, v0.h[4]     \n"
                        "fmla   v12.4h, %20.4h, v1.h[0]     \n"
                        "fmla   v13.4h, %20.4h, v1.h[4]     \n"
                        "fmla   v10.4h, v6.4h, v0.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[5]      \n"
                        "ext    v7.16b, %21.16b, %21.16b, #8 \n"
                        "fmla   v10.4h, %21.4h, v0.h[2]     \n"
                        "fmla   v11.4h, %21.4h, v0.h[6]     \n"
                        "fmla   v12.4h, %21.4h, v1.h[2]     \n"
                        "fmla   v13.4h, %21.4h, v1.h[6]     \n"
                        "fmla   v10.4h, v7.4h, v0.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[7]      \n"

                        "ext    v8.16b, %22.16b, %22.16b, #8 \n"
                        "fmla   v10.4h, %22.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %22.4h, v1.h[0]     \n"
                        "fmla   v12.4h, %22.4h, v1.h[4]     \n"
                        "fmla   v13.4h, %22.4h, v2.h[0]     \n"
                        "fmla   v10.4h, v8.4h, v0.h[5]      \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v12.4h, v8.4h, v1.h[5]      \n"
                        "fmla   v13.4h, v8.4h, v2.h[1]      \n"
                        "ext    v9.16b, %23.16b, %23.16b, #8 \n"
                        "fmla   v10.4h, %23.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %23.4h, v1.h[2]     \n"
                        "fmla   v12.4h, %23.4h, v1.h[6]     \n"
                        "fmla   v13.4h, %23.4h, v2.h[2]     \n"
                        "fmla   v10.4h, v9.4h, v0.h[7]      \n"
                        "fmla   v11.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v12.4h, v9.4h, v1.h[7]      \n"
                        "fmla   v13.4h, v9.4h, v2.h[3]      \n"

                        "ext    v6.16b, %24.16b, %24.16b, #8 \n"
                        "fmla   v10.4h, %24.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %24.4h, v1.h[4]     \n"
                        "fmla   v12.4h, %24.4h, v2.h[0]     \n"
                        "fmla   v13.4h, %24.4h, v2.h[4]     \n"

                        "add    %1, %1, #32                 \n"

                        "fmla   v10.4h, v6.4h, v1.h[1]      \n"
                        "fmla   v11.4h, v6.4h, v1.h[5]      \n"
                        "fmla   v12.4h, v6.4h, v2.h[1]      \n"
                        "fmla   v13.4h, v6.4h, v2.h[5]      \n"
                        "ext    v7.16b, %25.16b, %25.16b, #8 \n"
                        "fmla   v10.4h, %25.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %25.4h, v1.h[6]     \n"
                        "fmla   v12.4h, %25.4h, v2.h[2]     \n"
                        "fmla   v13.4h, %25.4h, v2.h[6]     \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v10.4h, v7.4h, v1.h[3]      \n"
                        "fmla   v11.4h, v7.4h, v1.h[7]      \n"
                        "fmla   v12.4h, v7.4h, v2.h[3]      \n"
                        "fmla   v13.4h, v7.4h, v2.h[7]      \n"

                        "add    %3, %3, #32                 \n"

                        "st1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00_01), // %8
                        "w"(_k00_23), // %9
                        "w"(_k01_01), // %10
                        "w"(_k01_23), // %11
                        "w"(_k02_01), // %12
                        "w"(_k02_23), // %13
                        "w"(_k10_01), // %14
                        "w"(_k10_23), // %15
                        "w"(_k11_01), // %16
                        "w"(_k11_23), // %17
                        "w"(_k12_01), // %18
                        "w"(_k12_23), // %19
                        "w"(_k20_01), // %20
                        "w"(_k20_23), // %21
                        "w"(_k21_01), // %22
                        "w"(_k21_23), // %23
                        "w"(_k22_01), // %24
                        "w"(_k22_23)  // %25
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%1]        \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v12.4h, v13.4h}, [%0]      \n" // sum0 sum1

                        "ext    v4.16b, %8.16b, %8.16b, #8  \n"
                        "fmul   v10.4h, %8.4h, v0.h[0]      \n"
                        "fmul   v11.4h, %8.4h, v0.h[4]      \n"
                        "fmla   v12.4h, v4.4h, v0.h[1]      \n"
                        "fmla   v13.4h, v4.4h, v0.h[5]      \n"
                        "ext    v5.16b, %9.16b, %9.16b, #8  \n"
                        "fmla   v10.4h, %9.4h, v0.h[2]      \n"
                        "fmla   v11.4h, %9.4h, v0.h[6]      \n"
                        "fmla   v12.4h, v5.4h, v0.h[3]      \n"
                        "fmla   v13.4h, v5.4h, v0.h[7]      \n"

                        "ext    v6.16b, %10.16b, %10.16b, #8 \n"
                        "fmla   v10.4h, %10.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %10.4h, v1.h[0]     \n"
                        "fmla   v12.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[1]      \n"
                        "ext    v7.16b, %11.16b, %11.16b, #8 \n"
                        "fmla   v10.4h, %11.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %11.4h, v1.h[2]     \n"
                        "fmla   v12.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[3]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v2.8h, v3.8h}, [%2]        \n" // r10 r11 r12 r13

                        "ext    v8.16b, %12.16b, %12.16b, #8 \n"
                        "fmla   v10.4h, %12.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %12.4h, v1.h[4]     \n"
                        "fmla   v12.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v1.h[5]      \n"
                        "ext    v9.16b, %13.16b, %13.16b, #8 \n"
                        "fmla   v10.4h, %13.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %13.4h, v1.h[6]     \n"
                        "fmla   v12.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v1.h[7]      \n"

                        "ext    v4.16b, %14.16b, %14.16b, #8 \n"
                        "fmla   v10.4h, %14.4h, v2.h[0]     \n"
                        "fmla   v11.4h, %14.4h, v2.h[4]     \n"
                        "fmla   v12.4h, v4.4h, v2.h[1]      \n"
                        "fmla   v13.4h, v4.4h, v2.h[5]      \n"
                        "ext    v5.16b, %15.16b, %15.16b, #8 \n"
                        "fmla   v10.4h, %15.4h, v2.h[2]     \n"
                        "fmla   v11.4h, %15.4h, v2.h[6]     \n"
                        "fmla   v12.4h, v5.4h, v2.h[3]      \n"
                        "fmla   v13.4h, v5.4h, v2.h[7]      \n"

                        "ext    v6.16b, %16.16b, %16.16b, #8 \n"
                        "fmla   v10.4h, %16.4h, v2.h[4]     \n"
                        "fmla   v11.4h, %16.4h, v3.h[0]     \n"
                        "fmla   v12.4h, v6.4h, v2.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v3.h[1]      \n"
                        "ext    v7.16b, %17.16b, %17.16b, #8 \n"
                        "fmla   v10.4h, %17.4h, v2.h[6]     \n"
                        "fmla   v11.4h, %17.4h, v3.h[2]     \n"
                        "fmla   v12.4h, v7.4h, v2.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v3.h[3]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%3]        \n" // r20 r21 r22 r23

                        "ext    v8.16b, %18.16b, %18.16b, #8 \n"
                        "fmla   v10.4h, %18.4h, v3.h[0]     \n"
                        "fmla   v11.4h, %18.4h, v3.h[4]     \n"
                        "fmla   v12.4h, v8.4h, v3.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v3.h[5]      \n"
                        "ext    v9.16b, %19.16b, %19.16b, #8 \n"
                        "fmla   v10.4h, %19.4h, v3.h[2]     \n"
                        "fmla   v11.4h, %19.4h, v3.h[6]     \n"
                        "fmla   v12.4h, v9.4h, v3.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v3.h[7]      \n"

                        "ext    v4.16b, %20.16b, %20.16b, #8 \n"
                        "fmla   v10.4h, %20.4h, v0.h[0]     \n"
                        "fmla   v11.4h, %20.4h, v0.h[4]     \n"
                        "fmla   v12.4h, v4.4h, v0.h[1]      \n"
                        "fmla   v13.4h, v4.4h, v0.h[5]      \n"
                        "ext    v5.16b, %21.16b, %21.16b, #8 \n"
                        "fmla   v10.4h, %21.4h, v0.h[2]     \n"
                        "fmla   v11.4h, %21.4h, v0.h[6]     \n"
                        "fmla   v12.4h, v5.4h, v0.h[3]      \n"
                        "fmla   v13.4h, v5.4h, v0.h[7]      \n"

                        "ext    v6.16b, %22.16b, %22.16b, #8 \n"
                        "fmla   v10.4h, %22.4h, v0.h[4]     \n"
                        "fmla   v11.4h, %22.4h, v1.h[0]     \n"
                        "fmla   v12.4h, v6.4h, v0.h[5]      \n"
                        "fmla   v13.4h, v6.4h, v1.h[1]      \n"
                        "ext    v7.16b, %23.16b, %23.16b, #8 \n"
                        "fmla   v10.4h, %23.4h, v0.h[6]     \n"
                        "fmla   v11.4h, %23.4h, v1.h[2]     \n"
                        "fmla   v12.4h, v7.4h, v0.h[7]      \n"
                        "fmla   v13.4h, v7.4h, v1.h[3]      \n"

                        "ext    v8.16b, %24.16b, %24.16b, #8 \n"
                        "fmla   v10.4h, %24.4h, v1.h[0]     \n"
                        "fmla   v11.4h, %24.4h, v1.h[4]     \n"
                        "fmla   v12.4h, v8.4h, v1.h[1]      \n"
                        "fmla   v13.4h, v8.4h, v1.h[5]      \n"
                        "ext    v9.16b, %25.16b, %25.16b, #8 \n"
                        "fmla   v10.4h, %25.4h, v1.h[2]     \n"
                        "fmla   v11.4h, %25.4h, v1.h[6]     \n"
                        "fmla   v12.4h, v9.4h, v1.h[3]      \n"
                        "fmla   v13.4h, v9.4h, v1.h[7]      \n"

                        "add    %1, %1, #16                 \n"

                        "fadd   v10.4h, v10.4h, v12.4h      \n"

                        "add    %2, %2, #16                 \n"

                        "fadd   v11.4h, v11.4h, v13.4h      \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v10.4h, v11.4h}, [%0], #16 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00_01), // %8
                        "w"(_k00_23), // %9
                        "w"(_k01_01), // %10
                        "w"(_k01_23), // %11
                        "w"(_k02_01), // %12
                        "w"(_k02_23), // %13
                        "w"(_k10_01), // %14
                        "w"(_k10_23), // %15
                        "w"(_k11_01), // %16
                        "w"(_k11_23), // %17
                        "w"(_k12_01), // %18
                        "w"(_k12_23), // %19
                        "w"(_k20_01), // %20
                        "w"(_k20_23), // %21
                        "w"(_k21_01), // %22
                        "w"(_k21_23), // %23
                        "w"(_k22_01), // %24
                        "w"(_k22_23)  // %25
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%1] \n" // r00 r01 r02

                        "prfm   pldl1keep, [%0, #64]        \n"
                        "ld1    {v13.4h}, [%0]              \n" // sum0

                        "ext    v6.16b, %8.16b, %8.16b, #8  \n"
                        "fmul   v10.4h, %8.4h, v0.h[0]      \n"
                        "fmul   v11.4h, v6.4h, v0.h[1]      \n"
                        "ext    v7.16b, %9.16b, %9.16b, #8  \n"
                        "fmul   v12.4h, %9.4h, v0.h[2]      \n"
                        "fmla   v13.4h, v7.4h, v0.h[3]      \n"

                        "ext    v8.16b, %10.16b, %10.16b, #8 \n"
                        "fmla   v10.4h, %10.4h, v1.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "ext    v9.16b, %11.16b, %11.16b, #8 \n"
                        "fmla   v12.4h, %11.4h, v1.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v1.h[3]      \n"

                        "prfm   pldl1keep, [%2, #192]       \n"
                        "ld1    {v3.4h, v4.4h, v5.4h}, [%2] \n" // r10 r11 r12

                        "ext    v6.16b, %12.16b, %12.16b, #8 \n"
                        "fmla   v10.4h, %12.4h, v2.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v2.h[1]      \n"
                        "ext    v7.16b, %13.16b, %13.16b, #8 \n"
                        "fmla   v12.4h, %13.4h, v2.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v2.h[3]      \n"

                        "ext    v8.16b, %14.16b, %14.16b, #8 \n"
                        "fmla   v10.4h, %14.4h, v3.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v3.h[1]      \n"
                        "ext    v9.16b, %15.16b, %15.16b, #8 \n"
                        "fmla   v12.4h, %15.4h, v3.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v3.h[3]      \n"

                        "ext    v6.16b, %16.16b, %16.16b, #8 \n"
                        "fmla   v10.4h, %16.4h, v4.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v4.h[1]      \n"
                        "ext    v7.16b, %17.16b, %17.16b, #8 \n"
                        "fmla   v12.4h, %17.4h, v4.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v4.h[3]      \n"

                        "prfm   pldl1keep, [%3, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%3] \n" // r20 r21 r22

                        "ext    v8.16b, %18.16b, %18.16b, #8 \n"
                        "fmla   v10.4h, %18.4h, v5.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v5.h[1]      \n"
                        "ext    v9.16b, %19.16b, %19.16b, #8 \n"
                        "fmla   v12.4h, %19.4h, v5.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v5.h[3]      \n"

                        "ext    v6.16b, %20.16b, %20.16b, #8 \n"
                        "fmla   v10.4h, %20.4h, v0.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v0.h[1]      \n"
                        "ext    v7.16b, %21.16b, %21.16b, #8 \n"
                        "fmla   v12.4h, %21.4h, v0.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v0.h[3]      \n"

                        "ext    v8.16b, %22.16b, %22.16b, #8 \n"
                        "fmla   v10.4h, %22.4h, v1.h[0]     \n"
                        "fmla   v11.4h, v8.4h, v1.h[1]      \n"
                        "ext    v9.16b, %23.16b, %23.16b, #8 \n"
                        "fmla   v12.4h, %23.4h, v1.h[2]     \n"
                        "fmla   v13.4h, v9.4h, v1.h[3]      \n"

                        "ext    v6.16b, %24.16b, %24.16b, #8 \n"
                        "fmla   v10.4h, %24.4h, v2.h[0]     \n"
                        "fmla   v11.4h, v6.4h, v2.h[1]      \n"
                        "ext    v7.16b, %25.16b, %25.16b, #8 \n"
                        "fmla   v12.4h, %25.4h, v2.h[2]     \n"
                        "fmla   v13.4h, v7.4h, v2.h[3]      \n"

                        "fadd   v10.4h, v10.4h, v11.4h      \n"

                        "add    %1, %1, #8                  \n"

                        "fadd   v12.4h, v12.4h, v13.4h      \n"

                        "add    %2, %2, #8                  \n"

                        "fadd   v10.4h, v10.4h, v12.4h      \n"

                        "add    %3, %3, #8                  \n"

                        "st1    {v10.4h}, [%0], #8          \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00_01), // %8
                        "w"(_k00_23), // %9
                        "w"(_k01_01), // %10
                        "w"(_k01_23), // %11
                        "w"(_k02_01), // %12
                        "w"(_k02_23), // %13
                        "w"(_k10_01), // %14
                        "w"(_k10_23), // %15
                        "w"(_k11_01), // %16
                        "w"(_k11_23), // %17
                        "w"(_k12_01), // %18
                        "w"(_k12_23), // %19
                        "w"(_k20_01), // %20
                        "w"(_k20_23), // %21
                        "w"(_k21_01), // %22
                        "w"(_k21_23), // %23
                        "w"(_k22_01), // %24
                        "w"(_k22_23)  // %25
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }

                r0 += 8;
                r1 += 8;
                r2 += 8;
            }
        }
    }
}
