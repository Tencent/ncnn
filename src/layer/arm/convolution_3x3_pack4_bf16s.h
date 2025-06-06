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

static void conv3x3s2_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    Mat top_blob_fp32(outw, outh, opt.num_threads, (size_t)4u * 4, 4, opt.workspace_allocator);

    const int tailstep = (w - 2 * outw + w) * 4;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob_fp32.channel(get_omp_thread_num());

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        int q = 0;
        for (; q < inch - 1; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const unsigned short* r0 = img0.row<const unsigned short>(0);
            const unsigned short* r1 = img0.row<const unsigned short>(1);
            const unsigned short* r2 = img0.row<const unsigned short>(2);

            const unsigned short* kptr = (const unsigned short*)kernel.channel(p).row<const unsigned short>(q);

#if __aarch64__
            // 16 * 9
            uint16x8_t _k00_01 = vld1q_u16(kptr);
            uint16x8_t _k00_23 = vld1q_u16(kptr + 8);
            uint16x8_t _k01_01 = vld1q_u16(kptr + 16);
            uint16x8_t _k01_23 = vld1q_u16(kptr + 24);
            uint16x8_t _k02_01 = vld1q_u16(kptr + 32);
            uint16x8_t _k02_23 = vld1q_u16(kptr + 40);
            uint16x8_t _k10_01 = vld1q_u16(kptr + 48);
            uint16x8_t _k10_23 = vld1q_u16(kptr + 56);
            uint16x8_t _k11_01 = vld1q_u16(kptr + 64);
            uint16x8_t _k11_23 = vld1q_u16(kptr + 72);
            uint16x8_t _k12_01 = vld1q_u16(kptr + 80);
            uint16x8_t _k12_23 = vld1q_u16(kptr + 88);
            uint16x8_t _k20_01 = vld1q_u16(kptr + 96);
            uint16x8_t _k20_23 = vld1q_u16(kptr + 104);
            uint16x8_t _k21_01 = vld1q_u16(kptr + 112);
            uint16x8_t _k21_23 = vld1q_u16(kptr + 120);
            uint16x8_t _k22_01 = vld1q_u16(kptr + 128);
            uint16x8_t _k22_23 = vld1q_u16(kptr + 136);
#endif // __aarch64__

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n" // r00 r01 r02 r03

                        "shll   v0.4s, v4.4h, #16           \n"
                        "shll2  v1.4s, v4.8h, #16           \n"
                        "shll   v2.4s, v5.4h, #16           \n"
                        "shll2  v3.4s, v5.8h, #16           \n"

                        "shll   v4.4s, v6.4h, #16           \n"
                        "shll2  v5.4s, v6.8h, #16           \n"
                        "shll   v6.4s, v7.4h, #16           \n"
                        "shll2  v7.4s, v7.8h, #16           \n"

                        "shll   v8.4s, %8.4h, #16           \n"
                        "shll2  v9.4s, %8.8h, #16           \n"

                        "fmla   v10.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[1]      \n"

                        "shll   v8.4s, %9.4h, #16           \n"
                        "shll2  v9.4s, %9.8h, #16           \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[3]      \n"

                        "shll   v8.4s, %10.4h, #16          \n"
                        "shll2  v9.4s, %10.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[1]      \n"

                        "shll   v8.4s, %11.4h, #16          \n"
                        "shll2  v9.4s, %11.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[3]      \n"

                        "prfm   pldl1keep, [%1, #64]        \n"
                        "ld1    {v0.4h}, [%1]               \n" // r08

                        "shll   v0.4s, v0.4h, #16           \n"

                        "shll   v8.4s, %12.4h, #16          \n"
                        "shll2  v9.4s, %12.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[1]      \n"

                        "shll   v8.4s, %13.4h, #16          \n"
                        "shll2  v9.4s, %13.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r10 r11 r12 r13

                        "shll   v0.4s, v4.4h, #16           \n"
                        "shll2  v1.4s, v4.8h, #16           \n"
                        "shll   v2.4s, v5.4h, #16           \n"
                        "shll2  v3.4s, v5.8h, #16           \n"

                        "shll   v4.4s, v6.4h, #16           \n"
                        "shll2  v5.4s, v6.8h, #16           \n"
                        "shll   v6.4s, v7.4h, #16           \n"
                        "shll2  v7.4s, v7.8h, #16           \n"

                        "shll   v8.4s, %14.4h, #16          \n"
                        "shll2  v9.4s, %14.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[1]      \n"

                        "shll   v8.4s, %15.4h, #16          \n"
                        "shll2  v9.4s, %15.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[3]      \n"

                        "shll   v8.4s, %16.4h, #16          \n"
                        "shll2  v9.4s, %16.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[1]      \n"

                        "shll   v8.4s, %17.4h, #16          \n"
                        "shll2  v9.4s, %17.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[3]      \n"

                        "prfm   pldl1keep, [%2, #64]        \n"
                        "ld1    {v0.4h}, [%2]               \n" // r18

                        "shll   v0.4s, v0.4h, #16           \n"

                        "shll   v8.4s, %18.4h, #16          \n"
                        "shll2  v9.4s, %18.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[1]      \n"

                        "shll   v8.4s, %19.4h, #16          \n"
                        "shll2  v9.4s, %19.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3], #64 \n" // r20 r21 r22 r23

                        "shll   v0.4s, v4.4h, #16           \n"
                        "shll2  v1.4s, v4.8h, #16           \n"
                        "shll   v2.4s, v5.4h, #16           \n"
                        "shll2  v3.4s, v5.8h, #16           \n"

                        "shll   v4.4s, v6.4h, #16           \n"
                        "shll2  v5.4s, v6.8h, #16           \n"
                        "shll   v6.4s, v7.4h, #16           \n"
                        "shll2  v7.4s, v7.8h, #16           \n"

                        "shll   v8.4s, %20.4h, #16          \n"
                        "shll2  v9.4s, %20.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[1]      \n"

                        "shll   v8.4s, %21.4h, #16          \n"
                        "shll2  v9.4s, %21.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[3]      \n"

                        "shll   v8.4s, %22.4h, #16          \n"
                        "shll2  v9.4s, %22.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[1]      \n"

                        "shll   v8.4s, %23.4h, #16          \n"
                        "shll2  v9.4s, %23.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[3]      \n"

                        "prfm   pldl1keep, [%3, #64]        \n"
                        "ld1    {v0.4h}, [%3]               \n" // r28

                        "shll   v0.4s, v0.4h, #16           \n"

                        "shll   v8.4s, %24.4h, #16          \n"
                        "shll2  v9.4s, %24.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[1]      \n"

                        "shll   v8.4s, %25.4h, #16          \n"
                        "shll2  v9.4s, %25.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "st1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%0], #64 \n"

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
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d24-d31}       \n" // sum0 sum1 sum2 sum3

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d8-d15}       \n" // r00 r01 r02 r03 r04 r05 r06 r07

                        "vshll.u16  q0, d8, #16         \n"
                        "vshll.u16  q1, d9, #16         \n"
                        "vshll.u16  q2, d10, #16        \n"
                        "vshll.u16  q3, d11, #16        \n"

                        "vshll.u16  q4, d12, #16        \n"
                        "vshll.u16  q5, d13, #16        \n"
                        "vshll.u16  q6, d14, #16        \n"
                        "vshll.u16  q7, d15, #16        \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%1, #64]           \n"
                        "vld1.f32   {d1}, [%1 :64]      \n" // r08

                        "vshll.u16  q0, d1, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d8-d15}       \n" // r10 r11 r12 r13 r14 r15 r16 r17

                        "vshll.u16  q0, d8, #16         \n"
                        "vshll.u16  q1, d9, #16         \n"
                        "vshll.u16  q2, d10, #16        \n"
                        "vshll.u16  q3, d11, #16        \n"

                        "vshll.u16  q4, d12, #16        \n"
                        "vshll.u16  q5, d13, #16        \n"
                        "vshll.u16  q6, d14, #16        \n"
                        "vshll.u16  q7, d15, #16        \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%2, #64]           \n"
                        "vld1.f32   {d1}, [%2 :64]      \n" // r18

                        "vshll.u16  q0, d1, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%3, #256]          \n"
                        "vldm       %3!, {d8-d15}       \n" // r20 r21 r22 r23 r24 r25 r26 r27

                        "vshll.u16  q0, d8, #16         \n"
                        "vshll.u16  q1, d9, #16         \n"
                        "vshll.u16  q2, d10, #16        \n"
                        "vshll.u16  q3, d11, #16        \n"

                        "vshll.u16  q4, d12, #16        \n"
                        "vshll.u16  q5, d13, #16        \n"
                        "vshll.u16  q6, d14, #16        \n"
                        "vshll.u16  q7, d15, #16        \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%3, #64]           \n"
                        "vld1.f32   {d1}, [%3 :64]      \n" // r28

                        "vshll.u16  q0, d1, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        //                         "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128] \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "sub        %4, %4, #256        \n" // kptr -= 8 * 16;

                        "vstm       %0!, {d24-d31}      \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v12.4s, v13.4s}, [%0]      \n" // sum0 sum1

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n" // r00 r01 r02 r03

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "shll   v6.4s, %8.4h, #16           \n"
                        "shll2  v7.4s, %8.8h, #16           \n"
                        "shll   v8.4s, %9.4h, #16           \n"
                        "shll2  v9.4s, %9.8h, #16           \n"

                        "fmul   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmul   v11.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v2.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%1, #64]        \n"
                        "ld1    {v4.4h}, [%1]               \n" // r04

                        "shll   v4.4s, v4.4h, #16           \n"

                        "shll   v6.4s, %10.4h, #16          \n"
                        "shll2  v7.4s, %10.8h, #16          \n"
                        "shll   v8.4s, %11.4h, #16          \n"
                        "shll2  v9.4s, %11.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %12.4h, #16          \n"
                        "shll2  v7.4s, %12.8h, #16          \n"
                        "shll   v8.4s, %13.4h, #16          \n"
                        "shll2  v9.4s, %13.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v4.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r10 r11 r12 r13

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "shll   v6.4s, %14.4h, #16          \n"
                        "shll2  v7.4s, %14.8h, #16          \n"
                        "shll   v8.4s, %15.4h, #16          \n"
                        "shll2  v9.4s, %15.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v2.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%2, #64]        \n"
                        "ld1    {v4.4h}, [%2]               \n" // r14

                        "shll   v4.4s, v4.4h, #16           \n"

                        "shll   v6.4s, %16.4h, #16          \n"
                        "shll2  v7.4s, %16.8h, #16          \n"
                        "shll   v8.4s, %17.4h, #16          \n"
                        "shll2  v9.4s, %17.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %18.4h, #16          \n"
                        "shll2  v7.4s, %18.8h, #16          \n"
                        "shll   v8.4s, %19.4h, #16          \n"
                        "shll2  v9.4s, %19.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v4.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r20 r21 r22 r23

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "shll   v6.4s, %20.4h, #16          \n"
                        "shll2  v7.4s, %20.8h, #16          \n"
                        "shll   v8.4s, %21.4h, #16          \n"
                        "shll2  v9.4s, %21.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v2.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%3, #64]        \n"
                        "ld1    {v4.4h}, [%3]               \n" // r24

                        "shll   v4.4s, v4.4h, #16           \n"

                        "shll   v6.4s, %22.4h, #16          \n"
                        "shll2  v7.4s, %22.8h, #16          \n"
                        "shll   v8.4s, %23.4h, #16          \n"
                        "shll2  v9.4s, %23.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %24.4h, #16          \n"
                        "shll2  v7.4s, %24.8h, #16          \n"
                        "shll   v8.4s, %25.4h, #16          \n"
                        "shll2  v9.4s, %25.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v4.s[1]      \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "fadd   v12.4s, v10.4s, v12.4s      \n"
                        "fadd   v13.4s, v11.4s, v13.4s      \n"

                        "st1    {v12.4s, v13.4s}, [%0], #32 \n"

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
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128] \n" // sum0 sum1

                        "pld        [%1, #256]          \n"
                        "vld1.u16   {d4-d7}, [%1 :64]!  \n" // r00 r01 r02 r03

                        "vshll.u16  q0, d4, #16         \n"
                        "vshll.u16  q1, d5, #16         \n"
                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmul.f32   q12, q8, d0[0]      \n"
                        "vmul.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%1, #64]           \n"
                        "vld1.f32   {d9}, [%1 :64]      \n" // r04

                        "vshll.u16  q4, d9, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%2, #256]          \n"
                        "vld1.u16   {d4-d7}, [%2 :64]!  \n" // r10 r11 r12 r13

                        "vshll.u16  q0, d4, #16         \n"
                        "vshll.u16  q1, d5, #16         \n"
                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%2, #64]           \n"
                        "vld1.f32   {d9}, [%2 :64]      \n" // r14

                        "vshll.u16  q4, d9, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%3, #256]          \n"
                        "vld1.u16   {d4-d7}, [%3 :64]!  \n" // r20 r21 r22 r23

                        "vshll.u16  q0, d4, #16         \n"
                        "vshll.u16  q1, d5, #16         \n"
                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%3, #64]           \n"
                        "vld1.f32   {d9}, [%3 :64]      \n" // r24

                        "vshll.u16  q4, d9, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        //                         "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128] \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "vadd.f32   q14, q12, q14       \n"
                        "vadd.f32   q15, q13, q15       \n"

                        "sub        %4, %4, #256        \n" // kptr -= 8 * 16;

                        "vst1.f32   {d28-d31}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v13.4s}, [%0]              \n" // sum0

                        "prfm   pldl1keep, [%1, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%1] \n" // r00 r01 r02

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "shll   v6.4s, %8.4h, #16           \n"
                        "shll2  v7.4s, %8.8h, #16           \n"

                        "fmul   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmul   v11.4s, v7.4s, v0.s[1]      \n"

                        "shll   v8.4s, %9.4h, #16           \n"
                        "shll2  v9.4s, %9.8h, #16           \n"

                        "fmul   v12.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "shll   v6.4s, %10.4h, #16          \n"
                        "shll2  v7.4s, %10.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"

                        "shll   v8.4s, %11.4h, #16          \n"
                        "shll2  v9.4s, %11.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v1.s[3]      \n"

                        "shll   v6.4s, %12.4h, #16          \n"
                        "shll2  v7.4s, %12.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %13.4h, #16          \n"
                        "shll2  v9.4s, %13.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%2, #192]       \n"
                        "ld1    {v3.4h, v4.4h, v5.4h}, [%2] \n" // r10 r11 r12

                        "shll   v3.4s, v3.4h, #16           \n"
                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "shll   v6.4s, %14.4h, #16          \n"
                        "shll2  v7.4s, %14.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v3.s[1]      \n"

                        "shll   v8.4s, %15.4h, #16          \n"
                        "shll2  v9.4s, %15.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %16.4h, #16          \n"
                        "shll2  v7.4s, %16.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v4.s[1]      \n"

                        "shll   v8.4s, %17.4h, #16          \n"
                        "shll2  v9.4s, %17.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "shll   v6.4s, %18.4h, #16          \n"
                        "shll2  v7.4s, %18.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v5.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v5.s[1]      \n"

                        "shll   v8.4s, %19.4h, #16          \n"
                        "shll2  v9.4s, %19.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v5.s[3]      \n"

                        "prfm   pldl1keep, [%3, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%3] \n" // r20 r21 r22

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "shll   v6.4s, %20.4h, #16          \n"
                        "shll2  v7.4s, %20.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v0.s[1]      \n"

                        "shll   v8.4s, %21.4h, #16          \n"
                        "shll2  v9.4s, %21.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "shll   v6.4s, %22.4h, #16          \n"
                        "shll2  v7.4s, %22.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"

                        "shll   v8.4s, %23.4h, #16          \n"
                        "shll2  v9.4s, %23.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v1.s[3]      \n"

                        "shll   v6.4s, %24.4h, #16          \n"
                        "shll2  v7.4s, %24.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %25.4h, #16          \n"
                        "shll2  v9.4s, %25.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "fadd   v11.4s, v10.4s, v11.4s      \n"

                        "add    %1, %1, #16                 \n"
                        "fadd   v13.4s, v12.4s, v13.4s      \n"

                        "add    %2, %2, #16                 \n"
                        "fadd   v13.4s, v11.4s, v13.4s      \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v13.4s}, [%0], #16         \n"

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
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d30-d31}, [%0 :128] \n" // sum0

                        "pld        [%1, #192]          \n"
                        "vld1.u16   {d2-d4}, [%1 :64]   \n" // r00 r01 r02

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshll.u16  q2, d4, #16         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmul.f32   q12, q8, d0[0]      \n"
                        "vmul.f32   q13, q9, d0[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmul.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%2, #192]          \n"
                        "vld1.u16   {d2-d4}, [%2 :64]   \n" // r10 r11 r12

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshll.u16  q2, d4, #16         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q9, d0[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%3, #192]          \n"
                        "vld1.u16   {d2-d4}, [%3 :64]   \n" // r20 r21 r22

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshll.u16  q2, d4, #16         \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q9, d0[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        //                         "pld        [%4, #256]          \n"
                        "vld1.u16   {d20-d23}, [%4 :128] \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "add        %1, %1, #16         \n"
                        "vadd.f32   q13, q12, q13       \n"

                        "add        %2, %2, #16         \n"
                        "vadd.f32   q15, q14, q15       \n"

                        "add        %3, %3, #16         \n"
                        "vadd.f32   q15, q13, q15       \n"

                        "sub        %4, %4, #256        \n" // kptr -= 8 * 16 * 2;

                        "vst1.f32   {d30-d31}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
        for (; q < inch; q++)
        {
            unsigned short* outptr0_bf16 = top_blob.channel(p);

            const float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const unsigned short* r0 = img0.row<const unsigned short>(0);
            const unsigned short* r1 = img0.row<const unsigned short>(1);
            const unsigned short* r2 = img0.row<const unsigned short>(2);

            const unsigned short* kptr = (const unsigned short*)kernel.channel(p).row<const unsigned short>(q);

#if __aarch64__
            // 16 * 9
            uint16x8_t _k00_01 = vld1q_u16(kptr);
            uint16x8_t _k00_23 = vld1q_u16(kptr + 8);
            uint16x8_t _k01_01 = vld1q_u16(kptr + 16);
            uint16x8_t _k01_23 = vld1q_u16(kptr + 24);
            uint16x8_t _k02_01 = vld1q_u16(kptr + 32);
            uint16x8_t _k02_23 = vld1q_u16(kptr + 40);
            uint16x8_t _k10_01 = vld1q_u16(kptr + 48);
            uint16x8_t _k10_23 = vld1q_u16(kptr + 56);
            uint16x8_t _k11_01 = vld1q_u16(kptr + 64);
            uint16x8_t _k11_23 = vld1q_u16(kptr + 72);
            uint16x8_t _k12_01 = vld1q_u16(kptr + 80);
            uint16x8_t _k12_23 = vld1q_u16(kptr + 88);
            uint16x8_t _k20_01 = vld1q_u16(kptr + 96);
            uint16x8_t _k20_23 = vld1q_u16(kptr + 104);
            uint16x8_t _k21_01 = vld1q_u16(kptr + 112);
            uint16x8_t _k21_23 = vld1q_u16(kptr + 120);
            uint16x8_t _k22_01 = vld1q_u16(kptr + 128);
            uint16x8_t _k22_23 = vld1q_u16(kptr + 136);
#endif // __aarch64__

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%1], #64 \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r00 r01 r02 r03

                        "shll   v0.4s, v4.4h, #16           \n"
                        "shll2  v1.4s, v4.8h, #16           \n"
                        "shll   v2.4s, v5.4h, #16           \n"
                        "shll2  v3.4s, v5.8h, #16           \n"

                        "shll   v4.4s, v6.4h, #16           \n"
                        "shll2  v5.4s, v6.8h, #16           \n"
                        "shll   v6.4s, v7.4h, #16           \n"
                        "shll2  v7.4s, v7.8h, #16           \n"

                        "shll   v8.4s, %10.4h, #16          \n"
                        "shll2  v9.4s, %10.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[1]      \n"

                        "shll   v8.4s, %11.4h, #16          \n"
                        "shll2  v9.4s, %11.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[3]      \n"

                        "shll   v8.4s, %12.4h, #16          \n"
                        "shll2  v9.4s, %12.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[1]      \n"

                        "shll   v8.4s, %13.4h, #16          \n"
                        "shll2  v9.4s, %13.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[3]      \n"

                        "prfm   pldl1keep, [%2, #64]        \n"
                        "ld1    {v0.4h}, [%2]               \n" // r08

                        "shll   v0.4s, v0.4h, #16           \n"

                        "shll   v8.4s, %14.4h, #16          \n"
                        "shll2  v9.4s, %14.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[1]      \n"

                        "shll   v8.4s, %15.4h, #16          \n"
                        "shll2  v9.4s, %15.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3], #64 \n" // r10 r11 r12 r13

                        "shll   v0.4s, v4.4h, #16           \n"
                        "shll2  v1.4s, v4.8h, #16           \n"
                        "shll   v2.4s, v5.4h, #16           \n"
                        "shll2  v3.4s, v5.8h, #16           \n"

                        "shll   v4.4s, v6.4h, #16           \n"
                        "shll2  v5.4s, v6.8h, #16           \n"
                        "shll   v6.4s, v7.4h, #16           \n"
                        "shll2  v7.4s, v7.8h, #16           \n"

                        "shll   v8.4s, %16.4h, #16          \n"
                        "shll2  v9.4s, %16.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[1]      \n"

                        "shll   v8.4s, %17.4h, #16          \n"
                        "shll2  v9.4s, %17.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[3]      \n"

                        "shll   v8.4s, %18.4h, #16          \n"
                        "shll2  v9.4s, %18.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[1]      \n"

                        "shll   v8.4s, %19.4h, #16          \n"
                        "shll2  v9.4s, %19.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[3]      \n"

                        "prfm   pldl1keep, [%3, #64]        \n"
                        "ld1    {v0.4h}, [%3]               \n" // r18

                        "shll   v0.4s, v0.4h, #16           \n"

                        "shll   v8.4s, %20.4h, #16          \n"
                        "shll2  v9.4s, %20.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[1]      \n"

                        "shll   v8.4s, %21.4h, #16          \n"
                        "shll2  v9.4s, %21.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%4], #64 \n" // r20 r21 r22 r23

                        "shll   v0.4s, v4.4h, #16           \n"
                        "shll2  v1.4s, v4.8h, #16           \n"
                        "shll   v2.4s, v5.4h, #16           \n"
                        "shll2  v3.4s, v5.8h, #16           \n"

                        "shll   v4.4s, v6.4h, #16           \n"
                        "shll2  v5.4s, v6.8h, #16           \n"
                        "shll   v6.4s, v7.4h, #16           \n"
                        "shll2  v7.4s, v7.8h, #16           \n"

                        "shll   v8.4s, %22.4h, #16          \n"
                        "shll2  v9.4s, %22.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[1]      \n"

                        "shll   v8.4s, %23.4h, #16          \n"
                        "shll2  v9.4s, %23.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v6.s[3]      \n"

                        "shll   v8.4s, %24.4h, #16          \n"
                        "shll2  v9.4s, %24.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[1]      \n"

                        "shll   v8.4s, %25.4h, #16          \n"
                        "shll2  v9.4s, %25.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v7.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v3.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v5.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v7.s[3]      \n"

                        "prfm   pldl1keep, [%4, #64]        \n"
                        "ld1    {v0.4h}, [%4]               \n" // r28

                        "shll   v0.4s, v0.4h, #16           \n"

                        "shll   v8.4s, %26.4h, #16          \n"
                        "shll2  v9.4s, %26.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[0]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[0]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[1]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[1]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[1]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[1]      \n"

                        "shll   v8.4s, %27.4h, #16          \n"
                        "shll2  v9.4s, %27.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v8.4s, v6.s[2]      \n"
                        "fmla   v13.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v10.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v11.4s, v9.4s, v4.s[3]      \n"
                        "fmla   v12.4s, v9.4s, v6.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "shrn   v10.4h, v10.4s, #16         \n"
                        "shrn   v11.4h, v11.4s, #16         \n"
                        "shrn   v12.4h, v12.4s, #16         \n"
                        "shrn   v13.4h, v13.4s, #16         \n"

                        "st1    {v10.4h, v11.4h, v12.4h, v13.4h}, [%0], #32 \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2)            // %4
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_01), // %10
                        "w"(_k00_23), // %11
                        "w"(_k01_01), // %12
                        "w"(_k01_23), // %13
                        "w"(_k02_01), // %14
                        "w"(_k02_23), // %15
                        "w"(_k10_01), // %16
                        "w"(_k10_23), // %17
                        "w"(_k11_01), // %18
                        "w"(_k11_23), // %19
                        "w"(_k12_01), // %20
                        "w"(_k12_23), // %21
                        "w"(_k20_01), // %22
                        "w"(_k20_23), // %23
                        "w"(_k21_01), // %24
                        "w"(_k21_23), // %25
                        "w"(_k22_01), // %26
                        "w"(_k22_23)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d24-d31}      \n" // sum0 sum1 sum2 sum3

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d8-d15}       \n" // r00 r01 r02 r03 r04 r05 r06 r07

                        "vshll.u16  q0, d8, #16         \n"
                        "vshll.u16  q1, d9, #16         \n"
                        "vshll.u16  q2, d10, #16        \n"
                        "vshll.u16  q3, d11, #16        \n"

                        "vshll.u16  q4, d12, #16        \n"
                        "vshll.u16  q5, d13, #16        \n"
                        "vshll.u16  q6, d14, #16        \n"
                        "vshll.u16  q7, d15, #16        \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%2, #64]           \n"
                        "vld1.f32   {d1}, [%2 :64]      \n" // r08

                        "vshll.u16  q0, d1, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d8-d15}       \n" // r10 r11 r12 r13 r14 r15 r16 r17

                        "vshll.u16  q0, d8, #16         \n"
                        "vshll.u16  q1, d9, #16         \n"
                        "vshll.u16  q2, d10, #16        \n"
                        "vshll.u16  q3, d11, #16        \n"

                        "vshll.u16  q4, d12, #16        \n"
                        "vshll.u16  q5, d13, #16        \n"
                        "vshll.u16  q6, d14, #16        \n"
                        "vshll.u16  q7, d15, #16        \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%3, #64]           \n"
                        "vld1.f32   {d1}, [%3 :64]      \n" // r18

                        "vshll.u16  q0, d1, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vldm       %4!, {d8-d15}       \n" // r20 r21 r22 r23 r24 r25 r26 r27

                        "vshll.u16  q0, d8, #16         \n"
                        "vshll.u16  q1, d9, #16         \n"
                        "vshll.u16  q2, d10, #16        \n"
                        "vshll.u16  q3, d11, #16        \n"

                        "vshll.u16  q4, d12, #16        \n"
                        "vshll.u16  q5, d13, #16        \n"
                        "vshll.u16  q6, d14, #16        \n"
                        "vshll.u16  q7, d15, #16        \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%4, #64]           \n"
                        "vld1.f32   {d1}, [%4 :64]      \n" // r28

                        "vshll.u16  q0, d1, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        //                         "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128] \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "sub        %5, %5, #256        \n" // kptr -= 8 * 16;

                        "vshrn.u32  d24, q12, #16       \n"
                        "vshrn.u32  d25, q13, #16       \n"
                        "vshrn.u32  d26, q14, #16       \n"
                        "vshrn.u32  d27, q15, #16       \n"

                        "vst1.f32   {d24-d27}, [%0 :64]! \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(kptr)          // %5
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v12.4s, v13.4s}, [%1], #32 \n" // sum0 sum1

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r00 r01 r02 r03

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "shll   v6.4s, %10.4h, #16          \n"
                        "shll2  v7.4s, %10.8h, #16          \n"

                        "fmul   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmul   v11.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %11.4h, #16          \n"
                        "shll2  v9.4s, %11.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%2, #64]        \n"
                        "ld1    {v4.4h}, [%2]               \n" // r04

                        "shll   v4.4s, v4.4h, #16           \n"

                        "shll   v6.4s, %12.4h, #16          \n"
                        "shll2  v7.4s, %12.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[1]      \n"

                        "shll   v8.4s, %13.4h, #16          \n"
                        "shll2  v9.4s, %13.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %14.4h, #16          \n"
                        "shll2  v7.4s, %14.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v4.s[1]      \n"

                        "shll   v8.4s, %15.4h, #16          \n"
                        "shll2  v9.4s, %15.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r10 r11 r12 r13

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "shll   v6.4s, %16.4h, #16          \n"
                        "shll2  v7.4s, %16.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %17.4h, #16          \n"
                        "shll2  v9.4s, %17.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%3, #64]        \n"
                        "ld1    {v4.4h}, [%3]               \n" // r14

                        "shll   v4.4s, v4.4h, #16           \n"

                        "shll   v6.4s, %18.4h, #16          \n"
                        "shll2  v7.4s, %18.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[1]      \n"

                        "shll   v8.4s, %19.4h, #16          \n"
                        "shll2  v9.4s, %19.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %20.4h, #16          \n"
                        "shll2  v7.4s, %20.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v4.s[1]      \n"

                        "shll   v8.4s, %21.4h, #16          \n"
                        "shll2  v9.4s, %21.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%4], #32 \n" // r20 r21 r22 r23

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "shll   v6.4s, %22.4h, #16          \n"
                        "shll2  v7.4s, %22.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v0.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %23.4h, #16          \n"
                        "shll2  v9.4s, %23.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v0.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%4, #64]        \n"
                        "ld1    {v4.4h}, [%4]               \n" // r24

                        "shll   v4.4s, v4.4h, #16           \n"

                        "shll   v6.4s, %24.4h, #16          \n"
                        "shll2  v7.4s, %24.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v1.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v3.s[1]      \n"

                        "shll   v8.4s, %25.4h, #16          \n"
                        "shll2  v9.4s, %25.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v1.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %26.4h, #16          \n"
                        "shll2  v7.4s, %26.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v12.4s, v7.4s, v2.s[1]      \n"
                        "fmla   v13.4s, v7.4s, v4.s[1]      \n"

                        "shll   v8.4s, %27.4h, #16          \n"
                        "shll2  v9.4s, %27.8h, #16          \n"

                        "fmla   v10.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v12.4s, v9.4s, v2.s[3]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "fadd   v12.4s, v10.4s, v12.4s      \n"
                        "fadd   v13.4s, v11.4s, v13.4s      \n"

                        "shrn   v12.4h, v12.4s, #16         \n"
                        "shrn   v13.4h, v13.4s, #16         \n"

                        "st1    {v12.4h, v13.4h}, [%0], #16 \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2)            // %4
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_01), // %10
                        "w"(_k00_23), // %11
                        "w"(_k01_01), // %12
                        "w"(_k01_23), // %13
                        "w"(_k02_01), // %14
                        "w"(_k02_23), // %15
                        "w"(_k10_01), // %16
                        "w"(_k10_23), // %17
                        "w"(_k11_01), // %18
                        "w"(_k11_23), // %19
                        "w"(_k12_01), // %20
                        "w"(_k12_23), // %21
                        "w"(_k20_01), // %22
                        "w"(_k20_23), // %23
                        "w"(_k21_01), // %24
                        "w"(_k21_23), // %25
                        "w"(_k22_01), // %26
                        "w"(_k22_23)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // sum0 sum1

                        "pld        [%2, #256]          \n"
                        "vld1.u16   {d4-d7}, [%2 :64]!  \n" // r00 r01 r02 r03

                        "vshll.u16  q0, d4, #16         \n"
                        "vshll.u16  q1, d5, #16         \n"
                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmul.f32   q12, q8, d0[0]      \n"
                        "vmul.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%2, #64]           \n"
                        "vld1.f32   {d9}, [%2 :64]      \n" // r04

                        "vshll.u16  q4, d9, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%3, #256]          \n"
                        "vld1.u16   {d4-d7}, [%3 :64]!  \n" // r10 r11 r12 r13

                        "vshll.u16  q0, d4, #16         \n"
                        "vshll.u16  q1, d5, #16         \n"
                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%3, #64]           \n"
                        "vld1.f32   {d9}, [%3 :64]      \n" // r14

                        "vshll.u16  q4, d9, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld1.u16   {d4-d7}, [%4 :64]!  \n" // r20 r21 r22 r23

                        "vshll.u16  q0, d4, #16         \n"
                        "vshll.u16  q1, d5, #16         \n"
                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"

                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "pld        [%4, #64]           \n"
                        "vld1.f32   {d9}, [%4 :64]      \n" // r24

                        "vshll.u16  q4, d9, #16         \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"

                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        //                         "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128] \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"

                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "vadd.f32   q14, q12, q14       \n"
                        "vadd.f32   q15, q13, q15       \n"

                        "sub        %5, %5, #256        \n" // kptr -= 8 * 16;

                        "vshrn.u32  d28, q14, #16       \n"
                        "vshrn.u32  d29, q15, #16       \n"

                        "vst1.f32   {d28-d29}, [%0 :64]! \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(kptr)          // %5
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v13.4s}, [%1], #16         \n" // sum0

                        "prfm   pldl1keep, [%2, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%2] \n" // r00 r01 r02

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "shll   v6.4s, %10.4h, #16          \n"
                        "shll2  v7.4s, %10.8h, #16          \n"

                        "fmul   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmul   v11.4s, v7.4s, v0.s[1]      \n"

                        "shll   v8.4s, %11.4h, #16          \n"
                        "shll2  v9.4s, %11.8h, #16          \n"

                        "fmul   v12.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "shll   v6.4s, %12.4h, #16          \n"
                        "shll2  v7.4s, %12.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"

                        "shll   v8.4s, %13.4h, #16          \n"
                        "shll2  v9.4s, %13.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v1.s[3]      \n"

                        "shll   v6.4s, %14.4h, #16          \n"
                        "shll2  v7.4s, %14.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %15.4h, #16          \n"
                        "shll2  v9.4s, %15.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "prfm   pldl1keep, [%3, #192]       \n"
                        "ld1    {v3.4h, v4.4h, v5.4h}, [%3] \n" // r10 r11 r12

                        "shll   v3.4s, v3.4h, #16           \n"
                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "shll   v6.4s, %16.4h, #16          \n"
                        "shll2  v7.4s, %16.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v3.s[1]      \n"

                        "shll   v8.4s, %17.4h, #16          \n"
                        "shll2  v9.4s, %17.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v3.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v3.s[3]      \n"

                        "shll   v6.4s, %18.4h, #16          \n"
                        "shll2  v7.4s, %18.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v4.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v4.s[1]      \n"

                        "shll   v8.4s, %19.4h, #16          \n"
                        "shll2  v9.4s, %19.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v4.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v4.s[3]      \n"

                        "shll   v6.4s, %20.4h, #16          \n"
                        "shll2  v7.4s, %20.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v5.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v5.s[1]      \n"

                        "shll   v8.4s, %21.4h, #16          \n"
                        "shll2  v9.4s, %21.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v5.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v5.s[3]      \n"

                        "prfm   pldl1keep, [%4, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%4] \n" // r20 r21 r22

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "shll   v6.4s, %22.4h, #16          \n"
                        "shll2  v7.4s, %22.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v0.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v0.s[1]      \n"

                        "shll   v8.4s, %23.4h, #16          \n"
                        "shll2  v9.4s, %23.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v0.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v0.s[3]      \n"

                        "shll   v6.4s, %24.4h, #16          \n"
                        "shll2  v7.4s, %24.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v1.s[1]      \n"

                        "shll   v8.4s, %25.4h, #16          \n"
                        "shll2  v9.4s, %25.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v1.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v1.s[3]      \n"

                        "shll   v6.4s, %26.4h, #16          \n"
                        "shll2  v7.4s, %26.8h, #16          \n"

                        "fmla   v10.4s, v6.4s, v2.s[0]      \n"
                        "fmla   v11.4s, v7.4s, v2.s[1]      \n"

                        "shll   v8.4s, %27.4h, #16          \n"
                        "shll2  v9.4s, %27.8h, #16          \n"

                        "fmla   v12.4s, v8.4s, v2.s[2]      \n"
                        "fmla   v13.4s, v9.4s, v2.s[3]      \n"

                        "fadd   v11.4s, v10.4s, v11.4s      \n"

                        "add    %2, %2, #16                 \n"
                        "fadd   v13.4s, v12.4s, v13.4s      \n"

                        "add    %3, %3, #16                 \n"
                        "fadd   v13.4s, v11.4s, v13.4s      \n"

                        "add    %4, %4, #16                 \n"
                        "shrn   v13.4h, v13.4s, #16         \n"

                        "st1    {v13.4h}, [%0], #8          \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2)            // %4
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_01), // %10
                        "w"(_k00_23), // %11
                        "w"(_k01_01), // %12
                        "w"(_k01_23), // %13
                        "w"(_k02_01), // %14
                        "w"(_k02_23), // %15
                        "w"(_k10_01), // %16
                        "w"(_k10_23), // %17
                        "w"(_k11_01), // %18
                        "w"(_k11_23), // %19
                        "w"(_k12_01), // %20
                        "w"(_k12_23), // %21
                        "w"(_k20_01), // %22
                        "w"(_k20_23), // %23
                        "w"(_k21_01), // %24
                        "w"(_k21_23), // %25
                        "w"(_k22_01), // %26
                        "w"(_k22_23)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d30-d31}, [%1 :128]! \n" // sum0

                        "pld        [%2, #192]          \n"
                        "vld1.u16   {d2-d4}, [%2 :64]   \n" // r00 r01 r02

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshll.u16  q2, d4, #16         \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmul.f32   q12, q8, d0[0]      \n"
                        "vmul.f32   q13, q9, d0[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmul.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%3, #192]          \n"
                        "vld1.u16   {d2-d4}, [%3 :64]   \n" // r10 r11 r12

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshll.u16  q2, d4, #16         \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q9, d0[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%4, #192]          \n"
                        "vld1.u16   {d2-d4}, [%4 :64]   \n" // r20 r21 r22

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshll.u16  q2, d4, #16         \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q9, d0[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128]! \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        //                         "pld        [%5, #256]          \n"
                        "vld1.u16   {d20-d23}, [%5 :128] \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"

                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "add        %2, %2, #16         \n"
                        "vadd.f32   q13, q12, q13       \n"

                        "add        %3, %3, #16         \n"
                        "vadd.f32   q15, q14, q15       \n"

                        "add        %4, %4, #16         \n"
                        "vadd.f32   q15, q13, q15       \n"

                        "sub        %5, %5, #256        \n" // kptr -= 8 * 16 * 2;

                        "vshrn.u32  d31, q15, #16       \n"

                        "vst1.u16   {d31}, [%0 :64]!    \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(kptr)          // %5
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    }
}
