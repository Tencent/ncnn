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

static void conv3x3s1_pack1to4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = bias ? vld1q_f32((const float*)bias + (p + 1) * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);
        out1.fill(_bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            float32x4_t _k00_0 = vld1q_f32(k0);
            float32x4_t _k01_0 = vld1q_f32(k0 + 4);
            float32x4_t _k02_0 = vld1q_f32(k0 + 8);
            float32x4_t _k10_0 = vld1q_f32(k0 + 12);
            float32x4_t _k11_0 = vld1q_f32(k0 + 16);
            float32x4_t _k12_0 = vld1q_f32(k0 + 20);
            float32x4_t _k20_0 = vld1q_f32(k0 + 24);
            float32x4_t _k21_0 = vld1q_f32(k0 + 28);
            float32x4_t _k22_0 = vld1q_f32(k0 + 32);

            float32x4_t _k00_1 = vld1q_f32(k1);
            float32x4_t _k01_1 = vld1q_f32(k1 + 4);
            float32x4_t _k02_1 = vld1q_f32(k1 + 8);
            float32x4_t _k10_1 = vld1q_f32(k1 + 12);
            float32x4_t _k11_1 = vld1q_f32(k1 + 16);
            float32x4_t _k12_1 = vld1q_f32(k1 + 20);
            float32x4_t _k20_1 = vld1q_f32(k1 + 24);
            float32x4_t _k21_1 = vld1q_f32(k1 + 28);
            float32x4_t _k22_1 = vld1q_f32(k1 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0] \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%1] \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v0.4s}, [%2], #16          \n"

                        "ld1    {v1.2s}, [%2]               \n"

                        "fmla   v24.4s, %10.4s, v0.s[0]     \n"
                        "fmla   v25.4s, %10.4s, v0.s[1]     \n"
                        "fmla   v26.4s, %10.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %10.4s, v0.s[3]     \n"
                        "fmla   v28.4s, %19.4s, v0.s[0]     \n"
                        "fmla   v29.4s, %19.4s, v0.s[1]     \n"
                        "fmla   v30.4s, %19.4s, v0.s[2]     \n"
                        "fmla   v31.4s, %19.4s, v0.s[3]     \n"

                        "fmla   v24.4s, %11.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %11.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %11.4s, v0.s[3]     \n"
                        "fmla   v27.4s, %11.4s, v1.s[0]     \n"
                        "fmla   v28.4s, %20.4s, v0.s[1]     \n"
                        "fmla   v29.4s, %20.4s, v0.s[2]     \n"
                        "fmla   v30.4s, %20.4s, v0.s[3]     \n"
                        "fmla   v31.4s, %20.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v2.4s}, [%3], #16          \n"

                        "ld1    {v3.2s}, [%3]               \n"

                        "fmla   v24.4s, %12.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %12.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %12.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %12.4s, v1.s[1]     \n"
                        "fmla   v28.4s, %21.4s, v0.s[2]     \n"
                        "fmla   v29.4s, %21.4s, v0.s[3]     \n"
                        "fmla   v30.4s, %21.4s, v1.s[0]     \n"
                        "fmla   v31.4s, %21.4s, v1.s[1]     \n"

                        "fmla   v24.4s, %13.4s, v2.s[0]     \n"
                        "fmla   v25.4s, %13.4s, v2.s[1]     \n"
                        "fmla   v26.4s, %13.4s, v2.s[2]     \n"
                        "fmla   v27.4s, %13.4s, v2.s[3]     \n"
                        "fmla   v28.4s, %22.4s, v2.s[0]     \n"
                        "fmla   v29.4s, %22.4s, v2.s[1]     \n"
                        "fmla   v30.4s, %22.4s, v2.s[2]     \n"
                        "fmla   v31.4s, %22.4s, v2.s[3]     \n"

                        "fmla   v24.4s, %14.4s, v2.s[1]     \n"
                        "fmla   v25.4s, %14.4s, v2.s[2]     \n"
                        "fmla   v26.4s, %14.4s, v2.s[3]     \n"
                        "fmla   v27.4s, %14.4s, v3.s[0]     \n"
                        "fmla   v28.4s, %23.4s, v2.s[1]     \n"
                        "fmla   v29.4s, %23.4s, v2.s[2]     \n"
                        "fmla   v30.4s, %23.4s, v2.s[3]     \n"
                        "fmla   v31.4s, %23.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v0.4s}, [%4], #16          \n"

                        "ld1    {v1.2s}, [%4]               \n"

                        "fmla   v24.4s, %15.4s, v2.s[2]     \n"
                        "fmla   v25.4s, %15.4s, v2.s[3]     \n"
                        "fmla   v26.4s, %15.4s, v3.s[0]     \n"
                        "fmla   v27.4s, %15.4s, v3.s[1]     \n"
                        "fmla   v28.4s, %24.4s, v2.s[2]     \n"
                        "fmla   v29.4s, %24.4s, v2.s[3]     \n"
                        "fmla   v30.4s, %24.4s, v3.s[0]     \n"
                        "fmla   v31.4s, %24.4s, v3.s[1]     \n"

                        "fmla   v24.4s, %16.4s, v0.s[0]     \n"
                        "fmla   v25.4s, %16.4s, v0.s[1]     \n"
                        "fmla   v26.4s, %16.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %16.4s, v0.s[3]     \n"
                        "fmla   v28.4s, %25.4s, v0.s[0]     \n"
                        "fmla   v29.4s, %25.4s, v0.s[1]     \n"
                        "fmla   v30.4s, %25.4s, v0.s[2]     \n"
                        "fmla   v31.4s, %25.4s, v0.s[3]     \n"

                        "fmla   v24.4s, %17.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %17.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %17.4s, v0.s[3]     \n"
                        "fmla   v27.4s, %17.4s, v1.s[0]     \n"
                        "fmla   v28.4s, %26.4s, v0.s[1]     \n"
                        "fmla   v29.4s, %26.4s, v0.s[2]     \n"
                        "fmla   v30.4s, %26.4s, v0.s[3]     \n"
                        "fmla   v31.4s, %26.4s, v1.s[0]     \n"

                        "fmla   v24.4s, %18.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %18.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %18.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %18.4s, v1.s[1]     \n"
                        "fmla   v28.4s, %27.4s, v0.s[2]     \n"
                        "fmla   v29.4s, %27.4s, v0.s[3]     \n"
                        "fmla   v30.4s, %27.4s, v1.s[0]     \n"
                        "fmla   v31.4s, %27.4s, v1.s[1]     \n"

                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%1], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_0), // %10
                        "w"(_k01_0), // %11
                        "w"(_k02_0), // %12
                        "w"(_k10_0), // %13
                        "w"(_k11_0), // %14
                        "w"(_k12_0), // %15
                        "w"(_k20_0), // %16
                        "w"(_k21_0), // %17
                        "w"(_k22_0), // %18
                        "w"(_k00_1), // %19
                        "w"(_k01_1), // %20
                        "w"(_k02_1), // %21
                        "w"(_k10_1), // %22
                        "w"(_k11_1), // %23
                        "w"(_k12_1), // %24
                        "w"(_k20_1), // %25
                        "w"(_k21_1), // %26
                        "w"(_k22_1)  // %27
                        : "memory", "v0", "v1", "v2", "v3", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v24.4s, v25.4s}, [%0]      \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v26.4s, v27.4s}, [%1]      \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v0.4s}, [%2]               \n"
                        "add    %2, %2, #8                  \n"

                        "fmla   v24.4s, %10.4s, v0.s[0]     \n"
                        "fmla   v25.4s, %10.4s, v0.s[1]     \n"
                        "fmla   v26.4s, %19.4s, v0.s[0]     \n"
                        "fmla   v27.4s, %19.4s, v0.s[1]     \n"

                        "fmla   v24.4s, %11.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %11.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %20.4s, v0.s[1]     \n"
                        "fmla   v27.4s, %20.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v1.4s}, [%3]               \n"

                        "fmla   v24.4s, %12.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %12.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %21.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %21.4s, v0.s[3]     \n"

                        "add    %3, %3, #8                  \n"

                        "fmla   v24.4s, %13.4s, v1.s[0]     \n"
                        "fmla   v25.4s, %13.4s, v1.s[1]     \n"
                        "fmla   v26.4s, %22.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %22.4s, v1.s[1]     \n"

                        "fmla   v24.4s, %14.4s, v1.s[1]     \n"
                        "fmla   v25.4s, %14.4s, v1.s[2]     \n"
                        "fmla   v26.4s, %23.4s, v1.s[1]     \n"
                        "fmla   v27.4s, %23.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v0.4s}, [%4]               \n"

                        "fmla   v24.4s, %15.4s, v1.s[2]     \n"
                        "fmla   v25.4s, %15.4s, v1.s[3]     \n"
                        "fmla   v26.4s, %24.4s, v1.s[2]     \n"
                        "fmla   v27.4s, %24.4s, v1.s[3]     \n"

                        "add    %4, %4, #8                  \n"

                        "fmla   v24.4s, %16.4s, v0.s[0]     \n"
                        "fmla   v25.4s, %16.4s, v0.s[1]     \n"
                        "fmla   v26.4s, %25.4s, v0.s[0]     \n"
                        "fmla   v27.4s, %25.4s, v0.s[1]     \n"

                        "fmla   v24.4s, %17.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %17.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %26.4s, v0.s[1]     \n"
                        "fmla   v27.4s, %26.4s, v0.s[2]     \n"

                        "fmla   v24.4s, %18.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %18.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %27.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %27.4s, v0.s[3]     \n"

                        "st1    {v24.4s, v25.4s}, [%0], #32 \n"
                        "st1    {v26.4s, v27.4s}, [%1], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00_0), // %10
                        "w"(_k01_0), // %11
                        "w"(_k02_0), // %12
                        "w"(_k10_0), // %13
                        "w"(_k11_0), // %14
                        "w"(_k12_0), // %15
                        "w"(_k20_0), // %16
                        "w"(_k21_0), // %17
                        "w"(_k22_0), // %18
                        "w"(_k00_1), // %19
                        "w"(_k01_1), // %20
                        "w"(_k02_1), // %21
                        "w"(_k10_1), // %22
                        "w"(_k11_1), // %23
                        "w"(_k12_1), // %24
                        "w"(_k20_1), // %25
                        "w"(_k21_1), // %26
                        "w"(_k22_1)  // %27
                        : "memory", "v0", "v1", "v24", "v25", "v26", "v27");
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum00 = vld1q_f32(outptr0);
                    float32x4_t _sum10 = vld1q_f32(outptr1);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);

                    _sum00 = vfmaq_laneq_f32(_sum00, _k00_0, _r0, 0);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k01_0, _r0, 1);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k02_0, _r0, 2);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k10_0, _r1, 0);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k11_0, _r1, 1);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k12_0, _r1, 2);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k20_0, _r2, 0);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k21_0, _r2, 1);
                    _sum00 = vfmaq_laneq_f32(_sum00, _k22_0, _r2, 2);

                    _sum10 = vfmaq_laneq_f32(_sum10, _k00_1, _r0, 0);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k01_1, _r0, 1);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k02_1, _r0, 2);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k10_1, _r1, 0);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k11_1, _r1, 1);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k12_1, _r1, 2);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k20_1, _r2, 0);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k21_1, _r2, 1);
                    _sum10 = vfmaq_laneq_f32(_sum10, _k22_1, _r2, 2);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr1, _sum10);

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 4;
                    outptr1 += 4;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 4;
            k1 += 9 * 4;
        }
    }
#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k01 = vld1q_f32(k0 + 4);
            float32x4_t _k02 = vld1q_f32(k0 + 8);
            float32x4_t _k10 = vld1q_f32(k0 + 12);
            float32x4_t _k11 = vld1q_f32(k0 + 16);
            float32x4_t _k12 = vld1q_f32(k0 + 20);
            float32x4_t _k20 = vld1q_f32(k0 + 24);
            float32x4_t _k21 = vld1q_f32(k0 + 28);
            float32x4_t _k22 = vld1q_f32(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

#if __aarch64__
                for (; j + 7 < outw; j += 8)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%1], #32   \n"

                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0] \n"

                        "fmla   v24.4s, %8.4s, v0.s[0]      \n"
                        "fmla   v25.4s, %8.4s, v0.s[1]      \n"
                        "fmla   v26.4s, %8.4s, v0.s[2]      \n"
                        "fmla   v27.4s, %8.4s, v0.s[3]      \n"
                        "fmla   v28.4s, %8.4s, v1.s[0]      \n"
                        "fmla   v29.4s, %8.4s, v1.s[1]      \n"
                        "fmla   v30.4s, %8.4s, v1.s[2]      \n"
                        "fmla   v31.4s, %8.4s, v1.s[3]      \n"

                        "ld1    {v2.2s}, [%1]               \n"

                        "fmla   v24.4s, %9.4s, v0.s[1]      \n"
                        "fmla   v25.4s, %9.4s, v0.s[2]      \n"
                        "fmla   v26.4s, %9.4s, v0.s[3]      \n"
                        "fmla   v27.4s, %9.4s, v1.s[0]      \n"
                        "fmla   v28.4s, %9.4s, v1.s[1]      \n"
                        "fmla   v29.4s, %9.4s, v1.s[2]      \n"
                        "fmla   v30.4s, %9.4s, v1.s[3]      \n"
                        "fmla   v31.4s, %9.4s, v2.s[0]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%2], #32   \n"

                        "fmla   v24.4s, %10.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %10.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %10.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %10.4s, v1.s[1]     \n"
                        "fmla   v28.4s, %10.4s, v1.s[2]     \n"
                        "fmla   v29.4s, %10.4s, v1.s[3]     \n"
                        "fmla   v30.4s, %10.4s, v2.s[0]     \n"
                        "fmla   v31.4s, %10.4s, v2.s[1]     \n"

                        "ld1    {v2.2s}, [%2]               \n"

                        "fmla   v24.4s, %11.4s, v4.s[0]     \n"
                        "fmla   v25.4s, %11.4s, v4.s[1]     \n"
                        "fmla   v26.4s, %11.4s, v4.s[2]     \n"
                        "fmla   v27.4s, %11.4s, v4.s[3]     \n"
                        "fmla   v28.4s, %11.4s, v5.s[0]     \n"
                        "fmla   v29.4s, %11.4s, v5.s[1]     \n"
                        "fmla   v30.4s, %11.4s, v5.s[2]     \n"
                        "fmla   v31.4s, %11.4s, v5.s[3]     \n"

                        "fmla   v24.4s, %12.4s, v4.s[1]     \n"
                        "fmla   v25.4s, %12.4s, v4.s[2]     \n"
                        "fmla   v26.4s, %12.4s, v4.s[3]     \n"
                        "fmla   v27.4s, %12.4s, v5.s[0]     \n"
                        "fmla   v28.4s, %12.4s, v5.s[1]     \n"
                        "fmla   v29.4s, %12.4s, v5.s[2]     \n"
                        "fmla   v30.4s, %12.4s, v5.s[3]     \n"
                        "fmla   v31.4s, %12.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%3], #32   \n"

                        "fmla   v24.4s, %13.4s, v4.s[2]     \n"
                        "fmla   v25.4s, %13.4s, v4.s[3]     \n"
                        "fmla   v26.4s, %13.4s, v5.s[0]     \n"
                        "fmla   v27.4s, %13.4s, v5.s[1]     \n"
                        "fmla   v28.4s, %13.4s, v5.s[2]     \n"
                        "fmla   v29.4s, %13.4s, v5.s[3]     \n"
                        "fmla   v30.4s, %13.4s, v2.s[0]     \n"
                        "fmla   v31.4s, %13.4s, v2.s[1]     \n"

                        "ld1    {v2.2s}, [%3]               \n"

                        "fmla   v24.4s, %14.4s, v0.s[0]     \n"
                        "fmla   v25.4s, %14.4s, v0.s[1]     \n"
                        "fmla   v26.4s, %14.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %14.4s, v0.s[3]     \n"
                        "fmla   v28.4s, %14.4s, v1.s[0]     \n"
                        "fmla   v29.4s, %14.4s, v1.s[1]     \n"
                        "fmla   v30.4s, %14.4s, v1.s[2]     \n"
                        "fmla   v31.4s, %14.4s, v1.s[3]     \n"

                        "fmla   v24.4s, %15.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %15.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %15.4s, v0.s[3]     \n"
                        "fmla   v27.4s, %15.4s, v1.s[0]     \n"
                        "fmla   v28.4s, %15.4s, v1.s[1]     \n"
                        "fmla   v29.4s, %15.4s, v1.s[2]     \n"
                        "fmla   v30.4s, %15.4s, v1.s[3]     \n"
                        "fmla   v31.4s, %15.4s, v2.s[0]     \n"

                        "sub    %0, %0, #64                 \n"

                        "fmla   v24.4s, %16.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %16.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %16.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %16.4s, v1.s[1]     \n"
                        "fmla   v28.4s, %16.4s, v1.s[2]     \n"
                        "fmla   v29.4s, %16.4s, v1.s[3]     \n"
                        "fmla   v30.4s, %16.4s, v2.s[0]     \n"
                        "fmla   v31.4s, %16.4s, v2.s[1]     \n"

                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v2", "v4", "v5", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
#endif // __aarch64__
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0] \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.4s}, [%1], #16          \n"

                        "fmla   v24.4s, %8.4s, v0.s[0]      \n"
                        "fmla   v25.4s, %8.4s, v0.s[1]      \n"
                        "fmla   v26.4s, %8.4s, v0.s[2]      \n"
                        "fmla   v27.4s, %8.4s, v0.s[3]      \n"

                        "ld1    {v1.2s}, [%1]               \n"

                        "fmla   v24.4s, %9.4s, v0.s[1]      \n"
                        "fmla   v25.4s, %9.4s, v0.s[2]      \n"
                        "fmla   v26.4s, %9.4s, v0.s[3]      \n"
                        "fmla   v27.4s, %9.4s, v1.s[0]      \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v2.4s}, [%2], #16          \n"

                        "fmla   v24.4s, %10.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %10.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %10.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %10.4s, v1.s[1]     \n"

                        "ld1    {v3.2s}, [%2]               \n"

                        "fmla   v24.4s, %11.4s, v2.s[0]     \n"
                        "fmla   v25.4s, %11.4s, v2.s[1]     \n"
                        "fmla   v26.4s, %11.4s, v2.s[2]     \n"
                        "fmla   v27.4s, %11.4s, v2.s[3]     \n"

                        "fmla   v24.4s, %12.4s, v2.s[1]     \n"
                        "fmla   v25.4s, %12.4s, v2.s[2]     \n"
                        "fmla   v26.4s, %12.4s, v2.s[3]     \n"
                        "fmla   v27.4s, %12.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.4s}, [%3], #16          \n"

                        "fmla   v24.4s, %13.4s, v2.s[2]     \n"
                        "fmla   v25.4s, %13.4s, v2.s[3]     \n"
                        "fmla   v26.4s, %13.4s, v3.s[0]     \n"
                        "fmla   v27.4s, %13.4s, v3.s[1]     \n"

                        "ld1    {v1.2s}, [%3]               \n"

                        "fmla   v24.4s, %14.4s, v0.s[0]     \n"
                        "fmla   v25.4s, %14.4s, v0.s[1]     \n"
                        "fmla   v26.4s, %14.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %14.4s, v0.s[3]     \n"

                        "fmla   v24.4s, %15.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %15.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %15.4s, v0.s[3]     \n"
                        "fmla   v27.4s, %15.4s, v1.s[0]     \n"

                        "fmla   v24.4s, %16.4s, v0.s[2]     \n"
                        "fmla   v25.4s, %16.4s, v0.s[3]     \n"
                        "fmla   v26.4s, %16.4s, v1.s[0]     \n"
                        "fmla   v27.4s, %16.4s, v1.s[1]     \n"

                        "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v2", "v3", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d24-d31}       \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1]!      \n"

                        "vmla.f32   q12, %q8, d0[0]     \n"
                        "vmla.f32   q13, %q8, d0[1]     \n"
                        "vmla.f32   q14, %q8, d1[0]     \n"
                        "vmla.f32   q15, %q8, d1[1]     \n"

                        "vld1.f32   {d2}, [%1]          \n"

                        "vmla.f32   q12, %q9, d0[1]     \n"
                        "vmla.f32   q13, %q9, d1[0]     \n"
                        "vmla.f32   q14, %q9, d1[1]     \n"
                        "vmla.f32   q15, %q9, d2[0]     \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d4-d5}, [%2]!      \n"

                        "vmla.f32   q12, %q10, d1[0]    \n"
                        "vmla.f32   q13, %q10, d1[1]    \n"
                        "vmla.f32   q14, %q10, d2[0]    \n"
                        "vmla.f32   q15, %q10, d2[1]    \n"

                        "vmla.f32   q12, %q11, d4[0]    \n"
                        "vmla.f32   q13, %q11, d4[1]    \n"
                        "vmla.f32   q14, %q11, d5[0]    \n"
                        "vmla.f32   q15, %q11, d5[1]    \n"

                        "vld1.f32   {d3}, [%2]          \n"

                        "vmla.f32   q12, %q12, d4[1]    \n"
                        "vmla.f32   q13, %q12, d5[0]    \n"
                        "vmla.f32   q14, %q12, d5[1]    \n"
                        "vmla.f32   q15, %q12, d3[0]    \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d0-d1}, [%3]!      \n"

                        "vmla.f32   q12, %q13, d5[0]    \n"
                        "vmla.f32   q13, %q13, d5[1]    \n"
                        "vmla.f32   q14, %q13, d3[0]    \n"
                        "vmla.f32   q15, %q13, d3[1]    \n"

                        "vmla.f32   q12, %q14, d0[0]    \n"
                        "vmla.f32   q13, %q14, d0[1]    \n"
                        "vmla.f32   q14, %q14, d1[0]    \n"
                        "vmla.f32   q15, %q14, d1[1]    \n"

                        "vld1.f32   {d2}, [%3]          \n"

                        "vmla.f32   q12, %q15, d0[1]    \n"
                        "vmla.f32   q13, %q15, d1[0]    \n"
                        "vmla.f32   q14, %q15, d1[1]    \n"
                        "vmla.f32   q15, %q15, d2[0]    \n"

                        "vmla.f32   q12, %q16, d1[0]    \n"
                        "vmla.f32   q13, %q16, d1[1]    \n"
                        "vmla.f32   q14, %q16, d2[0]    \n"
                        "vmla.f32   q15, %q16, d2[1]    \n"

                        "vstm       %0!, {d24-d31}      \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "q0", "q1", "q2", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v24.4s, v25.4s}, [%0]      \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.4s}, [%1]               \n"

                        "fmul   v26.4s, %8.4s, v0.s[0]      \n"
                        "fmul   v27.4s, %8.4s, v0.s[1]      \n"
                        "fmla   v24.4s, %9.4s, v0.s[1]      \n"
                        "fmla   v25.4s, %9.4s, v0.s[2]      \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v1.4s}, [%2]               \n"

                        "fmla   v26.4s, %10.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %10.4s, v0.s[3]     \n"
                        "fmla   v24.4s, %11.4s, v1.s[0]     \n"
                        "fmla   v25.4s, %11.4s, v1.s[1]     \n"

                        "add    %1, %1, #8                  \n"

                        "fmla   v26.4s, %12.4s, v1.s[1]     \n"
                        "fmla   v27.4s, %12.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.4s}, [%3]               \n"

                        "fmla   v24.4s, %13.4s, v1.s[2]     \n"
                        "fmla   v25.4s, %13.4s, v1.s[3]     \n"
                        "fmla   v26.4s, %14.4s, v0.s[0]     \n"
                        "fmla   v27.4s, %14.4s, v0.s[1]     \n"

                        "add    %2, %2, #8                  \n"

                        "fmla   v24.4s, %15.4s, v0.s[1]     \n"
                        "fmla   v25.4s, %15.4s, v0.s[2]     \n"
                        "fmla   v26.4s, %16.4s, v0.s[2]     \n"
                        "fmla   v27.4s, %16.4s, v0.s[3]     \n"

                        "add    %3, %3, #8                  \n"

                        "fadd   v24.4s, v24.4s, v26.4s      \n"
                        "fadd   v25.4s, v25.4s, v27.4s      \n"

                        "st1    {v24.4s, v25.4s}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "v0", "v1", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128] \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1]       \n"

                        "vmul.f32   q14, %q8, d0[0]     \n"
                        "vmul.f32   q15, %q8, d0[1]     \n"
                        "vmla.f32   q12, %q9, d0[1]     \n"
                        "vmla.f32   q13, %q9, d1[0]     \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d2-d3}, [%2]       \n"

                        "vmla.f32   q14, %q10, d1[0]    \n"
                        "vmla.f32   q15, %q10, d1[1]    \n"
                        "vmla.f32   q12, %q11, d2[0]    \n"
                        "vmla.f32   q13, %q11, d2[1]    \n"

                        "add        %1, %1, #8          \n"

                        "vmla.f32   q14, %q12, d2[1]    \n"
                        "vmla.f32   q15, %q12, d3[0]    \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d0-d1}, [%3]       \n"

                        "vmla.f32   q12, %q13, d3[0]    \n"
                        "vmla.f32   q13, %q13, d3[1]    \n"
                        "vmla.f32   q14, %q14, d0[0]    \n"
                        "vmla.f32   q15, %q14, d0[1]    \n"

                        "add        %2, %2, #8          \n"

                        "vmla.f32   q12, %q15, d0[1]    \n"
                        "vmla.f32   q13, %q15, d1[0]    \n"
                        "vmla.f32   q14, %q16, d1[0]    \n"
                        "vmla.f32   q15, %q16, d1[1]    \n"

                        "add        %3, %3, #8          \n"

                        "vadd.f32   q12, q12, q14       \n"
                        "vadd.f32   q13, q13, q15       \n"

                        "vst1.f32   {d24-d27}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00), // %8
                        "w"(_k01), // %9
                        "w"(_k02), // %10
                        "w"(_k10), // %11
                        "w"(_k11), // %12
                        "w"(_k12), // %13
                        "w"(_k20), // %14
                        "w"(_k21), // %15
                        "w"(_k22)  // %16
                        : "memory", "q0", "q1", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);

#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _k00, _r0, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k01, _r0, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k02, _r0, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k10, _r1, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k11, _r1, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k12, _r1, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k20, _r2, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k21, _r2, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k22, _r2, 2);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _k00, vget_low_f32(_r0), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k01, vget_low_f32(_r0), 1);
                    _sum0 = vmlaq_lane_f32(_sum0, _k02, vget_high_f32(_r0), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k10, vget_low_f32(_r1), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k11, vget_low_f32(_r1), 1);
                    _sum0 = vmlaq_lane_f32(_sum0, _k12, vget_high_f32(_r1), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k20, vget_low_f32(_r2), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k21, vget_low_f32(_r2), 1);
                    _sum0 = vmlaq_lane_f32(_sum0, _k22, vget_high_f32(_r2), 0);
#endif

                    vst1q_f32(outptr0, _sum0);

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                    outptr0 += 4;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * 4;
        }
    }
}

static void conv3x3s2_pack1to4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* bias = _bias;

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = bias ? vld1q_f32((const float*)bias + (p + 1) * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);
        out1.fill(_bias1);

        const float* k0 = kernel.channel(p);
        const float* k1 = kernel.channel(p + 1);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            float32x4_t _k00_0 = vld1q_f32(k0);
            float32x4_t _k01_0 = vld1q_f32(k0 + 4);
            float32x4_t _k02_0 = vld1q_f32(k0 + 8);
            float32x4_t _k10_0 = vld1q_f32(k0 + 12);
            float32x4_t _k11_0 = vld1q_f32(k0 + 16);
            float32x4_t _k12_0 = vld1q_f32(k0 + 20);
            float32x4_t _k20_0 = vld1q_f32(k0 + 24);
            float32x4_t _k21_0 = vld1q_f32(k0 + 28);
            float32x4_t _k22_0 = vld1q_f32(k0 + 32);

            float32x4_t _k00_1 = vld1q_f32(k1);
            float32x4_t _k01_1 = vld1q_f32(k1 + 4);
            float32x4_t _k02_1 = vld1q_f32(k1 + 8);
            float32x4_t _k10_1 = vld1q_f32(k1 + 12);
            float32x4_t _k11_1 = vld1q_f32(k1 + 16);
            float32x4_t _k12_1 = vld1q_f32(k1 + 20);
            float32x4_t _k20_1 = vld1q_f32(k1 + 24);
            float32x4_t _k21_1 = vld1q_f32(k1 + 28);
            float32x4_t _k22_1 = vld1q_f32(k1 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int nn = outw >> 2;
                int remain = outw & 3;

                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%1] \n" // sum0

                        // r0
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%3], #32   \n"
                        "ld1r   {v4.4s}, [%3]               \n"

                        "fmla   v6.4s, %12.4s, v0.s[0]      \n"
                        "fmla   v7.4s, %12.4s, v0.s[2]      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%2] \n" // sum1

                        "fmla   v8.4s, %12.4s, v1.s[0]      \n"
                        "fmla   v9.4s, %12.4s, v1.s[2]      \n"

                        "fmla   v10.4s, %21.4s, v0.s[0]     \n"
                        "fmla   v11.4s, %21.4s, v0.s[2]     \n"
                        "fmla   v12.4s, %21.4s, v1.s[0]     \n"
                        "fmla   v13.4s, %21.4s, v1.s[2]     \n"

                        "fmla   v6.4s, %13.4s, v0.s[1]      \n"
                        "fmla   v7.4s, %13.4s, v0.s[3]      \n"
                        "fmla   v8.4s, %13.4s, v1.s[1]      \n"
                        "fmla   v9.4s, %13.4s, v1.s[3]      \n"
                        "fmla   v10.4s, %22.4s, v0.s[1]     \n"
                        "fmla   v11.4s, %22.4s, v0.s[3]     \n"
                        "fmla   v12.4s, %22.4s, v1.s[1]     \n"
                        "fmla   v13.4s, %22.4s, v1.s[3]     \n"

                        // r1
                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v2.4s, v3.4s}, [%4], #32   \n"
                        "ld1r   {v5.4s}, [%4]               \n"

                        "fmla   v6.4s, %14.4s, v0.s[2]      \n"
                        "fmla   v7.4s, %14.4s, v1.s[0]      \n"
                        "fmla   v8.4s, %14.4s, v1.s[2]      \n"
                        "fmla   v9.4s, %14.4s, v4.s[0]      \n"
                        "fmla   v10.4s, %23.4s, v0.s[2]     \n"
                        "fmla   v11.4s, %23.4s, v1.s[0]     \n"
                        "fmla   v12.4s, %23.4s, v1.s[2]     \n"
                        "fmla   v13.4s, %23.4s, v4.s[0]     \n"

                        "fmla   v6.4s, %15.4s, v2.s[0]      \n"
                        "fmla   v7.4s, %15.4s, v2.s[2]      \n"
                        "fmla   v8.4s, %15.4s, v3.s[0]      \n"
                        "fmla   v9.4s, %15.4s, v3.s[2]      \n"
                        "fmla   v10.4s, %24.4s, v2.s[0]     \n"
                        "fmla   v11.4s, %24.4s, v2.s[2]     \n"
                        "fmla   v12.4s, %24.4s, v3.s[0]     \n"
                        "fmla   v13.4s, %24.4s, v3.s[2]     \n"

                        "fmla   v6.4s, %16.4s, v2.s[1]      \n"
                        "fmla   v7.4s, %16.4s, v2.s[3]      \n"
                        "fmla   v8.4s, %16.4s, v3.s[1]      \n"
                        "fmla   v9.4s, %16.4s, v3.s[3]      \n"
                        "fmla   v10.4s, %25.4s, v2.s[1]     \n"
                        "fmla   v11.4s, %25.4s, v2.s[3]     \n"
                        "fmla   v12.4s, %25.4s, v3.s[1]     \n"
                        "fmla   v13.4s, %25.4s, v3.s[3]     \n"

                        // r2
                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%5], #32   \n"
                        "ld1r   {v4.4s}, [%5]               \n"

                        "fmla   v6.4s, %17.4s, v2.s[2]      \n"
                        "fmla   v7.4s, %17.4s, v3.s[0]      \n"
                        "fmla   v8.4s, %17.4s, v3.s[2]      \n"
                        "fmla   v9.4s, %17.4s, v5.s[0]      \n"
                        "fmla   v10.4s, %26.4s, v2.s[2]     \n"
                        "fmla   v11.4s, %26.4s, v3.s[0]     \n"
                        "fmla   v12.4s, %26.4s, v3.s[2]     \n"
                        "fmla   v13.4s, %26.4s, v5.s[0]     \n"

                        "fmla   v6.4s, %18.4s, v0.s[0]      \n"
                        "fmla   v7.4s, %18.4s, v0.s[2]      \n"
                        "fmla   v8.4s, %18.4s, v1.s[0]      \n"
                        "fmla   v9.4s, %18.4s, v1.s[2]      \n"
                        "fmla   v10.4s, %27.4s, v0.s[0]     \n"
                        "fmla   v11.4s, %27.4s, v0.s[2]     \n"
                        "fmla   v12.4s, %27.4s, v1.s[0]     \n"
                        "fmla   v13.4s, %27.4s, v1.s[2]     \n"

                        "fmla   v6.4s, %19.4s, v0.s[1]      \n"
                        "fmla   v7.4s, %19.4s, v0.s[3]      \n"
                        "fmla   v8.4s, %19.4s, v1.s[1]      \n"
                        "fmla   v9.4s, %19.4s, v1.s[3]      \n"
                        "fmla   v10.4s, %28.4s, v0.s[1]     \n"
                        "fmla   v11.4s, %28.4s, v0.s[3]     \n"
                        "fmla   v12.4s, %28.4s, v1.s[1]     \n"
                        "fmla   v13.4s, %28.4s, v1.s[3]     \n"

                        "fmla   v6.4s, %20.4s, v0.s[2]      \n"
                        "fmla   v7.4s, %20.4s, v1.s[0]      \n"
                        "fmla   v8.4s, %20.4s, v1.s[2]      \n"
                        "fmla   v9.4s, %20.4s, v4.s[0]      \n"
                        "fmla   v10.4s, %29.4s, v0.s[2]     \n"
                        "fmla   v11.4s, %29.4s, v1.s[0]     \n"
                        "fmla   v12.4s, %29.4s, v1.s[2]     \n"
                        "fmla   v13.4s, %29.4s, v4.s[0]     \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%1], #64 \n"
                        "st1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%2], #64 \n"

                        "bne    0b                          \n"

                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2)       // %5
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "w"(_k00_0), // %12
                        "w"(_k01_0), // %13
                        "w"(_k02_0), // %14
                        "w"(_k10_0), // %15
                        "w"(_k11_0), // %16
                        "w"(_k12_0), // %17
                        "w"(_k20_0), // %18
                        "w"(_k21_0), // %19
                        "w"(_k22_0), // %20
                        "w"(_k00_1), // %21
                        "w"(_k01_1), // %22
                        "w"(_k02_1), // %23
                        "w"(_k10_1), // %24
                        "w"(_k11_1), // %25
                        "w"(_k12_1), // %26
                        "w"(_k20_1), // %27
                        "w"(_k21_1), // %28
                        "w"(_k22_1)  // %29
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13");
                }

                for (; remain > 0; remain--)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);
                    float32x4_t _sum1 = vld1q_f32(outptr1);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);

                    _sum0 = vfmaq_laneq_f32(_sum0, _k00_0, _r0, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k01_0, _r0, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k02_0, _r0, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k10_0, _r1, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k11_0, _r1, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k12_0, _r1, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k20_0, _r2, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k21_0, _r2, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k22_0, _r2, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _k00_1, _r0, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k01_1, _r0, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k02_1, _r0, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k10_1, _r1, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k11_1, _r1, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k12_1, _r1, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k20_1, _r2, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k21_1, _r2, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _k22_1, _r2, 2);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr1, _sum1);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 4;
                    outptr1 += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 4;
            k1 += 9 * 4;
        }
    }
#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k01 = vld1q_f32(k0 + 4);
            float32x4_t _k02 = vld1q_f32(k0 + 8);
            float32x4_t _k10 = vld1q_f32(k0 + 12);
            float32x4_t _k11 = vld1q_f32(k0 + 16);
            float32x4_t _k12 = vld1q_f32(k0 + 20);
            float32x4_t _k20 = vld1q_f32(k0 + 24);
            float32x4_t _k21 = vld1q_f32(k0 + 28);
            float32x4_t _k22 = vld1q_f32(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int nn = outw >> 2;
                int remain = outw & 3;

#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%1] \n" // sum0

                        // r0
                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%2], #32   \n"
                        "ld1r   {v4.4s}, [%2]               \n"

                        "fmla   v6.4s, %10.4s, v0.s[0]      \n"
                        "fmla   v7.4s, %10.4s, v0.s[2]      \n"
                        "fmla   v8.4s, %10.4s, v1.s[0]      \n"
                        "fmla   v9.4s, %10.4s, v1.s[2]      \n"

                        "fmla   v6.4s, %11.4s, v0.s[1]      \n"
                        "fmla   v7.4s, %11.4s, v0.s[3]      \n"
                        "fmla   v8.4s, %11.4s, v1.s[1]      \n"
                        "fmla   v9.4s, %11.4s, v1.s[3]      \n"

                        // r1
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v2.4s, v3.4s}, [%3], #32   \n"
                        "ld1r   {v5.4s}, [%3]               \n"

                        "fmla   v6.4s, %12.4s, v0.s[2]      \n"
                        "fmla   v7.4s, %12.4s, v1.s[0]      \n"
                        "fmla   v8.4s, %12.4s, v1.s[2]      \n"
                        "fmla   v9.4s, %12.4s, v4.s[0]      \n"

                        "fmla   v6.4s, %13.4s, v2.s[0]      \n"
                        "fmla   v7.4s, %13.4s, v2.s[2]      \n"
                        "fmla   v8.4s, %13.4s, v3.s[0]      \n"
                        "fmla   v9.4s, %13.4s, v3.s[2]      \n"

                        "fmla   v6.4s, %14.4s, v2.s[1]      \n"
                        "fmla   v7.4s, %14.4s, v2.s[3]      \n"
                        "fmla   v8.4s, %14.4s, v3.s[1]      \n"
                        "fmla   v9.4s, %14.4s, v3.s[3]      \n"

                        // r2
                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%4], #32   \n"
                        "ld1r   {v4.4s}, [%4]               \n"

                        "fmla   v6.4s, %15.4s, v2.s[2]      \n"
                        "fmla   v7.4s, %15.4s, v3.s[0]      \n"
                        "fmla   v8.4s, %15.4s, v3.s[2]      \n"
                        "fmla   v9.4s, %15.4s, v5.s[0]      \n"

                        "fmla   v6.4s, %16.4s, v0.s[0]      \n"
                        "fmla   v7.4s, %16.4s, v0.s[2]      \n"
                        "fmla   v8.4s, %16.4s, v1.s[0]      \n"
                        "fmla   v9.4s, %16.4s, v1.s[2]      \n"

                        "fmla   v6.4s, %17.4s, v0.s[1]      \n"
                        "fmla   v7.4s, %17.4s, v0.s[3]      \n"
                        "fmla   v8.4s, %17.4s, v1.s[1]      \n"
                        "fmla   v9.4s, %17.4s, v1.s[3]      \n"

                        "fmla   v6.4s, %18.4s, v0.s[2]      \n"
                        "fmla   v7.4s, %18.4s, v1.s[0]      \n"
                        "fmla   v8.4s, %18.4s, v1.s[2]      \n"
                        "fmla   v9.4s, %18.4s, v4.s[0]      \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%1], #64 \n"

                        "bne    0b                          \n"

                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00), // %10
                        "w"(_k01), // %11
                        "w"(_k02), // %12
                        "w"(_k10), // %13
                        "w"(_k11), // %14
                        "w"(_k12), // %15
                        "w"(_k20), // %16
                        "w"(_k21), // %17
                        "w"(_k22)  // %18
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
                }
#else  // __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1, {d0-d7}         \n" // sum0

                        // r0
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d8-d11}, [%2]!     \n"
                        "vld1.f32   {d12[]}, [%2]       \n"

                        "vmla.f32   q0, %q10, d8[0]     \n"
                        "vmla.f32   q1, %q10, d9[0]     \n"
                        "vmla.f32   q2, %q10, d10[0]    \n"
                        "vmla.f32   q3, %q10, d11[0]    \n"

                        "vmla.f32   q0, %q11, d8[1]     \n"
                        "vmla.f32   q1, %q11, d9[1]     \n"
                        "vmla.f32   q2, %q11, d10[1]    \n"
                        "vmla.f32   q3, %q11, d11[1]    \n"

                        "vmla.f32   q0, %q12, d9[0]     \n"
                        "vmla.f32   q1, %q12, d10[0]    \n"
                        "vmla.f32   q2, %q12, d11[0]    \n"

                        // r1
                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d8-d11}, [%3]!     \n"
                        "vld1.f32   {d13[]}, [%3]       \n"

                        "vmla.f32   q3, %q12, d12[0]    \n"

                        "vmla.f32   q0, %q13, d8[0]     \n"
                        "vmla.f32   q1, %q13, d9[0]     \n"
                        "vmla.f32   q2, %q13, d10[0]    \n"
                        "vmla.f32   q3, %q13, d11[0]    \n"

                        "vmla.f32   q0, %q14, d8[1]     \n"
                        "vmla.f32   q1, %q14, d9[1]     \n"
                        "vmla.f32   q2, %q14, d10[1]    \n"
                        "vmla.f32   q3, %q14, d11[1]    \n"

                        "vmla.f32   q0, %q15, d9[0]     \n"
                        "vmla.f32   q1, %q15, d10[0]    \n"
                        "vmla.f32   q2, %q15, d11[0]    \n"

                        // r2
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d8-d11}, [%4]!     \n"
                        "vld1.f32   {d12[]}, [%4]       \n"

                        "vmla.f32   q3, %q15, d13[0]    \n"

                        "vmla.f32   q0, %q16, d8[0]     \n"
                        "vmla.f32   q1, %q16, d9[0]     \n"
                        "vmla.f32   q2, %q16, d10[0]    \n"
                        "vmla.f32   q3, %q16, d11[0]    \n"

                        "vmla.f32   q0, %q17, d8[1]     \n"
                        "vmla.f32   q1, %q17, d9[1]     \n"
                        "vmla.f32   q2, %q17, d10[1]    \n"
                        "vmla.f32   q3, %q17, d11[1]    \n"

                        "vmla.f32   q0, %q18, d9[0]     \n"
                        "vmla.f32   q1, %q18, d10[0]    \n"
                        "vmla.f32   q2, %q18, d11[0]    \n"
                        "vmla.f32   q3, %q18, d12[0]    \n"

                        "subs       %0, %0, #1          \n"

                        "vstm       %1!, {d0-d7}        \n"

                        "bne        0b                  \n"

                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2)       // %4
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k00), // %10
                        "w"(_k01), // %11
                        "w"(_k02), // %12
                        "w"(_k10), // %13
                        "w"(_k11), // %14
                        "w"(_k12), // %15
                        "w"(_k20), // %16
                        "w"(_k21), // %17
                        "w"(_k22)  // %18
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
                }
#endif // __aarch64__

                for (; remain > 0; remain--)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _r2 = vld1q_f32(r2);

#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _k00, _r0, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k01, _r0, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k02, _r0, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k10, _r1, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k11, _r1, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k12, _r1, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k20, _r2, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k21, _r2, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _k22, _r2, 2);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _k00, vget_low_f32(_r0), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k01, vget_low_f32(_r0), 1);
                    _sum0 = vmlaq_lane_f32(_sum0, _k02, vget_high_f32(_r0), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k10, vget_low_f32(_r1), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k11, vget_low_f32(_r1), 1);
                    _sum0 = vmlaq_lane_f32(_sum0, _k12, vget_high_f32(_r1), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k20, vget_low_f32(_r2), 0);
                    _sum0 = vmlaq_lane_f32(_sum0, _k21, vget_low_f32(_r2), 1);
                    _sum0 = vmlaq_lane_f32(_sum0, _k22, vget_high_f32(_r2), 0);
#endif

                    vst1q_f32(outptr0, _sum0);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * 4;
        }
    }
}
