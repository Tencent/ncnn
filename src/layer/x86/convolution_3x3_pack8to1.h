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

static void conv3x3s1_pack8to1_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        out0.fill(bias0);
        out1.fill(bias1);

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

            __m256 _k00_0 = _mm256_loadu_ps(k0);
            __m256 _k01_0 = _mm256_loadu_ps(k0 + 8);
            __m256 _k02_0 = _mm256_loadu_ps(k0 + 16);
            __m256 _k10_0 = _mm256_loadu_ps(k0 + 24);
            __m256 _k11_0 = _mm256_loadu_ps(k0 + 32);
            __m256 _k12_0 = _mm256_loadu_ps(k0 + 40);
            __m256 _k20_0 = _mm256_loadu_ps(k0 + 48);
            __m256 _k21_0 = _mm256_loadu_ps(k0 + 56);
            __m256 _k22_0 = _mm256_loadu_ps(k0 + 64);

            __m256 _k00_1 = _mm256_loadu_ps(k1);
            __m256 _k01_1 = _mm256_loadu_ps(k1 + 8);
            __m256 _k02_1 = _mm256_loadu_ps(k1 + 16);
            __m256 _k10_1 = _mm256_loadu_ps(k1 + 24);
            __m256 _k11_1 = _mm256_loadu_ps(k1 + 32);
            __m256 _k12_1 = _mm256_loadu_ps(k1 + 40);
            __m256 _k20_1 = _mm256_loadu_ps(k1 + 48);
            __m256 _k21_1 = _mm256_loadu_ps(k1 + 56);
            __m256 _k22_1 = _mm256_loadu_ps(k1 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j < outw; j++)
                {
                    __m256 _sum00 = _mm256_loadu_ps(outptr0);
                    __m256 _sum10 = _mm256_loadu_ps(outptr1);

                    __m256 _r01 = _mm256_loadu_ps(r0);
                    __m256 _r02 = _mm256_loadu_ps(r0+8);
                    __m256 _r03 = _mm256_loadu_ps(r0+16);
                    __m256 _r11 = _mm256_loadu_ps(r1);
                    __m256 _r12 = _mm256_loadu_ps(r1+8);
                    __m256 _r13 = _mm256_loadu_ps(r1+16);
                    __m256 _r21 = _mm256_loadu_ps(r2);
                    __m256 _r22 = _mm256_loadu_ps(r2+8);
                    __m256 _r23 = _mm256_loadu_ps(r2+16);

                    _sum00 = _mm256_fmadd_ps(_r01,_k00_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r02,_k01_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r03,_k02_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r11,_k10_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r12,_k11_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r13,_k12_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r21,_k20_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r22,_k21_0,_sum00);
                    _sum00 = _mm256_fmadd_ps(_r23,_k22_0,_sum00);
                  

                    _sum10 = _mm256_fmadd_ps(_r01,_k00_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r02,_k01_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r03,_k02_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r11,_k10_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r12,_k11_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r13,_k12_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r21,_k20_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r22,_k21_1,_sum10);
                    _sum10 = _mm256_fmadd_ps(_r23,_k22_1,_sum10);

                    _mm256_storeu_ps(outptr0, _sum00);
                    _mm256_storeu_ps(outptr1, _sum10);
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += 8;
                    outptr1 += 8;
                }

                r0 += 2 * 8;
                r1 += 2 * 8;
                r2 += 2 * 8;
            }

            k0 += 9 * 8;
            k1 += 9 * 8;
        }
    }
#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;
        out0.fill(bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            __m256 _k00 = _mm256_loadu_ps(k0);
            __m256 _k01 = _mm256_loadu_ps(k0 + 4);
            __m256 _k02 = _mm256_loadu_ps(k0 + 8);
            __m256 _k10 = _mm256_loadu_ps(k0 + 12);
            __m256 _k11 = _mm256_loadu_ps(k0 + 16);
            __m256 _k12 = _mm256_loadu_ps(k0 + 20);
            __m256 _k20 = _mm256_loadu_ps(k0 + 24);
            __m256 _k21 = _mm256_loadu_ps(k0 + 28);
            __m256 _k22 = _mm256_loadu_ps(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

#if __aarch64__
                for (; j + 7 < outw; j += 8)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n" // r04 r05 r06 r07

                        "fmul   v16.4s, %8.4s, v0.4s        \n"
                        "fmul   v17.4s, %8.4s, v1.4s        \n"
                        "fmul   v18.4s, %8.4s, v2.4s        \n"
                        "fmul   v19.4s, %8.4s, v3.4s        \n"
                        "fmul   v20.4s, %8.4s, v4.4s        \n"
                        "fmul   v21.4s, %8.4s, v5.4s        \n"
                        "fmul   v22.4s, %8.4s, v6.4s        \n"
                        "fmul   v23.4s, %8.4s, v7.4s        \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%1]        \n" // r08 r09

                        "fmla   v16.4s, %9.4s, v1.4s        \n"
                        "fmla   v17.4s, %9.4s, v2.4s        \n"
                        "fmla   v18.4s, %9.4s, v3.4s        \n"
                        "fmla   v19.4s, %9.4s, v4.4s        \n"
                        "fmla   v20.4s, %9.4s, v5.4s        \n"
                        "fmla   v21.4s, %9.4s, v6.4s        \n"
                        "fmla   v22.4s, %9.4s, v7.4s        \n"
                        "fmla   v23.4s, %9.4s, v8.4s        \n"

                        "fmla   v16.4s, %10.4s, v2.4s       \n"
                        "fmla   v17.4s, %10.4s, v3.4s       \n"
                        "fmla   v18.4s, %10.4s, v4.4s       \n"
                        "fmla   v19.4s, %10.4s, v5.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v20.4s, %10.4s, v6.4s       \n"
                        "fmla   v21.4s, %10.4s, v7.4s       \n"
                        "fmla   v22.4s, %10.4s, v8.4s       \n"
                        "fmla   v23.4s, %10.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r14 r15 r16 r17

                        "fmla   v16.4s, %11.4s, v0.4s       \n"
                        "fmla   v17.4s, %11.4s, v1.4s       \n"
                        "fmla   v18.4s, %11.4s, v2.4s       \n"
                        "fmla   v19.4s, %11.4s, v3.4s       \n"
                        "fmla   v20.4s, %11.4s, v4.4s       \n"
                        "fmla   v21.4s, %11.4s, v5.4s       \n"
                        "fmla   v22.4s, %11.4s, v6.4s       \n"
                        "fmla   v23.4s, %11.4s, v7.4s       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r18 r19

                        "fmla   v16.4s, %12.4s, v1.4s       \n"
                        "fmla   v17.4s, %12.4s, v2.4s       \n"
                        "fmla   v18.4s, %12.4s, v3.4s       \n"
                        "fmla   v19.4s, %12.4s, v4.4s       \n"
                        "fmla   v20.4s, %12.4s, v5.4s       \n"
                        "fmla   v21.4s, %12.4s, v6.4s       \n"
                        "fmla   v22.4s, %12.4s, v7.4s       \n"
                        "fmla   v23.4s, %12.4s, v8.4s       \n"

                        "fmla   v16.4s, %13.4s, v2.4s       \n"
                        "fmla   v17.4s, %13.4s, v3.4s       \n"
                        "fmla   v18.4s, %13.4s, v4.4s       \n"
                        "fmla   v19.4s, %13.4s, v5.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v20.4s, %13.4s, v6.4s       \n"
                        "fmla   v21.4s, %13.4s, v7.4s       \n"
                        "fmla   v22.4s, %13.4s, v8.4s       \n"
                        "fmla   v23.4s, %13.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r24 r25 r26 r27

                        "fmla   v16.4s, %14.4s, v0.4s       \n"
                        "fmla   v17.4s, %14.4s, v1.4s       \n"
                        "fmla   v18.4s, %14.4s, v2.4s       \n"
                        "fmla   v19.4s, %14.4s, v3.4s       \n"
                        "fmla   v20.4s, %14.4s, v4.4s       \n"
                        "fmla   v21.4s, %14.4s, v5.4s       \n"
                        "fmla   v22.4s, %14.4s, v6.4s       \n"
                        "fmla   v23.4s, %14.4s, v7.4s       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r28 r29

                        "fmla   v16.4s, %15.4s, v1.4s       \n"
                        "fmla   v17.4s, %15.4s, v2.4s       \n"
                        "fmla   v18.4s, %15.4s, v3.4s       \n"
                        "fmla   v19.4s, %15.4s, v4.4s       \n"
                        "fmla   v20.4s, %15.4s, v5.4s       \n"
                        "fmla   v21.4s, %15.4s, v6.4s       \n"
                        "fmla   v22.4s, %15.4s, v7.4s       \n"
                        "fmla   v23.4s, %15.4s, v8.4s       \n"

                        "fmla   v16.4s, %16.4s, v2.4s       \n"
                        "fmla   v17.4s, %16.4s, v3.4s       \n"
                        "fmla   v18.4s, %16.4s, v4.4s       \n"
                        "fmla   v19.4s, %16.4s, v5.4s       \n"
                        "fmla   v20.4s, %16.4s, v6.4s       \n"
                        "fmla   v21.4s, %16.4s, v7.4s       \n"
                        "fmla   v22.4s, %16.4s, v8.4s       \n"
                        "fmla   v23.4s, %16.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%0]        \n" // sum0 sum1 sum2 sum3 sum4 sum5 sum6 sum7

                        "faddp  v16.4s, v16.4s, v17.4s      \n"
                        "faddp  v18.4s, v18.4s, v19.4s      \n"
                        "faddp  v20.4s, v20.4s, v21.4s      \n"
                        "faddp  v22.4s, v22.4s, v23.4s      \n"

                        "faddp  v16.4s, v16.4s, v18.4s      \n"
                        "faddp  v20.4s, v20.4s, v22.4s      \n"

                        "fadd   v0.4s, v0.4s, v16.4s        \n"
                        "fadd   v1.4s, v1.4s, v20.4s        \n"

                        "st1    {v0.4s, v1.4s}, [%0], #32   \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                }
#endif // __aarch64__
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%1]        \n" // r04 r05

                        "fmul   v16.4s, %8.4s, v0.4s        \n"
                        "fmul   v17.4s, %8.4s, v1.4s        \n"
                        "fmul   v18.4s, %8.4s, v2.4s        \n"
                        "fmul   v19.4s, %8.4s, v3.4s        \n"

                        "fmla   v16.4s, %9.4s, v1.4s        \n"
                        "fmla   v17.4s, %9.4s, v2.4s        \n"
                        "fmla   v18.4s, %9.4s, v3.4s        \n"
                        "fmla   v19.4s, %9.4s, v8.4s        \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v16.4s, %10.4s, v2.4s       \n"
                        "fmla   v17.4s, %10.4s, v3.4s       \n"
                        "fmla   v18.4s, %10.4s, v8.4s       \n"
                        "fmla   v19.4s, %10.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r14 r15

                        "fmla   v16.4s, %11.4s, v4.4s       \n"
                        "fmla   v17.4s, %11.4s, v5.4s       \n"
                        "fmla   v18.4s, %11.4s, v6.4s       \n"
                        "fmla   v19.4s, %11.4s, v7.4s       \n"

                        "fmla   v16.4s, %12.4s, v5.4s       \n"
                        "fmla   v17.4s, %12.4s, v6.4s       \n"
                        "fmla   v18.4s, %12.4s, v7.4s       \n"
                        "fmla   v19.4s, %12.4s, v8.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v16.4s, %13.4s, v6.4s       \n"
                        "fmla   v17.4s, %13.4s, v7.4s       \n"
                        "fmla   v18.4s, %13.4s, v8.4s       \n"
                        "fmla   v19.4s, %13.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r24 r25

                        "fmla   v16.4s, %14.4s, v0.4s       \n"
                        "fmla   v17.4s, %14.4s, v1.4s       \n"
                        "fmla   v18.4s, %14.4s, v2.4s       \n"
                        "fmla   v19.4s, %14.4s, v3.4s       \n"

                        "fmla   v16.4s, %15.4s, v1.4s       \n"
                        "fmla   v17.4s, %15.4s, v2.4s       \n"
                        "fmla   v18.4s, %15.4s, v3.4s       \n"
                        "fmla   v19.4s, %15.4s, v8.4s       \n"

                        "fmla   v16.4s, %16.4s, v2.4s       \n"
                        "fmla   v17.4s, %16.4s, v3.4s       \n"
                        "fmla   v18.4s, %16.4s, v8.4s       \n"
                        "fmla   v19.4s, %16.4s, v9.4s       \n"

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v0.4s}, [%0]               \n" // sum0 sum1 sum2 sum3

                        "faddp  v16.4s, v16.4s, v17.4s      \n"
                        "faddp  v18.4s, v18.4s, v19.4s      \n"

                        "faddp  v16.4s, v16.4s, v18.4s      \n"

                        "fadd   v0.4s, v0.4s, v16.4s        \n"

                        "st1    {v0.4s}, [%0], #16          \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r00 r01

                        "vmul.f32   q3, %q8, q0     \n"

                        "pld        [%1, #128]      \n"
                        "vld1.f32   {d4-d5}, [%1 :128]! \n" // r02

                        "vmul.f32   q4, %q8, q1     \n"
                        "vmla.f32   q3, %q9, q1     \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r03 r04

                        "vmul.f32   q5, %q8, q2     \n"
                        "vmla.f32   q4, %q9, q2     \n"
                        "vmla.f32   q3, %q10, q2    \n"

                        "vmul.f32   q6, %q8, q0     \n"
                        "vmla.f32   q5, %q9, q0     \n"
                        "vmla.f32   q4, %q10, q0    \n"

                        "pld        [%1, #128]      \n"
                        "vld1.f32   {d4-d5}, [%1 :128] \n" // r05

                        "vmla.f32   q6, %q9, q1     \n"
                        "vmla.f32   q5, %q10, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q6, %q10, q2    \n"

                        "vmla.f32   q3, %q11, q0    \n"

                        "pld        [%2, #128]      \n"
                        "vld1.f32   {d4-d5}, [%2 :128]! \n" // r12

                        "vmla.f32   q4, %q11, q1    \n"
                        "vmla.f32   q3, %q12, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r13 r14

                        "vmla.f32   q5, %q11, q2    \n"
                        "vmla.f32   q4, %q12, q2    \n"
                        "vmla.f32   q3, %q13, q2    \n"

                        "vmla.f32   q6, %q11, q0    \n"
                        "vmla.f32   q5, %q12, q0    \n"
                        "vmla.f32   q4, %q13, q0    \n"

                        "pld        [%2, #128]      \n"
                        "vld1.f32   {d4-d5}, [%2 :128] \n" // r15

                        "vmla.f32   q6, %q12, q1    \n"
                        "vmla.f32   q5, %q13, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q6, %q13, q2    \n"

                        "vmla.f32   q3, %q14, q0    \n"

                        "pld        [%3, #128]      \n"
                        "vld1.f32   {d4-d5}, [%3 :128]! \n" // r22

                        "vmla.f32   q4, %q14, q1    \n"
                        "vmla.f32   q3, %q15, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r23 r24

                        "vmla.f32   q5, %q14, q2    \n"
                        "vmla.f32   q4, %q15, q2    \n"
                        "vmla.f32   q3, %q16, q2    \n"

                        "vmla.f32   q6, %q14, q0    \n"
                        "vmla.f32   q5, %q15, q0    \n"
                        "vmla.f32   q4, %q16, q0    \n"

                        "pld        [%3, #128]      \n"
                        "vld1.f32   {d4-d5}, [%3 :128] \n" // r25

                        "vmla.f32   q6, %q15, q1    \n"
                        "vmla.f32   q5, %q16, q1    \n"

                        "vld1.f32   {d0-d1}, [%0]   \n" // sum0 sum1 sum2 sum3

                        "vmla.f32   q6, %q16, q2    \n"

                        "vadd.f32   d6, d6, d7      \n"
                        "vadd.f32   d8, d8, d9      \n"
                        "vadd.f32   d10, d10, d11   \n"
                        "vadd.f32   d12, d12, d13   \n"

                        "sub        %1, %1, #16     \n"

                        "vpadd.f32  d6, d6, d8      \n"
                        "vpadd.f32  d7, d10, d12    \n"

                        "sub        %2, %2, #16     \n"

                        "vadd.f32   q0, q0, q3      \n"

                        "sub        %3, %3, #16     \n"

                        "vst1.f32   {d0-d1}, [%0]!  \n"

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
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1] \n" // r00 r01 r02 r03

                        "fmul   v16.4s, %8.4s, v0.4s        \n"
                        "fmul   v17.4s, %8.4s, v1.4s        \n"
                        "fmul   v18.4s, %9.4s, v1.4s        \n"
                        "fmul   v19.4s, %9.4s, v2.4s        \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2] \n" // r10 r11 r12 r13

                        "fmla   v16.4s, %10.4s, v2.4s       \n"
                        "fmla   v17.4s, %10.4s, v3.4s       \n"

                        "fmla   v18.4s, %11.4s, v4.4s       \n"
                        "fmla   v19.4s, %11.4s, v5.4s       \n"
                        "fmla   v16.4s, %12.4s, v5.4s       \n"
                        "fmla   v17.4s, %12.4s, v6.4s       \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3] \n" // r20 r21 r22 r23

                        "fmla   v18.4s, %13.4s, v6.4s       \n"
                        "fmla   v19.4s, %13.4s, v7.4s       \n"

                        "fmla   v16.4s, %14.4s, v0.4s       \n"
                        "fmla   v17.4s, %14.4s, v1.4s       \n"
                        "fmla   v18.4s, %15.4s, v1.4s       \n"
                        "fmla   v19.4s, %15.4s, v2.4s       \n"
                        "fmla   v16.4s, %16.4s, v2.4s       \n"
                        "fmla   v17.4s, %16.4s, v3.4s       \n"

                        "ld1    {v0.2s}, [%0]               \n" // sum0 sum1

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "add    %1, %1, #32                 \n"

                        "faddp  v16.4s, v16.4s, v17.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "faddp  v16.4s, v16.4s, v16.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v0.2s, v0.2s, v16.2s        \n"

                        "st1    {v0.2s}, [%0], #8           \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r00 r01

                        "vmul.f32   q5, %q8, q0     \n"
                        "vmul.f32   q6, %q8, q1     \n"
                        "vmul.f32   q2, %q9, q1     \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1 :128] \n" // r02 r03

                        "vmul.f32   q3, %q9, q0     \n"
                        "vmla.f32   q5, %q10, q0    \n"
                        "vmla.f32   q6, %q10, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q2, %q11, q0    \n"
                        "vmla.f32   q3, %q11, q1    \n"
                        "vmla.f32   q5, %q12, q1    \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2 :128] \n" // r12 r13

                        "vmla.f32   q6, %q12, q0    \n"
                        "vmla.f32   q2, %q13, q0    \n"
                        "vmla.f32   q3, %q13, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q5, %q14, q0    \n"
                        "vmla.f32   q6, %q14, q1    \n"
                        "vmla.f32   q2, %q15, q1    \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3 :128] \n" // r22 r23

                        "vmla.f32   q3, %q15, q0    \n"
                        "vmla.f32   q5, %q16, q0    \n"
                        "vmla.f32   q6, %q16, q1    \n"

                        "vld1.f32   {d8}, [%0]      \n" // sum0 sum1

                        "vadd.f32   q5, q5, q2      \n"
                        "vadd.f32   q6, q6, q3      \n"

                        "vadd.f32   d10, d10, d11   \n"
                        "vadd.f32   d12, d12, d13   \n"

                        "vpadd.f32  d10, d10, d12   \n"

                        "vadd.f32   d8, d8, d10     \n"

                        "vst1.f32   {d8}, [%0]!     \n"

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
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%1] \n" // r00 r01 r02

                        "eor    v16.16b, v16.16b, v16.16b   \n"
                        "ld1    {v16.s}[0], [%0]            \n" // sum0

                        "fmul   v17.4s, %8.4s, v0.4s        \n"
                        "fmul   v18.4s, %9.4s, v1.4s        \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.4s, v4.4s, v5.4s}, [%2] \n" // r10 r11 r12

                        "fmla   v16.4s, %10.4s, v2.4s       \n"

                        "fmla   v17.4s, %11.4s, v3.4s       \n"
                        "fmla   v18.4s, %12.4s, v4.4s       \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%3] \n" // r20 r21 r22

                        "fmla   v16.4s, %13.4s, v5.4s       \n"

                        "fmla   v17.4s, %14.4s, v0.4s       \n"
                        "fmla   v18.4s, %15.4s, v1.4s       \n"
                        "fmla   v16.4s, %16.4s, v2.4s       \n"

                        "fadd   v17.4s, v17.4s, v18.4s      \n"
                        "fadd   v16.4s, v16.4s, v17.4s      \n"

                        "add    %1, %1, #16                 \n"

                        "faddp  v16.4s, v16.4s, v16.4s      \n"

                        "add    %2, %2, #16                 \n"

                        "faddp  v16.2s, v16.2s, v16.2s      \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v16.s}[0], [%0], #4        \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #384]      \n"
                        "vldm       %1, {d0-d5}     \n" // r00 r01 r02

                        "veor       q3, q3          \n"
                        "vld1.f32   {d6[0]}, [%0]   \n" // sum0

                        "vmul.f32   q4, %q8, q0     \n"
                        "vmul.f32   q5, %q9, q1     \n"
                        "vmla.f32   q3, %q10, q2    \n"

                        "pld        [%2, #384]      \n"
                        "vldm       %2, {d0-d5}     \n" // r10 r11 r12

                        "vmla.f32   q4, %q11, q0    \n"
                        "vmla.f32   q5, %q12, q1    \n"
                        "vmla.f32   q3, %q13, q2    \n"

                        "pld        [%3, #384]      \n"
                        "vldm       %3, {d0-d5}     \n" // r20 r21 r22

                        "vmla.f32   q4, %q14, q0    \n"
                        "vmla.f32   q5, %q15, q1    \n"
                        "vmla.f32   q3, %q16, q2    \n"

                        "vadd.f32   q4, q4, q5      \n"
                        "vadd.f32   q3, q3, q4      \n"

                        "add        %1, %1, #16     \n"

                        "vadd.f32   d6, d6, d7      \n"

                        "add        %2, %2, #16     \n"

                        "vpadd.f32  d6, d6, d6      \n"

                        "add        %3, %3, #16     \n"

                        "vst1.f32   {d6[0]}, [%0]!  \n"

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
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5");
#endif // __aarch64__
                }

                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
            }

            k0 += 9 * 4;
        }
    }
}
