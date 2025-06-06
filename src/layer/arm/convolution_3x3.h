// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

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

        const float* k0 = kernel + p * inch * 9;
        const float* k1 = kernel + (p + 1) * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr0n = outptr0 + outw;
            float* outptr1n = outptr1 + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

#if __ARM_NEON
            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0 + 3);
            float32x4_t _k06 = vld1q_f32(k0 + 6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1 + 3);
            float32x4_t _k16 = vld1q_f32(k1 + 6);
#endif // __ARM_NEON

            int i = 0;

            for (; i + 1 < outh; i += 2)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                        "add    %5, %5, #16                 \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                        "add    %8, %8, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" // _sum0

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n" // _sum1

                        "fmla   v6.4s, v8.4s, %18.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %21.s[0]      \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v12.4s}, [%3]              \n" // _sum0n

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v13.4s}, [%4]              \n" // _sum1n

                        "fmla   v12.4s, v14.4s, %20.s[0]    \n"
                        "fmla   v13.4s, v14.4s, %23.s[0]    \n"

                        "ext    v8.16b, v8.16b, v9.16b, #8  \n"
                        "ext    v9.16b, v14.16b, v15.16b, #4 \n"

                        "fmla   v6.4s, v10.4s, %18.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %21.s[1]     \n"
                        "fmla   v12.4s, v11.4s, %20.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %23.s[2]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%6]      \n" // r1
                        "add    %6, %6, #16                 \n"

                        "fmla   v6.4s, v8.4s, %18.s[2]      \n"
                        "fmla   v7.4s, v8.4s, %21.s[2]      \n"
                        "fmla   v12.4s, v9.4s, %20.s[1]     \n"
                        "fmla   v13.4s, v9.4s, %23.s[1]     \n"

                        "ext    v10.16b, v14.16b, v15.16b, #4 \n"

                        "fmla   v6.4s, v14.4s, %19.s[0]     \n"
                        "fmla   v7.4s, v14.4s, %22.s[0]     \n"
                        "fmla   v12.4s, v14.4s, %18.s[0]    \n"
                        "fmla   v13.4s, v14.4s, %21.s[0]    \n"

                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %19.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %22.s[1]     \n"
                        "fmla   v12.4s, v10.4s, %18.s[1]    \n"
                        "fmla   v13.4s, v10.4s, %21.s[1]    \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%7]        \n" // r2
                        "add    %7, %7, #16                 \n"

                        "fmla   v6.4s, v11.4s, %19.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %22.s[2]     \n"
                        "fmla   v12.4s, v11.4s, %18.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %21.s[2]    \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                        "fmla   v6.4s, v8.4s, %20.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %23.s[0]      \n"
                        "fmla   v12.4s, v8.4s, %19.s[0]     \n"
                        "fmla   v13.4s, v8.4s, %22.s[0]     \n"

                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %20.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %23.s[1]     \n"
                        "fmla   v12.4s, v10.4s, %19.s[1]    \n"
                        "fmla   v13.4s, v10.4s, %22.s[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                        "add    %5, %5, #16                 \n"

                        "fmla   v6.4s, v11.4s, %20.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %23.s[2]     \n"
                        "fmla   v12.4s, v11.4s, %19.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %22.s[2]    \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                        "add    %8, %8, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "st1    {v12.4s}, [%3], #16         \n"
                        "st1    {v13.4s}, [%4], #16         \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %5, %5, #16                 \n"
                        "sub    %8, %8, #16                 \n"
                        : "=r"(nn),       // %0
                        "=r"(outptr0),  // %1
                        "=r"(outptr1),  // %2
                        "=r"(outptr0n), // %3
                        "=r"(outptr1n), // %4
                        "=r"(r0),       // %5
                        "=r"(r1),       // %6
                        "=r"(r2),       // %7
                        "=r"(r3)        // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr0n),
                        "4"(outptr1n),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k00), // %18
                        "w"(_k03), // %19
                        "w"(_k06), // %20
                        "w"(_k10), // %21
                        "w"(_k13), // %22
                        "w"(_k16)  // %23
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5 :64] \n" // r0
                        "add        %5, #16             \n"

                        "pld        [%8, #192]          \n"
                        "vld1.f32   {d28-d30}, [%8]     \n" // r3
                        "add        %8, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q14, q15, #2   \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1 :64] \n" // _sum0

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2 :64] \n" // _sum1

                        "vmla.f32   q6, q8, %e18[0]     \n"
                        "vmla.f32   q7, q8, %e21[0]     \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d24-d25}, [%3]     \n" // _sum0n

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d26-d27}, [%4]     \n" // _sum1n

                        "vmla.f32   q12, q14, %e20[0]   \n"
                        "vmla.f32   q13, q14, %e23[0]   \n"

                        "vext.32    q8, q8, q9, #2      \n"
                        "vext.32    q9, q14, q15, #1    \n"

                        "vmla.f32   q6, q10, %e18[1]    \n"
                        "vmla.f32   q7, q10, %e21[1]    \n"
                        "vmla.f32   q12, q11, %f20[0]   \n"
                        "vmla.f32   q13, q11, %f23[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d28-d30}, [%6]     \n" // r1
                        "add        %6, #16             \n"

                        "vmla.f32   q6, q8, %f18[0]     \n"
                        "vmla.f32   q7, q8, %f21[0]     \n"
                        "vmla.f32   q12, q9, %e20[1]    \n"
                        "vmla.f32   q13, q9, %e23[1]    \n"

                        "vext.32    q10, q14, q15, #1   \n"

                        "vmla.f32   q6, q14, %e19[0]    \n"
                        "vmla.f32   q7, q14, %e22[0]    \n"
                        "vmla.f32   q12, q14, %e18[0]   \n"
                        "vmla.f32   q13, q14, %e21[0]   \n"

                        "vext.32    q11, q14, q15, #2   \n"

                        "vmla.f32   q6, q10, %e19[1]    \n"
                        "vmla.f32   q7, q10, %e22[1]    \n"
                        "vmla.f32   q12, q10, %e18[1]   \n"
                        "vmla.f32   q13, q10, %e21[1]   \n"

                        "pld        [%7, #192]          \n"
                        "vld1.f32   {d16-d18}, [%7 :64] \n" // r2
                        "add        %7, #16             \n"

                        "vmla.f32   q6, q11, %f19[0]    \n"
                        "vmla.f32   q7, q11, %f22[0]    \n"
                        "vmla.f32   q12, q11, %f18[0]   \n"
                        "vmla.f32   q13, q11, %f21[0]   \n"

                        "vext.32    q10, q8, q9, #1     \n"

                        "vmla.f32   q6, q8, %e20[0]     \n"
                        "vmla.f32   q7, q8, %e23[0]     \n"
                        "vmla.f32   q12, q8, %e19[0]    \n"
                        "vmla.f32   q13, q8, %e22[0]    \n"

                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e20[1]    \n"
                        "vmla.f32   q7, q10, %e23[1]    \n"
                        "vmla.f32   q12, q10, %e19[1]   \n"
                        "vmla.f32   q13, q10, %e22[1]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5 :64] \n" // r0
                        "add        %5, #16             \n"

                        "vmla.f32   q6, q11, %f20[0]    \n"
                        "vmla.f32   q7, q11, %f23[0]    \n"
                        "vmla.f32   q12, q11, %f19[0]   \n"
                        "vmla.f32   q13, q11, %f22[0]   \n"

                        "pld        [%8, #192]          \n"
                        "vld1.f32   {d28-d30}, [%8]     \n" // r3
                        "add        %8, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"

                        "vst1.f32   {d12-d13}, [%1 : 64]!\n"
                        "vst1.f32   {d14-d15}, [%2 : 64]!\n"

                        "vext.32    q11, q14, q15, #2   \n"

                        "vst1.f32   {d24-d25}, [%3]!    \n"
                        "vst1.f32   {d26-d27}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %5, #16             \n"
                        "sub        %8, #16             \n"
                        : "=r"(nn),       // %0
                        "=r"(outptr0),  // %1
                        "=r"(outptr1),  // %2
                        "=r"(outptr0n), // %3
                        "=r"(outptr1n), // %4
                        "=r"(r0),       // %5
                        "=r"(r1),       // %6
                        "=r"(r2),       // %7
                        "=r"(r3)        // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr0n),
                        "4"(outptr1n),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k00), // %18
                        "w"(_k03), // %19
                        "w"(_k06), // %20
                        "w"(_k10), // %21
                        "w"(_k13), // %22
                        "w"(_k16)  // %23
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    float32x4_t _sum0n = vmulq_f32(_r10, _k00);
                    float32x4_t _sum1n = vmulq_f32(_r10, _k10);
                    _sum0n = vmlaq_f32(_sum0n, _r20, _k03);
                    _sum1n = vmlaq_f32(_sum1n, _r20, _k13);
                    _sum0n = vmlaq_f32(_sum0n, _r30, _k06);
                    _sum1n = vmlaq_f32(_sum1n, _r30, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
                    _sum0n = vsetq_lane_f32(*outptr0n, _sum0n, 3);
                    _sum1n = vsetq_lane_f32(*outptr1n, _sum1n, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
                    *outptr0n = vaddvq_f32(_sum0n);
                    *outptr1n = vaddvq_f32(_sum1n);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss0n = vadd_f32(vget_low_f32(_sum0n), vget_high_f32(_sum0n));
                    float32x2_t _ss1n = vadd_f32(vget_low_f32(_sum1n), vget_high_f32(_sum1n));

                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);
                    float32x2_t _ss01n = vpadd_f32(_ss0n, _ss1n);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
                    *outptr0n = vget_lane_f32(_ss01n, 0);
                    *outptr1n = vget_lane_f32(_ss01n, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum0n = 0.f;
                    float sum1 = 0.f;
                    float sum1n = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    sum0n += r1[0] * k0[0];
                    sum0n += r1[1] * k0[1];
                    sum0n += r1[2] * k0[2];
                    sum0n += r2[0] * k0[3];
                    sum0n += r2[1] * k0[4];
                    sum0n += r2[2] * k0[5];
                    sum0n += r3[0] * k0[6];
                    sum0n += r3[1] * k0[7];
                    sum0n += r3[2] * k0[8];

                    sum1n += r1[0] * k1[0];
                    sum1n += r1[1] * k1[1];
                    sum1n += r1[2] * k1[2];
                    sum1n += r2[0] * k1[3];
                    sum1n += r2[1] * k1[4];
                    sum1n += r2[2] * k1[5];
                    sum1n += r3[0] * k1[6];
                    sum1n += r3[1] * k1[7];
                    sum1n += r3[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr0n += sum0n;
                    *outptr1n += sum1n;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr0++;
                    outptr1++;
                    outptr0n++;
                    outptr1n++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr1 += outw;
                outptr0n += outw;
                outptr1n += outw;
            }

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r0
                        "add    %3, %3, #16                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" // _sum0

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n" // _sum1

                        "fmul   v14.4s, v8.4s, %12.s[0]     \n"
                        "fmul   v15.4s, v8.4s, %15.s[0]     \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %12.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %15.s[1]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%4]        \n" // r1
                        "add    %4, %4, #16                 \n"

                        "fmla   v14.4s, v11.4s, %12.s[2]    \n"
                        "fmla   v15.4s, v11.4s, %15.s[2]    \n"

                        "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v14.4s, v10.4s, %13.s[1]    \n"
                        "fmla   v15.4s, v10.4s, %16.s[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r2
                        "add    %5, %5, #16                 \n"

                        "fmla   v6.4s, v11.4s, %13.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %16.s[2]     \n"

                        "fmla   v14.4s, v8.4s, %14.s[0]     \n"
                        "fmla   v15.4s, v8.4s, %17.s[0]     \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %14.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %17.s[1]     \n"

                        "fmla   v14.4s, v11.4s, %14.s[2]    \n"
                        "fmla   v15.4s, v11.4s, %17.s[2]    \n"

                        "fadd   v6.4s, v6.4s, v14.4s        \n"
                        "fadd   v7.4s, v7.4s, v15.4s        \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "subs   %w0, %w0, #1                \n"
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
                        "w"(_k00), // %12
                        "w"(_k03), // %13
                        "w"(_k06), // %14
                        "w"(_k10), // %15
                        "w"(_k13), // %16
                        "w"(_k16)  // %17
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n" // r0
                        "add        %3, #16             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1]     \n" // _sum0

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2]     \n" // _sum1

                        "vmul.f32   q14, q8, %e12[0]    \n"
                        "vmul.f32   q15, q8, %e15[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e12[1]    \n"
                        "vmla.f32   q7, q10, %e15[1]    \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n" // r1
                        "add        %4, #16             \n"

                        "vmla.f32   q14, q11, %f12[0]   \n"
                        "vmla.f32   q15, q11, %f15[0]   \n"

                        "vmla.f32   q6, q8, %e13[0]     \n"
                        "vmla.f32   q7, q8, %e16[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q14, q10, %e13[1]   \n"
                        "vmla.f32   q15, q10, %e16[1]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5]     \n" // r2
                        "add        %5, #16             \n"

                        "vmla.f32   q6, q11, %f13[0]    \n"
                        "vmla.f32   q7, q11, %f16[0]    \n"

                        "vmla.f32   q14, q8, %e14[0]    \n"
                        "vmla.f32   q15, q8, %e17[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e14[1]    \n"
                        "vmla.f32   q7, q10, %e17[1]    \n"

                        "vmla.f32   q14, q11, %f14[0]   \n"
                        "vmla.f32   q15, q11, %f17[0]   \n"

                        "vadd.f32   q6, q6, q14         \n"
                        "vadd.f32   q7, q7, q15         \n"

                        "vst1.f32   {d12-d13}, [%1]!    \n"

                        "vst1.f32   {d14-d15}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

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
                        "w"(_k00), // %12
                        "w"(_k03), // %13
                        "w"(_k06), // %14
                        "w"(_k10), // %15
                        "w"(_k13), // %16
                        "w"(_k16)  // %17
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                    outptr1++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* kernel0 = kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k3456 = vld1q_f32(kernel0 + 3);
            float32x4_t _k6789 = vld1q_f32(kernel0 + 6);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

            int i = 0;

            for (; i + 1 < outh; i += 2)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0
                        "add    %3, %3, #16                 \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v7.4s}, [%1]               \n" // _sum

                        "fmla   v7.4s, v9.4s, %14.s[0]      \n"
                        "fmul   v6.4s, v11.4s, %14.s[1]     \n"
                        "fmul   v13.4s, v12.4s, %14.s[2]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%4]       \n" // r1
                        "add    %4, %4, #16                 \n"

                        "fmla   v7.4s, v9.4s, %15.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v6.4s, v11.4s, %15.s[1]     \n"
                        "fmla   v13.4s, v12.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v8.4s}, [%2]               \n" // _sum2

                        "fmla   v8.4s, v9.4s, %14.s[0]      \n"
                        "fmul   v14.4s, v11.4s, %14.s[1]    \n"
                        "fmul   v15.4s, v12.4s, %14.s[2]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%5]       \n" // r2
                        "add    %5, %5, #16                 \n"

                        "fmla   v7.4s, v9.4s, %16.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v6.4s, v11.4s, %16.s[1]     \n"
                        "fmla   v13.4s, v12.4s, %16.s[2]    \n"

                        "fmla   v8.4s, v9.4s, %15.s[0]      \n"
                        "fmla   v14.4s, v11.4s, %15.s[1]    \n"
                        "fmla   v15.4s, v12.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%6]       \n" // r3
                        "add    %6, %6, #16                 \n"

                        "fmla   v8.4s, v9.4s, %16.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v14.4s, v11.4s, %16.s[1]    \n"
                        "fmla   v15.4s, v12.4s, %16.s[2]    \n"

                        "fadd   v7.4s, v7.4s, v6.4s         \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0

                        "fadd   v8.4s, v8.4s, v14.4s        \n"
                        "fadd   v7.4s, v7.4s, v13.4s        \n"
                        "fadd   v8.4s, v8.4s, v15.4s        \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v7.4s}, [%1], #16          \n"
                        "st1    {v8.4s}, [%2], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %3, %3, #16                 \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr),  // %1
                        "=r"(outptr2), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2),      // %5
                        "=r"(r3)       // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "w"(_k0123), // %14
                        "w"(_k3456), // %15
                        "w"(_k6789)  // %16
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n" // r0
                        "add        %3, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1 :64] \n" // _sum

                        "vmla.f32   q7, q9, %e14[0]     \n"
                        "vmul.f32   q6, q11, %e14[1]    \n"
                        "vmul.f32   q13, q12, %f14[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d18-d20}, [%4]     \n" // r1
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q9, %e15[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e15[1]    \n"
                        "vmla.f32   q13, q12, %f15[0]   \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d16-d17}, [%2]     \n" // _sum2

                        "vmla.f32   q8, q9, %e14[0]     \n"
                        "vmul.f32   q14, q11, %e14[1]   \n"
                        "vmul.f32   q15, q12, %f14[0]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d18-d20}, [%5 :64] \n" // r2
                        "add        %5, #16             \n"

                        "vmla.f32   q7, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e16[1]    \n"
                        "vmla.f32   q13, q12, %f16[0]   \n"

                        "vmla.f32   q8, q9, %e15[0]     \n"
                        "vmla.f32   q14, q11, %e15[1]   \n"
                        "vmla.f32   q15, q12, %f15[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d18-d20}, [%6]     \n" // r3
                        "add        %6, #16             \n"

                        "vmla.f32   q8, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q14, q11, %e16[1]   \n"
                        "vmla.f32   q15, q12, %f16[0]   \n"

                        "vadd.f32   q7, q7, q6          \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n" // r0

                        "vadd.f32   q8, q8, q14         \n"
                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q8, q8, q15         \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "add        %3, #16             \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"
                        "vst1.f32   {d16-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %3, #16             \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr),  // %1
                        "=r"(outptr2), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2),      // %5
                        "=r"(r3)       // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "w"(_k0123), // %14
                        "w"(_k3456), // %15
                        "w"(_k6789)  // %16
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    float32x4_t _sum2 = vmulq_f32(_r10, _k0123);
                    _sum2 = vmlaq_f32(_sum2, _r20, _k3456);
                    _sum2 = vmlaq_f32(_sum2, _r30, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);
                    _sum2 = vsetq_lane_f32(*outptr2, _sum2, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
                    *outptr2 = vaddvq_f32(_sum2);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));

                    float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

                    *outptr = vget_lane_f32(_sss2, 0);
                    *outptr2 = vget_lane_f32(_sss2, 1);
#endif // __aarch64__
#else
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                        "add    %2, %2, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v7.4s}, [%1]               \n" // _sum

                        "fmla   v7.4s, v8.4s, %10.s[0]      \n"
                        "fmul   v13.4s, v10.4s, %10.s[1]    \n"
                        "fmul   v14.4s, v11.4s, %10.s[2]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r1
                        "add    %3, %3, #16                 \n"

                        "fmla   v7.4s, v8.4s, %11.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v13.4s, v10.4s, %11.s[1]    \n"
                        "fmla   v14.4s, v11.4s, %11.s[2]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%4]        \n" // r2
                        "add    %4, %4, #16                 \n"

                        "fmla   v7.4s, v8.4s, %12.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v13.4s, v10.4s, %12.s[1]    \n"
                        "fmla   v14.4s, v11.4s, %12.s[2]    \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                        "add    %2, %2, #16                 \n"

                        "fadd   v7.4s, v7.4s, v13.4s        \n"
                        "fadd   v7.4s, v7.4s, v14.4s        \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "st1    {v7.4s}, [%1], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %2, %2, #16                 \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123), // %10
                        "w"(_k3456), // %11
                        "w"(_k6789)  // %12
                        : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" // r0
                        "add        %2, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1]     \n" // _sum

                        "vmla.f32   q7, q8, %e10[0]     \n"
                        "vmul.f32   q13, q10, %e10[1]   \n"
                        "vmul.f32   q14, q11, %f10[0]   \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n" // r1
                        "add        %3, #16             \n"

                        "vmla.f32   q7, q8, %e11[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e11[1]   \n"
                        "vmla.f32   q14, q11, %f11[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n" // r2
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q8, %e12[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e12[1]   \n"
                        "vmla.f32   q14, q11, %f12[0]   \n"

                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" // r0
                        "add        %2, #16             \n"

                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q7, q7, q14         \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #16             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123), // %10
                        "w"(_k3456), // %11
                        "w"(_k6789)  // %12
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif
                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
}

static void conv3x3s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

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

        const float* k0 = kernel + p * inch * 9;
        const float* k1 = kernel + (p + 1) * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

#if __ARM_NEON
            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0 + 3);
            float32x4_t _k06 = vld1q_f32(k0 + 6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1 + 3);
            float32x4_t _k16 = vld1q_f32(k1 + 6);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%3], #32   \n" // v8 v9 = r0

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" // v6 = _sum0

                        "fmul   v12.4s, v8.4s, %12.s[0]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n" // v7 = _sum1

                        "fmul   v13.4s, v8.4s, %15.s[0]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld2    {v10.4s, v11.4s}, [%3]      \n" // v10

                        "fmla   v6.4s, v9.4s, %12.s[1]      \n"

                        "ext    v14.16b, v8.16b, v10.16b, #4\n"

                        "fmla   v7.4s, v9.4s, %15.s[1]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%4], #32   \n" // r1

                        "fmla   v12.4s, v14.4s, %12.s[2]    \n"
                        "fmla   v13.4s, v14.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld2    {v10.4s, v11.4s}, [%4]      \n"

                        "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                        "ext    v14.16b, v8.16b, v10.16b, #4\n"

                        "fmla   v12.4s, v9.4s, %13.s[1]     \n"
                        "fmla   v13.4s, v9.4s, %16.s[1]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%5], #32   \n" // r2

                        "fmla   v6.4s, v14.4s, %13.s[2]     \n"
                        "fmla   v7.4s, v14.4s, %16.s[2]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld2    {v10.4s, v11.4s}, [%5]      \n"

                        "fmla   v12.4s, v8.4s, %14.s[0]     \n"
                        "fmla   v13.4s, v8.4s, %17.s[0]     \n"

                        "ext    v14.16b, v8.16b, v10.16b, #4\n"

                        "fmla   v6.4s, v9.4s, %14.s[1]      \n"
                        "fmla   v7.4s, v9.4s, %17.s[1]      \n"

                        "fmla   v12.4s, v14.4s, %14.s[2]    \n"
                        "fmla   v13.4s, v14.4s, %17.s[2]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%3], #32   \n" // v8 v9 = r0

                        "fadd   v6.4s, v6.4s, v12.4s        \n"
                        "fadd   v7.4s, v7.4s, v13.4s        \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "bne    0b                          \n"
                        "sub    %3, %3, #32                 \n"

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
                        "w"(_k00), // %12
                        "w"(_k03), // %13
                        "w"(_k06), // %14
                        "w"(_k10), // %15
                        "w"(_k13), // %16
                        "w"(_k16)  // %17
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d16-d19}, [%3]!    \n" // q8 q9 = r0

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1]     \n" // q6 = _sum0

                        "vmul.f32   q12, q8, %e12[0]    \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2]     \n" // q7 = _sum1

                        "vmul.f32   q13, q8, %e15[0]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d20-d21}, [%3]     \n" // q10

                        "vmla.f32   q6, q9, %e12[1]     \n"

                        "vext.32    q11, q8, q10, #1    \n"

                        "vmla.f32   q7, q9, %e15[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d16-d19}, [%4]!    \n" // r1

                        "vmla.f32   q12, q11, %f12[0]   \n"
                        "vmla.f32   q13, q11, %f15[0]   \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d20-d21}, [%4]     \n"

                        "vmla.f32   q6, q8, %e13[0]     \n"
                        "vmla.f32   q7, q8, %e16[0]     \n"

                        "vext.32    q11, q8, q10, #1    \n"

                        "vmla.f32   q12, q9, %e13[1]    \n"
                        "vmla.f32   q13, q9, %e16[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld2.f32   {d16-d19}, [%5]!    \n" // r2

                        "vmla.f32   q6, q11, %f13[0]    \n"
                        "vmla.f32   q7, q11, %f16[0]    \n"

                        "pld        [%5, #128]          \n"
                        "vld2.f32   {d20-d21}, [%5]     \n"

                        "vmla.f32   q12, q8, %e14[0]    \n"
                        "vmla.f32   q13, q8, %e17[0]    \n"

                        "vext.32    q11, q8, q10, #1    \n"

                        "vmla.f32   q6, q9, %e14[1]     \n"
                        "vmla.f32   q7, q9, %e17[1]     \n"

                        "vmla.f32   q12, q11, %f14[0]   \n"
                        "vmla.f32   q13, q11, %f17[0]   \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d16-d19}, [%3]!    \n" // q8 q9 = r0

                        "vadd.f32   q6, q6, q12         \n"
                        "vadd.f32   q7, q7, q13         \n"

                        "subs       %0, #1              \n"

                        "vst1.f32   {d12-d13}, [%1]!    \n"
                        "vst1.f32   {d14-d15}, [%2]!    \n"

                        "bne        0b                  \n"
                        "sub        %3, #32             \n"

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
                        "w"(_k00), // %12
                        "w"(_k03), // %13
                        "w"(_k06), // %14
                        "w"(_k10), // %15
                        "w"(_k13), // %16
                        "w"(_k16)  // %17
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* kernel0 = kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "0:                                        \n"

                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v0.4s}, [%1]                  \n"

                        "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                        "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%2]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                        "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%3]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                        "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%4]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                        "fadd       v0.4s, v0.4s, v10.4s           \n"
                        "fadd       v0.4s, v0.4s, v11.4s           \n"

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s}, [%1], #16             \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #32                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123), // %10
                        "w"(_k3456), // %11
                        "w"(_k6789)  // %12
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1]       \n"

                        "vmla.f32   q0, q2, %e10[0]     \n"
                        "vmul.f32   q10, q3, %e10[1]    \n"

                        "pld        [%2, #128]          \n"
                        "vld2.f32   {d16-d17}, [%2]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmul.f32   q11, q1, %f10[0]    \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d4-d7}, [%3]!      \n"

                        "vmla.f32   q0, q2, %e11[0]     \n"
                        "vmla.f32   q10, q3, %e11[1]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d16-d17}, [%3]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f11[0]    \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d4-d7}, [%4]!      \n"

                        "vmla.f32   q0, q2, %e12[0]     \n"
                        "vmla.f32   q10, q3, %e12[1]    \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d16-d17}, [%4]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f12[0]    \n"

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "vadd.f32   q0, q0, q10         \n"
                        "vadd.f32   q0, q0, q11         \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123), // %10
                        "w"(_k3456), // %11
                        "w"(_k6789)  // %12
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}

static void conv3x3s2_transform_kernel_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8 * 9, inch, outch / 8 + outch % 8);

    const float* kernel = _kernel;

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        const float* k0 = kernel + (p + 0) * inch * 9;
        const float* k1 = kernel + (p + 1) * inch * 9;
        const float* k2 = kernel + (p + 2) * inch * 9;
        const float* k3 = kernel + (p + 3) * inch * 9;
        const float* k4 = kernel + (p + 4) * inch * 9;
        const float* k5 = kernel + (p + 5) * inch * 9;
        const float* k6 = kernel + (p + 6) * inch * 9;
        const float* k7 = kernel + (p + 7) * inch * 9;

        float* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p < outch; p++)
    {
        const float* k0 = kernel + (p + 0) * inch * 9;

        float* ktmp = kernel_tm.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

static void conv3x3s2_packed_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    //     const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p + 0);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);
        Mat out4 = top_blob.channel(p + 4);
        Mat out5 = top_blob.channel(p + 5);
        Mat out6 = top_blob.channel(p + 6);
        Mat out7 = top_blob.channel(p + 7);

        const float bias0 = bias ? bias[p + 0] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;
        const float bias4 = bias ? bias[p + 4] : 0.f;
        const float bias5 = bias ? bias[p + 5] : 0.f;
        const float bias6 = bias ? bias[p + 6] : 0.f;
        const float bias7 = bias ? bias[p + 7] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);
        out4.fill(bias4);
        out5.fill(bias5);
        out6.fill(bias6);
        out7.fill(bias7);

        const float* ktmp = _kernel.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;
            float* outptr6 = out6;
            float* outptr7 = out7;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v8.4s}, [%1]               \n"
                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v9.4s}, [%2]               \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v10.4s}, [%3]              \n"
                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v11.4s}, [%4]              \n"

                        ///
                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld2    {v4.4s, v5.4s}, [%9], #32   \n" // v4=00 v5=01

                        "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "fmla   v9.4s, v4.4s, v0.s[1]       \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v12.4s}, [%5]              \n"
                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v13.4s}, [%6]              \n"

                        "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v4.4s, v0.s[3]      \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v14.4s}, [%7]              \n"
                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v15.4s}, [%8]              \n"

                        "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v4.4s, v1.s[3]      \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld2    {v6.4s, v7.4s}, [%9]        \n" // v6

                        "fmla   v8.4s, v5.4s, v2.s[0]       \n"
                        "fmla   v9.4s, v5.4s, v2.s[1]       \n"
                        "fmla   v10.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v5.4s, v2.s[3]      \n"

                        "ext    v6.16b, v4.16b, v6.16b, #4  \n" // v6=02

                        "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v5.4s, v3.s[0]      \n"
                        "fmla   v13.4s, v5.4s, v3.s[1]      \n"
                        "fmla   v14.4s, v5.4s, v3.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v3.s[3]      \n"

                        ///
                        "prfm   pldl1keep, [%10, #256]      \n"
                        "ld2    {v4.4s, v5.4s}, [%10], #32  \n" // v4=10 v5=11

                        "fmla   v8.4s, v6.4s, v0.s[0]       \n"
                        "fmla   v9.4s, v6.4s, v0.s[1]       \n"
                        "fmla   v10.4s, v6.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v6.4s, v0.s[3]      \n"

                        "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v13.4s, v6.4s, v1.s[1]      \n"
                        "fmla   v14.4s, v6.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v6.4s, v1.s[3]      \n"

                        "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                        "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                        "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v4.4s, v2.s[3]      \n"

                        "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                        "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                        "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                        "fmla   v15.4s, v4.4s, v3.s[3]      \n"

                        "prfm   pldl1keep, [%10, #256]      \n"
                        "ld2    {v6.4s, v7.4s}, [%10]       \n" // v6

                        "fmla   v8.4s, v5.4s, v0.s[0]       \n"
                        "fmla   v9.4s, v5.4s, v0.s[1]       \n"
                        "fmla   v10.4s, v5.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v5.4s, v0.s[3]      \n"

                        "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                        "ext    v6.16b, v4.16b, v6.16b, #4  \n" // v6=12

                        "fmla   v12.4s, v5.4s, v1.s[0]      \n"
                        "fmla   v13.4s, v5.4s, v1.s[1]      \n"
                        "fmla   v14.4s, v5.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v1.s[3]      \n"

                        ///
                        "prfm   pldl1keep, [%11, #256]      \n"
                        "ld2    {v4.4s, v5.4s}, [%11], #32  \n" // v4=20 v5=21

                        "fmla   v8.4s, v6.4s, v2.s[0]       \n"
                        "fmla   v9.4s, v6.4s, v2.s[1]       \n"
                        "fmla   v10.4s, v6.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v6.4s, v2.s[3]      \n"

                        "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v6.4s, v3.s[0]      \n"
                        "fmla   v13.4s, v6.4s, v3.s[1]      \n"
                        "fmla   v14.4s, v6.4s, v3.s[2]      \n"
                        "fmla   v15.4s, v6.4s, v3.s[3]      \n"

                        "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                        "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                        "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v4.4s, v0.s[3]      \n"

                        "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                        "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                        "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v4.4s, v1.s[3]      \n"

                        "prfm   pldl1keep, [%11, #256]      \n"
                        "ld2    {v6.4s, v7.4s}, [%11]       \n" // v6

                        "fmla   v8.4s, v5.4s, v2.s[0]       \n"
                        "fmla   v9.4s, v5.4s, v2.s[1]       \n"
                        "fmla   v10.4s, v5.4s, v2.s[2]      \n"
                        "fmla   v11.4s, v5.4s, v2.s[3]      \n"

                        "ext    v6.16b, v4.16b, v6.16b, #4  \n" // v6=22

                        "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                        "fmla   v12.4s, v5.4s, v3.s[0]      \n"
                        "fmla   v13.4s, v5.4s, v3.s[1]      \n"
                        "fmla   v14.4s, v5.4s, v3.s[2]      \n"
                        "fmla   v15.4s, v5.4s, v3.s[3]      \n"

                        "fmla   v8.4s, v6.4s, v0.s[0]       \n"
                        "fmla   v9.4s, v6.4s, v0.s[1]       \n"
                        "fmla   v10.4s, v6.4s, v0.s[2]      \n"
                        "fmla   v11.4s, v6.4s, v0.s[3]      \n"

                        "fmla   v12.4s, v6.4s, v1.s[0]      \n"
                        "fmla   v13.4s, v6.4s, v1.s[1]      \n"

                        "st1    {v8.4s}, [%1], #16          \n"
                        "st1    {v9.4s}, [%2], #16          \n"

                        "fmla   v14.4s, v6.4s, v1.s[2]      \n"
                        "fmla   v15.4s, v6.4s, v1.s[3]      \n"

                        "st1    {v10.4s}, [%3], #16         \n"
                        "st1    {v11.4s}, [%4], #16         \n"

                        "sub    %12, %12, #288              \n"

                        "st1    {v12.4s}, [%5], #16         \n"
                        "st1    {v13.4s}, [%6], #16         \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v14.4s}, [%7], #16         \n"
                        "st1    {v15.4s}, [%8], #16         \n"

                        "bne    0b                          \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7), // %8
                        "=r"(r0),      // %9
                        "=r"(r1),      // %10
                        "=r"(r2),      // %11
                        "=r"(ktmp)     // %12
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7),
                        "9"(r0),
                        "10"(r1),
                        "11"(r2),
                        "12"(ktmp)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else  // __aarch64__
                for (; nn > 0; nn--)
                {
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d16-d17}, [%0]     \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d18-d19}, [%1]     \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d20-d21}, [%2]     \n"
                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d22-d23}, [%3]     \n"

                        ///
                        "pld        [%8, #256]          \n"
                        "vld2.f32   {d8-d11}, [%8]!     \n" // q4=00 q5=01

                        "vld1.f32   {d0-d3}, [%11 :128]! \n"

                        "vmla.f32   q8, q4, d0[0]       \n"
                        "vmla.f32   q9, q4, d0[1]       \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d24-d25}, [%4]     \n"
                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d26-d27}, [%5]     \n"

                        "vmla.f32   q10, q4, d1[0]      \n"
                        "vmla.f32   q11, q4, d1[1]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.f32   {d28-d29}, [%6]     \n"
                        "pld        [%7, #128]          \n"
                        "vld1.f32   {d30-d31}, [%7]     \n"

                        "vld1.f32   {d4-d7}, [%11 :128]! \n"

                        "vmla.f32   q12, q4, d2[0]      \n"
                        "vmla.f32   q13, q4, d2[1]      \n"
                        "vmla.f32   q14, q4, d3[0]      \n"
                        "vmla.f32   q15, q4, d3[1]      \n"

                        "pld        [%8, #128]          \n"
                        "vld2.f32   {d12-d13}, [%8]     \n" // q6

                        "vmla.f32   q8, q5, d4[0]       \n"
                        "vmla.f32   q9, q5, d4[1]       \n"
                        "vmla.f32   q10, q5, d5[0]      \n"
                        "vmla.f32   q11, q5, d5[1]      \n"

                        "vext.f32   q6, q4, q6, #1      \n" // q6=02

                        "vld1.f32   {d0-d3}, [%11 :128]! \n"

                        "vmla.f32   q12, q5, d6[0]      \n"
                        "vmla.f32   q13, q5, d6[1]      \n"
                        "vmla.f32   q14, q5, d7[0]      \n"
                        "vmla.f32   q15, q5, d7[1]      \n"

                        ///
                        "pld        [%9, #256]          \n"
                        "vld2.f32   {d8-d11}, [%9]!     \n" // q4=10 q5=11

                        "vmla.f32   q8, q6, d0[0]       \n"
                        "vmla.f32   q9, q6, d0[1]       \n"
                        "vmla.f32   q10, q6, d1[0]      \n"
                        "vmla.f32   q11, q6, d1[1]      \n"

                        "vld1.f32   {d4-d7}, [%11 :128]! \n"

                        "vmla.f32   q12, q6, d2[0]      \n"
                        "vmla.f32   q13, q6, d2[1]      \n"
                        "vmla.f32   q14, q6, d3[0]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "vmla.f32   q8, q4, d4[0]       \n"
                        "vmla.f32   q9, q4, d4[1]       \n"
                        "vmla.f32   q10, q4, d5[0]      \n"
                        "vmla.f32   q11, q4, d5[1]      \n"

                        "vld1.f32   {d0-d3}, [%11 :128]! \n"

                        "vmla.f32   q12, q4, d6[0]      \n"
                        "vmla.f32   q13, q4, d6[1]      \n"
                        "vmla.f32   q14, q4, d7[0]      \n"
                        "vmla.f32   q15, q4, d7[1]      \n"

                        "pld        [%9, #128]          \n"
                        "vld2.f32   {d12-d13}, [%9]     \n" // q6

                        "vmla.f32   q8, q5, d0[0]       \n"
                        "vmla.f32   q9, q5, d0[1]       \n"
                        "vmla.f32   q10, q5, d1[0]      \n"
                        "vmla.f32   q11, q5, d1[1]      \n"

                        "vld1.f32   {d4-d7}, [%11 :128]! \n"

                        "vext.f32   q6, q4, q6, #1      \n" // q6=12

                        "vmla.f32   q12, q5, d2[0]      \n"
                        "vmla.f32   q13, q5, d2[1]      \n"
                        "vmla.f32   q14, q5, d3[0]      \n"
                        "vmla.f32   q15, q5, d3[1]      \n"

                        ///
                        "pld        [%10, #256]         \n"
                        "vld2.f32   {d8-d11}, [%10]!    \n" // q4=20 q5=21

                        "vmla.f32   q8, q6, d4[0]       \n"
                        "vmla.f32   q9, q6, d4[1]       \n"
                        "vmla.f32   q10, q6, d5[0]      \n"
                        "vmla.f32   q11, q6, d5[1]      \n"

                        "vld1.f32   {d0-d3}, [%11 :128]! \n"

                        "vmla.f32   q12, q6, d6[0]      \n"
                        "vmla.f32   q13, q6, d6[1]      \n"
                        "vmla.f32   q14, q6, d7[0]      \n"
                        "vmla.f32   q15, q6, d7[1]      \n"

                        "vmla.f32   q8, q4, d0[0]       \n"
                        "vmla.f32   q9, q4, d0[1]       \n"
                        "vmla.f32   q10, q4, d1[0]      \n"
                        "vmla.f32   q11, q4, d1[1]      \n"

                        "vld1.f32   {d4-d7}, [%11 :128]! \n"

                        "vmla.f32   q12, q4, d2[0]      \n"
                        "vmla.f32   q13, q4, d2[1]      \n"
                        "vmla.f32   q14, q4, d3[0]      \n"
                        "vmla.f32   q15, q4, d3[1]      \n"

                        "pld        [%10, #128]         \n"
                        "vld2.f32   {d12-d13}, [%10]    \n" // q6

                        "vmla.f32   q8, q5, d4[0]       \n"
                        "vmla.f32   q9, q5, d4[1]       \n"
                        "vmla.f32   q10, q5, d5[0]      \n"
                        "vmla.f32   q11, q5, d5[1]      \n"

                        "vext.f32   q6, q4, q6, #1      \n" // q6=22

                        "vld1.f32   {d0-d3}, [%11 :128]! \n"

                        "vmla.f32   q12, q5, d6[0]      \n"
                        "vmla.f32   q13, q5, d6[1]      \n"
                        "vmla.f32   q14, q5, d7[0]      \n"
                        "vmla.f32   q15, q5, d7[1]      \n"

                        "vmla.f32   q8, q6, d0[0]       \n"
                        "vmla.f32   q9, q6, d0[1]       \n"
                        "vmla.f32   q10, q6, d1[0]      \n"
                        "vmla.f32   q11, q6, d1[1]      \n"

                        "vmla.f32   q12, q6, d2[0]      \n"
                        "vmla.f32   q13, q6, d2[1]      \n"

                        "vst1.f32   {d16-d17}, [%0]!    \n"
                        "vst1.f32   {d18-d19}, [%1]!    \n"

                        "vmla.f32   q14, q6, d3[0]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "vst1.f32   {d20-d21}, [%2]!    \n"
                        "vst1.f32   {d22-d23}, [%3]!    \n"

                        "sub        %11, %11, #288      \n"

                        "vst1.f32   {d24-d25}, [%4]!    \n"
                        "vst1.f32   {d26-d27}, [%5]!    \n"
                        "vst1.f32   {d28-d29}, [%6]!    \n"
                        "vst1.f32   {d30-d31}, [%7]!    \n"
                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(outptr2), // %2
                        "=r"(outptr3), // %3
                        "=r"(outptr4), // %4
                        "=r"(outptr5), // %5
                        "=r"(outptr6), // %6
                        "=r"(outptr7), // %7
                        "=r"(r0),      // %8
                        "=r"(r1),      // %9
                        "=r"(r2),      // %10
                        "=r"(ktmp)     // %11
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(outptr2),
                        "3"(outptr3),
                        "4"(outptr4),
                        "5"(outptr5),
                        "6"(outptr6),
                        "7"(outptr7),
                        "8"(r0),
                        "9"(r1),
                        "10"(r2),
                        "11"(ktmp)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "prfm   pldl1keep, [%8, #128]   \n"
                        "ld1    {v0.4s}, [%8]           \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "ld1    {v8.s}[0], [%0]         \n"
                        "ld1    {v8.s}[1], [%1]         \n"
                        "ld1    {v8.s}[2], [%2]         \n"
                        "ld1    {v8.s}[3], [%3]         \n"

                        "fmul   v14.4s, v10.4s, v0.s[0] \n"
                        "fmul   v15.4s, v11.4s, v0.s[0] \n"

                        "ld1    {v9.s}[0], [%4]         \n"
                        "ld1    {v9.s}[1], [%5]         \n"
                        "ld1    {v9.s}[2], [%6]         \n"
                        "ld1    {v9.s}[3], [%7]         \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v0.s[1]  \n"
                        "fmla   v9.4s, v13.4s, v0.s[1]  \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "fmla   v14.4s, v10.4s, v0.s[2] \n"
                        "fmla   v15.4s, v11.4s, v0.s[2] \n"

                        "prfm   pldl1keep, [%9, #128]   \n"
                        "ld1    {v1.4s}, [%9]           \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v1.s[0]  \n"
                        "fmla   v9.4s, v13.4s, v1.s[0]  \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "fmla   v14.4s, v10.4s, v1.s[1] \n"
                        "fmla   v15.4s, v11.4s, v1.s[1] \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v1.s[2]  \n"
                        "fmla   v9.4s, v13.4s, v1.s[2]  \n"

                        "prfm   pldl1keep, [%10, #128]  \n"
                        "ld1    {v0.4s}, [%10]          \n"

                        "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                        "fmla   v14.4s, v10.4s, v0.s[0] \n"
                        "fmla   v15.4s, v11.4s, v0.s[0] \n"

                        "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                        "fmla   v8.4s, v12.4s, v0.s[1]  \n"
                        "fmla   v9.4s, v13.4s, v0.s[1]  \n"

                        "fmla   v14.4s, v10.4s, v0.s[2] \n"
                        "fmla   v15.4s, v11.4s, v0.s[2] \n"

                        "fadd   v8.4s, v8.4s, v14.4s    \n"
                        "fadd   v9.4s, v9.4s, v15.4s    \n"

                        "sub    %11, %11, #288          \n"

                        "st1    {v8.s}[0], [%0], #4     \n"
                        "st1    {v8.s}[1], [%1], #4     \n"
                        "st1    {v8.s}[2], [%2], #4     \n"
                        "st1    {v8.s}[3], [%3], #4     \n"

                        "st1    {v9.s}[0], [%4], #4     \n"
                        "st1    {v9.s}[1], [%5], #4     \n"
                        "st1    {v9.s}[2], [%6], #4     \n"
                        "st1    {v9.s}[3], [%7], #4     \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(outptr2), // %2
                        "=r"(outptr3), // %3
                        "=r"(outptr4), // %4
                        "=r"(outptr5), // %5
                        "=r"(outptr6), // %6
                        "=r"(outptr7), // %7
                        "=r"(r0),      // %8
                        "=r"(r1),      // %9
                        "=r"(r2),      // %10
                        "=r"(ktmp)     // %11
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(outptr2),
                        "3"(outptr3),
                        "4"(outptr4),
                        "5"(outptr5),
                        "6"(outptr6),
                        "7"(outptr7),
                        "8"(r0),
                        "9"(r1),
                        "10"(r2),
                        "11"(ktmp)
                        : "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
#else  // __aarch64__
                    asm volatile(
                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "pld        [%8, #128]      \n"
                        "vld1.f32   {d0-d1}, [%8]   \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vld1.f32   {d16[0]}, [%0]  \n"
                        "vld1.f32   {d16[1]}, [%1]  \n"
                        "vld1.f32   {d17[0]}, [%2]  \n"
                        "vld1.f32   {d17[1]}, [%3]  \n"

                        "vmul.f32   q14, q10, d0[0] \n"
                        "vmul.f32   q15, q11, d0[0] \n"

                        "vld1.f32   {d18[0]}, [%4]  \n"
                        "vld1.f32   {d18[1]}, [%5]  \n"
                        "vld1.f32   {d19[0]}, [%6]  \n"
                        "vld1.f32   {d19[1]}, [%7]  \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d0[1]  \n"
                        "vmla.f32   q9, q13, d0[1]  \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vmla.f32   q14, q10, d1[0] \n"
                        "vmla.f32   q15, q11, d1[0] \n"

                        "pld        [%9, #128]      \n"
                        "vld1.f32   {d2-d3}, [%9]   \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d2[0]  \n"
                        "vmla.f32   q9, q13, d2[0]  \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vmla.f32   q14, q10, d2[1] \n"
                        "vmla.f32   q15, q11, d2[1] \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d3[0]  \n"
                        "vmla.f32   q9, q13, d3[0]  \n"

                        "pld        [%10, #128]     \n"
                        "vld1.f32   {d0-d1}, [%10]  \n"

                        "vld1.f32   {d24-d27}, [%11 :128]! \n"

                        "vmla.f32   q14, q10, d0[0] \n"
                        "vmla.f32   q15, q11, d0[0] \n"

                        "vld1.f32   {d20-d23}, [%11 :128]! \n"

                        "vmla.f32   q8, q12, d0[1]  \n"
                        "vmla.f32   q9, q13, d0[1]  \n"

                        "vmla.f32   q14, q10, d1[0] \n"
                        "vmla.f32   q15, q11, d1[0] \n"

                        "vadd.f32   q8, q8, q14     \n"
                        "vadd.f32   q9, q9, q15     \n"

                        "sub        %11, %11, #288  \n"

                        "vst1.f32   {d16[0]}, [%0]! \n"
                        "vst1.f32   {d16[1]}, [%1]! \n"
                        "vst1.f32   {d17[0]}, [%2]! \n"
                        "vst1.f32   {d17[1]}, [%3]! \n"

                        "vst1.f32   {d18[0]}, [%4]! \n"
                        "vst1.f32   {d18[1]}, [%5]! \n"
                        "vst1.f32   {d19[0]}, [%6]! \n"
                        "vst1.f32   {d19[1]}, [%7]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(outptr2), // %2
                        "=r"(outptr3), // %3
                        "=r"(outptr4), // %4
                        "=r"(outptr5), // %5
                        "=r"(outptr6), // %6
                        "=r"(outptr7), // %7
                        "=r"(r0),      // %8
                        "=r"(r1),      // %9
                        "=r"(r2),      // %10
                        "=r"(ktmp)     // %11
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(outptr2),
                        "3"(outptr3),
                        "4"(outptr4),
                        "5"(outptr5),
                        "6"(outptr6),
                        "7"(outptr7),
                        "8"(r0),
                        "9"(r1),
                        "10"(r2),
                        "11"(ktmp)
                        : "memory", "q0", "q1", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // __ARM_NEON
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;
                    float sum4 = 0.f;
                    float sum5 = 0.f;
                    float sum6 = 0.f;
                    float sum7 = 0.f;

                    sum0 += r0[0] * ktmp[0];
                    sum1 += r0[0] * ktmp[1];
                    sum2 += r0[0] * ktmp[2];
                    sum3 += r0[0] * ktmp[3];
                    sum4 += r0[0] * ktmp[4];
                    sum5 += r0[0] * ktmp[5];
                    sum6 += r0[0] * ktmp[6];
                    sum7 += r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[1] * ktmp[0];
                    sum1 += r0[1] * ktmp[1];
                    sum2 += r0[1] * ktmp[2];
                    sum3 += r0[1] * ktmp[3];
                    sum4 += r0[1] * ktmp[4];
                    sum5 += r0[1] * ktmp[5];
                    sum6 += r0[1] * ktmp[6];
                    sum7 += r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[2] * ktmp[0];
                    sum1 += r0[2] * ktmp[1];
                    sum2 += r0[2] * ktmp[2];
                    sum3 += r0[2] * ktmp[3];
                    sum4 += r0[2] * ktmp[4];
                    sum5 += r0[2] * ktmp[5];
                    sum6 += r0[2] * ktmp[6];
                    sum7 += r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[0] * ktmp[0];
                    sum1 += r1[0] * ktmp[1];
                    sum2 += r1[0] * ktmp[2];
                    sum3 += r1[0] * ktmp[3];
                    sum4 += r1[0] * ktmp[4];
                    sum5 += r1[0] * ktmp[5];
                    sum6 += r1[0] * ktmp[6];
                    sum7 += r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[1] * ktmp[0];
                    sum1 += r1[1] * ktmp[1];
                    sum2 += r1[1] * ktmp[2];
                    sum3 += r1[1] * ktmp[3];
                    sum4 += r1[1] * ktmp[4];
                    sum5 += r1[1] * ktmp[5];
                    sum6 += r1[1] * ktmp[6];
                    sum7 += r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[2] * ktmp[0];
                    sum1 += r1[2] * ktmp[1];
                    sum2 += r1[2] * ktmp[2];
                    sum3 += r1[2] * ktmp[3];
                    sum4 += r1[2] * ktmp[4];
                    sum5 += r1[2] * ktmp[5];
                    sum6 += r1[2] * ktmp[6];
                    sum7 += r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[0] * ktmp[0];
                    sum1 += r2[0] * ktmp[1];
                    sum2 += r2[0] * ktmp[2];
                    sum3 += r2[0] * ktmp[3];
                    sum4 += r2[0] * ktmp[4];
                    sum5 += r2[0] * ktmp[5];
                    sum6 += r2[0] * ktmp[6];
                    sum7 += r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[1] * ktmp[0];
                    sum1 += r2[1] * ktmp[1];
                    sum2 += r2[1] * ktmp[2];
                    sum3 += r2[1] * ktmp[3];
                    sum4 += r2[1] * ktmp[4];
                    sum5 += r2[1] * ktmp[5];
                    sum6 += r2[1] * ktmp[6];
                    sum7 += r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[2] * ktmp[0];
                    sum1 += r2[2] * ktmp[1];
                    sum2 += r2[2] * ktmp[2];
                    sum3 += r2[2] * ktmp[3];
                    sum4 += r2[2] * ktmp[4];
                    sum5 += r2[2] * ktmp[5];
                    sum6 += r2[2] * ktmp[6];
                    sum7 += r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8 * 9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8 * 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        const float* ktmp = _kernel.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

            const float* k0 = ktmp;
            const float* k1 = ktmp + 3;
            const float* k2 = ktmp + 6;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);
#endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "0:                                        \n"

                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v0.4s}, [%1]                  \n"

                        "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                        "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%2]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                        "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%3]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                        "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%4]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                        "fadd       v0.4s, v0.4s, v10.4s           \n"
                        "fadd       v0.4s, v0.4s, v11.4s           \n"

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s}, [%1], #16             \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #32                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123), // %10
                        "w"(_k3456), // %11
                        "w"(_k6789)  // %12
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1]       \n"

                        "vmla.f32   q0, q2, %e10[0]     \n"
                        "vmul.f32   q10, q3, %e10[1]    \n"

                        "pld        [%2, #128]          \n"
                        "vld2.f32   {d16-d17}, [%2]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmul.f32   q11, q1, %f10[0]    \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d4-d7}, [%3]!      \n"

                        "vmla.f32   q0, q2, %e11[0]     \n"
                        "vmla.f32   q10, q3, %e11[1]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d16-d17}, [%3]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f11[0]    \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d4-d7}, [%4]!      \n"

                        "vmla.f32   q0, q2, %e12[0]     \n"
                        "vmla.f32   q10, q3, %e12[1]    \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d16-d17}, [%4]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f12[0]    \n"

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "vadd.f32   q0, q0, q10         \n"
                        "vadd.f32   q0, q0, q11         \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123), // %10
                        "w"(_k3456), // %11
                        "w"(_k6789)  // %12
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * ktmp[0];
                    sum += r0[1] * ktmp[1];
                    sum += r0[2] * ktmp[2];
                    sum += r1[0] * ktmp[3];
                    sum += r1[1] * ktmp[4];
                    sum += r1[2] * ktmp[5];
                    sum += r2[0] * ktmp[6];
                    sum += r2[1] * ktmp[7];
                    sum += r2[2] * ktmp[8];

                    *outptr += sum;
#endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
