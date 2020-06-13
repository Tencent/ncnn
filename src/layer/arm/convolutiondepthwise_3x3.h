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

static void convdw3x3s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g * 9;

        float* outptr = out;
        float* outptr2 = outptr + outw;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;
        const float* r3 = img0 + w * 3;

#if __ARM_NEON
        float32x4_t _k012x = vld1q_f32(kernel0);
        float32x4_t _k345x = vld1q_f32(kernel0 + 3);
        float32x4_t _k678x = vld1q_f32(kernel0 + 6);

        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        float32x4_t _bias0 = vdupq_n_f32(bias0);
#else
        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

        int i = 0;

        for (; i + 1 < outh; i += 2)
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
                    "prfm   pldl1keep, [%3, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%3]    \n" // r0
                    "add    %3, %3, #32                     \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v13.16b, v9.16b, v10.16b, #4    \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"
                    "ext    v14.16b, v9.16b, v10.16b, #8    \n"

                    "0:                                     \n"

                    "and    v4.16b, %17.16b, %17.16b        \n" // v4 = _bias0
                    "and    v5.16b, %17.16b, %17.16b        \n" // v5 = _bias0

                    "prfm   pldl1keep, [%6, #384]           \n"
                    "ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n" // r3
                    "add    %6, %6, #32                     \n"

                    "and    v6.16b, %17.16b, %17.16b        \n" // v6 = _bias0
                    "and    v7.16b, %17.16b, %17.16b        \n" // v7 = _bias0

                    "ext    v15.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v4.4s, v8.4s, %14.s[0]          \n"
                    "fmla   v5.4s, v9.4s, %14.s[0]          \n"

                    "ext    v20.16b, v17.16b, v18.16b, #4   \n"

                    "fmla   v6.4s, v16.4s, %16.s[0]         \n"
                    "fmla   v7.4s, v17.4s, %16.s[0]         \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v4.4s, v11.4s, %14.s[1]         \n"
                    "fmla   v5.4s, v13.4s, %14.s[1]         \n"

                    "ext    v21.16b, v17.16b, v18.16b, #8   \n"

                    "fmla   v6.4s, v15.4s, %16.s[1]         \n"
                    "fmla   v7.4s, v20.4s, %16.s[1]         \n"

                    "prfm   pldl1keep, [%4, #384]           \n"
                    "ld1    {v22.4s, v23.4s, v24.4s}, [%4]  \n" // r1

                    "fmla   v4.4s, v12.4s, %14.s[2]         \n"
                    "fmla   v5.4s, v14.4s, %14.s[2]         \n"

                    "add    %4, %4, #32                     \n"

                    "fmla   v6.4s, v19.4s, %16.s[2]         \n"
                    "fmla   v7.4s, v21.4s, %16.s[2]         \n"

                    "ext    v25.16b, v22.16b, v23.16b, #4   \n"

                    "fmla   v4.4s, v22.4s, %15.s[0]         \n"
                    "fmla   v5.4s, v23.4s, %15.s[0]         \n"

                    "ext    v27.16b, v23.16b, v24.16b, #4   \n"

                    "fmla   v6.4s, v22.4s, %14.s[0]         \n"
                    "fmla   v7.4s, v23.4s, %14.s[0]         \n"

                    "ext    v26.16b, v22.16b, v23.16b, #8   \n"

                    "fmla   v4.4s, v25.4s, %15.s[1]         \n"
                    "fmla   v5.4s, v27.4s, %15.s[1]         \n"

                    "ext    v28.16b, v23.16b, v24.16b, #8   \n"

                    "fmla   v6.4s, v25.4s, %14.s[1]         \n"
                    "fmla   v7.4s, v27.4s, %14.s[1]         \n"

                    "prfm   pldl1keep, [%5, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%5]    \n" // r2

                    "fmla   v4.4s, v26.4s, %15.s[2]         \n"
                    "fmla   v5.4s, v28.4s, %15.s[2]         \n"

                    "add    %5, %5, #32                     \n"

                    "fmla   v6.4s, v26.4s, %14.s[2]         \n"
                    "fmla   v7.4s, v28.4s, %14.s[2]         \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"

                    "fmla   v4.4s, v8.4s, %16.s[0]          \n"
                    "fmla   v5.4s, v9.4s, %16.s[0]          \n"

                    "ext    v13.16b, v9.16b, v10.16b, #4    \n"

                    "fmla   v6.4s, v8.4s, %15.s[0]          \n"
                    "fmla   v7.4s, v9.4s, %15.s[0]          \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v4.4s, v11.4s, %16.s[1]         \n"
                    "fmla   v5.4s, v13.4s, %16.s[1]         \n"

                    "ext    v14.16b, v9.16b, v10.16b, #8    \n"

                    "fmla   v6.4s, v11.4s, %15.s[1]         \n"
                    "fmla   v7.4s, v13.4s, %15.s[1]         \n"

                    "prfm   pldl1keep, [%3, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%3]    \n" // r0 next loop

                    "fmla   v4.4s, v12.4s, %16.s[2]         \n"
                    "fmla   v5.4s, v14.4s, %16.s[2]         \n"

                    "add    %3, %3, #32                     \n"
                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"

                    "fmla   v6.4s, v12.4s, %15.s[2]         \n"
                    "fmla   v7.4s, v14.4s, %15.s[2]         \n"

                    "ext    v13.16b, v9.16b, v10.16b, #4    \n"
                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"

                    "st1    {v4.4s, v5.4s}, [%1], #32       \n"

                    "ext    v14.16b, v9.16b, v10.16b, #8    \n"

                    "subs   %w0, %w0, #1                    \n"

                    "st1    {v6.4s, v7.4s}, [%2], #32       \n"

                    "bne    0b                              \n"
                    "sub    %3, %3, #32                     \n"
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
                    "w"(_k012x), // %14
                    "w"(_k345x), // %15
                    "w"(_k678x), // %16
                    "w"(_bias0)  // %17
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28");
            }

            if (remain >= 4)
            {
                remain -= 4;

                asm volatile(
                    "prfm   pldl1keep, [%2, #256]           \n"
                    "ld1    {v8.4s, v9.4s}, [%2]            \n" // r0
                    "add    %2, %2, #16                     \n"

                    "and    v4.16b, %15.16b, %15.16b        \n" // v4 = _bias0
                    "and    v6.16b, %15.16b, %15.16b        \n" // v6 = _bias0

                    "prfm   pldl1keep, [%5, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%5]          \n" // r3
                    "add    %5, %5, #16                     \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v15.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v4.4s, v8.4s, %12.s[0]          \n"
                    "fmla   v6.4s, v16.4s, %14.s[0]         \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"
                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v4.4s, v11.4s, %12.s[1]         \n"
                    "fmla   v6.4s, v15.4s, %14.s[1]         \n"

                    "prfm   pldl1keep, [%3, #256]           \n"
                    "ld1    {v22.4s, v23.4s}, [%3]          \n" // r1

                    "fmla   v4.4s, v12.4s, %12.s[2]         \n"

                    "add    %3, %3, #16                     \n"

                    "fmla   v6.4s, v19.4s, %14.s[2]         \n"

                    "ext    v25.16b, v22.16b, v23.16b, #4   \n"

                    "fmla   v4.4s, v22.4s, %13.s[0]         \n"
                    "fmla   v6.4s, v22.4s, %12.s[0]         \n"

                    "ext    v26.16b, v22.16b, v23.16b, #8   \n"

                    "fmla   v4.4s, v25.4s, %13.s[1]         \n"
                    "fmla   v6.4s, v25.4s, %12.s[1]         \n"

                    "prfm   pldl1keep, [%4, #256]           \n"
                    "ld1    {v8.4s, v9.4s}, [%4]            \n" // r2

                    "fmla   v4.4s, v26.4s, %13.s[2]         \n"

                    "add    %4, %4, #16                     \n"

                    "fmla   v6.4s, v26.4s, %12.s[2]         \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"

                    "fmla   v4.4s, v8.4s, %14.s[0]          \n"
                    "fmla   v6.4s, v8.4s, %13.s[0]          \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v4.4s, v11.4s, %14.s[1]         \n"
                    "fmla   v6.4s, v11.4s, %13.s[1]         \n"

                    "fmla   v4.4s, v12.4s, %14.s[2]         \n"
                    "fmla   v6.4s, v12.4s, %13.s[2]         \n"

                    "st1    {v4.4s}, [%0], #16              \n"
                    "st1    {v6.4s}, [%1], #16              \n"

                    : "=r"(outptr),  // %0
                    "=r"(outptr2), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3)       // %5
                    : "0"(outptr),
                    "1"(outptr2),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "w"(_k012x), // %12
                    "w"(_k345x), // %13
                    "w"(_k678x), // %14
                    "w"(_bias0)  // %15
                    : "cc", "memory", "v4", "v6", "v8", "v9", "v11", "v12", "v15", "v16", "v17", "v18", "v19", "v22", "v23", "v25", "v26");
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

                    "vmul.f32   q7, q9, %e14[0]     \n"

                    "vand       q13, %q17, %q17     \n" // q13 = _bias0
                    "vmul.f32   q6, q11, %e14[1]    \n"
                    "vmla.f32   q13, q12, %f14[0]   \n"

                    "pld        [%4, #192]          \n"
                    "vld1.f32   {d18-d20}, [%4]     \n" // r1
                    "add        %4, #16             \n"

                    "vmla.f32   q7, q9, %e15[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q6, q11, %e15[1]    \n"
                    "vmla.f32   q13, q12, %f15[0]   \n"

                    "vmul.f32   q8, q9, %e14[0]     \n"

                    "vand       q15, %q17, %q17     \n" // q15 = _bias0
                    "vmul.f32   q14, q11, %e14[1]   \n"
                    "vmla.f32   q15, q12, %f14[0]   \n"

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
                    "w"(_k012x), // %14
                    "w"(_k345x), // %15
                    "w"(_k678x), // %16
                    "w"(_bias0)  // %17
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

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                float32x4_t _sum2 = vmulq_f32(_r10, _k012x);
                _sum2 = vmlaq_f32(_sum2, _r20, _k345x);
                _sum2 = vmlaq_f32(_sum2, _r30, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);
                _sum2 = vsetq_lane_f32(bias0, _sum2, 3);
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
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                float sum2 = bias0;
                sum2 += r1[0] * k0[0];
                sum2 += r1[1] * k0[1];
                sum2 += r1[2] * k0[2];
                sum2 += r2[0] * k1[0];
                sum2 += r2[1] * k1[1];
                sum2 += r2[2] * k1[2];
                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];

                *outptr = sum;
                *outptr2 = sum2;
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
                    "prfm   pldl1keep, [%2, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%2]    \n" // r0
                    "add    %2, %2, #32                     \n"

                    "ext    v12.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v14.16b, v9.16b, v10.16b, #4    \n"

                    "0:                                     \n"

                    "fmul   v6.4s, v8.4s, %10.s[0]          \n"

                    "and    v4.16b, %13.16b, %13.16b        \n" // v4 = _bias0

                    "fmul   v7.4s, v9.4s, %10.s[0]          \n"

                    "and    v5.16b, %13.16b, %13.16b        \n" // v5 = _bias0

                    "fmla   v4.4s, v12.4s, %10.s[1]         \n"

                    "ext    v13.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v5.4s, v14.4s, %10.s[1]         \n"

                    "ext    v15.16b, v9.16b, v10.16b, #8    \n"

                    "fmla   v6.4s, v13.4s, %10.s[2]         \n"

                    "prfm   pldl1keep, [%3, #384]           \n"
                    "ld1    {v16.4s, v17.4s, v18.4s}, [%3]  \n" // r1

                    "fmla   v7.4s, v15.4s, %10.s[2]         \n"

                    "add    %3, %3, #32                     \n"

                    "fmla   v4.4s, v16.4s, %11.s[0]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v5.4s, v17.4s, %11.s[0]         \n"

                    "ext    v22.16b, v17.16b, v18.16b, #4   \n"

                    "fmla   v6.4s, v20.4s, %11.s[1]         \n"

                    "ext    v21.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v7.4s, v22.4s, %11.s[1]         \n"

                    "ext    v23.16b, v17.16b, v18.16b, #8   \n"

                    "fmla   v4.4s, v21.4s, %11.s[2]         \n"

                    "prfm   pldl1keep, [%4, #384]           \n"
                    "ld1    {v24.4s, v25.4s, v26.4s}, [%4]  \n" // r2

                    "fmla   v5.4s, v23.4s, %11.s[2]         \n"

                    "add    %4, %4, #32                     \n"

                    "fmla   v6.4s, v24.4s, %12.s[0]         \n"

                    "ext    v12.16b, v24.16b, v25.16b, #4   \n"

                    "fmla   v7.4s, v25.4s, %12.s[0]         \n"

                    "ext    v14.16b, v25.16b, v26.16b, #4   \n"

                    "fmla   v4.4s, v12.4s, %12.s[1]         \n"

                    "ext    v13.16b, v24.16b, v25.16b, #8   \n"

                    "fmla   v5.4s, v14.4s, %12.s[1]         \n"

                    "ext    v15.16b, v25.16b, v26.16b, #8   \n"

                    "fmla   v6.4s, v13.4s, %12.s[2]         \n"
                    "fmla   v7.4s, v15.4s, %12.s[2]         \n"

                    "prfm   pldl1keep, [%2, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%2]    \n" // r0 next loop

                    "fadd   v4.4s, v4.4s, v6.4s             \n"

                    "add    %2, %2, #32                     \n"

                    "fadd   v5.4s, v5.4s, v7.4s             \n"

                    "ext    v12.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v14.16b, v9.16b, v10.16b, #4    \n"

                    "subs   %w0, %w0, #1                    \n"

                    "st1    {v4.4s, v5.4s}, [%1], #32       \n"

                    "bne    0b                              \n"
                    "sub    %2, %2, #32                     \n"
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
                    "w"(_k012x), // %10
                    "w"(_k345x), // %11
                    "w"(_k678x), // %12
                    "w"(_bias0)  // %13
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26");
            }

            if (remain >= 4)
            {
                remain -= 4;

                asm volatile(
                    "prfm   pldl1keep, [%1, #192]           \n"
                    "ld1    {v8.4s, v9.4s}, [%1]            \n" // r0
                    "add    %1, %1, #16                     \n"

                    "and    v4.16b, %11.16b, %11.16b        \n" // v4 = _bias0

                    "ext    v12.16b, v8.16b, v9.16b, #4     \n"

                    "fmul   v6.4s, v8.4s, %8.s[0]           \n"

                    "ext    v13.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v4.4s, v12.4s, %8.s[1]          \n"

                    "prfm   pldl1keep, [%2, #192]           \n"
                    "ld1    {v16.4s, v17.4s}, [%2]          \n" // r1
                    "add    %2, %2, #16                     \n"

                    "fmla   v6.4s, v13.4s, %8.s[2]          \n"

                    "ext    v20.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v4.4s, v16.4s, %9.s[0]          \n"

                    "ext    v21.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v6.4s, v20.4s, %9.s[1]          \n"

                    "prfm   pldl1keep, [%3, #192]           \n"
                    "ld1    {v24.4s, v25.4s}, [%3]          \n" // r2
                    "add    %3, %3, #16                     \n"

                    "fmla   v4.4s, v21.4s, %9.s[2]          \n"

                    "ext    v12.16b, v24.16b, v25.16b, #4   \n"

                    "fmla   v6.4s, v24.4s, %10.s[0]         \n"

                    "ext    v13.16b, v24.16b, v25.16b, #8   \n"

                    "fmla   v4.4s, v12.4s, %10.s[1]         \n"

                    "fmla   v6.4s, v13.4s, %10.s[2]         \n"

                    "fadd   v4.4s, v4.4s, v6.4s             \n"

                    "st1    {v4.4s}, [%0], #16              \n"

                    : "=r"(outptr), // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2)      // %3
                    : "0"(outptr),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "w"(_k012x), // %8
                    "w"(_k345x), // %9
                    "w"(_k678x), // %10
                    "w"(_bias0)  // %11
                    : "cc", "memory", "v4", "v6", "v8", "v9", "v12", "v13", "v16", "v17", "v20", "v21", "v24", "v25");
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

                    "vmul.f32   q7, q8, %e10[0]     \n"

                    "vand       q14, %q13, %q13     \n" // q14 = _bias0
                    "vmul.f32   q13, q10, %e10[1]   \n"
                    "vmla.f32   q14, q11, %f10[0]   \n"

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
                    "w"(_k012x), // %10
                    "w"(_k345x), // %11
                    "w"(_k678x), // %12
                    "w"(_bias0)  // %13
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

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);
#if __aarch64__
                *outptr = vaddvq_f32(_sum);
#else
                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;
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
    }
}

static void convdw3x3s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g * 9;

        float* outptr = out;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;

#if __ARM_NEON
        float32x4_t _k012x = vld1q_f32(kernel0);
        float32x4_t _k345x = vld1q_f32(kernel0 + 3);
        float32x4_t _k678x = vld1q_f32(kernel0 + 6);

        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        float32x4_t _bias0 = vdupq_n_f32(bias0);
#else
        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;
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

                    "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                    "0:                                        \n"
                    "fmul       v0.4s,  v2.4s, %10.s[0]        \n"
                    "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %10.s[2]        \n"

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

                    "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

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
                    "w"(_k012x), // %10
                    "w"(_k345x), // %11
                    "w"(_k678x), // %12
                    "w"(_bias0)  // %13
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vand       q11, %q13, %q13     \n"

                    "0:                             \n"
                    "vmul.f32   q0, q2, %e10[0]     \n"
                    "vmul.f32   q10, q3, %e10[1]    \n"

                    "pld        [%2, #128]          \n"
                    "vld2.f32   {d16-d17}, [%2]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f10[0]    \n"

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

                    "vand       q11, %q13, %q13     \n"

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
                    "w"(_k012x), // %10
                    "w"(_k345x), // %11
                    "w"(_k678x), // %12
                    "w"(_bias0)  // %13
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

                float32x4_t _sum = vmulq_f32(_r00, _k012x);
                _sum = vmlaq_f32(_sum, _r10, _k345x);
                _sum = vmlaq_f32(_sum, _r20, _k678x);

                _sum = vsetq_lane_f32(bias0, _sum, 3);
#if __aarch64__
                *outptr = vaddvq_f32(_sum);
#else
                float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _ss = vpadd_f32(_ss, _ss);

                *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;
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
    }
}
