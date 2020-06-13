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

static void conv4x4s4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 4 * outw + w * 3;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch * 16 + q * 16;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0 + 4);
            float32x4_t _k891011 = vld1q_f32(kernel0 + 8);
            float32x4_t _k12131415 = vld1q_f32(kernel0 + 12);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 4;
            const float* k2 = kernel0 + 8;
            const float* k3 = kernel0 + 12;
#endif // __ARM_NEON

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%1, #128]          \n"
                        "0:                                        \n"

                        "prfm       pldl1keep, [%2, #512]          \n"
                        "prfm       pldl1keep, [%3, #512]          \n"

                        "ld1        {v7.4s}, [%1]                  \n" // v7 = outptr

                        "ld1        {v8.4s}, [%2], #16             \n" // v8  = r0
                        "ld1        {v9.4s}, [%3], #16             \n" // v9  = r1

                        "prfm       pldl1keep, [%4, #512]          \n"
                        "prfm       pldl1keep, [%5, #512]          \n"

                        "fmul       v12.4s, v8.4s, %12.4s          \n"
                        "fmul       v13.4s, v9.4s, %13.4s          \n"

                        "ld1        {v10.4s}, [%4], #16            \n" // v10 = r2
                        "ld1        {v11.4s}, [%5], #16            \n" // v11 = r3

                        "fmla       v12.4s, v10.4s, %14.4s         \n"
                        "fmla       v13.4s, v11.4s, %15.4s         \n"

                        "fadd       v5.4s, v12.4s, v13.4s          \n"

                        "ld1        {v8.4s}, [%2], #16             \n" // v8  = r0
                        "ld1        {v9.4s}, [%3], #16             \n" // v9  = r1

                        "fmul       v12.4s, v8.4s, %12.4s          \n"
                        "fmul       v13.4s, v9.4s, %13.4s          \n"

                        "ld1        {v10.4s}, [%4], #16            \n" // v10 = r2
                        "ld1        {v11.4s}, [%5], #16            \n" // v11 = r3

                        "fmla       v12.4s, v10.4s, %14.4s         \n"
                        "fmla       v13.4s, v11.4s, %15.4s         \n"

                        "fadd       v6.4s, v12.4s, v13.4s          \n"

                        "ld1        {v8.4s}, [%2], #16             \n" // v8  = r0
                        "ld1        {v9.4s}, [%3], #16             \n" // v9  = r1

                        "fmul       v12.4s, v8.4s, %12.4s          \n"
                        "fmul       v13.4s, v9.4s, %13.4s          \n"

                        "ld1        {v10.4s}, [%4], #16            \n" // v10 = r2
                        "ld1        {v11.4s}, [%5], #16            \n" // v11 = r3

                        "fmla       v12.4s, v10.4s, %14.4s         \n"
                        "fmla       v13.4s, v11.4s, %15.4s         \n"

                        "fadd       v14.4s, v12.4s, v13.4s         \n"
                        "faddp      v5.4s, v5.4s, v6.4s            \n" // Move to here to enhance ILP

                        "ld1        {v8.4s}, [%2], #16             \n" // v8  = r0
                        "ld1        {v9.4s}, [%3], #16             \n" // v9  = r1

                        "fmul       v12.4s, v8.4s, %12.4s          \n"
                        "fmul       v13.4s, v9.4s, %13.4s          \n"

                        "ld1        {v10.4s}, [%4], #16            \n" // v10 = r2
                        "ld1        {v11.4s}, [%5], #16            \n" // v11 = r3

                        "fmla       v12.4s, v10.4s, %14.4s         \n"
                        "fmla       v13.4s, v11.4s, %15.4s         \n"

                        "fadd       v15.4s, v12.4s, v13.4s         \n"

                        //                  "faddp      v5.4s ,  v5.4s,  v6.4s         \n"  // Move this line upward.
                        "faddp      v14.4s, v14.4s, v15.4s         \n"
                        "faddp      v5.4s ,  v5.4s, v14.4s         \n"

                        "fadd       v7.4s, v7.4s, v5.4s            \n"

                        "st1        {v7.4s}, [%1], #16             \n"

                        "prfm       pldl1keep, [%1, #128]          \n"

                        "subs       %w0, %w0, #1                   \n"
                        "bne        0b                             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k0123),    // %12
                        "w"(_k4567),    // %13
                        "w"(_k891011),  // %14
                        "w"(_k12131415) // %15
                        : "cc", "memory", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(

                        "pld        [%1, #128]          \n"

                        "0:                             \n"

                        "pld        [%2, #512]          \n"
                        "pld        [%3, #512]          \n"

                        "vld1.f32   {d14-d15}, [%1]     \n" // q7 = outptr

                        "vld1.f32   {d16-d17}, [%2]!    \n" // q8  = r0
                        "vld1.f32   {d18-d19}, [%3]!    \n" // q9  = r1

                        "pld        [%4, #512]          \n"
                        "pld        [%5, #512]          \n"

                        "vmul.f32   q12, q8, %q12       \n"
                        "vmul.f32   q13, q9, %q13       \n"

                        "vld1.f32   {d20-d21}, [%4]!    \n" // q10 = r2
                        "vld1.f32   {d22-d23}, [%5]!    \n" // q11 = r3

                        "vmla.f32   q12, q10, %q14      \n"
                        "vmla.f32   q13, q11, %q15      \n"

                        "vadd.f32   q5, q12, q13        \n"

                        "vld1.f32   {d16-d17}, [%2]!    \n" // q8  = r0
                        "vld1.f32   {d18-d19}, [%3]!    \n" // q9  = r1

                        "vmul.f32   q12, q8, %q12       \n"
                        "vmul.f32   q13, q9, %q13       \n"

                        "vld1.f32   {d20-d21}, [%4]!    \n" // q10 = r2
                        "vld1.f32   {d22-d23}, [%5]!    \n" // q11 = r3

                        "vmla.f32   q12, q10, %q14      \n"
                        "vmla.f32   q13, q11, %q15      \n"

                        "vadd.f32   q6, q12, q13        \n"

                        "vld1.f32   {d16-d17}, [%2]!    \n" // q8  = r0
                        "vld1.f32   {d18-d19}, [%3]!    \n" // q9  = r1

                        "vmul.f32   q12, q8, %q12       \n"
                        "vmul.f32   q13, q9, %q13       \n"

                        "vld1.f32   {d20-d21}, [%4]!    \n" // q10 = r2
                        "vld1.f32   {d22-d23}, [%5]!    \n" // q11 = r3

                        "vmla.f32   q12, q10, %q14      \n"
                        "vmla.f32   q13, q11, %q15      \n"

                        "vadd.f32   q14, q12, q13       \n"

                        "vld1.f32   {d16-d17}, [%2]!    \n" // q8  = r0
                        "vld1.f32   {d18-d19}, [%3]!    \n" // q9  = r1

                        "vmul.f32   q12, q8, %q12       \n"
                        "vmul.f32   q13, q9, %q13       \n"

                        "vld1.f32   {d20-d21}, [%4]!    \n" // q10 = r2
                        "vld1.f32   {d22-d23}, [%5]!    \n" // q11 = r3

                        "vmla.f32   q12, q10, %q14      \n"
                        "vmla.f32   q13, q11, %q15      \n"

                        "vadd.f32   q15, q12, q13       \n"

                        "vadd.f32   d10, d10, d11       \n"
                        "vadd.f32   d28, d28, d29       \n"
                        "vadd.f32   d11, d12, d13       \n"
                        "vadd.f32   d29, d30, d31       \n"

                        "vpadd.f32  d10, d10, d11       \n"
                        "vpadd.f32  d11, d28, d29       \n"

                        "vadd.f32   q7, q7, q5          \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "pld        [%1, #128]          \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k0123),    // %12
                        "w"(_k4567),    // %13
                        "w"(_k891011),  // %14
                        "w"(_k12131415) // %15
                        : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    float sum = 0.f;

                    asm volatile(
                        "ld1        {v8.4s}, [%0], #16             \n" // v8  = r0
                        "ld1        {v9.4s}, [%1], #16             \n" // v9  = r1

                        "fmul       v12.4s, v8.4s, %9.4s           \n"
                        "fmul       v13.4s, v9.4s, %10.4s          \n"

                        "ld1        {v10.4s}, [%2], #16            \n" // v10 = r2
                        "ld1        {v11.4s}, [%3], #16            \n" // v11 = r3

                        "fmla       v12.4s, v10.4s, %11.4s         \n"
                        "fmla       v13.4s, v11.4s, %12.4s         \n"

                        "fadd       v5.4s, v12.4s, v13.4s          \n"
                        "faddp      v5.4s, v5.4s, v5.4s            \n"
                        "faddp      s5, v5.2s                      \n"
                        "fmov       %w4, s5                        \n"
                        : "=r"(r0), // %0
                        "=r"(r1), // %1
                        "=r"(r2), // %2
                        "=r"(r3), // %3
                        "=r"(sum) // %4
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "w"(_k0123),    // %9
                        "w"(_k4567),    // %10
                        "w"(_k891011),  // %11
                        "w"(_k12131415) // %12
                        : "cc", "memory", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13");

                    *outptr += sum;
#else
                    float sum = 0.f;

                    asm volatile(
                        "vld1.f32   {d16-d17}, [%0]!    \n" // q8  = r0
                        "vld1.f32   {d18-d19}, [%1]!    \n" // q9  = r1

                        "vmul.f32   q12, q8, %q9        \n"
                        "vmul.f32   q13, q9, %q10       \n"

                        "vld1.f32   {d20-d21}, [%2]!    \n" // q10 = r2
                        "vld1.f32   {d22-d23}, [%3]!    \n" // q11 = r3

                        "vmla.f32   q12, q10, %q11      \n"
                        "vmla.f32   q13, q11, %q12      \n"

                        "vadd.f32   q5, q12, q13        \n"
                        "vadd.f32   d10, d10, d11       \n"
                        "vpadd.f32  d10, d10, d10       \n"
                        "vmov.f32   %4, d10[0]          \n"
                        : "=r"(r0), // %0
                        "=r"(r1), // %1
                        "=r"(r2), // %2
                        "=r"(r3), // %3
                        "=r"(sum) // %4
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "w"(_k0123),    // %9
                        "w"(_k4567),    // %10
                        "w"(_k891011),  // %11
                        "w"(_k12131415) // %12
                        : "cc", "memory", "q5", "q6", "q8", "q9", "q10", "q11", "q12", "q13");

                    *outptr += sum;
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];

                    *outptr += sum;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
#endif // __ARM_NEON
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }
        }
    }
}
