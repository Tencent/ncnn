// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

static void convdw3x3s1_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const signed char* kernel = (const signed char*)_kernel + p * 9;

        int* outptr0 = out;
        int* outptr0n = outptr0 + outw;

        const signed char* img0 = bottom_blob.channel(p);

        const signed char* r0 = img0;
        const signed char* r1 = img0 + w;
        const signed char* r2 = img0 + w * 2;
        const signed char* r3 = img0 + w * 3;

        int i = 0;

#if __ARM_NEON
        int8x16_t _k0123456789x = vld1q_s8(kernel);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON

        for (; i + 1 < outh; i += 2)
        {
#if __ARM_NEON
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "0:                                   \n"
                    "ld1    {v4.8b, v5.8b}, [%3]          \n"
                    "ld1    {v6.8b, v7.8b}, [%4]          \n"
                    "ld1    {v8.8b, v9.8b}, [%5]          \n"
                    "ld1    {v10.8b, v11.8b}, [%6]        \n"
                    "add    %3, %3, #8                    \n"
                    "add    %4, %4, #8                    \n"
                    "add    %5, %5, #8                    \n"
                    "add    %6, %6, #8                    \n"

                    "ext    v12.8b, v4.8b, v5.8b, #1      \n"
                    "ext    v13.8b, v4.8b, v5.8b, #2      \n"
                    "ext    v14.8b, v6.8b, v7.8b, #1      \n"
                    "ext    v15.8b, v6.8b, v7.8b, #2      \n"
                    "ext    v16.8b, v8.8b, v9.8b, #1      \n"
                    "ext    v17.8b, v8.8b, v9.8b, #2      \n"
                    "ext    v18.8b, v10.8b, v11.8b, #1    \n"
                    "ext    v19.8b, v10.8b, v11.8b, #2    \n"

                    "sshll  v4.8h, v4.8b, #0              \n" // r00
                    "sshll  v12.8h, v12.8b, #0            \n" // r01
                    "sshll  v13.8h, v13.8b, #0            \n" // r02
                    "sshll  v6.8h, v6.8b, #0              \n" // r10
                    "sshll  v14.8h, v14.8b, #0            \n" // r11
                    "sshll  v15.8h, v15.8b, #0            \n" // r12
                    "sshll  v8.8h, v8.8b, #0              \n" // r20
                    "sshll  v16.8h, v16.8b, #0            \n" // r21
                    "sshll  v17.8h, v17.8b, #0            \n" // r22
                    "sshll  v10.8h, v10.8b, #0            \n" // r30
                    "sshll  v18.8h, v18.8b, #0            \n" // r31
                    "sshll  v19.8h, v19.8b, #0            \n" // r32

                    // r0
                    "smull  v20.4s, v4.4h, %14.h[0]       \n" // (r00 - r07) * k00
                    "smull2  v21.4s, v4.8h, %14.h[0]      \n"
                    "smull  v22.4s, v12.4h, %14.h[1]      \n" // (r01 - r08) * k01
                    "smull2  v23.4s, v12.8h, %14.h[1]     \n"
                    "smull  v24.4s, v13.4h, %14.h[2]      \n" // (r02 - r09) * k02
                    "smull2  v25.4s, v13.8h, %14.h[2]     \n"

                    // r1
                    "smull  v26.4s, v6.4h, %14.h[0]       \n" // (r10 - r17) * k00
                    "smull2  v27.4s, v6.8h, %14.h[0]      \n"
                    "smull  v28.4s, v14.4h, %14.h[1]      \n" // (r11 - r18) * k01
                    "smull2  v29.4s, v14.8h, %14.h[1]     \n"
                    "smull  v30.4s, v15.4h, %14.h[2]      \n" // (r12 - r19) * k02
                    "smull2  v31.4s, v15.8h, %14.h[2]     \n"

                    "smlal  v20.4s, v6.4h, %14.h[3]       \n" // (r10 - r17) * k03
                    "smlal2  v21.4s, v6.8h, %14.h[3]      \n"
                    "smlal  v22.4s, v14.4h, %15.h[0]      \n" // (r11 - r18) * k04
                    "smlal2  v23.4s, v14.8h, %15.h[0]     \n"
                    "smlal  v24.4s, v15.4h, %15.h[1]      \n" // (r12 - r19) * k05
                    "smlal2  v25.4s, v15.8h, %15.h[1]     \n"

                    // r2
                    "smlal  v26.4s, v8.4h, %14.h[3]       \n" // (r20 - r27) * k03
                    "smlal2  v27.4s, v8.8h, %14.h[3]      \n"
                    "smlal  v28.4s, v16.4h, %15.h[0]      \n" // (r21 - r28) * k04
                    "smlal2  v29.4s, v16.8h, %15.h[0]     \n"
                    "smlal  v30.4s, v17.4h, %15.h[1]      \n" // (r22 - r29) * k05
                    "smlal2  v31.4s, v17.8h, %15.h[1]     \n"

                    "smlal  v20.4s, v8.4h, %15.h[2]       \n" // (r20 - r27) * k06
                    "smlal2  v21.4s, v8.8h, %15.h[2]      \n"
                    "smlal  v22.4s, v16.4h, %15.h[3]      \n" // (r21 - r28) * k07
                    "smlal2  v23.4s, v16.8h, %15.h[3]     \n"
                    "smlal  v24.4s, v17.4h, %16.h[0]      \n" // (r22 - r29) * k08
                    "smlal2  v25.4s, v17.8h, %16.h[0]     \n"

                    // r3
                    "smlal  v26.4s, v10.4h, %15.h[2]      \n" // (r30 - r37) * k06
                    "smlal2  v27.4s, v10.8h, %15.h[2]     \n"
                    "smlal  v28.4s, v18.4h, %15.h[3]      \n" // (r31 - r38) * k07
                    "smlal2  v29.4s, v18.8h, %15.h[3]     \n"
                    "smlal  v30.4s, v19.4h, %16.h[0]      \n" // (r32 - r39) * k08
                    "smlal2  v31.4s, v19.8h, %16.h[0]     \n"

                    // add and save
                    "add    v20.4s, v20.4s, v22.4s        \n"
                    "add    v21.4s, v21.4s, v23.4s        \n"
                    "add    v26.4s, v26.4s, v28.4s        \n"
                    "add    v27.4s, v27.4s, v29.4s        \n"
                    "add    v20.4s, v20.4s, v24.4s        \n"
                    "add    v21.4s, v21.4s, v25.4s        \n"
                    "add    v26.4s, v26.4s, v30.4s        \n"
                    "add    v27.4s, v27.4s, v31.4s        \n"

                    "st1    {v20.4s, v21.4s}, [%1], #32   \n"
                    "st1    {v26.4s, v27.4s}, [%2], #32   \n"

                    "subs   %w0, %w0, #1                  \n"
                    "bne    0b                            \n"

                    : "=r"(nn),       // %0
                    "=r"(outptr0),  // %1
                    "=r"(outptr0n), // %2
                    "=r"(r0),       // %3
                    "=r"(r1),       // %4
                    "=r"(r2),       // %5
                    "=r"(r3)        // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr0n),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2),
                    "6"(r3),
                    "w"(_k0123), // %14
                    "w"(_k4567), // %15
                    "w"(_k8xxx)  // %16
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                              \n"
                    // r0
                    "vld1.s8    {d30-d31}, [%3]      \n" // r0
                    "add    %3, %3, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r00
                    "vmovl.s8    q5, d10             \n" // r01
                    "vmovl.s8    q6, d12             \n" // r02
                    // sum0
                    "vmull.s16  q7, d30, %P14[0]     \n" // (r00 - r07) * k00
                    "vmull.s16  q8, d31, %P14[0]     \n"
                    "vmull.s16  q9, d10, %P14[1]     \n" // (r01 - r08) * k01
                    "vmull.s16  q10, d11, %P14[1]    \n"
                    "vmlal.s16  q7, d12, %P14[2]     \n" // (r02 - r09) * k02
                    "vmlal.s16  q8, d13, %P14[2]     \n"

                    // r1
                    "vld1.s8    {d30-d31}, [%4]      \n" // r1
                    "add    %4, %4, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r10
                    "vmovl.s8    q5, d10             \n" // r11
                    "vmovl.s8    q6, d12             \n" // r12
                    // sum0
                    "vmlal.s16  q7, d30, %P14[3]     \n" // (r10 - r17) * k03
                    "vmlal.s16  q8, d31, %P14[3]     \n"
                    "vmlal.s16  q9, d10, %P15[0]     \n" // (r11 - r18) * k04
                    "vmlal.s16  q10, d11, %P15[0]    \n"
                    "vmlal.s16  q7, d12, %P15[1]     \n" // (r12 - r19) * k05
                    "vmlal.s16  q8, d13, %P15[1]     \n"
                    // sum1
                    "vmull.s16  q11, d30, %P14[0]    \n" // (r10 - r17) * k00
                    "vmull.s16  q12, d31, %P14[0]    \n"
                    "vmull.s16  q13, d10, %P14[1]    \n" // (r11 - r18) * k01
                    "vmull.s16  q14, d11, %P14[1]    \n"
                    "vmlal.s16  q11, d12, %P14[2]    \n" // (r12 - r19) * k02
                    "vmlal.s16  q12, d13, %P14[2]    \n"

                    // r2
                    "vld1.s8    {d30-d31}, [%5]      \n" // r2
                    "add    %5, %5, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r20
                    "vmovl.s8    q5, d10             \n" // r21
                    "vmovl.s8    q6, d12             \n" // r22

                    // sum0
                    "vmlal.s16  q7, d30, %P15[2]     \n" // (r20 - r27) * k06
                    "vmlal.s16  q8, d31, %P15[2]     \n"
                    "vmlal.s16  q9, d10, %P15[3]     \n" // (r21 - r28) * k07
                    "vmlal.s16  q10, d11, %P15[3]    \n"
                    "vmlal.s16  q7, d12, %P16[0]     \n" // (r22 - r29) * k08
                    "vmlal.s16  q8, d13, %P16[0]     \n"
                    // sum1
                    "vmlal.s16  q11, d30, %P14[3]    \n" // (r20 - r27) * k03
                    "vmlal.s16  q12, d31, %P14[3]    \n"
                    "vmlal.s16  q13, d10, %P15[0]    \n" // (r21 - r28) * k04
                    "vmlal.s16  q14, d11, %P15[0]    \n"
                    "vmlal.s16  q11, d12, %P15[1]    \n" // (r22 - r29) * k05
                    "vmlal.s16  q12, d13, %P15[1]    \n"

                    // r3
                    "vld1.s8    {d30-d31}, [%6]      \n" // r3
                    "add    %6, %6, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r30
                    "vmovl.s8    q5, d10             \n" // r31
                    "vmovl.s8    q6, d12             \n" // r32

                    // sum1
                    "vmlal.s16  q11, d30, %P15[2]    \n" // (r30 - r37) * k06
                    "vmlal.s16  q12, d31, %P15[2]    \n"
                    "vmlal.s16  q13, d10, %P15[3]    \n" // (r31 - r38) * k07
                    "vmlal.s16  q14, d11, %P15[3]    \n"
                    "vmlal.s16  q11, d12, %P16[0]    \n" // (r32 - r39) * k08
                    "vmlal.s16  q12, d13, %P16[0]    \n"

                    "subs   %0, %0, #1               \n"

                    // add and save
                    "vadd.s32    q7, q7, q9          \n"
                    "vadd.s32    q8, q8, q10         \n"
                    "vadd.s32    q11, q11, q13       \n"
                    "vadd.s32    q12, q12, q14       \n"

                    "vst1.s32    {d14-d17}, [%1]!    \n"
                    "vst1.s32    {d22-d25}, [%2]!    \n"

                    "bne    0b                       \n"

                    : "=r"(nn),       // %0
                    "=r"(outptr0),  // %1
                    "=r"(outptr0n), // %2
                    "=r"(r0),       // %3
                    "=r"(r1),       // %4
                    "=r"(r2),       // %5
                    "=r"(r3)        // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr0n),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2),
                    "6"(r3),
                    "w"(_k0123), // %14
                    "w"(_k4567), // %15
                    "w"(_k8xxx)  // %16
                    : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                // TODO NEON
                int sum0 = 0;
                int sum0n = 0;

                sum0 += (int)r0[0] * kernel[0];
                sum0 += (int)r0[1] * kernel[1];
                sum0 += (int)r0[2] * kernel[2];
                sum0 += (int)r1[0] * kernel[3];
                sum0 += (int)r1[1] * kernel[4];
                sum0 += (int)r1[2] * kernel[5];
                sum0 += (int)r2[0] * kernel[6];
                sum0 += (int)r2[1] * kernel[7];
                sum0 += (int)r2[2] * kernel[8];

                sum0n += (int)r1[0] * kernel[0];
                sum0n += (int)r1[1] * kernel[1];
                sum0n += (int)r1[2] * kernel[2];
                sum0n += (int)r2[0] * kernel[3];
                sum0n += (int)r2[1] * kernel[4];
                sum0n += (int)r2[2] * kernel[5];
                sum0n += (int)r3[0] * kernel[6];
                sum0n += (int)r3[1] * kernel[7];
                sum0n += (int)r3[2] * kernel[8];

                *outptr0 = sum0;
                *outptr0n = sum0n;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr0n++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr0n += outw;
        }

        for (; i < outh; i++)
        {
#if __ARM_NEON
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "0:                                   \n"
                    "ld1    {v4.8b, v5.8b}, [%2]          \n"
                    "ld1    {v6.8b, v7.8b}, [%3]          \n"
                    "ld1    {v8.8b, v9.8b}, [%4]          \n"
                    "add    %2, %2, #8                    \n"
                    "add    %3, %3, #8                    \n"
                    "add    %4, %4, #8                    \n"

                    "ext    v12.8b, v4.8b, v5.8b, #1      \n"
                    "ext    v13.8b, v4.8b, v5.8b, #2      \n"
                    "ext    v14.8b, v6.8b, v7.8b, #1      \n"
                    "ext    v15.8b, v6.8b, v7.8b, #2      \n"
                    "ext    v16.8b, v8.8b, v9.8b, #1      \n"
                    "ext    v17.8b, v8.8b, v9.8b, #2      \n"

                    "sshll  v4.8h, v4.8b, #0              \n" // r00
                    "sshll  v12.8h, v12.8b, #0            \n" // r01
                    "sshll  v13.8h, v13.8b, #0            \n" // r02
                    "sshll  v6.8h, v6.8b, #0              \n" // r10
                    "sshll  v14.8h, v14.8b, #0            \n" // r11
                    "sshll  v15.8h, v15.8b, #0            \n" // r12
                    "sshll  v8.8h, v8.8b, #0              \n" // r20
                    "sshll  v16.8h, v16.8b, #0            \n" // r21
                    "sshll  v17.8h, v17.8b, #0            \n" // r22

                    // r0
                    "smull  v20.4s, v4.4h, %10.h[0]       \n" // (r00 - r07) * k00
                    "smull2  v21.4s, v4.8h, %10.h[0]      \n"
                    "smull  v22.4s, v12.4h, %10.h[1]      \n" // (r01 - r08) * k01
                    "smull2  v23.4s, v12.8h, %10.h[1]     \n"
                    "smull  v24.4s, v13.4h, %10.h[2]      \n" // (r02 - r09) * k02
                    "smull2  v25.4s, v13.8h, %10.h[2]     \n"

                    // r1
                    "smlal  v20.4s, v6.4h, %10.h[3]       \n" // (r10 - r17) * k03
                    "smlal2  v21.4s, v6.8h, %10.h[3]      \n"
                    "smlal  v22.4s, v14.4h, %11.h[0]      \n" // (r11 - r18) * k04
                    "smlal2  v23.4s, v14.8h, %11.h[0]     \n"
                    "smlal  v24.4s, v15.4h, %11.h[1]      \n" // (r12 - r19) * k05
                    "smlal2  v25.4s, v15.8h, %11.h[1]     \n"

                    // r2
                    "smlal  v20.4s, v8.4h, %11.h[2]       \n" // (r20 - r27) * k06
                    "smlal2  v21.4s, v8.8h, %11.h[2]      \n"
                    "smlal  v22.4s, v16.4h, %11.h[3]      \n" // (r21 - r28) * k07
                    "smlal2  v23.4s, v16.8h, %11.h[3]     \n"
                    "smlal  v24.4s, v17.4h, %12.h[0]      \n" // (r22 - r29) * k08
                    "smlal2  v25.4s, v17.8h, %12.h[0]     \n"

                    // add and save
                    "add    v20.4s, v20.4s, v22.4s        \n"
                    "add    v21.4s, v21.4s, v23.4s        \n"
                    "add    v20.4s, v20.4s, v24.4s        \n"
                    "add    v21.4s, v21.4s, v25.4s        \n"

                    "st1    {v20.4s, v21.4s}, [%1], #32   \n"

                    "subs   %w0, %w0, #1                  \n"
                    "bne    0b                            \n"

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
                    "w"(_k0123), // %10
                    "w"(_k4567), // %11
                    "w"(_k8xxx)  // %12
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                              \n"
                    // r0
                    "vld1.s8    {d30-d31}, [%2]        \n" // r0
                    "add    %2, %2, #8               \n"

                    "vext.s8    d10, d30, d31, #1      \n"
                    "vext.s8    d12, d30, d31, #2      \n"

                    "vmovl.s8    q15, d30              \n" // r00
                    "vmovl.s8    q5, d10             \n"   // r01
                    "vmovl.s8    q6, d12             \n"   // r02
                    // sum0
                    "vmull.s16  q7, d30, %P10[0]      \n" // (r00 - r07) * k00
                    "vmull.s16  q8, d31, %P10[0]      \n"
                    "vmull.s16  q9, d10, %P10[1]     \n" // (r01 - r08) * k01
                    "vmull.s16  q10, d11, %P10[1]    \n"
                    "vmlal.s16  q7, d12, %P10[2]     \n" // (r02 - r09) * k02
                    "vmlal.s16  q8, d13, %P10[2]     \n"

                    // r1
                    "vld1.s8    {d30-d31}, [%3]        \n" // r1
                    "add    %3, %3, #8               \n"

                    "vext.s8    d10, d30, d31, #1      \n"
                    "vext.s8    d12, d30, d31, #2      \n"

                    "vmovl.s8    q15, d30              \n" // r10
                    "vmovl.s8    q5, d10             \n"   // r11
                    "vmovl.s8    q6, d12             \n"   // r12
                    // sum0
                    "vmlal.s16  q7, d30, %P10[3]      \n" // (r10 - r17) * k03
                    "vmlal.s16  q8, d31, %P10[3]      \n"
                    "vmlal.s16  q9, d10, %P11[0]     \n" // (r11 - r18) * k04
                    "vmlal.s16  q10, d11, %P11[0]    \n"
                    "vmlal.s16  q7, d12, %P11[1]     \n" // (r12 - r19) * k05
                    "vmlal.s16  q8, d13, %P11[1]     \n"

                    // r2
                    "vld1.s8    {d30-d31}, [%4]        \n" // r2
                    "add    %4, %4, #8               \n"

                    "vext.s8    d10, d30, d31, #1      \n"
                    "vext.s8    d12, d30, d31, #2      \n"

                    "vmovl.s8    q15, d30              \n" // r20
                    "vmovl.s8    q5, d10             \n"   // r21
                    "vmovl.s8    q6, d12             \n"   // r22

                    // sum0
                    "vmlal.s16  q7, d30, %P11[2]      \n" // (r20 - r27) * k06
                    "vmlal.s16  q8, d31, %P11[2]      \n"
                    "vmlal.s16  q9, d10, %P11[3]     \n" // (r21 - r28) * k07
                    "vmlal.s16  q10, d11, %P11[3]    \n"
                    "vmlal.s16  q7, d12, %P12[0]     \n" // (r22 - r29) * k08
                    "vmlal.s16  q8, d13, %P12[0]     \n"

                    "subs   %0, %0, #1               \n"

                    // add and save
                    "vadd.s32    q7, q7, q9          \n"
                    "vadd.s32    q8, q8, q10         \n"

                    "vst1.s32    {d14-d17}, [%1]!    \n"

                    "bne    0b                       \n"

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
                    "w"(_k0123), // %10
                    "w"(_k4567), // %11
                    "w"(_k8xxx)  // %12
                    : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * kernel[0];
                sum += (int)r0[1] * kernel[1];
                sum += (int)r0[2] * kernel[2];
                sum += (int)r1[0] * kernel[3];
                sum += (int)r1[1] * kernel[4];
                sum += (int)r1[2] * kernel[5];
                sum += (int)r2[0] * kernel[6];
                sum += (int)r2[1] * kernel[7];
                sum += (int)r2[2] * kernel[8];

                *outptr0 = sum;

                r0++;
                r1++;
                r2++;
                outptr0++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static void convdw3x3s2_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const signed char* kernel = (const signed char*)_kernel + p * 9;

        int* outptr = out;

        const signed char* img = bottom_blob.channel(p);

        const signed char* r0 = img;
        const signed char* r1 = img + w;
        const signed char* r2 = img + w * 2;

        int i = 0;
#if __ARM_NEON
        int8x16_t _k0123456789x = vld1q_s8(kernel);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON
        for (; i < outh; i++)
        {
#if __ARM_NEON
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "0:                                   \n"
                    "ld2    {v4.8b, v5.8b}, [%2], #16     \n"
                    "ld2    {v6.8b, v7.8b}, [%2]          \n"
                    "ld2    {v8.8b, v9.8b}, [%3], #16     \n"
                    "ld2    {v10.8b, v11.8b}, [%3]        \n"
                    "ld2    {v12.8b, v13.8b}, [%4], #16   \n"
                    "ld2    {v14.8b, v15.8b}, [%4]        \n"

                    "ext    v6.8b, v4.8b, v6.8b, #1       \n"
                    "ext    v10.8b, v8.8b, v10.8b, #1     \n"
                    "ext    v14.8b, v12.8b, v14.8b, #1    \n"

                    "sshll  v4.8h, v4.8b, #0              \n" // r00
                    "sshll  v5.8h, v5.8b, #0              \n" // r01
                    "sshll  v6.8h, v6.8b, #0              \n" // r02
                    "sshll  v8.8h, v8.8b, #0              \n" // r10
                    "sshll  v9.8h, v9.8b, #0              \n" // r11
                    "sshll  v10.8h, v10.8b, #0            \n" // r12
                    "sshll  v12.8h, v12.8b, #0            \n" // r20
                    "sshll  v13.8h, v13.8b, #0            \n" // r21
                    "sshll  v14.8h, v14.8b, #0            \n" // r22

                    // r0
                    "smull  v20.4s, v4.4h, %10.h[0]       \n" // (r00 - r07) * k00
                    "smull2  v21.4s, v4.8h, %10.h[0]      \n"
                    "smull  v22.4s, v5.4h, %10.h[1]       \n" // (r01 - r08) * k01
                    "smull2  v23.4s, v5.8h, %10.h[1]      \n"
                    "smull  v24.4s, v6.4h, %10.h[2]       \n" // (r02 - r09) * k02
                    "smull2  v25.4s, v6.8h, %10.h[2]      \n"

                    // r1
                    "smlal  v20.4s, v8.4h, %10.h[3]       \n" // (r10 - r17) * k03
                    "smlal2  v21.4s, v8.8h, %10.h[3]      \n"
                    "smlal  v22.4s, v9.4h, %11.h[0]       \n" // (r11 - r18) * k04
                    "smlal2  v23.4s, v9.8h, %11.h[0]      \n"
                    "smlal  v24.4s, v10.4h, %11.h[1]      \n" // (r12 - r19) * k05
                    "smlal2  v25.4s, v10.8h, %11.h[1]     \n"

                    // r2
                    "smlal  v20.4s, v12.4h, %11.h[2]      \n" // (r20 - r27) * k06
                    "smlal2  v21.4s, v12.8h, %11.h[2]     \n"
                    "smlal  v22.4s, v13.4h, %11.h[3]      \n" // (r21 - r28) * k07
                    "smlal2  v23.4s, v13.8h, %11.h[3]     \n"
                    "smlal  v24.4s, v14.4h, %12.h[0]      \n" // (r22 - r29) * k08
                    "smlal2  v25.4s, v14.8h, %12.h[0]     \n"

                    // add and save
                    "add    v20.4s, v20.4s, v22.4s        \n"
                    "add    v21.4s, v21.4s, v23.4s        \n"
                    "add    v20.4s, v20.4s, v24.4s        \n"
                    "add    v21.4s, v21.4s, v25.4s        \n"

                    "st1    {v20.4s, v21.4s}, [%1], #32   \n"

                    "subs   %w0, %w0, #1                  \n"
                    "bne    0b                            \n"

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
                    "w"(_k4567), // %11
                    "w"(_k8xxx)  // %12
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                              \n"
                    // r0
                    "vld2.s8    {d30-d31}, [%2]!     \n" // r0
                    "vld2.s8    {d10-d11}, [%2]      \n"
                    "vext.s8    d12, d30, d10, #1    \n"

                    "vmovl.s8    q5, d31             \n" // r01
                    "vmovl.s8    q15, d30            \n" // r00
                    "vmovl.s8    q6, d12             \n" // r02
                    // sum0
                    "vmull.s16  q7, d30, %P10[0]     \n" // (r00 - r07) * k00
                    "vmull.s16  q8, d31, %P10[0]     \n"
                    "vmull.s16  q9, d10, %P10[1]     \n" // (r01 - r08) * k01
                    "vmull.s16  q10, d11, %P10[1]    \n"
                    "vmlal.s16  q7, d12, %P10[2]     \n" // (r02 - r09) * k02
                    "vmlal.s16  q8, d13, %P10[2]     \n"

                    // r1
                    "vld2.s8    {d30-d31}, [%3]!     \n" // r1
                    "vld2.s8    {d10-d11}, [%3]      \n"
                    "vext.s8    d12, d30, d10, #1    \n"

                    "vmovl.s8    q5, d31             \n" // r11
                    "vmovl.s8    q15, d30            \n" // r10
                    "vmovl.s8    q6, d12             \n" // r12
                    // sum0
                    "vmlal.s16  q7, d30, %P10[3]     \n" // (r10 - r17) * k03
                    "vmlal.s16  q8, d31, %P10[3]     \n"
                    "vmlal.s16  q9, d10, %P11[0]     \n" // (r11 - r18) * k04
                    "vmlal.s16  q10, d11, %P11[0]    \n"
                    "vmlal.s16  q7, d12, %P11[1]     \n" // (r12 - r19) * k05
                    "vmlal.s16  q8, d13, %P11[1]     \n"

                    // r2
                    "vld2.s8    {d30-d31}, [%4]!     \n" // r2
                    "vld2.s8    {d10-d11}, [%4]      \n"
                    "vext.s8    d12, d30, d10, #1    \n"

                    "vmovl.s8    q5, d31             \n" // r21
                    "vmovl.s8    q15, d30            \n" // r20
                    "vmovl.s8    q6, d12             \n" // r22

                    // sum0
                    "vmlal.s16  q7, d30, %P11[2]     \n" // (r20 - r27) * k06
                    "vmlal.s16  q8, d31, %P11[2]     \n"
                    "vmlal.s16  q9, d10, %P11[3]     \n" // (r21 - r28) * k07
                    "vmlal.s16  q10, d11, %P11[3]    \n"
                    "vmlal.s16  q7, d12, %P12[0]     \n" // (r22 - r29) * k08
                    "vmlal.s16  q8, d13, %P12[0]     \n"

                    "subs   %0, %0, #1               \n"

                    // add and save
                    "vadd.s32    q7, q7, q9          \n"
                    "vadd.s32    q8, q8, q10         \n"

                    "vst1.s32    {d14-d17}, [%1]!    \n"

                    "bne    0b                       \n"

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
                    "w"(_k4567), // %11
                    "w"(_k8xxx)  // %12
                    : "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * kernel[0];
                sum += (int)r0[1] * kernel[1];
                sum += (int)r0[2] * kernel[2];
                sum += (int)r1[0] * kernel[3];
                sum += (int)r1[1] * kernel[4];
                sum += (int)r1[2] * kernel[5];
                sum += (int)r2[0] * kernel[6];
                sum += (int)r2[1] * kernel[7];
                sum += (int)r2[2] * kernel[8];

                *outptr = sum;

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

static void convdw3x3s1_int8_requant_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, std::vector<float> scales_requant, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;
        const float scale_requant_in = scales_requant[2 * p];
        const float scale_requant_out = scales_requant[2 * p + 1];

        const signed char* kernel = (const signed char*)_kernel + p * 9;

        signed char* outptr0 = out;
        signed char* outptr0n = outptr0 + outw;

        const signed char* img0 = bottom_blob.channel(p);

        const signed char* r0 = img0;
        const signed char* r1 = img0 + w;
        const signed char* r2 = img0 + w * 2;
        const signed char* r3 = img0 + w * 3;

        int i = 0;

#if __ARM_NEON
        int8x16_t _k0123456789x = vld1q_s8(kernel);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON

        for (; i + 1 < outh; i += 2)
        {
#if __ARM_NEON
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "0:                                   \n"
                    "ld1    {v4.8b, v5.8b}, [%3]          \n"
                    "ld1    {v6.8b, v7.8b}, [%4]          \n"
                    "ld1    {v8.8b, v9.8b}, [%5]          \n"
                    "ld1    {v10.8b, v11.8b}, [%6]        \n"
                    "add    %3, %3, #8                    \n"
                    "add    %4, %4, #8                    \n"
                    "add    %5, %5, #8                    \n"
                    "add    %6, %6, #8                    \n"

                    "ext    v12.8b, v4.8b, v5.8b, #1      \n"
                    "ext    v13.8b, v4.8b, v5.8b, #2      \n"
                    "ext    v14.8b, v6.8b, v7.8b, #1      \n"
                    "ext    v15.8b, v6.8b, v7.8b, #2      \n"
                    "ext    v16.8b, v8.8b, v9.8b, #1      \n"
                    "ext    v17.8b, v8.8b, v9.8b, #2      \n"
                    "ext    v18.8b, v10.8b, v11.8b, #1    \n"
                    "ext    v19.8b, v10.8b, v11.8b, #2    \n"

                    "sshll  v4.8h, v4.8b, #0              \n" // r00
                    "sshll  v12.8h, v12.8b, #0            \n" // r01
                    "sshll  v13.8h, v13.8b, #0            \n" // r02
                    "sshll  v6.8h, v6.8b, #0              \n" // r10
                    "sshll  v14.8h, v14.8b, #0            \n" // r11
                    "sshll  v15.8h, v15.8b, #0            \n" // r12
                    "sshll  v8.8h, v8.8b, #0              \n" // r20
                    "sshll  v16.8h, v16.8b, #0            \n" // r21
                    "sshll  v17.8h, v17.8b, #0            \n" // r22
                    "sshll  v10.8h, v10.8b, #0            \n" // r30
                    "sshll  v18.8h, v18.8b, #0            \n" // r31
                    "sshll  v19.8h, v19.8b, #0            \n" // r32

                    // r0
                    "smull  v20.4s, v4.4h, %14.h[0]       \n" // (r00 - r07) * k00
                    "smull2  v21.4s, v4.8h, %14.h[0]      \n"
                    "smull  v22.4s, v12.4h, %14.h[1]      \n" // (r01 - r08) * k01
                    "smull2  v23.4s, v12.8h, %14.h[1]     \n"
                    "smull  v24.4s, v13.4h, %14.h[2]      \n" // (r02 - r09) * k02
                    "smull2  v25.4s, v13.8h, %14.h[2]     \n"

                    // r1
                    "smull  v26.4s, v6.4h, %14.h[0]       \n" // (r10 - r17) * k00
                    "smull2  v27.4s, v6.8h, %14.h[0]      \n"
                    "smull  v28.4s, v14.4h, %14.h[1]      \n" // (r11 - r18) * k01
                    "smull2  v29.4s, v14.8h, %14.h[1]     \n"
                    "smull  v30.4s, v15.4h, %14.h[2]      \n" // (r12 - r19) * k02
                    "smull2  v31.4s, v15.8h, %14.h[2]     \n"

                    "smlal  v20.4s, v6.4h, %14.h[3]       \n" // (r10 - r17) * k03
                    "smlal2  v21.4s, v6.8h, %14.h[3]      \n"
                    "smlal  v22.4s, v14.4h, %15.h[0]      \n" // (r11 - r18) * k04
                    "smlal2  v23.4s, v14.8h, %15.h[0]     \n"
                    "smlal  v24.4s, v15.4h, %15.h[1]      \n" // (r12 - r19) * k05
                    "smlal2  v25.4s, v15.8h, %15.h[1]     \n"

                    // r2
                    "smlal  v26.4s, v8.4h, %14.h[3]       \n" // (r20 - r27) * k03
                    "smlal2  v27.4s, v8.8h, %14.h[3]      \n"
                    "smlal  v28.4s, v16.4h, %15.h[0]      \n" // (r21 - r28) * k04
                    "smlal2  v29.4s, v16.8h, %15.h[0]     \n"
                    "smlal  v30.4s, v17.4h, %15.h[1]      \n" // (r22 - r29) * k05
                    "smlal2  v31.4s, v17.8h, %15.h[1]     \n"

                    "smlal  v20.4s, v8.4h, %15.h[2]       \n" // (r20 - r27) * k06
                    "smlal2  v21.4s, v8.8h, %15.h[2]      \n"
                    "smlal  v22.4s, v16.4h, %15.h[3]      \n" // (r21 - r28) * k07
                    "smlal2  v23.4s, v16.8h, %15.h[3]     \n"
                    "smlal  v24.4s, v17.4h, %16.h[0]      \n" // (r22 - r29) * k08
                    "smlal2  v25.4s, v17.8h, %16.h[0]     \n"

                    // r3
                    "smlal  v26.4s, v10.4h, %15.h[2]      \n" // (r30 - r37) * k06
                    "smlal2  v27.4s, v10.8h, %15.h[2]     \n"
                    "smlal  v28.4s, v18.4h, %15.h[3]      \n" // (r31 - r38) * k07
                    "smlal2  v29.4s, v18.8h, %15.h[3]     \n"
                    "smlal  v30.4s, v19.4h, %16.h[0]      \n" // (r32 - r39) * k08
                    "smlal2  v31.4s, v19.8h, %16.h[0]     \n"

                    // add and save
                    "add    v20.4s, v20.4s, v22.4s        \n"
                    "add    v21.4s, v21.4s, v23.4s        \n"
                    "add    v26.4s, v26.4s, v28.4s        \n"
                    "add    v27.4s, v27.4s, v29.4s        \n"
                    "add    v20.4s, v20.4s, v24.4s        \n"
                    "add    v21.4s, v21.4s, v25.4s        \n"
                    "add    v26.4s, v26.4s, v30.4s        \n"
                    "add    v27.4s, v27.4s, v31.4s        \n"

                    "dup    v4.4s, %w17                   \n" // bias
                    "dup    v5.4s, %w18                   \n" // scale_in
                    "dup    v6.4s, %w19                   \n" // scale_out

                    // top_s32 -> top_f32
                    "scvtf  v20.4s, v20.4s                 \n"
                    "scvtf  v21.4s, v21.4s                 \n"
                    "scvtf  v26.4s, v26.4s                 \n"
                    "scvtf  v27.4s, v27.4s                 \n"

                    // top_f32 = top_f32 * scale_in
                    "fmul   v20.4s, v20.4s, v5.4s          \n"
                    "fmul   v21.4s, v21.4s, v5.4s          \n"
                    "fmul   v26.4s, v26.4s, v5.4s          \n"
                    "fmul   v27.4s, v27.4s, v5.4s          \n"
                    // top_f32 = top_f32 + bias
                    "fadd   v20.4s, v20.4s, v4.4s          \n"
                    "fadd   v21.4s, v21.4s, v4.4s          \n"
                    "fadd   v26.4s, v26.4s, v4.4s          \n"
                    "fadd   v27.4s, v27.4s, v4.4s          \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v20.4s, v20.4s, v6.4s          \n"
                    "fmul   v21.4s, v21.4s, v6.4s          \n"
                    "fmul   v26.4s, v26.4s, v6.4s          \n"
                    "fmul   v27.4s, v27.4s, v6.4s          \n"
                    // top_f32 -> top_s32
                    "fcvtas v20.4s, v20.4s                 \n"
                    "fcvtas v21.4s, v21.4s                 \n"
                    "fcvtas v26.4s, v26.4s                 \n"
                    "fcvtas v27.4s, v27.4s                 \n"
                    // top_s32 -> top_s16
                    "sqxtn  v7.4h, v20.4s                 \n"
                    "sqxtn  v9.4h, v26.4s                 \n"
                    "sqxtn2 v7.8h, v21.4s                 \n"
                    "sqxtn2 v9.8h, v27.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v8.8b, v7.8h                  \n"
                    "sqxtn  v10.8b, v9.8h                 \n"
                    // save top_s8
                    "st1    {v8.8b}, [%1], #8             \n"
                    "st1    {v10.8b}, [%2], #8            \n"

                    "subs   %w0, %w0, #1                  \n"
                    "bne    0b                            \n"

                    : "=r"(nn),       // %0
                    "=r"(outptr0),  // %1
                    "=r"(outptr0n), // %2
                    "=r"(r0),       // %3
                    "=r"(r1),       // %4
                    "=r"(r2),       // %5
                    "=r"(r3)        // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr0n),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2),
                    "6"(r3),
                    "w"(_k0123),           // %14
                    "w"(_k4567),           // %15
                    "w"(_k8xxx),           // %16
                    "r"(bias0),            // %17
                    "r"(scale_requant_in), // %18
                    "r"(scale_requant_out) // %19
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                              \n"
                    // r0
                    "vld1.s8    {d30-d31}, [%3]      \n" // r0
                    "add    %3, %3, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r00
                    "vmovl.s8    q5, d10             \n" // r01
                    "vmovl.s8    q6, d12             \n" // r02
                    // sum0
                    "vmull.s16  q7, d30, %P14[0]     \n" // (r00 - r07) * k00
                    "vmull.s16  q8, d31, %P14[0]     \n"
                    "vmull.s16  q9, d10, %P14[1]     \n" // (r01 - r08) * k01
                    "vmull.s16  q10, d11, %P14[1]    \n"
                    "vmlal.s16  q7, d12, %P14[2]     \n" // (r02 - r09) * k02
                    "vmlal.s16  q8, d13, %P14[2]     \n"

                    // r1
                    "vld1.s8    {d30-d31}, [%4]      \n" // r1
                    "add    %4, %4, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r10
                    "vmovl.s8    q5, d10             \n" // r11
                    "vmovl.s8    q6, d12             \n" // r12
                    // sum0
                    "vmlal.s16  q7, d30, %P14[3]     \n" // (r10 - r17) * k03
                    "vmlal.s16  q8, d31, %P14[3]     \n"
                    "vmlal.s16  q9, d10, %P15[0]     \n" // (r11 - r18) * k04
                    "vmlal.s16  q10, d11, %P15[0]    \n"
                    "vmlal.s16  q7, d12, %P15[1]     \n" // (r12 - r19) * k05
                    "vmlal.s16  q8, d13, %P15[1]     \n"
                    // sum1
                    "vmull.s16  q11, d30, %P14[0]    \n" // (r10 - r17) * k00
                    "vmull.s16  q12, d31, %P14[0]    \n"
                    "vmull.s16  q13, d10, %P14[1]    \n" // (r11 - r18) * k01
                    "vmull.s16  q14, d11, %P14[1]    \n"
                    "vmlal.s16  q11, d12, %P14[2]    \n" // (r12 - r19) * k02
                    "vmlal.s16  q12, d13, %P14[2]    \n"

                    // r2
                    "vld1.s8    {d30-d31}, [%5]      \n" // r2
                    "add    %5, %5, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r20
                    "vmovl.s8    q5, d10             \n" // r21
                    "vmovl.s8    q6, d12             \n" // r22

                    // sum0
                    "vmlal.s16  q7, d30, %P15[2]     \n" // (r20 - r27) * k06
                    "vmlal.s16  q8, d31, %P15[2]     \n"
                    "vmlal.s16  q9, d10, %P15[3]     \n" // (r21 - r28) * k07
                    "vmlal.s16  q10, d11, %P15[3]    \n"
                    "vmlal.s16  q7, d12, %P16[0]     \n" // (r22 - r29) * k08
                    "vmlal.s16  q8, d13, %P16[0]     \n"
                    // sum1
                    "vmlal.s16  q11, d30, %P14[3]    \n" // (r20 - r27) * k03
                    "vmlal.s16  q12, d31, %P14[3]    \n"
                    "vmlal.s16  q13, d10, %P15[0]    \n" // (r21 - r28) * k04
                    "vmlal.s16  q14, d11, %P15[0]    \n"
                    "vmlal.s16  q11, d12, %P15[1]    \n" // (r22 - r29) * k05
                    "vmlal.s16  q12, d13, %P15[1]    \n"

                    // r3
                    "vld1.s8    {d30-d31}, [%6]      \n" // r3
                    "add    %6, %6, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r30
                    "vmovl.s8    q5, d10             \n" // r31
                    "vmovl.s8    q6, d12             \n" // r32

                    // sum1
                    "vmlal.s16  q11, d30, %P15[2]    \n" // (r30 - r37) * k06
                    "vmlal.s16  q12, d31, %P15[2]    \n"
                    "vmlal.s16  q13, d10, %P15[3]    \n" // (r31 - r38) * k07
                    "vmlal.s16  q14, d11, %P15[3]    \n"
                    "vmlal.s16  q11, d12, %P16[0]    \n" // (r32 - r39) * k08
                    "vmlal.s16  q12, d13, %P16[0]    \n"

                    "subs   %0, %0, #1               \n"

                    // add and save
                    "vadd.s32    q7, q7, q9          \n"
                    "vadd.s32    q8, q8, q10         \n"
                    "vadd.s32    q11, q11, q13       \n"
                    "vadd.s32    q12, q12, q14       \n"

                    "vdup.f32   q13, %17             \n" // bias
                    "vdup.f32   q14, %18             \n" // scale_in
                    "vdup.f32   q15, %19             \n" // scale_out

                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q7, q7            \n"
                    "vcvt.f32.s32 q8, q8            \n"
                    // top_f32 = top_f32 * scale_int
                    "vmul.f32   q0, q7, q14         \n"
                    "vmul.f32   q4, q8, q14         \n"
                    // top_f32 = top_f32 + bias
                    "vadd.f32   q0, q0, q13         \n"
                    "vadd.f32   q4, q4, q13         \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q15         \n"
                    "vmul.f32   q4, q4, q15         \n"
                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0           \n"
                    "vcvtr.s32.f32 s1, s1           \n"
                    "vcvtr.s32.f32 s2, s2           \n"
                    "vcvtr.s32.f32 s3, s3           \n"
                    "vcvtr.s32.f32 s16, s16           \n"
                    "vcvtr.s32.f32 s17, s17           \n"
                    "vcvtr.s32.f32 s18, s18           \n"
                    "vcvtr.s32.f32 s19, s19           \n"
                    // top_s32 -> top_s16
                    "vqmovn.s32 d14, q0             \n"
                    "vqmovn.s32 d15, q4             \n"
                    // top_s16 -> top_s8
                    "vqmovn.s16   d14, q7           \n"
                    // save top_s8
                    "vst1.8     {d14}, [%1]!        \n"

                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q11, q11          \n"
                    "vcvt.f32.s32 q12, q12          \n"
                    // top_f32 = top_f32 * scale_int
                    "vmul.f32   q0, q11, q14        \n"
                    "vmul.f32   q4, q12, q14        \n"
                    // top_f32 = top_f32 + bias
                    "vadd.f32   q0, q0, q13         \n"
                    "vadd.f32   q4, q4, q13         \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q15         \n"
                    "vmul.f32   q4, q4, q15         \n"
                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0           \n"
                    "vcvtr.s32.f32 s1, s1           \n"
                    "vcvtr.s32.f32 s2, s2           \n"
                    "vcvtr.s32.f32 s3, s3           \n"
                    "vcvtr.s32.f32 s16, s16           \n"
                    "vcvtr.s32.f32 s17, s17           \n"
                    "vcvtr.s32.f32 s18, s18           \n"
                    "vcvtr.s32.f32 s19, s19           \n"
                    // top_s32 -> top_s16
                    "vqmovn.s32 d14, q0             \n"
                    "vqmovn.s32 d15, q4             \n"
                    // top_s16 -> top_s8
                    "vqmovn.s16   d14, q7           \n"
                    // save top_s8
                    "vst1.8     {d14}, [%2]!        \n"

                    "bne    0b                      \n"

                    : "=r"(nn),       // %0
                    "=r"(outptr0),  // %1
                    "=r"(outptr0n), // %2
                    "=r"(r0),       // %3
                    "=r"(r1),       // %4
                    "=r"(r2),       // %5
                    "=r"(r3)        // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr0n),
                    "3"(r0),
                    "4"(r1),
                    "5"(r2),
                    "6"(r3),
                    "w"(_k0123),           // %14
                    "w"(_k4567),           // %15
                    "w"(_k8xxx),           // %16
                    "r"(bias0),            // %17
                    "r"(scale_requant_in), // %18
                    "r"(scale_requant_out) // %19
                    : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                // TODO NEON
                int sum0 = 0;
                int sum0n = 0;

                sum0 += (int)r0[0] * kernel[0];
                sum0 += (int)r0[1] * kernel[1];
                sum0 += (int)r0[2] * kernel[2];
                sum0 += (int)r1[0] * kernel[3];
                sum0 += (int)r1[1] * kernel[4];
                sum0 += (int)r1[2] * kernel[5];
                sum0 += (int)r2[0] * kernel[6];
                sum0 += (int)r2[1] * kernel[7];
                sum0 += (int)r2[2] * kernel[8];

                sum0n += (int)r1[0] * kernel[0];
                sum0n += (int)r1[1] * kernel[1];
                sum0n += (int)r1[2] * kernel[2];
                sum0n += (int)r2[0] * kernel[3];
                sum0n += (int)r2[1] * kernel[4];
                sum0n += (int)r2[2] * kernel[5];
                sum0n += (int)r3[0] * kernel[6];
                sum0n += (int)r3[1] * kernel[7];
                sum0n += (int)r3[2] * kernel[8];

                *outptr0 = float2int8(((float)sum0 * scale_requant_in + bias0) * scale_requant_out);
                *outptr0n = float2int8(((float)sum0n * scale_requant_in + bias0) * scale_requant_out);

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr0n++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr0n += outw;
        }

        for (; i < outh; i++)
        {
#if __ARM_NEON
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "dup    v26.4s, %w13                  \n"
                    "dup    v27.4s, %w14                  \n"
                    "dup    v28.4s, %w15                  \n"

                    "0:                                   \n"
                    "ld1    {v4.8b, v5.8b}, [%2]          \n"
                    "ld1    {v6.8b, v7.8b}, [%3]          \n"
                    "ld1    {v8.8b, v9.8b}, [%4]          \n"
                    "add    %2, %2, #8                    \n"
                    "add    %3, %3, #8                    \n"
                    "add    %4, %4, #8                    \n"

                    "ext    v12.8b, v4.8b, v5.8b, #1      \n"
                    "ext    v13.8b, v4.8b, v5.8b, #2      \n"
                    "ext    v14.8b, v6.8b, v7.8b, #1      \n"
                    "ext    v15.8b, v6.8b, v7.8b, #2      \n"
                    "ext    v16.8b, v8.8b, v9.8b, #1      \n"
                    "ext    v17.8b, v8.8b, v9.8b, #2      \n"

                    "sshll  v4.8h, v4.8b, #0              \n" // r00
                    "sshll  v12.8h, v12.8b, #0            \n" // r01
                    "sshll  v13.8h, v13.8b, #0            \n" // r02
                    "sshll  v6.8h, v6.8b, #0              \n" // r10
                    "sshll  v14.8h, v14.8b, #0            \n" // r11
                    "sshll  v15.8h, v15.8b, #0            \n" // r12
                    "sshll  v8.8h, v8.8b, #0              \n" // r20
                    "sshll  v16.8h, v16.8b, #0            \n" // r21
                    "sshll  v17.8h, v17.8b, #0            \n" // r22

                    // r0
                    "smull  v20.4s, v4.4h, %10.h[0]       \n" // (r00 - r07) * k00
                    "smull2  v21.4s, v4.8h, %10.h[0]      \n"
                    "smull  v22.4s, v12.4h, %10.h[1]      \n" // (r01 - r08) * k01
                    "smull2  v23.4s, v12.8h, %10.h[1]     \n"
                    "smull  v24.4s, v13.4h, %10.h[2]      \n" // (r02 - r09) * k02
                    "smull2  v25.4s, v13.8h, %10.h[2]     \n"

                    // r1
                    "smlal  v20.4s, v6.4h, %10.h[3]       \n" // (r10 - r17) * k03
                    "smlal2  v21.4s, v6.8h, %10.h[3]      \n"
                    "smlal  v22.4s, v14.4h, %11.h[0]      \n" // (r11 - r18) * k04
                    "smlal2  v23.4s, v14.8h, %11.h[0]     \n"
                    "smlal  v24.4s, v15.4h, %11.h[1]      \n" // (r12 - r19) * k05
                    "smlal2  v25.4s, v15.8h, %11.h[1]     \n"

                    // r2
                    "smlal  v20.4s, v8.4h, %11.h[2]       \n" // (r20 - r27) * k06
                    "smlal2  v21.4s, v8.8h, %11.h[2]      \n"
                    "smlal  v22.4s, v16.4h, %11.h[3]      \n" // (r21 - r28) * k07
                    "smlal2  v23.4s, v16.8h, %11.h[3]     \n"
                    "smlal  v24.4s, v17.4h, %12.h[0]      \n" // (r22 - r29) * k08
                    "smlal2  v25.4s, v17.8h, %12.h[0]     \n"

                    // add and save
                    "add    v20.4s, v20.4s, v22.4s        \n"
                    "add    v21.4s, v21.4s, v23.4s        \n"
                    "add    v20.4s, v20.4s, v24.4s        \n"
                    "add    v21.4s, v21.4s, v25.4s        \n"

                    // top_s32 -> top_f32
                    "scvtf  v20.4s, v20.4s                \n"
                    "scvtf  v21.4s, v21.4s                \n"
                    // top_f32 = top_f32 * scale_in
                    "fmul   v20.4s, v20.4s, v27.4s        \n"
                    "fmul   v21.4s, v21.4s, v27.4s        \n"
                    // top_f32 = top_f32 + bias
                    "fadd   v20.4s, v20.4s, v26.4s        \n"
                    "fadd   v21.4s, v21.4s, v26.4s        \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v20.4s, v20.4s, v28.4s        \n"
                    "fmul   v21.4s, v21.4s, v28.4s        \n"
                    // top_f32 -> top_s32
                    "fcvtas v20.4s, v20.4s                \n"
                    "fcvtas v21.4s, v21.4s                \n"
                    // top_s32 -> top_s16
                    "sqxtn  v7.4h, v20.4s                 \n"
                    "sqxtn2 v7.8h, v21.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v8.8b, v7.8h                  \n"
                    // save top_s8
                    "st1    {v8.8b}, [%1], #8             \n"

                    "subs   %w0, %w0, #1                  \n"
                    "bne    0b                            \n"

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
                    "w"(_k0123),           // %10
                    "w"(_k4567),           // %11
                    "w"(_k8xxx),           // %12
                    "r"(bias0),            // %13
                    "r"(scale_requant_in), // %14
                    "r"(scale_requant_out) // %15
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                              \n"
                    // r0
                    "vld1.s8    {d30-d31}, [%2]      \n" // r0
                    "add    %2, %2, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r00
                    "vmovl.s8    q5, d10             \n" // r01
                    "vmovl.s8    q6, d12             \n" // r02
                    // sum0
                    "vmull.s16  q7, d30, %P10[0]     \n" // (r00 - r07) * k00
                    "vmull.s16  q8, d31, %P10[0]     \n"
                    "vmull.s16  q9, d10, %P10[1]     \n" // (r01 - r08) * k01
                    "vmull.s16  q10, d11, %P10[1]    \n"
                    "vmlal.s16  q7, d12, %P10[2]     \n" // (r02 - r09) * k02
                    "vmlal.s16  q8, d13, %P10[2]     \n"

                    // r1
                    "vld1.s8    {d30-d31}, [%3]      \n" // r1
                    "add    %3, %3, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r10
                    "vmovl.s8    q5, d10             \n" // r11
                    "vmovl.s8    q6, d12             \n" // r12
                    // sum0
                    "vmlal.s16  q7, d30, %P10[3]     \n" // (r10 - r17) * k03
                    "vmlal.s16  q8, d31, %P10[3]     \n"
                    "vmlal.s16  q9, d10, %P11[0]     \n" // (r11 - r18) * k04
                    "vmlal.s16  q10, d11, %P11[0]    \n"
                    "vmlal.s16  q7, d12, %P11[1]     \n" // (r12 - r19) * k05
                    "vmlal.s16  q8, d13, %P11[1]     \n"

                    // r2
                    "vld1.s8    {d30-d31}, [%4]      \n" // r2
                    "add    %4, %4, #8               \n"

                    "vext.s8    d10, d30, d31, #1    \n"
                    "vext.s8    d12, d30, d31, #2    \n"

                    "vmovl.s8    q15, d30            \n" // r20
                    "vmovl.s8    q5, d10             \n" // r21
                    "vmovl.s8    q6, d12             \n" // r22

                    // sum0
                    "vmlal.s16  q7, d30, %P11[2]     \n" // (r20 - r27) * k06
                    "vmlal.s16  q8, d31, %P11[2]     \n"
                    "vmlal.s16  q9, d10, %P11[3]     \n" // (r21 - r28) * k07
                    "vmlal.s16  q10, d11, %P11[3]    \n"
                    "vmlal.s16  q7, d12, %P12[0]     \n" // (r22 - r29) * k08
                    "vmlal.s16  q8, d13, %P12[0]     \n"

                    "subs   %0, %0, #1               \n"

                    // add and save
                    "vadd.s32    q7, q7, q9          \n"
                    "vadd.s32    q8, q8, q10         \n"

                    "vdup.f32   q13, %13             \n" // bias
                    "vdup.f32   q14, %14             \n" // scale_in
                    "vdup.f32   q15, %15             \n" // scale_out

                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q7, q7            \n"
                    "vcvt.f32.s32 q8, q8            \n"
                    // top_f32 = top_f32 * scale_int
                    "vmul.f32   q0, q7, q14         \n"
                    "vmul.f32   q4, q8, q14         \n"
                    // top_f32 = top_f32 + bias
                    "vadd.f32   q0, q0, q13         \n"
                    "vadd.f32   q4, q4, q13         \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q15         \n"
                    "vmul.f32   q4, q4, q15         \n"
                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0           \n"
                    "vcvtr.s32.f32 s1, s1           \n"
                    "vcvtr.s32.f32 s2, s2           \n"
                    "vcvtr.s32.f32 s3, s3           \n"
                    "vcvtr.s32.f32 s16, s16           \n"
                    "vcvtr.s32.f32 s17, s17           \n"
                    "vcvtr.s32.f32 s18, s18           \n"
                    "vcvtr.s32.f32 s19, s19           \n"
                    // top_s32 -> top_s16
                    "vqmovn.s32 d14, q0             \n"
                    "vqmovn.s32 d15, q4             \n"
                    // top_s16 -> top_s8
                    "vqmovn.s16   d14, q7           \n"
                    // save top_s8
                    "vst1.8     {d14}, [%1]!        \n"

                    "bne    0b                      \n"

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
                    "w"(_k0123),           // %10
                    "w"(_k4567),           // %11
                    "w"(_k8xxx),           // %12
                    "r"(bias0),            // %13
                    "r"(scale_requant_in), // %14
                    "r"(scale_requant_out) // %15
                    : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * kernel[0];
                sum += (int)r0[1] * kernel[1];
                sum += (int)r0[2] * kernel[2];
                sum += (int)r1[0] * kernel[3];
                sum += (int)r1[1] * kernel[4];
                sum += (int)r1[2] * kernel[5];
                sum += (int)r2[0] * kernel[6];
                sum += (int)r2[1] * kernel[7];
                sum += (int)r2[2] * kernel[8];

                *outptr0 = float2int8(((float)sum * scale_requant_in + bias0) * scale_requant_out);

                r0++;
                r1++;
                r2++;
                outptr0++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static void convdw3x3s2_int8_requant_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, std::vector<float> scales_requant, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;
        const float scale_requant_in = scales_requant[2 * p];
        const float scale_requant_out = scales_requant[2 * p + 1];

        const signed char* kernel = (const signed char*)_kernel + p * 9;

        signed char* outptr = out;

        const signed char* img = bottom_blob.channel(p);

        const signed char* r0 = img;
        const signed char* r1 = img + w;
        const signed char* r2 = img + w * 2;

        int i = 0;
#if __ARM_NEON
        int8x16_t _k0123456789x = vld1q_s8(kernel);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON
        for (; i < outh; i++)
        {
#if __ARM_NEON
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "dup    v26.4s, %w13                  \n"
                    "dup    v27.4s, %w14                  \n"
                    "dup    v28.4s, %w15                  \n"
                    "0:                                   \n"
                    "ld2    {v4.8b, v5.8b}, [%2], #16     \n"
                    "ld2    {v6.8b, v7.8b}, [%2]          \n"
                    "ld2    {v8.8b, v9.8b}, [%3], #16     \n"
                    "ld2    {v10.8b, v11.8b}, [%3]        \n"
                    "ld2    {v12.8b, v13.8b}, [%4], #16   \n"
                    "ld2    {v14.8b, v15.8b}, [%4]        \n"

                    "ext    v6.8b, v4.8b, v6.8b, #1       \n"
                    "ext    v10.8b, v8.8b, v10.8b, #1     \n"
                    "ext    v14.8b, v12.8b, v14.8b, #1    \n"

                    "sshll  v4.8h, v4.8b, #0              \n" // r00
                    "sshll  v5.8h, v5.8b, #0              \n" // r01
                    "sshll  v6.8h, v6.8b, #0              \n" // r02
                    "sshll  v8.8h, v8.8b, #0              \n" // r10
                    "sshll  v9.8h, v9.8b, #0              \n" // r11
                    "sshll  v10.8h, v10.8b, #0            \n" // r12
                    "sshll  v12.8h, v12.8b, #0            \n" // r20
                    "sshll  v13.8h, v13.8b, #0            \n" // r21
                    "sshll  v14.8h, v14.8b, #0            \n" // r22

                    // r0
                    "smull  v20.4s, v4.4h, %10.h[0]       \n" // (r00 - r07) * k00
                    "smull2  v21.4s, v4.8h, %10.h[0]      \n"
                    "smull  v22.4s, v5.4h, %10.h[1]       \n" // (r01 - r08) * k01
                    "smull2  v23.4s, v5.8h, %10.h[1]      \n"
                    "smull  v24.4s, v6.4h, %10.h[2]       \n" // (r02 - r09) * k02
                    "smull2  v25.4s, v6.8h, %10.h[2]      \n"

                    // r1
                    "smlal  v20.4s, v8.4h, %10.h[3]       \n" // (r10 - r17) * k03
                    "smlal2  v21.4s, v8.8h, %10.h[3]      \n"
                    "smlal  v22.4s, v9.4h, %11.h[0]       \n" // (r11 - r18) * k04
                    "smlal2  v23.4s, v9.8h, %11.h[0]      \n"
                    "smlal  v24.4s, v10.4h, %11.h[1]      \n" // (r12 - r19) * k05
                    "smlal2  v25.4s, v10.8h, %11.h[1]     \n"

                    // r2
                    "smlal  v20.4s, v12.4h, %11.h[2]      \n" // (r20 - r27) * k06
                    "smlal2  v21.4s, v12.8h, %11.h[2]     \n"
                    "smlal  v22.4s, v13.4h, %11.h[3]      \n" // (r21 - r28) * k07
                    "smlal2  v23.4s, v13.8h, %11.h[3]     \n"
                    "smlal  v24.4s, v14.4h, %12.h[0]      \n" // (r22 - r29) * k08
                    "smlal2  v25.4s, v14.8h, %12.h[0]     \n"

                    // add and save
                    "add    v20.4s, v20.4s, v22.4s        \n"
                    "add    v21.4s, v21.4s, v23.4s        \n"
                    "add    v20.4s, v20.4s, v24.4s        \n"
                    "add    v21.4s, v21.4s, v25.4s        \n"

                    // top_s32 -> top_f32
                    "scvtf  v20.4s, v20.4s                \n"
                    "scvtf  v21.4s, v21.4s                \n"
                    // top_f32 = top_f32 * scale_in
                    "fmul   v20.4s, v20.4s, v27.4s        \n"
                    "fmul   v21.4s, v21.4s, v27.4s        \n"
                    // top_f32 = top_f32 + bias
                    "fadd   v20.4s, v20.4s, v26.4s        \n"
                    "fadd   v21.4s, v21.4s, v26.4s        \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v20.4s, v20.4s, v28.4s        \n"
                    "fmul   v21.4s, v21.4s, v28.4s        \n"
                    // top_f32 -> top_s32
                    "fcvtas v20.4s, v20.4s                \n"
                    "fcvtas v21.4s, v21.4s                \n"
                    // top_s32 -> top_s16
                    "sqxtn  v7.4h, v20.4s                 \n"
                    "sqxtn2 v7.8h, v21.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v8.8b, v7.8h                  \n"
                    // save top_s8
                    "st1    {v8.8b}, [%1], #8             \n"

                    "subs   %w0, %w0, #1                  \n"
                    "bne    0b                            \n"

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
                    "w"(_k0123),           // %10
                    "w"(_k4567),           // %11
                    "w"(_k8xxx),           // %12
                    "r"(bias0),            // %13
                    "r"(scale_requant_in), // %14
                    "r"(scale_requant_out) // %15
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                              \n"
                    // r0
                    "vld2.s8    {d30-d31}, [%2]!     \n" // r0
                    "vld2.s8    {d10-d11}, [%2]      \n"
                    "vext.s8    d12, d30, d10, #1    \n"

                    "vmovl.s8    q5, d31             \n" // r01
                    "vmovl.s8    q15, d30            \n" // r00
                    "vmovl.s8    q6, d12             \n" // r02
                    // sum0
                    "vmull.s16  q7, d30, %P10[0]     \n" // (r00 - r07) * k00
                    "vmull.s16  q8, d31, %P10[0]     \n"
                    "vmull.s16  q9, d10, %P10[1]     \n" // (r01 - r08) * k01
                    "vmull.s16  q10, d11, %P10[1]    \n"
                    "vmlal.s16  q7, d12, %P10[2]     \n" // (r02 - r09) * k02
                    "vmlal.s16  q8, d13, %P10[2]     \n"

                    // r1
                    "vld2.s8    {d30-d31}, [%3]!     \n" // r1
                    "vld2.s8    {d10-d11}, [%3]      \n"
                    "vext.s8    d12, d30, d10, #1    \n"

                    "vmovl.s8    q5, d31             \n" // r11
                    "vmovl.s8    q15, d30            \n" // r10
                    "vmovl.s8    q6, d12             \n" // r12
                    // sum0
                    "vmlal.s16  q7, d30, %P10[3]     \n" // (r10 - r17) * k03
                    "vmlal.s16  q8, d31, %P10[3]     \n"
                    "vmlal.s16  q9, d10, %P11[0]     \n" // (r11 - r18) * k04
                    "vmlal.s16  q10, d11, %P11[0]    \n"
                    "vmlal.s16  q7, d12, %P11[1]     \n" // (r12 - r19) * k05
                    "vmlal.s16  q8, d13, %P11[1]     \n"

                    // r2
                    "vld2.s8    {d30-d31}, [%4]!     \n" // r2
                    "vld2.s8    {d10-d11}, [%4]      \n"
                    "vext.s8    d12, d30, d10, #1    \n"

                    "vmovl.s8    q5, d31             \n" // r21
                    "vmovl.s8    q15, d30            \n" // r20
                    "vmovl.s8    q6, d12             \n" // r22

                    // sum0
                    "vmlal.s16  q7, d30, %P11[2]     \n" // (r20 - r27) * k06
                    "vmlal.s16  q8, d31, %P11[2]     \n"
                    "vmlal.s16  q9, d10, %P11[3]     \n" // (r21 - r28) * k07
                    "vmlal.s16  q10, d11, %P11[3]    \n"
                    "vmlal.s16  q7, d12, %P12[0]     \n" // (r22 - r29) * k08
                    "vmlal.s16  q8, d13, %P12[0]     \n"

                    "subs   %0, %0, #1               \n"

                    // add and save
                    "vadd.s32    q7, q7, q9          \n"
                    "vadd.s32    q8, q8, q10         \n"

                    "vdup.f32   q11, %13             \n" // bias
                    "vdup.f32   q12, %14             \n" // scale_in
                    "vdup.f32   q13, %15             \n" // scale_out

                    // top_s32 -> top_f32
                    "vcvt.f32.s32 q7, q7             \n"
                    "vcvt.f32.s32 q8, q8             \n"
                    // top_f32 = top_f32 * scale_int
                    "vmul.f32   q0, q7, q12          \n"
                    "vmul.f32   q4, q8, q12          \n"
                    // top_f32 = top_f32 + bias
                    "vadd.f32   q0, q0, q11          \n"
                    "vadd.f32   q4, q4, q11          \n"
                    // top_f32 = top_f32 * scale_out
                    "vmul.f32   q0, q0, q13          \n"
                    "vmul.f32   q4, q4, q13          \n"
                    // top_f32 -> top_s32
                    "vcvtr.s32.f32 s0, s0            \n"
                    "vcvtr.s32.f32 s1, s1            \n"
                    "vcvtr.s32.f32 s2, s2            \n"
                    "vcvtr.s32.f32 s3, s3            \n"
                    "vcvtr.s32.f32 s16, s16            \n"
                    "vcvtr.s32.f32 s17, s17            \n"
                    "vcvtr.s32.f32 s18, s18            \n"
                    "vcvtr.s32.f32 s19, s19            \n"
                    // top_s32 -> top_s16
                    "vqmovn.s32 d14, q0              \n"
                    "vqmovn.s32 d15, q4              \n"
                    // top_s16 -> top_s8
                    "vqmovn.s16   d14, q7            \n"
                    // save top_s8
                    "vst1.8     {d14}, [%1]!         \n"

                    "bne    0b                       \n"

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
                    "w"(_k0123),           // %10
                    "w"(_k4567),           // %11
                    "w"(_k8xxx),           // %12
                    "r"(bias0),            // %13
                    "r"(scale_requant_in), // %14
                    "r"(scale_requant_out) // %15
                    : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * kernel[0];
                sum += (int)r0[1] * kernel[1];
                sum += (int)r0[2] * kernel[2];
                sum += (int)r1[0] * kernel[3];
                sum += (int)r1[1] * kernel[4];
                sum += (int)r1[2] * kernel[5];
                sum += (int)r2[0] * kernel[6];
                sum += (int)r2[1] * kernel[7];
                sum += (int)r2[2] * kernel[8];

                *outptr = float2int8(((float)sum * scale_requant_in + bias0) * scale_requant_out);

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
