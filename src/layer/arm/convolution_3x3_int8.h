// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#if __aarch64__
static void conv3x3s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char *kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            int *outptr0 = out0;

            const signed char *img0 = bottom_blob.channel(q);

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
}

static void conv3x3s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char *kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            int *outptr0 = out0;

            const signed char *img0 = bottom_blob.channel(q);

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * (int)kernel0[0];
                    sum0 += (int)r0[1] * (int)kernel0[1];
                    sum0 += (int)r0[2] * (int)kernel0[2];
                    sum0 += (int)r1[0] * (int)kernel0[3];
                    sum0 += (int)r1[1] * (int)kernel0[4];
                    sum0 += (int)r1[2] * (int)kernel0[5];
                    sum0 += (int)r2[0] * (int)kernel0[6];
                    sum0 += (int)r2[1] * (int)kernel0[7];
                    sum0 += (int)r2[2] * (int)kernel0[8];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}
#else // __aarch64__
static void conv3x3s1_neon_s8(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1; 

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        out0.fill(0);
        out1.fill(0);

        const signed char* kernel0 = (const signed char *)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char *)kernel + (p + 1) * inch * 9;
        
        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr0n = outptr0 + outw;
            int* outptr1n = outptr1 + outw;
        
            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;
            const signed char* r3 = img0 + w * 3;

            const signed char* k00 = kernel0;
            const signed char* k03 = kernel0 + 3;
            const signed char* k06 = kernel0 + 6;
            const signed char* k10 = kernel1;
            const signed char* k13 = kernel1 + 3;
            const signed char* k16 = kernel1 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                if (nn > 0)
                {
                    asm volatile(
                        "vld1.8    {d26-d27}, [%0]    \n"
                        "vld1.8    {d28-d29}, [%1]    \n"
                        : "=r"(kernel0), // %0
                          "=r"(kernel1)  // %1
                        : "0"(kernel0),
                          "1"(kernel1)
                        : "cc", "memory"
                    );

                    asm volatile(
                        "0:                             \n"
                        "pld        [%5, #128]          \n"
                        "vld1.32    {d0-d1}, [%5]       \n"// r0
                        "add        %5, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d6-d7}, [%6]       \n"// r1
                        "add        %6, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%7, #128]          \n"
                        "vld1.32    {d10-d11}, [%7]     \n"// r2
                        "add        %7, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"     
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%8, #128]          \n"
                        "vld1.32    {d14-d15}, [%8]     \n"// r3
                        "add        %8, #8              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"     
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"     
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8    d30, d28[1]         \n"
                        "vdup.s8    d31, d28[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d30         \n"// k1n
                        "vmlal.s8   q2, d3, d31         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8    d30, d28[4]         \n"
                        "vdup.s8    d31, d28[5]         \n"     
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d30         \n"// k4n
                        "vmlal.s8   q2, d9, d31         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8    d30, d28[7]         \n"
                        "vdup.s8    d31, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d30        \n"// k7n
                        "vmlal.s8   q2, d13, d31        \n"// k8n

                        "pld        [%3, #128]          \n"
                        "vld1.32    {d18-d21}, [%3]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%3]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8    d30, d28[1]         \n"
                        "vdup.s8    d31, d28[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0n
                        "vmlal.s8   q2, d8, d30         \n"// k1n
                        "vmlal.s8   q2, d9, d31         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8    d30, d28[4]         \n"
                        "vdup.s8    d31, d28[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3n
                        "vmlal.s8   q2, d12, d30        \n"// k4n
                        "vmlal.s8   q2, d13, d31        \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8    d30, d28[7]         \n"
                        "vdup.s8    d31, d29[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6n
                        "vmlal.s8   q2, d16, d30        \n"// k7n
                        "vmlal.s8   q2, d17, d31        \n"// k8n

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d18-d21}, [%4]     \n"// sum1n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(outptr1),        // %3
                          "=r"(outptr1n),       // %4
                          "=r"(r0),             // %5
                          "=r"(r1),             // %6
                          "=r"(r2),             // %7
                          "=r"(r3)              // %8
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(outptr1),
                          "4"(outptr1n),
                          "5"(r0),
                          "6"(r1),
                          "7"(r2),
                          "8"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    int sum0n = 0;
                    int sum1 = 0;
                    int sum1n = 0;

                    //ToDo Neon
                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    sum1 += (int)r0[0] * kernel1[0];
                    sum1 += (int)r0[1] * kernel1[1];
                    sum1 += (int)r0[2] * kernel1[2];
                    sum1 += (int)r1[0] * kernel1[3];
                    sum1 += (int)r1[1] * kernel1[4];
                    sum1 += (int)r1[2] * kernel1[5];
                    sum1 += (int)r2[0] * kernel1[6];
                    sum1 += (int)r2[1] * kernel1[7];
                    sum1 += (int)r2[2] * kernel1[8];

                    sum0n += (int)r1[0] * kernel0[0];
                    sum0n += (int)r1[1] * kernel0[1];
                    sum0n += (int)r1[2] * kernel0[2];
                    sum0n += (int)r2[0] * kernel0[3];
                    sum0n += (int)r2[1] * kernel0[4];
                    sum0n += (int)r2[2] * kernel0[5];
                    sum0n += (int)r3[0] * kernel0[6];
                    sum0n += (int)r3[1] * kernel0[7];
                    sum0n += (int)r3[2] * kernel0[8];

                    sum1n += (int)r1[0] * kernel1[0];
                    sum1n += (int)r1[1] * kernel1[1];
                    sum1n += (int)r1[2] * kernel1[2];
                    sum1n += (int)r2[0] * kernel1[3];
                    sum1n += (int)r2[1] * kernel1[4];
                    sum1n += (int)r2[2] * kernel1[5];
                    sum1n += (int)r3[0] * kernel1[6];
                    sum1n += (int)r3[1] * kernel1[7];
                    sum1n += (int)r3[2] * kernel1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr0n += sum0n;
                    *outptr1n += sum1n;

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
                int nn = outw >> 3;
                int remain = outw & 7;

                if (nn > 0)
                {
                    asm volatile(
                        "vld1.8    {d26-d27}, [%0]    \n"
                        "vld1.8    {d28-d29}, [%1]    \n"
                        : "=r"(kernel0), // %0
                          "=r"(kernel1)  // %1
                        : "0"(kernel0),
                          "1"(kernel1)
                        : "cc", "memory"
                    );

                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8     d7, d28[1]         \n"
                        "vdup.s8    d11, d28[2]         \n" 
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d7          \n"// k1n
                        "vmlal.s8   q2, d3, d11         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8     d7, d28[4]         \n"
                        "vdup.s8    d11, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d7          \n"// k4n
                        "vmlal.s8   q2, d9, d11         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8     d7, d28[7]         \n"
                        "vdup.s8    d11, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d7         \n"// k7n
                        "vmlal.s8   q2, d13, d11        \n"// k8n

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr1),        // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2)              // %5
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr1),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    int sum1 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    sum1 += (int)r0[0] * kernel1[0];
                    sum1 += (int)r0[1] * kernel1[1];
                    sum1 += (int)r0[2] * kernel1[2];
                    sum1 += (int)r1[0] * kernel1[3];
                    sum1 += (int)r1[1] * kernel1[4];
                    sum1 += (int)r1[2] * kernel1[5];
                    sum1 += (int)r2[0] * kernel1[6];
                    sum1 += (int)r2[1] * kernel1[7];
                    sum1 += (int)r2[2] * kernel1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;

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

            kernel0 += 9;
            kernel1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q=0; q<inch; q++)
        {                   
            int* outptr0 = out0;
            int* outptr0n = outptr0 + outw;
        
            const signed char* img0 = bottom_blob.channel(q);
            
            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;
            const signed char* r3 = img0 + w * 3;

            const signed char* k00 = kernel0;
            const signed char* k03 = kernel0 + 3;
            const signed char* k06 = kernel0 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                if (nn > 0)
                {
                    asm volatile(
                        "vld1.8    {d26-d27}, [%0]    \n"
                        : "=r"(kernel0) // %0
                        : "0"(kernel0)
                        : "cc", "memory"
                    );

                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d14-d15}, [%6]     \n"// r3
                        "add        %6, #8              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2),             // %5
                          "=r"(r3)              // %6
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2),
                          "6"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    //Todo Neon

                    int sum0 = 0;
                    int sum0n = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    sum0n += (int)r1[0] * kernel0[0];
                    sum0n += (int)r1[1] * kernel0[1];
                    sum0n += (int)r1[2] * kernel0[2];
                    sum0n += (int)r2[0] * kernel0[3];
                    sum0n += (int)r2[1] * kernel0[4];
                    sum0n += (int)r2[2] * kernel0[5];
                    sum0n += (int)r3[0] * kernel0[6];
                    sum0n += (int)r3[1] * kernel0[7];
                    sum0n += (int)r3[2] * kernel0[8];

                    *outptr0 += sum0;
                    *outptr0n += sum0n;

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
                int nn = outw >> 3;
                int remain = outw & 7;

                if (nn > 0)
                {
                    asm volatile(
                        "vld1.8    {d26-d27}, [%0]    \n"
                        : "=r"(kernel0) // %0
                        : "0"(kernel0)
                        : "cc", "memory"
                    );

                    asm volatile(
                        "0:                             \n"
                        "pld        [%2, #128]          \n"
                        "vld1.32    {d0-d1}, [%2]       \n"// r0
                        "add        %2, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d6-d7}, [%3]       \n"// r1
                        "add        %3, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d10-d11}, [%4]     \n"// r2
                        "add        %4, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(r0),             // %2
                          "=r"(r1),             // %3
                          "=r"(r2)              // %4
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                }   

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }           
            kernel0 += 9;
        }       
    }
}

static void conv3x3s1_neon_s8_left4(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1; 

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        out0.fill(0);
        out1.fill(0);

        const signed char* kernel0 = (const signed char *)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char *)kernel + (p + 1) * inch * 9;
        
        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr0n = outptr0 + outw;
            int* outptr1n = outptr1 + outw;
        
            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;
            const signed char* r3 = img0 + w*3;

            const signed char* k00 = kernel0;
            const signed char* k03 = kernel0 + 3;
            const signed char* k06 = kernel0 + 6;
            const signed char* k10 = kernel1;
            const signed char* k13 = kernel1 + 3;
            const signed char* k16 = kernel1 + 6;

            int i = 0;
            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"  // k00 k01 k02 k03 k04 k05 k06 k07 k08
                    "vld1.8    {d28-d29}, [%1]    \n"  // k10 k11 k12 k13 k14 k15 k16 k17 k18
                    : "=r"(kernel0), // %0
                      "=r"(kernel1)  // %1
                    : "0"(kernel0),
                      "1"(kernel1)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%5, #128]          \n"
                        "vld1.32    {d0-d1}, [%5]       \n"// r0
                        "add        %5, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d6-d7}, [%6]       \n"// r1
                        "add        %6, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%7, #128]          \n"
                        "vld1.32    {d10-d11}, [%7]     \n"// r2
                        "add        %7, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%8, #128]          \n"
                        "vld1.32    {d14-d15}, [%8]     \n"// r3
                        "add        %8, #8              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8    d30, d28[1]         \n"
                        "vdup.s8    d31, d28[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d30         \n"// k1n
                        "vmlal.s8   q2, d3, d31         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8    d30, d28[4]         \n"
                        "vdup.s8    d31, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d30         \n"// k4n
                        "vmlal.s8   q2, d9, d31         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8    d30, d28[7]         \n"
                        "vdup.s8    d31, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d30        \n"// k7n
                        "vmlal.s8   q2, d13, d31        \n"// k8n

                        "pld        [%3, #128]          \n"
                        "vld1.32    {d18-d21}, [%3]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%3]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8    d30, d28[1]         \n"
                        "vdup.s8    d31, d28[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0n
                        "vmlal.s8   q2, d8, d30         \n"// k1n
                        "vmlal.s8   q2, d9, d31         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8    d30, d28[4]         \n"
                        "vdup.s8    d31, d28[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3n
                        "vmlal.s8   q2, d12, d30        \n"// k4n
                        "vmlal.s8   q2, d13, d31        \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8    d30, d28[7]         \n"
                        "vdup.s8    d31, d29[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6n
                        "vmlal.s8   q2, d16, d30        \n"// k7n
                        "vmlal.s8   q2, d17, d31        \n"// k8n

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d18-d21}, [%4]     \n"// sum1n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(outptr1),        // %3
                          "=r"(outptr1n),       // %4
                          "=r"(r0),             // %5
                          "=r"(r1),             // %6
                          "=r"(r2),             // %7
                          "=r"(r3)              // %8
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(outptr1),
                          "4"(outptr1n),
                          "5"(r0),
                          "6"(r1),
                          "7"(r2),
                          "8"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q15"
                    );
                }

                asm volatile(
                    "pld        [%5, #128]          \n"
                    "vld1.32    {d0-d1}, [%5]       \n"// r0
                    "add        %5, #4              \n"
                    "vext.8     d2, d0, d1, #1      \n"
                    "vext.8     d3, d0, d1, #2      \n"
                    
                    "vdup.s8     d1, d26[0]         \n"
                    "vdup.s8    d30, d26[1]         \n"
                    "vdup.s8    d31, d26[2]         \n"
                    "vmull.s8   q2, d0, d1          \n"// k0
                    "vmlal.s8   q2, d2, d30         \n"// k1
                    "vmlal.s8   q2, d3, d31         \n"// k2
                    
                    "pld        [%6, #128]          \n"
                    "vld1.32    {d6-d7}, [%6]       \n"// r1
                    "add        %6, #4              \n"
                    "vext.8     d8, d6, d7, #1      \n"
                    "vext.8     d9, d6, d7, #2      \n"
                    
                    "vdup.s8     d1, d26[3]         \n"
                    "vdup.s8    d30, d26[4]         \n"
                    "vdup.s8    d31, d26[5]         \n"
                    "vmlal.s8   q2, d6, d1          \n"// k3
                    "vmlal.s8   q2, d8, d30         \n"// k4
                    "vmlal.s8   q2, d9, d31         \n"// k5

                    "pld        [%7, #128]          \n"
                    "vld1.32    {d10-d11}, [%7]     \n"// r2
                    "add        %7, #4              \n"
                    "vext.8     d12, d10, d11, #1   \n"
                    "vext.8     d13, d10, d11, #2   \n"
                    
                    "vdup.s8     d1, d26[6]         \n"
                    "vdup.s8    d30, d26[7]         \n"
                    "vdup.s8    d31, d27[0]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k6
                    "vmlal.s8   q2, d12, d30        \n"// k7
                    "vmlal.s8   q2, d13, d31        \n"// k8
                    
                    "pld        [%8, #128]          \n"
                    "vld1.32    {d14-d15}, [%8]     \n"// r3
                    "add        %8, #4              \n"
                    "vext.8     d16, d14, d15, #1   \n"
                    "vext.8     d17, d14, d15, #2   \n"
                    
                    "pld        [%1, #128]          \n"
                    "vld1.32    {d18-d19}, [%1]     \n"// sum0
                    "vaddw.s16   q9, q9, d4         \n"
                    "vst1.32    {d18-d19}, [%1]!    \n"
                    
                    "vdup.s8     d1, d26[0]         \n"
                    "vdup.s8    d30, d26[1]         \n"
                    "vdup.s8    d31, d26[2]         \n"
                    "vmull.s8   q2, d6, d1          \n"// k0
                    "vmlal.s8   q2, d8, d30         \n"// k1
                    "vmlal.s8   q2, d9, d31         \n"// k2

                    "vdup.s8     d1, d26[3]         \n"
                    "vdup.s8    d30, d26[4]         \n"
                    "vdup.s8    d31, d26[5]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k3
                    "vmlal.s8   q2, d12, d30        \n"// k4
                    "vmlal.s8   q2, d13, d31        \n"// k5

                    "vdup.s8     d1, d26[6]         \n"
                    "vdup.s8    d30, d26[7]         \n"
                    "vdup.s8    d31, d27[0]         \n"
                    "vmlal.s8   q2, d14, d1         \n"// k6
                    "vmlal.s8   q2, d16, d30        \n"// k7
                    "vmlal.s8   q2, d17, d31        \n"// k8

                    "pld        [%2, #128]          \n"
                    "vld1.32    {d18-d19}, [%2]     \n"// sum0n
                    "vaddw.s16   q9, q9, d4         \n"
                    "vst1.32    {d18-d19}, [%2]!    \n"

                    "vdup.s8     d1, d28[0]         \n"
                    "vdup.s8    d30, d28[1]         \n"
                    "vdup.s8    d31, d28[2]         \n"
                    "vmull.s8   q2, d0, d1          \n"// k0n
                    "vmlal.s8   q2, d2, d30         \n"// k1n
                    "vmlal.s8   q2, d3, d31         \n"// k2n

                    "vdup.s8     d1, d28[3]         \n"
                    "vdup.s8    d30, d28[4]         \n"
                    "vdup.s8    d31, d28[5]         \n"
                    "vmlal.s8   q2, d6, d1          \n"// k3n
                    "vmlal.s8   q2, d8, d30         \n"// k4n
                    "vmlal.s8   q2, d9, d31         \n"// k5n

                    "vdup.s8     d1, d28[6]         \n"
                    "vdup.s8    d30, d28[7]         \n"
                    "vdup.s8    d31, d29[0]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k6n
                    "vmlal.s8   q2, d12, d30        \n"// k7n
                    "vmlal.s8   q2, d13, d31        \n"// k8n

                    "pld        [%3, #128]          \n"
                    "vld1.32    {d18-d19}, [%3]     \n"// sum1
                    "vaddw.s16   q9, q9, d4         \n"
                    "vst1.32    {d18-d19}, [%3]!    \n"
                    
                    "vdup.s8     d1, d28[0]         \n"
                    "vdup.s8    d30, d28[1]         \n"
                    "vdup.s8    d31, d28[2]         \n"
                    "vmull.s8   q2, d6, d1          \n"// k0n
                    "vmlal.s8   q2, d8, d30         \n"// k1n
                    "vmlal.s8   q2, d9, d31         \n"// k2n

                    "vdup.s8     d1, d28[3]         \n"
                    "vdup.s8    d30, d28[4]         \n"
                    "vdup.s8    d31, d28[5]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k3n
                    "vmlal.s8   q2, d12, d30        \n"// k4n
                    "vmlal.s8   q2, d13, d31        \n"// k5n

                    "vdup.s8     d1, d28[6]         \n"
                    "vdup.s8    d30, d28[7]         \n"
                    "vdup.s8    d31, d29[0]         \n"
                    "vmlal.s8   q2, d14, d1         \n"// k6n
                    "vmlal.s8   q2, d16, d30        \n"// k7n
                    "vmlal.s8   q2, d17, d31        \n"// k8n

                    "pld        [%4, #128]          \n"
                    "vld1.32    {d18-d19}, [%4]     \n"// sum1n
                    "vaddw.s16   q9, q9, d4         \n"
                    "vst1.32    {d18-d19}, [%4]!    \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0),        // %1
                      "=r"(outptr0n),       // %2
                      "=r"(outptr1),        // %3
                      "=r"(outptr1n),       // %4
                      "=r"(r0),             // %5
                      "=r"(r1),             // %6
                      "=r"(r2),             // %7
                      "=r"(r3)              // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr0n),
                      "3"(outptr1),
                      "4"(outptr1n),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q15"
                );

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
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    "vld1.8    {d28-d29}, [%1]    \n"
                    : "=r"(kernel0), // %0
                      "=r"(kernel1)  // %1
                    : "0"(kernel0),
                      "1"(kernel1)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8     d7, d28[1]         \n"
                        "vdup.s8    d11, d28[2]         \n" 
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d7          \n"// k1n
                        "vmlal.s8   q2, d3, d11         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8     d7, d28[4]         \n"
                        "vdup.s8    d11, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d7          \n"// k4n
                        "vmlal.s8   q2, d9, d11         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8     d7, d28[7]         \n"
                        "vdup.s8    d11, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d7         \n"// k7n
                        "vmlal.s8   q2, d13, d11        \n"// k8n

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr1),        // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2)              // %5
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr1),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                asm volatile(
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #4              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #4              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #4              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d19}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vst1.32    {d18-d19}, [%1]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8     d7, d28[1]         \n"
                        "vdup.s8    d11, d28[2]         \n" 
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d7          \n"// k1n
                        "vmlal.s8   q2, d3, d11         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8     d7, d28[4]         \n"
                        "vdup.s8    d11, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d7          \n"// k4n
                        "vmlal.s8   q2, d9, d11         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8     d7, d28[7]         \n"
                        "vdup.s8    d11, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d7         \n"// k7n
                        "vmlal.s8   q2, d13, d11        \n"// k8n

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d19}, [%2]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vst1.32    {d18-d19}, [%2]!    \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr1),        // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2)              // %5
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr1),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
            kernel1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q=0; q<inch; q++)
        {                   
            int* outptr0 = out0;
            int* outptr0n = outptr0 + outw;
        
            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;
            const signed char* r3 = img0 + w*3;

            const signed char* k00 = kernel0;
            const signed char* k03 = kernel0 + 3;
            const signed char* k06 = kernel0 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    : "=r"(kernel0) // %0
                    : "0"(kernel0)
                    : "cc", "memory"
                );
                
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d14-d15}, [%6]     \n"// r3
                        "add        %6, #8              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2),             // %5
                          "=r"(r3)              // %6
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2),
                          "6"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                asm volatile(
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #4              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #4              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #4              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d14-d15}, [%6]     \n"// r3
                        "add        %6, #4              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d19}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vst1.32    {d18-d19}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d19}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vst1.32    {d18-d19}, [%2]!    \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2),             // %5
                          "=r"(r3)              // %6
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2),
                          "6"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr0n += outw;
            }

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    : "=r"(kernel0) // %0
                    : "0"(kernel0)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%2, #128]          \n"
                        "vld1.32    {d0-d1}, [%2]       \n"// r0
                        "add        %2, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d6-d7}, [%3]       \n"// r1
                        "add        %3, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d10-d11}, [%4]     \n"// r2
                        "add        %4, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(r0),             // %2
                          "=r"(r1),             // %3
                          "=r"(r2)              // %4
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                asm volatile(
                        "pld        [%2, #128]          \n"
                        "vld1.32    {d0-d1}, [%2]       \n"// r0
                        "add        %2, #4              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d6-d7}, [%3]       \n"// r1
                        "add        %3, #4              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d10-d11}, [%4]     \n"// r2
                        "add        %4, #4              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d19}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vst1.32    {d18-d19}, [%1]!    \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(r0),             // %2
                          "=r"(r1),             // %3
                          "=r"(r2)              // %4
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }           
            kernel0 += 9;
        }       
    }
}

static void conv3x3s1_neon_s8_left6(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1; 

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        out0.fill(0);
        out1.fill(0);

        const signed char* kernel0 = (const signed char *)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char *)kernel + (p + 1) * inch * 9;
        
        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr0n = outptr0 + outw;
            int* outptr1n = outptr1 + outw;
        
            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;
            const signed char* r3 = img0 + w*3;

            const signed char* k00 = kernel0;
            const signed char* k03 = kernel0 + 3;
            const signed char* k06 = kernel0 + 6;
            const signed char* k10 = kernel1;
            const signed char* k13 = kernel1 + 3;
            const signed char* k16 = kernel1 + 6;

            int i = 0;
            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"  // k00 k01 k02 k03 k04 k05 k06 k07 k08
                    "vld1.8    {d28-d29}, [%1]    \n"  // k10 k11 k12 k13 k14 k15 k16 k17 k18
                    : "=r"(kernel0), // %0
                      "=r"(kernel1)  // %1
                    : "0"(kernel0),
                      "1"(kernel1)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%5, #128]          \n"
                        "vld1.32    {d0-d1}, [%5]       \n"// r0
                        "add        %5, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d6-d7}, [%6]       \n"// r1
                        "add        %6, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%7, #128]          \n"
                        "vld1.32    {d10-d11}, [%7]     \n"// r2
                        "add        %7, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%8, #128]          \n"
                        "vld1.32    {d14-d15}, [%8]     \n"// r3
                        "add        %8, #8              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8    d30, d28[1]         \n"
                        "vdup.s8    d31, d28[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d30         \n"// k1n
                        "vmlal.s8   q2, d3, d31         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8    d30, d28[4]         \n"
                        "vdup.s8    d31, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d30         \n"// k4n
                        "vmlal.s8   q2, d9, d31         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8    d30, d28[7]         \n"
                        "vdup.s8    d31, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d30        \n"// k7n
                        "vmlal.s8   q2, d13, d31        \n"// k8n

                        "pld        [%3, #128]          \n"
                        "vld1.32    {d18-d21}, [%3]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%3]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8    d30, d28[1]         \n"
                        "vdup.s8    d31, d28[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0n
                        "vmlal.s8   q2, d8, d30         \n"// k1n
                        "vmlal.s8   q2, d9, d31         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8    d30, d28[4]         \n"
                        "vdup.s8    d31, d28[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3n
                        "vmlal.s8   q2, d12, d30        \n"// k4n
                        "vmlal.s8   q2, d13, d31        \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8    d30, d28[7]         \n"
                        "vdup.s8    d31, d29[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6n
                        "vmlal.s8   q2, d16, d30        \n"// k7n
                        "vmlal.s8   q2, d17, d31        \n"// k8n

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d18-d21}, [%4]     \n"// sum1n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(outptr1),        // %3
                          "=r"(outptr1n),       // %4
                          "=r"(r0),             // %5
                          "=r"(r1),             // %6
                          "=r"(r2),             // %7
                          "=r"(r3)              // %8
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(outptr1),
                          "4"(outptr1n),
                          "5"(r0),
                          "6"(r1),
                          "7"(r2),
                          "8"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q15"
                    );
                }

                asm volatile(
                    "pld        [%5, #128]          \n"
                    "vld1.32    {d0-d1}, [%5]       \n"// r0
                    "add        %5, #6              \n"
                    "vext.8     d2, d0, d1, #1      \n"
                    "vext.8     d3, d0, d1, #2      \n"
                    
                    "vdup.s8     d1, d26[0]         \n"
                    "vdup.s8    d30, d26[1]         \n"
                    "vdup.s8    d31, d26[2]         \n"
                    "vmull.s8   q2, d0, d1          \n"// k0
                    "vmlal.s8   q2, d2, d30         \n"// k1
                    "vmlal.s8   q2, d3, d31         \n"// k2
                    
                    "pld        [%6, #128]          \n"
                    "vld1.32    {d6-d7}, [%6]       \n"// r1
                    "add        %6, #6              \n"
                    "vext.8     d8, d6, d7, #1      \n"
                    "vext.8     d9, d6, d7, #2      \n"
                    
                    "vdup.s8     d1, d26[3]         \n"
                    "vdup.s8    d30, d26[4]         \n"
                    "vdup.s8    d31, d26[5]         \n"
                    "vmlal.s8   q2, d6, d1          \n"// k3
                    "vmlal.s8   q2, d8, d30         \n"// k4
                    "vmlal.s8   q2, d9, d31         \n"// k5

                    "pld        [%7, #128]          \n"
                    "vld1.32    {d10-d11}, [%7]     \n"// r2
                    "add        %7, #6              \n"
                    "vext.8     d12, d10, d11, #1   \n"
                    "vext.8     d13, d10, d11, #2   \n"
                    
                    "vdup.s8     d1, d26[6]         \n"
                    "vdup.s8    d30, d26[7]         \n"
                    "vdup.s8    d31, d27[0]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k6
                    "vmlal.s8   q2, d12, d30        \n"// k7
                    "vmlal.s8   q2, d13, d31        \n"// k8
                    
                    "pld        [%8, #128]          \n"
                    "vld1.32    {d14-d15}, [%8]     \n"// r3
                    "add        %8, #6              \n"
                    "vext.8     d16, d14, d15, #1   \n"
                    "vext.8     d17, d14, d15, #2   \n"
                    
                    "pld        [%1, #128]          \n"
                    "vld1.32    {d18-d20}, [%1]     \n"// sum0
                    "vaddw.s16   q9,  q9, d4        \n"
                    "vaddw.s16  q10, q10, d5        \n"
                    "vst1.32    {d18-d20}, [%1]!    \n"
                    
                    "vdup.s8     d1, d26[0]         \n"
                    "vdup.s8    d30, d26[1]         \n"
                    "vdup.s8    d31, d26[2]         \n"
                    "vmull.s8   q2, d6, d1          \n"// k0
                    "vmlal.s8   q2, d8, d30         \n"// k1
                    "vmlal.s8   q2, d9, d31         \n"// k2

                    "vdup.s8     d1, d26[3]         \n"
                    "vdup.s8    d30, d26[4]         \n"
                    "vdup.s8    d31, d26[5]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k3
                    "vmlal.s8   q2, d12, d30        \n"// k4
                    "vmlal.s8   q2, d13, d31        \n"// k5

                    "vdup.s8     d1, d26[6]         \n"
                    "vdup.s8    d30, d26[7]         \n"
                    "vdup.s8    d31, d27[0]         \n" 
                    "vmlal.s8   q2, d14, d1         \n"// k6
                    "vmlal.s8   q2, d16, d30        \n"// k7
                    "vmlal.s8   q2, d17, d31        \n"// k8

                    "pld        [%2, #128]          \n"
                    "vld1.32    {d18-d20}, [%2]     \n"// sum0n
                    "vaddw.s16   q9,  q9, d4        \n"
                    "vaddw.s16  q10, q10, d5        \n"
                    "vst1.32    {d18-d20}, [%2]!    \n"
                    
                    "vdup.s8     d1, d28[0]         \n"
                    "vdup.s8    d30, d28[1]         \n"
                    "vdup.s8    d31, d28[2]         \n"
                    "vmull.s8   q2, d0, d1          \n"// k0n
                    "vmlal.s8   q2, d2, d30         \n"// k1n
                    "vmlal.s8   q2, d3, d31         \n"// k2n

                    "vdup.s8     d1, d28[3]         \n"
                    "vdup.s8    d30, d28[4]         \n"
                    "vdup.s8    d31, d28[5]         \n"
                    "vmlal.s8   q2, d6, d1          \n"// k3n
                    "vmlal.s8   q2, d8, d30         \n"// k4n
                    "vmlal.s8   q2, d9, d31         \n"// k5n

                    "vdup.s8     d1, d28[6]         \n"
                    "vdup.s8    d30, d28[7]         \n"
                    "vdup.s8    d31, d29[0]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k6n
                    "vmlal.s8   q2, d12, d30        \n"// k7n
                    "vmlal.s8   q2, d13, d31        \n"// k8n

                    "pld        [%3, #128]          \n"
                    "vld1.32    {d18-d20}, [%3]     \n"// sum1
                    "vaddw.s16   q9,  q9, d4        \n"
                    "vaddw.s16  q10, q10, d5        \n"
                    "vst1.32    {d18-d20}, [%3]!    \n"
                    
                    "vdup.s8     d1, d28[0]         \n"
                    "vdup.s8    d30, d28[1]         \n"
                    "vdup.s8    d31, d28[2]         \n"
                    "vmull.s8   q2, d6, d1          \n"// k0n
                    "vmlal.s8   q2, d8, d30         \n"// k1n
                    "vmlal.s8   q2, d9, d31         \n"// k2n

                    "vdup.s8     d1, d28[3]         \n"
                    "vdup.s8    d30, d28[4]         \n"
                    "vdup.s8    d31, d28[5]         \n"
                    "vmlal.s8   q2, d10, d1         \n"// k3n
                    "vmlal.s8   q2, d12, d30        \n"// k4n
                    "vmlal.s8   q2, d13, d31        \n"// k5n

                    "vdup.s8     d1, d28[6]         \n"
                    "vdup.s8    d30, d28[7]         \n"
                    "vdup.s8    d31, d29[0]         \n"
                    "vmlal.s8   q2, d14, d1         \n"// k6n
                    "vmlal.s8   q2, d16, d30        \n"// k7n
                    "vmlal.s8   q2, d17, d31        \n"// k8n

                    "pld        [%4, #128]          \n"
                    "vld1.32    {d18-d20}, [%4]     \n"// sum1n
                    "vaddw.s16   q9,  q9, d4        \n"
                    "vaddw.s16  q10, q10, d5        \n"
                    "vst1.32    {d18-d20}, [%4]!    \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0),        // %1
                      "=r"(outptr0n),       // %2
                      "=r"(outptr1),        // %3
                      "=r"(outptr1n),       // %4
                      "=r"(r0),             // %5
                      "=r"(r1),             // %6
                      "=r"(r2),             // %7
                      "=r"(r3)              // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr0n),
                      "3"(outptr1),
                      "4"(outptr1n),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q15"
                );

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
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    "vld1.8    {d28-d29}, [%1]    \n"
                    : "=r"(kernel0), // %0
                      "=r"(kernel1)  // %1
                    : "0"(kernel0),
                      "1"(kernel1)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8     d7, d28[1]         \n"
                        "vdup.s8    d11, d28[2]         \n" 
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d7          \n"// k1n
                        "vmlal.s8   q2, d3, d11         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8     d7, d28[4]         \n"
                        "vdup.s8    d11, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d7          \n"// k4n
                        "vmlal.s8   q2, d9, d11         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8     d7, d28[7]         \n"
                        "vdup.s8    d11, d29[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d7         \n"// k7n
                        "vmlal.s8   q2, d13, d11        \n"// k8n

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr1),        // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2)              // %5
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr1),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                asm volatile(
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #6              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #6              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #6              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d20}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d20}, [%1]!    \n"
                        
                        "vdup.s8     d1, d28[0]         \n"
                        "vdup.s8     d7, d28[1]         \n"
                        "vdup.s8    d11, d28[2]         \n" 
                        "vmull.s8   q2, d0, d1          \n"// k0n
                        "vmlal.s8   q2, d2, d7          \n"// k1n
                        "vmlal.s8   q2, d3, d11         \n"// k2n

                        "vdup.s8     d1, d28[3]         \n"
                        "vdup.s8     d7, d28[4]         \n"
                        "vdup.s8    d11, d28[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3n
                        "vmlal.s8   q2, d8, d7          \n"// k4n
                        "vmlal.s8   q2, d9, d11         \n"// k5n

                        "vdup.s8     d1, d28[6]         \n"
                        "vdup.s8     d7, d28[7]         \n"
                        "vdup.s8    d11, d29[0]         \n" 
                        "vmlal.s8   q2, d10, d1         \n"// k6n
                        "vmlal.s8   q2, d12, d7         \n"// k7n
                        "vmlal.s8   q2, d13, d11        \n"// k8n

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d20}, [%2]     \n"// sum1
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d20}, [%2]!    \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr1),        // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2)              // %5
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr1),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
            kernel1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q=0; q<inch; q++)
        {                   
            int* outptr0 = out0;
            int* outptr0n = outptr0 + outw;
        
            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;
            const signed char* r3 = img0 + w * 3;

            const signed char* k00 = kernel0;
            const signed char* k03 = kernel0 + 3;
            const signed char* k06 = kernel0 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    : "=r"(kernel0) // %0
                    : "0"(kernel0)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d14-d15}, [%6]     \n"// r3
                        "add        %6, #8              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d21}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),    // %1
                          "=r"(outptr0n),   // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2),             // %5
                          "=r"(r3)              // %6
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2),
                          "6"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                asm volatile(
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d0-d1}, [%3]       \n"// r0
                        "add        %3, #6              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%4, #128]          \n"
                        "vld1.32    {d6-d7}, [%4]       \n"// r1
                        "add        %4, #6              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%5, #128]          \n"
                        "vld1.32    {d10-d11}, [%5]     \n"// r2
                        "add        %5, #6              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%6, #128]          \n"
                        "vld1.32    {d14-d15}, [%6]     \n"// r3
                        "add        %6, #6              \n"
                        "vext.8     d16, d14, d15, #1   \n"
                        "vext.8     d17, d14, d15, #2   \n"
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d20}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d20}, [%1]!    \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d6, d1          \n"// k0
                        "vmlal.s8   q2, d8, d30         \n"// k1
                        "vmlal.s8   q2, d9, d31         \n"// k2

                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k3
                        "vmlal.s8   q2, d12, d30        \n"// k4
                        "vmlal.s8   q2, d13, d31        \n"// k5

                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d14, d1         \n"// k6
                        "vmlal.s8   q2, d16, d30        \n"// k7
                        "vmlal.s8   q2, d17, d31        \n"// k8

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d18-d20}, [%2]     \n"// sum0n
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d20}, [%2]!    \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(outptr0n),       // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2),             // %5
                          "=r"(r3)              // %6
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr0n),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2),
                          "6"(r3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr0n += outw;
            }

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    : "=r"(kernel0) // %0
                    : "0"(kernel0)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%2, #128]          \n"
                        "vld1.32    {d0-d1}, [%2]       \n"// r0
                        "add        %2, #8              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d6-d7}, [%3]       \n"// r1
                        "add        %3, #8              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d10-d11}, [%4]     \n"// r2
                        "add        %4, #8              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d21}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d21}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),    // %1
                          "=r"(r0),             // %2
                          "=r"(r1),             // %3
                          "=r"(r2)              // %4
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

                asm volatile(
                        "pld        [%2, #128]          \n"
                        "vld1.32    {d0-d1}, [%2]       \n"// r0
                        "add        %2, #6              \n"
                        "vext.8     d2, d0, d1, #1      \n"
                        "vext.8     d3, d0, d1, #2      \n"
                        
                        "vdup.s8     d1, d26[0]         \n"
                        "vdup.s8    d30, d26[1]         \n"
                        "vdup.s8    d31, d26[2]         \n"
                        "vmull.s8   q2, d0, d1          \n"// k0
                        "vmlal.s8   q2, d2, d30         \n"// k1
                        "vmlal.s8   q2, d3, d31         \n"// k2
                        
                        "pld        [%3, #128]          \n"
                        "vld1.32    {d6-d7}, [%3]       \n"// r1
                        "add        %3, #6              \n"
                        "vext.8     d8, d6, d7, #1      \n"
                        "vext.8     d9, d6, d7, #2      \n"
                        
                        "vdup.s8     d1, d26[3]         \n"
                        "vdup.s8    d30, d26[4]         \n"
                        "vdup.s8    d31, d26[5]         \n"
                        "vmlal.s8   q2, d6, d1          \n"// k3
                        "vmlal.s8   q2, d8, d30         \n"// k4
                        "vmlal.s8   q2, d9, d31         \n"// k5

                        "pld        [%4, #128]          \n"
                        "vld1.32    {d10-d11}, [%4]     \n"// r2
                        "add        %4, #6              \n"
                        "vext.8     d12, d10, d11, #1   \n"
                        "vext.8     d13, d10, d11, #2   \n"
                        
                        "vdup.s8     d1, d26[6]         \n"
                        "vdup.s8    d30, d26[7]         \n"
                        "vdup.s8    d31, d27[0]         \n"
                        "vmlal.s8   q2, d10, d1         \n"// k6
                        "vmlal.s8   q2, d12, d30        \n"// k7
                        "vmlal.s8   q2, d13, d31        \n"// k8
                        
                        "pld        [%1, #128]          \n"
                        "vld1.32    {d18-d20}, [%1]     \n"// sum0
                        "vaddw.s16   q9,  q9, d4        \n"
                        "vaddw.s16  q10, q10, d5        \n"
                        "vst1.32    {d18-d20}, [%1]!    \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),        // %1
                          "=r"(r0),             // %2
                          "=r"(r1),             // %3
                          "=r"(r2)              // %4
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
            kernel0 += 9;
        }       
    }
}

static void conv3x3s2_neon_s8(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char* kernel = _kernel;
    
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1; 

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);

        out0.fill(0.f);
        out1.fill(0.f);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char*)kernel + (p + 1) * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            const signed char* k00 = kernel0;
            const signed char* k01 = kernel0 + 3;
            const signed char* k02 = kernel0 + 6;

            const signed char* k10 = kernel1;
            const signed char* k11 = kernel1 + 3;
            const signed char* k12 = kernel1 + 6;

            int i = 0;

            for (; i < outh; i++)
            {                           
                int nn = outw >> 3;
                int remain = outw & 7;  

                asm volatile(
                    "vld1.s8    {d22-d23}, [%0]    \n"
                    "vld1.s8    {d24-d25}, [%1]    \n"
                    : "=r"(kernel0), // %0
                      "=r"(kernel1)  // %1
                    : "0"(kernel0),
                      "1"(kernel1)
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%3, #192]          \n"
                        "vld2.s8    {d0-d1}, [%3]!      \n" // r0
                        "vld2.s8    {d2-d3}, [%3]       \n"
                        "vext.8     d3, d0, d2, #1      \n"
            
                        "vdup.s8    d26, d22[0]         \n"
                        "vdup.s8    d27, d22[1]         \n"
                        "vdup.s8    d28, d22[2]         \n"
                        "vmull.s8   q2, d0, d26         \n" // k00
                        "vmlal.s8   q2, d1, d27         \n" // k01
                        "vmlal.s8   q2, d3, d28         \n" // k02
                        
                        "pld        [%4, #192]          \n"
                        "vld2.s8    {d6-d7}, [%4]!      \n" // r1
                        "vld2.s8    {d8-d9}, [%4]       \n"
                        "vext.8     d9, d6, d8, #1      \n"
                        
                        "vdup.s8    d26, d22[3]         \n"
                        "vdup.s8    d27, d22[4]         \n"
                        "vdup.s8    d28, d22[5]         \n"
                        "vmlal.s8   q2, d6, d26         \n" // k03
                        "vmlal.s8   q2, d7, d27         \n" // k04
                        "vmlal.s8   q2, d9, d28         \n" // k05

                        "pld        [%5, #192]          \n" 
                        "vld2.s8    {d10-d11}, [%5]!    \n" // r2
                        "vld2.s8    {d12-d13}, [%5]     \n"
                        "vext.8     d13, d10, d12, #1   \n"
                        
                        "vdup.s8    d26, d22[6]         \n"
                        "vdup.s8    d27, d22[7]         \n"
                        "vdup.s8    d28, d23[0]         \n"
                        "vmlal.s8   q2, d10, d26        \n" // k06
                        "vmlal.s8   q2, d11, d27        \n" // k07
                        "vmlal.s8   q2, d13, d28        \n" // k08

                        "pld        [%1, #256]          \n"
                        "vld1.32    {d14-d17}, [%1]     \n" //sum0
                        "vaddw.s16   q7, q7, d4         \n"
                        "vaddw.s16   q8, q8, d5         \n"
                        "vst1.32    {d14-d17}, [%1]!    \n"
                        
                        "vdup.s8    d26, d24[0]         \n"
                        "vdup.s8    d27, d24[1]         \n"
                        "vdup.s8    d28, d24[2]         \n"
                        "vmull.s8   q2, d0, d26         \n" // k00
                        "vmlal.s8   q2, d1, d27         \n" // k01
                        "vmlal.s8   q2, d3, d28         \n" // k02
                        
                        "vdup.s8    d26, d24[3]         \n"
                        "vdup.s8    d27, d24[4]         \n"
                        "vdup.s8    d28, d24[5]         \n"
                        "vmlal.s8   q2, d6, d26         \n" // k03
                        "vmlal.s8   q2, d7, d27         \n" // k04
                        "vmlal.s8   q2, d9, d28         \n" // k05
                        
                        "vdup.s8    d26, d24[6]         \n"
                        "vdup.s8    d27, d24[7]         \n"
                        "vdup.s8    d28, d25[0]         \n"
                        "vmlal.s8   q2, d10, d26        \n" // k06
                        "vmlal.s8   q2, d11, d27        \n" // k07
                        "vmlal.s8   q2, d13, d28        \n" // k08

                        "pld        [%2, #256]          \n"
                        "vld1.32    {d14-d17}, [%2]     \n" //sum1
                        "vaddw.s16   q7, q7, d4         \n"
                        "vaddw.s16   q8, q8, d5         \n"
                        "vst1.32    {d14-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),    // %1
                          "=r"(outptr1),    // %2
                          "=r"(r0),             // %3
                          "=r"(r1),             // %4
                          "=r"(r2)              // %5
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(outptr1),
                          "3"(r0),
                          "4"(r1),
                          "5"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q13", "q14", "q15"
                    );
                }           

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    int sum1 = 0;
                
                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    sum1 += (int)r0[0] * kernel1[0];
                    sum1 += (int)r0[1] * kernel1[1];
                    sum1 += (int)r0[2] * kernel1[2];
                    sum1 += (int)r1[0] * kernel1[3];
                    sum1 += (int)r1[1] * kernel1[4];
                    sum1 += (int)r1[2] * kernel1[5];
                    sum1 += (int)r2[0] * kernel1[6];
                    sum1 += (int)r2[1] * kernel1[7];
                    sum1 += (int)r2[2] * kernel1[8];
                
                    *outptr0 += sum0;
                    *outptr1 += sum1;

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

            kernel0 += 9;
            kernel1 += 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0.f);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            const signed char* k00 = kernel0;
            const signed char* k01 = kernel0 + 3;
            const signed char* k02 = kernel0 + 6;   

            int i = 0;

            for (; i < outh; i++)
            {           
                int nn = outw >> 3;
                int remain = outw & 7;  
                
                asm volatile(
                    "vld1.s8    {d22-d23}, [%0]    \n"
                    : "=r"(kernel0) // %0
                    : "0"(kernel0) 
                    : "cc", "memory"
                );

                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%2, #192]          \n"
                        "vld2.s8    {d0-d1}, [%2]!      \n" // r0
                        "vld2.s8    {d2-d3}, [%2]       \n"
                        "vext.8     d3, d0, d2, #1      \n"
            
                        "vdup.s8    d26, d22[0]         \n"
                        "vdup.s8    d27, d22[1]         \n"
                        "vdup.s8    d28, d22[2]         \n"
                        "vmull.s8   q2, d0, d26         \n" // k00
                        "vmlal.s8   q2, d1, d27         \n" // k01
                        "vmlal.s8   q2, d3, d28         \n" // k02
                        
                        "pld        [%3, #192]          \n"
                        "vld2.s8    {d6-d7}, [%3]!      \n" // r1
                        "vld2.s8    {d8-d9}, [%3]       \n"
                        "vext.8     d9, d6, d8, #1      \n"
                        
                        "vdup.s8    d26, d22[3]         \n"
                        "vdup.s8    d27, d22[4]         \n"
                        "vdup.s8    d28, d22[5]         \n"
                        "vmlal.s8   q2, d6, d26         \n" // k03
                        "vmlal.s8   q2, d7, d27         \n" // k04
                        "vmlal.s8   q2, d9, d28         \n" // k05

                        "pld        [%4, #192]          \n"
                        "vld2.s8    {d10-d11}, [%4]!    \n" // r2
                        "vld2.s8    {d12-d13}, [%4]     \n"
                        "vext.8     d13, d10, d12, #1   \n"
                        
                        "vdup.s8    d26, d22[6]         \n"
                        "vdup.s8    d27, d22[7]         \n"
                        "vdup.s8    d28, d23[0]         \n"
                        "vmlal.s8   q2, d10, d26        \n" // k06
                        "vmlal.s8   q2, d11, d27        \n" // k07
                        "vmlal.s8   q2, d13, d28        \n" // k08

                        "pld        [%1, #256]          \n"
                        "vld1.32    {d14-d17}, [%1]     \n" //sum0
                        "vaddw.s16   q7, q7, d4         \n"
                        "vaddw.s16   q8, q8, d5         \n"
                        "vst1.32    {d14-d17}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),             // %0
                          "=r"(outptr0),    // %1
                          "=r"(r0),             // %2
                          "=r"(r1),             // %3
                          "=r"(r2)              // %4
                        : "0"(nn),
                          "1"(outptr0),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q12", "q13", "q14"
                    );
                }           

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    
                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];
                    
                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }       
    }   
}

static void conv3x3s1_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int outw = top_blob.w;
    int remain = outw & 7;

    typedef void (*conv_func_int8)(const Mat&, Mat&, const Mat&, const Option&);

    conv_func_int8 conv_func_table[8] =
    {
        conv3x3s1_neon_s8,          //0
        conv3x3s1_neon_s8,          //1
        conv3x3s1_neon_s8,          //2
        conv3x3s1_neon_s8,          //3
        conv3x3s1_neon_s8_left4,    //4
        conv3x3s1_neon_s8,          //5
        conv3x3s1_neon_s8_left6,    //6
        conv3x3s1_neon_s8,          //7
    };   

    conv_func_int8 conv = conv_func_table[remain];

    conv(bottom_blob, top_blob, _kernel, opt);

    return;
}

static void conv3x3s2_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int outw = top_blob.w;
    int remain = outw & 7;
    
    typedef void (*conv_func_int8)(const Mat&, Mat&, const Mat&, const Option&);

    conv_func_int8 conv_func_table[8] =
    {
        conv3x3s2_neon_s8,      //0
        conv3x3s2_neon_s8,      //1
        conv3x3s2_neon_s8,      //2
        conv3x3s2_neon_s8,      //3
        conv3x3s2_neon_s8,      //4
        conv3x3s2_neon_s8,      //5
        conv3x3s2_neon_s8,      //6
        conv3x3s2_neon_s8,      //7
    };   

    conv_func_int8 conv = conv_func_table[remain];

    conv(bottom_blob, top_blob, _kernel, opt);

    return;
}
#endif