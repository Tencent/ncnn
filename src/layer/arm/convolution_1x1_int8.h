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
static void conv1x1s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            const signed char *r0 = bottom_blob.channel(q);
            const signed char *r1 = bottom_blob.channel(q + 1);
            const signed char *r2 = bottom_blob.channel(q + 2);
            const signed char *r3 = bottom_blob.channel(q + 3);
            const signed char *r4 = bottom_blob.channel(q + 4);
            const signed char *r5 = bottom_blob.channel(q + 5);
            const signed char *r6 = bottom_blob.channel(q + 6);
            const signed char *r7 = bottom_blob.channel(q + 7);

            int size = outw * outh;
            int remain = size;

            for (; remain > 0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * (int)kernel0[0] + (int)*r1 * (int)kernel0[1] +
                           (int)*r2 * (int)kernel0[2] + (int)*r3 * (int)kernel0[3] +
                           (int)*r4 * (int)kernel0[4] + (int)*r5 * (int)kernel0[5] +
                           (int)*r6 * (int)kernel0[6] + (int)*r7 * (int)kernel0[7];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char *r0 = bottom_blob.channel(q);

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;
            int remain = size;

            for (; remain > 0; remain--)
            {
                int sum0 = (int)(*r0) * (int)k0;

                *outptr0 += sum0;

                r0++;
                outptr0++;
            }
        }
    }
}
#else // __aarch64__
/*
 * Convolution 1x1 quantized with int8,unroll 8 x 4
 */
static void conv1x1s1_neon_s8(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            if (nn > 0)
            {
                asm volatile(
                    "vld1.s8    d18, [%0]   \n"
                    "vld1.s8    d19, [%1]   \n"
                    "vld1.s8    d24, [%2]   \n"
                    "vld1.s8    d25, [%3]   \n"
                    : "=r"(kernel0), // %0
                      "=r"(kernel1), // %1
                      "=r"(kernel2), // %2
                      "=r"(kernel3)  // %3
                    : "0"(kernel0),
                      "1"(kernel1),
                      "2"(kernel2),
                      "3"(kernel3)
                    :
                );

                asm volatile(
                    "0:                            \n"
                    //ld r0-r7
                    "pld        [%5, #64]          \n"
                    "vld1.s8    {d0}, [%5 :64]!    \n" //r0

                    "pld        [%6, #64]          \n"
                    "vld1.s8    {d1}, [%6 :64]!    \n" //r1

                    "pld        [%7, #64]          \n"
                    "vld1.s8    {d2}, [%7 :64]!    \n" //r2

                    "pld        [%8, #64]          \n"
                    "vld1.s8    {d3}, [%8 :64]!    \n" //r3

                    "pld        [%9, #64]          \n"
                    "vld1.s8    {d4}, [%9 :64]!    \n" //r4

                    "pld        [%10, #64]         \n"
                    "vld1.s8    {d5}, [%10 :64]!   \n" //r5

                    "pld        [%11, #64]         \n"
                    "vld1.s8    {d6}, [%11 :64]!   \n" //r6

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d7}, [%12 :64]!   \n" //r7
                    //###########################################
                    //load inch kernel_0 k0-k7
                    "vdup.s8    d8, d18[0]          \n"
                    "vdup.s8    d9, d18[1]          \n"
                    "vdup.s8    d10, d18[2]         \n"
                    "vdup.s8    d11, d18[3]         \n"
                    "vdup.s8    d12, d18[4]         \n"
                    "vdup.s8    d13, d18[5]         \n"
                    "vdup.s8    d14, d18[6]         \n"
                    "vdup.s8    d15, d18[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d20-d23}, [%1:128] \n" //outptr0_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%1:128]!\n"
                    //###########################################
                    //load inch kernel_1 k0-k7
                    "vdup.s8    d8, d19[0]          \n"
                    "vdup.s8    d9, d19[1]          \n"
                    "vdup.s8    d10, d19[2]         \n"
                    "vdup.s8    d11, d19[3]         \n"
                    "vdup.s8    d12, d19[4]         \n"
                    "vdup.s8    d13, d19[5]         \n"
                    "vdup.s8    d14, d19[6]         \n"
                    "vdup.s8    d15, d19[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr1_s32
                    "pld        [%2, #256]          \n"
                    "vld1.32    {d20-d23}, [%2:128] \n" //outptr1_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%2:128]!\n"
                    //############################################
                    //load inch kernel_2 k0-k7
                    "vdup.s8    d8, d24[0]          \n"
                    "vdup.s8    d9, d24[1]          \n"
                    "vdup.s8    d10, d24[2]         \n"
                    "vdup.s8    d11, d24[3]         \n"
                    "vdup.s8    d12, d24[4]         \n"
                    "vdup.s8    d13, d24[5]         \n"
                    "vdup.s8    d14, d24[6]         \n"
                    "vdup.s8    d15, d24[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr2_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32    {d20-d23}, [%3:128] \n" //outptr2_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%3:128]!\n"
                    //#############################################
                    //load inch kernel_3 k0-k7
                    "vdup.s8    d8, d25[0]          \n"
                    "vdup.s8    d9, d25[1]          \n"
                    "vdup.s8    d10, d25[2]         \n"
                    "vdup.s8    d11, d25[3]         \n"
                    "vdup.s8    d12, d25[4]         \n"
                    "vdup.s8    d13, d25[5]         \n"
                    "vdup.s8    d14, d25[6]         \n"
                    "vdup.s8    d15, d25[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr3_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32    {d20-d23}, [%4:128] \n" //outptr3_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%4:128]!\n"

                    //next
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),          // %0
                      "=r"(outptr0),     // %1
                      "=r"(outptr1),     // %2
                      "=r"(outptr2),     // %3
                      "=r"(outptr3),     // %4
                      "=r"(r0),          // %5
                      "=r"(r1),          // %6
                      "=r"(r2),          // %7
                      "=r"(r3),          // %8
                      "=r"(r4),          // %9
                      "=r"(r5),          // %10
                      "=r"(r6),          // %11
                      "=r"(r7)           // %12
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "9"(r4),
                      "10"(r5),
                      "11"(r6),
                      "12"(r7)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10", "q11", "q13", "q14", "q15"
                );
            }

            for (; remain>0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                int sum1 = (int)*r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                int sum2 = (int)*r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                int sum3 = (int)*r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* img0_s8 = bottom_blob.channel(q);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;

            const signed char k0 = kernel0[0];
            const signed char k1 = kernel1[0];
            const signed char k2 = kernel2[0];
            const signed char k3 = kernel3[0];

            const signed char* r0 = img0_s8;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);

            if (nn > 0)
            {
                asm volatile(
                    "0:                             \n"
                    //load r0
                    "pld        [%5, #64]           \n"
                    "vld1.s8    {d8}, [%5 :64]!     \n"

                    //mla
                    "vmull.s8   q5, d8, %12         \n"
                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d12-d15}, [%1]     \n"
                    "vmovl.s16  q8, d10             \n"
                    "vmovl.s16  q9, d11             \n"
                    "vadd.s32   q6, q8              \n"
                    "vadd.s32   q7, q9              \n"
                    "vst1.32    {d12-d15}, [%1]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %13         \n"
                    //outptr1_s32
                    "pld        [%2, #256]          \n"
                    "vld1.32    {d12-d15}, [%2]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%2]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %14         \n"
                    //outptr0_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32    {d12-d15}, [%3]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%3]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %15         \n"
                    //outptr0_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32    {d12-d15}, [%4]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%4]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0),        // %1
                      "=r"(outptr1),        // %2
                      "=r"(outptr2),        // %3
                      "=r"(outptr3),        // %4
                      "=r"(r0)              // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "w"(_k0),             // %12
                      "w"(_k1),             // %13
                      "w"(_k2),             // %14
                      "w"(_k3)              // %15
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * k0;
                int sum1 = (int)*r0 * k1;
                int sum2 = (int)*r0 * k2;
                int sum3 = (int)*r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            if (nn > 0)
            {
                //load inch kernel_0 k0-k7
                asm volatile(
                    "vld1.s8    d18, [%0]   \n"
                    : "=r"(kernel0) // %0
                    : "0" (kernel0)
                    :
                );

                asm volatile(
                    "0:                            \n"
                    //ld r0-r7
                    "pld        [%2, #64]          \n"
                    "vld1.s8    {d0}, [%2 :64]!    \n"  //r0
                    "pld        [%3, #64]          \n"
                    "vld1.s8    {d1}, [%3 :64]!    \n"  //r1
                    "pld        [%4, #64]          \n"
                    "vld1.s8    {d2}, [%4 :64]!    \n"  //r2
                    "pld        [%5, #64]          \n"
                    "vld1.s8    {d3}, [%5 :64]!    \n"  //r3
                    "pld        [%6, #64]          \n"
                    "vld1.s8    {d4}, [%6 :64]!    \n"  //r4
                    "pld        [%7, #64]          \n"
                    "vld1.s8    {d5}, [%7 :64]!    \n"  //r5
                    "pld        [%8, #64]          \n"
                    "vld1.s8    {d6}, [%8 :64]!    \n"  //r6
                    "pld        [%9, #64]          \n"
                    "vld1.s8    {d7}, [%9 :64]!    \n"  //r7

                    //load inch kernel_0 k0-k7
                    "vdup.s8    d8, d18[0]          \n"
                    "vdup.s8    d9, d18[1]          \n"
                    "vdup.s8    d10, d18[2]         \n"
                    "vdup.s8    d11, d18[3]         \n"
                    "vdup.s8    d12, d18[4]         \n"
                    "vdup.s8    d13, d18[5]         \n"
                    "vdup.s8    d14, d18[6]         \n"
                    "vdup.s8    d15, d18[7]         \n"

                    //mla
                    "vmull.s8   q14, d0,    d8          \n"
                    "vmlal.s8   q14, d1,    d9          \n"
                    "vmlal.s8   q14, d2,    d10         \n"
                    "vmlal.s8   q14, d3,    d11         \n"
                    "vmlal.s8   q14, d4,    d12         \n"
                    "vmlal.s8   q14, d5,    d13         \n"
                    "vmlal.s8   q14, d6,    d14         \n"
                    "vmlal.s8   q14, d7,    d15         \n"

                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d20-d23}, [%1]     \n" //outptr0_s32
                    "vaddw.s16   q10, q10, d28      \n"
                    "vaddw.s16   q11, q11, d29      \n"
                    "vst1.32    {d20-d23}, [%1]!    \n"

                    //next
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),          // %0
                      "=r"(outptr0),     // %1
                      "=r"(r0),          // %2
                      "=r"(r1),          // %3
                      "=r"(r2),          // %4
                      "=r"(r3),          // %5
                      "=r"(r4),          // %6
                      "=r"(r5),          // %7
                      "=r"(r6),          // %8
                      "=r"(r7)           // %9
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "9"(r7)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11", "q12", "q13", "q14"
                );
            }

            for (; remain>0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0_s8 = bottom_blob.channel(q);
            const signed char* r0 = img0_s8;

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);

            if (nn > 0)
            {
                asm volatile(
                    "0:                             \n"
                    //load r0
                    "pld        [%2, #64]           \n"
                    "vld1.s8    {d8}, [%2 :64]!     \n"

                    //mla
                    "vmull.s8   q10, d8, %6         \n"
                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d12-d15}, [%1]     \n"
                    "vaddw.s16   q6, q6, d20        \n"
                    "vaddw.s16   q7, q7, d21        \n"
                    "vst1.32    {d12-d15}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0),        // %1
                      "=r"(r0)              // %2
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(r0),
                      "w"(_k0)              // %6
                    : "cc", "memory", "q4", "q10", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                int sum0 = (int)*r0 * k0;

                *outptr0 += sum0;

                r0++;
                outptr0++;
            }
        }
    }
}

static void conv1x1s1_neon_s8_left4(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;

            int size = outw * outh;

            int nn = size >> 3;

            asm volatile(
                "vld1.s8    d18, [%0]   \n"
                "vld1.s8    d19, [%1]   \n"
                "vld1.s8    d24, [%2]   \n"
                "vld1.s8    d25, [%3]   \n"
                : "=r"(kernel0), // %0
                  "=r"(kernel1), // %1
                  "=r"(kernel2), // %2
                  "=r"(kernel3)  // %3
                : "0"(kernel0),
                  "1"(kernel1),
                  "2"(kernel2),
                  "3"(kernel3)
                :
            );

            if (nn > 0)
            {
                asm volatile(
                    "0:                            \n"
                    //ld r0-r7
                    "pld        [%5, #64]          \n"
                    "vld1.s8    {d0}, [%5 :64]!    \n"  //r0

                    "pld        [%6, #64]          \n"
                    "vld1.s8    {d1}, [%6 :64]!    \n"  //r1

                    "pld        [%7, #64]          \n"
                    "vld1.s8    {d2}, [%7 :64]!    \n"  //r2

                    "pld        [%8, #64]          \n"
                    "vld1.s8    {d3}, [%8 :64]!    \n"  //r3

                    "pld        [%9, #64]          \n"
                    "vld1.s8    {d4}, [%9 :64]!    \n"  //r4

                    "pld        [%10, #64]         \n"
                    "vld1.s8    {d5}, [%10 :64]!   \n"  //r5

                    "pld        [%11, #64]         \n"
                    "vld1.s8    {d6}, [%11 :64]!   \n"  //r6

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d7}, [%12 :64]!   \n"  //r7
                    //###########################################
                    //load inch kernel_0 k0-k7
                    "vdup.s8    d8, d18[0]          \n"
                    "vdup.s8    d9, d18[1]          \n"
                    "vdup.s8    d10, d18[2]         \n"
                    "vdup.s8    d11, d18[3]         \n"
                    "vdup.s8    d12, d18[4]         \n"
                    "vdup.s8    d13, d18[5]         \n"
                    "vdup.s8    d14, d18[6]         \n"
                    "vdup.s8    d15, d18[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d20-d23}, [%1:128] \n" //outptr0_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%1:128]!\n"
                    //###########################################
                    //load inch kernel_1 k0-k7
                    "vdup.s8    d8, d19[0]          \n"
                    "vdup.s8    d9, d19[1]          \n"
                    "vdup.s8    d10, d19[2]         \n"
                    "vdup.s8    d11, d19[3]         \n"
                    "vdup.s8    d12, d19[4]         \n"
                    "vdup.s8    d13, d19[5]         \n"
                    "vdup.s8    d14, d19[6]         \n"
                    "vdup.s8    d15, d19[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr1_s32
                    "pld        [%2, #256]          \n"
                    "vld1.32    {d20-d23}, [%2:128] \n" //outptr1_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%2:128]!\n"
                    //############################################
                    //load inch kernel_2 k0-k7
                    "vdup.s8    d8, d24[0]          \n"
                    "vdup.s8    d9, d24[1]          \n"
                    "vdup.s8    d10, d24[2]         \n"
                    "vdup.s8    d11, d24[3]         \n"
                    "vdup.s8    d12, d24[4]         \n"
                    "vdup.s8    d13, d24[5]         \n"
                    "vdup.s8    d14, d24[6]         \n"
                    "vdup.s8    d15, d24[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr2_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32    {d20-d23}, [%3:128] \n" //outptr2_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%3:128]!\n"
                    //#############################################
                    //load inch kernel_3 k0-k7
                    "vdup.s8    d8, d25[0]          \n"
                    "vdup.s8    d9, d25[1]          \n"
                    "vdup.s8    d10, d25[2]         \n"
                    "vdup.s8    d11, d25[3]         \n"
                    "vdup.s8    d12, d25[4]         \n"
                    "vdup.s8    d13, d25[5]         \n"
                    "vdup.s8    d14, d25[6]         \n"
                    "vdup.s8    d15, d25[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr3_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32    {d20-d23}, [%4:128] \n" //outptr3_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%4:128]!\n"

                    //next
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),          // %0
                      "=r"(outptr0),     // %1
                      "=r"(outptr1),     // %2
                      "=r"(outptr2),     // %3
                      "=r"(outptr3),     // %4
                      "=r"(r0),          // %5
                      "=r"(r1),          // %6
                      "=r"(r2),          // %7
                      "=r"(r3),          // %8
                      "=r"(r4),          // %9
                      "=r"(r5),          // %10
                      "=r"(r6),          // %11
                      "=r"(r7)           // %12
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "9"(r4),
                      "10"(r5),
                      "11"(r6),
                      "12"(r7)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10", "q11"
                );
            }

            asm volatile(
                    "0:                            \n"
                    //ld r0-r7
                    "pld        [%5, #64]          \n"
                    "vld1.s8    {d0}, [%5 :64]     \n"  //r0

                    "pld        [%6, #64]          \n"
                    "vld1.s8    {d1}, [%6 :64]     \n"  //r1

                    "pld        [%7, #64]          \n"
                    "vld1.s8    {d2}, [%7 :64]     \n"  //r2

                    "pld        [%8, #64]          \n"
                    "vld1.s8    {d3}, [%8 :64]     \n"  //r3

                    "pld        [%9, #64]          \n"
                    "vld1.s8    {d4}, [%9 :64]     \n"  //r4

                    "pld        [%10, #64]         \n"
                    "vld1.s8    {d5}, [%10 :64]    \n"  //r5

                    "pld        [%11, #64]         \n"
                    "vld1.s8    {d6}, [%11 :64]    \n"  //r6

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d7}, [%12 :64]    \n"  //r7

                    "add        %5, #4             \n"
                    "add        %6, #4             \n"
                    "add        %7, #4             \n"
                    "add        %8, #4             \n"
                    "add        %9, #4             \n"
                    "add        %10, #4            \n"
                    "add        %11, #4            \n"
                    "add        %12, #4            \n"
                    //###########################################
                    //load inch kernel_0 k0-k7
                    "vdup.s8    d8, d18[0]          \n"
                    "vdup.s8    d9, d18[1]          \n"
                    "vdup.s8    d10, d18[2]         \n"
                    "vdup.s8    d11, d18[3]         \n"
                    "vdup.s8    d12, d18[4]         \n"
                    "vdup.s8    d13, d18[5]         \n"
                    "vdup.s8    d14, d18[6]         \n"
                    "vdup.s8    d15, d18[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr0_s32
                    "pld        [%1, #128]          \n"
                    "vld1.32    {d20-d21}, [%1:128] \n" //outptr0_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%1:128]!\n"
                    //###########################################
                    //load inch kernel_1 k0-k7
                    "vdup.s8    d8, d19[0]          \n"
                    "vdup.s8    d9, d19[1]          \n"
                    "vdup.s8    d10, d19[2]         \n"
                    "vdup.s8    d11, d19[3]         \n"
                    "vdup.s8    d12, d19[4]         \n"
                    "vdup.s8    d13, d19[5]         \n"
                    "vdup.s8    d14, d19[6]         \n"
                    "vdup.s8    d15, d19[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr1_s32
                    "pld        [%2, #128]          \n"
                    "vld1.32    {d20-d21}, [%2:128] \n" //outptr1_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%2:128]!\n"
                    //############################################
                    //load inch kernel_2 k0-k7
                    "vdup.s8    d8, d24[0]          \n"
                    "vdup.s8    d9, d24[1]          \n"
                    "vdup.s8    d10, d24[2]         \n"
                    "vdup.s8    d11, d24[3]         \n"
                    "vdup.s8    d12, d24[4]         \n"
                    "vdup.s8    d13, d24[5]         \n"
                    "vdup.s8    d14, d24[6]         \n"
                    "vdup.s8    d15, d24[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr2_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32    {d20-d21}, [%3:128] \n" //outptr2_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%3:128]!\n"
                    //#############################################
                    //load inch kernel_3 k0-k7
                    "vdup.s8    d8, d25[0]          \n"
                    "vdup.s8    d9, d25[1]          \n"
                    "vdup.s8    d10, d25[2]         \n"
                    "vdup.s8    d11, d25[3]         \n"
                    "vdup.s8    d12, d25[4]         \n"
                    "vdup.s8    d13, d25[5]         \n"
                    "vdup.s8    d14, d25[6]         \n"
                    "vdup.s8    d15, d25[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr3_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32    {d20-d21}, [%4:128] \n" //outptr3_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vst1.32    {d20-d21}, [%4:128]!\n"
                    : "=r"(nn),          // %0
                      "=r"(outptr0),     // %1
                      "=r"(outptr1),     // %2
                      "=r"(outptr2),     // %3
                      "=r"(outptr3),     // %4
                      "=r"(r0),          // %5
                      "=r"(r1),          // %6
                      "=r"(r2),          // %7
                      "=r"(r3),          // %8
                      "=r"(r4),          // %9
                      "=r"(r5),          // %10
                      "=r"(r6),          // %11
                      "=r"(r7)           // %12
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "9"(r4),
                      "10"(r5),
                      "11"(r6),
                      "12"(r7)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q10", "q11"
                );

        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char* img0_s8 = bottom_blob.channel(q);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;

            const signed char k0 = kernel0[0];
            const signed char k1 = kernel1[0];
            const signed char k2 = kernel2[0];
            const signed char k3 = kernel3[0];

            const signed char* r0 = img0_s8;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);

            if (nn > 0)
            {
                asm volatile(
                    "0:                             \n"
                    //load r0
                    "pld        [%5, #64]           \n"
                    "vld1.s8    {d8}, [%5 :64]!     \n"

                    //mla
                    "vmull.s8   q5, d8, %12         \n"
                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d12-d15}, [%1]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%1]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %13         \n"
                    //outptr1_s32
                    "pld        [%2, #256]          \n"
                    "vld1.32    {d12-d15}, [%2]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%2]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %14         \n"
                    //outptr0_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32    {d12-d15}, [%3]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%3]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %15         \n"
                    //outptr0_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32    {d12-d15}, [%4]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%4]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0),        // %1
                      "=r"(outptr1),        // %2
                      "=r"(outptr2),        // %3
                      "=r"(outptr3),        // %4
                      "=r"(r0)              // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "w"(_k0),             // %12
                      "w"(_k1),             // %13
                      "w"(_k2),             // %14
                      "w"(_k3)              // %15
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * k0;
                int sum1 = (int)*r0 * k1;
                int sum2 = (int)*r0 * k2;
                int sum3 = (int)*r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            if (nn > 0)
            {
                //load inch kernel_0 k0-k7
                asm volatile(
                    "vld1.s8    d18, [%0]   \n"
                    : "=r"(kernel0) // %0
                    : "0" (kernel0)
                    :
                );

                asm volatile(
                    "0:                            \n"
                    //ld r0-r7
                    "pld        [%2, #64]          \n"
                    "vld1.s8    {d0}, [%2 :64]!    \n"  //r0
                    "pld        [%3, #64]          \n"
                    "vld1.s8    {d1}, [%3 :64]!    \n"  //r1
                    "pld        [%4, #64]          \n"
                    "vld1.s8    {d2}, [%4 :64]!    \n"  //r2
                    "pld        [%5, #64]          \n"
                    "vld1.s8    {d3}, [%5 :64]!    \n"  //r3
                    "pld        [%6, #64]          \n"
                    "vld1.s8    {d4}, [%6 :64]!    \n"  //r4
                    "pld        [%7, #64]          \n"
                    "vld1.s8    {d5}, [%7 :64]!    \n"  //r5
                    "pld        [%8, #64]          \n"
                    "vld1.s8    {d6}, [%8 :64]!    \n"  //r6
                    "pld        [%9, #64]          \n"
                    "vld1.s8    {d7}, [%9 :64]!    \n"  //r7

                    //load inch kernel_0 k0-k7
                    "vdup.s8    d8, d18[0]          \n"
                    "vdup.s8    d9, d18[1]          \n"
                    "vdup.s8    d10, d18[2]         \n"
                    "vdup.s8    d11, d18[3]         \n"
                    "vdup.s8    d12, d18[4]         \n"
                    "vdup.s8    d13, d18[5]         \n"
                    "vdup.s8    d14, d18[6]         \n"
                    "vdup.s8    d15, d18[7]         \n"

                    //mla
                    "vmull.s8   q8, d0, d8          \n"
                    "vmlal.s8   q8, d1, d9          \n"
                    "vmlal.s8   q8, d2, d10         \n"
                    "vmlal.s8   q8, d3, d11         \n"
                    "vmlal.s8   q8, d4, d12         \n"
                    "vmlal.s8   q8, d5, d13         \n"
                    "vmlal.s8   q8, d6, d14         \n"
                    "vmlal.s8   q8, d7, d15         \n"

                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d20-d23}, [%1]     \n" //outptr0_s32
                    "vaddw.s16   q10, q10, d16      \n"
                    "vaddw.s16   q11, q11, d17      \n"
                    "vst1.32    {d20-d23}, [%1]!    \n"

                    //next
                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),          // %0
                      "=r"(outptr0),     // %1
                      "=r"(r0),          // %2
                      "=r"(r1),          // %3
                      "=r"(r2),          // %4
                      "=r"(r3),          // %5
                      "=r"(r4),          // %6
                      "=r"(r5),          // %7
                      "=r"(r6),          // %8
                      "=r"(r7)           // %9
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "9"(r7)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q10", "q11", "q12", "q13"
                );
            }

            for (; remain>0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0_s8 = bottom_blob.channel(q);
            const signed char* r0 = img0_s8;

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);

            if (nn > 0)
            {
                asm volatile(
                    "0:                             \n"
                    //load r0
                    "pld        [%2, #64]           \n"
                    "vld1.s8    {d8}, [%2 :64]!     \n"

                    //mla
                    "vmull.s8   q5, d8, %6          \n"
                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32    {d12-d15}, [%1]     \n"
                    "vaddw.s16   q6, q6, d10        \n"
                    "vaddw.s16   q7, q7, d11        \n"
                    "vst1.32    {d12-d15}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0),        // %1
                      "=r"(r0)              // %2
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(r0),
                      "w"(_k0)              // %6
                    : "cc", "memory", "q4", "q5", "q7", "q8", "q9"
                );
            }

            for (; remain>0; remain--)
            {
                int sum0 = (int)*r0 * k0;

                *outptr0 += sum0;

                r0++;
                outptr0++;
            }
        }
    }
}

static void conv1x1s1_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int size = top_blob.h * top_blob.w;
    int remain = size & 7;

    typedef void (*conv_func_int8)(const Mat&, Mat&, const Mat&, const Option&);

    conv_func_int8 conv_func_table[8] =
    {
        conv1x1s1_neon_s8,          //0
        conv1x1s1_neon_s8,          //1
        conv1x1s1_neon_s8,          //2
        conv1x1s1_neon_s8,          //3
        conv1x1s1_neon_s8_left4,    //4
        conv1x1s1_neon_s8,          //5
        conv1x1s1_neon_s8,          //6
        conv1x1s1_neon_s8,          //7
    };

    conv_func_int8 conv = conv_func_table[remain];

    conv(bottom_blob, top_blob, _kernel, opt);

    return;
}
#endif // __aarch64__

static void conv1x1s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;
    const signed char *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            const signed char *r0 = bottom_blob.channel(q);
            const signed char *r1 = bottom_blob.channel(q + 1);
            const signed char *r2 = bottom_blob.channel(q + 2);
            const signed char *r3 = bottom_blob.channel(q + 3);
            const signed char *r4 = bottom_blob.channel(q + 4);
            const signed char *r5 = bottom_blob.channel(q + 5);
            const signed char *r6 = bottom_blob.channel(q + 6);
            const signed char *r7 = bottom_blob.channel(q + 7);

            for(int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    //ToDo Neon
                    int sum0 = (int)*r0 * (int)kernel0[0] + (int)*r1 * (int)kernel0[1] +
                            (int)*r2 * (int)kernel0[2] + (int)*r3 * (int)kernel0[3] +
                            (int)*r4 * (int)kernel0[4] + (int)*r5 * (int)kernel0[5] +
                            (int)*r6 * (int)kernel0[6] + (int)*r7 * (int)kernel0[7];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                    r7 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;
                r7 += tailstep;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char *r0 = bottom_blob.channel(q);

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            for(int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    //ToDo Neon
                    int sum0 = (int)*r0 * (int)kernel0[0];

                    *outptr0 += sum0;

                    r0 += 2;
                    outptr0++;
                }

                r0 += tailstep;
            }
        }
    }
}
