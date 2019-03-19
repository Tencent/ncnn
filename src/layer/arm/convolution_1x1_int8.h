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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static void conv1x1s1_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const signed char* kernel = _kernel;

#if __ARM_NEON && __aarch64__
    kernel_tm.create(4*8, inch/4 + inch%4, outch/8 + (outch%8)/4 + outch%4, (size_t)1u);
#else
    kernel_tm.create(4*4, inch/4 + inch%4, outch/4 + outch%4, (size_t)1u);
#endif // __ARM_NEON && __aarch64__    

    int p = 0;
#if __ARM_NEON && __aarch64__
    for (; p+7<outch; p+=8)
    {
        const signed char* kernel0 = kernel + (p+0)*inch;
        const signed char* kernel1 = kernel + (p+1)*inch;
        const signed char* kernel2 = kernel + (p+2)*inch;
        const signed char* kernel3 = kernel + (p+3)*inch;
        const signed char* kernel4 = kernel + (p+4)*inch;
        const signed char* kernel5 = kernel + (p+5)*inch;
        const signed char* kernel6 = kernel + (p+6)*inch;
        const signed char* kernel7 = kernel + (p+7)*inch;

        signed char* ktmp = kernel_tm.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            // kernel0...7 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];
            ktmp[4] = kernel4[0];
            ktmp[5] = kernel5[0];
            ktmp[6] = kernel6[0];
            ktmp[7] = kernel7[0];

            ktmp += 8;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
            kernel4 += 1;
            kernel5 += 1;
            kernel6 += 1;
            kernel7 += 1;
        }
    }
#endif // __ARM_NEON && __aarch64__    
    for (; p+3<outch; p+=4)
    {
        const signed char* kernel0 = kernel + (p+0)*inch;
        const signed char* kernel1 = kernel + (p+1)*inch;
        const signed char* kernel2 = kernel + (p+2)*inch;
        const signed char* kernel3 = kernel + (p+3)*inch;

#if __ARM_NEON && __aarch64__
        signed char* ktmp = kernel_tm.channel(p/8 + (p%8)/4);
#else
        signed char* ktmp = kernel_tm.channel(p/4);
#endif // __ARM_NEON && __aarch64__

        for (int q=0; q<inch; q++)
        {
            // kernel0...3 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];

            ktmp += 4;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
        }
    }

    for (; p<outch; p++)
    {
        const signed char* kernel0 = kernel + p*inch;

#if __ARM_NEON && __aarch64__
        signed char* ktmp = kernel_tm.channel(p/8 + (p%8)/4 + p%4);
#else
        signed char* ktmp = kernel_tm.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__

        for (int q=0; q<inch; q++)
        {
            ktmp[0] = kernel0[0];
            ktmp++;
            kernel0++;
        }
    }  
}

/*
 * Convolution 1x1 quantized with sgemm int8
 */
static void conv1x1s1_sgemm_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    // interleave
    Mat tmp(8*4, inch/4+inch%4, size/8 + (size%8)/4 + size%4, 1u, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i/8);

            for (int q=0; q<inch; q++)
            {
#if __ARM_NEON                
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]    \n"
                    "ld1    {v0.8b}, [%0]            \n"
                    "st1    {v0.8b}, [%1], #8        \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "cc", "memory", "v0"
                );
#else
                asm volatile(
                    "pld        [%0, #64]     \n"
                    "vld1.s8   {d0}, [%0]     \n"
                    "vst1.s8   {d0}, [%1]!    \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "d0"
                );
#endif // __aarch64__            
                img0 += bottom_blob.cstep;
#else                
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
                tmpptr[4] = img0[4];
                tmpptr[5] = img0[5];
                tmpptr[6] = img0[6];
                tmpptr[7] = img0[7];

                tmpptr += 8;
                img0 += bottom_blob.cstep;
#endif // __ARM_NEON__                
            }
        }

        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i/8 + (i%8)/4);

            for (int q=0; q<inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
            }            
        }        

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<size; i++)
        {
            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i/8 + (i%8)/4 + i%4);

            for (int q=0; q<inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }

    // sgemm process
    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p+1);
        int* outptr2 = top_blob.channel(p+2);
        int* outptr3 = top_blob.channel(p+3);
        int* outptr4 = top_blob.channel(p+4);
        int* outptr5 = top_blob.channel(p+5);
        int* outptr6 = top_blob.channel(p+6);
        int* outptr7 = top_blob.channel(p+7);

        int i = 0;

        for (; i+7<size; i+=8)
        {
            const signed char* tmpptr = tmp.channel(i/8);
            const signed char* kptr = kernel.channel(p/8);

#if __ARM_NEON && __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum0n
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum1
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum1n
                "eor    v20.16b, v20.16b, v20.16b    \n" // sum2
                "eor    v21.16b, v21.16b, v21.16b    \n" // sum2n
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum3
                "eor    v23.16b, v23.16b, v23.16b    \n" // sum3n
                "eor    v24.16b, v24.16b, v24.16b    \n" // sum4
                "eor    v25.16b, v25.16b, v25.16b    \n" // sum4n
                "eor    v26.16b, v26.16b, v26.16b    \n" // sum5
                "eor    v27.16b, v27.16b, v27.16b    \n" // sum5n
                "eor    v28.16b, v28.16b, v28.16b    \n" // sum6
                "eor    v29.16b, v29.16b, v29.16b    \n" // sum6n
                "eor    v30.16b, v30.16b, v30.16b    \n" // sum7
                "eor    v31.16b, v31.16b, v31.16b    \n" // sum7n

                // inch loop
                "lsr    w4, %w20, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%9, #128]                     \n" // k
                "ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%9], #32     \n"

                //"prfm   pldl1keep, [%8, #128]                     \n" // d
                "ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%8], #32   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
                "sshll    v1.8h, v1.8b, #0           \n" // k01 - k71
                "sshll    v2.8h, v2.8b, #0           \n" // k02 - k72
                "sshll    v3.8h, v3.8b, #0           \n" // k03 - k73

                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a70
                "sshll    v9.8h, v9.8b, #0           \n" // a01 - a71
                "sshll    v10.8h, v10.8b, #0         \n" // a02 - a72
                "sshll    v11.8h, v11.8b, #0         \n" // a03 - a73

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a70) * k00
                "smlal2   v17.4s, v8.8h, v0.h[0]     \n"//
                "smlal    v18.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a70) * k10
                "smlal2   v19.4s, v8.8h, v0.h[1]     \n"//
                "smlal    v20.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a70) * k20
                "smlal2   v21.4s, v8.8h, v0.h[2]     \n"//
                "smlal    v22.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a70) * k30
                "smlal2   v23.4s, v8.8h, v0.h[3]     \n"//
                "smlal    v24.4s, v8.4h, v0.h[4]     \n"// sum4 += (a00-a70) * k40
                "smlal2   v25.4s, v8.8h, v0.h[4]     \n"//
                "smlal    v26.4s, v8.4h, v0.h[5]     \n"// sum5 += (a00-a70) * k50
                "smlal2   v27.4s, v8.8h, v0.h[5]     \n"//
                "smlal    v28.4s, v8.4h, v0.h[6]     \n"// sum6 += (a00-a70) * k60
                "smlal2   v29.4s, v8.8h, v0.h[6]     \n"//
                "smlal    v30.4s, v8.4h, v0.h[7]     \n"// sum7 += (a00-a70) * k70
                "smlal2   v31.4s, v8.8h, v0.h[7]     \n"//
                // k1
                "smlal    v16.4s, v9.4h, v1.h[0]     \n"// sum0 += (a01-a71) * k01
                "smlal2   v17.4s, v9.8h, v1.h[0]     \n"//
                "smlal    v18.4s, v9.4h, v1.h[1]     \n"// sum1 += (a01-a71) * k11
                "smlal2   v19.4s, v9.8h, v1.h[1]     \n"//
                "smlal    v20.4s, v9.4h, v1.h[2]     \n"// sum2 += (a01-a71) * k21
                "smlal2   v21.4s, v9.8h, v1.h[2]     \n"//
                "smlal    v22.4s, v9.4h, v1.h[3]     \n"// sum3 += (a01-a71) * k31
                "smlal2   v23.4s, v9.8h, v1.h[3]     \n"//
                "smlal    v24.4s, v9.4h, v1.h[4]     \n"// sum4 += (a01-a71) * k41
                "smlal2   v25.4s, v9.8h, v1.h[4]     \n"//
                "smlal    v26.4s, v9.4h, v1.h[5]     \n"// sum5 += (a01-a71) * k51
                "smlal2   v27.4s, v9.8h, v1.h[5]     \n"//
                "smlal    v28.4s, v9.4h, v1.h[6]     \n"// sum6 += (a01-a71) * k61
                "smlal2   v29.4s, v9.8h, v1.h[6]     \n"//
                "smlal    v30.4s, v9.4h, v1.h[7]     \n"// sum7 += (a01-a71) * k71
                "smlal2   v31.4s, v9.8h, v1.h[7]     \n"//
                // k2
                "smlal    v16.4s, v10.4h, v2.h[0]    \n"// sum0 += (a02-a72) * k02
                "smlal2   v17.4s, v10.8h, v2.h[0]    \n"//
                "smlal    v18.4s, v10.4h, v2.h[1]    \n"// sum1 += (a02-a72) * k12
                "smlal2   v19.4s, v10.8h, v2.h[1]    \n"//
                "smlal    v20.4s, v10.4h, v2.h[2]    \n"// sum2 += (a02-a72) * k22
                "smlal2   v21.4s, v10.8h, v2.h[2]    \n"//
                "smlal    v22.4s, v10.4h, v2.h[3]    \n"// sum3 += (a02-a72) * k32
                "smlal2   v23.4s, v10.8h, v2.h[3]    \n"//
                "smlal    v24.4s, v10.4h, v2.h[4]    \n"// sum4 += (a02-a72) * k42
                "smlal2   v25.4s, v10.8h, v2.h[4]    \n"//
                "smlal    v26.4s, v10.4h, v2.h[5]    \n"// sum5 += (a02-a72) * k52
                "smlal2   v27.4s, v10.8h, v2.h[5]    \n"//
                "smlal    v28.4s, v10.4h, v2.h[6]    \n"// sum6 += (a02-a72) * k62
                "smlal2   v29.4s, v10.8h, v2.h[6]    \n"//
                "smlal    v30.4s, v10.4h, v2.h[7]    \n"// sum7 += (a02-a72) * k72
                "smlal2   v31.4s, v10.8h, v2.h[7]    \n"//

                "subs   w4, w4, #1                   \n"

                // k3
                "smlal    v16.4s, v11.4h, v3.h[0]    \n"// sum0 += (a03-a73) * k03
                "smlal2   v17.4s, v11.8h, v3.h[0]    \n"//
                "smlal    v18.4s, v11.4h, v3.h[1]    \n"// sum1 += (a03-a73) * k13
                "smlal2   v19.4s, v11.8h, v3.h[1]    \n"//
                "smlal    v20.4s, v11.4h, v3.h[2]    \n"// sum2 += (a03-a73) * k23
                "smlal2   v21.4s, v11.8h, v3.h[2]    \n"//
                "smlal    v22.4s, v11.4h, v3.h[3]    \n"// sum3 += (a03-a73) * k33
                "smlal2   v23.4s, v11.8h, v3.h[3]    \n"//
                "smlal    v24.4s, v11.4h, v3.h[4]    \n"// sum4 += (a03-a73) * k43
                "smlal2   v25.4s, v11.8h, v3.h[4]    \n"//
                "smlal    v26.4s, v11.4h, v3.h[5]    \n"// sum5 += (a03-a73) * k53
                "smlal2   v27.4s, v11.8h, v3.h[5]    \n"//
                "smlal    v28.4s, v11.4h, v3.h[6]    \n"// sum6 += (a03-a73) * k63
                "smlal2   v29.4s, v11.8h, v3.h[6]    \n"//
                "smlal    v30.4s, v11.4h, v3.h[7]    \n"// sum7 += (a03-a73) * k73
                "smlal2   v31.4s, v11.8h, v3.h[7]    \n"//

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w20, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%9, #128]      \n"
                "ld1    {v0.8b}, [%9], #8            \n"

                //"prfm   pldl1keep, [%8, #128]      \n"
                "ld1    {v8.8b}, [%8], #8            \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a70

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a70) * k00
                "smlal2   v17.4s, v8.8h, v0.h[0]     \n"//
                "smlal    v18.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a70) * k10
                "smlal2   v19.4s, v8.8h, v0.h[1]     \n"//
                "smlal    v20.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a70) * k20
                "smlal2   v21.4s, v8.8h, v0.h[2]     \n"//
                "smlal    v22.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a70) * k30
                "smlal2   v23.4s, v8.8h, v0.h[3]     \n"//
                "smlal    v24.4s, v8.4h, v0.h[4]     \n"// sum4 += (a00-a70) * k40
                "smlal2   v25.4s, v8.8h, v0.h[4]     \n"//
                "smlal    v26.4s, v8.4h, v0.h[5]     \n"// sum5 += (a00-a70) * k50
                "smlal2   v27.4s, v8.8h, v0.h[5]     \n"//
                "smlal    v28.4s, v8.4h, v0.h[6]     \n"// sum6 += (a00-a70) * k60
                "smlal2   v29.4s, v8.8h, v0.h[6]     \n"//
                "smlal    v30.4s, v8.4h, v0.h[7]     \n"// sum7 += (a00-a70) * k70
                "smlal2   v31.4s, v8.8h, v0.h[7]     \n"//

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v16.4s, v17.4s}, [%0], #32  \n"
                "st1    {v18.4s, v19.4s}, [%1], #32  \n"
                "st1    {v20.4s, v21.4s}, [%2], #32  \n"
                "st1    {v22.4s, v23.4s}, [%3], #32  \n"
                "st1    {v24.4s, v25.4s}, [%4], #32  \n"
                "st1    {v26.4s, v27.4s}, [%5], #32  \n"
                "st1    {v28.4s, v29.4s}, [%6], #32  \n"
                "st1    {v30.4s, v31.4s}, [%7], #32  \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(outptr4),    // %4
                  "=r"(outptr5),    // %5
                  "=r"(outptr6),    // %6
                  "=r"(outptr7),    // %7
                  "=r"(tmpptr),     // %8
                  "=r"(kptr)        // %9
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(outptr4),
                  "5"(outptr5),
                  "6"(outptr6),
                  "7"(outptr7),
                  "8"(tmpptr),
                  "9"(kptr),
                  "r"(inch)         // %20
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;
            int sum0_4 = 0;
            int sum0_5 = 0;
            int sum0_6 = 0;
            int sum0_7 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;
            int sum1_4 = 0;
            int sum1_5 = 0;
            int sum1_6 = 0;
            int sum1_7 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;
            int sum2_4 = 0;
            int sum2_5 = 0;
            int sum2_6 = 0;
            int sum2_7 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;
            int sum3_4 = 0;
            int sum3_5 = 0;
            int sum3_6 = 0;
            int sum3_7 = 0;

            int sum4_0 = 0;
            int sum4_1 = 0;
            int sum4_2 = 0;
            int sum4_3 = 0;
            int sum4_4 = 0;
            int sum4_5 = 0;
            int sum4_6 = 0;
            int sum4_7 = 0;

            int sum5_0 = 0;
            int sum5_1 = 0;
            int sum5_2 = 0;
            int sum5_3 = 0;
            int sum5_4 = 0;
            int sum5_5 = 0;
            int sum5_6 = 0;
            int sum5_7 = 0;

            int sum6_0 = 0;
            int sum6_1 = 0;
            int sum6_2 = 0;
            int sum6_3 = 0;
            int sum6_4 = 0;
            int sum6_5 = 0;
            int sum6_6 = 0;
            int sum6_7 = 0;

            int sum7_0 = 0;
            int sum7_1 = 0;
            int sum7_2 = 0;
            int sum7_3 = 0;
            int sum7_4 = 0;
            int sum7_5 = 0;
            int sum7_6 = 0;
            int sum7_7 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0_0 += (int)tmpptr[0] * kptr[0];
                sum0_1 += (int)tmpptr[1] * kptr[0];
                sum0_2 += (int)tmpptr[2] * kptr[0];
                sum0_3 += (int)tmpptr[3] * kptr[0];
                sum0_4 += (int)tmpptr[4] * kptr[0];
                sum0_5 += (int)tmpptr[5] * kptr[0];
                sum0_6 += (int)tmpptr[6] * kptr[0];
                sum0_7 += (int)tmpptr[7] * kptr[0];

                sum1_0 += (int)tmpptr[0] * kptr[1];
                sum1_1 += (int)tmpptr[1] * kptr[1];
                sum1_2 += (int)tmpptr[2] * kptr[1];
                sum1_3 += (int)tmpptr[3] * kptr[1];
                sum1_4 += (int)tmpptr[4] * kptr[1];
                sum1_5 += (int)tmpptr[5] * kptr[1];
                sum1_6 += (int)tmpptr[6] * kptr[1];
                sum1_7 += (int)tmpptr[7] * kptr[1];

                sum2_0 += (int)tmpptr[0] * kptr[2];
                sum2_1 += (int)tmpptr[1] * kptr[2];
                sum2_2 += (int)tmpptr[2] * kptr[2];
                sum2_3 += (int)tmpptr[3] * kptr[2];
                sum2_4 += (int)tmpptr[4] * kptr[2];
                sum2_5 += (int)tmpptr[5] * kptr[2];
                sum2_6 += (int)tmpptr[6] * kptr[2];
                sum2_7 += (int)tmpptr[7] * kptr[2];

                sum3_0 += (int)tmpptr[0] * kptr[3];
                sum3_1 += (int)tmpptr[1] * kptr[3];
                sum3_2 += (int)tmpptr[2] * kptr[3];
                sum3_3 += (int)tmpptr[3] * kptr[3];
                sum3_4 += (int)tmpptr[4] * kptr[3];
                sum3_5 += (int)tmpptr[5] * kptr[3];
                sum3_6 += (int)tmpptr[6] * kptr[3];
                sum3_7 += (int)tmpptr[7] * kptr[3];

                sum4_0 += (int)tmpptr[0] * kptr[4];
                sum4_1 += (int)tmpptr[1] * kptr[4];
                sum4_2 += (int)tmpptr[2] * kptr[4];
                sum4_3 += (int)tmpptr[3] * kptr[4];
                sum4_4 += (int)tmpptr[4] * kptr[4];
                sum4_5 += (int)tmpptr[5] * kptr[4];
                sum4_6 += (int)tmpptr[6] * kptr[4];
                sum4_7 += (int)tmpptr[7] * kptr[4];

                sum5_0 += (int)tmpptr[0] * kptr[5];
                sum5_1 += (int)tmpptr[1] * kptr[5];
                sum5_2 += (int)tmpptr[2] * kptr[5];
                sum5_3 += (int)tmpptr[3] * kptr[5];
                sum5_4 += (int)tmpptr[4] * kptr[5];
                sum5_5 += (int)tmpptr[5] * kptr[5];
                sum5_6 += (int)tmpptr[6] * kptr[5];
                sum5_7 += (int)tmpptr[7] * kptr[5];

                sum6_0 += (int)tmpptr[0] * kptr[6];
                sum6_1 += (int)tmpptr[1] * kptr[6];
                sum6_2 += (int)tmpptr[2] * kptr[6];
                sum6_3 += (int)tmpptr[3] * kptr[6];
                sum6_4 += (int)tmpptr[4] * kptr[6];
                sum6_5 += (int)tmpptr[5] * kptr[6];
                sum6_6 += (int)tmpptr[6] * kptr[6];
                sum6_7 += (int)tmpptr[7] * kptr[6];

                sum7_0 += (int)tmpptr[0] * kptr[7];
                sum7_1 += (int)tmpptr[1] * kptr[7];
                sum7_2 += (int)tmpptr[2] * kptr[7];
                sum7_3 += (int)tmpptr[3] * kptr[7];
                sum7_4 += (int)tmpptr[4] * kptr[7];
                sum7_5 += (int)tmpptr[5] * kptr[7];
                sum7_6 += (int)tmpptr[6] * kptr[7];
                sum7_7 += (int)tmpptr[7] * kptr[7];

                tmpptr += 8;
                kptr += 8;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;
            outptr0[4] = sum0_4;
            outptr0[5] = sum0_5;
            outptr0[6] = sum0_6;
            outptr0[7] = sum0_7;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;
            outptr1[4] = sum1_4;
            outptr1[5] = sum1_5;
            outptr1[6] = sum1_6;
            outptr1[7] = sum1_7;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;
            outptr2[4] = sum2_4;
            outptr2[5] = sum2_5;
            outptr2[6] = sum2_6;
            outptr2[7] = sum2_7;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
            outptr3[4] = sum3_4;
            outptr3[5] = sum3_5;
            outptr3[6] = sum3_6;
            outptr3[7] = sum3_7;

            outptr4[0] = sum4_0;
            outptr4[1] = sum4_1;
            outptr4[2] = sum4_2;
            outptr4[3] = sum4_3;
            outptr4[4] = sum4_4;
            outptr4[5] = sum4_5;
            outptr4[6] = sum4_6;
            outptr4[7] = sum4_7;

            outptr5[0] = sum5_0;
            outptr5[1] = sum5_1;
            outptr5[2] = sum5_2;
            outptr5[3] = sum5_3;
            outptr5[4] = sum5_4;
            outptr5[5] = sum5_5;
            outptr5[6] = sum5_6;
            outptr5[7] = sum5_7;

            outptr6[0] = sum6_0;
            outptr6[1] = sum6_1;
            outptr6[2] = sum6_2;
            outptr6[3] = sum6_3;
            outptr6[4] = sum6_4;
            outptr6[5] = sum6_5;
            outptr6[6] = sum6_6;
            outptr6[7] = sum6_7;

            outptr7[0] = sum7_0;
            outptr7[1] = sum7_1;
            outptr7[2] = sum7_2;
            outptr7[3] = sum7_3;
            outptr7[4] = sum7_4;
            outptr7[5] = sum7_5;
            outptr7[6] = sum7_6;
            outptr7[7] = sum7_7;

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
            outptr4 += 8;
            outptr5 += 8;
            outptr6 += 8;
            outptr7 += 8;
#endif            
        }

        for (; i+3<size; i+=4)
        {
            const signed char* tmpptr = tmp.channel(i/8 + (i%8)/4);
            const signed char* kptr = kernel.channel(p/8);

#if __ARM_NEON && __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3
                "eor    v20.16b, v20.16b, v20.16b    \n" // sum4
                "eor    v21.16b, v21.16b, v21.16b    \n" // sum5
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum6
                "eor    v23.16b, v23.16b, v23.16b    \n" // sum7

                // inch loop
                "lsr    w4, %w20, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%9, #128]                     \n" // k
                "ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%9], #32     \n"

                //"prfm   pldl1keep, [%8, #128]      \n" // d
                "ld1    {v8.8b, v9.8b}, [%8], #16    \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
                "sshll    v1.8h, v1.8b, #0           \n" // k01 - k71
                "sshll    v2.8h, v2.8b, #0           \n" // k02 - k72
                "sshll    v3.8h, v3.8b, #0           \n" // k03 - k73

                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a30,a01 - a31
                "sshll    v9.8h, v9.8b, #0           \n" // a02 - a32,a03 - a33

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a30) * k00
                "smlal    v17.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a30) * k10
                "smlal    v18.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a30) * k20
                "smlal    v19.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a30) * k30
                "smlal    v20.4s, v8.4h, v0.h[4]     \n"// sum4 += (a00-a30) * k40
                "smlal    v21.4s, v8.4h, v0.h[5]     \n"// sum5 += (a00-a30) * k50
                "smlal    v22.4s, v8.4h, v0.h[6]     \n"// sum6 += (a00-a30) * k60
                "smlal    v23.4s, v8.4h, v0.h[7]     \n"// sum7 += (a00-a30) * k70
                // k1
                "smlal2    v16.4s, v8.8h, v1.h[0]    \n"// sum0 += (a01-a31) * k01
                "smlal2    v17.4s, v8.8h, v1.h[1]    \n"// sum1 += (a01-a31) * k11
                "smlal2    v18.4s, v8.8h, v1.h[2]    \n"// sum2 += (a01-a31) * k21
                "smlal2    v19.4s, v8.8h, v1.h[3]    \n"// sum3 += (a01-a31) * k31
                "smlal2    v20.4s, v8.8h, v1.h[4]    \n"// sum4 += (a01-a31) * k41
                "smlal2    v21.4s, v8.8h, v1.h[5]    \n"// sum5 += (a01-a31) * k51
                "smlal2    v22.4s, v8.8h, v1.h[6]    \n"// sum6 += (a01-a31) * k61
                "smlal2    v23.4s, v8.8h, v1.h[7]    \n"// sum7 += (a01-a31) * k71
                // k2
                "smlal    v16.4s, v9.4h, v2.h[0]     \n"// sum0 += (a02-a32) * k02
                "smlal    v17.4s, v9.4h, v2.h[1]     \n"// sum1 += (a02-a32) * k12
                "smlal    v18.4s, v9.4h, v2.h[2]     \n"// sum2 += (a02-a32) * k22
                "smlal    v19.4s, v9.4h, v2.h[3]     \n"// sum3 += (a02-a32) * k32
                "smlal    v20.4s, v9.4h, v2.h[4]     \n"// sum4 += (a02-a32) * k42
                "smlal    v21.4s, v9.4h, v2.h[5]     \n"// sum5 += (a02-a32) * k52
                "smlal    v22.4s, v9.4h, v2.h[6]     \n"// sum6 += (a02-a32) * k62
                "smlal    v23.4s, v9.4h, v2.h[7]     \n"// sum7 += (a02-a32) * k72

                "subs   w4, w4, #1                   \n"

                // k3
                "smlal2    v16.4s, v9.8h, v3.h[0]    \n"// sum0 += (a03-a33) * k03
                "smlal2    v17.4s, v9.8h, v3.h[1]    \n"// sum1 += (a03-a33) * k13
                "smlal2    v18.4s, v9.8h, v3.h[2]    \n"// sum2 += (a03-a33) * k23
                "smlal2    v19.4s, v9.8h, v3.h[3]    \n"// sum3 += (a03-a33) * k33
                "smlal2    v20.4s, v9.8h, v3.h[4]    \n"// sum4 += (a03-a33) * k43
                "smlal2    v21.4s, v9.8h, v3.h[5]    \n"// sum5 += (a03-a33) * k53
                "smlal2    v22.4s, v9.8h, v3.h[6]    \n"// sum6 += (a03-a33) * k63
                "smlal2    v23.4s, v9.8h, v3.h[7]    \n"// sum7 += (a03-a33) * k73

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w20, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%9, #128]      \n"
                "ld1    {v0.8b}, [%9], #8            \n"

                //"prfm   pldl1keep, [%8, #128]      \n"
                "ld1    {v8.8b}, [%8], #8            \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a70

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a30) * k00
                "smlal    v17.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a30) * k10
                "smlal    v18.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a30) * k20
                "smlal    v19.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a30) * k30
                "smlal    v20.4s, v8.4h, v0.h[4]     \n"// sum4 += (a00-a30) * k40
                "smlal    v21.4s, v8.4h, v0.h[5]     \n"// sum5 += (a00-a30) * k50
                "smlal    v22.4s, v8.4h, v0.h[6]     \n"// sum6 += (a00-a30) * k60
                "smlal    v23.4s, v8.4h, v0.h[7]     \n"// sum7 += (a00-a30) * k70

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v16.4s}, [%0], #16          \n"
                "st1    {v17.4s}, [%1], #16          \n"
                "st1    {v18.4s}, [%2], #16          \n"
                "st1    {v19.4s}, [%3], #16          \n"
                "st1    {v20.4s}, [%4], #16          \n"
                "st1    {v21.4s}, [%5], #16          \n"
                "st1    {v22.4s}, [%6], #16          \n"
                "st1    {v23.4s}, [%7], #16          \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(outptr4),    // %4
                  "=r"(outptr5),    // %5
                  "=r"(outptr6),    // %6
                  "=r"(outptr7),    // %7
                  "=r"(tmpptr),     // %8
                  "=r"(kptr)        // %9
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(outptr4),
                  "5"(outptr5),
                  "6"(outptr6),
                  "7"(outptr7),
                  "8"(tmpptr),
                  "9"(kptr),
                  "r"(inch)         // %20
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            );
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;

            int sum4_0 = 0;
            int sum4_1 = 0;
            int sum4_2 = 0;
            int sum4_3 = 0;

            int sum5_0 = 0;
            int sum5_1 = 0;
            int sum5_2 = 0;
            int sum5_3 = 0;

            int sum6_0 = 0;
            int sum6_1 = 0;
            int sum6_2 = 0;
            int sum6_3 = 0;

            int sum7_0 = 0;
            int sum7_1 = 0;
            int sum7_2 = 0;
            int sum7_3 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0_0 += (int)tmpptr[0] * kptr[0];
                sum0_1 += (int)tmpptr[1] * kptr[0];
                sum0_2 += (int)tmpptr[2] * kptr[0];
                sum0_3 += (int)tmpptr[3] * kptr[0];

                sum1_0 += (int)tmpptr[0] * kptr[1];
                sum1_1 += (int)tmpptr[1] * kptr[1];
                sum1_2 += (int)tmpptr[2] * kptr[1];
                sum1_3 += (int)tmpptr[3] * kptr[1];

                sum2_0 += (int)tmpptr[0] * kptr[2];
                sum2_1 += (int)tmpptr[1] * kptr[2];
                sum2_2 += (int)tmpptr[2] * kptr[2];
                sum2_3 += (int)tmpptr[3] * kptr[2];

                sum3_0 += (int)tmpptr[0] * kptr[3];
                sum3_1 += (int)tmpptr[1] * kptr[3];
                sum3_2 += (int)tmpptr[2] * kptr[3];
                sum3_3 += (int)tmpptr[3] * kptr[3];

                sum4_0 += (int)tmpptr[0] * kptr[4];
                sum4_1 += (int)tmpptr[1] * kptr[4];
                sum4_2 += (int)tmpptr[2] * kptr[4];
                sum4_3 += (int)tmpptr[3] * kptr[4];

                sum5_0 += (int)tmpptr[0] * kptr[5];
                sum5_1 += (int)tmpptr[1] * kptr[5];
                sum5_2 += (int)tmpptr[2] * kptr[5];
                sum5_3 += (int)tmpptr[3] * kptr[5];

                sum6_0 += (int)tmpptr[0] * kptr[6];
                sum6_1 += (int)tmpptr[1] * kptr[6];
                sum6_2 += (int)tmpptr[2] * kptr[6];
                sum6_3 += (int)tmpptr[3] * kptr[6];

                sum7_0 += (int)tmpptr[0] * kptr[7];
                sum7_1 += (int)tmpptr[1] * kptr[7];
                sum7_2 += (int)tmpptr[2] * kptr[7];
                sum7_3 += (int)tmpptr[3] * kptr[7];

                tmpptr += 4;
                kptr += 8;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;

            outptr4[0] = sum4_0;
            outptr4[1] = sum4_1;
            outptr4[2] = sum4_2;
            outptr4[3] = sum4_3;

            outptr5[0] = sum5_0;
            outptr5[1] = sum5_1;
            outptr5[2] = sum5_2;
            outptr5[3] = sum5_3;

            outptr6[0] = sum6_0;
            outptr6[1] = sum6_1;
            outptr6[2] = sum6_2;
            outptr6[3] = sum6_3;

            outptr7[0] = sum7_0;
            outptr7[1] = sum7_1;
            outptr7[2] = sum7_2;
            outptr7[3] = sum7_3;

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            outptr4 += 4;
            outptr5 += 4;
            outptr6 += 4;
            outptr7 += 4;
#endif // __ARM_NEON && __aarch64__
        }

        for (; i<size; i++)
        {
            const signed char* tmpptr = tmp.channel(i/8 + (i%8)/4 + i%4);
            const signed char* kptr = kernel.channel(p/8);

#if __ARM_NEON && __aarch64__
            asm volatile(
                "eor    v14.16b, v14.16b, v14.16b    \n" // sum0_3
                "eor    v15.16b, v15.16b, v15.16b    \n" // sum4_7
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3
                "eor    v20.16b, v20.16b, v20.16b    \n" // sum4
                "eor    v21.16b, v21.16b, v21.16b    \n" // sum5
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum6
                "eor    v23.16b, v23.16b, v23.16b    \n" // sum7

                // inch loop
                "lsr    w4, %w20, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%9, #128]                       \n" // k
                "ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%9], #32     \n"

                //"prfm   pldl1keep, [%8, #64]                        \n" // d
                "ld1    {v4.8b}, [%8]                               \n"
                "add    %8, %8, #4                                  \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
                "sshll    v1.8h, v1.8b, #0           \n" // k01 - k71
                "sshll    v2.8h, v2.8b, #0           \n" // k02 - k72
                "sshll    v3.8h, v3.8b, #0           \n" // k03 - k73

                "sshll    v4.8h, v4.8b, #0           \n" // a00 - a30

                "subs   w4, w4, #1                   \n"

                //
                "smlal    v16.4s, v0.4h, v4.h[0]     \n"// sum0 += (k00-k70) * a00
                "smlal2   v17.4s, v0.8h, v4.h[0]     \n"// 
                "smlal    v18.4s, v1.4h, v4.h[1]     \n"// sum2 += (k01-k71) * a10
                "smlal2   v19.4s, v1.8h, v4.h[1]     \n"// 
                "smlal    v20.4s, v2.4h, v4.h[2]     \n"// sum4 += (k02-k72) * a20
                "smlal2   v21.4s, v2.8h, v4.h[2]     \n"// 
                "smlal    v22.4s, v3.4h, v4.h[3]     \n"// sum6 += (k03-k73) * a30
                "smlal2   v23.4s, v3.8h, v4.h[3]     \n"// 

                "bne    0b                           \n"

                "add      v16.4s, v16.4s, v18.4s     \n"
                "add      v17.4s, v17.4s, v19.4s     \n"
                "add      v20.4s, v20.4s, v22.4s     \n"
                "add      v21.4s, v21.4s, v23.4s     \n"
                "add      v14.4s, v16.4s, v20.4s     \n"
                "add      v15.4s, v17.4s, v21.4s     \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w20, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%9, #128]        \n"
                "ld1    {v0.8b}, [%9], #8            \n"// k
                "ld1    {v4.8b}, [%8]                \n"// d
                "add    %8, %8, #1                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
                "sshll    v4.8h, v4.8b, #0           \n" // a00 - a70

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v14.4s, v0.4h, v4.h[0]     \n"// sum0_3 += (k00-k30) * a00
                "smlal2   v15.4s, v8.8h, v4.h[0]     \n"// sum4_7 += (k40-k70) * a00

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v14.s}[0], [%0], #4         \n"
                "st1    {v14.s}[1], [%1], #4         \n"
                "st1    {v14.s}[2], [%2], #4         \n"
                "st1    {v14.s}[3], [%3], #4         \n"
                "st1    {v15.s}[0], [%4], #4         \n"
                "st1    {v15.s}[1], [%5], #4         \n"
                "st1    {v15.s}[2], [%6], #4         \n"
                "st1    {v15.s}[3], [%7], #4         \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(outptr4),    // %4
                  "=r"(outptr5),    // %5
                  "=r"(outptr6),    // %6
                  "=r"(outptr7),    // %7
                  "=r"(tmpptr),     // %8
                  "=r"(kptr)        // %9
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(outptr4),
                  "5"(outptr5),
                  "6"(outptr6),
                  "7"(outptr7),
                  "8"(tmpptr),
                  "9"(kptr),
                  "r"(inch)         // %20
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            );
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;            

            for (int q=0; q<inch; q++)
            {
                sum0 += (int)tmpptr[0] * kptr[0];
                sum1 += (int)tmpptr[0] * kptr[1];
                sum2 += (int)tmpptr[0] * kptr[2];
                sum3 += (int)tmpptr[0] * kptr[3];
                sum4 += (int)tmpptr[0] * kptr[4];
                sum5 += (int)tmpptr[0] * kptr[5];
                sum6 += (int)tmpptr[0] * kptr[6];
                sum7 += (int)tmpptr[0] * kptr[7];

                tmpptr++;
                kptr += 8;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;
            outptr4[0] = sum4;
            outptr5[0] = sum5;
            outptr6[0] = sum6;
            outptr7[0] = sum7;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
#endif            
        }
    }
#endif // __ARM_NEON && __aarch64__ 

    nn_outch = (outch - remain_outch_start) >> 2;  

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p+1);
        int* outptr2 = top_blob.channel(p+2);
        int* outptr3 = top_blob.channel(p+3);

        int i = 0;

        for (; i+7<size; i+=8)
        {
            const signed char* tmpptr = tmp.channel(i/8);
#if __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4);
#else
            const signed char* kptr = kernel.channel(p/4);
#endif // __ARM_NEON && __aarch64__
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3
                "eor    v20.16b, v20.16b, v20.16b    \n" // sum4
                "eor    v21.16b, v21.16b, v21.16b    \n" // sum5
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum6
                "eor    v23.16b, v23.16b, v23.16b    \n" // sum7

                // inch loop
                "lsr    w4, %w12, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%5, #128]                       \n" // k
                "ld1    {v0.8b, v1.8b}, [%5], #16    \n"

                //"prfm   pldl1keep, [%4, #128]                       \n" // d
                "ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%4], #32   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k30,k01 - k31
                "sshll    v1.8h, v1.8b, #0           \n" // k02 - k32,k03 - k33

                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a70
                "sshll    v9.8h, v9.8b, #0           \n" // a01 - a71
                "sshll    v10.8h, v10.8b, #0         \n" // a02 - a72
                "sshll    v11.8h, v11.8b, #0         \n" // a03 - a73

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a70) * k00
                "smlal2   v17.4s, v8.8h, v0.h[0]     \n"//
                "smlal    v18.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a70) * k10
                "smlal2   v19.4s, v8.8h, v0.h[1]     \n"//
                "smlal    v20.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a70) * k20
                "smlal2   v21.4s, v8.8h, v0.h[2]     \n"//
                "smlal    v22.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a70) * k30
                "smlal2   v23.4s, v8.8h, v0.h[3]     \n"//
                // k1
                "smlal    v16.4s, v9.4h, v0.h[4]     \n"// sum0 += (a01-a71) * k01
                "smlal2   v17.4s, v9.8h, v0.h[4]     \n"//
                "smlal    v18.4s, v9.4h, v0.h[5]     \n"// sum1 += (a01-a71) * k11
                "smlal2   v19.4s, v9.8h, v0.h[5]     \n"//
                "smlal    v20.4s, v9.4h, v0.h[6]     \n"// sum2 += (a01-a71) * k21
                "smlal2   v21.4s, v9.8h, v0.h[6]     \n"//
                "smlal    v22.4s, v9.4h, v0.h[7]     \n"// sum3 += (a01-a71) * k31
                "smlal2   v23.4s, v9.8h, v0.h[7]     \n"//
                // k2
                "smlal    v16.4s, v10.4h, v1.h[0]    \n"// sum0 += (a02-a72) * k02
                "smlal2   v17.4s, v10.8h, v1.h[0]    \n"//
                "smlal    v18.4s, v10.4h, v1.h[1]    \n"// sum1 += (a02-a72) * k12
                "smlal2   v19.4s, v10.8h, v1.h[1]    \n"//
                "smlal    v20.4s, v10.4h, v1.h[2]    \n"// sum2 += (a02-a72) * k22
                "smlal2   v21.4s, v10.8h, v1.h[2]    \n"//
                "smlal    v22.4s, v10.4h, v1.h[3]    \n"// sum3 += (a02-a72) * k32
                "smlal2   v23.4s, v10.8h, v1.h[3]    \n"//

                "subs   w4, w4, #1                   \n"

                // k3
                "smlal    v16.4s, v11.4h, v1.h[4]    \n"// sum0 += (a03-a73) * k03
                "smlal2   v17.4s, v11.8h, v1.h[4]    \n"//
                "smlal    v18.4s, v11.4h, v1.h[5]    \n"// sum1 += (a03-a73) * k13
                "smlal2   v19.4s, v11.8h, v1.h[5]    \n"//
                "smlal    v20.4s, v11.4h, v1.h[6]    \n"// sum2 += (a03-a73) * k23
                "smlal2   v21.4s, v11.8h, v1.h[6]    \n"//
                "smlal    v22.4s, v11.4h, v1.h[7]    \n"// sum3 += (a03-a73) * k33
                "smlal2   v23.4s, v11.8h, v1.h[7]    \n"//

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v0.8b}, [%5]                \n"
                //"prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v8.8b}, [%4], #8            \n"
                "add    %5, %5, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k30,k01 - k31
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a70

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a70) * k00
                "smlal2   v17.4s, v8.8h, v0.h[0]     \n"//
                "smlal    v18.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a70) * k10
                "smlal2   v19.4s, v8.8h, v0.h[1]     \n"//
                "smlal    v20.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a70) * k20
                "smlal2   v21.4s, v8.8h, v0.h[2]     \n"//
                "smlal    v22.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a70) * k30
                "smlal2   v23.4s, v8.8h, v0.h[3]     \n"//

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v16.4s, v17.4s}, [%0], #32  \n"
                "st1    {v18.4s, v19.4s}, [%1], #32  \n"
                "st1    {v20.4s, v21.4s}, [%2], #32  \n"
                "st1    {v22.4s, v23.4s}, [%3], #32  \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(tmpptr),     // %4
                  "=r"(kptr)        // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)         // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            );
#else
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"
                "vmov.s32    q8, #0            \n"
                "vmov.s32    q9, #0            \n"
                "vmov.s32    q10, #0           \n"
                "vmov.s32    q11, #0           \n"
                "vmov.s32    q12, #0           \n"
                "vmov.s32    q13, #0           \n"

                "lsr         r4, %12, #2       \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4-d7}, [%4]!    \n"// tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n"// a30-a37
                "vmovl.s8    q4, d6            \n"// a20-a27
                "vmovl.s8    q3, d5            \n"// a10-a17
                "vmovl.s8    q2, d4            \n"// a00-a07

                "vld1.s8     {d0-d1}, [%5]!    \n"// kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n"// k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n"// k00-k30,k01-k31

                "vmlal.s16   q6, d4, d0[0]     \n"// sum0 = (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q8, d4, d0[1]     \n"// sum1 = (a00-a07) * k10
                "vmlal.s16   q9, d5, d0[1]     \n"
                "vmlal.s16   q10, d4, d0[2]    \n"// sum2 = (a00-a07) * k20
                "vmlal.s16   q11, d5, d0[2]    \n"
                "vmlal.s16   q12, d4, d0[3]    \n"// sum3 = (a00-a07) * k30
                "vmlal.s16   q13, d5, d0[3]    \n"

                "vmlal.s16   q6, d6, d1[0]     \n"// sum0 += (a10-a17) * k01
                "vmlal.s16   q7, d7, d1[0]     \n"
                "vmlal.s16   q8, d6, d1[1]     \n"// sum1 += (a10-a17) * k11
                "vmlal.s16   q9, d7, d1[1]     \n"
                "vmlal.s16   q10, d6, d1[2]    \n"// sum2 += (a10-a17) * k21
                "vmlal.s16   q11, d7, d1[2]    \n"
                "vmlal.s16   q12, d6, d1[3]    \n"// sum3 += (a10-a17) * k31
                "vmlal.s16   q13, d7, d1[3]    \n"

                "vmlal.s16   q6, d8, d2[0]     \n"// sum0 += (a20-a27) * k02
                "vmlal.s16   q7, d9, d2[0]     \n"
                "vmlal.s16   q8, d8, d2[1]     \n"// sum1 += (a20-a27) * k12
                "vmlal.s16   q9, d9, d2[1]     \n"
                "vmlal.s16   q10, d8, d2[2]    \n"// sum2 += (a20-a27) * k22
                "vmlal.s16   q11, d9, d2[2]    \n"
                "vmlal.s16   q12, d8, d2[3]    \n"// sum3 += (a20-a27) * k32
                "vmlal.s16   q13, d9, d2[3]    \n"  

                "vmlal.s16   q6, d10, d3[0]    \n"// sum0 += (a30-a37) * k03
                "vmlal.s16   q7, d11, d3[0]    \n"
                "vmlal.s16   q8, d10, d3[1]    \n"// sum1 += (a30-a37) * k13
                "vmlal.s16   q9, d11, d3[1]    \n"
                "vmlal.s16   q10, d10, d3[2]   \n"// sum2 += (a30-a37) * k23
                "vmlal.s16   q11, d11, d3[2]   \n"
                "vmlal.s16   q12, d10, d3[3]   \n"// sum3 += (a30-a37) * k33
                "vmlal.s16   q13, d11, d3[3]   \n"                  

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for
 
                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]!       \n"// tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n"// kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q6, d2, d0[0]     \n"// sum0 += (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"
                "vmlal.s16   q8, d2, d0[1]     \n"// sum1 += (a00-a07) * k10
                "vmlal.s16   q9, d3, d0[1]     \n"
                "vmlal.s16   q10, d2, d0[2]    \n"// sum2 += (a00-a07) * k20
                "vmlal.s16   q11, d3, d0[2]    \n"
                "vmlal.s16   q12, d2, d0[3]    \n"// sum3 += (a00-a07) * k30
                "vmlal.s16   q13, d3, d0[3]    \n"    

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d12-d15}, [%0]!  \n"
                "vst1.s32    {d16-d19}, [%1]!  \n"
                "vst1.s32    {d20-d23}, [%2]!  \n"
                "vst1.s32    {d24-d27}, [%3]!  \n"

                : "=r"(outptr0), // %0
                  "=r"(outptr1), // %1
                  "=r"(outptr2), // %2
                  "=r"(outptr3), // %3
                  "=r"(tmpptr),  // %4
                  "=r"(kptr)     // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)      // %12  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif // __aarch64__            
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;
            int sum0_4 = 0;
            int sum0_5 = 0;
            int sum0_6 = 0;
            int sum0_7 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;
            int sum1_4 = 0;
            int sum1_5 = 0;
            int sum1_6 = 0;
            int sum1_7 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;
            int sum2_4 = 0;
            int sum2_5 = 0;
            int sum2_6 = 0;
            int sum2_7 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;
            int sum3_4 = 0;
            int sum3_5 = 0;
            int sum3_6 = 0;
            int sum3_7 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];
                sum0_4 += tmpptr[4] * kptr[0];
                sum0_5 += tmpptr[5] * kptr[0];
                sum0_6 += tmpptr[6] * kptr[0];
                sum0_7 += tmpptr[7] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];
                sum1_4 += tmpptr[4] * kptr[1];
                sum1_5 += tmpptr[5] * kptr[1];
                sum1_6 += tmpptr[6] * kptr[1];
                sum1_7 += tmpptr[7] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];
                sum2_4 += tmpptr[4] * kptr[2];
                sum2_5 += tmpptr[5] * kptr[2];
                sum2_6 += tmpptr[6] * kptr[2];
                sum2_7 += tmpptr[7] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];
                sum3_4 += tmpptr[4] * kptr[3];
                sum3_5 += tmpptr[5] * kptr[3];
                sum3_6 += tmpptr[6] * kptr[3];
                sum3_7 += tmpptr[7] * kptr[3];

                tmpptr += 8;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;
            outptr0[4] = sum0_4;
            outptr0[5] = sum0_5;
            outptr0[6] = sum0_6;
            outptr0[7] = sum0_7;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;
            outptr1[4] = sum1_4;
            outptr1[5] = sum1_5;
            outptr1[6] = sum1_6;
            outptr1[7] = sum1_7;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;
            outptr2[4] = sum2_4;
            outptr2[5] = sum2_5;
            outptr2[6] = sum2_6;
            outptr2[7] = sum2_7;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
            outptr3[4] = sum3_4;
            outptr3[5] = sum3_5;
            outptr3[6] = sum3_6;
            outptr3[7] = sum3_7;

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
#endif // __ARM_NEON            
        }    

        for (; i+3<size; i+=4)
        {
            const signed char* tmpptr = tmp.channel(i/8 + (i%8)/4);
#if __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4);
#else
            const signed char* kptr = kernel.channel(p/4);
#endif // __ARM_NEON && __aarch64__
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3

                // inch loop
                "lsr    w4, %w12, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%5, #128]      \n" // k
                "ld1    {v0.8b, v1.8b}, [%5], #16    \n"

                //"prfm   pldl1keep, [%4, #128]      \n" // d
                "ld1    {v8.8b, v9.8b}, [%4], #16    \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k30,k01 - k31
                "sshll    v1.8h, v1.8b, #0           \n" // k02 - k32,k03 - k33

                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a30,a01 - a31
                "sshll    v9.8h, v9.8b, #0           \n" // a02 - a32,a03 - a33

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a30) * k00
                "smlal    v17.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a30) * k10
                "smlal    v18.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a30) * k20
                "smlal    v19.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a30) * k30
                // k1
                "smlal2    v16.4s, v8.8h, v0.h[4]    \n"// sum0 += (a01-a31) * k01
                "smlal2    v17.4s, v8.8h, v0.h[5]    \n"// sum1 += (a01-a31) * k11
                "smlal2    v18.4s, v8.8h, v0.h[6]    \n"// sum2 += (a01-a31) * k21
                "smlal2    v19.4s, v8.8h, v0.h[7]    \n"// sum3 += (a01-a31) * k31
                // k2
                "smlal    v16.4s, v9.4h, v1.h[0]     \n"// sum0 += (a02-a32) * k02
                "smlal    v17.4s, v9.4h, v1.h[1]     \n"// sum1 += (a02-a32) * k12
                "smlal    v18.4s, v9.4h, v1.h[2]     \n"// sum2 += (a02-a32) * k22
                "smlal    v19.4s, v9.4h, v1.h[3]     \n"// sum3 += (a02-a32) * k32

                "subs   w4, w4, #1                   \n"

                // k3
                "smlal2    v16.4s, v9.8h, v1.h[4]    \n"// sum0 += (a03-a33) * k03
                "smlal2    v17.4s, v9.8h, v1.h[5]    \n"// sum1 += (a03-a33) * k13
                "smlal2    v18.4s, v9.8h, v1.h[6]    \n"// sum2 += (a03-a33) * k23
                "smlal2    v19.4s, v9.8h, v1.h[7]    \n"// sum3 += (a03-a33) * k33

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%5, #128]      \n"
                "ld1    {v0.8b}, [%5]                \n"
                //"prfm   pldl1keep, [%4, #128]      \n"
                "ld1    {v8.8b}, [%4]                \n"
                "add    %4, %4, #4                   \n"
                "add    %5, %5, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k30
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a30

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a30) * k00
                "smlal    v17.4s, v8.4h, v0.h[1]     \n"// sum1 += (a00-a30) * k10
                "smlal    v18.4s, v8.4h, v0.h[2]     \n"// sum2 += (a00-a30) * k20
                "smlal    v19.4s, v8.4h, v0.h[3]     \n"// sum3 += (a00-a30) * k30

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v16.4s}, [%0], #16          \n"
                "st1    {v17.4s}, [%1], #16          \n"
                "st1    {v18.4s}, [%2], #16          \n"
                "st1    {v19.4s}, [%3], #16          \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(tmpptr),     // %4
                  "=r"(kptr)        // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)         // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
            );
#else
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"
                "vmov.s32    q8, #0            \n"
                "vmov.s32    q9, #0            \n"

                "lsr         r4, %12, #2       \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4-d5}, [%4]!    \n"// tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
                "vmovl.s8    q3, d5            \n"// a20-a23,a30-a33
                "vmovl.s8    q2, d4            \n"// a00-a04,a10-a14

                "vld1.s8     {d0-d1}, [%5]!    \n"// kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n"// k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n"// k00-k30,k01-k31

                "vmlal.s16   q6, d4, d0[0]     \n"// sum0 = (a00-a03) * k00
                "vmlal.s16   q7, d4, d0[1]     \n"// sum1 = (a00-a03) * k10
                "vmlal.s16   q8, d4, d0[2]     \n"// sum2 = (a00-a03) * k20
                "vmlal.s16   q9, d4, d0[3]     \n"// sum3 = (a00-a03) * k30

                "vmlal.s16   q6, d5, d1[0]     \n"// sum0 += (a10-a13) * k01
                "vmlal.s16   q7, d5, d1[1]     \n"// sum1 += (a10-a13) * k11
                "vmlal.s16   q8, d5, d1[2]     \n"// sum2 += (a10-a13) * k21
                "vmlal.s16   q9, d5, d1[3]     \n"// sum3 += (a10-a13) * k31

                "vmlal.s16   q6, d6, d2[0]     \n"// sum0 += (a20-a23) * k02
                "vmlal.s16   q7, d6, d2[1]     \n"// sum1 += (a20-a23) * k12
                "vmlal.s16   q8, d6, d2[2]     \n"// sum2 += (a20-a23) * k22
                "vmlal.s16   q9, d6, d2[3]     \n"// sum3 += (a20-a23) * k32

                "vmlal.s16   q6, d7, d3[0]     \n"// sum0 += (a30-a33) * k03
                "vmlal.s16   q7, d7, d3[1]     \n"// sum1 += (a30-a33) * k13
                "vmlal.s16   q8, d7, d3[2]     \n"// sum2 += (a30-a33) * k23
                "vmlal.s16   q9, d7, d3[3]     \n"// sum3 += (a30-a33) * k33

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for
 
                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]        \n"// tmpr a00-a03    a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n"// kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %4, #4            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q6, d2, d0[0]     \n"// sum0 += (a00-a03) * k00
                "vmlal.s16   q7, d2, d0[1]     \n"// sum1 += (a00-a03) * k10
                "vmlal.s16   q8, d2, d0[2]     \n"// sum2 += (a00-a03) * k20
                "vmlal.s16   q9, d2, d0[3]     \n"// sum3 += (a00-a03) * k30

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d12-d13}, [%0]!  \n"
                "vst1.s32    {d14-d15}, [%1]!  \n"
                "vst1.s32    {d16-d17}, [%2]!  \n"
                "vst1.s32    {d18-d19}, [%3]!  \n"

                : "=r"(outptr0), // %0
                  "=r"(outptr1), // %1
                  "=r"(outptr2), // %2
                  "=r"(outptr3), // %3
                  "=r"(tmpptr),  // %4
                  "=r"(kptr)     // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)      // %12  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif // __aarch64__            
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
#endif // __ARM_NEON            
        }

        for (; i<size; i++)
        {
            const signed char* tmpptr = tmp.channel(i/8 + (i%8)/4 + i%4);
#if __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4);
#else
            const signed char* kptr = kernel.channel(p/4);
#endif // __ARM_NEON && __aarch64__
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v14.16b, v14.16b, v14.16b    \n" // sum0_3
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3

                // inch loop
                "lsr    w4, %w12, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%5, #128]      \n" // k
                "ld1    {v0.8b, v1.8b}, [%5], #16    \n"

                //"prfm   pldl1keep, [%4, #64]       \n" // d
                "ld1    {v4.8b}, [%4]                \n"
                "add    %4, %4, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k30,k01 - k31
                "sshll    v1.8h, v1.8b, #0           \n" // k02 - k32,k03 - k33

                "sshll    v4.8h, v4.8b, #0           \n" // a00 - a30

                "subs   w4, w4, #1                   \n"

                //
                "smlal    v16.4s, v0.4h, v4.h[0]     \n"// sum0 += (k00-k30) * a00
                "smlal2   v17.4s, v0.8h, v4.h[1]     \n"// sum1 += (k01-k31) * a10
                "smlal    v18.4s, v1.4h, v4.h[2]     \n"// sum2 += (k02-k32) * a20
                "smlal2   v19.4s, v1.8h, v4.h[3]     \n"// sum3 += (k03-k33) * a30

                "bne    0b                           \n"

                "add      v16.4s, v16.4s, v18.4s     \n"
                "add      v17.4s, v17.4s, v19.4s     \n"
                "add      v14.4s, v16.4s, v17.4s     \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%5, #128]      \n"
                "ld1    {v0.8b}, [%5]                \n"// k
                "ld1    {v4.8b}, [%4]                \n"// d
                "add    %4, %4, #1                   \n"
                "add    %5, %5, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k30
                "sshll    v4.8h, v4.8b, #0           \n" // a00 - a30

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v14.4s, v0.4h, v4.h[0]     \n"// sum0_3 += (k00-k30) * a00

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v14.s}[0], [%0], #4         \n"
                "st1    {v14.s}[1], [%1], #4         \n"
                "st1    {v14.s}[2], [%2], #4         \n"
                "st1    {v14.s}[3], [%3], #4         \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(tmpptr),     // %4
                  "=r"(kptr)        // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)         // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
            );
#else
            asm volatile(
                // inch loop
                "veor        q6, q6, q6        \n"
                "veor        q7, q7, q7        \n"
                "veor        q8, q8, q8        \n"
                "veor        q9, q9, q9        \n"
                "vmov.s32    q10, #0           \n"

                "lsr         r4, %12, #2       \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4}, [%4]        \n"// tmpr a00,a10,a20,a30    a(inch)(data)
                "add         %4, #4            \n"
                "vmovl.s8    q2, d4            \n"// a00,a10,a20,a30

                "vld1.s8     {d0-d1}, [%5]!    \n"// kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n"// k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n"// k00-k30,k01-k31

                "vmlal.s16   q6, d0, d4[0]     \n"// (k00-k30) * a00
                "vmlal.s16   q7, d1, d4[1]     \n"// (k01-k31) * a10
                "vmlal.s16   q8, d2, d4[2]     \n"// (k02-k32) * a20
                "vmlal.s16   q9, d3, d4[3]     \n"// (k03-k33) * a30

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for

                "vadd.s32    q6, q6, q7        \n"
                "vadd.s32    q9, q9, q8        \n"
                "vadd.s32    q10, q6, q9       \n"
 
                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]        \n"// tmpr a00        a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n"// kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %4, #1            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q10, d0, d2[0]    \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d20[0]}, [%0]!   \n"
                "vst1.s32    {d20[1]}, [%1]!   \n"
                "vst1.s32    {d21[0]}, [%2]!   \n"
                "vst1.s32    {d21[1]}, [%3]!   \n"

                : "=r"(outptr0), // %0
                  "=r"(outptr1), // %1
                  "=r"(outptr2), // %2
                  "=r"(outptr3), // %3
                  "=r"(tmpptr),  // %4
                  "=r"(kptr)     // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)      // %12  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif // __aarch64__            
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];

                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;  
#endif // __ARM_NEON
        }
    }

    remain_outch_start += nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        int* outptr0 = out0;

        int i = 0;

        for (; i+7<size; i+=8)
        {
            const signed char* tmpptr = tmp.channel(i/8);
#if __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4 + p%4);
#else
            const signed char* kptr = kernel.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum0n

                // inch loop
                "lsr    w4, %w6, #2                  \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%2, #128]      \n" // k
                "ld1    {v0.8b}, [%2]                \n"

                //"prfm   pldl1keep, [%1, #128]                     \n" // d
                "ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%1], #32   \n"
                "add    %2, %2, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k03

                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a07
                "sshll    v9.8h, v9.8b, #0           \n" // a10 - a17
                "sshll    v10.8h, v10.8b, #0         \n" // a20 - a27
                "sshll    v11.8h, v11.8b, #0         \n" // a30 - a37

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a07) * k00
                "smlal2   v17.4s, v8.8h, v0.h[0]     \n"//
                // k1
                "smlal    v16.4s, v9.4h, v0.h[1]     \n"// sum0 += (a10-a17) * k01
                "smlal2   v17.4s, v9.8h, v0.h[1]     \n"//
                // k2
                "smlal    v16.4s, v10.4h, v0.h[2]    \n"// sum0 += (a20-a27) * k02
                "smlal2   v17.4s, v10.8h, v0.h[2]    \n"//

                "subs   w4, w4, #1                   \n"

                // k3
                "smlal    v16.4s, v11.4h, v0.h[3]    \n"// sum0 += (a30-a37) * k03
                "smlal2   v17.4s, v11.8h, v0.h[3]    \n"//

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w6, #3                  \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v0.8b}, [%2]                \n"
                //"prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v8.8b}, [%1], #8            \n"
                "add    %2, %2, #1                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a07

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a07) * k00
                "smlal2   v17.4s, v8.8h, v0.h[0]     \n"//

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v16.4s, v17.4s}, [%0], #32  \n"

                : "=r"(outptr0),    // %0
                  "=r"(tmpptr),     // %1
                  "=r"(kptr)        // %2
                : "0"(outptr0),
                  "1"(tmpptr),
                  "2"(kptr),
                  "r"(inch)         // %6
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"
            );
#else
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"

                "lsr         r4, %6, #2        \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%1, #128]        \n"
                "vld1.s8     {d4-d7}, [%1]!    \n"// tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n"// a30-a37
                "vmovl.s8    q4, d6            \n"// a20-a27
                "vmovl.s8    q3, d5            \n"// a10-a17
                "vmovl.s8    q2, d4            \n"// a00-a07

                "vld1.s8     {d0}, [%2]        \n"// kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n"// k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n"// (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q6, d6, d0[1]     \n"// (a10-a17) * k01
                "vmlal.s16   q7, d7, d0[1]     \n"
                "vmlal.s16   q6, d8, d0[2]     \n"// (a20-a27) * k02
                "vmlal.s16   q7, d9, d0[2]     \n"
                "vmlal.s16   q6, d10, d0[3]    \n"// (a30-a37) * k03
                "vmlal.s16   q7, d11, d0[3]    \n"

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for
 
                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]!       \n"// tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n"// kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n"// (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"  

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d12-d15}, [%0]!  \n"

                : "=r"(outptr0), // %0
                  "=r"(tmpptr),  // %1
                  "=r"(kptr)     // %2
                : "0"(outptr0),
                  "1"(tmpptr),
                  "2"(kptr),
                  "r"(inch)      // %6  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
#endif // __aarch64__            
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];
                sum4 += tmpptr[4] * kptr[0];
                sum5 += tmpptr[5] * kptr[0];
                sum6 += tmpptr[6] * kptr[0];
                sum7 += tmpptr[7] * kptr[0];

                tmpptr += 8;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0[4] = sum4;
            outptr0[5] = sum5;
            outptr0[6] = sum6;
            outptr0[7] = sum7;

            outptr0 += 8;
#endif // __ARM_NEON
        }   

        for (; i+3<size; i+=4)
        {
            const signed char* tmpptr = tmp.channel(i/8 + (i%8)/4);   
#if __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4 + p%4);
#else
            const signed char* kptr = kernel.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__
#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0

                // inch loop
                "lsr    w4, %w6, #2                  \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%2, #128]      \n" // k
                "ld1    {v0.8b}, [%2]                \n"

                //"prfm   pldl1keep, [%1, #128]      \n" // d
                "ld1    {v8.8b, v9.8b}, [%1], #16    \n"
                "add    %2, %2, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00 - k03
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a03,a10 - a13
                "sshll    v9.8h, v9.8b, #0           \n" // a20 - a23,a30 - a33

                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a03) * k00
                "smlal    v16.4s, v9.4h, v0.h[1]     \n"// sum0 += (a10-a13) * k01
                "smlal    v16.4s, v10.4h, v0.h[2]    \n"// sum0 += (a20-a23) * k02

                "subs   w4, w4, #1                   \n"

                // k3
                "smlal    v16.4s, v11.4h, v0.h[3]    \n"// sum0 += (a30-a33) * k03

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w6, #3                  \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%2, #128]      \n"
                "ld1    {v0.8b}, [%2]                \n"
                //"prfm   pldl1keep, [%1, #128]      \n"
                "ld1    {v8.8b}, [%1]                \n"
                "add    %2, %2, #1                   \n"
                "add    %1, %1, #4                   \n"

                "sshll    v0.8h, v0.8b, #0           \n" // k00
                "sshll    v8.8h, v8.8b, #0           \n" // a00 - a03

                "subs   w4, w4, #1                   \n"

                // k0
                "smlal    v16.4s, v8.4h, v0.h[0]     \n"// sum0 += (a00-a03) * k00

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v16.4s}, [%0], #16          \n"

                : "=r"(outptr0),    // %0
                  "=r"(tmpptr),     // %1
                  "=r"(kptr)        // %2
                : "0"(outptr0),
                  "1"(tmpptr),
                  "2"(kptr),
                  "r"(inch)         // %6
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"
            );
#else
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"

                "lsr         r4, %6, #2        \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%2, #128]        \n"
                "vld1.s8     {d4-d5}, [%1]!    \n"// tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
                "vmovl.s8    q3, d5            \n"// a20-a23,a30-a33
                "vmovl.s8    q2, d4            \n"// a00-a03,a10-a13

                "vld1.s8     {d0}, [%2]        \n"// kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n"// k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n"// (a00-a03) * k00
                "vmlal.s16   q6, d5, d0[1]     \n"// (a10-a13) * k01
                "vmlal.s16   q6, d6, d0[2]     \n"// (a20-a23) * k02
                "vmlal.s16   q6, d7, d0[3]     \n"// (a30-a33) * k03

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for
 
                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]        \n"// tmpr a00-a03    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n"// kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %1, #4            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n"// (a00-a03) * k00

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d12-d13}, [%0]!  \n"

                : "=r"(outptr0), // %0
                  "=r"(tmpptr),  // %1
                  "=r"(kptr)     // %2
                : "0"(outptr0),
                  "1"(tmpptr),
                  "2"(kptr),
                  "r"(inch)      // %6  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6"
            );
#endif // __aarch64__            
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];

                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
#endif // __ARM_NEON
        }

        for (; i<size; i++)
        {
            const signed char* tmpptr = tmp.channel(i/8 + (i%8)/4 + i%4);   
#if __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4 + p%4);
#else
            const signed char* kptr = kernel.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__

            int q = 0;            
            int sum0 = 0;

            for (; q<inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }   

//     // NOTE sgemm int8
//     for (; p<outch; p++)
//     {
//         Mat out0 = top_blob.channel(p);
//
//         int* outptr0 = out0;
//
//         for (int i=0; i<size; i++)
//         {
//             int sum = 0;
//
//             const signed char* kptr = _kernel.channel(p/8 + p%8);
//
//             for (int q=0; q<inch; q++)
//             {
//                 const signed char* img0 = bottom_blob.channel(q);
//
//                 sum += img0[i] * kptr[0];
//                 kptr ++;
//             }
//
//             outptr0[i] = sum;
//         }
//     }
}

static void conv1x1s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}

static void conv1x1s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}
