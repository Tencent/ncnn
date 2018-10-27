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
/*
 * Convolution 1x1 quantized with int8,unroll 16 x 8
 */
static void conv1x1s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);
        Mat out4 = top_blob.channel(p+4);
        Mat out5 = top_blob.channel(p+5);
        Mat out6 = top_blob.channel(p+6);
        Mat out7 = top_blob.channel(p+7);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);
        out4.fill(0);
        out5.fill(0);
        out6.fill(0);
        out7.fill(0);

        int q = 0;

#ifdef __clang__
        for (; q+15<inch; q+=16)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;
            const signed char* kernel4 = (const signed char*)kernel + (p+4)*inch + q;
            const signed char* kernel5 = (const signed char*)kernel + (p+5)*inch + q;
            const signed char* kernel6 = (const signed char*)kernel + (p+6)*inch + q;
            const signed char* kernel7 = (const signed char*)kernel + (p+7)*inch + q;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);
            const signed char* r8 = bottom_blob.channel(q+8);
            const signed char* r9 = bottom_blob.channel(q+9);
            const signed char* r10 = bottom_blob.channel(q+10);
            const signed char* r11 = bottom_blob.channel(q+11);
            const signed char* r12 = bottom_blob.channel(q+12);
            const signed char* r13 = bottom_blob.channel(q+13);
            const signed char* r14 = bottom_blob.channel(q+14);
            const signed char* r15 = bottom_blob.channel(q+15);

            int size = outw * outh;

            int nn = size >> 4;
            int remain = size & 15;

            int8x16_t _k0 = vld1q_s8(kernel0);
            int8x16_t _k1 = vld1q_s8(kernel1);
            int8x16_t _k2 = vld1q_s8(kernel2);
            int8x16_t _k3 = vld1q_s8(kernel3);
            int8x16_t _k4 = vld1q_s8(kernel4);
            int8x16_t _k5 = vld1q_s8(kernel5);
            int8x16_t _k6 = vld1q_s8(kernel6);
            int8x16_t _k7 = vld1q_s8(kernel7);

            if (nn > 0)
            {
            asm volatile(
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"
                "ld1    {v8.16b}, [%9], #16          \n" // r0"
                "ld1    {v9.16b}, [%10], #16         \n" // r1"
                "ld1    {v10.16b}, [%11], #16        \n" // r2"
                "ld1    {v11.16b}, [%12], #16        \n" // r3"

                "dup    v24.16b, %50.b[0]            \n" // k00
                "dup    v25.16b, %50.b[1]            \n" // k01
                "dup    v26.16b, %50.b[2]            \n" // k02
                "dup    v27.16b, %50.b[3]            \n" // k03

                "0:                                  \n"
                "smull  v28.8h, v8.8b, v24.8b        \n" // r0 * k0
                "smull2  v31.8h, v8.16b, v24.16b     \n" // r0n * k0
                "prfm   pldl1keep, [%13, #128]       \n"
                "prfm   pldl1keep, [%14, #128]       \n"
                "prfm   pldl1keep, [%15, #128]       \n"

                "smlal  v28.8h, v9.8b, v25.8b        \n" // r0 * k1
                "smlal2  v31.8h, v9.16b, v25.16b     \n" // r0n * k1
                "prfm   pldl1keep, [%16, #128]       \n"
                "ld1    {v12.16b}, [%13], #16        \n" // r4"   
                "ld1    {v13.16b}, [%14], #16        \n" // r5"

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "ld1    {v14.16b}, [%15], #16        \n" // r6"
                "ld1    {v15.16b}, [%16], #16        \n" // r7"                               
                "dup    v24.16b, %50.b[4]            \n" // k04

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"
                "dup    v25.16b, %50.b[5]            \n" // k05
                "dup    v26.16b, %50.b[6]            \n" // k06
                "dup    v27.16b, %50.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n" // r4
                "smlal2  v31.8h, v12.16b, v24.16b    \n" // r4
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%1]       \n" // sum0  
                "prfm   pldl1keep, [%17, #128]       \n"

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "ld1    {v16.16b}, [%17], #16        \n" // r8"   

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "ld1    {v17.16b}, [%18], #16        \n" // r9"
                "ld1    {v18.16b}, [%19], #16        \n" // r10"
                "ld1    {v19.16b}, [%20], #16        \n" // r11"             

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v24.16b, %50.b[8]            \n" // k08
                "dup    v25.16b, %50.b[9]            \n" // k09
                "dup    v26.16b, %50.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n" // r8
                "smlal2  v31.8h, v16.16b, v24.16b    \n" // r8
                "dup    v27.16b, %50.b[11]           \n" // k11
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "ld1    {v20.16b}, [%21], #16        \n" // r12"

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "ld1    {v21.16b}, [%22], #16        \n" // r13"
                "ld1    {v22.16b}, [%23], #16        \n" // r14"
                "ld1    {v23.16b}, [%24], #16        \n" // r15"              

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v24.16b, %50.b[12]           \n" // k12
                "dup    v25.16b, %50.b[13]           \n" // k13
                "dup    v26.16b, %50.b[14]           \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n" // r12
                "smlal2  v31.8h, v20.16b, v24.16b    \n" // r12
                "dup    v27.16b, %50.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %51.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %51.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
                "dup    v26.16b, %51.b[2]            \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.16b, %51.b[3]            \n" // k03

                "st1    {v29.4s, v30.4s}, [%1], #32  \n" // sum0

                "ld1    {v29.4s, v30.4s}, [%1]       \n" // sum0
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n" 
                "st1    {v29.4s, v30.4s}, [%1], #32  \n" // sum0             
                //########################################### 
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %51.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %51.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %51.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"    
                "smlal2  v31.8h, v11.16b, v27.16b    \n"         
                "dup    v27.16b, %51.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%2]       \n" // sum1

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %51.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %51.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %51.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %51.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %51.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %51.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %51.b[14]           \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %51.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %52.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %52.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"        
                "smlal2  v31.8h, v23.16b, v27.16b    \n"     
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v26.16b, %52.b[2]            \n" // k02
                "dup    v27.16b, %52.b[3]            \n" // k03  

                "st1    {v29.4s, v30.4s}, [%2], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%2]       \n" // sum1
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"   
                "st1    {v29.4s, v30.4s}, [%2], #32  \n"             
                //########################################### // sum1

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %52.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %52.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %52.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %52.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%3]       \n" // sum2 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %52.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %52.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %52.b[10]           \n" // k10

                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %52.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %52.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %52.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %52.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %52.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %53.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %53.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v26.16b, %53.b[2]            \n" // k02

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                "dup    v27.16b, %53.b[3]            \n" // k03

                "st1    {v29.4s, v30.4s}, [%3], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%3]       \n" // sum2 
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"
                "st1    {v29.4s, v30.4s}, [%3], #32  \n"
                //########################################### //sum 2

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %53.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %53.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %53.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %53.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%4]       \n" // sum3 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %53.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %53.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %53.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %53.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %53.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %53.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %53.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %53.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %54.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %54.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"    
                "smlal2  v31.8h, v23.16b, v27.16b    \n"         
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v26.16b, %54.b[2]            \n" // k02

                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.16b, %54.b[3]            \n" // k03

                "st1    {v29.4s, v30.4s}, [%4], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%4]       \n" // sum3 
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"
                "st1    {v29.4s, v30.4s}, [%4], #32  \n"
                //########################################### // sum3
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %54.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %54.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %54.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"    
                "smlal2  v31.8h, v11.16b, v27.16b    \n"         
                "dup    v27.16b, %54.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%5]       \n" // sum4

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %54.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %54.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"    
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %54.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %54.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %54.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %54.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %54.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %54.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %55.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %55.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"
                "dup    v26.16b, %55.b[2]            \n" // k02    
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.16b, %55.b[3]            \n" // k03

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                
                "st1    {v29.4s, v30.4s}, [%5], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%5]       \n" // sum4
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"
                "st1    {v29.4s, v30.4s}, [%5], #32  \n"
                //########################################### // sum4
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %55.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %55.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %55.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %55.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%6]       \n" // sum5  

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %55.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %55.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %55.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %55.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %55.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %55.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %55.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %55.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %56.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %56.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v26.16b, %56.b[2]            \n" // k02

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                "dup    v27.16b, %56.b[3]            \n" // k03

                "st1    {v29.4s, v30.4s}, [%6], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%6]       \n" // sum5 
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"
                "st1    {v29.4s, v30.4s}, [%6], #32  \n"
                //########################################### // sum5
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %56.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %56.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %56.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"    
                "smlal2  v31.8h, v11.16b, v27.16b    \n"         
                "dup    v27.16b, %56.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%7]       \n" // sum6 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %56.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %56.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"     
                "dup    v26.16b, %56.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %56.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %56.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %56.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %56.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %56.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %57.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %57.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
                "dup    v26.16b, %57.b[2]            \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.16b, %57.b[3]            \n" // k03

                "st1    {v29.4s, v30.4s}, [%7], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%7]       \n" // sum6 
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"     
                "st1    {v29.4s, v30.4s}, [%7], #32  \n"                           
                //########################################### // sum6
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %57.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %57.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %57.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %57.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%8]       \n" // sum7 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %57.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %57.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %57.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %57.b[11]           \n" // k11
                
                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %57.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %57.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %57.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %57.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "ld1    {v8.16b}, [%9], #16          \n" // r0"

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "ld1    {v9.16b}, [%10], #16         \n" // r1"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"

                "smlal  v28.8h, v23.8b, v27.8b       \n"    
                "smlal2  v31.8h, v23.16b, v27.16b    \n"         
                "ld1    {v10.16b}, [%11], #16        \n" // r2"
                "ld1    {v11.16b}, [%12], #16        \n" // r3"
                "dup    v24.16b, %50.b[0]            \n" // k00                    

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v25.16b, %50.b[1]            \n" // k01

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                "dup    v26.16b, %50.b[2]            \n" // k02
                "dup    v27.16b, %50.b[3]            \n" // k03                

                "st1    {v29.4s, v30.4s}, [%8], #32  \n"

                "ld1    {v29.4s, v30.4s}, [%8]       \n" // sum7 
                "saddw  v29.4s, v29.4s, v31.4h       \n"
                "saddw2 v30.4s, v30.4s, v31.8h       \n"
                "st1    {v29.4s, v30.4s}, [%8], #32  \n"
                //########################################### // sum7
                "subs   %w0, %w0, #1                 \n"
                "bne    0b                           \n"
                "sub    %9, %9, #16                  \n"
                "sub    %10, %10, #16                \n"
                "sub    %11, %11, #16                \n"
                "sub    %12, %12, #16                \n"
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7),// %8
                  "=r"(r0),     // %9
                  "=r"(r1),     // %10
                  "=r"(r2),     // %11
                  "=r"(r3),     // %12
                  "=r"(r4),     // %13
                  "=r"(r5),     // %14
                  "=r"(r6),     // %15
                  "=r"(r7),     // %16
                  "=r"(r8),     // %17
                  "=r"(r9),     // %18
                  "=r"(r10),    // %19
                  "=r"(r11),    // %20
                  "=r"(r12),    // %21
                  "=r"(r13),    // %22
                  "=r"(r14),    // %23
                  "=r"(r15)     // %24
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
                  "12"(r3),
                  "13"(r4),
                  "14"(r5),
                  "15"(r6),
                  "16"(r7),
                  "17"(r8),
                  "18"(r9),
                  "19"(r10),
                  "20"(r11),
                  "21"(r12),
                  "22"(r13),
                  "23"(r14),
                  "24"(r15),
                  "w"(_k0),     // %50
                  "w"(_k1),     // %51
                  "w"(_k2),     // %52
                  "w"(_k3),     // %53
                  "w"(_k4),     // %54
                  "w"(_k5),     // %55
                  "w"(_k6),     // %56
                  "w"(_k7)      // %57
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );                     
            }

            if (remain >= 8)
            {
                remain -= 8;

            asm volatile(
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"
                "ld1    {v8.8b}, [%9], #8            \n" // r0"
                "ld1    {v9.8b}, [%10], #8           \n" // r1"
                "ld1    {v10.8b}, [%11], #8          \n" // r2"
                "ld1    {v11.8b}, [%12], #8          \n" // r3"

                "dup    v24.8b, %50.b[0]             \n" // k00
                "dup    v25.8b, %50.b[1]             \n" // k01
                "dup    v26.8b, %50.b[2]             \n" // k02
                "dup    v27.8b, %50.b[3]             \n" // k03

                "smull  v28.8h, v8.8b, v24.8b        \n" // r0
                "prfm   pldl1keep, [%13, #128]       \n"
                "prfm   pldl1keep, [%14, #128]       \n"
                "prfm   pldl1keep, [%15, #128]       \n"

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "prfm   pldl1keep, [%16, #128]       \n"
                "ld1    {v12.8b}, [%13], #8          \n" // r4" 
                "ld1    {v13.8b}, [%14], #8          \n" // r5"

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "ld1    {v14.8b}, [%15], #8          \n" // r6"
                "ld1    {v15.8b}, [%16], #8          \n" // r7"                         
                "dup    v24.8b, %50.b[4]             \n" // k04

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "dup    v25.8b, %50.b[5]             \n" // k05
                "dup    v26.8b, %50.b[6]             \n" // k06
                "dup    v27.8b, %50.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n" // r4
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%1]       \n" // sum0  
                "prfm   pldl1keep, [%17, #128]       \n"

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "ld1    {v16.8b}, [%17], #8          \n" // r8" 

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "ld1    {v17.8b}, [%18], #8          \n" // r9"
                "ld1    {v18.8b}, [%19], #8          \n" // r10"
                "ld1    {v19.8b}, [%20], #8          \n" // r11"

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[8]             \n" // k08
                "dup    v25.8b, %50.b[9]             \n" // k09
                "dup    v26.8b, %50.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n" // r8
                "dup    v27.8b, %50.b[11]            \n" // k11
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "ld1    {v20.8b}, [%21], #8          \n" // r12"

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "ld1    {v21.8b}, [%22], #8          \n" // r13"
                "ld1    {v22.8b}, [%23], #8          \n" // r14"
                "ld1    {v23.8b}, [%24], #8          \n" // r15" 

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[12]            \n" // k12
                "dup    v25.8b, %50.b[13]            \n" // k13
                "dup    v26.8b, %50.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n" // r12
                "dup    v27.8b, %50.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                 "dup    v24.8b, %51.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %51.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.8b, %51.b[3]             \n" // k03

                "st1    {v29.4s, v30.4s}, [%1], #32  \n" // sum0
                //########################################### 
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %51.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %51.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %51.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %51.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%2]       \n" // sum1

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v26.8b, %52.b[2]             \n" // k02
                "dup    v27.8b, %52.b[3]             \n" // k03  

                "st1    {v29.4s, v30.4s}, [%2], #32  \n"
                //########################################### // sum1

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %52.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %52.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %52.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %52.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%3]       \n" // sum2 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[10]            \n" // k10

                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v26.8b, %53.b[2]             \n" // k02

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                "dup    v27.8b, %53.b[3]             \n" // k03

                "st1    {v29.4s, v30.4s}, [%3], #32  \n"
                //########################################### //sum 2

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %53.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %53.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %53.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %53.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%4]       \n" // sum3 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v26.8b, %54.b[2]             \n" // k02

                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.8b, %54.b[3]             \n" // k03

                "st1    {v29.4s, v30.4s}, [%4], #32  \n"
                //########################################### // sum3
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %54.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %54.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %54.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %54.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%5]       \n" // sum4

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"    
                "dup    v26.8b, %54.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %54.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[2]             \n" // k02 
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %55.b[3]             \n" // k03

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                
                "st1    {v29.4s, v30.4s}, [%5], #32  \n"
                //########################################### // sum4
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %55.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %55.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %55.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %55.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%6]       \n" // sum5  

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v26.8b, %56.b[2]             \n" // k02

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                "dup    v27.8b, %56.b[3]             \n" // k03

                "st1    {v29.4s, v30.4s}, [%6], #32  \n"
                //########################################### // sum5
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %56.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %56.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %56.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %56.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%7]       \n" // sum6 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"        
                "dup    v26.8b, %56.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %56.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %57.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.8b, %57.b[3]             \n" // k03

                "st1    {v29.4s, v30.4s}, [%7], #32  \n"
                //########################################### // sum6
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %57.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %57.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %57.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %57.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v29.4s, v30.4s}, [%8]       \n" // sum7 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[11]            \n" // k11
                
                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "ld1    {v8.8b}, [%9], #8            \n" // r0"

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "ld1    {v9.8b}, [%10], #8           \n" // r1"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "ld1    {v10.8b}, [%11], #8          \n" // r2"
                "ld1    {v11.8b}, [%12], #8          \n" // r3"
                "dup    v24.8b, %50.b[0]             \n" // k00                     

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v25.8b, %50.b[1]             \n" // k01

                "saddw2 v30.4s, v30.4s, v28.8h       \n"
                "dup    v26.8b, %50.b[2]             \n" // k02
                "dup    v27.8b, %50.b[3]             \n" // k03                

                "st1    {v29.4s, v30.4s}, [%8], #32  \n"   
                //########################################### // sum7
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7),// %8
                  "=r"(r0),     // %9
                  "=r"(r1),     // %10
                  "=r"(r2),     // %11
                  "=r"(r3),     // %12
                  "=r"(r4),     // %13
                  "=r"(r5),     // %14
                  "=r"(r6),     // %15
                  "=r"(r7),     // %16
                  "=r"(r8),     // %17
                  "=r"(r9),     // %18
                  "=r"(r10),    // %19
                  "=r"(r11),    // %20
                  "=r"(r12),    // %21
                  "=r"(r13),    // %22
                  "=r"(r14),    // %23
                  "=r"(r15)     // %24
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
                  "12"(r3),
                  "13"(r4),
                  "14"(r5),
                  "15"(r6),
                  "16"(r7),
                  "17"(r8),
                  "18"(r9),
                  "19"(r10),
                  "20"(r11),
                  "21"(r12),
                  "22"(r13),
                  "23"(r14),
                  "24"(r15),
                  "w"(_k0),     // %50
                  "w"(_k1),     // %51
                  "w"(_k2),     // %52
                  "w"(_k3),     // %53
                  "w"(_k4),     // %54
                  "w"(_k5),     // %55
                  "w"(_k6),     // %56
                  "w"(_k7)      // %57
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );                             
            }

            if (remain >= 4)
            {
                remain -= 4;

            asm volatile(
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"
                "ld1    {v8.8b}, [%9], #8            \n" // r0"
                "ld1    {v9.8b}, [%10], #8           \n" // r1"
                "ld1    {v10.8b}, [%11], #8          \n" // r2"
                "ld1    {v11.8b}, [%12], #8          \n" // r3"

                "dup    v24.8b, %50.b[0]             \n" // k00
                "dup    v25.8b, %50.b[1]             \n" // k01
                "dup    v26.8b, %50.b[2]             \n" // k02
                "dup    v27.8b, %50.b[3]             \n" // k03

                "smull  v28.8h, v8.8b, v24.8b        \n" // r0
                "prfm   pldl1keep, [%13, #128]       \n"
                "prfm   pldl1keep, [%14, #128]       \n"
                "prfm   pldl1keep, [%15, #128]       \n"

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "prfm   pldl1keep, [%16, #128]       \n"
                "ld1    {v12.8b}, [%13], #8          \n" // r4" 
                "ld1    {v13.8b}, [%14], #8          \n" // r5"

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "ld1    {v14.8b}, [%15], #8          \n" // r6"
                "ld1    {v15.8b}, [%16], #8          \n" // r7"                         
                "dup    v24.8b, %50.b[4]             \n" // k04

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "dup    v25.8b, %50.b[5]             \n" // k05
                "dup    v26.8b, %50.b[6]             \n" // k06
                "dup    v27.8b, %50.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n" // r4
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v29.4s}, [%1]               \n" // sum0  
                "prfm   pldl1keep, [%17, #128]       \n"

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "ld1    {v16.8b}, [%17], #8          \n" // r8" 

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "ld1    {v17.8b}, [%18], #8          \n" // r9"
                "ld1    {v18.8b}, [%19], #8          \n" // r10"
                "ld1    {v19.8b}, [%20], #8          \n" // r11"

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[8]             \n" // k08
                "dup    v25.8b, %50.b[9]             \n" // k09
                "dup    v26.8b, %50.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n" // r8
                "dup    v27.8b, %50.b[11]            \n" // k11
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "ld1    {v20.8b}, [%21], #8          \n" // r12"

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "ld1    {v21.8b}, [%22], #8          \n" // r13"
                "ld1    {v22.8b}, [%23], #8          \n" // r14"
                "ld1    {v23.8b}, [%24], #8          \n" // r15" 

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[12]            \n" // k12
                "dup    v25.8b, %50.b[13]            \n" // k13
                "dup    v26.8b, %50.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n" // r12
                "dup    v27.8b, %50.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %51.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %51.b[3]             \n" // k03

                "st1    {v29.4s}, [%1], #16          \n" // sum0
                //########################################### 
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %51.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %51.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %51.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %51.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v29.4s}, [%2]               \n" // sum1

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %52.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %52.b[3]             \n" // k03  

                "st1    {v29.4s}, [%2], #16          \n"
                //########################################### // sum1

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %52.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %52.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %52.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %52.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v29.4s}, [%3]               \n" // sum2 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[10]            \n" // k10

                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %53.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %53.b[3]             \n" // k03

                "st1    {v29.4s}, [%3], #16          \n"
                //########################################### //sum 2

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %53.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %53.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %53.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %53.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v29.4s}, [%4]               \n" // sum3 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %54.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %54.b[3]             \n" // k03

                "st1    {v29.4s}, [%4], #16          \n"
                //########################################### // sum3
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %54.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %54.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %54.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %54.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v29.4s}, [%5]               \n" // sum4

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"    
                "dup    v26.8b, %54.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %54.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[2]             \n" // k02 
            
                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %55.b[3]             \n" // k03
                
                "st1    {v29.4s}, [%5], #16          \n"
                //########################################### // sum4
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %55.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %55.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %55.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %55.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v29.4s}, [%6]               \n" // sum5  

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %56.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "dup    v27.8b, %56.b[3]             \n" // k03

                "st1    {v29.4s}, [%6], #16          \n"
                //########################################### // sum5
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %56.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %56.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %56.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %56.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v29.4s}, [%7]               \n" // sum6 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"        
                "dup    v26.8b, %56.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %56.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %57.b[2]             \n" // k02

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "saddw2 v30.4s, v30.4s, v28.8h       \n"

                "dup    v27.8b, %57.b[3]             \n" // k03

                "st1    {v29.4s}, [%7], #16          \n"
                //########################################### // sum6
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %57.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %57.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %57.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %57.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v29.4s}, [%8]               \n" // sum7 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[11]            \n" // k11
                
                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "sub    %9, %9, #4                   \n"

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "sub    %10, %10, #4                 \n"
                "sub    %11, %11, #4                 \n"
                "sub    %12, %12, #4                 \n"

                "smlal  v28.8h, v23.8b, v27.8b       \n"    
                "sub    %13, %13, #4                 \n"
                "sub    %14, %14, #4                 \n"
                "sub    %15, %15, #4                 \n"
                "sub    %16, %16, #4                 \n"

                "saddw  v29.4s, v29.4s, v28.4h       \n"
                "sub    %17, %17, #4                 \n"
                "sub    %18, %18, #4                 \n"
                "sub    %19, %19, #4                 \n"
                "sub    %20, %20, #4                 \n"

                "st1    {v29.4s}, [%8], #16          \n"
                //########################################### // sum7
                "sub    %21, %21, #4                 \n"
                "sub    %22, %22, #4                 \n"
                "sub    %23, %23, #4                 \n"
                "sub    %24, %24, #4                 \n" 
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7),// %8
                  "=r"(r0),     // %9
                  "=r"(r1),     // %10
                  "=r"(r2),     // %11
                  "=r"(r3),     // %12
                  "=r"(r4),     // %13
                  "=r"(r5),     // %14
                  "=r"(r6),     // %15
                  "=r"(r7),     // %16
                  "=r"(r8),     // %17
                  "=r"(r9),     // %18
                  "=r"(r10),     // %19
                  "=r"(r11),     // %20
                  "=r"(r12),     // %21
                  "=r"(r13),     // %22
                  "=r"(r14),     // %23
                  "=r"(r15)      // %24
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
                  "12"(r3),
                  "13"(r4),
                  "14"(r5),
                  "15"(r6),
                  "16"(r7),
                  "17"(r8),
                  "18"(r9),
                  "19"(r10),
                  "20"(r11),
                  "21"(r12),
                  "22"(r13),
                  "23"(r14),
                  "24"(r15),
                  "w"(_k0),     // %50
                  "w"(_k1),     // %51
                  "w"(_k2),     // %52
                  "w"(_k3),     // %53
                  "w"(_k4),     // %54
                  "w"(_k5),     // %55
                  "w"(_k6),     // %56
                  "w"(_k7)      // %57
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            ); 
            }

            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7] + *r8 * kernel0[8] + *r9 * kernel0[9] + *r10 * kernel0[10] + *r11 * kernel0[11] + *r12 * kernel0[12] + *r13 * kernel0[13] + *r14 * kernel0[14] + *r15 * kernel0[15];
                int sum1 = (int)*r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7] + *r8 * kernel1[8] + *r9 * kernel1[9] + *r10 * kernel1[10] + *r11 * kernel1[11] + *r12 * kernel1[12] + *r13 * kernel1[13] + *r14 * kernel1[14] + *r15 * kernel1[15];
                int sum2 = (int)*r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7] + *r8 * kernel2[8] + *r9 * kernel2[9] + *r10 * kernel2[10] + *r11 * kernel2[11] + *r12 * kernel2[12] + *r13 * kernel2[13] + *r14 * kernel2[14] + *r15 * kernel2[15];
                int sum3 = (int)*r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7] + *r8 * kernel3[8] + *r9 * kernel3[9] + *r10 * kernel3[10] + *r11 * kernel3[11] + *r12 * kernel3[12] + *r13 * kernel3[13] + *r14 * kernel3[14] + *r15 * kernel3[15];
                int sum4 = (int)*r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7] + *r8 * kernel4[8] + *r9 * kernel4[9] + *r10 * kernel4[10] + *r11 * kernel4[11] + *r12 * kernel4[12] + *r13 * kernel4[13] + *r14 * kernel4[14] + *r15 * kernel4[15];
                int sum5 = (int)*r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7] + *r8 * kernel5[8] + *r9 * kernel5[9] + *r10 * kernel5[10] + *r11 * kernel5[11] + *r12 * kernel5[12] + *r13 * kernel5[13] + *r14 * kernel5[14] + *r15 * kernel5[15];
                int sum6 = (int)*r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7] + *r8 * kernel6[8] + *r9 * kernel6[9] + *r10 * kernel6[10] + *r11 * kernel6[11] + *r12 * kernel6[12] + *r13 * kernel6[13] + *r14 * kernel6[14] + *r15 * kernel6[15];
                int sum7 = (int)*r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7] + *r8 * kernel7[8] + *r9 * kernel7[9] + *r10 * kernel7[10] + *r11 * kernel7[11] + *r12 * kernel7[12] + *r13 * kernel7[13] + *r14 * kernel7[14] + *r15 * kernel7[15];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                r8++;
                r9++;
                r10++;
                r11++;
                r12++;
                r13++;
                r14++;
                r15++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;          
            }
        }
#else // f**k the gcc limit the num of asm operand less than 30 
        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;
            const signed char* kernel4 = (const signed char*)kernel + (p+4)*inch + q;
            const signed char* kernel5 = (const signed char*)kernel + (p+5)*inch + q;
            const signed char* kernel6 = (const signed char*)kernel + (p+6)*inch + q;
            const signed char* kernel7 = (const signed char*)kernel + (p+7)*inch + q;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);

            int size = outw * outh;

            int nn = size >> 4;
            int remain = size & 15;

            asm volatile(
                "ld1    {v0.16b}, [%0]    \n"
                "ld1    {v1.16b}, [%1]    \n"
                "ld1    {v2.16b}, [%2]    \n"
                "ld1    {v3.16b}, [%3]    \n"
                "ld1    {v4.16b}, [%4]    \n"
                "ld1    {v5.16b}, [%5]    \n"
                "ld1    {v6.16b}, [%6]    \n"
                "ld1    {v7.16b}, [%7]    \n"
                : 
                : "r"(kernel0),
                  "r"(kernel1),
                  "r"(kernel2),
                  "r"(kernel3),
                  "r"(kernel4),
                  "r"(kernel5),
                  "r"(kernel6),
                  "r"(kernel7)
                : "cc", "memory"
            );

	    if (nn > 0)
            {
            asm volatile(
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "prfm   pldl1keep, [%25, #128]       \n"
                "ld1    {v8.16b}, [%18], #16         \n" // r0"
                "ld1    {v9.16b}, [%19], #16         \n" // r1"
                "ld1    {v10.16b}, [%20], #16        \n" // r2"
                "ld1    {v11.16b}, [%21], #16        \n" // r3"
                "ld1    {v12.16b}, [%22], #16        \n" // r4"
                "ld1    {v13.16b}, [%23], #16        \n" // r5"
                "ld1    {v14.16b}, [%24], #16        \n" // r6"
                "ld1    {v15.16b}, [%25], #16        \n" // r7"
                
                "0:                                  \n"

                "dup    v16.16b, v0.16b[0]           \n" // k00
                "dup    v17.16b, v0.16b[1]           \n" // k01
                "dup    v18.16b, v0.16b[2]           \n" // k02
                "dup    v19.16b, v0.16b[3]           \n" // k03
                "dup    v20.16b, v0.16b[4]           \n" // k04
                "dup    v21.16b, v0.16b[5]           \n" // k05
                "dup    v22.16b, v0.16b[6]           \n" // k06
                "dup    v23.16b, v0.16b[7]           \n" // k07				

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v1.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v1.16b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v1.16b[2]           \n" // k02

                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%1]       \n" // sum0  
                                    
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v1.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v1.16b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"
                "dup    v21.16b, v1.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                
                "st1    {v26.4s, v27.4s}, [%1], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%1]       \n" // sum0n
                "dup    v22.16b, v1.16b[6]           \n" // k06
                "dup    v23.16b, v1.16b[7]           \n" // k07	
                
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n" 
                "st1    {v28.4s, v29.4s}, [%1], #32  \n"
                //###########################################
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n"
                "dup    v16.16b, v2.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v2.16b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v2.16b[2]           \n" // k02

                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%2]       \n" // sum1

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v2.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v2.16b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "dup    v21.16b, v2.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"

                "st1    {v26.4s, v27.4s}, [%2], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%2]       \n" // sum1n
                "dup    v22.16b, v2.16b[6]           \n" // k06
                "dup    v23.16b, v2.16b[7]           \n" // k07					
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n" 

                "st1    {v28.4s, v29.4s}, [%2], #32  \n"
                //###########################################

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v3.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v3.16b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v3.16b[2]           \n" // k02					

                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%3]       \n" // sum2

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v3.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v3.16b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "dup    v21.16b, v3.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%3], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%3]       \n" // sum2n
                "dup    v22.16b, v3.16b[6]           \n" // k06
                "dup    v23.16b, v3.16b[7]           \n" // k07					
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n"
                "st1    {v28.4s, v29.4s}, [%3], #32  \n"
                //##########################################

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v4.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v4.16b[1]            \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v4.16b[2]           \n" // k02					

                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%4]       \n" // sum3

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v4.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v4.16b[4]            \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n" 
                "dup    v21.16b, v4.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%4], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%4]       \n" // sum3n
                "dup    v22.16b, v4.16b[6]           \n" // k06
                "dup    v23.16b, v4.16b[7]           \n" // k07					
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n"
                
                "st1    {v28.4s, v29.4s}, [%4], #32  \n"
                //##########################################	

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v5.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v5.16b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v5.16b[2]           \n" // k02

                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%5]       \n" // sum4

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v5.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v5.16b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n" 
                "dup    v21.16b, v5.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"	
                "st1    {v26.4s, v27.4s}, [%5], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%5]       \n" // sum4n
                "dup    v22.16b, v5.16b[6]           \n" // k06
                "dup    v23.16b, v5.16b[7]           \n" // k07
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n"	
                "st1    {v28.4s, v29.4s}, [%5], #32  \n"
                //##########################################	

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v6.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v6.16b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v6.16b[2]           \n" // k02
                
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v6.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v6.16b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "dup    v21.16b, v6.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%6]       \n" // sum5
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%6], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%6]       \n" // sum5n
                "dup    v22.16b, v6.16b[6]           \n" // k06
                "dup    v23.16b, v6.16b[7]           \n" // k07
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n"
                "st1    {v28.4s, v29.4s}, [%6], #32  \n"
                //##########################################

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v7.16b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v7.16b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v7.16b[2]           \n" // k02					

                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%7]       \n" // sum6

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v7.16b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v7.16b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"
                "dup    v21.16b, v7.16b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%7], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%7]       \n" // sum6n
                "dup    v22.16b, v7.16b[6]           \n" // k06
                "dup    v23.16b, v7.16b[7]           \n" // k07
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n"
                "st1    {v28.4s, v29.4s}, [%7], #32  \n"
                //##########################################		

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "prfm   pldl1keep, [%20, #128]       \n"
                "prfm   pldl1keep, [%21, #128]       \n"
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "prfm   pldl1keep, [%22, #128]       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"

                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%8]       \n" // sum7
                                    
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "prfm   pldl1keep, [%25, #128]       \n"
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "ld1    {v8.16b}, [%18], #16         \n" // r0"
                "ld1    {v9.16b}, [%19], #16         \n" // r1"
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "ld1    {v10.16b}, [%20], #16        \n" // r2"
                "ld1    {v11.16b}, [%21], #16        \n" // r3"
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"   
                "ld1    {v12.16b}, [%22], #16        \n" // r4"
                "ld1    {v13.16b}, [%23], #16        \n" // r5"					
                
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"				
                "st1    {v26.4s, v27.4s}, [%8], #32  \n" 
                
                "ld1    {v28.4s, v29.4s}, [%8]       \n" // sum7n
                "ld1    {v14.16b}, [%24], #16        \n" // r6"
                "ld1    {v15.16b}, [%25], #16        \n" // r7"						
                "saddw  v28.4s, v28.4s, v25.4h       \n"
                "saddw2 v29.4s, v29.4s, v25.8h       \n"
                "st1    {v28.4s, v29.4s}, [%8], #32  \n"
                "subs   %w0, %w0, #1                 \n"
                "bne    0b                           \n"
                "sub    %18, %18, #16                \n"
                "sub    %19, %19, #16                \n"
                "sub    %20, %20, #16                \n"
                "sub    %21, %21, #16                \n"
                "sub    %22, %22, #16                \n"
                "sub    %23, %23, #16                \n"
                "sub    %24, %24, #16                \n"
                "sub    %25, %25, #16                \n"
                //##########################################					
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7) // %8
                : "0"(nn),      
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "r"(r0),              // %18
                  "r"(r1),		// %19
                  "r"(r2),		// %20
                  "r"(r3),		// %21
                  "r"(r4),		// %22
                  "r"(r5),		// %23
                  "r"(r6),		// %24
                  "r"(r7)		// %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
            );
			}

            if (remain == 8)
            {
                remain -= 8;

            asm volatile(
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "prfm   pldl1keep, [%25, #128]       \n"				
                "ld1    {v8.8b}, [%18], #8           \n" // r0"
                "ld1    {v9.8b}, [%19], #8           \n" // r1"
                "ld1    {v10.8b}, [%20], #8          \n" // r2"
                "ld1    {v11.8b}, [%21], #8          \n" // r3"
                "ld1    {v12.8b}, [%22], #8          \n" // r4"   
                "ld1    {v13.8b}, [%23], #8          \n" // r5"	
                "ld1    {v14.8b}, [%24], #8          \n" // r6"
                "ld1    {v15.8b}, [%25], #8          \n" // r7" 

                "dup    v16.8b, v0.16b[0]            \n" // k00
                "dup    v17.8b, v0.16b[1]            \n" // k01
                "dup    v18.8b, v0.16b[2]            \n" // k02
                "dup    v19.8b, v0.16b[3]            \n" // k03
                "dup    v20.8b, v0.16b[4]            \n" // k04
                "dup    v21.8b, v0.16b[5]            \n" // k05
                "dup    v22.8b, v0.16b[6]            \n" // k06
                "dup    v23.8b, v0.16b[7]            \n" // k07				

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%1]       \n" // sum0  
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%1], #32  \n" 
                //###########################################
                "dup    v16.8b, v1.16b[0]            \n" // k00
                "dup    v17.8b, v1.16b[1]            \n" // k01
                "dup    v18.8b, v1.16b[2]            \n" // k02
                "dup    v19.8b, v1.16b[3]            \n" // k03
                "dup    v20.8b, v1.16b[4]            \n" // k04
                "dup    v21.8b, v1.16b[5]            \n" // k05
                "dup    v22.8b, v1.16b[6]            \n" // k06
                "dup    v23.8b, v1.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%2]       \n" // sum1
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%2], #32  \n" 
                //###########################################
                "dup    v16.8b, v2.16b[0]            \n" // k00
                "dup    v17.8b, v2.16b[1]            \n" // k01
                "dup    v18.8b, v2.16b[2]            \n" // k02
                "dup    v19.8b, v2.16b[3]            \n" // k03
                "dup    v20.8b, v2.16b[4]            \n" // k04
                "dup    v21.8b, v2.16b[5]            \n" // k05
                "dup    v22.8b, v2.16b[6]            \n" // k06
                "dup    v23.8b, v2.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%3]       \n" // sum2
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%3], #32  \n" 
                //##########################################
                "dup    v16.8b, v3.16b[0]            \n" // k00
                "dup    v17.8b, v3.16b[1]            \n" // k01
                "dup    v18.8b, v3.16b[2]            \n" // k02
                "dup    v19.8b, v3.16b[3]            \n" // k03
                "dup    v20.8b, v3.16b[4]            \n" // k04
                "dup    v21.8b, v3.16b[5]            \n" // k05
                "dup    v22.8b, v3.16b[6]            \n" // k06
                "dup    v23.8b, v3.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%4]       \n" // sum3
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%4], #32  \n" 
                //##########################################	
                "dup    v16.8b, v4.16b[0]            \n" // k00
                "dup    v17.8b, v4.16b[1]            \n" // k01
                "dup    v18.8b, v4.16b[2]            \n" // k02
                "dup    v19.8b, v4.16b[3]            \n" // k03
                "dup    v20.8b, v4.16b[4]            \n" // k04
                "dup    v21.8b, v4.16b[5]            \n" // k05
                "dup    v22.8b, v4.16b[6]            \n" // k06
                "dup    v23.8b, v4.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%5]       \n" // sum4
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%5], #32  \n" 
                //##########################################	
                "dup    v16.8b, v5.16b[0]            \n" // k00
                "dup    v17.8b, v5.16b[1]            \n" // k01
                "dup    v18.8b, v5.16b[2]            \n" // k02
                "dup    v19.8b, v5.16b[3]            \n" // k03
                "dup    v20.8b, v5.16b[4]            \n" // k04
                "dup    v21.8b, v5.16b[5]            \n" // k05
                "dup    v22.8b, v5.16b[6]            \n" // k06
                "dup    v23.8b, v5.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%6]       \n" // sum5
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%6], #32  \n" 
                //##########################################
                "dup    v16.8b, v6.16b[0]            \n" // k00
                "dup    v17.8b, v6.16b[1]            \n" // k01
                "dup    v18.8b, v6.16b[2]            \n" // k02
                "dup    v19.8b, v6.16b[3]            \n" // k03
                "dup    v20.8b, v6.16b[4]            \n" // k04
                "dup    v21.8b, v6.16b[5]            \n" // k05
                "dup    v22.8b, v6.16b[6]            \n" // k06
                "dup    v23.8b, v6.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%7]       \n" // sum6
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%7], #32  \n" 
                //##########################################		
                "dup    v16.8b, v7.16b[0]            \n" // k00
                "dup    v17.8b, v7.16b[1]            \n" // k01
                "dup    v18.8b, v7.16b[2]            \n" // k02
                "dup    v19.8b, v7.16b[3]            \n" // k03
                "dup    v20.8b, v7.16b[4]            \n" // k04
                "dup    v21.8b, v7.16b[5]            \n" // k05
                "dup    v22.8b, v7.16b[6]            \n" // k06
                "dup    v23.8b, v7.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%8]       \n" // sum7
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%8], #32  \n" 
                //##########################################					
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7) // %8
                : "0"(nn),      
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "r"(r0),              // %18
                  "r"(r1),		// %19
                  "r"(r2),		// %20
                  "r"(r3),		// %21
                  "r"(r4),		// %22
                  "r"(r5),		// %23
                  "r"(r6),		// %24
                  "r"(r7)		// %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
            );
			}                               

            if (remain == 4)
            {
                remain -= 4;

            asm volatile(		
                "ld1    {v8.8b}, [%18], #8           \n" // r0"
                "ld1    {v9.8b}, [%19], #8           \n" // r1"
                "ld1    {v10.8b}, [%20], #8          \n" // r2"
                "ld1    {v11.8b}, [%21], #8          \n" // r3"
                "ld1    {v12.8b}, [%22], #8          \n" // r4"   
                "ld1    {v13.8b}, [%23], #8          \n" // r5"	
                "ld1    {v14.8b}, [%24], #8          \n" // r6"
                "ld1    {v15.8b}, [%25], #8          \n" // r7" 

                "dup    v16.8b, v0.16b[0]            \n" // k00
                "dup    v17.8b, v0.16b[1]            \n" // k01
                "dup    v18.8b, v0.16b[2]            \n" // k02
                "dup    v19.8b, v0.16b[3]            \n" // k03
                "dup    v20.8b, v0.16b[4]            \n" // k04
                "dup    v21.8b, v0.16b[5]            \n" // k05
                "dup    v22.8b, v0.16b[6]            \n" // k06
                "dup    v23.8b, v0.16b[7]            \n" // k07				

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v26.4s}, [%1]               \n" // sum0  
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%1], #16  	     \n" 
                //###########################################
                "dup    v16.8b, v1.16b[0]            \n" // k00
                "dup    v17.8b, v1.16b[1]            \n" // k01
                "dup    v18.8b, v1.16b[2]            \n" // k02
                "dup    v19.8b, v1.16b[3]            \n" // k03
                "dup    v20.8b, v1.16b[4]            \n" // k04
                "dup    v21.8b, v1.16b[5]            \n" // k05
                "dup    v22.8b, v1.16b[6]            \n" // k06
                "dup    v23.8b, v1.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v26.4s}, [%2]               \n" // sum1
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%2], #16  	     \n" 
                //###########################################
                "dup    v16.8b, v2.16b[0]            \n" // k00
                "dup    v17.8b, v2.16b[1]            \n" // k01
                "dup    v18.8b, v2.16b[2]            \n" // k02
                "dup    v19.8b, v2.16b[3]            \n" // k03
                "dup    v20.8b, v2.16b[4]            \n" // k04
                "dup    v21.8b, v2.16b[5]            \n" // k05
                "dup    v22.8b, v2.16b[6]            \n" // k06
                "dup    v23.8b, v2.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v26.4s}, [%3]       	     \n" // sum2
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%3], #16  	     \n" 
                //##########################################
                "dup    v16.8b, v3.16b[0]            \n" // k00
                "dup    v17.8b, v3.16b[1]            \n" // k01
                "dup    v18.8b, v3.16b[2]            \n" // k02
                "dup    v19.8b, v3.16b[3]            \n" // k03
                "dup    v20.8b, v3.16b[4]            \n" // k04
                "dup    v21.8b, v3.16b[5]            \n" // k05
                "dup    v22.8b, v3.16b[6]            \n" // k06
                "dup    v23.8b, v3.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v26.4s}, [%4]       	     \n" // sum3
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%4], #16  	     \n" 
                //##########################################	
                "dup    v16.8b, v4.16b[0]            \n" // k00
                "dup    v17.8b, v4.16b[1]            \n" // k01
                "dup    v18.8b, v4.16b[2]            \n" // k02
                "dup    v19.8b, v4.16b[3]            \n" // k03
                "dup    v20.8b, v4.16b[4]            \n" // k04
                "dup    v21.8b, v4.16b[5]            \n" // k05
                "dup    v22.8b, v4.16b[6]            \n" // k06
                "dup    v23.8b, v4.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v26.4s}, [%5]       	     \n" // sum4
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%5], #16  	     \n" 
                //##########################################	
                "dup    v16.8b, v5.16b[0]            \n" // k00
                "dup    v17.8b, v5.16b[1]            \n" // k01
                "dup    v18.8b, v5.16b[2]            \n" // k02
                "dup    v19.8b, v5.16b[3]            \n" // k03
                "dup    v20.8b, v5.16b[4]            \n" // k04
                "dup    v21.8b, v5.16b[5]            \n" // k05
                "dup    v22.8b, v5.16b[6]            \n" // k06
                "dup    v23.8b, v5.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v26.4s}, [%6]       	     \n" // sum5
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%6], #16  	     \n" 
                //##########################################
                "dup    v16.8b, v6.16b[0]            \n" // k00
                "dup    v17.8b, v6.16b[1]            \n" // k01
                "dup    v18.8b, v6.16b[2]            \n" // k02
                "dup    v19.8b, v6.16b[3]            \n" // k03
                "dup    v20.8b, v6.16b[4]            \n" // k04
                "dup    v21.8b, v6.16b[5]            \n" // k05
                "dup    v22.8b, v6.16b[6]            \n" // k06
                "dup    v23.8b, v6.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v26.4s}, [%7]       	     \n" // sum6
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%7], #16  	     \n" 
                //##########################################		
                "dup    v16.8b, v7.16b[0]            \n" // k00
                "dup    v17.8b, v7.16b[1]            \n" // k01
                "dup    v18.8b, v7.16b[2]            \n" // k02
                "dup    v19.8b, v7.16b[3]            \n" // k03
                "dup    v20.8b, v7.16b[4]            \n" // k04
                "dup    v21.8b, v7.16b[5]            \n" // k05
                "dup    v22.8b, v7.16b[6]            \n" // k06
                "dup    v23.8b, v7.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v26.4s}, [%8]       	     \n" // sum7
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%8], #16  	     \n" 
                "sub    %18, %18, #4                 \n"
                "sub    %19, %19, #4                 \n"
                "sub    %20, %20, #4                 \n"
                "sub    %21, %21, #4                 \n"
                "sub    %22, %22, #4                 \n"
                "sub    %23, %23, #4                 \n"
                "sub    %24, %24, #4                 \n"
                "sub    %25, %25, #4                 \n"
                //##########################################					
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7) // %8
                : "0"(nn),      
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "r"(r0),              // %18
                  "r"(r1),		// %19
                  "r"(r2),		// %20
                  "r"(r3),		// %21
                  "r"(r4),		// %22
                  "r"(r5),		// %23
                  "r"(r6),		// %24
                  "r"(r7)		// %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
            );                
            }
	
	    for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                int sum1 = (int)*r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                int sum2 = (int)*r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                int sum3 = (int)*r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];
                int sum4 = (int)*r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7];
                int sum5 = (int)*r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7];
                int sum6 = (int)*r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7];
                int sum7 = (int)*r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

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
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;          
            }
        }       
#endif

        for (; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;
            const signed char* kernel4 = (const signed char*)kernel + (p+4)*inch + q;
            const signed char* kernel5 = (const signed char*)kernel + (p+5)*inch + q;
            const signed char* kernel6 = (const signed char*)kernel + (p+6)*inch + q;
            const signed char* kernel7 = (const signed char*)kernel + (p+7)*inch + q;

            const signed char k0 = kernel0[0];
            const signed char k1 = kernel1[0];
            const signed char k2 = kernel2[0];
            const signed char k3 = kernel3[0];
            const signed char k4 = kernel4[0];
            const signed char k5 = kernel5[0];
            const signed char k6 = kernel6[0];
            const signed char k7 = kernel7[0];

            const signed char* r0 = img0;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);
            int8x8_t _k4 = vdup_n_s8(k4);
            int8x8_t _k5 = vdup_n_s8(k5);
            int8x8_t _k6 = vdup_n_s8(k6);
            int8x8_t _k7 = vdup_n_s8(k7);

            for (; nn>0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);

                int32x4_t _out0  = vld1q_s32(outptr0);
                int32x4_t _out0n = vld1q_s32(outptr0+4);
                int32x4_t _out1  = vld1q_s32(outptr1);
                int32x4_t _out1n = vld1q_s32(outptr1+4);
                int32x4_t _out2  = vld1q_s32(outptr2);
                int32x4_t _out2n = vld1q_s32(outptr2+4);
                int32x4_t _out3  = vld1q_s32(outptr3);
                int32x4_t _out3n = vld1q_s32(outptr3+4);
                int32x4_t _out4  = vld1q_s32(outptr4);
                int32x4_t _out4n = vld1q_s32(outptr4+4);
                int32x4_t _out5  = vld1q_s32(outptr5);
                int32x4_t _out5n = vld1q_s32(outptr5+4);
                int32x4_t _out6  = vld1q_s32(outptr6);
                int32x4_t _out6n = vld1q_s32(outptr6+4);
                int32x4_t _out7  = vld1q_s32(outptr7);
                int32x4_t _out7n = vld1q_s32(outptr7+4);

                int16x8_t _out0_s16 = vmull_s8(_r0, _k0);
                int16x8_t _out1_s16 = vmull_s8(_r0, _k1);
                int16x8_t _out2_s16 = vmull_s8(_r0, _k2);
                int16x8_t _out3_s16 = vmull_s8(_r0, _k3);
                int16x8_t _out4_s16 = vmull_s8(_r0, _k4);
                int16x8_t _out5_s16 = vmull_s8(_r0, _k5);
                int16x8_t _out6_s16 = vmull_s8(_r0, _k6);
                int16x8_t _out7_s16 = vmull_s8(_r0, _k7);

                _out0  = vaddw_s16(_out0, vget_low_s16(_out0_s16));
                _out0n = vaddw_s16(_out0n, vget_high_s16(_out0_s16));
                _out1  = vaddw_s16(_out1, vget_low_s16(_out1_s16));
                _out1n = vaddw_s16(_out1n, vget_high_s16(_out1_s16));
                _out2  = vaddw_s16(_out2, vget_low_s16(_out2_s16));
                _out2n = vaddw_s16(_out2n, vget_high_s16(_out2_s16));
                _out3  = vaddw_s16(_out3, vget_low_s16(_out3_s16));
                _out3n = vaddw_s16(_out3n, vget_high_s16(_out3_s16));
                _out4  = vaddw_s16(_out4, vget_low_s16(_out4_s16));
                _out4n = vaddw_s16(_out4n, vget_high_s16(_out4_s16));
                _out5  = vaddw_s16(_out5, vget_low_s16(_out5_s16));
                _out5n = vaddw_s16(_out5n, vget_high_s16(_out5_s16));
                _out6  = vaddw_s16(_out6, vget_low_s16(_out6_s16));
                _out6n = vaddw_s16(_out6n, vget_high_s16(_out6_s16));
                _out7  = vaddw_s16(_out7, vget_low_s16(_out7_s16));
                _out7n = vaddw_s16(_out7n, vget_high_s16(_out7_s16));

                vst1q_s32(outptr0, _out0);
                vst1q_s32(outptr0+4, _out0n);
                vst1q_s32(outptr1, _out1);
                vst1q_s32(outptr1+4, _out1n);
                vst1q_s32(outptr2, _out2);
                vst1q_s32(outptr2+4, _out2n);
                vst1q_s32(outptr3, _out3);
                vst1q_s32(outptr3+4, _out3n);
                vst1q_s32(outptr4, _out4);
                vst1q_s32(outptr4+4, _out4n);
                vst1q_s32(outptr5, _out5);
                vst1q_s32(outptr5+4, _out5n);
                vst1q_s32(outptr6, _out6);
                vst1q_s32(outptr6+4, _out6n);
                vst1q_s32(outptr7, _out7);
                vst1q_s32(outptr7+4, _out7n);

                r0 += 8;
                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
                outptr4 += 8;
                outptr5 += 8;
                outptr6 += 8;
                outptr7 += 8;
            }
            
            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = (int)*r0 * k0;
                int sum1 = (int)*r0 * k1;
                int sum2 = (int)*r0 * k2;
                int sum3 = (int)*r0 * k3;
                int sum4 = (int)*r0 * k4;
                int sum5 = (int)*r0 * k5;
                int sum6 = (int)*r0 * k6;
                int sum7 = (int)*r0 * k7;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);
            const signed char* img1 = bottom_blob.channel(q+1);
            const signed char* img2 = bottom_blob.channel(q+2);
            const signed char* img3 = bottom_blob.channel(q+3);
            const signed char* img4 = bottom_blob.channel(q+4);
            const signed char* img5 = bottom_blob.channel(q+5);
            const signed char* img6 = bottom_blob.channel(q+6);
            const signed char* img7 = bottom_blob.channel(q+7);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char k0 = kernel0[0];
            const signed char k1 = kernel0[1];
            const signed char k2 = kernel0[2];
            const signed char k3 = kernel0[3];
            const signed char k4 = kernel0[4];
            const signed char k5 = kernel0[5];
            const signed char k6 = kernel0[6];
            const signed char k7 = kernel0[7];

            const signed char* r0 = img0;
            const signed char* r1 = img1;
            const signed char* r2 = img2;
            const signed char* r3 = img3;
            const signed char* r4 = img4;
            const signed char* r5 = img5;
            const signed char* r6 = img6;
            const signed char* r7 = img7;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);
            int8x8_t _k4 = vdup_n_s8(k4);
            int8x8_t _k5 = vdup_n_s8(k5);
            int8x8_t _k6 = vdup_n_s8(k6);
            int8x8_t _k7 = vdup_n_s8(k7);

            for (; nn>0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);
                int8x8_t _r1 = vld1_s8(r1);
                int8x8_t _r2 = vld1_s8(r2);
                int8x8_t _r3 = vld1_s8(r3);
                int8x8_t _r4 = vld1_s8(r4);
                int8x8_t _r5 = vld1_s8(r5);
                int8x8_t _r6 = vld1_s8(r6);
                int8x8_t _r7 = vld1_s8(r7);

                int32x4_t _out0 = vld1q_s32(outptr);
                int32x4_t _out0n = vld1q_s32(outptr+4);

                int16x8_t _out0_s16 = vmull_s8(_r0, _k0);
                _out0_s16 = vmlal_s8(_out0_s16, _r1, _k1);
                _out0_s16 = vmlal_s8(_out0_s16, _r2, _k2);
                _out0_s16 = vmlal_s8(_out0_s16, _r3, _k3);
                _out0_s16 = vmlal_s8(_out0_s16, _r4, _k4);
                _out0_s16 = vmlal_s8(_out0_s16, _r5, _k5);
                _out0_s16 = vmlal_s8(_out0_s16, _r6, _k6);
                _out0_s16 = vmlal_s8(_out0_s16, _r7, _k7);

                _out0 = vaddw_s16(_out0, vget_low_s16(_out0_s16));
                _out0n = vaddw_s16(_out0n, vget_high_s16(_out0_s16));

                vst1q_s32(outptr, _out0);
                vst1q_s32(outptr+4, _out0n);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                r5 += 8;
                r6 += 8;
                r7 += 8;
                outptr += 8;
            }

            for (; remain>0; remain--)
            {
                int sum  = (int)*r0 * k0;
                int sum1 = (int)*r1 * k1;
                int sum2 = (int)*r2 * k2;
                int sum3 = (int)*r3 * k3;
                int sum4 = (int)*r4 * k4;
                int sum5 = (int)*r5 * k5;
                int sum6 = (int)*r6 * k6;
                int sum7 = (int)*r7 * k7;

                *outptr += sum + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr++;
            }

        }

        for (; q<inch; q++)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);
            const signed char* kernel0 = (const signed char*)kernel + p*inch  + q;
            const signed char k0 = kernel0[0];
            const signed char* r0 = img0;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);

            if (nn > 0)
            {
                int8x8_t _r0 = vld1_s8(r0);

                int32x4_t _out0 = vld1q_s32(outptr);
                int32x4_t _out0n = vld1q_s32(outptr+4);

                int16x8_t _out0_s16 = vmull_s8(_r0, _k0);

                _out0 = vaddw_s16(_out0, vget_low_s16(_out0_s16));
                _out0n = vaddw_s16(_out0n, vget_high_s16(_out0_s16));

                vst1q_s32(outptr, _out0);
                vst1q_s32(outptr+4, _out0n);

                r0 += 8;
                outptr += 8;
            }

            for (; remain>0; remain--)
            {
                int sum = (int)*r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
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
