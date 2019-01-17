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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static void conv7x7s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*49  + q*49;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;
            const float* r4 = img0 + w*4;
            const float* r5 = img0 + w*5;
            const float* r6 = img0 + w*6;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 7;
            const float* k2 = kernel0 + 14;
            const float* k3 = kernel0 + 21;
            const float* k4 = kernel0 + 28;
            const float* k5 = kernel0 + 35;
            const float* k6 = kernel0 + 42;

            int i = 0;

            for (; i < outh; i++)
            {

#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _k0123 = vld1q_f32(k0);
                float32x4_t _k4567 = vld1q_f32(k0 + 4);
                float32x4_t _k78910 = vld1q_f32(k1);
                float32x4_t _k11121314 = vld1q_f32(k1 + 4);
                float32x4_t _k14151617 = vld1q_f32(k2);
                float32x4_t _k18192021 = vld1q_f32(k2 + 4);
                float32x4_t _k21222324 = vld1q_f32(k3);
                float32x4_t _k25262728 = vld1q_f32(k3 + 4);
                float32x4_t _k28293031 = vld1q_f32(k4);
                float32x4_t _k32333435 = vld1q_f32(k4 + 4);
                float32x4_t _k35363738 = vld1q_f32(k5);
                float32x4_t _k39404142 = vld1q_f32(k5 + 4);
                float32x4_t _k42434445 = vld1q_f32(k6);
                float32x4_t _k46474849 = vld1q_f32(k6 + 4);
#ifdef __clang__    // __ARM_NEON && __aarch64__ && __clang__
                if (nn > 0)
                {
                asm volatile(
                    // v0:  input / final output
                    // v1 v2 v3: = ri0 ri4 ri0n , i <-  1-7
                    // v4 = ri1 / ri3 / ri6
                    // v5 = ri2 / ri5
                    // v9 = intermediate sum register
                    "0:                                        \n"                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    //i = 1
                    "prfm       pldl1keep, [%2, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%2]    \n"
                    "add        %2, %2, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmul       v9.4s, v1.4s, %18.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %18.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %18.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %18.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %19.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %19.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %19.s[2]         \n"

                    //i = 2
                    "prfm       pldl1keep, [%3, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%3]    \n" // v1 v2 v3: = r20 r24 r20n
                    "add        %3, %3, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n" // v4 = r21
                    "fmla       v9.4s, v1.4s, %20.s[0]         \n" // *+ r10
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n" // v5 = r22
                    "fmla       v0.4s, v4.4s, %20.s[1]         \n" // *+ r11
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n" // v4 = r23
                    "fmla       v9.4s, v5.4s, %20.s[2]         \n" // *+ r1
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n" // v5 = r25
                    "fmla       v0.4s, v4.4s, %20.s[3]         \n" // *+ r13
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n" // v4 = r26
                    "fmla       v9.4s, v2.4s, %21.s[0]         \n" // *+ r14
                    "fmla       v0.4s, v5.4s, %21.s[1]         \n" // *+ r15
                    "fmla       v9.4s, v4.4s, %21.s[2]         \n" // *+ r16

                    //i = 3
                    "prfm       pldl1keep, [%4, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%4]    \n"
                    "add        %4, %4, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %22.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %22.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %22.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %22.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %23.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %23.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %23.s[2]         \n"

                    //i = 4
                    "prfm       pldl1keep, [%5, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%5]    \n"
                    "add        %5, %5, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %24.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %24.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %24.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %24.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %25.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %25.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %25.s[2]         \n"

                    //i = 5
                    "prfm       pldl1keep, [%6, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%6]    \n"
                    "add        %6, %6, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %26.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %26.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %26.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %26.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %27.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %27.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %27.s[2]         \n"

                    //i = 6
                    "prfm       pldl1keep, [%7, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%7]    \n"
                    "add        %7, %7, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %28.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %28.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %28.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %28.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %29.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %29.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %29.s[2]         \n"
                    
                    //i = 7
                    "prfm       pldl1keep, [%8, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%8]    \n"
                    "add        %8, %8, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %30.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %30.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %30.s[2]         \n"                    
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %30.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %31.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %31.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %31.s[2]         \n"

                    "fadd       v0.4s, v0.4s, v9.4s            \n"                    
                    "st1        {v0.4s}, [%1], #16             \n"                    
                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"
                    
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5),         // %7
                      "=r"(r6)          // %8
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "w"(_k0123),     // %18
                      "w"(_k4567),     // %19
                      "w"(_k78910),    // %20
                      "w"(_k11121314), // %21
                      "w"(_k14151617), // %22
                      "w"(_k18192021), // %23
                      "w"(_k21222324), // %24
                      "w"(_k25262728), // %25
                      "w"(_k28293031), // %26
                      "w"(_k32333435), // %27
                      "w"(_k35363738), // %28
                      "w"(_k39404142), // %29
                      "w"(_k42434445), // %30
                      "w"(_k46474849)  // %31
                    : "cc", "memory","v0", "v1", "v2", "v3", "v4", "v5", "v9"
                );
                }                    
#else   // __ARM_NEON && __aarch64__ defined, but __clang__ not defined
// When compiled with gcc, gcc does not accept over 30 operands
                for (; nn>0; nn--)
                {
                    float32x4_t _sum = vld1q_f32(outptr);

                    float32x4_t _r00 = vld1q_f32(r0);// 0 1 2 3
                    float32x4_t _r04 = vld1q_f32(r0 + 4);// 4 5 6 7
                    float32x4_t _r00n = vld1q_f32(r0 + 8);// 8 9 10 11
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1);// 1 2 3 4
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2);// 2 3 4 5
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3);// 3 4 5 6
                    float32x4_t _r05 = vextq_f32(_r04, _r00n, 1);// 5 6 7 8
                    float32x4_t _r06 = vextq_f32(_r04, _r00n, 2);// 6 7 8 9

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r05, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r06, _k4567, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r14 = vld1q_f32(r1 + 4);
                    float32x4_t _r10n = vld1q_f32(r1 + 8);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);
                    float32x4_t _r15 = vextq_f32(_r14, _r10n, 1);
                    float32x4_t _r16 = vextq_f32(_r14, _r10n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k78910, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k78910, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k78910, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k78910, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k11121314, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r15, _k11121314, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r16, _k11121314, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r24 = vld1q_f32(r2 + 4);
                    float32x4_t _r20n = vld1q_f32(r2 + 8);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);
                    float32x4_t _r25 = vextq_f32(_r24, _r20n, 1);
                    float32x4_t _r26 = vextq_f32(_r24, _r20n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k14151617, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k14151617, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k14151617, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k14151617, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k18192021, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r25, _k18192021, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r26, _k18192021, 2);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r34 = vld1q_f32(r3 + 4);
                    float32x4_t _r30n = vld1q_f32(r3 + 8);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);
                    float32x4_t _r35 = vextq_f32(_r34, _r30n, 1);
                    float32x4_t _r36 = vextq_f32(_r34, _r30n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k21222324, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k21222324, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k21222324, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k21222324, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k25262728, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r35, _k25262728, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r36, _k25262728, 2);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r44 = vld1q_f32(r4 + 4);
                    float32x4_t _r40n = vld1q_f32(r4 + 8);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);
                    float32x4_t _r45 = vextq_f32(_r44, _r40n, 1);
                    float32x4_t _r46 = vextq_f32(_r44, _r40n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k28293031, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k28293031, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k28293031, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k28293031, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k32333435, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r45, _k32333435, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r46, _k32333435, 2);

                    float32x4_t _r50 = vld1q_f32(r5);
                    float32x4_t _r54 = vld1q_f32(r5 + 4);
                    float32x4_t _r50n = vld1q_f32(r5 + 8);
                    float32x4_t _r51 = vextq_f32(_r50, _r54, 1);
                    float32x4_t _r52 = vextq_f32(_r50, _r54, 2);
                    float32x4_t _r53 = vextq_f32(_r50, _r54, 3);
                    float32x4_t _r55 = vextq_f32(_r54, _r50n, 1);
                    float32x4_t _r56 = vextq_f32(_r54, _r50n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r50, _k35363738, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r51, _k35363738, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r52, _k35363738, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r53, _k35363738, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r54, _k39404142, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r55, _k39404142, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r56, _k39404142, 2);

                    float32x4_t _r60 = vld1q_f32(r6);
                    float32x4_t _r64 = vld1q_f32(r6 + 4);
                    float32x4_t _r60n = vld1q_f32(r6 + 8);
                    float32x4_t _r61 = vextq_f32(_r60, _r64, 1);
                    float32x4_t _r62 = vextq_f32(_r60, _r64, 2);
                    float32x4_t _r63 = vextq_f32(_r60, _r64, 3);
                    float32x4_t _r65 = vextq_f32(_r64, _r60n, 1);
                    float32x4_t _r66 = vextq_f32(_r64, _r60n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r60, _k42434445, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r61, _k42434445, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r62, _k42434445, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r63, _k42434445, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r64, _k46474849, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r65, _k46474849, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r66, _k46474849, 2);

                    vst1q_f32(outptr, _sum);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    r6 += 4;
                    outptr += 4;
                }
#endif   // __clang__
#else //__aarch32__
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d24-d25}, [%1]     \n"// _sum
//                     "veor       q13, q13            \n"// _sum2 = 0;
//                     "veor       q14, q14            \n"// _sum3 = 0;
//                     "veor       q15, q15            \n"// _sum4 = 0;

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k0123 k4567
                    "add        %9, #28             \n"

                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%2]!      \n"// q0 = 0  1  2  3
                    "vmla.f32   q12, q0, d8[0]      \n"

                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2]       \n"// q2 = 4  5  6  7  q3 = 8  9 10 11
                    "vmul.f32   q13, q2, d10[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"// q1 = 1  2  3  4
                    "vext.32    q10, q2, q3, #1     \n"// q10= 5  6  7  8
                    "vmul.f32   q14, q1, d8[1]      \n"
                    "vmul.f32   q15, q10, d10[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"// q8 = 2  3  4  5
                    "vext.32    q11, q2, q3, #2     \n"// q11= 6  7  8  9
                    "vmla.f32   q12, q8, d9[0]      \n"
                    "vmla.f32   q13, q11, d11[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"// q9 = 3  4  5  6
                    "vmla.f32   q14, q9, d9[1]      \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d12-d15}, [%9]     \n"// q6 q7 = k78910 k11121314
                    "add        %9, #28             \n"

                    "pld        [%3, #128]          \n"
                    "vld1.f32   {d0-d1}, [%3]!      \n"
                    "vmla.f32   q15, q0, d12[0]     \n"

                    "pld        [%3, #256]          \n"
                    "vld1.f32   {d4-d7}, [%3]       \n"
                    "vmla.f32   q12, q2, d14[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"
                    "vext.32    q10, q2, q3, #1     \n"
                    "vmla.f32   q13, q1, d12[1]     \n"
                    "vmla.f32   q14, q10, d14[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"
                    "vext.32    q11, q2, q3, #2     \n"
                    "vmla.f32   q15, q8, d13[0]     \n"
                    "vmla.f32   q12, q11, d15[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"
                    "vmla.f32   q13, q9, d13[1]     \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k14151617 k18192021
                    "add        %9, #28             \n"

                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d0-d1}, [%4]!      \n"
                    "vmla.f32   q14, q0, d8[0]      \n"

                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d4-d7}, [%4]       \n"
                    "vmla.f32   q15, q2, d10[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"
                    "vext.32    q10, q2, q3, #1     \n"
                    "vmla.f32   q12, q1, d8[1]      \n"
                    "vmla.f32   q13, q10, d10[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"
                    "vext.32    q11, q2, q3, #2     \n"
                    "vmla.f32   q14, q8, d9[0]      \n"
                    "vmla.f32   q15, q11, d11[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"
                    "vmla.f32   q12, q9, d9[1]      \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d12-d15}, [%9]     \n"// q6 q7 = k21222324 k25262728
                    "add        %9, #28             \n"

                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"
                    "vmla.f32   q13, q0, d12[0]     \n"

                    "pld        [%5, #256]          \n"
                    "vld1.f32   {d4-d7}, [%5]       \n"
                    "vmla.f32   q14, q2, d14[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"
                    "vext.32    q10, q2, q3, #1     \n"
                    "vmla.f32   q15, q1, d12[1]     \n"
                    "vmla.f32   q12, q10, d14[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"
                    "vext.32    q11, q2, q3, #2     \n"
                    "vmla.f32   q13, q8, d13[0]     \n"
                    "vmla.f32   q14, q11, d15[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"
                    "vmla.f32   q15, q9, d13[1]     \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k28293031 k32333435
                    "add        %9, #28             \n"

                    "pld        [%6, #128]          \n"
                    "vld1.f32   {d0-d1}, [%6]!      \n"
                    "vmla.f32   q12, q0, d8[0]      \n"

                    "pld        [%6, #256]          \n"
                    "vld1.f32   {d4-d7}, [%6]       \n"
                    "vmla.f32   q13, q2, d10[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"
                    "vext.32    q10, q2, q3, #1     \n"
                    "vmla.f32   q14, q1, d8[1]      \n"
                    "vmla.f32   q15, q10, d10[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"
                    "vext.32    q11, q2, q3, #2     \n"
                    "vmla.f32   q12, q8, d9[0]      \n"
                    "vmla.f32   q13, q11, d11[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"
                    "vmla.f32   q14, q9, d9[1]      \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d12-d15}, [%9]     \n"// q6 q7 = k35363738 k39404142
                    "add        %9, #28             \n"

                    "pld        [%7, #128]          \n"
                    "vld1.f32   {d0-d1}, [%7]!      \n"
                    "vmla.f32   q15, q0, d12[0]     \n"

                    "pld        [%7, #256]          \n"
                    "vld1.f32   {d4-d7}, [%7]       \n"
                    "vmla.f32   q12, q2, d14[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"
                    "vext.32    q10, q2, q3, #1     \n"
                    "vmla.f32   q13, q1, d12[1]     \n"
                    "vmla.f32   q14, q10, d14[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"
                    "vext.32    q11, q2, q3, #2     \n"
                    "vmla.f32   q15, q8, d13[0]     \n"
                    "vmla.f32   q12, q11, d15[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"
                    "vmla.f32   q13, q9, d13[1]     \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k42434445 k46474849
                    "sub        %9, #168            \n"// restore k0

                    "pld        [%8, #128]          \n"
                    "vld1.f32   {d0-d1}, [%8]!      \n"
                    "vmla.f32   q14, q0, d8[0]      \n"

                    "pld        [%8, #256]          \n"
                    "vld1.f32   {d4-d7}, [%8]       \n"
                    "vmla.f32   q15, q2, d10[0]     \n"

                    "vext.32    q1, q0, q2, #1      \n"
                    "vext.32    q10, q2, q3, #1     \n"
                    "vmla.f32   q12, q1, d8[1]      \n"
                    "vmla.f32   q13, q10, d10[1]    \n"

                    "vext.32    q8, q0, q2, #2      \n"
                    "vext.32    q11, q2, q3, #2     \n"
                    "vmla.f32   q14, q8, d9[0]      \n"
                    "vmla.f32   q15, q11, d11[0]    \n"

                    "vext.32    q9, q0, q2, #3      \n"
                    "vmla.f32   q12, q9, d9[1]      \n"

                    "vadd.f32   q13, q13, q14       \n"
                    "vadd.f32   q13, q13, q15       \n"
                    "vadd.f32   q12, q12, q13       \n"

                    "vst1.f32   {d24-d25}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5),         // %7
                      "=r"(r6),         // %8
                      "=r"(k0)          // %9
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "9"(k0)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON

                for (; remain>0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];
                    sum += r0[5] * k0[5];
                    sum += r0[6] * k0[6];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];
                    sum += r1[5] * k1[5];
                    sum += r1[6] * k1[6];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];
                    sum += r2[5] * k2[5];
                    sum += r2[6] * k2[6];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];
                    sum += r3[5] * k3[5];
                    sum += r3[6] * k3[6];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
                    sum += r4[5] * k4[5];
                    sum += r4[6] * k4[6];

                    sum += r5[0] * k5[0];
                    sum += r5[1] * k5[1];
                    sum += r5[2] * k5[2];
                    sum += r5[3] * k5[3];
                    sum += r5[4] * k5[4];
                    sum += r5[5] * k5[5];
                    sum += r5[6] * k5[6];

                    sum += r6[0] * k6[0];
                    sum += r6[1] * k6[1];
                    sum += r6[2] * k6[2];
                    sum += r6[3] * k6[3];
                    sum += r6[4] * k6[4];
                    sum += r6[5] * k6[5];
                    sum += r6[6] * k6[6];

                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    r5++;
                    r6++;
                    outptr++;
                }

                r0 += 6;
                r1 += 6;
                r2 += 6;
                r3 += 6;
                r4 += 6;
                r5 += 6;
                r6 += 6;

            }

        }
    }

}

static void conv7x7s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*49  + q*49;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;
            const float* r4 = img0 + w*4;
            const float* r5 = img0 + w*5;
            const float* r6 = img0 + w*6;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 7;
            const float* k2 = kernel0 + 14;
            const float* k3 = kernel0 + 21;
            const float* k4 = kernel0 + 28;
            const float* k5 = kernel0 + 35;
            const float* k6 = kernel0 + 42;

            int i = 0;

            for (; i < outh; i++)
            {

#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _k0123 = vld1q_f32(k0);
                float32x4_t _k4567 = vld1q_f32(k0 + 4);
                float32x4_t _k78910 = vld1q_f32(k1);
                float32x4_t _k11121314 = vld1q_f32(k1 + 4);
                float32x4_t _k14151617 = vld1q_f32(k2);
                float32x4_t _k18192021 = vld1q_f32(k2 + 4);
                float32x4_t _k21222324 = vld1q_f32(k3);
                float32x4_t _k25262728 = vld1q_f32(k3 + 4);
                float32x4_t _k28293031 = vld1q_f32(k4);
                float32x4_t _k32333435 = vld1q_f32(k4 + 4);
                float32x4_t _k35363738 = vld1q_f32(k5);
                float32x4_t _k39404142 = vld1q_f32(k5 + 4);
                float32x4_t _k42434445 = vld1q_f32(k6);
                float32x4_t _k46474849 = vld1q_f32(k6 + 4);
#ifdef __clang__    // __ARM_NEON && __aarch64__ && __clang__
                if (nn > 0)
                {
                asm volatile(
                    // v0:  input / final output
                    // v1 v2: = _ri0/_ri1  first 
                    // v3 v4: =                  then _r0_8101214/_r0_9111315
                    // v5 = ri2 / ri4 / ri6
                    // v6 = ri3 / ri5
                    // v9 = intermediate sum register
                    "0:                                        \n"                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    //i = 1
                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%2]           \n" // v1  v2 = _r00  _r01
                    "add        %2, %2, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%2]           \n" // v3  v4 = _r0_8101214 / _r0_9111315     
                    "fmul       v9.4s, v1.4s, %18.s[0]         \n" // *+ _r00                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n" // v5 = _r02
                    "fmla       v0.4s, v2.4s, %18.s[1]         \n" // *+ _r01
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n" // v6 = _r03
                    "fmla       v9.4s, v5.4s, %18.s[2]         \n" // *+ _r02
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n" // v5 = _r04
                    "fmla       v0.4s, v6.4s, %18.s[3]         \n" // *+ _r03
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n" // v6 = _r05
                    "fmla       v9.4s, v5.4s, %19.s[0]         \n" // *+ _r04                    
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n" // v5 = _r06
                    "fmla       v0.4s, v6.4s, %19.s[1]         \n" // *+ _r05
                    "fmla       v9.4s, v5.4s, %19.s[2]         \n" // *+ _r06

                    //i = 2
                    "prfm       pldl1keep, [%3, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%3]           \n"
                    "add        %3, %3, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%3]           \n"    
                    "fmla       v9.4s, v1.4s, %20.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %20.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %20.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %20.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %21.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %21.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %21.s[2]         \n"

                    //i = 3
                    "prfm       pldl1keep, [%4, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%4]           \n"
                    "add        %4, %4, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%4]           \n"    
                    "fmla       v9.4s, v1.4s, %22.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %22.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %22.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %22.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %23.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %23.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %23.s[2]         \n"

                    //i = 4
                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%5]           \n"
                    "add        %5, %5, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%5]           \n"    
                    "fmla       v9.4s, v1.4s, %24.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %24.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %24.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %24.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %25.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %25.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %25.s[2]         \n"

                    //i = 5
                    "prfm       pldl1keep, [%6, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%6]           \n"
                    "add        %6, %6, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%6]           \n"    
                    "fmla       v9.4s, v1.4s, %26.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %26.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %26.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %26.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %27.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %27.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %27.s[2]         \n"

                    //i = 6
                    "prfm       pldl1keep, [%7, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%7]           \n"
                    "add        %7, %7, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%7]           \n"    
                    "fmla       v9.4s, v1.4s, %28.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %28.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %28.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %28.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %29.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %29.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %29.s[2]         \n"

                    //i = 7
                    "prfm       pldl1keep, [%8, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%8]           \n"
                    "add        %8, %8, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%8]           \n"    
                    "fmla       v9.4s, v1.4s, %30.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %30.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %30.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %30.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %31.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %31.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %31.s[2]         \n"

                    "fadd       v0.4s, v0.4s, v9.4s            \n"                    
                    "st1        {v0.4s}, [%1], #16             \n"                    
                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5),         // %7
                      "=r"(r6)          // %8
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "w"(_k0123),     // %18
                      "w"(_k4567),     // %19
                      "w"(_k78910),    // %20
                      "w"(_k11121314), // %21
                      "w"(_k14151617), // %22
                      "w"(_k18192021), // %23
                      "w"(_k21222324), // %24
                      "w"(_k25262728), // %25
                      "w"(_k28293031), // %26
                      "w"(_k32333435), // %27
                      "w"(_k35363738), // %28
                      "w"(_k39404142), // %29
                      "w"(_k42434445), // %30
                      "w"(_k46474849)  // %31
                    : "cc", "memory","v0", "v1", "v2", "v3", "v4", "v5", "v6", "v9"
                );
                }    
#else   // __ARM_NEON && __aarch64__ defined, but __clang__ not defined
// When compiled with gcc, gcc does not accept over 30 operands
                for (; nn>0; nn--)
                {
                    float32x4_t _sum = vld1q_f32(outptr);

                    float32x4x2_t _r00_02461357 = vld2q_f32(r0);
                    float32x4x2_t _r00nx2 = vld2q_f32(r0 + 8);
                    float32x4_t _r0_8101214 = _r00nx2.val[0];// 8 10 12 14
                    float32x4_t _r0_9111315 = _r00nx2.val[1];// 9 11 13 15
                    float32x4_t _r00 = _r00_02461357.val[0];// 0 2 4 6
                    float32x4_t _r01 = _r00_02461357.val[1];// 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0_8101214, 1);// 2 4 6 8
                    float32x4_t _r03 = vextq_f32(_r01, _r0_9111315, 1);// 3 5 7 9
                    float32x4_t _r04 = vextq_f32(_r00, _r0_8101214, 2);// 4 6 8 10
                    float32x4_t _r05 = vextq_f32(_r01, _r0_9111315, 2);// 5 7 9 11
                    float32x4_t _r06 = vextq_f32(_r00, _r0_8101214, 3);// 6 8 10 12

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r05, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r06, _k4567, 2);

                    float32x4x2_t _r10_02461357 = vld2q_f32(r1);
                    float32x4x2_t _r10nx2 = vld2q_f32(r1 + 8);
                    float32x4_t _r1_8101214 = _r10nx2.val[0];
                    float32x4_t _r1_9111315 = _r10nx2.val[1];
                    float32x4_t _r10 = _r10_02461357.val[0];
                    float32x4_t _r11 = _r10_02461357.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1_8101214, 1);
                    float32x4_t _r13 = vextq_f32(_r11, _r1_9111315, 1);
                    float32x4_t _r14 = vextq_f32(_r10, _r1_8101214, 2);
                    float32x4_t _r15 = vextq_f32(_r11, _r1_9111315, 2);
                    float32x4_t _r16 = vextq_f32(_r10, _r1_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k78910, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k78910, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k78910, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k78910, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k11121314, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r15, _k11121314, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r16, _k11121314, 2);

                    float32x4x2_t _r20_02461357 = vld2q_f32(r2);
                    float32x4x2_t _r20nx2 = vld2q_f32(r2 + 8);
                    float32x4_t _r2_8101214 = _r20nx2.val[0];
                    float32x4_t _r2_9111315 = _r20nx2.val[1];
                    float32x4_t _r20 = _r20_02461357.val[0];
                    float32x4_t _r21 = _r20_02461357.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2_8101214, 1);
                    float32x4_t _r23 = vextq_f32(_r21, _r2_9111315, 1);
                    float32x4_t _r24 = vextq_f32(_r20, _r2_8101214, 2);
                    float32x4_t _r25 = vextq_f32(_r21, _r2_9111315, 2);
                    float32x4_t _r26 = vextq_f32(_r20, _r2_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k14151617, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k14151617, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k14151617, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k14151617, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k18192021, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r25, _k18192021, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r26, _k18192021, 2);

                    float32x4x2_t _r30_02461357 = vld2q_f32(r3);
                    float32x4x2_t _r30nx2 = vld2q_f32(r3 + 8);
                    float32x4_t _r3_8101214 = _r30nx2.val[0];
                    float32x4_t _r3_9111315 = _r30nx2.val[1];
                    float32x4_t _r30 = _r30_02461357.val[0];
                    float32x4_t _r31 = _r30_02461357.val[1];
                    float32x4_t _r32 = vextq_f32(_r30, _r3_8101214, 1);
                    float32x4_t _r33 = vextq_f32(_r31, _r3_9111315, 1);
                    float32x4_t _r34 = vextq_f32(_r30, _r3_8101214, 2);
                    float32x4_t _r35 = vextq_f32(_r31, _r3_9111315, 2);
                    float32x4_t _r36 = vextq_f32(_r30, _r3_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k21222324, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k21222324, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k21222324, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k21222324, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k25262728, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r35, _k25262728, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r36, _k25262728, 2);

                    float32x4x2_t _r40_02461357 = vld2q_f32(r4);
                    float32x4x2_t _r40nx2 = vld2q_f32(r4 + 8);
                    float32x4_t _r4_8101214 = _r40nx2.val[0];
                    float32x4_t _r4_9111315 = _r40nx2.val[1];
                    float32x4_t _r40 = _r40_02461357.val[0];
                    float32x4_t _r41 = _r40_02461357.val[1];
                    float32x4_t _r42 = vextq_f32(_r40, _r4_8101214, 1);
                    float32x4_t _r43 = vextq_f32(_r41, _r4_9111315, 1);
                    float32x4_t _r44 = vextq_f32(_r40, _r4_8101214, 2);
                    float32x4_t _r45 = vextq_f32(_r41, _r4_9111315, 2);
                    float32x4_t _r46 = vextq_f32(_r40, _r4_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k28293031, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k28293031, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k28293031, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k28293031, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k32333435, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r45, _k32333435, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r46, _k32333435, 2);

                    float32x4x2_t _r50_02461357 = vld2q_f32(r5);
                    float32x4x2_t _r50nx2 = vld2q_f32(r5 + 8);
                    float32x4_t _r5_8101214 = _r50nx2.val[0];
                    float32x4_t _r5_9111315 = _r50nx2.val[1];
                    float32x4_t _r50 = _r50_02461357.val[0];
                    float32x4_t _r51 = _r50_02461357.val[1];
                    float32x4_t _r52 = vextq_f32(_r50, _r5_8101214, 1);
                    float32x4_t _r53 = vextq_f32(_r51, _r5_9111315, 1);
                    float32x4_t _r54 = vextq_f32(_r50, _r5_8101214, 2);
                    float32x4_t _r55 = vextq_f32(_r51, _r5_9111315, 2);
                    float32x4_t _r56 = vextq_f32(_r50, _r5_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r50, _k35363738, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r51, _k35363738, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r52, _k35363738, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r53, _k35363738, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r54, _k39404142, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r55, _k39404142, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r56, _k39404142, 2);

                    float32x4x2_t _r60_02461357 = vld2q_f32(r6);
                    float32x4x2_t _r60nx2 = vld2q_f32(r6 + 8);
                    float32x4_t _r6_8101214 = _r60nx2.val[0];
                    float32x4_t _r6_9111315 = _r60nx2.val[1];
                    float32x4_t _r60 = _r60_02461357.val[0];
                    float32x4_t _r61 = _r60_02461357.val[1];
                    float32x4_t _r62 = vextq_f32(_r60, _r6_8101214, 1);
                    float32x4_t _r63 = vextq_f32(_r61, _r6_9111315, 1);
                    float32x4_t _r64 = vextq_f32(_r60, _r6_8101214, 2);
                    float32x4_t _r65 = vextq_f32(_r61, _r6_9111315, 2);
                    float32x4_t _r66 = vextq_f32(_r60, _r6_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r60, _k42434445, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r61, _k42434445, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r62, _k42434445, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r63, _k42434445, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r64, _k46474849, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r65, _k46474849, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r66, _k46474849, 2);

                    vst1q_f32(outptr, _sum);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                    outptr += 4;
                }
#endif   // __clang__
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d26-d27}, [%1]     \n"// _sum
//                     "veor       q14, q14            \n"// _sum2 = 0;
//                     "veor       q15, q15            \n"// _sum3 = 0;

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k0123 k4567
                    "add        %9, #28             \n"

                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d0-d3}, [%2]!      \n"// q0 = 0  2  4  6  q1 = 1  3  5  7
                    "vmla.f32   q13, q0, d8[0]      \n"
                    "vmul.f32   q14, q1, d8[1]      \n"

                    "vld2.f32   {d4-d7}, [%2]       \n"// q2 = 8 10 12 14  q3 = 9 11 13 15
                    "vext.32    q8, q0, q2, #1      \n"// q8 = 2  4  6  8
                    "vext.32    q9, q1, q3, #1      \n"// q9 = 3  5  7  9
                    "vmul.f32   q15, q8, d9[0]      \n"
                    "vmla.f32   q13, q9, d9[1]      \n"

                    "vext.32    q10, q0, q2, #2     \n"// q10= 4  6  8 10
                    "vext.32    q11, q1, q3, #2     \n"// q11= 5  7  9 11
                    "vmla.f32   q14, q10, d10[0]    \n"
                    "vmla.f32   q15, q11, d10[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"// q12= 6  8 10 12
                    "vmla.f32   q13, q12, d11[0]    \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d12-d15}, [%9]     \n"// q6 q7 = k78910 k11121314
                    "add        %9, #28             \n"

                    "pld        [%3, #512]          \n"
                    "vld2.f32   {d0-d3}, [%3]!      \n"
                    "vmla.f32   q14, q0, d12[0]     \n"
                    "vmla.f32   q15, q1, d12[1]     \n"

                    "vld2.f32   {d4-d7}, [%3]       \n"
                    "vext.32    q8, q0, q2, #1      \n"
                    "vext.32    q9, q1, q3, #1      \n"
                    "vmla.f32   q13, q8, d13[0]     \n"
                    "vmla.f32   q14, q9, d13[1]     \n"

                    "vext.32    q10, q0, q2, #2     \n"
                    "vext.32    q11, q1, q3, #2     \n"
                    "vmla.f32   q15, q10, d14[0]    \n"
                    "vmla.f32   q13, q11, d14[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"
                    "vmla.f32   q14, q12, d15[0]    \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k14151617 k18192021
                    "add        %9, #28             \n"

                    "pld        [%4, #512]          \n"
                    "vld2.f32   {d0-d3}, [%4]!      \n"
                    "vmla.f32   q15, q0, d8[0]      \n"
                    "vmla.f32   q13, q1, d8[1]      \n"

                    "vld2.f32   {d4-d7}, [%4]       \n"
                    "vext.32    q8, q0, q2, #1      \n"
                    "vext.32    q9, q1, q3, #1      \n"
                    "vmla.f32   q14, q8, d9[0]      \n"
                    "vmla.f32   q15, q9, d9[1]      \n"

                    "vext.32    q10, q0, q2, #2     \n"
                    "vext.32    q11, q1, q3, #2     \n"
                    "vmla.f32   q13, q10, d10[0]    \n"
                    "vmla.f32   q14, q11, d10[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"
                    "vmla.f32   q15, q12, d11[0]    \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d12-d15}, [%9]     \n"// q6 q7 = k21222324 k25262728
                    "add        %9, #28             \n"

                    "pld        [%5, #512]          \n"
                    "vld2.f32   {d0-d3}, [%5]!      \n"
                    "vmla.f32   q13, q0, d12[0]     \n"
                    "vmla.f32   q14, q1, d12[1]     \n"

                    "vld2.f32   {d4-d7}, [%5]       \n"
                    "vext.32    q8, q0, q2, #1      \n"
                    "vext.32    q9, q1, q3, #1      \n"
                    "vmla.f32   q15, q8, d13[0]     \n"
                    "vmla.f32   q13, q9, d13[1]     \n"

                    "vext.32    q10, q0, q2, #2     \n"
                    "vext.32    q11, q1, q3, #2     \n"
                    "vmla.f32   q14, q10, d14[0]    \n"
                    "vmla.f32   q15, q11, d14[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"
                    "vmla.f32   q13, q12, d15[0]    \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k28293031 k32333435
                    "add        %9, #28             \n"

                    "pld        [%6, #512]          \n"
                    "vld2.f32   {d0-d3}, [%6]!      \n"
                    "vmla.f32   q14, q0, d8[0]      \n"
                    "vmla.f32   q15, q1, d8[1]      \n"

                    "vld2.f32   {d4-d7}, [%6]       \n"
                    "vext.32    q8, q0, q2, #1      \n"
                    "vext.32    q9, q1, q3, #1      \n"
                    "vmla.f32   q13, q8, d9[0]      \n"
                    "vmla.f32   q14, q9, d9[1]      \n"

                    "vext.32    q10, q0, q2, #2     \n"
                    "vext.32    q11, q1, q3, #2     \n"
                    "vmla.f32   q15, q10, d10[0]    \n"
                    "vmla.f32   q13, q11, d10[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"
                    "vmla.f32   q14, q12, d11[0]    \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d12-d15}, [%9]     \n"// q6 q7 = k35363738 k39404142
                    "add        %9, #28             \n"

                    "pld        [%7, #512]          \n"
                    "vld2.f32   {d0-d3}, [%7]!      \n"
                    "vmla.f32   q15, q0, d12[0]     \n"
                    "vmla.f32   q13, q1, d12[1]     \n"

                    "vld2.f32   {d4-d7}, [%7]       \n"
                    "vext.32    q8, q0, q2, #1      \n"
                    "vext.32    q9, q1, q3, #1      \n"
                    "vmla.f32   q14, q8, d13[0]     \n"
                    "vmla.f32   q15, q9, d13[1]     \n"

                    "vext.32    q10, q0, q2, #2     \n"
                    "vext.32    q11, q1, q3, #2     \n"
                    "vmla.f32   q13, q10, d14[0]    \n"
                    "vmla.f32   q14, q11, d14[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"
                    "vmla.f32   q15, q12, d15[0]    \n"

                    "pld        [%9, #256]          \n"
                    "vld1.f32   {d8-d11}, [%9]      \n"// q4 q5 = k42434445 k46474849
                    "sub        %9, #168            \n"// restore k0

                    "pld        [%8, #512]          \n"
                    "vld2.f32   {d0-d3}, [%8]!      \n"
                    "vmla.f32   q13, q0, d8[0]      \n"
                    "vmla.f32   q14, q1, d8[1]      \n"

                    "vld2.f32   {d4-d7}, [%8]       \n"
                    "vext.32    q8, q0, q2, #1      \n"
                    "vext.32    q9, q1, q3, #1      \n"
                    "vmla.f32   q15, q8, d9[0]      \n"
                    "vmla.f32   q13, q9, d9[1]      \n"

                    "vext.32    q10, q0, q2, #2     \n"
                    "vext.32    q11, q1, q3, #2     \n"
                    "vmla.f32   q14, q10, d10[0]    \n"
                    "vmla.f32   q15, q11, d10[1]    \n"

                    "vext.32    q12, q0, q2, #3     \n"
                    "vmla.f32   q13, q12, d11[0]    \n"

                    "vadd.f32   q14, q14, q15       \n"
                    "vadd.f32   q13, q13, q14       \n"

                    "vst1.f32   {d26-d27}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5),         // %7
                      "=r"(r6),         // %8
                      "=r"(k0)          // %9
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "9"(k0)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON

                for (; remain>0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];
                    sum += r0[5] * k0[5];
                    sum += r0[6] * k0[6];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];
                    sum += r1[5] * k1[5];
                    sum += r1[6] * k1[6];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];
                    sum += r2[5] * k2[5];
                    sum += r2[6] * k2[6];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];
                    sum += r3[5] * k3[5];
                    sum += r3[6] * k3[6];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
                    sum += r4[5] * k4[5];
                    sum += r4[6] * k4[6];

                    sum += r5[0] * k5[0];
                    sum += r5[1] * k5[1];
                    sum += r5[2] * k5[2];
                    sum += r5[3] * k5[3];
                    sum += r5[4] * k5[4];
                    sum += r5[5] * k5[5];
                    sum += r5[6] * k5[6];

                    sum += r6[0] * k6[0];
                    sum += r6[1] * k6[1];
                    sum += r6[2] * k6[2];
                    sum += r6[3] * k6[3];
                    sum += r6[4] * k6[4];
                    sum += r6[5] * k6[5];
                    sum += r6[6] * k6[6];

                    *outptr += sum;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;

            }

        }
    }

}
