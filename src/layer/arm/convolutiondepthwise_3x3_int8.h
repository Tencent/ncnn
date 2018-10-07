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
static void convdw3x3s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const signed char* kernel = (const signed char *)_kernel + p*9;
        
        int* outptr0 = out;
        int* outptr0n = outptr0 + outw;
    
        const signed char* img0 = bottom_blob.channel(p);
        
        const signed char* r0 = img0;
        const signed char* r1 = img0 + w;
        const signed char* r2 = img0 + w*2;
        const signed char* r3 = img0 + w*3;

        int i = 0;

        int8x8_t _k0 = vdup_n_s8(kernel[0]);
        int8x8_t _k1 = vdup_n_s8(kernel[1]);
        int8x8_t _k2 = vdup_n_s8(kernel[2]);

        int8x8_t _k3 = vdup_n_s8(kernel[3]);
        int8x8_t _k4 = vdup_n_s8(kernel[4]);
        int8x8_t _k5 = vdup_n_s8(kernel[5]);

        int8x8_t _k6 = vdup_n_s8(kernel[6]);
        int8x8_t _k7 = vdup_n_s8(kernel[7]);
        int8x8_t _k8 = vdup_n_s8(kernel[8]);

        for (; i+1 < outh; i+=2)
        {
            int nn = outw >> 3;
            int remain = outw & 7;

            for (; nn >0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);
                int8x8_t _r0n = vld1_s8(r0+8);
                int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r0, _r0n, 2);

                int16x8_t _sum0 = vmull_s8(_r0, _k0);
                _sum0 = vmlal_s8(_sum0, _r01, _k1);
                _sum0 = vmlal_s8(_sum0, _r02, _k2);

                int8x8_t _r1 = vld1_s8(r1);
                int8x8_t _r1n = vld1_s8(r1+8);
                int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                _sum0 = vmlal_s8(_sum0, _r1, _k3);
                _sum0 = vmlal_s8(_sum0, _r11, _k4);
                _sum0 = vmlal_s8(_sum0, _r12, _k5);

                int16x8_t _sum1 = vmull_s8(_r1, _k0);
                _sum1 = vmlal_s8(_sum1, _r11, _k1);
                _sum1 = vmlal_s8(_sum1, _r12, _k2);

                int8x8_t _r2 = vld1_s8(r2);
                int8x8_t _r2n = vld1_s8(r2+8);
                int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                _sum0 = vmlal_s8(_sum0, _r2, _k6);
                _sum0 = vmlal_s8(_sum0, _r21, _k7);
                _sum0 = vmlal_s8(_sum0, _r22, _k8);

                _sum1 = vmlal_s8(_sum1, _r2, _k3);
                _sum1 = vmlal_s8(_sum1, _r21, _k4);
                _sum1 = vmlal_s8(_sum1, _r22, _k5);

                int8x8_t _r3 = vld1_s8(r3);
                int8x8_t _r3n = vld1_s8(r3+8);
                int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                _sum1 = vmlal_s8(_sum1, _r3, _k6);
                _sum1 = vmlal_s8(_sum1, _r31, _k7);
                _sum1 = vmlal_s8(_sum1, _r32, _k8);

                int32x4_t sum0_s32 = vmovl_s16(vget_low_s16(_sum0));
                int32x4_t sum0n_s32 = vmovl_s16(vget_high_s16(_sum0));

                vst1q_s32(outptr0, sum0_s32);
                vst1q_s32(outptr0+4, sum0n_s32);

                int32x4_t sum1_s32 = vmovl_s16(vget_low_s16(_sum1));
                int32x4_t sum1n_s32 = vmovl_s16(vget_high_s16(_sum1));

                vst1q_s32(outptr0n, sum1_s32);
                vst1q_s32(outptr0n+4, sum1n_s32);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr0 += 8;
                outptr0n += 8;
            }

            for (; remain>0; remain--)
            {
                //Todo Neon

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
            int nn = outw >> 3;
            int remain = outw & 7;

            for (; nn >0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);
                int8x8_t _r0n = vld1_s8(r0+8);
                int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r0, _r0n, 2);

                int16x8_t _sum0 = vmull_s8(_r0, _k0);
                _sum0 = vmlal_s8(_sum0, _r01, _k1);
                _sum0 = vmlal_s8(_sum0, _r02, _k2);

                int8x8_t _r1 = vld1_s8(r1);
                int8x8_t _r1n = vld1_s8(r1+8);
                int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                _sum0 = vmlal_s8(_sum0, _r1, _k3);
                _sum0 = vmlal_s8(_sum0, _r11, _k4);
                _sum0 = vmlal_s8(_sum0, _r12, _k5);

                int8x8_t _r2 = vld1_s8(r2);
                int8x8_t _r2n = vld1_s8(r2+8);
                int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                _sum0 = vmlal_s8(_sum0, _r2, _k6);
                _sum0 = vmlal_s8(_sum0, _r21, _k7);
                _sum0 = vmlal_s8(_sum0, _r22, _k8);

                int32x4_t sum0_s32 = vmovl_s16(vget_low_s16(_sum0));
                int32x4_t sum0n_s32 = vmovl_s16(vget_high_s16(_sum0));

                vst1q_s32(outptr0, sum0_s32);
                vst1q_s32(outptr0+4, sum0n_s32);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }

            for (; remain>0; remain--)
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

static void convdw3x3s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const signed char* kernel = (const signed char*)_kernel + p*9;

        int* outptr = out;

        const signed char* img = bottom_blob.channel(p);

        const signed char* r0 = img;
        const signed char* r1 = img + w;
        const signed char* r2 = img + w*2;

        int i = 0;

        int8x8_t _k0 = vdup_n_s8(kernel[0]);
        int8x8_t _k1 = vdup_n_s8(kernel[1]);
        int8x8_t _k2 = vdup_n_s8(kernel[2]);
        int8x8_t _k3 = vdup_n_s8(kernel[3]);
        int8x8_t _k4 = vdup_n_s8(kernel[4]);
        int8x8_t _k5 = vdup_n_s8(kernel[5]);
        int8x8_t _k6 = vdup_n_s8(kernel[6]);
        int8x8_t _k7 = vdup_n_s8(kernel[7]);
        int8x8_t _k8 = vdup_n_s8(kernel[8]);

        for (; i < outh; i++)
        {           
            int nn = outw >> 3;
            int remain = outw & 7;

            for (; nn > 0; nn--)
            {
                int8x8x2_t _r0 = vld2_s8(r0);
                int8x8x2_t _r0n = vld2_s8(r0+16);
                int8x8_t _r00 = _r0.val[0];
                int8x8_t _r01 = _r0.val[1];
                int8x8_t _r02 = vext_s8(_r00, _r0n.val[0], 1);

                int16x8_t _sum = vmull_s8(_r00, _k0);
                _sum = vmlal_s8(_sum, _r01, _k1);
                _sum = vmlal_s8(_sum, _r02, _k2);

                int8x8x2_t _r1 = vld2_s8(r1);
                int8x8x2_t _r1n = vld2_s8(r1+16);
                int8x8_t _r10 = _r1.val[0];
                int8x8_t _r11 = _r1.val[1];
                int8x8_t _r12 = vext_s8(_r10, _r1n.val[0], 1);
                _sum = vmlal_s8(_sum, _r10, _k3);
                _sum = vmlal_s8(_sum, _r11, _k4);
                _sum = vmlal_s8(_sum, _r12, _k5);

                int8x8x2_t _r2 = vld2_s8(r2);
                int8x8x2_t _r2n = vld2_s8(r2+16);
                int8x8_t _r20 = _r2.val[0];
                int8x8_t _r21 = _r2.val[1];
                int8x8_t _r22 = vext_s8(_r20, _r2n.val[0], 1);
                _sum = vmlal_s8(_sum, _r20, _k6);
                _sum = vmlal_s8(_sum, _r21, _k7);
                _sum = vmlal_s8(_sum, _r22, _k8);

                int32x4_t sum0_s32 = vmovl_s16(vget_low_s16(_sum));
                int32x4_t sum0n_s32 = vmovl_s16(vget_high_s16(_sum));

                vst1q_s32(outptr, sum0_s32);
                vst1q_s32(outptr+4, sum0n_s32);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr += 8;
            }       

            for (; remain>0; remain--)
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
#else // __aarch64__
static void convdw3x3s1_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const signed char* kernel = (const signed char *)_kernel + p*9;
        
        int* outptr0_s32 = out0;
        int* outptr0n_s32 = outptr0_s32 + outw;
    
        const signed char* img0 = bottom_blob.channel(p);
        
        const signed char* r0 = img0;
        const signed char* r1 = img0 + w;
        const signed char* r2 = img0 + w*2;
        const signed char* r3 = img0 + w*3;

        int i = 0;

        for (; i+1 < outh; i+=2)
        {
            int nn = outw >> 3;
            int remain = outw & 7;

            if (nn > 0)
            {
                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    : "=r"(kernel) // %0
                    : "0"(kernel)
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
                    
                    "vmovl.s16  q9, d4              \n"
                    "vmovl.s16  q10, d5             \n"
                    "vst1.32    {d18-d21}, [%1]!    \n"// sum0
                    
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

                    "vmovl.s16  q9, d4              \n"
                    "vmovl.s16  q10, d5             \n"
                    "vst1.32    {d18-d21}, [%2]!    \n"// sum0n

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0_s32),    // %1
                      "=r"(outptr0n_s32),   // %2
                      "=r"(r0),             // %3
                      "=r"(r1),             // %4
                      "=r"(r2),             // %5
                      "=r"(r3)              // %6
                    : "0"(nn),
                      "1"(outptr0_s32),
                      "2"(outptr0n_s32),
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

                *outptr0_s32 = sum0;
                *outptr0n_s32 = sum0n;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0_s32++;
                outptr0n_s32++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0_s32 += outw;
            outptr0n_s32 += outw;
        }

        for (; i < outh; i++)
        {
            int nn = outw >> 3;
            int remain = outw & 7;

            if (nn > 0)
            {
                asm volatile(
                    "vld1.8    {d26-d27}, [%0]    \n"
                    : "=r"(kernel) // %0
                    : "0"(kernel)
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
                    
                    "vmovl.s16  q9, d4              \n"
                    "vmovl.s16  q10, d5             \n"
                    "vst1.32    {d18-d21}, [%1]!    \n"// sum0

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr0_s32),    // %1
                      "=r"(r0),             // %2
                      "=r"(r1),             // %3
                      "=r"(r2)              // %4
                    : "0"(nn),
                      "1"(outptr0_s32),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }

            for (; remain>0; remain--)
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

                *outptr0_s32 = sum;

                r0++;
                r1++;
                r2++;
                outptr0_s32++;
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
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const signed char* kernel = (const signed char*)_kernel + p*9;

        int* outptr_s32 = out;

        const signed char* img = bottom_blob.channel(p);

        const signed char* r0 = img;
        const signed char* r1 = img + w;
        const signed char* r2 = img + w*2;

        int i = 0;

        int8x8_t _k0 = vdup_n_s8(kernel[0]);
        int8x8_t _k1 = vdup_n_s8(kernel[1]);
        int8x8_t _k2 = vdup_n_s8(kernel[2]);
        int8x8_t _k3 = vdup_n_s8(kernel[3]);
        int8x8_t _k4 = vdup_n_s8(kernel[4]);
        int8x8_t _k5 = vdup_n_s8(kernel[5]);
        int8x8_t _k6 = vdup_n_s8(kernel[6]);
        int8x8_t _k7 = vdup_n_s8(kernel[7]);
        int8x8_t _k8 = vdup_n_s8(kernel[8]);

        for (; i < outh; i++)
        {           
            int nn = outw >> 3;
            int remain = outw & 7;  

            if (nn > 0)
            {
                asm volatile(
                    "0:                             \n"
                    "pld        [%2, #192]          \n"
                    "vld2.s8    {d0-d1}, [%2]!      \n" // r0
                    "vld2.s8    {d2-d3}, [%2]       \n" //
                    "vext.8     d3, d0, d2, #1      \n"
        
                    "vmull.s8   q2, d0, %P10        \n" // k00
                    "vmull.s8   q3, d1, %P11        \n" // k01
                    "vmull.s8   q4, d3, %P12        \n" // k02

                    "veor       q7, q0, q0          \n"
                    "veor       q8, q0, q0          \n"

                    "pld        [%3, #192]          \n"
                    "vld2.s8    {d0-d1}, [%3]!      \n" // r1
                    "vld2.s8    {d2-d3}, [%3]       \n" //
                    "vext.8     d3, d0, d2, #1      \n"
                                    
                    "vmlal.s8   q2, d0, %P13        \n" // k03
                    "vmlal.s8   q3, d1, %P14        \n" // k04
                    "vmlal.s8   q4, d3, %P15        \n" // k05

                    "pld        [%4, #192]          \n" 
                    "vld2.s8    {d0-d1}, [%4]!      \n" // r2
                    "vld2.s8    {d2-d3}, [%4]       \n" //
                    "vext.8     d3, d0, d2, #1      \n"
                                            
                    "vmlal.s8   q2, d0, %P16        \n" // k06
                    "vmlal.s8   q3, d1, %P17        \n" // k07
                    "vmlal.s8   q4, d3, %P18        \n" // k08

                    "vadd.s16   q2, q2, q3          \n"
                    "vadd.s16   q2, q2, q4          \n"

                    "vaddw.s16  q7, q7, d4          \n"
                    "vaddw.s16  q8, q8, d5          \n"

                    "vst1.32    {d14-d17}, [%1]!    \n" // sum

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),             // %0
                      "=r"(outptr_s32),     // %1
                      "=r"(r0),             // %2
                      "=r"(r1),             // %3
                      "=r"(r2)              // %4
                    : "0"(nn),
                      "1"(outptr_s32),
                      "2"(r0),              // %7
                      "3"(r1),              // %8
                      "4"(r2),              // %9
                      "w"(_k0),             // %10
                      "w"(_k1),             // %11
                      "w"(_k2),             // %12
                      "w"(_k3),             // %13
                      "w"(_k4),             // %14
                      "w"(_k5),             // %15
                      "w"(_k6),             // %16
                      "w"(_k7),             // %17
                      "w"(_k8)              // %18
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q6", "q7", "q8", "q13", "q14"
                );               
            }           

            for (; remain>0; remain--)
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
                
                *outptr_s32 = sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr_s32++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
#endif
