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

// Saturate to 8 bits
#define  ROUND_S8(a)   ((a)>127 ? (127):((a)<(-128) ? (-128):(a)))	

static void conv1x1_quantize_int8_transform_kernel(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, const stQuantizeParams& scale)
{
    float ufKernelFactor = 0.f;

    //initial quantized kernel Mat
    kernel_tm.create(1*inch*outch);

    const float* kernel = _kernel;
    ufKernelFactor = scale.weightScale;

    //quantize the kernel weight
    signed char* kernel_s8 = (signed char*)kernel_tm.data;
    for (int p=0; p<outch; p++)
    {
        for (int q=0; q<inch; q++)
        {
            int tmp = p*inch*1 + q*1;
            signed char tmp2;

            for (int idx=0; idx<1; idx++)
            {
                if(kernel[tmp + idx]>=0)
                {
                    tmp2 = (signed char)(ufKernelFactor*kernel[tmp + idx]+0.5);
                }
                else
                {
                    tmp2 = (signed char)(ufKernelFactor*kernel[tmp + idx]-0.5);
                }
                
                kernel_s8[tmp + idx] = ROUND_S8(tmp2);
            }
        }
    }	
}

/*
 * Convolution 1x1 quantized with int8,unroll 8 x 4 
 */
static void conv1x1s1_neon_s8(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const stQuantizeParams& scale)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;	

    #pragma omp parallel for
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;
        const float bias2 = bias ? bias[p+2] : 0.f;
        const float bias3 = bias ? bias[p+3] : 0.f;

        out0.fill(0.f);
        out1.fill(0.f);
        out2.fill(0.f);
        out3.fill(0.f);

        int q = 0;
        
        for (; q+7<inch; q+=8)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            int* outptr0_s32 = (int*)outptr0;
            int* outptr1_s32 = (int*)outptr1;
            int* outptr2_s32 = (int*)outptr2;
            int* outptr3_s32 = (int*)outptr3;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);
            const float* img4 = bottom_blob.channel(q+4);
            const float* img5 = bottom_blob.channel(q+5);
            const float* img6 = bottom_blob.channel(q+6);
            const float* img7 = bottom_blob.channel(q+7);			

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;

            const signed char* pk0 = kernel0;
            const signed char* pk1 = kernel1;
            const signed char* pk2 = kernel2;
            const signed char* pk3 = kernel3;

            const signed char* r0 = (signed char *)img0;
            const signed char* r1 = (signed char *)img1;
            const signed char* r2 = (signed char *)img2;
            const signed char* r3 = (signed char *)img3;
            const signed char* r4 = (signed char *)img4;
            const signed char* r5 = (signed char *)img5;
            const signed char* r6 = (signed char *)img6;
            const signed char* r7 = (signed char *)img7;			

            int size = outw * outh;
        
            int nn = size >> 3;
            int remain = size & 7;

            asm volatile(
                "vld1.s8	d18, [%0]	\n"
                "vld1.s8	d19, [%1]	\n"
                "vld1.s8	d24, [%2]	\n"
                "vld1.s8	d25, [%3]	\n"
                : "=r"(pk0), // %0
                  "=r"(pk1), // %1
                  "=r"(pk2), // %2
                  "=r"(pk3)  // %3				  
                : "0"(pk0),
                  "1"(pk1),
                  "2"(pk2),
                  "3"(pk3)
                :
            );				

            if (nn > 0)
            {	
                asm volatile(
                    "0:  						   \n"
                    //ld r0-r7
                    "pld        [%5, #64]          \n"
                    "vld1.s8	{d0}, [%5 :64]!    \n"	//r0

                    "pld        [%6, #64]          \n"
                    "vld1.s8	{d1}, [%6 :64]!    \n"	//r1

                    "pld        [%7, #64]          \n"
                    "vld1.s8	{d2}, [%7 :64]!    \n"	//r2

                    "pld        [%8, #64]          \n"
                    "vld1.s8	{d3}, [%8 :64]!    \n"	//r3

                    "pld        [%9, #64]          \n"
                    "vld1.s8	{d4}, [%9 :64]!    \n"	//r4

                    "pld        [%10, #64]         \n"
                    "vld1.s8	{d5}, [%10 :64]!   \n"	//r5

                    "pld        [%11, #64]         \n"
                    "vld1.s8	{d6}, [%11 :64]!   \n"	//r6

                    "pld        [%12, #64]         \n"
                    "vld1.s8	{d7}, [%12 :64]!   \n"	//r7					
                    //###########################################
                    //load inch kernel_0 k0-k7
                    "vdup.s8	d8, d18[0]			\n"			
                    "vdup.s8	d9, d18[1]			\n"	
                    "vdup.s8	d10, d18[2]			\n"	
                    "vdup.s8	d11, d18[3]			\n"			
                    "vdup.s8	d12, d18[4]			\n"			
                    "vdup.s8	d13, d18[5]			\n"	
                    "vdup.s8	d14, d18[6]			\n"	
                    "vdup.s8	d15, d18[7]			\n"										
                    
                    //mla
                    "vmull.s8	q8, d0,	d8			\n"
                    "vmlal.s8	q8, d1,	d9			\n"
                    "vmlal.s8	q8, d2,	d10			\n"
                    "vmlal.s8	q8, d3,	d11			\n"		
                    "vmlal.s8	q8, d4,	d12			\n"
                    "vmlal.s8	q8, d5,	d13			\n"		
                    "vmlal.s8	q8, d6,	d14			\n"
                    "vmlal.s8	q8, d7,	d15			\n"	

                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32   	{d20-d23}, [%1:128] \n"	//outptr0_s32
                    "vmovl.s16	q4, d16				\n"
                    "vmovl.s16	q5, d17				\n"
                    "vadd.s32	q10, q4				\n"
                    "vadd.s32	q11, q5				\n"
                    "vst1.32   	{d20-d23}, [%1:128]!\n"
                    //###########################################
                    //load inch kernel_1 k0-k7
                    "vdup.s8	d8, d19[0]			\n"			
                    "vdup.s8	d9, d19[1]			\n"	
                    "vdup.s8	d10, d19[2]			\n"	
                    "vdup.s8	d11, d19[3]			\n"		
                    "vdup.s8	d12, d19[4]			\n"			
                    "vdup.s8	d13, d19[5]			\n"	
                    "vdup.s8	d14, d19[6]			\n"	
                    "vdup.s8	d15, d19[7]			\n"												
                    
                    //mla
                    "vmull.s8	q8, d0,	d8			\n"
                    "vmlal.s8	q8, d1,	d9			\n"
                    "vmlal.s8	q8, d2,	d10			\n"
                    "vmlal.s8	q8, d3,	d11			\n"		
                    "vmlal.s8	q8, d4,	d12			\n"
                    "vmlal.s8	q8, d5,	d13			\n"		
                    "vmlal.s8	q8, d6,	d14			\n"
                    "vmlal.s8	q8, d7,	d15			\n"			

                    //outptr1_s32
                    "pld        [%2, #256]          \n"
                    "vld1.32   	{d20-d23}, [%2:128]	\n"	//outptr1_s32
                    "vmovl.s16	q4, d16				\n"
                    "vmovl.s16	q5, d17				\n"
                    "vadd.s32	q10, q4				\n"
                    "vadd.s32	q11, q5				\n"
                    "vst1.32   	{d20-d23}, [%2:128]!\n"	
                    //############################################
                    //load inch kernel_2 k0-k7
                    "vdup.s8	d8, d24[0]			\n"			
                    "vdup.s8	d9, d24[1]			\n"	
                    "vdup.s8	d10, d24[2]			\n"	
                    "vdup.s8	d11, d24[3]			\n"			
                    "vdup.s8	d12, d24[4]			\n"			
                    "vdup.s8	d13, d24[5]			\n"	
                    "vdup.s8	d14, d24[6]			\n"	
                    "vdup.s8	d15, d24[7]			\n"											
                    
                    //mla
                    "vmull.s8	q8, d0,	d8			\n"
                    "vmlal.s8	q8, d1,	d9			\n"
                    "vmlal.s8	q8, d2,	d10			\n"
                    "vmlal.s8	q8, d3,	d11			\n"		
                    "vmlal.s8	q8, d4,	d12			\n"
                    "vmlal.s8	q8, d5,	d13			\n"		
                    "vmlal.s8	q8, d6,	d14			\n"
                    "vmlal.s8	q8, d7,	d15			\n"					

                    //outptr2_s32
                    "pld        [%3, #256]          \n"
                    "vld1.32   	{d20-d23}, [%3:128] \n"	//outptr2_s32
                    "vmovl.s16	q4, d16				\n"
                    "vmovl.s16	q5, d17				\n"
                    "vadd.s32	q10, q4				\n"
                    "vadd.s32	q11, q5				\n"
                    "vst1.32   	{d20-d23}, [%3:128]!\n"							
                    //#############################################
                    //load inch kernel_3 k0-k7
                    "vdup.s8	d8, d25[0]			\n"			
                    "vdup.s8	d9, d25[1]			\n"	
                    "vdup.s8	d10, d25[2]			\n"	
                    "vdup.s8	d11, d25[3]			\n"	
                    "vdup.s8	d12, d25[4]			\n"			
                    "vdup.s8	d13, d25[5]			\n"	
                    "vdup.s8	d14, d25[6]			\n"	
                    "vdup.s8	d15, d25[7]			\n"													
                    
                    //mla
                    "vmull.s8	q8, d0,	d8			\n"
                    "vmlal.s8	q8, d1,	d9			\n"
                    "vmlal.s8	q8, d2,	d10			\n"
                    "vmlal.s8	q8, d3,	d11			\n"		
                    "vmlal.s8	q8, d4,	d12			\n"
                    "vmlal.s8	q8, d5,	d13			\n"		
                    "vmlal.s8	q8, d6,	d14			\n"
                    "vmlal.s8	q8, d7,	d15			\n"				

                    //outptr3_s32
                    "pld        [%4, #256]          \n"
                    "vld1.32   	{d20-d23}, [%4:128] \n"	//outptr3_s32
                    "vmovl.s16	q4, d16				\n"
                    "vmovl.s16	q5, d17				\n"
                    "vadd.s32	q10, q4				\n"
                    "vadd.s32	q11, q5				\n"
                    "vst1.32   	{d20-d23}, [%4:128]!\n"						
                    
                    //next
                    "subs       %0, #1              \n"					
                    "bne        0b                  \n"
                    : "=r"(nn),     	 // %0
                      "=r"(outptr0_s32), // %1
                      "=r"(outptr1_s32), // %2
                      "=r"(outptr2_s32), // %3
                      "=r"(outptr3_s32), // %4
                      "=r"(r0),     	 // %5
                      "=r"(r1),     	 // %6
                      "=r"(r2),     	 // %7
                      "=r"(r3),     	 // %8
                      "=r"(r4),	  	 // %9
                      "=r"(r5),	  	 // %10
                      "=r"(r6),	  	 // %11
                      "=r"(r7)	  		 // %12					  
                    : "0"(nn),
                      "1"(outptr0_s32),
                      "2"(outptr1_s32),
                      "3"(outptr2_s32),
                      "4"(outptr3_s32),
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

            for (; remain>0; remain--)
            {
                //ToDo Neon;
                int sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                int sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                int sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                int sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];																													

                *outptr0_s32 += sum0;
                *outptr1_s32 += sum1;
                *outptr2_s32 += sum2;
                *outptr3_s32 += sum3;								

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0_s32++;
                outptr1_s32++;
                outptr2_s32++;
                outptr3_s32++;
            }	
        }
        
        for (; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            
            int* outptr0_s32 = (int*)outptr0;
            int* outptr1_s32 = (int*)outptr1;
            int* outptr2_s32 = (int*)outptr2;
            int* outptr3_s32 = (int*)outptr3;

            const float* img0 = bottom_blob.channel(q);
            const signed char* img0_s8 = (signed char *)img0;

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
                    "vld1.s8    {d8}, [%5 :64]!  	\n"

                    //mla
                    "vmull.s8   q5, d8, %12         \n"
                    //outptr0_s32
                    "pld		[%1, #256]			\n"
                    "vld1.32	{d12-d15}, [%1]     \n"
                    "vmovl.s16	q8, d10				\n"
                    "vmovl.s16	q9, d11				\n"
                    "vadd.s32	q6, q8				\n"
                    "vadd.s32	q7, q9				\n"
                    "vst1.32   	{d12-d15}, [%1]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %13         \n"
                    //outptr1_s32
                    "pld		[%2, #256]			\n"
                    "vld1.32	{d12-d15}, [%2]     \n"
                    "vmovl.s16	q8, d10				\n"
                    "vmovl.s16	q9, d11				\n"
                    "vadd.s32	q6, q8				\n"
                    "vadd.s32	q7, q9				\n"
                    "vst1.32   	{d12-d15}, [%2]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %14         \n"
                    //outptr0_s32
                    "pld		[%3, #256]			\n"
                    "vld1.32	{d12-d15}, [%3]     \n"
                    "vmovl.s16	q8, d10				\n"
                    "vmovl.s16	q9, d11				\n"
                    "vadd.s32	q6, q8				\n"
                    "vadd.s32	q7, q9				\n"
                    "vst1.32   	{d12-d15}, [%3]!    \n"

                    //mla
                    "vmull.s8   q5, d8, %15         \n"
                    //outptr0_s32
                    "pld		[%4, #256]			\n"
                    "vld1.32	{d12-d15}, [%4]     \n"
                    "vmovl.s16	q8, d10				\n"
                    "vmovl.s16	q9, d11				\n"
                    "vadd.s32	q6, q8				\n"
                    "vadd.s32	q7, q9				\n"
                    "vst1.32   	{d12-d15}, [%4]!    \n"					

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     		// %0
                      "=r"(outptr0_s32), 	// %1
                      "=r"(outptr1_s32), 	// %2
                      "=r"(outptr2_s32), 	// %3
                      "=r"(outptr3_s32), 	// %4
                      "=r"(r0)      		// %5
                    : "0"(nn),
                      "1"(outptr0_s32),
                      "2"(outptr1_s32),
                      "3"(outptr2_s32),
                      "4"(outptr3_s32),
                      "5"(r0),
                      "w"(_k0),      		// %12
                      "w"(_k1),      		// %13
                      "w"(_k2),      		// %14
                      "w"(_k3)      		// %15
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9"
                );
            }
            
            for (; remain>0; remain--)
            {
                // TODO neon optimize
                int sum0 = *r0 * k0;
                int sum1 = *r0 * k1;
                int sum2 = *r0 * k2;
                int sum3 = *r0 * k3;

                *outptr0_s32 += sum0;
                *outptr1_s32 += sum1;
                *outptr2_s32 += sum2;
                *outptr3_s32 += sum3;

                r0++;
                outptr0_s32++;
                outptr1_s32++;
                outptr2_s32++;
                outptr3_s32++;
            }
        }
    }

    #pragma omp parallel for
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out0.fill(0.f);

        int q = 0;
        
        for (; q+7<inch; q+=8)
        {
            float* outptr0 = out0;

            int* outptr0_s32 = (int*)outptr0;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);
            const float* img4 = bottom_blob.channel(q+4);
            const float* img5 = bottom_blob.channel(q+5);
            const float* img6 = bottom_blob.channel(q+6);
            const float* img7 = bottom_blob.channel(q+7);			

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* k0 = kernel0;

            const signed char* r0 = (signed char *)img0;
            const signed char* r1 = (signed char *)img1;
            const signed char* r2 = (signed char *)img2;
            const signed char* r3 = (signed char *)img3;
            const signed char* r4 = (signed char *)img4;
            const signed char* r5 = (signed char *)img5;
            const signed char* r6 = (signed char *)img6;
            const signed char* r7 = (signed char *)img7;			

            int size = outw * outh;
        
            int nn = size >> 3;
            int remain = size & 7;

            //load inch kernel_0 k0-k7
            asm volatile(
                "vld1.s8	d18, [%0]			\n"
                "vdup.s8	 d8, d18[0]			\n"			
                "vdup.s8	 d9, d18[1]			\n"	
                "vdup.s8	d10, d18[2]			\n"	
                "vdup.s8	d11, d18[3]			\n"			
                "vdup.s8	d12, d18[4]			\n"			
                "vdup.s8	d13, d18[5]			\n"	
                "vdup.s8	d14, d18[6]			\n"	
                "vdup.s8	d15, d18[7]			\n"	
                : "=r"(k0)
                : "0"(k0)
                : 
            );

            if (nn > 0)
            {	
                asm volatile(
                    "0:  						   \n"
                    //ld r0-r7
                    "pld        [%2, #64]          \n"
                    "vld1.s8	{d0}, [%2 :64]!    \n"	//r0
                    "pld        [%3, #64]          \n"
                    "vld1.s8	{d1}, [%3 :64]!    \n"  //r1
                    "pld        [%4, #64]          \n"
                    "vld1.s8	{d2}, [%4 :64]!    \n"  //r2
                    "pld        [%5, #64]          \n"
                    "vld1.s8	{d3}, [%5 :64]!    \n"  //r3
                    "pld        [%6, #64]          \n"
                    "vld1.s8	{d4}, [%6 :64]!    \n"  //r4
                    "pld        [%7, #64]          \n"
                    "vld1.s8	{d5}, [%7 :64]!    \n"	//r5
                    "pld        [%8, #64]          \n"
                    "vld1.s8	{d6}, [%8 :64]!    \n"	//r6
                    "pld        [%9, #64]          \n"
                    "vld1.s8	{d7}, [%9 :64]!    \n"	//r7												
                    
                    //mla
                    "vmull.s8	q8, d0,	d8			\n"
                    "vmlal.s8	q8, d1,	d9			\n"
                    "vmlal.s8	q8, d2,	d10			\n"
                    "vmlal.s8	q8, d3,	d11			\n"		
                    "vmlal.s8	q8, d4,	d12			\n"
                    "vmlal.s8	q8, d5,	d13			\n"		
                    "vmlal.s8	q8, d6,	d14			\n"
                    "vmlal.s8	q8, d7,	d15			\n"	

                    //outptr0_s32
                    "pld        [%1, #256]          \n"
                    "vld1.32   	{d20-d23}, [%1]     \n"	//outptr0_s32
                    "vmovl.s16	q12, d16			\n"
                    "vmovl.s16	q13, d17			\n"
                    "vadd.s32	q10, q12			\n"
                    "vadd.s32	q11, q13			\n"
                    "vst1.32   	{d20-d23}, [%1]!    \n"				
                    
                    //next
                    "subs       %0, #1              \n"					
                    "bne        0b                  \n"
                    : "=r"(nn),     	 // %0
                      "=r"(outptr0_s32), // %1
                      "=r"(r0),     	 // %2
                      "=r"(r1),     	 // %3
                      "=r"(r2),     	 // %4
                      "=r"(r3),     	 // %5
                      "=r"(r4),	  	 	 // %6
                      "=r"(r5),	  	 	 // %7
                      "=r"(r6),	  	 	 // %8
                      "=r"(r7)	  		 // %9					  
                    : "0"(nn),
                      "1"(outptr0_s32),
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
                int sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];																												

                *outptr0_s32 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0_s32++;
            }	
        }
        
        for (; q<inch; q++)
        {
            float* outptr0 = out0;
            
            int* outptr0_s32 = (int*)outptr0;

            const float* img0 = bottom_blob.channel(q);
            const signed char* img0_s8 = (signed char *)img0;
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
                    "vld1.s8    {d8}, [%2 :64]!  	\n"

                    //mla
                    "vmull.s8   q5, d8, %6          \n"
                    //outptr0_s32
                    "pld		[%1, #256]			\n"
                    "vld1.32	{d12-d15}, [%1]     \n"
                    "vmovl.s16	q8, d10				\n"
                    "vmovl.s16	q9, d11				\n"
                    "vadd.s32	q6, q8				\n"
                    "vadd.s32	q7, q9				\n"
                    "vst1.32   	{d12-d15}, [%1]!    \n"				

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     		// %0
                      "=r"(outptr0_s32), 	// %1
                      "=r"(r0)      		// %2
                    : "0"(nn),
                      "1"(outptr0_s32),
                      "2"(r0),
                      "w"(_k0)      		// %6
                    : "cc", "memory", "q4", "q5", "q7", "q8", "q9"
                );
            }
            
            for (; remain>0; remain--)
            {
                int sum0 = *r0 * k0;

                *outptr0_s32 += sum0;

                r0++;
                outptr0_s32++;
            }
        }
    }	
}

static void conv1x1s1_neon_s8_inter(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, \
                                    const stQuantizeParams& scale)
{
    Mat bottom_blob_s8;

    bottom_blob_s8.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, 1);

    //Quantize Float32 to Int8
    conv_quantize_neon(bottom_blob, bottom_blob_s8, scale);

    //Convolution with Int8
    conv1x1s1_int8_neon(bottom_blob_s8, top_blob, _kernel, _bias, scale);

    //Dequantize Int8 to Float32
    conv_dequantize_neon(top_blob, _bias, scale);

    return;
}

static void conv1x1s2_neon_s8_inter(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, \
                                    const stQuantizeParams& scale)
{
    Mat bottom_blob_s8;

    bottom_blob_s8.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, 1);

    //Quantize Float32 to Int8
    conv_quantize_neon(bottom_blob, bottom_blob_s8, scale);

    //Convolution with Int8
    conv1x1s1_int8_neon(bottom_blob_s8, top_blob, _kernel, _bias, scale);

    //Dequantize Int8 to Float32
    conv_dequantize_neon(top_blob, _bias, scale);

    return;
}
