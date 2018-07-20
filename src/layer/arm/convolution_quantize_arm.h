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

static void conv_quantize_neon(const Mat &bottom_blob, Mat &bottom_blob_s8, const float dataScale)
{
    float ufDataFactor = dataScale;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int size = w * h;
    int FPSCR_value = 0;

#if NCNN_INT8_INFO
    fprintf(stderr, "scale %f\n", dataScale);
#endif      

    asm volatile(   
        "VMRS    r10, FPSCR              \n"
        "MOV     %0,  r10                \n"
        "BIC     r10, r10,#0x00c00000    \n"
        "VMSR FPSCR,r10                  \n"
        : "=r"(FPSCR_value)
        : "0"(FPSCR_value)
        : "cc", "r10"
    );

    size = w*h;

    #pragma omp parallel for
    for (int qidx=0; qidx<inch; qidx++)
    {
        const float* img0 = bottom_blob.channel(qidx);
        signed char* img0_s8 = bottom_blob_s8.channel(qidx);

        int nn = size >> 3;
        int remain = size & 7;

        if(nn > 0)
        {
            asm volatile(
                "PLD        [%1, #256]          \n"
                "VLD1.F32   {D0-D3}, [%1]!      \n"
                "VDUP.32    Q10, %3             \n"

                "0:                             \n"
                "VMUL.F32   Q0,Q0,Q10           \n"
                "VMUL.F32   Q1,Q1,Q10           \n"

                "VCVTR.S32.F32 S0,S0            \n"
                "VCVTR.S32.F32 S1,S1            \n"
                "VCVTR.S32.F32 S2,S2            \n"
                "VCVTR.S32.F32 S3,S3            \n"
                "VCVTR.S32.F32 S4,S4            \n"
                "VCVTR.S32.F32 S5,S5            \n"
                "VCVTR.S32.F32 S6,S6            \n"
                "VCVTR.S32.F32 S7,S7            \n"

                "VQMOVN.S32 D4,Q0               \n"
                "VQMOVN.S32 D5,Q1               \n"
                            
                "PLD        [%1, #256]          \n"
                "VLD1.F32   {D0-D3}, [%1]!      \n"
                            
                "VQMOVN.S16 D4,Q2               \n"
                "VST1.8     {D4}, [%2]!         \n"
                
                "SUBS       %0, #1              \n"
                "BNE        0b                  \n"
                
                "SUB        %1, #32             \n"
                : "=r"(nn),                     // %0
                  "=r"(img0),                   // %1
                  "=r"(img0_s8),                // %2
                  "=r"(ufDataFactor)            // %3
                : "0"(nn),
                  "1"(img0),
                  "2"(img0_s8),
                  "3"(ufDataFactor)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11"
            );
        }


        if(remain > 0)
        {
            asm volatile(   
                "VLD1.F32   {D0[0]}, [%1]!      \n"
                "VDUP.32    Q10, %3             \n"
                            
                "0:                             \n"
                "VMUL.F32   Q0,Q0,Q10           \n"
                "VCVTR.S32.F32 S0,S0            \n"
                
                "VQMOVN.S32 D4,Q0               \n" 
                "VLD1.F32   {D0[0]}, [%1]!      \n"
                  
                "VQMOVN.S16 D4,Q2               \n"
                "VST1.8     {D4[0]}, [%2]!      \n"
                
                "SUBS       %0, #1              \n"
                "BNE        0b                  \n"
                : "=r"(remain),                 // %0
                  "=r"(img0),                   // %1
                  "=r"(img0_s8),                // %2
                  "=r"(ufDataFactor)            // %3
                : "0"(remain),
                  "1"(img0),
                  "2"(img0_s8),
                  "3"(ufDataFactor)
                : "cc", "memory", "q0", "q1", "q2", "q10"
            );
        }
    }

    //ncnn_comm_print_blob(bottom_blob, PRINT_BLOB_TYPE_S16);
    asm volatile(   
        "MOV   r10,   %0              \n"
        "VMSR  FPSCR, r10             \n"
        : "=r"(FPSCR_value)
        : "0"(FPSCR_value)
        : "cc", "r10"
    );
}

static void conv_dequantize_neon(Mat &top_blob, const Mat &_bias, const float dataScale, const float weightScale)
{
    float ufReverseFactor = 0.f;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    int size = outh * outw;

    const float *bias = _bias;

    if (0 != dataScale * weightScale)
    {
        ufReverseFactor = 1 / (dataScale * weightScale);
    }

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        const float* img0 = top_blob.channel(p);
        int* img0_s32 = (int*)img0;
        float* img0_f32 = (float*)img0;
        
        float bias0 = bias ? bias[p] : 0.f;

        int nn = size >> 3;
        int remain = size & 7;  

        if(nn > 0)
        {
            asm volatile(   
                "PLD        [%1, #256]          \n"
                "VLD1.S32   {D0-D3}, [%1]!      \n" //Q0-Q1 data
                "VDUP.F32   Q10, %3             \n" //Q10 scale
                "VDUP.F32   Q12, %4             \n" //Q12 bias
                
                "0:                             \n"
                "VCVTR.F32.S32 Q0,Q0            \n"
                "VCVTR.F32.S32 Q1,Q1            \n"

                "VMUL.F32   Q0,Q0,Q10           \n"
                "VMUL.F32   Q1,Q1,Q10           \n"

                "VADD.F32   Q2,Q0,Q12           \n"
                "VADD.F32   Q3,Q1,Q12           \n"
                            
                "PLD        [%1, #256]          \n"
                "VLD1.S32   {D0-D3}, [%1]!      \n"
                "VST1.F32   {D4-D7}, [%2]!      \n"
                            
                "SUBS       %0, #1              \n"
                "BNE        0b                  \n"
                        
                "SUB        %1, #32             \n"
                : "=r"(nn),                     // %0
                  "=r"(img0_s32),               // %1
                  "=r"(img0_f32),               // %2
                  "=r"(ufReverseFactor),        // %3
                  "=r"(bias0)                   // %4
                : "0"(nn),
                  "1"(img0_s32),
                  "2"(img0_f32),
                  "3"(ufReverseFactor),
                  "4"(bias0)
                : "cc", "memory", "q0", "q1", "q2", "q4", "q10", "q12"
            );
        }
        
        if(remain > 0)
        {
            asm volatile(   
                "VLD1.F32   {D0[0]}, [%1]!      \n" //D0 data
                "VDUP.32    Q10, %3             \n" //Q10 scale
                "VDUP.32    Q12, %4             \n" //Q12 bias
                
                "0:                             \n"
                "VCVTR.F32.S32 S0,S0            \n"
                "VMUL.F32   Q0,Q0,Q10           \n"
                "VADD.F32   Q2,Q0,Q12           \n"
                
                //store
                "VLD1.F32   {D0[0]}, [%1]!      \n"
                "VST1.F32   {D4[0]}, [%2]!      \n"
                
                "SUBS       %0, #1              \n"
                "BNE        0b                  \n"
                : "=r"(remain),                 // %0
                  "=r"(img0_s32),               // %1
                  "=r"(img0_f32),               // %2
                  "=r"(ufReverseFactor),        // %3
                  "=r"(bias0)                   // %4                             
                : "0"(remain),
                  "1"(img0_s32),
                  "2"(img0_f32),
                  "3"(ufReverseFactor),
                  "4"(bias0)                          
                : "cc", "memory", "q0", "q1", "q2", "q4", "q10", "q12"
            );
        }   
    }
}
