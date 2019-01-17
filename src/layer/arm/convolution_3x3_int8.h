// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(4*9, inch, outch/4 + outch%4, (size_t)1u);

    const signed char* kernel = _kernel;

    int p=0;
    for (; p+3<outch; p+=4)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;
        const signed char* k1 = kernel + (p+1)*inch*9;
        const signed char* k2 = kernel + (p+2)*inch*9;
        const signed char* k3 = kernel + (p+3)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/4);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp += 4;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
        }
    }
    for (; p<outch; p++)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/4 + p%4);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

static void conv3x3s2_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*9, inch, outch/8 + outch%8, (size_t)1u);

    const signed char* kernel = _kernel;

    int p=0;
    for (; p+7<outch; p+=8)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;
        const signed char* k1 = kernel + (p+1)*inch*9;
        const signed char* k2 = kernel + (p+2)*inch*9;
        const signed char* k3 = kernel + (p+3)*inch*9;
        const signed char* k4 = kernel + (p+4)*inch*9;
        const signed char* k5 = kernel + (p+5)*inch*9;
        const signed char* k6 = kernel + (p+6)*inch*9;
        const signed char* k7 = kernel + (p+7)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p<outch; p++)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

#if __aarch64__
static void conv3x3s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
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

            int i = 0;

            int8x8_t _k00 = vdup_n_s8(kernel0[0]);
            int8x8_t _k01 = vdup_n_s8(kernel0[1]);
            int8x8_t _k02 = vdup_n_s8(kernel0[2]);
            int8x8_t _k03 = vdup_n_s8(kernel0[3]);
            int8x8_t _k04 = vdup_n_s8(kernel0[4]);
            int8x8_t _k05 = vdup_n_s8(kernel0[5]);
            int8x8_t _k06 = vdup_n_s8(kernel0[6]);
            int8x8_t _k07 = vdup_n_s8(kernel0[7]);
            int8x8_t _k08 = vdup_n_s8(kernel0[8]);

            int8x8_t _k10 = vdup_n_s8(kernel1[0]);
            int8x8_t _k11 = vdup_n_s8(kernel1[1]);
            int8x8_t _k12 = vdup_n_s8(kernel1[2]);
            int8x8_t _k13 = vdup_n_s8(kernel1[3]);
            int8x8_t _k14 = vdup_n_s8(kernel1[4]);
            int8x8_t _k15 = vdup_n_s8(kernel1[5]);
            int8x8_t _k16 = vdup_n_s8(kernel1[6]);
            int8x8_t _k17 = vdup_n_s8(kernel1[7]);
            int8x8_t _k18 = vdup_n_s8(kernel1[8]); 

            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                for (; nn > 0; nn--)
                {
                    // outch 0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);

                    int16x8_t _sum0 = vmull_s8(_r0, _k00);
                    _sum0 = vmlal_s8(_sum0, _r01, _k01);
                    _sum0 = vmlal_s8(_sum0, _r02, _k02);

                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    _sum0 = vmlal_s8(_sum0, _r1, _k03);
                    _sum0 = vmlal_s8(_sum0, _r11, _k04);
                    _sum0 = vmlal_s8(_sum0, _r12, _k05);

                    int16x8_t _sum1 = vmull_s8(_r1, _k00);
                    _sum1 = vmlal_s8(_sum1, _r11, _k01);
                    _sum1 = vmlal_s8(_sum1, _r12, _k02);

                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    _sum0 = vmlal_s8(_sum0, _r2, _k06);
                    _sum0 = vmlal_s8(_sum0, _r21, _k07);
                    _sum0 = vmlal_s8(_sum0, _r22, _k08);

                    _sum1 = vmlal_s8(_sum1, _r2, _k03);
                    _sum1 = vmlal_s8(_sum1, _r21, _k04);
                    _sum1 = vmlal_s8(_sum1, _r22, _k05);

                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    _sum1 = vmlal_s8(_sum1, _r3, _k06);
                    _sum1 = vmlal_s8(_sum1, _r31, _k07);
                    _sum1 = vmlal_s8(_sum1, _r32, _k08);

                    int32x4_t sum0_s32 = vld1q_s32(outptr0);
                    int32x4_t sum0n_s32 = vld1q_s32(outptr0+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum0));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum0));

                    vst1q_s32(outptr0, sum0_s32);
                    vst1q_s32(outptr0+4, sum0n_s32);

                    int32x4_t sum1_s32 = vld1q_s32(outptr0n);
                    int32x4_t sum1n_s32 = vld1q_s32(outptr0n+4);

                    sum1_s32 = vaddw_s16(sum1_s32, vget_low_s16(_sum1));
                    sum1n_s32 = vaddw_s16(sum1n_s32, vget_high_s16(_sum1));

                    vst1q_s32(outptr0n, sum1_s32);
                    vst1q_s32(outptr0n+4, sum1n_s32);

                    // outch 1
                    _sum0 = vmull_s8(_r0, _k10);
                    _sum0 = vmlal_s8(_sum0, _r01, _k11);
                    _sum0 = vmlal_s8(_sum0, _r02, _k12);

                    _sum0 = vmlal_s8(_sum0, _r1, _k13);
                    _sum0 = vmlal_s8(_sum0, _r11, _k14);
                    _sum0 = vmlal_s8(_sum0, _r12, _k15);

                    _sum0 = vmlal_s8(_sum0, _r2, _k16);
                    _sum0 = vmlal_s8(_sum0, _r21, _k17);
                    _sum0 = vmlal_s8(_sum0, _r22, _k18);

                    _sum1 = vmull_s8(_r1, _k10);
                    _sum1 = vmlal_s8(_sum1, _r11, _k11);
                    _sum1 = vmlal_s8(_sum1, _r12, _k12);

                    _sum1 = vmlal_s8(_sum1, _r2, _k13);
                    _sum1 = vmlal_s8(_sum1, _r21, _k14);
                    _sum1 = vmlal_s8(_sum1, _r22, _k15);

                    _sum1 = vmlal_s8(_sum1, _r3, _k16);
                    _sum1 = vmlal_s8(_sum1, _r31, _k17);
                    _sum1 = vmlal_s8(_sum1, _r32, _k18);

                    sum0_s32 = vld1q_s32(outptr1);
                    sum0n_s32 = vld1q_s32(outptr1+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum0));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum0));

                    vst1q_s32(outptr1, sum0_s32);
                    vst1q_s32(outptr1+4, sum0n_s32);

                    sum1_s32 = vld1q_s32(outptr1n);
                    sum1n_s32 = vld1q_s32(outptr1n+4);

                    sum1_s32 = vaddw_s16(sum1_s32, vget_low_s16(_sum1));
                    sum1n_s32 = vaddw_s16(sum1n_s32, vget_high_s16(_sum1));

                    vst1q_s32(outptr1n, sum1_s32);
                    vst1q_s32(outptr1n+4, sum1n_s32);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr0n += 8;
                    outptr1n += 8;
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

                for (; nn > 0; nn--)
                {
                    // outch 0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);

                    int16x8_t _sum0 = vmull_s8(_r0, _k00);
                    _sum0 = vmlal_s8(_sum0, _r01, _k01);
                    _sum0 = vmlal_s8(_sum0, _r02, _k02);

                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    _sum0 = vmlal_s8(_sum0, _r1, _k03);
                    _sum0 = vmlal_s8(_sum0, _r11, _k04);
                    _sum0 = vmlal_s8(_sum0, _r12, _k05);

                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    _sum0 = vmlal_s8(_sum0, _r2, _k06);
                    _sum0 = vmlal_s8(_sum0, _r21, _k07);
                    _sum0 = vmlal_s8(_sum0, _r22, _k08);

                    int32x4_t sum0_s32 = vld1q_s32(outptr0);
                    int32x4_t sum0n_s32 = vld1q_s32(outptr0+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum0));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum0));

                    vst1q_s32(outptr0, sum0_s32);
                    vst1q_s32(outptr0+4, sum0n_s32);

                    // outch 1
                    _sum0 = vmull_s8(_r0, _k10);
                    _sum0 = vmlal_s8(_sum0, _r01, _k11);
                    _sum0 = vmlal_s8(_sum0, _r02, _k12);

                    _sum0 = vmlal_s8(_sum0, _r1, _k13);
                    _sum0 = vmlal_s8(_sum0, _r11, _k14);
                    _sum0 = vmlal_s8(_sum0, _r12, _k15);

                    _sum0 = vmlal_s8(_sum0, _r2, _k16);
                    _sum0 = vmlal_s8(_sum0, _r21, _k17);
                    _sum0 = vmlal_s8(_sum0, _r22, _k18);

                    sum0_s32 = vld1q_s32(outptr1);
                    sum0n_s32 = vld1q_s32(outptr1+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum0));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum0));

                    vst1q_s32(outptr1, sum0_s32);
                    vst1q_s32(outptr1+4, sum0n_s32);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
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

            int i = 0;

            int8x8_t _k00 = vdup_n_s8(kernel0[0]);
            int8x8_t _k01 = vdup_n_s8(kernel0[1]);
            int8x8_t _k02 = vdup_n_s8(kernel0[2]);
            int8x8_t _k03 = vdup_n_s8(kernel0[3]);
            int8x8_t _k04 = vdup_n_s8(kernel0[4]);
            int8x8_t _k05 = vdup_n_s8(kernel0[5]);
            int8x8_t _k06 = vdup_n_s8(kernel0[6]);
            int8x8_t _k07 = vdup_n_s8(kernel0[7]);
            int8x8_t _k08 = vdup_n_s8(kernel0[8]);

            for (; i+1 < outh; i+=2)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                for (; nn > 0; nn--)
                {
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);

                    int16x8_t _sum0 = vmull_s8(_r0, _k00);
                    _sum0 = vmlal_s8(_sum0, _r01, _k01);
                    _sum0 = vmlal_s8(_sum0, _r02, _k02);

                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    _sum0 = vmlal_s8(_sum0, _r1, _k03);
                    _sum0 = vmlal_s8(_sum0, _r11, _k04);
                    _sum0 = vmlal_s8(_sum0, _r12, _k05);

                    int16x8_t _sum1 = vmull_s8(_r1, _k00);
                    _sum1 = vmlal_s8(_sum1, _r11, _k01);
                    _sum1 = vmlal_s8(_sum1, _r12, _k02);

                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    _sum0 = vmlal_s8(_sum0, _r2, _k06);
                    _sum0 = vmlal_s8(_sum0, _r21, _k07);
                    _sum0 = vmlal_s8(_sum0, _r22, _k08);

                    _sum1 = vmlal_s8(_sum1, _r2, _k03);
                    _sum1 = vmlal_s8(_sum1, _r21, _k04);
                    _sum1 = vmlal_s8(_sum1, _r22, _k05);

                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    _sum1 = vmlal_s8(_sum1, _r3, _k06);
                    _sum1 = vmlal_s8(_sum1, _r31, _k07);
                    _sum1 = vmlal_s8(_sum1, _r32, _k08);

                    int32x4_t sum0_s32 = vld1q_s32(outptr0);
                    int32x4_t sum0n_s32 = vld1q_s32(outptr0+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum0));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum0)); 

                    vst1q_s32(outptr0, sum0_s32);
                    vst1q_s32(outptr0+4, sum0n_s32);

                    int32x4_t sum1_s32 = vld1q_s32(outptr0n);
                    int32x4_t sum1n_s32 = vld1q_s32(outptr0n+4);

                    sum1_s32 = vaddw_s16(sum1_s32, vget_low_s16(_sum1));
                    sum1n_s32 = vaddw_s16(sum1n_s32, vget_high_s16(_sum1));

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
                    // Todo neon
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

                for (; nn > 0; nn--)
                {
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);

                    int16x8_t _sum0 = vmull_s8(_r0, _k00);
                    _sum0 = vmlal_s8(_sum0, _r01, _k01);
                    _sum0 = vmlal_s8(_sum0, _r02, _k02);

                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    _sum0 = vmlal_s8(_sum0, _r1, _k03);
                    _sum0 = vmlal_s8(_sum0, _r11, _k04);
                    _sum0 = vmlal_s8(_sum0, _r12, _k05);

                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    _sum0 = vmlal_s8(_sum0, _r2, _k06);
                    _sum0 = vmlal_s8(_sum0, _r21, _k07);
                    _sum0 = vmlal_s8(_sum0, _r22, _k08);

                    int32x4_t sum0_s32 = vld1q_s32(outptr0);
                    int32x4_t sum0n_s32 = vld1q_s32(outptr0+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum0));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum0));

                    vst1q_s32(outptr0, sum0_s32);
                    vst1q_s32(outptr0+4, sum0n_s32);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 8;
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

static void conv3x3s2_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char* kernel = _kernel;
    
    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2; 

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);       
        
        out0.fill(0.f);
        out1.fill(0.f);
        out2.fill(0.f);
        out3.fill(0.f);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;
        const signed char* kernel1 = (const signed char*)kernel + (p + 1) * inch * 9;
        const signed char* kernel2 = (const signed char*)kernel + (p + 2) * inch * 9;
        const signed char* kernel3 = (const signed char*)kernel + (p + 3) * inch * 9;

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3; 

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            int i = 0;

            int8x16_t _k0 = vld1q_s8(kernel0);
            int8x16_t _k1 = vld1q_s8(kernel1);
            int8x16_t _k2 = vld1q_s8(kernel2);
            int8x16_t _k3 = vld1q_s8(kernel3);

            for (; i < outh; i++)
            {
                int nn = outw >> 3;
                int remain = outw & 7;

                if (nn > 0)
                {
                asm volatile(
                    "0:                                \n"
                    // r0
                    "prfm   pldl1keep, [%5, #128]      \n"
                    "ld2    {v4.8b, v5.8b}, [%5], #16  \n"
                    "ld2    {v6.8b, v7.8b}, [%5]       \n"
                    "ext    v8.8b, v4.8b, v6.8b, #1    \n"

                    "dup    v9.8b,  %16.b[0]           \n"
                    "dup    v10.8b, %17.b[0]           \n"
                    "dup    v11.8b, %18.b[0]           \n"
                    "dup    v12.8b, %19.b[0]           \n"

                    "smull  v13.8h, v4.8b, v9.8b       \n"
                    "smull  v14.8h, v4.8b, v10.8b      \n"
                    "smull  v15.8h, v4.8b, v11.8b      \n"
                    "smull  v16.8h, v4.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[1]            \n"
                    "dup    v10.8b, %17.b[1]           \n"
                    "dup    v11.8b, %18.b[1]           \n"
                    "dup    v12.8b, %19.b[1]           \n"

                    "smlal  v13.8h, v5.8b, v9.8b       \n"
                    "smlal  v14.8h, v5.8b, v10.8b      \n"
                    "smlal  v15.8h, v5.8b, v11.8b      \n"
                    "smlal  v16.8h, v5.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[2]            \n"
                    "dup    v10.8b, %17.b[2]           \n"
                    "dup    v11.8b, %18.b[2]           \n"
                    "dup    v12.8b, %19.b[2]           \n"

                    "smlal  v13.8h, v8.8b, v9.8b       \n"
                    "smlal  v14.8h, v8.8b, v10.8b      \n"
                    "smlal  v15.8h, v8.8b, v11.8b      \n"
                    "smlal  v16.8h, v8.8b, v12.8b      \n"
                    // r1
                    "prfm   pldl1keep, [%6, #128]      \n"
                    "ld2    {v4.8b, v5.8b}, [%6], #16  \n"
                    "ld2    {v6.8b, v7.8b}, [%6]       \n"
                    "ext    v8.8b, v4.8b, v6.8b, #1    \n"

                    "dup    v9.8b, %16.b[3]            \n"
                    "dup    v10.8b, %17.b[3]           \n"
                    "dup    v11.8b, %18.b[3]           \n"
                    "dup    v12.8b, %19.b[3]           \n"

                    "smlal  v13.8h, v4.8b, v9.8b       \n"
                    "smlal  v14.8h, v4.8b, v10.8b      \n"
                    "smlal  v15.8h, v4.8b, v11.8b      \n"
                    "smlal  v16.8h, v4.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[4]            \n"
                    "dup    v10.8b, %17.b[4]           \n"
                    "dup    v11.8b, %18.b[4]           \n"
                    "dup    v12.8b, %19.b[4]           \n"

                    "smlal  v13.8h, v5.8b, v9.8b       \n"
                    "smlal  v14.8h, v5.8b, v10.8b      \n"
                    "smlal  v15.8h, v5.8b, v11.8b      \n"
                    "smlal  v16.8h, v5.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[5]            \n"
                    "dup    v10.8b, %17.b[5]           \n"
                    "dup    v11.8b, %18.b[5]           \n"
                    "dup    v12.8b, %19.b[5]           \n"

                    "smlal  v13.8h, v8.8b, v9.8b       \n"
                    "smlal  v14.8h, v8.8b, v10.8b      \n"
                    "smlal  v15.8h, v8.8b, v11.8b      \n"
                    "smlal  v16.8h, v8.8b, v12.8b      \n"
                    // r2
                    "prfm   pldl1keep, [%7, #128]      \n"
                    "ld2    {v4.8b, v5.8b}, [%7], #16  \n"
                    "ld2    {v6.8b, v7.8b}, [%7]       \n"
                    "ext    v8.8b, v4.8b, v6.8b, #1    \n"

                    "dup    v9.8b, %16.b[6]            \n"
                    "dup    v10.8b, %17.b[6]           \n"
                    "dup    v11.8b, %18.b[6]           \n"
                    "dup    v12.8b, %19.b[6]           \n"

                    "smlal  v13.8h, v4.8b, v9.8b       \n"
                    "smlal  v14.8h, v4.8b, v10.8b      \n"
                    "smlal  v15.8h, v4.8b, v11.8b      \n"
                    "smlal  v16.8h, v4.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[7]            \n"
                    "dup    v10.8b, %17.b[7]           \n"
                    "dup    v11.8b, %18.b[7]           \n"
                    "dup    v12.8b, %19.b[7]           \n"

                    "smlal  v13.8h, v5.8b, v9.8b       \n"
                    "smlal  v14.8h, v5.8b, v10.8b      \n"
                    "smlal  v15.8h, v5.8b, v11.8b      \n"
                    "smlal  v16.8h, v5.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[8]            \n"
                    "dup    v10.8b, %17.b[8]           \n"
                    "dup    v11.8b, %18.b[8]           \n"
                    "dup    v12.8b, %19.b[8]           \n"

                    "smlal  v13.8h, v8.8b, v9.8b       \n"
                    "smlal  v14.8h, v8.8b, v10.8b      \n"
                    "smlal  v15.8h, v8.8b, v11.8b      \n"
                    "smlal  v16.8h, v8.8b, v12.8b      \n"
                    // sum0 - sum3
                    "prfm   pldl1keep, [%1, #128]      \n"
                    "prfm   pldl1keep, [%2, #128]      \n"
                    "prfm   pldl1keep, [%3, #128]      \n"
                    "prfm   pldl1keep, [%4, #128]      \n"
                    "ld1    {v17.4s, v18.4s}, [%1]     \n"
                    "ld1    {v19.4s, v20.4s}, [%2]     \n"
                    "ld1    {v21.4s, v22.4s}, [%3]     \n"
                    "ld1    {v23.4s, v24.4s}, [%4]     \n"

                    "saddw  v17.4s, v17.4s, v13.4h     \n"
                    "saddw2  v18.4s, v18.4s, v13.8h    \n"
                    "saddw  v19.4s, v19.4s, v14.4h     \n"
                    "saddw2  v20.4s, v20.4s, v14.8h    \n"
                    "saddw  v21.4s, v21.4s, v15.4h     \n"
                    "saddw2  v22.4s, v22.4s, v15.8h    \n"
                    "saddw  v23.4s, v23.4s, v16.4h     \n"
                    "saddw2  v24.4s, v24.4s, v16.8h    \n"
                    "st1    {v17.4s, v18.4s}, [%1], #32\n"
                    "st1    {v19.4s, v20.4s}, [%2], #32\n"
                    "st1    {v21.4s, v22.4s}, [%3], #32\n"
                    "st1    {v23.4s, v24.4s}, [%4], #32\n"
                    "subs   %w0, %w0, #1               \n"
                    "bne    0b                         \n"
                    : "=r"(nn),         //%0
                      "=r"(outptr0),    //%1
                      "=r"(outptr1),    //%2
                      "=r"(outptr2),    //%3
                      "=r"(outptr3),    //%4
                      "=r"(r0),         //%5
                      "=r"(r1),         //%6
                      "=r"(r2)          //%7
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "w"(_k0),         //%16
                      "w"(_k1),         //%17
                      "w"(_k2),         //%18
                      "w"(_k3)          //%19
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24"
                );
                }

                if (remain >= 4)
                {
                    remain -= 4;

                asm volatile(
                    // r0
                    "prfm   pldl1keep, [%5, #128]      \n"
                    "ld2    {v4.8b, v5.8b}, [%5], #16  \n"
                    "ld2    {v6.8b, v7.8b}, [%5]       \n"
                    "ext    v8.8b, v4.8b, v6.8b, #1    \n"

                    "dup    v9.8b,  %16.b[0]           \n"
                    "dup    v10.8b, %17.b[0]           \n"
                    "dup    v11.8b, %18.b[0]           \n"
                    "dup    v12.8b, %19.b[0]           \n"

                    "smull  v13.8h, v4.8b, v9.8b       \n"
                    "smull  v14.8h, v4.8b, v10.8b      \n"
                    "smull  v15.8h, v4.8b, v11.8b      \n"
                    "smull  v16.8h, v4.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[1]            \n"
                    "dup    v10.8b, %17.b[1]           \n"
                    "dup    v11.8b, %18.b[1]           \n"
                    "dup    v12.8b, %19.b[1]           \n"

                    "smlal  v13.8h, v5.8b, v9.8b       \n"
                    "smlal  v14.8h, v5.8b, v10.8b      \n"
                    "smlal  v15.8h, v5.8b, v11.8b      \n"
                    "smlal  v16.8h, v5.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[2]            \n"
                    "dup    v10.8b, %17.b[2]           \n"
                    "dup    v11.8b, %18.b[2]           \n"
                    "dup    v12.8b, %19.b[2]           \n"

                    "smlal  v13.8h, v8.8b, v9.8b       \n"
                    "smlal  v14.8h, v8.8b, v10.8b      \n"
                    "smlal  v15.8h, v8.8b, v11.8b      \n"
                    "smlal  v16.8h, v8.8b, v12.8b      \n"
                    // r1
                    "prfm   pldl1keep, [%6, #128]      \n"
                    "ld2    {v4.8b, v5.8b}, [%6], #16  \n"
                    "ld2    {v6.8b, v7.8b}, [%6]       \n"
                    "ext    v8.8b, v4.8b, v6.8b, #1    \n"

                    "dup    v9.8b, %16.b[3]            \n"
                    "dup    v10.8b, %17.b[3]           \n"
                    "dup    v11.8b, %18.b[3]           \n"
                    "dup    v12.8b, %19.b[3]           \n"

                    "smlal  v13.8h, v4.8b, v9.8b       \n"
                    "smlal  v14.8h, v4.8b, v10.8b      \n"
                    "smlal  v15.8h, v4.8b, v11.8b      \n"
                    "smlal  v16.8h, v4.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[4]            \n"
                    "dup    v10.8b, %17.b[4]           \n"
                    "dup    v11.8b, %18.b[4]           \n"
                    "dup    v12.8b, %19.b[4]           \n"

                    "smlal  v13.8h, v5.8b, v9.8b       \n"
                    "smlal  v14.8h, v5.8b, v10.8b      \n"
                    "smlal  v15.8h, v5.8b, v11.8b      \n"
                    "smlal  v16.8h, v5.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[5]            \n"
                    "dup    v10.8b, %17.b[5]           \n"
                    "dup    v11.8b, %18.b[5]           \n"
                    "dup    v12.8b, %19.b[5]           \n"

                    "smlal  v13.8h, v8.8b, v9.8b       \n"
                    "smlal  v14.8h, v8.8b, v10.8b      \n"
                    "smlal  v15.8h, v8.8b, v11.8b      \n"
                    "smlal  v16.8h, v8.8b, v12.8b      \n"
                    // r2
                    "prfm   pldl1keep, [%7, #128]      \n"
                    "ld2    {v4.8b, v5.8b}, [%7], #16  \n"
                    "ld2    {v6.8b, v7.8b}, [%7]       \n"
                    "ext    v8.8b, v4.8b, v6.8b, #1    \n"

                    "dup    v9.8b, %16.b[6]            \n"
                    "dup    v10.8b, %17.b[6]           \n"
                    "dup    v11.8b, %18.b[6]           \n"
                    "dup    v12.8b, %19.b[6]           \n"

                    "smlal  v13.8h, v4.8b, v9.8b       \n"
                    "smlal  v14.8h, v4.8b, v10.8b      \n"
                    "smlal  v15.8h, v4.8b, v11.8b      \n"
                    "smlal  v16.8h, v4.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[7]            \n"
                    "dup    v10.8b, %17.b[7]           \n"
                    "dup    v11.8b, %18.b[7]           \n"
                    "dup    v12.8b, %19.b[7]           \n"
 
                    "smlal  v13.8h, v5.8b, v9.8b       \n"
                    "smlal  v14.8h, v5.8b, v10.8b      \n"
                    "smlal  v15.8h, v5.8b, v11.8b      \n"
                    "smlal  v16.8h, v5.8b, v12.8b      \n"

                    "dup    v9.8b, %16.b[8]            \n"
                    "dup    v10.8b, %17.b[8]           \n"
                    "dup    v11.8b, %18.b[8]           \n"
                    "dup    v12.8b, %19.b[8]           \n"

                    "smlal  v13.8h, v8.8b, v9.8b       \n"
                    "smlal  v14.8h, v8.8b, v10.8b      \n"
                    "smlal  v15.8h, v8.8b, v11.8b      \n"
                    "smlal  v16.8h, v8.8b, v12.8b      \n"
                    // sum0 - sum3
                    "prfm   pldl1keep, [%1, #128]      \n"
                    "prfm   pldl1keep, [%2, #128]      \n"
                    "prfm   pldl1keep, [%3, #128]      \n"
                    "prfm   pldl1keep, [%4, #128]      \n"
                    "ld1    {v17.4s}, [%1]             \n"
                    "ld1    {v19.4s}, [%2]             \n"
                    "ld1    {v21.4s}, [%3]             \n"
                    "ld1    {v23.4s}, [%4]             \n"

                    "saddw  v17.4s, v17.4s, v13.4h     \n"
                    "saddw  v19.4s, v19.4s, v14.4h     \n"
                    "saddw  v21.4s, v21.4s, v15.4h     \n"
                    "saddw  v23.4s, v23.4s, v16.4h     \n"

                    "st1    {v17.4s}, [%1], #16        \n"
                    "st1    {v19.4s}, [%2], #16        \n"
                    "st1    {v21.4s}, [%3], #16        \n"
                    "st1    {v23.4s}, [%4], #16        \n"
                    "sub    %5, %5, #8                 \n"
                    "sub    %6, %6, #8                 \n"
                    "sub    %7, %7, #8                 \n"
                    : "=r"(nn),         //%0
                      "=r"(outptr0),    //%1
                      "=r"(outptr1),    //%2
                      "=r"(outptr2),    //%3
                      "=r"(outptr3),    //%4
                      "=r"(r0),         //%5
                      "=r"(r1),         //%6
                      "=r"(r2)          //%7
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "w"(_k0),         //%16
                      "w"(_k1),         //%17
                      "w"(_k2),         //%18
                      "w"(_k3)          //%19
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24"
                );
                }

                for (; remain>0; remain--)
                {
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

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

                    sum2 += (int)r0[0] * kernel2[0];
                    sum2 += (int)r0[1] * kernel2[1];
                    sum2 += (int)r0[2] * kernel2[2];
                    sum2 += (int)r1[0] * kernel2[3];
                    sum2 += (int)r1[1] * kernel2[4];
                    sum2 += (int)r1[2] * kernel2[5];
                    sum2 += (int)r2[0] * kernel2[6];
                    sum2 += (int)r2[1] * kernel2[7];
                    sum2 += (int)r2[2] * kernel2[8];

                    sum3 += (int)r0[0] * kernel3[0];
                    sum3 += (int)r0[1] * kernel3[1];
                    sum3 += (int)r0[2] * kernel3[2];
                    sum3 += (int)r1[0] * kernel3[3];
                    sum3 += (int)r1[1] * kernel3[4];
                    sum3 += (int)r1[2] * kernel3[5];
                    sum3 += (int)r2[0] * kernel3[6];
                    sum3 += (int)r2[1] * kernel3[7];
                    sum3 += (int)r2[2] * kernel3[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }       

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
            kernel1 += 9;
            kernel2 += 9;
            kernel3 += 9;
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

            int i = 0;

            int8x8_t _k0 = vdup_n_s8(kernel0[0]);
            int8x8_t _k1 = vdup_n_s8(kernel0[1]);
            int8x8_t _k2 = vdup_n_s8(kernel0[2]);
            int8x8_t _k3 = vdup_n_s8(kernel0[3]);
            int8x8_t _k4 = vdup_n_s8(kernel0[4]);
            int8x8_t _k5 = vdup_n_s8(kernel0[5]);
            int8x8_t _k6 = vdup_n_s8(kernel0[6]);
            int8x8_t _k7 = vdup_n_s8(kernel0[7]);
            int8x8_t _k8 = vdup_n_s8(kernel0[8]);

            for (; i < outh; i++)
            {  
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn >0; nn--)
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

                    int32x4_t sum0_s32 = vld1q_s32(outptr0);
                    int32x4_t sum0n_s32 = vld1q_s32(outptr0+4);

                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum));
                    sum0n_s32 = vaddw_s16(sum0n_s32, vget_high_s16(_sum));

                    vst1q_s32(outptr0, sum0_s32);
                    vst1q_s32(outptr0+4, sum0n_s32);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr0 += 8;
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;

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

                    int32x4_t sum0_s32 = vld1q_s32(outptr0);
                    sum0_s32 = vaddw_s16(sum0_s32, vget_low_s16(_sum));
                    vst1q_s32(outptr0, sum0_s32);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 4;
                }                
#endif
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
#else // __aarch64__
static void conv3x3s1_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
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

static void conv3x3s2_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
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
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q13", "q14", "q15"
                    );
                }           

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
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
                        
                        "sub        %3, #8              \n"
                        "sub        %4, #8              \n"
                        "sub        %5, #8              \n"
                        
                        "vdup.s8    d26, d22[6]         \n"
                        "vdup.s8    d27, d22[7]         \n"
                        "vdup.s8    d28, d23[0]         \n"
                        "vmlal.s8   q2, d10, d26        \n" // k06
                        "vmlal.s8   q2, d11, d27        \n" // k07
                        "vmlal.s8   q2, d13, d28        \n" // k08

                        "pld        [%1, #128]          \n"
                        "vld1.32    {d14-d15}, [%1]     \n" //sum0
                        "vaddw.s16   q7, q7, d4         \n"
                        "vst1.32    {d14-d15}, [%1]!    \n"
                        
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

                        "pld        [%2, #128]          \n"
                        "vld1.32    {d14-d15}, [%2]     \n" //sum1
                        "vaddw.s16   q7, q7, d4         \n"
                        "vst1.32    {d14-d15}, [%2]!    \n"
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
                
                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
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

                        "sub        %2, #8              \n"
                        "sub        %3, #8              \n"
                        "sub        %4, #8              \n"                         
                        
                        "vdup.s8    d26, d22[6]         \n"
                        "vdup.s8    d27, d22[7]         \n"
                        "vdup.s8    d28, d23[0]         \n"
                        "vmlal.s8   q2, d10, d26        \n" // k06
                        "vmlal.s8   q2, d11, d27        \n" // k07
                        "vmlal.s8   q2, d13, d28        \n" // k08

                        "pld        [%1, #128]          \n"
                        "vld1.32    {d14-d15}, [%1]     \n" //sum0
                        "vaddw.s16   q7, q7, d4         \n"
                        "vst1.32    {d14-d15}, [%1]!    \n"
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

static void conv3x3s1_packed_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p+0);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);

        const signed char* ktmp = _kernel.channel(p/4);

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char *img0 = bottom_blob.channel(q);

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;

            int i = 0;    

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                if (nn > 0)
                {
                asm volatile(
                    "0:                         \n"
                    "vld1.s8    {d0-d3}, [%8]!  \n"// d0=(k00-k30 k01-k31) d1=(k02-k32 k03-k33) d2=(k04-k34 k05-k35) d3=(k06-k36 k07-k37)
                    // r0
                    "pld        [%5, #128]      \n"
                    "vld1.s8    {d8-d9}, [%5]   \n"// d8=r00-r07 d9=r08-r015 q4
                    "add        %5, #8          \n"
                    "pld        [%1, #128]      \n"
                    "vld1.s32   {d12-d15}, [%1] \n"// sum00-sum07 q6 q7
                    "pld        [%2, #128]      \n"
                    "vld1.s32   {d16-d19}, [%2] \n"// sum10-sum17 q8 q9
                    "pld        [%3, #128]      \n"
                    "vld1.s32   {d20-d23}, [%3] \n"// sum20-sum27 q10 q11
                    "pld        [%4, #128]      \n"
                    "vld1.s32   {d24-d27}, [%4] \n"// sum30-sum37 q12 q13
                    
                    "vmovl.s8   q3, d3          \n"// d6(k06-k36) d7(k07-k37) 
                    "vmovl.s8   q2, d2          \n"// d4(k04-k34) d5(k05-k35)
                    "vmovl.s8   q1, d1          \n"// d2(k02-k32) d3(k03-k33)
                    "vmovl.s8   q0, d0          \n"// d0(k00-k30) d1(k01-k31)
                    "vmovl.s8   q5, d8          \n"// d10(r00-r03) d11(r04-r07)
                    
                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-07) += (r00-r07) * k00
                    "vmlal.s16  q7, d11, d0[0]  \n"
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-17) += (r00-r07) * k10
                    "vmlal.s16  q9, d11, d0[1]  \n"     
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-27) += (r00-r07) * k20
                    "vmlal.s16  q11, d11, d0[2] \n"
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-37) += (r00-r07) * k30
                    "vmlal.s16  q13, d11, d0[3] \n"

                    "vext.s8    q4, q4, #1      \n"// d8=r01-r08 q4
                    "vmovl.s8   q5, d8          \n"// d10(r01-r04) d11(r05-r08)
                    
                    "vmlal.s16  q6, d10, d1[0]  \n"// sum(00-07) += (r01-r08) * k01
                    "vmlal.s16  q7, d11, d1[0]  \n"
                    "vmlal.s16  q8, d10, d1[1]  \n"// sum(10-17) += (r01-r08) * k11
                    "vmlal.s16  q9, d11, d1[1]  \n"     
                    "vmlal.s16  q10, d10, d1[2] \n"// sum(20-27) += (r01-r08) * k21
                    "vmlal.s16  q11, d11, d1[2] \n"
                    "vmlal.s16  q12, d10, d1[3] \n"// sum(30-37) += (r01-r08) * k31
                    "vmlal.s16  q13, d11, d1[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r02-r09 q4
                    "vmovl.s8   q5, d8          \n"// d10(r02-r05) d11(r06-r09)
                    
                    "vmlal.s16  q6, d10, d2[0]  \n"// sum(00-07) += (r02-r09) * k02
                    "vmlal.s16  q7, d11, d2[0]  \n"
                    "vmlal.s16  q8, d10, d2[1]  \n"// sum(10-17) += (r02-r09) * k12
                    "vmlal.s16  q9, d11, d2[1]  \n"     
                    "vmlal.s16  q10, d10, d2[2] \n"// sum(20-27) += (r02-r09) * k22
                    "vmlal.s16  q11, d11, d2[2] \n"
                    "vmlal.s16  q12, d10, d2[3] \n"// sum(30-37) += (r02-r09) * k32
                    "vmlal.s16  q13, d11, d2[3] \n"                 
                    
                    // r1
                    "pld        [%6, #128]      \n"
                    "vld1.s8    {d8-d9}, [%6]   \n"// d8=r10-r17 d9=r18-r115 q4
                    "add        %6, #8          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r10-r13) d11(r14-r17)
                    
                    "vmlal.s16  q6, d10, d3[0]  \n"// sum(00-07) += (r10-r17) * k03
                    "vmlal.s16  q7, d11, d3[0]  \n"
                    "vmlal.s16  q8, d10, d3[1]  \n"// sum(10-17) += (r10-r17) * k13
                    "vmlal.s16  q9, d11, d3[1]  \n"     
                    "vmlal.s16  q10, d10, d3[2] \n"// sum(20-27) += (r10-r17) * k23
                    "vmlal.s16  q11, d11, d3[2] \n"
                    "vmlal.s16  q12, d10, d3[3] \n"// sum(30-37) += (r10-r17) * k33
                    "vmlal.s16  q13, d11, d3[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r11-r18 q4
                    "vmovl.s8   q5, d8          \n"// d10(r11-r14) d11(r15-r18)

                    "vmlal.s16  q6, d10, d4[0]  \n"// sum(00-07) += (r11-r18) * k04
                    "vmlal.s16  q7, d11, d4[0]  \n"
                    "vmlal.s16  q8, d10, d4[1]  \n"// sum(10-17) += (r11-r18) * k14
                    "vmlal.s16  q9, d11, d4[1]  \n"     
                    "vmlal.s16  q10, d10, d4[2] \n"// sum(20-27) += (r11-r18) * k24
                    "vmlal.s16  q11, d11, d4[2] \n"
                    "vmlal.s16  q12, d10, d4[3] \n"// sum(30-37) += (r11-r18) * k34
                    "vmlal.s16  q13, d11, d4[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r12-r19 q4
                    "vmovl.s8   q5, d8          \n"// d10(r12-r15) d11(r16-r19)

                    "vmlal.s16  q6, d10, d5[0]  \n"// sum(00-07) += (r12-r19) * k05
                    "vmlal.s16  q7, d11, d5[0]  \n"
                    "vmlal.s16  q8, d10, d5[1]  \n"// sum(10-17) += (r12-r19) * k15
                    "vmlal.s16  q9, d11, d5[1]  \n"     
                    "vmlal.s16  q10, d10, d5[2] \n"// sum(20-27) += (r12-r19) * k25
                    "vmlal.s16  q11, d11, d5[2] \n"
                    "vmlal.s16  q12, d10, d5[3] \n"// sum(30-37) += (r12-r19) * k35
                    "vmlal.s16  q13, d11, d5[3] \n"

                    // r2
                    "pld        [%7, #128]      \n"
                    "vld1.s8    {d8-d9}, [%7]   \n"// d8=r20-r27 d9=r28-r215 q4
                    "add        %7, #8          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r20-r23) d11(r24-r27)

                    "vmlal.s16  q6, d10, d6[0]  \n"// sum(00-07) += (r20-r27) * k06
                    "vmlal.s16  q7, d11, d6[0]  \n"
                    "vmlal.s16  q8, d10, d6[1]  \n"// sum(10-17) += (r20-r27) * k16
                    "vmlal.s16  q9, d11, d6[1]  \n"     
                    "vmlal.s16  q10, d10, d6[2] \n"// sum(20-27) += (r20-r27) * k26
                    "vmlal.s16  q11, d11, d6[2] \n"
                    "vmlal.s16  q12, d10, d6[3] \n"// sum(30-37) += (r20-r27) * k36
                    "vmlal.s16  q13, d11, d6[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r21-r28 q4
                    "vmovl.s8   q5, d8          \n"// d10(r21-r24) d11(r25-r28)

                    "vmlal.s16  q6, d10, d7[0]  \n"// sum(00-07) += (r21-r28) * k07
                    "vmlal.s16  q7, d11, d7[0]  \n"
                    "vmlal.s16  q8, d10, d7[1]  \n"// sum(10-17) += (r21-r28) * k17
                    "vmlal.s16  q9, d11, d7[1]  \n"     
                    "vmlal.s16  q10, d10, d7[2] \n"// sum(20-27) += (r21-r28) * k27
                    "vmlal.s16  q11, d11, d7[2] \n"
                    "vmlal.s16  q12, d10, d7[3] \n"// sum(30-37) += (r21-r28) * k37
                    "vmlal.s16  q13, d11, d7[3] \n"
                    
                    "vld1.s8    {d0}, [%8]      \n"// d0(k08-k38 xx-xx)
                    "add        %8, #4          \n"
                    "vmovl.s8   q0, d0          \n"// d0(k08-k38) d1(xx-xx)

                    "vext.s8    q4, q4, #1      \n"// d8=r22-r29 q4
                    "vmovl.s8   q5, d8          \n"// d10(r22-r25) d11(r26-r29)

                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-07) += (r22-r29) * k08
                    "vmlal.s16  q7, d11, d0[0]  \n"
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-17) += (r22-r29) * k18
                    "vmlal.s16  q9, d11, d0[1]  \n"     
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-27) += (r22-r29) * k28
                    "vmlal.s16  q11, d11, d0[2] \n"
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-37) += (r22-r29) * k38
                    "vmlal.s16  q13, d11, d0[3] \n"
                    
                    "vst1.s32   {d12-d15}, [%1]! \n"// sum00-sum07 q6 q7
                    "vst1.s32   {d16-d19}, [%2]! \n"// sum10-sum17 q8 q9
                    "vst1.s32   {d20-d23}, [%3]! \n"// sum20-sum27 q10 q11
                    "vst1.s32   {d24-d27}, [%4]! \n"// sum30-sum37 q12 q13

                    "sub        %8, #36          \n"
                    "subs       %0, #1           \n"

                    "bne        0b               \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(r0),         // %5
                      "=r"(r1),         // %6
                      "=r"(r2),         // %7
                      "=r"(ktmp)        // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" //q14 q15 not be used...
                );
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;
                asm volatile(
                    "vld1.s8    {d0-d3}, [%7]!  \n"// d0=(k00-k30 k01-k31) d1=(k02-k32 k03-k33) d2=(k04-k34 k05-k35) d3=(k06-k36 k07-k37)
                    // r0
                    "vld1.s8    {d8}, [%4]      \n"// d8=r00-r07
                    "add        %4, #4          \n"
                    "vld1.s32   {d12-d13}, [%0] \n"// sum00-sum03 q6
                    "vld1.s32   {d16-d17}, [%1] \n"// sum10-sum13 q8
                    "vld1.s32   {d20-d21}, [%2] \n"// sum20-sum23 q10
                    "vld1.s32   {d24-d25}, [%3] \n"// sum30-sum33 q12
                    
                    "vmovl.s8   q3, d3          \n"// d6(k06-k36) d7(k07-k37) 
                    "vmovl.s8   q2, d2          \n"// d4(k04-k34) d5(k05-k35)
                    "vmovl.s8   q1, d1          \n"// d2(k02-k32) d3(k03-k33)
                    "vmovl.s8   q0, d0          \n"// d0(k00-k30) d1(k01-k31)
                    "vmovl.s8   q5, d8          \n"// d10(r00-r03)
                    
                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-03) += (r00-r03) * k00
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-13) += (r00-r03) * k10
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-23) += (r00-r03) * k20
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-33) += (r00-r03) * k30

                    "vext.s8    d8, d8, #1      \n"// d8=r01-r08
                    "vmovl.s8   q5, d8          \n"// d10(r01-r04)
                    
                    "vmlal.s16  q6, d10, d1[0]  \n"// sum(00-03) += (r01-r04) * k01
                    "vmlal.s16  q8, d10, d1[1]  \n"// sum(10-13) += (r01-r04) * k11
                    "vmlal.s16  q10, d10, d1[2] \n"// sum(20-23) += (r01-r04) * k21
                    "vmlal.s16  q12, d10, d1[3] \n"// sum(30-33) += (r01-r04) * k31
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r02-r09
                    "vmovl.s8   q5, d8          \n"// d10(r02-r05)
                    
                    "vmlal.s16  q6, d10, d2[0]  \n"// sum(00-03) += (r02-r05) * k02
                    "vmlal.s16  q8, d10, d2[1]  \n"// sum(10-13) += (r02-r05) * k12
                    "vmlal.s16  q10, d10, d2[2] \n"// sum(20-23) += (r02-r05) * k22
                    "vmlal.s16  q12, d10, d2[3] \n"// sum(30-33) += (r02-r05) * k32
                    
                    // r1
                    "vld1.s8    {d8}, [%5]      \n"// d8=r10-r17
                    "add        %5, #4          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r10-r13)
                    
                    "vmlal.s16  q6, d10, d3[0]  \n"// sum(00-03) += (r10-r13) * k03
                    "vmlal.s16  q8, d10, d3[1]  \n"// sum(10-13) += (r10-r13) * k13
                    "vmlal.s16  q10, d10, d3[2] \n"// sum(20-23) += (r10-r13) * k23
                    "vmlal.s16  q12, d10, d3[3] \n"// sum(30-33) += (r10-r13) * k33
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r11-r18
                    "vmovl.s8   q5, d8          \n"// d10(r11-r14)

                    "vmlal.s16  q6, d10, d4[0]  \n"// sum(00-03) += (r11-r14) * k04
                    "vmlal.s16  q8, d10, d4[1]  \n"// sum(10-13) += (r11-r14) * k14
                    "vmlal.s16  q10, d10, d4[2] \n"// sum(20-23) += (r11-r14) * k24
                    "vmlal.s16  q12, d10, d4[3] \n"// sum(30-33) += (r11-r14) * k34
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r12-r19 q4
                    "vmovl.s8   q5, d8          \n"// d10(r12-r15)

                    "vmlal.s16  q6, d10, d5[0]  \n"// sum(00-03) += (r12-r15) * k05
                    "vmlal.s16  q8, d10, d5[1]  \n"// sum(10-13) += (r12-r15) * k15
                    "vmlal.s16  q10, d10, d5[2] \n"// sum(20-23) += (r12-r15) * k25
                    "vmlal.s16  q12, d10, d5[3] \n"// sum(30-33) += (r12-r15) * k35

                    // r2
                    "vld1.s8    {d8}, [%6]      \n"// d8=r20-r27
                    "add        %6, #4          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r20-r23)

                    "vmlal.s16  q6, d10, d6[0]  \n"// sum(00-03) += (r20-r23) * k06
                    "vmlal.s16  q8, d10, d6[1]  \n"// sum(10-13) += (r20-r23) * k16
                    "vmlal.s16  q10, d10, d6[2] \n"// sum(20-23) += (r20-r23) * k26
                    "vmlal.s16  q12, d10, d6[3] \n"// sum(30-33) += (r20-r23) * k36
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r21-r28 q4
                    "vmovl.s8   q5, d8          \n"// d10(r21-r24)

                    "vmlal.s16  q6, d10, d7[0]  \n"// sum(00-03) += (r21-r24) * k07
                    "vmlal.s16  q8, d10, d7[1]  \n"// sum(10-13) += (r21-r24) * k17
                    "vmlal.s16  q10, d10, d7[2] \n"// sum(20-23) += (r21-r24) * k27
                    "vmlal.s16  q12, d10, d7[3] \n"// sum(30-33) += (r21-r24) * k37
                    
                    "vld1.s8    {d0}, [%7]      \n"// d0(k08-k38 xx-xx)
                    "add        %7, #4          \n"
                    "vmovl.s8   q0, d0          \n"// d0(k08-k38) d1(xx-xx)

                    "vext.s8    d8, d8, #1      \n"// d8=r22-r25
                    "vmovl.s8   q5, d8          \n"// d10(r22-r25)

                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-03) += (r22-r25) * k08
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-13) += (r22-r25) * k18
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-23) += (r22-r25) * k28
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-33) += (r22-r25) * k38
                    
                    "vst1.s32   {d12-d13}, [%0]! \n"// sum00-sum03 q6
                    "vst1.s32   {d16-d17}, [%1]! \n"// sum10-sum13 q8
                    "vst1.s32   {d20-d21}, [%2]! \n"// sum20-sum23 q10
                    "vst1.s32   {d24-d25}, [%3]! \n"// sum30-sum33 q12 

                    "sub        %7, #36          \n"

                    : "=r"(outptr0),    // %0
                      "=r"(outptr1),    // %1
                      "=r"(outptr2),    // %2
                      "=r"(outptr3),    // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(r2),         // %6
                      "=r"(ktmp)        // %7
                    : "0"(outptr0),
                      "1"(outptr1),
                      "2"(outptr2),
                      "3"(outptr3),
                      "4"(r0),
                      "5"(r1),
                      "6"(r2),
                      "7"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13" //q14 q15 not be used...
                );
                }
#endif
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[]}, [%4]!       \n"// d0(r00)
                        "vld1.s8    {d1[]}, [%4]!       \n"// d1(r01)

                        "vld1.s8    {d4-d7}, [%7]!      \n"// d4(k00-k30 k01-k31) d5(k02-k32 k03-k33) d6(k04-k34 k05-k35) d7(k06-k36 k07-k37)

                        "vsli.64    d0, d1, #32         \n"// d0(r00 r00 r00 r00 r01 r01 r01 r01)

                        "vld1.s8    {d2[]}, [%4]        \n"// d2(r02 r02 r02 r02 r02 r02 r02 r02)
                        "sub        %4, %4, #2          \n"
                        "vld1.s8    {d3[]}, [%5]!       \n"// d3(r10 r10 r10 r10 r10 r10 r10 r10)
                        
                        "vmovl.s8   q5, d7              \n"// d10(k06-k36) d11(k07-k37)
                        "vmovl.s8   q4, d6              \n"// d8(k04-k34) d9(k05-k35)
                        "vmovl.s8   q3, d5              \n"// d6(k02-k32) d7(k03-k33)
                        "vmovl.s8   q2, d4              \n"// d4(k00-k30) d5(k01-k31)
                        
                        "vmovl.s8   q0, d0              \n"// d0(r00 r00 r00 r00) d1(r01 r01 r01 r01)

                        "vsli.64    d2, d3, #32         \n"// d2(r02 r02 r02 r02 r10 r10 r10 r10)
                        
                        "vmull.s16  q8, d0, d4          \n"// (r00) * (k00-k30)
                        "vmull.s16  q9, d1, d5          \n"// (r01) * (k01-k31)
                        
                        "vmovl.s8   q10, d2             \n"// d20(r02 r02 r02 r02) d21(r10 r10 r10 r10)

                        "vld1.s8    {d0[]}, [%5]!       \n"// d0(r11 r11 r11 r11 r11 r11 r11 r11)
                        "vld1.s8    {d1[]}, [%5]        \n"// d1(r12 r12 r12 r12 r12 r12 r12 r12)
                        "sub        %5, %5, #2          \n"

                        "vsli.64    d0, d1, #32         \n"// d0(r11 r11 r11 r11 r12 r12 r12 r12)

                        "vmlal.s16  q8, d20, d6         \n"// (r02) * (k02-k32)
                        "vmlal.s16  q9, d21, d7         \n"// (r10) * (k03-k33)
                        
                        "vmovl.s8   q0, d0              \n"// d0(r11 r11 r11 r11 ) d1(r12 r12 r12 r12)

                        "vld1.s8    {d2[]}, [%6]!       \n"// d2(r20 r20 r20 r20 r20 r20 r20 r20)
                        "vld1.s8    {d3[]}, [%6]!       \n"// d3(r21 r21 r21 r21 r21 r21 r21 r21)

                        "vsli.64    d2, d3, #32         \n"// d2(r20 r20 r20 r20 r21 r21 r21 r21)

                        "vmlal.s16  q8, d0, d8          \n"// (r11) * (k04-k34)
                        "vmlal.s16  q9, d1, d9          \n"// (r12) * (k05-k35)     

                        "vmovl.s8   q2, d2              \n"// d4(r20 r20 r20 r20) d5(r21 r21 r21 r21)

                        "vld1.s8    {d0[]}, [%6]        \n"// d0(r22 r22 r22 r22 r22 r22 r22 r22)
                        "sub        %6, %6, #2          \n"
                        "veor       d1, d1, d1          \n"// d1 = 0

                        "vld1.s8    {d6}, [%7]          \n"// d6 = k08-k38 xxxx
                        "sub        %7, #32             \n"

                        "vsli.64    d0, d1, #32         \n"// d0(r22 r22 r22 r22 0 0 0 0)
                        "vmovl.s8   q4, d6              \n"// d8(k08-k38)
                        "vmovl.s8   q0, d0              \n"// d0(r22 r22 r22 r22) d1(0 0 0 0)

                        "vmlal.s16  q8, d4, d10         \n"// (r20) * (k06-k36)
                        "vmlal.s16  q9, d5, d11         \n"// (r21) * (k07-k37)

                        "vld1.s32   {d20[0]}, [%0]      \n"

                        "vmlal.s16  q8, d0, d8          \n"// (r22) * (k08-k38)

                        "vld1.s32   {d20[1]}, [%1]      \n"

                        "vadd.s32   q8, q8, q9          \n"

                        "vld1.s32   {d21[0]}, [%2]      \n"
                        "vld1.s32   {d21[1]}, [%3]      \n"

                        "vadd.s32   q10, q10, q8        \n"

                        "vst1.s32   {d20[0]}, [%0]!     \n"
                        "vst1.s32   {d20[1]}, [%1]!     \n"
                        "vst1.s32   {d21[0]}, [%2]!     \n"
                        "vst1.s32   {d21[1]}, [%3]!     \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(ktmp)        // %7
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9", "q10"
                    );
#else
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum1 += r0[0] * ktmp[1];
                    sum2 += r0[0] * ktmp[2];
                    sum3 += r0[0] * ktmp[3];

                    sum0 += r0[1] * ktmp[4];
                    sum1 += r0[1] * ktmp[5];
                    sum2 += r0[1] * ktmp[6];
                    sum3 += r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[2] * ktmp[0];
                    sum1 += r0[2] * ktmp[1];
                    sum2 += r0[2] * ktmp[2];
                    sum3 += r0[2] * ktmp[3];

                    sum0 += r1[0] * ktmp[4];
                    sum1 += r1[0] * ktmp[5];
                    sum2 += r1[0] * ktmp[6];
                    sum3 += r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[1] * ktmp[0];
                    sum1 += r1[1] * ktmp[1];
                    sum2 += r1[1] * ktmp[2];
                    sum3 += r1[1] * ktmp[3];

                    sum0 += r1[2] * ktmp[4];
                    sum1 += r1[2] * ktmp[5];
                    sum2 += r1[2] * ktmp[6];
                    sum3 += r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[0] * ktmp[0];
                    sum1 += r2[0] * ktmp[1];
                    sum2 += r2[0] * ktmp[2];
                    sum3 += r2[0] * ktmp[3];

                    sum0 += r2[1] * ktmp[4];
                    sum1 += r2[1] * ktmp[5];
                    sum2 += r2[1] * ktmp[6];
                    sum3 += r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[2] * ktmp[0];
                    sum1 += r2[2] * ktmp[1];
                    sum2 += r2[2] * ktmp[2];
                    sum3 += r2[2] * ktmp[3];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    ktmp -= 8*5;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
#endif
                    r0++;
                    r1++;
                    r2++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            ktmp += 4*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* ktmp = _kernel.channel(p/4 + p%4);

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr0n = outptr0 + outw;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;
            const signed char* r3 = img0 + w * 3;

            int i = 0;

#if __ARM_NEON
            int8x16_t _k0123456789x = vld1q_s8(ktmp);
            int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
            int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

            int16x4_t _k0123 = vget_low_s16(_k_s16);
            int16x4_t _k4567 = vget_high_s16(_k_s16);
            int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON

            for (; i+1 < outh; i+=2)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn >0; nn--)
                {
                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08 
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum0n = vmull_lane_s16(vget_high_s16(_r0_s16), _k0123, 0);

                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum1n = vmull_lane_s16(vget_high_s16(_r01_s16), _k0123, 1);

                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02
                    int32x4_t _sum2n = vmull_lane_s16(vget_high_s16(_r02_s16), _k0123, 2);

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r1_s16), _k0123, 3);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r11_s16), _k4567, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r12_s16), _k4567, 1); 

                    int32x4_t _sum4 = vmull_lane_s16(vget_low_s16(_r1_s16), _k0123, 0); // (r10 - r17) * k00
                    int32x4_t _sum4n = vmull_lane_s16(vget_high_s16(_r1_s16), _k0123, 0);

                    int32x4_t _sum5 = vmull_lane_s16(vget_low_s16(_r11_s16), _k0123, 1); // (r11 - r18) * k01
                    int32x4_t _sum5n = vmull_lane_s16(vget_high_s16(_r11_s16), _k0123, 1);

                    int32x4_t _sum6 = vmull_lane_s16(vget_low_s16(_r12_s16), _k0123, 2); // (r12 - r19) * k02
                    int32x4_t _sum6n = vmull_lane_s16(vget_high_s16(_r12_s16), _k0123, 2);

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r2_s16), _k4567, 2);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r21_s16), _k4567, 3);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r22_s16), _k8xxx, 0);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r2_s16), _k0123, 3); // (r20 - r27) * k03
                    _sum4n = vmlal_lane_s16(_sum4n, vget_high_s16(_r2_s16), _k0123, 3);

                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r21_s16), _k4567, 0); // (r21 - r28) * k04
                    _sum5n = vmlal_lane_s16(_sum5n, vget_high_s16(_r21_s16), _k4567, 0);

                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r22_s16), _k4567, 1); // (r22 - r29) * k05
                    _sum6n = vmlal_lane_s16(_sum6n, vget_high_s16(_r22_s16), _k4567, 1);

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out01 = vld1q_s32(outptr0+4);
                    int32x4_t _out10 = vld1q_s32(outptr0n);
                    int32x4_t _out11 = vld1q_s32(outptr0n+4);

                    // r3
                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    int16x8_t _r3_s16 = vmovl_s8(_r3);   // r30 - r37
                    int16x8_t _r31_s16 = vmovl_s8(_r31); // r31 - r38
                    int16x8_t _r32_s16 = vmovl_s8(_r32); // r32 - r39

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _sum2n = vaddq_s32(_sum2n, _sum0n);

                    _out00 = vaddq_s32(_out00, _sum2);
                    _out01 = vaddq_s32(_out01, _sum2n);

                    vst1q_s32(outptr0, _out00);
                    vst1q_s32(outptr0+4, _out01);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r3_s16), _k4567, 2); // (r30 - r37) * k06
                    _sum4n = vmlal_lane_s16(_sum4n, vget_high_s16(_r3_s16), _k4567, 2);

                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r31_s16), _k4567, 3); // (r31 - r38) * k07
                    _sum5n = vmlal_lane_s16(_sum5n, vget_high_s16(_r31_s16), _k4567, 3);

                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r32_s16), _k8xxx, 0); // (r32 - r39) * k08
                    _sum6n = vmlal_lane_s16(_sum6n, vget_high_s16(_r32_s16), _k8xxx, 0); 

                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum4n = vaddq_s32(_sum4n, _sum5n);
                    _sum6 = vaddq_s32(_sum6, _sum4);
                    _sum6n = vaddq_s32(_sum6n, _sum4n);

                    _out10 = vaddq_s32(_out10, _sum6);
                    _out11 = vaddq_s32(_out11, _sum6n);

                    vst1q_s32(outptr0n, _out10);
                    vst1q_s32(outptr0n+4, _out11);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr0n += 8;
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;

                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05

                    int32x4_t _sum4 = vmull_lane_s16(vget_low_s16(_r1_s16), _k0123, 0); // (r10 - r17) * k00
                    int32x4_t _sum5 = vmull_lane_s16(vget_low_s16(_r11_s16), _k0123, 1); // (r11 - r18) * k01
                    int32x4_t _sum6 = vmull_lane_s16(vget_low_s16(_r12_s16), _k0123, 2); // (r12 - r19) * k02

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r2_s16), _k0123, 3); // (r20 - r27) * k03
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r21_s16), _k4567, 0); // (r21 - r28) * k04
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r22_s16), _k4567, 1); // (r22 - r29) * k05

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out10 = vld1q_s32(outptr0n);

                    // r3
                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    int16x8_t _r3_s16 = vmovl_s8(_r3);   // r30 - r37
                    int16x8_t _r31_s16 = vmovl_s8(_r31); // r31 - r38
                    int16x8_t _r32_s16 = vmovl_s8(_r32); // r32 - r39

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _out00 = vaddq_s32(_out00, _sum2);

                    vst1q_s32(outptr0, _out00);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r3_s16), _k4567, 2); // (r30 - r37) * k06
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r31_s16), _k4567, 3); // (r31 - r38) * k07
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r32_s16), _k8xxx, 0); // (r32 - r39) * k08

                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum6 = vaddq_s32(_sum6, _sum4);

                    _out10 = vaddq_s32(_out10, _sum6);

                    vst1q_s32(outptr0n, _out10);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr0 += 4;
                    outptr0n += 4;
                }
#endif
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[0]}, [%2]!  \n"
                        "vld1.s8    {d0[1]}, [%2]!  \n"
                        "vld1.s8    {d0[2]}, [%2]   \n"
                        "sub        %2, #2          \n"

                        "vld1.s8    {d0[3]}, [%3]!  \n"
                        "vld1.s8    {d0[4]}, [%3]!  \n"
                        "vld1.s8    {d0[5]}, [%3]   \n"
                        "sub        %3, #2          \n"

                        "vld1.s8    {d0[6]}, [%4]!  \n"
                        "vld1.s8    {d0[7]}, [%4]!  \n"// d0(r00 r01 r02 r10 r11 r12 r22 r21)

                        "vld1.s8    {d4[]}, [%4]    \n"// d4(r22 r22 r22 r22 r22 r22 r22 r22) 
                        "sub        %4, #2          \n"

                        "vext.s8    d1, d0, d4, #3  \n"// d1(r10 r11 r12 r22 r21 r22 r22 r22)

                        "vld1.s8    {d1[6]}, [%5]!  \n"
                        "vld1.s8    {d1[7]}, [%5]!  \n"// d1(r10 r11 r12 r22 r21 r22 r30 r31)

                        "vld1.s8    {d2}, [%6]!     \n"// d2(k00 k01 k02 k10 k11 k12 k20 k21)

                        "vld1.s8    {d5[]}, [%5]    \n"// d5(r32 r32 r32 r32 r32 r32 r32 r32)
                        "sub        %5, #2          \n"

                        "veor       d3, d3          \n"// d3(00 00 00 00 00 00 00 00)

                        "vmull.s8   q8, d0, d2      \n"// sum0 = (r00 - r21) * (k00 - k21)
                        "vmull.s8   q9, d1, d2      \n"// sum1 = (r10 - r31) * (k00 - k21)

                        "vld1.s8    {d3[0]}, [%6]   \n"// d3(k22 00 00 00 00 00 00 00)
                        "sub        %6, #8          \n"

                        "vmull.s8   q10, d4, d3     \n"// r22 * k22
                        "vmull.s8   q11, d5, d3     \n"// r22 * k22

                        "vld1.s32   {d6[0]}, [%0]   \n"

                        "vaddl.s16  q10, d16, d18   \n"
                        "vaddl.s16  q11, d18, d22   \n"
                        "vaddw.s16  q10, q10, d17   \n"
                        "vaddw.s16  q11, q11, d19   \n"

                        "vld1.s32   {d6[1]}, [%1]   \n"

                        "vpadd.s32  d20, d20, d21   \n"
                        "vpadd.s32  d22, d22, d23   \n"
                        "vpadd.s32  d20, d20, d22   \n"
                        "vpadd.s32  d6, d6, d20     \n"

                        "vst1.s32   {d6[0]}, [%0]!  \n"
                        "vst1.s32   {d6[1]}, [%1]!  \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr0n),   // %1
                          "=r"(r0),         // %2
                          "=r"(r1),         // %3
                          "=r"(r2),         // %4
                          "=r"(r3),         // %5
                          "=r"(ktmp)        // %6
                        : "0"(outptr0),
                          "1"(outptr0n),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2),
                          "5"(r3),
                          "6"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
#else
                    int sum0 = 0;
                    int sum0n = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum0 += r0[1] * ktmp[1];
                    sum0 += r0[2] * ktmp[2];
                    sum0 += r1[0] * ktmp[3];
                    sum0 += r1[1] * ktmp[4];
                    sum0 += r1[2] * ktmp[5];
                    sum0 += r2[0] * ktmp[6];
                    sum0 += r2[1] * ktmp[7];

                    sum0 += r2[2] * ktmp[8];

                    sum0n += r1[0] * ktmp[0];
                    sum0n += r1[1] * ktmp[1];
                    sum0n += r1[2] * ktmp[2];
                    sum0n += r2[0] * ktmp[3];
                    sum0n += r2[1] * ktmp[4];
                    sum0n += r2[2] * ktmp[5];
                    sum0n += r3[0] * ktmp[6];
                    sum0n += r3[1] * ktmp[7];

                    sum0n += r3[2] * ktmp[8];

                    *outptr0 += sum0;
                    *outptr0n += sum0n;

                    outptr0++;
                    outptr0n++;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
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
                for (; nn >0; nn--)
                {
                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08 
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum0n = vmull_lane_s16(vget_high_s16(_r0_s16), _k0123, 0);

                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum1n = vmull_lane_s16(vget_high_s16(_r01_s16), _k0123, 1);

                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02
                    int32x4_t _sum2n = vmull_lane_s16(vget_high_s16(_r02_s16), _k0123, 2);

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r1_s16), _k0123, 3);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r11_s16), _k4567, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r12_s16), _k4567, 1);

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r2_s16), _k4567, 2);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r21_s16), _k4567, 3);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r22_s16), _k8xxx, 0); 

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out01 = vld1q_s32(outptr0+4);

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _sum2n = vaddq_s32(_sum2n, _sum0n);

                    _out00 = vaddq_s32(_out00, _sum2);
                    _out01 = vaddq_s32(_out01, _sum2n);

                    vst1q_s32(outptr0, _out00);
                    vst1q_s32(outptr0+4, _out01);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr0n += 8;
                }
#endif
                for (; remain>0; remain--)
                {
                    int sum0 = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum0 += r0[1] * ktmp[1];
                    sum0 += r0[2] * ktmp[2];
                    sum0 += r1[0] * ktmp[3];
                    sum0 += r1[1] * ktmp[4];
                    sum0 += r1[2] * ktmp[5];
                    sum0 += r2[0] * ktmp[6];
                    sum0 += r2[1] * ktmp[7];
                    sum0 += r2[2] * ktmp[8];

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

            ktmp += 9;
        }
    }
}

static void conv3x3s2_packed_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p+0);
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

        const signed char* ktmp = _kernel.channel(p/8);

        for (int q=0; q<inch; q++)
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

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    // TODO
                }
#else // __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.s32   {d16-d17}, [%1]     \n"// out0
                    "pld        [%2, #128]          \n"
                    "vld1.s32   {d18-d19}, [%2]     \n"// out1
                    "pld        [%3, #128]          \n"
                    "vld1.s32   {d20-d21}, [%3]     \n"// out2
                    "pld        [%4, #128]          \n"
                    "vld1.s32   {d22-d23}, [%4]     \n"// out3 

                    // r0
                    "pld        [%9, #64]          \n"
                    "vld2.s8    {d8-d9}, [%9]       \n"// d8(a00 a02 a04 a06 a08 a010 a012 a014), d9(a01 a03 a05 a07 a09 a011 a013 a015)
                    "add        %9, #8              \n"
                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d0-d2}, [%12]!     \n"// d0(k00-k70) d1(k01-k71) d2(k02-k72)

                    "pld        [%5, #128]          \n"
                    "vld1.s32   {d24-d25}, [%5]     \n"// out4
                    "pld        [%6, #128]          \n"
                    "vld1.s32   {d26-d27}, [%6]     \n"// out5

                    "vmovl.s8   q2, d2              \n"// q2(k02-k72)
                    "vmovl.s8   q1, d1              \n"// q1(k01-k71)
                    "vmovl.s8   q0, d0              \n"// q0(k00-k70)
                    "vext.s8    d12, d8, d8, #1     \n"// d12(a02 a04 a06 a08 x x x x)

                    "pld        [%7, #128]          \n"
                    "vld1.s32   {d28-d29}, [%7]     \n"// out6

                    "vmovl.s8   q5, d9              \n"// q5(a01 a03 a05 a07 a09 a011 a013 a015) d11
                    "vmovl.s8   q4, d8              \n"// q4(a00 a02 a04 a06 a08 a010 a012 a014) d9
                    "vmovl.s8   q6, d12             \n"// q6(a02 a04 a06 a08 a010 a012 a014 a016) d13

                    "pld        [%8, #128]          \n"
                    "vld1.s32   {d30-d31}, [%8]     \n"// out7  

                    "vmlal.s16  q8, d8, d0[0]       \n"// sum0 += (a00 a02 a04 a06) * k00
                    "vmlal.s16  q9, d8, d0[1]       \n"// sum1 += (a00 a02 a04 a06) * k10
                    "vmlal.s16  q10, d8, d0[2]      \n"// sum2 += (a00 a02 a04 a06) * k20
                    "vmlal.s16  q11, d8, d0[3]      \n"// sum3 += (a00 a02 a04 a06) * k30
                    "vmlal.s16  q12, d8, d1[0]      \n"// sum4 += (a00 a02 a04 a06) * k40
                    "vmlal.s16  q13, d8, d1[1]      \n"// sum5 += (a00 a02 a04 a06) * k50
                    "vmlal.s16  q14, d8, d1[2]      \n"// sum6 += (a00 a02 a04 a06) * k60
                    "vmlal.s16  q15, d8, d1[3]      \n"// sum7 += (a00 a02 a04 a06) * k70

                    "vmlal.s16  q8, d10, d2[0]      \n"// sum0 += (a01-a07) * k01
                    "vmlal.s16  q9, d10, d2[1]      \n"// sum1 += (a01-a07) * k11
                    "vmlal.s16  q10, d10, d2[2]     \n"// sum2 += (a01-a07) * k21
                    "vmlal.s16  q11, d10, d2[3]     \n"// sum3 += (a01-a07) * k31
                    "vmlal.s16  q12, d10, d3[0]     \n"// sum4 += (a01-a07) * k41
                    "vmlal.s16  q13, d10, d3[1]     \n"// sum5 += (a01-a07) * k51
                    "vmlal.s16  q14, d10, d3[2]     \n"// sum6 += (a01-a07) * k61
                    "vmlal.s16  q15, d10, d3[3]     \n"// sum7 += (a01-a07) * k71   

                    "pld        [%10, #64]         \n"
                    "vld2.s8    {d8-d9}, [%10]      \n"// d8(a10 a12 a14 a16 a18 a110 a112 a114), d9(a11 a13 a15 a17 a19 a111 a113 a115)
                    "add        %10, #8             \n"

                    "vmlal.s16  q8, d12, d4[0]      \n"// sum0 += (a02-a08) * k02
                    "vmlal.s16  q9, d12, d4[1]      \n"// sum1 += (a02-a08) * k12
                    "vmlal.s16  q10, d12, d4[2]     \n"// sum2 += (a02-a08) * k22
                    "vmlal.s16  q11, d12, d4[3]     \n"// sum3 += (a02-a08) * k32

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d0-d2}, [%12]!     \n"// d0(k03-k73) d1(k04-k74) d2(k05-k75)

                    "vmlal.s16  q12, d12, d5[0]     \n"// sum4 += (a02-a08) * k42
                    "vmlal.s16  q13, d12, d5[1]     \n"// sum5 += (a02-a08) * k52
                    "vmlal.s16  q14, d12, d5[2]     \n"// sum6 += (a02-a08) * k62
                    "vmlal.s16  q15, d12, d5[3]     \n"// sum7 += (a02-a08) * k72

                    // r1
                    "vext.s8    d12, d8, d8, #1     \n"// d12(a12 a14 a16 a18 x x x x)

                    "vmovl.s8   q2, d2              \n"// q2(k05-k75)
                    "vmovl.s8   q1, d1              \n"// q1(k04-k74)
                    "vmovl.s8   q0, d0              \n"// q0(k03-k73)
                    "vmovl.s8   q5, d9              \n"// q5(a11-a115)
                    "vmovl.s8   q4, d8              \n"// q4(a10-a114)
                    "vmovl.s8   q6, d12             \n"// q6(a12-a116)

                    "vmlal.s16  q8, d8, d0[0]       \n"// sum0 += (a10-a16) * k03
                    "vmlal.s16  q9, d8, d0[1]       \n"// sum1 += (a10-a16) * k13
                    "vmlal.s16  q10, d8, d0[2]      \n"// sum2 += (a10-a16) * k23
                    "vmlal.s16  q11, d8, d0[3]      \n"// sum3 += (a10-a16) * k33
                    "vmlal.s16  q12, d8, d1[0]      \n"// sum4 += (a10-a16) * k43
                    "vmlal.s16  q13, d8, d1[1]      \n"// sum5 += (a10-a16) * k53
                    "vmlal.s16  q14, d8, d1[2]      \n"// sum6 += (a10-a16) * k63
                    "vmlal.s16  q15, d8, d1[3]      \n"// sum7 += (a10-a16) * k73

                    "vmlal.s16  q8, d10, d2[0]      \n"// sum0 += (a11-a17) * k04
                    "vmlal.s16  q9, d10, d2[1]      \n"// sum1 += (a11-a17) * k14
                    "vmlal.s16  q10, d10, d2[2]     \n"// sum2 += (a11-a17) * k24
                    "vmlal.s16  q11, d10, d2[3]     \n"// sum3 += (a11-a17) * k34
                    "vmlal.s16  q12, d10, d3[0]     \n"// sum4 += (a11-a17) * k44
                    "vmlal.s16  q13, d10, d3[1]     \n"// sum5 += (a11-a17) * k54
                    "vmlal.s16  q14, d10, d3[2]     \n"// sum6 += (a11-a17) * k64
                    "vmlal.s16  q15, d10, d3[3]     \n"// sum7 += (a11-a17) * k74

                    "pld        [%11, #64]         \n"
                    "vld2.s8    {d8-d9}, [%11]      \n"// d8(a20 a22 a24 a26 a28 a210 a212 a214), d9(a21 a23 a25 a27 a29 a211 a213 a215)
                    "add        %11, #8             \n"

                    "vmlal.s16  q8, d12, d4[0]      \n"// sum0 += (a12-a18) * k05
                    "vmlal.s16  q9, d12, d4[1]      \n"// sum1 += (a12-a18) * k15
                    "vmlal.s16  q10, d12, d4[2]     \n"// sum2 += (a12-a18) * k25
                    "vmlal.s16  q11, d12, d4[3]     \n"// sum3 += (a12-a18) * k35

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d0-d2}, [%12]!     \n"// d0(k06-k76) d1(k07-k77) d2(k08-k78)

                    "vmlal.s16  q12, d12, d5[0]     \n"// sum4 += (a12-a18) * k45
                    "vmlal.s16  q13, d12, d5[1]     \n"// sum5 += (a12-a18) * k55
                    "vmlal.s16  q14, d12, d5[2]     \n"// sum6 += (a12-a18) * k65
                    "vmlal.s16  q15, d12, d5[3]     \n"// sum7 += (a12-a18) * k75

                    // r2
                    "vext.s8    d12, d8, d8, #1     \n"// d12(a22 a24 a26 a28 x x x x)
                    
                    "vmovl.s8   q2, d2              \n"// q2(k08-k78)
                    "vmovl.s8   q1, d1              \n"// q1(k07-k77)
                    "vmovl.s8   q0, d0              \n"// q0(k06-k76) 
                    "vmovl.s8   q5, d9              \n"// q5(a21-a215)
                    "vmovl.s8   q4, d8              \n"// q4(a20-a214)
                    "vmovl.s8   q6, d12             \n"// q6(a22-a216)

                    "vmlal.s16  q8, d8, d0[0]       \n"// sum0 += (a20-a26) * k06
                    "vmlal.s16  q9, d8, d0[1]       \n"// sum1 += (a20-a26) * k16
                    "vmlal.s16  q10, d8, d0[2]      \n"// sum2 += (a20-a26) * k26
                    "vmlal.s16  q11, d8, d0[3]      \n"// sum3 += (a20-a26) * k36
                    "vmlal.s16  q12, d8, d1[0]      \n"// sum4 += (a20-a26) * k46
                    "vmlal.s16  q13, d8, d1[1]      \n"// sum5 += (a20-a26) * k56
                    "vmlal.s16  q14, d8, d1[2]      \n"// sum6 += (a20-a26) * k66
                    "vmlal.s16  q15, d8, d1[3]      \n"// sum7 += (a20-a26) * k76

                    "vmlal.s16  q8, d10, d2[0]      \n"// sum0 += (a21-a27) * k07
                    "vmlal.s16  q9, d10, d2[1]      \n"// sum1 += (a21-a27) * k17
                    "vmlal.s16  q10, d10, d2[2]     \n"// sum2 += (a21-a27) * k27
                    "vmlal.s16  q11, d10, d2[3]     \n"// sum3 += (a21-a27) * k37
                    "vmlal.s16  q12, d10, d3[0]     \n"// sum4 += (a21-a27) * k47
                    "vmlal.s16  q13, d10, d3[1]     \n"// sum5 += (a21-a27) * k57
                    "vmlal.s16  q14, d10, d3[2]     \n"// sum6 += (a21-a27) * k67
                    "vmlal.s16  q15, d10, d3[3]     \n"// sum7 += (a21-a27) * k77

                    "vmlal.s16  q8, d12, d4[0]      \n"// sum0 += (a22-a28) * k08
                    "vmlal.s16  q9, d12, d4[1]      \n"// sum1 += (a22-a28) * k18
                    "vmlal.s16  q10, d12, d4[2]     \n"// sum2 += (a22-a28) * k28
                    "vmlal.s16  q11, d12, d4[3]     \n"// sum3 += (a22-a28) * k38
                    "vmlal.s16  q12, d12, d5[0]     \n"// sum4 += (a22-a28) * k48
                    "vmlal.s16  q13, d12, d5[1]     \n"// sum5 += (a22-a28) * k58
                    "vmlal.s16  q14, d12, d5[2]     \n"// sum6 += (a22-a28) * k68
                    "vmlal.s16  q15, d12, d5[3]     \n"// sum7 += (a22-a28) * k78

                    // save s32 to memory
                    "sub        %12, %12, #72       \n"
                    "vst1.s32   {d16-d17}, [%1]!    \n"// out0
                    "vst1.s32   {d18-d19}, [%2]!    \n"// out1
                    "vst1.s32   {d20-d21}, [%3]!    \n"// out2
                    "vst1.s32   {d22-d23}, [%4]!    \n"// out3
                    "subs       %0, #1              \n"
                    "vst1.s32   {d24-d25}, [%5]!    \n"// out4
                    "vst1.s32   {d26-d27}, [%6]!    \n"// out5
                    "vst1.s32   {d28-d29}, [%7]!    \n"// out6
                    "vst1.s32   {d30-d31}, [%8]!    \n"// out7
                                                 
                    "bne        0b                  \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(outptr4),    // %5
                      "=r"(outptr5),    // %6
                      "=r"(outptr6),    // %7
                      "=r"(outptr7),    // %8
                      "=r"(r0),         // %9
                      "=r"(r1),         // %10
                      "=r"(r2),         // %11
                      "=r"(ktmp)        // %12
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
                      "12"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    // TODO
#else // __aarch64__
                    asm volatile(
                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d0}, [%8]         \n"// d0(a00 a01 a02 ....)
                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d2}, [%9]         \n"// d2(a10 a11 a12 ....)
                        "pld        [%10, #64]         \n"
                        "vld1.s8    {d4}, [%10]        \n"// d4(a20 a21 a22 ....)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n"// d6(k00-k70) d7(k01-k71) d8(k02-k72)

                        "vmovl.s8   q0, d0             \n"// d0(a00 a01 a02 x) 
                        "vmovl.s8   q1, d2             \n"// d2(a10 a11 a12 x)
                        "vmovl.s8   q2, d4             \n"// d4(a20 a21 a22 x)

                        "vmovl.s8   q5, d8             \n"// d10(k02-k32) d11(k42-k72)
                        "vmovl.s8   q4, d7             \n"// d8(k01-k31) d9(k41-k71)
                        "vmovl.s8   q3, d6             \n"// d6(k00-k30) d7(k40-k70)

                        "vld1.s32   {d20[0]}, [%0]     \n"// out0 q10
                        "vld1.s32   {d20[1]}, [%1]     \n"// out1
                        "vld1.s32   {d21[0]}, [%2]     \n"// out2 
                        "vld1.s32   {d21[1]}, [%3]     \n"// out3

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d24-d26}, [%11]!  \n"
                        "vmovl.s8   q14, d26           \n"// d28(k05-k35) d29(k45-k75)
                        "vmovl.s8   q13, d25           \n"// d26(k04-k34) d27(k44-k74)
                        "vmovl.s8   q12, d24           \n"// d24(k03-k33) d25(k43-k73)

                        "vld1.s32   {d22[0]}, [%4]     \n"// out4 q11
                        "vld1.s32   {d22[1]}, [%5]     \n"// out5
                        "vld1.s32   {d23[0]}, [%6]     \n"// out6
                        "vld1.s32   {d23[1]}, [%7]     \n"// out7

                        "vmull.s16  q6, d6, d0[0]      \n"// a00 x (k00-k30)
                        "vmull.s16  q7, d7, d0[0]      \n"// a00 x (k40-k70)
                        "vmull.s16  q8, d8, d0[1]      \n"// a01 x (k01-k31)
                        "vmull.s16  q9, d9, d0[1]      \n"// a01 x (k41-k71)
                        "vmlal.s16  q10, d10, d0[2]    \n"// a02 x (k02-k32)
                        "vmlal.s16  q11, d11, d0[2]    \n"// a02 x (k42-k72)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n"
                        "vmovl.s8   q5, d8             \n"// d10(k08-k38) d11(k48-k78)
                        "vmovl.s8   q4, d7             \n"// d8(k07-k37) d9(k47-k77)
                        "vmovl.s8   q3, d6             \n"// d6(k06-k36) d7(k46-k76)

                        "vmlal.s16  q6, d24, d2[0]     \n"// a10 x (k03-k33)
                        "vmlal.s16  q7, d25, d2[0]     \n"// a10 x (k43-k73)
                        "vmlal.s16  q8, d26, d2[1]     \n"// a11 x (k04-k34)
                        "vmlal.s16  q9, d27, d2[1]     \n"// a11 x (k44-k74)
                        "vmlal.s16  q10, d28, d2[2]    \n"// a12 x (k05-k35)
                        "vmlal.s16  q11, d29, d2[2]    \n"// a12 x (k45-k75)

                        "vmlal.s16  q6, d6, d4[0]      \n"// a20 x (k06-k36)
                        "vmlal.s16  q7, d7, d4[0]      \n"// a20 x (k46-k76)
                        "vmlal.s16  q8, d8, d4[1]      \n"// a21 x (k07-k37)
                        "vmlal.s16  q9, d9, d4[1]      \n"// a21 x (k47-k77)
                        "vmlal.s16  q10, d10, d4[2]    \n"// a22 x (k08-k38)
                        "vmlal.s16  q11, d11, d4[2]    \n"// a22 x (k48-k78)

                        "vadd.s32   q8, q8, q6         \n"
                        "vadd.s32   q9, q9, q7         \n"

                        "sub        %11, %11, #72      \n"

                        "vadd.s32   q10, q10, q8       \n"
                        "vadd.s32   q11, q11, q9       \n"

                        "vst1.s32   {d20[0]}, [%0]!    \n"// out0
                        "vst1.s32   {d20[1]}, [%1]!    \n"// out1
                        "vst1.s32   {d21[0]}, [%2]!    \n"// out2
                        "vst1.s32   {d21[1]}, [%3]!    \n"// out3
                        "vst1.s32   {d22[0]}, [%4]!    \n"// out4
                        "vst1.s32   {d22[1]}, [%5]!    \n"// out5
                        "vst1.s32   {d23[0]}, [%6]!    \n"// out6
                        "vst1.s32   {d23[1]}, [%7]!    \n"// out7

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(outptr4),    // %4
                          "=r"(outptr5),    // %5
                          "=r"(outptr6),    // %6
                          "=r"(outptr7),    // %7
                          "=r"(r0),         // %8
                          "=r"(r1),         // %9
                          "=r"(r2),         // %10
                          "=r"(ktmp)        // %11
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(outptr4),
                          "5"(outptr5),
                          "6"(outptr6),
                          "7"(outptr7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(ktmp)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else // __ARM_NEON
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;
                    int sum4 = 0;
                    int sum5 = 0;
                    int sum6 = 0;
                    int sum7 = 0;

                    sum0 += (int)r0[0] * ktmp[0];
                    sum1 += (int)r0[0] * ktmp[1];
                    sum2 += (int)r0[0] * ktmp[2];
                    sum3 += (int)r0[0] * ktmp[3];
                    sum4 += (int)r0[0] * ktmp[4];
                    sum5 += (int)r0[0] * ktmp[5];
                    sum6 += (int)r0[0] * ktmp[6];
                    sum7 += (int)r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[1] * ktmp[0];
                    sum1 += (int)r0[1] * ktmp[1];
                    sum2 += (int)r0[1] * ktmp[2];
                    sum3 += (int)r0[1] * ktmp[3];
                    sum4 += (int)r0[1] * ktmp[4];
                    sum5 += (int)r0[1] * ktmp[5];
                    sum6 += (int)r0[1] * ktmp[6];
                    sum7 += (int)r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[2] * ktmp[0];
                    sum1 += (int)r0[2] * ktmp[1];
                    sum2 += (int)r0[2] * ktmp[2];
                    sum3 += (int)r0[2] * ktmp[3];
                    sum4 += (int)r0[2] * ktmp[4];
                    sum5 += (int)r0[2] * ktmp[5];
                    sum6 += (int)r0[2] * ktmp[6];
                    sum7 += (int)r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[0] * ktmp[0];
                    sum1 += (int)r1[0] * ktmp[1];
                    sum2 += (int)r1[0] * ktmp[2];
                    sum3 += (int)r1[0] * ktmp[3];
                    sum4 += (int)r1[0] * ktmp[4];
                    sum5 += (int)r1[0] * ktmp[5];
                    sum6 += (int)r1[0] * ktmp[6];
                    sum7 += (int)r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[1] * ktmp[0];
                    sum1 += (int)r1[1] * ktmp[1];
                    sum2 += (int)r1[1] * ktmp[2];
                    sum3 += (int)r1[1] * ktmp[3];
                    sum4 += (int)r1[1] * ktmp[4];
                    sum5 += (int)r1[1] * ktmp[5];
                    sum6 += (int)r1[1] * ktmp[6];
                    sum7 += (int)r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[2] * ktmp[0];
                    sum1 += (int)r1[2] * ktmp[1];
                    sum2 += (int)r1[2] * ktmp[2];
                    sum3 += (int)r1[2] * ktmp[3];
                    sum4 += (int)r1[2] * ktmp[4];
                    sum5 += (int)r1[2] * ktmp[5];
                    sum6 += (int)r1[2] * ktmp[6];
                    sum7 += (int)r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[0] * ktmp[0];
                    sum1 += (int)r2[0] * ktmp[1];
                    sum2 += (int)r2[0] * ktmp[2];
                    sum3 += (int)r2[0] * ktmp[3];
                    sum4 += (int)r2[0] * ktmp[4];
                    sum5 += (int)r2[0] * ktmp[5];
                    sum6 += (int)r2[0] * ktmp[6];
                    sum7 += (int)r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[1] * ktmp[0];
                    sum1 += (int)r2[1] * ktmp[1];
                    sum2 += (int)r2[1] * ktmp[2];
                    sum3 += (int)r2[1] * ktmp[3];
                    sum4 += (int)r2[1] * ktmp[4];
                    sum5 += (int)r2[1] * ktmp[5];
                    sum6 += (int)r2[1] * ktmp[6];
                    sum7 += (int)r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[2] * ktmp[0];
                    sum1 += (int)r2[2] * ktmp[1];
                    sum2 += (int)r2[2] * ktmp[2];
                    sum3 += (int)r2[2] * ktmp[3];
                    sum4 += (int)r2[2] * ktmp[4];
                    sum5 += (int)r2[2] * ktmp[5];
                    sum6 += (int)r2[2] * ktmp[6];
                    sum7 += (int)r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8*9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        const signed char* ktmp = _kernel.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;

            int i = 0;

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
                // TODO
#else
                if (nn > 0)
                {
                asm volatile(
                    "vld1.s8    {d0-d1}, [%5]       \n"// d0(k0 - k7) d1(k8 ...)
                    "vmovl.s8   q1, d1              \n"// d2(k8 ...)
                    "vmovl.s8   q0, d0              \n"// d0(k0 - k3) d1(k4 - k7)
                    "0:                             \n"
                    "pld        [%2, #192]          \n"
                    "vld2.s8    {d4-d5}, [%2]!      \n"// r0 d4(a00 a02 ... a014) d5(a01 a03 ... a015)
                    "vld2.s8    {d8-d9}, [%2]       \n"//    d8(a016 ....)
                    "vld2.s8    {d10-d11}, [%3]!    \n"// r1 d10(a10 a12 ... a114) d11(a11 a13 ... a115)
                    "vld2.s8    {d14-d15}, [%3]     \n"//    d14(a116 ....)
                    "vld2.s8    {d16-d17}, [%4]!    \n"// r2 d16(a20 a22 ... a214) d17(a21 a23 ... a215)
                    "vld2.s8    {d20-d21}, [%4]     \n"//    d20(a216 ....)
                    "vld1.s32   {d22-d25}, [%1]     \n"// q11(out0 - out3) q12(out4 - out7)

                    "vext.s8    d8, d4, d8, #1      \n"//  d8(a02 a04 ... a016)
                    "vext.s8    d14, d10, d14, #1   \n"// d14(a12 a14 ... a116)
                    "vext.s8    d20, d16, d20, #1   \n"// d20(a22 a24 ... a216)

                    "vmovl.s8   q3, d5              \n"// q3(a01 a03 ... a015)
                    "vmovl.s8   q2, d4              \n"// q2(a00 a02 ... a014)
                    "vmovl.s8   q4, d8              \n"// q4(a02 a04 ... a016)

                    "vmovl.s8   q6, d11             \n"// q6(a11 a13 ... a115)
                    "vmovl.s8   q5, d10             \n"// q5(a10 a12 ... a114)
                    "vmovl.s8   q7, d14             \n"// q7(a12 a14 ... a116)

                    "vmovl.s8   q9, d17             \n"// q9(a21 a23 ... a215)
                    "vmovl.s8   q8, d16             \n"// q8(a20 a22 ... a214)
                    "vmovl.s8   q10, d20            \n"// q10(a22 a24 ... a216)
        
                    "vmlal.s16  q11, d4, d0[0]      \n"// k0
                    "vmlal.s16  q12, d5, d0[0]      \n"
                    "vmull.s16  q13, d6, d0[1]      \n"// k1
                    "vmull.s16  q14, d7, d0[1]      \n"
                    "vmlal.s16  q11, d8, d0[2]      \n"// k2
                    "vmlal.s16  q12, d9, d0[2]      \n"

                    "vmlal.s16  q13, d12, d1[0]     \n"// k4
                    "vmlal.s16  q14, d13, d1[0]     \n"
                    "vmlal.s16  q11, d10, d0[3]     \n"// k3
                    "vmlal.s16  q12, d11, d0[3]     \n"
                    "vmlal.s16  q13, d14, d1[1]     \n"// k5
                    "vmlal.s16  q14, d15, d1[1]     \n"

                    "vmlal.s16  q11, d16, d1[2]     \n"// k6
                    "vmlal.s16  q12, d17, d1[2]     \n"
                    "vmlal.s16  q13, d18, d1[3]     \n"// k7 
                    "vmlal.s16  q14, d19, d1[3]     \n"
                    "vmlal.s16  q11, d20, d2[0]     \n"// k8 
                    "vmlal.s16  q12, d21, d2[0]     \n"

                    "vadd.s32   q11, q11, q13       \n"
                    "vadd.s32   q12, q12, q14       \n"
                    
                    "vst1.32    {d22-d25}, [%1]!    \n"     

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2),     // %4
                      "=r"(ktmp)    // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                if (remain > 0)
                {
#if __ARM_NEON
                    int8x8_t _k01234567s8 = vld1_s8(ktmp);
                    int8x8_t _k8xxxxxxxs8 = vld1_s8(ktmp+8);
                    int8x8_t _k34567xxxs8 = vext_s8(_k01234567s8, _k01234567s8, 3);
                    int8x8_t _k678xxxxxs8 = vext_s8(_k01234567s8, _k8xxxxxxxs8, 6);
                    int16x8_t _k0123_s16 = vmovl_s8(_k01234567s8);
                    int16x8_t _k3456_s16 = vmovl_s8(_k34567xxxs8);
                    int16x8_t _k678x_s16 = vmovl_s8(_k678xxxxxs8);
#endif
                    for (; remain>0; remain--)
                    {
#if __ARM_NEON
                        int8x8_t _r00s8 = vld1_s8(r0);
                        int8x8_t _r10s8 = vld1_s8(r1);
                        int8x8_t _r20s8 = vld1_s8(r2);

                        int16x8_t _r00s16 = vmovl_s8(_r00s8);
                        int16x8_t _r10s16 = vmovl_s8(_r10s8);
                        int16x8_t _r20s16 = vmovl_s8(_r20s8);

                        int32x4_t _sum = vmull_s16(vget_low_s16(_r00s16), vget_low_s16(_k0123_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r10s16), vget_low_s16(_k3456_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r20s16), vget_low_s16(_k678x_s16));

                        _sum = vsetq_lane_s32(*outptr, _sum, 3);

#if __aarch64__
                        *outptr = vaddvq_s32(_sum);
#else
                        int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                        _ss = vpadd_s32(_ss, _ss);

                        *outptr = vget_lane_s32(_ss, 0);
#endif // __aarch64__
#else
                        int sum = 0;

                        sum += (int)r0[0] * ktmp[0];
                        sum += (int)r0[1] * ktmp[1];
                        sum += (int)r0[2] * ktmp[2];
                        sum += (int)r1[0] * ktmp[3];
                        sum += (int)r1[1] * ktmp[4];
                        sum += (int)r1[2] * ktmp[5];
                        sum += (int)r2[0] * ktmp[6];
                        sum += (int)r2[1] * ktmp[7];
                        sum += (int)r2[2] * ktmp[8];

                        *outptr += sum;
#endif // __ARM_NEON
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
#endif
