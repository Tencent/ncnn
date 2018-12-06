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

                remain = outw;

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

static void conv3x3s1_packed_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
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
            const signed char *r3 = img0 + w * 3;

            int i = 0;

            for (; i+1 < outh; i+=2)
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
//                     "pld        [%9, #256]      \n"
                    "vld1.s8    {d0-d3}, [%9]!  \n"// d0=k00 k01  d1=k02 k10  d2=k11 k12  d3=k20 k21

                    "pld        [%5, #128]      \n"
                    "vld1.s8    {d4-d5}, [%5]   \n"// d4=r00 d5=r00n
                    "add        %5, #8          \n"

                    "vdup.s8    d8, d0[0]       \n"
                    "vdup.s8    d9, d0[1]       \n"

                    "pld        [%6, #128]      \n"
                    "vld1.s8    {d6-d7}, [%6]   \n"// d6=r10 d7=r10n
                    "add        %6, #8          \n"

                    "vdup.s8    d10, d0[2]      \n"
                    "vdup.s8    d11, d0[3]      \n"

                    "vmull.s8   q8, d4, d8      \n"
                    "vmull.s8   q9, d4, d9      \n"

                    "vdup.s8    d12, d0[4]      \n"
                    "vdup.s8    d13, d0[5]      \n"

                    "vmull.s8   q10, d4, d10    \n"
                    "vmull.s8   q11, d4, d11    \n"

                    "vdup.s8    d14, d0[6]      \n"
                    "vdup.s8    d15, d0[7]      \n"

                    "vmull.s8   q12, d6, d8     \n"
                    "vmull.s8   q13, d6, d9     \n"

                    "vext.s8    q2, q2, q2, #1  \n"// d4=r01

                    "vmull.s8   q14, d6, d10    \n"
                    "vmull.s8   q15, d6, d11    \n"

                    "vext.s8    q3, q3, q3, #1  \n"// d6=r11

                    "vmlal.s8   q8, d4, d12     \n"
                    "vmlal.s8   q9, d4, d13     \n"

                    "vdup.s8    d8, d1[0]       \n"
                    "vdup.s8    d9, d1[1]       \n"

                    "vmlal.s8   q10, d4, d14    \n"
                    "vmlal.s8   q11, d4, d15    \n"

                    "vdup.s8    d10, d1[2]      \n"
                    "vdup.s8    d11, d1[3]      \n"

                    "vmlal.s8   q12, d6, d12    \n"
                    "vmlal.s8   q13, d6, d13    \n"

                    "vext.s8    q2, q2, q2, #1  \n"// d4=r02

                    "vmlal.s8   q14, d6, d14    \n"
                    "vmlal.s8   q15, d6, d15    \n"

                    "vext.s8    q3, q3, q3, #1  \n"// d6=r12

                    "vmlal.s8   q8, d4, d8      \n"
                    "vmlal.s8   q9, d4, d9      \n"

                    "vdup.s8    d12, d1[4]      \n"
                    "vdup.s8    d13, d1[5]      \n"

                    "vmlal.s8   q10, d4, d10    \n"
                    "vmlal.s8   q11, d4, d11    \n"

                    "vdup.s8    d14, d1[6]      \n"
                    "vdup.s8    d15, d1[7]      \n"

                    "vmlal.s8   q12, d6, d8     \n"
                    "vmlal.s8   q13, d6, d9     \n"

                    "pld        [%7, #128]      \n"
                    "vld1.s8    {d4-d5}, [%7]   \n"// d4=r20 d5=r20n
                    "add        %7, #8          \n"

                    "vmlal.s8   q14, d6, d10    \n"
                    "vmlal.s8   q15, d6, d11    \n"

                    ///
                    "vext.s8    q3, q3, q3, #14 \n"// d6=r10

                    "vmlal.s8   q8, d6, d12     \n"
                    "vmlal.s8   q9, d6, d13     \n"

                    "vdup.s8    d8, d2[0]       \n"
                    "vdup.s8    d9, d2[1]       \n"

                    "vmlal.s8   q10, d6, d14    \n"
                    "vmlal.s8   q11, d6, d15    \n"

                    "vdup.s8    d10, d2[2]      \n"
                    "vdup.s8    d11, d2[3]      \n"

                    "vmlal.s8   q12, d4, d12    \n"
                    "vmlal.s8   q13, d4, d13    \n"

                    "vext.s8    q3, q3, q3, #1  \n"// d6=r11

                    "vmlal.s8   q14, d4, d14    \n"
                    "vmlal.s8   q15, d4, d15    \n"

                    "vext.s8    q2, q2, q2, #1  \n"// d4=r21

                    "vmlal.s8   q8, d6, d8      \n"
                    "vmlal.s8   q9, d6, d9      \n"

                    "vdup.s8    d12, d2[4]      \n"
                    "vdup.s8    d13, d2[5]      \n"

                    "vmlal.s8   q10, d6, d10    \n"
                    "vmlal.s8   q11, d6, d11    \n"

                    "vdup.s8    d14, d2[6]      \n"
                    "vdup.s8    d15, d2[7]      \n"

                    "vmlal.s8   q12, d4, d8     \n"
                    "vmlal.s8   q13, d4, d9     \n"

                    "vext.s8    q3, q3, q3, #1  \n"// d6=r12

                    "vmlal.s8   q14, d4, d10    \n"
                    "vmlal.s8   q15, d4, d11    \n"

                    "vext.s8    q2, q2, q2, #1  \n"// d4=r22

                    "vmlal.s8   q8, d6, d12     \n"
                    "vmlal.s8   q9, d6, d13     \n"

                    "vdup.s8    d8, d3[0]       \n"
                    "vdup.s8    d9, d3[1]       \n"

                    "vmlal.s8   q10, d6, d14    \n"
                    "vmlal.s8   q11, d6, d15    \n"

                    "vdup.s8    d10, d3[2]      \n"
                    "vdup.s8    d11, d3[3]      \n"

                    "vmlal.s8   q12, d4, d12    \n"
                    "vmlal.s8   q13, d4, d13    \n"

                    "pld        [%8, #128]      \n"
                    "vld1.s8    {d6-d7}, [%8]   \n"// d6=r30 d6=r30n
                    "add        %8, #8          \n"

                    "vmlal.s8   q14, d4, d14    \n"
                    "vmlal.s8   q15, d4, d15    \n"

                    ///
                    "vext.s8    q2, q2, q2, #14 \n"// d4=r20

                    "vmlal.s8   q8, d4, d8      \n"
                    "vmlal.s8   q9, d4, d9      \n"

                    "vdup.s8    d12, d3[4]      \n"
                    "vdup.s8    d13, d3[5]      \n"

                    "vmlal.s8   q10, d4, d10    \n"
                    "vmlal.s8   q11, d4, d11    \n"

                    "vdup.s8    d14, d3[6]      \n"
                    "vdup.s8    d15, d3[7]      \n"

                    "vmlal.s8   q12, d6, d8     \n"
                    "vmlal.s8   q13, d6, d9     \n"

                    "vext.s8    q2, q2, q2, #1  \n"// d4=r21

                    "vmlal.s8   q14, d6, d10    \n"
                    "vmlal.s8   q15, d6, d11    \n"

                    "vext.s8    q3, q3, q3, #1  \n"// d6=r31

//                     "pld        [%9, #128]      \n"
                    "vld1.s8    {d0}, [%9]      \n"
                    "add        %9, #4          \n"

                    "vmlal.s8   q8, d4, d12     \n"
                    "vmlal.s8   q9, d4, d13     \n"

                    "vdup.s8    d8, d0[0]       \n"
                    "vdup.s8    d9, d0[1]       \n"

                    "vmlal.s8   q10, d4, d14    \n"
                    "vmlal.s8   q11, d4, d15    \n"

                    "vdup.s8    d10, d0[2]      \n"
                    "vdup.s8    d11, d0[3]      \n"

                    "vmlal.s8   q12, d6, d12    \n"
                    "vmlal.s8   q13, d6, d13    \n"

                    "vext.s8    q2, q2, q2, #1  \n"// d4=r22

                    "vmlal.s8   q14, d6, d14    \n"
                    "vmlal.s8   q15, d6, d15    \n"

                    "vext.s8    q3, q3, q3, #1  \n"// d6=r32

                    "vmlal.s8   q8, d4, d8      \n"
                    "vmlal.s8   q9, d4, d9      \n"

                    "pld        [%1, #256]      \n"
                    "vld1.s32   {d12-d15}, [%1] \n"

                    "vmlal.s8   q10, d4, d10    \n"
                    "vmlal.s8   q11, d4, d11    \n"

                    "pld        [%2, #256]      \n"
                    "vld1.s32   {d0-d3}, [%2]   \n"

                    "vaddw.s16  q6, q6, d16     \n"
                    "vaddw.s16  q7, q7, d17     \n"
                    "vaddw.s16  q0, q0, d18     \n"
                    "vaddw.s16  q1, q1, d19     \n"

                    "pld        [%3, #256]      \n"
                    "vld1.s32   {d16-d19}, [%3]  \n"

                    "vmlal.s8   q12, d6, d8     \n"
                    "vmlal.s8   q13, d6, d9     \n"

                    "vst1.s32   {d12-d15}, [%1] \n"
                    "add        %1, %1, %20, lsl #2 \n"

                    "vmlal.s8   q14, d6, d10    \n"
                    "vmlal.s8   q15, d6, d11    \n"

                    "pld        [%4, #256]      \n"
                    "vld1.s32   {d4-d7}, [%4]   \n"

                    "vst1.s32   {d0-d3}, [%2]   \n"
                    "add        %2, %2, %20, lsl #2 \n"

                    "vaddw.s16  q8, q8, d20     \n"
                    "vaddw.s16  q9, q9, d21     \n"

                    "pld        [%1, #256]      \n"
                    "vld1.s32   {d12-d15}, [%1] \n"

                    "vaddw.s16  q2, q2, d22     \n"
                    "vaddw.s16  q3, q3, d23     \n"

                    ///
                    "pld        [%2, #256]      \n"
                    "vld1.s32   {d0-d3}, [%2]   \n"

                    "vaddw.s16  q6, q6, d24     \n"

                    "vst1.s32   {d16-d19}, [%3] \n"
                    "add        %3, %3, %20, lsl #2 \n"

                    "vaddw.s16  q7, q7, d25     \n"

                    "pld        [%3, #256]      \n"
                    "vld1.s32   {d8-d11}, [%3] \n"

                    "vaddw.s16  q0, q0, d26     \n"

                    "vst1.s32   {d4-d7}, [%4]   \n"
                    "add        %4, %4, %20, lsl #2 \n"

                    ///
                    "vaddw.s16  q1, q1, d27     \n"

                    "pld        [%4, #256]      \n"
                    "vld1.s32   {d4-d7}, [%4]   \n"

                    "vaddw.s16  q4, q4, d28     \n"

                    "vst1.s32   {d12-d15}, [%1]! \n"

                    "vaddw.s16  q5, q5, d29     \n"

                    "vst1.s32   {d0-d3}, [%2]!  \n"

                    "vaddw.s16  q2, q2, d30     \n"

                    "vst1.s32   {d8-d11}, [%3]! \n"

                    "vaddw.s16  q3, q3, d31     \n"

                    "sub        %9, #36         \n"
                    "subs       %0, #1          \n"

                    "sub        %1, %1, %20, lsl #2 \n"
                    "sub        %2, %2, %20, lsl #2 \n"
                    "sub        %3, %3, %20, lsl #2 \n"

                    "vst1.s32   {d4-d7}, [%4]!  \n"

                    "sub        %4, %4, %20, lsl #2 \n"

                    "bne        0b              \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(r0),         // %5
                      "=r"(r1),         // %6
                      "=r"(r2),         // %7
                      "=r"(r3),         // %8
                      "=r"(ktmp)        // %9
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "9"(ktmp),
                      "r"(outw)         // %20
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                        "vld1.s8    {d0}, [%4]          \n"// d0 = r00 r00n
                        "add        %4, #4              \n"
                        "vld1.s8    {d2}, [%5]          \n"// d2 = r10 r10n
                        "add        %5, #4              \n"

                        "vld1.s8    {d1}, [%6]          \n"// d1 = r20 r20n
                        "add        %6, #4              \n"
                        "vld1.s8    {d3}, [%7]          \n"// d3 = r30 r30n
                        "add        %7, #4              \n"

                        "vext.s8    d4, d0, d0, #1      \n"// d4 = r01
                        "vext.s8    d6, d2, d2, #1      \n"// d6 = r11
                        "vext.s8    d5, d1, d1, #1      \n"// d5 = r21
                        "vext.s8    d7, d3, d3, #1      \n"// d7 = r31

                        "vext.s8    d8, d0, d0, #2      \n"// d8 = r02
                        "vext.s8    d10, d2, d2, #2     \n"// d10 = r12
                        "vext.s8    d9, d1, d1, #2      \n"// d9 = r22
                        "vext.s8    d11, d3, d3, #2     \n"// d11 = r32

                        "vld1.s8    {d12-d15}, [%8]!    \n"// d12=k00 k01  d13=k02 k10  d14=k11 k12  d15=k20 k21

                        "vsli.64    q0, q1, #32         \n"// d0 = r00 r10  d1 = r20 r30

                        /// r00 r20 r10
                        "vdup.s8    d24, d12[0]         \n"
                        "vdup.s8    d25, d12[1]         \n"
                        "vdup.s8    d26, d12[2]         \n"
                        "vdup.s8    d27, d12[3]         \n"

                        "vmull.s8   q8, d0, d24         \n"
                        "vmull.s8   q9, d0, d25         \n"

                        "vdup.s8    d28, d15[0]         \n"
                        "vdup.s8    d29, d15[1]         \n"

                        "vmull.s8   q10, d0, d26        \n"
                        "vmull.s8   q11, d0, d27        \n"

                        "vdup.s8    d30, d15[2]         \n"
                        "vdup.s8    d31, d15[3]         \n"

                        "vmlal.s8   q8, d1, d28         \n"
                        "vmlal.s8   q9, d1, d29         \n"

                        "vext.s8    d0, d0, d1, #4      \n"// d0 = r10 r20

                        "vdup.s8    d24, d13[4]         \n"
                        "vdup.s8    d25, d13[5]         \n"

                        "vmlal.s8   q10, d1, d30        \n"
                        "vmlal.s8   q11, d1, d31        \n"

                        "vdup.s8    d26, d13[6]         \n"
                        "vdup.s8    d27, d13[7]         \n"

                        "vmlal.s8   q8, d0, d24         \n"
                        "vmlal.s8   q9, d0, d25         \n"

                        /// r01 r21 r11
                        "vsli.64    q2, q3, #32         \n"// d4 = r01 r11  d5 = r21 r31

                        "vdup.s8    d28, d12[4]         \n"
                        "vdup.s8    d29, d12[5]         \n"

                        "vmlal.s8   q10, d0, d26        \n"
                        "vmlal.s8   q11, d0, d27        \n"

                        "vdup.s8    d30, d12[6]         \n"
                        "vdup.s8    d31, d12[7]         \n"

                        "vmlal.s8   q8, d4, d28         \n"
                        "vmlal.s8   q9, d4, d29         \n"

                        "vdup.s8    d24, d15[4]         \n"
                        "vdup.s8    d25, d15[5]         \n"

                        "vmlal.s8   q10, d4, d30        \n"
                        "vmlal.s8   q11, d4, d31        \n"

                        "vdup.s8    d26, d15[6]         \n"
                        "vdup.s8    d27, d15[7]         \n"

                        "vmlal.s8   q8, d5, d24         \n"
                        "vmlal.s8   q9, d5, d25         \n"

                        "vext.s8    d4, d4, d5, #4      \n"// d4 = r11 r21

                        "vdup.s8    d28, d14[0]         \n"
                        "vdup.s8    d29, d14[1]         \n"

                        "vmlal.s8   q10, d5, d26        \n"
                        "vmlal.s8   q11, d5, d27        \n"

                        "vdup.s8    d30, d14[2]         \n"
                        "vdup.s8    d31, d14[3]         \n"

                        "vmlal.s8   q8, d4, d28         \n"
                        "vmlal.s8   q9, d4, d29         \n"

                        /// r02 r22 r12
                        "vsli.64    q4, q5, #32         \n"// d8 = r02 r12  d9 = r22 r32

                        "vld1.s8    {d12}, [%8]         \n"// d12=k22
                        "add        %8, #4              \n"

                        "vdup.s8    d24, d13[0]         \n"
                        "vdup.s8    d25, d13[1]         \n"

                        "vmlal.s8   q10, d4, d30        \n"
                        "vmlal.s8   q11, d4, d31        \n"

                        "vdup.s8    d26, d13[2]         \n"
                        "vdup.s8    d27, d13[3]         \n"

                        "vmlal.s8   q8, d8, d24         \n"
                        "vmlal.s8   q9, d8, d25         \n"

                        "vdup.s8    d28, d12[0]         \n"
                        "vdup.s8    d29, d12[1]         \n"

                        "vmlal.s8   q10, d8, d26        \n"
                        "vmlal.s8   q11, d8, d27        \n"

                        "vdup.s8    d30, d12[2]         \n"
                        "vdup.s8    d31, d12[3]         \n"

                        "vmlal.s8   q8, d9, d28         \n"
                        "vmlal.s8   q9, d9, d29         \n"

                        "vext.s8    d8, d8, d9, #4      \n"// d8 = r12 r22

                        "vdup.s8    d24, d14[4]         \n"
                        "vdup.s8    d25, d14[5]         \n"

                        "vld1.s32   {d0-d1}, [%0]       \n"

                        "vmlal.s8   q10, d9, d30        \n"
                        "vmlal.s8   q11, d9, d31        \n"

                        "vdup.s8    d26, d14[6]         \n"
                        "vdup.s8    d27, d14[7]         \n"

                        "vld1.s32   {d2-d3}, [%1]       \n"

                        "vmlal.s8   q8, d8, d24         \n"
                        "vmlal.s8   q9, d8, d25         \n"

                        "vld1.s32   {d4-d5}, [%2]       \n"

                        "vmlal.s8   q10, d8, d26        \n"
                        "vmlal.s8   q11, d8, d27        \n"

                        "vld1.s32   {d6-d7}, [%3]       \n"

                        "vaddw.s16  q0, q0, d16         \n"
                        "vaddw.s16  q1, q1, d18         \n"

                        "vst1.s32   {d0-d1}, [%0]       \n"
                        "add        %0, %0, %18, lsl #2 \n"

                        "vaddw.s16  q2, q2, d20         \n"

                        "vst1.s32   {d2-d3}, [%1]       \n"
                        "add        %1, %1, %18, lsl #2 \n"

                        "vaddw.s16  q3, q3, d22         \n"

                        "vld1.s32   {d8-d9}, [%0]       \n"

                        "vld1.s32   {d10-d11}, [%1]     \n"

                        "vst1.s32   {d4-d5}, [%2]       \n"
                        "add        %2, %2, %18, lsl #2 \n"
                        "vst1.s32   {d6-d7}, [%3]       \n"
                        "add        %3, %3, %18, lsl #2 \n"

                        "vld1.s32   {d28-d29}, [%2]     \n"

                        "vld1.s32   {d30-d31}, [%3]     \n"

                        "vaddw.s16  q4, q4, d17         \n"
                        "vaddw.s16  q5, q5, d19         \n"

                        "sub        %8, #36             \n"

                        "vst1.s32   {d8-d9}, [%0]!      \n"

                        "vaddw.s16  q14, q14, d21       \n"
                        "vaddw.s16  q15, q15, d23       \n"

                        "vst1.s32   {d10-d11}, [%1]!    \n"

                        "sub        %0, %0, %18, lsl #2 \n"
                        "sub        %1, %1, %18, lsl #2 \n"

                        "vst1.s32   {d28-d29}, [%2]!    \n"
                        "vst1.s32   {d30-d31}, [%3]!    \n"

                        "sub        %2, %2, %18, lsl #2 \n"
                        "sub        %3, %3, %18, lsl #2 \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(r3),         // %7
                          "=r"(ktmp)        // %8
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(r3),
                          "8"(ktmp),
                          "r"(outw)         // %18
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[]}, [%4]!       \n"// d0 = 00 00
                        "vld1.s8    {d1[]}, [%4]!       \n"// d1 = 01 01
                        "vld1.s8    {d2[]}, [%4]        \n"// d2 = 02 02
                        "sub        %4, %4, #2          \n"

                        "vld1.s8    {d3[]}, [%5]!       \n"// d3 = 10 10
                        "vld1.s8    {d4[]}, [%5]!       \n"// d4 = 11 11
                        "vld1.s8    {d5[]}, [%5]        \n"// d5 = 12 12
                        "sub        %5, %5, #2          \n"

                        "vld1.s8    {d6[]}, [%6]!       \n"// d6 = 20 20
                        "vld1.s8    {d7[]}, [%6]!       \n"// d7 = 21 21
                        "vld1.s8    {d8[]}, [%6]        \n"// d8 = 22 22
                        "sub        %6, %6, #2          \n"

                        "vld1.s8    {d9[]}, [%7]!       \n"// d9 = 30 30
                        "vld1.s8    {d10[]}, [%7]!      \n"// d10 = 31 31
                        "vld1.s8    {d11[]}, [%7]       \n"// d11 = 32 32
                        "sub        %7, %7, #2          \n"

                        "vld1.s8    {d12-d15}, [%8]!    \n"// d12 d13 d14 d15 = 0~7

                        "vsli.64    d0, d1, #32         \n"// d0 = 00 01

                        "vsli.64    d3, d4, #32         \n"// d3 = 10 11

                        "vmull.s8   q8, d0, d12         \n"

                        "vsli.64    d2, d3, #32         \n"// d2 = 02 10

                        "vmull.s8   q9, d3, d12         \n"

                        "vsli.64    d5, d6, #32         \n"// d5 = 12 20

                        "vmlal.s8   q8, d2, d13         \n"

                        "vsli.64    d4, d5, #32         \n"// d4 = 11 12

                        "vmlal.s8   q9, d5, d13         \n"

                        "vsli.64    d7, d8, #32         \n"// d7 = 21 22

                        "vmlal.s8   q8, d4, d14         \n"

                        "vsli.64    d6, d7, #32         \n"// d6 = 20 21

                        "vmlal.s8   q9, d7, d14         \n"

                        "vsli.64    d9, d10, #32        \n"// d9 = 30 31

                        "vld1.s32   {d20[0]}, [%0]      \n"
                        "vld1.s32   {d20[1]}, [%1]      \n"

                        "add        %0, %0, %18, lsl #2 \n"
                        "add        %1, %1, %18, lsl #2 \n"

                        "vmlal.s8   q8, d6, d15         \n"

                        "vsli.64    d8, d11, #32        \n"// d8 = 22 32

                        "vmlal.s8   q9, d9, d15         \n"

                        "vld1.s8    {d14}, [%8]         \n"
                        "add        %8, #4              \n"

                        "vld1.s32   {d21[0]}, [%2]      \n"
                        "vld1.s32   {d21[1]}, [%3]      \n"

                        "add        %2, %2, %18, lsl #2 \n"
                        "add        %3, %3, %18, lsl #2 \n"

                        "vadd.s16   d12, d16, d17       \n"

                        "vadd.s16   d13, d18, d19       \n"// q6 = sum0123 sum0123n

                        "vsli.64    d14, d14, #32       \n"// d14 = 0~3 0~3

                        "vld1.s32   {d22[0]}, [%0]      \n"
                        "vld1.s32   {d22[1]}, [%1]      \n"

                        "vmlal.s8   q6, d8, d14         \n"

                        "sub        %8, #36             \n"

                        ///
                        "vld1.s32   {d23[0]}, [%2]      \n"
                        "vld1.s32   {d23[1]}, [%3]      \n"

                        "sub        %0, %0, %18, lsl #2 \n"
                        "sub        %1, %1, %18, lsl #2 \n"

                        // addw
                        "vaddw.s16  q10, q10, d12       \n"
                        "vaddw.s16  q11, q11, d13       \n"

                        "sub        %2, %2, %18, lsl #2 \n"
                        "sub        %3, %3, %18, lsl #2 \n"

                        "vst1.s32   {d20[0]}, [%0]      \n"
                        "vst1.s32   {d20[1]}, [%1]      \n"

                        "add        %0, %0, %18, lsl #2 \n"
                        "add        %1, %1, %18, lsl #2 \n"

                        "vst1.s32   {d21[0]}, [%2]      \n"
                        "vst1.s32   {d21[1]}, [%3]      \n"

                        "add        %2, %2, %18, lsl #2 \n"
                        "add        %3, %3, %18, lsl #2 \n"

                        "vst1.s32   {d22[0]}, [%0]!     \n"
                        "vst1.s32   {d22[1]}, [%1]!     \n"

                        "sub        %0, %0, %18, lsl #2 \n"
                        "sub        %1, %1, %18, lsl #2 \n"

                        "vst1.s32   {d23[0]}, [%2]!     \n"
                        "vst1.s32   {d23[1]}, [%3]!     \n"

                        "sub        %2, %2, %18, lsl #2 \n"
                        "sub        %3, %3, %18, lsl #2 \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(r3),         // %7
                          "=r"(ktmp)        // %8
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(r3),
                          "8"(ktmp),
                          "r"(outw)         // %18
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                    );
#else
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    int sum0n = 0;
                    int sum1n = 0;
                    int sum2n = 0;
                    int sum3n = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum1 += r0[0] * ktmp[1];
                    sum2 += r0[0] * ktmp[2];
                    sum3 += r0[0] * ktmp[3];
                    sum0 += r0[1] * ktmp[4];
                    sum1 += r0[1] * ktmp[5];
                    sum2 += r0[1] * ktmp[6];
                    sum3 += r0[1] * ktmp[7];

                    sum0n += r1[0] * ktmp[0];
                    sum1n += r1[0] * ktmp[1];
                    sum2n += r1[0] * ktmp[2];
                    sum3n += r1[0] * ktmp[3];
                    sum0n += r1[1] * ktmp[4];
                    sum1n += r1[1] * ktmp[5];
                    sum2n += r1[1] * ktmp[6];
                    sum3n += r1[1] * ktmp[7];

                    ktmp += 8;

                    sum0 += r0[2] * ktmp[0];
                    sum1 += r0[2] * ktmp[1];
                    sum2 += r0[2] * ktmp[2];
                    sum3 += r0[2] * ktmp[3];
                    sum0 += r1[0] * ktmp[4];
                    sum1 += r1[0] * ktmp[5];
                    sum2 += r1[0] * ktmp[6];
                    sum3 += r1[0] * ktmp[7];

                    sum0n += r1[2] * ktmp[0];
                    sum1n += r1[2] * ktmp[1];
                    sum2n += r1[2] * ktmp[2];
                    sum3n += r1[2] * ktmp[3];
                    sum0n += r2[0] * ktmp[4];
                    sum1n += r2[0] * ktmp[5];
                    sum2n += r2[0] * ktmp[6];
                    sum3n += r2[0] * ktmp[7];

                    ktmp += 8;

                    sum0 += r1[1] * ktmp[0];
                    sum1 += r1[1] * ktmp[1];
                    sum2 += r1[1] * ktmp[2];
                    sum3 += r1[1] * ktmp[3];
                    sum0 += r1[2] * ktmp[4];
                    sum1 += r1[2] * ktmp[5];
                    sum2 += r1[2] * ktmp[6];
                    sum3 += r1[2] * ktmp[7];

                    sum0n += r2[1] * ktmp[0];
                    sum1n += r2[1] * ktmp[1];
                    sum2n += r2[1] * ktmp[2];
                    sum3n += r2[1] * ktmp[3];
                    sum0n += r2[2] * ktmp[4];
                    sum1n += r2[2] * ktmp[5];
                    sum2n += r2[2] * ktmp[6];
                    sum3n += r2[2] * ktmp[7];

                    ktmp += 8;

                    ///
                    sum0 += r2[0] * ktmp[0];
                    sum1 += r2[0] * ktmp[1];
                    sum2 += r2[0] * ktmp[2];
                    sum3 += r2[0] * ktmp[3];
                    sum0 += r2[1] * ktmp[4];
                    sum1 += r2[1] * ktmp[5];
                    sum2 += r2[1] * ktmp[6];
                    sum3 += r2[1] * ktmp[7];

                    sum0n += r3[0] * ktmp[0];
                    sum1n += r3[0] * ktmp[1];
                    sum2n += r3[0] * ktmp[2];
                    sum3n += r3[0] * ktmp[3];
                    sum0n += r3[1] * ktmp[4];
                    sum1n += r3[1] * ktmp[5];
                    sum2n += r3[1] * ktmp[6];
                    sum3n += r3[1] * ktmp[7];

                    ktmp += 8;

                    sum0 += r2[2] * ktmp[0];
                    sum1 += r2[2] * ktmp[1];
                    sum2 += r2[2] * ktmp[2];
                    sum3 += r2[2] * ktmp[3];

                    sum0n += r3[2] * ktmp[0];
                    sum1n += r3[2] * ktmp[1];
                    sum2n += r3[2] * ktmp[2];
                    sum3n += r3[2] * ktmp[3];

                    ktmp += 4;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    *(outptr0 + outw) += sum0n;
                    *(outptr1 + outw) += sum1n;
                    *(outptr2 + outw) += sum2n;
                    *(outptr3 + outw) += sum3n;

                    ktmp -= 12*3;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
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
                outptr1 += outw;
                outptr2 += outw;
                outptr3 += outw;
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
                if (nn > 0)
                {
                asm volatile(
                    "0:                         \n"
//                     "pld        [%8, #256]      \n"
                    "vld1.s8    {d0-d3}, [%8]!  \n"// d0=k00 k01  d1=k02 k10  d2=k11 k12  d3=k20 k21

                    "pld        [%5, #128]      \n"
                    "vld1.s8    {d4-d5}, [%5]   \n"// d4=r00 d5=r00n
                    "add        %5, #8          \n"

                    "vdup.s8    d8, d0[0]       \n"
                    "vdup.s8    d9, d0[1]       \n"
                    "vdup.s8    d10, d0[2]      \n"
                    "vdup.s8    d11, d0[3]      \n"

                    "vmull.s8   q8, d4, d8      \n"
                    "vmull.s8   q9, d4, d9      \n"

                    "vext.s8    d24, d4, d5, #1 \n"// d24=r01

                    "vdup.s8    d12, d0[4]      \n"
                    "vdup.s8    d13, d0[5]      \n"

                    "vmull.s8   q10, d4, d10    \n"
                    "vmull.s8   q11, d4, d11    \n"

                    "vdup.s8    d14, d0[6]      \n"
                    "vdup.s8    d15, d0[7]      \n"

                    "vmlal.s8   q8, d24, d12    \n"
                    "vmlal.s8   q9, d24, d13    \n"

                    "vext.s8    d25, d4, d5, #2 \n"// d25=r02

                    "vdup.s8    d8, d1[0]       \n"
                    "vdup.s8    d9, d1[1]       \n"

                    "vmlal.s8   q10, d24, d14   \n"
                    "vmlal.s8   q11, d24, d15   \n"

                    "vdup.s8    d10, d1[2]      \n"
                    "vdup.s8    d11, d1[3]      \n"

                    "vmlal.s8   q8, d25, d8     \n"
                    "vmlal.s8   q9, d25, d9     \n"

                    "pld        [%6, #128]      \n"
                    "vld1.s8    {d6-d7}, [%6]   \n"// d6=r10 d7=r10n
                    "add        %6, #8          \n"

                    "vdup.s8    d12, d1[4]      \n"
                    "vdup.s8    d13, d1[5]      \n"

                    "vmlal.s8   q10, d25, d10   \n"
                    "vmlal.s8   q11, d25, d11   \n"

                    "vdup.s8    d14, d1[6]      \n"
                    "vdup.s8    d15, d1[7]      \n"

                    "vmlal.s8   q8, d6, d12     \n"
                    "vmlal.s8   q9, d6, d13     \n"

                    "vext.s8    d26, d6, d7, #1 \n"// d26=r11

                    "vdup.s8    d8, d2[0]       \n"
                    "vdup.s8    d9, d2[1]       \n"

                    "vmlal.s8   q10, d6, d14    \n"
                    "vmlal.s8   q11, d6, d15    \n"

                    "vdup.s8    d10, d2[2]      \n"
                    "vdup.s8    d11, d2[3]      \n"

                    "vmlal.s8   q8, d26, d8     \n"
                    "vmlal.s8   q9, d26, d9     \n"

                    "vext.s8    d27, d6, d7, #2 \n"// d27=r12

                    "vdup.s8    d12, d2[4]      \n"
                    "vdup.s8    d13, d2[5]      \n"

                    "vmlal.s8   q10, d26, d10   \n"
                    "vmlal.s8   q11, d26, d11   \n"

                    "vdup.s8    d14, d2[6]      \n"
                    "vdup.s8    d15, d2[7]      \n"

                    "vmlal.s8   q8, d27, d12    \n"
                    "vmlal.s8   q9, d27, d13    \n"

                    "pld        [%7, #128]      \n"
                    "vld1.s8    {d4-d5}, [%7]   \n"// d4=r20 d5=r20n
                    "add        %7, #8          \n"

                    "vdup.s8    d8, d3[0]       \n"
                    "vdup.s8    d9, d3[1]       \n"

                    "vmlal.s8   q10, d27, d14   \n"
                    "vmlal.s8   q11, d27, d15   \n"

                    "vdup.s8    d10, d3[2]      \n"
                    "vdup.s8    d11, d3[3]      \n"

                    "vmlal.s8   q8, d4, d8      \n"
                    "vmlal.s8   q9, d4, d9      \n"

                    "vext.s8    d24, d4, d5, #1 \n"// d24=r21

                    "vdup.s8    d12, d3[4]      \n"
                    "vdup.s8    d13, d3[5]      \n"

                    "vmlal.s8   q10, d4, d10    \n"
                    "vmlal.s8   q11, d4, d11    \n"

                    "vdup.s8    d14, d3[6]      \n"
                    "vdup.s8    d15, d3[7]      \n"

                    "vmlal.s8   q8, d24, d12    \n"
                    "vmlal.s8   q9, d24, d13    \n"

//                     "pld        [%8, #128]      \n"
                    "vld1.s8    {d0}, [%8]      \n"
                    "add        %8, #4          \n"

                    "vext.s8    d25, d4, d5, #2 \n"// d25=r22

                    "vdup.s8    d8, d0[0]       \n"
                    "vdup.s8    d9, d0[1]       \n"

                    "vmlal.s8   q10, d24, d14   \n"
                    "vmlal.s8   q11, d24, d15   \n"

                    "vdup.s8    d10, d0[2]      \n"
                    "vdup.s8    d11, d0[3]      \n"

                    "pld        [%1, #256]      \n"
                    "vld1.s32   {d12-d15}, [%1] \n"

                    "vmlal.s8   q8, d25, d8     \n"
                    "vmlal.s8   q9, d25, d9     \n"

                    "pld        [%2, #256]      \n"
                    "vld1.s32   {d0-d3}, [%2]   \n"

                    "vaddw.s16  q6, q6, d16     \n"
                    "vaddw.s16  q7, q7, d17     \n"

                    "vmlal.s8   q10, d25, d10   \n"
                    "vmlal.s8   q11, d25, d11   \n"

                    "vaddw.s16  q0, q0, d18     \n"
                    "vaddw.s16  q1, q1, d19     \n"

                    "pld        [%3, #256]      \n"
                    "vld1.s32   {d16-d19}, [%3]  \n"

                    "vst1.s32   {d12-d15}, [%1]! \n"

                    "pld        [%4, #256]      \n"
                    "vld1.s32   {d4-d7}, [%4]   \n"

                    "vst1.s32   {d0-d3}, [%2]!  \n"

                    "vaddw.s16  q8, q8, d20     \n"
                    "vaddw.s16  q9, q9, d21     \n"
                    "vaddw.s16  q2, q2, d22     \n"
                    "vaddw.s16  q3, q3, d23     \n"

                    "sub        %8, #36         \n"

                    "vst1.s32   {d16-d19}, [%3]! \n"

                    "subs       %0, #1          \n"

                    "vst1.s32   {d4-d7}, [%4]!  \n"

                    "bne        0b              \n"

                    : "=r"(nn),
                      "=r"(outptr0),
                      "=r"(outptr1),
                      "=r"(outptr2),
                      "=r"(outptr3),
                      "=r"(r0),
                      "=r"(r1),
                      "=r"(r2),
                      "=r"(ktmp)
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                        "vld1.s8    {d0}, [%4]          \n"// d0 = r00 r00n
                        "add        %4, #4              \n"
                        "vld1.s8    {d3}, [%5]          \n"// d3 = r10 r10n
                        "add        %5, #4              \n"
                        "vld1.s8    {d6}, [%6]          \n"// d6 = r20 r20n
                        "add        %6, #4              \n"

                        "vld1.s8    {d12-d15}, [%7]!    \n"// d12=k00 k01  d13=k02 k10  d14=k11 k12  d15=k20 k21

                        /// r00 r01 r02
                        "vext.s8    d1, d0, d0, #1      \n"
                        "vext.s8    d2, d0, d0, #2      \n"

                        "vdup.s8    d24, d12[0]         \n"
                        "vdup.s8    d26, d12[1]         \n"
                        "vdup.s8    d25, d12[2]         \n"
                        "vdup.s8    d27, d12[3]         \n"

                        "vsli.64    d0, d0, #32         \n"// d0 = r00 r00
                        "vsli.64    d1, d1, #32         \n"// d1 = r01 r01
                        "vsli.64    d2, d2, #32         \n"// d2 = r02 r02

                        "vsli.64    q12, q13, #32       \n"

                        "vdup.s8    d28, d12[4]         \n"
                        "vdup.s8    d30, d12[5]         \n"
                        "vdup.s8    d29, d12[6]         \n"
                        "vdup.s8    d31, d12[7]         \n"

                        "vsli.64    q14, q15, #32       \n"

                        "vmull.s8   q8, d0, d24         \n"
                        "vmull.s8   q9, d0, d25         \n"

                        "vdup.s8    d24, d13[0]         \n"
                        "vdup.s8    d26, d13[1]         \n"
                        "vdup.s8    d25, d13[2]         \n"
                        "vdup.s8    d27, d13[3]         \n"

                        "vsli.64    q12, q13, #32       \n"

                        "vmlal.s8   q8, d1, d28         \n"
                        "vmlal.s8   q9, d1, d29         \n"

                        /// r10 r11 r12
                        "vext.s8    d4, d3, d3, #1      \n"
                        "vext.s8    d5, d3, d3, #2      \n"

                        "vdup.s8    d28, d13[4]         \n"
                        "vdup.s8    d30, d13[5]         \n"
                        "vdup.s8    d29, d13[6]         \n"
                        "vdup.s8    d31, d13[7]         \n"

                        "vsli.64    d3, d3, #32         \n"// d3 = r10 r10
                        "vsli.64    d4, d4, #32         \n"// d4 = r11 r11
                        "vsli.64    d5, d5, #32         \n"// d5 = r12 r12

                        "vsli.64    q14, q15, #32       \n"

                        "vmlal.s8   q8, d2, d24         \n"
                        "vmlal.s8   q9, d2, d25         \n"

                        "vdup.s8    d24, d14[0]         \n"
                        "vdup.s8    d26, d14[1]         \n"
                        "vdup.s8    d25, d14[2]         \n"
                        "vdup.s8    d27, d14[3]         \n"

                        "vsli.64    q12, q13, #32       \n"

                        "vmlal.s8   q8, d3, d28         \n"
                        "vmlal.s8   q9, d3, d29         \n"

                        "vdup.s8    d28, d14[4]         \n"
                        "vdup.s8    d30, d14[5]         \n"
                        "vdup.s8    d29, d14[6]         \n"
                        "vdup.s8    d31, d14[7]         \n"

                        "vsli.64    q14, q15, #32       \n"

                        "vmlal.s8   q8, d4, d24         \n"
                        "vmlal.s8   q9, d4, d25         \n"

                        /// r20 r21 r22
                        "vext.s8    d7, d6, d6, #1      \n"
                        "vext.s8    d8, d6, d6, #2      \n"

                        "vdup.s8    d24, d15[0]         \n"
                        "vdup.s8    d26, d15[1]         \n"
                        "vdup.s8    d25, d15[2]         \n"
                        "vdup.s8    d27, d15[3]         \n"

                        "vsli.64    d6, d6, #32         \n"// d6 = r20 r20
                        "vsli.64    d7, d7, #32         \n"// d7 = r21 r21
                        "vsli.64    d8, d8, #32         \n"// d8 = r22 r22

                        "vsli.64    q12, q13, #32       \n"

                        "vmlal.s8   q8, d5, d28         \n"
                        "vmlal.s8   q9, d5, d29         \n"

                        "vdup.s8    d28, d15[4]         \n"
                        "vdup.s8    d30, d15[5]         \n"
                        "vdup.s8    d29, d15[6]         \n"
                        "vdup.s8    d31, d15[7]         \n"

                        "vsli.64    q14, q15, #32       \n"

                        "vld1.s8    {d12}, [%7]         \n"// d12=k22
                        "add        %7, #4              \n"

                        "vmlal.s8   q8, d6, d24         \n"
                        "vmlal.s8   q9, d6, d25         \n"

                        "vdup.s8    d24, d12[0]         \n"
                        "vdup.s8    d26, d12[1]         \n"
                        "vdup.s8    d25, d12[2]         \n"
                        "vdup.s8    d27, d12[3]         \n"

                        "vsli.64    q12, q13, #32       \n"

                        "vmlal.s8   q8, d7, d28         \n"
                        "vmlal.s8   q9, d7, d29         \n"

                        "vld1.s32   {d0-d1}, [%0]       \n"
                        "vld1.s32   {d2-d3}, [%1]       \n"

                        "vmlal.s8   q8, d8, d24         \n"
                        "vmlal.s8   q9, d8, d25         \n"

                        ///
                        "vld1.s32   {d4-d5}, [%2]       \n"
                        "vld1.s32   {d6-d7}, [%3]       \n"

                        "vaddw.s16  q0, q0, d16         \n"
                        "vaddw.s16  q1, q1, d17         \n"
                        "vaddw.s16  q2, q2, d18         \n"
                        "vaddw.s16  q3, q3, d19         \n"

                        "vst1.s32   {d0-d1}, [%0]!      \n"
                        "vst1.s32   {d2-d3}, [%1]!      \n"

                        "sub        %7, #36             \n"

                        "vst1.s32   {d4-d5}, [%2]!      \n"
                        "vst1.s32   {d6-d7}, [%3]!      \n"

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
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15"
                    );
                }
#endif
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[]}, [%4]!       \n"
                        "vld1.s8    {d1[]}, [%4]!       \n"

                        "vld1.s8    {d4-d7}, [%7]!      \n"// d4 d5 d6 d7 = 0~7

                        "vsli.64    d0, d1, #32         \n"// d0 = 00 01

                        "vld1.s8    {d2[]}, [%4]        \n"
                        "sub        %4, %4, #2          \n"
                        "vld1.s8    {d3[]}, [%5]!       \n"

                        "vsli.64    d2, d3, #32         \n"// d2 = 02 10

                        "vmull.s8   q8, d0, d4          \n"

                        "vld1.s8    {d0[]}, [%5]!       \n"
                        "vld1.s8    {d1[]}, [%5]        \n"
                        "sub        %5, %5, #2          \n"

                        "vsli.64    d0, d1, #32         \n"// d0 = 11 12

                        "vmlal.s8   q8, d2, d5          \n"

                        "vld1.s8    {d2[]}, [%6]!       \n"
                        "vld1.s8    {d3[]}, [%6]!       \n"

                        "vsli.64    d2, d3, #32         \n"// d2 = 20 21

                        "vmlal.s8   q8, d0, d6          \n"

                        "vld1.s8    {d0[]}, [%6]        \n"
                        "sub        %6, %6, #2          \n"
                        "veor       d1, d1, d1          \n"

                        "vld1.s8    {d4}, [%7]          \n"// d4 = 0~4 xxxx
                        "sub        %7, #32             \n"

                        "vsli.64    d0, d1, #32         \n"// d0 = 22 zero

                        "vmlal.s8   q8, d2, d7          \n"

                        "vld1.s32   {d20[0]}, [%0]      \n"

                        "vmlal.s8   q8, d0, d4          \n"

                        "vld1.s32   {d20[1]}, [%1]      \n"

                        "vadd.s16   d16, d16, d17       \n"

                        "vld1.s32   {d21[0]}, [%2]      \n"
                        "vld1.s32   {d21[1]}, [%3]      \n"

                        "vaddw.s16  q10, q10, d16       \n"

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
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q10"
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

            int8x8_t _k00 = vdup_n_s8(ktmp[0]);
            int8x8_t _k01 = vdup_n_s8(ktmp[1]);
            int8x8_t _k02 = vdup_n_s8(ktmp[2]);
            int8x8_t _k10 = vdup_n_s8(ktmp[3]);
            int8x8_t _k11 = vdup_n_s8(ktmp[4]);
            int8x8_t _k12 = vdup_n_s8(ktmp[5]);
            int8x8_t _k20 = vdup_n_s8(ktmp[6]);
            int8x8_t _k21 = vdup_n_s8(ktmp[7]);
            int8x8_t _k22 = vdup_n_s8(ktmp[8]);

            int i = 0;

            for (; i+1 < outh; i+=2)
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

                    "pld        [%3, #128]      \n"
                    "vld1.s8    {d4-d5}, [%3]   \n"// d4=r00 d5=r00n
                    "add        %3, #8          \n"

                    "pld        [%6, #128]      \n"
                    "vld1.s8    {d6-d7}, [%6]   \n"// d6=r30 d7=r30n
                    "add        %6, #8          \n"

                    "vext.s8    d8, d4, d5, #1  \n"// d8=r01
                    "vext.s8    d10, d6, d7, #1 \n"// d10=r31

                    "vmull.s8   q8, d4, %P14    \n"
                    "vmull.s8   q9, d6, %P20    \n"

                    "vext.s8    d9, d4, d5, #2  \n"// d9=r02
                    "vext.s8    d11, d6, d7, #2 \n"// d11=r32

                    "vmlal.s8   q8, d8, %P15    \n"
                    "vmlal.s8   q9, d10, %P21   \n"

                    "pld        [%4, #128]      \n"
                    "vld1.s8    {d4-d5}, [%4]   \n"// d4=r10 d5=r10n
                    "add        %4, #8          \n"

                    "vmlal.s8   q8, d9, %P16    \n"
                    "vmlal.s8   q9, d11, %P22   \n"

                    "vext.s8    d8, d4, d5, #1  \n"// d8=r11

                    "vmlal.s8   q8, d4, %P17    \n"
                    "vmlal.s8   q9, d4, %P14    \n"

                    "vext.s8    d9, d4, d5, #2  \n"// d9=r12

                    "vmlal.s8   q8, d8, %P18    \n"
                    "vmlal.s8   q9, d8, %P15    \n"

                    "pld        [%5, #128]      \n"
                    "vld1.s8    {d6-d7}, [%5]   \n"// d6=r20 d7=r20n
                    "add        %5, #8          \n"

                    "vmlal.s8   q8, d9, %P19    \n"
                    "vmlal.s8   q9, d9, %P16    \n"

                    "vext.s8    d10, d6, d7, #1 \n"// d10=r21

                    "vmlal.s8   q8, d6, %P20    \n"
                    "vmlal.s8   q9, d6, %P17    \n"

                    "vext.s8    d11, d6, d7, #2 \n"// d11=r22

                    "vmlal.s8   q8, d10, %P21   \n"
                    "vmlal.s8   q9, d10, %P18   \n"

                    "pld        [%1, #256]      \n"
                    "vld1.s32   {d0-d3}, [%1]   \n"

                    "vmlal.s8   q8, d11, %P22   \n"
                    "vmlal.s8   q9, d11, %P19   \n"

                    "pld        [%2, #256]      \n"
                    "vld1.s32   {d12-d15}, [%2] \n"

                    "vaddw.s16  q0, q0, d16     \n"
                    "vaddw.s16  q1, q1, d17     \n"
                    "vaddw.s16  q6, q6, d18     \n"
                    "vaddw.s16  q7, q7, d19     \n"

                    "vst1.s32   {d0-d3}, [%1]!  \n"

                    "subs       %0, #1          \n"

                    "vst1.s32   {d12-d15}, [%2]! \n"

                    "bne        0b              \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr0n),   // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2),         // %5
                      "=r"(r3)          // %6
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr0n),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "6"(r3),
                      "w"(_k00),        // %14
                      "w"(_k01),        // %15
                      "w"(_k02),        // %16
                      "w"(_k10),        // %17
                      "w"(_k11),        // %18
                      "w"(_k12),        // %19
                      "w"(_k20),        // %20
                      "w"(_k21),        // %21
                      "w"(_k22)         // %22
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"
                );
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(

                        "vld1.s8    {d12}, [%6]!        \n"// d12=0~7

                        /// r00 r01 r02
                        "vld1.s8    {d0}, [%2]          \n"// d0 = r00 r00n
                        "add        %2, #4              \n"

                        "vdup.s8    d24, d12[0]         \n"

                        "vext.s8    d1, d0, d0, #1      \n"
                        "vext.s8    d2, d0, d0, #2      \n"

                        /// r10 r11 r12
                        "vld1.s8    {d3}, [%3]          \n"// d3 = r10 r10n
                        "add        %3, #4              \n"

                        "vdup.s8    d25, d12[1]         \n"

                        "vext.s8    d4, d3, d3, #1      \n"
                        "vext.s8    d5, d3, d3, #2      \n"

                        "vdup.s8    d26, d12[2]         \n"

                        "vsli.64    d0, d3, #32         \n"// d0 = r00 r10
                        "vsli.64    d1, d4, #32         \n"// d1 = r01 r11
                        "vsli.64    d2, d5, #32         \n"// d2 = r02 r12

                        "vmull.s8   q8, d0, d24         \n"
                        "vmull.s8   q9, d1, d25         \n"

                        /// r20 r21 r22
                        "vld1.s8    {d6}, [%4]          \n"// d6 = r20 r20n
                        "add        %4, #4              \n"

                        "vdup.s8    d27, d12[3]         \n"
                        "vdup.s8    d28, d12[4]         \n"

                        "vext.s8    d7, d6, d6, #1      \n"
                        "vext.s8    d8, d6, d6, #2      \n"

                        "vsli.64    d3, d6, #32         \n"// d3 = r10 r20

                        "vdup.s8    d29, d12[5]         \n"

                        "vmlal.s8   q8, d2, d26         \n"
                        "vmlal.s8   q9, d3, d27         \n"

                        "vsli.64    d4, d7, #32         \n"// d4 = r11 r21
                        "vsli.64    d5, d8, #32         \n"// d5 = r12 r22

                        /// r30 r31 r32
                        "vld1.s8    {d9}, [%5]          \n"// d9 = r30 r30n
                        "add        %5, #4              \n"

                        "vdup.s8    d30, d12[6]         \n"
                        "vdup.s8    d31, d12[7]         \n"

                        "vmlal.s8   q8, d4, d28         \n"
                        "vmlal.s8   q9, d5, d29         \n"

                        "vext.s8    d10, d9, d9, #1     \n"
                        "vext.s8    d11, d9, d9, #2     \n"

                        "vsli.64    d6, d9, #32         \n"// d6 = r20 r30
                        "vsli.64    d7, d10, #32        \n"// d7 = r21 r31

                        "vmlal.s8   q8, d6, d30         \n"
                        "vmlal.s8   q9, d7, d31         \n"

                        "vld1.s8    {d13[]}, [%6]!      \n"
                        "vsli.64    d8, d11, #32        \n"// d8 = r22 r32

                        "vmlal.s8   q8, d8, d13         \n"

                        ///
                        "vld1.s32   {d0-d1}, [%0]       \n"

                        "vadd.s16   q8, q8, q9          \n"

                        "vld1.s32   {d2-d3}, [%1]       \n"

                        "vaddw.s16  q0, q0, d16         \n"
                        "vaddw.s16  q1, q1, d17         \n"

                        "sub        %6, #9              \n"

                        "vst1.s32   {d0-d1}, [%0]!      \n"
                        "vst1.s32   {d2-d3}, [%1]!      \n"

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
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
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
                        "vld1.s8    {d0[7]}, [%4]!  \n"// d0=r

                        "vld1.s8    {d4[]}, [%4]    \n"// d4=r22
                        "sub        %4, #2          \n"

                        "vext.s8    d1, d0, d4, #3  \n"

                        "vld1.s8    {d1[6]}, [%5]!  \n"
                        "vld1.s8    {d1[7]}, [%5]!  \n"// d1=rn

                        "vld1.s8    {d2}, [%6]!     \n"// d2=k01234567

                        "vld1.s8    {d5[]}, [%5]    \n"// d5=r32
                        "sub        %5, #2          \n"

                        "veor       d3, d3          \n"

                        "vmull.s8   q8, d0, d2      \n"
                        "vmull.s8   q9, d1, d2      \n"

                        "vld1.s8    {d3[0]}, [%6]   \n"// d3=k8 ... zeros
                        "sub        %6, #8          \n"

                        "vmlal.s8   q8, d4, d3      \n"
                        "vmlal.s8   q9, d5, d3      \n"

                        "vld1.s32   {d6[0]}, [%0]   \n"

                        "vadd.s16   d16, d16, d17   \n"
                        "vadd.s16   d18, d18, d19   \n"

                        "vld1.s32   {d6[1]}, [%1]   \n"

                        "vpadd.s16  d16, d16, d18   \n"
                        "vpadal.s16 d6, d16         \n"

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
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
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
                if (nn > 0)
                {
                asm volatile(
                    "0:                         \n"

                    "pld        [%2, #128]      \n"
                    "vld1.s8    {d4-d5}, [%2]   \n"// d4=r00 d5=r00n
                    "add        %2, #8          \n"

                    "vext.s8    d8, d4, d5, #1  \n"// d8=r01

                    "vmull.s8   q8, d4, %P10    \n"

                    "vext.s8    d9, d4, d5, #2  \n"// d9=r02

                    "vmull.s8   q9, d8, %P11    \n"

                    "pld        [%3, #128]      \n"
                    "vld1.s8    {d6-d7}, [%3]   \n"// d6=r10 d7=r10n
                    "add        %3, #8          \n"

                    "vmlal.s8   q8, d9, %P12    \n"

                    "vext.s8    d10, d6, d7, #1 \n"// d10=r11

                    "vmlal.s8   q9, d6, %P13    \n"

                    "vext.s8    d11, d6, d7, #2 \n"// d11=r12

                    "vmlal.s8   q8, d10, %P14   \n"

                    "pld        [%4, #128]      \n"
                    "vld1.s8    {d4-d5}, [%4]   \n"// d4=r20 d5=r20n
                    "add        %4, #8          \n"

                    "vmlal.s8   q9, d11, %P15   \n"

                    "vext.s8    d8, d4, d5, #1  \n"// d8=r21

                    "vmlal.s8   q8, d4, %P16    \n"

                    "vext.s8    d9, d4, d5, #2  \n"// d9=r22

                    "vmlal.s8   q9, d8, %P17    \n"

                    "vmlal.s8   q8, d9, %P18    \n"

                    "pld        [%1, #256]      \n"
                    "vld1.s32   {d0-d3}, [%1]   \n"

                    "vadd.s16   q8, q8, q9      \n"

                    "vaddw.s16  q0, q0, d16     \n"
                    "vaddw.s16  q1, q1, d17     \n"

                    "subs       %0, #1          \n"

                    "vst1.s32   {d0-d3}, [%1]!  \n"

                    "bne        0b              \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2)          // %4
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k00),        // %10
                      "w"(_k01),        // %11
                      "w"(_k02),        // %12
                      "w"(_k10),        // %13
                      "w"(_k11),        // %14
                      "w"(_k12),        // %15
                      "w"(_k20),        // %16
                      "w"(_k21),        // %17
                      "w"(_k22)         // %18
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9"
                );
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
