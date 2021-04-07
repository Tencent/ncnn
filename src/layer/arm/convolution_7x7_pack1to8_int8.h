// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv7x7s2_pack1to8_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        int32x4_t _bias0 = vdupq_n_s32(0);
        out0.fill(_bias0, _bias0);

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const signed char* r0 = img0.row<const signed char>(0);
            const signed char* r1 = img0.row<const signed char>(1);
            const signed char* r2 = img0.row<const signed char>(2);
            const signed char* r3 = img0.row<const signed char>(3);
            const signed char* r4 = img0.row<const signed char>(4);
            const signed char* r5 = img0.row<const signed char>(5);
            const signed char* r6 = img0.row<const signed char>(6);

            const signed char* k0 = kernel.channel(p).row<const signed char>(q);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j < outw; j++)
                {
                    int32x4_t _sum0 = vld1q_s32(outptr0);
                    int32x4_t _sum1 = vld1q_s32(outptr0 + 4);

                    int8x8_t _r00 = vld1_dup_s8(r0);
                    int8x8_t _r01 = vld1_dup_s8(r0 + 1);
                    int8x8_t _r02 = vld1_dup_s8(r0 + 2);
                    int8x8_t _r03 = vld1_dup_s8(r0 + 3);

                    int8x8_t _k00 = vld1_s8(k0);
                    int8x8_t _k01 = vld1_s8(k0 + 8);
                    int8x8_t _k02 = vld1_s8(k0 + 16);
                    int8x8_t _k03 = vld1_s8(k0 + 24);
                    k0 += 32;

                    int16x8_t _s00 = vmull_s8(_r00, _k00);
                    int16x8_t _s01 = vmull_s8(_r01, _k01);
                    int16x8_t _s02 = vmull_s8(_r02, _k02);
                    int16x8_t _s03 = vmull_s8(_r03, _k03);

                    int8x8_t _r04 = vld1_dup_s8(r0 + 4);
                    int8x8_t _r05 = vld1_dup_s8(r0 + 5);
                    int8x8_t _r06 = vld1_dup_s8(r0 + 6);
                    int8x8_t _r10 = vld1_dup_s8(r1);

                    int8x8_t _k04 = vld1_s8(k0);
                    int8x8_t _k05 = vld1_s8(k0 + 8);
                    int8x8_t _k06 = vld1_s8(k0 + 16);
                    int8x8_t _k10 = vld1_s8(k0 + 24);
                    k0 += 32;

                    _s00 = vmlal_s8(_s00, _r04, _k04);
                    _s01 = vmlal_s8(_s01, _r05, _k05);
                    _s02 = vmlal_s8(_s02, _r06, _k06);
                    _s03 = vmlal_s8(_s03, _r10, _k10);

                    int32x4_t _s0001l = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _s0001h = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _s0203l = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s03));
                    int32x4_t _s0203h = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s03));
                    _sum0 = vaddq_s32(_sum0, _s0001l);
                    _sum1 = vaddq_s32(_sum1, _s0001h);
                    _sum0 = vaddq_s32(_sum0, _s0203l);
                    _sum1 = vaddq_s32(_sum1, _s0203h);

                    int8x8_t _r11 = vld1_dup_s8(r1 + 1);
                    int8x8_t _r12 = vld1_dup_s8(r1 + 2);
                    int8x8_t _r13 = vld1_dup_s8(r1 + 3);
                    int8x8_t _r14 = vld1_dup_s8(r1 + 4);

                    int8x8_t _k11 = vld1_s8(k0);
                    int8x8_t _k12 = vld1_s8(k0 + 8);
                    int8x8_t _k13 = vld1_s8(k0 + 16);
                    int8x8_t _k14 = vld1_s8(k0 + 24);
                    k0 += 32;

                    int16x8_t _s11 = vmull_s8(_r11, _k11);
                    int16x8_t _s12 = vmull_s8(_r12, _k12);
                    int16x8_t _s13 = vmull_s8(_r13, _k13);
                    int16x8_t _s14 = vmull_s8(_r14, _k14);

                    int8x8_t _r15 = vld1_dup_s8(r1 + 5);
                    int8x8_t _r16 = vld1_dup_s8(r1 + 6);
                    int8x8_t _r20 = vld1_dup_s8(r2);
                    int8x8_t _r21 = vld1_dup_s8(r2 + 1);

                    int8x8_t _k15 = vld1_s8(k0);
                    int8x8_t _k16 = vld1_s8(k0 + 8);
                    int8x8_t _k20 = vld1_s8(k0 + 16);
                    int8x8_t _k21 = vld1_s8(k0 + 24);
                    k0 += 32;

                    _s11 = vmlal_s8(_s11, _r15, _k15);
                    _s12 = vmlal_s8(_s12, _r16, _k16);
                    _s13 = vmlal_s8(_s13, _r20, _k20);
                    _s14 = vmlal_s8(_s14, _r21, _k21);

                    int32x4_t _s1112l = vaddl_s16(vget_low_s16(_s11), vget_low_s16(_s12));
                    int32x4_t _s1112h = vaddl_s16(vget_high_s16(_s11), vget_high_s16(_s12));
                    int32x4_t _s1314l = vaddl_s16(vget_low_s16(_s13), vget_low_s16(_s14));
                    int32x4_t _s1314h = vaddl_s16(vget_high_s16(_s13), vget_high_s16(_s14));
                    _sum0 = vaddq_s32(_sum0, _s1112l);
                    _sum1 = vaddq_s32(_sum1, _s1112h);
                    _sum0 = vaddq_s32(_sum0, _s1314l);
                    _sum1 = vaddq_s32(_sum1, _s1314h);

                    int8x8_t _r22 = vld1_dup_s8(r2 + 2);
                    int8x8_t _r23 = vld1_dup_s8(r2 + 3);
                    int8x8_t _r24 = vld1_dup_s8(r2 + 4);
                    int8x8_t _r25 = vld1_dup_s8(r2 + 5);

                    int8x8_t _k22 = vld1_s8(k0);
                    int8x8_t _k23 = vld1_s8(k0 + 8);
                    int8x8_t _k24 = vld1_s8(k0 + 16);
                    int8x8_t _k25 = vld1_s8(k0 + 24);
                    k0 += 32;

                    int16x8_t _s22 = vmull_s8(_r22, _k22);
                    int16x8_t _s23 = vmull_s8(_r23, _k23);
                    int16x8_t _s24 = vmull_s8(_r24, _k24);
                    int16x8_t _s25 = vmull_s8(_r25, _k25);

                    int8x8_t _r26 = vld1_dup_s8(r2 + 6);
                    int8x8_t _r30 = vld1_dup_s8(r3);
                    int8x8_t _r31 = vld1_dup_s8(r3 + 1);
                    int8x8_t _r32 = vld1_dup_s8(r3 + 2);

                    int8x8_t _k26 = vld1_s8(k0);
                    int8x8_t _k30 = vld1_s8(k0 + 8);
                    int8x8_t _k31 = vld1_s8(k0 + 16);
                    int8x8_t _k32 = vld1_s8(k0 + 24);
                    k0 += 32;

                    _s22 = vmlal_s8(_s22, _r26, _k26);
                    _s23 = vmlal_s8(_s23, _r30, _k30);
                    _s24 = vmlal_s8(_s24, _r31, _k31);
                    _s25 = vmlal_s8(_s25, _r32, _k32);

                    int32x4_t _s2223l = vaddl_s16(vget_low_s16(_s22), vget_low_s16(_s23));
                    int32x4_t _s2223h = vaddl_s16(vget_high_s16(_s22), vget_high_s16(_s23));
                    int32x4_t _s2425l = vaddl_s16(vget_low_s16(_s24), vget_low_s16(_s25));
                    int32x4_t _s2425h = vaddl_s16(vget_high_s16(_s24), vget_high_s16(_s25));
                    _sum0 = vaddq_s32(_sum0, _s2223l);
                    _sum1 = vaddq_s32(_sum1, _s2223h);
                    _sum0 = vaddq_s32(_sum0, _s2425l);
                    _sum1 = vaddq_s32(_sum1, _s2425h);

                    int8x8_t _r33 = vld1_dup_s8(r3 + 3);
                    int8x8_t _r34 = vld1_dup_s8(r3 + 4);
                    int8x8_t _r35 = vld1_dup_s8(r3 + 5);
                    int8x8_t _r36 = vld1_dup_s8(r3 + 6);

                    int8x8_t _k33 = vld1_s8(k0);
                    int8x8_t _k34 = vld1_s8(k0 + 8);
                    int8x8_t _k35 = vld1_s8(k0 + 16);
                    int8x8_t _k36 = vld1_s8(k0 + 24);
                    k0 += 32;

                    int16x8_t _s33 = vmull_s8(_r33, _k33);
                    int16x8_t _s34 = vmull_s8(_r34, _k34);
                    int16x8_t _s35 = vmull_s8(_r35, _k35);
                    int16x8_t _s36 = vmull_s8(_r36, _k36);

                    int8x8_t _r40 = vld1_dup_s8(r4);
                    int8x8_t _r41 = vld1_dup_s8(r4 + 1);
                    int8x8_t _r42 = vld1_dup_s8(r4 + 2);
                    int8x8_t _r43 = vld1_dup_s8(r4 + 3);

                    int8x8_t _k40 = vld1_s8(k0);
                    int8x8_t _k41 = vld1_s8(k0 + 8);
                    int8x8_t _k42 = vld1_s8(k0 + 16);
                    int8x8_t _k43 = vld1_s8(k0 + 24);
                    k0 += 32;

                    _s33 = vmlal_s8(_s33, _r40, _k40);
                    _s34 = vmlal_s8(_s34, _r41, _k41);
                    _s35 = vmlal_s8(_s35, _r42, _k42);
                    _s36 = vmlal_s8(_s36, _r43, _k43);

                    int32x4_t _s3334l = vaddl_s16(vget_low_s16(_s33), vget_low_s16(_s34));
                    int32x4_t _s3334h = vaddl_s16(vget_high_s16(_s33), vget_high_s16(_s34));
                    int32x4_t _s3536l = vaddl_s16(vget_low_s16(_s35), vget_low_s16(_s36));
                    int32x4_t _s3536h = vaddl_s16(vget_high_s16(_s35), vget_high_s16(_s36));
                    _sum0 = vaddq_s32(_sum0, _s3334l);
                    _sum1 = vaddq_s32(_sum1, _s3334h);
                    _sum0 = vaddq_s32(_sum0, _s3536l);
                    _sum1 = vaddq_s32(_sum1, _s3536h);

                    int8x8_t _r44 = vld1_dup_s8(r4 + 4);
                    int8x8_t _r45 = vld1_dup_s8(r4 + 5);
                    int8x8_t _r46 = vld1_dup_s8(r4 + 6);
                    int8x8_t _r50 = vld1_dup_s8(r5);

                    int8x8_t _k44 = vld1_s8(k0);
                    int8x8_t _k45 = vld1_s8(k0 + 8);
                    int8x8_t _k46 = vld1_s8(k0 + 16);
                    int8x8_t _k50 = vld1_s8(k0 + 24);
                    k0 += 32;

                    int16x8_t _s44 = vmull_s8(_r44, _k44);
                    int16x8_t _s45 = vmull_s8(_r45, _k45);
                    int16x8_t _s46 = vmull_s8(_r46, _k46);
                    int16x8_t _s50 = vmull_s8(_r50, _k50);

                    int8x8_t _r51 = vld1_dup_s8(r5 + 1);
                    int8x8_t _r52 = vld1_dup_s8(r5 + 2);
                    int8x8_t _r53 = vld1_dup_s8(r5 + 3);
                    int8x8_t _r54 = vld1_dup_s8(r5 + 4);

                    int8x8_t _k51 = vld1_s8(k0);
                    int8x8_t _k52 = vld1_s8(k0 + 8);
                    int8x8_t _k53 = vld1_s8(k0 + 16);
                    int8x8_t _k54 = vld1_s8(k0 + 24);
                    k0 += 32;

                    _s44 = vmlal_s8(_s44, _r51, _k51);
                    _s45 = vmlal_s8(_s45, _r52, _k52);
                    _s46 = vmlal_s8(_s46, _r53, _k53);
                    _s50 = vmlal_s8(_s50, _r54, _k54);

                    int32x4_t _s4445l = vaddl_s16(vget_low_s16(_s44), vget_low_s16(_s45));
                    int32x4_t _s4445h = vaddl_s16(vget_high_s16(_s44), vget_high_s16(_s45));
                    int32x4_t _s4650l = vaddl_s16(vget_low_s16(_s46), vget_low_s16(_s50));
                    int32x4_t _s4650h = vaddl_s16(vget_high_s16(_s46), vget_high_s16(_s50));
                    _sum0 = vaddq_s32(_sum0, _s4445l);
                    _sum1 = vaddq_s32(_sum1, _s4445h);
                    _sum0 = vaddq_s32(_sum0, _s4650l);
                    _sum1 = vaddq_s32(_sum1, _s4650h);

                    int8x8_t _r55 = vld1_dup_s8(r5 + 5);
                    int8x8_t _r56 = vld1_dup_s8(r5 + 6);
                    int8x8_t _r60 = vld1_dup_s8(r6);
                    int8x8_t _r61 = vld1_dup_s8(r6 + 1);

                    int8x8_t _k55 = vld1_s8(k0);
                    int8x8_t _k56 = vld1_s8(k0 + 8);
                    int8x8_t _k60 = vld1_s8(k0 + 16);
                    int8x8_t _k61 = vld1_s8(k0 + 24);
                    k0 += 32;

                    int16x8_t _s55 = vmull_s8(_r55, _k55);
                    int16x8_t _s56 = vmull_s8(_r56, _k56);
                    int16x8_t _s60 = vmull_s8(_r60, _k60);
                    int16x8_t _s61 = vmull_s8(_r61, _k61);

                    int8x8_t _r62 = vld1_dup_s8(r6 + 2);
                    int8x8_t _r63 = vld1_dup_s8(r6 + 3);
                    int8x8_t _r64 = vld1_dup_s8(r6 + 4);
                    int8x8_t _r65 = vld1_dup_s8(r6 + 5);

                    int8x8_t _k62 = vld1_s8(k0);
                    int8x8_t _k63 = vld1_s8(k0 + 8);
                    int8x8_t _k64 = vld1_s8(k0 + 16);
                    int8x8_t _k65 = vld1_s8(k0 + 24);
                    k0 += 32;

                    _s55 = vmlal_s8(_s55, _r62, _k62);
                    _s56 = vmlal_s8(_s56, _r63, _k63);
                    _s60 = vmlal_s8(_s60, _r64, _k64);
                    _s61 = vmlal_s8(_s61, _r65, _k65);

                    int32x4_t _s5556l = vaddl_s16(vget_low_s16(_s55), vget_low_s16(_s56));
                    int32x4_t _s5556h = vaddl_s16(vget_high_s16(_s55), vget_high_s16(_s56));
                    int32x4_t _s6061l = vaddl_s16(vget_low_s16(_s60), vget_low_s16(_s61));
                    int32x4_t _s6061h = vaddl_s16(vget_high_s16(_s60), vget_high_s16(_s61));
                    _sum0 = vaddq_s32(_sum0, _s5556l);
                    _sum1 = vaddq_s32(_sum1, _s5556h);
                    _sum0 = vaddq_s32(_sum0, _s6061l);
                    _sum1 = vaddq_s32(_sum1, _s6061h);

                    int8x8_t _r66 = vld1_dup_s8(r6 + 6);
                    int8x8_t _k66 = vld1_s8(k0);
                    k0 -= 384;

                    int16x8_t _s66 = vmull_s8(_r66, _k66);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s66));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s66));

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                    outptr0 += 8;
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
