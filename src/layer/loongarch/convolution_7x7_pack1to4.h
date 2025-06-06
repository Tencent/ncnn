// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void conv7x7s2_pack1to4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        __m128 _bias0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);
            const float* r3 = img0.row(3);
            const float* r4 = img0.row(4);
            const float* r5 = img0.row(5);
            const float* r6 = img0.row(6);

            const float* kptr = kernel.channel(p).row(q);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    __m128 _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    __m128 _sum2 = (__m128)__lsx_vld(outptr0 + 4 * 2, 0);
                    __m128 _sum3 = (__m128)__lsx_vld(outptr0 + 4 * 3, 0);

                    __m128 _k00 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k01 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k02 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k03 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k04 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k05 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k06 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);
                    __m128i _r0nn = __lsx_vld(r0 + 8, 0);

                    __m128 _r00 = (__m128)__lsx_vreplvei_w(_r0, 0);
                    __m128 _r01 = (__m128)__lsx_vreplvei_w(_r0, 1);
                    __m128 _r02 = (__m128)__lsx_vreplvei_w(_r0, 2);
                    __m128 _r03 = (__m128)__lsx_vreplvei_w(_r0, 3);
                    __m128 _r04 = (__m128)__lsx_vreplvei_w(_r0n, 0);
                    __m128 _r05 = (__m128)__lsx_vreplvei_w(_r0n, 1);
                    __m128 _r06 = (__m128)__lsx_vreplvei_w(_r0n, 2);
                    __m128 _r07 = (__m128)__lsx_vreplvei_w(_r0n, 3);
                    __m128 _r08 = (__m128)__lsx_vreplvei_w(_r0nn, 0);
                    __m128 _r09 = (__m128)__lsx_vreplvei_w(_r0nn, 1);
                    __m128 _r0a = (__m128)__lsx_vreplvei_w(_r0nn, 2);
                    __m128 _r0b = (__m128)__lsx_vreplvei_w(_r0nn, 3);
                    __m128 _r0c = __lsx_vreplfr2vr_s(r0[12]);

                    _sum0 = __lsx_vfmadd_s(_k00, _r00, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k00, _r02, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k00, _r04, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k00, _r06, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k01, _r01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k01, _r03, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k01, _r05, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k01, _r07, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k02, _r02, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k02, _r04, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k02, _r06, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k02, _r08, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k03, _r03, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k03, _r05, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k03, _r07, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k03, _r09, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k04, _r04, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k04, _r06, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k04, _r08, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k04, _r0a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k05, _r05, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k05, _r07, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k05, _r09, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k05, _r0b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k06, _r06, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k06, _r08, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k06, _r0a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k06, _r0c, _sum3);

                    __m128 _k10 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k11 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k12 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k13 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k14 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k15 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k16 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r1n = __lsx_vld(r1 + 4, 0);
                    __m128i _r1nn = __lsx_vld(r1 + 8, 0);

                    __m128 _r10 = (__m128)__lsx_vreplvei_w(_r1, 0);
                    __m128 _r11 = (__m128)__lsx_vreplvei_w(_r1, 1);
                    __m128 _r12 = (__m128)__lsx_vreplvei_w(_r1, 2);
                    __m128 _r13 = (__m128)__lsx_vreplvei_w(_r1, 3);
                    __m128 _r14 = (__m128)__lsx_vreplvei_w(_r1n, 0);
                    __m128 _r15 = (__m128)__lsx_vreplvei_w(_r1n, 1);
                    __m128 _r16 = (__m128)__lsx_vreplvei_w(_r1n, 2);
                    __m128 _r17 = (__m128)__lsx_vreplvei_w(_r1n, 3);
                    __m128 _r18 = (__m128)__lsx_vreplvei_w(_r1nn, 0);
                    __m128 _r19 = (__m128)__lsx_vreplvei_w(_r1nn, 1);
                    __m128 _r1a = (__m128)__lsx_vreplvei_w(_r1nn, 2);
                    __m128 _r1b = (__m128)__lsx_vreplvei_w(_r1nn, 3);
                    __m128 _r1c = __lsx_vreplfr2vr_s(r1[12]);

                    _sum0 = __lsx_vfmadd_s(_k10, _r10, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k10, _r12, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k10, _r14, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k10, _r16, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k11, _r11, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k11, _r13, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k11, _r15, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k11, _r17, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k12, _r12, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k12, _r14, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k12, _r16, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k12, _r18, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k13, _r13, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k13, _r15, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k13, _r17, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k13, _r19, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k14, _r14, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k14, _r16, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k14, _r18, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k14, _r1a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k15, _r15, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k15, _r17, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k15, _r19, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k15, _r1b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k16, _r16, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k16, _r18, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k16, _r1a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k16, _r1c, _sum3);

                    __m128 _k20 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k21 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k22 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k23 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k24 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k25 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k26 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r2n = __lsx_vld(r2 + 4, 0);
                    __m128i _r2nn = __lsx_vld(r2 + 8, 0);

                    __m128 _r20 = (__m128)__lsx_vreplvei_w(_r2, 0);
                    __m128 _r21 = (__m128)__lsx_vreplvei_w(_r2, 1);
                    __m128 _r22 = (__m128)__lsx_vreplvei_w(_r2, 2);
                    __m128 _r23 = (__m128)__lsx_vreplvei_w(_r2, 3);
                    __m128 _r24 = (__m128)__lsx_vreplvei_w(_r2n, 0);
                    __m128 _r25 = (__m128)__lsx_vreplvei_w(_r2n, 1);
                    __m128 _r26 = (__m128)__lsx_vreplvei_w(_r2n, 2);
                    __m128 _r27 = (__m128)__lsx_vreplvei_w(_r2n, 3);
                    __m128 _r28 = (__m128)__lsx_vreplvei_w(_r2nn, 0);
                    __m128 _r29 = (__m128)__lsx_vreplvei_w(_r2nn, 1);
                    __m128 _r2a = (__m128)__lsx_vreplvei_w(_r2nn, 2);
                    __m128 _r2b = (__m128)__lsx_vreplvei_w(_r2nn, 3);
                    __m128 _r2c = __lsx_vreplfr2vr_s(r2[12]);

                    _sum0 = __lsx_vfmadd_s(_k20, _r20, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k20, _r22, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k20, _r24, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k20, _r26, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k21, _r21, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k21, _r23, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k21, _r25, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k21, _r27, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k22, _r22, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k22, _r24, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k22, _r26, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k22, _r28, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k23, _r23, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k23, _r25, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k23, _r27, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k23, _r29, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k24, _r24, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k24, _r26, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k24, _r28, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k24, _r2a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k25, _r25, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k25, _r27, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k25, _r29, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k25, _r2b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k26, _r26, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k26, _r28, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k26, _r2a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k26, _r2c, _sum3);

                    __m128 _k30 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k31 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k32 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k33 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k34 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k35 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k36 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r3 = __lsx_vld(r3, 0);
                    __m128i _r3n = __lsx_vld(r3 + 4, 0);
                    __m128i _r3nn = __lsx_vld(r3 + 8, 0);

                    __m128 _r30 = (__m128)__lsx_vreplvei_w(_r3, 0);
                    __m128 _r31 = (__m128)__lsx_vreplvei_w(_r3, 1);
                    __m128 _r32 = (__m128)__lsx_vreplvei_w(_r3, 2);
                    __m128 _r33 = (__m128)__lsx_vreplvei_w(_r3, 3);
                    __m128 _r34 = (__m128)__lsx_vreplvei_w(_r3n, 0);
                    __m128 _r35 = (__m128)__lsx_vreplvei_w(_r3n, 1);
                    __m128 _r36 = (__m128)__lsx_vreplvei_w(_r3n, 2);
                    __m128 _r37 = (__m128)__lsx_vreplvei_w(_r3n, 3);
                    __m128 _r38 = (__m128)__lsx_vreplvei_w(_r3nn, 0);
                    __m128 _r39 = (__m128)__lsx_vreplvei_w(_r3nn, 1);
                    __m128 _r3a = (__m128)__lsx_vreplvei_w(_r3nn, 2);
                    __m128 _r3b = (__m128)__lsx_vreplvei_w(_r3nn, 3);
                    __m128 _r3c = __lsx_vreplfr2vr_s(r3[12]);

                    _sum0 = __lsx_vfmadd_s(_k30, _r30, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k30, _r32, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k30, _r34, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k30, _r36, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k31, _r31, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k31, _r33, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k31, _r35, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k31, _r37, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k32, _r32, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k32, _r34, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k32, _r36, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k32, _r38, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k33, _r33, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k33, _r35, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k33, _r37, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k33, _r39, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k34, _r34, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k34, _r36, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k34, _r38, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k34, _r3a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k35, _r35, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k35, _r37, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k35, _r39, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k35, _r3b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k36, _r36, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k36, _r38, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k36, _r3a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k36, _r3c, _sum3);

                    __m128 _k40 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k41 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k42 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k43 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k44 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k45 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k46 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r4 = __lsx_vld(r4, 0);
                    __m128i _r4n = __lsx_vld(r4 + 4, 0);
                    __m128i _r4nn = __lsx_vld(r4 + 8, 0);

                    __m128 _r40 = (__m128)__lsx_vreplvei_w(_r4, 0);
                    __m128 _r41 = (__m128)__lsx_vreplvei_w(_r4, 1);
                    __m128 _r42 = (__m128)__lsx_vreplvei_w(_r4, 2);
                    __m128 _r43 = (__m128)__lsx_vreplvei_w(_r4, 3);
                    __m128 _r44 = (__m128)__lsx_vreplvei_w(_r4n, 0);
                    __m128 _r45 = (__m128)__lsx_vreplvei_w(_r4n, 1);
                    __m128 _r46 = (__m128)__lsx_vreplvei_w(_r4n, 2);
                    __m128 _r47 = (__m128)__lsx_vreplvei_w(_r4n, 3);
                    __m128 _r48 = (__m128)__lsx_vreplvei_w(_r4nn, 0);
                    __m128 _r49 = (__m128)__lsx_vreplvei_w(_r4nn, 1);
                    __m128 _r4a = (__m128)__lsx_vreplvei_w(_r4nn, 2);
                    __m128 _r4b = (__m128)__lsx_vreplvei_w(_r4nn, 3);
                    __m128 _r4c = __lsx_vreplfr2vr_s(r4[12]);

                    _sum0 = __lsx_vfmadd_s(_k40, _r40, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k40, _r42, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k40, _r44, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k40, _r46, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k41, _r41, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k41, _r43, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k41, _r45, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k41, _r47, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k42, _r42, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k42, _r44, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k42, _r46, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k42, _r48, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k43, _r43, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k43, _r45, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k43, _r47, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k43, _r49, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k44, _r44, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k44, _r46, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k44, _r48, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k44, _r4a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k45, _r45, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k45, _r47, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k45, _r49, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k45, _r4b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k46, _r46, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k46, _r48, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k46, _r4a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k46, _r4c, _sum3);

                    __m128 _k50 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k51 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k52 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k53 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k54 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k55 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k56 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r5 = __lsx_vld(r5, 0);
                    __m128i _r5n = __lsx_vld(r5 + 4, 0);
                    __m128i _r5nn = __lsx_vld(r5 + 8, 0);

                    __m128 _r50 = (__m128)__lsx_vreplvei_w(_r5, 0);
                    __m128 _r51 = (__m128)__lsx_vreplvei_w(_r5, 1);
                    __m128 _r52 = (__m128)__lsx_vreplvei_w(_r5, 2);
                    __m128 _r53 = (__m128)__lsx_vreplvei_w(_r5, 3);
                    __m128 _r54 = (__m128)__lsx_vreplvei_w(_r5n, 0);
                    __m128 _r55 = (__m128)__lsx_vreplvei_w(_r5n, 1);
                    __m128 _r56 = (__m128)__lsx_vreplvei_w(_r5n, 2);
                    __m128 _r57 = (__m128)__lsx_vreplvei_w(_r5n, 3);
                    __m128 _r58 = (__m128)__lsx_vreplvei_w(_r5nn, 0);
                    __m128 _r59 = (__m128)__lsx_vreplvei_w(_r5nn, 1);
                    __m128 _r5a = (__m128)__lsx_vreplvei_w(_r5nn, 2);
                    __m128 _r5b = (__m128)__lsx_vreplvei_w(_r5nn, 3);
                    __m128 _r5c = __lsx_vreplfr2vr_s(r5[12]);

                    _sum0 = __lsx_vfmadd_s(_k50, _r50, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k50, _r52, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k50, _r54, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k50, _r56, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k51, _r51, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k51, _r53, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k51, _r55, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k51, _r57, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k52, _r52, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k52, _r54, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k52, _r56, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k52, _r58, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k53, _r53, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k53, _r55, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k53, _r57, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k53, _r59, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k54, _r54, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k54, _r56, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k54, _r58, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k54, _r5a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k55, _r55, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k55, _r57, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k55, _r59, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k55, _r5b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k56, _r56, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k56, _r58, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k56, _r5a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k56, _r5c, _sum3);

                    __m128 _k60 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k61 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k62 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k63 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k64 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k65 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k66 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr -= 4 * 42;

                    __m128i _r6 = __lsx_vld(r6, 0);
                    __m128i _r6n = __lsx_vld(r6 + 4, 0);
                    __m128i _r6nn = __lsx_vld(r6 + 8, 0);

                    __m128 _r60 = (__m128)__lsx_vreplvei_w(_r6, 0);
                    __m128 _r61 = (__m128)__lsx_vreplvei_w(_r6, 1);
                    __m128 _r62 = (__m128)__lsx_vreplvei_w(_r6, 2);
                    __m128 _r63 = (__m128)__lsx_vreplvei_w(_r6, 3);
                    __m128 _r64 = (__m128)__lsx_vreplvei_w(_r6n, 0);
                    __m128 _r65 = (__m128)__lsx_vreplvei_w(_r6n, 1);
                    __m128 _r66 = (__m128)__lsx_vreplvei_w(_r6n, 2);
                    __m128 _r67 = (__m128)__lsx_vreplvei_w(_r6n, 3);
                    __m128 _r68 = (__m128)__lsx_vreplvei_w(_r6nn, 0);
                    __m128 _r69 = (__m128)__lsx_vreplvei_w(_r6nn, 1);
                    __m128 _r6a = (__m128)__lsx_vreplvei_w(_r6nn, 2);
                    __m128 _r6b = (__m128)__lsx_vreplvei_w(_r6nn, 3);
                    __m128 _r6c = __lsx_vreplfr2vr_s(r6[12]);

                    _sum0 = __lsx_vfmadd_s(_k60, _r60, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k60, _r62, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k60, _r64, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k60, _r66, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k61, _r61, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k61, _r63, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k61, _r65, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k61, _r67, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k62, _r62, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k62, _r64, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k62, _r66, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k62, _r68, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k63, _r63, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k63, _r65, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k63, _r67, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k63, _r69, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k64, _r64, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k64, _r66, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k64, _r68, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k64, _r6a, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k65, _r65, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k65, _r67, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k65, _r69, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k65, _r6b, _sum3);
                    _sum0 = __lsx_vfmadd_s(_k66, _r66, _sum0);
                    _sum1 = __lsx_vfmadd_s(_k66, _r68, _sum1);
                    _sum2 = __lsx_vfmadd_s(_k66, _r6a, _sum2);
                    _sum3 = __lsx_vfmadd_s(_k66, _r6c, _sum3);

                    __lsx_vst(_sum0, outptr0, 0);
                    __lsx_vst(_sum1, outptr0 + 4, 0);
                    __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst(_sum3, outptr0 + 4 * 3, 0);

                    outptr0 += 4 * 4;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                }
                for (; j < outw; j++)
                {
                    __m128 _sum0 = (__m128)__lsx_vld(outptr0, 0);

                    __m128 _k00 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k01 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k02 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k03 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k04 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k05 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k06 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r0n = __lsx_vld(r0 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k00, (__m128)__lsx_vreplvei_w(_r0, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k01, (__m128)__lsx_vreplvei_w(_r0, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k02, (__m128)__lsx_vreplvei_w(_r0, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k03, (__m128)__lsx_vreplvei_w(_r0, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k04, (__m128)__lsx_vreplvei_w(_r0n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k05, (__m128)__lsx_vreplvei_w(_r0n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k06, (__m128)__lsx_vreplvei_w(_r0n, 2), _sum0);

                    __m128 _k10 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k11 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k12 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k13 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k14 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k15 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k16 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r1n = __lsx_vld(r1 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k10, (__m128)__lsx_vreplvei_w(_r1, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k11, (__m128)__lsx_vreplvei_w(_r1, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k12, (__m128)__lsx_vreplvei_w(_r1, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k13, (__m128)__lsx_vreplvei_w(_r1, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k14, (__m128)__lsx_vreplvei_w(_r1n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k15, (__m128)__lsx_vreplvei_w(_r1n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k16, (__m128)__lsx_vreplvei_w(_r1n, 2), _sum0);

                    __m128 _k20 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k21 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k22 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k23 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k24 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k25 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k26 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r2n = __lsx_vld(r2 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k20, (__m128)__lsx_vreplvei_w(_r2, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k21, (__m128)__lsx_vreplvei_w(_r2, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k22, (__m128)__lsx_vreplvei_w(_r2, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k23, (__m128)__lsx_vreplvei_w(_r2, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k24, (__m128)__lsx_vreplvei_w(_r2n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k25, (__m128)__lsx_vreplvei_w(_r2n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k26, (__m128)__lsx_vreplvei_w(_r2n, 2), _sum0);

                    __m128 _k30 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k31 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k32 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k33 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k34 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k35 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k36 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r3 = __lsx_vld(r3, 0);
                    __m128i _r3n = __lsx_vld(r3 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k30, (__m128)__lsx_vreplvei_w(_r3, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k31, (__m128)__lsx_vreplvei_w(_r3, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k32, (__m128)__lsx_vreplvei_w(_r3, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k33, (__m128)__lsx_vreplvei_w(_r3, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k34, (__m128)__lsx_vreplvei_w(_r3n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k35, (__m128)__lsx_vreplvei_w(_r3n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k36, (__m128)__lsx_vreplvei_w(_r3n, 2), _sum0);

                    __m128 _k40 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k41 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k42 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k43 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k44 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k45 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k46 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r4 = __lsx_vld(r4, 0);
                    __m128i _r4n = __lsx_vld(r4 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k40, (__m128)__lsx_vreplvei_w(_r4, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k41, (__m128)__lsx_vreplvei_w(_r4, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k42, (__m128)__lsx_vreplvei_w(_r4, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k43, (__m128)__lsx_vreplvei_w(_r4, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k44, (__m128)__lsx_vreplvei_w(_r4n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k45, (__m128)__lsx_vreplvei_w(_r4n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k46, (__m128)__lsx_vreplvei_w(_r4n, 2), _sum0);

                    __m128 _k50 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k51 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k52 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k53 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k54 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k55 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k56 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr += 4 * 7;

                    __m128i _r5 = __lsx_vld(r5, 0);
                    __m128i _r5n = __lsx_vld(r5 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k50, (__m128)__lsx_vreplvei_w(_r5, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k51, (__m128)__lsx_vreplvei_w(_r5, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k52, (__m128)__lsx_vreplvei_w(_r5, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k53, (__m128)__lsx_vreplvei_w(_r5, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k54, (__m128)__lsx_vreplvei_w(_r5n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k55, (__m128)__lsx_vreplvei_w(_r5n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k56, (__m128)__lsx_vreplvei_w(_r5n, 2), _sum0);

                    __m128 _k60 = (__m128)__lsx_vld(kptr, 0);
                    __m128 _k61 = (__m128)__lsx_vld(kptr + 4, 0);
                    __m128 _k62 = (__m128)__lsx_vld(kptr + 4 * 2, 0);
                    __m128 _k63 = (__m128)__lsx_vld(kptr + 4 * 3, 0);
                    __m128 _k64 = (__m128)__lsx_vld(kptr + 4 * 4, 0);
                    __m128 _k65 = (__m128)__lsx_vld(kptr + 4 * 5, 0);
                    __m128 _k66 = (__m128)__lsx_vld(kptr + 4 * 6, 0);

                    kptr -= 4 * 42;

                    __m128i _r6 = __lsx_vld(r6, 0);
                    __m128i _r6n = __lsx_vld(r6 + 4, 0);

                    _sum0 = __lsx_vfmadd_s(_k60, (__m128)__lsx_vreplvei_w(_r6, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k61, (__m128)__lsx_vreplvei_w(_r6, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k62, (__m128)__lsx_vreplvei_w(_r6, 2), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k63, (__m128)__lsx_vreplvei_w(_r6, 3), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k64, (__m128)__lsx_vreplvei_w(_r6n, 0), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k65, (__m128)__lsx_vreplvei_w(_r6n, 1), _sum0);
                    _sum0 = __lsx_vfmadd_s(_k66, (__m128)__lsx_vreplvei_w(_r6n, 2), _sum0);

                    __lsx_vst(_sum0, outptr0, 0);

                    outptr0 += 4;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
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
