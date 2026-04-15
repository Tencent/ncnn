// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convdw3x3s1_pack8_lasx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m256 _bias0 = bias ? (__m256)__lasx_xvld((const float*)bias + g * 8, 0) : (__m256)__lasx_xvldi(0);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum00 = _bias0;
                __m256 _sum01 = _bias0;
                __m256 _sum02 = _bias0;
                __m256 _sum03 = _bias0;
                __m256 _sum10 = _bias0;
                __m256 _sum11 = _bias0;
                __m256 _sum12 = _bias0;
                __m256 _sum13 = _bias0;

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);
                __m256 _r04 = (__m256)__lasx_xvld(r0 + 32, 0);
                __m256 _r05 = (__m256)__lasx_xvld(r0 + 40, 0);

                _sum00 = __lasx_xvfmadd_s(_k00, _r00, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k00, _r01, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k00, _r02, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k00, _r03, _sum03);
                _sum00 = __lasx_xvfmadd_s(_k01, _r01, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k01, _r02, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k01, _r03, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k01, _r04, _sum03);
                _sum00 = __lasx_xvfmadd_s(_k02, _r02, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k02, _r03, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k02, _r04, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k02, _r05, _sum03);

                __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);
                __m256 _r14 = (__m256)__lasx_xvld(r1 + 32, 0);
                __m256 _r15 = (__m256)__lasx_xvld(r1 + 40, 0);

                _sum10 = __lasx_xvfmadd_s(_k00, _r10, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k00, _r11, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k00, _r12, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k00, _r13, _sum13);
                _sum00 = __lasx_xvfmadd_s(_k10, _r10, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k10, _r11, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k10, _r12, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k10, _r13, _sum03);

                _sum10 = __lasx_xvfmadd_s(_k01, _r11, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k01, _r12, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k01, _r13, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k01, _r14, _sum13);
                _sum00 = __lasx_xvfmadd_s(_k11, _r11, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k11, _r12, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k11, _r13, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k11, _r14, _sum03);

                _sum10 = __lasx_xvfmadd_s(_k02, _r12, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k02, _r13, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k02, _r14, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k02, _r15, _sum13);
                _sum00 = __lasx_xvfmadd_s(_k12, _r12, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k12, _r13, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k12, _r14, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k12, _r15, _sum03);

                __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);
                __m256 _r24 = (__m256)__lasx_xvld(r2 + 32, 0);
                __m256 _r25 = (__m256)__lasx_xvld(r2 + 40, 0);

                _sum10 = __lasx_xvfmadd_s(_k10, _r20, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k10, _r21, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k10, _r22, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k10, _r23, _sum13);
                _sum00 = __lasx_xvfmadd_s(_k20, _r20, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k20, _r21, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k20, _r22, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k20, _r23, _sum03);

                _sum10 = __lasx_xvfmadd_s(_k11, _r21, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k11, _r22, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k11, _r23, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k11, _r24, _sum13);
                _sum00 = __lasx_xvfmadd_s(_k21, _r21, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k21, _r22, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k21, _r23, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k21, _r24, _sum03);

                _sum10 = __lasx_xvfmadd_s(_k12, _r22, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k12, _r23, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k12, _r24, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k12, _r25, _sum13);
                _sum00 = __lasx_xvfmadd_s(_k22, _r22, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k22, _r23, _sum01);
                _sum02 = __lasx_xvfmadd_s(_k22, _r24, _sum02);
                _sum03 = __lasx_xvfmadd_s(_k22, _r25, _sum03);

                __m256 _r30 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r31 = (__m256)__lasx_xvld(r3 + 8, 0);
                __m256 _r32 = (__m256)__lasx_xvld(r3 + 16, 0);
                __m256 _r33 = (__m256)__lasx_xvld(r3 + 24, 0);
                __m256 _r34 = (__m256)__lasx_xvld(r3 + 32, 0);
                __m256 _r35 = (__m256)__lasx_xvld(r3 + 40, 0);

                _sum10 = __lasx_xvfmadd_s(_k20, _r30, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k20, _r31, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k20, _r32, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k20, _r33, _sum13);
                _sum10 = __lasx_xvfmadd_s(_k21, _r31, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k21, _r32, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k21, _r33, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k21, _r34, _sum13);
                _sum10 = __lasx_xvfmadd_s(_k22, _r32, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k22, _r33, _sum11);
                _sum12 = __lasx_xvfmadd_s(_k22, _r34, _sum12);
                _sum13 = __lasx_xvfmadd_s(_k22, _r35, _sum13);

                __lasx_xvst((__m256i)(_sum00), outptr0, 0);
                __lasx_xvst((__m256i)(_sum01), outptr0 + 8, 0);
                __lasx_xvst((__m256i)(_sum02), outptr0 + 16, 0);
                __lasx_xvst((__m256i)(_sum03), outptr0 + 24, 0);
                __lasx_xvst((__m256i)(_sum10), outptr1, 0);
                __lasx_xvst((__m256i)(_sum11), outptr1 + 8, 0);
                __lasx_xvst((__m256i)(_sum12), outptr1 + 16, 0);
                __lasx_xvst((__m256i)(_sum13), outptr1 + 24, 0);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                r3 += 32;
                outptr0 += 32;
                outptr1 += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum00 = _bias0;
                __m256 _sum01 = _bias0;
                __m256 _sum10 = _bias0;
                __m256 _sum11 = _bias0;

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);

                _sum00 = __lasx_xvfmadd_s(_k00, _r00, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k00, _r01, _sum01);
                _sum00 = __lasx_xvfmadd_s(_k01, _r01, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k01, _r02, _sum01);
                _sum00 = __lasx_xvfmadd_s(_k02, _r02, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k02, _r03, _sum01);

                __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);

                _sum00 = __lasx_xvfmadd_s(_k10, _r10, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k10, _r11, _sum01);
                _sum10 = __lasx_xvfmadd_s(_k00, _r10, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k00, _r11, _sum11);

                _sum00 = __lasx_xvfmadd_s(_k11, _r11, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k11, _r12, _sum01);
                _sum10 = __lasx_xvfmadd_s(_k01, _r11, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k01, _r12, _sum11);

                _sum00 = __lasx_xvfmadd_s(_k12, _r12, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k12, _r13, _sum01);
                _sum10 = __lasx_xvfmadd_s(_k02, _r12, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k02, _r13, _sum11);

                __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);

                _sum00 = __lasx_xvfmadd_s(_k20, _r20, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k20, _r21, _sum01);
                _sum10 = __lasx_xvfmadd_s(_k10, _r20, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k10, _r21, _sum11);

                _sum00 = __lasx_xvfmadd_s(_k21, _r21, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k21, _r22, _sum01);
                _sum10 = __lasx_xvfmadd_s(_k11, _r21, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k11, _r22, _sum11);

                _sum00 = __lasx_xvfmadd_s(_k22, _r22, _sum00);
                _sum01 = __lasx_xvfmadd_s(_k22, _r23, _sum01);
                _sum10 = __lasx_xvfmadd_s(_k12, _r22, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k12, _r23, _sum11);

                __m256 _r30 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r31 = (__m256)__lasx_xvld(r3 + 8, 0);
                __m256 _r32 = (__m256)__lasx_xvld(r3 + 16, 0);
                __m256 _r33 = (__m256)__lasx_xvld(r3 + 24, 0);

                _sum10 = __lasx_xvfmadd_s(_k20, _r30, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k20, _r31, _sum11);
                _sum10 = __lasx_xvfmadd_s(_k21, _r31, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k21, _r32, _sum11);
                _sum10 = __lasx_xvfmadd_s(_k22, _r32, _sum10);
                _sum11 = __lasx_xvfmadd_s(_k22, _r33, _sum11);

                __lasx_xvst((__m256i)(_sum00), outptr0, 0);
                __lasx_xvst((__m256i)(_sum01), outptr0 + 8, 0);
                __lasx_xvst((__m256i)(_sum10), outptr1, 0);
                __lasx_xvst((__m256i)(_sum11), outptr1 + 8, 0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                outptr0 += 16;
                outptr1 += 16;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);

                __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k00, _r10, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k01, _r11, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k02, _r12, _sum1);

                __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k10, _r20, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k11, _r21, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k12, _r22, _sum1);

                __m256 _r30 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r31 = (__m256)__lasx_xvld(r3 + 8, 0);
                __m256 _r32 = (__m256)__lasx_xvld(r3 + 16, 0);

                _sum1 = __lasx_xvfmadd_s(_k20, _r30, _sum1);
                _sum1 = __lasx_xvfmadd_s(_k21, _r31, _sum1);
                _sum1 = __lasx_xvfmadd_s(_k22, _r32, _sum1);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);
                __lasx_xvst((__m256i)(_sum1), outptr1, 0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr0 += 8;
                outptr1 += 8;
            }

            r0 += 2 * 8 + w * 8;
            r1 += 2 * 8 + w * 8;
            r2 += 2 * 8 + w * 8;
            r3 += 2 * 8 + w * 8;

            outptr0 += outw * 8;
            outptr1 += outw * 8;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;
                __m256 _sum2 = _bias0;
                __m256 _sum3 = _bias0;

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);
                __m256 _r04 = (__m256)__lasx_xvld(r0 + 32, 0);
                __m256 _r05 = (__m256)__lasx_xvld(r0 + 40, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k00, _r01, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k00, _r02, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k00, _r03, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k01, _r02, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k01, _r03, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k01, _r04, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k02, _r03, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k02, _r04, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k02, _r05, _sum3);

                __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);
                __m256 _r14 = (__m256)__lasx_xvld(r1 + 32, 0);
                __m256 _r15 = (__m256)__lasx_xvld(r1 + 40, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k10, _r11, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k10, _r12, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k10, _r13, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k11, _r12, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k11, _r13, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k11, _r14, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k12, _r13, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k12, _r14, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k12, _r15, _sum3);

                __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);
                __m256 _r24 = (__m256)__lasx_xvld(r2 + 32, 0);
                __m256 _r25 = (__m256)__lasx_xvld(r2 + 40, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k20, _r21, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k20, _r22, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k20, _r23, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k21, _r22, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k21, _r23, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k21, _r24, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k22, _r23, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k22, _r24, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k22, _r25, _sum3);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);
                __lasx_xvst((__m256i)(_sum1), outptr0 + 8, 0);
                __lasx_xvst((__m256i)(_sum2), outptr0 + 16, 0);
                __lasx_xvst((__m256i)(_sum3), outptr0 + 24, 0);

                r0 += 32;
                r1 += 32;
                r2 += 32;
                outptr0 += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k00, _r01, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k01, _r02, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k02, _r03, _sum1);

                __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k10, _r11, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k11, _r12, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k12, _r13, _sum1);

                __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k20, _r21, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k21, _r22, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k22, _r23, _sum1);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);
                __lasx_xvst((__m256i)(_sum1), outptr0 + 8, 0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr0 += 16;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);

                __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);

                __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }

            r0 += 2 * 8;
            r1 += 2 * 8;
            r2 += 2 * 8;
        }
    }
}

static void convdw3x3s2_pack8_lasx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __m256 _bias0 = bias ? (__m256)__lasx_xvld((const float*)bias + g * 8, 0) : (__m256)__lasx_xvldi(0);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
        __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
        __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);
        __m256 _k10 = (__m256)__lasx_xvld(k0 + 24, 0);
        __m256 _k11 = (__m256)__lasx_xvld(k0 + 32, 0);
        __m256 _k12 = (__m256)__lasx_xvld(k0 + 40, 0);
        __m256 _k20 = (__m256)__lasx_xvld(k0 + 48, 0);
        __m256 _k21 = (__m256)__lasx_xvld(k0 + 56, 0);
        __m256 _k22 = (__m256)__lasx_xvld(k0 + 64, 0);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;
                __m256 _sum2 = _bias0;
                __m256 _sum3 = _bias0;

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);
                __m256 _r04 = (__m256)__lasx_xvld(r0 + 32, 0);
                __m256 _r05 = (__m256)__lasx_xvld(r0 + 40, 0);
                __m256 _r06 = (__m256)__lasx_xvld(r0 + 48, 0);
                __m256 _r07 = (__m256)__lasx_xvld(r0 + 56, 0);
                __m256 _r08 = (__m256)__lasx_xvld(r0 + 64, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k00, _r02, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k00, _r04, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k00, _r06, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k01, _r03, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k01, _r05, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k01, _r07, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k02, _r04, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k02, _r06, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k02, _r08, _sum3);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);
                __m256 _r14 = (__m256)__lasx_xvld(r1 + 32, 0);
                __m256 _r15 = (__m256)__lasx_xvld(r1 + 40, 0);
                __m256 _r16 = (__m256)__lasx_xvld(r1 + 48, 0);
                __m256 _r17 = (__m256)__lasx_xvld(r1 + 56, 0);
                __m256 _r18 = (__m256)__lasx_xvld(r1 + 64, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k10, _r12, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k10, _r14, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k10, _r16, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k11, _r13, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k11, _r15, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k11, _r17, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k12, _r14, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k12, _r16, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k12, _r18, _sum3);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);
                __m256 _r24 = (__m256)__lasx_xvld(r2 + 32, 0);
                __m256 _r25 = (__m256)__lasx_xvld(r2 + 40, 0);
                __m256 _r26 = (__m256)__lasx_xvld(r2 + 48, 0);
                __m256 _r27 = (__m256)__lasx_xvld(r2 + 56, 0);
                __m256 _r28 = (__m256)__lasx_xvld(r2 + 64, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k20, _r22, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k20, _r24, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k20, _r26, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k21, _r23, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k21, _r25, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k21, _r27, _sum3);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k22, _r24, _sum1);
                _sum2 = __lasx_xvfmadd_s(_k22, _r26, _sum2);
                _sum3 = __lasx_xvfmadd_s(_k22, _r28, _sum3);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);
                __lasx_xvst((__m256i)(_sum1), outptr0 + 8, 0);
                __lasx_xvst((__m256i)(_sum2), outptr0 + 16, 0);
                __lasx_xvst((__m256i)(_sum3), outptr0 + 24, 0);

                r0 += 2 * 32;
                r1 += 2 * 32;
                r2 += 2 * 32;
                outptr0 += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                __m256 _sum0 = _bias0;
                __m256 _sum1 = _bias0;

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);
                __m256 _r04 = (__m256)__lasx_xvld(r0 + 32, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k00, _r02, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k01, _r03, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k02, _r04, _sum1);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);
                __m256 _r14 = (__m256)__lasx_xvld(r1 + 32, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k10, _r12, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k11, _r13, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k12, _r14, _sum1);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);
                __m256 _r24 = (__m256)__lasx_xvld(r2 + 32, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k20, _r22, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k21, _r23, _sum1);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum1 = __lasx_xvfmadd_s(_k22, _r24, _sum1);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);
                __lasx_xvst((__m256i)(_sum1), outptr0 + 8, 0);

                r0 += 2 * 16;
                r1 += 2 * 16;
                r2 += 2 * 16;
                outptr0 += 16;
            }
            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);

                r0 += 2 * 8;
                r1 += 2 * 8;
                r2 += 2 * 8;
                outptr0 += 8;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
