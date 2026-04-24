// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convdw5x5s1_pack8_lasx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
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

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);
        const float* r4 = img0.row(4);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);
                __m256 _r04 = (__m256)__lasx_xvld(r0 + 32, 0);

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k03 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k04 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k03, _r03, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k04, _r04, _sum0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);
                __m256 _r14 = (__m256)__lasx_xvld(r1 + 32, 0);

                __m256 _k10 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k13 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k14 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k13, _r13, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k14, _r14, _sum0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);
                __m256 _r24 = (__m256)__lasx_xvld(r2 + 32, 0);

                __m256 _k20 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k23 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k24 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k23, _r23, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k24, _r24, _sum0);

                __m256 _r30 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r31 = (__m256)__lasx_xvld(r3 + 8, 0);
                __m256 _r32 = (__m256)__lasx_xvld(r3 + 16, 0);
                __m256 _r33 = (__m256)__lasx_xvld(r3 + 24, 0);
                __m256 _r34 = (__m256)__lasx_xvld(r3 + 32, 0);

                __m256 _k30 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k31 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k32 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k33 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k34 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k30, _r30, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k31, _r31, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k32, _r32, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k33, _r33, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k34, _r34, _sum0);

                __m256 _r40 = (__m256)__lasx_xvld(r4, 0);
                __m256 _r41 = (__m256)__lasx_xvld(r4 + 8, 0);
                __m256 _r42 = (__m256)__lasx_xvld(r4 + 16, 0);
                __m256 _r43 = (__m256)__lasx_xvld(r4 + 24, 0);
                __m256 _r44 = (__m256)__lasx_xvld(r4 + 32, 0);

                __m256 _k40 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k41 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k42 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k43 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k44 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 -= 160;

                _sum0 = __lasx_xvfmadd_s(_k40, _r40, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k41, _r41, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k42, _r42, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k43, _r43, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k44, _r44, _sum0);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                outptr0 += 8;
            }

            r0 += 4 * 8;
            r1 += 4 * 8;
            r2 += 4 * 8;
            r3 += 4 * 8;
            r4 += 4 * 8;
        }
    }
}

static void convdw5x5s2_pack8_lasx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
        const float* r3 = img0.row(3);
        const float* r4 = img0.row(4);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                __m256 _sum0 = _bias0;

                __m256 _r00 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r01 = (__m256)__lasx_xvld(r0 + 8, 0);
                __m256 _r02 = (__m256)__lasx_xvld(r0 + 16, 0);
                __m256 _r03 = (__m256)__lasx_xvld(r0 + 24, 0);
                __m256 _r04 = (__m256)__lasx_xvld(r0 + 32, 0);

                __m256 _k00 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k01 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k02 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k03 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k04 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k00, _r00, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k01, _r01, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k02, _r02, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k03, _r03, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k04, _r04, _sum0);

                __m256 _r10 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r11 = (__m256)__lasx_xvld(r1 + 8, 0);
                __m256 _r12 = (__m256)__lasx_xvld(r1 + 16, 0);
                __m256 _r13 = (__m256)__lasx_xvld(r1 + 24, 0);
                __m256 _r14 = (__m256)__lasx_xvld(r1 + 32, 0);

                __m256 _k10 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k11 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k12 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k13 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k14 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k10, _r10, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k11, _r11, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k12, _r12, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k13, _r13, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k14, _r14, _sum0);

                __m256 _r20 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r21 = (__m256)__lasx_xvld(r2 + 8, 0);
                __m256 _r22 = (__m256)__lasx_xvld(r2 + 16, 0);
                __m256 _r23 = (__m256)__lasx_xvld(r2 + 24, 0);
                __m256 _r24 = (__m256)__lasx_xvld(r2 + 32, 0);

                __m256 _k20 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k21 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k22 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k23 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k24 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k20, _r20, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k21, _r21, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k22, _r22, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k23, _r23, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k24, _r24, _sum0);

                __m256 _r30 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r31 = (__m256)__lasx_xvld(r3 + 8, 0);
                __m256 _r32 = (__m256)__lasx_xvld(r3 + 16, 0);
                __m256 _r33 = (__m256)__lasx_xvld(r3 + 24, 0);
                __m256 _r34 = (__m256)__lasx_xvld(r3 + 32, 0);

                __m256 _k30 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k31 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k32 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k33 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k34 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 += 40;

                _sum0 = __lasx_xvfmadd_s(_k30, _r30, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k31, _r31, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k32, _r32, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k33, _r33, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k34, _r34, _sum0);

                __m256 _r40 = (__m256)__lasx_xvld(r4, 0);
                __m256 _r41 = (__m256)__lasx_xvld(r4 + 8, 0);
                __m256 _r42 = (__m256)__lasx_xvld(r4 + 16, 0);
                __m256 _r43 = (__m256)__lasx_xvld(r4 + 24, 0);
                __m256 _r44 = (__m256)__lasx_xvld(r4 + 32, 0);

                __m256 _k40 = (__m256)__lasx_xvld(k0, 0);
                __m256 _k41 = (__m256)__lasx_xvld(k0 + 8, 0);
                __m256 _k42 = (__m256)__lasx_xvld(k0 + 16, 0);
                __m256 _k43 = (__m256)__lasx_xvld(k0 + 24, 0);
                __m256 _k44 = (__m256)__lasx_xvld(k0 + 32, 0);
                k0 -= 160;

                _sum0 = __lasx_xvfmadd_s(_k40, _r40, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k41, _r41, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k42, _r42, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k43, _r43, _sum0);
                _sum0 = __lasx_xvfmadd_s(_k44, _r44, _sum0);

                __lasx_xvst((__m256i)(_sum0), outptr0, 0);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                r4 += 16;
                outptr0 += 8;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }
}
