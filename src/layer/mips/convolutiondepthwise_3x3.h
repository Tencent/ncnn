// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convdw3x3s1_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g * 9;

        float* outptr0 = out;
        float* outptr1 = outptr0 + outw;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;
        const float* r3 = img0 + w * 3;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;

        int i = 0;

        for (; i + 1 < outh; i += 2)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = bias0;
                float sum2 = bias0;

                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum2 += r1[0] * k0[0];
                sum2 += r1[1] * k0[1];
                sum2 += r1[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum2 += r2[0] * k1[0];
                sum2 += r2[1] * k1[1];
                sum2 += r2[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];

                *outptr0 = sum;
                *outptr1 = sum2;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr1 += outw;
        }

        for (; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

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

static void convdw3x3s2_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g * 9;

        float* outptr = out;

        const float* img0 = bottom_blob.channel(g);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;

        int i = 0;

        for (; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

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
