// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// Copyright (C) 2019 BUG1989. All rights reserved.
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

static void conv3x3s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch * 9 + q * 9;

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
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_sse(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(4 * 4, inch, outch);

    // G
    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[4][3];
            for (int i = 0; i < 4; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 4; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++)
                {
                    kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt_b);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4 * 4, tiles, inch, 4u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const float* img = bottom_blob_bordered.channel(q);
            float* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 2;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
#if __AVX__
                    __m128 _d0, _d1, _d2, _d3;
                    __m128 _w0, _w1, _w2, _w3;

                    // load
                    _d0 = _mm_loadu_ps(r0);
                    _d1 = _mm_loadu_ps(r1);
                    _d2 = _mm_loadu_ps(r2);
                    _d3 = _mm_loadu_ps(r3);

                    // w = B_t * d
                    _w0 = _mm_sub_ps(_d0, _d2);
                    _w1 = _mm_add_ps(_d1, _d2);
                    _w2 = _mm_sub_ps(_d2, _d1);
                    _w3 = _mm_sub_ps(_d3, _d1);

                    // transpose d to d_t
                    _MM_TRANSPOSE4_PS(_w0, _w1, _w2, _w3);

                    // d = B_t * d_t
                    _d0 = _mm_sub_ps(_w0, _w2);
                    _d1 = _mm_add_ps(_w1, _w2);
                    _d2 = _mm_sub_ps(_w2, _w1);
                    _d3 = _mm_sub_ps(_w3, _w1);

                    // save to out_tm
                    _mm_storeu_ps(out_tm0, _d0);
                    _mm_storeu_ps(out_tm0 + 4, _d1);
                    _mm_storeu_ps(out_tm0 + 8, _d2);
                    _mm_storeu_ps(out_tm0 + 12, _d3);
#else
                    float d0[4], d1[4], d2[4], d3[4];
                    float w0[4], w1[4], w2[4], w3[4];
                    float t0[4], t1[4], t2[4], t3[4];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                    }
                    // d = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm0[n + 4] = d1[n];
                        out_tm0[n + 8] = d2[n];
                        out_tm0[n + 12] = d3[n];
                    }
#endif
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;

                    out_tm0 += 16;
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p + 1);
            Mat out2_tm = top_blob_tm.channel(p + 2);
            Mat out3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p + 1);
            const Mat kernel2_tm = kernel_tm.channel(p + 2);
            const Mat kernel3_tm = kernel_tm.channel(p + 3);

            for (int i = 0; i < tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);
                float* output1_tm = out1_tm.row(i);
                float* output2_tm = out2_tm.row(i);
                float* output3_tm = out3_tm.row(i);

#if __AVX__
                float zero_val = 0.f;

                __m256 _sum0 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum0n = _mm256_broadcast_ss(&zero_val);
                __m256 _sum1 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum1n = _mm256_broadcast_ss(&zero_val);
                __m256 _sum2 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum2n = _mm256_broadcast_ss(&zero_val);
                __m256 _sum3 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum3n = _mm256_broadcast_ss(&zero_val);

                int q = 0;

                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r0n = _mm256_loadu_ps(r0 + 8);
                    // k0
                    __m256 _k0 = _mm256_loadu_ps(k0);
                    __m256 _k0n = _mm256_loadu_ps(k0 + 8);
                    __m256 _k1 = _mm256_loadu_ps(k1);
                    __m256 _k1n = _mm256_loadu_ps(k1 + 8);
                    __m256 _k2 = _mm256_loadu_ps(k2);
                    __m256 _k2n = _mm256_loadu_ps(k2 + 8);
                    __m256 _k3 = _mm256_loadu_ps(k3);
                    __m256 _k3n = _mm256_loadu_ps(k3 + 8);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);

                    // k1
                    _r0 = _mm256_loadu_ps(r1);
                    _r0n = _mm256_loadu_ps(r1 + 8);
                    _k0 = _mm256_loadu_ps(k0 + 16);
                    _k0n = _mm256_loadu_ps(k0 + 24);
                    _k1 = _mm256_loadu_ps(k1 + 16);
                    _k1n = _mm256_loadu_ps(k1 + 24);
                    _k2 = _mm256_loadu_ps(k2 + 16);
                    _k2n = _mm256_loadu_ps(k2 + 24);
                    _k3 = _mm256_loadu_ps(k3 + 16);
                    _k3n = _mm256_loadu_ps(k3 + 24);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                    // k2
                    _r0 = _mm256_loadu_ps(r2);
                    _r0n = _mm256_loadu_ps(r2 + 8);
                    _k0 = _mm256_loadu_ps(k0 + 32);
                    _k0n = _mm256_loadu_ps(k0 + 40);
                    _k1 = _mm256_loadu_ps(k1 + 32);
                    _k1n = _mm256_loadu_ps(k1 + 40);
                    _k2 = _mm256_loadu_ps(k2 + 32);
                    _k2n = _mm256_loadu_ps(k2 + 40);
                    _k3 = _mm256_loadu_ps(k3 + 32);
                    _k3n = _mm256_loadu_ps(k3 + 40);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                    // k3
                    _r0 = _mm256_loadu_ps(r3);
                    _r0n = _mm256_loadu_ps(r3 + 8);
                    _k0 = _mm256_loadu_ps(k0 + 48);
                    _k0n = _mm256_loadu_ps(k0 + 56);
                    _k1 = _mm256_loadu_ps(k1 + 48);
                    _k1n = _mm256_loadu_ps(k1 + 56);
                    _k2 = _mm256_loadu_ps(k2 + 48);
                    _k2n = _mm256_loadu_ps(k2 + 56);
                    _k3 = _mm256_loadu_ps(k3 + 48);
                    _k3n = _mm256_loadu_ps(k3 + 56);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r0n = _mm256_loadu_ps(r0 + 8);
                    __m256 _k0 = _mm256_loadu_ps(k0);
                    __m256 _k0n = _mm256_loadu_ps(k0 + 8);
                    __m256 _k1 = _mm256_loadu_ps(k1);
                    __m256 _k1n = _mm256_loadu_ps(k1 + 8);
                    __m256 _k2 = _mm256_loadu_ps(k2);
                    __m256 _k2n = _mm256_loadu_ps(k2 + 8);
                    __m256 _k3 = _mm256_loadu_ps(k3);
                    __m256 _k3n = _mm256_loadu_ps(k3 + 8);

                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                }

                _mm256_storeu_ps(output0_tm, _sum0);
                _mm256_storeu_ps(output0_tm + 8, _sum0n);
                _mm256_storeu_ps(output1_tm, _sum1);
                _mm256_storeu_ps(output1_tm + 8, _sum1n);
                _mm256_storeu_ps(output2_tm, _sum2);
                _mm256_storeu_ps(output2_tm + 8, _sum2n);
                _mm256_storeu_ps(output3_tm, _sum3);
                _mm256_storeu_ps(output3_tm + 8, _sum3n);
#else
                float sum0[16] = {0.0f};
                float sum1[16] = {0.0f};
                float sum2[16] = {0.0f};
                float sum3[16] = {0.0f};

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r1[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r2[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r3[n] * k0[n];
                        k0 -= 16 * 3;

                        sum1[n] += r0[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r1[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r2[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r3[n] * k1[n];
                        k1 -= 16 * 3;

                        sum2[n] += r0[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r1[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r2[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r3[n] * k2[n];
                        k2 -= 16 * 3;

                        sum3[n] += r0[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r1[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r2[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r3[n] * k3[n];
                        k3 -= 16 * 3;
                    }
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum1[n] += r0[n] * k1[n];
                        sum2[n] += r0[n] * k2[n];
                        sum3[n] += r0[n] * k3[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                    output1_tm[n] = sum1[n];
                    output2_tm[n] = sum2[n];
                    output3_tm[n] = sum3[n];
                }
#endif
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i = 0; i < tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);

                float sum0[16] = {0.0f};

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel0_tm.row(q + 1);
                    const float* k2 = kernel0_tm.row(q + 2);
                    const float* k3 = kernel0_tm.row(q + 3);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum0[n] += r1[n] * k1[n];
                        sum0[n] += r2[n] * k2[n];
                        sum0[n] += r3[n] * k3[n];
                    }
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* k0 = kernel0_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    }
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // };

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            for (int j = 0; j < nColBlocks; j++)
            {
                float* outRow0 = out.row(j * 2);
                float* outRow1 = out.row(j * 2 + 1);

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float* out_tile = out_tm.row(j * nRowBlocks + i);

                    float s0[4], s1[4], s2[4], s3[4];
                    float w0[4], w1[4];
                    float d0[2], d1[2], d2[2], d3[2];
                    float o0[2], o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 4];
                        s2[n] = out_tile[n + 8];
                        s3[n] = out_tile[n + 12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n] + bias0;
                        o1[n] = d1[n] - d2[n] + d3[n] + bias0;
                    }
                    // save to top blob tm
                    outRow0[0] = o0[0];
                    outRow0[1] = o0[1];
                    outRow1[0] = o1[0];
                    outRow1[1] = o1[1];

                    outRow0 += 2;
                    outRow1 += 2;
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_sse(const Mat& kernel, std::vector<Mat>& kernel_tm2, int inch, int outch)
{
    Mat kernel_tm(6 * 6, inch, outch);

    // G
    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},
        {-1.0f / 6, -1.0f / 6, -1.0f / 6},
        {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6},
        {1.0f / 24, -1.0f / 12, 1.0f / 6},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    for (int r = 0; r < 9; r++)
    {
        Mat kernel_tm_test(4 * 8, inch, outch / 8 + (outch % 8) / 4 + outch % 4);

        int p = 0;
        for (; p + 7 < outch; p += 8)
        {
            const float* kernel0 = (const float*)kernel_tm.channel(p);
            const float* kernel1 = (const float*)kernel_tm.channel(p + 1);
            const float* kernel2 = (const float*)kernel_tm.channel(p + 2);
            const float* kernel3 = (const float*)kernel_tm.channel(p + 3);
            const float* kernel4 = (const float*)kernel_tm.channel(p + 4);
            const float* kernel5 = (const float*)kernel_tm.channel(p + 5);
            const float* kernel6 = (const float*)kernel_tm.channel(p + 6);
            const float* kernel7 = (const float*)kernel_tm.channel(p + 7);

            float* ktmp = kernel_tm_test.channel(p / 8);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp[16] = kernel4[r * 4 + 0];
                ktmp[17] = kernel4[r * 4 + 1];
                ktmp[18] = kernel4[r * 4 + 2];
                ktmp[19] = kernel4[r * 4 + 3];

                ktmp[20] = kernel5[r * 4 + 0];
                ktmp[21] = kernel5[r * 4 + 1];
                ktmp[22] = kernel5[r * 4 + 2];
                ktmp[23] = kernel5[r * 4 + 3];

                ktmp[24] = kernel6[r * 4 + 0];
                ktmp[25] = kernel6[r * 4 + 1];
                ktmp[26] = kernel6[r * 4 + 2];
                ktmp[27] = kernel6[r * 4 + 3];

                ktmp[28] = kernel7[r * 4 + 0];
                ktmp[29] = kernel7[r * 4 + 1];
                ktmp[30] = kernel7[r * 4 + 2];
                ktmp[31] = kernel7[r * 4 + 3];

                ktmp += 32;
                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
                kernel4 += 36;
                kernel5 += 36;
                kernel6 += 36;
                kernel7 += 36;
            }
        }

        for (; p + 3 < outch; p += 4)
        {
            const float* kernel0 = (const float*)kernel_tm.channel(p);
            const float* kernel1 = (const float*)kernel_tm.channel(p + 1);
            const float* kernel2 = (const float*)kernel_tm.channel(p + 2);
            const float* kernel3 = (const float*)kernel_tm.channel(p + 3);

            float* ktmp = kernel_tm_test.channel(p / 8 + (p % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp += 16;
                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
            }
        }

        for (; p < outch; p++)
        {
            const float* kernel0 = (const float*)kernel_tm.channel(p);

            float* ktmp = kernel_tm_test.channel(p / 8 + (p % 8) / 4 + p % 4);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp += 4;
                kernel0 += 36;
            }
        }
        kernel_tm2.push_back(kernel_tm_test);
    }
}

static void conv3x3s1_winograd43_sse(const Mat& bottom_blob, Mat& top_blob, const std::vector<Mat>& kernel_tm_test, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    size_t elemsize = bottom_blob.elemsize;
    const float* bias = _bias;

    // pad to 4n+2, winograd F(4,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;

    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt_b);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4, inch, tiles * 9, elemsize, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =	4 * r00  - 5 * r02	+ r04
        // 1 = -4 * (r01 + r02)  + r03 + r04
        // 2 =	4 * (r01 - r02)  - r03 + r04
        // 3 = -2 * r01 - r02 + 2 * r03 + r04
        // 4 =	2 * r01 - r02 - 2 * r03 + r04
        // 5 =	4 * r01 - 5 * r03 + r05

        // 0 =	4 * r00  - 5 * r02	+ r04
        // 1 = -4 * (r01 + r02)  + r03 + r04
        // 2 =	4 * (r01 - r02)  - r03 + r04
        // 3 = -2 * r01 - r02 + 2 * r03 + r04
        // 4 =	2 * r01 - r02 - 2 * r03 + r04
        // 5 =	4 * r01 - 5 * r03 + r05

#if __AVX__
        __m256 _1_n = _mm256_set1_ps(-1);
        __m256 _2_p = _mm256_set1_ps(2);
        __m256 _2_n = _mm256_set1_ps(-2);
        __m256 _4_p = _mm256_set1_ps(4);
        __m256 _4_n = _mm256_set1_ps(-4);
        __m256 _5_n = _mm256_set1_ps(-5);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const float* img = bottom_blob_bordered.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 4;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;
                const float* r4 = r3 + w;
                const float* r5 = r4 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float* out_tm0 = bottom_blob_tm.channel(tiles * 0 + j * nRowBlocks + i).row(q);
                    float* out_tm1 = bottom_blob_tm.channel(tiles * 1 + j * nRowBlocks + i).row(q);
                    float* out_tm2 = bottom_blob_tm.channel(tiles * 2 + j * nRowBlocks + i).row(q);
                    float* out_tm3 = bottom_blob_tm.channel(tiles * 3 + j * nRowBlocks + i).row(q);
                    float* out_tm4 = bottom_blob_tm.channel(tiles * 4 + j * nRowBlocks + i).row(q);
                    float* out_tm5 = bottom_blob_tm.channel(tiles * 5 + j * nRowBlocks + i).row(q);
                    float* out_tm6 = bottom_blob_tm.channel(tiles * 6 + j * nRowBlocks + i).row(q);
                    float* out_tm7 = bottom_blob_tm.channel(tiles * 7 + j * nRowBlocks + i).row(q);
                    float* out_tm8 = bottom_blob_tm.channel(tiles * 8 + j * nRowBlocks + i).row(q);
#if __AVX__
                    __m256 _d0, _d1, _d2, _d3, _d4, _d5;
                    __m256 _w0, _w1, _w2, _w3, _w4, _w5;
                    __m256 _t0, _t1, _t2, _t3, _t4, _t5;
                    __m256 _n0, _n1, _n2, _n3, _n4, _n5;
                    // load
                    _d0 = _mm256_loadu_ps(r0);
                    _d1 = _mm256_loadu_ps(r1);
                    _d2 = _mm256_loadu_ps(r2);
                    _d3 = _mm256_loadu_ps(r3);
                    _d4 = _mm256_loadu_ps(r4);
                    _d5 = _mm256_loadu_ps(r5);

                    // w = B_t * d
                    _w0 = _mm256_mul_ps(_d0, _4_p);
                    _w0 = _mm256_fmadd_ps(_d2, _5_n, _w0);
                    _w0 = _mm256_add_ps(_w0, _d4);

                    _w1 = _mm256_mul_ps(_d1, _4_n);
                    _w1 = _mm256_fmadd_ps(_d2, _4_n, _w1);
                    _w1 = _mm256_add_ps(_w1, _d3);
                    _w1 = _mm256_add_ps(_w1, _d4);

                    _w2 = _mm256_mul_ps(_d1, _4_p);
                    _w2 = _mm256_fmadd_ps(_d2, _4_n, _w2);
                    _w2 = _mm256_fmadd_ps(_d3, _1_n, _w2);
                    _w2 = _mm256_add_ps(_w2, _d4);

                    _w3 = _mm256_mul_ps(_d1, _2_n);
                    _w3 = _mm256_fmadd_ps(_d2, _1_n, _w3);
                    _w3 = _mm256_fmadd_ps(_d3, _2_p, _w3);
                    _w3 = _mm256_add_ps(_w3, _d4);

                    _w4 = _mm256_mul_ps(_d1, _2_p);
                    _w4 = _mm256_fmadd_ps(_d2, _1_n, _w4);
                    _w4 = _mm256_fmadd_ps(_d3, _2_n, _w4);
                    _w4 = _mm256_add_ps(_w4, _d4);

                    _w5 = _mm256_mul_ps(_d1, _4_p);
                    _w5 = _mm256_fmadd_ps(_d3, _5_n, _w5);
                    _w5 = _mm256_add_ps(_w5, _d5);
                    // transpose d to d_t
#ifdef _WIN32
                    {
                        _t0.m256_f32[0] = _w0.m256_f32[0];
                        _t1.m256_f32[0] = _w0.m256_f32[1];
                        _t2.m256_f32[0] = _w0.m256_f32[2];
                        _t3.m256_f32[0] = _w0.m256_f32[3];
                        _t4.m256_f32[0] = _w0.m256_f32[4];
                        _t5.m256_f32[0] = _w0.m256_f32[5];
                        _t0.m256_f32[1] = _w1.m256_f32[0];
                        _t1.m256_f32[1] = _w1.m256_f32[1];
                        _t2.m256_f32[1] = _w1.m256_f32[2];
                        _t3.m256_f32[1] = _w1.m256_f32[3];
                        _t4.m256_f32[1] = _w1.m256_f32[4];
                        _t5.m256_f32[1] = _w1.m256_f32[5];
                        _t0.m256_f32[2] = _w2.m256_f32[0];
                        _t1.m256_f32[2] = _w2.m256_f32[1];
                        _t2.m256_f32[2] = _w2.m256_f32[2];
                        _t3.m256_f32[2] = _w2.m256_f32[3];
                        _t4.m256_f32[2] = _w2.m256_f32[4];
                        _t5.m256_f32[2] = _w2.m256_f32[5];
                        _t0.m256_f32[3] = _w3.m256_f32[0];
                        _t1.m256_f32[3] = _w3.m256_f32[1];
                        _t2.m256_f32[3] = _w3.m256_f32[2];
                        _t3.m256_f32[3] = _w3.m256_f32[3];
                        _t4.m256_f32[3] = _w3.m256_f32[4];
                        _t5.m256_f32[3] = _w3.m256_f32[5];
                        _t0.m256_f32[4] = _w4.m256_f32[0];
                        _t1.m256_f32[4] = _w4.m256_f32[1];
                        _t2.m256_f32[4] = _w4.m256_f32[2];
                        _t3.m256_f32[4] = _w4.m256_f32[3];
                        _t4.m256_f32[4] = _w4.m256_f32[4];
                        _t5.m256_f32[4] = _w4.m256_f32[5];
                        _t0.m256_f32[5] = _w5.m256_f32[0];
                        _t1.m256_f32[5] = _w5.m256_f32[1];
                        _t2.m256_f32[5] = _w5.m256_f32[2];
                        _t3.m256_f32[5] = _w5.m256_f32[3];
                        _t4.m256_f32[5] = _w5.m256_f32[4];
                        _t5.m256_f32[5] = _w5.m256_f32[5];
                    }
#else
                    {
                        _t0[0] = _w0[0];
                        _t1[0] = _w0[1];
                        _t2[0] = _w0[2];
                        _t3[0] = _w0[3];
                        _t4[0] = _w0[4];
                        _t5[0] = _w0[5];
                        _t0[1] = _w1[0];
                        _t1[1] = _w1[1];
                        _t2[1] = _w1[2];
                        _t3[1] = _w1[3];
                        _t4[1] = _w1[4];
                        _t5[1] = _w1[5];
                        _t0[2] = _w2[0];
                        _t1[2] = _w2[1];
                        _t2[2] = _w2[2];
                        _t3[2] = _w2[3];
                        _t4[2] = _w2[4];
                        _t5[2] = _w2[5];
                        _t0[3] = _w3[0];
                        _t1[3] = _w3[1];
                        _t2[3] = _w3[2];
                        _t3[3] = _w3[3];
                        _t4[3] = _w3[4];
                        _t5[3] = _w3[5];
                        _t0[4] = _w4[0];
                        _t1[4] = _w4[1];
                        _t2[4] = _w4[2];
                        _t3[4] = _w4[3];
                        _t4[4] = _w4[4];
                        _t5[4] = _w4[5];
                        _t0[5] = _w5[0];
                        _t1[5] = _w5[1];
                        _t2[5] = _w5[2];
                        _t3[5] = _w5[3];
                        _t4[5] = _w5[4];
                        _t5[5] = _w5[5];
                    }
#endif
                    // d = B_t * d_t
                    _n0 = _mm256_mul_ps(_t0, _4_p);
                    _n0 = _mm256_fmadd_ps(_t2, _5_n, _n0);
                    _n0 = _mm256_add_ps(_n0, _t4);

                    _n1 = _mm256_mul_ps(_t1, _4_n);
                    _n1 = _mm256_fmadd_ps(_t2, _4_n, _n1);
                    _n1 = _mm256_add_ps(_n1, _t3);
                    _n1 = _mm256_add_ps(_n1, _t4);

                    _n2 = _mm256_mul_ps(_t1, _4_p);
                    _n2 = _mm256_fmadd_ps(_t2, _4_n, _n2);
                    _n2 = _mm256_fmadd_ps(_t3, _1_n, _n2);
                    _n2 = _mm256_add_ps(_n2, _t4);

                    _n3 = _mm256_mul_ps(_t1, _2_n);
                    _n3 = _mm256_fmadd_ps(_t2, _1_n, _n3);
                    _n3 = _mm256_fmadd_ps(_t3, _2_p, _n3);
                    _n3 = _mm256_add_ps(_n3, _t4);

                    _n4 = _mm256_mul_ps(_t1, _2_p);
                    _n4 = _mm256_fmadd_ps(_t2, _1_n, _n4);
                    _n4 = _mm256_fmadd_ps(_t3, _2_n, _n4);
                    _n4 = _mm256_add_ps(_n4, _t4);

                    _n5 = _mm256_mul_ps(_t1, _4_p);
                    _n5 = _mm256_fmadd_ps(_t3, _5_n, _n5);
                    _n5 = _mm256_add_ps(_n5, _t5);
                    // save to out_tm
                    float output_n0[8] = {0.f};
                    _mm256_storeu_ps(output_n0, _n0);
                    float output_n1[8] = {0.f};
                    _mm256_storeu_ps(output_n1, _n1);
                    float output_n2[8] = {0.f};
                    _mm256_storeu_ps(output_n2, _n2);
                    float output_n3[8] = {0.f};
                    _mm256_storeu_ps(output_n3, _n3);
                    float output_n4[8] = {0.f};
                    _mm256_storeu_ps(output_n4, _n4);
                    float output_n5[8] = {0.f};
                    _mm256_storeu_ps(output_n5, _n5);

                    out_tm0[0] = output_n0[0];
                    out_tm0[1] = output_n0[1];
                    out_tm0[2] = output_n0[2];
                    out_tm0[3] = output_n0[3];
                    out_tm1[0] = output_n0[4];
                    out_tm1[1] = output_n0[5];
                    out_tm1[2] = output_n1[0];
                    out_tm1[3] = output_n1[1];
                    out_tm2[0] = output_n1[2];
                    out_tm2[1] = output_n1[3];
                    out_tm2[2] = output_n1[4];
                    out_tm2[3] = output_n1[5];

                    out_tm3[0] = output_n2[0];
                    out_tm3[1] = output_n2[1];
                    out_tm3[2] = output_n2[2];
                    out_tm3[3] = output_n2[3];
                    out_tm4[0] = output_n2[4];
                    out_tm4[1] = output_n2[5];
                    out_tm4[2] = output_n3[0];
                    out_tm4[3] = output_n3[1];
                    out_tm5[0] = output_n3[2];
                    out_tm5[1] = output_n3[3];
                    out_tm5[2] = output_n3[4];
                    out_tm5[3] = output_n3[5];

                    out_tm6[0] = output_n4[0];
                    out_tm6[1] = output_n4[1];
                    out_tm6[2] = output_n4[2];
                    out_tm6[3] = output_n4[3];
                    out_tm7[0] = output_n4[4];
                    out_tm7[1] = output_n4[5];
                    out_tm7[2] = output_n5[0];
                    out_tm7[3] = output_n5[1];
                    out_tm8[0] = output_n5[2];
                    out_tm8[1] = output_n5[3];
                    out_tm8[2] = output_n5[4];
                    out_tm8[3] = output_n5[5];
#else
                    float d0[6], d1[6], d2[6], d3[6], d4[6], d5[6];
                    float w0[6], w1[6], w2[6], w3[6], w4[6], w5[6];
                    float t0[6], t1[6], t2[6], t3[6], t4[6], t5[6];

                    // load
                    for (int n = 0; n < 6; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                        d4[n] = r4[n];
                        d5[n] = r5[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 6; n++)
                    {
                        w0[n] = 4 * d0[n] - 5 * d2[n] + d4[n];
                        w1[n] = -4 * d1[n] - 4 * d2[n] + d3[n] + d4[n];
                        w2[n] = 4 * d1[n] - 4 * d2[n] - d3[n] + d4[n];
                        w3[n] = -2 * d1[n] - d2[n] + 2 * d3[n] + d4[n];
                        w4[n] = 2 * d1[n] - d2[n] - 2 * d3[n] + d4[n];
                        w5[n] = 4 * d1[n] - 5 * d3[n] + d5[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t4[0] = w0[4];
                        t5[0] = w0[5];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t4[1] = w1[4];
                        t5[1] = w1[5];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t4[2] = w2[4];
                        t5[2] = w2[5];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                        t4[3] = w3[4];
                        t5[3] = w3[5];
                        t0[4] = w4[0];
                        t1[4] = w4[1];
                        t2[4] = w4[2];
                        t3[4] = w4[3];
                        t4[4] = w4[4];
                        t5[4] = w4[5];
                        t0[5] = w5[0];
                        t1[5] = w5[1];
                        t2[5] = w5[2];
                        t3[5] = w5[3];
                        t4[5] = w5[4];
                        t5[5] = w5[5];
                    }
                    // d = B_t * d_t
                    for (int n = 0; n < 6; n++)
                    {
                        d0[n] = 4 * t0[n] - 5 * t2[n] + t4[n];
                        d1[n] = -4 * t1[n] - 4 * t2[n] + t3[n] + t4[n];
                        d2[n] = 4 * t1[n] - 4 * t2[n] - t3[n] + t4[n];
                        d3[n] = -2 * t1[n] - t2[n] + 2 * t3[n] + t4[n];
                        d4[n] = 2 * t1[n] - t2[n] - 2 * t3[n] + t4[n];
                        d5[n] = 4 * t1[n] - 5 * t3[n] + t5[n];
                    }
                    // save to out_tm
                    {
                        out_tm0[0] = d0[0];
                        out_tm0[1] = d0[1];
                        out_tm0[2] = d0[2];
                        out_tm0[3] = d0[3];
                        out_tm1[0] = d0[4];
                        out_tm1[1] = d0[5];
                        out_tm1[2] = d1[0];
                        out_tm1[3] = d1[1];
                        out_tm2[0] = d1[2];
                        out_tm2[1] = d1[3];
                        out_tm2[2] = d1[4];
                        out_tm2[3] = d1[5];

                        out_tm3[0] = d2[0];
                        out_tm3[1] = d2[1];
                        out_tm3[2] = d2[2];
                        out_tm3[3] = d2[3];
                        out_tm4[0] = d2[4];
                        out_tm4[1] = d2[5];
                        out_tm4[2] = d3[0];
                        out_tm4[3] = d3[1];
                        out_tm5[0] = d3[2];
                        out_tm5[1] = d3[3];
                        out_tm5[2] = d3[4];
                        out_tm5[3] = d3[5];

                        out_tm6[0] = d4[0];
                        out_tm6[1] = d4[1];
                        out_tm6[2] = d4[2];
                        out_tm6[3] = d4[3];
                        out_tm7[0] = d4[4];
                        out_tm7[1] = d4[5];
                        out_tm7[2] = d5[0];
                        out_tm7[3] = d5[1];
                        out_tm8[0] = d5[2];
                        out_tm8[1] = d5[3];
                        out_tm8[2] = d5[4];
                        out_tm8[3] = d5[5];
                    }
#endif // __AVX__
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(36, tiles, outch, elemsize, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 9; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp * 8;

                float* output0_tm = top_blob_tm.channel(p);
                float* output1_tm = top_blob_tm.channel(p + 1);
                float* output2_tm = top_blob_tm.channel(p + 2);
                float* output3_tm = top_blob_tm.channel(p + 3);
                float* output4_tm = top_blob_tm.channel(p + 4);
                float* output5_tm = top_blob_tm.channel(p + 5);
                float* output6_tm = top_blob_tm.channel(p + 6);
                float* output7_tm = top_blob_tm.channel(p + 7);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;
                output4_tm = output4_tm + r * 4;
                output5_tm = output5_tm + r * 4;
                output6_tm = output6_tm + r * 4;
                output7_tm = output7_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm_test[r].channel(p / 8);
                    const float* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __AVX__ || __SSE__
#if __AVX__
                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum1 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum2 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum3 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum4 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum5 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum6 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum7 = _mm_broadcast_ss(&zero_val);
#else
                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);
                    __m128 _sum4 = _mm_set1_ps(0.f);
                    __m128 _sum5 = _mm_set1_ps(0.f);
                    __m128 _sum6 = _mm_set1_ps(0.f);
                    __m128 _sum7 = _mm_set1_ps(0.f);
#endif
                    int q = 0;
                    for (; q + 3 < inch; q = q + 4)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _r1 = _mm_loadu_ps(r0 + 4);
                        __m128 _r2 = _mm_loadu_ps(r0 + 8);
                        __m128 _r3 = _mm_loadu_ps(r0 + 12);

                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
                        __m128 _k4 = _mm_loadu_ps(kptr + 16);
                        __m128 _k5 = _mm_loadu_ps(kptr + 20);
                        __m128 _k6 = _mm_loadu_ps(kptr + 24);
                        __m128 _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r0, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r0, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r0, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r0, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r0, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r0, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r0, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r0, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r0, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r0, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r0, _k7));
#endif
                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r1, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r1, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r1, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r1, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r1, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r1, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r1, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r1, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r1, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r1, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r1, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r1, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r1, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r1, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r1, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r1, _k7));
#endif

                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r2, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r2, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r2, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r2, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r2, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r2, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r2, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r2, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r2, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r2, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r2, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r2, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r2, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r2, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r2, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r2, _k7));
#endif
                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r3, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r3, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r3, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r3, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r3, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r3, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r3, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r3, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r3, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r3, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r3, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r3, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r3, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r3, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r3, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r3, _k7));
#endif
                        kptr += 32;
                        r0 += 16;
                    }

                    for (; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
                        __m128 _k4 = _mm_loadu_ps(kptr + 16);
                        __m128 _k5 = _mm_loadu_ps(kptr + 20);
                        __m128 _k6 = _mm_loadu_ps(kptr + 24);
                        __m128 _k7 = _mm_loadu_ps(kptr + 28);

#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r0, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r0, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r0, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r0, _k7, _sum7);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r0, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r0, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r0, _k3));
                        _sum4 = _mm_add_ps(_sum4, _mm_mul_ps(_r0, _k4));
                        _sum5 = _mm_add_ps(_sum5, _mm_mul_ps(_r0, _k5));
                        _sum6 = _mm_add_ps(_sum6, _mm_mul_ps(_r0, _k6));
                        _sum7 = _mm_add_ps(_sum7, _mm_mul_ps(_r0, _k7));
#endif

                        kptr += 32;
                        r0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
                    _mm_storeu_ps(output4_tm, _sum4);
                    _mm_storeu_ps(output5_tm, _sum5);
                    _mm_storeu_ps(output6_tm, _sum6);
                    _mm_storeu_ps(output7_tm, _sum7);
#else
                    float sum0[4] = {0};
                    float sum1[4] = {0};
                    float sum2[4] = {0};
                    float sum3[4] = {0};
                    float sum4[4] = {0};
                    float sum5[4] = {0};
                    float sum6[4] = {0};
                    float sum7[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += r0[n] * kptr[n];
                            sum1[n] += r0[n] * kptr[n + 4];
                            sum2[n] += r0[n] * kptr[n + 8];
                            sum3[n] += r0[n] * kptr[n + 12];
                            sum4[n] += r0[n] * kptr[n + 16];
                            sum5[n] += r0[n] * kptr[n + 20];
                            sum6[n] += r0[n] * kptr[n + 24];
                            sum7[n] += r0[n] * kptr[n + 28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }
#endif // __AVX__
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                    output4_tm += 36;
                    output5_tm += 36;
                    output6_tm += 36;
                    output7_tm += 36;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                float* output0_tm = top_blob_tm.channel(p);
                float* output1_tm = top_blob_tm.channel(p + 1);
                float* output2_tm = top_blob_tm.channel(p + 2);
                float* output3_tm = top_blob_tm.channel(p + 3);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4);
                    const float* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __AVX__ || __SSE__
#if __AVX__
                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum1 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum2 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum3 = _mm_broadcast_ss(&zero_val);
#else
                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);
#endif
                    for (int q = 0; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
                        _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_r0, _k1));
                        _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_r0, _k2));
                        _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_r0, _k3));
#endif
                        kptr += 16;
                        r0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
#else
                    float sum0[4] = {0};
                    float sum1[4] = {0};
                    float sum2[4] = {0};
                    float sum3[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += r0[n] * kptr[n];
                            sum1[n] += r0[n] * kptr[n + 4];
                            sum2[n] += r0[n] * kptr[n + 8];
                            sum3[n] += r0[n] * kptr[n + 12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
#endif // __AVX__
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                }
            }

            remain_outch_start += nn_outch << 2;

            for (int p = remain_outch_start; p < outch; p++)
            {
                float* output0_tm = top_blob_tm.channel(p);

                output0_tm = output0_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4 + p % 4);
                    const float* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __AVX__ || __SSE__
#if __AVX__
                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
#else
                    __m128 _sum0 = _mm_set1_ps(0.f);
#endif

                    for (int q = 0; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
#if __AVX__
                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
#else
                        _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_r0, _k0));
#endif
                        kptr += 16;
                        r0 += 4;
                    }
                    _mm_storeu_ps(output0_tm, _sum0);
#else
                    float sum0[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                        }
                        kptr += 4;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }
#endif // __AVX__ || __SSE__
                    output0_tm += 36;
                }
            }

            // for (int p=0; p<outch; p++)
            // {
            //     Mat out0_tm = top_blob_tm.channel(p);
            //     const Mat kernel0_tm = kernel_tm.channel(p);

            //     for (int i=0; i<tiles; i++)
            //     {
            //         float* output0_tm = out0_tm.row<int>(i);

            //         int sum0[36] = {0};

            //         for (int q=0; q<inch; q++)
            //         {
            //             const float* r0 = bottom_blob_tm.channel(q).row<float>(i);
            //             const float* k0 = kernel0_tm.row<float>(q);

            //             for (int n=0; n<36; n++)
            //             {
            //                 sum0[n] += (int)r0[n] * k0[n];
            //             }
            //         }

            //         for (int n=0; n<36; n++)
            //         {
            //             output0_tm[n] = sum0[n];
            //         }
            //     }
            // }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, opt.workspace_allocator);
    }
    {
        // AT
        // const float itm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 =	r00 + r01 + r02 + r03 +	r04
        // 1 =		  r01 - r02 + 2 * (r03 - r04)
        // 2 =		  r01 + r02 + 4 * (r03 + r04)
        // 3 =		  r01 - r02 + 8 * (r03 - r04)  + r05

        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            float* out_tile = top_blob_tm.channel(p);
            float* outRow0 = top_blob_bordered.channel(p);
            float* outRow1 = outRow0 + outw;
            float* outRow2 = outRow0 + outw * 2;
            float* outRow3 = outRow0 + outw * 3;

            const float bias0 = bias ? bias[p] : 0.f;

            for (int j = 0; j < nColBlocks; j++)
            {
                for (int i = 0; i < nRowBlocks; i++)
                {
                    // TODO AVX2
                    float s0[6], s1[6], s2[6], s3[6], s4[6], s5[6];
                    float w0[6], w1[6], w2[6], w3[6];
                    float d0[4], d1[4], d2[4], d3[4], d4[4], d5[4];
                    float o0[4], o1[4], o2[4], o3[4];

                    // load
                    for (int n = 0; n < 6; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 6];
                        s2[n] = out_tile[n + 12];
                        s3[n] = out_tile[n + 18];
                        s4[n] = out_tile[n + 24];
                        s5[n] = out_tile[n + 30];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 6; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
                        w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
                        w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
                        w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + s5[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d0[2] = w2[0];
                        d0[3] = w3[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d1[2] = w2[1];
                        d1[3] = w3[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d2[2] = w2[2];
                        d2[3] = w3[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                        d3[2] = w2[3];
                        d3[3] = w3[3];
                        d4[0] = w0[4];
                        d4[1] = w1[4];
                        d4[2] = w2[4];
                        d4[3] = w3[4];
                        d5[0] = w0[5];
                        d5[1] = w1[5];
                        d5[2] = w2[5];
                        d5[3] = w3[5];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 4; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n] + d3[n] + d4[n];
                        o1[n] = d1[n] - d2[n] + 2 * d3[n] - 2 * d4[n];
                        o2[n] = d1[n] + d2[n] + 4 * d3[n] + 4 * d4[n];
                        o3[n] = d1[n] - d2[n] + 8 * d3[n] - 8 * d4[n] + d5[n];
                    }
                    // save to top blob tm
                    for (int n = 0; n < 4; n++)
                    {
                        outRow0[n] = o0[n] + bias0;
                        outRow1[n] = o1[n] + bias0;
                        outRow2[n] = o2[n] + bias0;
                        outRow3[n] = o3[n] + bias0;
                    }

                    out_tile += 36;

                    outRow0 += 4;
                    outRow1 += 4;
                    outRow2 += 4;
                    outRow3 += 4;
                }

                outRow0 += outw * 3;
                outRow1 += outw * 3;
                outRow2 += outw * 3;
                outRow3 += outw * 3;
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;

            const float* img = bottom_blob.channel(q);
            const float* kernel0 = kernel + p * inch * 9 + q * 9;

            const float* r0 = img;
            const float* r1 = img + w;
            const float* r2 = img + w * 2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;

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
}
