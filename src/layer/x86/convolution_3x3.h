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

static void conv3x3s1_winograd23_transform_kernel_sse(const Mat& kernel, Mat& kernel_tm2, int inch, int outch, const Option& opt)
{
    Mat kernel_tm(4 * 4, inch, outch);

    // G
    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
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

    // interleave
    // src = 16-inch-outch
    // dst = inch-16-outch
#if __SSE2__
    kernel_tm2.create(8 * inch, 16, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm2.create(inch, 16, outch);
#endif

    int q = 0;
#if __SSE2__
    for (; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm2.channel(q / 8);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif
    for (; q < outch; q++)
    {
#if __SSE2__
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        Mat g0 = kernel_tm2.channel(q);
#endif

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd23_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, 4u, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_sse(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        const int tiles = h_tm / 4 * w_tm / 4;

        // permute
        Mat bottom_blob_tm2;
#if __AVX__
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 16, 4u, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 16, 4u, opt.workspace_allocator);
        else
            bottom_blob_tm2.create(1 * inch, tiles, 16, 4u, opt.workspace_allocator);
#elif __SSE2__
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 16, 4u, opt.workspace_allocator);
        else
            bottom_blob_tm2.create(1 * inch, tiles, 16, 4u, opt.workspace_allocator);
#else
        bottom_blob_tm2.create(1 * inch, tiles, 16, 4u, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 16; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
#if __SSE2__
#if __AVX__
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    _mm256_storeu_ps(tmpptr, _r0);

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 8;
                }
            }
#endif // __AVX__
            for (; i + 3 < tiles; i += 4)
            {
#if __AVX__
                float* tmpptr = tm2.row(i / 8 + (i % 8) / 4);
#else
                float* tmpptr = tm2.row(i / 4);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    __m128 _r0 = _mm_loadu_ps(r0);
                    _mm_storeu_ps(tmpptr, _r0);

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 4;
                }
            }
#endif // __SSE2__
            for (; i < tiles; i++)
            {
#if __AVX__
                float* tmpptr = tm2.row(i / 8 + (i % 8) / 4 + i % 4);
#elif __SSE2__
                float* tmpptr = tm2.row(i / 4 + i % 4);
#else
                float* tmpptr = tm2.row(i);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    tmpptr[0] = r0[0];

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 1;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 16, outch, 4u, opt.workspace_allocator);

#if __SSE2__
        int nn_outch = outch >> 3;
        int remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
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

            const Mat kernel0_tm = kernel_tm.channel(p / 8);

            for (int r = 0; r < 16; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(k0 + 4);
                        __m256 _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(k0 + 6);
                        __m256 _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(k0 + 4);
                        __m256 _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(k0 + 6);
                        __m256 _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output3_tm, _sum3);
                    _mm256_storeu_ps(output4_tm, _sum4);
                    _mm256_storeu_ps(output5_tm, _sum5);
                    _mm256_storeu_ps(output6_tm, _sum6);
                    _mm256_storeu_ps(output7_tm, _sum7);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                    output4_tm += 8;
                    output5_tm += 8;
                    output6_tm += 8;
                    output7_tm += 8;
                }
#endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val = _mm_loadu_ps(r0);

                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        __m128 _w4 = _mm_load1_ps(k0 + 4);
                        __m128 _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        __m128 _w6 = _mm_load1_ps(k0 + 6);
                        __m128 _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);

                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        __m128 _w4 = _mm_load1_ps(k0 + 4);
                        __m128 _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        __m128 _w6 = _mm_load1_ps(k0 + 6);
                        __m128 _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
                    _mm_storeu_ps(output4_tm, _sum4);
                    _mm_storeu_ps(output5_tm, _sum5);
                    _mm_storeu_ps(output6_tm, _sum6);
                    _mm_storeu_ps(output7_tm, _sum7);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                    output4_tm += 4;
                    output5_tm += 4;
                    output6_tm += 4;
                    output7_tm += 4;
                }
                for (; i < tiles; i++)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#else
                    const float* r0 = bb2.row(i / 4 + i % 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

#if __AVX__
                    __m256 _sum = _mm256_setzero_ps();
#else
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
#endif

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
#if __AVX__
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _w0 = _mm256_loadu_ps(k0);
                        _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        __m256 _w1 = _mm256_loadu_ps(k0 + 8);
                        _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);

                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _w2 = _mm256_loadu_ps(k0 + 16);
                        _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);

                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        __m256 _w3 = _mm256_loadu_ps(k0 + 24);
                        _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
#else
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w00 = _mm_loadu_ps(k0);
                        __m128 _w01 = _mm_loadu_ps(k0 + 4);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w00, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val0, _w01, _sum1);

                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _w10 = _mm_loadu_ps(k0 + 8);
                        __m128 _w11 = _mm_loadu_ps(k0 + 12);
                        _sum0 = _mm_comp_fmadd_ps(_val1, _w10, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w11, _sum1);

                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _w20 = _mm_loadu_ps(k0 + 16);
                        __m128 _w21 = _mm_loadu_ps(k0 + 20);
                        _sum0 = _mm_comp_fmadd_ps(_val2, _w20, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val2, _w21, _sum1);

                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _w30 = _mm_loadu_ps(k0 + 24);
                        __m128 _w31 = _mm_loadu_ps(k0 + 28);
                        _sum0 = _mm_comp_fmadd_ps(_val3, _w30, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val3, _w31, _sum1);
#endif
                        r0 += 4;
                        k0 += 32;
                    }
                    for (; j < nn; j++)
                    {
#if __AVX__
                        __m256 _val = _mm256_broadcast_ss(r0);
                        __m256 _w = _mm256_loadu_ps(k0);
                        _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
#else
                        __m128 _val = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        __m128 _w1 = _mm_loadu_ps(k0 + 4);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
#endif
                        r0 += 1;
                        k0 += 8;
                    }

                    float sum[8];
#if __AVX__
                    _mm256_storeu_ps(sum, _sum);
#else
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
#endif

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];
                    output4_tm[0] = sum[4];
                    output5_tm[0] = sum[5];
                    output6_tm[0] = sum[6];
                    output7_tm[0] = sum[7];

                    output0_tm++;
                    output1_tm++;
                    output2_tm++;
                    output3_tm++;
                    output4_tm++;
                    output5_tm++;
                    output6_tm++;
                    output7_tm++;
                }
            }
        }

        nn_outch = (outch - remain_outch_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = remain_outch_start + pp * 4;

            float* output0_tm = top_blob_tm.channel(p);
            float* output1_tm = top_blob_tm.channel(p + 1);
            float* output2_tm = top_blob_tm.channel(p + 2);
            float* output3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4);

            for (int r = 0; r < 16; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output3_tm, _sum3);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                }
#endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                }
                for (; i < tiles; i++)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#else
                    const float* r0 = bb2.row(i / 4 + i % 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _w1 = _mm_loadu_ps(k0 + 4);
                        _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);

                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _w2 = _mm_loadu_ps(k0 + 8);
                        _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);

                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _w3 = _mm_loadu_ps(k0 + 12);
                        _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);

                        r0 += 4;
                        k0 += 16;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        _sum = _mm_comp_fmadd_ps(_val, _w0, _sum);

                        r0 += 1;
                        k0 += 4;
                    }

                    float sum[4];
                    _mm_storeu_ps(sum, _sum);

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];

                    output0_tm++;
                    output1_tm++;
                    output2_tm++;
                    output3_tm++;
                }
            }
        }

        remain_outch_start += nn_outch << 2;
#else
        int remain_outch_start = 0;
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

#if __SSE2__
            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const Mat kernel0_tm = kernel_tm.channel(p);
#endif

            for (int r = 0; r < 16; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __SSE2__
#if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val0 = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        __m256 _val1 = _mm256_loadu_ps(r0 + 8);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val1, _w1, _sum0);

                        __m256 _val2 = _mm256_loadu_ps(r0 + 16);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        _sum0 = _mm256_comp_fmadd_ps(_val2, _w2, _sum0);

                        __m256 _val3 = _mm256_loadu_ps(r0 + 24);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val3, _w3, _sum0);

                        r0 += 32;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        r0 += 8;
                        k0++;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    output0_tm += 8;
                }
#endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val0 = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                        __m128 _val1 = _mm_loadu_ps(r0 + 4);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val1, _w1, _sum0);

                        __m128 _val2 = _mm_loadu_ps(r0 + 8);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        _sum0 = _mm_comp_fmadd_ps(_val2, _w2, _sum0);

                        __m128 _val3 = _mm_loadu_ps(r0 + 12);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val3, _w3, _sum0);

                        r0 += 16;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        r0 += 4;
                        k0++;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    output0_tm += 4;
                }
#endif // __SSE2__
                for (; i < tiles; i++)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#elif __SSE2__
                    const float* r0 = bb2.row(i / 4 + i % 4);
#else
                    const float* r0 = bb2.row(i);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    float sum = 0.f;

                    for (int j = 0; j < nn; j++)
                    {
                        float w0 = k0[0];
                        float val0 = r0[0];
                        sum += val0 * w0;

                        r0 += 1;
                        k0 += 1;
                    }

                    output0_tm[0] = sum;
                    output0_tm += 1;
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
        conv3x3s1_winograd23_transform_output_sse(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_sse(const Mat& kernel, Mat& kernel_tm2, int inch, int outch, const Option& opt)
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

    #pragma omp parallel for num_threads(opt.num_threads)
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

    // interleave
    // src = 36-inch-outch
    // dst = inch-36-outch
#if __SSE2__
    kernel_tm2.create(8 * inch, 36, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm2.create(inch, 36, outch);
#endif

    int q = 0;
#if __SSE2__
    for (; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm2.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif
    for (; q < outch; q++)
    {
#if __SSE2__
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        Mat g0 = kernel_tm2.channel(q);
#endif

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd43_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

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
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, 4u, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_sse(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = h_tm / 6 * w_tm / 6;

        // permute
        Mat bottom_blob_tm2;
#if __AVX__
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 36, 4u, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 36, 4u, opt.workspace_allocator);
        else
            bottom_blob_tm2.create(1 * inch, tiles, 36, 4u, opt.workspace_allocator);
#elif __SSE2__
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 36, 4u, opt.workspace_allocator);
        else
            bottom_blob_tm2.create(1 * inch, tiles, 36, 4u, opt.workspace_allocator);
#else
        bottom_blob_tm2.create(1 * inch, tiles, 36, 4u, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
#if __SSE2__
#if __AVX__
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    _mm256_storeu_ps(tmpptr, _r0);

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 8;
                }
            }
#endif // __AVX__
            for (; i + 3 < tiles; i += 4)
            {
#if __AVX__
                float* tmpptr = tm2.row(i / 8 + (i % 8) / 4);
#else
                float* tmpptr = tm2.row(i / 4);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    __m128 _r0 = _mm_loadu_ps(r0);
                    _mm_storeu_ps(tmpptr, _r0);

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 4;
                }
            }
#endif // __SSE2__
            for (; i < tiles; i++)
            {
#if __AVX__
                float* tmpptr = tm2.row(i / 8 + (i % 8) / 4 + i % 4);
#elif __SSE2__
                float* tmpptr = tm2.row(i / 4 + i % 4);
#else
                float* tmpptr = tm2.row(i);
#endif

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    tmpptr[0] = r0[0];

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 1;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 36, outch, 4u, opt.workspace_allocator);

#if __SSE2__
        int nn_outch = outch >> 3;
        int remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
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

            const Mat kernel0_tm = kernel_tm.channel(p / 8);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(k0 + 4);
                        __m256 _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(k0 + 6);
                        __m256 _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(k0 + 4);
                        __m256 _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(k0 + 6);
                        __m256 _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output3_tm, _sum3);
                    _mm256_storeu_ps(output4_tm, _sum4);
                    _mm256_storeu_ps(output5_tm, _sum5);
                    _mm256_storeu_ps(output6_tm, _sum6);
                    _mm256_storeu_ps(output7_tm, _sum7);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                    output4_tm += 8;
                    output5_tm += 8;
                    output6_tm += 8;
                    output7_tm += 8;
                }
#endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val = _mm_loadu_ps(r0);

                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        __m128 _w4 = _mm_load1_ps(k0 + 4);
                        __m128 _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        __m128 _w6 = _mm_load1_ps(k0 + 6);
                        __m128 _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);

                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        __m128 _w4 = _mm_load1_ps(k0 + 4);
                        __m128 _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        __m128 _w6 = _mm_load1_ps(k0 + 6);
                        __m128 _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
                    _mm_storeu_ps(output4_tm, _sum4);
                    _mm_storeu_ps(output5_tm, _sum5);
                    _mm_storeu_ps(output6_tm, _sum6);
                    _mm_storeu_ps(output7_tm, _sum7);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                    output4_tm += 4;
                    output5_tm += 4;
                    output6_tm += 4;
                    output7_tm += 4;
                }
                for (; i < tiles; i++)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#else
                    const float* r0 = bb2.row(i / 4 + i % 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

#if __AVX__
                    __m256 _sum = _mm256_setzero_ps();
#else
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
#endif

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
#if __AVX__
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _w0 = _mm256_loadu_ps(k0);
                        _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        __m256 _w1 = _mm256_loadu_ps(k0 + 8);
                        _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);

                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _w2 = _mm256_loadu_ps(k0 + 16);
                        _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);

                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        __m256 _w3 = _mm256_loadu_ps(k0 + 24);
                        _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
#else
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w00 = _mm_loadu_ps(k0);
                        __m128 _w01 = _mm_loadu_ps(k0 + 4);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w00, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val0, _w01, _sum1);

                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _w10 = _mm_loadu_ps(k0 + 8);
                        __m128 _w11 = _mm_loadu_ps(k0 + 12);
                        _sum0 = _mm_comp_fmadd_ps(_val1, _w10, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w11, _sum1);

                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _w20 = _mm_loadu_ps(k0 + 16);
                        __m128 _w21 = _mm_loadu_ps(k0 + 20);
                        _sum0 = _mm_comp_fmadd_ps(_val2, _w20, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val2, _w21, _sum1);

                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _w30 = _mm_loadu_ps(k0 + 24);
                        __m128 _w31 = _mm_loadu_ps(k0 + 28);
                        _sum0 = _mm_comp_fmadd_ps(_val3, _w30, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val3, _w31, _sum1);
#endif
                        r0 += 4;
                        k0 += 32;
                    }
                    for (; j < nn; j++)
                    {
#if __AVX__
                        __m256 _val = _mm256_broadcast_ss(r0);
                        __m256 _w = _mm256_loadu_ps(k0);
                        _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
#else
                        __m128 _val = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        __m128 _w1 = _mm_loadu_ps(k0 + 4);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
#endif
                        r0 += 1;
                        k0 += 8;
                    }

                    float sum[8];
#if __AVX__
                    _mm256_storeu_ps(sum, _sum);
#else
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
#endif

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];
                    output4_tm[0] = sum[4];
                    output5_tm[0] = sum[5];
                    output6_tm[0] = sum[6];
                    output7_tm[0] = sum[7];

                    output0_tm++;
                    output1_tm++;
                    output2_tm++;
                    output3_tm++;
                    output4_tm++;
                    output5_tm++;
                    output6_tm++;
                    output7_tm++;
                }
            }
        }

        nn_outch = (outch - remain_outch_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = remain_outch_start + pp * 4;

            float* output0_tm = top_blob_tm.channel(p);
            float* output1_tm = top_blob_tm.channel(p + 1);
            float* output2_tm = top_blob_tm.channel(p + 2);
            float* output3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output3_tm, _sum3);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                }
#endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                }
                for (; i < tiles; i++)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#else
                    const float* r0 = bb2.row(i / 4 + i % 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _w1 = _mm_loadu_ps(k0 + 4);
                        _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);

                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _w2 = _mm_loadu_ps(k0 + 8);
                        _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);

                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _w3 = _mm_loadu_ps(k0 + 12);
                        _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);

                        r0 += 4;
                        k0 += 16;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        _sum = _mm_comp_fmadd_ps(_val, _w0, _sum);

                        r0 += 1;
                        k0 += 4;
                    }

                    float sum[4];
                    _mm_storeu_ps(sum, _sum);

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];

                    output0_tm++;
                    output1_tm++;
                    output2_tm++;
                    output3_tm++;
                }
            }
        }

        remain_outch_start += nn_outch << 2;
#else
        int remain_outch_start = 0;
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

#if __SSE2__
            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const Mat kernel0_tm = kernel_tm.channel(p);
#endif

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __SSE2__
#if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val0 = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        __m256 _val1 = _mm256_loadu_ps(r0 + 8);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val1, _w1, _sum0);

                        __m256 _val2 = _mm256_loadu_ps(r0 + 16);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        _sum0 = _mm256_comp_fmadd_ps(_val2, _w2, _sum0);

                        __m256 _val3 = _mm256_loadu_ps(r0 + 24);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val3, _w3, _sum0);

                        r0 += 32;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        r0 += 8;
                        k0++;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    output0_tm += 8;
                }
#endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);
#else
                    const float* r0 = bb2.row(i / 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val0 = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                        __m128 _val1 = _mm_loadu_ps(r0 + 4);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val1, _w1, _sum0);

                        __m128 _val2 = _mm_loadu_ps(r0 + 8);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        _sum0 = _mm_comp_fmadd_ps(_val2, _w2, _sum0);

                        __m128 _val3 = _mm_loadu_ps(r0 + 12);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val3, _w3, _sum0);

                        r0 += 16;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        r0 += 4;
                        k0++;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    output0_tm += 4;
                }
#endif // __SSE2__
                for (; i < tiles; i++)
                {
#if __AVX__
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);
#elif __SSE2__
                    const float* r0 = bb2.row(i / 4 + i % 4);
#else
                    const float* r0 = bb2.row(i);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    float sum = 0.f;

                    for (int j = 0; j < nn; j++)
                    {
                        float w0 = k0[0];
                        float val0 = r0[0];
                        sum += val0 * w0;

                        r0 += 1;
                        k0 += 1;
                    }

                    output0_tm[0] = sum;
                    output0_tm += 1;
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
        conv3x3s1_winograd43_transform_output_sse(top_blob_tm, top_blob_bordered, bias, opt);
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
