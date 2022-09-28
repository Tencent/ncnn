// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_winograd_dot_sse(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 4u, 1, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __AVX__
    if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, batch, 4u, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 4u, opt.workspace_allocator);
    else
        bottom_blob_tm2.create(1 * inch, tiles, batch, 4u, opt.workspace_allocator);
#elif __SSE2__
    if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 4u, opt.workspace_allocator);
    else
        bottom_blob_tm2.create(1 * inch, tiles, batch, 4u, opt.workspace_allocator);
#else
    bottom_blob_tm2.create(1 * inch, tiles, batch, 4u, opt.workspace_allocator);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
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

    top_blob_tm.create(tiles, batch, outch, 4u, opt.workspace_allocator);

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

        for (int r = 0; r < batch; r++)
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

        for (int r = 0; r < batch; r++)
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

        for (int r = 0; r < batch; r++)
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
