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

static void im2col_sgemm_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
#if __SSE2__
#if __AVX__
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
#else
    if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
#endif
    {
#if __AVX__
        int nn_size = size / 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    __m256 _r0 = _mm256_loadu_ps(img0);
                    _mm256_storeu_ps(tmpptr, _r0);
                    img0 += size;
                    tmpptr += 8;
                }
            }
        }

        int remain_size_start = nn_size * 8;
        nn_size = (size - remain_size_start) / 4;
#else
        int nn_size = size / 4;
        int remain_size_start = 0;
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

#if __AVX__
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#else
            float* tmpptr = tmp.channel(i / 4);
#endif

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    __m128 _r0 = _mm_loadu_ps(img0);
                    _mm_storeu_ps(tmpptr, _r0);
                    img0 += size;
                    tmpptr += 4;
                }
            }
        }

        remain_size_start += nn_size * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
#if __AVX__
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#else
            float* tmpptr = tmp.channel(i / 4 + i % 4);
#endif

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }
#else // __SSE2__
    tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            float* tmpptr = tmp.channel(i);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }
#endif // __SSE2__

#if __SSE2__
    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);
        float* outptr4 = top_blob.channel(p + 4);
        float* outptr5 = top_blob.channel(p + 5);
        float* outptr6 = top_blob.channel(p + 6);
        float* outptr7 = top_blob.channel(p + 7);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
#if __AVX__
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            __m256 _sum0 = _mm256_broadcast_ss(biasptr);
            __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
            __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
            __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);
            __m256 _sum4 = _mm256_broadcast_ss(biasptr + 4);
            __m256 _sum5 = _mm256_broadcast_ss(biasptr + 5);
            __m256 _sum6 = _mm256_broadcast_ss(biasptr + 6);
            __m256 _sum7 = _mm256_broadcast_ss(biasptr + 7);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m256 _val = _mm256_loadu_ps(tmpptr);

                __m256 _w0 = _mm256_broadcast_ss(kptr);
                __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                __m256 _w4 = _mm256_broadcast_ss(kptr + 4);
                __m256 _w5 = _mm256_broadcast_ss(kptr + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                __m256 _w6 = _mm256_broadcast_ss(kptr + 6);
                __m256 _w7 = _mm256_broadcast_ss(kptr + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 8;
                kptr += 8;

                _val = _mm256_loadu_ps(tmpptr);

                _w0 = _mm256_broadcast_ss(kptr);
                _w1 = _mm256_broadcast_ss(kptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _w2 = _mm256_broadcast_ss(kptr + 2);
                _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                _w4 = _mm256_broadcast_ss(kptr + 4);
                _w5 = _mm256_broadcast_ss(kptr + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                _w6 = _mm256_broadcast_ss(kptr + 6);
                _w7 = _mm256_broadcast_ss(kptr + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 8;
                kptr += 8;

                _val = _mm256_loadu_ps(tmpptr);

                _w0 = _mm256_broadcast_ss(kptr);
                _w1 = _mm256_broadcast_ss(kptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _w2 = _mm256_broadcast_ss(kptr + 2);
                _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                _w4 = _mm256_broadcast_ss(kptr + 4);
                _w5 = _mm256_broadcast_ss(kptr + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                _w6 = _mm256_broadcast_ss(kptr + 6);
                _w7 = _mm256_broadcast_ss(kptr + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 8;
                kptr += 8;

                _val = _mm256_loadu_ps(tmpptr);

                _w0 = _mm256_broadcast_ss(kptr);
                _w1 = _mm256_broadcast_ss(kptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _w2 = _mm256_broadcast_ss(kptr + 2);
                _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                _w4 = _mm256_broadcast_ss(kptr + 4);
                _w5 = _mm256_broadcast_ss(kptr + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                _w6 = _mm256_broadcast_ss(kptr + 6);
                _w7 = _mm256_broadcast_ss(kptr + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 8;
                kptr += 8;
            }
            for (; j < nn; j++)
            {
                __m256 _val = _mm256_loadu_ps(tmpptr);

                __m256 _w0 = _mm256_broadcast_ss(kptr);
                __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                __m256 _w4 = _mm256_broadcast_ss(kptr + 4);
                __m256 _w5 = _mm256_broadcast_ss(kptr + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                __m256 _w6 = _mm256_broadcast_ss(kptr + 6);
                __m256 _w7 = _mm256_broadcast_ss(kptr + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 8;
                kptr += 8;
            }

            _mm256_storeu_ps(outptr0, _sum0);
            _mm256_storeu_ps(outptr1, _sum1);
            _mm256_storeu_ps(outptr2, _sum2);
            _mm256_storeu_ps(outptr3, _sum3);
            _mm256_storeu_ps(outptr4, _sum4);
            _mm256_storeu_ps(outptr5, _sum5);
            _mm256_storeu_ps(outptr6, _sum6);
            _mm256_storeu_ps(outptr7, _sum7);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
            outptr4 += 8;
            outptr5 += 8;
            outptr6 += 8;
            outptr7 += 8;
        }
#endif
        for (; i + 3 < size; i += 4)
        {
#if __AVX__
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#else
            const float* tmpptr = tmp.channel(i / 4);
#endif
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum0 = _mm_set1_ps(biasptr[0]);
            __m128 _sum1 = _mm_set1_ps(biasptr[1]);
            __m128 _sum2 = _mm_set1_ps(biasptr[2]);
            __m128 _sum3 = _mm_set1_ps(biasptr[3]);
            __m128 _sum4 = _mm_set1_ps(biasptr[4]);
            __m128 _sum5 = _mm_set1_ps(biasptr[5]);
            __m128 _sum6 = _mm_set1_ps(biasptr[6]);
            __m128 _sum7 = _mm_set1_ps(biasptr[7]);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m128 _val = _mm_loadu_ps(tmpptr);

                __m128 _w0 = _mm_load1_ps(kptr);
                __m128 _w1 = _mm_load1_ps(kptr + 1);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                __m128 _w2 = _mm_load1_ps(kptr + 2);
                __m128 _w3 = _mm_load1_ps(kptr + 3);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                __m128 _w4 = _mm_load1_ps(kptr + 4);
                __m128 _w5 = _mm_load1_ps(kptr + 5);
                _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                __m128 _w6 = _mm_load1_ps(kptr + 6);
                __m128 _w7 = _mm_load1_ps(kptr + 7);
                _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 4;
                kptr += 8;

                _val = _mm_loadu_ps(tmpptr);

                _w0 = _mm_load1_ps(kptr);
                _w1 = _mm_load1_ps(kptr + 1);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _w2 = _mm_load1_ps(kptr + 2);
                _w3 = _mm_load1_ps(kptr + 3);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                _w4 = _mm_load1_ps(kptr + 4);
                _w5 = _mm_load1_ps(kptr + 5);
                _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                _w6 = _mm_load1_ps(kptr + 6);
                _w7 = _mm_load1_ps(kptr + 7);
                _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 4;
                kptr += 8;

                _val = _mm_loadu_ps(tmpptr);

                _w0 = _mm_load1_ps(kptr);
                _w1 = _mm_load1_ps(kptr + 1);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _w2 = _mm_load1_ps(kptr + 2);
                _w3 = _mm_load1_ps(kptr + 3);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                _w4 = _mm_load1_ps(kptr + 4);
                _w5 = _mm_load1_ps(kptr + 5);
                _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                _w6 = _mm_load1_ps(kptr + 6);
                _w7 = _mm_load1_ps(kptr + 7);
                _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 4;
                kptr += 8;

                _val = _mm_loadu_ps(tmpptr);

                _w0 = _mm_load1_ps(kptr);
                _w1 = _mm_load1_ps(kptr + 1);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _w2 = _mm_load1_ps(kptr + 2);
                _w3 = _mm_load1_ps(kptr + 3);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                _w4 = _mm_load1_ps(kptr + 4);
                _w5 = _mm_load1_ps(kptr + 5);
                _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                _w6 = _mm_load1_ps(kptr + 6);
                _w7 = _mm_load1_ps(kptr + 7);
                _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 4;
                kptr += 8;
            }
            for (; j < nn; j++)
            {
                __m128 _val = _mm_loadu_ps(tmpptr);

                __m128 _w0 = _mm_load1_ps(kptr);
                __m128 _w1 = _mm_load1_ps(kptr + 1);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                __m128 _w2 = _mm_load1_ps(kptr + 2);
                __m128 _w3 = _mm_load1_ps(kptr + 3);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                __m128 _w4 = _mm_load1_ps(kptr + 4);
                __m128 _w5 = _mm_load1_ps(kptr + 5);
                _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                __m128 _w6 = _mm_load1_ps(kptr + 6);
                __m128 _w7 = _mm_load1_ps(kptr + 7);
                _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                tmpptr += 4;
                kptr += 8;
            }

            _mm_storeu_ps(outptr0, _sum0);
            _mm_storeu_ps(outptr1, _sum1);
            _mm_storeu_ps(outptr2, _sum2);
            _mm_storeu_ps(outptr3, _sum3);
            _mm_storeu_ps(outptr4, _sum4);
            _mm_storeu_ps(outptr5, _sum5);
            _mm_storeu_ps(outptr6, _sum6);
            _mm_storeu_ps(outptr7, _sum7);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            outptr4 += 4;
            outptr5 += 4;
            outptr6 += 4;
            outptr7 += 4;
        }
        for (; i < size; i++)
        {
#if __AVX__
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#else
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
#endif
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

#if __AVX__
            __m256 _sum = _mm256_loadu_ps(biasptr);
#else
            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _mm_loadu_ps(biasptr + 4);
#endif

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
#if __AVX__
                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                __m256 _w0 = _mm256_loadu_ps(kptr);
                _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);

                __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);

                __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
#else
                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _w00 = _mm_loadu_ps(kptr);
                __m128 _w01 = _mm_loadu_ps(kptr + 4);
                _sum0 = _mm_comp_fmadd_ps(_val0, _w00, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val0, _w01, _sum1);

                __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                __m128 _w10 = _mm_loadu_ps(kptr + 8);
                __m128 _w11 = _mm_loadu_ps(kptr + 12);
                _sum0 = _mm_comp_fmadd_ps(_val1, _w10, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w11, _sum1);

                __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                __m128 _w20 = _mm_loadu_ps(kptr + 16);
                __m128 _w21 = _mm_loadu_ps(kptr + 20);
                _sum0 = _mm_comp_fmadd_ps(_val2, _w20, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val2, _w21, _sum1);

                __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                __m128 _w30 = _mm_loadu_ps(kptr + 24);
                __m128 _w31 = _mm_loadu_ps(kptr + 28);
                _sum0 = _mm_comp_fmadd_ps(_val3, _w30, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val3, _w31, _sum1);
#endif
                tmpptr += 4;
                kptr += 32;
            }
            for (; j < nn; j++)
            {
#if __AVX__
                __m256 _val = _mm256_broadcast_ss(tmpptr);
                __m256 _w = _mm256_loadu_ps(kptr);
                _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
#else
                __m128 _val = _mm_load1_ps(tmpptr);
                __m128 _w0 = _mm_loadu_ps(kptr);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
#endif
                tmpptr += 1;
                kptr += 8;
            }

            float sum[8];
#if __AVX__
            _mm256_storeu_ps(sum, _sum);
#else
            _mm_storeu_ps(sum, _sum0);
            _mm_storeu_ps(sum + 4, _sum1);
#endif

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];
            outptr4[0] = sum[4];
            outptr5[0] = sum[5];
            outptr6[0] = sum[6];
            outptr7[0] = sum[7];

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
        }
    }

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
#if __AVX__
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            __m256 _sum0 = _mm256_broadcast_ss(biasptr);
            __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
            __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
            __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m256 _val = _mm256_loadu_ps(tmpptr);
                __m256 _w0 = _mm256_broadcast_ss(kptr);
                __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 8;
                kptr += 4;

                _val = _mm256_loadu_ps(tmpptr);
                _w0 = _mm256_broadcast_ss(kptr);
                _w1 = _mm256_broadcast_ss(kptr + 1);
                _w2 = _mm256_broadcast_ss(kptr + 2);
                _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 8;
                kptr += 4;

                _val = _mm256_loadu_ps(tmpptr);
                _w0 = _mm256_broadcast_ss(kptr);
                _w1 = _mm256_broadcast_ss(kptr + 1);
                _w2 = _mm256_broadcast_ss(kptr + 2);
                _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 8;
                kptr += 4;

                _val = _mm256_loadu_ps(tmpptr);
                _w0 = _mm256_broadcast_ss(kptr);
                _w1 = _mm256_broadcast_ss(kptr + 1);
                _w2 = _mm256_broadcast_ss(kptr + 2);
                _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 8;
                kptr += 4;
            }
            for (; j < nn; j++)
            {
                __m256 _val = _mm256_loadu_ps(tmpptr);
                __m256 _w0 = _mm256_broadcast_ss(kptr);
                __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 8;
                kptr += 4;
            }

            _mm256_storeu_ps(outptr0, _sum0);
            _mm256_storeu_ps(outptr1, _sum1);
            _mm256_storeu_ps(outptr2, _sum2);
            _mm256_storeu_ps(outptr3, _sum3);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
        }
#endif
        for (; i + 3 < size; i += 4)
        {
#if __AVX__
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#else
            const float* tmpptr = tmp.channel(i / 4);
#endif
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum0 = _mm_set1_ps(biasptr[0]);
            __m128 _sum1 = _mm_set1_ps(biasptr[1]);
            __m128 _sum2 = _mm_set1_ps(biasptr[2]);
            __m128 _sum3 = _mm_set1_ps(biasptr[3]);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m128 _val = _mm_loadu_ps(tmpptr);
                __m128 _w0 = _mm_load1_ps(kptr);
                __m128 _w1 = _mm_load1_ps(kptr + 1);
                __m128 _w2 = _mm_load1_ps(kptr + 2);
                __m128 _w3 = _mm_load1_ps(kptr + 3);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 4;
                kptr += 4;

                _val = _mm_loadu_ps(tmpptr);
                _w0 = _mm_load1_ps(kptr);
                _w1 = _mm_load1_ps(kptr + 1);
                _w2 = _mm_load1_ps(kptr + 2);
                _w3 = _mm_load1_ps(kptr + 3);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 4;
                kptr += 4;

                _val = _mm_loadu_ps(tmpptr);
                _w0 = _mm_load1_ps(kptr);
                _w1 = _mm_load1_ps(kptr + 1);
                _w2 = _mm_load1_ps(kptr + 2);
                _w3 = _mm_load1_ps(kptr + 3);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 4;
                kptr += 4;

                _val = _mm_loadu_ps(tmpptr);
                _w0 = _mm_load1_ps(kptr);
                _w1 = _mm_load1_ps(kptr + 1);
                _w2 = _mm_load1_ps(kptr + 2);
                _w3 = _mm_load1_ps(kptr + 3);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 4;
                kptr += 4;
            }
            for (; j < nn; j++)
            {
                __m128 _val = _mm_loadu_ps(tmpptr);
                __m128 _w0 = _mm_load1_ps(kptr);
                __m128 _w1 = _mm_load1_ps(kptr + 1);
                __m128 _w2 = _mm_load1_ps(kptr + 2);
                __m128 _w3 = _mm_load1_ps(kptr + 3);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                tmpptr += 4;
                kptr += 4;
            }

            _mm_storeu_ps(outptr0, _sum0);
            _mm_storeu_ps(outptr1, _sum1);
            _mm_storeu_ps(outptr2, _sum2);
            _mm_storeu_ps(outptr3, _sum3);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }
        for (; i < size; i++)
        {
#if __AVX__
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#else
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
#endif
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum = _mm_loadu_ps(biasptr);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _w0 = _mm_loadu_ps(kptr);
                _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);

                __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);

                __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);

                tmpptr += 4;
                kptr += 16;
            }
            for (; j < nn; j++)
            {
                __m128 _val = _mm_load1_ps(tmpptr);
                __m128 _w0 = _mm_loadu_ps(kptr);
                _sum = _mm_comp_fmadd_ps(_val, _w0, _sum);

                tmpptr += 1;
                kptr += 4;
            }

            float sum[4];
            _mm_storeu_ps(sum, _sum);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
        }
    }

    remain_outch_start += nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
#if __AVX__
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            __m256 _sum0 = _mm256_set1_ps(bias0);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m256 _val0 = _mm256_loadu_ps(tmpptr);
                __m256 _w0 = _mm256_broadcast_ss(kptr);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                __m256 _val1 = _mm256_loadu_ps(tmpptr + 8);
                __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val1, _w1, _sum0);

                __m256 _val2 = _mm256_loadu_ps(tmpptr + 16);
                __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                _sum0 = _mm256_comp_fmadd_ps(_val2, _w2, _sum0);

                __m256 _val3 = _mm256_loadu_ps(tmpptr + 24);
                __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                _sum0 = _mm256_comp_fmadd_ps(_val3, _w3, _sum0);

                tmpptr += 32;
                kptr += 4;
            }
            for (; j < nn; j++)
            {
                __m256 _val = _mm256_loadu_ps(tmpptr);
                __m256 _w0 = _mm256_broadcast_ss(kptr);
                _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                tmpptr += 8;
                kptr++;
            }

            _mm256_storeu_ps(outptr0, _sum0);

            outptr0 += 8;
        }
#endif
        for (; i + 3 < size; i += 4)
        {
#if __AVX__
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#else
            const float* tmpptr = tmp.channel(i / 4);
#endif
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum0 = _mm_set1_ps(bias0);

            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                __m128 _val0 = _mm_loadu_ps(tmpptr);
                __m128 _w0 = _mm_load1_ps(kptr);
                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                __m128 _val1 = _mm_loadu_ps(tmpptr + 4);
                __m128 _w1 = _mm_load1_ps(kptr + 1);
                _sum0 = _mm_comp_fmadd_ps(_val1, _w1, _sum0);

                __m128 _val2 = _mm_loadu_ps(tmpptr + 8);
                __m128 _w2 = _mm_load1_ps(kptr + 2);
                _sum0 = _mm_comp_fmadd_ps(_val2, _w2, _sum0);

                __m128 _val3 = _mm_loadu_ps(tmpptr + 12);
                __m128 _w3 = _mm_load1_ps(kptr + 3);
                _sum0 = _mm_comp_fmadd_ps(_val3, _w3, _sum0);

                tmpptr += 16;
                kptr += 4;
            }
            for (; j < nn; j++)
            {
                __m128 _val = _mm_loadu_ps(tmpptr);
                __m128 _w0 = _mm_load1_ps(kptr);
                _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                tmpptr += 4;
                kptr++;
            }

            _mm_storeu_ps(outptr0, _sum0);

            outptr0 += 4;
        }
        for (; i < size; i++)
        {
#if __AVX__
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#else
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
#endif
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            for (int j = 0; j < nn; j++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
#else // __SSE2__
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        for (int i = 0; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i);
            const float* kptr = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            for (int j = 0; j < nn; j++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
#endif // __SSE2__
}

static void convolution_im2col_sgemm_transform_kernel_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-maxk-inch-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __SSE2__
    kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + outch % 4);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);
        const Mat k4 = kernel.channel(q + 4);
        const Mat k5 = kernel.channel(q + 5);
        const Mat k6 = kernel.channel(q + 6);
        const Mat k7 = kernel.channel(q + 7);

        float* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);
            const float* k40 = k4.row(p);
            const float* k50 = k5.row(p);
            const float* k60 = k6.row(p);
            const float* k70 = k7.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00 += 8;
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + q % 4);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];

                g00 += 1;
            }
        }
    }
#else
    kernel_tm = kernel;
#endif // __SSE2__
}

static void convolution_im2col_sgemm_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_sse(bottom_im2col, top_blob, kernel, _bias, opt);
}
