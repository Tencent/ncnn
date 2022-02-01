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

static void im2col_sgemm_pack4to1_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u * 4, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    Mat tmp;
    if (size >= 12)
        tmp.create(12 * maxk, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + size % 12 % 4, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u * 4, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u * 4, 4, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size / 12;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 12;

            float* tmpptr = tmp.channel(i / 12);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x12
                    __m128 _r0 = _mm_load_ps(img0);
                    __m128 _r1 = _mm_load_ps(img0 + 4);
                    __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(img0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(img0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(img0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(img0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(img0 + 4 * 7);
                    __m128 _r8 = _mm_load_ps(img0 + 4 * 8);
                    __m128 _r9 = _mm_load_ps(img0 + 4 * 9);
                    __m128 _ra = _mm_load_ps(img0 + 4 * 10);
                    __m128 _rb = _mm_load_ps(img0 + 4 * 11);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r4);
                    _mm_store_ps(tmpptr + 4 * 2, _r8);
                    _mm_store_ps(tmpptr + 4 * 3, _r1);
                    _mm_store_ps(tmpptr + 4 * 4, _r5);
                    _mm_store_ps(tmpptr + 4 * 5, _r9);
                    _mm_store_ps(tmpptr + 4 * 6, _r2);
                    _mm_store_ps(tmpptr + 4 * 7, _r6);
                    _mm_store_ps(tmpptr + 4 * 8, _ra);
                    _mm_store_ps(tmpptr + 4 * 9, _r3);
                    _mm_store_ps(tmpptr + 4 * 10, _r7);
                    _mm_store_ps(tmpptr + 4 * 11, _rb);

                    img0 += size * 4;
                    tmpptr += 48;
                }
            }
        }

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x8
                    __m128 _r0 = _mm_load_ps(img0);
                    __m128 _r1 = _mm_load_ps(img0 + 4);
                    __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(img0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(img0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(img0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(img0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(img0 + 4 * 7);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r4);
                    _mm_store_ps(tmpptr + 4 * 2, _r1);
                    _mm_store_ps(tmpptr + 4 * 3, _r5);
                    _mm_store_ps(tmpptr + 4 * 4, _r2);
                    _mm_store_ps(tmpptr + 4 * 5, _r6);
                    _mm_store_ps(tmpptr + 4 * 6, _r3);
                    _mm_store_ps(tmpptr + 4 * 7, _r7);

                    img0 += size * 4;
                    tmpptr += 32;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_load_ps(img0);
                    __m128 _r1 = _mm_load_ps(img0 + 4);
                    __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(img0 + 4 * 3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r1);
                    _mm_store_ps(tmpptr + 4 * 2, _r2);
                    _mm_store_ps(tmpptr + 4 * 3, _r3);

                    img0 += size * 4;
                    tmpptr += 16;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    __m128 _val = _mm_load_ps(img0);
                    _mm_store_ps(tmpptr, _val);

                    img0 += size * 4;
                    tmpptr += 4;
                }
            }
        }
    }

    int nn_outch = outch / 4;
    int remain_outch_start = nn_outch * 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);

        const float zeros[4] = {0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_load1_ps(biasptr);
            __m128 _sum1 = _mm_load1_ps(biasptr);
            __m128 _sum2 = _mm_load1_ps(biasptr);
            __m128 _sum3 = _mm_load1_ps(biasptr + 1);
            __m128 _sum4 = _mm_load1_ps(biasptr + 1);
            __m128 _sum5 = _mm_load1_ps(biasptr + 1);
            __m128 _sum6 = _mm_load1_ps(biasptr + 2);
            __m128 _sum7 = _mm_load1_ps(biasptr + 2);
            __m128 _sum8 = _mm_load1_ps(biasptr + 2);
            __m128 _sum9 = _mm_load1_ps(biasptr + 3);
            __m128 _suma = _mm_load1_ps(biasptr + 3);
            __m128 _sumb = _mm_load1_ps(biasptr + 3);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);
                __m128 _val1 = _mm_load_ps(tmpptr + 4);
                __m128 _val2 = _mm_load_ps(tmpptr + 8);

                __m128 _w0 = _mm_load1_ps(kptr0);
                __m128 _w1 = _mm_load1_ps(kptr0 + 1);
                __m128 _w2 = _mm_load1_ps(kptr0 + 2);
                __m128 _w3 = _mm_load1_ps(kptr0 + 3);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val0, _w1, _sum3);
                _sum4 = _mm_comp_fmadd_ps(_val1, _w1, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val2, _w1, _sum5);
                _sum6 = _mm_comp_fmadd_ps(_val0, _w2, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val1, _w2, _sum7);
                _sum8 = _mm_comp_fmadd_ps(_val2, _w2, _sum8);
                _sum9 = _mm_comp_fmadd_ps(_val0, _w3, _sum9);
                _suma = _mm_comp_fmadd_ps(_val1, _w3, _suma);
                _sumb = _mm_comp_fmadd_ps(_val2, _w3, _sumb);

                tmpptr += 12;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            _mm_store_ps(outptr0 + 8, _sum2);
            _mm_store_ps(outptr1, _sum3);
            _mm_store_ps(outptr1 + 4, _sum4);
            _mm_store_ps(outptr1 + 8, _sum5);
            _mm_store_ps(outptr2, _sum6);
            _mm_store_ps(outptr2 + 4, _sum7);
            _mm_store_ps(outptr2 + 8, _sum8);
            _mm_store_ps(outptr3, _sum9);
            _mm_store_ps(outptr3 + 4, _suma);
            _mm_store_ps(outptr3 + 8, _sumb);

            outptr0 += 12;
            outptr1 += 12;
            outptr2 += 12;
            outptr3 += 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_load1_ps(biasptr);
            __m128 _sum1 = _mm_load1_ps(biasptr);
            __m128 _sum2 = _mm_load1_ps(biasptr + 1);
            __m128 _sum3 = _mm_load1_ps(biasptr + 1);
            __m128 _sum4 = _mm_load1_ps(biasptr + 2);
            __m128 _sum5 = _mm_load1_ps(biasptr + 2);
            __m128 _sum6 = _mm_load1_ps(biasptr + 3);
            __m128 _sum7 = _mm_load1_ps(biasptr + 3);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);
                __m128 _val1 = _mm_load_ps(tmpptr + 4);

                __m128 _w0 = _mm_load1_ps(kptr0);
                __m128 _w1 = _mm_load1_ps(kptr0 + 1);
                __m128 _w2 = _mm_load1_ps(kptr0 + 2);
                __m128 _w3 = _mm_load1_ps(kptr0 + 3);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val0, _w1, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val1, _w1, _sum3);
                _sum4 = _mm_comp_fmadd_ps(_val0, _w2, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val1, _w2, _sum5);
                _sum6 = _mm_comp_fmadd_ps(_val0, _w3, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val1, _w3, _sum7);

                tmpptr += 8;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            _mm_store_ps(outptr1, _sum2);
            _mm_store_ps(outptr1 + 4, _sum3);
            _mm_store_ps(outptr2, _sum4);
            _mm_store_ps(outptr2 + 4, _sum5);
            _mm_store_ps(outptr3, _sum6);
            _mm_store_ps(outptr3 + 4, _sum7);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_load1_ps(biasptr);
            __m128 _sum1 = _mm_load1_ps(biasptr + 1);
            __m128 _sum2 = _mm_load1_ps(biasptr + 2);
            __m128 _sum3 = _mm_load1_ps(biasptr + 3);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);

                __m128 _w0 = _mm_load1_ps(kptr0);
                __m128 _w1 = _mm_load1_ps(kptr0 + 1);
                __m128 _w2 = _mm_load1_ps(kptr0 + 2);
                __m128 _w3 = _mm_load1_ps(kptr0 + 3);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val0, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val0, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val0, _w3, _sum3);

                tmpptr += 4;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr1, _sum1);
            _mm_store_ps(outptr2, _sum2);
            _mm_store_ps(outptr3, _sum3);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum = _mm_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _w0 = _mm_load_ps(kptr0);
                _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                tmpptr += 1;
                kptr0 += 4;
            }

            float sum[4];
            _mm_storeu_ps(sum, _sum);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];

            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_set1_ps(bias0);
            __m128 _sum1 = _mm_set1_ps(bias0);
            __m128 _sum2 = _mm_set1_ps(bias0);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);
                __m128 _val1 = _mm_load_ps(tmpptr + 4);
                __m128 _val2 = _mm_load_ps(tmpptr + 8);
                __m128 _w0 = _mm_load1_ps(kptr0);
                _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_w0, _val1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_w0, _val2, _sum2);

                tmpptr += 12;
                kptr0 += 1;
            }

            _mm_storeu_ps(outptr0, _sum0);
            _mm_storeu_ps(outptr0 + 4, _sum1);
            _mm_storeu_ps(outptr0 + 8, _sum2);

            outptr0 += 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_set1_ps(bias0);
            __m128 _sum1 = _mm_set1_ps(bias0);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);
                __m128 _val1 = _mm_load_ps(tmpptr + 4);
                __m128 _w0 = _mm_load1_ps(kptr0);
                _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_w0, _val1, _sum1);

                tmpptr += 8;
                kptr0 += 1;
            }

            _mm_storeu_ps(outptr0, _sum0);
            _mm_storeu_ps(outptr0 + 4, _sum1);

            outptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_set1_ps(bias0);

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);
                __m128 _w0 = _mm_load1_ps(kptr0);
                _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);

                tmpptr += 4;
                kptr0 += 1;
            }

            _mm_storeu_ps(outptr0, _sum0);

            outptr0 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            __m128 _sum0 = _mm_setzero_ps();

            for (int j = 0; j < nn; j++)
            {
                __m128 _val0 = _mm_load_ps(tmpptr);
                __m128 _w0 = _mm_load_ps(kptr0);
                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                tmpptr += 4;
                kptr0 += 4;
            }

            sum0 += _mm_reduce_add_ps(_sum0);

            outptr0[0] = sum0;

            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4to1_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(4 * 4 * maxk, inch / 4, outch / 4 + outch % 4);

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        float* g00 = kernel_tm.channel(q / 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

        float* g00 = kernel_tm.channel(q / 4 + q % 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4to1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u * 4, 4, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row(dilation_h * u) + dilation_w * v * 4;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            __m128 _val = _mm_load_ps(sptr);
                            _mm_store_ps(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4to1_sse(bottom_im2col, top_blob, kernel, _bias, opt);
}
