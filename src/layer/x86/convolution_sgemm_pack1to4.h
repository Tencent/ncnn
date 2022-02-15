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

static void im2col_sgemm_pack1to4_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    __m128 _r0 = _mm_loadu_ps(img0);
                    __m128 _r1 = _mm_loadu_ps(img0 + 4);
                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r1);

                    img0 += size;
                    tmpptr += 8;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    __m128 _r0 = _mm_loadu_ps(img0);
                    _mm_store_ps(tmpptr, _r0);

                    img0 += size;
                    tmpptr += 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _sum0;
            __m128 _sum2 = _sum0;
            __m128 _sum3 = _sum0;
            __m128 _sum4 = _sum0;
            __m128 _sum5 = _sum0;
            __m128 _sum6 = _sum0;
            __m128 _sum7 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m128 _w0 = _mm_load_ps(kptr0);

                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                __m128 _val4 = _mm_load1_ps(tmpptr + 4);
                __m128 _val5 = _mm_load1_ps(tmpptr + 5);
                __m128 _val6 = _mm_load1_ps(tmpptr + 6);
                __m128 _val7 = _mm_load1_ps(tmpptr + 7);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                tmpptr += 8;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            _mm_store_ps(outptr0 + 8, _sum2);
            _mm_store_ps(outptr0 + 12, _sum3);
            _mm_store_ps(outptr0 + 16, _sum4);
            _mm_store_ps(outptr0 + 20, _sum5);
            _mm_store_ps(outptr0 + 24, _sum6);
            _mm_store_ps(outptr0 + 28, _sum7);
            outptr0 += 32;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _sum0;
            __m128 _sum2 = _sum0;
            __m128 _sum3 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m128 _w0 = _mm_load_ps(kptr0);

                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                __m128 _val3 = _mm_load1_ps(tmpptr + 3);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                tmpptr += 4;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            _mm_store_ps(outptr0 + 8, _sum2);
            _mm_store_ps(outptr0 + 12, _sum3);
            outptr0 += 16;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            __m128 _sum = _mm_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m128 _w0 = _mm_load_ps(kptr0);
                __m128 _val = _mm_load1_ps(tmpptr);
                _sum = _mm_comp_fmadd_ps(_w0, _val, _sum);

                tmpptr += 1;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum);
            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack1to4_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(4 * maxk, inch, outch / 4);

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        float* g00 = kernel_tm.channel(q / 4);

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
}

static void convolution_im2col_sgemm_pack1to4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                        for (; j + 3 < outw; j += 4)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];
                            ptr[2] = sptr[stride_w * 2];
                            ptr[3] = sptr[stride_w * 3];

                            sptr += stride_w * 4;
                            ptr += 4;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];

                            sptr += stride_w * 2;
                            ptr += 2;
                        }
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

    im2col_sgemm_pack1to4_sse(bottom_im2col, top_blob, kernel, _bias, opt);
}
