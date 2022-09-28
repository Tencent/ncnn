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

static void im2col_sgemm_pack4to8_avx(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 16u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + size % 8, 16u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 16u, 4, opt.workspace_allocator);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + i % 8);

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 8 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m256 _sum0 = _mm256_loadu_ps(biasptr);
            __m256 _sum1 = _sum0;
            __m256 _sum2 = _sum0;
            __m256 _sum3 = _sum0;
            __m256 _sum4 = _sum0;
            __m256 _sum5 = _sum0;
            __m256 _sum6 = _sum0;
            __m256 _sum7 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m256 _w0 = _mm256_load_ps(kptr);

                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                kptr += 8;
                tmpptr += 8;
            }

            _mm256_store_ps(outptr0, _sum0);
            _mm256_store_ps(outptr0 + 8, _sum1);
            _mm256_store_ps(outptr0 + 8 * 2, _sum2);
            _mm256_store_ps(outptr0 + 8 * 3, _sum3);
            _mm256_store_ps(outptr0 + 8 * 4, _sum4);
            _mm256_store_ps(outptr0 + 8 * 5, _sum5);
            _mm256_store_ps(outptr0 + 8 * 6, _sum6);
            _mm256_store_ps(outptr0 + 8 * 7, _sum7);

            outptr0 += 8 * 8;
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + i % 8);
            const float* kptr = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m256 _sum0 = _mm256_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m256 _w0 = _mm256_load_ps(kptr);
                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                kptr += 8;
                tmpptr += 1;
            }

            _mm256_store_ps(outptr0, _sum0);
            outptr0 += 8;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4to8_avx(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-4a-maxk-inch/4a-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(32 * maxk, inch / 4, outch / 8, (size_t)4u);

    for (int q = 0; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4to8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 16u, 4, opt.workspace_allocator);
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

    im2col_sgemm_pack4to8_avx(bottom_im2col, top_blob, kernel, _bias, opt);
}
