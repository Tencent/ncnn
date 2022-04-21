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

static void im2col_sgemm_pack8to1_avx(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u * 8, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + size % 8, 4u * 8, 8, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u * 8, 8, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 8;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 8x8
                    __m256 _r0 = _mm256_load_ps(img0);
                    __m256 _r1 = _mm256_load_ps(img0 + 8);
                    __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(img0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(img0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(img0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(img0 + 8 * 7);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                    _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);

                    img0 += size * 8;
                    tmpptr += 64;
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
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 8;

                for (int k = 0; k < maxk; k++)
                {
                    __m256 _val = _mm256_load_ps(img0);
                    _mm256_store_ps(tmpptr, _val);

                    img0 += size * 8;
                    tmpptr += 8;
                }
            }
        }
    }

    int nn_outch = outch / 8;
    int remain_outch_start = nn_outch * 8;

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
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = kernel.channel(p / 8);

            int nn = inch * maxk * 8; // inch always > 0

            __m256 _sum0 = _mm256_broadcast_ss(biasptr);
            __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
            __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
            __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);
            __m256 _sum4 = _mm256_broadcast_ss(biasptr + 4);
            __m256 _sum5 = _mm256_broadcast_ss(biasptr + 5);
            __m256 _sum6 = _mm256_broadcast_ss(biasptr + 6);
            __m256 _sum7 = _mm256_broadcast_ss(biasptr + 7);

            for (int j = 0; j < nn; j++)
            {
                __m256 _val0 = _mm256_load_ps(tmpptr);

                __m256 _w0 = _mm256_broadcast_ss(kptr0);
                __m256 _w1 = _mm256_broadcast_ss(kptr0 + 1);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val0, _w1, _sum1);
                __m256 _w2 = _mm256_broadcast_ss(kptr0 + 2);
                __m256 _w3 = _mm256_broadcast_ss(kptr0 + 3);
                _sum2 = _mm256_comp_fmadd_ps(_val0, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val0, _w3, _sum3);
                __m256 _w4 = _mm256_broadcast_ss(kptr0 + 4);
                __m256 _w5 = _mm256_broadcast_ss(kptr0 + 5);
                _sum4 = _mm256_comp_fmadd_ps(_val0, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val0, _w5, _sum5);
                __m256 _w6 = _mm256_broadcast_ss(kptr0 + 6);
                __m256 _w7 = _mm256_broadcast_ss(kptr0 + 7);
                _sum6 = _mm256_comp_fmadd_ps(_val0, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val0, _w7, _sum7);

                tmpptr += 8;
                kptr0 += 8;
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
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + i % 8);
            const float* kptr0 = kernel.channel(p / 8);

            int nn = inch * maxk * 8; // inch always > 0

            __m256 _sum = _mm256_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                __m256 _w0 = _mm256_load_ps(kptr0);
                _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                tmpptr += 1;
                kptr0 += 8;
            }

            float sum[8];
            _mm256_storeu_ps(sum, _sum);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];
            outptr4[0] = sum[4];
            outptr5[0] = sum[5];
            outptr6[0] = sum[6];
            outptr7[0] = sum[7];

            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
            outptr4 += 1;
            outptr5 += 1;
            outptr6 += 1;
            outptr7 += 1;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = kernel.channel(p / 8 + p % 8);

            int nn = inch * maxk * 8; // inch always > 0

            __m256 _sum0 = _mm256_set1_ps(bias0);

            for (int j = 0; j < nn; j++)
            {
                __m256 _val0 = _mm256_load_ps(tmpptr);
                __m256 _w0 = _mm256_broadcast_ss(kptr0);
                _sum0 = _mm256_comp_fmadd_ps(_w0, _val0, _sum0);

                tmpptr += 8;
                kptr0 += 1;
            }

            _mm256_storeu_ps(outptr0, _sum0);
            outptr0 += 8;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + i % 8);
            const float* kptr0 = kernel.channel(p / 8 + p % 8);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            __m256 _sum0 = _mm256_setzero_ps();

            for (int j = 0; j < nn; j++)
            {
                __m256 _val0 = _mm256_load_ps(tmpptr);
                __m256 _w0 = _mm256_load_ps(kptr0);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                tmpptr += 8;
                kptr0 += 8;
            }

            sum0 += _mm256_reduce_add_ps(_sum0);

            outptr0[0] = sum0;
            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack8to1_avx(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(8 * 8 * maxk, inch / 8, outch / 8 + outch % 8);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
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
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

        float* g00 = kernel_tm.channel(q / 8 + q % 8);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 8; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack8to1_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u * 8, 8, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row(dilation_h * u) + dilation_w * v * 8;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            __m256 _val = _mm256_load_ps(sptr);
                            _mm256_store_ps(ptr, _val);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack8to1_avx(bottom_im2col, top_blob, kernel, _bias, opt);
}
