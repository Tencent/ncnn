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

static void im2col_sgemm_pack16to8_avx512(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 64u, 16, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 64u, 16, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 64u, 16, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 64u, 16, opt.workspace_allocator);
    {
        int nn_size = size / 8;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 16;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 16x8
                    __m512 _r0 = _mm512_loadu_ps(img0);
                    __m512 _r1 = _mm512_loadu_ps(img0 + 16);
                    __m512 _r2 = _mm512_loadu_ps(img0 + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(img0 + 16 * 3);
                    __m512 _r4 = _mm512_loadu_ps(img0 + 16 * 4);
                    __m512 _r5 = _mm512_loadu_ps(img0 + 16 * 5);
                    __m512 _r6 = _mm512_loadu_ps(img0 + 16 * 6);
                    __m512 _r7 = _mm512_loadu_ps(img0 + 16 * 7);

                    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
                    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_r2, _r3);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_r2, _r3);
                    __m512 _tmp4 = _mm512_unpacklo_ps(_r4, _r5);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_r4, _r5);
                    __m512 _tmp6 = _mm512_unpacklo_ps(_r6, _r7);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_r6, _r7);

                    __m512 _tmp8 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp9 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpa = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpb = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpc = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpd = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpe = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpf = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

                    _tmp0 = _mm512_shuffle_f32x4(_tmp8, _tmpc, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_tmp9, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_tmpa, _tmpe, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_tmpb, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_f32x4(_tmp8, _tmpc, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_f32x4(_tmp9, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_tmpa, _tmpe, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_f32x4(_tmpb, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _r2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _r4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _r5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _r6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _r7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(tmpptr, _r0);
                    _mm512_storeu_ps(tmpptr + 16, _r1);
                    _mm512_storeu_ps(tmpptr + 16 * 2, _r2);
                    _mm512_storeu_ps(tmpptr + 16 * 3, _r3);
                    _mm512_storeu_ps(tmpptr + 16 * 4, _r4);
                    _mm512_storeu_ps(tmpptr + 16 * 5, _r5);
                    _mm512_storeu_ps(tmpptr + 16 * 6, _r6);
                    _mm512_storeu_ps(tmpptr + 16 * 7, _r7);

                    img0 += size * 16;
                    tmpptr += 128;
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
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 16;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 16x4
                    __m512 _r0 = _mm512_loadu_ps(img0);
                    __m512 _r1 = _mm512_loadu_ps(img0 + 16);
                    __m512 _r2 = _mm512_loadu_ps(img0 + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(img0 + 16 * 3);

                    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
                    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_r2, _r3);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_r2, _r3);

                    __m512 _tmp4 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp7 = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));

                    _tmp0 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _r2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _r3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(tmpptr, _r0);
                    _mm512_storeu_ps(tmpptr + 16, _r1);
                    _mm512_storeu_ps(tmpptr + 16 * 2, _r2);
                    _mm512_storeu_ps(tmpptr + 16 * 3, _r3);

                    img0 += size * 16;
                    tmpptr += 64;
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
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 16;

                for (int k = 0; k < maxk; k++)
                {
                    __m512 _val = _mm512_load_ps(img0);
                    _mm512_store_ps(tmpptr, _val);

                    img0 += size * 16;
                    tmpptr += 16;
                }
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);

        const float zeros[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 8 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p / 2);

            int nn = inch * maxk * 16; // inch always > 0

            __m512 _sum0 = _mm512_loadu_ps(biasptr);
            __m512 _sum1 = _sum0;
            __m512 _sum2 = _sum0;
            __m512 _sum3 = _sum0;
            __m512 _sum4 = _sum0;
            __m512 _sum5 = _sum0;
            __m512 _sum6 = _sum0;
            __m512 _sum7 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m512 _w0 = _mm512_load_ps(kptr);

                __m512 _val0 = _mm512_set1_ps(tmpptr[0]);
                __m512 _val1 = _mm512_set1_ps(tmpptr[1]);
                __m512 _val2 = _mm512_set1_ps(tmpptr[2]);
                __m512 _val3 = _mm512_set1_ps(tmpptr[3]);
                __m512 _val4 = _mm512_set1_ps(tmpptr[4]);
                __m512 _val5 = _mm512_set1_ps(tmpptr[5]);
                __m512 _val6 = _mm512_set1_ps(tmpptr[6]);
                __m512 _val7 = _mm512_set1_ps(tmpptr[7]);
                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w0, _sum3);
                _sum4 = _mm512_fmadd_ps(_val4, _w0, _sum4);
                _sum5 = _mm512_fmadd_ps(_val5, _w0, _sum5);
                _sum6 = _mm512_fmadd_ps(_val6, _w0, _sum6);
                _sum7 = _mm512_fmadd_ps(_val7, _w0, _sum7);

                tmpptr += 8;
                kptr += 16;
            }

            _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
            _mm256_store_ps(outptr0 + 8, _mm512_extractf32x8_ps(_sum1, 0));
            _mm256_store_ps(outptr0 + 8 * 2, _mm512_extractf32x8_ps(_sum2, 0));
            _mm256_store_ps(outptr0 + 8 * 3, _mm512_extractf32x8_ps(_sum3, 0));
            _mm256_store_ps(outptr0 + 8 * 4, _mm512_extractf32x8_ps(_sum4, 0));
            _mm256_store_ps(outptr0 + 8 * 5, _mm512_extractf32x8_ps(_sum5, 0));
            _mm256_store_ps(outptr0 + 8 * 6, _mm512_extractf32x8_ps(_sum6, 0));
            _mm256_store_ps(outptr0 + 8 * 7, _mm512_extractf32x8_ps(_sum7, 0));
            _mm256_store_ps(outptr1, _mm512_extractf32x8_ps(_sum0, 1));
            _mm256_store_ps(outptr1 + 8, _mm512_extractf32x8_ps(_sum1, 1));
            _mm256_store_ps(outptr1 + 8 * 2, _mm512_extractf32x8_ps(_sum2, 1));
            _mm256_store_ps(outptr1 + 8 * 3, _mm512_extractf32x8_ps(_sum3, 1));
            _mm256_store_ps(outptr1 + 8 * 4, _mm512_extractf32x8_ps(_sum4, 1));
            _mm256_store_ps(outptr1 + 8 * 5, _mm512_extractf32x8_ps(_sum5, 1));
            _mm256_store_ps(outptr1 + 8 * 6, _mm512_extractf32x8_ps(_sum6, 1));
            _mm256_store_ps(outptr1 + 8 * 7, _mm512_extractf32x8_ps(_sum7, 1));

            outptr0 += 64;
            outptr1 += 64;
        }
        for (; i + 3 < size; i += 4)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr = kernel.channel(p / 2);

            int nn = inch * maxk * 16; // inch always > 0

            __m512 _sum0 = _mm512_loadu_ps(biasptr);
            __m512 _sum1 = _sum0;
            __m512 _sum2 = _sum0;
            __m512 _sum3 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m512 _w0 = _mm512_load_ps(kptr);

                __m512 _val0 = _mm512_set1_ps(tmpptr[0]);
                __m512 _val1 = _mm512_set1_ps(tmpptr[1]);
                __m512 _val2 = _mm512_set1_ps(tmpptr[2]);
                __m512 _val3 = _mm512_set1_ps(tmpptr[3]);
                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w0, _sum3);

                tmpptr += 4;
                kptr += 16;
            }

            _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
            _mm256_store_ps(outptr0 + 8, _mm512_extractf32x8_ps(_sum1, 0));
            _mm256_store_ps(outptr0 + 16, _mm512_extractf32x8_ps(_sum2, 0));
            _mm256_store_ps(outptr0 + 24, _mm512_extractf32x8_ps(_sum3, 0));
            _mm256_store_ps(outptr1, _mm512_extractf32x8_ps(_sum0, 1));
            _mm256_store_ps(outptr1 + 8, _mm512_extractf32x8_ps(_sum1, 1));
            _mm256_store_ps(outptr1 + 16, _mm512_extractf32x8_ps(_sum2, 1));
            _mm256_store_ps(outptr1 + 24, _mm512_extractf32x8_ps(_sum3, 1));

            outptr0 += 32;
            outptr1 += 32;
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const float* kptr = kernel.channel(p / 2);

            int nn = inch * maxk * 16; // inch always > 0

            __m512 _sum = _mm512_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m512 _w0 = _mm512_load_ps(kptr);
                __m512 _val0 = _mm512_set1_ps(tmpptr[0]);
                _sum = _mm512_fmadd_ps(_val0, _w0, _sum);

                tmpptr += 1;
                kptr += 16;
            }

            _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_sum, 0));
            _mm256_store_ps(outptr1, _mm512_extractf32x8_ps(_sum, 1));

            outptr0 += 8;
            outptr1 += 8;
        }
    }

    remain_outch_start += nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 8 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk * 16; // inch always > 0

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
                __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                _sum0 = _mm256_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm256_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm256_fmadd_ps(_val3, _w0, _sum3);
                _sum4 = _mm256_fmadd_ps(_val4, _w0, _sum4);
                _sum5 = _mm256_fmadd_ps(_val5, _w0, _sum5);
                _sum6 = _mm256_fmadd_ps(_val6, _w0, _sum6);
                _sum7 = _mm256_fmadd_ps(_val7, _w0, _sum7);

                tmpptr += 8;
                kptr += 8;
            }

            _mm256_store_ps(outptr0, _sum0);
            _mm256_store_ps(outptr0 + 8, _sum1);
            _mm256_store_ps(outptr0 + 8 * 2, _sum2);
            _mm256_store_ps(outptr0 + 8 * 3, _sum3);
            _mm256_store_ps(outptr0 + 8 * 4, _sum4);
            _mm256_store_ps(outptr0 + 8 * 5, _sum5);
            _mm256_store_ps(outptr0 + 8 * 6, _sum6);
            _mm256_store_ps(outptr0 + 8 * 7, _sum7);

            outptr0 += 64;
        }
        for (; i + 3 < size; i += 4)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk * 16; // inch always > 0

            __m256 _sum0 = _mm256_loadu_ps(biasptr);
            __m256 _sum1 = _sum0;
            __m256 _sum2 = _sum0;
            __m256 _sum3 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m256 _w0 = _mm256_load_ps(kptr);

                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                _sum0 = _mm256_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm256_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm256_fmadd_ps(_val3, _w0, _sum3);

                tmpptr += 4;
                kptr += 8;
            }

            _mm256_store_ps(outptr0, _sum0);
            _mm256_store_ps(outptr0 + 8, _sum1);
            _mm256_store_ps(outptr0 + 16, _sum2);
            _mm256_store_ps(outptr0 + 24, _sum3);

            outptr0 += 32;
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const float* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk * 16; // inch always > 0

            __m256 _sum = _mm256_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m256 _w0 = _mm256_load_ps(kptr);
                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                _sum = _mm256_fmadd_ps(_val0, _w0, _sum);

                tmpptr += 1;
                kptr += 8;
            }

            _mm256_store_ps(outptr0, _sum);

            outptr0 += 8;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack16to8_avx512(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 16b-16a-maxk-inch/16a-outch/16b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(16 * 16 * maxk, inch / 16, outch / 16 + (outch % 16) / 8, (size_t)4u);

    int q = 0;
    for (; q + 15 < outch; q += 16)
    {
        float* g00 = kernel_tm.channel(q / 16);

        for (int p = 0; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
    for (; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8);

        for (int p = 0; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 16; i++)
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

static void convolution_im2col_sgemm_pack16to8_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 64u, 16, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * 16;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row(dilation_h * u) + dilation_w * v * 16;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            __m512 _v = _mm512_load_ps(sptr);
                            _mm512_store_ps(ptr, _v);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack16to8_avx512(bottom_im2col, top_blob, kernel, _bias, opt);
}
