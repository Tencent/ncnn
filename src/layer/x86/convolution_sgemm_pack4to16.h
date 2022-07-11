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

static void im2col_sgemm_pack4to16_avx512(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 16u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 16)
        tmp.create(16 * maxk, inch, size / 16 + size % 16, 16u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 16u, 4, opt.workspace_allocator);
    {
        int nn_size = size >> 4;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 16;

            float* tmpptr = tmp.channel(i / 16);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x16
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
                    __m128 _rc = _mm_load_ps(img0 + 4 * 12);
                    __m128 _rd = _mm_load_ps(img0 + 4 * 13);
                    __m128 _re = _mm_load_ps(img0 + 4 * 14);
                    __m128 _rf = _mm_load_ps(img0 + 4 * 15);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                    _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r4);
                    _mm_store_ps(tmpptr + 4 * 2, _r8);
                    _mm_store_ps(tmpptr + 4 * 3, _rc);
                    _mm_store_ps(tmpptr + 4 * 4, _r1);
                    _mm_store_ps(tmpptr + 4 * 5, _r5);
                    _mm_store_ps(tmpptr + 4 * 6, _r9);
                    _mm_store_ps(tmpptr + 4 * 7, _rd);
                    _mm_store_ps(tmpptr + 4 * 8, _r2);
                    _mm_store_ps(tmpptr + 4 * 9, _r6);
                    _mm_store_ps(tmpptr + 4 * 10, _ra);
                    _mm_store_ps(tmpptr + 4 * 11, _re);
                    _mm_store_ps(tmpptr + 4 * 12, _r3);
                    _mm_store_ps(tmpptr + 4 * 13, _r7);
                    _mm_store_ps(tmpptr + 4 * 14, _rb);
                    _mm_store_ps(tmpptr + 4 * 15, _rf);

                    img0 += size * 4;
                    tmpptr += 64;
                }
            }
        }

        remain_size_start += nn_size << 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 16 + i % 16);

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

        const float zeros[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 16 : zeros;

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float* tmpptr = tmp.channel(i / 16);
            const float* kptr = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m512 _sum0 = _mm512_loadu_ps(biasptr);
            __m512 _sum1 = _sum0;
            __m512 _sum2 = _sum0;
            __m512 _sum3 = _sum0;
            __m512 _sum4 = _sum0;
            __m512 _sum5 = _sum0;
            __m512 _sum6 = _sum0;
            __m512 _sum7 = _sum0;
            __m512 _sum8 = _sum0;
            __m512 _sum9 = _sum0;
            __m512 _suma = _sum0;
            __m512 _sumb = _sum0;
            __m512 _sumc = _sum0;
            __m512 _sumd = _sum0;
            __m512 _sume = _sum0;
            __m512 _sumf = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m512 _w0 = _mm512_load_ps(kptr);

                __m512 _val0 = _mm512_set1_ps(tmpptr[0]);
                __m512 _val1 = _mm512_set1_ps(tmpptr[1]);
                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w0, _sum1);
                __m512 _val2 = _mm512_set1_ps(tmpptr[2]);
                __m512 _val3 = _mm512_set1_ps(tmpptr[3]);
                _sum2 = _mm512_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w0, _sum3);
                __m512 _val4 = _mm512_set1_ps(tmpptr[4]);
                __m512 _val5 = _mm512_set1_ps(tmpptr[5]);
                _sum4 = _mm512_fmadd_ps(_val4, _w0, _sum4);
                _sum5 = _mm512_fmadd_ps(_val5, _w0, _sum5);
                __m512 _val6 = _mm512_set1_ps(tmpptr[6]);
                __m512 _val7 = _mm512_set1_ps(tmpptr[7]);
                _sum6 = _mm512_fmadd_ps(_val6, _w0, _sum6);
                _sum7 = _mm512_fmadd_ps(_val7, _w0, _sum7);
                __m512 _val8 = _mm512_set1_ps(tmpptr[8]);
                __m512 _val9 = _mm512_set1_ps(tmpptr[9]);
                _sum8 = _mm512_fmadd_ps(_val8, _w0, _sum8);
                _sum9 = _mm512_fmadd_ps(_val9, _w0, _sum9);
                __m512 _vala = _mm512_set1_ps(tmpptr[10]);
                __m512 _valb = _mm512_set1_ps(tmpptr[11]);
                _suma = _mm512_fmadd_ps(_vala, _w0, _suma);
                _sumb = _mm512_fmadd_ps(_valb, _w0, _sumb);
                __m512 _valc = _mm512_set1_ps(tmpptr[12]);
                __m512 _vald = _mm512_set1_ps(tmpptr[13]);
                _sumc = _mm512_fmadd_ps(_valc, _w0, _sumc);
                _sumd = _mm512_fmadd_ps(_vald, _w0, _sumd);
                __m512 _vale = _mm512_set1_ps(tmpptr[14]);
                __m512 _valf = _mm512_set1_ps(tmpptr[15]);
                _sume = _mm512_fmadd_ps(_vale, _w0, _sume);
                _sumf = _mm512_fmadd_ps(_valf, _w0, _sumf);

                kptr += 16;
                tmpptr += 16;
            }

            _mm512_store_ps(outptr0, _sum0);
            _mm512_store_ps(outptr0 + 16, _sum1);
            _mm512_store_ps(outptr0 + 16 * 2, _sum2);
            _mm512_store_ps(outptr0 + 16 * 3, _sum3);
            _mm512_store_ps(outptr0 + 16 * 4, _sum4);
            _mm512_store_ps(outptr0 + 16 * 5, _sum5);
            _mm512_store_ps(outptr0 + 16 * 6, _sum6);
            _mm512_store_ps(outptr0 + 16 * 7, _sum7);
            _mm512_store_ps(outptr0 + 16 * 8, _sum8);
            _mm512_store_ps(outptr0 + 16 * 9, _sum9);
            _mm512_store_ps(outptr0 + 16 * 10, _suma);
            _mm512_store_ps(outptr0 + 16 * 11, _sumb);
            _mm512_store_ps(outptr0 + 16 * 12, _sumc);
            _mm512_store_ps(outptr0 + 16 * 13, _sumd);
            _mm512_store_ps(outptr0 + 16 * 14, _sume);
            _mm512_store_ps(outptr0 + 16 * 15, _sumf);

            outptr0 += 16 * 16;
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 16 + i % 16);
            const float* kptr = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m512 _sum0 = _mm512_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m512 _w0 = _mm512_load_ps(kptr);
                __m512 _val0 = _mm512_set1_ps(tmpptr[0]);
                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);

                kptr += 16;
                tmpptr += 1;
            }

            _mm512_store_ps(outptr0, _sum0);
            outptr0 += 16;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4to16_avx512(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 16b-4a-maxk-inch/4a-outch/16b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(16 * 4 * maxk, inch / 4, outch / 16, (size_t)4u);

    for (int q = 0; q + 15 < outch; q += 16)
    {
        float* g00 = kernel_tm.channel(q / 16);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
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
}

static void convolution_im2col_sgemm_pack4to16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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

    im2col_sgemm_pack4to16_avx512(bottom_im2col, top_blob, kernel, _bias, opt);
}
