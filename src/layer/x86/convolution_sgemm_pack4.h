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

static void im2col_sgemm_pack4_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u * 4, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 12)
        tmp.create(12 * maxk, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 4u * 4, 4, opt.workspace_allocator);
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
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x2
                    __m128 _r0 = _mm_load_ps(img0);
                    __m128 _r1 = _mm_load_ps(img0 + 4);

                    __m128 _r01_0 = _mm_unpacklo_ps(_r0, _r1);
                    __m128 _r01_1 = _mm_unpackhi_ps(_r0, _r1);

                    _mm_store_ps(tmpptr, _r01_0);
                    _mm_store_ps(tmpptr + 4, _r01_1);

                    img0 += size * 4;
                    tmpptr += 8;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

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

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _sum0;
            __m128 _sum2 = _sum0;
            __m128 _sum3 = _sum0;
            __m128 _sum4 = _sum0;
            __m128 _sum5 = _sum0;
            __m128 _sum6 = _sum0;
            __m128 _sum7 = _sum0;
            __m128 _sum8 = _sum0;
            __m128 _sum9 = _sum0;
            __m128 _suma = _sum0;
            __m128 _sumb = _sum0;

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
                __m128 _val8 = _mm_load1_ps(tmpptr + 8);
                __m128 _val9 = _mm_load1_ps(tmpptr + 9);
                __m128 _vala = _mm_load1_ps(tmpptr + 10);
                __m128 _valb = _mm_load1_ps(tmpptr + 11);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);
                _sum8 = _mm_comp_fmadd_ps(_val8, _w0, _sum8);
                _sum9 = _mm_comp_fmadd_ps(_val9, _w0, _sum9);
                _suma = _mm_comp_fmadd_ps(_vala, _w0, _suma);
                _sumb = _mm_comp_fmadd_ps(_valb, _w0, _sumb);

                tmpptr += 12;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            _mm_store_ps(outptr0 + 4 * 2, _sum2);
            _mm_store_ps(outptr0 + 4 * 3, _sum3);
            _mm_store_ps(outptr0 + 4 * 4, _sum4);
            _mm_store_ps(outptr0 + 4 * 5, _sum5);
            _mm_store_ps(outptr0 + 4 * 6, _sum6);
            _mm_store_ps(outptr0 + 4 * 7, _sum7);
            _mm_store_ps(outptr0 + 4 * 8, _sum8);
            _mm_store_ps(outptr0 + 4 * 9, _sum9);
            _mm_store_ps(outptr0 + 4 * 10, _suma);
            _mm_store_ps(outptr0 + 4 * 11, _sumb);

            outptr0 += 4 * 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

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
            _mm_store_ps(outptr0 + 4 * 2, _sum2);
            _mm_store_ps(outptr0 + 4 * 3, _sum3);
            _mm_store_ps(outptr0 + 4 * 4, _sum4);
            _mm_store_ps(outptr0 + 4 * 5, _sum5);
            _mm_store_ps(outptr0 + 4 * 6, _sum6);
            _mm_store_ps(outptr0 + 4 * 7, _sum7);

            outptr0 += 4 * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

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
            _mm_store_ps(outptr0 + 4 * 2, _sum2);
            _mm_store_ps(outptr0 + 4 * 3, _sum3);

            outptr0 += 4 * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __m128 _w0 = _mm_load_ps(kptr0);

                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _val1 = _mm_load1_ps(tmpptr + 1);

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);

                tmpptr += 2;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);

            outptr0 += 4 * 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum = _mm_loadu_ps(biasptr);

            for (int j = 0; j < nn; j++)
            {
                __m128 _w0 = _mm_load_ps(kptr0);
                __m128 _val0 = _mm_load1_ps(tmpptr);
                _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                tmpptr += 1;
                kptr0 += 4;
            }

            _mm_store_ps(outptr0, _sum);

            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(16 * maxk, inch / 4, outch / 4);

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        float* g00 = kernel_tm.channel(q / 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);

            const float* k10 = k1.row(p);
            const float* k11 = k1.row(p + 1);
            const float* k12 = k1.row(p + 2);
            const float* k13 = k1.row(p + 3);

            const float* k20 = k2.row(p);
            const float* k21 = k2.row(p + 1);
            const float* k22 = k2.row(p + 2);
            const float* k23 = k2.row(p + 3);

            const float* k30 = k3.row(p);
            const float* k31 = k3.row(p + 1);
            const float* k32 = k3.row(p + 2);
            const float* k33 = k3.row(p + 3);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k01[k];
                g00[5] = k11[k];
                g00[6] = k21[k];
                g00[7] = k31[k];

                g00[8] = k02[k];
                g00[9] = k12[k];
                g00[10] = k22[k];
                g00[11] = k32[k];

                g00[12] = k03[k];
                g00[13] = k13[k];
                g00[14] = k23[k];
                g00[15] = k33[k];

                g00 += 16;
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v * 4;

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

    im2col_sgemm_pack4_sse(bottom_im2col, top_blob, kernel, _bias, opt);
}
