// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void im2col_sgemm_pack4_lsx(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
                    __m128i _r0 = __lsx_vld(img0, 0);
                    __m128i _r1 = __lsx_vld(img0 + 4, 0);
                    __m128i _r2 = __lsx_vld(img0 + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(img0 + 4 * 3, 0);
                    __m128i _r4 = __lsx_vld(img0 + 4 * 4, 0);
                    __m128i _r5 = __lsx_vld(img0 + 4 * 5, 0);
                    __m128i _r6 = __lsx_vld(img0 + 4 * 6, 0);
                    __m128i _r7 = __lsx_vld(img0 + 4 * 7, 0);
                    __m128i _r8 = __lsx_vld(img0 + 4 * 8, 0);
                    __m128i _r9 = __lsx_vld(img0 + 4 * 9, 0);
                    __m128i _ra = __lsx_vld(img0 + 4 * 10, 0);
                    __m128i _rb = __lsx_vld(img0 + 4 * 11, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r45r = __lsx_vilvl_w(_r5, _r4);
                    __m128i _r45l = __lsx_vilvh_w(_r5, _r4);
                    __m128i _r67r = __lsx_vilvl_w(_r7, _r6);
                    __m128i _r67l = __lsx_vilvh_w(_r7, _r6);
                    __m128i _r89r = __lsx_vilvl_w(_r9, _r8);
                    __m128i _r89l = __lsx_vilvh_w(_r9, _r8);
                    __m128i _rabr = __lsx_vilvl_w(_rb, _ra);
                    __m128i _rabl = __lsx_vilvh_w(_rb, _ra);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);
                    __m128i _r4567_0 = __lsx_vilvl_d(_r67r, _r45r);
                    __m128i _r4567_1 = __lsx_vilvh_d(_r67r, _r45r);
                    __m128i _r4567_2 = __lsx_vilvl_d(_r67l, _r45l);
                    __m128i _r4567_3 = __lsx_vilvh_d(_r67l, _r45l);
                    __m128i _r89ab_0 = __lsx_vilvl_d(_rabr, _r89r);
                    __m128i _r89ab_1 = __lsx_vilvh_d(_rabr, _r89r);
                    __m128i _r89ab_2 = __lsx_vilvl_d(_rabl, _r89l);
                    __m128i _r89ab_3 = __lsx_vilvh_d(_rabl, _r89l);

                    __lsx_vst(_r0123_0, tmpptr, 0);
                    __lsx_vst(_r4567_0, tmpptr + 4, 0);
                    __lsx_vst(_r89ab_0, tmpptr + 4 * 2, 0);
                    __lsx_vst(_r0123_1, tmpptr + 4 * 3, 0);
                    __lsx_vst(_r4567_1, tmpptr + 4 * 4, 0);
                    __lsx_vst(_r89ab_1, tmpptr + 4 * 5, 0);
                    __lsx_vst(_r0123_2, tmpptr + 4 * 6, 0);
                    __lsx_vst(_r4567_2, tmpptr + 4 * 7, 0);
                    __lsx_vst(_r89ab_2, tmpptr + 4 * 8, 0);
                    __lsx_vst(_r0123_3, tmpptr + 4 * 9, 0);
                    __lsx_vst(_r4567_3, tmpptr + 4 * 10, 0);
                    __lsx_vst(_r89ab_3, tmpptr + 4 * 11, 0);

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
                    __m128i _r0 = __lsx_vld(img0, 0);
                    __m128i _r1 = __lsx_vld(img0 + 4, 0);
                    __m128i _r2 = __lsx_vld(img0 + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(img0 + 4 * 3, 0);
                    __m128i _r4 = __lsx_vld(img0 + 4 * 4, 0);
                    __m128i _r5 = __lsx_vld(img0 + 4 * 5, 0);
                    __m128i _r6 = __lsx_vld(img0 + 4 * 6, 0);
                    __m128i _r7 = __lsx_vld(img0 + 4 * 7, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r45r = __lsx_vilvl_w(_r5, _r4);
                    __m128i _r45l = __lsx_vilvh_w(_r5, _r4);
                    __m128i _r67r = __lsx_vilvl_w(_r7, _r6);
                    __m128i _r67l = __lsx_vilvh_w(_r7, _r6);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);
                    __m128i _r4567_0 = __lsx_vilvl_d(_r67r, _r45r);
                    __m128i _r4567_1 = __lsx_vilvh_d(_r67r, _r45r);
                    __m128i _r4567_2 = __lsx_vilvl_d(_r67l, _r45l);
                    __m128i _r4567_3 = __lsx_vilvh_d(_r67l, _r45l);

                    __lsx_vst(_r0123_0, tmpptr, 0);
                    __lsx_vst(_r4567_0, tmpptr + 4, 0);
                    __lsx_vst(_r0123_1, tmpptr + 4 * 2, 0);
                    __lsx_vst(_r4567_1, tmpptr + 4 * 3, 0);
                    __lsx_vst(_r0123_2, tmpptr + 4 * 4, 0);
                    __lsx_vst(_r4567_2, tmpptr + 4 * 5, 0);
                    __lsx_vst(_r0123_3, tmpptr + 4 * 6, 0);
                    __lsx_vst(_r4567_3, tmpptr + 4 * 7, 0);

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
                    __m128i _r0 = __lsx_vld(img0, 0);
                    __m128i _r1 = __lsx_vld(img0 + 4, 0);
                    __m128i _r2 = __lsx_vld(img0 + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(img0 + 4 * 3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, tmpptr, 0);
                    __lsx_vst(_r0123_1, tmpptr + 4, 0);
                    __lsx_vst(_r0123_2, tmpptr + 4 * 2, 0);
                    __lsx_vst(_r0123_3, tmpptr + 4 * 3, 0);

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
                    __m128i _r0 = __lsx_vld(img0, 0);
                    __m128i _r1 = __lsx_vld(img0 + 4, 0);

                    __m128i _r01_0 = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01_1 = __lsx_vilvh_w(_r1, _r0);

                    __lsx_vst(_r01_0, tmpptr, 0);
                    __lsx_vst(_r01_1, tmpptr + 4, 0);

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
                    __m128i _val = __lsx_vld(img0, 0);
                    __lsx_vst(_val, tmpptr, 0);

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

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
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
                __builtin_prefetch(tmpptr + 48);
                __builtin_prefetch(kptr0 + 16);
                __m128i _val0123 = __lsx_vld(tmpptr, 0);
                __m128i _val4567 = __lsx_vld(tmpptr + 4, 0);
                __m128i _val89ab = __lsx_vld(tmpptr + 8, 0);
                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 0), _sum0);
                _sum1 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 1), _sum1);
                _sum2 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 2), _sum2);
                _sum3 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 3), _sum3);
                _sum4 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 0), _sum4);
                _sum5 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 1), _sum5);
                _sum6 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 2), _sum6);
                _sum7 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 3), _sum7);
                _sum8 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val89ab, 0), _sum8);
                _sum9 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val89ab, 1), _sum9);
                _suma = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val89ab, 2), _suma);
                _sumb = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val89ab, 3), _sumb);

                tmpptr += 12;
                kptr0 += 4;
            }

            __lsx_vst(_sum0, outptr0, 0);
            __lsx_vst(_sum1, outptr0 + 4, 0);
            __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
            __lsx_vst(_sum3, outptr0 + 4 * 3, 0);
            __lsx_vst(_sum4, outptr0 + 4 * 4, 0);
            __lsx_vst(_sum5, outptr0 + 4 * 5, 0);
            __lsx_vst(_sum6, outptr0 + 4 * 6, 0);
            __lsx_vst(_sum7, outptr0 + 4 * 7, 0);
            __lsx_vst(_sum8, outptr0 + 4 * 8, 0);
            __lsx_vst(_sum9, outptr0 + 4 * 9, 0);
            __lsx_vst(_suma, outptr0 + 4 * 10, 0);
            __lsx_vst(_sumb, outptr0 + 4 * 11, 0);

            outptr0 += 4 * 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = _sum0;
            __m128 _sum2 = _sum0;
            __m128 _sum3 = _sum0;
            __m128 _sum4 = _sum0;
            __m128 _sum5 = _sum0;
            __m128 _sum6 = _sum0;
            __m128 _sum7 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 32);
                __builtin_prefetch(kptr0 + 16);
                __m128i _val0123 = __lsx_vld(tmpptr, 0);
                __m128i _val4567 = __lsx_vld(tmpptr + 4, 0);
                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 0), _sum0);
                _sum1 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 1), _sum1);
                _sum2 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 2), _sum2);
                _sum3 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 3), _sum3);
                _sum4 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 0), _sum4);
                _sum5 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 1), _sum5);
                _sum6 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 2), _sum6);
                _sum7 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 3), _sum7);

                tmpptr += 8;
                kptr0 += 4;
            }

            __lsx_vst(_sum0, outptr0, 0);
            __lsx_vst(_sum1, outptr0 + 4, 0);
            __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
            __lsx_vst(_sum3, outptr0 + 4 * 3, 0);
            __lsx_vst(_sum4, outptr0 + 4 * 4, 0);
            __lsx_vst(_sum5, outptr0 + 4 * 5, 0);
            __lsx_vst(_sum6, outptr0 + 4 * 6, 0);
            __lsx_vst(_sum7, outptr0 + 4 * 7, 0);

            outptr0 += 4 * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = _sum0;
            __m128 _sum2 = _sum0;
            __m128 _sum3 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr0 + 16);
                __m128i _val0123 = __lsx_vld(tmpptr, 0);
                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 0), _sum0);
                _sum1 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 1), _sum1);
                _sum2 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 2), _sum2);
                _sum3 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 3), _sum3);

                tmpptr += 4;
                kptr0 += 4;
            }

            __lsx_vst(_sum0, outptr0, 0);
            __lsx_vst(_sum1, outptr0 + 4, 0);
            __lsx_vst(_sum2, outptr0 + 4 * 2, 0);
            __lsx_vst(_sum3, outptr0 + 4 * 3, 0);

            outptr0 += 4 * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum0 = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 8);
                __builtin_prefetch(kptr0 + 16);
                __m128 _val0 = __lsx_vreplfr2vr_s(*tmpptr++);
                __m128 _val1 = __lsx_vreplfr2vr_s(*tmpptr++);
                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                _sum0 = __lsx_vfmadd_s(_w0, _val0, _sum0);
                _sum1 = __lsx_vfmadd_s(_w0, _val1, _sum1);

                kptr0 += 4;
            }

            __lsx_vst(_sum0, outptr0, 0);
            __lsx_vst(_sum1, outptr0 + 4, 0);

            outptr0 += 4 * 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            __m128 _sum = bias ? (__m128)__lsx_vld(bias + p * 4, 0) : (__m128)__lsx_vreplgr2vr_w(0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 4);
                __builtin_prefetch(kptr0 + 16);
                __m128 _val0 = __lsx_vreplfr2vr_s(*tmpptr++);
                __m128 _w0 = (__m128)__lsx_vld(kptr0, 0);
                _sum = __lsx_vfmadd_s(_w0, _val0, _sum);

                kptr0 += 4;
            }

            __lsx_vst(_sum, outptr0, 0);

            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_pack4_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                            __m128 _val = (__m128)__lsx_vld(sptr, 0);
                            __lsx_vst(_val, ptr, 0);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4_lsx(bottom_im2col, top_blob, kernel, _bias, opt);
}
