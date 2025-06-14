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

static void convolution_winograd_dot_pack4_lsx(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 16u, 4, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 12)
        bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 4, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 11 < tiles; i += 12)
        {
            float* tmpptr = tm2.row(i / 12);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x8
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r0 + 4, 0);
                __m128i _r2 = __lsx_vld(r0 + 4 * 2, 0);
                __m128i _r3 = __lsx_vld(r0 + 4 * 3, 0);
                __m128i _r4 = __lsx_vld(r0 + 4 * 4, 0);
                __m128i _r5 = __lsx_vld(r0 + 4 * 5, 0);
                __m128i _r6 = __lsx_vld(r0 + 4 * 6, 0);
                __m128i _r7 = __lsx_vld(r0 + 4 * 7, 0);
                __m128i _r8 = __lsx_vld(r0 + 4 * 8, 0);
                __m128i _r9 = __lsx_vld(r0 + 4 * 9, 0);
                __m128i _ra = __lsx_vld(r0 + 4 * 10, 0);
                __m128i _rb = __lsx_vld(r0 + 4 * 11, 0);

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

                r0 += bottom_blob_tm.cstep * 4;
                tmpptr += 48;
            }
        }
        for (; i + 7 < tiles; i += 8)
        {
            float* tmpptr = tm2.row(i / 12 + (i % 12) / 8);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x8
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r0 + 4, 0);
                __m128i _r2 = __lsx_vld(r0 + 4 * 2, 0);
                __m128i _r3 = __lsx_vld(r0 + 4 * 3, 0);
                __m128i _r4 = __lsx_vld(r0 + 4 * 4, 0);
                __m128i _r5 = __lsx_vld(r0 + 4 * 5, 0);
                __m128i _r6 = __lsx_vld(r0 + 4 * 6, 0);
                __m128i _r7 = __lsx_vld(r0 + 4 * 7, 0);

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

                r0 += bottom_blob_tm.cstep * 4;
                tmpptr += 32;
            }
        }
        for (; i + 3 < tiles; i += 4)
        {
            float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x4
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r0 + 4, 0);
                __m128i _r2 = __lsx_vld(r0 + 4 * 2, 0);
                __m128i _r3 = __lsx_vld(r0 + 4 * 3, 0);

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

                r0 += bottom_blob_tm.cstep * 4;
                tmpptr += 16;
            }
        }
        for (; i + 1 < tiles; i += 2)
        {
            float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x2
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r0 + 4, 0);

                __m128i _r01_0 = __lsx_vilvl_w(_r1, _r0);
                __m128i _r01_1 = __lsx_vilvh_w(_r1, _r0);

                __lsx_vst(_r01_0, tmpptr, 0);
                __lsx_vst(_r01_1, tmpptr + 4, 0);

                r0 += bottom_blob_tm.cstep * 4;
                tmpptr += 8;
            }
        }
        for (; i < tiles; i++)
        {
            float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * 4;

            for (int q = 0; q < inch; q++)
            {
                __m128i _val = __lsx_vld(r0, 0);
                __lsx_vst(_val, tmpptr, 0);

                r0 += bottom_blob_tm.cstep * 4;
                tmpptr += 4;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 16u, 4, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                const float* r0 = bb2.row(i / 12);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum4 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum5 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum6 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum7 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum8 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum9 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _suma = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sumb = (__m128)__lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 48);
                    __builtin_prefetch(k0 + 16);
                    __m128i _val0123 = __lsx_vld(r0, 0);
                    __m128i _val4567 = __lsx_vld(r0 + 4, 0);
                    __m128i _val89ab = __lsx_vld(r0 + 8, 0);
                    __m128 _w0 = (__m128)__lsx_vld(k0, 0);
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

                    r0 += 12;
                    k0 += 4;
                }

                __lsx_vst(_sum0, output0_tm, 0);
                __lsx_vst(_sum1, output0_tm + 4, 0);
                __lsx_vst(_sum2, output0_tm + 4 * 2, 0);
                __lsx_vst(_sum3, output0_tm + 4 * 3, 0);
                __lsx_vst(_sum4, output0_tm + 4 * 4, 0);
                __lsx_vst(_sum5, output0_tm + 4 * 5, 0);
                __lsx_vst(_sum6, output0_tm + 4 * 6, 0);
                __lsx_vst(_sum7, output0_tm + 4 * 7, 0);
                __lsx_vst(_sum8, output0_tm + 4 * 8, 0);
                __lsx_vst(_sum9, output0_tm + 4 * 9, 0);
                __lsx_vst(_suma, output0_tm + 4 * 10, 0);
                __lsx_vst(_sumb, output0_tm + 4 * 11, 0);

                output0_tm += 4 * 12;
            }
            for (; i + 7 < tiles; i += 8)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum4 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum5 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum6 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum7 = (__m128)__lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 32);
                    __builtin_prefetch(k0 + 16);
                    __m128i _val0123 = __lsx_vld(r0, 0);
                    __m128i _val4567 = __lsx_vld(r0 + 4, 0);
                    __m128 _w0 = (__m128)__lsx_vld(k0, 0);
                    _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 0), _sum0);
                    _sum1 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 1), _sum1);
                    _sum2 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 2), _sum2);
                    _sum3 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 3), _sum3);
                    _sum4 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 0), _sum4);
                    _sum5 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 1), _sum5);
                    _sum6 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 2), _sum6);
                    _sum7 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val4567, 3), _sum7);

                    r0 += 8;
                    k0 += 4;
                }

                __lsx_vst(_sum0, output0_tm, 0);
                __lsx_vst(_sum1, output0_tm + 4, 0);
                __lsx_vst(_sum2, output0_tm + 4 * 2, 0);
                __lsx_vst(_sum3, output0_tm + 4 * 3, 0);
                __lsx_vst(_sum4, output0_tm + 4 * 4, 0);
                __lsx_vst(_sum5, output0_tm + 4 * 5, 0);
                __lsx_vst(_sum6, output0_tm + 4 * 6, 0);
                __lsx_vst(_sum7, output0_tm + 4 * 7, 0);

                output0_tm += 4 * 8;
            }
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 16);
                    __builtin_prefetch(k0 + 16);
                    __m128i _val0123 = __lsx_vld(r0, 0);
                    __m128 _w0 = (__m128)__lsx_vld(k0, 0);
                    _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 0), _sum0);
                    _sum1 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 1), _sum1);
                    _sum2 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 2), _sum2);
                    _sum3 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val0123, 3), _sum3);

                    r0 += 4;
                    k0 += 4;
                }

                __lsx_vst(_sum0, output0_tm, 0);
                __lsx_vst(_sum1, output0_tm + 4, 0);
                __lsx_vst(_sum2, output0_tm + 4 * 2, 0);
                __lsx_vst(_sum3, output0_tm + 4 * 3, 0);

                output0_tm += 4 * 4;
            }
            for (; i + 1 < tiles; i += 2)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 8);
                    __builtin_prefetch(k0 + 16);
                    __m128 _val0 = __lsx_vreplfr2vr_s(*r0++);
                    __m128 _val1 = __lsx_vreplfr2vr_s(*r0++);
                    __m128 _w0 = (__m128)__lsx_vld(k0, 0);
                    _sum0 = __lsx_vfmadd_s(_w0, _val0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_w0, _val1, _sum1);

                    k0 += 4;
                }

                __lsx_vst(_sum0, output0_tm, 0);
                __lsx_vst(_sum1, output0_tm + 4, 0);

                output0_tm += 4 * 2;
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 4);
                    __builtin_prefetch(k0 + 16);
                    __m128 _val0 = __lsx_vreplfr2vr_s(*r0++);
                    __m128 _w0 = (__m128)__lsx_vld(k0, 0);
                    _sum = __lsx_vfmadd_s(_w0, _val0, _sum);

                    k0 += 4;
                }

                __lsx_vst(_sum, output0_tm, 0);

                output0_tm += 4;
            }
        }
    }
}
