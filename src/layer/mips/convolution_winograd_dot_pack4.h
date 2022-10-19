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

static void convolution_winograd_dot_pack4_msa(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
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
                v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                v4f32 _r4 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);
                v4f32 _r5 = (v4f32)__msa_ld_w(r0 + 4 * 5, 0);
                v4f32 _r6 = (v4f32)__msa_ld_w(r0 + 4 * 6, 0);
                v4f32 _r7 = (v4f32)__msa_ld_w(r0 + 4 * 7, 0);
                v4f32 _r8 = (v4f32)__msa_ld_w(r0 + 4 * 8, 0);
                v4f32 _r9 = (v4f32)__msa_ld_w(r0 + 4 * 9, 0);
                v4f32 _ra = (v4f32)__msa_ld_w(r0 + 4 * 10, 0);
                v4f32 _rb = (v4f32)__msa_ld_w(r0 + 4 * 11, 0);

                v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _r45r = __msa_ilvr_w((v4i32)_r5, (v4i32)_r4);
                v4i32 _r45l = __msa_ilvl_w((v4i32)_r5, (v4i32)_r4);
                v4i32 _r67r = __msa_ilvr_w((v4i32)_r7, (v4i32)_r6);
                v4i32 _r67l = __msa_ilvl_w((v4i32)_r7, (v4i32)_r6);
                v4i32 _r89r = __msa_ilvr_w((v4i32)_r9, (v4i32)_r8);
                v4i32 _r89l = __msa_ilvl_w((v4i32)_r9, (v4i32)_r8);
                v4i32 _rabr = __msa_ilvr_w((v4i32)_rb, (v4i32)_ra);
                v4i32 _rabl = __msa_ilvl_w((v4i32)_rb, (v4i32)_ra);
                v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);
                v2i64 _r4567_0 = __msa_ilvr_d((v2i64)_r67r, (v2i64)_r45r);
                v2i64 _r4567_1 = __msa_ilvl_d((v2i64)_r67r, (v2i64)_r45r);
                v2i64 _r4567_2 = __msa_ilvr_d((v2i64)_r67l, (v2i64)_r45l);
                v2i64 _r4567_3 = __msa_ilvl_d((v2i64)_r67l, (v2i64)_r45l);
                v2i64 _r89ab_0 = __msa_ilvr_d((v2i64)_rabr, (v2i64)_r89r);
                v2i64 _r89ab_1 = __msa_ilvl_d((v2i64)_rabr, (v2i64)_r89r);
                v2i64 _r89ab_2 = __msa_ilvr_d((v2i64)_rabl, (v2i64)_r89l);
                v2i64 _r89ab_3 = __msa_ilvl_d((v2i64)_rabl, (v2i64)_r89l);

                __msa_st_w((v4i32)_r0123_0, tmpptr, 0);
                __msa_st_w((v4i32)_r4567_0, tmpptr + 4, 0);
                __msa_st_w((v4i32)_r89ab_0, tmpptr + 4 * 2, 0);
                __msa_st_w((v4i32)_r0123_1, tmpptr + 4 * 3, 0);
                __msa_st_w((v4i32)_r4567_1, tmpptr + 4 * 4, 0);
                __msa_st_w((v4i32)_r89ab_1, tmpptr + 4 * 5, 0);
                __msa_st_w((v4i32)_r0123_2, tmpptr + 4 * 6, 0);
                __msa_st_w((v4i32)_r4567_2, tmpptr + 4 * 7, 0);
                __msa_st_w((v4i32)_r89ab_2, tmpptr + 4 * 8, 0);
                __msa_st_w((v4i32)_r0123_3, tmpptr + 4 * 9, 0);
                __msa_st_w((v4i32)_r4567_3, tmpptr + 4 * 10, 0);
                __msa_st_w((v4i32)_r89ab_3, tmpptr + 4 * 11, 0);

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
                v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);
                v4f32 _r4 = (v4f32)__msa_ld_w(r0 + 4 * 4, 0);
                v4f32 _r5 = (v4f32)__msa_ld_w(r0 + 4 * 5, 0);
                v4f32 _r6 = (v4f32)__msa_ld_w(r0 + 4 * 6, 0);
                v4f32 _r7 = (v4f32)__msa_ld_w(r0 + 4 * 7, 0);

                v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _r45r = __msa_ilvr_w((v4i32)_r5, (v4i32)_r4);
                v4i32 _r45l = __msa_ilvl_w((v4i32)_r5, (v4i32)_r4);
                v4i32 _r67r = __msa_ilvr_w((v4i32)_r7, (v4i32)_r6);
                v4i32 _r67l = __msa_ilvl_w((v4i32)_r7, (v4i32)_r6);
                v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);
                v2i64 _r4567_0 = __msa_ilvr_d((v2i64)_r67r, (v2i64)_r45r);
                v2i64 _r4567_1 = __msa_ilvl_d((v2i64)_r67r, (v2i64)_r45r);
                v2i64 _r4567_2 = __msa_ilvr_d((v2i64)_r67l, (v2i64)_r45l);
                v2i64 _r4567_3 = __msa_ilvl_d((v2i64)_r67l, (v2i64)_r45l);

                __msa_st_w((v4i32)_r0123_0, tmpptr, 0);
                __msa_st_w((v4i32)_r4567_0, tmpptr + 4, 0);
                __msa_st_w((v4i32)_r0123_1, tmpptr + 4 * 2, 0);
                __msa_st_w((v4i32)_r4567_1, tmpptr + 4 * 3, 0);
                __msa_st_w((v4i32)_r0123_2, tmpptr + 4 * 4, 0);
                __msa_st_w((v4i32)_r4567_2, tmpptr + 4 * 5, 0);
                __msa_st_w((v4i32)_r0123_3, tmpptr + 4 * 6, 0);
                __msa_st_w((v4i32)_r4567_3, tmpptr + 4 * 7, 0);

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
                v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(r0 + 4, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);

                v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                __msa_st_w((v4i32)_r0123_0, tmpptr, 0);
                __msa_st_w((v4i32)_r0123_1, tmpptr + 4, 0);
                __msa_st_w((v4i32)_r0123_2, tmpptr + 4 * 2, 0);
                __msa_st_w((v4i32)_r0123_3, tmpptr + 4 * 3, 0);

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
                v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(r0 + 4, 0);

                v4i32 _r01_0 = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r01_1 = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);

                __msa_st_w((v4i32)_r01_0, tmpptr, 0);
                __msa_st_w((v4i32)_r01_1, tmpptr + 4, 0);

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
                v4f32 _val = (v4f32)__msa_ld_w(r0, 0);
                __msa_st_w((v4i32)_val, tmpptr, 0);

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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);
                v4f32 _sum4 = (v4f32)__msa_fill_w(0);
                v4f32 _sum5 = (v4f32)__msa_fill_w(0);
                v4f32 _sum6 = (v4f32)__msa_fill_w(0);
                v4f32 _sum7 = (v4f32)__msa_fill_w(0);
                v4f32 _sum8 = (v4f32)__msa_fill_w(0);
                v4f32 _sum9 = (v4f32)__msa_fill_w(0);
                v4f32 _suma = (v4f32)__msa_fill_w(0);
                v4f32 _sumb = (v4f32)__msa_fill_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 48);
                    __builtin_prefetch(k0 + 16);
                    v4i32 _val0123 = __msa_ld_w(r0, 0);
                    v4i32 _val4567 = __msa_ld_w(r0 + 4, 0);
                    v4i32 _val89ab = __msa_ld_w(r0 + 8, 0);
                    v4f32 _w0 = (v4f32)__msa_ld_w(k0, 0);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val0123, 0), _w0);
                    _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val0123, 1), _w0);
                    _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val0123, 2), _w0);
                    _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val0123, 3), _w0);
                    _sum4 = __msa_fmadd_w(_sum4, (v4f32)__msa_splati_w(_val4567, 0), _w0);
                    _sum5 = __msa_fmadd_w(_sum5, (v4f32)__msa_splati_w(_val4567, 1), _w0);
                    _sum6 = __msa_fmadd_w(_sum6, (v4f32)__msa_splati_w(_val4567, 2), _w0);
                    _sum7 = __msa_fmadd_w(_sum7, (v4f32)__msa_splati_w(_val4567, 3), _w0);
                    _sum8 = __msa_fmadd_w(_sum8, (v4f32)__msa_splati_w(_val89ab, 0), _w0);
                    _sum9 = __msa_fmadd_w(_sum9, (v4f32)__msa_splati_w(_val89ab, 1), _w0);
                    _suma = __msa_fmadd_w(_suma, (v4f32)__msa_splati_w(_val89ab, 2), _w0);
                    _sumb = __msa_fmadd_w(_sumb, (v4f32)__msa_splati_w(_val89ab, 3), _w0);

                    r0 += 12;
                    k0 += 4;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                __msa_st_w((v4i32)_sum1, output0_tm + 4, 0);
                __msa_st_w((v4i32)_sum2, output0_tm + 4 * 2, 0);
                __msa_st_w((v4i32)_sum3, output0_tm + 4 * 3, 0);
                __msa_st_w((v4i32)_sum4, output0_tm + 4 * 4, 0);
                __msa_st_w((v4i32)_sum5, output0_tm + 4 * 5, 0);
                __msa_st_w((v4i32)_sum6, output0_tm + 4 * 6, 0);
                __msa_st_w((v4i32)_sum7, output0_tm + 4 * 7, 0);
                __msa_st_w((v4i32)_sum8, output0_tm + 4 * 8, 0);
                __msa_st_w((v4i32)_sum9, output0_tm + 4 * 9, 0);
                __msa_st_w((v4i32)_suma, output0_tm + 4 * 10, 0);
                __msa_st_w((v4i32)_sumb, output0_tm + 4 * 11, 0);

                output0_tm += 4 * 12;
            }
            for (; i + 7 < tiles; i += 8)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);
                v4f32 _sum4 = (v4f32)__msa_fill_w(0);
                v4f32 _sum5 = (v4f32)__msa_fill_w(0);
                v4f32 _sum6 = (v4f32)__msa_fill_w(0);
                v4f32 _sum7 = (v4f32)__msa_fill_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 32);
                    __builtin_prefetch(k0 + 16);
                    v4i32 _val0123 = __msa_ld_w(r0, 0);
                    v4i32 _val4567 = __msa_ld_w(r0 + 4, 0);
                    v4f32 _w0 = (v4f32)__msa_ld_w(k0, 0);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val0123, 0), _w0);
                    _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val0123, 1), _w0);
                    _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val0123, 2), _w0);
                    _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val0123, 3), _w0);
                    _sum4 = __msa_fmadd_w(_sum4, (v4f32)__msa_splati_w(_val4567, 0), _w0);
                    _sum5 = __msa_fmadd_w(_sum5, (v4f32)__msa_splati_w(_val4567, 1), _w0);
                    _sum6 = __msa_fmadd_w(_sum6, (v4f32)__msa_splati_w(_val4567, 2), _w0);
                    _sum7 = __msa_fmadd_w(_sum7, (v4f32)__msa_splati_w(_val4567, 3), _w0);

                    r0 += 8;
                    k0 += 4;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                __msa_st_w((v4i32)_sum1, output0_tm + 4, 0);
                __msa_st_w((v4i32)_sum2, output0_tm + 4 * 2, 0);
                __msa_st_w((v4i32)_sum3, output0_tm + 4 * 3, 0);
                __msa_st_w((v4i32)_sum4, output0_tm + 4 * 4, 0);
                __msa_st_w((v4i32)_sum5, output0_tm + 4 * 5, 0);
                __msa_st_w((v4i32)_sum6, output0_tm + 4 * 6, 0);
                __msa_st_w((v4i32)_sum7, output0_tm + 4 * 7, 0);

                output0_tm += 4 * 8;
            }
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 16);
                    __builtin_prefetch(k0 + 16);
                    v4i32 _val0123 = __msa_ld_w(r0, 0);
                    v4f32 _w0 = (v4f32)__msa_ld_w(k0, 0);
                    _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val0123, 0), _w0);
                    _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val0123, 1), _w0);
                    _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val0123, 2), _w0);
                    _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val0123, 3), _w0);

                    r0 += 4;
                    k0 += 4;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                __msa_st_w((v4i32)_sum1, output0_tm + 4, 0);
                __msa_st_w((v4i32)_sum2, output0_tm + 4 * 2, 0);
                __msa_st_w((v4i32)_sum3, output0_tm + 4 * 3, 0);

                output0_tm += 4 * 4;
            }
            for (; i + 1 < tiles; i += 2)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 8);
                    __builtin_prefetch(k0 + 16);
                    v4f32 _val0 = __msa_fill_w_f32(*r0++);
                    v4f32 _val1 = __msa_fill_w_f32(*r0++);
                    v4f32 _w0 = (v4f32)__msa_ld_w(k0, 0);
                    _sum0 = __msa_fmadd_w(_sum0, _val0, _w0);
                    _sum1 = __msa_fmadd_w(_sum1, _val1, _w0);

                    k0 += 4;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                __msa_st_w((v4i32)_sum1, output0_tm + 4, 0);

                output0_tm += 4 * 2;
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch * 4; // inch always > 0

                v4f32 _sum = (v4f32)__msa_fill_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 4);
                    __builtin_prefetch(k0 + 16);
                    v4f32 _val0 = __msa_fill_w_f32(*r0++);
                    v4f32 _w0 = (v4f32)__msa_ld_w(k0, 0);
                    _sum = __msa_fmadd_w(_sum, _val0, _w0);

                    k0 += 4;
                }

                __msa_st_w((v4i32)_sum, output0_tm, 0);

                output0_tm += 4;
            }
        }
    }
}
