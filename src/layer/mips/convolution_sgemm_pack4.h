// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void im2col_sgemm_pack4_msa(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
                    v4f32 _r0 = (v4f32)__msa_ld_w(img0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(img0 + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(img0 + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(img0 + 4 * 3, 0);
                    v4f32 _r4 = (v4f32)__msa_ld_w(img0 + 4 * 4, 0);
                    v4f32 _r5 = (v4f32)__msa_ld_w(img0 + 4 * 5, 0);
                    v4f32 _r6 = (v4f32)__msa_ld_w(img0 + 4 * 6, 0);
                    v4f32 _r7 = (v4f32)__msa_ld_w(img0 + 4 * 7, 0);
                    v4f32 _r8 = (v4f32)__msa_ld_w(img0 + 4 * 8, 0);
                    v4f32 _r9 = (v4f32)__msa_ld_w(img0 + 4 * 9, 0);
                    v4f32 _ra = (v4f32)__msa_ld_w(img0 + 4 * 10, 0);
                    v4f32 _rb = (v4f32)__msa_ld_w(img0 + 4 * 11, 0);

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
                    v4f32 _r0 = (v4f32)__msa_ld_w(img0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(img0 + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(img0 + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(img0 + 4 * 3, 0);
                    v4f32 _r4 = (v4f32)__msa_ld_w(img0 + 4 * 4, 0);
                    v4f32 _r5 = (v4f32)__msa_ld_w(img0 + 4 * 5, 0);
                    v4f32 _r6 = (v4f32)__msa_ld_w(img0 + 4 * 6, 0);
                    v4f32 _r7 = (v4f32)__msa_ld_w(img0 + 4 * 7, 0);

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
                    v4f32 _r0 = (v4f32)__msa_ld_w(img0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(img0 + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(img0 + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(img0 + 4 * 3, 0);

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
                    v4f32 _r0 = (v4f32)__msa_ld_w(img0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(img0 + 4, 0);

                    v4i32 _r01_0 = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01_1 = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);

                    __msa_st_w((v4i32)_r01_0, tmpptr, 0);
                    __msa_st_w((v4i32)_r01_1, tmpptr + 4, 0);

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
                    v4f32 _val = (v4f32)__msa_ld_w(img0, 0);
                    __msa_st_w((v4i32)_val, tmpptr, 0);

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

            v4f32 _sum0 = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = _sum0;
            v4f32 _sum2 = _sum0;
            v4f32 _sum3 = _sum0;
            v4f32 _sum4 = _sum0;
            v4f32 _sum5 = _sum0;
            v4f32 _sum6 = _sum0;
            v4f32 _sum7 = _sum0;
            v4f32 _sum8 = _sum0;
            v4f32 _sum9 = _sum0;
            v4f32 _suma = _sum0;
            v4f32 _sumb = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 48);
                __builtin_prefetch(kptr0 + 16);
                v4i32 _val0123 = __msa_ld_w(tmpptr, 0);
                v4i32 _val4567 = __msa_ld_w(tmpptr + 4, 0);
                v4i32 _val89ab = __msa_ld_w(tmpptr + 8, 0);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
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

                tmpptr += 12;
                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
            __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);
            __msa_st_w((v4i32)_sum4, outptr0 + 4 * 4, 0);
            __msa_st_w((v4i32)_sum5, outptr0 + 4 * 5, 0);
            __msa_st_w((v4i32)_sum6, outptr0 + 4 * 6, 0);
            __msa_st_w((v4i32)_sum7, outptr0 + 4 * 7, 0);
            __msa_st_w((v4i32)_sum8, outptr0 + 4 * 8, 0);
            __msa_st_w((v4i32)_sum9, outptr0 + 4 * 9, 0);
            __msa_st_w((v4i32)_suma, outptr0 + 4 * 10, 0);
            __msa_st_w((v4i32)_sumb, outptr0 + 4 * 11, 0);

            outptr0 += 4 * 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum0 = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = _sum0;
            v4f32 _sum2 = _sum0;
            v4f32 _sum3 = _sum0;
            v4f32 _sum4 = _sum0;
            v4f32 _sum5 = _sum0;
            v4f32 _sum6 = _sum0;
            v4f32 _sum7 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 32);
                __builtin_prefetch(kptr0 + 16);
                v4i32 _val0123 = __msa_ld_w(tmpptr, 0);
                v4i32 _val4567 = __msa_ld_w(tmpptr + 4, 0);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val0123, 0), _w0);
                _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val0123, 1), _w0);
                _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val0123, 2), _w0);
                _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val0123, 3), _w0);
                _sum4 = __msa_fmadd_w(_sum4, (v4f32)__msa_splati_w(_val4567, 0), _w0);
                _sum5 = __msa_fmadd_w(_sum5, (v4f32)__msa_splati_w(_val4567, 1), _w0);
                _sum6 = __msa_fmadd_w(_sum6, (v4f32)__msa_splati_w(_val4567, 2), _w0);
                _sum7 = __msa_fmadd_w(_sum7, (v4f32)__msa_splati_w(_val4567, 3), _w0);

                tmpptr += 8;
                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
            __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);
            __msa_st_w((v4i32)_sum4, outptr0 + 4 * 4, 0);
            __msa_st_w((v4i32)_sum5, outptr0 + 4 * 5, 0);
            __msa_st_w((v4i32)_sum6, outptr0 + 4 * 6, 0);
            __msa_st_w((v4i32)_sum7, outptr0 + 4 * 7, 0);

            outptr0 += 4 * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum0 = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = _sum0;
            v4f32 _sum2 = _sum0;
            v4f32 _sum3 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr0 + 16);
                v4i32 _val0123 = __msa_ld_w(tmpptr, 0);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val0123, 0), _w0);
                _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val0123, 1), _w0);
                _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val0123, 2), _w0);
                _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val0123, 3), _w0);

                tmpptr += 4;
                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
            __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);

            outptr0 += 4 * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum0 = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = _sum0;

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 8);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = __msa_fill_w_f32(*tmpptr++);
                v4f32 _val1 = __msa_fill_w_f32(*tmpptr++);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val0, _w0);
                _sum1 = __msa_fmadd_w(_sum1, _val1, _w0);

                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);

            outptr0 += 4 * 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum = bias ? (v4f32)__msa_ld_w(bias + p * 4, 0) : (v4f32)__msa_fill_w(0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 4);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = __msa_fill_w_f32(*tmpptr++);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
                _sum = __msa_fmadd_w(_sum, _val0, _w0);

                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum, outptr0, 0);

            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                            v4f32 _val = (v4f32)__msa_ld_w(sptr, 0);
                            __msa_st_w((v4i32)_val, ptr, 0);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4_msa(bottom_im2col, top_blob, kernel, _bias, opt);
}
