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

static void im2col_sgemm_pack4to1_msa(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u * 4, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    Mat tmp;
    if (size >= 12)
        tmp.create(12 * maxk, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + size % 12 % 4, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 4u * 4, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u * 4, 4, opt.workspace_allocator);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);

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

    int nn_outch = outch / 4;
    int remain_outch_start = nn_outch * 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);

        const float zeros[4] = {0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4i32 _bias = __msa_ld_w(biasptr, 0);
            v4f32 _sum0 = (v4f32)__msa_splati_w(_bias, 0);
            v4f32 _sum1 = (v4f32)__msa_splati_w(_bias, 0);
            v4f32 _sum2 = (v4f32)__msa_splati_w(_bias, 0);
            v4f32 _sum3 = (v4f32)__msa_splati_w(_bias, 1);
            v4f32 _sum4 = (v4f32)__msa_splati_w(_bias, 1);
            v4f32 _sum5 = (v4f32)__msa_splati_w(_bias, 1);
            v4f32 _sum6 = (v4f32)__msa_splati_w(_bias, 2);
            v4f32 _sum7 = (v4f32)__msa_splati_w(_bias, 2);
            v4f32 _sum8 = (v4f32)__msa_splati_w(_bias, 2);
            v4f32 _sum9 = (v4f32)__msa_splati_w(_bias, 3);
            v4f32 _suma = (v4f32)__msa_splati_w(_bias, 3);
            v4f32 _sumb = (v4f32)__msa_splati_w(_bias, 3);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 48);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4f32 _val1 = (v4f32)__msa_ld_w(tmpptr + 4, 0);
                v4f32 _val2 = (v4f32)__msa_ld_w(tmpptr + 8, 0);
                v4i32 _w0123 = __msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val0, (v4f32)__msa_splati_w(_w0123, 0));
                _sum1 = __msa_fmadd_w(_sum1, _val1, (v4f32)__msa_splati_w(_w0123, 0));
                _sum2 = __msa_fmadd_w(_sum2, _val2, (v4f32)__msa_splati_w(_w0123, 0));
                _sum3 = __msa_fmadd_w(_sum3, _val0, (v4f32)__msa_splati_w(_w0123, 1));
                _sum4 = __msa_fmadd_w(_sum4, _val1, (v4f32)__msa_splati_w(_w0123, 1));
                _sum5 = __msa_fmadd_w(_sum5, _val2, (v4f32)__msa_splati_w(_w0123, 1));
                _sum6 = __msa_fmadd_w(_sum6, _val0, (v4f32)__msa_splati_w(_w0123, 2));
                _sum7 = __msa_fmadd_w(_sum7, _val1, (v4f32)__msa_splati_w(_w0123, 2));
                _sum8 = __msa_fmadd_w(_sum8, _val2, (v4f32)__msa_splati_w(_w0123, 2));
                _sum9 = __msa_fmadd_w(_sum9, _val0, (v4f32)__msa_splati_w(_w0123, 3));
                _suma = __msa_fmadd_w(_suma, _val1, (v4f32)__msa_splati_w(_w0123, 3));
                _sumb = __msa_fmadd_w(_sumb, _val2, (v4f32)__msa_splati_w(_w0123, 3));

                tmpptr += 12;
                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr0 + 8, 0);
            __msa_st_w((v4i32)_sum3, outptr1, 0);
            __msa_st_w((v4i32)_sum4, outptr1 + 4, 0);
            __msa_st_w((v4i32)_sum5, outptr1 + 8, 0);
            __msa_st_w((v4i32)_sum6, outptr2, 0);
            __msa_st_w((v4i32)_sum7, outptr2 + 4, 0);
            __msa_st_w((v4i32)_sum8, outptr2 + 8, 0);
            __msa_st_w((v4i32)_sum9, outptr3, 0);
            __msa_st_w((v4i32)_suma, outptr3 + 4, 0);
            __msa_st_w((v4i32)_sumb, outptr3 + 8, 0);

            outptr0 += 12;
            outptr1 += 12;
            outptr2 += 12;
            outptr3 += 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4i32 _bias = __msa_ld_w(biasptr, 0);
            v4f32 _sum0 = (v4f32)__msa_splati_w(_bias, 0);
            v4f32 _sum1 = (v4f32)__msa_splati_w(_bias, 0);
            v4f32 _sum2 = (v4f32)__msa_splati_w(_bias, 1);
            v4f32 _sum3 = (v4f32)__msa_splati_w(_bias, 1);
            v4f32 _sum4 = (v4f32)__msa_splati_w(_bias, 2);
            v4f32 _sum5 = (v4f32)__msa_splati_w(_bias, 2);
            v4f32 _sum6 = (v4f32)__msa_splati_w(_bias, 3);
            v4f32 _sum7 = (v4f32)__msa_splati_w(_bias, 3);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 32);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4f32 _val1 = (v4f32)__msa_ld_w(tmpptr + 4, 0);
                v4i32 _w0123 = __msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val0, (v4f32)__msa_splati_w(_w0123, 0));
                _sum1 = __msa_fmadd_w(_sum1, _val1, (v4f32)__msa_splati_w(_w0123, 0));
                _sum2 = __msa_fmadd_w(_sum2, _val0, (v4f32)__msa_splati_w(_w0123, 1));
                _sum3 = __msa_fmadd_w(_sum3, _val1, (v4f32)__msa_splati_w(_w0123, 1));
                _sum4 = __msa_fmadd_w(_sum4, _val0, (v4f32)__msa_splati_w(_w0123, 2));
                _sum5 = __msa_fmadd_w(_sum5, _val1, (v4f32)__msa_splati_w(_w0123, 2));
                _sum6 = __msa_fmadd_w(_sum6, _val0, (v4f32)__msa_splati_w(_w0123, 3));
                _sum7 = __msa_fmadd_w(_sum7, _val1, (v4f32)__msa_splati_w(_w0123, 3));

                tmpptr += 8;
                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr1, 0);
            __msa_st_w((v4i32)_sum3, outptr1 + 4, 0);
            __msa_st_w((v4i32)_sum4, outptr2, 0);
            __msa_st_w((v4i32)_sum5, outptr2 + 4, 0);
            __msa_st_w((v4i32)_sum6, outptr3, 0);
            __msa_st_w((v4i32)_sum7, outptr3 + 4, 0);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4i32 _bias = __msa_ld_w(biasptr, 0);
            v4f32 _sum0 = (v4f32)__msa_splati_w(_bias, 0);
            v4f32 _sum1 = (v4f32)__msa_splati_w(_bias, 1);
            v4f32 _sum2 = (v4f32)__msa_splati_w(_bias, 2);
            v4f32 _sum3 = (v4f32)__msa_splati_w(_bias, 3);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4i32 _w0123 = __msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val0, (v4f32)__msa_splati_w(_w0123, 0));
                _sum1 = __msa_fmadd_w(_sum1, _val0, (v4f32)__msa_splati_w(_w0123, 1));
                _sum2 = __msa_fmadd_w(_sum2, _val0, (v4f32)__msa_splati_w(_w0123, 2));
                _sum3 = __msa_fmadd_w(_sum3, _val0, (v4f32)__msa_splati_w(_w0123, 3));

                tmpptr += 4;
                kptr0 += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr1, 0);
            __msa_st_w((v4i32)_sum2, outptr2, 0);
            __msa_st_w((v4i32)_sum3, outptr3, 0);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr0 = kernel.channel(p / 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum = (v4f32)__msa_ld_w(biasptr, 0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 4);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = __msa_fill_w_f32(*tmpptr++);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
                _sum = __msa_fmadd_w(_sum, _val0, _w0);

                kptr0 += 4;
            }

            outptr0[0] = _sum[0];
            outptr1[0] = _sum[1];
            outptr2[0] = _sum[2];
            outptr3[0] = _sum[3];

            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum0 = __msa_fill_w_f32(bias0);
            v4f32 _sum1 = __msa_fill_w_f32(bias0);
            v4f32 _sum2 = __msa_fill_w_f32(bias0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 48);
                __builtin_prefetch(kptr0 + 4);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4f32 _val1 = (v4f32)__msa_ld_w(tmpptr + 4, 0);
                v4f32 _val2 = (v4f32)__msa_ld_w(tmpptr + 8, 0);
                v4f32 _w0 = __msa_fill_w_f32(*kptr0);
                _sum0 = __msa_fmadd_w(_sum0, _w0, _val0);
                _sum1 = __msa_fmadd_w(_sum1, _w0, _val1);
                _sum2 = __msa_fmadd_w(_sum2, _w0, _val2);

                tmpptr += 12;
                kptr0 += 1;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr0 + 8, 0);

            outptr0 += 12;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum0 = __msa_fill_w_f32(bias0);
            v4f32 _sum1 = __msa_fill_w_f32(bias0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 32);
                __builtin_prefetch(kptr0 + 4);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4f32 _val1 = (v4f32)__msa_ld_w(tmpptr + 4, 0);
                v4f32 _w0 = __msa_fill_w_f32(*kptr0);
                _sum0 = __msa_fmadd_w(_sum0, _w0, _val0);
                _sum1 = __msa_fmadd_w(_sum1, _w0, _val1);

                tmpptr += 8;
                kptr0 += 1;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);

            outptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk * 4; // inch always > 0

            v4f32 _sum0 = __msa_fill_w_f32(bias0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr0 + 4);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4f32 _w0 = __msa_fill_w_f32(*kptr0);
                _sum0 = __msa_fmadd_w(_sum0, _w0, _val0);

                tmpptr += 4;
                kptr0 += 1;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);

            outptr0 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            v4f32 _sum0 = (v4f32)__msa_fill_w(0);

            for (int j = 0; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr0 + 16);
                v4f32 _val0 = (v4f32)__msa_ld_w(tmpptr, 0);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr0, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val0, _w0);
                tmpptr += 4;
                kptr0 += 4;
            }

            sum0 += __msa_reduce_fadd_w(_sum0);

            outptr0[0] = sum0;

            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4to1_msa(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(4 * 4 * maxk, inch / 4, outch / 4 + outch % 4);

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        float* g00 = kernel_tm.channel(q / 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
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

        float* g00 = kernel_tm.channel(q / 4 + q % 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4to1_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                    const float* sptr = img.row(dilation_h * u) + dilation_w * v * 4;

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

    im2col_sgemm_pack4to1_msa(bottom_im2col, top_blob, kernel, _bias, opt);
}
