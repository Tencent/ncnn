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

static void gridsample_2d_bilinear_align0_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m512 two = _mm512_set1_ps(2.f);

                // x
                gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);

                // y
                gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);
            }

            __m512 x_w = _mm512_floor_ps(gx);
            __m512 y_n = _mm512_floor_ps(gy);

            __m512 w = _mm512_sub_ps(gx, x_w);
            __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
            __m512 n = _mm512_sub_ps(gy, y_n);
            __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);

            __m512 nw = _mm512_mul_ps(s, e);
            __m512 ne = _mm512_mul_ps(s, w);
            __m512 sw = _mm512_mul_ps(n, e);
            __m512 se = _mm512_mul_ps(n, w);

            __m512i x0 = _mm512_cvtps_epi32(x_w);
            __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
            __m512i y0 = _mm512_cvtps_epi32(y_n);
            __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);

            __mmask16 x0_in_range = _mm512_cmpgt_epi32_mask(x0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x0);
            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 y0_in_range = _mm512_cmpgt_epi32_mask(y0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y0);
            __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);

            __mmask16 v00_in_range = x0_in_range & y0_in_range;
            __mmask16 v01_in_range = x0_in_range & y1_in_range;
            __mmask16 v10_in_range = x1_in_range & y0_in_range;
            __mmask16 v11_in_range = x1_in_range & y1_in_range;

            // (W*y + x) * elempack + vec(8)
            __m512i i_nw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                   _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i i_ne_offset = _mm512_add_epi32(i_nw_offset, vElempacki);
            __m512i i_sw_offset = _mm512_add_epi32(i_nw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
            __m512i i_se_offset = _mm512_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                __m512 nw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v00_in_range, i_nw_offset, src.channel(q), sizeof(float));
                __m512 ne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v10_in_range, i_ne_offset, src.channel(q), sizeof(float));
                __m512 sw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v01_in_range, i_sw_offset, src.channel(q), sizeof(float));
                __m512 se_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v11_in_range, i_se_offset, src.channel(q), sizeof(float));

                __m512 _v = _mm512_mul_ps(nw_val, nw);
                _v = _mm512_fmadd_ps(ne_val, ne, _v);
                _v = _mm512_fmadd_ps(sw_val, sw, _v);
                _v = _mm512_fmadd_ps(se_val, se, _v);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align1_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m512 two = _mm512_set1_ps(2.f);

                // x
                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));

                // y
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));
            }

            __m512 x_w = _mm512_floor_ps(gx);
            __m512 y_n = _mm512_floor_ps(gy);

            __m512 w = _mm512_sub_ps(gx, x_w);
            __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
            __m512 n = _mm512_sub_ps(gy, y_n);
            __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);

            __m512 nw = _mm512_mul_ps(s, e);
            __m512 ne = _mm512_mul_ps(s, w);
            __m512 sw = _mm512_mul_ps(n, e);
            __m512 se = _mm512_mul_ps(n, w);

            __m512i x0 = _mm512_cvtps_epi32(x_w);
            __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
            __m512i y0 = _mm512_cvtps_epi32(y_n);
            __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);

            __mmask16 x0_in_range = _mm512_cmpgt_epi32_mask(x0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x0);
            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 y0_in_range = _mm512_cmpgt_epi32_mask(y0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y0);
            __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);

            __mmask16 v00_in_range = x0_in_range & y0_in_range;
            __mmask16 v01_in_range = x0_in_range & y1_in_range;
            __mmask16 v10_in_range = x1_in_range & y0_in_range;
            __mmask16 v11_in_range = x1_in_range & y1_in_range;

            // (W*y + x) * elempack + vec(8)
            __m512i i_nw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                   _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i i_ne_offset = _mm512_add_epi32(i_nw_offset, vElempacki);
            __m512i i_sw_offset = _mm512_add_epi32(i_nw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
            __m512i i_se_offset = _mm512_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                __m512 nw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v00_in_range, i_nw_offset, src.channel(q), sizeof(float));
                __m512 ne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v10_in_range, i_ne_offset, src.channel(q), sizeof(float));
                __m512 sw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v01_in_range, i_sw_offset, src.channel(q), sizeof(float));
                __m512 se_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v11_in_range, i_se_offset, src.channel(q), sizeof(float));

                __m512 _v = _mm512_mul_ps(nw_val, nw);
                _v = _mm512_fmadd_ps(ne_val, ne, _v);
                _v = _mm512_fmadd_ps(sw_val, sw, _v);
                _v = _mm512_fmadd_ps(se_val, se, _v);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align0_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m512 two = _mm512_set1_ps(2.f);

                // x
                gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);

                const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                // y
                gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);

                const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
            }

            __m512 x_w = _mm512_floor_ps(gx);
            __m512 y_n = _mm512_floor_ps(gy);

            __m512 w = _mm512_sub_ps(gx, x_w);
            __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
            __m512 n = _mm512_sub_ps(gy, y_n);
            __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);

            __m512 nw = _mm512_mul_ps(s, e);
            __m512 ne = _mm512_mul_ps(s, w);
            __m512 sw = _mm512_mul_ps(n, e);
            __m512 se = _mm512_mul_ps(n, w);

            __m512i x0 = _mm512_cvtps_epi32(x_w);
            __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
            __m512i y0 = _mm512_cvtps_epi32(y_n);
            __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);

            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);

            __mmask16 v11_in_range = x1_in_range & y1_in_range;

            // (W*y + x) * elempack + vec(8)
            __m512i i_nw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                   _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i i_ne_offset = _mm512_add_epi32(i_nw_offset, vElempacki);
            __m512i i_sw_offset = _mm512_add_epi32(i_nw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
            __m512i i_se_offset = _mm512_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                __m512 nw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_nw_offset, src.channel(q), sizeof(float));
                __m512 ne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_ne_offset, src.channel(q), sizeof(float));
                __m512 sw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_sw_offset, src.channel(q), sizeof(float));
                __m512 se_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v11_in_range, i_se_offset, src.channel(q), sizeof(float));

                __m512 _v = _mm512_mul_ps(nw_val, nw);
                _v = _mm512_fmadd_ps(ne_val, ne, _v);
                _v = _mm512_fmadd_ps(sw_val, sw, _v);
                _v = _mm512_fmadd_ps(se_val, se, _v);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align1_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m512 two = _mm512_set1_ps(2.f);

                // x
                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));

                const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                // y
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));

                const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
            }

            __m512 x_w = _mm512_floor_ps(gx);
            __m512 y_n = _mm512_floor_ps(gy);

            __m512 w = _mm512_sub_ps(gx, x_w);
            __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
            __m512 n = _mm512_sub_ps(gy, y_n);
            __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);

            __m512 nw = _mm512_mul_ps(s, e);
            __m512 ne = _mm512_mul_ps(s, w);
            __m512 sw = _mm512_mul_ps(n, e);
            __m512 se = _mm512_mul_ps(n, w);

            __m512i x0 = _mm512_cvtps_epi32(x_w);
            __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
            __m512i y0 = _mm512_cvtps_epi32(y_n);
            __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);

            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);

            __mmask16 v11_in_range = x1_in_range & y1_in_range;

            // (W*y + x) * elempack + vec(8)
            __m512i i_nw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                   _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i i_ne_offset = _mm512_add_epi32(i_nw_offset, vElempacki);
            __m512i i_sw_offset = _mm512_add_epi32(i_nw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
            __m512i i_se_offset = _mm512_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                __m512 nw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_nw_offset, src.channel(q), sizeof(float));
                __m512 ne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_ne_offset, src.channel(q), sizeof(float));
                __m512 sw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_sw_offset, src.channel(q), sizeof(float));
                __m512 se_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v11_in_range, i_se_offset, src.channel(q), sizeof(float));

                __m512 _v = _mm512_mul_ps(nw_val, nw);
                _v = _mm512_fmadd_ps(ne_val, ne, _v);
                _v = _mm512_fmadd_ps(sw_val, sw, _v);
                _v = _mm512_fmadd_ps(se_val, se, _v);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align0_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m512 two = _mm512_set1_ps(2.f);

                // x
                gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);

                const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                __m512 v0p5fp16 = _mm512_set1_ps(0.5f);
                gx = _mm512_add_ps(gx, v0p5fp16);

                gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                gx = _mm512_sub_ps(vImgWf, reflectx_v);

                gx = _mm512_sub_ps(gx, v0p5fp16);

                _mm512_sub_ps(gx, v0p5fp16);

                gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                // y
                gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);

                const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                gy = _mm512_add_ps(gy, v0p5fp16);

                gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, vImgHf), *(__m512*)_ps512_inv_sign_mask);
                gy = _mm512_sub_ps(vImgHf, reflecty_v);

                gy = _mm512_sub_ps(gy, v0p5fp16);

                _mm512_sub_ps(gy, v0p5fp16);

                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
            }

            __m512 x_w = _mm512_floor_ps(gx);
            __m512 y_n = _mm512_floor_ps(gy);

            __m512 w = _mm512_sub_ps(gx, x_w);
            __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
            __m512 n = _mm512_sub_ps(gy, y_n);
            __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);

            __m512 nw = _mm512_mul_ps(s, e);
            __m512 ne = _mm512_mul_ps(s, w);
            __m512 sw = _mm512_mul_ps(n, e);
            __m512 se = _mm512_mul_ps(n, w);

            __m512i x0 = _mm512_cvtps_epi32(x_w);
            __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
            __m512i y0 = _mm512_cvtps_epi32(y_n);
            __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);

            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);

            __mmask16 v11_in_range = x1_in_range & y1_in_range;

            // (W*y + x) * elempack + vec(8)
            __m512i i_nw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                   _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i i_ne_offset = _mm512_add_epi32(i_nw_offset, vElempacki);
            __m512i i_sw_offset = _mm512_add_epi32(i_nw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
            __m512i i_se_offset = _mm512_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                __m512 nw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_nw_offset, src.channel(q), sizeof(float));
                __m512 ne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_ne_offset, src.channel(q), sizeof(float));
                __m512 sw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_sw_offset, src.channel(q), sizeof(float));
                __m512 se_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v11_in_range, i_se_offset, src.channel(q), sizeof(float));

                __m512 _v = _mm512_mul_ps(nw_val, nw);
                _v = _mm512_fmadd_ps(ne_val, ne, _v);
                _v = _mm512_fmadd_ps(sw_val, sw, _v);
                _v = _mm512_fmadd_ps(se_val, se, _v);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align1_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m512 two = _mm512_set1_ps(2.f);

                // x
                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));

                const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, border_x), *(__m512*)_ps512_inv_sign_mask);
                gx = _mm512_sub_ps(border_x, reflectx_v);

                // y
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));

                const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, border_y), *(__m512*)_ps512_inv_sign_mask);
                gy = _mm512_sub_ps(border_y, reflecty_v);
            }

            __m512 x_w = _mm512_floor_ps(gx);
            __m512 y_n = _mm512_floor_ps(gy);

            __m512 w = _mm512_sub_ps(gx, x_w);
            __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
            __m512 n = _mm512_sub_ps(gy, y_n);
            __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);

            __m512 nw = _mm512_mul_ps(s, e);
            __m512 ne = _mm512_mul_ps(s, w);
            __m512 sw = _mm512_mul_ps(n, e);
            __m512 se = _mm512_mul_ps(n, w);

            __m512i x0 = _mm512_cvtps_epi32(x_w);
            __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
            __m512i y0 = _mm512_cvtps_epi32(y_n);
            __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);

            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);

            __mmask16 v11_in_range = x1_in_range & y1_in_range;

            // (W*y + x) * elempack + vec(8)
            __m512i i_nw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                   _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i i_ne_offset = _mm512_add_epi32(i_nw_offset, vElempacki);
            __m512i i_sw_offset = _mm512_add_epi32(i_nw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
            __m512i i_se_offset = _mm512_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                __m512 nw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_nw_offset, src.channel(q), sizeof(float));
                __m512 ne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_ne_offset, src.channel(q), sizeof(float));
                __m512 sw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_sw_offset, src.channel(q), sizeof(float));
                __m512 se_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v11_in_range, i_se_offset, src.channel(q), sizeof(float));

                __m512 _v = _mm512_mul_ps(nw_val, nw);
                _v = _mm512_fmadd_ps(ne_val, ne, _v);
                _v = _mm512_fmadd_ps(sw_val, sw, _v);
                _v = _mm512_fmadd_ps(se_val, se, _v);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_3d_bilinear_align0_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512 vImgDf = _mm512_set1_ps(src.d);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);
    const __m512i vImgDi = _mm512_set1_epi32(src.d);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m512 gx = _mm512_set1_ps(gridptr[0]);
                __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);
                __m512 gz = _mm512_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m512 two = _mm512_set1_ps(2.f);

                    // x
                    gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);

                    // y
                    gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);

                    // z
                    gz = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gz, *(__m512*)_ps512_1), vImgDf, *(__m512*)_ps512_1), two);
                }

                __m512 x_w = _mm512_floor_ps(gx);
                __m512 y_n = _mm512_floor_ps(gy);
                __m512 z_t = _mm512_floor_ps(gz);

                __m512 w = _mm512_sub_ps(gx, x_w);
                __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
                __m512 n = _mm512_sub_ps(gy, y_n);
                __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);
                __m512 t = _mm512_sub_ps(gz, z_t);
                __m512 b = _mm512_sub_ps(*(__m512*)_ps512_1, t);

                __m512 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m512 nw = _mm512_mul_ps(s, e);
                    __m512 ne = _mm512_mul_ps(s, w);
                    __m512 sw = _mm512_mul_ps(n, e);
                    __m512 se = _mm512_mul_ps(n, w);

                    tnw = _mm512_mul_ps(b, nw);
                    tne = _mm512_mul_ps(b, ne);
                    tsw = _mm512_mul_ps(b, sw);
                    tse = _mm512_mul_ps(b, se);

                    bnw = _mm512_mul_ps(t, nw);
                    bne = _mm512_mul_ps(t, ne);
                    bsw = _mm512_mul_ps(t, sw);
                    bse = _mm512_mul_ps(t, se);
                }

                __m512i x0 = _mm512_cvtps_epi32(x_w);
                __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
                __m512i y0 = _mm512_cvtps_epi32(y_n);
                __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);
                __m512i z0 = _mm512_cvtps_epi32(z_t);
                __m512i z1 = _mm512_add_epi32(z0, *(__m512i*)_pi32_512_1);

                __mmask16 x0_in_range = _mm512_cmpgt_epi32_mask(x0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x0);
                __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
                __mmask16 y0_in_range = _mm512_cmpgt_epi32_mask(y0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y0);
                __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);
                __mmask16 z0_in_range = _mm512_cmpgt_epi32_mask(z0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z0);
                __mmask16 z1_in_range = _mm512_cmpgt_epi32_mask(z1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z1);

                __mmask16 v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __mmask16 v00_in_range = x0_in_range & y0_in_range;
                    __mmask16 v01_in_range = x0_in_range & y1_in_range;
                    __mmask16 v10_in_range = x1_in_range & y0_in_range;
                    __mmask16 v11_in_range = x1_in_range & y1_in_range;

                    v000_in_range = v00_in_range & z0_in_range;
                    v010_in_range = v01_in_range & z0_in_range;
                    v100_in_range = v10_in_range & z0_in_range;
                    v110_in_range = v11_in_range & z0_in_range;

                    v001_in_range = v00_in_range & z1_in_range;
                    v011_in_range = v01_in_range & z1_in_range;
                    v101_in_range = v10_in_range & z1_in_range;
                    v111_in_range = v11_in_range & z1_in_range;
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                __m512i i_tnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), z0), _mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                __m512i i_tne_offset = _mm512_add_epi32(i_tnw_offset, vElempacki);
                __m512i i_tsw_offset = _mm512_add_epi32(i_tnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_tse_offset = _mm512_add_epi32(i_tsw_offset, vElempacki);

                __m512i i_bnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m512i i_bne_offset = _mm512_add_epi32(i_bnw_offset, vElempacki);
                __m512i i_bsw_offset = _mm512_add_epi32(i_bnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_bse_offset = _mm512_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 tnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v000_in_range, i_tnw_offset, src.channel(q), sizeof(float));
                    __m512 tne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v100_in_range, i_tne_offset, src.channel(q), sizeof(float));
                    __m512 tsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v010_in_range, i_tsw_offset, src.channel(q), sizeof(float));
                    __m512 tse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v110_in_range, i_tse_offset, src.channel(q), sizeof(float));

                    __m512 bnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v001_in_range, i_bnw_offset, src.channel(q), sizeof(float));
                    __m512 bne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v101_in_range, i_bne_offset, src.channel(q), sizeof(float));
                    __m512 bsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v011_in_range, i_bsw_offset, src.channel(q), sizeof(float));
                    __m512 bse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v111_in_range, i_bse_offset, src.channel(q), sizeof(float));

                    __m512 _v = _mm512_mul_ps(tnw_val, tnw);
                    _v = _mm512_fmadd_ps(tne_val, tne, _v);
                    _v = _mm512_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm512_fmadd_ps(tse_val, tse, _v);

                    _v = _mm512_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm512_fmadd_ps(bne_val, bne, _v);
                    _v = _mm512_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm512_fmadd_ps(bse_val, bse, _v);

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align1_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512 vImgDf = _mm512_set1_ps(src.d);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);
    const __m512i vImgDi = _mm512_set1_epi32(src.d);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m512 gx = _mm512_set1_ps(gridptr[0]);
                __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);
                __m512 gz = _mm512_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m512 two = _mm512_set1_ps(2.f);

                    // x
                    gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));

                    // y
                    gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));

                    // z
                    gz = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gz, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1));
                }

                __m512 x_w = _mm512_floor_ps(gx);
                __m512 y_n = _mm512_floor_ps(gy);
                __m512 z_t = _mm512_floor_ps(gz);

                __m512 w = _mm512_sub_ps(gx, x_w);
                __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
                __m512 n = _mm512_sub_ps(gy, y_n);
                __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);
                __m512 t = _mm512_sub_ps(gz, z_t);
                __m512 b = _mm512_sub_ps(*(__m512*)_ps512_1, t);

                __m512 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m512 nw = _mm512_mul_ps(s, e);
                    __m512 ne = _mm512_mul_ps(s, w);
                    __m512 sw = _mm512_mul_ps(n, e);
                    __m512 se = _mm512_mul_ps(n, w);

                    tnw = _mm512_mul_ps(b, nw);
                    tne = _mm512_mul_ps(b, ne);
                    tsw = _mm512_mul_ps(b, sw);
                    tse = _mm512_mul_ps(b, se);

                    bnw = _mm512_mul_ps(t, nw);
                    bne = _mm512_mul_ps(t, ne);
                    bsw = _mm512_mul_ps(t, sw);
                    bse = _mm512_mul_ps(t, se);
                }

                __m512i x0 = _mm512_cvtps_epi32(x_w);
                __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
                __m512i y0 = _mm512_cvtps_epi32(y_n);
                __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);
                __m512i z0 = _mm512_cvtps_epi32(z_t);
                __m512i z1 = _mm512_add_epi32(z0, *(__m512i*)_pi32_512_1);

                __mmask16 x0_in_range = _mm512_cmpgt_epi32_mask(x0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x0);
                __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
                __mmask16 y0_in_range = _mm512_cmpgt_epi32_mask(y0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y0);
                __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);
                __mmask16 z0_in_range = _mm512_cmpgt_epi32_mask(z0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z0);
                __mmask16 z1_in_range = _mm512_cmpgt_epi32_mask(z1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z1);

                __mmask16 v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __mmask16 v00_in_range = x0_in_range & y0_in_range;
                    __mmask16 v01_in_range = x0_in_range & y1_in_range;
                    __mmask16 v10_in_range = x1_in_range & y0_in_range;
                    __mmask16 v11_in_range = x1_in_range & y1_in_range;

                    v000_in_range = v00_in_range & z0_in_range;
                    v010_in_range = v01_in_range & z0_in_range;
                    v100_in_range = v10_in_range & z0_in_range;
                    v110_in_range = v11_in_range & z0_in_range;

                    v001_in_range = v00_in_range & z1_in_range;
                    v011_in_range = v01_in_range & z1_in_range;
                    v101_in_range = v10_in_range & z1_in_range;
                    v111_in_range = v11_in_range & z1_in_range;
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                __m512i i_tnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), z0), _mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                __m512i i_tne_offset = _mm512_add_epi32(i_tnw_offset, vElempacki);
                __m512i i_tsw_offset = _mm512_add_epi32(i_tnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_tse_offset = _mm512_add_epi32(i_tsw_offset, vElempacki);

                __m512i i_bnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m512i i_bne_offset = _mm512_add_epi32(i_bnw_offset, vElempacki);
                __m512i i_bsw_offset = _mm512_add_epi32(i_bnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_bse_offset = _mm512_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 tnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v000_in_range, i_tnw_offset, src.channel(q), sizeof(float));
                    __m512 tne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v100_in_range, i_tne_offset, src.channel(q), sizeof(float));
                    __m512 tsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v010_in_range, i_tsw_offset, src.channel(q), sizeof(float));
                    __m512 tse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v110_in_range, i_tse_offset, src.channel(q), sizeof(float));

                    __m512 bnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v001_in_range, i_bnw_offset, src.channel(q), sizeof(float));
                    __m512 bne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v101_in_range, i_bne_offset, src.channel(q), sizeof(float));
                    __m512 bsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v011_in_range, i_bsw_offset, src.channel(q), sizeof(float));
                    __m512 bse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v111_in_range, i_bse_offset, src.channel(q), sizeof(float));

                    __m512 _v = _mm512_mul_ps(tnw_val, tnw);
                    _v = _mm512_fmadd_ps(tne_val, tne, _v);
                    _v = _mm512_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm512_fmadd_ps(tse_val, tse, _v);

                    _v = _mm512_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm512_fmadd_ps(bne_val, bne, _v);
                    _v = _mm512_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm512_fmadd_ps(bse_val, bse, _v);

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align0_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512 vImgDf = _mm512_set1_ps(src.d);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);
    const __m512i vImgDi = _mm512_set1_epi32(src.d);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m512 gx = _mm512_set1_ps(gridptr[0]);
                __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);
                __m512 gz = _mm512_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m512 two = _mm512_set1_ps(2.f);

                    // x
                    gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);

                    const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                    gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                    // y
                    gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);

                    const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                    // z
                    gz = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gz, *(__m512*)_ps512_1), vImgDf, *(__m512*)_ps512_1), two);

                    const __m512 border_z = _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1);

                    gz = _mm512_min_ps(border_z, _mm512_max_ps(gz, _mm512_setzero_ps()));
                }

                __m512 x_w = _mm512_floor_ps(gx);
                __m512 y_n = _mm512_floor_ps(gy);
                __m512 z_t = _mm512_floor_ps(gz);

                __m512 w = _mm512_sub_ps(gx, x_w);
                __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
                __m512 n = _mm512_sub_ps(gy, y_n);
                __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);
                __m512 t = _mm512_sub_ps(gz, z_t);
                __m512 b = _mm512_sub_ps(*(__m512*)_ps512_1, t);

                __m512 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m512 nw = _mm512_mul_ps(s, e);
                    __m512 ne = _mm512_mul_ps(s, w);
                    __m512 sw = _mm512_mul_ps(n, e);
                    __m512 se = _mm512_mul_ps(n, w);

                    tnw = _mm512_mul_ps(b, nw);
                    tne = _mm512_mul_ps(b, ne);
                    tsw = _mm512_mul_ps(b, sw);
                    tse = _mm512_mul_ps(b, se);

                    bnw = _mm512_mul_ps(t, nw);
                    bne = _mm512_mul_ps(t, ne);
                    bsw = _mm512_mul_ps(t, sw);
                    bse = _mm512_mul_ps(t, se);
                }

                __m512i x0 = _mm512_cvtps_epi32(x_w);
                __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
                __m512i y0 = _mm512_cvtps_epi32(y_n);
                __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);
                __m512i z0 = _mm512_cvtps_epi32(z_t);
                __m512i z1 = _mm512_add_epi32(z0, *(__m512i*)_pi32_512_1);

                __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
                __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);
                __mmask16 z1_in_range = _mm512_cmpgt_epi32_mask(z1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z1);

                __mmask16 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __mmask16 v11_in_range = x1_in_range & y1_in_range;

                    v110_in_range = x1_in_range & y1_in_range;

                    v011_in_range = y1_in_range & z1_in_range;
                    v101_in_range = x1_in_range & z1_in_range;
                    v111_in_range = v11_in_range & z1_in_range;
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                __m512i i_tnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), z0), _mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                __m512i i_tne_offset = _mm512_add_epi32(i_tnw_offset, vElempacki);
                __m512i i_tsw_offset = _mm512_add_epi32(i_tnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_tse_offset = _mm512_add_epi32(i_tsw_offset, vElempacki);

                __m512i i_bnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m512i i_bne_offset = _mm512_add_epi32(i_bnw_offset, vElempacki);
                __m512i i_bsw_offset = _mm512_add_epi32(i_bnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_bse_offset = _mm512_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 tnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_tnw_offset, src.channel(q), sizeof(float));
                    __m512 tne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_tne_offset, src.channel(q), sizeof(float));
                    __m512 tsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_tsw_offset, src.channel(q), sizeof(float));
                    __m512 tse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v110_in_range, i_tse_offset, src.channel(q), sizeof(float));

                    __m512 bnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), z1_in_range, i_bnw_offset, src.channel(q), sizeof(float));
                    __m512 bne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v101_in_range, i_bne_offset, src.channel(q), sizeof(float));
                    __m512 bsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v011_in_range, i_bsw_offset, src.channel(q), sizeof(float));
                    __m512 bse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v111_in_range, i_bse_offset, src.channel(q), sizeof(float));

                    __m512 _v = _mm512_mul_ps(tnw_val, tnw);
                    _v = _mm512_fmadd_ps(tne_val, tne, _v);
                    _v = _mm512_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm512_fmadd_ps(tse_val, tse, _v);

                    _v = _mm512_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm512_fmadd_ps(bne_val, bne, _v);
                    _v = _mm512_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm512_fmadd_ps(bse_val, bse, _v);

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align1_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512 vImgDf = _mm512_set1_ps(src.d);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);
    const __m512i vImgDi = _mm512_set1_epi32(src.d);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m512 gx = _mm512_set1_ps(gridptr[0]);
                __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);
                __m512 gz = _mm512_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m512 two = _mm512_set1_ps(2.f);

                    // x
                    gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));

                    const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                    gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                    // y
                    gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));

                    const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                    // z
                    gz = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gz, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1));

                    const __m512 border_z = _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1);

                    gz = _mm512_min_ps(border_z, _mm512_max_ps(gz, _mm512_setzero_ps()));
                }

                __m512 x_w = _mm512_floor_ps(gx);
                __m512 y_n = _mm512_floor_ps(gy);
                __m512 z_t = _mm512_floor_ps(gz);

                __m512 w = _mm512_sub_ps(gx, x_w);
                __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
                __m512 n = _mm512_sub_ps(gy, y_n);
                __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);
                __m512 t = _mm512_sub_ps(gz, z_t);
                __m512 b = _mm512_sub_ps(*(__m512*)_ps512_1, t);

                __m512 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m512 nw = _mm512_mul_ps(s, e);
                    __m512 ne = _mm512_mul_ps(s, w);
                    __m512 sw = _mm512_mul_ps(n, e);
                    __m512 se = _mm512_mul_ps(n, w);

                    tnw = _mm512_mul_ps(b, nw);
                    tne = _mm512_mul_ps(b, ne);
                    tsw = _mm512_mul_ps(b, sw);
                    tse = _mm512_mul_ps(b, se);

                    bnw = _mm512_mul_ps(t, nw);
                    bne = _mm512_mul_ps(t, ne);
                    bsw = _mm512_mul_ps(t, sw);
                    bse = _mm512_mul_ps(t, se);
                }

                __m512i x0 = _mm512_cvtps_epi32(x_w);
                __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
                __m512i y0 = _mm512_cvtps_epi32(y_n);
                __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);
                __m512i z0 = _mm512_cvtps_epi32(z_t);
                __m512i z1 = _mm512_add_epi32(z0, *(__m512i*)_pi32_512_1);

                __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
                __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);
                __mmask16 z1_in_range = _mm512_cmpgt_epi32_mask(z1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z1);

                __mmask16 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __mmask16 v11_in_range = x1_in_range & y1_in_range;

                    v110_in_range = x1_in_range & y1_in_range;

                    v011_in_range = y1_in_range & z1_in_range;
                    v101_in_range = x1_in_range & z1_in_range;
                    v111_in_range = v11_in_range & z1_in_range;
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                __m512i i_tnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), z0), _mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                __m512i i_tne_offset = _mm512_add_epi32(i_tnw_offset, vElempacki);
                __m512i i_tsw_offset = _mm512_add_epi32(i_tnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_tse_offset = _mm512_add_epi32(i_tsw_offset, vElempacki);

                __m512i i_bnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m512i i_bne_offset = _mm512_add_epi32(i_bnw_offset, vElempacki);
                __m512i i_bsw_offset = _mm512_add_epi32(i_bnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_bse_offset = _mm512_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 tnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_tnw_offset, src.channel(q), sizeof(float));
                    __m512 tne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_tne_offset, src.channel(q), sizeof(float));
                    __m512 tsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_tsw_offset, src.channel(q), sizeof(float));
                    __m512 tse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v110_in_range, i_tse_offset, src.channel(q), sizeof(float));

                    __m512 bnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), z1_in_range, i_bnw_offset, src.channel(q), sizeof(float));
                    __m512 bne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v101_in_range, i_bne_offset, src.channel(q), sizeof(float));
                    __m512 bsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v011_in_range, i_bsw_offset, src.channel(q), sizeof(float));
                    __m512 bse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v111_in_range, i_bse_offset, src.channel(q), sizeof(float));

                    __m512 _v = _mm512_mul_ps(tnw_val, tnw);
                    _v = _mm512_fmadd_ps(tne_val, tne, _v);
                    _v = _mm512_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm512_fmadd_ps(tse_val, tse, _v);

                    _v = _mm512_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm512_fmadd_ps(bne_val, bne, _v);
                    _v = _mm512_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm512_fmadd_ps(bse_val, bse, _v);

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align0_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512 vImgDf = _mm512_set1_ps(src.d);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);
    const __m512i vImgDi = _mm512_set1_epi32(src.d);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m512 gx = _mm512_set1_ps(gridptr[0]);
                __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);
                __m512 gz = _mm512_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m512 two = _mm512_set1_ps(2.f);

                    // x
                    gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);
                    const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                    __m512 v0p5fp16 = _mm512_set1_ps(0.5f);
                    gx = _mm512_add_ps(gx, v0p5fp16);

                    gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                    gx = _mm512_sub_ps(vImgWf, reflectx_v);

                    gx = _mm512_sub_ps(gx, v0p5fp16);

                    _mm512_sub_ps(gx, v0p5fp16);

                    gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                    // y
                    gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);
                    const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                    gy = _mm512_add_ps(gy, v0p5fp16);

                    gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, vImgHf), *(__m512*)_ps512_inv_sign_mask);
                    gy = _mm512_sub_ps(vImgHf, reflecty_v);

                    gy = _mm512_sub_ps(gy, v0p5fp16);

                    _mm512_sub_ps(gy, v0p5fp16);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                    // z
                    gz = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gz, *(__m512*)_ps512_1), vImgDf, *(__m512*)_ps512_1), two);
                    const __m512 border_z = _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1);

                    gz = _mm512_add_ps(gz, v0p5fp16);

                    gz = _mm512_and_ps(gz, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectz_v = _mm512_and_ps(_mm512_sub_ps(gz, vImgDf), *(__m512*)_ps512_inv_sign_mask);
                    gz = _mm512_sub_ps(vImgDf, reflectz_v);

                    gz = _mm512_sub_ps(gz, v0p5fp16);

                    _mm512_sub_ps(gz, v0p5fp16);

                    gz = _mm512_min_ps(border_z, _mm512_max_ps(gz, _mm512_setzero_ps()));
                }

                __m512 x_w = _mm512_floor_ps(gx);
                __m512 y_n = _mm512_floor_ps(gy);
                __m512 z_t = _mm512_floor_ps(gz);

                __m512 w = _mm512_sub_ps(gx, x_w);
                __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
                __m512 n = _mm512_sub_ps(gy, y_n);
                __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);
                __m512 t = _mm512_sub_ps(gz, z_t);
                __m512 b = _mm512_sub_ps(*(__m512*)_ps512_1, t);

                __m512 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m512 nw = _mm512_mul_ps(s, e);
                    __m512 ne = _mm512_mul_ps(s, w);
                    __m512 sw = _mm512_mul_ps(n, e);
                    __m512 se = _mm512_mul_ps(n, w);

                    tnw = _mm512_mul_ps(b, nw);
                    tne = _mm512_mul_ps(b, ne);
                    tsw = _mm512_mul_ps(b, sw);
                    tse = _mm512_mul_ps(b, se);

                    bnw = _mm512_mul_ps(t, nw);
                    bne = _mm512_mul_ps(t, ne);
                    bsw = _mm512_mul_ps(t, sw);
                    bse = _mm512_mul_ps(t, se);
                }

                __m512i x0 = _mm512_cvtps_epi32(x_w);
                __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
                __m512i y0 = _mm512_cvtps_epi32(y_n);
                __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);
                __m512i z0 = _mm512_cvtps_epi32(z_t);
                __m512i z1 = _mm512_add_epi32(z0, *(__m512i*)_pi32_512_1);

                __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
                __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);
                __mmask16 z1_in_range = _mm512_cmpgt_epi32_mask(z1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z1);

                __mmask16 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __mmask16 v11_in_range = x1_in_range & y1_in_range;

                    v110_in_range = x1_in_range & y1_in_range;

                    v011_in_range = y1_in_range & z1_in_range;
                    v101_in_range = x1_in_range & z1_in_range;
                    v111_in_range = v11_in_range & z1_in_range;
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                __m512i i_tnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), z0), _mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                __m512i i_tne_offset = _mm512_add_epi32(i_tnw_offset, vElempacki);
                __m512i i_tsw_offset = _mm512_add_epi32(i_tnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_tse_offset = _mm512_add_epi32(i_tsw_offset, vElempacki);

                __m512i i_bnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m512i i_bne_offset = _mm512_add_epi32(i_bnw_offset, vElempacki);
                __m512i i_bsw_offset = _mm512_add_epi32(i_bnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_bse_offset = _mm512_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 tnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_tnw_offset, src.channel(q), sizeof(float));
                    __m512 tne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_tne_offset, src.channel(q), sizeof(float));
                    __m512 tsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_tsw_offset, src.channel(q), sizeof(float));
                    __m512 tse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v110_in_range, i_tse_offset, src.channel(q), sizeof(float));

                    __m512 bnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), z1_in_range, i_bnw_offset, src.channel(q), sizeof(float));
                    __m512 bne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v101_in_range, i_bne_offset, src.channel(q), sizeof(float));
                    __m512 bsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v011_in_range, i_bsw_offset, src.channel(q), sizeof(float));
                    __m512 bse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v111_in_range, i_bse_offset, src.channel(q), sizeof(float));

                    __m512 _v = _mm512_mul_ps(tnw_val, tnw);
                    _v = _mm512_fmadd_ps(tne_val, tne, _v);
                    _v = _mm512_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm512_fmadd_ps(tse_val, tse, _v);

                    _v = _mm512_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm512_fmadd_ps(bne_val, bne, _v);
                    _v = _mm512_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm512_fmadd_ps(bse_val, bse, _v);

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align1_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512 vImgDf = _mm512_set1_ps(src.d);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);
    const __m512i vImgDi = _mm512_set1_epi32(src.d);

    const __m512i vElempacki = _mm512_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m512 gx = _mm512_set1_ps(gridptr[0]);
                __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);
                __m512 gz = _mm512_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m512 two = _mm512_set1_ps(2.f);

                    // x
                    gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));
                    const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                    gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, border_x), *(__m512*)_ps512_inv_sign_mask);
                    gx = _mm512_sub_ps(border_x, reflectx_v);

                    // y
                    gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));
                    const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                    gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, border_y), *(__m512*)_ps512_inv_sign_mask);
                    gy = _mm512_sub_ps(border_y, reflecty_v);

                    // z
                    gz = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gz, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1));
                    const __m512 border_z = _mm512_sub_ps(vImgDf, *(__m512*)_ps512_1);

                    gz = _mm512_and_ps(gz, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectz_v = _mm512_and_ps(_mm512_sub_ps(gz, border_z), *(__m512*)_ps512_inv_sign_mask);
                    gz = _mm512_sub_ps(border_z, reflectz_v);
                }

                __m512 x_w = _mm512_floor_ps(gx);
                __m512 y_n = _mm512_floor_ps(gy);
                __m512 z_t = _mm512_floor_ps(gz);

                __m512 w = _mm512_sub_ps(gx, x_w);
                __m512 e = _mm512_sub_ps(*(__m512*)_ps512_1, w);
                __m512 n = _mm512_sub_ps(gy, y_n);
                __m512 s = _mm512_sub_ps(*(__m512*)_ps512_1, n);
                __m512 t = _mm512_sub_ps(gz, z_t);
                __m512 b = _mm512_sub_ps(*(__m512*)_ps512_1, t);

                __m512 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m512 nw = _mm512_mul_ps(s, e);
                    __m512 ne = _mm512_mul_ps(s, w);
                    __m512 sw = _mm512_mul_ps(n, e);
                    __m512 se = _mm512_mul_ps(n, w);

                    tnw = _mm512_mul_ps(b, nw);
                    tne = _mm512_mul_ps(b, ne);
                    tsw = _mm512_mul_ps(b, sw);
                    tse = _mm512_mul_ps(b, se);

                    bnw = _mm512_mul_ps(t, nw);
                    bne = _mm512_mul_ps(t, ne);
                    bsw = _mm512_mul_ps(t, sw);
                    bse = _mm512_mul_ps(t, se);
                }

                __m512i x0 = _mm512_cvtps_epi32(x_w);
                __m512i x1 = _mm512_add_epi32(x0, *(__m512i*)_pi32_512_1);
                __m512i y0 = _mm512_cvtps_epi32(y_n);
                __m512i y1 = _mm512_add_epi32(y0, *(__m512i*)_pi32_512_1);
                __m512i z0 = _mm512_cvtps_epi32(z_t);
                __m512i z1 = _mm512_add_epi32(z0, *(__m512i*)_pi32_512_1);

                __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
                __mmask16 y1_in_range = _mm512_cmpgt_epi32_mask(y1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y1);
                __mmask16 z1_in_range = _mm512_cmpgt_epi32_mask(z1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgDi, z1);

                __mmask16 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __mmask16 v11_in_range = x1_in_range & y1_in_range;

                    v110_in_range = x1_in_range & y1_in_range;

                    v011_in_range = y1_in_range & z1_in_range;
                    v101_in_range = x1_in_range & z1_in_range;
                    v111_in_range = v11_in_range & z1_in_range;
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                __m512i i_tnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), z0), _mm512_add_epi32(_mm512_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                __m512i i_tne_offset = _mm512_add_epi32(i_tnw_offset, vElempacki);
                __m512i i_tsw_offset = _mm512_add_epi32(i_tnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_tse_offset = _mm512_add_epi32(i_tsw_offset, vElempacki);

                __m512i i_bnw_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m512i i_bne_offset = _mm512_add_epi32(i_bnw_offset, vElempacki);
                __m512i i_bsw_offset = _mm512_add_epi32(i_bnw_offset, _mm512_mullo_epi32(vImgWi, vElempacki));
                __m512i i_bse_offset = _mm512_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 tnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0b1111111111111111, i_tnw_offset, src.channel(q), sizeof(float));
                    __m512 tne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), x1_in_range, i_tne_offset, src.channel(q), sizeof(float));
                    __m512 tsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), y1_in_range, i_tsw_offset, src.channel(q), sizeof(float));
                    __m512 tse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v110_in_range, i_tse_offset, src.channel(q), sizeof(float));

                    __m512 bnw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), z1_in_range, i_bnw_offset, src.channel(q), sizeof(float));
                    __m512 bne_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v101_in_range, i_bne_offset, src.channel(q), sizeof(float));
                    __m512 bsw_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v011_in_range, i_bsw_offset, src.channel(q), sizeof(float));
                    __m512 bse_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v111_in_range, i_bse_offset, src.channel(q), sizeof(float));

                    __m512 _v = _mm512_mul_ps(tnw_val, tnw);
                    _v = _mm512_fmadd_ps(tne_val, tne, _v);
                    _v = _mm512_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm512_fmadd_ps(tse_val, tse, _v);

                    _v = _mm512_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm512_fmadd_ps(bne_val, bne, _v);
                    _v = _mm512_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm512_fmadd_ps(bse_val, bse, _v);

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}