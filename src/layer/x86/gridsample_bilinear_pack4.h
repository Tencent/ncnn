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

static void gridsample_2d_bilinear_align0_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m128 two = _mm_set1_ps(2.f);

                // x
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                // y
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);
            }

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128 x_w = _mm_floor_ps(gx);
            __m128 y_n = _mm_floor_ps(gy);
#else
            __m128 x_w = floor_ps(gx);
            __m128 y_n = floor_ps(gy);
#endif // __SSE4_1__

            __m128 w = _mm_sub_ps(gx, x_w);
            __m128 e = _mm_sub_ps(v1fp4, w);
            __m128 n = _mm_sub_ps(gy, y_n);
            __m128 s = _mm_sub_ps(v1fp4, n);

            __m128 nw = _mm_mul_ps(s, e);
            __m128 ne = _mm_mul_ps(s, w);
            __m128 sw = _mm_mul_ps(n, e);
            __m128 se = _mm_mul_ps(n, w);

            __m128i x0 = _mm_cvtps_epi32(x_w);
            __m128i x1 = _mm_add_epi32(x0, v1ip4);
            __m128i y0 = _mm_cvtps_epi32(y_n);
            __m128i y1 = _mm_add_epi32(y0, v1ip4);

            __m128i x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
            __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            __m128i v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
            __m128i v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
            __m128i v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
            __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                _mm_set_epi32(3, 2, 1, 0));
            __m128i i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            __m128i i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            __m128i i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);
#else
            __m128 nw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w), vElempackf),
                                          _mm_set_ps(3, 2, 1, 0));
            __m128 ne_offset = _mm_add_ps(nw_offset, vElempackf);
            __m128 sw_offset = _mm_add_ps(nw_offset, _mm_mul_ps(vImgWf, vElempackf));
            __m128 se_offset = _mm_add_ps(sw_offset, vElempackf);

            __m128i i_nw_offset = _mm_cvtps_epi32(nw_offset);
            __m128i i_ne_offset = _mm_cvtps_epi32(ne_offset);
            __m128i i_sw_offset = _mm_cvtps_epi32(sw_offset);
            __m128i i_se_offset = _mm_cvtps_epi32(se_offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 nw_val = mask_gather_ps(src.channel(q), i_nw_offset, _mm_castsi128_ps(v00_in_range));
                __m128 ne_val = mask_gather_ps(src.channel(q), i_ne_offset, _mm_castsi128_ps(v10_in_range));
                __m128 sw_val = mask_gather_ps(src.channel(q), i_sw_offset, _mm_castsi128_ps(v01_in_range));
                __m128 se_val = mask_gather_ps(src.channel(q), i_se_offset, _mm_castsi128_ps(v11_in_range));

                __m128 _v = _mm_mul_ps(nw_val, nw);
                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align1_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m128 two = _mm_set1_ps(2.f);

                // x
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                // y
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));
            }

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128 x_w = _mm_floor_ps(gx);
            __m128 y_n = _mm_floor_ps(gy);
#else
            __m128 x_w = floor_ps(gx);
            __m128 y_n = floor_ps(gy);
#endif // __SSE4_1__

            __m128 w = _mm_sub_ps(gx, x_w);
            __m128 e = _mm_sub_ps(v1fp4, w);
            __m128 n = _mm_sub_ps(gy, y_n);
            __m128 s = _mm_sub_ps(v1fp4, n);

            __m128 nw = _mm_mul_ps(s, e);
            __m128 ne = _mm_mul_ps(s, w);
            __m128 sw = _mm_mul_ps(n, e);
            __m128 se = _mm_mul_ps(n, w);

            __m128i x0 = _mm_cvtps_epi32(x_w);
            __m128i x1 = _mm_add_epi32(x0, v1ip4);
            __m128i y0 = _mm_cvtps_epi32(y_n);
            __m128i y1 = _mm_add_epi32(y0, v1ip4);

            __m128i x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
            __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            __m128i v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
            __m128i v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
            __m128i v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
            __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                _mm_set_epi32(3, 2, 1, 0));
            __m128i i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            __m128i i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            __m128i i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);
#else
            __m128 nw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w), vElempackf),
                                          _mm_set_ps(3, 2, 1, 0));
            __m128 ne_offset = _mm_add_ps(nw_offset, vElempackf);
            __m128 sw_offset = _mm_add_ps(nw_offset, _mm_mul_ps(vImgWf, vElempackf));
            __m128 se_offset = _mm_add_ps(sw_offset, vElempackf);

            __m128i i_nw_offset = _mm_cvtps_epi32(nw_offset);
            __m128i i_ne_offset = _mm_cvtps_epi32(ne_offset);
            __m128i i_sw_offset = _mm_cvtps_epi32(sw_offset);
            __m128i i_se_offset = _mm_cvtps_epi32(se_offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 nw_val = mask_gather_ps(src.channel(q), i_nw_offset, _mm_castsi128_ps(v00_in_range));
                __m128 ne_val = mask_gather_ps(src.channel(q), i_ne_offset, _mm_castsi128_ps(v10_in_range));
                __m128 sw_val = mask_gather_ps(src.channel(q), i_sw_offset, _mm_castsi128_ps(v01_in_range));
                __m128 se_val = mask_gather_ps(src.channel(q), i_se_offset, _mm_castsi128_ps(v11_in_range));

                __m128 _v = _mm_mul_ps(nw_val, nw);
                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align0_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m128 two = _mm_set1_ps(2.f);

                // x
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                // y
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128 x_w = _mm_floor_ps(gx);
            __m128 y_n = _mm_floor_ps(gy);
#else
            __m128 x_w = floor_ps(gx);
            __m128 y_n = floor_ps(gy);
#endif // __SSE4_1__

            __m128 w = _mm_sub_ps(gx, x_w);
            __m128 e = _mm_sub_ps(v1fp4, w);
            __m128 n = _mm_sub_ps(gy, y_n);
            __m128 s = _mm_sub_ps(v1fp4, n);

            __m128 nw = _mm_mul_ps(s, e);
            __m128 ne = _mm_mul_ps(s, w);
            __m128 sw = _mm_mul_ps(n, e);
            __m128 se = _mm_mul_ps(n, w);

            __m128i x0 = _mm_cvtps_epi32(x_w);
            __m128i x1 = _mm_add_epi32(x0, v1ip4);
            __m128i y0 = _mm_cvtps_epi32(y_n);
            __m128i y1 = _mm_add_epi32(y0, v1ip4);

            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                _mm_set_epi32(3, 2, 1, 0));
            __m128i i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            __m128i i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            __m128i i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);
#else
            __m128 nw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w), vElempackf),
                                          _mm_set_ps(3, 2, 1, 0));
            __m128 ne_offset = _mm_add_ps(nw_offset, vElempackf);
            __m128 sw_offset = _mm_add_ps(nw_offset, _mm_mul_ps(vImgWf, vElempackf));
            __m128 se_offset = _mm_add_ps(sw_offset, vElempackf);

            __m128i i_nw_offset = _mm_cvtps_epi32(nw_offset);
            __m128i i_ne_offset = _mm_cvtps_epi32(ne_offset);
            __m128i i_sw_offset = _mm_cvtps_epi32(sw_offset);
            __m128i i_se_offset = _mm_cvtps_epi32(se_offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                __m128 ne_val = mask_gather_ps(src.channel(q), i_ne_offset, _mm_castsi128_ps(x1_in_range));
                __m128 sw_val = mask_gather_ps(src.channel(q), i_sw_offset, _mm_castsi128_ps(y1_in_range));
                __m128 se_val = mask_gather_ps(src.channel(q), i_se_offset, _mm_castsi128_ps(v11_in_range));

                __m128 _v = _mm_mul_ps(nw_val, nw);
                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align1_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m128 two = _mm_set1_ps(2.f);

                // x
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                // y
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128 x_w = _mm_floor_ps(gx);
            __m128 y_n = _mm_floor_ps(gy);
#else
            __m128 x_w = floor_ps(gx);
            __m128 y_n = floor_ps(gy);
#endif // __SSE4_1__

            __m128 w = _mm_sub_ps(gx, x_w);
            __m128 e = _mm_sub_ps(v1fp4, w);
            __m128 n = _mm_sub_ps(gy, y_n);
            __m128 s = _mm_sub_ps(v1fp4, n);

            __m128 nw = _mm_mul_ps(s, e);
            __m128 ne = _mm_mul_ps(s, w);
            __m128 sw = _mm_mul_ps(n, e);
            __m128 se = _mm_mul_ps(n, w);

            __m128i x0 = _mm_cvtps_epi32(x_w);
            __m128i x1 = _mm_add_epi32(x0, v1ip4);
            __m128i y0 = _mm_cvtps_epi32(y_n);
            __m128i y1 = _mm_add_epi32(y0, v1ip4);

            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                _mm_set_epi32(3, 2, 1, 0));
            __m128i i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            __m128i i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            __m128i i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);
#else
            __m128 nw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w), vElempackf),
                                          _mm_set_ps(3, 2, 1, 0));
            __m128 ne_offset = _mm_add_ps(nw_offset, vElempackf);
            __m128 sw_offset = _mm_add_ps(nw_offset, _mm_mul_ps(vImgWf, vElempackf));
            __m128 se_offset = _mm_add_ps(sw_offset, vElempackf);

            __m128i i_nw_offset = _mm_cvtps_epi32(nw_offset);
            __m128i i_ne_offset = _mm_cvtps_epi32(ne_offset);
            __m128i i_sw_offset = _mm_cvtps_epi32(sw_offset);
            __m128i i_se_offset = _mm_cvtps_epi32(se_offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                __m128 ne_val = mask_gather_ps(src.channel(q), i_ne_offset, _mm_castsi128_ps(x1_in_range));
                __m128 sw_val = mask_gather_ps(src.channel(q), i_sw_offset, _mm_castsi128_ps(y1_in_range));
                __m128 se_val = mask_gather_ps(src.channel(q), i_se_offset, _mm_castsi128_ps(v11_in_range));

                __m128 _v = _mm_mul_ps(nw_val, nw);
                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align0_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m128 two = _mm_set1_ps(2.f);

                // x
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                __m128 v0p5fp4 = _mm_set1_ps(0.5f);
                gx = _mm_add_ps(gx, v0p5fp4);

                gx = _mm_and_ps(gx, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx_v = _mm_and_ps(_mm_sub_ps(gx, vImgWf), *(__m128*)_ps_inv_sign_mask);
                gx = _mm_sub_ps(vImgWf, reflectx_v);

                gx = _mm_sub_ps(gx, v0p5fp4);

                _mm_sub_ps(gx, v0p5fp4);

                gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                // y
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_add_ps(gy, v0p5fp4);

                gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps_inv_sign_mask);
                gy = _mm_sub_ps(vImgHf, reflecty_v);

                gy = _mm_sub_ps(gy, v0p5fp4);

                _mm_sub_ps(gy, v0p5fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128 x_w = _mm_floor_ps(gx);
            __m128 y_n = _mm_floor_ps(gy);
#else
            __m128 x_w = floor_ps(gx);
            __m128 y_n = floor_ps(gy);
#endif // __SSE4_1__

            __m128 w = _mm_sub_ps(gx, x_w);
            __m128 e = _mm_sub_ps(v1fp4, w);
            __m128 n = _mm_sub_ps(gy, y_n);
            __m128 s = _mm_sub_ps(v1fp4, n);

            __m128 nw = _mm_mul_ps(s, e);
            __m128 ne = _mm_mul_ps(s, w);
            __m128 sw = _mm_mul_ps(n, e);
            __m128 se = _mm_mul_ps(n, w);

            __m128i x0 = _mm_cvtps_epi32(x_w);
            __m128i x1 = _mm_add_epi32(x0, v1ip4);
            __m128i y0 = _mm_cvtps_epi32(y_n);
            __m128i y1 = _mm_add_epi32(y0, v1ip4);

            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                _mm_set_epi32(3, 2, 1, 0));
            __m128i i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            __m128i i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            __m128i i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);
#else
            __m128 nw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w), vElempackf),
                                          _mm_set_ps(3, 2, 1, 0));
            __m128 ne_offset = _mm_add_ps(nw_offset, vElempackf);
            __m128 sw_offset = _mm_add_ps(nw_offset, _mm_mul_ps(vImgWf, vElempackf));
            __m128 se_offset = _mm_add_ps(sw_offset, vElempackf);

            __m128i i_nw_offset = _mm_cvtps_epi32(nw_offset);
            __m128i i_ne_offset = _mm_cvtps_epi32(ne_offset);
            __m128i i_sw_offset = _mm_cvtps_epi32(sw_offset);
            __m128i i_se_offset = _mm_cvtps_epi32(se_offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                __m128 ne_val = mask_gather_ps(src.channel(q), i_ne_offset, _mm_castsi128_ps(x1_in_range));
                __m128 sw_val = mask_gather_ps(src.channel(q), i_sw_offset, _mm_castsi128_ps(y1_in_range));
                __m128 se_val = mask_gather_ps(src.channel(q), i_se_offset, _mm_castsi128_ps(v11_in_range));

                __m128 _v = _mm_mul_ps(nw_val, nw);
                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bilinear_align1_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m128 two = _mm_set1_ps(2.f);

                // x
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_and_ps(gx, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx_v = _mm_and_ps(_mm_sub_ps(gx, border_x), *(__m128*)_ps_inv_sign_mask);
                gx = _mm_sub_ps(border_x, reflectx_v);

                // y
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps_inv_sign_mask);
                gy = _mm_sub_ps(border_y, reflecty_v);
            }

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128 x_w = _mm_floor_ps(gx);
            __m128 y_n = _mm_floor_ps(gy);
#else
            __m128 x_w = floor_ps(gx);
            __m128 y_n = floor_ps(gy);
#endif // __SSE4_1__

            __m128 w = _mm_sub_ps(gx, x_w);
            __m128 e = _mm_sub_ps(v1fp4, w);
            __m128 n = _mm_sub_ps(gy, y_n);
            __m128 s = _mm_sub_ps(v1fp4, n);

            __m128 nw = _mm_mul_ps(s, e);
            __m128 ne = _mm_mul_ps(s, w);
            __m128 sw = _mm_mul_ps(n, e);
            __m128 se = _mm_mul_ps(n, w);

            __m128i x0 = _mm_cvtps_epi32(x_w);
            __m128i x1 = _mm_add_epi32(x0, v1ip4);
            __m128i y0 = _mm_cvtps_epi32(y_n);
            __m128i y1 = _mm_add_epi32(y0, v1ip4);

            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                                _mm_set_epi32(3, 2, 1, 0));
            __m128i i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            __m128i i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            __m128i i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);
#else
            __m128 nw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w), vElempackf),
                                          _mm_set_ps(3, 2, 1, 0));
            __m128 ne_offset = _mm_add_ps(nw_offset, vElempackf);
            __m128 sw_offset = _mm_add_ps(nw_offset, _mm_mul_ps(vImgWf, vElempackf));
            __m128 se_offset = _mm_add_ps(sw_offset, vElempackf);

            __m128i i_nw_offset = _mm_cvtps_epi32(nw_offset);
            __m128i i_ne_offset = _mm_cvtps_epi32(ne_offset);
            __m128i i_sw_offset = _mm_cvtps_epi32(sw_offset);
            __m128i i_se_offset = _mm_cvtps_epi32(se_offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                __m128 ne_val = mask_gather_ps(src.channel(q), i_ne_offset, _mm_castsi128_ps(x1_in_range));
                __m128 sw_val = mask_gather_ps(src.channel(q), i_sw_offset, _mm_castsi128_ps(y1_in_range));
                __m128 se_val = mask_gather_ps(src.channel(q), i_se_offset, _mm_castsi128_ps(v11_in_range));

                __m128 _v = _mm_mul_ps(nw_val, nw);
                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_3d_bilinear_align0_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m128 gx = _mm_set1_ps(gridptr[0]);
                __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);
                __m128 gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m128 two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                    // y
                    gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                    // z
                    gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);
                }

#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128 x_w = _mm_floor_ps(gx);
                __m128 y_n = _mm_floor_ps(gy);
                __m128 z_t = _mm_floor_ps(gz);
#else
                __m128 x_w = floor_ps(gx);
                __m128 y_n = floor_ps(gy);
                __m128 z_t = floor_ps(gz);
#endif // __SSE4_1__

                __m128 w = _mm_sub_ps(gx, x_w);
                __m128 e = _mm_sub_ps(v1fp4, w);
                __m128 n = _mm_sub_ps(gy, y_n);
                __m128 s = _mm_sub_ps(v1fp4, n);
                __m128 t = _mm_sub_ps(gz, z_t);
                __m128 b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m128 nw = _mm_mul_ps(s, e);
                    __m128 ne = _mm_mul_ps(s, w);
                    __m128 sw = _mm_mul_ps(n, e);
                    __m128 se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                __m128i x0 = _mm_cvtps_epi32(x_w);
                __m128i x1 = _mm_add_epi32(x0, v1ip4);
                __m128i y0 = _mm_cvtps_epi32(y_n);
                __m128i y1 = _mm_add_epi32(y0, v1ip4);
                __m128i z0 = _mm_cvtps_epi32(z_t);
                __m128i z1 = _mm_add_epi32(z0, v1ip4);

                __m128i x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
                __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                __m128i y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
                __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                __m128i z0_in_range = _mm_and_si128(_mm_cmpgt_epi32(z0, vn1ip4), _mm_cmpgt_epi32(vImgDi, z0));
                __m128i z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m128i v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
                    __m128i v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
                    __m128i v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
                    __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v000_in_range = _mm_and_si128(v00_in_range, z0_in_range);
                    v010_in_range = _mm_and_si128(v01_in_range, z0_in_range);
                    v100_in_range = _mm_and_si128(v10_in_range, z0_in_range);
                    v110_in_range = _mm_and_si128(v11_in_range, z0_in_range);

                    v001_in_range = _mm_and_si128(v00_in_range, z1_in_range);
                    v011_in_range = _mm_and_si128(v01_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(v10_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128i i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                __m128i i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                __m128i i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                __m128i i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m128i i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                __m128i i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);
#else
                __m128 tnw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), z_t), _mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w)), vElempackf), _mm_set_ps(3, 2, 1, 0));
                __m128 tne_offset = _mm_add_ps(tnw_offset, vElempackf);
                __m128 tsw_offset = _mm_add_ps(tnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 tse_offset = _mm_add_ps(tsw_offset, vElempackf);

                __m128 bnw_offset = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), vElempackf), tnw_offset);
                __m128 bne_offset = _mm_add_ps(bnw_offset, vElempackf);
                __m128 bsw_offset = _mm_add_ps(bnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 bse_offset = _mm_add_ps(bsw_offset, vElempackf);

                __m128i i_tnw_offset = _mm_cvtps_epi32(tnw_offset);
                __m128i i_tne_offset = _mm_cvtps_epi32(tne_offset);
                __m128i i_tsw_offset = _mm_cvtps_epi32(tsw_offset);
                __m128i i_tse_offset = _mm_cvtps_epi32(tse_offset);

                __m128i i_bnw_offset = _mm_cvtps_epi32(bnw_offset);
                __m128i i_bne_offset = _mm_cvtps_epi32(bne_offset);
                __m128i i_bsw_offset = _mm_cvtps_epi32(bsw_offset);
                __m128i i_bse_offset = _mm_cvtps_epi32(bse_offset);
#endif // __SSE4_1__

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, _mm_castsi128_ps(v000_in_range));
                    __m128 tne_val = mask_gather_ps(src.channel(q), i_tne_offset, _mm_castsi128_ps(v100_in_range));
                    __m128 tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, _mm_castsi128_ps(v010_in_range));
                    __m128 tse_val = mask_gather_ps(src.channel(q), i_tse_offset, _mm_castsi128_ps(v110_in_range));

                    __m128 bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, _mm_castsi128_ps(v001_in_range));
                    __m128 bne_val = mask_gather_ps(src.channel(q), i_bne_offset, _mm_castsi128_ps(v101_in_range));
                    __m128 bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, _mm_castsi128_ps(v011_in_range));
                    __m128 bse_val = mask_gather_ps(src.channel(q), i_bse_offset, _mm_castsi128_ps(v111_in_range));

                    __m128 _v = _mm_mul_ps(tnw_val, tnw);
                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align1_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m128 gx = _mm_set1_ps(gridptr[0]);
                __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);
                __m128 gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m128 two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                    // y
                    gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                    // z
                    gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));
                }

#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128 x_w = _mm_floor_ps(gx);
                __m128 y_n = _mm_floor_ps(gy);
                __m128 z_t = _mm_floor_ps(gz);
#else
                __m128 x_w = floor_ps(gx);
                __m128 y_n = floor_ps(gy);
                __m128 z_t = floor_ps(gz);
#endif // __SSE4_1__

                __m128 w = _mm_sub_ps(gx, x_w);
                __m128 e = _mm_sub_ps(v1fp4, w);
                __m128 n = _mm_sub_ps(gy, y_n);
                __m128 s = _mm_sub_ps(v1fp4, n);
                __m128 t = _mm_sub_ps(gz, z_t);
                __m128 b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m128 nw = _mm_mul_ps(s, e);
                    __m128 ne = _mm_mul_ps(s, w);
                    __m128 sw = _mm_mul_ps(n, e);
                    __m128 se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                __m128i x0 = _mm_cvtps_epi32(x_w);
                __m128i x1 = _mm_add_epi32(x0, v1ip4);
                __m128i y0 = _mm_cvtps_epi32(y_n);
                __m128i y1 = _mm_add_epi32(y0, v1ip4);
                __m128i z0 = _mm_cvtps_epi32(z_t);
                __m128i z1 = _mm_add_epi32(z0, v1ip4);

                __m128i x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
                __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                __m128i y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
                __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                __m128i z0_in_range = _mm_and_si128(_mm_cmpgt_epi32(z0, vn1ip4), _mm_cmpgt_epi32(vImgDi, z0));
                __m128i z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m128i v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
                    __m128i v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
                    __m128i v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
                    __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v000_in_range = _mm_and_si128(v00_in_range, z0_in_range);
                    v010_in_range = _mm_and_si128(v01_in_range, z0_in_range);
                    v100_in_range = _mm_and_si128(v10_in_range, z0_in_range);
                    v110_in_range = _mm_and_si128(v11_in_range, z0_in_range);

                    v001_in_range = _mm_and_si128(v00_in_range, z1_in_range);
                    v011_in_range = _mm_and_si128(v01_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(v10_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128i i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                __m128i i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                __m128i i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                __m128i i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m128i i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                __m128i i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);
#else
                __m128 tnw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), z_t), _mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w)), vElempackf), _mm_set_ps(3, 2, 1, 0));
                __m128 tne_offset = _mm_add_ps(tnw_offset, vElempackf);
                __m128 tsw_offset = _mm_add_ps(tnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 tse_offset = _mm_add_ps(tsw_offset, vElempackf);

                __m128 bnw_offset = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), vElempackf), tnw_offset);
                __m128 bne_offset = _mm_add_ps(bnw_offset, vElempackf);
                __m128 bsw_offset = _mm_add_ps(bnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 bse_offset = _mm_add_ps(bsw_offset, vElempackf);

                __m128i i_tnw_offset = _mm_cvtps_epi32(tnw_offset);
                __m128i i_tne_offset = _mm_cvtps_epi32(tne_offset);
                __m128i i_tsw_offset = _mm_cvtps_epi32(tsw_offset);
                __m128i i_tse_offset = _mm_cvtps_epi32(tse_offset);

                __m128i i_bnw_offset = _mm_cvtps_epi32(bnw_offset);
                __m128i i_bne_offset = _mm_cvtps_epi32(bne_offset);
                __m128i i_bsw_offset = _mm_cvtps_epi32(bsw_offset);
                __m128i i_bse_offset = _mm_cvtps_epi32(bse_offset);
#endif // __SSE4_1__

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, _mm_castsi128_ps(v000_in_range));
                    __m128 tne_val = mask_gather_ps(src.channel(q), i_tne_offset, _mm_castsi128_ps(v100_in_range));
                    __m128 tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, _mm_castsi128_ps(v010_in_range));
                    __m128 tse_val = mask_gather_ps(src.channel(q), i_tse_offset, _mm_castsi128_ps(v110_in_range));

                    __m128 bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, _mm_castsi128_ps(v001_in_range));
                    __m128 bne_val = mask_gather_ps(src.channel(q), i_bne_offset, _mm_castsi128_ps(v101_in_range));
                    __m128 bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, _mm_castsi128_ps(v011_in_range));
                    __m128 bse_val = mask_gather_ps(src.channel(q), i_bse_offset, _mm_castsi128_ps(v111_in_range));

                    __m128 _v = _mm_mul_ps(tnw_val, tnw);
                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align0_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m128 gx = _mm_set1_ps(gridptr[0]);
                __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);
                __m128 gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m128 two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                    const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                    // y
                    gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                    const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);

                    const __m128 border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                __m128 x_w = floor_ps(gx);
                __m128 y_n = floor_ps(gy);
                __m128 z_t = floor_ps(gz);

                __m128 w = _mm_sub_ps(gx, x_w);
                __m128 e = _mm_sub_ps(v1fp4, w);
                __m128 n = _mm_sub_ps(gy, y_n);
                __m128 s = _mm_sub_ps(v1fp4, n);
                __m128 t = _mm_sub_ps(gz, z_t);
                __m128 b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m128 nw = _mm_mul_ps(s, e);
                    __m128 ne = _mm_mul_ps(s, w);
                    __m128 sw = _mm_mul_ps(n, e);
                    __m128 se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                __m128i x0 = _mm_cvtps_epi32(x_w);
                __m128i x1 = _mm_add_epi32(x0, v1ip4);
                __m128i y0 = _mm_cvtps_epi32(y_n);
                __m128i y1 = _mm_add_epi32(y0, v1ip4);
                __m128i z0 = _mm_cvtps_epi32(z_t);
                __m128i z1 = _mm_add_epi32(z0, v1ip4);

                __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                __m128i z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128i i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                __m128i i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                __m128i i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                __m128i i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m128i i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                __m128i i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);
#else
                __m128 tnw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), z_t), _mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w)), vElempackf), _mm_set_ps(3, 2, 1, 0));
                __m128 tne_offset = _mm_add_ps(tnw_offset, vElempackf);
                __m128 tsw_offset = _mm_add_ps(tnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 tse_offset = _mm_add_ps(tsw_offset, vElempackf);

                __m128 bnw_offset = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), vElempackf), tnw_offset);
                __m128 bne_offset = _mm_add_ps(bnw_offset, vElempackf);
                __m128 bsw_offset = _mm_add_ps(bnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 bse_offset = _mm_add_ps(bsw_offset, vElempackf);

                __m128i i_tnw_offset = _mm_cvtps_epi32(tnw_offset);
                __m128i i_tne_offset = _mm_cvtps_epi32(tne_offset);
                __m128i i_tsw_offset = _mm_cvtps_epi32(tsw_offset);
                __m128i i_tse_offset = _mm_cvtps_epi32(tse_offset);

                __m128i i_bnw_offset = _mm_cvtps_epi32(bnw_offset);
                __m128i i_bne_offset = _mm_cvtps_epi32(bne_offset);
                __m128i i_bsw_offset = _mm_cvtps_epi32(bsw_offset);
                __m128i i_bse_offset = _mm_cvtps_epi32(bse_offset);
#endif // __SSE4_1__

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    __m128 tne_val = mask_gather_ps(src.channel(q), i_tne_offset, _mm_castsi128_ps(x1_in_range));
                    __m128 tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, _mm_castsi128_ps(y1_in_range));
                    __m128 tse_val = mask_gather_ps(src.channel(q), i_tse_offset, _mm_castsi128_ps(v110_in_range));

                    __m128 bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, _mm_castsi128_ps(z1_in_range));
                    __m128 bne_val = mask_gather_ps(src.channel(q), i_bne_offset, _mm_castsi128_ps(v101_in_range));
                    __m128 bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, _mm_castsi128_ps(v011_in_range));
                    __m128 bse_val = mask_gather_ps(src.channel(q), i_bse_offset, _mm_castsi128_ps(v111_in_range));

                    __m128 _v = _mm_mul_ps(tnw_val, tnw);
                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align1_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m128 gx = _mm_set1_ps(gridptr[0]);
                __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);
                __m128 gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m128 two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                    const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                    // y
                    gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                    const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));

                    const __m128 border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                __m128 x_w = floor_ps(gx);
                __m128 y_n = floor_ps(gy);
                __m128 z_t = floor_ps(gz);

                __m128 w = _mm_sub_ps(gx, x_w);
                __m128 e = _mm_sub_ps(v1fp4, w);
                __m128 n = _mm_sub_ps(gy, y_n);
                __m128 s = _mm_sub_ps(v1fp4, n);
                __m128 t = _mm_sub_ps(gz, z_t);
                __m128 b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m128 nw = _mm_mul_ps(s, e);
                    __m128 ne = _mm_mul_ps(s, w);
                    __m128 sw = _mm_mul_ps(n, e);
                    __m128 se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                __m128i x0 = _mm_cvtps_epi32(x_w);
                __m128i x1 = _mm_add_epi32(x0, v1ip4);
                __m128i y0 = _mm_cvtps_epi32(y_n);
                __m128i y1 = _mm_add_epi32(y0, v1ip4);
                __m128i z0 = _mm_cvtps_epi32(z_t);
                __m128i z1 = _mm_add_epi32(z0, v1ip4);

                __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                __m128i z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128i i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                __m128i i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                __m128i i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                __m128i i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m128i i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                __m128i i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);
#else
                __m128 tnw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), z_t), _mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w)), vElempackf), _mm_set_ps(3, 2, 1, 0));
                __m128 tne_offset = _mm_add_ps(tnw_offset, vElempackf);
                __m128 tsw_offset = _mm_add_ps(tnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 tse_offset = _mm_add_ps(tsw_offset, vElempackf);

                __m128 bnw_offset = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), vElempackf), tnw_offset);
                __m128 bne_offset = _mm_add_ps(bnw_offset, vElempackf);
                __m128 bsw_offset = _mm_add_ps(bnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 bse_offset = _mm_add_ps(bsw_offset, vElempackf);

                __m128i i_tnw_offset = _mm_cvtps_epi32(tnw_offset);
                __m128i i_tne_offset = _mm_cvtps_epi32(tne_offset);
                __m128i i_tsw_offset = _mm_cvtps_epi32(tsw_offset);
                __m128i i_tse_offset = _mm_cvtps_epi32(tse_offset);

                __m128i i_bnw_offset = _mm_cvtps_epi32(bnw_offset);
                __m128i i_bne_offset = _mm_cvtps_epi32(bne_offset);
                __m128i i_bsw_offset = _mm_cvtps_epi32(bsw_offset);
                __m128i i_bse_offset = _mm_cvtps_epi32(bse_offset);
#endif // __SSE4_1__

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    __m128 tne_val = mask_gather_ps(src.channel(q), i_tne_offset, _mm_castsi128_ps(x1_in_range));
                    __m128 tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, _mm_castsi128_ps(y1_in_range));
                    __m128 tse_val = mask_gather_ps(src.channel(q), i_tse_offset, _mm_castsi128_ps(v110_in_range));

                    __m128 bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, _mm_castsi128_ps(z1_in_range));
                    __m128 bne_val = mask_gather_ps(src.channel(q), i_bne_offset, _mm_castsi128_ps(v101_in_range));
                    __m128 bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, _mm_castsi128_ps(v011_in_range));
                    __m128 bse_val = mask_gather_ps(src.channel(q), i_bse_offset, _mm_castsi128_ps(v111_in_range));

                    __m128 _v = _mm_mul_ps(tnw_val, tnw);
                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align0_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m128 gx = _mm_set1_ps(gridptr[0]);
                __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);
                __m128 gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m128 two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);
                    const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                    __m128 v0p5fp4 = _mm_set1_ps(0.5f);
                    gx = _mm_add_ps(gx, v0p5fp4);

                    gx = _mm_and_ps(gx, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectx_v = _mm_and_ps(_mm_sub_ps(gx, vImgWf), *(__m128*)_ps_inv_sign_mask);
                    gx = _mm_sub_ps(vImgWf, reflectx_v);

                    gx = _mm_sub_ps(gx, v0p5fp4);

                    _mm_sub_ps(gx, v0p5fp4);

                    gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                    // y
                    gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);
                    const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_add_ps(gy, v0p5fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps_inv_sign_mask);
                    gy = _mm_sub_ps(vImgHf, reflecty_v);

                    gy = _mm_sub_ps(gy, v0p5fp4);

                    _mm_sub_ps(gy, v0p5fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);
                    const __m128 border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_add_ps(gz, v0p5fp4);

                    gz = _mm_and_ps(gz, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectz_v = _mm_and_ps(_mm_sub_ps(gz, vImgDf), *(__m128*)_ps_inv_sign_mask);
                    gz = _mm_sub_ps(vImgDf, reflectz_v);

                    gz = _mm_sub_ps(gz, v0p5fp4);

                    _mm_sub_ps(gz, v0p5fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                __m128 x_w = floor_ps(gx);
                __m128 y_n = floor_ps(gy);
                __m128 z_t = floor_ps(gz);

                __m128 w = _mm_sub_ps(gx, x_w);
                __m128 e = _mm_sub_ps(v1fp4, w);
                __m128 n = _mm_sub_ps(gy, y_n);
                __m128 s = _mm_sub_ps(v1fp4, n);
                __m128 t = _mm_sub_ps(gz, z_t);
                __m128 b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m128 nw = _mm_mul_ps(s, e);
                    __m128 ne = _mm_mul_ps(s, w);
                    __m128 sw = _mm_mul_ps(n, e);
                    __m128 se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                __m128i x0 = _mm_cvtps_epi32(x_w);
                __m128i x1 = _mm_add_epi32(x0, v1ip4);
                __m128i y0 = _mm_cvtps_epi32(y_n);
                __m128i y1 = _mm_add_epi32(y0, v1ip4);
                __m128i z0 = _mm_cvtps_epi32(z_t);
                __m128i z1 = _mm_add_epi32(z0, v1ip4);

                __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                __m128i z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128i i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                __m128i i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                __m128i i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                __m128i i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m128i i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                __m128i i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);
#else
                __m128 tnw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), z_t), _mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w)), vElempackf), _mm_set_ps(3, 2, 1, 0));
                __m128 tne_offset = _mm_add_ps(tnw_offset, vElempackf);
                __m128 tsw_offset = _mm_add_ps(tnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 tse_offset = _mm_add_ps(tsw_offset, vElempackf);

                __m128 bnw_offset = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), vElempackf), tnw_offset);
                __m128 bne_offset = _mm_add_ps(bnw_offset, vElempackf);
                __m128 bsw_offset = _mm_add_ps(bnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 bse_offset = _mm_add_ps(bsw_offset, vElempackf);

                __m128i i_tnw_offset = _mm_cvtps_epi32(tnw_offset);
                __m128i i_tne_offset = _mm_cvtps_epi32(tne_offset);
                __m128i i_tsw_offset = _mm_cvtps_epi32(tsw_offset);
                __m128i i_tse_offset = _mm_cvtps_epi32(tse_offset);

                __m128i i_bnw_offset = _mm_cvtps_epi32(bnw_offset);
                __m128i i_bne_offset = _mm_cvtps_epi32(bne_offset);
                __m128i i_bsw_offset = _mm_cvtps_epi32(bsw_offset);
                __m128i i_bse_offset = _mm_cvtps_epi32(bse_offset);
#endif // __SSE4_1__

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    __m128 tne_val = mask_gather_ps(src.channel(q), i_tne_offset, _mm_castsi128_ps(x1_in_range));
                    __m128 tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, _mm_castsi128_ps(y1_in_range));
                    __m128 tse_val = mask_gather_ps(src.channel(q), i_tse_offset, _mm_castsi128_ps(v110_in_range));

                    __m128 bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, _mm_castsi128_ps(z1_in_range));
                    __m128 bne_val = mask_gather_ps(src.channel(q), i_bne_offset, _mm_castsi128_ps(v101_in_range));
                    __m128 bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, _mm_castsi128_ps(v011_in_range));
                    __m128 bse_val = mask_gather_ps(src.channel(q), i_bse_offset, _mm_castsi128_ps(v111_in_range));

                    __m128 _v = _mm_mul_ps(tnw_val, tnw);
                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_bilinear_align1_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
#if !((_MSC_VER && __AVX__) || __SSE4_1__)
    const __m128 vElempackf = _mm_set1_ps(src.elempack);
#endif // !__SSE4_1__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m128 gx = _mm_set1_ps(gridptr[0]);
                __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);
                __m128 gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m128 two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
                    const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_and_ps(gx, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectx_v = _mm_and_ps(_mm_sub_ps(gx, border_x), *(__m128*)_ps_inv_sign_mask);
                    gx = _mm_sub_ps(border_x, reflectx_v);

                    // y
                    gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));
                    const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps_inv_sign_mask);
                    gy = _mm_sub_ps(border_y, reflecty_v);

                    // z
                    gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));
                    const __m128 border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_and_ps(gz, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectz_v = _mm_and_ps(_mm_sub_ps(gz, border_z), *(__m128*)_ps_inv_sign_mask);
                    gz = _mm_sub_ps(border_z, reflectz_v);
                }

                __m128 x_w = floor_ps(gx);
                __m128 y_n = floor_ps(gy);
                __m128 z_t = floor_ps(gz);

                __m128 w = _mm_sub_ps(gx, x_w);
                __m128 e = _mm_sub_ps(v1fp4, w);
                __m128 n = _mm_sub_ps(gy, y_n);
                __m128 s = _mm_sub_ps(v1fp4, n);
                __m128 t = _mm_sub_ps(gz, z_t);
                __m128 b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    __m128 nw = _mm_mul_ps(s, e);
                    __m128 ne = _mm_mul_ps(s, w);
                    __m128 sw = _mm_mul_ps(n, e);
                    __m128 se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                __m128i x0 = _mm_cvtps_epi32(x_w);
                __m128i x1 = _mm_add_epi32(x0, v1ip4);
                __m128i y0 = _mm_cvtps_epi32(y_n);
                __m128i y1 = _mm_add_epi32(y0, v1ip4);
                __m128i z0 = _mm_cvtps_epi32(z_t);
                __m128i z1 = _mm_add_epi32(z0, v1ip4);

                __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                __m128i y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                __m128i z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m128i v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(4)
#if (_MSC_VER && __AVX__) || __SSE4_1__
                __m128i i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                __m128i i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                __m128i i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                __m128i i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m128i i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                __m128i i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                __m128i i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);
#else
                __m128 tnw_offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), z_t), _mm_add_ps(_mm_mul_ps(y_n, vImgWf), x_w)), vElempackf), _mm_set_ps(3, 2, 1, 0));
                __m128 tne_offset = _mm_add_ps(tnw_offset, vElempackf);
                __m128 tsw_offset = _mm_add_ps(tnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 tse_offset = _mm_add_ps(tsw_offset, vElempackf);

                __m128 bnw_offset = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), vElempackf), tnw_offset);
                __m128 bne_offset = _mm_add_ps(bnw_offset, vElempackf);
                __m128 bsw_offset = _mm_add_ps(bnw_offset, _mm_mul_ps(vImgWf, vElempackf));
                __m128 bse_offset = _mm_add_ps(bsw_offset, vElempackf);

                __m128i i_tnw_offset = _mm_cvtps_epi32(tnw_offset);
                __m128i i_tne_offset = _mm_cvtps_epi32(tne_offset);
                __m128i i_tsw_offset = _mm_cvtps_epi32(tsw_offset);
                __m128i i_tse_offset = _mm_cvtps_epi32(tse_offset);

                __m128i i_bnw_offset = _mm_cvtps_epi32(bnw_offset);
                __m128i i_bne_offset = _mm_cvtps_epi32(bne_offset);
                __m128i i_bsw_offset = _mm_cvtps_epi32(bsw_offset);
                __m128i i_bse_offset = _mm_cvtps_epi32(bse_offset);
#endif // __SSE4_1__

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    __m128 tne_val = mask_gather_ps(src.channel(q), i_tne_offset, _mm_castsi128_ps(x1_in_range));
                    __m128 tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, _mm_castsi128_ps(y1_in_range));
                    __m128 tse_val = mask_gather_ps(src.channel(q), i_tse_offset, _mm_castsi128_ps(v110_in_range));

                    __m128 bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, _mm_castsi128_ps(z1_in_range));
                    __m128 bne_val = mask_gather_ps(src.channel(q), i_bne_offset, _mm_castsi128_ps(v101_in_range));
                    __m128 bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, _mm_castsi128_ps(v011_in_range));
                    __m128 bse_val = mask_gather_ps(src.channel(q), i_bse_offset, _mm_castsi128_ps(v111_in_range));

                    __m128 _v = _mm_mul_ps(tnw_val, tnw);
                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}