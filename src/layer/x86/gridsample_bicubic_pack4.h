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

static NCNN_FORCEINLINE __m128 cubic_interp1d_p4(const __m128& x0_v, const __m128& x1_v, const __m128& x2_v, const __m128& x3_v, const __m128& tx)
{
    const __m128 A = _mm_set1_ps(-0.75f);

    const __m128 x0 = _mm_add_ps(tx, v1fp4);
    const __m128& x1 = tx;
    const __m128 x2 = _mm_sub_ps(v1fp4, tx);
    //const __m128 x3 = _mm_add_ps(x2, v1fp4);

    const __m128 coeffs0 = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(A, x0), _mm_mul_ps(_mm_set1_ps(5.0f), A)), x0), _mm_mul_ps(_mm_set1_ps(8.0f), A)), x0), _mm_mul_ps(_mm_set1_ps(4), A));
    const __m128 coeffs1 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set1_ps(2.0f)), x1), _mm_add_ps(A, _mm_set1_ps(3.0f))), x1), x1), v1fp4);
    const __m128 coeffs2 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set1_ps(2.0f)), x2), _mm_add_ps(A, _mm_set1_ps(3.0f))), x2), x2), v1fp4);
    const __m128 coeffs3 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(v1fp4, coeffs0), coeffs1), coeffs2);

    __m128 _v = _mm_mul_ps(coeffs0, x0_v);
    _v = _mm_comp_fmadd_ps(coeffs1, x1_v, _v);
    _v = _mm_comp_fmadd_ps(coeffs2, x2_v, _v);
    _v = _mm_comp_fmadd_ps(coeffs3, x3_v, _v);

    return _v;
}

static void gridsample_2d_bicubic_align0_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

            __m128 gx_floor = floor_ps(gx);
            __m128 gy_floor = floor_ps(gy);

            const __m128 tx = _mm_sub_ps(gx, gx_floor);
            const __m128 ty = _mm_sub_ps(gy, gy_floor);

            __m128 coefficients[4];

            __m128 gx0 = _mm_add_ps(gx_floor, vn1fp4);
            __m128 gx1 = gx_floor;
            __m128 gx2 = _mm_add_ps(gx_floor, v1fp4);
            __m128 gx3 = _mm_add_ps(gx_floor, _mm_set1_ps(2.0f));

            __m128i x0 = _mm_cvtps_epi32(gx0);
            __m128i x1 = _mm_cvtps_epi32(gx1);
            __m128i x2 = _mm_cvtps_epi32(gx2);
            __m128i x3 = _mm_cvtps_epi32(gx3);

            __m128i x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i x2_in_range = _mm_and_si128(_mm_cmpgt_epi32(x2, vn1ip4), _mm_cmpgt_epi32(vImgWi, x2));
            __m128i x3_in_range = _mm_and_si128(_mm_cmpgt_epi32(x3, vn1ip4), _mm_cmpgt_epi32(vImgWi, x3));

            __m128i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4],
                    v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i));

                __m128i y = _mm_cvtps_epi32(gy);

                __m128i y_in_range = _mm_and_si128(_mm_cmpgt_epi32(y, vn1ip4), _mm_cmpgt_epi32(vImgHi, y));

                v0_in_range[i] = _mm_and_si128(x0_in_range, y_in_range);
                v1_in_range[i] = _mm_and_si128(x1_in_range, y_in_range);
                v2_in_range[i] = _mm_and_si128(x2_in_range, y_in_range);
                v3_in_range[i] = _mm_and_si128(x3_in_range, y_in_range);

                __m128 v0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m128 x0_val = mask_gather_ps(src.channel(q), v0_offset[i], _mm_castsi128_ps(v0_in_range[i]));
                    __m128 x1_val = mask_gather_ps(src.channel(q), v1_offset[i], _mm_castsi128_ps(v1_in_range[i]));
                    __m128 x2_val = mask_gather_ps(src.channel(q), v2_offset[i], _mm_castsi128_ps(v2_in_range[i]));
                    __m128 x3_val = mask_gather_ps(src.channel(q), v3_offset[i], _mm_castsi128_ps(v3_in_range[i]));

                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m128 _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);

    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));
            }

            __m128 gx_floor = floor_ps(gx);
            __m128 gy_floor = floor_ps(gy);

            const __m128 tx = _mm_sub_ps(gx, gx_floor);
            const __m128 ty = _mm_sub_ps(gy, gy_floor);

            __m128 coefficients[4];

            __m128 gx0 = _mm_add_ps(gx_floor, vn1fp4);
            __m128 gx1 = gx_floor;
            __m128 gx2 = _mm_add_ps(gx_floor, v1fp4);
            __m128 gx3 = _mm_add_ps(gx_floor, _mm_set1_ps(2.0f));

            __m128i x0 = _mm_cvtps_epi32(gx0);
            __m128i x1 = _mm_cvtps_epi32(gx1);
            __m128i x2 = _mm_cvtps_epi32(gx2);
            __m128i x3 = _mm_cvtps_epi32(gx3);

            __m128i x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
            __m128i x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            __m128i x2_in_range = _mm_and_si128(_mm_cmpgt_epi32(x2, vn1ip4), _mm_cmpgt_epi32(vImgWi, x2));
            __m128i x3_in_range = _mm_and_si128(_mm_cmpgt_epi32(x3, vn1ip4), _mm_cmpgt_epi32(vImgWi, x3));

            __m128i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4],
                    v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i));

                __m128i y = _mm_cvtps_epi32(gy);

                __m128i y_in_range = _mm_and_si128(_mm_cmpgt_epi32(y, vn1ip4), _mm_cmpgt_epi32(vImgHi, y));

                v0_in_range[i] = _mm_and_si128(x0_in_range, y_in_range);
                v1_in_range[i] = _mm_and_si128(x1_in_range, y_in_range);
                v2_in_range[i] = _mm_and_si128(x2_in_range, y_in_range);
                v3_in_range[i] = _mm_and_si128(x3_in_range, y_in_range);

                __m128 v0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m128 x0_val = mask_gather_ps(src.channel(q), v0_offset[i], _mm_castsi128_ps(v0_in_range[i]));
                    __m128 x1_val = mask_gather_ps(src.channel(q), v1_offset[i], _mm_castsi128_ps(v1_in_range[i]));
                    __m128 x2_val = mask_gather_ps(src.channel(q), v2_offset[i], _mm_castsi128_ps(v2_in_range[i]));
                    __m128 x3_val = mask_gather_ps(src.channel(q), v3_offset[i], _mm_castsi128_ps(v3_in_range[i]));

                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m128 _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);

    const __m128 vElempackf = _mm_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            const __m128 two = _mm_set1_ps(2.f);
            const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);
            const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);
            gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);
            gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

            __m128 gx_floor = floor_ps(gx);
            __m128 gy_floor = floor_ps(gy);

            const __m128 tx = _mm_sub_ps(gx, gx_floor);
            const __m128 ty = _mm_sub_ps(gy, gy_floor);

            __m128 coefficients[4];

            __m128 gx0 = _mm_add_ps(gx_floor, vn1fp4);
            __m128 gx1 = gx_floor;
            __m128 gx2 = _mm_add_ps(gx_floor, v1fp4);
            __m128 gx3 = _mm_add_ps(gx_floor, _mm_set1_ps(2.0f));

            gx0 = _mm_min_ps(border_x, _mm_max_ps(gx0, _mm_setzero_ps()));
            gx1 = _mm_min_ps(border_x, _mm_max_ps(gx1, _mm_setzero_ps()));
            gx2 = _mm_min_ps(border_x, _mm_max_ps(gx2, _mm_setzero_ps()));
            gx3 = _mm_min_ps(border_x, _mm_max_ps(gx3, _mm_setzero_ps()));

            __m128i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i));
                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                __m128 v0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m128 x0_val = mask_gather_ps(src.channel(q), v0_offset[i], vn1fp4);
                    __m128 x1_val = mask_gather_ps(src.channel(q), v1_offset[i], vn1fp4);
                    __m128 x2_val = mask_gather_ps(src.channel(q), v2_offset[i], vn1fp4);
                    __m128 x3_val = mask_gather_ps(src.channel(q), v3_offset[i], vn1fp4);

                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m128 _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);

    const __m128 vElempackf = _mm_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            const __m128 two = _mm_set1_ps(2.f);
            const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);
            const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

            gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
            gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

            __m128 gx_floor = floor_ps(gx);
            __m128 gy_floor = floor_ps(gy);

            const __m128 tx = _mm_sub_ps(gx, gx_floor);
            const __m128 ty = _mm_sub_ps(gy, gy_floor);

            __m128 coefficients[4];

            __m128 gx0 = _mm_add_ps(gx_floor, vn1fp4);
            __m128 gx1 = gx_floor;
            __m128 gx2 = _mm_add_ps(gx_floor, v1fp4);
            __m128 gx3 = _mm_add_ps(gx_floor, _mm_set1_ps(2.0f));

            gx0 = _mm_min_ps(border_x, _mm_max_ps(gx0, _mm_setzero_ps()));
            gx1 = _mm_min_ps(border_x, _mm_max_ps(gx1, _mm_setzero_ps()));
            gx2 = _mm_min_ps(border_x, _mm_max_ps(gx2, _mm_setzero_ps()));
            gx3 = _mm_min_ps(border_x, _mm_max_ps(gx3, _mm_setzero_ps()));

            __m128i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i));
                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                __m128 v0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m128 x0_val = mask_gather_ps(src.channel(q), v0_offset[i], vn1fp4);
                    __m128 x1_val = mask_gather_ps(src.channel(q), v1_offset[i], vn1fp4);
                    __m128 x2_val = mask_gather_ps(src.channel(q), v2_offset[i], vn1fp4);
                    __m128 x3_val = mask_gather_ps(src.channel(q), v3_offset[i], vn1fp4);

                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m128 _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);

    const __m128 vElempackf = _mm_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            const __m128 two = _mm_set1_ps(2.f);
            const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);
            const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);
            gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);
            gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

            __m128 gx_floor = floor_ps(gx);
            __m128 gy_floor = floor_ps(gy);

            const __m128 tx = _mm_sub_ps(gx, gx_floor);
            const __m128 ty = _mm_sub_ps(gy, gy_floor);

            __m128 coefficients[4];

            __m128 gx0 = _mm_add_ps(gx_floor, vn1fp4);
            __m128 gx1 = gx_floor;
            __m128 gx2 = _mm_add_ps(gx_floor, v1fp4);
            __m128 gx3 = _mm_add_ps(gx_floor, _mm_set1_ps(2.0f));
            const __m128 v0p5fp4 = _mm_set1_ps(0.5f);
            {
                // x0
                gx0 = _mm_add_ps(gx0, v0p5fp4);

                gx0 = _mm_and_ps(gx0, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx0_v = _mm_and_ps(_mm_sub_ps(gx0, vImgWf), *(__m128*)_ps_inv_sign_mask);
                gx0 = _mm_sub_ps(vImgWf, reflectx0_v);

                gx0 = _mm_sub_ps(gx0, v0p5fp4);

                _mm_sub_ps(gx0, v0p5fp4);

                gx0 = _mm_min_ps(border_x, _mm_max_ps(gx0, _mm_setzero_ps()));

                // x1
                gx1 = _mm_add_ps(gx1, v0p5fp4);

                gx1 = _mm_and_ps(gx1, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx1_v = _mm_and_ps(_mm_sub_ps(gx1, vImgWf), *(__m128*)_ps_inv_sign_mask);
                gx1 = _mm_sub_ps(vImgWf, reflectx1_v);

                gx1 = _mm_sub_ps(gx1, v0p5fp4);

                _mm_sub_ps(gx1, v0p5fp4);

                gx1 = _mm_min_ps(border_x, _mm_max_ps(gx1, _mm_setzero_ps()));

                // x2
                gx2 = _mm_add_ps(gx2, v0p5fp4);

                gx2 = _mm_and_ps(gx2, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx2_v = _mm_and_ps(_mm_sub_ps(gx2, vImgWf), *(__m128*)_ps_inv_sign_mask);
                gx2 = _mm_sub_ps(vImgWf, reflectx2_v);

                gx2 = _mm_sub_ps(gx2, v0p5fp4);

                _mm_sub_ps(gx2, v0p5fp4);

                gx2 = _mm_min_ps(border_x, _mm_max_ps(gx2, _mm_setzero_ps()));

                // x3
                gx3 = _mm_add_ps(gx3, v0p5fp4);

                gx3 = _mm_and_ps(gx3, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx3_v = _mm_and_ps(_mm_sub_ps(gx3, vImgWf), *(__m128*)_ps_inv_sign_mask);
                gx3 = _mm_sub_ps(vImgWf, reflectx3_v);

                gx3 = _mm_sub_ps(gx3, v0p5fp4);

                _mm_sub_ps(gx3, v0p5fp4);

                gx3 = _mm_min_ps(border_x, _mm_max_ps(gx3, _mm_setzero_ps()));
            }

            __m128i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i));

                {
                    //y
                    gy = _mm_add_ps(gy, v0p5fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps_inv_sign_mask);
                    gy = _mm_sub_ps(vImgHf, reflecty_v);

                    gy = _mm_sub_ps(gy, v0p5fp4);

                    _mm_sub_ps(gy, v0p5fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
                }

                __m128 v0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m128 x0_val = mask_gather_ps(src.channel(q), v0_offset[i], vn1fp4);
                    __m128 x1_val = mask_gather_ps(src.channel(q), v1_offset[i], vn1fp4);
                    __m128 x2_val = mask_gather_ps(src.channel(q), v2_offset[i], vn1fp4);
                    __m128 x3_val = mask_gather_ps(src.channel(q), v3_offset[i], vn1fp4);

                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m128 _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);

    const __m128 vElempackf = _mm_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m128 gx = _mm_set1_ps(gridptr[0]);
            __m128 gy = _mm_set1_ps(gridptr[grid.elempack]);

            const __m128 two = _mm_set1_ps(2.f);
            const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);
            const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

            gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
            gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

            __m128 gx_floor = floor_ps(gx);
            __m128 gy_floor = floor_ps(gy);

            const __m128 tx = _mm_sub_ps(gx, gx_floor);
            const __m128 ty = _mm_sub_ps(gy, gy_floor);

            __m128 coefficients[4];

            __m128 gx0 = _mm_add_ps(gx_floor, vn1fp4);
            __m128 gx1 = gx_floor;
            __m128 gx2 = _mm_add_ps(gx_floor, v1fp4);
            __m128 gx3 = _mm_add_ps(gx_floor, _mm_set1_ps(2.0f));
            {
                // x0
                gx0 = _mm_and_ps(gx0, *(__m128*)_ps_inv_sign_mask);
                __m128 reflectx0_v = _mm_and_ps(_mm_sub_ps(gx0, border_x), *(__m128*)_ps_inv_sign_mask);
                gx0 = _mm_sub_ps(border_x, reflectx0_v);

                // x1
                gx1 = _mm_and_ps(gx1, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx1_v = _mm_and_ps(_mm_sub_ps(gx1, border_x), *(__m128*)_ps_inv_sign_mask);
                gx1 = _mm_sub_ps(border_x, reflectx1_v);

                // x2
                gx2 = _mm_and_ps(gx2, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx2_v = _mm_and_ps(_mm_sub_ps(gx2, border_x), *(__m128*)_ps_inv_sign_mask);
                gx2 = _mm_sub_ps(border_x, reflectx2_v);

                // x3
                gx3 = _mm_and_ps(gx3, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx3_v = _mm_and_ps(_mm_sub_ps(gx3, border_x), *(__m128*)_ps_inv_sign_mask);
                gx3 = _mm_sub_ps(border_x, reflectx3_v);
            }

            __m128i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i));

                {
                    //y
                    gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps_inv_sign_mask);
                    gy = _mm_sub_ps(border_y, reflecty_v);
                }

                __m128 v0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                __m128 v3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m128 x0_val = mask_gather_ps(src.channel(q), v0_offset[i], vn1fp4);
                    __m128 x1_val = mask_gather_ps(src.channel(q), v1_offset[i], vn1fp4);
                    __m128 x2_val = mask_gather_ps(src.channel(q), v2_offset[i], vn1fp4);
                    __m128 x3_val = mask_gather_ps(src.channel(q), v3_offset[i], vn1fp4);

                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m128 _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}