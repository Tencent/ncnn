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

static NCNN_FORCEINLINE __m256 cubic_interp1d_p8(const __m256& x0_v, const __m256& x1_v, const __m256& x2_v, const __m256& x3_v, const __m256& tx)
{
    const __m256 A = _mm256_set1_ps(-0.75f);

    const __m256 x0 = _mm256_add_ps(tx, v1fp8);
    const __m256& x1 = tx;
    const __m256 x2 = _mm256_sub_ps(v1fp8, tx);
    //const __m256 x3 = _mm256_add_ps(x2, v1fp8);

    const __m256 coeffs0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(A, x0), _mm256_mul_ps(_mm256_set1_ps(5.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(8.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(4), A));
    const __m256 coeffs1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x1), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x1), x1), v1fp8);
    const __m256 coeffs2 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x2), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x2), x2), v1fp8);
    const __m256 coeffs3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(v1fp8, coeffs0), coeffs1), coeffs2);

    __m256 _v = _mm256_mul_ps(coeffs0, x0_v);
    _v = _mm256_comp_fmadd_ps(coeffs1, x1_v, _v);
    _v = _mm256_comp_fmadd_ps(coeffs2, x2_v, _v);
    _v = _mm256_comp_fmadd_ps(coeffs3, x3_v, _v);

    return _v;
}

static void gridsample_2d_bicubic_align0_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256 vElempackf = _mm256_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m256 gx = _mm256_set1_ps(gridptr[0]);
            __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m256 two = _mm256_set1_ps(2.f);

                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);
            }

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, vn1fp8);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, v1fp8);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            __m256i x0 = _mm256_cvtps_epi32(gx0);
            __m256i x1 = _mm256_cvtps_epi32(gx1);
            __m256i x2 = _mm256_cvtps_epi32(gx2);
            __m256i x3 = _mm256_cvtps_epi32(gx3);
                  
            __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x0));
            __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
            __m256i x2_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x2, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x2));
            __m256i x3_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x3, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x3));

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4],
                    v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                __m256i y = _mm256_cvtps_epi32(gy);

                __m256i y_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y));

                v0_in_range[i] = _mm256_and_si256(x0_in_range, y_in_range);
                v1_in_range[i] = _mm256_and_si256(x1_in_range, y_in_range);
                v2_in_range[i] = _mm256_and_si256(x2_in_range, y_in_range);
                v3_in_range[i] = _mm256_and_si256(x3_in_range, y_in_range);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v0_offset[i], *reinterpret_cast<__m256*>(&v0_in_range[i]), sizeof(float));
                    __m256 x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v1_offset[i], *reinterpret_cast<__m256*>(&v1_in_range[i]), sizeof(float));
                    __m256 x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v2_offset[i], *reinterpret_cast<__m256*>(&v2_in_range[i]), sizeof(float));
                    __m256 x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v3_offset[i], *reinterpret_cast<__m256*>(&v3_in_range[i]), sizeof(float));

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256 vElempackf = _mm256_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m256 gx = _mm256_set1_ps(gridptr[0]);
            __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const __m256 two = _mm256_set1_ps(2.f);

                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));
            }

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, vn1fp8);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, v1fp8);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            __m256i x0 = _mm256_cvtps_epi32(gx0);
            __m256i x1 = _mm256_cvtps_epi32(gx1);
            __m256i x2 = _mm256_cvtps_epi32(gx2);
            __m256i x3 = _mm256_cvtps_epi32(gx3);

            __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x0));
            __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
            __m256i x2_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x2, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x2));
            __m256i x3_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x3, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x3));

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4],
                    v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                __m256i y = _mm256_cvtps_epi32(gy);

                __m256i y_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y));

                v0_in_range[i] = _mm256_and_si256(x0_in_range, y_in_range);
                v1_in_range[i] = _mm256_and_si256(x1_in_range, y_in_range);
                v2_in_range[i] = _mm256_and_si256(x2_in_range, y_in_range);
                v3_in_range[i] = _mm256_and_si256(x3_in_range, y_in_range);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v0_offset[i], *reinterpret_cast<__m256*>(&v0_in_range[i]), sizeof(float));
                    __m256 x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v1_offset[i], *reinterpret_cast<__m256*>(&v1_in_range[i]), sizeof(float));
                    __m256 x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v2_offset[i], *reinterpret_cast<__m256*>(&v2_in_range[i]), sizeof(float));
                    __m256 x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v3_offset[i], *reinterpret_cast<__m256*>(&v3_in_range[i]), sizeof(float));

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256 vElempackf = _mm256_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m256 gx = _mm256_set1_ps(gridptr[0]);
            __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

            const __m256 two = _mm256_set1_ps(2.f);
            const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);
            const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);
            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, vn1fp8);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, v1fp8);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            gx0 = _mm256_min_ps(border_x, _mm256_max_ps(gx0, _mm256_setzero_ps()));
            gx1 = _mm256_min_ps(border_x, _mm256_max_ps(gx1, _mm256_setzero_ps()));
            gx2 = _mm256_min_ps(border_x, _mm256_max_ps(gx2, _mm256_setzero_ps()));
            gx3 = _mm256_min_ps(border_x, _mm256_max_ps(gx3, _mm256_setzero_ps()));

            __m256i x0 = _mm256_cvtps_epi32(gx0);
            __m256i x1 = _mm256_cvtps_epi32(gx1);
            __m256i x2 = _mm256_cvtps_epi32(gx2);
            __m256i x3 = _mm256_cvtps_epi32(gx3);

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                __m256i y = _mm256_cvtps_epi32(gy);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v0_offset[i], vn1fp8, sizeof(float));
                    __m256 x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v1_offset[i], vn1fp8, sizeof(float));
                    __m256 x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v2_offset[i], vn1fp8, sizeof(float));
                    __m256 x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v3_offset[i], vn1fp8, sizeof(float));

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256 vElempackf = _mm256_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m256 gx = _mm256_set1_ps(gridptr[0]);
            __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

            const __m256 two = _mm256_set1_ps(2.f);
            const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);
            const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, vn1fp8);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, v1fp8);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            gx0 = _mm256_min_ps(border_x, _mm256_max_ps(gx0, _mm256_setzero_ps()));
            gx1 = _mm256_min_ps(border_x, _mm256_max_ps(gx1, _mm256_setzero_ps()));
            gx2 = _mm256_min_ps(border_x, _mm256_max_ps(gx2, _mm256_setzero_ps()));
            gx3 = _mm256_min_ps(border_x, _mm256_max_ps(gx3, _mm256_setzero_ps()));

            __m256i x0 = _mm256_cvtps_epi32(gx0);
            __m256i x1 = _mm256_cvtps_epi32(gx1);
            __m256i x2 = _mm256_cvtps_epi32(gx2);
            __m256i x3 = _mm256_cvtps_epi32(gx3);

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                __m256i y = _mm256_cvtps_epi32(gy);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v0_offset[i], vn1fp8, sizeof(float));
                    __m256 x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v1_offset[i], vn1fp8, sizeof(float));
                    __m256 x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v2_offset[i], vn1fp8, sizeof(float));
                    __m256 x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v3_offset[i], vn1fp8, sizeof(float));

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256 vElempackf = _mm256_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m256 gx = _mm256_set1_ps(gridptr[0]);
            __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

            const __m256 two = _mm256_set1_ps(2.f);
            const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);
            const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);
            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, vn1fp8);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, v1fp8);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));
            const __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
            {
                // x0
                const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx0 = _mm256_add_ps(gx0, v0p5fp8);

                gx0 = _mm256_and_ps(gx0, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx0_v = _mm256_and_ps(_mm256_sub_ps(gx0, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx0 = _mm256_sub_ps(vImgWf, reflectx0_v);

                gx0 = _mm256_sub_ps(gx0, v0p5fp8);

                _mm256_sub_ps(gx0, v0p5fp8);

                gx0 = _mm256_min_ps(border_x, _mm256_max_ps(gx0, _mm256_setzero_ps()));

                // x1
                gx1 = _mm256_add_ps(gx1, v0p5fp8);

                gx1 = _mm256_and_ps(gx1, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx1_v = _mm256_and_ps(_mm256_sub_ps(gx1, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx1 = _mm256_sub_ps(vImgWf, reflectx1_v);

                gx1 = _mm256_sub_ps(gx1, v0p5fp8);

                _mm256_sub_ps(gx1, v0p5fp8);

                gx1 = _mm256_min_ps(border_x, _mm256_max_ps(gx1, _mm256_setzero_ps()));

                // x2
                gx2 = _mm256_add_ps(gx2, v0p5fp8);

                gx2 = _mm256_and_ps(gx2, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx2_v = _mm256_and_ps(_mm256_sub_ps(gx2, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx2 = _mm256_sub_ps(vImgWf, reflectx2_v);

                gx2 = _mm256_sub_ps(gx2, v0p5fp8);

                _mm256_sub_ps(gx2, v0p5fp8);

                gx2 = _mm256_min_ps(border_x, _mm256_max_ps(gx2, _mm256_setzero_ps()));

                // x3
                gx3 = _mm256_add_ps(gx3, v0p5fp8);

                gx3 = _mm256_and_ps(gx3, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx3_v = _mm256_and_ps(_mm256_sub_ps(gx3, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx3 = _mm256_sub_ps(vImgWf, reflectx3_v);

                gx3 = _mm256_sub_ps(gx3, v0p5fp8);

                _mm256_sub_ps(gx3, v0p5fp8);

                gx3 = _mm256_min_ps(border_x, _mm256_max_ps(gx3, _mm256_setzero_ps()));
            }

            __m256i x0 = _mm256_cvtps_epi32(gx0);
            __m256i x1 = _mm256_cvtps_epi32(gx1);
            __m256i x2 = _mm256_cvtps_epi32(gx2);
            __m256i x3 = _mm256_cvtps_epi32(gx3);

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                {
                    //y
                    const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                    gy = _mm256_add_ps(gy, v0p5fp8);

                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                    gy = _mm256_sub_ps(gy, v0p5fp8);

                    _mm256_sub_ps(gy, v0p5fp8);

                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                }

                __m256i y = _mm256_cvtps_epi32(gy);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v0_offset[i], vn1fp8, sizeof(float));
                    __m256 x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v1_offset[i], vn1fp8, sizeof(float));
                    __m256 x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v2_offset[i], vn1fp8, sizeof(float));
                    __m256 x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v3_offset[i], vn1fp8, sizeof(float));

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    float* outptr = static_cast<float*>(dst.data);

    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256 vElempackf = _mm256_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m256 gx = _mm256_set1_ps(gridptr[0]);
            __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

            const __m256 two = _mm256_set1_ps(2.f);
            const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);
            const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, vn1fp8);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, v1fp8);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));
            const __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
            {
                // x0
                const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx0 = _mm256_and_ps(gx0, *(__m256*)_ps256_inv_sign_mask);
                __m256 reflectx0_v = _mm256_and_ps(_mm256_sub_ps(gx0, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx0 = _mm256_sub_ps(border_x, reflectx0_v);

                // x1
                gx1 = _mm256_and_ps(gx1, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx1_v = _mm256_and_ps(_mm256_sub_ps(gx1, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx1 = _mm256_sub_ps(border_x, reflectx1_v);

                // x2
                gx2 = _mm256_and_ps(gx2, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx2_v = _mm256_and_ps(_mm256_sub_ps(gx2, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx2 = _mm256_sub_ps(border_x, reflectx2_v);

                // x3
                gx3 = _mm256_and_ps(gx3, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx3_v = _mm256_and_ps(_mm256_sub_ps(gx3, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx3 = _mm256_sub_ps(border_x, reflectx3_v);
            }

            __m256i x0 = _mm256_cvtps_epi32(gx0);
            __m256i x1 = _mm256_cvtps_epi32(gx1);
            __m256i x2 = _mm256_cvtps_epi32(gx2);
            __m256i x3 = _mm256_cvtps_epi32(gx3);

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                {
                    //y
                    const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                    gy = _mm256_sub_ps(border_y, reflecty_v);
                }

                __m256i y = _mm256_cvtps_epi32(gy);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                 _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v0_offset[i], vn1fp8, sizeof(float));
                    __m256 x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v1_offset[i], vn1fp8, sizeof(float));
                    __m256 x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v2_offset[i], vn1fp8, sizeof(float));
                    __m256 x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), v3_offset[i], vn1fp8, sizeof(float));

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}