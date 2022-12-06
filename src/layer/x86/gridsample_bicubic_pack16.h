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

static NCNN_FORCEINLINE __m512 cubic_interp1d_p16(const __m512& x0_v, const __m512& x1_v, const __m512& x2_v, const __m512& x3_v, const __m512& tx)
{
    const __m512 A = _mm512_set1_ps(-0.75f);

    const __m512 x0 = _mm512_add_ps(tx, *(__m512*)_ps512_1);
    const __m512& x1 = tx;
    const __m512 x2 = _mm512_sub_ps(*(__m512*)_ps512_1, tx);
    //const __m512 x3 = _mm512_add_ps(x2, *(__m512*)_ps512_1);

    const __m512 coeffs0 = _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(A, x0), _mm512_mul_ps(_mm512_set1_ps(5.0f), A)), x0), _mm512_mul_ps(_mm512_set1_ps(8.0f), A)), x0), _mm512_mul_ps(_mm512_set1_ps(4), A));
    const __m512 coeffs1 = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(A, _mm512_set1_ps(2.0f)), x1), _mm512_add_ps(A, _mm512_set1_ps(3.0f))), x1), x1), *(__m512*)_ps512_1);
    const __m512 coeffs2 = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(A, _mm512_set1_ps(2.0f)), x2), _mm512_add_ps(A, _mm512_set1_ps(3.0f))), x2), x2), *(__m512*)_ps512_1);
    const __m512 coeffs3 = _mm512_sub_ps(_mm512_sub_ps(_mm512_sub_ps(*(__m512*)_ps512_1, coeffs0), coeffs1), coeffs2);

    __m512 _v = _mm512_mul_ps(coeffs0, x0_v);
    _v = _mm512_fmadd_ps(coeffs1, x1_v, _v);
    _v = _mm512_fmadd_ps(coeffs2, x2_v, _v);
    _v = _mm512_fmadd_ps(coeffs3, x3_v, _v);

    return _v;
}

static void gridsample_2d_bicubic_align0_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512 vElempackf = _mm512_set1_ps(src.elempack);

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

            __m512 gx_floor = _mm512_roundscale_ps(gx, _MM_FROUND_TO_NEG_INF);
            __m512 gy_floor = _mm512_roundscale_ps(gy, _MM_FROUND_TO_NEG_INF);

            const __m512 tx = _mm512_sub_ps(gx, gx_floor);
            const __m512 ty = _mm512_sub_ps(gy, gy_floor);

            __m512 coefficients[4];

            __m512 gx0 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_n1);
            __m512 gx1 = gx_floor;
            __m512 gx2 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_1);
            __m512 gx3 = _mm512_add_ps(gx_floor, _mm512_set1_ps(2.0f));

            __m512i x0 = _mm512_cvtps_epi32(gx0);
            __m512i x1 = _mm512_cvtps_epi32(gx1);
            __m512i x2 = _mm512_cvtps_epi32(gx2);
            __m512i x3 = _mm512_cvtps_epi32(gx3);

            __mmask16 x0_in_range = _mm512_cmpgt_epi32_mask(x0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x0);
            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 x2_in_range = _mm512_cmpgt_epi32_mask(x2, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x2);
            __mmask16 x3_in_range = _mm512_cmpgt_epi32_mask(x3, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x3);

            __m512i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            __mmask16 v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm512_add_ps(gy_floor, _mm512_set1_ps(-1.0f + i));

                __m512i y = _mm512_cvtps_epi32(gy);

                __mmask16 y_in_range = _mm512_cmpgt_epi32_mask(y, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y);

                v0_in_range[i] = x0_in_range & y_in_range;
                v1_in_range[i] = x1_in_range & y_in_range;
                v2_in_range[i] = x2_in_range & y_in_range;
                v3_in_range[i] = x3_in_range & y_in_range;

                __m512 v0_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v1_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v2_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v3_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm512_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm512_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm512_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm512_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v0_in_range[i], v0_offset[i], src.channel(q), sizeof(float));
                    __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v1_in_range[i], v1_offset[i], src.channel(q), sizeof(float));
                    __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v2_in_range[i], v2_offset[i], src.channel(q), sizeof(float));
                    __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v3_in_range[i], v3_offset[i], src.channel(q), sizeof(float));

                    coefficients[i] = cubic_interp1d_p16(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m512 _v = cubic_interp1d_p16(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512 vElempackf = _mm512_set1_ps(src.elempack);

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

                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));
            }

            __m512 gx_floor = _mm512_roundscale_ps(gx, _MM_FROUND_TO_NEG_INF);
            __m512 gy_floor = _mm512_roundscale_ps(gy, _MM_FROUND_TO_NEG_INF);

            const __m512 tx = _mm512_sub_ps(gx, gx_floor);
            const __m512 ty = _mm512_sub_ps(gy, gy_floor);

            __m512 coefficients[4];

            __m512 gx0 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_n1);
            __m512 gx1 = gx_floor;
            __m512 gx2 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_1);
            __m512 gx3 = _mm512_add_ps(gx_floor, _mm512_set1_ps(2.0f));

            __m512i x0 = _mm512_cvtps_epi32(gx0);
            __m512i x1 = _mm512_cvtps_epi32(gx1);
            __m512i x2 = _mm512_cvtps_epi32(gx2);
            __m512i x3 = _mm512_cvtps_epi32(gx3);

            __mmask16 x0_in_range = _mm512_cmpgt_epi32_mask(x0, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x0);
            __mmask16 x1_in_range = _mm512_cmpgt_epi32_mask(x1, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x1);
            __mmask16 x2_in_range = _mm512_cmpgt_epi32_mask(x2, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x2);
            __mmask16 x3_in_range = _mm512_cmpgt_epi32_mask(x3, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgWi, x3);

            __m512i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            __mmask16 v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm512_add_ps(gy_floor, _mm512_set1_ps(-1.0f + i));

                __m512i y = _mm512_cvtps_epi32(gy);

                __mmask16 y_in_range = _mm512_cmpgt_epi32_mask(y, *(__m512i*)_pi32_512_n1) & _mm512_cmpgt_epi32_mask(vImgHi, y);

                v0_in_range[i] = x0_in_range & y_in_range;
                v1_in_range[i] = x1_in_range & y_in_range;
                v2_in_range[i] = x2_in_range & y_in_range;
                v3_in_range[i] = x3_in_range & y_in_range;

                __m512 v0_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v1_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v2_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v3_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm512_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm512_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm512_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm512_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v0_in_range[i], v0_offset[i], src.channel(q), sizeof(float));
                    __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v1_in_range[i], v1_offset[i], src.channel(q), sizeof(float));
                    __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v2_in_range[i], v2_offset[i], src.channel(q), sizeof(float));
                    __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v3_in_range[i], v3_offset[i], src.channel(q), sizeof(float));

                    coefficients[i] = cubic_interp1d_p16(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m512 _v = cubic_interp1d_p16(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512 vElempackf = _mm512_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            const __m512 two = _mm512_set1_ps(2.f);
            const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);
            const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);
            gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);
            gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);

            __m512 gx_floor = _mm512_roundscale_ps(gx, _MM_FROUND_TO_NEG_INF);
            __m512 gy_floor = _mm512_roundscale_ps(gy, _MM_FROUND_TO_NEG_INF);

            const __m512 tx = _mm512_sub_ps(gx, gx_floor);
            const __m512 ty = _mm512_sub_ps(gy, gy_floor);

            __m512 coefficients[4];

            __m512 gx0 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_n1);
            __m512 gx1 = gx_floor;
            __m512 gx2 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_1);
            __m512 gx3 = _mm512_add_ps(gx_floor, _mm512_set1_ps(2.0f));

            gx0 = _mm512_min_ps(border_x, _mm512_max_ps(gx0, _mm512_setzero_ps()));
            gx1 = _mm512_min_ps(border_x, _mm512_max_ps(gx1, _mm512_setzero_ps()));
            gx2 = _mm512_min_ps(border_x, _mm512_max_ps(gx2, _mm512_setzero_ps()));
            gx3 = _mm512_min_ps(border_x, _mm512_max_ps(gx3, _mm512_setzero_ps()));

            __m512i x0 = _mm512_cvtps_epi32(gx0);
            __m512i x1 = _mm512_cvtps_epi32(gx1);
            __m512i x2 = _mm512_cvtps_epi32(gx2);
            __m512i x3 = _mm512_cvtps_epi32(gx3);

            __m512i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm512_add_ps(gy_floor, _mm512_set1_ps(-1.0f + i));
                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                __m512i y = _mm512_cvtps_epi32(gy);

                __m512 v0_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v1_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v2_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v3_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm512_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm512_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm512_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm512_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v0_offset[i], src.channel(q), sizeof(float));
                    __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v1_offset[i], src.channel(q), sizeof(float));
                    __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v2_offset[i], src.channel(q), sizeof(float));
                    __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v3_offset[i], src.channel(q), sizeof(float));

                    coefficients[i] = cubic_interp1d_p16(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m512 _v = cubic_interp1d_p16(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512 vElempackf = _mm512_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            const __m512 two = _mm512_set1_ps(2.f);
            const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);
            const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

            gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));
            gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));

            __m512 gx_floor = _mm512_roundscale_ps(gx, _MM_FROUND_TO_NEG_INF);
            __m512 gy_floor = _mm512_roundscale_ps(gy, _MM_FROUND_TO_NEG_INF);

            const __m512 tx = _mm512_sub_ps(gx, gx_floor);
            const __m512 ty = _mm512_sub_ps(gy, gy_floor);

            __m512 coefficients[4];

            __m512 gx0 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_n1);
            __m512 gx1 = gx_floor;
            __m512 gx2 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_1);
            __m512 gx3 = _mm512_add_ps(gx_floor, _mm512_set1_ps(2.0f));

            gx0 = _mm512_min_ps(border_x, _mm512_max_ps(gx0, _mm512_setzero_ps()));
            gx1 = _mm512_min_ps(border_x, _mm512_max_ps(gx1, _mm512_setzero_ps()));
            gx2 = _mm512_min_ps(border_x, _mm512_max_ps(gx2, _mm512_setzero_ps()));
            gx3 = _mm512_min_ps(border_x, _mm512_max_ps(gx3, _mm512_setzero_ps()));

            __m512i x0 = _mm512_cvtps_epi32(gx0);
            __m512i x1 = _mm512_cvtps_epi32(gx1);
            __m512i x2 = _mm512_cvtps_epi32(gx2);
            __m512i x3 = _mm512_cvtps_epi32(gx3);

            __m512i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm512_add_ps(gy_floor, _mm512_set1_ps(-1.0f + i));
                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                __m512i y = _mm512_cvtps_epi32(gy);

                __m512 v0_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v1_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v2_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v3_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm512_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm512_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm512_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm512_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v0_offset[i], src.channel(q), sizeof(float));
                    __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v1_offset[i], src.channel(q), sizeof(float));
                    __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v2_offset[i], src.channel(q), sizeof(float));
                    __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v3_offset[i], src.channel(q), sizeof(float));

                    coefficients[i] = cubic_interp1d_p16(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m512 _v = cubic_interp1d_p16(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512 vElempackf = _mm512_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            const __m512 two = _mm512_set1_ps(2.f);
            const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);
            const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);
            gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), vImgWf, *(__m512*)_ps512_1), two);
            gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), vImgHf, *(__m512*)_ps512_1), two);

            __m512 gx_floor = _mm512_roundscale_ps(gx, _MM_FROUND_TO_NEG_INF);
            __m512 gy_floor = _mm512_roundscale_ps(gy, _MM_FROUND_TO_NEG_INF);

            const __m512 tx = _mm512_sub_ps(gx, gx_floor);
            const __m512 ty = _mm512_sub_ps(gy, gy_floor);

            __m512 coefficients[4];

            __m512 gx0 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_n1);
            __m512 gx1 = gx_floor;
            __m512 gx2 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_1);
            __m512 gx3 = _mm512_add_ps(gx_floor, _mm512_set1_ps(2.0f));
            const __m512 v0p5fp16 = _mm512_set1_ps(0.5f);
            {
                // x0
                const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                gx0 = _mm512_add_ps(gx0, v0p5fp16);

                gx0 = _mm512_and_ps(gx0, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx0_v = _mm512_and_ps(_mm512_sub_ps(gx0, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                gx0 = _mm512_sub_ps(vImgWf, reflectx0_v);

                gx0 = _mm512_sub_ps(gx0, v0p5fp16);

                _mm512_sub_ps(gx0, v0p5fp16);

                gx0 = _mm512_min_ps(border_x, _mm512_max_ps(gx0, _mm512_setzero_ps()));

                // x1
                gx1 = _mm512_add_ps(gx1, v0p5fp16);

                gx1 = _mm512_and_ps(gx1, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx1_v = _mm512_and_ps(_mm512_sub_ps(gx1, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                gx1 = _mm512_sub_ps(vImgWf, reflectx1_v);

                gx1 = _mm512_sub_ps(gx1, v0p5fp16);

                _mm512_sub_ps(gx1, v0p5fp16);

                gx1 = _mm512_min_ps(border_x, _mm512_max_ps(gx1, _mm512_setzero_ps()));

                // x2
                gx2 = _mm512_add_ps(gx2, v0p5fp16);

                gx2 = _mm512_and_ps(gx2, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx2_v = _mm512_and_ps(_mm512_sub_ps(gx2, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                gx2 = _mm512_sub_ps(vImgWf, reflectx2_v);

                gx2 = _mm512_sub_ps(gx2, v0p5fp16);

                _mm512_sub_ps(gx2, v0p5fp16);

                gx2 = _mm512_min_ps(border_x, _mm512_max_ps(gx2, _mm512_setzero_ps()));

                // x3
                gx3 = _mm512_add_ps(gx3, v0p5fp16);

                gx3 = _mm512_and_ps(gx3, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx3_v = _mm512_and_ps(_mm512_sub_ps(gx3, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                gx3 = _mm512_sub_ps(vImgWf, reflectx3_v);

                gx3 = _mm512_sub_ps(gx3, v0p5fp16);

                _mm512_sub_ps(gx3, v0p5fp16);

                gx3 = _mm512_min_ps(border_x, _mm512_max_ps(gx3, _mm512_setzero_ps()));
            }

            __m512i x0 = _mm512_cvtps_epi32(gx0);
            __m512i x1 = _mm512_cvtps_epi32(gx1);
            __m512i x2 = _mm512_cvtps_epi32(gx2);
            __m512i x3 = _mm512_cvtps_epi32(gx3);

            __m512i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm512_add_ps(gy_floor, _mm512_set1_ps(-1.0f + i));

                {
                    //y
                    const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                    gy = _mm512_add_ps(gy, v0p5fp16);

                    gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, vImgHf), *(__m512*)_ps512_inv_sign_mask);
                    gy = _mm512_sub_ps(vImgHf, reflecty_v);

                    gy = _mm512_sub_ps(gy, v0p5fp16);

                    _mm512_sub_ps(gy, v0p5fp16);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
                }

                __m512i y = _mm512_cvtps_epi32(gy);

                __m512 v0_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v1_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v2_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v3_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm512_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm512_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm512_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm512_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v0_offset[i], src.channel(q), sizeof(float));
                    __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v1_offset[i], src.channel(q), sizeof(float));
                    __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v2_offset[i], src.channel(q), sizeof(float));
                    __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v3_offset[i], src.channel(q), sizeof(float));

                    coefficients[i] = cubic_interp1d_p16(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m512 _v = cubic_interp1d_p16(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    float* outptr = static_cast<float*>(dst.data);

    const __m512 vImgWf = _mm512_set1_ps(src.w);
    const __m512 vImgHf = _mm512_set1_ps(src.h);
    const __m512i vImgWi = _mm512_set1_epi32(src.w);
    const __m512i vImgHi = _mm512_set1_epi32(src.h);

    const __m512 vElempackf = _mm512_set1_ps(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            __m512 gx = _mm512_set1_ps(gridptr[0]);
            __m512 gy = _mm512_set1_ps(gridptr[grid.elempack]);

            const __m512 two = _mm512_set1_ps(2.f);
            const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);
            const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

            gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1));
            gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, *(__m512*)_ps512_1), two), _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1));

            __m512 gx_floor = _mm512_roundscale_ps(gx, _MM_FROUND_TO_NEG_INF);
            __m512 gy_floor = _mm512_roundscale_ps(gy, _MM_FROUND_TO_NEG_INF);

            const __m512 tx = _mm512_sub_ps(gx, gx_floor);
            const __m512 ty = _mm512_sub_ps(gy, gy_floor);

            __m512 coefficients[4];

            __m512 gx0 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_n1);
            __m512 gx1 = gx_floor;
            __m512 gx2 = _mm512_add_ps(gx_floor, *(__m512*)_ps512_1);
            __m512 gx3 = _mm512_add_ps(gx_floor, _mm512_set1_ps(2.0f));
            const __m512 v0p5fp16 = _mm512_set1_ps(0.5f);
            {
                // x0
                const __m512 border_x = _mm512_sub_ps(vImgWf, *(__m512*)_ps512_1);

                gx0 = _mm512_and_ps(gx0, *(__m512*)_ps512_inv_sign_mask);
                __m512 reflectx0_v = _mm512_and_ps(_mm512_sub_ps(gx0, border_x), *(__m512*)_ps512_inv_sign_mask);
                gx0 = _mm512_sub_ps(border_x, reflectx0_v);

                // x1
                gx1 = _mm512_and_ps(gx1, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx1_v = _mm512_and_ps(_mm512_sub_ps(gx1, border_x), *(__m512*)_ps512_inv_sign_mask);
                gx1 = _mm512_sub_ps(border_x, reflectx1_v);

                // x2
                gx2 = _mm512_and_ps(gx2, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx2_v = _mm512_and_ps(_mm512_sub_ps(gx2, border_x), *(__m512*)_ps512_inv_sign_mask);
                gx2 = _mm512_sub_ps(border_x, reflectx2_v);

                // x3
                gx3 = _mm512_and_ps(gx3, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx3_v = _mm512_and_ps(_mm512_sub_ps(gx3, border_x), *(__m512*)_ps512_inv_sign_mask);
                gx3 = _mm512_sub_ps(border_x, reflectx3_v);
            }

            __m512i x0 = _mm512_cvtps_epi32(gx0);
            __m512i x1 = _mm512_cvtps_epi32(gx1);
            __m512i x2 = _mm512_cvtps_epi32(gx2);
            __m512i x3 = _mm512_cvtps_epi32(gx3);

            __m512i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm512_add_ps(gy_floor, _mm512_set1_ps(-1.0f + i));

                {
                    //y
                    const __m512 border_y = _mm512_sub_ps(vImgHf, *(__m512*)_ps512_1);

                    gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, border_y), *(__m512*)_ps512_inv_sign_mask);
                    gy = _mm512_sub_ps(border_y, reflecty_v);
                }

                __m512i y = _mm512_cvtps_epi32(gy);

                __m512 v0_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx0), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v1_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx1), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v2_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx2), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                __m512 v3_offset_f = _mm512_add_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(gy, vImgWf), gx3), vElempackf),
                                                   _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                v0_offset[i] = _mm512_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm512_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm512_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm512_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < dst.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v0_offset[i], src.channel(q), sizeof(float));
                    __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v1_offset[i], src.channel(q), sizeof(float));
                    __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v2_offset[i], src.channel(q), sizeof(float));
                    __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, v3_offset[i], src.channel(q), sizeof(float));

                    coefficients[i] = cubic_interp1d_p16(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m512 _v = cubic_interp1d_p16(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}