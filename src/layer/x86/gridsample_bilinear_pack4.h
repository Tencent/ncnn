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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm_set1_ps(gridptr[0]);
            auto gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm_set1_ps(2.f);

                // x
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                // y
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);
            }

            auto x_w = _mm_floor_ps(gx);
            auto y_n = _mm_floor_ps(gy);

            auto w = _mm_sub_ps(gx, x_w);
            auto e = _mm_sub_ps(v1fp4, w);
            auto n = _mm_sub_ps(gy, y_n);
            auto s = _mm_sub_ps(v1fp4, n);

            auto nw = _mm_mul_ps(s, e);
            auto ne = _mm_mul_ps(s, w);
            auto sw = _mm_mul_ps(n, e);
            auto se = _mm_mul_ps(n, w);

            auto x0 = _mm_cvtps_epi32(x_w);
            auto x1 = _mm_add_epi32(x0, v1ip4);
            auto y0 = _mm_cvtps_epi32(y_n);
            auto y1 = _mm_add_epi32(y0, v1ip4);

            auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
            auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            auto y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
            auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            auto v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
            auto v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
            auto v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
            auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(8)
            auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
            auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                auto nw_val = mask_gather_ps(src.channel(q), i_nw_offset, *reinterpret_cast<__m128*>(&v00_in_range));
                auto ne_val = mask_gather_ps(src.channel(q), i_ne_offset, *reinterpret_cast<__m128*>(&v10_in_range));
                auto sw_val = mask_gather_ps(src.channel(q), i_sw_offset, *reinterpret_cast<__m128*>(&v01_in_range));
                auto se_val = mask_gather_ps(src.channel(q), i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                auto _v = _mm_mul_ps(nw_val, nw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm_set1_ps(gridptr[0]);
            auto gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm_set1_ps(2.f);

                // x
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                // y
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));
            }

            auto x_w = _mm_floor_ps(gx);
            auto y_n = _mm_floor_ps(gy);

            auto w = _mm_sub_ps(gx, x_w);
            auto e = _mm_sub_ps(v1fp4, w);
            auto n = _mm_sub_ps(gy, y_n);
            auto s = _mm_sub_ps(v1fp4, n);

            auto nw = _mm_mul_ps(s, e);
            auto ne = _mm_mul_ps(s, w);
            auto sw = _mm_mul_ps(n, e);
            auto se = _mm_mul_ps(n, w);

            auto x0 = _mm_cvtps_epi32(x_w);
            auto x1 = _mm_add_epi32(x0, v1ip4);
            auto y0 = _mm_cvtps_epi32(y_n);
            auto y1 = _mm_add_epi32(y0, v1ip4);

            auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
            auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            auto y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
            auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            auto v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
            auto v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
            auto v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
            auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(8)
            auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
            auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                auto nw_val = mask_gather_ps(src.channel(q), i_nw_offset, *reinterpret_cast<__m128*>(&v00_in_range));
                auto ne_val = mask_gather_ps(src.channel(q), i_ne_offset, *reinterpret_cast<__m128*>(&v10_in_range));
                auto sw_val = mask_gather_ps(src.channel(q), i_sw_offset, *reinterpret_cast<__m128*>(&v01_in_range));
                auto se_val = mask_gather_ps(src.channel(q), i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                auto _v = _mm_mul_ps(nw_val, nw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm_set1_ps(gridptr[0]);
            auto gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm_set1_ps(2.f);

                // x
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                // y
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

            auto x_w = _mm_floor_ps(gx);
            auto y_n = _mm_floor_ps(gy);

            auto w = _mm_sub_ps(gx, x_w);
            auto e = _mm_sub_ps(v1fp4, w);
            auto n = _mm_sub_ps(gy, y_n);
            auto s = _mm_sub_ps(v1fp4, n);

            auto nw = _mm_mul_ps(s, e);
            auto ne = _mm_mul_ps(s, w);
            auto sw = _mm_mul_ps(n, e);
            auto se = _mm_mul_ps(n, w);

            auto x0 = _mm_cvtps_epi32(x_w);
            auto x1 = _mm_add_epi32(x0, v1ip4);
            auto y0 = _mm_cvtps_epi32(y_n);
            auto y1 = _mm_add_epi32(y0, v1ip4);

            auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(8)
            auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
            auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                auto nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                auto ne_val = mask_gather_ps(src.channel(q), i_ne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                auto sw_val = mask_gather_ps(src.channel(q), i_sw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                auto se_val = mask_gather_ps(src.channel(q), i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                auto _v = _mm_mul_ps(nw_val, nw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm_set1_ps(gridptr[0]);
            auto gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm_set1_ps(2.f);

                // x
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                // y
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

            auto x_w = _mm_floor_ps(gx);
            auto y_n = _mm_floor_ps(gy);

            auto w = _mm_sub_ps(gx, x_w);
            auto e = _mm_sub_ps(v1fp4, w);
            auto n = _mm_sub_ps(gy, y_n);
            auto s = _mm_sub_ps(v1fp4, n);

            auto nw = _mm_mul_ps(s, e);
            auto ne = _mm_mul_ps(s, w);
            auto sw = _mm_mul_ps(n, e);
            auto se = _mm_mul_ps(n, w);

            auto x0 = _mm_cvtps_epi32(x_w);
            auto x1 = _mm_add_epi32(x0, v1ip4);
            auto y0 = _mm_cvtps_epi32(y_n);
            auto y1 = _mm_add_epi32(y0, v1ip4);

            auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(8)
            auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
            auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                auto nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                auto ne_val = mask_gather_ps(src.channel(q), i_ne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                auto sw_val = mask_gather_ps(src.channel(q), i_sw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                auto se_val = mask_gather_ps(src.channel(q), i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                auto _v = _mm_mul_ps(nw_val, nw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm_set1_ps(gridptr[0]);
            auto gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm_set1_ps(2.f);

                // x
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                auto v0p5fp4 = _mm_set1_ps(0.5f);
                gx = _mm_add_ps(gx, v0p5fp4);

                gx = _mm_and_ps(gx, *(__m128*)_ps256_inv_sign_mask);

                auto reflectx_v = _mm_and_ps(_mm_sub_ps(gx, vImgWf), *(__m128*)_ps256_inv_sign_mask);
                gx = _mm_sub_ps(vImgWf, reflectx_v);

                gx = _mm_sub_ps(gx, v0p5fp4);

                _mm_sub_ps(gx, v0p5fp4);

                gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                // y
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_add_ps(gy, v0p5fp4);

                gy = _mm_and_ps(gy, *(__m128*)_ps256_inv_sign_mask);

                auto reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps256_inv_sign_mask);
                gy = _mm_sub_ps(vImgHf, reflecty_v);

                gy = _mm_sub_ps(gy, v0p5fp4);

                _mm_sub_ps(gy, v0p5fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

            auto x_w = _mm_floor_ps(gx);
            auto y_n = _mm_floor_ps(gy);

            auto w = _mm_sub_ps(gx, x_w);
            auto e = _mm_sub_ps(v1fp4, w);
            auto n = _mm_sub_ps(gy, y_n);
            auto s = _mm_sub_ps(v1fp4, n);

            auto nw = _mm_mul_ps(s, e);
            auto ne = _mm_mul_ps(s, w);
            auto sw = _mm_mul_ps(n, e);
            auto se = _mm_mul_ps(n, w);

            auto x0 = _mm_cvtps_epi32(x_w);
            auto x1 = _mm_add_epi32(x0, v1ip4);
            auto y0 = _mm_cvtps_epi32(y_n);
            auto y1 = _mm_add_epi32(y0, v1ip4);

            auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(8)
            auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
            auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                auto nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                auto ne_val = mask_gather_ps(src.channel(q), i_ne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                auto sw_val = mask_gather_ps(src.channel(q), i_sw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                auto se_val = mask_gather_ps(src.channel(q), i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                auto _v = _mm_mul_ps(nw_val, nw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm_set1_ps(gridptr[0]);
            auto gy = _mm_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm_set1_ps(2.f);

                // x
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_and_ps(gx, *(__m128*)_ps256_inv_sign_mask);

                auto reflectx_v = _mm_and_ps(_mm_sub_ps(gx, border_x), *(__m128*)_ps256_inv_sign_mask);
                gx = _mm_sub_ps(border_x, reflectx_v);

                // y
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_and_ps(gy, *(__m128*)_ps256_inv_sign_mask);

                auto reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps256_inv_sign_mask);
                gy = _mm_sub_ps(border_y, reflecty_v);
            }

            auto x_w = _mm_floor_ps(gx);
            auto y_n = _mm_floor_ps(gy);

            auto w = _mm_sub_ps(gx, x_w);
            auto e = _mm_sub_ps(v1fp4, w);
            auto n = _mm_sub_ps(gy, y_n);
            auto s = _mm_sub_ps(v1fp4, n);

            auto nw = _mm_mul_ps(s, e);
            auto ne = _mm_mul_ps(s, w);
            auto sw = _mm_mul_ps(n, e);
            auto se = _mm_mul_ps(n, w);

            auto x0 = _mm_cvtps_epi32(x_w);
            auto x1 = _mm_add_epi32(x0, v1ip4);
            auto y0 = _mm_cvtps_epi32(y_n);
            auto y1 = _mm_add_epi32(y0, v1ip4);

            auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
            auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));

            auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

            // (W*y + x) * elempack + vec(8)
            auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
            auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
            auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
            auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

            for (int q = 0; q < dst.c; q++)
            {
                auto nw_val = mask_gather_ps(src.channel(q), i_nw_offset, vn1fp4);
                auto ne_val = mask_gather_ps(src.channel(q), i_ne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                auto sw_val = mask_gather_ps(src.channel(q), i_sw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                auto se_val = mask_gather_ps(src.channel(q), i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                auto _v = _mm_mul_ps(nw_val, nw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgDf = _mm_set1_ps(src.d);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);
    const auto vImgDi = _mm_set1_epi32(src.d);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                auto gx = _mm_set1_ps(gridptr[0]);
                auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const auto two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                    // y
                    gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                    // z
                    gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);
                }

                auto x_w = _mm_floor_ps(gx);
                auto y_n = _mm_floor_ps(gy);
                auto z_t = _mm_floor_ps(gz);

                auto w = _mm_sub_ps(gx, x_w);
                auto e = _mm_sub_ps(v1fp4, w);
                auto n = _mm_sub_ps(gy, y_n);
                auto s = _mm_sub_ps(v1fp4, n);
                auto t = _mm_sub_ps(gz, z_t);
                auto b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    auto nw = _mm_mul_ps(s, e);
                    auto ne = _mm_mul_ps(s, w);
                    auto sw = _mm_mul_ps(n, e);
                    auto se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                auto x0 = _mm_cvtps_epi32(x_w);
                auto x1 = _mm_add_epi32(x0, v1ip4);
                auto y0 = _mm_cvtps_epi32(y_n);
                auto y1 = _mm_add_epi32(y0, v1ip4);
                auto z0 = _mm_cvtps_epi32(z_t);
                auto z1 = _mm_add_epi32(z0, v1ip4);

                auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                auto y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                auto z0_in_range = _mm_and_si128(_mm_cmpgt_epi32(z0, vn1ip4), _mm_cmpgt_epi32(vImgDi, z0));
                auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    auto v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
                    auto v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
                    auto v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
                    auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v000_in_range = _mm_and_si128(v00_in_range, z0_in_range);
                    v010_in_range = _mm_and_si128(v01_in_range, z0_in_range);
                    v100_in_range = _mm_and_si128(v10_in_range, z0_in_range);
                    v110_in_range = _mm_and_si128(v11_in_range, z0_in_range);

                    v001_in_range = _mm_and_si128(v00_in_range, z1_in_range);
                    v011_in_range = _mm_and_si128(v01_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(v10_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    auto tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, *reinterpret_cast<__m128*>(&v000_in_range));
                    auto tne_val = mask_gather_ps(src.channel(q), i_tne_offset, *reinterpret_cast<__m128*>(&v100_in_range));
                    auto tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, *reinterpret_cast<__m128*>(&v010_in_range));
                    auto tse_val = mask_gather_ps(src.channel(q), i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                    auto bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, *reinterpret_cast<__m128*>(&v001_in_range));
                    auto bne_val = mask_gather_ps(src.channel(q), i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                    auto bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                    auto bse_val = mask_gather_ps(src.channel(q), i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                    auto _v = _mm_mul_ps(tnw_val, tnw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgDf = _mm_set1_ps(src.d);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);
    const auto vImgDi = _mm_set1_epi32(src.d);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                auto gx = _mm_set1_ps(gridptr[0]);
                auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const auto two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                    // y
                    gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                    // z
                    gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));
                }

                auto x_w = _mm_floor_ps(gx);
                auto y_n = _mm_floor_ps(gy);
                auto z_t = _mm_floor_ps(gz);

                auto w = _mm_sub_ps(gx, x_w);
                auto e = _mm_sub_ps(v1fp4, w);
                auto n = _mm_sub_ps(gy, y_n);
                auto s = _mm_sub_ps(v1fp4, n);
                auto t = _mm_sub_ps(gz, z_t);
                auto b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    auto nw = _mm_mul_ps(s, e);
                    auto ne = _mm_mul_ps(s, w);
                    auto sw = _mm_mul_ps(n, e);
                    auto se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                auto x0 = _mm_cvtps_epi32(x_w);
                auto x1 = _mm_add_epi32(x0, v1ip4);
                auto y0 = _mm_cvtps_epi32(y_n);
                auto y1 = _mm_add_epi32(y0, v1ip4);
                auto z0 = _mm_cvtps_epi32(z_t);
                auto z1 = _mm_add_epi32(z0, v1ip4);

                auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWi, x0));
                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                auto y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHi, y0));
                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                auto z0_in_range = _mm_and_si128(_mm_cmpgt_epi32(z0, vn1ip4), _mm_cmpgt_epi32(vImgDi, z0));
                auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    auto v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
                    auto v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
                    auto v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
                    auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v000_in_range = _mm_and_si128(v00_in_range, z0_in_range);
                    v010_in_range = _mm_and_si128(v01_in_range, z0_in_range);
                    v100_in_range = _mm_and_si128(v10_in_range, z0_in_range);
                    v110_in_range = _mm_and_si128(v11_in_range, z0_in_range);

                    v001_in_range = _mm_and_si128(v00_in_range, z1_in_range);
                    v011_in_range = _mm_and_si128(v01_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(v10_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    auto tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, *reinterpret_cast<__m128*>(&v000_in_range));
                    auto tne_val = mask_gather_ps(src.channel(q), i_tne_offset, *reinterpret_cast<__m128*>(&v100_in_range));
                    auto tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, *reinterpret_cast<__m128*>(&v010_in_range));
                    auto tse_val = mask_gather_ps(src.channel(q), i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                    auto bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, *reinterpret_cast<__m128*>(&v001_in_range));
                    auto bne_val = mask_gather_ps(src.channel(q), i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                    auto bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                    auto bse_val = mask_gather_ps(src.channel(q), i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                    auto _v = _mm_mul_ps(tnw_val, tnw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgDf = _mm_set1_ps(src.d);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);
    const auto vImgDi = _mm_set1_epi32(src.d);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                auto gx = _mm_set1_ps(gridptr[0]);
                auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const auto two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);

                    const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                    // y
                    gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

                    const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);

                    const auto border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                auto x_w = _mm_floor_ps(gx);
                auto y_n = _mm_floor_ps(gy);
                auto z_t = _mm_floor_ps(gz);

                auto w = _mm_sub_ps(gx, x_w);
                auto e = _mm_sub_ps(v1fp4, w);
                auto n = _mm_sub_ps(gy, y_n);
                auto s = _mm_sub_ps(v1fp4, n);
                auto t = _mm_sub_ps(gz, z_t);
                auto b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    auto nw = _mm_mul_ps(s, e);
                    auto ne = _mm_mul_ps(s, w);
                    auto sw = _mm_mul_ps(n, e);
                    auto se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                auto x0 = _mm_cvtps_epi32(x_w);
                auto x1 = _mm_add_epi32(x0, v1ip4);
                auto y0 = _mm_cvtps_epi32(y_n);
                auto y1 = _mm_add_epi32(y0, v1ip4);
                auto z0 = _mm_cvtps_epi32(z_t);
                auto z1 = _mm_add_epi32(z0, v1ip4);

                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    auto tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    auto tne_val = mask_gather_ps(src.channel(q), i_tne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                    auto tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                    auto tse_val = mask_gather_ps(src.channel(q), i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                    auto bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, *reinterpret_cast<__m128*>(&z1_in_range));
                    auto bne_val = mask_gather_ps(src.channel(q), i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                    auto bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                    auto bse_val = mask_gather_ps(src.channel(q), i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                    auto _v = _mm_mul_ps(tnw_val, tnw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgDf = _mm_set1_ps(src.d);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);
    const auto vImgDi = _mm_set1_epi32(src.d);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                auto gx = _mm_set1_ps(gridptr[0]);
                auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const auto two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));

                    const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                    // y
                    gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

                    const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));

                    const auto border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                auto x_w = _mm_floor_ps(gx);
                auto y_n = _mm_floor_ps(gy);
                auto z_t = _mm_floor_ps(gz);

                auto w = _mm_sub_ps(gx, x_w);
                auto e = _mm_sub_ps(v1fp4, w);
                auto n = _mm_sub_ps(gy, y_n);
                auto s = _mm_sub_ps(v1fp4, n);
                auto t = _mm_sub_ps(gz, z_t);
                auto b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    auto nw = _mm_mul_ps(s, e);
                    auto ne = _mm_mul_ps(s, w);
                    auto sw = _mm_mul_ps(n, e);
                    auto se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                auto x0 = _mm_cvtps_epi32(x_w);
                auto x1 = _mm_add_epi32(x0, v1ip4);
                auto y0 = _mm_cvtps_epi32(y_n);
                auto y1 = _mm_add_epi32(y0, v1ip4);
                auto z0 = _mm_cvtps_epi32(z_t);
                auto z1 = _mm_add_epi32(z0, v1ip4);

                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    auto tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    auto tne_val = mask_gather_ps(src.channel(q), i_tne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                    auto tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                    auto tse_val = mask_gather_ps(src.channel(q), i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                    auto bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, *reinterpret_cast<__m128*>(&z1_in_range));
                    auto bne_val = mask_gather_ps(src.channel(q), i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                    auto bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                    auto bse_val = mask_gather_ps(src.channel(q), i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                    auto _v = _mm_mul_ps(tnw_val, tnw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgDf = _mm_set1_ps(src.d);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);
    const auto vImgDi = _mm_set1_epi32(src.d);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                auto gx = _mm_set1_ps(gridptr[0]);
                auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const auto two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);
                    const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                    auto v0p5fp4 = _mm_set1_ps(0.5f);
                    gx = _mm_add_ps(gx, v0p5fp4);

                    gx = _mm_and_ps(gx, *(__m128*)_ps256_inv_sign_mask);

                    auto reflectx_v = _mm_and_ps(_mm_sub_ps(gx, vImgWf), *(__m128*)_ps256_inv_sign_mask);
                    gx = _mm_sub_ps(vImgWf, reflectx_v);

                    gx = _mm_sub_ps(gx, v0p5fp4);

                    _mm_sub_ps(gx, v0p5fp4);

                    gx = _mm_min_ps(border_x, _mm_max_ps(gx, _mm_setzero_ps()));

                    // y
                    gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);
                    const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_add_ps(gy, v0p5fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps256_inv_sign_mask);

                    auto reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps256_inv_sign_mask);
                    gy = _mm_sub_ps(vImgHf, reflecty_v);

                    gy = _mm_sub_ps(gy, v0p5fp4);

                    _mm_sub_ps(gy, v0p5fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);
                    const auto border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_add_ps(gz, v0p5fp4);

                    gz = _mm_and_ps(gz, *(__m128*)_ps256_inv_sign_mask);

                    auto reflectz_v = _mm_and_ps(_mm_sub_ps(gz, vImgDf), *(__m128*)_ps256_inv_sign_mask);
                    gz = _mm_sub_ps(vImgDf, reflectz_v);

                    gz = _mm_sub_ps(gz, v0p5fp4);

                    _mm_sub_ps(gz, v0p5fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                auto x_w = _mm_floor_ps(gx);
                auto y_n = _mm_floor_ps(gy);
                auto z_t = _mm_floor_ps(gz);

                auto w = _mm_sub_ps(gx, x_w);
                auto e = _mm_sub_ps(v1fp4, w);
                auto n = _mm_sub_ps(gy, y_n);
                auto s = _mm_sub_ps(v1fp4, n);
                auto t = _mm_sub_ps(gz, z_t);
                auto b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    auto nw = _mm_mul_ps(s, e);
                    auto ne = _mm_mul_ps(s, w);
                    auto sw = _mm_mul_ps(n, e);
                    auto se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                auto x0 = _mm_cvtps_epi32(x_w);
                auto x1 = _mm_add_epi32(x0, v1ip4);
                auto y0 = _mm_cvtps_epi32(y_n);
                auto y1 = _mm_add_epi32(y0, v1ip4);
                auto z0 = _mm_cvtps_epi32(z_t);
                auto z1 = _mm_add_epi32(z0, v1ip4);

                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    auto tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    auto tne_val = mask_gather_ps(src.channel(q), i_tne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                    auto tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                    auto tse_val = mask_gather_ps(src.channel(q), i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                    auto bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, *reinterpret_cast<__m128*>(&z1_in_range));
                    auto bne_val = mask_gather_ps(src.channel(q), i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                    auto bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                    auto bse_val = mask_gather_ps(src.channel(q), i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                    auto _v = _mm_mul_ps(tnw_val, tnw);
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
    const auto vImgWf = _mm_set1_ps(src.w);
    const auto vImgHf = _mm_set1_ps(src.h);
    const auto vImgDf = _mm_set1_ps(src.d);
    const auto vImgWi = _mm_set1_epi32(src.w);
    const auto vImgHi = _mm_set1_epi32(src.h);
    const auto vImgDi = _mm_set1_epi32(src.d);

    const auto vElempacki = _mm_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                auto gx = _mm_set1_ps(gridptr[0]);
                auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const auto two = _mm_set1_ps(2.f);

                    // x
                    gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
                    const auto border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_and_ps(gx, *(__m128*)_ps256_inv_sign_mask);

                    auto reflectx_v = _mm_and_ps(_mm_sub_ps(gx, border_x), *(__m128*)_ps256_inv_sign_mask);
                    gx = _mm_sub_ps(border_x, reflectx_v);

                    // y
                    gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));
                    const auto border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps256_inv_sign_mask);

                    auto reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps256_inv_sign_mask);
                    gy = _mm_sub_ps(border_y, reflecty_v);

                    // z
                    gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));
                    const auto border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_and_ps(gz, *(__m128*)_ps256_inv_sign_mask);

                    auto reflectz_v = _mm_and_ps(_mm_sub_ps(gz, border_z), *(__m128*)_ps256_inv_sign_mask);
                    gz = _mm_sub_ps(border_z, reflectz_v);
                }

                auto x_w = _mm_floor_ps(gx);
                auto y_n = _mm_floor_ps(gy);
                auto z_t = _mm_floor_ps(gz);

                auto w = _mm_sub_ps(gx, x_w);
                auto e = _mm_sub_ps(v1fp4, w);
                auto n = _mm_sub_ps(gy, y_n);
                auto s = _mm_sub_ps(v1fp4, n);
                auto t = _mm_sub_ps(gz, z_t);
                auto b = _mm_sub_ps(v1fp4, t);

                __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                {
                    auto nw = _mm_mul_ps(s, e);
                    auto ne = _mm_mul_ps(s, w);
                    auto sw = _mm_mul_ps(n, e);
                    auto se = _mm_mul_ps(n, w);

                    tnw = _mm_mul_ps(b, nw);
                    tne = _mm_mul_ps(b, ne);
                    tsw = _mm_mul_ps(b, sw);
                    tse = _mm_mul_ps(b, se);

                    bnw = _mm_mul_ps(t, nw);
                    bne = _mm_mul_ps(t, ne);
                    bsw = _mm_mul_ps(t, sw);
                    bse = _mm_mul_ps(t, se);
                }

                auto x0 = _mm_cvtps_epi32(x_w);
                auto x1 = _mm_add_epi32(x0, v1ip4);
                auto y0 = _mm_cvtps_epi32(y_n);
                auto y1 = _mm_add_epi32(y0, v1ip4);
                auto z0 = _mm_cvtps_epi32(z_t);
                auto z1 = _mm_add_epi32(z0, v1ip4);

                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWi, x1));
                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHi, y1));
                auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDi, z1));

                __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                    v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                    v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                    v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                }

                // (W*H*z + W*y + x) * elempack + vec(8)
                auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWi, vElempacki));
                auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                for (int q = 0; q < dst.c; q++)
                {
                    auto tnw_val = mask_gather_ps(src.channel(q), i_tnw_offset, vn1fp4);
                    auto tne_val = mask_gather_ps(src.channel(q), i_tne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                    auto tsw_val = mask_gather_ps(src.channel(q), i_tsw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                    auto tse_val = mask_gather_ps(src.channel(q), i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                    auto bnw_val = mask_gather_ps(src.channel(q), i_bnw_offset, *reinterpret_cast<__m128*>(&z1_in_range));
                    auto bne_val = mask_gather_ps(src.channel(q), i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                    auto bsw_val = mask_gather_ps(src.channel(q), i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                    auto bse_val = mask_gather_ps(src.channel(q), i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                    auto _v = _mm_mul_ps(tnw_val, tnw);
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