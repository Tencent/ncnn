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

static void gridsample_nearest_align0_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const auto vImgWf = _mm256_set1_ps(src.w);
    const auto vImgHf = _mm256_set1_ps(src.h);
    const auto vImgWi = _mm256_set1_epi32(src.w);
    const auto vImgHi = _mm256_set1_epi32(src.h);

    const auto vElempacki = _mm256_set1_epi32(src.elempack);

#pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm256_set1_ps(gridptr[0]);
            auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm256_set1_ps(2.f);

                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            auto ix = _mm256_cvtps_epi32(gx);
            auto iy = _mm256_cvtps_epi32(gy);

            auto v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));

            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_nearest_align1_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const auto vImgWf = _mm256_set1_ps(src.w);
    const auto vImgHf = _mm256_set1_ps(src.h);
    const auto vImgWi = _mm256_set1_epi32(src.w);
    const auto vImgHi = _mm256_set1_epi32(src.h);

    const auto vElempacki = _mm256_set1_epi32(src.elempack);

#pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm256_set1_ps(gridptr[0]);
            auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm256_set1_ps(2.f);

                // x
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            auto ix = _mm256_cvtps_epi32(gx);
            auto iy = _mm256_cvtps_epi32(gy);

            auto v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));

            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_nearest_align0_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const auto vImgWf = _mm256_set1_ps(src.w);
    const auto vImgHf = _mm256_set1_ps(src.h);
    const auto vImgWi = _mm256_set1_epi32(src.w);
    const auto vImgHi = _mm256_set1_epi32(src.h);

    const auto vElempacki = _mm256_set1_epi32(src.elempack);

#pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm256_set1_ps(gridptr[0]);
            auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm256_set1_ps(2.f);

                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);

                const auto border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));


                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

                const auto border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            auto ix = _mm256_cvtps_epi32(gx);
            auto iy = _mm256_cvtps_epi32(gy);

            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_nearest_align1_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const auto vImgWf = _mm256_set1_ps(src.w);
    const auto vImgHf = _mm256_set1_ps(src.h);
    const auto vImgWi = _mm256_set1_epi32(src.w);
    const auto vImgHi = _mm256_set1_epi32(src.h);

    const auto vElempacki = _mm256_set1_epi32(src.elempack);

#pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm256_set1_ps(gridptr[0]);
            auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

            // compute coord
            {
                const auto two = _mm256_set1_ps(2.f);

                // x
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));

                const auto border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));


                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

                const auto border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            auto ix = _mm256_cvtps_epi32(gx);
            auto iy = _mm256_cvtps_epi32(gy);

            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_nearest_align0_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const auto vImgWf = _mm256_set1_ps(src.w);
    const auto vImgHf = _mm256_set1_ps(src.h);
    const auto vImgWi = _mm256_set1_epi32(src.w);
    const auto vImgHi = _mm256_set1_epi32(src.h);

    const auto vElempacki = _mm256_set1_epi32(src.elempack);

#pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm256_set1_ps(gridptr[0]);
            auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

            const auto two = _mm256_set1_ps(2.f);
            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const auto border_x = _mm256_sub_ps(vImgWf, v1fp8);

                auto v0p5fp8 = _mm256_set1_ps(0.5f);
                gx = _mm256_add_ps(gx, v0p5fp8);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                auto reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(vImgWf, reflectx_v);

                gx = _mm256_sub_ps(gx, v0p5fp8);

                _mm256_sub_ps(gx, v0p5fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));


                // y
                const auto border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_add_ps(gy, v0p5fp8);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                auto reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(vImgHf, reflecty_v);

                gy = _mm256_sub_ps(gy, v0p5fp8);

                _mm256_sub_ps(gy, v0p5fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            auto ix = _mm256_cvtps_epi32(gx);
            auto iy = _mm256_cvtps_epi32(gy);

            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_nearest_align1_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    float* outptr = static_cast<float*>(dst.data);

    const auto vImgWf = _mm256_set1_ps(src.w);
    const auto vImgHf = _mm256_set1_ps(src.h);
    const auto vImgWi = _mm256_set1_epi32(src.w);
    const auto vImgHi = _mm256_set1_epi32(src.h);

    const auto vElempacki = _mm256_set1_epi32(src.elempack);

#pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < dst.h; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            //grid tensor has been packed
            const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
            auto gx = _mm256_set1_ps(gridptr[0]);
            auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

            const auto two = _mm256_set1_ps(2.f);
            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const auto border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                auto reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(border_x, reflectx_v);


                // y
                const auto border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                auto reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(border_y, reflecty_v);
            }

            auto ix = _mm256_cvtps_epi32(gx);
            auto iy = _mm256_cvtps_epi32(gy);

            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}