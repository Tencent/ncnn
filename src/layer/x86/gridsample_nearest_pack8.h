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

static void gridsample_2d_nearest_align0_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

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

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256i ix = _mm256_cvtps_epi32(gx);
            __m256i iy = _mm256_cvtps_epi32(gy);

            __m256i v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                                  _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));

            __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

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
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256i ix = _mm256_cvtps_epi32(gx);
            __m256i iy = _mm256_cvtps_epi32(gy);

            __m256i v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                                  _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));

            __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align0_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

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

                const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

                const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256i ix = _mm256_cvtps_epi32(gx);
            __m256i iy = _mm256_cvtps_epi32(gy);

            __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

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
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));

                const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

                const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256i ix = _mm256_cvtps_epi32(gx);
            __m256i iy = _mm256_cvtps_epi32(gy);

            __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align0_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

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
            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                gx = _mm256_add_ps(gx, v0p5fp8);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(vImgWf, reflectx_v);

                gx = _mm256_sub_ps(gx, v0p5fp8);

                _mm256_sub_ps(gx, v0p5fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_add_ps(gy, v0p5fp8);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(vImgHf, reflecty_v);

                gy = _mm256_sub_ps(gy, v0p5fp8);

                _mm256_sub_ps(gy, v0p5fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            __m256i ix = _mm256_cvtps_epi32(gx);
            __m256i iy = _mm256_cvtps_epi32(gy);

            __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

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
            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(border_x, reflectx_v);

                // y
                const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(border_y, reflecty_v);
            }

            __m256i ix = _mm256_cvtps_epi32(gx);
            __m256i iy = _mm256_cvtps_epi32(gy);

            __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                _mm256_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_3d_nearest_align0_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);
    const __m256i vImgDi = _mm256_set1_epi32(src.d);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m256 gx = _mm256_set1_ps(gridptr[0]);
                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);
                __m256 gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m256 two = _mm256_set1_ps(2.f);

                    // x
                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);

                    // y
                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

                    // z
                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, v1fp8), vImgDf, v1fp8), two);
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                __m256i ix = _mm256_cvtps_epi32(gx);
                __m256i iy = _mm256_cvtps_epi32(gy);
                __m256i iz = _mm256_cvtps_epi32(gz);

                __m256i v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                                      _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));
                v_in_range = _mm256_and_si256(v_in_range, _mm256_and_si256(_mm256_cmpgt_epi32(iz, vn1ip8), _mm256_cmpgt_epi32(vImgDi, iz)));

                __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                    _mm256_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_zeros_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);
    const __m256i vImgDi = _mm256_set1_epi32(src.d);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m256 gx = _mm256_set1_ps(gridptr[0]);
                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);
                __m256 gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m256 two = _mm256_set1_ps(2.f);

                    // x
                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));

                    // y
                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

                    // z
                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, v1fp8), two), _mm256_sub_ps(vImgDf, v1fp8));
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                __m256i ix = _mm256_cvtps_epi32(gx);
                __m256i iy = _mm256_cvtps_epi32(gy);
                __m256i iz = _mm256_cvtps_epi32(gz);

                __m256i v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                                      _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));
                v_in_range = _mm256_and_si256(v_in_range, _mm256_and_si256(_mm256_cmpgt_epi32(iz, vn1ip8), _mm256_cmpgt_epi32(vImgDi, iz)));

                __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                    _mm256_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align0_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);
    const __m256i vImgDi = _mm256_set1_epi32(src.d);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m256 gx = _mm256_set1_ps(gridptr[0]);
                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);
                __m256 gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m256 two = _mm256_set1_ps(2.f);

                    // x
                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);

                    const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                    // y
                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);

                    const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                    // z
                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, v1fp8), vImgDf, v1fp8), two);

                    const __m256 border_z = _mm256_sub_ps(vImgDf, v1fp8);

                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                __m256i ix = _mm256_cvtps_epi32(gx);
                __m256i iy = _mm256_cvtps_epi32(gy);
                __m256i iz = _mm256_cvtps_epi32(gz);

                __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                    _mm256_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_border_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);
    const __m256i vImgDi = _mm256_set1_epi32(src.d);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m256 gx = _mm256_set1_ps(gridptr[0]);
                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);
                __m256 gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                // compute coord
                {
                    const __m256 two = _mm256_set1_ps(2.f);

                    // x
                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));

                    const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                    // y
                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));

                    const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                    // z
                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, v1fp8), two), _mm256_sub_ps(vImgDf, v1fp8));

                    const __m256 border_z = _mm256_sub_ps(vImgDf, v1fp8);

                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                __m256i ix = _mm256_cvtps_epi32(gx);
                __m256i iy = _mm256_cvtps_epi32(gy);
                __m256i iz = _mm256_cvtps_epi32(gz);

                __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                    _mm256_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align0_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);
    const __m256i vImgDi = _mm256_set1_epi32(src.d);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m256 gx = _mm256_set1_ps(gridptr[0]);
                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);
                __m256 gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                const __m256 two = _mm256_set1_ps(2.f);
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, v1fp8), vImgWf, v1fp8), two);
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, v1fp8), vImgHf, v1fp8), two);
                gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, v1fp8), vImgDf, v1fp8), two);

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                // compute coord
                {
                    // x
                    const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                    __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                    gx = _mm256_add_ps(gx, v0p5fp8);

                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                    gx = _mm256_sub_ps(vImgWf, reflectx_v);

                    gx = _mm256_sub_ps(gx, v0p5fp8);

                    _mm256_sub_ps(gx, v0p5fp8);

                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                    // y
                    const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                    gy = _mm256_add_ps(gy, v0p5fp8);

                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                    gy = _mm256_sub_ps(gy, v0p5fp8);

                    _mm256_sub_ps(gy, v0p5fp8);

                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                    // z
                    const __m256 border_z = _mm256_sub_ps(vImgDf, v1fp8);

                    gz = _mm256_add_ps(gz, v0p5fp8);

                    gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, vImgDf), *(__m256*)_ps256_inv_sign_mask);
                    gz = _mm256_sub_ps(vImgDf, reflectz_v);

                    gz = _mm256_sub_ps(gz, v0p5fp8);

                    _mm256_sub_ps(gz, v0p5fp8);

                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                }

                __m256i ix = _mm256_cvtps_epi32(gx);
                __m256i iy = _mm256_cvtps_epi32(gy);
                __m256i iz = _mm256_cvtps_epi32(gz);

                __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                    _mm256_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_reflection_blob_pack8(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
    const __m256i vImgWi = _mm256_set1_epi32(src.w);
    const __m256i vImgHi = _mm256_set1_epi32(src.h);
    const __m256i vImgDi = _mm256_set1_epi32(src.d);

    const __m256i vElempacki = _mm256_set1_epi32(src.elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < dst.d; z++)
    {
        for (int y = 0; y < dst.h; y++)
        {
            for (int x = 0; x < dst.w; x++)
            {
                //grid tensor has been packed
                const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;
                __m256 gx = _mm256_set1_ps(gridptr[0]);
                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);
                __m256 gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                const __m256 two = _mm256_set1_ps(2.f);
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, v1fp8), two), _mm256_sub_ps(vImgWf, v1fp8));
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, v1fp8), two), _mm256_sub_ps(vImgHf, v1fp8));
                gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, v1fp8), two), _mm256_sub_ps(vImgDf, v1fp8));

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                // compute coord
                {
                    // x
                    const __m256 border_x = _mm256_sub_ps(vImgWf, v1fp8);

                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                    gx = _mm256_sub_ps(border_x, reflectx_v);

                    // y
                    const __m256 border_y = _mm256_sub_ps(vImgHf, v1fp8);

                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                    gy = _mm256_sub_ps(border_y, reflecty_v);

                    // z
                    const __m256 border_z = _mm256_sub_ps(vImgDf, v1fp8);

                    gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, border_z), *(__m256*)_ps256_inv_sign_mask);
                    gz = _mm256_sub_ps(border_z, reflectz_v);
                }

                __m256i ix = _mm256_cvtps_epi32(gx);
                __m256i iy = _mm256_cvtps_epi32(gy);
                __m256i iz = _mm256_cvtps_epi32(gz);

                __m256i i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m256 _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src.channel(q), i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                    _mm256_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}