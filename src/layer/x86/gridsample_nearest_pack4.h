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

static void gridsample_2d_nearest_align0_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
            gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

            __m128i ix = _mm_cvtps_epi32(gx);
            __m128i iy = _mm_cvtps_epi32(gy);

            __m128i v_in_range = _mm_and_si128(_mm_and_si128(_mm_cmpgt_epi32(ix, vn1ip4), _mm_cmpgt_epi32(vImgWi, ix)),
                                               _mm_and_si128(_mm_cmpgt_epi32(iy, vn1ip4), _mm_cmpgt_epi32(vImgHi, iy)));

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
#else
            __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx), vElempackf),
                                       _mm_set_ps(3, 2, 1, 0));
            __m128i i_offset = _mm_cvtps_epi32(offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 _v = mask_gather_ps(src.channel(q), i_offset, *reinterpret_cast<__m128*>(&v_in_range));

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
            gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

            __m128i ix = _mm_cvtps_epi32(gx);
            __m128i iy = _mm_cvtps_epi32(gy);

            __m128i v_in_range = _mm_and_si128(_mm_and_si128(_mm_cmpgt_epi32(ix, vn1ip4), _mm_cmpgt_epi32(vImgWi, ix)),
                                               _mm_and_si128(_mm_cmpgt_epi32(iy, vn1ip4), _mm_cmpgt_epi32(vImgHi, iy)));

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
#else
            __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx), vElempackf),
                                       _mm_set_ps(3, 2, 1, 0));
            __m128i i_offset = _mm_cvtps_epi32(offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 _v = mask_gather_ps(src.channel(q), i_offset, *reinterpret_cast<__m128*>(&v_in_range));

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align0_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
            gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

            __m128i ix = _mm_cvtps_epi32(gx);
            __m128i iy = _mm_cvtps_epi32(gy);

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
#else
            __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx), vElempackf),
                                       _mm_set_ps(3, 2, 1, 0));
            __m128i i_offset = _mm_cvtps_epi32(offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
            gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

            __m128i ix = _mm_cvtps_epi32(gx);
            __m128i iy = _mm_cvtps_epi32(gy);

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
#else
            __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx), vElempackf),
                                       _mm_set_ps(3, 2, 1, 0));
            __m128i i_offset = _mm_cvtps_epi32(offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align0_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m128 two = _mm_set1_ps(2.f);
            gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);
            gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);

            gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
            gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

            // compute coord
            {
                // x
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
                const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_add_ps(gy, v0p5fp4);

                gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps_inv_sign_mask);
                gy = _mm_sub_ps(vImgHf, reflecty_v);

                gy = _mm_sub_ps(gy, v0p5fp4);

                _mm_sub_ps(gy, v0p5fp4);

                gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));
            }

            __m128i ix = _mm_cvtps_epi32(gx);
            __m128i iy = _mm_cvtps_epi32(gy);

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
#else
            __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx), vElempackf),
                                       _mm_set_ps(3, 2, 1, 0));
            __m128i i_offset = _mm_cvtps_epi32(offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m128 two = _mm_set1_ps(2.f);
            gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
            gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));

            gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
            gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                gx = _mm_and_ps(gx, *(__m128*)_ps_inv_sign_mask);

                __m128 reflectx_v = _mm_and_ps(_mm_sub_ps(gx, border_x), *(__m128*)_ps_inv_sign_mask);
                gx = _mm_sub_ps(border_x, reflectx_v);

                // y
                const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps_inv_sign_mask);
                gy = _mm_sub_ps(border_y, reflecty_v);
            }

            __m128i ix = _mm_cvtps_epi32(gx);
            __m128i iy = _mm_cvtps_epi32(gy);

#if (_MSC_VER && __AVX__) || __SSE4_1__
            __m128i i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm_set_epi32(3, 2, 1, 0));
#else
            __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWf), gx), vElempackf),
                                       _mm_set_ps(3, 2, 1, 0));
            __m128i i_offset = _mm_cvtps_epi32(offset);
#endif // __SSE4_1__

            for (int q = 0; q < dst.c; q++)
            {
                __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                _mm_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_3d_nearest_align0_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                gz = floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                __m128i ix = _mm_cvtps_epi32(gx);
                __m128i iy = _mm_cvtps_epi32(gy);
                __m128i iz = _mm_cvtps_epi32(gz);

                __m128i v_in_range = _mm_and_si128(_mm_and_si128(_mm_cmpgt_epi32(ix, vn1ip4), _mm_cmpgt_epi32(vImgWi, ix)),
                                                   _mm_and_si128(_mm_cmpgt_epi32(iy, vn1ip4), _mm_cmpgt_epi32(vImgHi, iy)));
                v_in_range = _mm_and_si128(v_in_range, _mm_and_si128(_mm_cmpgt_epi32(iz, vn1ip4), _mm_cmpgt_epi32(vImgDi, iz)));

                __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), gz),
                                                      _mm_add_ps(_mm_mul_ps(gy, vImgWf), gx)),
                                                      vElempackf),
                                           _mm_set_ps(3, 2, 1, 0));
                __m128i i_offset = _mm_cvtps_epi32(offset);

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 _v = mask_gather_ps(src.channel(q), i_offset, *reinterpret_cast<__m128*>(&v_in_range));

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_zeros_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                gz = floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                __m128i ix = _mm_cvtps_epi32(gx);
                __m128i iy = _mm_cvtps_epi32(gy);
                __m128i iz = _mm_cvtps_epi32(gz);

                __m128i v_in_range = _mm_and_si128(_mm_and_si128(_mm_cmpgt_epi32(ix, vn1ip4), _mm_cmpgt_epi32(vImgWi, ix)),
                                                   _mm_and_si128(_mm_cmpgt_epi32(iy, vn1ip4), _mm_cmpgt_epi32(vImgHi, iy)));
                v_in_range = _mm_and_si128(v_in_range, _mm_and_si128(_mm_cmpgt_epi32(iz, vn1ip4), _mm_cmpgt_epi32(vImgDi, iz)));

                __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), gz),
                                                      _mm_add_ps(_mm_mul_ps(gy, vImgWf), gx)),
                                                      vElempackf),
                                           _mm_set_ps(3, 2, 1, 0));
                __m128i i_offset = _mm_cvtps_epi32(offset);

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 _v = mask_gather_ps(src.channel(q), i_offset, *reinterpret_cast<__m128*>(&v_in_range));

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align0_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                gz = floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                __m128i ix = _mm_cvtps_epi32(gx);
                __m128i iy = _mm_cvtps_epi32(gy);
                __m128i iz = _mm_cvtps_epi32(gz);

                __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), gz),
                                                      _mm_add_ps(_mm_mul_ps(gy, vImgWf), gx)),
                                                      vElempackf),
                                           _mm_set_ps(3, 2, 1, 0));
                __m128i i_offset = _mm_cvtps_epi32(offset);

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_border_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                gz = floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                __m128i ix = _mm_cvtps_epi32(gx);
                __m128i iy = _mm_cvtps_epi32(gy);
                __m128i iz = _mm_cvtps_epi32(gz);

                __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), gz),
                                                      _mm_add_ps(_mm_mul_ps(gy, vImgWf), gx)),
                                                      vElempackf),
                                           _mm_set_ps(3, 2, 1, 0));
                __m128i i_offset = _mm_cvtps_epi32(offset);

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align0_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                const __m128 two = _mm_set1_ps(2.f);
                gx = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gx, v1fp4), vImgWf, v1fp4), two);
                gy = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gy, v1fp4), vImgHf, v1fp4), two);
                gz = _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(gz, v1fp4), vImgDf, v1fp4), two);

                gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                gz = floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                // compute coord
                {
                    // x
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
                    const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_add_ps(gy, v0p5fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, vImgHf), *(__m128*)_ps_inv_sign_mask);
                    gy = _mm_sub_ps(vImgHf, reflecty_v);

                    gy = _mm_sub_ps(gy, v0p5fp4);

                    _mm_sub_ps(gy, v0p5fp4);

                    gy = _mm_min_ps(border_y, _mm_max_ps(gy, _mm_setzero_ps()));

                    // z
                    const __m128 border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_add_ps(gz, v0p5fp4);

                    gz = _mm_and_ps(gz, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectz_v = _mm_and_ps(_mm_sub_ps(gz, vImgDf), *(__m128*)_ps_inv_sign_mask);
                    gz = _mm_sub_ps(vImgDf, reflectz_v);

                    gz = _mm_sub_ps(gz, v0p5fp4);

                    _mm_sub_ps(gz, v0p5fp4);

                    gz = _mm_min_ps(border_z, _mm_max_ps(gz, _mm_setzero_ps()));
                }

                __m128i ix = _mm_cvtps_epi32(gx);
                __m128i iy = _mm_cvtps_epi32(gy);
                __m128i iz = _mm_cvtps_epi32(gz);

                __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), gz),
                                                      _mm_add_ps(_mm_mul_ps(gy, vImgWf), gx)),
                                                      vElempackf),
                                           _mm_set_ps(3, 2, 1, 0));
                __m128i i_offset = _mm_cvtps_epi32(offset);

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_reflection_blob_pack4(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const __m128 vImgWf = _mm_set1_ps(src.w);
    const __m128 vImgHf = _mm_set1_ps(src.h);
    const __m128 vImgDf = _mm_set1_ps(src.d);
    const __m128i vImgWi = _mm_set1_epi32(src.w);
    const __m128i vImgHi = _mm_set1_epi32(src.h);
    const __m128i vImgDi = _mm_set1_epi32(src.d);

    const __m128i vElempacki = _mm_set1_epi32(src.elempack);
    const __m128 vElempackf = _mm_set1_ps(src.elempack);

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

                const __m128 two = _mm_set1_ps(2.f);
                gx = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gx, v1fp4), two), _mm_sub_ps(vImgWf, v1fp4));
                gy = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gy, v1fp4), two), _mm_sub_ps(vImgHf, v1fp4));
                gz = _mm_mul_ps(_mm_div_ps(_mm_add_ps(gz, v1fp4), two), _mm_sub_ps(vImgDf, v1fp4));

                gx = floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                gy = floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                gz = floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                // compute coord
                {
                    // x
                    const __m128 border_x = _mm_sub_ps(vImgWf, v1fp4);

                    gx = _mm_and_ps(gx, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectx_v = _mm_and_ps(_mm_sub_ps(gx, border_x), *(__m128*)_ps_inv_sign_mask);
                    gx = _mm_sub_ps(border_x, reflectx_v);

                    // y
                    const __m128 border_y = _mm_sub_ps(vImgHf, v1fp4);

                    gy = _mm_and_ps(gy, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflecty_v = _mm_and_ps(_mm_sub_ps(gy, border_y), *(__m128*)_ps_inv_sign_mask);
                    gy = _mm_sub_ps(border_y, reflecty_v);

                    // z
                    const __m128 border_z = _mm_sub_ps(vImgDf, v1fp4);

                    gz = _mm_and_ps(gz, *(__m128*)_ps_inv_sign_mask);

                    __m128 reflectz_v = _mm_and_ps(_mm_sub_ps(gz, border_z), *(__m128*)_ps_inv_sign_mask);
                    gz = _mm_sub_ps(border_z, reflectz_v);
                }

                __m128i ix = _mm_cvtps_epi32(gx);
                __m128i iy = _mm_cvtps_epi32(gy);
                __m128i iz = _mm_cvtps_epi32(gz);

                __m128 offset = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(vImgWf, vImgHf), gz),
                                                      _mm_add_ps(_mm_mul_ps(gy, vImgWf), gx)),
                                                      vElempackf),
                                           _mm_set_ps(3, 2, 1, 0));
                __m128i i_offset = _mm_cvtps_epi32(offset);

                for (int q = 0; q < dst.c; q++)
                {
                    __m128 _v = mask_gather_ps(src.channel(q), i_offset, _mm_set1_ps(-1.0f));

                    _mm_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}