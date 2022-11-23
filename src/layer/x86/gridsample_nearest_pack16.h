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

static void gridsample_2d_nearest_align0_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, v1fp16), vImgWf, v1fp16), two);

                // y
                gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, v1fp16), vImgHf, v1fp16), two);
            }

            gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
            gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));

            __m512i ix = _mm512_cvtps_epi32(gx);
            __m512i iy = _mm512_cvtps_epi32(gy);

            __mmask16 v_in_range = (_mm512_cmpgt_epi32_mask(ix, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgWi, ix)) & (_mm512_cmpgt_epi32_mask(iy, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgHi, iy));

            __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v_in_range, i_offset, src.channel(q), sizeof(float));

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, v1fp16), two), _mm512_sub_ps(vImgWf, v1fp16));

                // y
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, v1fp16), two), _mm512_sub_ps(vImgHf, v1fp16));
            }

            gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
            gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));

            __m512i ix = _mm512_cvtps_epi32(gx);
            __m512i iy = _mm512_cvtps_epi32(gy);

            __mmask16 v_in_range = (_mm512_cmpgt_epi32_mask(ix, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgWi, ix)) & (_mm512_cmpgt_epi32_mask(iy, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgHi, iy));


            __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v_in_range, i_offset, src.channel(q), sizeof(float));

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align0_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, v1fp16), vImgWf, v1fp16), two);

                const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                // y
                gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, v1fp16), vImgHf, v1fp16), two);

                const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
            }

            gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
            gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));

            __m512i ix = _mm512_cvtps_epi32(gx);
            __m512i iy = _mm512_cvtps_epi32(gy);

            __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, v1fp16), two), _mm512_sub_ps(vImgWf, v1fp16));

                const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                // y
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, v1fp16), two), _mm512_sub_ps(vImgHf, v1fp16));

                const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
            }

            gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
            gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));

            __m512i ix = _mm512_cvtps_epi32(gx);
            __m512i iy = _mm512_cvtps_epi32(gy);

            __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align0_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m512 two = _mm512_set1_ps(2.f);
            gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, v1fp16), vImgWf, v1fp16), two);
            gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, v1fp16), vImgHf, v1fp16), two);

            gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
            gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                __m512 v0p5fp16 = _mm512_set1_ps(0.5f);
                gx = _mm512_add_ps(gx, v0p5fp16);

                gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                gx = _mm512_sub_ps(vImgWf, reflectx_v);

                gx = _mm512_sub_ps(gx, v0p5fp16);

                _mm512_sub_ps(gx, v0p5fp16);

                gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                // y
                const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                gy = _mm512_add_ps(gy, v0p5fp16);

                gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, vImgHf), *(__m512*)_ps512_inv_sign_mask);
                gy = _mm512_sub_ps(vImgHf, reflecty_v);

                gy = _mm512_sub_ps(gy, v0p5fp16);

                _mm512_sub_ps(gy, v0p5fp16);

                gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));
            }

            __m512i ix = _mm512_cvtps_epi32(gx);
            __m512i iy = _mm512_cvtps_epi32(gy);

            __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_2d_nearest_align1_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m512 two = _mm512_set1_ps(2.f);
            gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, v1fp16), two), _mm512_sub_ps(vImgWf, v1fp16));
            gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, v1fp16), two), _mm512_sub_ps(vImgHf, v1fp16));

            gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
            gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, border_x), *(__m512*)_ps512_inv_sign_mask);
                gx = _mm512_sub_ps(border_x, reflectx_v);

                // y
                const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, border_y), *(__m512*)_ps512_inv_sign_mask);
                gy = _mm512_sub_ps(border_y, reflecty_v);
            }

            __m512i ix = _mm512_cvtps_epi32(gx);
            __m512i iy = _mm512_cvtps_epi32(gy);

            __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                             _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            for (int q = 0; q < dst.c; q++)
            {
                __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                _mm512_storeu_ps(dst.channel(q).row(y) + x * dst.elempack, _v);
            }
        }
    }
}

static void gridsample_3d_nearest_align0_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                    gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, v1fp16), vImgWf, v1fp16), two);

                    // y
                    gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, v1fp16), vImgHf, v1fp16), two);

                    // z
                    gz = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gz, v1fp16), vImgDf, v1fp16), two);
                }

                gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
                gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));
                gz = _mm512_floor_ps(_mm512_add_ps(gz, _mm512_set1_ps(0.5f)));

                __m512i ix = _mm512_cvtps_epi32(gx);
                __m512i iy = _mm512_cvtps_epi32(gy);
                __m512i iz = _mm512_cvtps_epi32(gz);

                __mmask16 v_in_range = (_mm512_cmpgt_epi32_mask(ix, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgWi, ix)) & (_mm512_cmpgt_epi32_mask(iy, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgHi, iy));
                v_in_range = v_in_range & (_mm512_cmpgt_epi32_mask(iz, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgDi, iz));

                __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), iz), _mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v_in_range, i_offset, src.channel(q), sizeof(float));

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_zeros_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                    gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, v1fp16), two), _mm512_sub_ps(vImgWf, v1fp16));

                    // y
                    gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, v1fp16), two), _mm512_sub_ps(vImgHf, v1fp16));

                    // z
                    gz = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gz, v1fp16), two), _mm512_sub_ps(vImgDf, v1fp16));
                }

                gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
                gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));
                gz = _mm512_floor_ps(_mm512_add_ps(gz, _mm512_set1_ps(0.5f)));

                __m512i ix = _mm512_cvtps_epi32(gx);
                __m512i iy = _mm512_cvtps_epi32(gy);
                __m512i iz = _mm512_cvtps_epi32(gz);

                __mmask16 v_in_range = (_mm512_cmpgt_epi32_mask(ix, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgWi, ix)) & (_mm512_cmpgt_epi32_mask(iy, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgHi, iy));
                v_in_range = v_in_range & (_mm512_cmpgt_epi32_mask(iz, vn1ip16) & _mm512_cmpgt_epi32_mask(vImgDi, iz));

                __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), iz), _mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), v_in_range, i_offset, src.channel(q), sizeof(float));

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align0_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                    gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, v1fp16), vImgWf, v1fp16), two);

                    const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                    gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                    // y
                    gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, v1fp16), vImgHf, v1fp16), two);

                    const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                    // z
                    gz = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gz, v1fp16), vImgDf, v1fp16), two);

                    const __m512 border_z = _mm512_sub_ps(vImgDf, v1fp16);

                    gz = _mm512_min_ps(border_z, _mm512_max_ps(gz, _mm512_setzero_ps()));
                }

                gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
                gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));
                gz = _mm512_floor_ps(_mm512_add_ps(gz, _mm512_set1_ps(0.5f)));

                __m512i ix = _mm512_cvtps_epi32(gx);
                __m512i iy = _mm512_cvtps_epi32(gy);
                __m512i iz = _mm512_cvtps_epi32(gz);

                __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), iz), _mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_border_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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
                    gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, v1fp16), two), _mm512_sub_ps(vImgWf, v1fp16));

                    const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                    gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                    // y
                    gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, v1fp16), two), _mm512_sub_ps(vImgHf, v1fp16));

                    const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                    // z
                    gz = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gz, v1fp16), two), _mm512_sub_ps(vImgDf, v1fp16));

                    const __m512 border_z = _mm512_sub_ps(vImgDf, v1fp16);

                    gz = _mm512_min_ps(border_z, _mm512_max_ps(gz, _mm512_setzero_ps()));
                }

                gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
                gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));
                gz = _mm512_floor_ps(_mm512_add_ps(gz, _mm512_set1_ps(0.5f)));

                __m512i ix = _mm512_cvtps_epi32(gx);
                __m512i iy = _mm512_cvtps_epi32(gy);
                __m512i iz = _mm512_cvtps_epi32(gz);

                __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), iz), _mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align0_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

                const __m512 two = _mm512_set1_ps(2.f);
                gx = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gx, v1fp16), vImgWf, v1fp16), two);
                gy = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gy, v1fp16), vImgHf, v1fp16), two);
                gz = _mm512_div_ps(_mm512_fmsub_ps(_mm512_add_ps(gz, v1fp16), vImgDf, v1fp16), two);

                gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
                gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));
                gz = _mm512_floor_ps(_mm512_add_ps(gz, _mm512_set1_ps(0.5f)));

                // compute coord
                {
                    // x
                    const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                    __m512 v0p5fp16 = _mm512_set1_ps(0.5f);
                    gx = _mm512_add_ps(gx, v0p5fp16);

                    gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, vImgWf), *(__m512*)_ps512_inv_sign_mask);
                    gx = _mm512_sub_ps(vImgWf, reflectx_v);

                    gx = _mm512_sub_ps(gx, v0p5fp16);

                    _mm512_sub_ps(gx, v0p5fp16);

                    gx = _mm512_min_ps(border_x, _mm512_max_ps(gx, _mm512_setzero_ps()));

                    // y
                    const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                    gy = _mm512_add_ps(gy, v0p5fp16);

                    gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, vImgHf), *(__m512*)_ps512_inv_sign_mask);
                    gy = _mm512_sub_ps(vImgHf, reflecty_v);

                    gy = _mm512_sub_ps(gy, v0p5fp16);

                    _mm512_sub_ps(gy, v0p5fp16);

                    gy = _mm512_min_ps(border_y, _mm512_max_ps(gy, _mm512_setzero_ps()));

                    // z
                    const __m512 border_z = _mm512_sub_ps(vImgDf, v1fp16);

                    gz = _mm512_add_ps(gz, v0p5fp16);

                    gz = _mm512_and_ps(gz, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectz_v = _mm512_and_ps(_mm512_sub_ps(gz, vImgDf), *(__m512*)_ps512_inv_sign_mask);
                    gz = _mm512_sub_ps(vImgDf, reflectz_v);

                    gz = _mm512_sub_ps(gz, v0p5fp16);

                    _mm512_sub_ps(gz, v0p5fp16);

                    gz = _mm512_min_ps(border_z, _mm512_max_ps(gz, _mm512_setzero_ps()));
                }

                __m512i ix = _mm512_cvtps_epi32(gx);
                __m512i iy = _mm512_cvtps_epi32(gy);
                __m512i iz = _mm512_cvtps_epi32(gz);

                __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), iz), _mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}

static void gridsample_3d_nearest_align1_reflection_blob_pack16(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

                const __m512 two = _mm512_set1_ps(2.f);
                gx = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gx, v1fp16), two), _mm512_sub_ps(vImgWf, v1fp16));
                gy = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gy, v1fp16), two), _mm512_sub_ps(vImgHf, v1fp16));
                gz = _mm512_mul_ps(_mm512_div_ps(_mm512_add_ps(gz, v1fp16), two), _mm512_sub_ps(vImgDf, v1fp16));

                gx = _mm512_floor_ps(_mm512_add_ps(gx, _mm512_set1_ps(0.5f)));
                gy = _mm512_floor_ps(_mm512_add_ps(gy, _mm512_set1_ps(0.5f)));
                gz = _mm512_floor_ps(_mm512_add_ps(gz, _mm512_set1_ps(0.5f)));

                // compute coord
                {
                    // x
                    const __m512 border_x = _mm512_sub_ps(vImgWf, v1fp16);

                    gx = _mm512_and_ps(gx, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectx_v = _mm512_and_ps(_mm512_sub_ps(gx, border_x), *(__m512*)_ps512_inv_sign_mask);
                    gx = _mm512_sub_ps(border_x, reflectx_v);

                    // y
                    const __m512 border_y = _mm512_sub_ps(vImgHf, v1fp16);

                    gy = _mm512_and_ps(gy, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflecty_v = _mm512_and_ps(_mm512_sub_ps(gy, border_y), *(__m512*)_ps512_inv_sign_mask);
                    gy = _mm512_sub_ps(border_y, reflecty_v);

                    // z
                    const __m512 border_z = _mm512_sub_ps(vImgDf, v1fp16);

                    gz = _mm512_and_ps(gz, *(__m512*)_ps512_inv_sign_mask);

                    __m512 reflectz_v = _mm512_and_ps(_mm512_sub_ps(gz, border_z), *(__m512*)_ps512_inv_sign_mask);
                    gz = _mm512_sub_ps(border_z, reflectz_v);
                }

                __m512i ix = _mm512_cvtps_epi32(gx);
                __m512i iy = _mm512_cvtps_epi32(gy);
                __m512i iz = _mm512_cvtps_epi32(gz);
                
                __m512i i_offset = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_mullo_epi32(vImgWi, vImgHi), iz), _mm512_add_epi32(_mm512_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

                for (int q = 0; q < dst.c; q++)
                {
                    __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 65535, i_offset, src.channel(q), sizeof(float));

                    _mm512_storeu_ps(dst.channel(q).depth(z).row(y) + x * dst.elempack, _v);
                }
            }
        }
    }
}