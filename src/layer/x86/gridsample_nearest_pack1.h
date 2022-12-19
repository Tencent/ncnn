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

static void gridsample_2d_nearest_align0_zeros_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 15 < nn; x += 16)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
            gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
            tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

            gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
            gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

            // compute coord
            {
                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                              _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, v_in_range);

                _mm256_storeu_ps(dst.channel(q).row(y) + x / 2, _v);
            }
        }

        nn = grid_size & 15;
#endif // __AVX__

        for (int x = grid_size - nn; x < grid_size; x += 2)
        {
            float sample_x = gridptr[x];
            float sample_y = gridptr[x + 1];

            sample_x = ((sample_x + 1) * src.w - 1) / 2.f;
            sample_y = ((sample_y + 1) * src.h - 1) / 2.f;

            int x0 = static_cast<int>(floor(sample_x + 0.5f));
            int y0 = static_cast<int>(floor(sample_y + 0.5f));

            bool v00_in_range = (x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h);

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                dst.channel(q).row(y)[x / 2] = v00_in_range ? image.row(y0)[x0] : 0;
            }
        }
    }
}

static void gridsample_2d_nearest_align1_zeros_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 15 < grid_size; x += 16)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
            gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
            tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

            gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
            gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

            // compute coord
            {
                // x
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                              _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, v_in_range);

                _mm256_storeu_ps(dst.channel(q).row(y) + x / 2, _v);
            }
        }

        nn = grid_size & 15;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 2)
        {
            float sample_x = gridptr[x];
            float sample_y = gridptr[x + 1];

            sample_x = (sample_x + 1) / 2.f * (src.w - 1);
            sample_y = (sample_y + 1) / 2.f * (src.h - 1);

            int x0 = static_cast<int>(floor(sample_x + 0.5f));
            int y0 = static_cast<int>(floor(sample_y + 0.5f));

            bool v00_in_range = (x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h);

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                dst.channel(q).row(y)[x / 2] = v00_in_range ? image.row(y0)[x0] : 0;
            }
        }
    }
}

static void gridsample_2d_nearest_align0_border_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 15 < nn; x += 16)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
            gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
            tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

            gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
            gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

            // compute coord
            {
                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
            __m256i i_offset = _mm256_cvtps_epi32(offset);
            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(dst.channel(q).row(y) + x / 2, _v);
            }
        }

        nn = grid_size & 15;
#endif // __AVX__

        for (int x = grid_size - nn; x < grid_size; x += 2)
        {
            float sample_x = gridptr[x];
            float sample_y = gridptr[x + 1];

            sample_x = ((sample_x + 1) * src.w - 1) / 2.f;
            sample_y = ((sample_y + 1) * src.h - 1) / 2.f;

            sample_x = std::min(src.w - 1.0f, std::max(sample_x, 0.0f));
            sample_y = std::min(src.h - 1.0f, std::max(sample_y, 0.0f));

            int x0 = static_cast<int>(floor(sample_x + 0.5f));
            int y0 = static_cast<int>(floor(sample_y + 0.5f));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                dst.channel(q).row(y)[x / 2] = image.row(y0)[x0];
            }
        }
    }
}

static void gridsample_2d_nearest_align1_border_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 15 < grid_size; x += 16)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
            gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
            tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

            gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
            gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

            // compute coord
            {
                // x
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
            __m256i i_offset = _mm256_cvtps_epi32(offset);
            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(dst.channel(q).row(y) + x / 2, _v);
            }
        }

        nn = grid_size & 15;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 2)
        {
            float sample_x = gridptr[x];
            float sample_y = gridptr[x + 1];

            sample_x = (sample_x + 1) / 2.f * (src.w - 1);
            sample_y = (sample_y + 1) / 2.f * (src.h - 1);

            sample_x = std::min(src.w - 1.0f, std::max(sample_x, 0.0f));
            sample_y = std::min(src.h - 1.0f, std::max(sample_y, 0.0f));

            int x0 = static_cast<int>(floor(sample_x + 0.5f));
            int y0 = static_cast<int>(floor(sample_y + 0.5f));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                dst.channel(q).row(y)[x / 2] = image.row(y0)[x0];
            }
        }
    }
}

static void gridsample_2d_nearest_align0_reflection_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 15 < nn; x += 16)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
            gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
            tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

            gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
            gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                gx = _mm256_add_ps(gx, v0p5fp8);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(vImgWf, reflectx_v);

                gx = _mm256_sub_ps(gx, v0p5fp8);

                _mm256_sub_ps(gx, v0p5fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_add_ps(gy, v0p5fp8);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(vImgHf, reflecty_v);

                gy = _mm256_sub_ps(gy, v0p5fp8);

                _mm256_sub_ps(gy, v0p5fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
            }

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
            __m256i i_offset = _mm256_cvtps_epi32(offset);
            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(dst.channel(q).row(y) + x / 2, _v);
            }
        }

        nn = grid_size & 15;
#endif // __AVX__

        for (int x = grid_size - nn; x < grid_size; x += 2)
        {
            float sample_x = gridptr[x];
            float sample_y = gridptr[x + 1];

            sample_x = ((sample_x + 1) * src.w - 1) / 2.f;
            sample_y = ((sample_y + 1) * src.h - 1) / 2.f;

            sample_x = floor(sample_x + 0.5f);
            sample_y = floor(sample_y + 0.5f);

            sample_x = abs(sample_x + 0.5f);
            sample_x = src.w - abs(sample_x - src.w) - 0.5;

            sample_y = abs(sample_y + 0.5f);
            sample_y = src.h - abs(sample_y - src.h) - 0.5;

            int x0 = std::min(src.w - 1.0f, std::max(sample_x, 0.0f));
            int y0 = std::min(src.h - 1.0f, std::max(sample_y, 0.0f));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                dst.channel(q).row(y)[x / 2] = image.row(y0)[x0];
            }
        }
    }
}

static void gridsample_2d_nearest_align1_reflection_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 15 < grid_size; x += 16)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
            gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
            tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

            gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
            gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(border_x, reflectx_v);

                // y
                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(border_y, reflecty_v);
            }

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
            __m256i i_offset = _mm256_cvtps_epi32(offset);
            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(dst.channel(q).row(y) + x / 2, _v);
            }
        }

        nn = grid_size & 15;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 2)
        {
            float sample_x = gridptr[x];
            float sample_y = gridptr[x + 1];

            sample_x = (sample_x + 1) / 2.f * (src.w - 1);
            sample_y = (sample_y + 1) / 2.f * (src.h - 1);

            sample_x = floor(sample_x + 0.5f);
            sample_y = floor(sample_y + 0.5f);

            sample_x = abs(sample_x);
            int x0 = (src.w - 1) - abs(sample_x - (src.w - 1));

            sample_y = abs(sample_y);
            int y0 = (src.h - 1) - abs(sample_y - (src.h - 1));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                dst.channel(q).row(y)[x / 2] = image.row(y0)[x0];
            }
        }
    }
}

static void gridsample_3d_nearest_align0_zeros_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 23 < nn; x += 24)
        {
            //upzip (3)
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
            __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
            __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
            gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

            tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
            tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

            gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
            gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
            gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

            // compute coord
            {
                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                // z
                gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                              _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));
            v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, gz, _CMP_GT_OS)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                          _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, v_in_range);

                _mm256_storeu_ps(static_cast<float*>(dst.channel(q).depth(y).data) + x / 3, _v);
            }
        }

        nn = grid_size % 24;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 3)
        {
            float gx = gridptr[x];
            float gy = gridptr[x + 1];
            float gz = gridptr[x + 2];

            gx = ((gx + 1) * src.w - 1) / 2.f;
            gy = ((gy + 1) * src.h - 1) / 2.f;
            gz = ((gz + 1) * src.d - 1) / 2.f;

            // bilinear interpolate
            int x0 = static_cast<int>(floor(gx + 0.5f));
            int y0 = static_cast<int>(floor(gy + 0.5f));
            int z0 = static_cast<int>(floor(gz + 0.5f));

            bool v_in_range = (x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h) && (z0 > -1) && (z0 < src.d);

            for (int q = 0; q < src.c; q++)
            {
                dst.channel(q).depth(y)[x / 3] = v_in_range ? src.channel(q).depth(z0).row(y0)[x0] : 0;
            }
        }
    }
}

static void gridsample_3d_nearest_align1_zeros_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 23 < nn; x += 24)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
            __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
            __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
            gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

            tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
            tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

            gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
            gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
            gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

            // compute coord
            {
                // x
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                // z
                gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                              _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));
            v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, gz, _CMP_GT_OS)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                          _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, v_in_range);

                _mm256_storeu_ps(static_cast<float*>(dst.channel(q).depth(y).data) + x / 3, _v);
            }
        }
        nn = grid_size % 24;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 3)
        {
            float gx = gridptr[x];
            float gy = gridptr[x + 1];
            float gz = gridptr[x + 2];

            gx = (gx + 1) / 2.f * (src.w - 1);
            gy = (gy + 1) / 2.f * (src.h - 1);
            gz = (gz + 1) / 2.f * (src.d - 1);

            int x0 = static_cast<int>(floor(gx + 0.5f));
            int y0 = static_cast<int>(floor(gy + 0.5f));
            int z0 = static_cast<int>(floor(gz + 0.5f));

            bool v_in_range = (x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h) && (z0 > -1) && (z0 < src.d);

            for (int q = 0; q < src.c; q++)
            {
                dst.channel(q).depth(y)[x / 3] = v_in_range ? src.channel(q).depth(z0).row(y0)[x0] : 0;
            }
        }
    }
}

static void gridsample_3d_nearest_align0_border_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 23 < nn; x += 24)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
            __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
            __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
            gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

            tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
            tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

            gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
            gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
            gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

            // compute coord
            {
                // x
                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                // z
                gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                          _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(static_cast<float*>(dst.channel(q).depth(y).data) + x / 3, _v);
            }
        }
        nn = grid_size % 24;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 3)
        {
            float gx = gridptr[x];
            float gy = gridptr[x + 1];
            float gz = gridptr[x + 2];

            gx = ((gx + 1) * src.w - 1) / 2.f;
            gy = ((gy + 1) * src.h - 1) / 2.f;
            gz = ((gz + 1) * src.d - 1) / 2.f;

            gx = std::min(src.w - 1.0f, std::max(gx, 0.0f));
            gy = std::min(src.h - 1.0f, std::max(gy, 0.0f));
            gz = std::min(src.d - 1.0f, std::max(gz, 0.0f));

            int x0 = static_cast<int>(floor(gx + 0.5f));
            int y0 = static_cast<int>(floor(gy + 0.5f));
            int z0 = static_cast<int>(floor(gz + 0.5f));

            for (int q = 0; q < src.c; q++)
            {
                dst.channel(q).depth(y)[x / 3] = src.channel(q).depth(z0).row(y0)[x0];
            }
        }
    }
}

static void gridsample_3d_nearest_align1_border_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 23 < nn; x += 24)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
            __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
            __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
            gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

            tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
            tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

            gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
            gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
            gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

            // compute coord
            {
                // x
                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                // z
                gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));

                const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
            }

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                          _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(static_cast<float*>(dst.channel(q).depth(y).data) + x / 3, _v);
            }
        }
        nn = grid_size % 24;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 3)
        {
            float gx = gridptr[x];
            float gy = gridptr[x + 1];
            float gz = gridptr[x + 2];

            gx = (gx + 1) / 2.f * (src.w - 1);
            gy = (gy + 1) / 2.f * (src.h - 1);
            gz = (gz + 1) / 2.f * (src.d - 1);

            gx = std::min(src.w - 1.0f, std::max(gx, 0.0f));
            gy = std::min(src.h - 1.0f, std::max(gy, 0.0f));
            gz = std::min(src.d - 1.0f, std::max(gz, 0.0f));

            int x0 = static_cast<int>(floor(gx + 0.5f));
            int y0 = static_cast<int>(floor(gy + 0.5f));
            int z0 = static_cast<int>(floor(gz + 0.5f));

            for (int q = 0; q < src.c; q++)
            {
                dst.channel(q).depth(y)[x / 3] = src.channel(q).depth(z0).row(y0)[x0];
            }
        }
    }
}

static void gridsample_3d_nearest_align0_reflection_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 23 < nn; x += 24)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
            __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
            __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
            gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

            tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
            tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

            gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
            gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
            gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            {
                // x
                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                gx = _mm256_add_ps(gx, v0p5fp8);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(vImgWf, reflectx_v);

                gx = _mm256_sub_ps(gx, v0p5fp8);

                _mm256_sub_ps(gx, v0p5fp8);

                gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                // y
                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_add_ps(gy, v0p5fp8);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(vImgHf, reflecty_v);

                gy = _mm256_sub_ps(gy, v0p5fp8);

                _mm256_sub_ps(gy, v0p5fp8);

                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                // z
                const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                gz = _mm256_add_ps(gz, v0p5fp8);

                gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, vImgDf), *(__m256*)_ps256_inv_sign_mask);
                gz = _mm256_sub_ps(vImgDf, reflectz_v);

                gz = _mm256_sub_ps(gz, v0p5fp8);

                _mm256_sub_ps(gz, v0p5fp8);

                gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
            }

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                          _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(static_cast<float*>(dst.channel(q).depth(y).data) + x / 3, _v);
            }
        }
        nn = grid_size % 24;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 3)
        {
            float gx = gridptr[x];
            float gy = gridptr[x + 1];
            float gz = gridptr[x + 2];

            gx = ((gx + 1) * src.w - 1) / 2.f;
            gy = ((gy + 1) * src.h - 1) / 2.f;
            gz = ((gz + 1) * src.d - 1) / 2.f;

            gx = floor(gx + 0.5f);
            gy = floor(gy + 0.5f);
            gz = floor(gz + 0.5f);

            gx = abs(gx + 0.5f);
            gx = src.w - abs(gx - src.w) - 0.5;

            gy = abs(gy + 0.5f);
            gy = src.h - abs(gy - src.h) - 0.5;

            gz = abs(gz + 0.5f);
            gz = src.d - abs(gz - src.d) - 0.5;

            int x0 = std::min(src.w - 1.0f, std::max(gx, 0.0f));
            int y0 = std::min(src.h - 1.0f, std::max(gy, 0.0f));
            int z0 = std::min(src.d - 1.0f, std::max(gz, 0.0f));

            for (int q = 0; q < src.c; q++)
            {
                dst.channel(q).depth(y)[x / 3] = src.channel(q).depth(z0).row(y0)[x0];
            }
        }
    }
}

static void gridsample_3d_nearest_align1_reflection_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
    const __m256 vImgWf = _mm256_set1_ps(src.w);
    const __m256 vImgHf = _mm256_set1_ps(src.h);
    const __m256 vImgDf = _mm256_set1_ps(src.d);
#endif // __AVX__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int y = 0; y < grid.c; y++)
    {
        const float* gridptr = grid.channel(y);
        int nn = grid_size;
#if __AVX__
        for (int x = 0; x + 23 < nn; x += 24)
        {
            __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
            __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
            __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

            __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
            __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
            gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

            tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
            tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

            gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
            gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
            gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
            gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            // compute coord
            {
                // x
                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                gx = _mm256_sub_ps(border_x, reflectx_v);

                // y
                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                gy = _mm256_sub_ps(border_y, reflecty_v);

                // z
                const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, border_z), *(__m256*)_ps256_inv_sign_mask);
                gz = _mm256_sub_ps(border_z, reflectz_v);
            }

            __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                          _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
            __m256i i_offset = _mm256_cvtps_epi32(offset);

            for (int q = 0; q < src.c; q++)
            {
                __m256 _v = mask_gather_ps256(src.channel(q), i_offset, *(__m256*)_ps256_n1);

                _mm256_storeu_ps(static_cast<float*>(dst.channel(q).depth(y).data) + x / 3, _v);
            }
        }
        nn = grid_size % 24;
#endif // __AVX__
        for (int x = grid_size - nn; x < grid_size; x += 3)
        {
            float gx = gridptr[x];
            float gy = gridptr[x + 1];
            float gz = gridptr[x + 2];

            gx = (gx + 1) / 2.f * (src.w - 1);
            gy = (gy + 1) / 2.f * (src.h - 1);
            gz = (gz + 1) / 2.f * (src.d - 1);

            gx = floor(gx + 0.5f);
            gy = floor(gy + 0.5f);
            gz = floor(gz + 0.5f);

            gx = abs(gx);
            gx = (src.w - 1) - abs(gx - (src.w - 1));

            gy = abs(gy);
            gy = (src.h - 1) - abs(gy - (src.h - 1));

            gz = abs(gz);
            gz = (src.d - 1) - abs(gz - (src.d - 1));

            int x0 = std::min(src.w - 1.0f, std::max(gx, 0.0f));
            int y0 = std::min(src.h - 1.0f, std::max(gy, 0.0f));
            int z0 = std::min(src.d - 1.0f, std::max(gz, 0.0f));

            for (int q = 0; q < src.c; q++)
            {
                dst.channel(q).depth(y)[x / 3] = src.channel(q).depth(z0).row(y0)[x0];
            }
        }
    }
}