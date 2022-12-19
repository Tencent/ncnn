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

static void gridsample_2d_bicubic_align0_zeros_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(gx0, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx0, _CMP_GT_OS));
            __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(gx1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx1, _CMP_GT_OS));
            __m256 x2_in_range = _mm256_and_ps(_mm256_cmp_ps(gx2, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx2, _CMP_GT_OS));
            __m256 x3_in_range = _mm256_and_ps(_mm256_cmp_ps(gx3, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx3, _CMP_GT_OS));

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            __m256 v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS));

                v0_in_range[i] = _mm256_and_ps(x0_in_range, y_in_range);
                v1_in_range[i] = _mm256_and_ps(x1_in_range, y_in_range);
                v2_in_range[i] = _mm256_and_ps(x2_in_range, y_in_range);
                v3_in_range[i] = _mm256_and_ps(x3_in_range, y_in_range);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < src.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = mask_gather_ps256(src.channel(q), v0_offset[i], v0_in_range[i]);
                    __m256 x1_val = mask_gather_ps256(src.channel(q), v1_offset[i], v1_in_range[i]);
                    __m256 x2_val = mask_gather_ps256(src.channel(q), v2_offset[i], v2_in_range[i]);
                    __m256 x3_val = mask_gather_ps256(src.channel(q), v3_offset[i], v3_in_range[i]);

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

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

            int x1 = floor(sample_x);
            int y1 = floor(sample_y);
            int x0 = x1 - 1;
            int y0 = y1 - 1;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x3 = x1 + 2;
            int y3 = y1 + 2;

            bool x1_in_range = (x1 > -1) & (x1 < src.w);
            bool y1_in_range = (y1 > -1) & (y1 < src.h);
            bool x0_in_range = (x0 > -1) & (x0 < src.w);
            bool y0_in_range = (y0 > -1) & (y0 < src.h);
            bool x2_in_range = (x2 > -1) & (x2 < src.w);
            bool y2_in_range = (y2 > -1) & (y2 < src.h);
            bool x3_in_range = (x3 > -1) & (x3 < src.w);
            bool y3_in_range = (y3 > -1) & (y3 < src.h);

            bool v00_in_range = x0_in_range & y0_in_range;
            bool v01_in_range = x1_in_range & y0_in_range;
            bool v02_in_range = x2_in_range & y0_in_range;
            bool v03_in_range = x3_in_range & y0_in_range;
            bool v10_in_range = x0_in_range & y1_in_range;
            bool v11_in_range = x1_in_range & y1_in_range;
            bool v12_in_range = x2_in_range & y1_in_range;
            bool v13_in_range = x3_in_range & y1_in_range;
            bool v20_in_range = x0_in_range & y2_in_range;
            bool v21_in_range = x1_in_range & y2_in_range;
            bool v22_in_range = x2_in_range & y2_in_range;
            bool v23_in_range = x3_in_range & y2_in_range;
            bool v30_in_range = x0_in_range & y3_in_range;
            bool v31_in_range = x1_in_range & y3_in_range;
            bool v32_in_range = x2_in_range & y3_in_range;
            bool v33_in_range = x3_in_range & y3_in_range;

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                float v00 = v00_in_range ? image.row(y0)[x0] : 0;
                float v01 = v01_in_range ? image.row(y0)[x1] : 0;
                float v02 = v02_in_range ? image.row(y0)[x2] : 0;
                float v03 = v03_in_range ? image.row(y0)[x3] : 0;
                float v10 = v10_in_range ? image.row(y1)[x0] : 0;
                float v11 = v11_in_range ? image.row(y1)[x1] : 0;
                float v12 = v12_in_range ? image.row(y1)[x2] : 0;
                float v13 = v13_in_range ? image.row(y1)[x3] : 0;
                float v20 = v20_in_range ? image.row(y2)[x0] : 0;
                float v21 = v21_in_range ? image.row(y2)[x1] : 0;
                float v22 = v22_in_range ? image.row(y2)[x2] : 0;
                float v23 = v23_in_range ? image.row(y2)[x3] : 0;
                float v30 = v30_in_range ? image.row(y3)[x0] : 0;
                float v31 = v31_in_range ? image.row(y3)[x1] : 0;
                float v32 = v32_in_range ? image.row(y3)[x2] : 0;
                float v33 = v33_in_range ? image.row(y3)[x3] : 0;

                float x_coeffs[4];
                float y_coeffs[4];
                interpolate_cubic(sample_x - x1, x_coeffs);
                interpolate_cubic(sample_y - y1, y_coeffs);

                float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                dst.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_zeros_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(gx0, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx0, _CMP_GT_OS));
            __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(gx1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx1, _CMP_GT_OS));
            __m256 x2_in_range = _mm256_and_ps(_mm256_cmp_ps(gx2, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx2, _CMP_GT_OS));
            __m256 x3_in_range = _mm256_and_ps(_mm256_cmp_ps(gx3, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx3, _CMP_GT_OS));

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            __m256 v0_in_range[4], v1_in_range[4], v2_in_range[4], v3_in_range[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS));

                v0_in_range[i] = _mm256_and_ps(x0_in_range, y_in_range);
                v1_in_range[i] = _mm256_and_ps(x1_in_range, y_in_range);
                v2_in_range[i] = _mm256_and_ps(x2_in_range, y_in_range);
                v3_in_range[i] = _mm256_and_ps(x3_in_range, y_in_range);

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < src.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = mask_gather_ps256(src.channel(q), v0_offset[i], v0_in_range[i]);
                    __m256 x1_val = mask_gather_ps256(src.channel(q), v1_offset[i], v1_in_range[i]);
                    __m256 x2_val = mask_gather_ps256(src.channel(q), v2_offset[i], v2_in_range[i]);
                    __m256 x3_val = mask_gather_ps256(src.channel(q), v3_offset[i], v3_in_range[i]);

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

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

            int x1 = floor(sample_x);
            int y1 = floor(sample_y);
            int x0 = x1 - 1;
            int y0 = y1 - 1;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x3 = x1 + 2;
            int y3 = y1 + 2;

            bool x1_in_range = (x1 > -1) & (x1 < src.w);
            bool y1_in_range = (y1 > -1) & (y1 < src.h);
            bool x0_in_range = (x0 > -1) & (x0 < src.w);
            bool y0_in_range = (y0 > -1) & (y0 < src.h);
            bool x2_in_range = (x2 > -1) & (x2 < src.w);
            bool y2_in_range = (y2 > -1) & (y2 < src.h);
            bool x3_in_range = (x3 > -1) & (x3 < src.w);
            bool y3_in_range = (y3 > -1) & (y3 < src.h);

            bool v00_in_range = x0_in_range & y0_in_range;
            bool v01_in_range = x1_in_range & y0_in_range;
            bool v02_in_range = x2_in_range & y0_in_range;
            bool v03_in_range = x3_in_range & y0_in_range;
            bool v10_in_range = x0_in_range & y1_in_range;
            bool v11_in_range = x1_in_range & y1_in_range;
            bool v12_in_range = x2_in_range & y1_in_range;
            bool v13_in_range = x3_in_range & y1_in_range;
            bool v20_in_range = x0_in_range & y2_in_range;
            bool v21_in_range = x1_in_range & y2_in_range;
            bool v22_in_range = x2_in_range & y2_in_range;
            bool v23_in_range = x3_in_range & y2_in_range;
            bool v30_in_range = x0_in_range & y3_in_range;
            bool v31_in_range = x1_in_range & y3_in_range;
            bool v32_in_range = x2_in_range & y3_in_range;
            bool v33_in_range = x3_in_range & y3_in_range;

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                float v00 = v00_in_range ? image.row(y0)[x0] : 0;
                float v01 = v01_in_range ? image.row(y0)[x1] : 0;
                float v02 = v02_in_range ? image.row(y0)[x2] : 0;
                float v03 = v03_in_range ? image.row(y0)[x3] : 0;
                float v10 = v10_in_range ? image.row(y1)[x0] : 0;
                float v11 = v11_in_range ? image.row(y1)[x1] : 0;
                float v12 = v12_in_range ? image.row(y1)[x2] : 0;
                float v13 = v13_in_range ? image.row(y1)[x3] : 0;
                float v20 = v20_in_range ? image.row(y2)[x0] : 0;
                float v21 = v21_in_range ? image.row(y2)[x1] : 0;
                float v22 = v22_in_range ? image.row(y2)[x2] : 0;
                float v23 = v23_in_range ? image.row(y2)[x3] : 0;
                float v30 = v30_in_range ? image.row(y3)[x0] : 0;
                float v31 = v31_in_range ? image.row(y3)[x1] : 0;
                float v32 = v32_in_range ? image.row(y3)[x2] : 0;
                float v33 = v33_in_range ? image.row(y3)[x3] : 0;

                float x_coeffs[4];
                float y_coeffs[4];
                interpolate_cubic(sample_x - x1, x_coeffs);
                interpolate_cubic(sample_y - y1, y_coeffs);

                float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                dst.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_border_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);
            const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            gx0 = _mm256_min_ps(border_x, _mm256_max_ps(gx0, _mm256_setzero_ps()));
            gx1 = _mm256_min_ps(border_x, _mm256_max_ps(gx1, _mm256_setzero_ps()));
            gx2 = _mm256_min_ps(border_x, _mm256_max_ps(gx2, _mm256_setzero_ps()));
            gx3 = _mm256_min_ps(border_x, _mm256_max_ps(gx3, _mm256_setzero_ps()));

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < src.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = mask_gather_ps256(src.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                    __m256 x1_val = mask_gather_ps256(src.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                    __m256 x2_val = mask_gather_ps256(src.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                    __m256 x3_val = mask_gather_ps256(src.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

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

            int x_floor = floor(sample_x);
            int y_floor = floor(sample_y);

            int x1 = x_floor;
            int y1 = y_floor;
            int x0 = x1 - 1;
            int y0 = y1 - 1;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x3 = x1 + 2;
            int y3 = y1 + 2;

            x1 = std::min(src.w - 1, std::max(x1, 0));
            y1 = std::min(src.h - 1, std::max(y1, 0));
            x0 = std::min(src.w - 1, std::max(x0, 0));
            y0 = std::min(src.h - 1, std::max(y0, 0));
            x2 = std::min(src.w - 1, std::max(x2, 0));
            y2 = std::min(src.h - 1, std::max(y2, 0));
            x3 = std::min(src.w - 1, std::max(x3, 0));
            y3 = std::min(src.h - 1, std::max(y3, 0));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                float v00 = image.row(y0)[x0];
                float v01 = image.row(y0)[x1];
                float v02 = image.row(y0)[x2];
                float v03 = image.row(y0)[x3];
                float v10 = image.row(y1)[x0];
                float v11 = image.row(y1)[x1];
                float v12 = image.row(y1)[x2];
                float v13 = image.row(y1)[x3];
                float v20 = image.row(y2)[x0];
                float v21 = image.row(y2)[x1];
                float v22 = image.row(y2)[x2];
                float v23 = image.row(y2)[x3];
                float v30 = image.row(y3)[x0];
                float v31 = image.row(y3)[x1];
                float v32 = image.row(y3)[x2];
                float v33 = image.row(y3)[x3];

                float x_coeffs[4];
                float y_coeffs[4];
                interpolate_cubic(sample_x - x_floor, x_coeffs);
                interpolate_cubic(sample_y - y_floor, y_coeffs);

                float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                dst.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_border_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
            const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

            gx0 = _mm256_min_ps(border_x, _mm256_max_ps(gx0, _mm256_setzero_ps()));
            gx1 = _mm256_min_ps(border_x, _mm256_max_ps(gx1, _mm256_setzero_ps()));
            gx2 = _mm256_min_ps(border_x, _mm256_max_ps(gx2, _mm256_setzero_ps()));
            gx3 = _mm256_min_ps(border_x, _mm256_max_ps(gx3, _mm256_setzero_ps()));

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < src.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = mask_gather_ps256(src.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                    __m256 x1_val = mask_gather_ps256(src.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                    __m256 x2_val = mask_gather_ps256(src.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                    __m256 x3_val = mask_gather_ps256(src.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

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

            int x_floor = floor(sample_x);
            int y_floor = floor(sample_y);

            int x1 = x_floor;
            int y1 = y_floor;
            int x0 = x1 - 1;
            int y0 = y1 - 1;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x3 = x1 + 2;
            int y3 = y1 + 2;

            x1 = std::min(src.w - 1, std::max(x1, 0));
            y1 = std::min(src.h - 1, std::max(y1, 0));
            x0 = std::min(src.w - 1, std::max(x0, 0));
            y0 = std::min(src.h - 1, std::max(y0, 0));
            x2 = std::min(src.w - 1, std::max(x2, 0));
            y2 = std::min(src.h - 1, std::max(y2, 0));
            x3 = std::min(src.w - 1, std::max(x3, 0));
            y3 = std::min(src.h - 1, std::max(y3, 0));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                float v00 = image.row(y0)[x0];
                float v01 = image.row(y0)[x1];
                float v02 = image.row(y0)[x2];
                float v03 = image.row(y0)[x3];
                float v10 = image.row(y1)[x0];
                float v11 = image.row(y1)[x1];
                float v12 = image.row(y1)[x2];
                float v13 = image.row(y1)[x3];
                float v20 = image.row(y2)[x0];
                float v21 = image.row(y2)[x1];
                float v22 = image.row(y2)[x2];
                float v23 = image.row(y2)[x3];
                float v30 = image.row(y3)[x0];
                float v31 = image.row(y3)[x1];
                float v32 = image.row(y3)[x2];
                float v33 = image.row(y3)[x3];

                float x_coeffs[4];
                float y_coeffs[4];
                interpolate_cubic(sample_x - x_floor, x_coeffs);
                interpolate_cubic(sample_y - y_floor, y_coeffs);

                float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                dst.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
            }
        }
    }
}

static void gridsample_2d_bicubic_align0_reflection_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);
            const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
            gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
            gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));
            const __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
            {
                // x0
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

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                {
                    //y
                    gy = _mm256_add_ps(gy, v0p5fp8);

                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                    gy = _mm256_sub_ps(gy, v0p5fp8);

                    _mm256_sub_ps(gy, v0p5fp8);

                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                }

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < src.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = mask_gather_ps256(src.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                    __m256 x1_val = mask_gather_ps256(src.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                    __m256 x2_val = mask_gather_ps256(src.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                    __m256 x3_val = mask_gather_ps256(src.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

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

            int x_floor = floor(sample_x);
            int y_floor = floor(sample_y);

            int x1 = x_floor;
            int y1 = y_floor;
            int x0 = x1 - 1;
            int y0 = y1 - 1;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x3 = x1 + 2;
            int y3 = y1 + 2;

            x0 = static_cast<int>(reflect_coord(x0 + 0.5, src.w) - 0.5);

            y0 = static_cast<int>(reflect_coord(y0 + 0.5, src.h) - 0.5);

            x0 = std::min(src.w - 1, std::max(x0, 0));
            y0 = std::min(src.h - 1, std::max(y0, 0));

            x1 = static_cast<int>(reflect_coord(x1 + 0.5, src.w) - 0.5);

            y1 = static_cast<int>(reflect_coord(y1 + 0.5, src.h) - 0.5);

            x1 = std::min(src.w - 1, std::max(x1, 0));
            y1 = std::min(src.h - 1, std::max(y1, 0));

            x2 = static_cast<int>(reflect_coord(x2 + 0.5, src.w) - 0.5);

            y2 = static_cast<int>(reflect_coord(y2 + 0.5, src.h) - 0.5);

            x2 = std::min(src.w - 1, std::max(x2, 0));
            y2 = std::min(src.h - 1, std::max(y2, 0));

            x3 = static_cast<int>(reflect_coord(x3 + 0.5, src.w) - 0.5);

            y3 = static_cast<int>(reflect_coord(y3 + 0.5, src.h) - 0.5);

            x3 = std::min(src.w - 1, std::max(x3, 0));
            y3 = std::min(src.h - 1, std::max(y3, 0));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                float v00 = image.row(y0)[x0];
                float v01 = image.row(y0)[x1];
                float v02 = image.row(y0)[x2];
                float v03 = image.row(y0)[x3];
                float v10 = image.row(y1)[x0];
                float v11 = image.row(y1)[x1];
                float v12 = image.row(y1)[x2];
                float v13 = image.row(y1)[x3];
                float v20 = image.row(y2)[x0];
                float v21 = image.row(y2)[x1];
                float v22 = image.row(y2)[x2];
                float v23 = image.row(y2)[x3];
                float v30 = image.row(y3)[x0];
                float v31 = image.row(y3)[x1];
                float v32 = image.row(y3)[x2];
                float v33 = image.row(y3)[x3];

                float x_coeffs[4];
                float y_coeffs[4];
                interpolate_cubic(sample_x - x_floor, x_coeffs);
                interpolate_cubic(sample_y - y_floor, y_coeffs);

                float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                dst.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
            }
        }
    }
}

static void gridsample_2d_bicubic_align1_reflection_blob_pack1(const Mat& src, Mat& dst, const Mat& grid, const Option& opt)
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

            const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
            const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

            gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
            gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 coefficients[4];

            __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
            __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));
            {
                // x0
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

            __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                {
                    //y
                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                    gy = _mm256_sub_ps(border_y, reflecty_v);
                }

                __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
            }

            for (int q = 0; q < src.c; q++)
            {
                for (int i = 0; i < 4; i++)
                {
                    __m256 x0_val = mask_gather_ps256(src.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                    __m256 x1_val = mask_gather_ps256(src.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                    __m256 x2_val = mask_gather_ps256(src.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                    __m256 x3_val = mask_gather_ps256(src.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                }

                __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

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

            int x_floor = floor(sample_x);
            int y_floor = floor(sample_y);

            int x1 = x_floor;
            int y1 = y_floor;
            int x0 = x1 - 1;
            int y0 = y1 - 1;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x3 = x1 + 2;
            int y3 = y1 + 2;

            x0 = static_cast<int>(reflect_coord(x0, src.w - 1));
            y0 = static_cast<int>(reflect_coord(y0, src.h - 1));
            x1 = static_cast<int>(reflect_coord(x1, src.w - 1));
            y1 = static_cast<int>(reflect_coord(y1, src.h - 1));
            x2 = static_cast<int>(reflect_coord(x2, src.w - 1));
            y2 = static_cast<int>(reflect_coord(y2, src.h - 1));
            x3 = static_cast<int>(reflect_coord(x3, src.w - 1));
            y3 = static_cast<int>(reflect_coord(y3, src.h - 1));

            for (int q = 0; q < src.c; q++)
            {
                const Mat& image = src.channel(q);

                float v00 = image.row(y0)[x0];
                float v01 = image.row(y0)[x1];
                float v02 = image.row(y0)[x2];
                float v03 = image.row(y0)[x3];
                float v10 = image.row(y1)[x0];
                float v11 = image.row(y1)[x1];
                float v12 = image.row(y1)[x2];
                float v13 = image.row(y1)[x3];
                float v20 = image.row(y2)[x0];
                float v21 = image.row(y2)[x1];
                float v22 = image.row(y2)[x2];
                float v23 = image.row(y2)[x3];
                float v30 = image.row(y3)[x0];
                float v31 = image.row(y3)[x1];
                float v32 = image.row(y3)[x2];
                float v33 = image.row(y3)[x3];

                float x_coeffs[4];
                float y_coeffs[4];
                interpolate_cubic(sample_x - x_floor, x_coeffs);
                interpolate_cubic(sample_y - y_floor, y_coeffs);

                float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                dst.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
            }
        }
    }
}