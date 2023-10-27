// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

template<GridSample::PaddingMode pd, bool align_corner>
void gridsample_2d_bilinear_compute_blob(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int grid_size = grid.w * grid.h;

    float* offset_value_ptr = offset_value.channel(0);

    grid_sample_unormalize<align_corner> unormalize;
    compute_coord<pd, align_corner> get_coord;

    if (permute_fusion == 0)
    {
        for (int y = 0; y < grid.c; y++)
        {
            const float* gridptr = grid.channel(y);
            int x = 0;
#if __AVX__
            for (; x + 15 < grid_size; x += 16)
            {
                __m256 gx = _mm256_loadu_ps(gridptr);
                __m256 gy = _mm256_loadu_ps(gridptr + 8);

                transpose2x8_ps(gx, gy);

                gx = unormalize(_mm256_set1_ps(src.w), gx);
                gx = get_coord(_mm256_set1_ps(src.w), gx);

                gy = unormalize(_mm256_set1_ps(src.h), gy);
                gy = get_coord(_mm256_set1_ps(src.h), gy);

                __m256 x_w = _mm256_floor_ps(gx);
                __m256 y_n = _mm256_floor_ps(gy);

                __m256 x1 = _mm256_add_ps(x_w, _mm256_set1_ps(1));
                __m256 y1 = _mm256_add_ps(y_n, _mm256_set1_ps(1));

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x_w, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x1, _CMP_GT_OS));
                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y_n, _CMP_GT_OS));
                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y1, _CMP_GT_OS));

                __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                __m256 v01_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                __m256 v10_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                __m256 nw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(y_n, _mm256_set1_ps(src.w), x_w), _mm256_set1_ps(src.elempack));
                __m256 ne_offset = _mm256_add_ps(nw_offset, _mm256_set1_ps(src.elempack));
                __m256 sw_offset = _mm256_comp_fmadd_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.elempack), nw_offset);
                __m256 se_offset = _mm256_add_ps(sw_offset, _mm256_set1_ps(src.elempack));

                nw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), nw_offset, v00_in_range);
                ne_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), ne_offset, v01_in_range);
                sw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), sw_offset, v10_in_range);
                se_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), se_offset, v11_in_range);

                nw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(nw_offset));
                ne_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(ne_offset));
                sw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(sw_offset));
                se_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(se_offset));

                __m256 alpha = _mm256_sub_ps(gx, x_w);
                __m256 beta = _mm256_sub_ps(gy, y_n);

                transpose8x6_ps(nw_offset, ne_offset, sw_offset, se_offset, alpha, beta);

                _mm256_storeu_ps(offset_value_ptr, nw_offset);
                _mm256_storeu_ps(offset_value_ptr + 8, ne_offset);
                _mm256_storeu_ps(offset_value_ptr + 16, sw_offset);
                _mm256_storeu_ps(offset_value_ptr + 24, se_offset);

                _mm256_storeu_ps(offset_value_ptr + 32, alpha);
                _mm256_storeu_ps(offset_value_ptr + 40, beta);

                gridptr += 16;
                offset_value_ptr += 48;
            }
#endif // __AVX__

            for (; x < grid_size; x += 2)
            {
                float sample_x = *gridptr;
                float sample_y = *(gridptr + 1);

                sample_x = unormalize(src.w, sample_x);
                sample_x = get_coord(src.w, sample_x);

                sample_y = unormalize(src.h, sample_y);
                sample_y = get_coord(src.h, sample_y);

                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                bool x0_in_bound = (x0 > -1) & (x0 < src.w);
                bool x1_in_bound = (x1 > -1) & (x1 < src.w);
                bool y0_in_bound = (y0 > -1) & (y0 < src.h);
                bool y1_in_bound = (y1 > -1) & (y1 < src.h);

                bool in_bound_00 = x0_in_bound & y0_in_bound;
                bool in_bound_01 = x1_in_bound & y0_in_bound;
                bool in_bound_10 = x0_in_bound & y1_in_bound;
                bool in_bound_11 = x1_in_bound & y1_in_bound;

                int* offset_ptr = (int*)offset_value_ptr;
                float* value_ptr = offset_value_ptr + 4;

                offset_ptr[0] = in_bound_00 ? (x0 + y0 * src.w) * src.elempack : -1.0;
                offset_ptr[1] = in_bound_01 ? (x1 + y0 * src.w) * src.elempack : -1.0;
                offset_ptr[2] = in_bound_10 ? (x0 + y1 * src.w) * src.elempack : -1.0;
                offset_ptr[3] = in_bound_11 ? (x1 + y1 * src.w) * src.elempack : -1.0;

                value_ptr[0] = sample_x - x0;
                value_ptr[1] = sample_y - y0;

                gridptr += 2;
                offset_value_ptr += 6;
            }
        }
    }
    else
    {
        const float* gridptr_x = grid.channel(0);
        const float* gridptr_y = grid.channel(1);

        int x = 0;
#if __AVX__
        for (; x + 7 < grid_size; x += 8)
        {
            __m256 gx = _mm256_loadu_ps(gridptr_x);
            __m256 gy = _mm256_loadu_ps(gridptr_y);

            gx = unormalize(_mm256_set1_ps(src.w), gx);
            gx = get_coord(_mm256_set1_ps(src.w), gx);

            gy = unormalize(_mm256_set1_ps(src.h), gy);
            gy = get_coord(_mm256_set1_ps(src.h), gy);

            __m256 x_w = _mm256_floor_ps(gx);
            __m256 y_n = _mm256_floor_ps(gy);

            __m256 x1 = _mm256_add_ps(x_w, _mm256_set1_ps(1));
            __m256 y1 = _mm256_add_ps(y_n, _mm256_set1_ps(1));

            __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x_w, _CMP_GT_OS));
            __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x1, _CMP_GT_OS));
            __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y_n, _CMP_GT_OS));
            __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y1, _CMP_GT_OS));

            __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
            __m256 v01_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
            __m256 v10_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
            __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

            __m256 nw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(y_n, _mm256_set1_ps(src.w), x_w), _mm256_set1_ps(src.elempack));
            __m256 ne_offset = _mm256_add_ps(nw_offset, _mm256_set1_ps(src.elempack));
            __m256 sw_offset = _mm256_comp_fmadd_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.elempack), nw_offset);
            __m256 se_offset = _mm256_add_ps(sw_offset, _mm256_set1_ps(src.elempack));

            nw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), nw_offset, v00_in_range);
            ne_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), ne_offset, v01_in_range);
            sw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), sw_offset, v10_in_range);
            se_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), se_offset, v11_in_range);

            nw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(nw_offset));
            ne_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(ne_offset));
            sw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(sw_offset));
            se_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(se_offset));

            __m256 alpha = _mm256_sub_ps(gx, x_w);
            __m256 beta = _mm256_sub_ps(gy, y_n);

            transpose8x6_ps(nw_offset, ne_offset, sw_offset, se_offset, alpha, beta);

            _mm256_storeu_ps(offset_value_ptr, nw_offset);
            _mm256_storeu_ps(offset_value_ptr + 8, ne_offset);
            _mm256_storeu_ps(offset_value_ptr + 16, sw_offset);
            _mm256_storeu_ps(offset_value_ptr + 24, se_offset);

            _mm256_storeu_ps(offset_value_ptr + 32, alpha);
            _mm256_storeu_ps(offset_value_ptr + 40, beta);

            gridptr_x += 8;
            gridptr_y += 8;
            offset_value_ptr += 48;
        }

#endif // __AVX__

        for (; x < grid_size; x++)
        {
            float sample_x = *gridptr_x;
            float sample_y = *gridptr_y;

            sample_x = unormalize(src.w, sample_x);
            sample_x = get_coord(src.w, sample_x);

            sample_y = unormalize(src.h, sample_y);
            sample_y = get_coord(src.h, sample_y);

            int x0 = (int)floorf(sample_x);
            int y0 = (int)floorf(sample_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            bool x0_in_bound = (x0 > -1) & (x0 < src.w);
            bool x1_in_bound = (x1 > -1) & (x1 < src.w);
            bool y0_in_bound = (y0 > -1) & (y0 < src.h);
            bool y1_in_bound = (y1 > -1) & (y1 < src.h);

            bool in_bound_00 = x0_in_bound & y0_in_bound;
            bool in_bound_01 = x1_in_bound & y0_in_bound;
            bool in_bound_10 = x0_in_bound & y1_in_bound;
            bool in_bound_11 = x1_in_bound & y1_in_bound;

            int* offset_ptr = (int*)offset_value_ptr;
            float* value_ptr = offset_value_ptr + 4;

            offset_ptr[0] = in_bound_00 ? (x0 + y0 * src.w) * src.elempack : -1.0;
            offset_ptr[1] = in_bound_01 ? (x1 + y0 * src.w) * src.elempack : -1.0;
            offset_ptr[2] = in_bound_10 ? (x0 + y1 * src.w) * src.elempack : -1.0;
            offset_ptr[3] = in_bound_11 ? (x1 + y1 * src.w) * src.elempack : -1.0;

            value_ptr[0] = sample_x - x0;
            value_ptr[1] = sample_y - y0;

            gridptr_x++;
            gridptr_y++;
            offset_value_ptr += 6;
        }
    }
}

template<GridSample::PaddingMode pd, bool align_corner>
void gridsample_3d_bilinear_compute_blob(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int grid_size = grid.w * grid.h * grid.d;

    float* offset_value_ptr = offset_value.channel(0);

    grid_sample_unormalize<align_corner> unormalize;
    compute_coord<pd, align_corner> get_coord;

    if (permute_fusion == 0)
    {
        for (int y = 0; y < grid.c; y++)
        {
            const float* gridptr = grid.channel(y);
            int x = 0;
#if __AVX__
            for (; x + 23 < grid_size; x += 24)
            {
                __m256 gx = _mm256_loadu_ps(gridptr);
                __m256 gy = _mm256_loadu_ps(gridptr + 8);
                __m256 gz = _mm256_loadu_ps(gridptr + 16);

                transpose3x8_ps(gx, gy, gz);

                gx = unormalize(_mm256_set1_ps(src.w), gx);
                gx = get_coord(_mm256_set1_ps(src.w), gx);

                gy = unormalize(_mm256_set1_ps(src.h), gy);
                gy = get_coord(_mm256_set1_ps(src.h), gy);

                gz = unormalize(_mm256_set1_ps(src.d), gz);
                gz = get_coord(_mm256_set1_ps(src.d), gz);

                __m256 x_w = _mm256_floor_ps(gx);
                __m256 y_n = _mm256_floor_ps(gy);
                __m256 z_t = _mm256_floor_ps(gz);

                __m256 x1 = _mm256_add_ps(x_w, _mm256_set1_ps(1));
                __m256 y1 = _mm256_add_ps(y_n, _mm256_set1_ps(1));
                __m256 z1 = _mm256_add_ps(z_t, _mm256_set1_ps(1));

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x_w, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x1, _CMP_GT_OS));
                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y_n, _CMP_GT_OS));
                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y1, _CMP_GT_OS));
                __m256 z0_in_range = _mm256_and_ps(_mm256_cmp_ps(z_t, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.d), z_t, _CMP_GT_OS));
                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.d), z1, _CMP_GT_OS));

                __m256 v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                    __m256 v01_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                    __m256 v10_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                    v000_in_range = _mm256_and_ps(v00_in_range, z0_in_range);
                    v001_in_range = _mm256_and_ps(v01_in_range, z0_in_range);
                    v010_in_range = _mm256_and_ps(v10_in_range, z0_in_range);
                    v011_in_range = _mm256_and_ps(v11_in_range, z0_in_range);

                    v100_in_range = _mm256_and_ps(v00_in_range, z1_in_range);
                    v101_in_range = _mm256_and_ps(v01_in_range, z1_in_range);
                    v110_in_range = _mm256_and_ps(v10_in_range, z1_in_range);
                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                }

                __m256 tnw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.h)), z_t,
                                                  _mm256_comp_fmadd_ps(y_n, _mm256_set1_ps(src.w), x_w)),
                                                  _mm256_set1_ps(src.elempack));
                __m256 tne_offset = _mm256_add_ps(tnw_offset, _mm256_set1_ps(src.elempack));
                __m256 tsw_offset = _mm256_add_ps(tnw_offset, _mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.elempack)));
                __m256 tse_offset = _mm256_add_ps(tsw_offset, _mm256_set1_ps(src.elempack));

                __m256 bnw_offset = _mm256_comp_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.h)), _mm256_set1_ps(src.elempack), tnw_offset);
                __m256 bne_offset = _mm256_add_ps(bnw_offset, _mm256_set1_ps(src.elempack));
                __m256 bsw_offset = _mm256_add_ps(bnw_offset, _mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.elempack)));
                __m256 bse_offset = _mm256_add_ps(bsw_offset, _mm256_set1_ps(src.elempack));

                tnw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tnw_offset, v000_in_range);
                tne_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tne_offset, v001_in_range);
                tsw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tsw_offset, v010_in_range);
                tse_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tse_offset, v011_in_range);

                bnw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bnw_offset, v100_in_range);
                bne_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bne_offset, v101_in_range);
                bsw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bsw_offset, v110_in_range);
                bse_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bse_offset, v111_in_range);

                tnw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tnw_offset));
                tne_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tne_offset));
                tsw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tsw_offset));
                tse_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tse_offset));

                bnw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bnw_offset));
                bne_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bne_offset));
                bsw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bsw_offset));
                bse_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bse_offset));

                __m256 alpha = _mm256_sub_ps(gx, x_w);
                __m256 beta = _mm256_sub_ps(gy, y_n);
                __m256 gamma = _mm256_sub_ps(gz, z_t);

                transpose8x11_ps(tnw_offset, tne_offset, tsw_offset, tse_offset, bnw_offset, bne_offset, bsw_offset, bse_offset, alpha, beta, gamma);

                _mm256_storeu_ps(offset_value_ptr, tnw_offset);
                _mm256_storeu_ps(offset_value_ptr + 8, tne_offset);
                _mm256_storeu_ps(offset_value_ptr + 16, tsw_offset);
                _mm256_storeu_ps(offset_value_ptr + 24, tse_offset);

                _mm256_storeu_ps(offset_value_ptr + 32, bnw_offset);
                _mm256_storeu_ps(offset_value_ptr + 40, bne_offset);
                _mm256_storeu_ps(offset_value_ptr + 48, bsw_offset);
                _mm256_storeu_ps(offset_value_ptr + 56, bse_offset);

                _mm256_storeu_ps(offset_value_ptr + 64, alpha);
                _mm256_storeu_ps(offset_value_ptr + 72, beta);
                _mm256_storeu_ps(offset_value_ptr + 80, gamma);

                gridptr += 24;

                offset_value_ptr += 88;
            }
#endif // __AVX__

            for (; x < grid_size; x += 3)
            {
                float sample_x = *gridptr;
                float sample_y = *(gridptr + 1);
                float sample_z = *(gridptr + 2);

                sample_x = unormalize(src.w, sample_x);
                sample_x = get_coord(src.w, sample_x);

                sample_y = unormalize(src.h, sample_y);
                sample_y = get_coord(src.h, sample_y);

                sample_z = unormalize(src.d, sample_z);
                sample_z = get_coord(src.d, sample_z);

                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int z0 = (int)floorf(sample_z);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                bool x0_in_range = (x0 > -1) & (x0 < src.w);
                bool y0_in_range = (y0 > -1) & (y0 < src.h);
                bool z0_in_range = (z0 > -1) & (z0 < src.d);
                bool x1_in_range = (x1 > -1) & (x1 < src.w);
                bool y1_in_range = (y1 > -1) & (y1 < src.h);
                bool z1_in_range = (z1 > -1) & (z1 < src.d);

                bool v00_in_range = x0_in_range & y0_in_range;
                bool v01_in_range = x1_in_range & y0_in_range;
                bool v10_in_range = x0_in_range & y1_in_range;
                bool v11_in_range = x1_in_range & y1_in_range;

                bool in_bound_000 = v00_in_range & z0_in_range;
                bool in_bound_001 = v01_in_range & z0_in_range;
                bool in_bound_010 = v10_in_range & z0_in_range;
                bool in_bound_011 = v11_in_range & z0_in_range;

                bool in_bound_100 = v00_in_range & z1_in_range;
                bool in_bound_101 = v01_in_range & z1_in_range;
                bool in_bound_110 = v10_in_range & z1_in_range;
                bool in_bound_111 = v11_in_range & z1_in_range;

                int* offset_ptr = (int*)offset_value_ptr;
                float* value_ptr = offset_value_ptr + 8;

                offset_ptr[0] = in_bound_000 ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;
                offset_ptr[1] = in_bound_001 ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;
                offset_ptr[2] = in_bound_010 ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;
                offset_ptr[3] = in_bound_011 ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;

                offset_ptr[4] = in_bound_100 ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;
                offset_ptr[5] = in_bound_101 ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;
                offset_ptr[6] = in_bound_110 ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;
                offset_ptr[7] = in_bound_111 ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;

                value_ptr[0] = sample_x - x0;
                value_ptr[1] = sample_y - y0;
                value_ptr[2] = sample_z - z0;

                gridptr += 3;
                offset_value_ptr += 11;
            }
        }
    }
    else
    {
        const float* gridptr_x = grid.channel(0);
        const float* gridptr_y = grid.channel(1);
        const float* gridptr_z = grid.channel(2);

        int x = 0;
#if __AVX__
        for (; x + 7 < grid_size; x += 8)
        {
            __m256 gx = _mm256_loadu_ps(gridptr_x);
            __m256 gy = _mm256_loadu_ps(gridptr_y);
            __m256 gz = _mm256_loadu_ps(gridptr_z);

            gx = unormalize(_mm256_set1_ps(src.w), gx);
            gx = get_coord(_mm256_set1_ps(src.w), gx);

            gy = unormalize(_mm256_set1_ps(src.h), gy);
            gy = get_coord(_mm256_set1_ps(src.h), gy);

            gz = unormalize(_mm256_set1_ps(src.d), gz);
            gz = get_coord(_mm256_set1_ps(src.d), gz);

            __m256 x_w = _mm256_floor_ps(gx);
            __m256 y_n = _mm256_floor_ps(gy);
            __m256 z_t = _mm256_floor_ps(gz);

            __m256 x1 = _mm256_add_ps(x_w, _mm256_set1_ps(1));
            __m256 y1 = _mm256_add_ps(y_n, _mm256_set1_ps(1));
            __m256 z1 = _mm256_add_ps(z_t, _mm256_set1_ps(1));

            __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x_w, _CMP_GT_OS));
            __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), x1, _CMP_GT_OS));
            __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y_n, _CMP_GT_OS));
            __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), y1, _CMP_GT_OS));
            __m256 z0_in_range = _mm256_and_ps(_mm256_cmp_ps(z_t, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.d), z_t, _CMP_GT_OS));
            __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.d), z1, _CMP_GT_OS));

            __m256 v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
            {
                __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                __m256 v01_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                __m256 v10_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                v000_in_range = _mm256_and_ps(v00_in_range, z0_in_range);
                v001_in_range = _mm256_and_ps(v01_in_range, z0_in_range);
                v010_in_range = _mm256_and_ps(v10_in_range, z0_in_range);
                v011_in_range = _mm256_and_ps(v11_in_range, z0_in_range);

                v100_in_range = _mm256_and_ps(v00_in_range, z1_in_range);
                v101_in_range = _mm256_and_ps(v01_in_range, z1_in_range);
                v110_in_range = _mm256_and_ps(v10_in_range, z1_in_range);
                v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
            }

            __m256 tnw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.h)), z_t,
                                              _mm256_comp_fmadd_ps(y_n, _mm256_set1_ps(src.w), x_w)),
                                              _mm256_set1_ps(src.elempack));
            __m256 tne_offset = _mm256_add_ps(tnw_offset, _mm256_set1_ps(src.elempack));
            __m256 tsw_offset = _mm256_add_ps(tnw_offset, _mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.elempack)));
            __m256 tse_offset = _mm256_add_ps(tsw_offset, _mm256_set1_ps(src.elempack));

            __m256 bnw_offset = _mm256_comp_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.h)), _mm256_set1_ps(src.elempack), tnw_offset);
            __m256 bne_offset = _mm256_add_ps(bnw_offset, _mm256_set1_ps(src.elempack));
            __m256 bsw_offset = _mm256_add_ps(bnw_offset, _mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.elempack)));
            __m256 bse_offset = _mm256_add_ps(bsw_offset, _mm256_set1_ps(src.elempack));

            tnw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tnw_offset, v000_in_range);
            tne_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tne_offset, v001_in_range);
            tsw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tsw_offset, v010_in_range);
            tse_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), tse_offset, v011_in_range);

            bnw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bnw_offset, v100_in_range);
            bne_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bne_offset, v101_in_range);
            bsw_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bsw_offset, v110_in_range);
            bse_offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), bse_offset, v111_in_range);

            tnw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tnw_offset));
            tne_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tne_offset));
            tsw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tsw_offset));
            tse_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(tse_offset));

            bnw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bnw_offset));
            bne_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bne_offset));
            bsw_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bsw_offset));
            bse_offset = _mm256_castsi256_ps(_mm256_cvtps_epi32(bse_offset));

            __m256 alpha = _mm256_sub_ps(gx, x_w);
            __m256 beta = _mm256_sub_ps(gy, y_n);
            __m256 gamma = _mm256_sub_ps(gz, z_t);

            transpose8x11_ps(tnw_offset, tne_offset, tsw_offset, tse_offset, bnw_offset, bne_offset, bsw_offset, bse_offset, alpha, beta, gamma);

            _mm256_storeu_ps(offset_value_ptr, tnw_offset);
            _mm256_storeu_ps(offset_value_ptr + 8, tne_offset);
            _mm256_storeu_ps(offset_value_ptr + 16, tsw_offset);
            _mm256_storeu_ps(offset_value_ptr + 24, tse_offset);

            _mm256_storeu_ps(offset_value_ptr + 32, bnw_offset);
            _mm256_storeu_ps(offset_value_ptr + 40, bne_offset);
            _mm256_storeu_ps(offset_value_ptr + 48, bsw_offset);
            _mm256_storeu_ps(offset_value_ptr + 56, bse_offset);

            _mm256_storeu_ps(offset_value_ptr + 64, alpha);
            _mm256_storeu_ps(offset_value_ptr + 72, beta);
            _mm256_storeu_ps(offset_value_ptr + 80, gamma);

            gridptr_x += 8;
            gridptr_y += 8;
            gridptr_z += 8;

            offset_value_ptr += 88;
        }
#endif // __AVX__

        for (; x < grid_size; x++)
        {
            float sample_x = *gridptr_x;
            float sample_y = *gridptr_y;
            float sample_z = *gridptr_z;

            sample_x = unormalize(src.w, sample_x);
            sample_x = get_coord(src.w, sample_x);

            sample_y = unormalize(src.h, sample_y);
            sample_y = get_coord(src.h, sample_y);

            sample_z = unormalize(src.d, sample_z);
            sample_z = get_coord(src.d, sample_z);

            int x0 = (int)floorf(sample_x);
            int y0 = (int)floorf(sample_y);
            int z0 = (int)floorf(sample_z);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            int z1 = z0 + 1;

            bool x0_in_range = (x0 > -1) & (x0 < src.w);
            bool y0_in_range = (y0 > -1) & (y0 < src.h);
            bool z0_in_range = (z0 > -1) & (z0 < src.d);
            bool x1_in_range = (x1 > -1) & (x1 < src.w);
            bool y1_in_range = (y1 > -1) & (y1 < src.h);
            bool z1_in_range = (z1 > -1) & (z1 < src.d);

            bool v00_in_range = x0_in_range & y0_in_range;
            bool v01_in_range = x1_in_range & y0_in_range;
            bool v10_in_range = x0_in_range & y1_in_range;
            bool v11_in_range = x1_in_range & y1_in_range;

            bool in_bound_000 = v00_in_range & z0_in_range;
            bool in_bound_001 = v01_in_range & z0_in_range;
            bool in_bound_010 = v10_in_range & z0_in_range;
            bool in_bound_011 = v11_in_range & z0_in_range;

            bool in_bound_100 = v00_in_range & z1_in_range;
            bool in_bound_101 = v01_in_range & z1_in_range;
            bool in_bound_110 = v10_in_range & z1_in_range;
            bool in_bound_111 = v11_in_range & z1_in_range;

            int* offset_ptr = (int*)offset_value_ptr;
            float* value_ptr = offset_value_ptr + 8;

            offset_ptr[0] = in_bound_000 ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;
            offset_ptr[1] = in_bound_001 ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;
            offset_ptr[2] = in_bound_010 ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;
            offset_ptr[3] = in_bound_011 ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;

            offset_ptr[4] = in_bound_100 ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;
            offset_ptr[5] = in_bound_101 ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;
            offset_ptr[6] = in_bound_110 ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;
            offset_ptr[7] = in_bound_111 ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0;

            value_ptr[0] = sample_x - x0;
            value_ptr[1] = sample_y - y0;
            value_ptr[2] = sample_z - z0;

            gridptr_x++;
            gridptr_y++;
            gridptr_z++;
            offset_value_ptr += 11;
        }
    }
}
