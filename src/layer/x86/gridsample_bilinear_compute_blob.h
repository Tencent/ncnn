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
void gridsample_2d_bilinear_compute_blob(const Mat& src, const Mat& grid, Mat& offset, Mat& value, int permute_fusion, const Option& opt)
{
    const int grid_size = grid.w * grid.h;

    float* offset_ptr_00 = offset.channel(0);
    float* offset_ptr_01 = offset.channel(1);
    float* offset_ptr_10 = offset.channel(2);
    float* offset_ptr_11 = offset.channel(3);

    float* value_ptr_alpha = value.channel(0);
    float* value_ptr_beta = value.channel(1);

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
                __m256 tmp_x = _mm256_loadu_ps(gridptr);
                __m256 gy = _mm256_loadu_ps(gridptr + 8);

                __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
                gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
                tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

                gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
                gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

                {
                    gx = unormalize(_mm256_set1_ps(src.w), gx);
                    gx = get_coord(_mm256_set1_ps(src.w), gx);

                    gy = unormalize(_mm256_set1_ps(src.h), gy);
                    gy = get_coord(_mm256_set1_ps(src.h), gy);
                }

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

                _mm256_storeu_ps(offset_ptr_00, nw_offset);
                _mm256_storeu_ps(offset_ptr_01, ne_offset);
                _mm256_storeu_ps(offset_ptr_10, sw_offset);
                _mm256_storeu_ps(offset_ptr_11, se_offset);

                __m256 alpha = _mm256_sub_ps(gx, x_w);
                __m256 beta = _mm256_sub_ps(gy, y_n);

                _mm256_storeu_ps(value_ptr_alpha, alpha);
                _mm256_storeu_ps(value_ptr_beta, beta);

                _mm256_storeu_ps(value_ptr_alpha, alpha);
                _mm256_storeu_ps(value_ptr_beta, beta);

                gridptr += 16;

                offset_ptr_00 += 8;
                offset_ptr_01 += 8;
                offset_ptr_10 += 8;
                offset_ptr_11 += 8;

                value_ptr_alpha += 8;
                value_ptr_beta += 8;
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

                int x0 = (int)floor(sample_x);
                int y0 = (int)floor(sample_y);
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

                *offset_ptr_00 = in_bound_00 ? (x0 + y0 * src.w) * src.elempack : -1.0f;
                *offset_ptr_01 = in_bound_01 ? (x1 + y0 * src.w) * src.elempack : -1.0f;
                *offset_ptr_10 = in_bound_10 ? (x0 + y1 * src.w) * src.elempack : -1.0f;
                *offset_ptr_11 = in_bound_11 ? (x1 + y1 * src.w) * src.elempack : -1.0f;

                *value_ptr_alpha = sample_x - x0;
                *value_ptr_beta = sample_y - y0;

                gridptr += 2;

                offset_ptr_00++;
                offset_ptr_01++;
                offset_ptr_10++;
                offset_ptr_11++;

                value_ptr_alpha++;
                value_ptr_beta++;
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

            {
                gx = unormalize(_mm256_set1_ps(src.w), gx);
                gx = get_coord(_mm256_set1_ps(src.w), gx);

                gy = unormalize(_mm256_set1_ps(src.h), gy);
                gy = get_coord(_mm256_set1_ps(src.h), gy);
            }

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

            _mm256_storeu_ps(offset_ptr_00, nw_offset);
            _mm256_storeu_ps(offset_ptr_01, ne_offset);
            _mm256_storeu_ps(offset_ptr_10, sw_offset);
            _mm256_storeu_ps(offset_ptr_11, se_offset);

            __m256 alpha = _mm256_sub_ps(gx, x_w);
            __m256 beta = _mm256_sub_ps(gy, y_n);

            _mm256_storeu_ps(value_ptr_alpha, alpha);
            _mm256_storeu_ps(value_ptr_beta, beta);

            gridptr_x += 8;
            gridptr_y += 8;

            offset_ptr_00 += 8;
            offset_ptr_01 += 8;
            offset_ptr_10 += 8;
            offset_ptr_11 += 8;

            value_ptr_alpha += 8;
            value_ptr_beta += 8;
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

            int x0 = (int)floor(sample_x);
            int y0 = (int)floor(sample_y);
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

            *offset_ptr_00 = in_bound_00 ? (x0 + y0 * src.w) * src.elempack : -1.0f;
            *offset_ptr_01 = in_bound_01 ? (x1 + y0 * src.w) * src.elempack : -1.0f;
            *offset_ptr_10 = in_bound_10 ? (x0 + y1 * src.w) * src.elempack : -1.0f;
            *offset_ptr_11 = in_bound_11 ? (x1 + y1 * src.w) * src.elempack : -1.0f;

            *value_ptr_alpha = sample_x - x0;
            *value_ptr_beta = sample_y - y0;

            gridptr_x++;
            gridptr_y++;

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;
        }
    }
}

template<GridSample::PaddingMode pd, bool align_corner>
void gridsample_3d_bilinear_compute_blob(const Mat& src, const Mat& grid, Mat& offset, Mat& value, int permute_fusion, const Option& opt)
{
    const int grid_size = grid.w * grid.h * grid.d;

    float* offset_ptr_000 = offset.channel(0);
    float* offset_ptr_001 = offset.channel(1);
    float* offset_ptr_010 = offset.channel(2);
    float* offset_ptr_011 = offset.channel(3);

    float* offset_ptr_100 = offset.channel(4);
    float* offset_ptr_101 = offset.channel(5);
    float* offset_ptr_110 = offset.channel(6);
    float* offset_ptr_111 = offset.channel(7);

    float* value_ptr_alpha = value.channel(0);
    float* value_ptr_beta = value.channel(1);
    float* value_ptr_gamma = value.channel(2);

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
                __m256 tmp_x = _mm256_loadu_ps(gridptr);
                __m256 tmp_y = _mm256_loadu_ps(gridptr + 8);
                __m256 gz = _mm256_loadu_ps(gridptr + 16);

                __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
                __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
                gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

                tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
                tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

                gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
                gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
                gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

                {
                    gx = unormalize(_mm256_set1_ps(src.w), gx);
                    gx = get_coord(_mm256_set1_ps(src.w), gx);

                    gy = unormalize(_mm256_set1_ps(src.h), gy);
                    gy = get_coord(_mm256_set1_ps(src.h), gy);

                    gz = unormalize(_mm256_set1_ps(src.d), gz);
                    gz = get_coord(_mm256_set1_ps(src.d), gz);
                }

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

                _mm256_storeu_ps(offset_ptr_000, tnw_offset);
                _mm256_storeu_ps(offset_ptr_001, tne_offset);
                _mm256_storeu_ps(offset_ptr_010, tsw_offset);
                _mm256_storeu_ps(offset_ptr_011, tse_offset);

                _mm256_storeu_ps(offset_ptr_100, bnw_offset);
                _mm256_storeu_ps(offset_ptr_101, bne_offset);
                _mm256_storeu_ps(offset_ptr_110, bsw_offset);
                _mm256_storeu_ps(offset_ptr_111, bse_offset);

                __m256 alpha = _mm256_sub_ps(gx, x_w);
                __m256 beta = _mm256_sub_ps(gy, y_n);
                __m256 gamma = _mm256_sub_ps(gz, z_t);

                _mm256_storeu_ps(value_ptr_alpha, alpha);
                _mm256_storeu_ps(value_ptr_beta, beta);
                _mm256_storeu_ps(value_ptr_gamma, gamma);

                gridptr += 24;

                offset_ptr_000 += 8;
                offset_ptr_001 += 8;
                offset_ptr_010 += 8;
                offset_ptr_011 += 8;

                offset_ptr_100 += 8;
                offset_ptr_101 += 8;
                offset_ptr_110 += 8;
                offset_ptr_111 += 8;

                value_ptr_alpha += 8;
                value_ptr_beta += 8;
                value_ptr_gamma += 8;
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

                int x0 = (int)floor(sample_x);
                int y0 = (int)floor(sample_y);
                int z0 = (int)floor(sample_z);
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

                *offset_ptr_000 = in_bound_000 ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;
                *offset_ptr_001 = in_bound_001 ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;
                *offset_ptr_010 = in_bound_010 ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;
                *offset_ptr_011 = in_bound_011 ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;

                *offset_ptr_100 = in_bound_100 ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;
                *offset_ptr_101 = in_bound_101 ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;
                *offset_ptr_110 = in_bound_110 ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;
                *offset_ptr_111 = in_bound_111 ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;

                *value_ptr_alpha = sample_x - x0;
                *value_ptr_beta = sample_y - y0;
                *value_ptr_gamma = sample_z - z0;

                gridptr += 3;

                offset_ptr_000++;
                offset_ptr_001++;
                offset_ptr_010++;
                offset_ptr_011++;

                offset_ptr_100++;
                offset_ptr_101++;
                offset_ptr_110++;
                offset_ptr_111++;

                value_ptr_alpha++;
                value_ptr_beta++;
                value_ptr_gamma++;
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

            {
                gx = unormalize(_mm256_set1_ps(src.w), gx);
                gx = get_coord(_mm256_set1_ps(src.w), gx);

                gy = unormalize(_mm256_set1_ps(src.h), gy);
                gy = get_coord(_mm256_set1_ps(src.h), gy);

                gz = unormalize(_mm256_set1_ps(src.d), gz);
                gz = get_coord(_mm256_set1_ps(src.d), gz);
            }

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

            _mm256_storeu_ps(offset_ptr_000, tnw_offset);
            _mm256_storeu_ps(offset_ptr_001, tne_offset);
            _mm256_storeu_ps(offset_ptr_010, tsw_offset);
            _mm256_storeu_ps(offset_ptr_011, tse_offset);

            _mm256_storeu_ps(offset_ptr_100, bnw_offset);
            _mm256_storeu_ps(offset_ptr_101, bne_offset);
            _mm256_storeu_ps(offset_ptr_110, bsw_offset);
            _mm256_storeu_ps(offset_ptr_111, bse_offset);

            __m256 alpha = _mm256_sub_ps(gx, x_w);
            __m256 beta = _mm256_sub_ps(gy, y_n);
            __m256 gamma = _mm256_sub_ps(gz, z_t);

            _mm256_storeu_ps(value_ptr_alpha, alpha);
            _mm256_storeu_ps(value_ptr_beta, beta);
            _mm256_storeu_ps(value_ptr_gamma, gamma);

            gridptr_x += 8;
            gridptr_y += 8;
            gridptr_z += 8;

            offset_ptr_000 += 8;
            offset_ptr_001 += 8;
            offset_ptr_010 += 8;
            offset_ptr_011 += 8;

            offset_ptr_100 += 8;
            offset_ptr_101 += 8;
            offset_ptr_110 += 8;
            offset_ptr_111 += 8;

            value_ptr_alpha += 8;
            value_ptr_beta += 8;
            value_ptr_gamma += 8;
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

            int x0 = (int)floor(sample_x);
            int y0 = (int)floor(sample_y);
            int z0 = (int)floor(sample_z);
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

            *offset_ptr_000 = in_bound_000 ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;
            *offset_ptr_001 = in_bound_001 ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;
            *offset_ptr_010 = in_bound_010 ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;
            *offset_ptr_011 = in_bound_011 ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1.0f;

            *offset_ptr_100 = in_bound_100 ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;
            *offset_ptr_101 = in_bound_101 ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;
            *offset_ptr_110 = in_bound_110 ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;
            *offset_ptr_111 = in_bound_111 ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1.0f;

            *value_ptr_alpha = sample_x - x0;
            *value_ptr_beta = sample_y - y0;
            *value_ptr_gamma = sample_z - z0;

            gridptr_x++;
            gridptr_y++;
            gridptr_z++;

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;
        }
    }
}
