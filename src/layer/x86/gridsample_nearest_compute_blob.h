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
void gridsample_2d_nearest_compute_blob(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int grid_size = grid.w * grid.h;

    float* offset_ptr = offset_value.channel(0);

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

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx, _CMP_GT_OS)),
                                                  _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS)));

                __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(gy, _mm256_set1_ps(src.w), gx), _mm256_set1_ps(src.elempack));

                offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_castsi256_ps(_mm256_cvtps_epi32(offset)), v_in_range);

                _mm256_storeu_ps(offset_ptr, offset);

                gridptr += 16;
                offset_ptr += 8;
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

                int x0 = static_cast<int>(floorf(sample_x + 0.5f));
                int y0 = static_cast<int>(floorf(sample_y + 0.5f));

                bool in_bound = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h));

                int* iptr = (int*)offset_ptr;
                *iptr = in_bound ? (x0 + y0 * src.w) * src.elempack : -1.0;

                gridptr += 2;
                offset_ptr++;
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

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

            __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx, _CMP_GT_OS)),
                                              _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS)));

            __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(gy, _mm256_set1_ps(src.w), gx), _mm256_set1_ps(src.elempack));

            offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_castsi256_ps(_mm256_cvtps_epi32(offset)), v_in_range);

            _mm256_storeu_ps(offset_ptr, offset);

            gridptr_x += 8;
            gridptr_y += 8;
            offset_ptr += 8;
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

            int x0 = static_cast<int>(floorf(sample_x + 0.5f));
            int y0 = static_cast<int>(floorf(sample_y + 0.5f));

            bool in_bound = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h));

            int* iptr = (int*)offset_ptr;
            *iptr = in_bound ? (x0 + y0 * src.w) * src.elempack : -1.0;

            gridptr_x++;
            gridptr_y++;

            offset_ptr++;
        }
    }
}

template<GridSample::PaddingMode pd, bool align_corner>
void gridsample_3d_nearest_compute_blob(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int grid_size = grid.w * grid.h * grid.d;

    float* offset_ptr = offset_value.channel(0);

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

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx, _CMP_GT_OS)),
                                                  _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS)));
                v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.d), gz, _CMP_GT_OS)));

                __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.h)), gz,
                                              _mm256_comp_fmadd_ps(gy, _mm256_set1_ps(src.w), gx)),
                                              _mm256_set1_ps(src.elempack));

                offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_castsi256_ps(_mm256_cvtps_epi32(offset)), v_in_range);

                _mm256_storeu_ps(offset_ptr, offset);

                gridptr += 24;
                offset_ptr += 8;
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

                int x0 = static_cast<int>(floorf(sample_x + 0.5f));
                int y0 = static_cast<int>(floorf(sample_y + 0.5f));
                int z0 = static_cast<int>(floorf(sample_z + 0.5f));

                bool in_bound = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h) & (z0 > -1) & (z0 < src.d));

                int* iptr = (int*)offset_ptr;
                *iptr = in_bound ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;

                gridptr += 3;
                offset_ptr++;
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

            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
            gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

            __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx, _CMP_GT_OS)),
                                              _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS)));
            v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.d), gz, _CMP_GT_OS)));

            __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(src.w), _mm256_set1_ps(src.h)), gz,
                                          _mm256_comp_fmadd_ps(gy, _mm256_set1_ps(src.w), gx)),
                                          _mm256_set1_ps(src.elempack));

            offset = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_castsi256_ps(_mm256_cvtps_epi32(offset)), v_in_range);

            _mm256_storeu_ps(offset_ptr, offset);

            gridptr_x += 8;
            gridptr_y += 8;
            gridptr_z += 8;

            offset_ptr += 8;
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

            int x0 = static_cast<int>(floorf(sample_x + 0.5f));
            int y0 = static_cast<int>(floorf(sample_y + 0.5f));
            int z0 = static_cast<int>(floorf(sample_z + 0.5f));

            bool in_bound = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h) & (z0 > -1) & (z0 < src.d));

            int* iptr = (int*)offset_ptr;
            *iptr = in_bound ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1.0;

            gridptr_x++;
            gridptr_y++;
            gridptr_z++;

            offset_ptr++;
        }
    }
}
