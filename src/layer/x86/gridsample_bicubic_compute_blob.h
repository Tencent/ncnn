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
void gridsample_2d_bicubic_compute_blob(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
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
                gy = unormalize(_mm256_set1_ps(src.h), gy);

                __m256 gx_floor = _mm256_floor_ps(gx);
                __m256 gy_floor = _mm256_floor_ps(gy);

                __m256 tx = _mm256_sub_ps(gx, gx_floor);
                __m256 ty = _mm256_sub_ps(gy, gy_floor);

                __m256 gx0 = _mm256_add_ps(gx_floor, _mm256_set1_ps(-1));
                __m256 gx1 = gx_floor;
                __m256 gx2 = _mm256_add_ps(gx_floor, _mm256_set1_ps(1));
                __m256 gx3 = _mm256_add_ps(gx2, _mm256_set1_ps(1));

                gx0 = get_coord(_mm256_set1_ps(src.w), gx0);
                gx1 = get_coord(_mm256_set1_ps(src.w), gx1);
                gx2 = get_coord(_mm256_set1_ps(src.w), gx2);
                gx3 = get_coord(_mm256_set1_ps(src.w), gx3);

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(gx0, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx0, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(gx1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx1, _CMP_GT_OS));
                __m256 x2_in_range = _mm256_and_ps(_mm256_cmp_ps(gx2, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx2, _CMP_GT_OS));
                __m256 x3_in_range = _mm256_and_ps(_mm256_cmp_ps(gx3, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx3, _CMP_GT_OS));
                __m256 v0_offset_f[4], v1_offset_f[4], v2_offset_f[4], v3_offset_f[4];
                for (int i = 0; i < 4; i++)
                {
                    gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                    gy = get_coord(_mm256_set1_ps(src.h), gy);

                    __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS));

                    __m256 gy_offset = _mm256_mul_ps(gy, _mm256_set1_ps(src.w));

                    v0_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx0), _mm256_set1_ps(src.elempack));
                    v1_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx1), _mm256_set1_ps(src.elempack));
                    v2_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx2), _mm256_set1_ps(src.elempack));
                    v3_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx3), _mm256_set1_ps(src.elempack));

                    v0_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v0_offset_f[i], _mm256_and_ps(x0_in_range, y_in_range));
                    v1_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v1_offset_f[i], _mm256_and_ps(x1_in_range, y_in_range));
                    v2_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v2_offset_f[i], _mm256_and_ps(x2_in_range, y_in_range));
                    v3_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v3_offset_f[i], _mm256_and_ps(x3_in_range, y_in_range));

                    v0_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v0_offset_f[i]));
                    v1_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v1_offset_f[i]));
                    v2_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v2_offset_f[i]));
                    v3_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v3_offset_f[i]));
                }

                transpose8x18_ps(tx, ty, v0_offset_f[0], v1_offset_f[0], v2_offset_f[0], v3_offset_f[0], v0_offset_f[1], v1_offset_f[1], v2_offset_f[1], v3_offset_f[1], v0_offset_f[2], v1_offset_f[2], v2_offset_f[2], v3_offset_f[2], v0_offset_f[3], v1_offset_f[3], v2_offset_f[3], v3_offset_f[3]);

                _mm256_storeu_ps(offset_value_ptr, tx);
                _mm256_storeu_ps(offset_value_ptr + 8, ty);
                offset_value_ptr += 16;
                for (int i = 0; i < 4; i++)
                {
                    _mm256_storeu_ps(offset_value_ptr, v0_offset_f[i]);
                    _mm256_storeu_ps(offset_value_ptr + 8, v1_offset_f[i]);
                    _mm256_storeu_ps(offset_value_ptr + 16, v2_offset_f[i]);
                    _mm256_storeu_ps(offset_value_ptr + 24, v3_offset_f[i]);
                    offset_value_ptr += 32;
                }
                gridptr += 16;
            }

#endif // __AVX__

            for (; x < grid_size; x += 2)
            {
                float sample_x = *gridptr;
                float sample_y = *(gridptr + 1);

                sample_x = unormalize(src.w, sample_x);
                sample_y = unormalize(src.h, sample_y);

                int x1 = floorf(sample_x);
                int y1 = floorf(sample_y);
                int x0 = x1 - 1;
                int x2 = x1 + 1;
                int x3 = x1 + 2;

                offset_value_ptr[0] = sample_x - static_cast<float>(x1);
                offset_value_ptr[1] = sample_y - static_cast<float>(y1);

                x1 = get_coord(src.w, x1);
                x0 = get_coord(src.w, x0);
                x2 = get_coord(src.w, x2);
                x3 = get_coord(src.w, x3);

                bool x1_in_range = (x1 > -1) & (x1 < src.w);
                bool x0_in_range = (x0 > -1) & (x0 < src.w);
                bool x2_in_range = (x2 > -1) & (x2 < src.w);
                bool x3_in_range = (x3 > -1) & (x3 < src.w);

                int* offset_ptr = (int*)offset_value_ptr + 2;

                for (int i = 0; i < 4; i++)
                {
                    int gy = y1 + i - 1;
                    gy = get_coord(src.h, gy);
                    int offset_y = gy * src.w;

                    bool y_in_range = (gy > -1) & (gy < src.h);

                    bool v0_in_bound = (x0_in_range & y_in_range);
                    bool v1_in_bound = (x1_in_range & y_in_range);
                    bool v2_in_bound = (x2_in_range & y_in_range);
                    bool v3_in_bound = (x3_in_range & y_in_range);

                    offset_ptr[0] = v0_in_bound ? (offset_y + x0) * src.elempack : -1.0;
                    offset_ptr[1] = v1_in_bound ? (offset_y + x1) * src.elempack : -1.0;
                    offset_ptr[2] = v2_in_bound ? (offset_y + x2) * src.elempack : -1.0;
                    offset_ptr[3] = v3_in_bound ? (offset_y + x3) * src.elempack : -1.0;

                    offset_ptr += 4;
                }

                gridptr += 2;
                offset_value_ptr += 18;
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
            gy = unormalize(_mm256_set1_ps(src.h), gy);

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            __m256 tx = _mm256_sub_ps(gx, gx_floor);
            __m256 ty = _mm256_sub_ps(gy, gy_floor);

            __m256 gx0 = _mm256_add_ps(gx_floor, _mm256_set1_ps(-1));
            __m256 gx1 = gx_floor;
            __m256 gx2 = _mm256_add_ps(gx_floor, _mm256_set1_ps(1));
            __m256 gx3 = _mm256_add_ps(gx2, _mm256_set1_ps(1));

            gx0 = get_coord(_mm256_set1_ps(src.w), gx0);
            gx1 = get_coord(_mm256_set1_ps(src.w), gx1);
            gx2 = get_coord(_mm256_set1_ps(src.w), gx2);
            gx3 = get_coord(_mm256_set1_ps(src.w), gx3);

            __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(gx0, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx0, _CMP_GT_OS));
            __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(gx1, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx1, _CMP_GT_OS));
            __m256 x2_in_range = _mm256_and_ps(_mm256_cmp_ps(gx2, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx2, _CMP_GT_OS));
            __m256 x3_in_range = _mm256_and_ps(_mm256_cmp_ps(gx3, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.w), gx3, _CMP_GT_OS));

            __m256 v0_offset_f[4], v1_offset_f[4], v2_offset_f[4], v3_offset_f[4];
            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                gy = get_coord(_mm256_set1_ps(src.h), gy);

                __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS));

                __m256 gy_offset = _mm256_mul_ps(gy, _mm256_set1_ps(src.w));

                v0_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx0), _mm256_set1_ps(src.elempack));
                v1_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx1), _mm256_set1_ps(src.elempack));
                v2_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx2), _mm256_set1_ps(src.elempack));
                v3_offset_f[i] = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx3), _mm256_set1_ps(src.elempack));

                v0_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v0_offset_f[i], _mm256_and_ps(x0_in_range, y_in_range));
                v1_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v1_offset_f[i], _mm256_and_ps(x1_in_range, y_in_range));
                v2_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v2_offset_f[i], _mm256_and_ps(x2_in_range, y_in_range));
                v3_offset_f[i] = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v3_offset_f[i], _mm256_and_ps(x3_in_range, y_in_range));

                v0_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v0_offset_f[i]));
                v1_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v1_offset_f[i]));
                v2_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v2_offset_f[i]));
                v3_offset_f[i] = _mm256_castsi256_ps(_mm256_cvtps_epi32(v3_offset_f[i]));
            }

            transpose8x18_ps(tx, ty, v0_offset_f[0], v1_offset_f[0], v2_offset_f[0], v3_offset_f[0], v0_offset_f[1], v1_offset_f[1], v2_offset_f[1], v3_offset_f[1], v0_offset_f[2], v1_offset_f[2], v2_offset_f[2], v3_offset_f[2], v0_offset_f[3], v1_offset_f[3], v2_offset_f[3], v3_offset_f[3]);

            _mm256_storeu_ps(offset_value_ptr, tx);
            _mm256_storeu_ps(offset_value_ptr + 8, ty);
            offset_value_ptr += 16;
            for (int i = 0; i < 4; i++)
            {
                _mm256_storeu_ps(offset_value_ptr, v0_offset_f[i]);
                _mm256_storeu_ps(offset_value_ptr + 8, v1_offset_f[i]);
                _mm256_storeu_ps(offset_value_ptr + 16, v2_offset_f[i]);
                _mm256_storeu_ps(offset_value_ptr + 24, v3_offset_f[i]);
                offset_value_ptr += 32;
            }

            gridptr_x += 8;
            gridptr_y += 8;
        }

#endif // __AVX__

        for (; x < grid_size; x++)
        {
            float sample_x = *gridptr_x;
            float sample_y = *gridptr_y;

            sample_x = unormalize(src.w, sample_x);
            sample_y = unormalize(src.h, sample_y);

            int x1 = floorf(sample_x);
            int y1 = floorf(sample_y);
            int x0 = x1 - 1;
            int x2 = x1 + 1;
            int x3 = x1 + 2;

            offset_value_ptr[0] = sample_x - static_cast<float>(x1);
            offset_value_ptr[1] = sample_y - static_cast<float>(y1);

            x1 = get_coord(src.w, x1);
            x0 = get_coord(src.w, x0);
            x2 = get_coord(src.w, x2);
            x3 = get_coord(src.w, x3);

            bool x1_in_range = (x1 > -1) & (x1 < src.w);
            bool x0_in_range = (x0 > -1) & (x0 < src.w);
            bool x2_in_range = (x2 > -1) & (x2 < src.w);
            bool x3_in_range = (x3 > -1) & (x3 < src.w);

            int* offset_ptr = (int*)offset_value_ptr + 2;

            for (int i = 0; i < 4; i++)
            {
                int gy = y1 + i - 1;
                gy = get_coord(src.h, gy);
                int offset_y = gy * src.w;

                bool y_in_range = (gy > -1) & (gy < src.h);

                bool v0_in_bound = (x0_in_range & y_in_range);
                bool v1_in_bound = (x1_in_range & y_in_range);
                bool v2_in_bound = (x2_in_range & y_in_range);
                bool v3_in_bound = (x3_in_range & y_in_range);

                offset_ptr[0] = v0_in_bound ? (offset_y + x0) * src.elempack : -1.0;
                offset_ptr[1] = v1_in_bound ? (offset_y + x1) * src.elempack : -1.0;
                offset_ptr[2] = v2_in_bound ? (offset_y + x2) * src.elempack : -1.0;
                offset_ptr[3] = v3_in_bound ? (offset_y + x3) * src.elempack : -1.0;

                offset_ptr += 4;
            }

            gridptr_x++;
            gridptr_y++;

            offset_value_ptr += 18;
        }
    }
}
