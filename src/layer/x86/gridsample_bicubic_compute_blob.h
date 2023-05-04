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
void gridsample_2d_bicubic_compute_blob(const Mat& src, const Mat& grid, Mat& offset, Mat& value, int permute_fusion, const Option& opt)
{
    const int grid_size = grid.w * grid.h;

    float *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

    float *v0_in_bound_ptr[4], *v1_in_bound_ptr[4], *v2_in_bound_ptr[4], *v3_in_bound_ptr[4];

    float* value_x = value.channel(0);
    float* value_y = value.channel(1);

    for (int i = 0; i < 4; i++)
    {
        v0_offset_ptr[i] = offset.channel(i * 4 + 0);
        v1_offset_ptr[i] = offset.channel(i * 4 + 1);
        v2_offset_ptr[i] = offset.channel(i * 4 + 2);
        v3_offset_ptr[i] = offset.channel(i * 4 + 3);
    }

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

                // compute coord
                {
                    // x
                    gx = unormalize(_mm256_set1_ps(src.w), gx);
                    // y
                    gy = unormalize(_mm256_set1_ps(src.h), gy);
                }

                __m256 gx_floor = _mm256_floor_ps(gx);
                __m256 gy_floor = _mm256_floor_ps(gy);

                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

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

                for (int i = 0; i < 4; i++)
                {
                    gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                    gy = get_coord(_mm256_set1_ps(src.h), gy);

                    __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS));

                    __m256 gy_offset = _mm256_mul_ps(gy, _mm256_set1_ps(src.w));

                    volatile float epack = src.elempack;
                    __m256 v0_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx0), _mm256_set1_ps(epack));
                    __m256 v1_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx1), _mm256_set1_ps(epack));
                    __m256 v2_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx2), _mm256_set1_ps(epack));
                    __m256 v3_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx3), _mm256_set1_ps(epack));

                    v0_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v0_offset_f, _mm256_and_ps(x0_in_range, y_in_range));
                    v1_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v1_offset_f, _mm256_and_ps(x1_in_range, y_in_range));
                    v2_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v2_offset_f, _mm256_and_ps(x2_in_range, y_in_range));
                    v3_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v3_offset_f, _mm256_and_ps(x3_in_range, y_in_range));

                    _mm256_storeu_ps(v0_offset_ptr[i], v0_offset_f);
                    _mm256_storeu_ps(v1_offset_ptr[i], v1_offset_f);
                    _mm256_storeu_ps(v2_offset_ptr[i], v2_offset_f);
                    _mm256_storeu_ps(v3_offset_ptr[i], v3_offset_f);

                    v0_offset_ptr[i] += 8;
                    v1_offset_ptr[i] += 8;
                    v2_offset_ptr[i] += 8;
                    v3_offset_ptr[i] += 8;
                }

                _mm256_storeu_ps(value_x, tx);
                _mm256_storeu_ps(value_y, ty);

                value_x += 8;
                value_y += 8;

                gridptr += 16;
            }

#endif // __AVX__

            for (; x < grid_size; x += 2)
            {
                float sample_x = *gridptr;
                float sample_y = *(gridptr + 1);

                // x
                sample_x = unormalize(src.w, sample_x);
                // y
                sample_y = unormalize(src.h, sample_y);

                int x1 = floor(sample_x);
                int y1 = floor(sample_y);
                int x0 = x1 - 1;
                int x2 = x1 + 1;
                int x3 = x1 + 2;

                *value_x = sample_x - static_cast<float>(x1);
                *value_y = sample_y - static_cast<float>(y1);

                x1 = get_coord(src.w, x1);
                x0 = get_coord(src.w, x0);
                x2 = get_coord(src.w, x2);
                x3 = get_coord(src.w, x3);

                bool x1_in_range = (x1 > -1) & (x1 < src.w);
                bool x0_in_range = (x0 > -1) & (x0 < src.w);
                bool x2_in_range = (x2 > -1) & (x2 < src.w);
                bool x3_in_range = (x3 > -1) & (x3 < src.w);

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

                    *v0_offset_ptr[i] = v0_in_bound ? (offset_y + x0) * src.elempack : -1.0f;
                    *v1_offset_ptr[i] = v1_in_bound ? (offset_y + x1) * src.elempack : -1.0f;
                    *v2_offset_ptr[i] = v2_in_bound ? (offset_y + x2) * src.elempack : -1.0f;
                    *v3_offset_ptr[i] = v3_in_bound ? (offset_y + x3) * src.elempack : -1.0f;

                    v0_offset_ptr[i]++;
                    v1_offset_ptr[i]++;
                    v2_offset_ptr[i]++;
                    v3_offset_ptr[i]++;
                }

                value_x++;
                value_y++;

                gridptr += 2;
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

            // compute coord
            {
                // x
                gx = unormalize(_mm256_set1_ps(src.w), gx);
                // y
                gy = unormalize(_mm256_set1_ps(src.h), gy);
            }

            __m256 gx_floor = _mm256_floor_ps(gx);
            __m256 gy_floor = _mm256_floor_ps(gy);

            const __m256 tx = _mm256_sub_ps(gx, gx_floor);
            const __m256 ty = _mm256_sub_ps(gy, gy_floor);

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

            for (int i = 0; i < 4; i++)
            {
                gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                gy = get_coord(_mm256_set1_ps(src.h), gy);

                __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, _mm256_set1_ps(-1), _CMP_GT_OS), _mm256_cmp_ps(_mm256_set1_ps(src.h), gy, _CMP_GT_OS));

                __m256 gy_offset = _mm256_mul_ps(gy, _mm256_set1_ps(src.w));

                volatile float epack = src.elempack;
                __m256 v0_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx0), _mm256_set1_ps(epack));
                __m256 v1_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx1), _mm256_set1_ps(epack));
                __m256 v2_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx2), _mm256_set1_ps(epack));
                __m256 v3_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx3), _mm256_set1_ps(epack));

                v0_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v0_offset_f, _mm256_and_ps(x0_in_range, y_in_range));
                v1_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v1_offset_f, _mm256_and_ps(x1_in_range, y_in_range));
                v2_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v2_offset_f, _mm256_and_ps(x2_in_range, y_in_range));
                v3_offset_f = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), v3_offset_f, _mm256_and_ps(x3_in_range, y_in_range));

                _mm256_storeu_ps(v0_offset_ptr[i], v0_offset_f);
                _mm256_storeu_ps(v1_offset_ptr[i], v1_offset_f);
                _mm256_storeu_ps(v2_offset_ptr[i], v2_offset_f);
                _mm256_storeu_ps(v3_offset_ptr[i], v3_offset_f);

                v0_offset_ptr[i] += 8;
                v1_offset_ptr[i] += 8;
                v2_offset_ptr[i] += 8;
                v3_offset_ptr[i] += 8;
            }

            _mm256_storeu_ps(value_x, tx);
            _mm256_storeu_ps(value_y, ty);

            value_x += 8;
            value_y += 8;

            gridptr_x += 8;
            gridptr_y += 8;
        }

#endif // __AVX__

        for (; x < grid_size; x++)
        {
            float sample_x = *gridptr_x;
            float sample_y = *gridptr_y;

            // x
            sample_x = unormalize(src.w, sample_x);
            // y
            sample_y = unormalize(src.h, sample_y);

            int x1 = floor(sample_x);
            int y1 = floor(sample_y);
            int x0 = x1 - 1;
            int x2 = x1 + 1;
            int x3 = x1 + 2;

            *value_x = sample_x - static_cast<float>(x1);
            *value_y = sample_y - static_cast<float>(y1);

            x1 = get_coord(src.w, x1);
            x0 = get_coord(src.w, x0);
            x2 = get_coord(src.w, x2);
            x3 = get_coord(src.w, x3);

            bool x1_in_range = (x1 > -1) & (x1 < src.w);
            bool x0_in_range = (x0 > -1) & (x0 < src.w);
            bool x2_in_range = (x2 > -1) & (x2 < src.w);
            bool x3_in_range = (x3 > -1) & (x3 < src.w);

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

                *v0_offset_ptr[i] = v0_in_bound ? (offset_y + x0) * src.elempack : -1.0f;
                *v1_offset_ptr[i] = v1_in_bound ? (offset_y + x1) * src.elempack : -1.0f;
                *v2_offset_ptr[i] = v2_in_bound ? (offset_y + x2) * src.elempack : -1.0f;
                *v3_offset_ptr[i] = v3_in_bound ? (offset_y + x3) * src.elempack : -1.0f;

                v0_offset_ptr[i]++;
                v1_offset_ptr[i]++;
                v2_offset_ptr[i]++;
                v3_offset_ptr[i]++;
            }

            value_x++;
            value_y++;

            gridptr_x++;
            gridptr_y++;
        }
    }
}
