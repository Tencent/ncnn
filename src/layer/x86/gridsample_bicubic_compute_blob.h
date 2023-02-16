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

template<PaddingMode pd, bool align_corner>
struct gridsample_2d_bicubic_compute_blob
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#endif // __AVX__

        int *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];
        float *v0_in_bound_ptr[4], *v1_in_bound_ptr[4], *v2_in_bound_ptr[4], *v3_in_bound_ptr[4];
        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);

            v0_in_bound_ptr[i] = in_bound.channel(i * 4 + 0);
            v1_in_bound_ptr[i] = in_bound.channel(i * 4 + 1);
            v2_in_bound_ptr[i] = in_bound.channel(i * 4 + 2);
            v3_in_bound_ptr[i] = in_bound.channel(i * 4 + 3);
        }

        float* value_x = value.channel(0);
        float* value_y = value.channel(1);

        grid_sample_unormalize<align_corner> unormalize;
        compute_coord<pd, align_corner> get_coord;

        if (permute_fusion == 0)
        {
            for (int y = 0; y < grid.c; y++)
            {
                const float* gridptr = grid.channel(y);
                int nn = grid_size;
#if __AVX__
                for (int x = 0; x + 15 < nn; x += 16)
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
                        gx = unormalize(vImgWf, gx);

                        // y
                        gy = unormalize(vImgHf, gy);
                    }

                    __m256 gx_floor = _mm256_floor_ps(gx);
                    __m256 gy_floor = _mm256_floor_ps(gy);

                    const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                    const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                    __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                    __m256 gx1 = gx_floor;
                    __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                    __m256 gx3 = _mm256_add_ps(gx2, *(__m256*)_ps256_1);

                    gx0 = get_coord(vImgWf, gx0);
                    gx1 = get_coord(vImgWf, gx1);
                    gx2 = get_coord(vImgWf, gx2);
                    gx3 = get_coord(vImgWf, gx3);

                    for (int i = 0; i < 4; i++)
                    {
                        gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                        gy = get_coord(vImgHf, gy);

                        __m256 gy_offset = _mm256_mul_ps(gy, vImgWf);

                        __m256 v0_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx0), vElempackf);
                        __m256 v1_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx1), vElempackf);
                        __m256 v2_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx2), vElempackf);
                        __m256 v3_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx3), vElempackf);

                        _mm256_storeu_epi32(v0_offset_ptr[i], _mm256_cvtps_epi32(v0_offset_f));
                        _mm256_storeu_epi32(v1_offset_ptr[i], _mm256_cvtps_epi32(v1_offset_f));
                        _mm256_storeu_epi32(v2_offset_ptr[i], _mm256_cvtps_epi32(v2_offset_f));
                        _mm256_storeu_epi32(v3_offset_ptr[i], _mm256_cvtps_epi32(v3_offset_f));

                        _mm256_storeu_ps(v0_in_bound_ptr[i], *(__m256*)_ps256_n1);
                        _mm256_storeu_ps(v1_in_bound_ptr[i], *(__m256*)_ps256_n1);
                        _mm256_storeu_ps(v2_in_bound_ptr[i], *(__m256*)_ps256_n1);
                        _mm256_storeu_ps(v3_in_bound_ptr[i], *(__m256*)_ps256_n1);

                        v0_in_bound_ptr[i] += 8;
                        v1_in_bound_ptr[i] += 8;
                        v2_in_bound_ptr[i] += 8;
                        v3_in_bound_ptr[i] += 8;

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

                nn = grid_size & 15;
#endif // __AVX__

                for (int x = grid_size - nn; x < grid_size; x += 2)
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

                    for (int i = 0; i < 4; i++)
                    {
                        int offset_y = get_coord(src.h, y1 + i - 1) * src.w;

                        *v0_offset_ptr[i] = (offset_y + x0) * src.elempack;
                        *v1_offset_ptr[i] = (offset_y + x1) * src.elempack;
                        *v2_offset_ptr[i] = (offset_y + x2) * src.elempack;
                        *v3_offset_ptr[i] = (offset_y + x3) * src.elempack;

                        *v0_in_bound_ptr[i]++ = -1.0f;
                        *v1_in_bound_ptr[i]++ = -1.0f;
                        *v2_in_bound_ptr[i]++ = -1.0f;
                        *v3_in_bound_ptr[i]++ = -1.0f;

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

            int nn = grid_size;
#if __AVX__
            for (int x = 0; x + 7 < nn; x += 8)
            {
                __m256 gx = _mm256_loadu_ps(gridptr_x);
                __m256 gy = _mm256_loadu_ps(gridptr_y);

                // compute coord
                {
                    // x
                    gx = unormalize(vImgWf, gx);

                    // y
                    gy = unormalize(vImgHf, gy);
                }

                __m256 gx_floor = _mm256_floor_ps(gx);
                __m256 gy_floor = _mm256_floor_ps(gy);

                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                __m256 gx1 = gx_floor;
                __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                __m256 gx3 = _mm256_add_ps(gx2, *(__m256*)_ps256_1);

                gx0 = get_coord(vImgWf, gx0);
                gx1 = get_coord(vImgWf, gx1);
                gx2 = get_coord(vImgWf, gx2);
                gx3 = get_coord(vImgWf, gx3);

                for (int i = 0; i < 4; i++)
                {
                    gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));
                    gy = get_coord(vImgHf, gy);

                    __m256 gy_offset = _mm256_mul_ps(gy, vImgWf);

                    __m256 v0_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx0), vElempackf);
                    __m256 v1_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx1), vElempackf);
                    __m256 v2_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx2), vElempackf);
                    __m256 v3_offset_f = _mm256_mul_ps(_mm256_add_ps(gy_offset, gx3), vElempackf);

                    _mm256_storeu_epi32(v0_offset_ptr[i], _mm256_cvtps_epi32(v0_offset_f));
                    _mm256_storeu_epi32(v1_offset_ptr[i], _mm256_cvtps_epi32(v1_offset_f));
                    _mm256_storeu_epi32(v2_offset_ptr[i], _mm256_cvtps_epi32(v2_offset_f));
                    _mm256_storeu_epi32(v3_offset_ptr[i], _mm256_cvtps_epi32(v3_offset_f));

                    _mm256_storeu_ps(v0_in_bound_ptr[i], *(__m256*)_ps256_n1);
                    _mm256_storeu_ps(v1_in_bound_ptr[i], *(__m256*)_ps256_n1);
                    _mm256_storeu_ps(v2_in_bound_ptr[i], *(__m256*)_ps256_n1);
                    _mm256_storeu_ps(v3_in_bound_ptr[i], *(__m256*)_ps256_n1);

                    v0_in_bound_ptr[i] += 8;
                    v1_in_bound_ptr[i] += 8;
                    v2_in_bound_ptr[i] += 8;
                    v3_in_bound_ptr[i] += 8;

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

            nn = grid_size & 7;
#endif // __AVX__

            for (int x = grid_size - nn; x < grid_size; x++)
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

                for (int i = 0; i < 4; i++)
                {
                    int offset_y = static_cast<int>(get_coord(src.h, y1 + i - 1)) * src.w;

                    *v0_offset_ptr[i] = (offset_y + x0) * src.elempack;
                    *v1_offset_ptr[i] = (offset_y + x1) * src.elempack;
                    *v2_offset_ptr[i] = (offset_y + x2) * src.elempack;
                    *v3_offset_ptr[i] = (offset_y + x3) * src.elempack;

                    *v0_in_bound_ptr[i]++ = -1.0f;
                    *v1_in_bound_ptr[i]++ = -1.0f;
                    *v2_in_bound_ptr[i]++ = -1.0f;
                    *v3_in_bound_ptr[i]++ = -1.0f;

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
};

template<bool align_corner>
struct gridsample_2d_bicubic_compute_blob<PaddingMode::Zeros, align_corner>
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#endif // __AVX__

        int *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

        float *v0_in_bound_ptr[4], *v1_in_bound_ptr[4], *v2_in_bound_ptr[4], *v3_in_bound_ptr[4];

        float* value_x = value.channel(0);
        float* value_y = value.channel(1);

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);

            v0_in_bound_ptr[i] = in_bound.channel(i * 4 + 0);
            v1_in_bound_ptr[i] = in_bound.channel(i * 4 + 1);
            v2_in_bound_ptr[i] = in_bound.channel(i * 4 + 2);
            v3_in_bound_ptr[i] = in_bound.channel(i * 4 + 3);
        }

        grid_sample_unormalize<align_corner> unormalize;

        if (permute_fusion == 0)
        {
            for (int y = 0; y < grid.c; y++)
            {
                const float* gridptr = grid.channel(y);
                int nn = grid_size;
#if __AVX__
                for (int x = 0; x + 15 < nn; x += 16)
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
                        gx = unormalize(vImgWf, gx);
                        // y
                        gy = unormalize(vImgHf, gy);
                    }

                    __m256 gx_floor = _mm256_floor_ps(gx);
                    __m256 gy_floor = _mm256_floor_ps(gy);

                    const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                    const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                    __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                    __m256 gx1 = gx_floor;
                    __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                    __m256 gx3 = _mm256_add_ps(gx2, *(__m256*)_ps256_1);

                    __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(gx0, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx0, _CMP_GT_OS));
                    __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(gx1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx1, _CMP_GT_OS));
                    __m256 x2_in_range = _mm256_and_ps(_mm256_cmp_ps(gx2, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx2, _CMP_GT_OS));
                    __m256 x3_in_range = _mm256_and_ps(_mm256_cmp_ps(gx3, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx3, _CMP_GT_OS));

                    for (int i = 0; i < 4; i++)
                    {
                        gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                        __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS));

                        _mm256_storeu_ps(v0_in_bound_ptr[i], _mm256_and_ps(x0_in_range, y_in_range));
                        _mm256_storeu_ps(v1_in_bound_ptr[i], _mm256_and_ps(x1_in_range, y_in_range));
                        _mm256_storeu_ps(v2_in_bound_ptr[i], _mm256_and_ps(x2_in_range, y_in_range));
                        _mm256_storeu_ps(v3_in_bound_ptr[i], _mm256_and_ps(x3_in_range, y_in_range));

                        __m256 v0_offset_f = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf);
                        __m256 v1_offset_f = _mm256_add_ps(v0_offset_f, vElempackf);
                        __m256 v2_offset_f = _mm256_add_ps(v1_offset_f, vElempackf);
                        __m256 v3_offset_f = _mm256_add_ps(v2_offset_f, vElempackf);

                        _mm256_storeu_epi32(v0_offset_ptr[i], _mm256_cvtps_epi32(v0_offset_f));
                        _mm256_storeu_epi32(v1_offset_ptr[i], _mm256_cvtps_epi32(v1_offset_f));
                        _mm256_storeu_epi32(v2_offset_ptr[i], _mm256_cvtps_epi32(v2_offset_f));
                        _mm256_storeu_epi32(v3_offset_ptr[i], _mm256_cvtps_epi32(v3_offset_f));

                        v0_offset_ptr[i] += 8;
                        v1_offset_ptr[i] += 8;
                        v2_offset_ptr[i] += 8;
                        v3_offset_ptr[i] += 8;

                        v0_in_bound_ptr[i] += 8;
                        v1_in_bound_ptr[i] += 8;
                        v2_in_bound_ptr[i] += 8;
                        v3_in_bound_ptr[i] += 8;
                    }

                    _mm256_storeu_ps(value_x, tx);
                    _mm256_storeu_ps(value_y, ty);

                    value_x += 8;
                    value_y += 8;

                    gridptr += 16;
                }

                nn = grid_size & 15;
#endif // __AVX__

                for (int x = grid_size - nn; x < grid_size; x += 2)
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

                    bool x1_in_range = (x1 > -1) & (x1 < src.w);
                    bool x0_in_range = (x0 > -1) & (x0 < src.w);
                    bool x2_in_range = (x2 > -1) & (x2 < src.w);
                    bool x3_in_range = (x3 > -1) & (x3 < src.w);

                    for (int i = 0; i < 4; i++)
                    {
                        int gy = y1 + i - 1;
                        int offset_y = gy * src.w;

                        bool y_in_range = (gy > -1) & (gy < src.h);

                        *v0_in_bound_ptr[i] = (x0_in_range & y_in_range) ? -1.0f : 0.0f;
                        *v1_in_bound_ptr[i] = (x1_in_range & y_in_range) ? -1.0f : 0.0f;
                        *v2_in_bound_ptr[i] = (x2_in_range & y_in_range) ? -1.0f : 0.0f;
                        *v3_in_bound_ptr[i] = (x3_in_range & y_in_range) ? -1.0f : 0.0f;

                        *v0_offset_ptr[i] = (offset_y + x0) * src.elempack;
                        *v1_offset_ptr[i] = (offset_y + x1) * src.elempack;
                        *v2_offset_ptr[i] = (offset_y + x2) * src.elempack;
                        *v3_offset_ptr[i] = (offset_y + x3) * src.elempack;

                        v0_offset_ptr[i]++;
                        v1_offset_ptr[i]++;
                        v2_offset_ptr[i]++;
                        v3_offset_ptr[i]++;

                        v0_in_bound_ptr[i]++;
                        v1_in_bound_ptr[i]++;
                        v2_in_bound_ptr[i]++;
                        v3_in_bound_ptr[i]++;
                    }

                    *value_x = sample_x - static_cast<float>(x1);
                    *value_y = sample_y - static_cast<float>(y1);

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

            int nn = grid_size;
#if __AVX__
            for (int x = 0; x + 7 < nn; x += 8)
            {
                __m256 gx = _mm256_loadu_ps(gridptr_x);
                __m256 gy = _mm256_loadu_ps(gridptr_y);

                // compute coord
                {
                    // x
                    gx = unormalize(vImgWf, gx);
                    // y
                    gy = unormalize(vImgHf, gy);
                }

                __m256 gx_floor = _mm256_floor_ps(gx);
                __m256 gy_floor = _mm256_floor_ps(gy);

                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                __m256 gx1 = gx_floor;
                __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                __m256 gx3 = _mm256_add_ps(gx2, *(__m256*)_ps256_1);

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(gx0, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx0, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(gx1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx1, _CMP_GT_OS));
                __m256 x2_in_range = _mm256_and_ps(_mm256_cmp_ps(gx2, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx2, _CMP_GT_OS));
                __m256 x3_in_range = _mm256_and_ps(_mm256_cmp_ps(gx3, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx3, _CMP_GT_OS));

                for (int i = 0; i < 4; i++)
                {
                    gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                    __m256 y_in_range = _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS));

                    _mm256_storeu_ps(v0_in_bound_ptr[i], _mm256_and_ps(x0_in_range, y_in_range));
                    _mm256_storeu_ps(v1_in_bound_ptr[i], _mm256_and_ps(x1_in_range, y_in_range));
                    _mm256_storeu_ps(v2_in_bound_ptr[i], _mm256_and_ps(x2_in_range, y_in_range));
                    _mm256_storeu_ps(v3_in_bound_ptr[i], _mm256_and_ps(x3_in_range, y_in_range));

                    __m256 v0_offset_f = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf);
                    __m256 v1_offset_f = _mm256_add_ps(v0_offset_f, vElempackf);
                    __m256 v2_offset_f = _mm256_add_ps(v1_offset_f, vElempackf);
                    __m256 v3_offset_f = _mm256_add_ps(v2_offset_f, vElempackf);

                    _mm256_storeu_epi32(v0_offset_ptr[i], _mm256_cvtps_epi32(v0_offset_f));
                    _mm256_storeu_epi32(v1_offset_ptr[i], _mm256_cvtps_epi32(v1_offset_f));
                    _mm256_storeu_epi32(v2_offset_ptr[i], _mm256_cvtps_epi32(v2_offset_f));
                    _mm256_storeu_epi32(v3_offset_ptr[i], _mm256_cvtps_epi32(v3_offset_f));

                    v0_offset_ptr[i] += 8;
                    v1_offset_ptr[i] += 8;
                    v2_offset_ptr[i] += 8;
                    v3_offset_ptr[i] += 8;

                    v0_in_bound_ptr[i] += 8;
                    v1_in_bound_ptr[i] += 8;
                    v2_in_bound_ptr[i] += 8;
                    v3_in_bound_ptr[i] += 8;
                }

                _mm256_storeu_ps(value_x, tx);
                _mm256_storeu_ps(value_y, ty);

                value_x += 8;
                value_y += 8;

                gridptr_x += 8;
                gridptr_y += 8;
            }

            nn = grid_size & 7;
#endif // __AVX__

            for (int x = grid_size - nn; x < grid_size; x++)
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

                bool x1_in_range = (x1 > -1) & (x1 < src.w);
                bool x0_in_range = (x0 > -1) & (x0 < src.w);
                bool x2_in_range = (x2 > -1) & (x2 < src.w);
                bool x3_in_range = (x3 > -1) & (x3 < src.w);

                for (int i = 0; i < 4; i++)
                {
                    int gy = y1 + i - 1;
                    int offset_y = gy * src.w;

                    bool y_in_range = (gy > -1) & (gy < src.h);

                    *v0_in_bound_ptr[i] = (x0_in_range & y_in_range) ? -1.0f : 0.0f;
                    *v1_in_bound_ptr[i] = (x1_in_range & y_in_range) ? -1.0f : 0.0f;
                    *v2_in_bound_ptr[i] = (x2_in_range & y_in_range) ? -1.0f : 0.0f;
                    *v3_in_bound_ptr[i] = (x3_in_range & y_in_range) ? -1.0f : 0.0f;

                    *v0_offset_ptr[i] = (offset_y + x0) * src.elempack;
                    *v1_offset_ptr[i] = (offset_y + x1) * src.elempack;
                    *v2_offset_ptr[i] = (offset_y + x2) * src.elempack;
                    *v3_offset_ptr[i] = (offset_y + x3) * src.elempack;

                    v0_offset_ptr[i]++;
                    v1_offset_ptr[i]++;
                    v2_offset_ptr[i]++;
                    v3_offset_ptr[i]++;

                    v0_in_bound_ptr[i]++;
                    v1_in_bound_ptr[i]++;
                    v2_in_bound_ptr[i]++;
                    v3_in_bound_ptr[i]++;
                }

                *value_x = sample_x - static_cast<float>(x1);
                *value_y = sample_y - static_cast<float>(y1);

                value_x++;
                value_y++;

                gridptr_x++;
                gridptr_y++;
            }
        }
    }
};

#if __AVX__
static void cubic_interp1d_p8(__m256& coeffs0, __m256& coeffs1, __m256& coeffs2, __m256& coeffs3, const __m256& tx)
{
    const __m256 A = _mm256_set1_ps(-0.75f);

    const __m256 x0 = _mm256_add_ps(tx, *(__m256*)_ps256_1);
    const __m256& x1 = tx;
    const __m256 x2 = _mm256_sub_ps(*(__m256*)_ps256_1, tx);
    //const __m256 x3 = _mm256_add_ps(x2, *(__m256*)_ps256_1);

    coeffs0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(A, x0), _mm256_mul_ps(_mm256_set1_ps(5.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(8.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(4), A));
    coeffs1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x1), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x1), x1), *(__m256*)_ps256_1);
    coeffs2 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x2), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x2), x2), *(__m256*)_ps256_1);
    coeffs3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(*(__m256*)_ps256_1, coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p8(const Mat& src, Mat& dst, Mat& offset, Mat& in_bound, const Mat& value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    __m256 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
    __m256 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
    __m256 value_f[4];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        int *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

        float *v0_in_bound_ptr[4], *v1_in_bound_ptr[4], *v2_in_bound_ptr[4], *v3_in_bound_ptr[4];

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);

            v0_in_bound_ptr[i] = in_bound.channel(i * 4 + 0);
            v1_in_bound_ptr[i] = in_bound.channel(i * 4 + 1);
            v2_in_bound_ptr[i] = in_bound.channel(i * 4 + 2);
            v3_in_bound_ptr[i] = in_bound.channel(i * 4 + 3);
        }

        const float* value_x = value.channel(0);
        const float* value_y = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            cubic_interp1d_p8(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm256_set1_ps(*value_x));
            for (int ii = 0; ii < 4; ii++)
            {
                __m256 x0_val = mask_gather_ps256(srcptr, _mm256_add_epi32(_mm256_set1_epi32(*v0_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)), _mm256_set1_ps(*v0_in_bound_ptr[ii]));
                __m256 x1_val = mask_gather_ps256(srcptr, _mm256_add_epi32(_mm256_set1_epi32(*v1_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)), _mm256_set1_ps(*v1_in_bound_ptr[ii]));
                __m256 x2_val = mask_gather_ps256(srcptr, _mm256_add_epi32(_mm256_set1_epi32(*v2_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)), _mm256_set1_ps(*v2_in_bound_ptr[ii]));
                __m256 x3_val = mask_gather_ps256(srcptr, _mm256_add_epi32(_mm256_set1_epi32(*v3_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)), _mm256_set1_ps(*v3_in_bound_ptr[ii]));

                value_f[ii] = _mm256_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                v0_offset_ptr[ii]++;
                v1_offset_ptr[ii]++;
                v2_offset_ptr[ii]++;
                v3_offset_ptr[ii]++;

                v0_in_bound_ptr[ii]++;
                v1_in_bound_ptr[ii]++;
                v2_in_bound_ptr[ii]++;
                v3_in_bound_ptr[ii]++;
            }

            cubic_interp1d_p8(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm256_set1_ps(*value_y));

            __m256 _v = _mm256_mul_ps(y_coeffs0, value_f[0]);
            _v = _mm256_comp_fmadd_ps(y_coeffs1, value_f[1], _v);
            _v = _mm256_comp_fmadd_ps(y_coeffs2, value_f[2], _v);
            _v = _mm256_comp_fmadd_ps(y_coeffs3, value_f[3], _v);
            _mm256_storeu_ps(dstptr, _v);

            value_x++;
            value_y++;

            dstptr += 8;
        }
    }
}

#endif // __AVX__
