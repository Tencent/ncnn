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
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(src.w);
        const __m256i vImgHi = _mm256_set1_epi32(src.h);
#endif // __AVX2__
#endif // __AVX__

        int *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4]; 

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i * 4 + 0] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i * 4 + 1] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i * 4 + 2] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i * 4 + 3] = offset.channel(i * 4 + 3);
        }

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
                        gx = unormalize(vImgWf, gx);
                        gx = get_coord(vImgWf, gx);

                        // y
                        gy = unormalize(vImgHf, gy);
                        gy = get_coord(vImgHf, gy);
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

                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);
                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

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
                    sample_y = unormalize(src.h, sample_x);

                    int x1 = floor(sample_x);
                    int y1 = floor(sample_y);
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
                    gx = get_coord(vImgWf, gx);

                    // y
                    gy = unormalize(vImgHf, gy);
                    gy = get_coord(vImgHf, gy);
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

                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);
                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

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
                sample_y = unormalize(src.h, sample_x);

                int x1 = floor(sample_x);
                int y1 = floor(sample_y);
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
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(src.w);
        const __m256i vImgHi = _mm256_set1_epi32(src.h);
#endif // __AVX2__
#endif // __AVX__

        int *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4]; 

        float *v0_in_bound_ptr[4], *v1_in_bound_ptr[4], *v2_in_bound_ptr[4], *v3_in_bound_ptr[4];

        for (int i = 0; i < 4; i ++)
        {
            v0_offset_ptr[i * 4 + 0] = offset.channel(i * 4 + 0);
            v0_offset_ptr[i * 4 + 1] = offset.channel(i * 4 + 1);
            v0_offset_ptr[i * 4 + 2] = offset.channel(i * 4 + 2);
            v0_offset_ptr[i * 4 + 3] = offset.channel(i * 4 + 3);

            v0_in_bound_ptr[i * 4 + 0] = in_bound.channel(i * 4 + 0);
            v0_in_bound_ptr[i * 4 + 1] = in_bound.channel(i * 4 + 1);
            v0_in_bound_ptr[i * 4 + 2] = in_bound.channel(i * 4 + 2);
            v0_in_bound_ptr[i * 4 + 3] = in_bound.channel(i * 4 + 3);
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
                        gx = unormalize(vImgWf, gx);
                        // y
                        gy = unormalize(vImgHf, gy);
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
                    sample_y = unormalize(src.h, sample_x);

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
                sample_y = unormalize(src.h, sample_x);

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

                gridptr_x++;
                gridptr_y++;
            }
        }
    }
};


