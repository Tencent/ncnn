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
struct gridsample_2d_nearest_compute_blob
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

        int* offset_ptr = offset.channel(0);

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

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                    __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                    __m256i i_offset = _mm256_cvtps_epi32(offset);

                    _mm256_storeu_epi32(offset_ptr, i_offset);

                    gridptr += 16;

                    offset_ptr += 8;
                }

                nn = grid_size & 15;
#endif // __AVX__

                for (int x = grid_size - nn; x < grid_size; x += 2)
                {
                    float sample_x = *gridptr;
                    float sample_y = *(gridptr + 1);

                    // x
                    sample_x = unormalize(src.w, sample_x);
                    sample_x = get_coord(src.w, sample_x);

                    // y
                    sample_y = unormalize(src.h, sample_x);
                    sample_y = get_coord(src.h, sample_x);

                    int x0 = static_cast<int>(floor(sample_x + 0.5f));
                    int y0 = static_cast<int>(floor(sample_y + 0.5f));

                    *offset_ptr = x0 + y0 * src.w;

                    gridptr += 2;

                    offset_ptr++;
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

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                __m256i i_offset = _mm256_cvtps_epi32(offset);

                _mm256_storeu_epi32(offset_ptr, i_offset);

                gridptr_x += 8;
                gridptr_y += 8;

                offset_ptr += 8;
            }

            nn = grid_size & 7;
#endif // __AVX__

            for (int x = grid_size - nn; x < grid_size; x++)
            {
                float sample_x = *gridptr_x;
                float sample_y = *gridptr_y;

                // x
                sample_x = unormalize(src.w, sample_x);
                sample_x = get_coord(src.w, sample_x);

                // y
                sample_y = unormalize(src.h, sample_x);
                sample_y = get_coord(src.h, sample_x);

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                *offset_ptr = x0 + y0 * src.w;

                gridptr_x++;
                gridptr_y++;

                offset_ptr++;
            }
        }
    }
};

template<bool align_corner>
struct gridsample_2d_nearest_compute_blob<PaddingMode::Zeros, align_corner>
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

        int* offset_ptr = offset.channel(0);

        float* in_bound_ptr = in_bound.channel(0);

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

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                    __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                      _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

                    __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                    __m256i i_offset = _mm256_cvtps_epi32(offset);

                    _mm256_storeu_ps(in_bound_ptr, v_in_range);
                    _mm256_storeu_epi32(offset_ptr, i_offset);

                    gridptr += 16;
                    offset_ptr += 8;
                    in_bound_ptr += 8;
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

                    int x0 = static_cast<int>(floor(sample_x + 0.5f));
                    int y0 = static_cast<int>(floor(sample_y + 0.5f));

                    *in_bound_ptr = (x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h);
                    *offset_ptr = x0 + y0 * src.w;

                    gridptr += 2;
                    offset_ptr++;
                    in_bound_ptr++;
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

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                __m256i i_offset = _mm256_cvtps_epi32(offset);

                _mm256_storeu_ps(in_bound_ptr, v_in_range);
                _mm256_storeu_epi32(offset_ptr, i_offset);

                gridptr_x += 8;
                gridptr_y += 8;
                offset_ptr += 8;
                in_bound_ptr += 8;
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

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                *in_bound_ptr = (x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h);

                *offset_ptr = x0 + y0 * src.w;

                gridptr_x++;
                gridptr_y++;

                offset_ptr++;

                in_bound_ptr++;
            }
        }
    }
};
