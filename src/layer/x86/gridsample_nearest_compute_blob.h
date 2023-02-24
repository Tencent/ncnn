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
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#endif // __AVX__

        float* offset_ptr = offset.channel(0);

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
                        gx = unormalize(vImgWf, gx);
                        gx = get_coord(vImgWf, gx);

                        gy = unormalize(vImgHf, gy);
                        gy = get_coord(vImgHf, gy);
                    }

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                    __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(gy, vImgWf, gx), vElempackf);

                    _mm256_storeu_ps(offset_ptr, offset);

                    gridptr += 16;

                    offset_ptr += 8;
                }

                nn = grid_size & 15;
#endif // __AVX__

                for (int x = grid_size - nn; x < grid_size; x += 2)
                {
                    float sample_x = *gridptr;
                    float sample_y = *(gridptr + 1);

                    sample_x = unormalize(src.w, sample_x);
                    sample_x = get_coord(src.w, sample_x);

                    sample_y = unormalize(src.h, sample_y);
                    sample_y = get_coord(src.h, sample_y);

                    int x0 = static_cast<int>(floor(sample_x + 0.5f));
                    int y0 = static_cast<int>(floor(sample_y + 0.5f));

                    *offset_ptr = (x0 + y0 * src.w) * src.elempack;

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
                    gx = unormalize(vImgWf, gx);
                    gx = get_coord(vImgWf, gx);

                    gy = unormalize(vImgHf, gy);
                    gy = get_coord(vImgHf, gy);
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(gy, vImgWf, gx), vElempackf);

                _mm256_storeu_ps(offset_ptr, offset);

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

                sample_x = unormalize(src.w, sample_x);
                sample_x = get_coord(src.w, sample_x);

                sample_y = unormalize(src.h, sample_y);
                sample_y = get_coord(src.h, sample_y);

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                *offset_ptr = (x0 + y0 * src.w) * src.elempack;

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
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#endif // __AVX__

        float* offset_ptr = offset.channel(0);

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
                    __m256 tmp_x = _mm256_loadu_ps(gridptr);
                    __m256 gy = _mm256_loadu_ps(gridptr + 8);

                    __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
                    gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
                    tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

                    gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
                    gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

                    // compute coord
                    {
                        gx = unormalize(vImgWf, gx);
                        gy = unormalize(vImgHf, gy);
                    }

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                    __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                      _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

                    __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(gy, vImgWf, gx), vElempackf);

                    _mm256_storeu_ps(in_bound_ptr, v_in_range);
                    _mm256_storeu_ps(offset_ptr, offset);

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

                    sample_x = unormalize(src.w, sample_x);

                    sample_y = unormalize(src.h, sample_y);

                    int x0 = static_cast<int>(floor(sample_x + 0.5f));
                    int y0 = static_cast<int>(floor(sample_y + 0.5f));

                    *in_bound_ptr = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h)) ? -1.0f : 0.0f;
                    *offset_ptr = (x0 + y0 * src.w) * src.elempack;

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
                    gx = unormalize(vImgWf, gx);
                    gy = unormalize(vImgHf, gy);
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

                __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(gy, vImgWf, gx), vElempackf);

                _mm256_storeu_ps(in_bound_ptr, v_in_range);
                _mm256_storeu_ps(offset_ptr, offset);

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

                sample_x = unormalize(src.w, sample_x);
                sample_y = unormalize(src.h, sample_y);

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                *in_bound_ptr = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h)) ? -1.0f : 0.0f;

                *offset_ptr = (x0 + y0 * src.w) * src.elempack;

                gridptr_x++;
                gridptr_y++;

                offset_ptr++;

                in_bound_ptr++;
            }
        }
    }
};

template<PaddingMode pd, bool align_corner>
struct gridsample_3d_nearest_compute_blob
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vImgDf = _mm256_set1_ps(src.d);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#endif // __AVX__

        float* offset_ptr = offset.channel(0);

        grid_sample_unormalize<align_corner> unormalize;
        compute_coord<pd, align_corner> get_coord;

        if (permute_fusion == 0)
        {
            for (int y = 0; y < grid.c; y++)
            {
                const float* gridptr = grid.channel(y);
                int nn = grid_size;
#if __AVX__
                for (int x = 0; x + 23 < nn; x += 24)
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

                    // compute coord
                    {
                        gx = unormalize(vImgWf, gx);
                        gx = get_coord(vImgWf, gx);

                        gy = unormalize(vImgHf, gy);
                        gy = get_coord(vImgHf, gy);

                        gz = unormalize(vImgDf, gz);
                        gz = get_coord(vImgDf, gz);
                    }

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                    gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                    __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), gz,
                                                  _mm256_comp_fmadd_ps(gy, vImgWf, gx)),
                                                  vElempackf);

                    _mm256_storeu_ps(offset_ptr, offset);

                    gridptr += 24;

                    offset_ptr += 8;
                }

                nn = grid_size % 24;
#endif // __AVX__

                for (int x = grid_size - nn; x < grid_size; x += 3)
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

                    int x0 = static_cast<int>(floor(sample_x + 0.5f));
                    int y0 = static_cast<int>(floor(sample_y + 0.5f));
                    int z0 = static_cast<int>(floor(sample_z + 0.5f));

                    *offset_ptr = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;

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

            int nn = grid_size;
#if __AVX__
            for (int x = 0; x + 7 < nn; x += 8)
            {
                __m256 gx = _mm256_loadu_ps(gridptr_x);
                __m256 gy = _mm256_loadu_ps(gridptr_y);
                __m256 gz = _mm256_loadu_ps(gridptr_z);

                // compute coord
                {
                    gx = unormalize(vImgWf, gx);
                    gx = get_coord(vImgWf, gx);

                    gy = unormalize(vImgHf, gy);
                    gy = get_coord(vImgHf, gy);

                    gz = unormalize(vImgDf, gz);
                    gz = get_coord(vImgDf, gz);
                }

                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), gz,
                                              _mm256_comp_fmadd_ps(gy, vImgWf, gx)),
                                              vElempackf);

                _mm256_storeu_ps(offset_ptr, offset);

                gridptr_x += 8;
                gridptr_y += 8;
                gridptr_z += 8;

                offset_ptr += 8;
            }

            nn = grid_size & 7;
#endif // __AVX__

            for (int x = grid_size - nn; x < grid_size; x++)
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

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));
                int z0 = static_cast<int>(floor(sample_z + 0.5f));

                *offset_ptr = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;

                gridptr_x++;
                gridptr_y++;
                gridptr_z++;

                offset_ptr++;
            }
        }
    }
};

template<bool align_corner>
struct gridsample_3d_nearest_compute_blob<PaddingMode::Zeros, align_corner>
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vImgDf = _mm256_set1_ps(src.d);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#endif // __AVX__

        float* offset_ptr = offset.channel(0);

        float* in_bound_ptr = in_bound.channel(0);

        grid_sample_unormalize<align_corner> unormalize;

        if (permute_fusion == 0)
        {
            for (int y = 0; y < grid.c; y++)
            {
                const float* gridptr = grid.channel(y);
                int nn = grid_size;
#if __AVX__
                for (int x = 0; x + 23 < nn; x += 24)
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

                    // compute coord
                    {
                        gx = unormalize(vImgWf, gx);
                        gy = unormalize(vImgHf, gy);
                        gz = unormalize(vImgDf, gz);
                    }

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                    gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                    __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                      _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));
                    v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, gz, _CMP_GT_OS)));

                    __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), gz,
                                                  _mm256_comp_fmadd_ps(gy, vImgWf, gx)),
                                                  vElempackf);

                    _mm256_storeu_ps(in_bound_ptr, v_in_range);
                    _mm256_storeu_ps(offset_ptr, offset);

                    gridptr += 24;
                    offset_ptr += 8;
                    in_bound_ptr += 8;
                }

                nn = grid_size % 24;
#endif // __AVX__

                for (int x = grid_size - nn; x < grid_size; x += 3)
                {
                    float sample_x = *gridptr;
                    float sample_y = *(gridptr + 1);
                    float sample_z = *(gridptr + 2);

                    sample_x = unormalize(src.w, sample_x);
                    sample_y = unormalize(src.h, sample_y);
                    sample_z = unormalize(src.d, sample_z);

                    int x0 = static_cast<int>(floor(sample_x + 0.5f));
                    int y0 = static_cast<int>(floor(sample_y + 0.5f));
                    int z0 = static_cast<int>(floor(sample_z + 0.5f));

                    *in_bound_ptr = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h) & (z0 > -1) & (z0 < src.d)) ? -1.0f : 0.0f;
                    *offset_ptr = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;

                    gridptr += 3;
                    offset_ptr++;
                    in_bound_ptr++;
                }
            }
        }
        else
        {
            const float* gridptr_x = grid.channel(0);
            const float* gridptr_y = grid.channel(1);
            const float* gridptr_z = grid.channel(2);

            int nn = grid_size;
#if __AVX__
            for (int x = 0; x + 7 < nn; x += 8)
            {
                __m256 gx = _mm256_loadu_ps(gridptr_x);
                __m256 gy = _mm256_loadu_ps(gridptr_y);
                __m256 gz = _mm256_loadu_ps(gridptr_z);

                // compute coord=
                {
                    gx = unormalize(vImgWf, gx);
                    gy = unormalize(vImgHf, gy);
                    gz = unormalize(vImgDf, gz);

                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                    gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));
                }

                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));
                v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, gz, _CMP_GT_OS)));

                __m256 offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), gz,
                                              _mm256_comp_fmadd_ps(gy, vImgWf, gx)),
                                              vElempackf);

                _mm256_storeu_ps(in_bound_ptr, v_in_range);
                _mm256_storeu_ps(offset_ptr, offset);

                gridptr_x += 8;
                gridptr_y += 8;
                gridptr_z += 8;

                offset_ptr += 8;
                in_bound_ptr += 8;
            }

            nn = grid_size & 7;
#endif // __AVX__

            for (int x = grid_size - nn; x < grid_size; x++)
            {
                float sample_x = *gridptr_x;
                float sample_y = *gridptr_y;
                float sample_z = *gridptr_z;

                sample_x = unormalize(src.w, sample_x);
                sample_y = unormalize(src.h, sample_y);
                sample_z = unormalize(src.d, sample_z);

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));
                int z0 = static_cast<int>(floor(sample_z + 0.5f));

                *in_bound_ptr = ((x0 > -1) & (x0 < src.w) & (y0 > -1) & (y0 < src.h) & (z0 > -1) & (z0 < src.d)) ? -1.0f : 0.0f;

                *offset_ptr = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;

                gridptr_x++;
                gridptr_y++;
                gridptr_z++;

                offset_ptr++;

                in_bound_ptr++;
            }
        }
    }
};

#if __SSE2__
#if __AVX__
#if __AVX512F__
static void gridsample_nearest_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset.channel(0);

        const float* in_bound_ptr = in_bound.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(in_bound_ptr) < 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)), srcptr, sizeof(float));

            _mm512_storeu_ps(dstptr, _v);

            offset_ptr++;
            in_bound_ptr++;
            dstptr += 16;
        }
    }
}
#endif // __AVX512F__
static void gridsample_nearest_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset.channel(0);

        const float* in_bound_ptr = in_bound.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
#if __AVX2__
            __m256i _offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
            __m256i _offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__
            __m256 _v = mask_gather_ps256(srcptr, _offset, _mm256_set1_ps(*in_bound_ptr));

            _mm256_storeu_ps(dstptr, _v);

            offset_ptr++;
            in_bound_ptr++;
            dstptr += 8;
        }
    }
}
#endif // __AVX__
static void gridsample_nearest_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset.channel(0);

        const float* in_bound_ptr = in_bound.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m128 _v = mask_gather_ps(srcptr, _mm_add_epi32(_mm_set1_epi32(*offset_ptr), _mm_set_epi32(3, 2, 1, 0)), _mm_set1_ps(*in_bound_ptr));

            _mm_storeu_ps(dstptr, _v);

            offset_ptr++;
            in_bound_ptr++;
            dstptr += 4;
        }
    }
}

#endif // __SSE2__

static void gridsample_nearest_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset.channel(0);

        const float* in_bound_ptr = in_bound.channel(0);

        int nn = grid_size;
#if __SSE2__
#if __AVX__
        for (int i = 0; i + 7 < grid_size; i += 8)
        {
            __m256 _v = mask_gather_ps256(srcptr, _mm256_set_epi32(*(offset_ptr + 7), *(offset_ptr + 6), *(offset_ptr + 5), *(offset_ptr + 4), *(offset_ptr + 3), *(offset_ptr + 2), *(offset_ptr + 1), *offset_ptr), _mm256_loadu_ps(in_bound_ptr));

            _mm256_storeu_ps(dstptr, _v);

            offset_ptr += 8;
            in_bound_ptr += 8;
            dstptr += 8;
        }
        nn = grid_size & 7;
#endif // __AVX__
        for (int i = grid_size - nn; i + 3 < grid_size; i += 4)
        {
            __m128 _v = mask_gather_ps(srcptr, _mm_set_epi32(*(offset_ptr + 3), *(offset_ptr + 2), *(offset_ptr + 1), *offset_ptr), _mm_loadu_ps(in_bound_ptr));

            _mm_storeu_ps(dstptr, _v);

            offset_ptr += 4;
            in_bound_ptr += 4;
            dstptr += 4;
        }
        nn = grid_size & 3;
#endif // __SSE2__
        for (int i = grid_size - nn; i < grid_size; i++)
        {
            *dstptr = *reinterpret_cast<const int*>(in_bound_ptr) < 0 ? *(srcptr + static_cast<int>(*offset_ptr)) : 0;

            in_bound_ptr++;
            offset_ptr++;
            dstptr++;
        }
    }
}
