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
struct gridsample_2d_bilinear_compute_blob
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(src.w);
        const __m256i vImgHi = _mm256_set1_epi32(src.h);
        const __m256i vElempacki = _mm256_set1_epi32(src.elempack);
#endif // __AVX2__
#endif // __AVX__

        int* offset_ptr_00 = offset.channel(0);
        int* offset_ptr_01 = offset.channel(1);
        int* offset_ptr_10 = offset.channel(2);
        int* offset_ptr_11 = offset.channel(3);

        float* in_bound_ptr_01 = in_bound.channel(1);
        float* in_bound_ptr_10 = in_bound.channel(2);
        float* in_bound_ptr_11 = in_bound.channel(3);

        float* value_ptr_alpha = value.channel(0);
        float* value_ptr_beta = value.channel(1);

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
                        gx = get_coord(vImgWf, gx);

                        // y
                        gy = unormalize(vImgHf, gy);
                        gy = get_coord(vImgHf, gy);
                    }

                    __m256 x_w = _mm256_floor_ps(gx);
                    __m256 y_n = _mm256_floor_ps(gy);

#if __AVX2__
                    __m256i x0 = _mm256_cvtps_epi32(x_w);
                    __m256i y0 = _mm256_cvtps_epi32(y_n);
                    __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                    __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                    __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                    __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                    __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                    __m256i i_nw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0), vElempacki);
                    __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                    __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                    __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                    _mm256_storeu_epi32(in_bound_ptr_01, x1_in_range);
                    _mm256_storeu_epi32(in_bound_ptr_10, y1_in_range);
                    _mm256_storeu_epi32(in_bound_ptr_11, v11_in_range);
#else
                    __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                    __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                    __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                    __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                    __m256 nw_offset = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w), vElempackf);
                    __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                    __m256 sw_offset = _mm256_add_ps(nw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                    __m256 se_offset = _mm256_add_ps(sw_offset, vElempackf);

                    __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                    __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                    __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                    __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);

                    _mm256_storeu_ps(in_bound_ptr_01, x1_in_range);
                    _mm256_storeu_ps(in_bound_ptr_10, y1_in_range);
                    _mm256_storeu_ps(in_bound_ptr_11, v11_in_range);
#endif

                    _mm256_storeu_epi32(offset_ptr_00, i_nw_offset);
                    _mm256_storeu_epi32(offset_ptr_01, i_ne_offset);
                    _mm256_storeu_epi32(offset_ptr_10, i_sw_offset);
                    _mm256_storeu_epi32(offset_ptr_11, i_se_offset);

                    __m256 alpha = _mm256_sub_ps(gx, x_w);
                    __m256 beta = _mm256_sub_ps(gy, y_n);

                    _mm256_storeu_ps(value_ptr_alpha, alpha);
                    _mm256_storeu_ps(value_ptr_beta, beta);

                    gridptr += 16;

                    offset_ptr_00 += 8;
                    offset_ptr_01 += 8;
                    offset_ptr_10 += 8;
                    offset_ptr_11 += 8;

                    in_bound_ptr_01 += 8;
                    in_bound_ptr_10 += 8;
                    in_bound_ptr_11 += 8;

                    value_ptr_alpha += 8;
                    value_ptr_beta += 8;
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
                    sample_y = unormalize(src.h, sample_y);
                    sample_y = get_coord(src.h, sample_y);

                    int x0 = (int)floor(sample_x);
                    int y0 = (int)floor(sample_y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    bool x1_in_bound = (x1 > -1) & (x1 < src.w);
                    bool y1_in_bound = (y1 > -1) & (y1 < src.h);

                    *in_bound_ptr_01 = x1_in_bound ? -1.0f : 0.0f;
                    *in_bound_ptr_10 = y1_in_bound ? -1.0f : 0.0f;
                    *in_bound_ptr_11 = (x1_in_bound & y1_in_bound) ? -1.0f : 0.0f;

                    *offset_ptr_00 = (x0 + y0 * src.w) * src.elempack;
                    *offset_ptr_01 = (x1 + y0 * src.w) * src.elempack;
                    *offset_ptr_10 = (x0 + y1 * src.w) * src.elempack;
                    *offset_ptr_11 = (x1 + y1 * src.w) * src.elempack;

                    *value_ptr_alpha = sample_x - x0;
                    *value_ptr_beta = sample_y - y0;

                    gridptr += 2;

                    offset_ptr_00++;
                    offset_ptr_01++;
                    offset_ptr_10++;
                    offset_ptr_11++;

                    in_bound_ptr_01++;
                    in_bound_ptr_10++;
                    in_bound_ptr_11++;

                    value_ptr_alpha++;
                    value_ptr_beta++;
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

                __m256 x_w = _mm256_floor_ps(gx);
                __m256 y_n = _mm256_floor_ps(gy);

#if __AVX2__
                __m256i x0 = _mm256_cvtps_epi32(x_w);
                __m256i y0 = _mm256_cvtps_epi32(y_n);
                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                __m256i i_nw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0), vElempacki);
                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                _mm256_storeu_epi32(in_bound_ptr_01, x1_in_range);
                _mm256_storeu_epi32(in_bound_ptr_10, y1_in_range);
                _mm256_storeu_epi32(in_bound_ptr_11, v11_in_range);
#else
                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                __m256 nw_offset = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w), vElempackf);
                __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                __m256 sw_offset = _mm256_add_ps(nw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                __m256 se_offset = _mm256_add_ps(sw_offset, vElempackf);

                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);

                _mm256_storeu_ps(in_bound_ptr_01, x1_in_range);
                _mm256_storeu_ps(in_bound_ptr_10, y1_in_range);
                _mm256_storeu_ps(in_bound_ptr_11, v11_in_range);
#endif

                _mm256_storeu_epi32(offset_ptr_00, i_nw_offset);
                _mm256_storeu_epi32(offset_ptr_01, i_ne_offset);
                _mm256_storeu_epi32(offset_ptr_10, i_sw_offset);
                _mm256_storeu_epi32(offset_ptr_11, i_se_offset);

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

                in_bound_ptr_01 += 8;
                in_bound_ptr_10 += 8;
                in_bound_ptr_11 += 8;

                value_ptr_alpha += 8;
                value_ptr_beta += 8;
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
                sample_y = unormalize(src.h, sample_y);
                sample_y = get_coord(src.h, sample_y);

                int x0 = (int)floor(sample_x);
                int y0 = (int)floor(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                bool x1_in_bound = (x1 > -1) & (x1 < src.w);
                bool y1_in_bound = (y1 > -1) & (y1 < src.h);

                *in_bound_ptr_01 = x1_in_bound ? -1.0f : 0.0f;
                *in_bound_ptr_10 = y1_in_bound ? -1.0f : 0.0f;
                *in_bound_ptr_11 = (x1_in_bound & y1_in_bound) ? -1.0f : 0.0f;

                *offset_ptr_00 = (x0 + y0 * src.w) * src.elempack;
                *offset_ptr_01 = (x1 + y0 * src.w) * src.elempack;
                *offset_ptr_10 = (x0 + y1 * src.w) * src.elempack;
                *offset_ptr_11 = (x1 + y1 * src.w) * src.elempack;

                *value_ptr_alpha = sample_x - x0;
                *value_ptr_beta = sample_y - y0;

                gridptr_x++;
                gridptr_y++;

                offset_ptr_00++;
                offset_ptr_01++;
                offset_ptr_10++;
                offset_ptr_11++;

                in_bound_ptr_01++;
                in_bound_ptr_10++;
                in_bound_ptr_11++;

                value_ptr_alpha++;
                value_ptr_beta++;
            }
        }
    }
};

template<bool align_corner>
struct gridsample_2d_bilinear_compute_blob<PaddingMode::Zeros, align_corner>
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(src.w);
        const __m256i vImgHi = _mm256_set1_epi32(src.h);
        const __m256i vElempacki = _mm256_set1_epi32(src.elempack);
#endif // __AVX2__
#endif // __AVX__

        int* offset_ptr_00 = offset.channel(0);
        int* offset_ptr_01 = offset.channel(1);
        int* offset_ptr_10 = offset.channel(2);
        int* offset_ptr_11 = offset.channel(3);

        float* in_bound_ptr_00 = in_bound.channel(0);
        float* in_bound_ptr_01 = in_bound.channel(1);
        float* in_bound_ptr_10 = in_bound.channel(2);
        float* in_bound_ptr_11 = in_bound.channel(3);

        float* value_ptr_alpha = value.channel(0);
        float* value_ptr_beta = value.channel(1);

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

                    __m256 x_w = _mm256_floor_ps(gx);
                    __m256 y_n = _mm256_floor_ps(gy);

#if __AVX2__
                    __m256i x0 = _mm256_cvtps_epi32(x_w);
                    __m256i y0 = _mm256_cvtps_epi32(y_n);
                    __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                    __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                    __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x0));
                    __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                    __m256i y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y0));
                    __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                    __m256i v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                    __m256i v01_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                    __m256i v10_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                    __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                    __m256i i_nw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0), vElempacki);
                    __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                    __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                    __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                    _mm256_storeu_ps(in_bound_ptr_00, _mm256_castsi256_ps(v00_in_range));
                    _mm256_storeu_ps(in_bound_ptr_01, _mm256_castsi256_ps(v01_in_range));
                    _mm256_storeu_ps(in_bound_ptr_10, _mm256_castsi256_ps(v10_in_range));
                    _mm256_storeu_ps(in_bound_ptr_11, _mm256_castsi256_ps(v11_in_range));
#else
                    __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                    __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                    __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                    __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                    __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                    __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                    __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                    __m256 v01_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                    __m256 v10_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                    __m256 nw_offset = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w), vElempackf);
                    __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                    __m256 sw_offset = _mm256_add_ps(nw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                    __m256 se_offset = _mm256_add_ps(sw_offset, vElempackf);

                    __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                    __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                    __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                    __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);

                    _mm256_storeu_ps(in_bound_ptr_00, v00_in_range);
                    _mm256_storeu_ps(in_bound_ptr_01, v01_in_range);
                    _mm256_storeu_ps(in_bound_ptr_10, v10_in_range);
                    _mm256_storeu_ps(in_bound_ptr_11, v11_in_range);
#endif // __AVX2__

                    _mm256_storeu_epi32(offset_ptr_00, i_nw_offset);
                    _mm256_storeu_epi32(offset_ptr_01, i_ne_offset);
                    _mm256_storeu_epi32(offset_ptr_10, i_sw_offset);
                    _mm256_storeu_epi32(offset_ptr_11, i_se_offset);

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

                    in_bound_ptr_00 += 8;
                    in_bound_ptr_01 += 8;
                    in_bound_ptr_10 += 8;
                    in_bound_ptr_11 += 8;

                    value_ptr_alpha += 8;
                    value_ptr_beta += 8;
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

                    int x0 = (int)floor(sample_x);
                    int y0 = (int)floor(sample_y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    bool x0_in_bound = (x0 > -1) & (x0 < src.w);
                    bool x1_in_bound = (x1 > -1) & (x1 < src.w);
                    bool y0_in_bound = (y0 > -1) & (y0 < src.h);
                    bool y1_in_bound = (y1 > -1) & (y1 < src.h);

                    *in_bound_ptr_00 = (x0_in_bound & y0_in_bound) ? -1.0f : 0.0f;
                    *in_bound_ptr_01 = (x1_in_bound & y0_in_bound) ? -1.0f : 0.0f;
                    *in_bound_ptr_10 = (x0_in_bound & y1_in_bound) ? -1.0f : 0.0f;
                    *in_bound_ptr_11 = (x1_in_bound & y1_in_bound) ? -1.0f : 0.0f;

                    *offset_ptr_00 = (x0 + y0 * src.w) * src.elempack;
                    *offset_ptr_01 = (x1 + y0 * src.w) * src.elempack;
                    *offset_ptr_10 = (x0 + y1 * src.w) * src.elempack;
                    *offset_ptr_11 = (x1 + y1 * src.w) * src.elempack;

                    *value_ptr_alpha = sample_x - x0;
                    *value_ptr_beta = sample_y - y0;

                    gridptr += 2;

                    offset_ptr_00++;
                    offset_ptr_01++;
                    offset_ptr_10++;
                    offset_ptr_11++;

                    in_bound_ptr_00++;
                    in_bound_ptr_01++;
                    in_bound_ptr_10++;
                    in_bound_ptr_11++;

                    value_ptr_alpha++;
                    value_ptr_beta++;
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

                __m256 x_w = _mm256_floor_ps(gx);
                __m256 y_n = _mm256_floor_ps(gy);

#if __AVX2__
                __m256i x0 = _mm256_cvtps_epi32(x_w);
                __m256i y0 = _mm256_cvtps_epi32(y_n);
                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x0));
                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                __m256i y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y0));
                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                __m256i v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                __m256i v01_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                __m256i v10_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                __m256i i_nw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0), vElempacki);
                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                _mm256_storeu_ps(in_bound_ptr_00, _mm256_castsi256_ps(v00_in_range));
                _mm256_storeu_ps(in_bound_ptr_01, _mm256_castsi256_ps(v01_in_range));
                _mm256_storeu_ps(in_bound_ptr_10, _mm256_castsi256_ps(v10_in_range));
                _mm256_storeu_ps(in_bound_ptr_11, _mm256_castsi256_ps(v11_in_range));
#else
                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                __m256 v01_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                __m256 v10_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                __m256 nw_offset = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w), vElempackf);
                __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                __m256 sw_offset = _mm256_add_ps(nw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                __m256 se_offset = _mm256_add_ps(sw_offset, vElempackf);

                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);

                _mm256_storeu_ps(in_bound_ptr_00, v00_in_range);
                _mm256_storeu_ps(in_bound_ptr_01, v01_in_range);
                _mm256_storeu_ps(in_bound_ptr_10, v10_in_range);
                _mm256_storeu_ps(in_bound_ptr_11, v11_in_range);
#endif // __AVX2__

                _mm256_storeu_epi32(offset_ptr_00, i_nw_offset);
                _mm256_storeu_epi32(offset_ptr_01, i_ne_offset);
                _mm256_storeu_epi32(offset_ptr_10, i_sw_offset);
                _mm256_storeu_epi32(offset_ptr_11, i_se_offset);

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

                in_bound_ptr_00 += 8;
                in_bound_ptr_01 += 8;
                in_bound_ptr_10 += 8;
                in_bound_ptr_11 += 8;

                value_ptr_alpha += 8;
                value_ptr_beta += 8;
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

                int x0 = (int)floor(sample_x);
                int y0 = (int)floor(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                bool x0_in_bound = (x0 > -1) & (x0 < src.w);
                bool x1_in_bound = (x1 > -1) & (x1 < src.w);
                bool y0_in_bound = (y0 > -1) & (y0 < src.h);
                bool y1_in_bound = (y1 > -1) & (y1 < src.h);

                *in_bound_ptr_00 = (x0_in_bound & y0_in_bound) ? -1.0f : 0.0f;
                *in_bound_ptr_01 = (x1_in_bound & y0_in_bound) ? -1.0f : 0.0f;
                *in_bound_ptr_10 = (x0_in_bound & y1_in_bound) ? -1.0f : 0.0f;
                *in_bound_ptr_11 = (x1_in_bound & y1_in_bound) ? -1.0f : 0.0f;

                *offset_ptr_00 = (x0 + y0 * src.w) * src.elempack;
                *offset_ptr_01 = (x1 + y0 * src.w) * src.elempack;
                *offset_ptr_10 = (x0 + y1 * src.w) * src.elempack;
                *offset_ptr_11 = (x1 + y1 * src.w) * src.elempack;

                *value_ptr_alpha = sample_x - x0;
                *value_ptr_beta = sample_y - y0;

                gridptr_x++;
                gridptr_y++;

                offset_ptr_00++;
                offset_ptr_01++;
                offset_ptr_10++;
                offset_ptr_11++;

                in_bound_ptr_00++;
                in_bound_ptr_01++;
                in_bound_ptr_10++;
                in_bound_ptr_11++;

                value_ptr_alpha++;
                value_ptr_beta++;
            }
        }
    }
};

#if __AVX__
static void gridsample_2d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Mat& value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const int* offset_ptr_00 = offset.channel(0);
        const int* offset_ptr_01 = offset.channel(1);
        const int* offset_ptr_10 = offset.channel(2);
        const int* offset_ptr_11 = offset.channel(3);

        const float* in_bound_ptr_00 = in_bound.channel(0);
        const float* in_bound_ptr_01 = in_bound.channel(1);
        const float* in_bound_ptr_10 = in_bound.channel(2);
        const float* in_bound_ptr_11 = in_bound.channel(3);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            __m256i v00_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_00), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v01_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_01), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v10_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_10), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v11_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_11), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            __m256 v00_in_range = _mm256_set1_ps(*in_bound_ptr_00);
            __m256 v01_in_range = _mm256_set1_ps(*in_bound_ptr_01);
            __m256 v10_in_range = _mm256_set1_ps(*in_bound_ptr_10);
            __m256 v11_in_range = _mm256_set1_ps(*in_bound_ptr_11);

            __m256 v00_val = mask_gather_ps256(srcptr, v00_offset, v00_in_range);
            __m256 v01_val = mask_gather_ps256(srcptr, v01_offset, v01_in_range);
            __m256 v10_val = mask_gather_ps256(srcptr, v10_offset, v10_in_range);
            __m256 v11_val = mask_gather_ps256(srcptr, v11_offset, v11_in_range);

            __m256 alpha = _mm256_set1_ps(*value_ptr_alpha);
            __m256 beta = _mm256_set1_ps(*value_ptr_beta);

            __m256 v0 = _mm256_comp_fmadd_ps(v01_val, alpha, _mm256_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m256 v1 = _mm256_comp_fmadd_ps(v11_val, alpha, _mm256_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m256 _v = _mm256_comp_fmadd_ps(v1, beta, _mm256_comp_fnmadd_ps(v0, beta, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            in_bound_ptr_00++;
            in_bound_ptr_01++;
            in_bound_ptr_10++;
            in_bound_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;

            dstptr += 8;
        }
    }
}
#endif // __AVX__