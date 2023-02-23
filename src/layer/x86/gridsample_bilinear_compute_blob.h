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

                    __m256 nw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(y_n, vImgWf, x_w), vElempackf);
                    __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                    __m256 sw_offset = _mm256_comp_fmadd_ps(vImgWf, vElempackf, nw_offset);
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

                    __m256 nw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(y_n, vImgWf, x_w), vElempackf);
                    __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                    __m256 sw_offset = _mm256_comp_fmadd_ps(vImgWf, vElempackf, nw_offset);
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

                __m256 nw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(y_n, vImgWf, x_w), vElempackf);
                __m256 ne_offset = _mm256_add_ps(nw_offset, vElempackf);
                __m256 sw_offset = _mm256_comp_fmadd_ps(vImgWf, vElempackf, nw_offset);
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

template<PaddingMode pd, bool align_corner>
struct gridsample_3d_bilinear_compute_blob
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vImgDf = _mm256_set1_ps(src.d);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(src.w);
        const __m256i vImgHi = _mm256_set1_epi32(src.h);
        const __m256i vImgDi = _mm256_set1_epi32(src.d);
        const __m256i vElempacki = _mm256_set1_epi32(src.elempack);
#endif // __AVX2__
#endif // __AVX__

        int* offset_ptr_000 = offset.channel(0);
        int* offset_ptr_001 = offset.channel(1);
        int* offset_ptr_010 = offset.channel(2);
        int* offset_ptr_011 = offset.channel(3);

        int* offset_ptr_100 = offset.channel(4);
        int* offset_ptr_101 = offset.channel(5);
        int* offset_ptr_110 = offset.channel(6);
        int* offset_ptr_111 = offset.channel(7);

        float* in_bound_ptr_000 = in_bound.channel(0);
        float* in_bound_ptr_001 = in_bound.channel(1);
        float* in_bound_ptr_010 = in_bound.channel(2);
        float* in_bound_ptr_011 = in_bound.channel(3);
        float* in_bound_ptr_100 = in_bound.channel(4);
        float* in_bound_ptr_101 = in_bound.channel(5);
        float* in_bound_ptr_110 = in_bound.channel(6);
        float* in_bound_ptr_111 = in_bound.channel(7);

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

                    __m256 x_w = _mm256_floor_ps(gx);
                    __m256 y_n = _mm256_floor_ps(gy);
                    __m256 z_t = _mm256_floor_ps(gz);
#if __AVX2__
                    __m256i x0 = _mm256_cvtps_epi32(x_w);
                    __m256i y0 = _mm256_cvtps_epi32(y_n);
                    __m256i z0 = _mm256_cvtps_epi32(z_t);

                    __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                    __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);
                    __m256i z1 = _mm256_add_epi32(z0, *(__m256i*)_pi32_256_1);

                    __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                    __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));
                    __m256i z1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z1));

                    __m256i v011_in_range, v110_in_range, v101_in_range, v111_in_range;
                    {
                        v011_in_range = _mm256_and_si256(x1_in_range, y1_in_range);
                        v101_in_range = _mm256_and_si256(x1_in_range, z1_in_range);
                        v110_in_range = _mm256_and_si256(y1_in_range, z1_in_range);
                        v111_in_range = _mm256_and_si256(v011_in_range, z1_in_range);
                    }

                    __m256i i_tnw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0)), vElempacki);
                    __m256i i_tne_offset = _mm256_add_epi32(i_tnw_offset, vElempacki);
                    __m256i i_tsw_offset = _mm256_add_epi32(i_tnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                    __m256i i_tse_offset = _mm256_add_epi32(i_tsw_offset, vElempacki);

                    __m256i i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                    __m256i i_bne_offset = _mm256_add_epi32(i_bnw_offset, vElempacki);
                    __m256i i_bsw_offset = _mm256_add_epi32(i_bnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                    __m256i i_bse_offset = _mm256_add_epi32(i_bsw_offset, vElempacki);

                    _mm256_storeu_ps(in_bound_ptr_000, *(__m256*)_ps256_n1);
                    _mm256_storeu_ps(in_bound_ptr_001, _mm256_castsi256_ps(x1_in_range));
                    _mm256_storeu_ps(in_bound_ptr_010, _mm256_castsi256_ps(y1_in_range));
                    _mm256_storeu_ps(in_bound_ptr_011, _mm256_castsi256_ps(v011_in_range));

                    _mm256_storeu_ps(in_bound_ptr_100, _mm256_castsi256_ps(z1_in_range));
                    _mm256_storeu_ps(in_bound_ptr_101, _mm256_castsi256_ps(v101_in_range));
                    _mm256_storeu_ps(in_bound_ptr_110, _mm256_castsi256_ps(v110_in_range));
                    _mm256_storeu_ps(in_bound_ptr_111, _mm256_castsi256_ps(v111_in_range));
#else
                    __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                    __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                    __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                    __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                    __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                    __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                    __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                    __m256 z0_in_range = _mm256_and_ps(_mm256_cmp_ps(z_t, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z_t, _CMP_GT_OS));
                    __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

                    __m256 v011_in_range, v110_in_range, v101_in_range, v111_in_range;
                    {
                        v011_in_range = _mm256_and_ps(x1_in_range, y1_in_range);
                        v101_in_range = _mm256_and_ps(x1_in_range, z1_in_range);
                        v110_in_range = _mm256_and_ps(y1_in_range, z1_in_range);
                        v111_in_range = _mm256_and_ps(v011_in_range, z1_in_range);
                    }

                    __m256 tnw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t,
                                                      _mm256_comp_fmadd_ps(y_n, vImgWf, x_w)),
                                                      vElempackf);
                    __m256 tne_offset = _mm256_add_ps(tnw_offset, vElempackf);
                    __m256 tsw_offset = _mm256_add_ps(tnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                    __m256 tse_offset = _mm256_add_ps(tsw_offset, vElempackf);

                    __m256 bnw_offset = _mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), vElempackf, tnw_offset);
                    __m256 bne_offset = _mm256_add_ps(bnw_offset, vElempackf);
                    __m256 bsw_offset = _mm256_add_ps(bnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                    __m256 bse_offset = _mm256_add_ps(bsw_offset, vElempackf);

                    __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                    __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                    __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                    __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                    __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                    __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                    __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                    __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                    _mm256_storeu_ps(in_bound_ptr_000, *(__m256*)_ps256_n1);
                    _mm256_storeu_ps(in_bound_ptr_001, x1_in_range);
                    _mm256_storeu_ps(in_bound_ptr_010, y1_in_range);
                    _mm256_storeu_ps(in_bound_ptr_011, v011_in_range);

                    _mm256_storeu_ps(in_bound_ptr_100, z1_in_range);
                    _mm256_storeu_ps(in_bound_ptr_101, v101_in_range);
                    _mm256_storeu_ps(in_bound_ptr_110, v110_in_range);
                    _mm256_storeu_ps(in_bound_ptr_111, v111_in_range);
#endif // __AVX2__
                    _mm256_storeu_epi32(offset_ptr_000, i_tnw_offset);
                    _mm256_storeu_epi32(offset_ptr_001, i_tne_offset);
                    _mm256_storeu_epi32(offset_ptr_010, i_tsw_offset);
                    _mm256_storeu_epi32(offset_ptr_011, i_tse_offset);

                    _mm256_storeu_epi32(offset_ptr_100, i_bnw_offset);
                    _mm256_storeu_epi32(offset_ptr_101, i_bne_offset);
                    _mm256_storeu_epi32(offset_ptr_110, i_bsw_offset);
                    _mm256_storeu_epi32(offset_ptr_111, i_bse_offset);

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

                    in_bound_ptr_000 += 8;
                    in_bound_ptr_001 += 8;
                    in_bound_ptr_010 += 8;
                    in_bound_ptr_011 += 8;

                    in_bound_ptr_100 += 8;
                    in_bound_ptr_101 += 8;
                    in_bound_ptr_110 += 8;
                    in_bound_ptr_111 += 8;

                    value_ptr_alpha += 8;
                    value_ptr_beta += 8;
                    value_ptr_gamma += 8;
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

                    int x0 = (int)floor(sample_x);
                    int y0 = (int)floor(sample_y);
                    int z0 = (int)floor(sample_z);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    bool x1_in_range = (x1 > -1) & (x1 < src.w);
                    bool y1_in_range = (y1 > -1) & (y1 < src.h);
                    bool z1_in_range = (z1 > -1) & (z1 < src.d);

                    bool v11_in_range = x1_in_range & y1_in_range;

                    *in_bound_ptr_000 = -1.0f;
                    *in_bound_ptr_001 = x1_in_range ? -1.0f : 0.0f;
                    *in_bound_ptr_010 = y1_in_range ? -1.0f : 0.0f;
                    *in_bound_ptr_011 = v11_in_range ? -1.0f : 0.0f;

                    *in_bound_ptr_100 = z1_in_range ? -1.0f : 0.0f;
                    *in_bound_ptr_101 = (x1_in_range & z1_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_110 = (y1_in_range & z1_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_111 = (v11_in_range & z1_in_range) ? -1.0f : 0.0f;

                    *offset_ptr_000 = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                    *offset_ptr_001 = (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                    *offset_ptr_010 = (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack;
                    *offset_ptr_011 = (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack;

                    *offset_ptr_100 = (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                    *offset_ptr_101 = (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                    *offset_ptr_110 = (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack;
                    *offset_ptr_111 = (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack;

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

                    in_bound_ptr_000++;
                    in_bound_ptr_001++;
                    in_bound_ptr_010++;
                    in_bound_ptr_011++;

                    in_bound_ptr_100++;
                    in_bound_ptr_101++;
                    in_bound_ptr_110++;
                    in_bound_ptr_111++;

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

                __m256 x_w = _mm256_floor_ps(gx);
                __m256 y_n = _mm256_floor_ps(gy);
                __m256 z_t = _mm256_floor_ps(gz);
#if __AVX2__
                __m256i x0 = _mm256_cvtps_epi32(x_w);
                __m256i y0 = _mm256_cvtps_epi32(y_n);
                __m256i z0 = _mm256_cvtps_epi32(z_t);

                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);
                __m256i z1 = _mm256_add_epi32(z0, *(__m256i*)_pi32_256_1);

                __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x0));
                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                __m256i y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y0));
                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));
                __m256i z0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z0));
                __m256i z1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z1));

                __m256i v011_in_range, v110_in_range, v101_in_range, v111_in_range;
                {
                    v011_in_range = _mm256_and_si256(x1_in_range, y1_in_range);
                    v101_in_range = _mm256_and_si256(x1_in_range, z1_in_range);
                    v110_in_range = _mm256_and_si256(y1_in_range, z1_in_range);
                    v111_in_range = _mm256_and_si256(v011_in_range, z1_in_range);
                }

                __m256i i_tnw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0)), vElempacki);
                __m256i i_tne_offset = _mm256_add_epi32(i_tnw_offset, vElempacki);
                __m256i i_tsw_offset = _mm256_add_epi32(i_tnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                __m256i i_tse_offset = _mm256_add_epi32(i_tsw_offset, vElempacki);

                __m256i i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m256i i_bne_offset = _mm256_add_epi32(i_bnw_offset, vElempacki);
                __m256i i_bsw_offset = _mm256_add_epi32(i_bnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                __m256i i_bse_offset = _mm256_add_epi32(i_bsw_offset, vElempacki);

                _mm256_storeu_ps(in_bound_ptr_000, *(__m256*)_ps256_n1);
                _mm256_storeu_ps(in_bound_ptr_001, _mm256_castsi256_ps(x1_in_range));
                _mm256_storeu_ps(in_bound_ptr_010, _mm256_castsi256_ps(y1_in_range));
                _mm256_storeu_ps(in_bound_ptr_011, _mm256_castsi256_ps(v011_in_range));

                _mm256_storeu_ps(in_bound_ptr_100, _mm256_castsi256_ps(z1_in_range));
                _mm256_storeu_ps(in_bound_ptr_101, _mm256_castsi256_ps(v101_in_range));
                _mm256_storeu_ps(in_bound_ptr_110, _mm256_castsi256_ps(v110_in_range));
                _mm256_storeu_ps(in_bound_ptr_111, _mm256_castsi256_ps(v111_in_range));
#else
                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                __m256 z0_in_range = _mm256_and_ps(_mm256_cmp_ps(z_t, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z_t, _CMP_GT_OS));
                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

                __m256 v011_in_range, v110_in_range, v101_in_range, v111_in_range;
                {
                    v011_in_range = _mm256_and_ps(x1_in_range, y1_in_range);
                    v101_in_range = _mm256_and_ps(x1_in_range, z1_in_range);
                    v110_in_range = _mm256_and_ps(y1_in_range, z1_in_range);
                    v111_in_range = _mm256_and_ps(v011_in_range, z1_in_range);
                }

                __m256 tnw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t,
                                                  _mm256_comp_fmadd_ps(y_n, vImgWf, x_w)),
                                                  vElempackf);
                __m256 tne_offset = _mm256_add_ps(tnw_offset, vElempackf);
                __m256 tsw_offset = _mm256_add_ps(tnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                __m256 tse_offset = _mm256_add_ps(tsw_offset, vElempackf);

                __m256 bnw_offset = _mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), vElempackf, tnw_offset);
                __m256 bne_offset = _mm256_add_ps(bnw_offset, vElempackf);
                __m256 bsw_offset = _mm256_add_ps(bnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                __m256 bse_offset = _mm256_add_ps(bsw_offset, vElempackf);

                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                _mm256_storeu_ps(in_bound_ptr_000, *(__m256*)_ps256_n1);
                _mm256_storeu_ps(in_bound_ptr_001, x1_in_range);
                _mm256_storeu_ps(in_bound_ptr_010, y1_in_range);
                _mm256_storeu_ps(in_bound_ptr_011, v011_in_range);

                _mm256_storeu_ps(in_bound_ptr_100, z1_in_range);
                _mm256_storeu_ps(in_bound_ptr_101, v101_in_range);
                _mm256_storeu_ps(in_bound_ptr_110, v110_in_range);
                _mm256_storeu_ps(in_bound_ptr_111, v111_in_range);
#endif // __AVX2__
                _mm256_storeu_epi32(offset_ptr_000, i_tnw_offset);
                _mm256_storeu_epi32(offset_ptr_001, i_tne_offset);
                _mm256_storeu_epi32(offset_ptr_010, i_tsw_offset);
                _mm256_storeu_epi32(offset_ptr_011, i_tse_offset);

                _mm256_storeu_epi32(offset_ptr_100, i_bnw_offset);
                _mm256_storeu_epi32(offset_ptr_101, i_bne_offset);
                _mm256_storeu_epi32(offset_ptr_110, i_bsw_offset);
                _mm256_storeu_epi32(offset_ptr_111, i_bse_offset);

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

                in_bound_ptr_000 += 8;
                in_bound_ptr_001 += 8;
                in_bound_ptr_010 += 8;
                in_bound_ptr_011 += 8;

                in_bound_ptr_100 += 8;
                in_bound_ptr_101 += 8;
                in_bound_ptr_110 += 8;
                in_bound_ptr_111 += 8;

                value_ptr_alpha += 8;
                value_ptr_beta += 8;
                value_ptr_gamma += 8;
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

                int x0 = (int)floor(sample_x);
                int y0 = (int)floor(sample_y);
                int z0 = (int)floor(sample_z);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                bool x1_in_range = (x1 > -1) & (x1 < src.w);
                bool y1_in_range = (y1 > -1) & (y1 < src.h);
                bool z1_in_range = (z1 > -1) & (z1 < src.d);

                bool v11_in_range = x1_in_range & y1_in_range;

                *in_bound_ptr_000 = -1.0f;
                *in_bound_ptr_001 = x1_in_range ? -1.0f : 0.0f;
                *in_bound_ptr_010 = y1_in_range ? -1.0f : 0.0f;
                *in_bound_ptr_011 = v11_in_range ? -1.0f : 0.0f;

                *in_bound_ptr_100 = z1_in_range ? -1.0f : 0.0f;
                *in_bound_ptr_101 = (x1_in_range & z1_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_110 = (y1_in_range & z1_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_111 = (v11_in_range & z1_in_range) ? -1.0f : 0.0f;

                *offset_ptr_000 = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                *offset_ptr_001 = (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                *offset_ptr_010 = (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack;
                *offset_ptr_011 = (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack;

                *offset_ptr_100 = (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                *offset_ptr_101 = (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                *offset_ptr_110 = (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack;
                *offset_ptr_111 = (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack;

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

                in_bound_ptr_000++;
                in_bound_ptr_001++;
                in_bound_ptr_010++;
                in_bound_ptr_011++;

                in_bound_ptr_100++;
                in_bound_ptr_101++;
                in_bound_ptr_110++;
                in_bound_ptr_111++;

                value_ptr_alpha++;
                value_ptr_beta++;
                value_ptr_gamma++;
            }
        }
    }
};

template<bool align_corner>
struct gridsample_3d_bilinear_compute_blob<PaddingMode::Zeros, align_corner>
{
    void operator()(const Mat& src, const Mat& grid, Mat& offset, Mat& in_bound, Mat& value, int permute_fusion, const Option& opt)
    {
        const int grid_size = grid.w * grid.h * grid.d;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(src.w);
        const __m256 vImgHf = _mm256_set1_ps(src.h);
        const __m256 vImgDf = _mm256_set1_ps(src.d);
        const __m256 vElempackf = _mm256_set1_ps(src.elempack);
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(src.w);
        const __m256i vImgHi = _mm256_set1_epi32(src.h);
        const __m256i vImgDi = _mm256_set1_epi32(src.d);
        const __m256i vElempacki = _mm256_set1_epi32(src.elempack);
#endif // __AVX2__
#endif // __AVX__

        int* offset_ptr_000 = offset.channel(0);
        int* offset_ptr_001 = offset.channel(1);
        int* offset_ptr_010 = offset.channel(2);
        int* offset_ptr_011 = offset.channel(3);

        int* offset_ptr_100 = offset.channel(4);
        int* offset_ptr_101 = offset.channel(5);
        int* offset_ptr_110 = offset.channel(6);
        int* offset_ptr_111 = offset.channel(7);

        float* in_bound_ptr_000 = in_bound.channel(0);
        float* in_bound_ptr_001 = in_bound.channel(1);
        float* in_bound_ptr_010 = in_bound.channel(2);
        float* in_bound_ptr_011 = in_bound.channel(3);
        float* in_bound_ptr_100 = in_bound.channel(4);
        float* in_bound_ptr_101 = in_bound.channel(5);
        float* in_bound_ptr_110 = in_bound.channel(6);
        float* in_bound_ptr_111 = in_bound.channel(7);

        float* value_ptr_alpha = value.channel(0);
        float* value_ptr_beta = value.channel(1);
        float* value_ptr_gamma = value.channel(2);

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

                    __m256 x_w = _mm256_floor_ps(gx);
                    __m256 y_n = _mm256_floor_ps(gy);
                    __m256 z_t = _mm256_floor_ps(gz);
#if __AVX2__
                    __m256i x0 = _mm256_cvtps_epi32(x_w);
                    __m256i y0 = _mm256_cvtps_epi32(y_n);
                    __m256i z0 = _mm256_cvtps_epi32(z_t);

                    __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                    __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);
                    __m256i z1 = _mm256_add_epi32(z0, *(__m256i*)_pi32_256_1);

                    __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x0));
                    __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                    __m256i y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y0));
                    __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));
                    __m256i z0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z0));
                    __m256i z1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z1));

                    __m256i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                    {
                        __m256i v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                        __m256i v01_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                        __m256i v10_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                        __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                        v000_in_range = _mm256_and_si256(v00_in_range, z0_in_range);
                        v001_in_range = _mm256_and_si256(v01_in_range, z0_in_range);
                        v010_in_range = _mm256_and_si256(v10_in_range, z0_in_range);
                        v011_in_range = _mm256_and_si256(v11_in_range, z0_in_range);

                        v100_in_range = _mm256_and_si256(v00_in_range, z1_in_range);
                        v101_in_range = _mm256_and_si256(v01_in_range, z1_in_range);
                        v110_in_range = _mm256_and_si256(v10_in_range, z1_in_range);
                        v111_in_range = _mm256_and_si256(v11_in_range, z1_in_range);
                    }

                    __m256i i_tnw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0)), vElempacki);
                    __m256i i_tne_offset = _mm256_add_epi32(i_tnw_offset, vElempacki);
                    __m256i i_tsw_offset = _mm256_add_epi32(i_tnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                    __m256i i_tse_offset = _mm256_add_epi32(i_tsw_offset, vElempacki);

                    __m256i i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                    __m256i i_bne_offset = _mm256_add_epi32(i_bnw_offset, vElempacki);
                    __m256i i_bsw_offset = _mm256_add_epi32(i_bnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                    __m256i i_bse_offset = _mm256_add_epi32(i_bsw_offset, vElempacki);

                    _mm256_storeu_ps(in_bound_ptr_000, _mm256_castsi256_ps(v000_in_range));
                    _mm256_storeu_ps(in_bound_ptr_001, _mm256_castsi256_ps(v001_in_range));
                    _mm256_storeu_ps(in_bound_ptr_010, _mm256_castsi256_ps(v010_in_range));
                    _mm256_storeu_ps(in_bound_ptr_011, _mm256_castsi256_ps(v011_in_range));

                    _mm256_storeu_ps(in_bound_ptr_100, _mm256_castsi256_ps(v100_in_range));
                    _mm256_storeu_ps(in_bound_ptr_101, _mm256_castsi256_ps(v101_in_range));
                    _mm256_storeu_ps(in_bound_ptr_110, _mm256_castsi256_ps(v110_in_range));
                    _mm256_storeu_ps(in_bound_ptr_111, _mm256_castsi256_ps(v111_in_range));
#else
                    __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                    __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                    __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                    __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                    __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                    __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                    __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                    __m256 z0_in_range = _mm256_and_ps(_mm256_cmp_ps(z_t, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z_t, _CMP_GT_OS));
                    __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

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

                    __m256 tnw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t,
                                                      _mm256_comp_fmadd_ps(y_n, vImgWf, x_w)),
                                                      vElempackf);
                    __m256 tne_offset = _mm256_add_ps(tnw_offset, vElempackf);
                    __m256 tsw_offset = _mm256_add_ps(tnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                    __m256 tse_offset = _mm256_add_ps(tsw_offset, vElempackf);

                    __m256 bnw_offset = _mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), vElempackf, tnw_offset);
                    __m256 bne_offset = _mm256_add_ps(bnw_offset, vElempackf);
                    __m256 bsw_offset = _mm256_add_ps(bnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                    __m256 bse_offset = _mm256_add_ps(bsw_offset, vElempackf);

                    __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                    __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                    __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                    __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                    __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                    __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                    __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                    __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                    _mm256_storeu_ps(in_bound_ptr_000, v000_in_range);
                    _mm256_storeu_ps(in_bound_ptr_001, v001_in_range);
                    _mm256_storeu_ps(in_bound_ptr_010, v010_in_range);
                    _mm256_storeu_ps(in_bound_ptr_011, v011_in_range);

                    _mm256_storeu_ps(in_bound_ptr_100, v100_in_range);
                    _mm256_storeu_ps(in_bound_ptr_101, v101_in_range);
                    _mm256_storeu_ps(in_bound_ptr_110, v110_in_range);
                    _mm256_storeu_ps(in_bound_ptr_111, v111_in_range);
#endif // __AVX2__
                    _mm256_storeu_epi32(offset_ptr_000, i_tnw_offset);
                    _mm256_storeu_epi32(offset_ptr_001, i_tne_offset);
                    _mm256_storeu_epi32(offset_ptr_010, i_tsw_offset);
                    _mm256_storeu_epi32(offset_ptr_011, i_tse_offset);

                    _mm256_storeu_epi32(offset_ptr_100, i_bnw_offset);
                    _mm256_storeu_epi32(offset_ptr_101, i_bne_offset);
                    _mm256_storeu_epi32(offset_ptr_110, i_bsw_offset);
                    _mm256_storeu_epi32(offset_ptr_111, i_bse_offset);

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

                    in_bound_ptr_000 += 8;
                    in_bound_ptr_001 += 8;
                    in_bound_ptr_010 += 8;
                    in_bound_ptr_011 += 8;

                    in_bound_ptr_100 += 8;
                    in_bound_ptr_101 += 8;
                    in_bound_ptr_110 += 8;
                    in_bound_ptr_111 += 8;

                    value_ptr_alpha += 8;
                    value_ptr_beta += 8;
                    value_ptr_gamma += 8;
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

                    *in_bound_ptr_000 = (v00_in_range & z0_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_001 = (v01_in_range & z0_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_010 = (v10_in_range & z0_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_011 = (v11_in_range & z0_in_range) ? -1.0f : 0.0f;

                    *in_bound_ptr_100 = (v00_in_range & z1_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_101 = (v01_in_range & z1_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_110 = (v10_in_range & z1_in_range) ? -1.0f : 0.0f;
                    *in_bound_ptr_111 = (v11_in_range & z1_in_range) ? -1.0f : 0.0f;

                    *offset_ptr_000 = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                    *offset_ptr_001 = (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                    *offset_ptr_010 = (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack;
                    *offset_ptr_011 = (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack;

                    *offset_ptr_100 = (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                    *offset_ptr_101 = (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                    *offset_ptr_110 = (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack;
                    *offset_ptr_111 = (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack;

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

                    in_bound_ptr_000++;
                    in_bound_ptr_001++;
                    in_bound_ptr_010++;
                    in_bound_ptr_011++;

                    in_bound_ptr_100++;
                    in_bound_ptr_101++;
                    in_bound_ptr_110++;
                    in_bound_ptr_111++;

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
                    gy = unormalize(vImgHf, gy);
                    gz = unormalize(vImgDf, gz);
                }

                __m256 x_w = _mm256_floor_ps(gx);
                __m256 y_n = _mm256_floor_ps(gy);
                __m256 z_t = _mm256_floor_ps(gz);
#if __AVX2__
                __m256i x0 = _mm256_cvtps_epi32(x_w);
                __m256i y0 = _mm256_cvtps_epi32(y_n);
                __m256i z0 = _mm256_cvtps_epi32(z_t);

                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);
                __m256i z1 = _mm256_add_epi32(z0, *(__m256i*)_pi32_256_1);

                __m256i x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x0));
                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                __m256i y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y0));
                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));
                __m256i z0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z0, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z0));
                __m256i z1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgDi, z1));

                __m256i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                {
                    __m256i v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                    __m256i v01_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                    __m256i v10_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                    __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                    v000_in_range = _mm256_and_si256(v00_in_range, z0_in_range);
                    v001_in_range = _mm256_and_si256(v01_in_range, z0_in_range);
                    v010_in_range = _mm256_and_si256(v10_in_range, z0_in_range);
                    v011_in_range = _mm256_and_si256(v11_in_range, z0_in_range);

                    v100_in_range = _mm256_and_si256(v00_in_range, z1_in_range);
                    v101_in_range = _mm256_and_si256(v01_in_range, z1_in_range);
                    v110_in_range = _mm256_and_si256(v10_in_range, z1_in_range);
                    v111_in_range = _mm256_and_si256(v11_in_range, z1_in_range);
                }

                __m256i i_tnw_offset = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0)), vElempacki);
                __m256i i_tne_offset = _mm256_add_epi32(i_tnw_offset, vElempacki);
                __m256i i_tsw_offset = _mm256_add_epi32(i_tnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                __m256i i_tse_offset = _mm256_add_epi32(i_tsw_offset, vElempacki);

                __m256i i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                __m256i i_bne_offset = _mm256_add_epi32(i_bnw_offset, vElempacki);
                __m256i i_bsw_offset = _mm256_add_epi32(i_bnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                __m256i i_bse_offset = _mm256_add_epi32(i_bsw_offset, vElempacki);

                _mm256_storeu_ps(in_bound_ptr_000, _mm256_castsi256_ps(v000_in_range));
                _mm256_storeu_ps(in_bound_ptr_001, _mm256_castsi256_ps(v001_in_range));
                _mm256_storeu_ps(in_bound_ptr_010, _mm256_castsi256_ps(v010_in_range));
                _mm256_storeu_ps(in_bound_ptr_011, _mm256_castsi256_ps(v011_in_range));

                _mm256_storeu_ps(in_bound_ptr_100, _mm256_castsi256_ps(v100_in_range));
                _mm256_storeu_ps(in_bound_ptr_101, _mm256_castsi256_ps(v101_in_range));
                _mm256_storeu_ps(in_bound_ptr_110, _mm256_castsi256_ps(v110_in_range));
                _mm256_storeu_ps(in_bound_ptr_111, _mm256_castsi256_ps(v111_in_range));
#else
                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                __m256 z0_in_range = _mm256_and_ps(_mm256_cmp_ps(z_t, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z_t, _CMP_GT_OS));
                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

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

                __m256 tnw_offset = _mm256_mul_ps(_mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t,
                                                  _mm256_comp_fmadd_ps(y_n, vImgWf, x_w)),
                                                  vElempackf);
                __m256 tne_offset = _mm256_add_ps(tnw_offset, vElempackf);
                __m256 tsw_offset = _mm256_add_ps(tnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                __m256 tse_offset = _mm256_add_ps(tsw_offset, vElempackf);

                __m256 bnw_offset = _mm256_comp_fmadd_ps(_mm256_mul_ps(vImgWf, vImgHf), vElempackf, tnw_offset);
                __m256 bne_offset = _mm256_add_ps(bnw_offset, vElempackf);
                __m256 bsw_offset = _mm256_add_ps(bnw_offset, _mm256_mul_ps(vImgWf, vElempackf));
                __m256 bse_offset = _mm256_add_ps(bsw_offset, vElempackf);

                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                _mm256_storeu_ps(in_bound_ptr_000, v000_in_range);
                _mm256_storeu_ps(in_bound_ptr_001, v001_in_range);
                _mm256_storeu_ps(in_bound_ptr_010, v010_in_range);
                _mm256_storeu_ps(in_bound_ptr_011, v011_in_range);

                _mm256_storeu_ps(in_bound_ptr_100, v100_in_range);
                _mm256_storeu_ps(in_bound_ptr_101, v101_in_range);
                _mm256_storeu_ps(in_bound_ptr_110, v110_in_range);
                _mm256_storeu_ps(in_bound_ptr_111, v111_in_range);
#endif // __AVX2__
                _mm256_storeu_epi32(offset_ptr_000, i_tnw_offset);
                _mm256_storeu_epi32(offset_ptr_001, i_tne_offset);
                _mm256_storeu_epi32(offset_ptr_010, i_tsw_offset);
                _mm256_storeu_epi32(offset_ptr_011, i_tse_offset);

                _mm256_storeu_epi32(offset_ptr_100, i_bnw_offset);
                _mm256_storeu_epi32(offset_ptr_101, i_bne_offset);
                _mm256_storeu_epi32(offset_ptr_110, i_bsw_offset);
                _mm256_storeu_epi32(offset_ptr_111, i_bse_offset);

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

                in_bound_ptr_000 += 8;
                in_bound_ptr_001 += 8;
                in_bound_ptr_010 += 8;
                in_bound_ptr_011 += 8;

                in_bound_ptr_100 += 8;
                in_bound_ptr_101 += 8;
                in_bound_ptr_110 += 8;
                in_bound_ptr_111 += 8;

                value_ptr_alpha += 8;
                value_ptr_beta += 8;
                value_ptr_gamma += 8;
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

                *in_bound_ptr_000 = (v00_in_range & z0_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_001 = (v01_in_range & z0_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_010 = (v10_in_range & z0_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_011 = (v11_in_range & z0_in_range) ? -1.0f : 0.0f;

                *in_bound_ptr_100 = (v00_in_range & z1_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_101 = (v01_in_range & z1_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_110 = (v10_in_range & z1_in_range) ? -1.0f : 0.0f;
                *in_bound_ptr_111 = (v11_in_range & z1_in_range) ? -1.0f : 0.0f;

                *offset_ptr_000 = (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                *offset_ptr_001 = (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack;
                *offset_ptr_010 = (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack;
                *offset_ptr_011 = (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack;

                *offset_ptr_100 = (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                *offset_ptr_101 = (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack;
                *offset_ptr_110 = (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack;
                *offset_ptr_111 = (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack;

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

                in_bound_ptr_000++;
                in_bound_ptr_001++;
                in_bound_ptr_010++;
                in_bound_ptr_011++;

                in_bound_ptr_100++;
                in_bound_ptr_101++;
                in_bound_ptr_110++;
                in_bound_ptr_111++;

                value_ptr_alpha++;
                value_ptr_beta++;
                value_ptr_gamma++;
            }
        }
    }
};

#if __SSE2__
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
static void gridsample_3d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Mat& value, const Option& opt)
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

        const int* offset_ptr_000 = offset.channel(0);
        const int* offset_ptr_001 = offset.channel(1);
        const int* offset_ptr_010 = offset.channel(2);
        const int* offset_ptr_011 = offset.channel(3);
        const int* offset_ptr_100 = offset.channel(4);
        const int* offset_ptr_101 = offset.channel(5);
        const int* offset_ptr_110 = offset.channel(6);
        const int* offset_ptr_111 = offset.channel(7);

        const float* in_bound_ptr_000 = in_bound.channel(0);
        const float* in_bound_ptr_001 = in_bound.channel(1);
        const float* in_bound_ptr_010 = in_bound.channel(2);
        const float* in_bound_ptr_011 = in_bound.channel(3);
        const float* in_bound_ptr_100 = in_bound.channel(4);
        const float* in_bound_ptr_101 = in_bound.channel(5);
        const float* in_bound_ptr_110 = in_bound.channel(6);
        const float* in_bound_ptr_111 = in_bound.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        for (int i = 0; i < grid_size; i++)
        {
            __m256i v000_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_000), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v001_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_001), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v010_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_010), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v011_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_011), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v100_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_100), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v101_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_101), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v110_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_110), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v111_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_111), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

            __m256 v000_in_range = _mm256_set1_ps(*in_bound_ptr_000);
            __m256 v001_in_range = _mm256_set1_ps(*in_bound_ptr_001);
            __m256 v010_in_range = _mm256_set1_ps(*in_bound_ptr_010);
            __m256 v011_in_range = _mm256_set1_ps(*in_bound_ptr_011);
            __m256 v100_in_range = _mm256_set1_ps(*in_bound_ptr_100);
            __m256 v101_in_range = _mm256_set1_ps(*in_bound_ptr_101);
            __m256 v110_in_range = _mm256_set1_ps(*in_bound_ptr_110);
            __m256 v111_in_range = _mm256_set1_ps(*in_bound_ptr_111);

            __m256 v000_val = mask_gather_ps256(srcptr, v000_offset, v000_in_range);
            __m256 v001_val = mask_gather_ps256(srcptr, v001_offset, v001_in_range);
            __m256 v010_val = mask_gather_ps256(srcptr, v010_offset, v010_in_range);
            __m256 v011_val = mask_gather_ps256(srcptr, v011_offset, v011_in_range);
            __m256 v100_val = mask_gather_ps256(srcptr, v100_offset, v100_in_range);
            __m256 v101_val = mask_gather_ps256(srcptr, v101_offset, v101_in_range);
            __m256 v110_val = mask_gather_ps256(srcptr, v110_offset, v110_in_range);
            __m256 v111_val = mask_gather_ps256(srcptr, v111_offset, v111_in_range);

            __m256 alpha = _mm256_set1_ps(*value_ptr_alpha);
            __m256 beta = _mm256_set1_ps(*value_ptr_beta);
            __m256 gamma = _mm256_set1_ps(*value_ptr_gamma);

            __m256 v00 = _mm256_comp_fmadd_ps(v001_val, alpha, _mm256_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m256 v01 = _mm256_comp_fmadd_ps(v011_val, alpha, _mm256_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m256 v10 = _mm256_comp_fmadd_ps(v101_val, alpha, _mm256_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m256 v11 = _mm256_comp_fmadd_ps(v111_val, alpha, _mm256_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m256 v0 = _mm256_comp_fmadd_ps(v01, beta, _mm256_comp_fnmadd_ps(v00, beta, v00));
            __m256 v1 = _mm256_comp_fmadd_ps(v11, beta, _mm256_comp_fnmadd_ps(v10, beta, v10));

            __m256 _v = _mm256_comp_fmadd_ps(v1, gamma, _mm256_comp_fnmadd_ps(v0, gamma, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            in_bound_ptr_000++;
            in_bound_ptr_001++;
            in_bound_ptr_010++;
            in_bound_ptr_011++;

            in_bound_ptr_100++;
            in_bound_ptr_101++;
            in_bound_ptr_110++;
            in_bound_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;

            dstptr += 8;
        }
    }
}
#endif // __AVX__
static void gridsample_2d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Mat& value, const Option& opt)
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
            __m128i v00_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_00), _mm_set_epi32(3, 2, 1, 0));
            __m128i v01_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_01), _mm_set_epi32(3, 2, 1, 0));
            __m128i v10_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_10), _mm_set_epi32(3, 2, 1, 0));
            __m128i v11_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_11), _mm_set_epi32(3, 2, 1, 0));

            __m128 v00_in_range = _mm_set1_ps(*in_bound_ptr_00);
            __m128 v01_in_range = _mm_set1_ps(*in_bound_ptr_01);
            __m128 v10_in_range = _mm_set1_ps(*in_bound_ptr_10);
            __m128 v11_in_range = _mm_set1_ps(*in_bound_ptr_11);

            __m128 v00_val = mask_gather_ps(srcptr, v00_offset, v00_in_range);
            __m128 v01_val = mask_gather_ps(srcptr, v01_offset, v01_in_range);
            __m128 v10_val = mask_gather_ps(srcptr, v10_offset, v10_in_range);
            __m128 v11_val = mask_gather_ps(srcptr, v11_offset, v11_in_range);

            __m128 alpha = _mm_set1_ps(*value_ptr_alpha);
            __m128 beta = _mm_set1_ps(*value_ptr_beta);

            __m128 v0 = _mm_comp_fmadd_ps(v01_val, alpha, _mm_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m128 v1 = _mm_comp_fmadd_ps(v11_val, alpha, _mm_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m128 _v = _mm_comp_fmadd_ps(v1, beta, _mm_comp_fnmadd_ps(v0, beta, v0));
            _mm_storeu_ps(dstptr, _v);

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

            dstptr += 4;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Mat& value, const Option& opt)
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

        const int* offset_ptr_000 = offset.channel(0);
        const int* offset_ptr_001 = offset.channel(1);
        const int* offset_ptr_010 = offset.channel(2);
        const int* offset_ptr_011 = offset.channel(3);
        const int* offset_ptr_100 = offset.channel(4);
        const int* offset_ptr_101 = offset.channel(5);
        const int* offset_ptr_110 = offset.channel(6);
        const int* offset_ptr_111 = offset.channel(7);

        const float* in_bound_ptr_000 = in_bound.channel(0);
        const float* in_bound_ptr_001 = in_bound.channel(1);
        const float* in_bound_ptr_010 = in_bound.channel(2);
        const float* in_bound_ptr_011 = in_bound.channel(3);
        const float* in_bound_ptr_100 = in_bound.channel(4);
        const float* in_bound_ptr_101 = in_bound.channel(5);
        const float* in_bound_ptr_110 = in_bound.channel(6);
        const float* in_bound_ptr_111 = in_bound.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        for (int i = 0; i < grid_size; i++)
        {
            __m128i v000_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_000), _mm_set_epi32(3, 2, 1, 0));
            __m128i v001_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_001), _mm_set_epi32(3, 2, 1, 0));
            __m128i v010_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_010), _mm_set_epi32(3, 2, 1, 0));
            __m128i v011_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_011), _mm_set_epi32(3, 2, 1, 0));
            __m128i v100_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_100), _mm_set_epi32(3, 2, 1, 0));
            __m128i v101_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_101), _mm_set_epi32(3, 2, 1, 0));
            __m128i v110_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_110), _mm_set_epi32(3, 2, 1, 0));
            __m128i v111_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_111), _mm_set_epi32(3, 2, 1, 0));

            __m128 v000_in_range = _mm_set1_ps(*in_bound_ptr_000);
            __m128 v001_in_range = _mm_set1_ps(*in_bound_ptr_001);
            __m128 v010_in_range = _mm_set1_ps(*in_bound_ptr_010);
            __m128 v011_in_range = _mm_set1_ps(*in_bound_ptr_011);
            __m128 v100_in_range = _mm_set1_ps(*in_bound_ptr_100);
            __m128 v101_in_range = _mm_set1_ps(*in_bound_ptr_101);
            __m128 v110_in_range = _mm_set1_ps(*in_bound_ptr_110);
            __m128 v111_in_range = _mm_set1_ps(*in_bound_ptr_111);

            __m128 v000_val = mask_gather_ps(srcptr, v000_offset, v000_in_range);
            __m128 v001_val = mask_gather_ps(srcptr, v001_offset, v001_in_range);
            __m128 v010_val = mask_gather_ps(srcptr, v010_offset, v010_in_range);
            __m128 v011_val = mask_gather_ps(srcptr, v011_offset, v011_in_range);
            __m128 v100_val = mask_gather_ps(srcptr, v100_offset, v100_in_range);
            __m128 v101_val = mask_gather_ps(srcptr, v101_offset, v101_in_range);
            __m128 v110_val = mask_gather_ps(srcptr, v110_offset, v110_in_range);
            __m128 v111_val = mask_gather_ps(srcptr, v111_offset, v111_in_range);

            __m128 alpha = _mm_set1_ps(*value_ptr_alpha);
            __m128 beta = _mm_set1_ps(*value_ptr_beta);
            __m128 gamma = _mm_set1_ps(*value_ptr_gamma);

            __m128 v00 = _mm_comp_fmadd_ps(v001_val, alpha, _mm_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m128 v01 = _mm_comp_fmadd_ps(v011_val, alpha, _mm_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m128 v10 = _mm_comp_fmadd_ps(v101_val, alpha, _mm_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m128 v11 = _mm_comp_fmadd_ps(v111_val, alpha, _mm_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m128 v0 = _mm_comp_fmadd_ps(v01, beta, _mm_comp_fnmadd_ps(v00, beta, v00));
            __m128 v1 = _mm_comp_fmadd_ps(v11, beta, _mm_comp_fnmadd_ps(v10, beta, v10));

            __m128 _v = _mm_comp_fmadd_ps(v1, gamma, _mm_comp_fnmadd_ps(v0, gamma, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            in_bound_ptr_000++;
            in_bound_ptr_001++;
            in_bound_ptr_010++;
            in_bound_ptr_011++;

            in_bound_ptr_100++;
            in_bound_ptr_101++;
            in_bound_ptr_110++;
            in_bound_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;

            dstptr += 4;
        }
    }
}
#endif // __SSE2__

static void gridsample_2d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Mat& value, const Option& opt)
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

        int nn = grid_size;
#if __SSE2__
#if __AVX__

        for (int i = 0; i + 7 < grid_size; i += 8)
        {
            __m256i v00_offset = _mm256_loadu_epi32(offset_ptr_00);
            __m256i v01_offset = _mm256_loadu_epi32(offset_ptr_01);
            __m256i v10_offset = _mm256_loadu_epi32(offset_ptr_10);
            __m256i v11_offset = _mm256_loadu_epi32(offset_ptr_11);

            __m256 v00_in_range = _mm256_loadu_ps(in_bound_ptr_00);
            __m256 v01_in_range = _mm256_loadu_ps(in_bound_ptr_01);
            __m256 v10_in_range = _mm256_loadu_ps(in_bound_ptr_10);
            __m256 v11_in_range = _mm256_loadu_ps(in_bound_ptr_11);

            __m256 v00_val = mask_gather_ps256(srcptr, v00_offset, v00_in_range);
            __m256 v01_val = mask_gather_ps256(srcptr, v01_offset, v01_in_range);
            __m256 v10_val = mask_gather_ps256(srcptr, v10_offset, v10_in_range);
            __m256 v11_val = mask_gather_ps256(srcptr, v11_offset, v11_in_range);

            __m256 alpha = _mm256_loadu_ps(value_ptr_alpha);
            __m256 beta = _mm256_loadu_ps(value_ptr_beta);

            __m256 v0 = _mm256_comp_fmadd_ps(v01_val, alpha, _mm256_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m256 v1 = _mm256_comp_fmadd_ps(v11_val, alpha, _mm256_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m256 _v = _mm256_comp_fmadd_ps(v1, beta, _mm256_comp_fnmadd_ps(v0, beta, v0));
            _mm256_storeu_ps(dstptr, _v);

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

            dstptr += 8;
        }
        nn = grid_size & 7;
#endif // __AVX__
        for (int i = grid_size - nn; i + 3 < grid_size; i += 4)
        {
            __m128i v00_offset = _mm_loadu_epi32(offset_ptr_00);
            __m128i v01_offset = _mm_loadu_epi32(offset_ptr_01);
            __m128i v10_offset = _mm_loadu_epi32(offset_ptr_10);
            __m128i v11_offset = _mm_loadu_epi32(offset_ptr_11);

            __m128 v00_in_range = _mm_loadu_ps(in_bound_ptr_00);
            __m128 v01_in_range = _mm_loadu_ps(in_bound_ptr_01);
            __m128 v10_in_range = _mm_loadu_ps(in_bound_ptr_10);
            __m128 v11_in_range = _mm_loadu_ps(in_bound_ptr_11);

            __m128 v00_val = mask_gather_ps(srcptr, v00_offset, v00_in_range);
            __m128 v01_val = mask_gather_ps(srcptr, v01_offset, v01_in_range);
            __m128 v10_val = mask_gather_ps(srcptr, v10_offset, v10_in_range);
            __m128 v11_val = mask_gather_ps(srcptr, v11_offset, v11_in_range);

            __m128 alpha = _mm_loadu_ps(value_ptr_alpha);
            __m128 beta = _mm_loadu_ps(value_ptr_beta);

            __m128 v0 = _mm_comp_fmadd_ps(v01_val, alpha, _mm_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m128 v1 = _mm_comp_fmadd_ps(v11_val, alpha, _mm_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m128 _v = _mm_comp_fmadd_ps(v1, beta, _mm_comp_fnmadd_ps(v0, beta, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_00 += 4;
            offset_ptr_01 += 4;
            offset_ptr_10 += 4;
            offset_ptr_11 += 4;

            in_bound_ptr_00 += 4;
            in_bound_ptr_01 += 4;
            in_bound_ptr_10 += 4;
            in_bound_ptr_11 += 4;

            value_ptr_alpha += 4;
            value_ptr_beta += 4;

            dstptr += 4;
        }
        nn = grid_size & 3;
#endif // __SSE2__
        for (int i = grid_size - nn; i < grid_size; i++)
        {
            float v00 = *in_bound_ptr_00 < 0 ? *(srcptr + *offset_ptr_00) : 0;
            float v01 = *in_bound_ptr_01 < 0 ? *(srcptr + *offset_ptr_01) : 0;
            float v10 = *in_bound_ptr_10 < 0 ? *(srcptr + *offset_ptr_10) : 0;
            float v11 = *in_bound_ptr_11 < 0 ? *(srcptr + *offset_ptr_11) : 0;

            float v0 = v00 * (1 - *value_ptr_alpha) + v01 * *value_ptr_alpha;
            float v1 = v10 * (1 - *value_ptr_alpha) + v11 * *value_ptr_alpha;

            *dstptr = v0 * (1 - *value_ptr_beta) + v1 * *value_ptr_beta;

            in_bound_ptr_00++;
            in_bound_ptr_01++;
            in_bound_ptr_10++;
            in_bound_ptr_11++;

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;
            dstptr++;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset, const Mat& in_bound, const Mat& value, const Option& opt)
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

        const int* offset_ptr_000 = offset.channel(0);
        const int* offset_ptr_001 = offset.channel(1);
        const int* offset_ptr_010 = offset.channel(2);
        const int* offset_ptr_011 = offset.channel(3);
        const int* offset_ptr_100 = offset.channel(4);
        const int* offset_ptr_101 = offset.channel(5);
        const int* offset_ptr_110 = offset.channel(6);
        const int* offset_ptr_111 = offset.channel(7);

        const float* in_bound_ptr_000 = in_bound.channel(0);
        const float* in_bound_ptr_001 = in_bound.channel(1);
        const float* in_bound_ptr_010 = in_bound.channel(2);
        const float* in_bound_ptr_011 = in_bound.channel(3);
        const float* in_bound_ptr_100 = in_bound.channel(4);
        const float* in_bound_ptr_101 = in_bound.channel(5);
        const float* in_bound_ptr_110 = in_bound.channel(6);
        const float* in_bound_ptr_111 = in_bound.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        int nn = grid_size;
#if __SSE2__
#if __AVX__
        for (int i = 0; i + 7 < grid_size; i += 8)
        {
            __m256i v000_offset = _mm256_loadu_epi32(offset_ptr_000);
            __m256i v001_offset = _mm256_loadu_epi32(offset_ptr_001);
            __m256i v010_offset = _mm256_loadu_epi32(offset_ptr_010);
            __m256i v011_offset = _mm256_loadu_epi32(offset_ptr_011);
            __m256i v100_offset = _mm256_loadu_epi32(offset_ptr_100);
            __m256i v101_offset = _mm256_loadu_epi32(offset_ptr_101);
            __m256i v110_offset = _mm256_loadu_epi32(offset_ptr_110);
            __m256i v111_offset = _mm256_loadu_epi32(offset_ptr_111);

            __m256 v000_in_range = _mm256_loadu_ps(in_bound_ptr_000);
            __m256 v001_in_range = _mm256_loadu_ps(in_bound_ptr_001);
            __m256 v010_in_range = _mm256_loadu_ps(in_bound_ptr_010);
            __m256 v011_in_range = _mm256_loadu_ps(in_bound_ptr_011);
            __m256 v100_in_range = _mm256_loadu_ps(in_bound_ptr_100);
            __m256 v101_in_range = _mm256_loadu_ps(in_bound_ptr_101);
            __m256 v110_in_range = _mm256_loadu_ps(in_bound_ptr_110);
            __m256 v111_in_range = _mm256_loadu_ps(in_bound_ptr_111);

            __m256 v000_val = mask_gather_ps256(srcptr, v000_offset, v000_in_range);
            __m256 v001_val = mask_gather_ps256(srcptr, v001_offset, v001_in_range);
            __m256 v010_val = mask_gather_ps256(srcptr, v010_offset, v010_in_range);
            __m256 v011_val = mask_gather_ps256(srcptr, v011_offset, v011_in_range);
            __m256 v100_val = mask_gather_ps256(srcptr, v100_offset, v100_in_range);
            __m256 v101_val = mask_gather_ps256(srcptr, v101_offset, v101_in_range);
            __m256 v110_val = mask_gather_ps256(srcptr, v110_offset, v110_in_range);
            __m256 v111_val = mask_gather_ps256(srcptr, v111_offset, v111_in_range);

            __m256 alpha = _mm256_loadu_ps(value_ptr_alpha);
            __m256 beta = _mm256_loadu_ps(value_ptr_beta);
            __m256 gamma = _mm256_loadu_ps(value_ptr_gamma);

            __m256 v00 = _mm256_comp_fmadd_ps(v001_val, alpha, _mm256_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m256 v01 = _mm256_comp_fmadd_ps(v011_val, alpha, _mm256_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m256 v10 = _mm256_comp_fmadd_ps(v101_val, alpha, _mm256_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m256 v11 = _mm256_comp_fmadd_ps(v111_val, alpha, _mm256_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m256 v0 = _mm256_comp_fmadd_ps(v01, beta, _mm256_comp_fnmadd_ps(v00, beta, v00));
            __m256 v1 = _mm256_comp_fmadd_ps(v11, beta, _mm256_comp_fnmadd_ps(v10, beta, v10));

            __m256 _v = _mm256_comp_fmadd_ps(v1, gamma, _mm256_comp_fnmadd_ps(v0, gamma, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_000 += 8;
            offset_ptr_001 += 8;
            offset_ptr_010 += 8;
            offset_ptr_011 += 8;

            offset_ptr_100 += 8;
            offset_ptr_101 += 8;
            offset_ptr_110 += 8;
            offset_ptr_111 += 8;

            in_bound_ptr_000 += 8;
            in_bound_ptr_001 += 8;
            in_bound_ptr_010 += 8;
            in_bound_ptr_011 += 8;

            in_bound_ptr_100 += 8;
            in_bound_ptr_101 += 8;
            in_bound_ptr_110 += 8;
            in_bound_ptr_111 += 8;

            value_ptr_alpha += 8;
            value_ptr_beta += 8;
            value_ptr_gamma += 8;

            dstptr += 8;
        }

        nn = grid_size & 7;
#endif // __AVX__
        for (int i = grid_size - nn; i + 3 < grid_size; i += 4)
        {
            __m128i v000_offset = _mm_loadu_epi32(offset_ptr_000);
            __m128i v001_offset = _mm_loadu_epi32(offset_ptr_001);
            __m128i v010_offset = _mm_loadu_epi32(offset_ptr_010);
            __m128i v011_offset = _mm_loadu_epi32(offset_ptr_011);
            __m128i v100_offset = _mm_loadu_epi32(offset_ptr_100);
            __m128i v101_offset = _mm_loadu_epi32(offset_ptr_101);
            __m128i v110_offset = _mm_loadu_epi32(offset_ptr_110);
            __m128i v111_offset = _mm_loadu_epi32(offset_ptr_111);

            __m128 v000_in_range = _mm_loadu_ps(in_bound_ptr_000);
            __m128 v001_in_range = _mm_loadu_ps(in_bound_ptr_001);
            __m128 v010_in_range = _mm_loadu_ps(in_bound_ptr_010);
            __m128 v011_in_range = _mm_loadu_ps(in_bound_ptr_011);
            __m128 v100_in_range = _mm_loadu_ps(in_bound_ptr_100);
            __m128 v101_in_range = _mm_loadu_ps(in_bound_ptr_101);
            __m128 v110_in_range = _mm_loadu_ps(in_bound_ptr_110);
            __m128 v111_in_range = _mm_loadu_ps(in_bound_ptr_111);

            __m128 v000_val = mask_gather_ps(srcptr, v000_offset, v000_in_range);
            __m128 v001_val = mask_gather_ps(srcptr, v001_offset, v001_in_range);
            __m128 v010_val = mask_gather_ps(srcptr, v010_offset, v010_in_range);
            __m128 v011_val = mask_gather_ps(srcptr, v011_offset, v011_in_range);
            __m128 v100_val = mask_gather_ps(srcptr, v100_offset, v100_in_range);
            __m128 v101_val = mask_gather_ps(srcptr, v101_offset, v101_in_range);
            __m128 v110_val = mask_gather_ps(srcptr, v110_offset, v110_in_range);
            __m128 v111_val = mask_gather_ps(srcptr, v111_offset, v111_in_range);

            __m128 alpha = _mm_loadu_ps(value_ptr_alpha);
            __m128 beta = _mm_loadu_ps(value_ptr_beta);
            __m128 gamma = _mm_loadu_ps(value_ptr_gamma);

            __m128 v00 = _mm_comp_fmadd_ps(v001_val, alpha, _mm_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m128 v01 = _mm_comp_fmadd_ps(v011_val, alpha, _mm_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m128 v10 = _mm_comp_fmadd_ps(v101_val, alpha, _mm_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m128 v11 = _mm_comp_fmadd_ps(v111_val, alpha, _mm_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m128 v0 = _mm_comp_fmadd_ps(v01, beta, _mm_comp_fnmadd_ps(v00, beta, v00));
            __m128 v1 = _mm_comp_fmadd_ps(v11, beta, _mm_comp_fnmadd_ps(v10, beta, v10));

            __m128 _v = _mm_comp_fmadd_ps(v1, gamma, _mm_comp_fnmadd_ps(v0, gamma, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_000 += 4;
            offset_ptr_001 += 4;
            offset_ptr_010 += 4;
            offset_ptr_011 += 4;

            offset_ptr_100 += 4;
            offset_ptr_101 += 4;
            offset_ptr_110 += 4;
            offset_ptr_111 += 4;

            in_bound_ptr_000 += 4;
            in_bound_ptr_001 += 4;
            in_bound_ptr_010 += 4;
            in_bound_ptr_011 += 4;

            in_bound_ptr_100 += 4;
            in_bound_ptr_101 += 4;
            in_bound_ptr_110 += 4;
            in_bound_ptr_111 += 4;

            value_ptr_alpha += 4;
            value_ptr_beta += 4;
            value_ptr_gamma += 4;

            dstptr += 4;
        }
        nn = grid_size & 3;
#endif // __SSE2__
        for (int i = grid_size - nn; i < grid_size; i++)
        {
            float v000 = *in_bound_ptr_000 < 0 ? *(srcptr + *offset_ptr_000) : 0;
            float v001 = *in_bound_ptr_001 < 0 ? *(srcptr + *offset_ptr_001) : 0;
            float v010 = *in_bound_ptr_010 < 0 ? *(srcptr + *offset_ptr_010) : 0;
            float v011 = *in_bound_ptr_011 < 0 ? *(srcptr + *offset_ptr_011) : 0;

            float v100 = *in_bound_ptr_100 < 0 ? *(srcptr + *offset_ptr_100) : 0;
            float v101 = *in_bound_ptr_101 < 0 ? *(srcptr + *offset_ptr_101) : 0;
            float v110 = *in_bound_ptr_110 < 0 ? *(srcptr + *offset_ptr_110) : 0;
            float v111 = *in_bound_ptr_111 < 0 ? *(srcptr + *offset_ptr_111) : 0;

            float v00 = v000 * (1 - *value_ptr_alpha) + v001 * *value_ptr_alpha;
            float v01 = v010 * (1 - *value_ptr_alpha) + v011 * *value_ptr_alpha;
            float v10 = v100 * (1 - *value_ptr_alpha) + v101 * *value_ptr_alpha;
            float v11 = v110 * (1 - *value_ptr_alpha) + v111 * *value_ptr_alpha;

            float v0 = v00 * (1 - *value_ptr_beta) + v01 * *value_ptr_beta;
            float v1 = v10 * (1 - *value_ptr_beta) + v11 * *value_ptr_beta;

            *dstptr = v0 * (1 - *value_ptr_gamma) + v1 * *value_ptr_gamma;

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            in_bound_ptr_000++;
            in_bound_ptr_001++;
            in_bound_ptr_010++;
            in_bound_ptr_011++;

            in_bound_ptr_100++;
            in_bound_ptr_101++;
            in_bound_ptr_110++;
            in_bound_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;
            dstptr++;
        }
    }
}