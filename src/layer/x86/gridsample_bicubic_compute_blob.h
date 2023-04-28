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

#if __SSE2__
#if __AVX__
#if __AVX512F__
static void cubic_interp1d_p16(__m512& coeffs0, __m512& coeffs1, __m512& coeffs2, __m512& coeffs3, const __m512& tx)
{
    const __m512 A = _mm512_set1_ps(-0.75f);

    const __m512 x0 = _mm512_add_ps(tx, *(__m512*)_ps512_1);
    const __m512& x1 = tx;
    const __m512 x2 = _mm512_sub_ps(*(__m512*)_ps512_1, tx);
    //const __m512 x3 = _mm512_add_ps(x2, *(__m512*)_ps512_1);

    coeffs0 = _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(A, x0), _mm512_mul_ps(_mm512_set1_ps(5.0f), A)), x0), _mm512_mul_ps(_mm512_set1_ps(8.0f), A)), x0), _mm512_mul_ps(_mm512_set1_ps(4), A));
    coeffs1 = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(A, _mm512_set1_ps(2.0f)), x1), _mm512_add_ps(A, _mm512_set1_ps(3.0f))), x1), x1), *(__m512*)_ps512_1);
    coeffs2 = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(A, _mm512_set1_ps(2.0f)), x2), _mm512_add_ps(A, _mm512_set1_ps(3.0f))), x2), x2), *(__m512*)_ps512_1);
    coeffs3 = _mm512_sub_ps(_mm512_sub_ps(_mm512_sub_ps(*(__m512*)_ps512_1, coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p16(const Mat& src, Mat& dst, Mat& offset, const Mat& value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    __m512 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
    __m512 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
    __m512 value_f[4];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        float *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);
        }

        const float* value_x = value.channel(0);
        const float* value_y = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            cubic_interp1d_p16(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm512_set1_ps(*value_x));
            for (int ii = 0; ii < 4; ii++)
            {
                __m512 x0_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(v0_offset_ptr[ii]) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), _mm512_add_epi32(_mm512_set1_epi32(*v0_offset_ptr[ii]), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)), srcptr, sizeof(float));
                __m512 x1_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(v1_offset_ptr[ii]) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), _mm512_add_epi32(_mm512_set1_epi32(*v1_offset_ptr[ii]), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)), srcptr, sizeof(float));
                __m512 x2_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(v2_offset_ptr[ii]) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), _mm512_add_epi32(_mm512_set1_epi32(*v2_offset_ptr[ii]), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)), srcptr, sizeof(float));
                __m512 x3_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(v3_offset_ptr[ii]) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), _mm512_add_epi32(_mm512_set1_epi32(*v3_offset_ptr[ii]), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)), srcptr, sizeof(float));

                value_f[ii] = _mm512_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm512_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm512_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm512_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                v0_offset_ptr[ii]++;
                v1_offset_ptr[ii]++;
                v2_offset_ptr[ii]++;
                v3_offset_ptr[ii]++;
            }

            cubic_interp1d_p16(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm512_set1_ps(*value_y));

            __m512 _v = _mm512_mul_ps(y_coeffs0, value_f[0]);
            _v = _mm512_fmadd_ps(y_coeffs1, value_f[1], _v);
            _v = _mm512_fmadd_ps(y_coeffs2, value_f[2], _v);
            _v = _mm512_fmadd_ps(y_coeffs3, value_f[3], _v);
            _mm512_storeu_ps(dstptr, _v);

            value_x++;
            value_y++;

            dstptr += 16;
        }
    }
}
#endif // __AVX512F__
static void cubic_interp1d_p8(__m256& coeffs0, __m256& coeffs1, __m256& coeffs2, __m256& coeffs3, const __m256& tx)
{
    const __m256 A = _mm256_set1_ps(-0.75f);

    const __m256 x0 = _mm256_add_ps(tx, _mm256_set1_ps(1));
    const __m256& x1 = tx;
    const __m256 x2 = _mm256_sub_ps(_mm256_set1_ps(1), tx);
    //const __m256 x3 = _mm256_add_ps(x2, _mm256_set1_ps(1));

    coeffs0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(A, x0), _mm256_mul_ps(_mm256_set1_ps(5.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(8.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(4), A));
    coeffs1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x1), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x1), x1), _mm256_set1_ps(1));
    coeffs2 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x2), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x2), x2), _mm256_set1_ps(1));
    coeffs3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(1), coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p8(const Mat& src, Mat& dst, Mat& offset, const Mat& value, const Option& opt)
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

        float *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);
        }

        const float* value_x = value.channel(0);
        const float* value_y = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            cubic_interp1d_p8(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm256_set1_ps(*value_x));
            for (int ii = 0; ii < 4; ii++)
            {
                float v0_in_bound = *reinterpret_cast<const int*>(v0_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;
                float v1_in_bound = *reinterpret_cast<const int*>(v1_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;
                float v2_in_bound = *reinterpret_cast<const int*>(v2_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;
                float v3_in_bound = *reinterpret_cast<const int*>(v3_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;

#if __AVX2__
                __m256i v0_offset = _mm256_add_epi32(_mm256_set1_epi32(*v0_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                __m256i v1_offset = _mm256_add_epi32(_mm256_set1_epi32(*v1_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                __m256i v2_offset = _mm256_add_epi32(_mm256_set1_epi32(*v2_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                __m256i v3_offset = _mm256_add_epi32(_mm256_set1_epi32(*v3_offset_ptr[ii]), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
                __m256i v0_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*v0_offset_ptr[ii]), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
                __m256i v1_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*v1_offset_ptr[ii]), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
                __m256i v2_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*v2_offset_ptr[ii]), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
                __m256i v3_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*v3_offset_ptr[ii]), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__

                __m256 x0_val = mask_gather_ps256(srcptr, v0_offset, _mm256_set1_ps(v0_in_bound));
                __m256 x1_val = mask_gather_ps256(srcptr, v1_offset, _mm256_set1_ps(v1_in_bound));
                __m256 x2_val = mask_gather_ps256(srcptr, v2_offset, _mm256_set1_ps(v2_in_bound));
                __m256 x3_val = mask_gather_ps256(srcptr, v3_offset, _mm256_set1_ps(v3_in_bound));

                value_f[ii] = _mm256_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                v0_offset_ptr[ii]++;
                v1_offset_ptr[ii]++;
                v2_offset_ptr[ii]++;
                v3_offset_ptr[ii]++;
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
static void cubic_interp1d_p4(__m128& coeffs0, __m128& coeffs1, __m128& coeffs2, __m128& coeffs3, const __m128& tx)
{
    const __m128 A = _mm_set1_ps(-0.75f);

    const __m128 x0 = _mm_add_ps(tx, *(__m128*)_ps_1);
    const __m128& x1 = tx;
    const __m128 x2 = _mm_sub_ps(*(__m128*)_ps_1, tx);
    //const __m128 x3 = _mm_add_ps(x2, *(__m128*)_ps_1);

    coeffs0 = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(A, x0), _mm_mul_ps(_mm_set1_ps(5.0f), A)), x0), _mm_mul_ps(_mm_set1_ps(8.0f), A)), x0), _mm_mul_ps(_mm_set1_ps(4), A));
    coeffs1 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set1_ps(2.0f)), x1), _mm_add_ps(A, _mm_set1_ps(3.0f))), x1), x1), *(__m128*)_ps_1);
    coeffs2 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set1_ps(2.0f)), x2), _mm_add_ps(A, _mm_set1_ps(3.0f))), x2), x2), *(__m128*)_ps_1);
    coeffs3 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(*(__m128*)_ps_1, coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p4(const Mat& src, Mat& dst, Mat& offset, const Mat& value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    __m128 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
    __m128 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
    __m128 value_f[4];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        float *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);
        }

        const float* value_x = value.channel(0);
        const float* value_y = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            cubic_interp1d_p4(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm_set1_ps(*value_x));
            for (int ii = 0; ii < 4; ii++)
            {
                float v0_in_bound = *reinterpret_cast<const int*>(v0_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;
                float v1_in_bound = *reinterpret_cast<const int*>(v1_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;
                float v2_in_bound = *reinterpret_cast<const int*>(v2_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;
                float v3_in_bound = *reinterpret_cast<const int*>(v3_offset_ptr[ii]) >= 0 ? -1.0f : 0.0f;

                __m128 x0_val = mask_gather_ps(srcptr, _mm_add_epi32(_mm_set1_epi32(*v0_offset_ptr[ii]), _mm_set_epi32(3, 2, 1, 0)), _mm_set1_ps(v0_in_bound));
                __m128 x1_val = mask_gather_ps(srcptr, _mm_add_epi32(_mm_set1_epi32(*v1_offset_ptr[ii]), _mm_set_epi32(3, 2, 1, 0)), _mm_set1_ps(v1_in_bound));
                __m128 x2_val = mask_gather_ps(srcptr, _mm_add_epi32(_mm_set1_epi32(*v2_offset_ptr[ii]), _mm_set_epi32(3, 2, 1, 0)), _mm_set1_ps(v2_in_bound));
                __m128 x3_val = mask_gather_ps(srcptr, _mm_add_epi32(_mm_set1_epi32(*v3_offset_ptr[ii]), _mm_set_epi32(3, 2, 1, 0)), _mm_set1_ps(v3_in_bound));

                value_f[ii] = _mm_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                v0_offset_ptr[ii]++;
                v1_offset_ptr[ii]++;
                v2_offset_ptr[ii]++;
                v3_offset_ptr[ii]++;
            }

            cubic_interp1d_p4(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm_set1_ps(*value_y));

            __m128 _v = _mm_mul_ps(y_coeffs0, value_f[0]);
            _v = _mm_comp_fmadd_ps(y_coeffs1, value_f[1], _v);
            _v = _mm_comp_fmadd_ps(y_coeffs2, value_f[2], _v);
            _v = _mm_comp_fmadd_ps(y_coeffs3, value_f[3], _v);
            _mm_storeu_ps(dstptr, _v);

            value_x++;
            value_y++;

            dstptr += 4;
        }
    }
}
#endif // __SSE2__

static inline void cubic_interp1d(float& coeffs0, float& coeffs1, float& coeffs2, float& coeffs3, float fx)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs0 = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs1 = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs2 = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs3 = 1.f - coeffs0 - coeffs1 - coeffs2;
}

static void gridsample_2d_bicubic_apply_interpolation_p1(const Mat& src, Mat& dst, Mat& offset, const Mat& value, const Option& opt)
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

        float *v0_offset_ptr[4], *v1_offset_ptr[4], *v2_offset_ptr[4], *v3_offset_ptr[4];

        for (int i = 0; i < 4; i++)
        {
            v0_offset_ptr[i] = offset.channel(i * 4 + 0);
            v1_offset_ptr[i] = offset.channel(i * 4 + 1);
            v2_offset_ptr[i] = offset.channel(i * 4 + 2);
            v3_offset_ptr[i] = offset.channel(i * 4 + 3);
        }

        const float* value_x = value.channel(0);
        const float* value_y = value.channel(1);

        int x = 0;
#if __SSE2__
#if __AVX__
        {
            __m256 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            __m256 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            __m256 value_f[4];
            for (; x + 7 < grid_size; x += 8)
            {
                cubic_interp1d_p8(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm256_loadu_ps(value_x));
                for (int ii = 0; ii < 4; ii++)
                {
                    __m256 v0_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(v0_offset_ptr[ii]), _mm256_set1_ps(-1.0f));
                    __m256 v1_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(v1_offset_ptr[ii]), _mm256_set1_ps(-1.0f));
                    __m256 v2_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(v2_offset_ptr[ii]), _mm256_set1_ps(-1.0f));
                    __m256 v3_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(v3_offset_ptr[ii]), _mm256_set1_ps(-1.0f));

                    __m256 x0_val = mask_gather_ps256(srcptr, _mm256_set_epi32(*(v0_offset_ptr[ii] + 7), *(v0_offset_ptr[ii] + 6), *(v0_offset_ptr[ii] + 5), *(v0_offset_ptr[ii] + 4), *(v0_offset_ptr[ii] + 3), *(v0_offset_ptr[ii] + 2), *(v0_offset_ptr[ii] + 1), *v0_offset_ptr[ii]), v0_in_bound);
                    __m256 x1_val = mask_gather_ps256(srcptr, _mm256_set_epi32(*(v1_offset_ptr[ii] + 7), *(v1_offset_ptr[ii] + 6), *(v1_offset_ptr[ii] + 5), *(v1_offset_ptr[ii] + 4), *(v1_offset_ptr[ii] + 3), *(v1_offset_ptr[ii] + 2), *(v1_offset_ptr[ii] + 1), *v1_offset_ptr[ii]), v1_in_bound);
                    __m256 x2_val = mask_gather_ps256(srcptr, _mm256_set_epi32(*(v2_offset_ptr[ii] + 7), *(v2_offset_ptr[ii] + 6), *(v2_offset_ptr[ii] + 5), *(v2_offset_ptr[ii] + 4), *(v2_offset_ptr[ii] + 3), *(v2_offset_ptr[ii] + 2), *(v2_offset_ptr[ii] + 1), *v2_offset_ptr[ii]), v2_in_bound);
                    __m256 x3_val = mask_gather_ps256(srcptr, _mm256_set_epi32(*(v3_offset_ptr[ii] + 7), *(v3_offset_ptr[ii] + 6), *(v3_offset_ptr[ii] + 5), *(v3_offset_ptr[ii] + 4), *(v3_offset_ptr[ii] + 3), *(v3_offset_ptr[ii] + 2), *(v3_offset_ptr[ii] + 1), *v3_offset_ptr[ii]), v3_in_bound);

                    value_f[ii] = _mm256_mul_ps(x_coeffs0, x0_val);
                    value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                    value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                    value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                    v0_offset_ptr[ii] += 8;
                    v1_offset_ptr[ii] += 8;
                    v2_offset_ptr[ii] += 8;
                    v3_offset_ptr[ii] += 8;
                }

                cubic_interp1d_p8(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm256_loadu_ps(value_y));

                __m256 _v = _mm256_mul_ps(y_coeffs0, value_f[0]);
                _v = _mm256_comp_fmadd_ps(y_coeffs1, value_f[1], _v);
                _v = _mm256_comp_fmadd_ps(y_coeffs2, value_f[2], _v);
                _v = _mm256_comp_fmadd_ps(y_coeffs3, value_f[3], _v);
                _mm256_storeu_ps(dstptr, _v);

                value_x += 8;
                value_y += 8;

                dstptr += 8;
            }
        }
#endif // __AVX__
        {
            __m128 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            __m128 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            __m128 value_f[4];
            for (; x + 3 < grid_size; x += 4)
            {
                cubic_interp1d_p4(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm_loadu_ps(value_x));
                for (int ii = 0; ii < 4; ii++)
                {
                    __m128 v0_in_bound = _mm_andnot_ps(_mm_loadu_ps(v0_offset_ptr[ii]), _mm_set1_ps(-1.0f));
                    __m128 v1_in_bound = _mm_andnot_ps(_mm_loadu_ps(v1_offset_ptr[ii]), _mm_set1_ps(-1.0f));
                    __m128 v2_in_bound = _mm_andnot_ps(_mm_loadu_ps(v2_offset_ptr[ii]), _mm_set1_ps(-1.0f));
                    __m128 v3_in_bound = _mm_andnot_ps(_mm_loadu_ps(v3_offset_ptr[ii]), _mm_set1_ps(-1.0f));

                    __m128 x0_val = mask_gather_ps(srcptr, _mm_set_epi32(*(v0_offset_ptr[ii] + 3), *(v0_offset_ptr[ii] + 2), *(v0_offset_ptr[ii] + 1), *v0_offset_ptr[ii]), v0_in_bound);
                    __m128 x1_val = mask_gather_ps(srcptr, _mm_set_epi32(*(v1_offset_ptr[ii] + 3), *(v1_offset_ptr[ii] + 2), *(v1_offset_ptr[ii] + 1), *v1_offset_ptr[ii]), v1_in_bound);
                    __m128 x2_val = mask_gather_ps(srcptr, _mm_set_epi32(*(v2_offset_ptr[ii] + 3), *(v2_offset_ptr[ii] + 2), *(v2_offset_ptr[ii] + 1), *v2_offset_ptr[ii]), v2_in_bound);
                    __m128 x3_val = mask_gather_ps(srcptr, _mm_set_epi32(*(v3_offset_ptr[ii] + 3), *(v3_offset_ptr[ii] + 2), *(v3_offset_ptr[ii] + 1), *v3_offset_ptr[ii]), v3_in_bound);

                    value_f[ii] = _mm_mul_ps(x_coeffs0, x0_val);
                    value_f[ii] = _mm_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                    value_f[ii] = _mm_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                    value_f[ii] = _mm_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                    v0_offset_ptr[ii] += 4;
                    v1_offset_ptr[ii] += 4;
                    v2_offset_ptr[ii] += 4;
                    v3_offset_ptr[ii] += 4;
                }

                cubic_interp1d_p4(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm_loadu_ps(value_y));

                __m128 _v = _mm_mul_ps(y_coeffs0, value_f[0]);
                _v = _mm_comp_fmadd_ps(y_coeffs1, value_f[1], _v);
                _v = _mm_comp_fmadd_ps(y_coeffs2, value_f[2], _v);
                _v = _mm_comp_fmadd_ps(y_coeffs3, value_f[3], _v);
                _mm_storeu_ps(dstptr, _v);

                value_x += 4;
                value_y += 4;

                dstptr += 4;
            }
        }
#endif // __SSE2__
        float x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
        float y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
        float value_f[4];

        for (; x < grid_size; x++)
        {
            cubic_interp1d(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, *value_x);
            for (int ii = 0; ii < 4; ii++)
            {
                float x0_val = *reinterpret_cast<const int*>(v0_offset_ptr[ii]) >= 0 ? *(srcptr + static_cast<int>(*v0_offset_ptr[ii])) : 0;
                float x1_val = *reinterpret_cast<const int*>(v1_offset_ptr[ii]) >= 0 ? *(srcptr + static_cast<int>(*v1_offset_ptr[ii])) : 0;
                float x2_val = *reinterpret_cast<const int*>(v2_offset_ptr[ii]) >= 0 ? *(srcptr + static_cast<int>(*v2_offset_ptr[ii])) : 0;
                float x3_val = *reinterpret_cast<const int*>(v3_offset_ptr[ii]) >= 0 ? *(srcptr + static_cast<int>(*v3_offset_ptr[ii])) : 0;

                value_f[ii] = x_coeffs0 * x0_val;
                value_f[ii] = x_coeffs1 * x1_val + value_f[ii];
                value_f[ii] = x_coeffs2 * x2_val + value_f[ii];
                value_f[ii] = x_coeffs3 * x3_val + value_f[ii];

                v0_offset_ptr[ii]++;
                v1_offset_ptr[ii]++;
                v2_offset_ptr[ii]++;
                v3_offset_ptr[ii]++;
            }

            cubic_interp1d(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, *value_y);

            float _v = y_coeffs0 * value_f[0];
            _v = y_coeffs1 * value_f[1] + _v;
            _v = y_coeffs2 * value_f[2] + _v;
            _v = y_coeffs3 * value_f[3] + _v;
            *dstptr = _v;

            value_x++;
            value_y++;

            dstptr++;
        }
    }
}