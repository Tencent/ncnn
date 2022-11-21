// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gridsample_x86.h"

#if __SSE2__
#include <smmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

GridSample_x86::GridSample_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

#if __SSE2__
#if __AVX__
const __m256 v1fp8 = *(__m256*)_ps256_1;
const auto vn1fp8 = _mm256_set1_ps(-1.0f);
const auto v1ip8 = _mm256_set1_epi32(1);
const auto vn1ip8 = _mm256_set1_epi32(-1);

#include "gridsample_bilinear_pack8.h"

static __m256 NCNN_FORCEINLINE
grid_sample_unormalize_p8(const __m256& w, const __m256& coordx, int align_corner)
{
    __m256 two = _mm256_set1_ps(2.f);

    if (align_corner)
        return _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(coordx, v1fp8), two), _mm256_sub_ps(w, v1fp8));
    else
        return _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(coordx, v1fp8), w, v1fp8), two);
}

static NCNN_FORCEINLINE __m256 border_coord_p8(const __m256& coord, const __m256& border)
{
    return _mm256_min_ps(border, _mm256_max_ps(coord, _mm256_setzero_ps()));
}

static NCNN_FORCEINLINE __m256 reflect_coord_p8(__m256 x, const __m256& high)
{
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);

    __m256 reflect_v = _mm256_and_ps(_mm256_sub_ps(x, high), *(__m256*)_ps256_inv_sign_mask);
    x = _mm256_sub_ps(high, reflect_v);
    return x;
}

static NCNN_FORCEINLINE __m256 compute_coord_p8(__m256 sx, const __m256& w, int padding_mode, int align_corner)
{
    if (padding_mode == 2) // border
    {
        sx = border_coord_p8(sx, _mm256_sub_ps(w, v1fp8));
    }
    else if (padding_mode == 3) // reflection
    {
        if (align_corner)
        {
            sx = reflect_coord_p8(sx, _mm256_sub_ps(w, v1fp8));
        }
        else
        {
            __m256 v0p5f = _mm256_set1_ps(0.5f);
            sx = _mm256_sub_ps(reflect_coord_p8(_mm256_add_ps(sx, v0p5f), w), v0p5f);
            sx = border_coord_p8(sx, _mm256_sub_ps(w, v1fp8));
        }
    }

    return sx;
}

static NCNN_FORCEINLINE __m256 get_coord_p8(const __m256& x, const __m256& w, int padding_mode, int align_corner)
{
    // compute the origin coordinates
    __m256 sx = grid_sample_unormalize_p8(w, x, align_corner);

    // correct the coordinates according to the padding_mode
    __m256 coord = compute_coord_p8(sx, w, padding_mode, align_corner);

    return coord;
}

static NCNN_FORCEINLINE __m256 cubic_interp1d_p8(const __m256& x0_v, const __m256& x1_v, const __m256& x2_v, const __m256& x3_v, const __m256& tx)
{
    const auto A = _mm256_set1_ps(-0.75f);

    const auto x0 = _mm256_add_ps(tx, v1fp8);
    const auto& x1 = tx;
    const auto x2 = _mm256_sub_ps(v1fp8, tx);
    //const auto x3 = _mm256_add_ps(x2, v1fp8);

    const __m256 coeffs0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(A, x0), _mm256_mul_ps(_mm256_set1_ps(5.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(8.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(4), A));
    const __m256 coeffs1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x1), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x1), x1), v1fp8);
    const __m256 coeffs2 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x2), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x2), x2), v1fp8);
    const __m256 coeffs3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(v1fp8, coeffs0), coeffs1), coeffs2);

    auto _v = _mm256_mul_ps(coeffs0, x0_v);
    _v = _mm256_comp_fmadd_ps(coeffs1, x1_v, _v);
    _v = _mm256_comp_fmadd_ps(coeffs2, x2_v, _v);
    _v = _mm256_comp_fmadd_ps(coeffs3, x3_v, _v);

    return _v;
}

#endif // __AVX__

const __m128 v1fp4 = _mm_set1_ps(1.0f);

static __m128 NCNN_FORCEINLINE
grid_sample_unormalize_p4(const __m128& w, const __m128& coordx, int align_corner)
{
    __m128 two = _mm_set1_ps(2.f);

    if (align_corner)
        return _mm_mul_ps(_mm_div_ps(_mm_add_ps(coordx, v1fp4), two), _mm_sub_ps(w, v1fp4));
    else
        return _mm_div_ps(_mm_comp_fmsub_ps(_mm_add_ps(coordx, v1fp4), w, v1fp4), two);
}

static NCNN_FORCEINLINE __m128 border_coord_p4(const __m128& coord, const __m128& border)
{
    return _mm_min_ps(border, _mm_max_ps(coord, _mm_setzero_ps()));
}

static NCNN_FORCEINLINE __m128 reflect_coord_p4(__m128 x, const __m128& high)
{
    /* take the absolute value */
    x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);

    __m128 reflect_v = _mm_and_ps(_mm_sub_ps(x, high), *(__m128*)_ps_inv_sign_mask);
    x = _mm_sub_ps(high, reflect_v);
    return x;
}

static NCNN_FORCEINLINE __m128 compute_coord_p4(__m128 sx, const __m128& w, int padding_mode, int align_corner)
{
    if (padding_mode == 2) // border
    {
        sx = border_coord_p4(sx, _mm_sub_ps(w, v1fp4));
    }
    else if (padding_mode == 3) // reflection
    {
        if (align_corner)
        {
            sx = reflect_coord_p4(sx, _mm_sub_ps(w, v1fp4));
        }
        else
        {
            __m128 v0p5f = *(__m128*)_ps_0p5;
            sx = _mm_sub_ps(reflect_coord_p4(_mm_add_ps(sx, v0p5f), w), v0p5f);
            sx = border_coord_p4(sx, _mm_sub_ps(w, v1fp4));
        }
    }

    return sx;
}

static NCNN_FORCEINLINE __m128 get_coord_p4(const __m128& x, const __m128& w, int padding_mode, int align_corner)
{
    // compute the origin coordinates
    __m128 sx = grid_sample_unormalize_p4(w, x, align_corner);

    // correct the coordinates according to the padding_mode
    __m128 coord = compute_coord_p4(sx, w, padding_mode, align_corner);

    return coord;
}

static NCNN_FORCEINLINE __m128 cubic_interp1d_p4(const __m128& x0_v, const __m128& x1_v, const __m128& x2_v, const __m128& x3_v, const __m128& tx)
{
    const auto A = _mm_set1_ps(-0.75f);

    const auto x0 = _mm_add_ps(tx, v1fp4);
    const auto& x1 = tx;
    const auto x2 = _mm_sub_ps(v1fp4, tx);
    //const auto x3 = _mm_add_ps(x2, v1fp4);

    const __m128 coeffs0 = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(A, x0), _mm_mul_ps(_mm_set1_ps(5.0f), A)), x0), _mm_mul_ps(_mm_set1_ps(8.0f), A)), x0), _mm_mul_ps(_mm_set1_ps(4), A));
    const __m128 coeffs1 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set1_ps(2.0f)), x1), _mm_add_ps(A, _mm_set1_ps(3.0f))), x1), x1), v1fp4);
    const __m128 coeffs2 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set1_ps(2.0f)), x2), _mm_add_ps(A, _mm_set1_ps(3.0f))), x2), x2), v1fp4);
    const __m128 coeffs3 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(v1fp4, coeffs0), coeffs1), coeffs2);

    auto _v = _mm_mul_ps(coeffs0, x0_v);
    _v = _mm_comp_fmadd_ps(coeffs1, x1_v, _v);
    _v = _mm_comp_fmadd_ps(coeffs2, x2_v, _v);
    _v = _mm_comp_fmadd_ps(coeffs3, x3_v, _v);

    return _v;
}

static NCNN_FORCEINLINE __m128 mask_gather_ps(const float* ptr, __m128i offset, __m128 mask)
{
#if __AVX__
    __m128 v = _mm_mask_i32gather_ps(_mm_setzero_ps(), ptr, offset, mask, sizeof(float));
#else
    int offseti[4], maski[4];
    memcpy(offseti, &offset, 4 * sizeof(int));
    memcpy(maski, &mask, 4 * sizeof(int));

    float data[4];
    for (int i = 0; i < 4; i++)
    {
        if (maski[i] & 0x01)
        {
            data[i] = *(ptr + offseti[i]);
        }
    }

    __m128 v = _mm_loadu_ps(data);
#endif // __AVX__

    return v;
}

#endif // __SSE2__

int GridSample_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];
    const int elempack = bottom_blob.elempack;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

#if __SSE2__
#if __AVX__

    if (elempack == 8)
    {
        const auto vImgWf = _mm256_set1_ps(w);
        const auto vImgHf = _mm256_set1_ps(h);
        const auto vImgWi = _mm256_set1_epi32(w);
        const auto vImgHi = _mm256_set1_epi32(h);

        const auto vElemsizei = _mm256_set1_epi32(elemsize / 8);
        const auto vElempacki = _mm256_set1_epi32(elempack);
        const auto vElempackf = _mm256_set1_ps(elempack);

        if (dims == 3)
        {
            const auto outW = grid.h;
            const auto outH = grid.c * grid.elempack;

            top_blob.create(outW, outH, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1) //zeros
            {
                if (padding_mode == 1) //zeros
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        Mat dst = top_blob.channel(q);
                        const Mat image = bottom_blob.channel(q);

                        gridsample_bilinear_image_pack8(image, dst, grid, padding_mode, align_corner);
                    }
                }
                else //border reflection
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm256_set1_ps(gridptr[0]);
                                auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);

                                auto x_w = _mm256_floor_ps(gx);
                                auto y_n = _mm256_floor_ps(gy);

                                auto w = _mm256_sub_ps(gx, x_w);
                                auto e = _mm256_sub_ps(v1fp8, w);
                                auto n = _mm256_sub_ps(gy, y_n);
                                auto s = _mm256_sub_ps(v1fp8, n);

                                auto nw = _mm256_mul_ps(s, e);
                                auto ne = _mm256_mul_ps(s, w);
                                auto sw = _mm256_mul_ps(n, e);
                                auto se = _mm256_mul_ps(n, w);

                                auto x0 = _mm256_cvtps_epi32(x_w);
                                auto x1 = _mm256_add_epi32(x0, v1ip8);
                                auto y0 = _mm256_cvtps_epi32(y_n);
                                auto y1 = _mm256_add_epi32(y0, v1ip8);

                                auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                                auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y1));

                                auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                auto i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0), vElempacki),
                                    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                                auto i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                                auto i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                                auto i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                                auto nw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_nw_offset, vn1fp8, sizeof(float));
                                auto ne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_ne_offset, *reinterpret_cast<__m256*>(&x1_in_range), sizeof(float));
                                auto sw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_sw_offset, *reinterpret_cast<__m256*>(&y1_in_range), sizeof(float));
                                auto se_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_se_offset, *reinterpret_cast<__m256*>(&v11_in_range), sizeof(float));

                                auto _v = _mm256_mul_ps(nw_val, nw);
                                _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                _mm256_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1) //zeros
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm256_set1_ps(gridptr[0]);
                                auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                auto ix = _mm256_cvtps_epi32(gx);
                                auto iy = _mm256_cvtps_epi32(gy);

                                auto v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                    _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));

                                auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                    i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                                _mm256_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
                else //border reflection
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm256_set1_ps(gridptr[0]);
                                auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                                gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                gx = compute_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                gy = compute_coord_p8(gy, vImgHf, padding_mode, align_corner);

                                auto ix = _mm256_cvtps_epi32(gx);
                                auto iy = _mm256_cvtps_epi32(gy);

                                auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix), vElempacki),
                                    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                    i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                                _mm256_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
            }

            if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm256_set1_ps(gridptr[0]);
                                auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                                gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);

                                auto gx_floor = _mm256_floor_ps(gx);
                                auto gy_floor = _mm256_floor_ps(gy);

                                const auto tx = _mm256_sub_ps(gx, gx_floor);
                                const auto ty = _mm256_sub_ps(gy, gy_floor);

                                __m256 coefficients[4];

                                for (int i = 0; i < 4; i++)
                                {
                                    auto gx0 = compute_coord_p8(_mm256_add_ps(gx_floor, vn1fp8), vImgWf, padding_mode, align_corner);
                                    auto gx1 = compute_coord_p8(gx_floor, vImgWf, padding_mode, align_corner);
                                    auto gx2 = compute_coord_p8(_mm256_add_ps(gx_floor, v1fp8), vImgWf, padding_mode, align_corner);
                                    auto gx3 = compute_coord_p8(_mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f)), vImgWf, padding_mode, align_corner);

                                    gy = compute_coord_p8(_mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i)), vImgHf, padding_mode, align_corner);

                                    auto x0 = _mm256_cvtps_epi32(gx0);
                                    auto x1 = _mm256_cvtps_epi32(gx1);
                                    auto x2 = _mm256_cvtps_epi32(gx2);
                                    auto x3 = _mm256_cvtps_epi32(gx3);

                                    auto y = _mm256_cvtps_epi32(gy);

                                    auto x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x0));
                                    auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                                    auto x2_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x2, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x2));
                                    auto x3_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x3, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x3));

                                    auto y_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y));

                                    auto v0_in_range = _mm256_and_si256(x0_in_range, y_in_range);
                                    auto v1_in_range = _mm256_and_si256(x1_in_range, y_in_range);
                                    auto v2_in_range = _mm256_and_si256(x2_in_range, y_in_range);
                                    auto v3_in_range = _mm256_and_si256(x3_in_range, y_in_range);

                                    auto x0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                                    auto x0_offset = _mm256_cvtps_epi32(x0_offset_f);
                                    auto x1_offset = _mm256_cvtps_epi32(x1_offset_f);
                                    auto x2_offset = _mm256_cvtps_epi32(x2_offset_f);
                                    auto x3_offset = _mm256_cvtps_epi32(x3_offset_f);

                                    auto x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x0_offset, *reinterpret_cast<__m256*>(&v0_in_range), sizeof(float));
                                    auto x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x1_offset, *reinterpret_cast<__m256*>(&v1_in_range), sizeof(float));
                                    auto x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x2_offset, *reinterpret_cast<__m256*>(&v2_in_range), sizeof(float));
                                    auto x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x3_offset, *reinterpret_cast<__m256*>(&v3_in_range), sizeof(float));

                                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                }

                                auto _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                _mm256_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm256_set1_ps(gridptr[0]);
                                auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                                gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);

                                auto gx_floor = _mm256_floor_ps(gx);
                                auto gy_floor = _mm256_floor_ps(gy);

                                const auto tx = _mm256_sub_ps(gx, gx_floor);
                                const auto ty = _mm256_sub_ps(gy, gy_floor);

                                __m256 coefficients[4];

                                for (int i = 0; i < 4; i++)
                                {
                                    auto gx0 = compute_coord_p8(_mm256_add_ps(gx_floor, vn1fp8), vImgWf, padding_mode, align_corner);
                                    auto gx1 = compute_coord_p8(gx_floor, vImgWf, padding_mode, align_corner);
                                    auto gx2 = compute_coord_p8(_mm256_add_ps(gx_floor, v1fp8), vImgWf, padding_mode, align_corner);
                                    auto gx3 = compute_coord_p8(_mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f)), vImgWf, padding_mode, align_corner);

                                    gy = compute_coord_p8(_mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i)), vImgHf, padding_mode, align_corner);

                                    auto x0 = _mm256_cvtps_epi32(gx0);
                                    auto x1 = _mm256_cvtps_epi32(gx1);
                                    auto x2 = _mm256_cvtps_epi32(gx2);
                                    auto x3 = _mm256_cvtps_epi32(gx3);

                                    auto y = _mm256_cvtps_epi32(gy);

                                    auto x0_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x1_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x2_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x3_offset_f = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3), vElempackf),
                                        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));

                                    auto x0_offset = _mm256_cvtps_epi32(x0_offset_f);
                                    auto x1_offset = _mm256_cvtps_epi32(x1_offset_f);
                                    auto x2_offset = _mm256_cvtps_epi32(x2_offset_f);
                                    auto x3_offset = _mm256_cvtps_epi32(x3_offset_f);

                                    auto x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x0_offset, vn1fp8, sizeof(float));
                                    auto x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x1_offset, vn1fp8, sizeof(float));
                                    auto x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x2_offset, vn1fp8, sizeof(float));
                                    auto x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x3_offset, vn1fp8, sizeof(float));

                                    coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                }

                                auto _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                _mm256_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
            }
        }

        if (dims == 4)
        {
            const int outW = grid.h;
            const int outH = grid.d;
            const int outD = grid.c * grid.elempack;

            top_blob.create(outW, outH, outD, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const auto vImgDf = _mm256_set1_ps(d);
            const auto vImgDi = _mm256_set1_epi32(d);

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm256_set1_ps(gridptr[0]);
                                    auto gy = _mm256_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                    gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);
                                    gz = get_coord_p8(gz, vImgDf, padding_mode, align_corner);

                                    auto x_w = _mm256_floor_ps(gx);
                                    auto y_n = _mm256_floor_ps(gy);
                                    auto z_t = _mm256_floor_ps(gz);

                                    auto w = _mm256_sub_ps(gx, x_w);
                                    auto e = _mm256_sub_ps(v1fp8, w);
                                    auto n = _mm256_sub_ps(gy, y_n);
                                    auto s = _mm256_sub_ps(v1fp8, n);
                                    auto t = _mm256_sub_ps(gz, z_t);
                                    auto b = _mm256_sub_ps(v1fp8, t);

                                    __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                    {
                                        auto nw = _mm256_mul_ps(s, e);
                                        auto ne = _mm256_mul_ps(s, w);
                                        auto sw = _mm256_mul_ps(n, e);
                                        auto se = _mm256_mul_ps(n, w);

                                        tnw = _mm256_mul_ps(b, nw);
                                        tne = _mm256_mul_ps(b, ne);
                                        tsw = _mm256_mul_ps(b, sw);
                                        tse = _mm256_mul_ps(b, se);

                                        bnw = _mm256_mul_ps(t, nw);
                                        bne = _mm256_mul_ps(t, ne);
                                        bsw = _mm256_mul_ps(t, sw);
                                        bse = _mm256_mul_ps(t, se);
                                    }

                                    auto x0 = _mm256_cvtps_epi32(x_w);
                                    auto x1 = _mm256_add_epi32(x0, v1ip8);
                                    auto y0 = _mm256_cvtps_epi32(y_n);
                                    auto y1 = _mm256_add_epi32(y0, v1ip8);
                                    auto z0 = _mm256_cvtps_epi32(z_t);
                                    auto z1 = _mm256_add_epi32(z0, v1ip8);

                                    auto x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x0));
                                    auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                                    auto y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y0));
                                    auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y1));
                                    auto z0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z0, vn1ip8), _mm256_cmpgt_epi32(vImgDi, z0));
                                    auto z1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z1, vn1ip8), _mm256_cmpgt_epi32(vImgDi, z1));

                                    __m256i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                                    {
                                        auto v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                                        auto v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                                        auto v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                                        auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                        v000_in_range = _mm256_and_si256(v00_in_range, z0_in_range);
                                        v010_in_range = _mm256_and_si256(v01_in_range, z0_in_range);
                                        v100_in_range = _mm256_and_si256(v10_in_range, z0_in_range);
                                        v110_in_range = _mm256_and_si256(v11_in_range, z0_in_range);

                                        v001_in_range = _mm256_and_si256(v00_in_range, z1_in_range);
                                        v011_in_range = _mm256_and_si256(v01_in_range, z1_in_range);
                                        v101_in_range = _mm256_and_si256(v10_in_range, z1_in_range);
                                        v111_in_range = _mm256_and_si256(v11_in_range, z1_in_range);
                                    }

                                    // (W*H*z + W*y + x) * elempack + vec(8)
                                    auto i_tnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                                    auto i_tne_offset = _mm256_add_epi32(i_tnw_offset, vElempacki);
                                    auto i_tsw_offset = _mm256_add_epi32(i_tnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                                    auto i_tse_offset = _mm256_add_epi32(i_tsw_offset, vElempacki);

                                    auto i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                                    auto i_bne_offset = _mm256_add_epi32(i_bnw_offset, vElempacki);
                                    auto i_bsw_offset = _mm256_add_epi32(i_bnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                                    auto i_bse_offset = _mm256_add_epi32(i_bsw_offset, vElempacki);

                                    auto tnw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tnw_offset, *reinterpret_cast<__m256*>(&v000_in_range), sizeof(float));
                                    auto tne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tne_offset, *reinterpret_cast<__m256*>(&v100_in_range), sizeof(float));
                                    auto tsw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tsw_offset, *reinterpret_cast<__m256*>(&v010_in_range), sizeof(float));
                                    auto tse_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tse_offset, *reinterpret_cast<__m256*>(&v110_in_range), sizeof(float));

                                    auto bnw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bnw_offset, *reinterpret_cast<__m256*>(&v001_in_range), sizeof(float));
                                    auto bne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bne_offset, *reinterpret_cast<__m256*>(&v101_in_range), sizeof(float));
                                    auto bsw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bsw_offset, *reinterpret_cast<__m256*>(&v011_in_range), sizeof(float));
                                    auto bse_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bse_offset, *reinterpret_cast<__m256*>(&v111_in_range), sizeof(float));

                                    auto _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm256_set1_ps(gridptr[0]);
                                    auto gy = _mm256_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                    gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);
                                    gz = get_coord_p8(gz, vImgDf, padding_mode, align_corner);

                                    auto x_w = _mm256_floor_ps(gx);
                                    auto y_n = _mm256_floor_ps(gy);
                                    auto z_t = _mm256_floor_ps(gz);

                                    auto w = _mm256_sub_ps(gx, x_w);
                                    auto e = _mm256_sub_ps(v1fp8, w);
                                    auto n = _mm256_sub_ps(gy, y_n);
                                    auto s = _mm256_sub_ps(v1fp8, n);
                                    auto t = _mm256_sub_ps(gz, z_t);
                                    auto b = _mm256_sub_ps(v1fp8, t);

                                    __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                    {
                                        auto nw = _mm256_mul_ps(s, e);
                                        auto ne = _mm256_mul_ps(s, w);
                                        auto sw = _mm256_mul_ps(n, e);
                                        auto se = _mm256_mul_ps(n, w);

                                        tnw = _mm256_mul_ps(b, nw);
                                        tne = _mm256_mul_ps(b, ne);
                                        tsw = _mm256_mul_ps(b, sw);
                                        tse = _mm256_mul_ps(b, se);

                                        bnw = _mm256_mul_ps(t, nw);
                                        bne = _mm256_mul_ps(t, ne);
                                        bsw = _mm256_mul_ps(t, sw);
                                        bse = _mm256_mul_ps(t, se);
                                    }

                                    auto x0 = _mm256_cvtps_epi32(x_w);
                                    auto x1 = _mm256_add_epi32(x0, v1ip8);
                                    auto y0 = _mm256_cvtps_epi32(y_n);
                                    auto y1 = _mm256_add_epi32(y0, v1ip8);
                                    auto z0 = _mm256_cvtps_epi32(z_t);
                                    auto z1 = _mm256_add_epi32(z0, v1ip8);

                                    auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                                    auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y1));
                                    auto z1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(z1, vn1ip8), _mm256_cmpgt_epi32(vImgDi, z1));

                                    __m256i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                                    {
                                        auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                        v110_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                        v011_in_range = _mm256_and_si256(y1_in_range, z1_in_range);
                                        v101_in_range = _mm256_and_si256(x1_in_range, z1_in_range);
                                        v111_in_range = _mm256_and_si256(v11_in_range, z1_in_range);
                                    }

                                    // (W*H*z + W*y + x) * elempack + vec(8)
                                    auto i_tnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                                    auto i_tne_offset = _mm256_add_epi32(i_tnw_offset, vElempacki);
                                    auto i_tsw_offset = _mm256_add_epi32(i_tnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                                    auto i_tse_offset = _mm256_add_epi32(i_tsw_offset, vElempacki);

                                    auto i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), vElempacki), i_tnw_offset);
                                    auto i_bne_offset = _mm256_add_epi32(i_bnw_offset, vElempacki);
                                    auto i_bsw_offset = _mm256_add_epi32(i_bnw_offset, _mm256_mullo_epi32(vImgWi, vElempacki));
                                    auto i_bse_offset = _mm256_add_epi32(i_bsw_offset, vElempacki);

                                    auto tnw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tnw_offset, vn1fp8, sizeof(float));
                                    auto tne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tne_offset, *reinterpret_cast<__m256*>(&x1_in_range), sizeof(float));
                                    auto tsw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tsw_offset, *reinterpret_cast<__m256*>(&y1_in_range), sizeof(float));
                                    auto tse_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_tse_offset, *reinterpret_cast<__m256*>(&v110_in_range), sizeof(float));

                                    auto bnw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bnw_offset, *reinterpret_cast<__m256*>(&z1_in_range), sizeof(float));
                                    auto bne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bne_offset, *reinterpret_cast<__m256*>(&v101_in_range), sizeof(float));
                                    auto bsw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bsw_offset, *reinterpret_cast<__m256*>(&v011_in_range), sizeof(float));
                                    auto bse_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_bse_offset, *reinterpret_cast<__m256*>(&v111_in_range), sizeof(float));

                                    auto _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm256_set1_ps(gridptr[0]);
                                    auto gy = _mm256_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                    gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);
                                    gz = get_coord_p8(gz, vImgDf, padding_mode, align_corner);

                                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                    gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                    auto ix = _mm256_cvtps_epi32(gx);
                                    auto iy = _mm256_cvtps_epi32(gy);
                                    auto iz = _mm256_cvtps_epi32(gz);

                                    auto v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                        _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));
                                    v_in_range = _mm256_and_si256(v_in_range, _mm256_and_si256(_mm256_cmpgt_epi32(iz, vn1ip8), _mm256_cmpgt_epi32(vImgDi, iz)));

                                    auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                                    auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                        i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                                    _mm256_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm256_set1_ps(gridptr[0]);
                                    auto gy = _mm256_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm256_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                                    gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);
                                    gz = grid_sample_unormalize_p8(vImgDf, gz, align_corner);

                                    gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                    gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                    gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                    gx = compute_coord_p8(gx, vImgWf, padding_mode, align_corner);
                                    gy = compute_coord_p8(gy, vImgHf, padding_mode, align_corner);
                                    gy = compute_coord_p8(gz, vImgDf, padding_mode, align_corner);

                                    auto ix = _mm256_cvtps_epi32(gx);
                                    auto iy = _mm256_cvtps_epi32(gy);
                                    auto iz = _mm256_cvtps_epi32(gz);

                                    auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), iz), _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix)), vElempacki), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                                    auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                        i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                                    _mm256_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
            }

            if (sample_type == 3)
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -1;
            }
        }
    }

#endif // __AVX__

    if (elempack == 4)
    {
        const auto vn1fp4 = _mm_set1_ps(-1.0f);
        const auto v1ip4 = _mm_set1_epi32(1);
        const auto vn1ip4 = _mm_set1_epi32(-1);

        const auto vImgWfp4 = _mm_set1_ps(w);
        const auto vImgHfp4 = _mm_set1_ps(h);
        const auto vImgWip4 = _mm_set1_epi32(w);
        const auto vImgHip4 = _mm_set1_epi32(h);

        const auto vElemsizei = _mm_set1_epi32(elemsize / 8);
        const auto vElempacki = _mm_set1_epi32(elempack);
        const auto vElempackf = _mm_set1_ps(elempack);

        if (dims == 3)
        {
            const auto outW = grid.h;
            const auto outH = grid.c * grid.elempack;

            top_blob.create(outW, outH, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1) //zeros
            {
                if (padding_mode == 1) //zeros
                {
#pragma parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                //grid tensor has been packed
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm_set1_ps(gridptr[0]);
                                auto gy = _mm_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);

                                auto x_w = _mm_floor_ps(gx);
                                auto y_n = _mm_floor_ps(gy);

                                auto w = _mm_sub_ps(gx, x_w);
                                auto e = _mm_sub_ps(v1fp4, w);
                                auto n = _mm_sub_ps(gy, y_n);
                                auto s = _mm_sub_ps(v1fp4, n);

                                auto nw = _mm_mul_ps(s, e);
                                auto ne = _mm_mul_ps(s, w);
                                auto sw = _mm_mul_ps(n, e);
                                auto se = _mm_mul_ps(n, w);

                                auto x0 = _mm_cvtps_epi32(x_w);
                                auto x1 = _mm_add_epi32(x0, v1ip4);
                                auto y0 = _mm_cvtps_epi32(y_n);
                                auto y1 = _mm_add_epi32(y0, v1ip4);

                                auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x0));
                                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x1));
                                auto y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y0));
                                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y1));

                                auto v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
                                auto v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
                                auto v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
                                auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                                // (W*y + x) * elempack + vec(8)
                                auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWip4), x0), vElempacki),
                                    _mm_set_epi32(3, 2, 1, 0));
                                auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
                                auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWip4, vElempacki));
                                auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

                                auto nw_val = mask_gather_ps(ptr, i_nw_offset, *reinterpret_cast<__m128*>(&v00_in_range));
                                auto ne_val = mask_gather_ps(ptr, i_ne_offset, *reinterpret_cast<__m128*>(&v10_in_range));
                                auto sw_val = mask_gather_ps(ptr, i_sw_offset, *reinterpret_cast<__m128*>(&v01_in_range));
                                auto se_val = mask_gather_ps(ptr, i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                                auto _v = _mm_mul_ps(nw_val, nw);
                                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                                _mm_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
                else //border reflection
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm_set1_ps(gridptr[0]);
                                auto gy = _mm_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);

                                auto x_w = _mm_floor_ps(gx);
                                auto y_n = _mm_floor_ps(gy);

                                auto w = _mm_sub_ps(gx, x_w);
                                auto e = _mm_sub_ps(v1fp4, w);
                                auto n = _mm_sub_ps(gy, y_n);
                                auto s = _mm_sub_ps(v1fp4, n);

                                auto nw = _mm_mul_ps(s, e);
                                auto ne = _mm_mul_ps(s, w);
                                auto sw = _mm_mul_ps(n, e);
                                auto se = _mm_mul_ps(n, w);

                                auto x0 = _mm_cvtps_epi32(x_w);
                                auto x1 = _mm_add_epi32(x0, v1ip4);
                                auto y0 = _mm_cvtps_epi32(y_n);
                                auto y1 = _mm_add_epi32(y0, v1ip4);

                                auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x1));
                                auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y1));

                                auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                                auto i_nw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(y0, vImgWip4), x0), vElempacki),
                                    _mm_set_epi32(3, 2, 1, 0));
                                auto i_ne_offset = _mm_add_epi32(i_nw_offset, vElempacki);
                                auto i_sw_offset = _mm_add_epi32(i_nw_offset, _mm_mullo_epi32(vImgWip4, vElempacki));
                                auto i_se_offset = _mm_add_epi32(i_sw_offset, vElempacki);

                                auto nw_val = mask_gather_ps(ptr, i_nw_offset, vn1fp4);
                                auto ne_val = mask_gather_ps(ptr, i_ne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                                auto sw_val = mask_gather_ps(ptr, i_sw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                                auto se_val = mask_gather_ps(ptr, i_se_offset, *reinterpret_cast<__m128*>(&v11_in_range));

                                auto _v = _mm_mul_ps(nw_val, nw);
                                _v = _mm_comp_fmadd_ps(ne_val, ne, _v);
                                _v = _mm_comp_fmadd_ps(sw_val, sw, _v);
                                _v = _mm_comp_fmadd_ps(se_val, se, _v);

                                _mm_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1) //zeros
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm_set1_ps(gridptr[0]);
                                auto gy = _mm_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);

                                gx = _mm_floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                                gy = _mm_floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

                                auto ix = _mm_cvtps_epi32(gx);
                                auto iy = _mm_cvtps_epi32(gy);

                                auto v_in_range = _mm_and_si128(_mm_and_si128(_mm_cmpgt_epi32(ix, vn1ip4), _mm_cmpgt_epi32(vImgWip4, ix)),
                                    _mm_and_si128(_mm_cmpgt_epi32(iy, vn1ip4), _mm_cmpgt_epi32(vImgHip4, iy)));

                                auto i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWip4), ix), vElempacki),
                                    _mm_set_epi32(3, 2, 1, 0));

                                auto _v = mask_gather_ps(static_cast<float*>(bottom_blob.channel(q).data),
                                    i_offset, *reinterpret_cast<__m128*>(&v_in_range));

                                _mm_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
                else //border reflection
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm_set1_ps(gridptr[0]);
                                auto gy = _mm_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);

                                gx = _mm_floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                                gy = _mm_floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));

                                auto ix = _mm_cvtps_epi32(gx);
                                auto iy = _mm_cvtps_epi32(gy);

                                auto i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(iy, vImgWip4), ix), vElempacki),
                                    _mm_set_epi32(3, 2, 1, 0));

                                auto _v = mask_gather_ps(static_cast<float*>(bottom_blob.channel(q).data),
                                    i_offset, _mm_set1_ps(-1.0f));

                                _mm_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
            }

            if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm_set1_ps(gridptr[0]);
                                auto gy = _mm_set1_ps(gridptr[grid.elempack]);

                                gx = grid_sample_unormalize_p4(vImgWfp4, gx, align_corner);
                                gy = grid_sample_unormalize_p4(vImgHfp4, gy, align_corner);

                                auto gx_floor = _mm_floor_ps(gx);
                                auto gy_floor = _mm_floor_ps(gy);

                                const auto tx = _mm_sub_ps(gx, gx_floor);
                                const auto ty = _mm_sub_ps(gy, gy_floor);

                                __m128 coefficients[4];

                                for (int i = 0; i < 4; i++)
                                {
                                    auto gx0 = compute_coord_p4(_mm_add_ps(gx_floor, vn1fp4), vImgWfp4, padding_mode, align_corner);
                                    auto gx1 = compute_coord_p4(gx_floor, vImgWfp4, padding_mode, align_corner);
                                    auto gx2 = compute_coord_p4(_mm_add_ps(gx_floor, v1fp4), vImgWfp4, padding_mode, align_corner);
                                    auto gx3 = compute_coord_p4(_mm_add_ps(gx_floor, _mm_set1_ps(2.0f)), vImgWfp4, padding_mode, align_corner);

                                    gy = compute_coord_p4(_mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i)), vImgHfp4, padding_mode, align_corner);

                                    auto x0 = _mm_cvtps_epi32(gx0);
                                    auto x1 = _mm_cvtps_epi32(gx1);
                                    auto x2 = _mm_cvtps_epi32(gx2);
                                    auto x3 = _mm_cvtps_epi32(gx3);

                                    auto y = _mm_cvtps_epi32(gy);

                                    auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x0));
                                    auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x1));
                                    auto x2_in_range = _mm_and_si128(_mm_cmpgt_epi32(x2, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x2));
                                    auto x3_in_range = _mm_and_si128(_mm_cmpgt_epi32(x3, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x3));

                                    auto y_in_range = _mm_and_si128(_mm_cmpgt_epi32(y, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y));

                                    auto v0_in_range = _mm_and_si128(x0_in_range, y_in_range);
                                    auto v1_in_range = _mm_and_si128(x1_in_range, y_in_range);
                                    auto v2_in_range = _mm_and_si128(x2_in_range, y_in_range);
                                    auto v3_in_range = _mm_and_si128(x3_in_range, y_in_range);

                                    auto x0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx0), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx1), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx2), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx3), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                                    auto x0_offset = _mm_cvtps_epi32(x0_offset_f);
                                    auto x1_offset = _mm_cvtps_epi32(x1_offset_f);
                                    auto x2_offset = _mm_cvtps_epi32(x2_offset_f);
                                    auto x3_offset = _mm_cvtps_epi32(x3_offset_f);

                                    auto x0_val = mask_gather_ps(ptr, x0_offset, *reinterpret_cast<__m128*>(&v0_in_range));
                                    auto x1_val = mask_gather_ps(ptr, x1_offset, *reinterpret_cast<__m128*>(&v1_in_range));
                                    auto x2_val = mask_gather_ps(ptr, x2_offset, *reinterpret_cast<__m128*>(&v2_in_range));
                                    auto x3_val = mask_gather_ps(ptr, x3_offset, *reinterpret_cast<__m128*>(&v3_in_range));

                                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                                }

                                auto _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                _mm_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm_set1_ps(gridptr[0]);
                                auto gy = _mm_set1_ps(gridptr[grid.elempack]);

                                gx = grid_sample_unormalize_p4(vImgWfp4, gx, align_corner);
                                gy = grid_sample_unormalize_p4(vImgHfp4, gy, align_corner);

                                auto gx_floor = _mm_floor_ps(gx);
                                auto gy_floor = _mm_floor_ps(gy);

                                const auto tx = _mm_sub_ps(gx, gx_floor);
                                const auto ty = _mm_sub_ps(gy, gy_floor);

                                __m128 coefficients[4];

                                for (int i = 0; i < 4; i++)
                                {
                                    auto gx0 = compute_coord_p4(_mm_add_ps(gy_floor, vn1fp4), vImgWfp4, padding_mode, align_corner);
                                    auto gx1 = compute_coord_p4(gy_floor, vImgWfp4, padding_mode, align_corner);
                                    auto gx2 = compute_coord_p4(_mm_add_ps(gy_floor, v1fp4), vImgWfp4, padding_mode, align_corner);
                                    auto gx3 = compute_coord_p4(_mm_add_ps(gy_floor, _mm_set1_ps(2.0f)), vImgWfp4, padding_mode, align_corner);

                                    gy = compute_coord_p4(_mm_add_ps(gy_floor, _mm_set1_ps(-1.0f + i)), vImgHfp4, padding_mode, align_corner);

                                    auto x0 = _mm_cvtps_epi32(gx0);
                                    auto x1 = _mm_cvtps_epi32(gx1);
                                    auto x2 = _mm_cvtps_epi32(gx2);
                                    auto x3 = _mm_cvtps_epi32(gx3);

                                    auto y = _mm_cvtps_epi32(gy);

                                    auto x0_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx0), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x1_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx1), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x2_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx2), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));
                                    auto x3_offset_f = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(gy, vImgWfp4), gx3), vElempackf),
                                        _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f));

                                    auto x0_offset = _mm_cvtps_epi32(x0_offset_f);
                                    auto x1_offset = _mm_cvtps_epi32(x1_offset_f);
                                    auto x2_offset = _mm_cvtps_epi32(x2_offset_f);
                                    auto x3_offset = _mm_cvtps_epi32(x3_offset_f);

                                    auto x0_val = mask_gather_ps(ptr, x0_offset, vn1fp4);
                                    auto x1_val = mask_gather_ps(ptr, x1_offset, vn1fp4);
                                    auto x2_val = mask_gather_ps(ptr, x2_offset, vn1fp4);
                                    auto x3_val = mask_gather_ps(ptr, x3_offset, vn1fp4);

                                    coefficients[i] = cubic_interp1d_p4(x0_val, x1_val, x2_val, x3_val, tx);
                                }

                                auto _v = cubic_interp1d_p4(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                _mm_storeu_ps(outptr, _v);

                                outptr += elempack;
                            }
                        }
                    }
                }
            }
        }

        if (dims == 4)
        {
            const int outW = grid.h;
            const int outH = grid.d;
            const int outD = grid.c * grid.elempack;

            top_blob.create(outW, outH, outD, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const auto vImgDfp4 = _mm_set1_ps(d);
            const auto vImgDip4 = _mm_set1_epi32(d);

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm_set1_ps(gridptr[0]);
                                    auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                    gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);
                                    gz = get_coord_p4(gz, vImgDfp4, padding_mode, align_corner);

                                    auto x_w = _mm_floor_ps(gx);
                                    auto y_n = _mm_floor_ps(gy);
                                    auto z_t = _mm_floor_ps(gz);

                                    auto w = _mm_sub_ps(gx, x_w);
                                    auto e = _mm_sub_ps(v1fp4, w);
                                    auto n = _mm_sub_ps(gy, y_n);
                                    auto s = _mm_sub_ps(v1fp4, n);
                                    auto t = _mm_sub_ps(gz, z_t);
                                    auto b = _mm_sub_ps(v1fp4, t);

                                    __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                    {
                                        auto nw = _mm_mul_ps(s, e);
                                        auto ne = _mm_mul_ps(s, w);
                                        auto sw = _mm_mul_ps(n, e);
                                        auto se = _mm_mul_ps(n, w);

                                        tnw = _mm_mul_ps(b, nw);
                                        tne = _mm_mul_ps(b, ne);
                                        tsw = _mm_mul_ps(b, sw);
                                        tse = _mm_mul_ps(b, se);

                                        bnw = _mm_mul_ps(t, nw);
                                        bne = _mm_mul_ps(t, ne);
                                        bsw = _mm_mul_ps(t, sw);
                                        bse = _mm_mul_ps(t, se);
                                    }

                                    auto x0 = _mm_cvtps_epi32(x_w);
                                    auto x1 = _mm_add_epi32(x0, v1ip4);
                                    auto y0 = _mm_cvtps_epi32(y_n);
                                    auto y1 = _mm_add_epi32(y0, v1ip4);
                                    auto z0 = _mm_cvtps_epi32(z_t);
                                    auto z1 = _mm_add_epi32(z0, v1ip4);

                                    auto x0_in_range = _mm_and_si128(_mm_cmpgt_epi32(x0, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x0));
                                    auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x1));
                                    auto y0_in_range = _mm_and_si128(_mm_cmpgt_epi32(y0, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y0));
                                    auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y1));
                                    auto z0_in_range = _mm_and_si128(_mm_cmpgt_epi32(z0, vn1ip4), _mm_cmpgt_epi32(vImgDip4, z0));
                                    auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDip4, z1));

                                    __m128i v000_in_range, v010_in_range, v100_in_range, v110_in_range, v001_in_range, v011_in_range, v101_in_range, v111_in_range;
                                    {
                                        auto v00_in_range = _mm_and_si128(x0_in_range, y0_in_range);
                                        auto v01_in_range = _mm_and_si128(x0_in_range, y1_in_range);
                                        auto v10_in_range = _mm_and_si128(x1_in_range, y0_in_range);
                                        auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                                        v000_in_range = _mm_and_si128(v00_in_range, z0_in_range);
                                        v010_in_range = _mm_and_si128(v01_in_range, z0_in_range);
                                        v100_in_range = _mm_and_si128(v10_in_range, z0_in_range);
                                        v110_in_range = _mm_and_si128(v11_in_range, z0_in_range);

                                        v001_in_range = _mm_and_si128(v00_in_range, z1_in_range);
                                        v011_in_range = _mm_and_si128(v01_in_range, z1_in_range);
                                        v101_in_range = _mm_and_si128(v10_in_range, z1_in_range);
                                        v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                                    }

                                    // (W*H*z + W*y + x) * elempack + vec(8)
                                    auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWip4, vImgHip4), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWip4), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                                    auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                                    auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWip4, vElempacki));
                                    auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                                    auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWip4, vImgHip4), vElempacki), i_tnw_offset);
                                    auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                                    auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWip4, vElempacki));
                                    auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                                    auto tnw_val = mask_gather_ps(ptr, i_tnw_offset, *reinterpret_cast<__m128*>(&v000_in_range));
                                    auto tne_val = mask_gather_ps(ptr, i_tne_offset, *reinterpret_cast<__m128*>(&v100_in_range));
                                    auto tsw_val = mask_gather_ps(ptr, i_tsw_offset, *reinterpret_cast<__m128*>(&v010_in_range));
                                    auto tse_val = mask_gather_ps(ptr, i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                                    auto bnw_val = mask_gather_ps(ptr, i_bnw_offset, *reinterpret_cast<__m128*>(&v001_in_range));
                                    auto bne_val = mask_gather_ps(ptr, i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                                    auto bsw_val = mask_gather_ps(ptr, i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                                    auto bse_val = mask_gather_ps(ptr, i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                                    auto _v = _mm_mul_ps(tnw_val, tnw);
                                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm_set1_ps(gridptr[0]);
                                    auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                    gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);
                                    gz = get_coord_p4(gz, vImgDfp4, padding_mode, align_corner);

                                    auto x_w = _mm_floor_ps(gx);
                                    auto y_n = _mm_floor_ps(gy);
                                    auto z_t = _mm_floor_ps(gz);

                                    auto w = _mm_sub_ps(gx, x_w);
                                    auto e = _mm_sub_ps(v1fp4, w);
                                    auto n = _mm_sub_ps(gy, y_n);
                                    auto s = _mm_sub_ps(v1fp4, n);
                                    auto t = _mm_sub_ps(gz, z_t);
                                    auto b = _mm_sub_ps(v1fp4, t);

                                    __m128 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                    {
                                        auto nw = _mm_mul_ps(s, e);
                                        auto ne = _mm_mul_ps(s, w);
                                        auto sw = _mm_mul_ps(n, e);
                                        auto se = _mm_mul_ps(n, w);

                                        tnw = _mm_mul_ps(b, nw);
                                        tne = _mm_mul_ps(b, ne);
                                        tsw = _mm_mul_ps(b, sw);
                                        tse = _mm_mul_ps(b, se);

                                        bnw = _mm_mul_ps(t, nw);
                                        bne = _mm_mul_ps(t, ne);
                                        bsw = _mm_mul_ps(t, sw);
                                        bse = _mm_mul_ps(t, se);
                                    }

                                    auto x0 = _mm_cvtps_epi32(x_w);
                                    auto x1 = _mm_add_epi32(x0, v1ip4);
                                    auto y0 = _mm_cvtps_epi32(y_n);
                                    auto y1 = _mm_add_epi32(y0, v1ip4);
                                    auto z0 = _mm_cvtps_epi32(z_t);
                                    auto z1 = _mm_add_epi32(z0, v1ip4);

                                    auto x1_in_range = _mm_and_si128(_mm_cmpgt_epi32(x1, vn1ip4), _mm_cmpgt_epi32(vImgWip4, x1));
                                    auto y1_in_range = _mm_and_si128(_mm_cmpgt_epi32(y1, vn1ip4), _mm_cmpgt_epi32(vImgHip4, y1));
                                    auto z1_in_range = _mm_and_si128(_mm_cmpgt_epi32(z1, vn1ip4), _mm_cmpgt_epi32(vImgDip4, z1));

                                    __m128i v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                                    {
                                        auto v11_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                                        v110_in_range = _mm_and_si128(x1_in_range, y1_in_range);

                                        v011_in_range = _mm_and_si128(y1_in_range, z1_in_range);
                                        v101_in_range = _mm_and_si128(x1_in_range, z1_in_range);
                                        v111_in_range = _mm_and_si128(v11_in_range, z1_in_range);
                                    }

                                    // (W*H*z + W*y + x) * elempack + vec(8)
                                    auto i_tnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWip4, vImgHip4), z0), _mm_add_epi32(_mm_mullo_epi32(y0, vImgWip4), x0)), vElempacki), _mm_set_epi32(3, 2, 1, 0));
                                    auto i_tne_offset = _mm_add_epi32(i_tnw_offset, vElempacki);
                                    auto i_tsw_offset = _mm_add_epi32(i_tnw_offset, _mm_mullo_epi32(vImgWip4, vElempacki));
                                    auto i_tse_offset = _mm_add_epi32(i_tsw_offset, vElempacki);

                                    auto i_bnw_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWip4, vImgHip4), vElempacki), i_tnw_offset);
                                    auto i_bne_offset = _mm_add_epi32(i_bnw_offset, vElempacki);
                                    auto i_bsw_offset = _mm_add_epi32(i_bnw_offset, _mm_mullo_epi32(vImgWip4, vElempacki));
                                    auto i_bse_offset = _mm_add_epi32(i_bsw_offset, vElempacki);

                                    auto tnw_val = mask_gather_ps(ptr, i_tnw_offset, vn1fp4);
                                    auto tne_val = mask_gather_ps(ptr, i_tne_offset, *reinterpret_cast<__m128*>(&x1_in_range));
                                    auto tsw_val = mask_gather_ps(ptr, i_tsw_offset, *reinterpret_cast<__m128*>(&y1_in_range));
                                    auto tse_val = mask_gather_ps(ptr, i_tse_offset, *reinterpret_cast<__m128*>(&v110_in_range));

                                    auto bnw_val = mask_gather_ps(ptr, i_bnw_offset, *reinterpret_cast<__m128*>(&z1_in_range));
                                    auto bne_val = mask_gather_ps(ptr, i_bne_offset, *reinterpret_cast<__m128*>(&v101_in_range));
                                    auto bsw_val = mask_gather_ps(ptr, i_bsw_offset, *reinterpret_cast<__m128*>(&v011_in_range));
                                    auto bse_val = mask_gather_ps(ptr, i_bse_offset, *reinterpret_cast<__m128*>(&v111_in_range));

                                    auto _v = _mm_mul_ps(tnw_val, tnw);
                                    _v = _mm_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm_set1_ps(gridptr[0]);
                                    auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                    gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);
                                    gz = get_coord_p4(gz, vImgDfp4, padding_mode, align_corner);

                                    gx = _mm_floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                                    gy = _mm_floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                                    gz = _mm_floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                                    auto ix = _mm_cvtps_epi32(gx);
                                    auto iy = _mm_cvtps_epi32(gy);
                                    auto iz = _mm_cvtps_epi32(gz);

                                    auto v_in_range = _mm_and_si128(_mm_and_si128(_mm_cmpgt_epi32(ix, vn1ip4), _mm_cmpgt_epi32(vImgWip4, ix)),
                                        _mm_and_si128(_mm_cmpgt_epi32(iy, vn1ip4), _mm_cmpgt_epi32(vImgHip4, iy)));
                                    v_in_range = _mm_and_si128(v_in_range, _mm_and_si128(_mm_cmpgt_epi32(iz, vn1ip4), _mm_cmpgt_epi32(vImgDip4, iz)));

                                    auto i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWip4, vImgHip4), iz), _mm_add_epi32(_mm_mullo_epi32(iy, vImgWip4), ix)), vElempacki), _mm_set_epi32(3, 2, 1, 0));

                                    auto _v = mask_gather_ps(static_cast<float*>(bottom_blob.channel(q).data),
                                        i_offset, *reinterpret_cast<__m128*>(&v_in_range));

                                    _mm_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
                        for (int z = 0; z < outD; z++)
                        {
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(z / grid.elempack).depth(y).row(x) + z % grid.elempack;

                                    auto gx = _mm_set1_ps(gridptr[0]);
                                    auto gy = _mm_set1_ps(gridptr[grid.elempack]);
                                    auto gz = _mm_set1_ps(gridptr[grid.elempack * 2]);

                                    gx = get_coord_p4(gx, vImgWfp4, padding_mode, align_corner);
                                    gy = get_coord_p4(gy, vImgHfp4, padding_mode, align_corner);
                                    gz = get_coord_p4(gz, vImgDfp4, padding_mode, align_corner);

                                    gx = _mm_floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5f)));
                                    gy = _mm_floor_ps(_mm_add_ps(gy, _mm_set1_ps(0.5f)));
                                    gz = _mm_floor_ps(_mm_add_ps(gz, _mm_set1_ps(0.5f)));

                                    auto ix = _mm_cvtps_epi32(gx);
                                    auto iy = _mm_cvtps_epi32(gy);
                                    auto iz = _mm_cvtps_epi32(gz);

                                    auto i_offset = _mm_add_epi32(_mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_mullo_epi32(vImgWip4, vImgHip4), iz), _mm_add_epi32(_mm_mullo_epi32(iy, vImgWip4), ix)), vElempacki), _mm_set_epi32(3, 2, 1, 0));

                                    auto _v = mask_gather_ps(static_cast<float*>(bottom_blob.channel(q).data),
                                        i_offset, _mm_set1_ps(-1.0f));

                                    _mm_storeu_ps(outptr, _v);

                                    outptr += elempack;
                                }
                            }
                        }
                    }
                }
            }

            if (sample_type == 3)
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -1;
            }
        }
    }

#endif // __SSE2__

    if (elempack == 1)
    {
#if !__SSE2__
        ncnn::Mat grid_tmp;

        if (grid.elempack != 1)
        {
            ncnn::convert_packing(grid, grid_tmp, 1, opt);
        }

        ncnn::Mat grid_p1 = (grid.elempack == 1) ? grid : grid_tmp;
#if __AVX__
        const auto vn1fp8 = _mm256_set1_ps(-1.0f);
        const auto v1ip8 = _mm256_set1_epi32(1);
        const auto vn1ip8 = _mm256_set1_epi32(-1);

        const auto vImgWf = _mm256_set1_ps(w);
        const auto vImgHf = _mm256_set1_ps(h);
        const auto vImgWi = _mm256_set1_epi32(w);
        const auto vImgHi = _mm256_set1_epi32(h);
#endif // __AVX__

        if (dims == 3)
        {
            int size = w * h;
            const float* gridptr = static_cast<float*>(grid_p1.data);

            top_blob.create(grid.h, grid.c, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -1;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        int j = 0;
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
#if __AVX__
                        for (; j + 7 < size; j += 8)
                        {
                            auto tmp_x = _mm256_loadu_ps(gridptr + j);
                            auto gy = _mm256_loadu_ps(gridptr + j + 8);

                            auto gx = _mm256_shuffle_ps(tmp_x, gy, 0x10001000);
                            gy = _mm256_shuffle_ps(tmp_x, gy, 0x11011101);

                            gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                            gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);

                            auto x_w = _mm256_floor_ps(gx);
                            auto y_n = _mm256_floor_ps(gy);

                            auto w = _mm256_sub_ps(gx, x_w);
                            auto e = _mm256_sub_ps(v1fp8, w);
                            auto n = _mm256_sub_ps(gy, y_n);
                            auto s = _mm256_sub_ps(v1fp8, n);

                            auto nw = _mm256_mul_ps(s, e);
                            auto ne = _mm256_mul_ps(s, w);
                            auto sw = _mm256_mul_ps(n, e);
                            auto se = _mm256_mul_ps(n, w);

                            auto x0 = _mm256_cvtps_epi32(x_w);
                            auto x1 = _mm256_add_epi32(x0, v1ip8);
                            auto y0 = _mm256_cvtps_epi32(y_n);
                            auto y1 = _mm256_add_epi32(y0, v1ip8);

                            auto x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x0));
                            auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                            auto y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y0));
                            auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y1));

                            auto v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                            auto v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                            auto v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                            auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                            // (W*y + x) * elempack + vec(8)
                            auto i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                            auto i_ne_offset = _mm256_add_epi32(i_nw_offset, v1ip8);
                            auto i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                            auto i_se_offset = _mm256_add_epi32(i_sw_offset, v1ip8);

                            auto nw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_nw_offset, *reinterpret_cast<__m256*>(&v00_in_range), sizeof(float));
                            auto ne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_ne_offset, *reinterpret_cast<__m256*>(&v10_in_range), sizeof(float));
                            auto sw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_sw_offset, *reinterpret_cast<__m256*>(&v01_in_range), sizeof(float));
                            auto se_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_se_offset, *reinterpret_cast<__m256*>(&v11_in_range), sizeof(float));

                            auto _v = _mm256_mul_ps(nw_val, nw);
                            _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                            _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                            _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                            _mm256_storeu_ps(outptr, _v);

                            outptr += 8;
                        }
#endif // __AVX__
                        for (; j < size; j++)
                        {

                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        int j = 0;
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
#if __AVX__
                        for (; j + 7 < size; j += 8)
                        {
                            auto tmp_x = _mm256_loadu_ps(gridptr + j);
                            auto gy = _mm256_loadu_ps(gridptr + j + 8);

                            auto gx = _mm256_shuffle_ps(tmp_x, gy, 0x10001000);
                            gy = _mm256_shuffle_ps(tmp_x, gy, 0x11011101);

                            gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                            gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);

                            auto x_w = _mm256_floor_ps(gx);
                            auto y_n = _mm256_floor_ps(gy);

                            auto w = _mm256_sub_ps(gx, x_w);
                            auto e = _mm256_sub_ps(v1fp8, w);
                            auto n = _mm256_sub_ps(gy, y_n);
                            auto s = _mm256_sub_ps(v1fp8, n);

                            auto nw = _mm256_mul_ps(s, e);
                            auto ne = _mm256_mul_ps(s, w);
                            auto sw = _mm256_mul_ps(n, e);
                            auto se = _mm256_mul_ps(n, w);

                            auto x0 = _mm256_cvtps_epi32(x_w);
                            auto x1 = _mm256_add_epi32(x0, v1ip8);
                            auto y0 = _mm256_cvtps_epi32(y_n);
                            auto y1 = _mm256_add_epi32(y0, v1ip8);

                            auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                            auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y1));

                            auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                            auto i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                            auto i_ne_offset = _mm256_add_epi32(i_nw_offset, v1ip8);
                            auto i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                            auto i_se_offset = _mm256_add_epi32(i_sw_offset, v1ip8);

                            auto nw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_nw_offset, vn1fp8, sizeof(float));
                            auto ne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_ne_offset, *reinterpret_cast<__m256*>(&x1_in_range), sizeof(float));
                            auto sw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_sw_offset, *reinterpret_cast<__m256*>(&y1_in_range), sizeof(float));
                            auto se_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, i_se_offset, *reinterpret_cast<__m256*>(&v11_in_range), sizeof(float));

                            auto _v = _mm256_mul_ps(nw_val, nw);
                            _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                            _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                            _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                            _mm256_storeu_ps(outptr, _v);

                            outptr += 8;
                        }
#endif // __AVX__
                        for (; j < size; j++)
                        {

                        }
                    }
                }
            }
            else if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        int j = 0;
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
#if __AVX__
                        for (; j + 7 < size; j += 8)
                        {
                            auto tmp_x = _mm256_loadu_ps(gridptr + j);
                            auto gy = _mm256_loadu_ps(gridptr + j + 8);

                            auto gx = _mm256_shuffle_ps(tmp_x, gy, 0x10001000);
                            gy = _mm256_shuffle_ps(tmp_x, gy, 0x11011101);

                            gx = get_coord_p8(gx, vImgWf, padding_mode, align_corner);
                            gy = get_coord_p8(gy, vImgHf, padding_mode, align_corner);

                            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                            auto ix = _mm256_cvtps_epi32(gx);
                            auto iy = _mm256_cvtps_epi32(gy);

                            auto v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vImgWi, ix)),
                                _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vImgHi, iy)));

                            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix);

                            auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                            _mm256_storeu_ps(outptr, _v);

                            outptr += 8;
                        }
#endif // __AVX__
                        for (; j < size; j++)
                        {

                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        int j = 0;
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
#if __AVX__
                        for (; j + 7 < size; j += 8)
                        {
                            auto tmp_x = _mm256_loadu_ps(gridptr + j);
                            auto gy = _mm256_loadu_ps(gridptr + j + 8);

                            auto gx = _mm256_shuffle_ps(tmp_x, gy, 0x10001000);
                            gy = _mm256_shuffle_ps(tmp_x, gy, 0x11011101);

                            gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                            gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);

                            gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                            gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                            gx = compute_coord_p8(gx, vImgWf, padding_mode, align_corner);
                            gy = compute_coord_p8(gy, vImgHf, padding_mode, align_corner);

                            auto ix = _mm256_cvtps_epi32(gx);
                            auto iy = _mm256_cvtps_epi32(gy);

                            auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(iy, vImgWi), ix);

                            auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                            _mm256_storeu_ps(outptr, _v);

                            outptr += 8;
                        }
#endif // __AVX__
                        for (; j < size; j++)
                        {

                        }
                    }
                }
            }
            else if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        int j = 0;
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
#if __AVX__
                        for (; j + 7 < size; j += 8)
                        {
                            auto tmp_x = _mm256_loadu_ps(gridptr + j);
                            auto gy = _mm256_loadu_ps(gridptr + j + 8);

                            auto gx = _mm256_shuffle_ps(tmp_x, gy, 0x10001000);
                            gy = _mm256_shuffle_ps(tmp_x, gy, 0x11011101);

                            gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                            gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);

                            auto gx_floor = _mm256_floor_ps(gx);
                            auto gy_floor = _mm256_floor_ps(gy);

                            const auto tx = _mm256_sub_ps(gx, gx_floor);
                            const auto ty = _mm256_sub_ps(gy, gy_floor);

                            __m256 coefficients[4];

                            for (int i = 0; i < 4; i++)
                            {
                                auto gx0 = compute_coord_p8(_mm256_add_ps(gx_floor, vn1fp8), vImgWf, padding_mode, align_corner);
                                auto gx1 = compute_coord_p8(gx_floor, vImgWf, padding_mode, align_corner);
                                auto gx2 = compute_coord_p8(_mm256_add_ps(gx_floor, v1fp8), vImgWf, padding_mode, align_corner);
                                auto gx3 = compute_coord_p8(_mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f)), vImgWf, padding_mode, align_corner);

                                gy = compute_coord_p8(_mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i)), vImgHf, padding_mode, align_corner);

                                auto x0 = _mm256_cvtps_epi32(gx0);
                                auto x1 = _mm256_cvtps_epi32(gx1);
                                auto x2 = _mm256_cvtps_epi32(gx2);
                                auto x3 = _mm256_cvtps_epi32(gx3);

                                auto y = _mm256_cvtps_epi32(gy);

                                auto x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x0));
                                auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x1));
                                auto x2_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x2, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x2));
                                auto x3_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x3, vn1ip8), _mm256_cmpgt_epi32(vImgWi, x3));

                                auto y_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y, vn1ip8), _mm256_cmpgt_epi32(vImgHi, y));

                                auto v0_in_range = _mm256_and_si256(x0_in_range, y_in_range);
                                auto v1_in_range = _mm256_and_si256(x1_in_range, y_in_range);
                                auto v2_in_range = _mm256_and_si256(x2_in_range, y_in_range);
                                auto v3_in_range = _mm256_and_si256(x3_in_range, y_in_range);

                                auto x0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                                auto x1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                                auto x2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                                auto x3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                                auto x0_offset = _mm256_cvtps_epi32(x0_offset_f);
                                auto x1_offset = _mm256_cvtps_epi32(x1_offset_f);
                                auto x2_offset = _mm256_cvtps_epi32(x2_offset_f);
                                auto x3_offset = _mm256_cvtps_epi32(x3_offset_f);

                                auto x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x0_offset, *reinterpret_cast<__m256*>(&v0_in_range), sizeof(float));
                                auto x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x1_offset, *reinterpret_cast<__m256*>(&v1_in_range), sizeof(float));
                                auto x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x2_offset, *reinterpret_cast<__m256*>(&v2_in_range), sizeof(float));
                                auto x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x3_offset, *reinterpret_cast<__m256*>(&v3_in_range), sizeof(float));

                                coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                            }

                            auto _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                            _mm256_storeu_ps(outptr, _v);

                            outptr += 8;
                        }
#endif // __AVX__
                        for (; j < size; j++)
                        {

                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        int j = 0;
                        float* outptr = top_blob.channel(q);
                        const float* ptr = static_cast<float*>(bottom_blob.channel(q).data);
#if __AVX__
                        for (; j + 7 < size; j += 8)
                        {
                            auto tmp_x = _mm256_loadu_ps(gridptr + j);
                            auto gy = _mm256_loadu_ps(gridptr + j + 8);

                            auto gx = _mm256_shuffle_ps(tmp_x, gy, 0x10001000);
                            gy = _mm256_shuffle_ps(tmp_x, gy, 0x11011101);

                            gx = grid_sample_unormalize_p8(vImgWf, gx, align_corner);
                            gy = grid_sample_unormalize_p8(vImgHf, gy, align_corner);

                            auto gx_floor = _mm256_floor_ps(gx);
                            auto gy_floor = _mm256_floor_ps(gy);

                            const auto tx = _mm256_sub_ps(gx, gx_floor);
                            const auto ty = _mm256_sub_ps(gy, gy_floor);

                            __m256 coefficients[4];

                            for (int i = 0; i < 4; i++)
                            {
                                auto gx0 = compute_coord_p8(_mm256_add_ps(gx_floor, vn1fp8), vImgWf, padding_mode, align_corner);
                                auto gx1 = compute_coord_p8(gx_floor, vImgWf, padding_mode, align_corner);
                                auto gx2 = compute_coord_p8(_mm256_add_ps(gx_floor, v1fp8), vImgWf, padding_mode, align_corner);
                                auto gx3 = compute_coord_p8(_mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f)), vImgWf, padding_mode, align_corner);

                                gy = compute_coord_p8(_mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i)), vImgHf, padding_mode, align_corner);

                                auto x0 = _mm256_cvtps_epi32(gx0);
                                auto x1 = _mm256_cvtps_epi32(gx1);
                                auto x2 = _mm256_cvtps_epi32(gx2);
                                auto x3 = _mm256_cvtps_epi32(gx3);

                                auto y = _mm256_cvtps_epi32(gy);

                                auto x0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                                auto x1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                                auto x2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                                auto x3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                                auto x0_offset = _mm256_cvtps_epi32(x0_offset_f);
                                auto x1_offset = _mm256_cvtps_epi32(x1_offset_f);
                                auto x2_offset = _mm256_cvtps_epi32(x2_offset_f);
                                auto x3_offset = _mm256_cvtps_epi32(x3_offset_f);

                                auto x0_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x0_offset, vn1fp8, sizeof(float));
                                auto x1_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x1_offset, vn1fp8, sizeof(float));
                                auto x2_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x2_offset, vn1fp8, sizeof(float));
                                auto x3_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, x3_offset, vn1fp8, sizeof(float));

                                coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                            }

                            auto _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                            _mm256_storeu_ps(outptr, _v);

                            outptr += 8;
                        }
#endif // __AVX__
                        for (; j < size; j++)
                        {

                        }
                    }
                }
            }
        }

        if (dims == 4)
        {
            int size = w * h * d;
            if (sample_type == 1)
            {

            }
            else if (sample_type == 2)
            {

            }
            else
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -1;
            }
        }
        return 0;
#endif // __SSE2__

        return GridSample::forward(bottom_blobs, top_blobs, opt);
    }

    return 0;
}

} // namespace ncnn
