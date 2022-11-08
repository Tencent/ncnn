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
#include <emmintrin.h>
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
const __m256i v1ip8 = _mm256_set1_epi32(1);
const __m256i vn1ip8 = _mm256_set1_epi32(-1);

static __m256 NCNN_FORCEINLINE
grid_sample_unormalize(__m256 w, __m256 coordx, int align_corner)
{
    __m256 two = _mm256_set1_ps(2.f);

    if (align_corner)
        return _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(coordx, v1fp8), two), _mm256_sub_ps(w, v1fp8));
    else
        return _mm256_div_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(coordx, v1fp8), w), v1fp8), two);
}

static NCNN_FORCEINLINE __m256 border_coord(__m256 coord, __m256 border)
{
    return _mm256_min_ps(border, _mm256_max_ps(coord, _mm256_setzero_ps()));
}

static __m256 reflect_coord(__m256 x, __m256 high)
{
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);

    __m256 reflect_v = _mm256_and_ps(_mm256_sub_ps(x, high), *(__m256*)_ps256_inv_sign_mask);
    x = _mm256_sub_ps(high, reflect_v);
    return x;
}

static __m256 compute_coord(__m256 sx, __m256 w, int padding_mode, int align_corner)
{
    if (padding_mode == 2) // border
    {
        sx = border_coord(sx, _mm256_sub_ps(w, v1fp8));
    }
    else if (padding_mode == 3) // reflection
    {
        if (align_corner)
        {
            sx = reflect_coord(sx, _mm256_sub_ps(w, v1fp8));
        }
        else
        {
            __m256 v0p5f = *(__m256*)_ps256_0p5;
            sx = _mm256_sub_ps(reflect_coord(_mm256_add_ps(sx, v0p5f), w), v0p5f);
            sx = border_coord(sx, _mm256_sub_ps(w, v1fp8));
        }
    }

    return sx;
}

static __m256 get_coord(__m256 x, __m256 w, int padding_mode, int align_corner)
{
    // compute the origin coordinates
    __m256 sx = grid_sample_unormalize(w, x, align_corner);

    // correct the coordinates according to the padding_mode
    __m256 coord = compute_coord(sx, w, padding_mode, align_corner);

    return coord;
}

#endif // __AVX__

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
        const auto vElemsizei = _mm256_set1_epi32(elemsize / 8);
        if (dims == 3)
        {
            const auto outW = grid.h;
            const auto outH = grid.c * grid.elempack;

            const auto vWi = _mm256_set1_epi32(outW);
            const auto vHi = _mm256_set1_epi32(outH);

            const auto vHf = _mm256_set1_ps(outH);
            const auto vWf = _mm256_set1_ps(outW);

            top_blob.create(outW, outH, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const auto vElempacki = _mm256_set1_epi32(elempack);

            if (resize_type == 1) //zeros
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
                                //grid tensor has been packed
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                auto gx = _mm256_set1_ps(gridptr[0]);
                                auto gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord(gx, vWf, padding_mode, align_corner);
                                gy = get_coord(gy, vHf, padding_mode, align_corner);

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

                                auto x0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x0, vn1ip8), _mm256_cmpgt_epi32(vWi, x0));
                                auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vWi, x1));
                                auto y0_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y0, vn1ip8), _mm256_cmpgt_epi32(vHi, y0));
                                auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vHi, y1));

                                auto v00_in_range = _mm256_and_si256(x0_in_range, y0_in_range);
                                auto v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                                auto v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                                auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                auto i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vWi), x0), vElempacki),
                                                                    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                                auto i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                                auto i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vWi, vElempacki));
                                auto i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                                auto nw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_nw_offset, *reinterpret_cast<__m256*>(&v00_in_range), sizeof(float));
                                auto ne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_ne_offset, *reinterpret_cast<__m256*>(&v10_in_range), sizeof(float));
                                auto sw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_sw_offset, *reinterpret_cast<__m256*>(&v01_in_range), sizeof(float));
                                auto se_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_se_offset, *reinterpret_cast<__m256*>(&v11_in_range), sizeof(float));

                                auto _v = _mm256_mul_ps(nw_val, nw);
                                _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                _mm256_storeu_ps(outptr, _v);

                                outptr += 8;
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

                                gx = get_coord(gx, vWf, padding_mode, align_corner);
                                gy = get_coord(gy, vHf, padding_mode, align_corner);

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

                                auto x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, vn1ip8), _mm256_cmpgt_epi32(vWi, x1));
                                auto y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, vn1ip8), _mm256_cmpgt_epi32(vHi, y1));

                                auto v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                auto i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(y0, vWi), x0), vElempacki),
                                                                    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                                auto i_ne_offset = _mm256_add_epi32(i_nw_offset, vElempacki);
                                auto i_sw_offset = _mm256_add_epi32(i_nw_offset, _mm256_mullo_epi32(vWi, vElempacki));
                                auto i_se_offset = _mm256_add_epi32(i_sw_offset, vElempacki);

                                auto nw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_nw_offset, _mm256_set1_ps(-1.0f), sizeof(float));
                                auto ne_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_ne_offset, *reinterpret_cast<__m256*>(&x1_in_range), sizeof(float));
                                auto sw_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_sw_offset, *reinterpret_cast<__m256*>(&y1_in_range), sizeof(float));
                                auto se_val = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data), i_se_offset, *reinterpret_cast<__m256*>(&v11_in_range), sizeof(float));

                                auto _v = _mm256_mul_ps(nw_val, nw);
                                _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                _mm256_storeu_ps(outptr, _v);

                                outptr += 8;
                            }
                        }
                    }
                }
            }

            if (resize_type == 2)
            {
                if (padding_mode == 1) //zeros
                {
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

                                gx = get_coord(gx, vWf, padding_mode, align_corner);
                                gy = get_coord(gy, vHf, padding_mode, align_corner);

                                gx = _mm256_round_ps(gx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                                gy = _mm256_round_ps(gy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                                auto ix = _mm256_cvtps_epi32(gx);
                                auto iy = _mm256_cvtps_epi32(gy);

                                auto v_in_range = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi32(ix, vn1ip8), _mm256_cmpgt_epi32(vWi, ix)),
                                                                   _mm256_and_si256(_mm256_cmpgt_epi32(iy, vn1ip8), _mm256_cmpgt_epi32(vHi, iy)));

                                auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vWi), ix), vElempacki),
                                                                 _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                                                   i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                                _mm256_storeu_ps(outptr, _v);

                                outptr += 8;
                            }
                        }
                    }
                }
                else //border reflection
                {
                    for (int q = 0; q < channels; q++)
                    {
                        float* outptr = top_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                const float* gridptr = grid.channel(y / grid.elempack).row(x) + y % grid.elempack;
                                __m256 gx = _mm256_set1_ps(gridptr[0]);
                                __m256 gy = _mm256_set1_ps(gridptr[grid.elempack]);

                                gx = get_coord(gx, vWf, padding_mode, align_corner);
                                gy = get_coord(gy, vHf, padding_mode, align_corner);

                                gx = _mm256_round_ps(gx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                                gy = _mm256_round_ps(gy, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                                auto ix = _mm256_cvtps_epi32(gx);
                                auto iy = _mm256_cvtps_epi32(gy);

                                auto i_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(iy, vWi), ix), vElempacki),
                                                                 _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

                                auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), static_cast<float*>(bottom_blob.channel(q).data),
                                                                   i_offset, _mm256_set1_ps(-1.0f), sizeof(float));

                                _mm256_storeu_ps(outptr, _v);

                                outptr += 8;
                            }
                        }
                    }
                }
            }

            if (resize_type == 3)
            {
                for (int q = 0; q < channels; q++)
                {
                    const float* outptr = bottom_blob.channel(q);
                    for (int y = 0; y < outH; y++)
                    {
                        for (int x = 0; x < outW; x++)
                        {
                            outptr += 8;
                        }
                    }
                }
            }
        }

        if (dims == 4)
        {
        }
    }

#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 3)
        {
            if (dims == 3)
            {
                const int outW = grid.h;
                const int outH = grid.c;

                top_blob.create(outW, outH, channels, elemsize, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (resize_type == 1)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* outptr = bottom_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                outptr += 8;
                            }
                        }
                    }
                }

                if (resize_type == 2)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* outptr = bottom_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                outptr++;
                            }
                        }
                    }
                }

                if (resize_type == 3)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* outptr = bottom_blob.channel(q);
                        for (int y = 0; y < outH; y++)
                        {
                            for (int x = 0; x < outW; x++)
                            {
                                outptr += 8;
                            }
                        }
                    }
                }
            }
        }

        if (dims == 4)
        {
        }
    }

#endif // __SSE2__

    if (elempack == 1)
    {
        return GridSample::forward(bottom_blobs, top_blobs, opt);
    }

    return 0;
}

} // namespace ncnn
