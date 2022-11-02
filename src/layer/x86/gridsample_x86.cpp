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
    const __m256 v1f = *(__m256*)_ps256_1;

    static __m256 NCNN_FORCEINLINE
        grid_sample_unormalize(__m256 w, __m256 coordx, int align_corner)
    {
        __m256 two = _mm256_set1_ps(2.f);

        if (align_corner)
            return _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(coordx, v1f), two), _mm256_sub_ps(w, v1f));
        else
            return _mm256_div_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(coordx, v1f), w), v1f), two);
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
        x = _mm256_sub_ps(x, reflect_v);
        return x;
    }

    static __m256 compute_coord(__m256 sx, __m256 w, int padding_mode, int align_corner)
    {
        if (padding_mode == 2) // border
        {
            sx = border_coord(sx, _mm256_sub_ps(w, v1f));
        }
        else if (padding_mode == 3) // reflection
        {
            if (align_corner)
            {
                sx = reflect_coord(sx, _mm256_sub_ps(w, v1f));
            }
            else
            {
                __m256 v0p5f = *(__m256*)_ps256_0p5;
                sx = _mm256_sub_ps(reflect_coord(_mm256_add_ps(sx, v0p5f), w), v0p5f);
                sx = border_coord(sx, _mm256_sub_ps(w, v1f));
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

    int GridSample_x86::forward_inplace(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

            if (dims == 3)
            {
                const int outW = grid.h;
                const int outH = grid.c;

                top_blob.create(outW, outH, channels, elemsize, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (resize_type == 1)
                {
                    if (padding_mode == 1) //zeros
                    {
#pragma omp parallel for num_threads(opt.num_threads)
                        for (int q = 0; q < channels; q++)
                        {
                            const float* outptr = bottom_blob.channel(q);
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(outH).row(outW);
                                    __m256 gx = _mm256_set1_ps(gridptr[0]);
                                    __m256 gy = _mm256_set1_ps(gridptr[1]);

                                    __m256 vecH = _mm256_set1_ps(outH);
                                    __m256 vecW = _mm256_set1_ps(outW);

                                    gx = get_coord(gx, vecW, padding_mode, align_corner);
                                    gx = get_coord(gy, vecH, padding_mode, align_corner);

                                    auto x_w = _mm256_floor_ps(gx);
                                    auto y_n = _mm256_floor_ps(gy);

                                    auto w = _mm256_sub_ps(gx, x_w);
                                    auto e = _mm256_sub_ps(v1f, w);
                                    auto n = _mm256_sub_ps(gy, y_n);
                                    auto s = _mm256_sub_ps(v1f, n);

                                    auto nw = _mm256_mul_ps(s, e);
                                    auto ne = _mm256_mul_ps(s, w);
                                    auto sw = _mm256_mul_ps(n, e);
                                    auto se = _mm256_mul_ps(n, w);



                                    outptr++;
                                }
                            }
                        }
                    }
                    else //border bilinear
                    {
#pragma omp parallel for num_threads(opt.num_threads)
                        for (int q = 0; q < channels; q++)
                        {
                            const float* outptr = bottom_blob.channel(q);
                            for (int y = 0; y < outH; y++)
                            {
                                for (int x = 0; x < outW; x++)
                                {
                                    const float* gridptr = grid.channel(outH).row(outW);
                                    __m256 gx = _mm256_set1_ps(gridptr[0]);
                                    __m256 gy = _mm256_set1_ps(gridptr[1]);

                                    __m256 vecH = _mm256_set1_ps(outH);
                                    __m256 vecW = _mm256_set1_ps(outW);

                                    gx = get_coord(gx, vecW, padding_mode, align_corner);
                                    gx = get_coord(gy, vecH, padding_mode, align_corner);

                                    auto x_w = _mm256_floor_ps(gx);
                                    auto y_n = _mm256_floor_ps(gy);

                                    auto w = _mm256_sub_ps(gx, x_w);
                                    auto e = _mm256_sub_ps(v1f, w);
                                    auto n = _mm256_sub_ps(gy, y_n);
                                    auto s = _mm256_sub_ps(v1f, n);

                                    auto nw = _mm256_mul_ps(s, e);
                                    auto ne = _mm256_mul_ps(s, w);
                                    auto sw = _mm256_mul_ps(n, e);
                                    auto se = _mm256_mul_ps(n, w);



                                    outptr++;
                                }
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



                                outptr++;
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



                                    outptr++;
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



                                    outptr++;
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
            return forward(bottom_blobs, top_blobs, opt);
        }

        return 0;
    }

} // namespace ncnn
