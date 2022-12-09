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
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
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
#if __AVX512F__

_PS512_CONST(n1, -1.0f);
_PI32_CONST512(n1, -1);

#include "gridsample_bilinear_pack16.h"
#include "gridsample_nearest_pack16.h"
#include "gridsample_bicubic_pack16.h"

#endif // __AVX512F__

_PS256_CONST(n1, -1.0f);
_PS256_CONST(2, 2.0f);
_PI32_CONST256(n1, -1);

static NCNN_FORCEINLINE __m256 mask_gather_ps256(const float* ptr, __m256i offset, __m256 mask)
{
#if __AVX2__
    __m256 v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, offset, mask, sizeof(float));
#else
    int offseti[8], maski[8];
    memcpy(offseti, &offset, 8 * sizeof(int));
    memcpy(maski, &mask, 8 * sizeof(int));

    float data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 8; i++)
    {
        if (maski[i] & 0xF0000000)
        {
            data[i] = *(ptr + offseti[i]);
        }
    }

    __m256 v = _mm256_loadu_ps(data);
#endif // __AVX__

    return v;
}

static NCNN_FORCEINLINE __m256 cubic_interp1d_p8(const __m256& x0_v, const __m256& x1_v, const __m256& x2_v, const __m256& x3_v, const __m256& tx)
{
    const __m256 A = _mm256_set1_ps(-0.75f);

    const __m256 x0 = _mm256_add_ps(tx, *(__m256*)_ps256_1);
    const __m256& x1 = tx;
    const __m256 x2 = _mm256_sub_ps(*(__m256*)_ps256_1, tx);
    //const __m256 x3 = _mm256_add_ps(x2, *(__m256*)_ps256_1);

    const __m256 coeffs0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(A, x0), _mm256_mul_ps(_mm256_set1_ps(5.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(8.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(4), A));
    const __m256 coeffs1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x1), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x1), x1), *(__m256*)_ps256_1);
    const __m256 coeffs2 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x2), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x2), x2), *(__m256*)_ps256_1);
    const __m256 coeffs3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(*(__m256*)_ps256_1, coeffs0), coeffs1), coeffs2);

    __m256 _v = _mm256_mul_ps(coeffs0, x0_v);
    _v = _mm256_comp_fmadd_ps(coeffs1, x1_v, _v);
    _v = _mm256_comp_fmadd_ps(coeffs2, x2_v, _v);
    _v = _mm256_comp_fmadd_ps(coeffs3, x3_v, _v);

    return _v;
}

#include "gridsample_bilinear_pack8.h"
#include "gridsample_nearest_pack8.h"
#include "gridsample_bicubic_pack8.h"

#endif // __AVX__

const __m128 v1fp4 = _mm_set1_ps(1.0f);
const __m128 vn1fp4 = _mm_set1_ps(-1.0f);
const __m128i v1ip4 = _mm_set1_epi32(1);
const __m128i vn1ip4 = _mm_set1_epi32(-1);

static NCNN_FORCEINLINE __m128 mask_gather_ps(const float* ptr, __m128i offset, __m128 mask)
{
#if __AVX2__
    __m128 v = _mm_mask_i32gather_ps(_mm_setzero_ps(), ptr, offset, mask, sizeof(float));
#else
    int offseti[4], maski[4];
    memcpy(offseti, &offset, 4 * sizeof(int));
    memcpy(maski, &mask, 4 * sizeof(int));

    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; i++)
    {
        if (maski[i] & 0xF0000000)
        {
            data[i] = *(ptr + offseti[i]);
        }
    }

    __m128 v = _mm_loadu_ps(data);
#endif // __AVX__

    return v;
}

#include "gridsample_bicubic_pack4.h"
#include "gridsample_bilinear_pack4.h"
#include "gridsample_nearest_pack4.h"

static inline void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline float reflect_coord(float x, int high)
{
    x = abs(x);
    x = high - abs(x - high);
    return x;
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
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 3)
        {
            top_blob.create(grid.h, grid.c * grid.elempack, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }

            if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }
        }

        if (dims == 4)
        {
            const int outW = grid.h;
            const int outH = grid.d;
            const int outD = grid.c * grid.elempack;

            top_blob.create(grid.h, grid.d, grid.c * grid.elempack, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample sample_type error\n");
                    return -100;
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_zeros_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_border_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_reflection_blob_pack16(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample sample_type error\n");
                    return -100;
                }
            }

            if (sample_type == 3)
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -100;
            }
        }
    }
#endif // __AVX512F__

#if __AVX__

    if (elempack == 8)
    {
        if (dims == 3)
        {
            top_blob.create(grid.h, grid.c * grid.elempack, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }

            if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }
        }

        if (dims == 4)
        {
            top_blob.create(grid.h, grid.d, grid.c * grid.elempack, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample sample_type error\n");
                    return -100;
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_zeros_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_border_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_reflection_blob_pack8(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample sample_type error\n");
                    return -100;
                }
            }

            if (sample_type == 3)
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -100;
            }
        }
    }

#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 3)
        {
            top_blob.create(grid.h, grid.c * grid.elempack, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample sample_type error\n");
                    return -100;
                }
            }

            if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }
        }

        if (dims == 4)
        {
            top_blob.create(grid.h, grid.d, grid.c * grid.elempack, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }

            if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_zeros_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_border_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_reflection_blob_pack4(bottom_blob, top_blob, grid, opt);
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample sample_type error\n");
                    return -100;
                }
            }

            if (sample_type == 3)
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -100;
            }
        }
    }

#endif // __SSE2__

    if (elempack == 1)
    {
#if __SSE2__
        ncnn::Mat grid_tmp;

        if (grid.elempack != 1)
        {
            ncnn::convert_packing(grid, grid_tmp, 1, opt);
        }

        ncnn::Mat grid_p1 = (grid.elempack == 1) ? grid : grid_tmp;
#if __AVX__
        const __m256 vImgWf = _mm256_set1_ps(w);
        const __m256 vImgHf = _mm256_set1_ps(h);
#if __AVX2__
        const __m256i vImgWi = _mm256_set1_epi32(w);
        const __m256i vImgHi = _mm256_set1_epi32(h);
#endif //__AVX2__
#endif // __AVX__

        if (dims == 3)
        {
            const int grid_size = grid_p1.w * grid_p1.h;

            top_blob.create(grid_p1.h, grid_p1.c, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);

                                __m256 nw = _mm256_mul_ps(s, e);
                                __m256 ne = _mm256_mul_ps(s, w);
                                __m256 sw = _mm256_mul_ps(n, e);
                                __m256 se = _mm256_mul_ps(n, w);

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
                                __m256i v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                                __m256i v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                __m256i i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, *(__m256i*)_pi32_256_1);
#else
                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                                __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                                __m256 v01_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                                __m256 v10_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                __m256 nw_offset = _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w);
                                __m256 ne_offset = _mm256_add_ps(nw_offset, *(__m256*)_ps256_1);
                                __m256 sw_offset = _mm256_add_ps(nw_offset, vImgWf);
                                __m256 se_offset = _mm256_add_ps(sw_offset, *(__m256*)_ps256_1);

                                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);
#endif // __AVX2__

                                for (int q = 0; q < channels; q++)
                                {
#if __AVX2__
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, _mm256_castsi256_ps(v00_in_range));
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, _mm256_castsi256_ps(v10_in_range));
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, _mm256_castsi256_ps(v01_in_range));
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, _mm256_castsi256_ps(v11_in_range));
#else
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, v00_in_range);
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, v10_in_range);
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, v01_in_range);
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, v11_in_range);
#endif // __AVX2__

                                    __m256 _v = _mm256_mul_ps(nw_val, nw);
                                    _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                    _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                    _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                // bilinear interpolate
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;

                                bool v00_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h);
                                bool v01_in_range = (x1 > -1) & (x1 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h);
                                bool v10_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y1 > -1) & (y1 < bottom_blob.h);
                                bool v11_in_range = (x1 > -1) & (x1 < bottom_blob.w) & (y1 > -1) & (y1 < bottom_blob.h);

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v00 = v00_in_range ? image.row(y0)[x0] : 0;
                                    float v01 = v01_in_range ? image.row(y0)[x1] : 0;
                                    float v10 = v10_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;

                                    float v0 = v00 * (1 - alpha) + v01 * alpha;
                                    float v1 = v10 * (1 - alpha) + v11 * alpha;

                                    top_blob.channel(q).row(y)[x / 2] = v0 * (1 - beta) + v1 * beta;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
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
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);

                                __m256 nw = _mm256_mul_ps(s, e);
                                __m256 ne = _mm256_mul_ps(s, w);
                                __m256 sw = _mm256_mul_ps(n, e);
                                __m256 se = _mm256_mul_ps(n, w);

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
                                __m256i v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                                __m256i v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                __m256i i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, *(__m256i*)_pi32_256_1);
#else
                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                                __m256 x0_in_range = _mm256_and_ps(_mm256_cmp_ps(x_w, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x_w, _CMP_GT_OS));
                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y0_in_range = _mm256_and_ps(_mm256_cmp_ps(y_n, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y_n, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                                __m256 v00_in_range = _mm256_and_ps(x0_in_range, y0_in_range);
                                __m256 v01_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                                __m256 v10_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                __m256 nw_offset = _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w);
                                __m256 ne_offset = _mm256_add_ps(nw_offset, *(__m256*)_ps256_1);
                                __m256 sw_offset = _mm256_add_ps(nw_offset, vImgWf);
                                __m256 se_offset = _mm256_add_ps(sw_offset, *(__m256*)_ps256_1);

                                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);
#endif // __AVX2__

                                for (int q = 0; q < channels; q++)
                                {
#if __AVX2__
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, _mm256_castsi256_ps(v00_in_range));
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, _mm256_castsi256_ps(v10_in_range));
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, _mm256_castsi256_ps(v01_in_range));
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, _mm256_castsi256_ps(v11_in_range));
#else
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, v00_in_range);
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, v10_in_range);
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, v01_in_range);
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, v11_in_range);
#endif // __AVX2__

                                    __m256 _v = _mm256_mul_ps(nw_val, nw);
                                    _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                    _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                    _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                // bilinear interpolate
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;

                                bool v00_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h);
                                bool v01_in_range = (x1 > -1) & (x1 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h);
                                bool v10_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y1 > -1) & (y1 < bottom_blob.h);
                                bool v11_in_range = (x1 > -1) & (x1 < bottom_blob.w) & (y1 > -1) & (y1 < bottom_blob.h);

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v00 = v00_in_range ? image.row(y0)[x0] : 0;
                                    float v01 = v01_in_range ? image.row(y0)[x1] : 0;
                                    float v10 = v10_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;

                                    float v0 = v00 * (1 - alpha) + v01 * alpha;
                                    float v1 = v10 * (1 - alpha) + v11 * alpha;

                                    top_blob.channel(q).row(y)[x / 2] = v0 * (1 - beta) + v1 * beta;
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);

                                __m256 nw = _mm256_mul_ps(s, e);
                                __m256 ne = _mm256_mul_ps(s, w);
                                __m256 sw = _mm256_mul_ps(n, e);
                                __m256 se = _mm256_mul_ps(n, w);

#if __AVX2__
                                __m256i x0 = _mm256_cvtps_epi32(x_w);
                                __m256i y0 = _mm256_cvtps_epi32(y_n);
                                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                __m256i i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, *(__m256i*)_pi32_256_1);
#else
                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                __m256 nw_offset = _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w);
                                __m256 ne_offset = _mm256_add_ps(nw_offset, *(__m256*)_ps256_1);
                                __m256 sw_offset = _mm256_add_ps(nw_offset, vImgWf);
                                __m256 se_offset = _mm256_add_ps(sw_offset, *(__m256*)_ps256_1);

                                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);
#endif

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, *(__m256*)_ps256_n1);
#if __AVX2__
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, _mm256_castsi256_ps(x1_in_range));
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, _mm256_castsi256_ps(y1_in_range));
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, _mm256_castsi256_ps(v11_in_range));
#else
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, x1_in_range);
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, y1_in_range);
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, v11_in_range);
#endif

                                    __m256 _v = _mm256_mul_ps(nw_val, nw);
                                    _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                    _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                    _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                sample_x = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                sample_y = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < bottom_blob.w);
                                bool y1_in_range = (y1 > -1) & (y1 < bottom_blob.h);
                                bool v11_in_range = x1_in_range & y1_in_range;

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v00 = image.row(y0)[x0];
                                    float v01 = x1_in_range ? image.row(y0)[x1] : 0;
                                    float v10 = y1_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;

                                    float v0 = v00 * (1 - alpha) + v01 * alpha;
                                    float v1 = v10 * (1 - alpha) + v11 * alpha;

                                    top_blob.channel(q).row(y)[x / 2] = v0 * (1 - beta) + v1 * beta;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
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
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);

                                __m256 nw = _mm256_mul_ps(s, e);
                                __m256 ne = _mm256_mul_ps(s, w);
                                __m256 sw = _mm256_mul_ps(n, e);
                                __m256 se = _mm256_mul_ps(n, w);

#if __AVX2__
                                __m256i x0 = _mm256_cvtps_epi32(x_w);
                                __m256i y0 = _mm256_cvtps_epi32(y_n);
                                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                __m256i i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, *(__m256i*)_pi32_256_1);
#else
                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                __m256 nw_offset = _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w);
                                __m256 ne_offset = _mm256_add_ps(nw_offset, *(__m256*)_ps256_1);
                                __m256 sw_offset = _mm256_add_ps(nw_offset, vImgWf);
                                __m256 se_offset = _mm256_add_ps(sw_offset, *(__m256*)_ps256_1);

                                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);
#endif

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, *(__m256*)_ps256_n1);
#if __AVX2__
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, _mm256_castsi256_ps(x1_in_range));
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, _mm256_castsi256_ps(y1_in_range));
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, _mm256_castsi256_ps(v11_in_range));
#else
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, x1_in_range);
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, y1_in_range);
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, v11_in_range);
#endif

                                    __m256 _v = _mm256_mul_ps(nw_val, nw);
                                    _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                    _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                    _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                sample_x = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                sample_y = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < bottom_blob.w);
                                bool y1_in_range = (y1 > -1) & (y1 < bottom_blob.h);
                                bool v11_in_range = x1_in_range & y1_in_range;

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v00 = image.row(y0)[x0];
                                    float v01 = x1_in_range ? image.row(y0)[x1] : 0;
                                    float v10 = y1_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;

                                    float v0 = v00 * (1 - alpha) + v01 * alpha;
                                    float v1 = v10 * (1 - alpha) + v11 * alpha;

                                    top_blob.channel(q).row(y)[x / 2] = v0 * (1 - beta) + v1 * beta;
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                                    gx = _mm256_add_ps(gx, v0p5fp8);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(vImgWf, reflectx_v);

                                    gx = _mm256_sub_ps(gx, v0p5fp8);

                                    _mm256_sub_ps(gx, v0p5fp8);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_add_ps(gy, v0p5fp8);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                                    gy = _mm256_sub_ps(gy, v0p5fp8);

                                    _mm256_sub_ps(gy, v0p5fp8);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);

                                __m256 nw = _mm256_mul_ps(s, e);
                                __m256 ne = _mm256_mul_ps(s, w);
                                __m256 sw = _mm256_mul_ps(n, e);
                                __m256 se = _mm256_mul_ps(n, w);

#if __AVX2__
                                __m256i x0 = _mm256_cvtps_epi32(x_w);
                                __m256i y0 = _mm256_cvtps_epi32(y_n);
                                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                __m256i i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, *(__m256i*)_pi32_256_1);
#else
                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                __m256 nw_offset = _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w);
                                __m256 ne_offset = _mm256_add_ps(nw_offset, *(__m256*)_ps256_1);
                                __m256 sw_offset = _mm256_add_ps(nw_offset, vImgWf);
                                __m256 se_offset = _mm256_add_ps(sw_offset, *(__m256*)_ps256_1);

                                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);
#endif

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, *(__m256*)_ps256_n1);
#if __AVX2__
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, _mm256_castsi256_ps(x1_in_range));
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, _mm256_castsi256_ps(y1_in_range));
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, _mm256_castsi256_ps(v11_in_range));
#else
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, x1_in_range);
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, y1_in_range);
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, v11_in_range);
#endif

                                    __m256 _v = _mm256_mul_ps(nw_val, nw);
                                    _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                    _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                    _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                sample_x = abs(sample_x + 0.5f);
                                sample_x = w - abs(sample_x - w) - 0.5;

                                sample_y = abs(sample_y + 0.5f);
                                sample_y = h - abs(sample_y - h) - 0.5;

                                sample_x = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                sample_y = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < bottom_blob.w);
                                bool y1_in_range = (y1 > -1) & (y1 < bottom_blob.h);
                                bool v11_in_range = x1_in_range & y1_in_range;

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v00 = image.row(y0)[x0];
                                    float v01 = x1_in_range ? image.row(y0)[x1] : 0;
                                    float v10 = y1_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;

                                    float v0 = v00 * (1 - alpha) + v01 * alpha;
                                    float v1 = v10 * (1 - alpha) + v11 * alpha;

                                    top_blob.channel(q).row(y)[x / 2] = v0 * (1 - beta) + v1 * beta;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
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
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(border_x, reflectx_v);

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(border_y, reflecty_v);
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);

                                __m256 nw = _mm256_mul_ps(s, e);
                                __m256 ne = _mm256_mul_ps(s, w);
                                __m256 sw = _mm256_mul_ps(n, e);
                                __m256 se = _mm256_mul_ps(n, w);

#if __AVX2__
                                __m256i x0 = _mm256_cvtps_epi32(x_w);
                                __m256i y0 = _mm256_cvtps_epi32(y_n);
                                __m256i x1 = _mm256_add_epi32(x0, *(__m256i*)_pi32_256_1);
                                __m256i y1 = _mm256_add_epi32(y0, *(__m256i*)_pi32_256_1);

                                __m256i x1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(x1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgWi, x1));
                                __m256i y1_in_range = _mm256_and_si256(_mm256_cmpgt_epi32(y1, *(__m256i*)_pi32_256_n1), _mm256_cmpgt_epi32(vImgHi, y1));

                                __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                __m256i i_nw_offset = _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0);
                                __m256i i_ne_offset = _mm256_add_epi32(i_nw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_sw_offset = _mm256_add_epi32(i_nw_offset, vImgWi);
                                __m256i i_se_offset = _mm256_add_epi32(i_sw_offset, *(__m256i*)_pi32_256_1);
#else
                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));

                                __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                __m256 nw_offset = _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w);
                                __m256 ne_offset = _mm256_add_ps(nw_offset, *(__m256*)_ps256_1);
                                __m256 sw_offset = _mm256_add_ps(nw_offset, vImgWf);
                                __m256 se_offset = _mm256_add_ps(sw_offset, *(__m256*)_ps256_1);

                                __m256i i_nw_offset = _mm256_cvtps_epi32(nw_offset);
                                __m256i i_ne_offset = _mm256_cvtps_epi32(ne_offset);
                                __m256i i_sw_offset = _mm256_cvtps_epi32(sw_offset);
                                __m256i i_se_offset = _mm256_cvtps_epi32(se_offset);
#endif

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 nw_val = mask_gather_ps256(bottom_blob.channel(q), i_nw_offset, *(__m256*)_ps256_n1);
#if __AVX2__
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, _mm256_castsi256_ps(x1_in_range));
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, _mm256_castsi256_ps(y1_in_range));
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, _mm256_castsi256_ps(v11_in_range));
#else
                                    __m256 ne_val = mask_gather_ps256(bottom_blob.channel(q), i_ne_offset, x1_in_range);
                                    __m256 sw_val = mask_gather_ps256(bottom_blob.channel(q), i_sw_offset, y1_in_range);
                                    __m256 se_val = mask_gather_ps256(bottom_blob.channel(q), i_se_offset, v11_in_range);
#endif

                                    __m256 _v = _mm256_mul_ps(nw_val, nw);
                                    _v = _mm256_comp_fmadd_ps(ne_val, ne, _v);
                                    _v = _mm256_comp_fmadd_ps(sw_val, sw, _v);
                                    _v = _mm256_comp_fmadd_ps(se_val, se, _v);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                sample_x = abs(sample_x);
                                sample_x = (w - 1) - abs(sample_x - (w - 1));

                                sample_y = abs(sample_y);
                                sample_y = (h - 1) - abs(sample_y - (h - 1));

                                sample_x = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                sample_y = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < bottom_blob.w);
                                bool y1_in_range = (y1 > -1) & (y1 < bottom_blob.h);
                                bool v11_in_range = x1_in_range & y1_in_range;

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v00 = image.row(y0)[x0];
                                    float v01 = x1_in_range ? image.row(y0)[x1] : 0;
                                    float v10 = y1_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;

                                    float v0 = v00 * (1 - alpha) + v01 * alpha;
                                    float v1 = v10 * (1 - alpha) + v11 * alpha;

                                    top_blob.channel(q).row(y)[x / 2] = v0 * (1 - beta) + v1 * beta;
                                }
                            }
                        }
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }
            else if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, v_in_range);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                                bool v00_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h);

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    top_blob.channel(q).row(y)[x / 2] = v00_in_range ? image.row(y0)[x0] : 0;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
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
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, v_in_range);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                                bool v00_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h);

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    top_blob.channel(q).row(y)[x / 2] = v00_in_range ? image.row(y0)[x0] : 0;
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                                __m256i i_offset = _mm256_cvtps_epi32(offset);
                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                sample_x = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                sample_y = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    top_blob.channel(q).row(y)[x / 2] = image.row(y0)[x0];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
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
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                                __m256i i_offset = _mm256_cvtps_epi32(offset);
                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                sample_x = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                sample_y = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    top_blob.channel(q).row(y)[x / 2] = image.row(y0)[x0];
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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

                                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                // compute coord
                                {
                                    // x
                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                                    gx = _mm256_add_ps(gx, v0p5fp8);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(vImgWf, reflectx_v);

                                    gx = _mm256_sub_ps(gx, v0p5fp8);

                                    _mm256_sub_ps(gx, v0p5fp8);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_add_ps(gy, v0p5fp8);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                                    gy = _mm256_sub_ps(gy, v0p5fp8);

                                    _mm256_sub_ps(gy, v0p5fp8);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                }

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                                __m256i i_offset = _mm256_cvtps_epi32(offset);
                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                sample_x = floor(sample_x + 0.5f);
                                sample_y = floor(sample_y + 0.5f);

                                sample_x = abs(sample_x + 0.5f);
                                sample_x = w - abs(sample_x - w) - 0.5;

                                sample_y = abs(sample_y + 0.5f);
                                sample_y = h - abs(sample_y - h) - 0.5;

                                int x0 = std::min(w - 1.0f, std::max(sample_x, 0.0f));
                                int y0 = std::min(h - 1.0f, std::max(sample_y, 0.0f));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    top_blob.channel(q).row(y)[x / 2] = image.row(y0)[x0];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

                                __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
                                gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
                                tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

                                gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
                                gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

                                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
                                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));

                                // compute coord
                                {
                                    // x
                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(border_x, reflectx_v);

                                    // y
                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(border_y, reflecty_v);
                                }

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx);
                                __m256i i_offset = _mm256_cvtps_epi32(offset);
                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                sample_x = floor(sample_x + 0.5f);
                                sample_y = floor(sample_y + 0.5f);

                                sample_x = abs(sample_x);
                                int x0 = (w - 1) - abs(sample_x - (w - 1));

                                sample_y = abs(sample_y);
                                int y0 = (h - 1) - abs(sample_y - (h - 1));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    top_blob.channel(q).row(y)[x / 2] = image.row(y0)[x0];
                                }
                            }
                        }
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }
            else if (sample_type == 3)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
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

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    for (int i = 0; i < 4; i++)
                                    {
                                        __m256 x0_val = mask_gather_ps256(bottom_blob.channel(q), v0_offset[i], v0_in_range[i]);
                                        __m256 x1_val = mask_gather_ps256(bottom_blob.channel(q), v1_offset[i], v1_in_range[i]);
                                        __m256 x2_val = mask_gather_ps256(bottom_blob.channel(q), v2_offset[i], v2_in_range[i]);
                                        __m256 x3_val = mask_gather_ps256(bottom_blob.channel(q), v3_offset[i], v3_in_range[i]);

                                        coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                    }

                                    __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                int x1 = floor(sample_x);
                                int y1 = floor(sample_y);
                                int x0 = x1 - 1;
                                int y0 = y1 - 1;
                                int x2 = x1 + 1;
                                int y2 = y1 + 1;
                                int x3 = x1 + 2;
                                int y3 = y1 + 2;

                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool x0_in_range = (x0 > -1) & (x0 < w);
                                bool y0_in_range = (y0 > -1) & (y0 < h);
                                bool x2_in_range = (x2 > -1) & (x2 < w);
                                bool y2_in_range = (y2 > -1) & (y2 < h);
                                bool x3_in_range = (x3 > -1) & (x3 < w);
                                bool y3_in_range = (y3 > -1) & (y3 < h);

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

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    float v00 = v00_in_range ? image.row(y0)[x0] : 0;
                                    float v01 = v01_in_range ? image.row(y0)[x1] : 0;
                                    float v02 = v02_in_range ? image.row(y0)[x2] : 0;
                                    float v03 = v03_in_range ? image.row(y0)[x3] : 0;
                                    float v10 = v10_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;
                                    float v12 = v12_in_range ? image.row(y1)[x2] : 0;
                                    float v13 = v13_in_range ? image.row(y1)[x3] : 0;
                                    float v20 = v20_in_range ? image.row(y2)[x0] : 0;
                                    float v21 = v21_in_range ? image.row(y2)[x1] : 0;
                                    float v22 = v22_in_range ? image.row(y2)[x2] : 0;
                                    float v23 = v23_in_range ? image.row(y2)[x3] : 0;
                                    float v30 = v30_in_range ? image.row(y3)[x0] : 0;
                                    float v31 = v31_in_range ? image.row(y3)[x1] : 0;
                                    float v32 = v32_in_range ? image.row(y3)[x2] : 0;
                                    float v33 = v33_in_range ? image.row(y3)[x3] : 0;

                                    float x_coeffs[4];
                                    float y_coeffs[4];
                                    interpolate_cubic(sample_x - x1, x_coeffs);
                                    interpolate_cubic(sample_y - y1, y_coeffs);

                                    float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                                    float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                                    float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                                    float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                                    top_blob.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
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
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
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

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    for (int i = 0; i < 4; i++)
                                    {
                                        __m256 x0_val = mask_gather_ps256(bottom_blob.channel(q), v0_offset[i], v0_in_range[i]);
                                        __m256 x1_val = mask_gather_ps256(bottom_blob.channel(q), v1_offset[i], v1_in_range[i]);
                                        __m256 x2_val = mask_gather_ps256(bottom_blob.channel(q), v2_offset[i], v2_in_range[i]);
                                        __m256 x3_val = mask_gather_ps256(bottom_blob.channel(q), v3_offset[i], v3_in_range[i]);

                                        coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                    }

                                    __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                int x1 = floor(sample_x);
                                int y1 = floor(sample_y);
                                int x0 = x1 - 1;
                                int y0 = y1 - 1;
                                int x2 = x1 + 1;
                                int y2 = y1 + 1;
                                int x3 = x1 + 2;
                                int y3 = y1 + 2;

                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool x0_in_range = (x0 > -1) & (x0 < w);
                                bool y0_in_range = (y0 > -1) & (y0 < h);
                                bool x2_in_range = (x2 > -1) & (x2 < w);
                                bool y2_in_range = (y2 > -1) & (y2 < h);
                                bool x3_in_range = (x3 > -1) & (x3 < w);
                                bool y3_in_range = (y3 > -1) & (y3 < h);

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

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    float v00 = v00_in_range ? image.row(y0)[x0] : 0;
                                    float v01 = v01_in_range ? image.row(y0)[x1] : 0;
                                    float v02 = v02_in_range ? image.row(y0)[x2] : 0;
                                    float v03 = v03_in_range ? image.row(y0)[x3] : 0;
                                    float v10 = v10_in_range ? image.row(y1)[x0] : 0;
                                    float v11 = v11_in_range ? image.row(y1)[x1] : 0;
                                    float v12 = v12_in_range ? image.row(y1)[x2] : 0;
                                    float v13 = v13_in_range ? image.row(y1)[x3] : 0;
                                    float v20 = v20_in_range ? image.row(y2)[x0] : 0;
                                    float v21 = v21_in_range ? image.row(y2)[x1] : 0;
                                    float v22 = v22_in_range ? image.row(y2)[x2] : 0;
                                    float v23 = v23_in_range ? image.row(y2)[x3] : 0;
                                    float v30 = v30_in_range ? image.row(y3)[x0] : 0;
                                    float v31 = v31_in_range ? image.row(y3)[x1] : 0;
                                    float v32 = v32_in_range ? image.row(y3)[x2] : 0;
                                    float v33 = v33_in_range ? image.row(y3)[x3] : 0;

                                    float x_coeffs[4];
                                    float y_coeffs[4];
                                    interpolate_cubic(sample_x - x1, x_coeffs);
                                    interpolate_cubic(sample_y - y1, y_coeffs);

                                    float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                                    float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                                    float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                                    float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                                    top_blob.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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

                                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);
                                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
                                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                __m256 gx_floor = _mm256_floor_ps(gx);
                                __m256 gy_floor = _mm256_floor_ps(gy);

                                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                                __m256 coefficients[4];

                                __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                                __m256 gx1 = gx_floor;
                                __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                                __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

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

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    for (int i = 0; i < 4; i++)
                                    {
                                        __m256 x0_val = mask_gather_ps256(bottom_blob.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x1_val = mask_gather_ps256(bottom_blob.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x2_val = mask_gather_ps256(bottom_blob.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x3_val = mask_gather_ps256(bottom_blob.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                                        coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                    }

                                    __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                int x_floor = floor(sample_x);
                                int y_floor = floor(sample_y);

                                int x1 = x_floor;
                                int y1 = y_floor;
                                int x0 = x1 - 1;
                                int y0 = y1 - 1;
                                int x2 = x1 + 1;
                                int y2 = y1 + 1;
                                int x3 = x1 + 2;
                                int y3 = y1 + 2;

                                x1 = std::min(w - 1, std::max(x1, 0));
                                y1 = std::min(h - 1, std::max(y1, 0));
                                x0 = std::min(w - 1, std::max(x0, 0));
                                y0 = std::min(h - 1, std::max(y0, 0));
                                x2 = std::min(w - 1, std::max(x2, 0));
                                y2 = std::min(h - 1, std::max(y2, 0));
                                x3 = std::min(w - 1, std::max(x3, 0));
                                y3 = std::min(h - 1, std::max(y3, 0));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    float v00 = image.row(y0)[x0];
                                    float v01 = image.row(y0)[x1];
                                    float v02 = image.row(y0)[x2];
                                    float v03 = image.row(y0)[x3];
                                    float v10 = image.row(y1)[x0];
                                    float v11 = image.row(y1)[x1];
                                    float v12 = image.row(y1)[x2];
                                    float v13 = image.row(y1)[x3];
                                    float v20 = image.row(y2)[x0];
                                    float v21 = image.row(y2)[x1];
                                    float v22 = image.row(y2)[x2];
                                    float v23 = image.row(y2)[x3];
                                    float v30 = image.row(y3)[x0];
                                    float v31 = image.row(y3)[x1];
                                    float v32 = image.row(y3)[x2];
                                    float v33 = image.row(y3)[x3];

                                    float x_coeffs[4];
                                    float y_coeffs[4];
                                    interpolate_cubic(sample_x - x_floor, x_coeffs);
                                    interpolate_cubic(sample_y - y_floor, y_coeffs);

                                    float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                                    float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                                    float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                                    float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                                    top_blob.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

                                __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
                                gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
                                tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

                                gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
                                gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

                                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
                                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
                                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                __m256 gx_floor = _mm256_floor_ps(gx);
                                __m256 gy_floor = _mm256_floor_ps(gy);

                                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                                __m256 coefficients[4];

                                __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                                __m256 gx1 = gx_floor;
                                __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                                __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));

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

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    for (int i = 0; i < 4; i++)
                                    {
                                        __m256 x0_val = mask_gather_ps256(bottom_blob.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x1_val = mask_gather_ps256(bottom_blob.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x2_val = mask_gather_ps256(bottom_blob.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x3_val = mask_gather_ps256(bottom_blob.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                                        coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                    }

                                    __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                int x_floor = floor(sample_x);
                                int y_floor = floor(sample_y);

                                int x1 = x_floor;
                                int y1 = y_floor;
                                int x0 = x1 - 1;
                                int y0 = y1 - 1;
                                int x2 = x1 + 1;
                                int y2 = y1 + 1;
                                int x3 = x1 + 2;
                                int y3 = y1 + 2;

                                x1 = std::min(w - 1, std::max(x1, 0));
                                y1 = std::min(h - 1, std::max(y1, 0));
                                x0 = std::min(w - 1, std::max(x0, 0));
                                y0 = std::min(h - 1, std::max(y0, 0));
                                x2 = std::min(w - 1, std::max(x2, 0));
                                y2 = std::min(h - 1, std::max(y2, 0));
                                x3 = std::min(w - 1, std::max(x3, 0));
                                y3 = std::min(h - 1, std::max(y3, 0));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    float v00 = image.row(y0)[x0];
                                    float v01 = image.row(y0)[x1];
                                    float v02 = image.row(y0)[x2];
                                    float v03 = image.row(y0)[x3];
                                    float v10 = image.row(y1)[x0];
                                    float v11 = image.row(y1)[x1];
                                    float v12 = image.row(y1)[x2];
                                    float v13 = image.row(y1)[x3];
                                    float v20 = image.row(y2)[x0];
                                    float v21 = image.row(y2)[x1];
                                    float v22 = image.row(y2)[x2];
                                    float v23 = image.row(y2)[x3];
                                    float v30 = image.row(y3)[x0];
                                    float v31 = image.row(y3)[x1];
                                    float v32 = image.row(y3)[x2];
                                    float v33 = image.row(y3)[x3];

                                    float x_coeffs[4];
                                    float y_coeffs[4];
                                    interpolate_cubic(sample_x - x_floor, x_coeffs);
                                    interpolate_cubic(sample_y - y_floor, y_coeffs);

                                    float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                                    float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                                    float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                                    float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                                    top_blob.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
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

                                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);
                                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
                                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                __m256 gx_floor = _mm256_floor_ps(gx);
                                __m256 gy_floor = _mm256_floor_ps(gy);

                                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                                __m256 coefficients[4];

                                __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                                __m256 gx1 = gx_floor;
                                __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                                __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));
                                const __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                                {
                                    // x0
                                    gx0 = _mm256_add_ps(gx0, v0p5fp8);

                                    gx0 = _mm256_and_ps(gx0, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx0_v = _mm256_and_ps(_mm256_sub_ps(gx0, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx0 = _mm256_sub_ps(vImgWf, reflectx0_v);

                                    gx0 = _mm256_sub_ps(gx0, v0p5fp8);

                                    _mm256_sub_ps(gx0, v0p5fp8);

                                    gx0 = _mm256_min_ps(border_x, _mm256_max_ps(gx0, _mm256_setzero_ps()));

                                    // x1
                                    gx1 = _mm256_add_ps(gx1, v0p5fp8);

                                    gx1 = _mm256_and_ps(gx1, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx1_v = _mm256_and_ps(_mm256_sub_ps(gx1, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx1 = _mm256_sub_ps(vImgWf, reflectx1_v);

                                    gx1 = _mm256_sub_ps(gx1, v0p5fp8);

                                    _mm256_sub_ps(gx1, v0p5fp8);

                                    gx1 = _mm256_min_ps(border_x, _mm256_max_ps(gx1, _mm256_setzero_ps()));

                                    // x2
                                    gx2 = _mm256_add_ps(gx2, v0p5fp8);

                                    gx2 = _mm256_and_ps(gx2, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx2_v = _mm256_and_ps(_mm256_sub_ps(gx2, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx2 = _mm256_sub_ps(vImgWf, reflectx2_v);

                                    gx2 = _mm256_sub_ps(gx2, v0p5fp8);

                                    _mm256_sub_ps(gx2, v0p5fp8);

                                    gx2 = _mm256_min_ps(border_x, _mm256_max_ps(gx2, _mm256_setzero_ps()));

                                    // x3
                                    gx3 = _mm256_add_ps(gx3, v0p5fp8);

                                    gx3 = _mm256_and_ps(gx3, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx3_v = _mm256_and_ps(_mm256_sub_ps(gx3, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx3 = _mm256_sub_ps(vImgWf, reflectx3_v);

                                    gx3 = _mm256_sub_ps(gx3, v0p5fp8);

                                    _mm256_sub_ps(gx3, v0p5fp8);

                                    gx3 = _mm256_min_ps(border_x, _mm256_max_ps(gx3, _mm256_setzero_ps()));
                                }

                                __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
                                for (int i = 0; i < 4; i++)
                                {
                                    gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                                    {
                                        //y
                                        gy = _mm256_add_ps(gy, v0p5fp8);

                                        gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                        __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                                        gy = _mm256_sub_ps(vImgHf, reflecty_v);

                                        gy = _mm256_sub_ps(gy, v0p5fp8);

                                        _mm256_sub_ps(gy, v0p5fp8);

                                        gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));
                                    }

                                    __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                                    __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                                    __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                                    __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                                    v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                                    v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                                    v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                                    v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
                                }

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    for (int i = 0; i < 4; i++)
                                    {
                                        __m256 x0_val = mask_gather_ps256(bottom_blob.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x1_val = mask_gather_ps256(bottom_blob.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x2_val = mask_gather_ps256(bottom_blob.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x3_val = mask_gather_ps256(bottom_blob.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                                        coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                    }

                                    __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__

                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = ((sample_x + 1) * w - 1) / 2.f;
                                sample_y = ((sample_y + 1) * h - 1) / 2.f;

                                int x_floor = floor(sample_x);
                                int y_floor = floor(sample_y);

                                int x1 = x_floor;
                                int y1 = y_floor;
                                int x0 = x1 - 1;
                                int y0 = y1 - 1;
                                int x2 = x1 + 1;
                                int y2 = y1 + 1;
                                int x3 = x1 + 2;
                                int y3 = y1 + 2;

                                x0 = static_cast<int>(reflect_coord(x0 + 0.5, w) - 0.5);

                                y0 = static_cast<int>(reflect_coord(y0 + 0.5, h) - 0.5);

                                x0 = std::min(w - 1, std::max(x0, 0));
                                y0 = std::min(h - 1, std::max(y0, 0));

                                x1 = static_cast<int>(reflect_coord(x1 + 0.5, w) - 0.5);

                                y1 = static_cast<int>(reflect_coord(y1 + 0.5, h) - 0.5);

                                x1 = std::min(w - 1, std::max(x1, 0));
                                y1 = std::min(h - 1, std::max(y1, 0));

                                x2 = static_cast<int>(reflect_coord(x2 + 0.5, w) - 0.5);

                                y2 = static_cast<int>(reflect_coord(y2 + 0.5, h) - 0.5);

                                x2 = std::min(w - 1, std::max(x2, 0));
                                y2 = std::min(h - 1, std::max(y2, 0));

                                x3 = static_cast<int>(reflect_coord(x3 + 0.5, w) - 0.5);

                                y3 = static_cast<int>(reflect_coord(y3 + 0.5, h) - 0.5);

                                x3 = std::min(w - 1, std::max(x3, 0));
                                y3 = std::min(h - 1, std::max(y3, 0));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    float v00 = image.row(y0)[x0];
                                    float v01 = image.row(y0)[x1];
                                    float v02 = image.row(y0)[x2];
                                    float v03 = image.row(y0)[x3];
                                    float v10 = image.row(y1)[x0];
                                    float v11 = image.row(y1)[x1];
                                    float v12 = image.row(y1)[x2];
                                    float v13 = image.row(y1)[x3];
                                    float v20 = image.row(y2)[x0];
                                    float v21 = image.row(y2)[x1];
                                    float v22 = image.row(y2)[x2];
                                    float v23 = image.row(y2)[x3];
                                    float v30 = image.row(y3)[x0];
                                    float v31 = image.row(y3)[x1];
                                    float v32 = image.row(y3)[x2];
                                    float v33 = image.row(y3)[x3];

                                    float x_coeffs[4];
                                    float y_coeffs[4];
                                    interpolate_cubic(sample_x - x_floor, x_coeffs);
                                    interpolate_cubic(sample_y - y_floor, y_coeffs);

                                    float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                                    float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                                    float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                                    float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                                    top_blob.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 15 < grid_size; x += 16)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 gy = _mm256_loadu_ps(gridptr + x + 8);

                                __m256 gx = _mm256_permute2f128_ps(tmp_x, gy, 0b00100000);
                                gy = _mm256_permute2f128_ps(tmp_x, gy, 0b00110001);
                                tmp_x = _mm256_or_ps(gx, _mm256_setzero_ps());

                                gx = _mm256_shuffle_ps(gx, gy, 0b10001000);
                                gy = _mm256_shuffle_ps(tmp_x, gy, 0b11011101);

                                const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);
                                const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
                                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                __m256 gx_floor = _mm256_floor_ps(gx);
                                __m256 gy_floor = _mm256_floor_ps(gy);

                                const __m256 tx = _mm256_sub_ps(gx, gx_floor);
                                const __m256 ty = _mm256_sub_ps(gy, gy_floor);

                                __m256 coefficients[4];

                                __m256 gx0 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_n1);
                                __m256 gx1 = gx_floor;
                                __m256 gx2 = _mm256_add_ps(gx_floor, *(__m256*)_ps256_1);
                                __m256 gx3 = _mm256_add_ps(gx_floor, _mm256_set1_ps(2.0f));
                                {
                                    // x0
                                    gx0 = _mm256_and_ps(gx0, *(__m256*)_ps256_inv_sign_mask);
                                    __m256 reflectx0_v = _mm256_and_ps(_mm256_sub_ps(gx0, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx0 = _mm256_sub_ps(border_x, reflectx0_v);

                                    // x1
                                    gx1 = _mm256_and_ps(gx1, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx1_v = _mm256_and_ps(_mm256_sub_ps(gx1, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx1 = _mm256_sub_ps(border_x, reflectx1_v);

                                    // x2
                                    gx2 = _mm256_and_ps(gx2, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx2_v = _mm256_and_ps(_mm256_sub_ps(gx2, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx2 = _mm256_sub_ps(border_x, reflectx2_v);

                                    // x3
                                    gx3 = _mm256_and_ps(gx3, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx3_v = _mm256_and_ps(_mm256_sub_ps(gx3, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx3 = _mm256_sub_ps(border_x, reflectx3_v);
                                }

                                __m256i v0_offset[4], v1_offset[4], v2_offset[4], v3_offset[4];
                                for (int i = 0; i < 4; i++)
                                {
                                    gy = _mm256_add_ps(gy_floor, _mm256_set1_ps(-1.0f + i));

                                    {
                                        //y
                                        gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                        __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                                        gy = _mm256_sub_ps(border_y, reflecty_v);
                                    }

                                    __m256 v0_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx0);
                                    __m256 v1_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx1);
                                    __m256 v2_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx2);
                                    __m256 v3_offset_f = _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx3);

                                    v0_offset[i] = _mm256_cvtps_epi32(v0_offset_f);
                                    v1_offset[i] = _mm256_cvtps_epi32(v1_offset_f);
                                    v2_offset[i] = _mm256_cvtps_epi32(v2_offset_f);
                                    v3_offset[i] = _mm256_cvtps_epi32(v3_offset_f);
                                }

                                for (int q = 0; q < bottom_blob.c; q++)
                                {
                                    for (int i = 0; i < 4; i++)
                                    {
                                        __m256 x0_val = mask_gather_ps256(bottom_blob.channel(q), v0_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x1_val = mask_gather_ps256(bottom_blob.channel(q), v1_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x2_val = mask_gather_ps256(bottom_blob.channel(q), v2_offset[i], *(__m256*)_ps256_n1);
                                        __m256 x3_val = mask_gather_ps256(bottom_blob.channel(q), v3_offset[i], *(__m256*)_ps256_n1);

                                        coefficients[i] = cubic_interp1d_p8(x0_val, x1_val, x2_val, x3_val, tx);
                                    }

                                    __m256 _v = cubic_interp1d_p8(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);

                                    _mm256_storeu_ps(top_blob.channel(q).row(y) + x / 2, _v);
                                }
                            }

                            nn = grid_size & 15;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 2)
                            {
                                float sample_x = gridptr[x];
                                float sample_y = gridptr[x + 1];

                                sample_x = (sample_x + 1) / 2.f * (w - 1);
                                sample_y = (sample_y + 1) / 2.f * (h - 1);

                                int x_floor = floor(sample_x);
                                int y_floor = floor(sample_y);

                                int x1 = x_floor;
                                int y1 = y_floor;
                                int x0 = x1 - 1;
                                int y0 = y1 - 1;
                                int x2 = x1 + 1;
                                int y2 = y1 + 1;
                                int x3 = x1 + 2;
                                int y3 = y1 + 2;

                                x0 = static_cast<int>(reflect_coord(x0, w - 1));
                                y0 = static_cast<int>(reflect_coord(y0, h - 1));
                                x1 = static_cast<int>(reflect_coord(x1, w - 1));
                                y1 = static_cast<int>(reflect_coord(y1, h - 1));
                                x2 = static_cast<int>(reflect_coord(x2, w - 1));
                                y2 = static_cast<int>(reflect_coord(y2, h - 1));
                                x3 = static_cast<int>(reflect_coord(x3, w - 1));
                                y3 = static_cast<int>(reflect_coord(y3, h - 1));

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);

                                    float v00 = image.row(y0)[x0];
                                    float v01 = image.row(y0)[x1];
                                    float v02 = image.row(y0)[x2];
                                    float v03 = image.row(y0)[x3];
                                    float v10 = image.row(y1)[x0];
                                    float v11 = image.row(y1)[x1];
                                    float v12 = image.row(y1)[x2];
                                    float v13 = image.row(y1)[x3];
                                    float v20 = image.row(y2)[x0];
                                    float v21 = image.row(y2)[x1];
                                    float v22 = image.row(y2)[x2];
                                    float v23 = image.row(y2)[x3];
                                    float v30 = image.row(y3)[x0];
                                    float v31 = image.row(y3)[x1];
                                    float v32 = image.row(y3)[x2];
                                    float v33 = image.row(y3)[x3];

                                    float x_coeffs[4];
                                    float y_coeffs[4];
                                    interpolate_cubic(sample_x - x_floor, x_coeffs);
                                    interpolate_cubic(sample_y - y_floor, y_coeffs);

                                    float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                                    float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                                    float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                                    float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                                    top_blob.channel(q).row(y)[x / 2] = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                                }
                            }
                        }
                    }
                }
                else
                {
                    NCNN_LOGE("gridsample padding_mode error\n");
                    return -100;
                }
            }
        }

        if (dims == 4)
        {
#if __AVX__
            const __m256 vImgDf = _mm256_set1_ps(d);
#if __AVX2__
            const __m256i vImgDi = _mm256_set1_epi32(d);
#endif // __AVX2__
#endif // __AVX__
            int grid_size = grid_p1.w * grid_p1.h * grid_p1.d;

            top_blob.create(grid_p1.h, grid_p1.d, grid_p1.c, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                //upzip (3)
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // z
                                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);
                                __m256 z_t = _mm256_floor_ps(gz);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);
                                __m256 t = _mm256_sub_ps(gz, z_t);
                                __m256 b = _mm256_sub_ps(*(__m256*)_ps256_1, t);

                                __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                {
                                    __m256 nw = _mm256_mul_ps(s, e);
                                    __m256 ne = _mm256_mul_ps(s, w);
                                    __m256 sw = _mm256_mul_ps(n, e);
                                    __m256 se = _mm256_mul_ps(n, w);

                                    tnw = _mm256_mul_ps(b, nw);
                                    tne = _mm256_mul_ps(b, ne);
                                    tsw = _mm256_mul_ps(b, sw);
                                    tse = _mm256_mul_ps(b, se);

                                    bnw = _mm256_mul_ps(t, nw);
                                    bne = _mm256_mul_ps(t, ne);
                                    bsw = _mm256_mul_ps(t, sw);
                                    bse = _mm256_mul_ps(t, se);
                                }

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
                                    __m256i v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                                    __m256i v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                                    __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                    v000_in_range = _mm256_and_si256(v00_in_range, z0_in_range);
                                    v010_in_range = _mm256_and_si256(v01_in_range, z0_in_range);
                                    v100_in_range = _mm256_and_si256(v10_in_range, z0_in_range);
                                    v110_in_range = _mm256_and_si256(v11_in_range, z0_in_range);

                                    v001_in_range = _mm256_and_si256(v00_in_range, z1_in_range);
                                    v011_in_range = _mm256_and_si256(v01_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_si256(v10_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_si256(v11_in_range, z1_in_range);
                                }

                                __m256i i_tnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0));
                                __m256i i_tne_offset = _mm256_add_epi32(i_tnw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_tsw_offset = _mm256_add_epi32(i_tnw_offset, vImgWi);
                                __m256i i_tse_offset = _mm256_add_epi32(i_tsw_offset, *(__m256i*)_pi32_256_1);

                                __m256i i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), i_tnw_offset);
                                __m256i i_bne_offset = _mm256_add_epi32(i_bnw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_bsw_offset = _mm256_add_epi32(i_bnw_offset, vImgWi);
                                __m256i i_bse_offset = _mm256_add_epi32(i_bsw_offset, *(__m256i*)_pi32_256_1);
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
                                    __m256 v01_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                                    __m256 v10_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v000_in_range = _mm256_and_ps(v00_in_range, z0_in_range);
                                    v010_in_range = _mm256_and_ps(v01_in_range, z0_in_range);
                                    v100_in_range = _mm256_and_ps(v10_in_range, z0_in_range);
                                    v110_in_range = _mm256_and_ps(v11_in_range, z0_in_range);

                                    v001_in_range = _mm256_and_ps(v00_in_range, z1_in_range);
                                    v011_in_range = _mm256_and_ps(v01_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_ps(v10_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                                }

                                __m256 tnw_offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t),
                                                                  _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w));
                                __m256 tne_offset = _mm256_add_ps(tnw_offset, *(__m256*)_ps256_1);
                                __m256 tsw_offset = _mm256_add_ps(tnw_offset, vImgWf);
                                __m256 tse_offset = _mm256_add_ps(tsw_offset, *(__m256*)_ps256_1);

                                __m256 bnw_offset = _mm256_add_ps(_mm256_mul_ps(vImgWf, vImgHf), tnw_offset);
                                __m256 bne_offset = _mm256_add_ps(bnw_offset, *(__m256*)_ps256_1);
                                __m256 bsw_offset = _mm256_add_ps(bnw_offset, vImgWf);
                                __m256 bse_offset = _mm256_add_ps(bsw_offset, *(__m256*)_ps256_1);

                                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);
#endif // __AVX2__

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
#if __AVX2__
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, _mm256_castsi256_ps(v000_in_range));
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, _mm256_castsi256_ps(v100_in_range));
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, _mm256_castsi256_ps(v010_in_range));
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, _mm256_castsi256_ps(v110_in_range));

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, _mm256_castsi256_ps(v001_in_range));
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, _mm256_castsi256_ps(v101_in_range));
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, _mm256_castsi256_ps(v011_in_range));
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, _mm256_castsi256_ps(v111_in_range));
#else
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, v000_in_range);
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, v100_in_range);
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, v010_in_range);
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, v110_in_range);

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, v001_in_range);
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, v101_in_range);
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, v011_in_range);
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, v111_in_range);
#endif

                                    __m256 _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }

                            nn = grid_size % 24;

#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = ((gx + 1) * w - 1) / 2.f;
                                gy = ((gy + 1) * h - 1) / 2.f;
                                gz = ((gz + 1) * d - 1) / 2.f;

                                // bilinear interpolate
                                int x0 = (int)floor(gx);
                                int y0 = (int)floor(gy);
                                int z0 = (int)floor(gz);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                bool x0_in_range = (x0 > -1) & (x0 < w);
                                bool y0_in_range = (y0 > -1) & (y0 < h);
                                bool z0_in_range = (z0 > -1) & (z0 < d);
                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool z1_in_range = (z1 > -1) & (z1 < d);

                                bool v00_in_range = x0_in_range & y0_in_range;
                                bool v01_in_range = x1_in_range & y0_in_range;
                                bool v10_in_range = x0_in_range & y1_in_range;
                                bool v11_in_range = x1_in_range & y1_in_range;

                                bool v000_in_range = v00_in_range & z0_in_range;
                                bool v010_in_range = v10_in_range & z0_in_range;
                                bool v100_in_range = v00_in_range & z1_in_range;
                                bool v110_in_range = v10_in_range & z1_in_range;

                                bool v001_in_range = v01_in_range & z0_in_range;
                                bool v011_in_range = v11_in_range & z0_in_range;
                                bool v101_in_range = v01_in_range & z1_in_range;
                                bool v111_in_range = v11_in_range & z1_in_range;

                                float alpha = gx - x0;
                                float beta = gy - y0;
                                float gamma = gz - z0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v000 = v000_in_range ? image.depth(z0).row(y0)[x0] : 0;
                                    float v010 = v010_in_range ? image.depth(z0).row(y1)[x0] : 0;
                                    float v100 = v100_in_range ? image.depth(z1).row(y0)[x0] : 0;
                                    float v110 = v110_in_range ? image.depth(z1).row(y1)[x0] : 0;

                                    float v001 = v001_in_range ? image.depth(z0).row(y0)[x1] : 0;
                                    float v011 = v011_in_range ? image.depth(z0).row(y1)[x1] : 0;
                                    float v101 = v101_in_range ? image.depth(z1).row(y0)[x1] : 0;
                                    float v111 = v111_in_range ? image.depth(z1).row(y1)[x1] : 0;

                                    float v00 = v000 * (1 - alpha) + v001 * alpha;
                                    float v01 = v010 * (1 - alpha) + v011 * alpha;
                                    float v10 = v100 * (1 - alpha) + v101 * alpha;
                                    float v11 = v110 * (1 - alpha) + v111 * alpha;

                                    float v0 = v00 * (1 - beta) + v01 * beta;
                                    float v1 = v10 * (1 - beta) + v11 * beta;

                                    top_blob.channel(q).depth(y)[x / 3] = v0 * (1 - gamma) + v1 * gamma;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    // z
                                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);
                                __m256 z_t = _mm256_floor_ps(gz);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);
                                __m256 t = _mm256_sub_ps(gz, z_t);
                                __m256 b = _mm256_sub_ps(*(__m256*)_ps256_1, t);

                                __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                {
                                    __m256 nw = _mm256_mul_ps(s, e);
                                    __m256 ne = _mm256_mul_ps(s, w);
                                    __m256 sw = _mm256_mul_ps(n, e);
                                    __m256 se = _mm256_mul_ps(n, w);

                                    tnw = _mm256_mul_ps(b, nw);
                                    tne = _mm256_mul_ps(b, ne);
                                    tsw = _mm256_mul_ps(b, sw);
                                    tse = _mm256_mul_ps(b, se);

                                    bnw = _mm256_mul_ps(t, nw);
                                    bne = _mm256_mul_ps(t, ne);
                                    bsw = _mm256_mul_ps(t, sw);
                                    bse = _mm256_mul_ps(t, se);
                                }

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
                                    __m256i v01_in_range = _mm256_and_si256(x0_in_range, y1_in_range);
                                    __m256i v10_in_range = _mm256_and_si256(x1_in_range, y0_in_range);
                                    __m256i v11_in_range = _mm256_and_si256(x1_in_range, y1_in_range);

                                    v000_in_range = _mm256_and_si256(v00_in_range, z0_in_range);
                                    v010_in_range = _mm256_and_si256(v01_in_range, z0_in_range);
                                    v100_in_range = _mm256_and_si256(v10_in_range, z0_in_range);
                                    v110_in_range = _mm256_and_si256(v11_in_range, z0_in_range);

                                    v001_in_range = _mm256_and_si256(v00_in_range, z1_in_range);
                                    v011_in_range = _mm256_and_si256(v01_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_si256(v10_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_si256(v11_in_range, z1_in_range);
                                }

                                __m256i i_tnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), z0), _mm256_add_epi32(_mm256_mullo_epi32(y0, vImgWi), x0));
                                __m256i i_tne_offset = _mm256_add_epi32(i_tnw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_tsw_offset = _mm256_add_epi32(i_tnw_offset, vImgWi);
                                __m256i i_tse_offset = _mm256_add_epi32(i_tsw_offset, *(__m256i*)_pi32_256_1);

                                __m256i i_bnw_offset = _mm256_add_epi32(_mm256_mullo_epi32(vImgWi, vImgHi), i_tnw_offset);
                                __m256i i_bne_offset = _mm256_add_epi32(i_bnw_offset, *(__m256i*)_pi32_256_1);
                                __m256i i_bsw_offset = _mm256_add_epi32(i_bnw_offset, vImgWi);
                                __m256i i_bse_offset = _mm256_add_epi32(i_bsw_offset, *(__m256i*)_pi32_256_1);
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
                                    __m256 v01_in_range = _mm256_and_ps(x0_in_range, y1_in_range);
                                    __m256 v10_in_range = _mm256_and_ps(x1_in_range, y0_in_range);
                                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v000_in_range = _mm256_and_ps(v00_in_range, z0_in_range);
                                    v010_in_range = _mm256_and_ps(v01_in_range, z0_in_range);
                                    v100_in_range = _mm256_and_ps(v10_in_range, z0_in_range);
                                    v110_in_range = _mm256_and_ps(v11_in_range, z0_in_range);

                                    v001_in_range = _mm256_and_ps(v00_in_range, z1_in_range);
                                    v011_in_range = _mm256_and_ps(v01_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_ps(v10_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                                }

                                __m256 tnw_offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t),
                                                                  _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w));
                                __m256 tne_offset = _mm256_add_ps(tnw_offset, *(__m256*)_ps256_1);
                                __m256 tsw_offset = _mm256_add_ps(tnw_offset, vImgWf);
                                __m256 tse_offset = _mm256_add_ps(tsw_offset, *(__m256*)_ps256_1);

                                __m256 bnw_offset = _mm256_add_ps(_mm256_mul_ps(vImgWf, vImgHf), tnw_offset);
                                __m256 bne_offset = _mm256_add_ps(bnw_offset, *(__m256*)_ps256_1);
                                __m256 bsw_offset = _mm256_add_ps(bnw_offset, vImgWf);
                                __m256 bse_offset = _mm256_add_ps(bsw_offset, *(__m256*)_ps256_1);

                                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);
#endif // __AVX2__

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
#if __AVX2__
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, _mm256_castsi256_ps(v000_in_range));
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, _mm256_castsi256_ps(v100_in_range));
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, _mm256_castsi256_ps(v010_in_range));
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, _mm256_castsi256_ps(v110_in_range));

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, _mm256_castsi256_ps(v001_in_range));
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, _mm256_castsi256_ps(v101_in_range));
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, _mm256_castsi256_ps(v011_in_range));
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, _mm256_castsi256_ps(v111_in_range));
#else
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, v000_in_range);
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, v100_in_range);
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, v010_in_range);
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, v110_in_range);

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, v001_in_range);
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, v101_in_range);
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, v011_in_range);
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, v111_in_range);
#endif

                                    __m256 _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = (gx + 1) / 2.f * (w - 1);
                                gy = (gy + 1) / 2.f * (h - 1);
                                gz = (gz + 1) / 2.f * (d - 1);

                                // bilinear interpolate
                                int x0 = (int)floor(gx);
                                int y0 = (int)floor(gy);
                                int z0 = (int)floor(gz);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                bool x0_in_range = (x0 > -1) & (x0 < w);
                                bool y0_in_range = (y0 > -1) & (y0 < h);
                                bool z0_in_range = (z0 > -1) & (z0 < d);
                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool z1_in_range = (z1 > -1) & (z1 < d);

                                bool v00_in_range = x0_in_range & y0_in_range;
                                bool v01_in_range = x1_in_range & y0_in_range;
                                bool v10_in_range = x0_in_range & y1_in_range;
                                bool v11_in_range = x1_in_range & y1_in_range;

                                bool v000_in_range = v00_in_range & z0_in_range;
                                bool v010_in_range = v10_in_range & z0_in_range;
                                bool v100_in_range = v00_in_range & z1_in_range;
                                bool v110_in_range = v10_in_range & z1_in_range;

                                bool v001_in_range = v01_in_range & z0_in_range;
                                bool v011_in_range = v11_in_range & z0_in_range;
                                bool v101_in_range = v01_in_range & z1_in_range;
                                bool v111_in_range = v11_in_range & z1_in_range;

                                float alpha = gx - x0;
                                float beta = gy - y0;
                                float gamma = gz - z0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v000 = v000_in_range ? image.depth(z0).row(y0)[x0] : 0;
                                    float v010 = v010_in_range ? image.depth(z0).row(y1)[x0] : 0;
                                    float v100 = v100_in_range ? image.depth(z1).row(y0)[x0] : 0;
                                    float v110 = v110_in_range ? image.depth(z1).row(y1)[x0] : 0;

                                    float v001 = v001_in_range ? image.depth(z0).row(y0)[x1] : 0;
                                    float v011 = v011_in_range ? image.depth(z0).row(y1)[x1] : 0;
                                    float v101 = v101_in_range ? image.depth(z1).row(y0)[x1] : 0;
                                    float v111 = v111_in_range ? image.depth(z1).row(y1)[x1] : 0;

                                    float v00 = v000 * (1 - alpha) + v001 * alpha;
                                    float v01 = v010 * (1 - alpha) + v011 * alpha;
                                    float v10 = v100 * (1 - alpha) + v101 * alpha;
                                    float v11 = v110 * (1 - alpha) + v111 * alpha;

                                    float v0 = v00 * (1 - beta) + v01 * beta;
                                    float v1 = v10 * (1 - beta) + v11 * beta;

                                    top_blob.channel(q).depth(y)[x / 3] = v0 * (1 - gamma) + v1 * gamma;
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                                    // z
                                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);
                                __m256 z_t = _mm256_floor_ps(gz);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);
                                __m256 t = _mm256_sub_ps(gz, z_t);
                                __m256 b = _mm256_sub_ps(*(__m256*)_ps256_1, t);

                                __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                {
                                    __m256 nw = _mm256_mul_ps(s, e);
                                    __m256 ne = _mm256_mul_ps(s, w);
                                    __m256 sw = _mm256_mul_ps(n, e);
                                    __m256 se = _mm256_mul_ps(n, w);

                                    tnw = _mm256_mul_ps(b, nw);
                                    tne = _mm256_mul_ps(b, ne);
                                    tsw = _mm256_mul_ps(b, sw);
                                    tse = _mm256_mul_ps(b, se);

                                    bnw = _mm256_mul_ps(t, nw);
                                    bne = _mm256_mul_ps(t, ne);
                                    bsw = _mm256_mul_ps(t, sw);
                                    bse = _mm256_mul_ps(t, se);
                                }

                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                                __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

                                __m256 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                                {
                                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v110_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v011_in_range = _mm256_and_ps(y1_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_ps(x1_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                                }

                                __m256 tnw_offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t),
                                                                  _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w));
                                __m256 tne_offset = _mm256_add_ps(tnw_offset, *(__m256*)_ps256_1);
                                __m256 tsw_offset = _mm256_add_ps(tnw_offset, vImgWf);
                                __m256 tse_offset = _mm256_add_ps(tsw_offset, *(__m256*)_ps256_1);

                                __m256 bnw_offset = _mm256_add_ps(_mm256_mul_ps(vImgWf, vImgHf), tnw_offset);
                                __m256 bne_offset = _mm256_add_ps(bnw_offset, *(__m256*)_ps256_1);
                                __m256 bsw_offset = _mm256_add_ps(bnw_offset, vImgWf);
                                __m256 bse_offset = _mm256_add_ps(bsw_offset, *(__m256*)_ps256_1);

                                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, *(__m256*)_ps256_n1);
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, x1_in_range);
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, y1_in_range);
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, v110_in_range);

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, z1_in_range);
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, v101_in_range);
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, v011_in_range);
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, v111_in_range);

                                    __m256 _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = ((gx + 1) * w - 1) / 2.f;
                                gy = ((gy + 1) * h - 1) / 2.f;
                                gz = ((gz + 1) * d - 1) / 2.f;

                                gx = std::min(w - 1.0f, std::max(gx, 0.0f));
                                gy = std::min(h - 1.0f, std::max(gy, 0.0f));
                                gz = std::min(d - 1.0f, std::max(gz, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(gx);
                                int y0 = (int)floor(gy);
                                int z0 = (int)floor(gz);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool z1_in_range = (z1 > -1) & (z1 < d);

                                bool v11_in_range = x1_in_range & y1_in_range;

                                bool v110_in_range = y1_in_range & z1_in_range;

                                bool v101_in_range = x1_in_range & z1_in_range;
                                bool v111_in_range = v11_in_range & z1_in_range;

                                float alpha = gx - x0;
                                float beta = gy - y0;
                                float gamma = gz - z0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v000 = image.depth(z0).row(y0)[x0];
                                    float v010 = y1_in_range ? image.depth(z0).row(y1)[x0] : 0;
                                    float v100 = z1_in_range ? image.depth(z1).row(y0)[x0] : 0;
                                    float v110 = v110_in_range ? image.depth(z1).row(y1)[x0] : 0;

                                    float v001 = x1_in_range ? image.depth(z0).row(y0)[x1] : 0;
                                    float v011 = v11_in_range ? image.depth(z0).row(y1)[x1] : 0;
                                    float v101 = v101_in_range ? image.depth(z1).row(y0)[x1] : 0;
                                    float v111 = v111_in_range ? image.depth(z1).row(y1)[x1] : 0;

                                    float v00 = v000 * (1 - alpha) + v001 * alpha;
                                    float v01 = v010 * (1 - alpha) + v011 * alpha;
                                    float v10 = v100 * (1 - alpha) + v101 * alpha;
                                    float v11 = v110 * (1 - alpha) + v111 * alpha;

                                    float v0 = v00 * (1 - beta) + v01 * beta;
                                    float v1 = v10 * (1 - beta) + v11 * beta;

                                    top_blob.channel(q).depth(y)[x / 3] = v0 * (1 - gamma) + v1 * gamma;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                                    // z
                                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));

                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);
                                __m256 z_t = _mm256_floor_ps(gz);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);
                                __m256 t = _mm256_sub_ps(gz, z_t);
                                __m256 b = _mm256_sub_ps(*(__m256*)_ps256_1, t);

                                __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                {
                                    __m256 nw = _mm256_mul_ps(s, e);
                                    __m256 ne = _mm256_mul_ps(s, w);
                                    __m256 sw = _mm256_mul_ps(n, e);
                                    __m256 se = _mm256_mul_ps(n, w);

                                    tnw = _mm256_mul_ps(b, nw);
                                    tne = _mm256_mul_ps(b, ne);
                                    tsw = _mm256_mul_ps(b, sw);
                                    tse = _mm256_mul_ps(b, se);

                                    bnw = _mm256_mul_ps(t, nw);
                                    bne = _mm256_mul_ps(t, ne);
                                    bsw = _mm256_mul_ps(t, sw);
                                    bse = _mm256_mul_ps(t, se);
                                }

                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                                __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

                                __m256 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                                {
                                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v110_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v011_in_range = _mm256_and_ps(y1_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_ps(x1_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                                }

                                __m256 tnw_offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t),
                                                                  _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w));
                                __m256 tne_offset = _mm256_add_ps(tnw_offset, *(__m256*)_ps256_1);
                                __m256 tsw_offset = _mm256_add_ps(tnw_offset, vImgWf);
                                __m256 tse_offset = _mm256_add_ps(tsw_offset, *(__m256*)_ps256_1);

                                __m256 bnw_offset = _mm256_add_ps(_mm256_mul_ps(vImgWf, vImgHf), tnw_offset);
                                __m256 bne_offset = _mm256_add_ps(bnw_offset, *(__m256*)_ps256_1);
                                __m256 bsw_offset = _mm256_add_ps(bnw_offset, vImgWf);
                                __m256 bse_offset = _mm256_add_ps(bsw_offset, *(__m256*)_ps256_1);

                                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, *(__m256*)_ps256_n1);
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, x1_in_range);
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, y1_in_range);
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, v110_in_range);

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, z1_in_range);
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, v101_in_range);
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, v011_in_range);
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, v111_in_range);

                                    __m256 _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = (gx + 1) / 2.f * (w - 1);
                                gy = (gy + 1) / 2.f * (h - 1);
                                gz = (gz + 1) / 2.f * (d - 1);

                                gx = std::min(w - 1.0f, std::max(gx, 0.0f));
                                gy = std::min(h - 1.0f, std::max(gy, 0.0f));
                                gz = std::min(d - 1.0f, std::max(gz, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(gx);
                                int y0 = (int)floor(gy);
                                int z0 = (int)floor(gz);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool z1_in_range = (z1 > -1) & (z1 < d);

                                bool v11_in_range = x1_in_range & y1_in_range;

                                bool v110_in_range = y1_in_range & z1_in_range;

                                bool v101_in_range = x1_in_range & z1_in_range;
                                bool v111_in_range = v11_in_range & z1_in_range;

                                float alpha = gx - x0;
                                float beta = gy - y0;
                                float gamma = gz - z0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v000 = image.depth(z0).row(y0)[x0];
                                    float v010 = y1_in_range ? image.depth(z0).row(y1)[x0] : 0;
                                    float v100 = z1_in_range ? image.depth(z1).row(y0)[x0] : 0;
                                    float v110 = v110_in_range ? image.depth(z1).row(y1)[x0] : 0;

                                    float v001 = x1_in_range ? image.depth(z0).row(y0)[x1] : 0;
                                    float v011 = v11_in_range ? image.depth(z0).row(y1)[x1] : 0;
                                    float v101 = v101_in_range ? image.depth(z1).row(y0)[x1] : 0;
                                    float v111 = v111_in_range ? image.depth(z1).row(y1)[x1] : 0;

                                    float v00 = v000 * (1 - alpha) + v001 * alpha;
                                    float v01 = v010 * (1 - alpha) + v011 * alpha;
                                    float v10 = v100 * (1 - alpha) + v101 * alpha;
                                    float v11 = v110 * (1 - alpha) + v111 * alpha;

                                    float v0 = v00 * (1 - beta) + v01 * beta;
                                    float v1 = v10 * (1 - beta) + v11 * beta;

                                    top_blob.channel(q).depth(y)[x / 3] = v0 * (1 - gamma) + v1 * gamma;
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                                    gx = _mm256_add_ps(gx, v0p5fp8);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(vImgWf, reflectx_v);

                                    gx = _mm256_sub_ps(gx, v0p5fp8);

                                    _mm256_sub_ps(gx, v0p5fp8);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_add_ps(gy, v0p5fp8);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                                    gy = _mm256_sub_ps(gy, v0p5fp8);

                                    _mm256_sub_ps(gy, v0p5fp8);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                                    // z
                                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_add_ps(gz, v0p5fp8);

                                    gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, vImgDf), *(__m256*)_ps256_inv_sign_mask);
                                    gz = _mm256_sub_ps(vImgDf, reflectz_v);

                                    gz = _mm256_sub_ps(gz, v0p5fp8);

                                    _mm256_sub_ps(gz, v0p5fp8);

                                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);
                                __m256 z_t = _mm256_floor_ps(gz);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);
                                __m256 t = _mm256_sub_ps(gz, z_t);
                                __m256 b = _mm256_sub_ps(*(__m256*)_ps256_1, t);

                                __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                {
                                    __m256 nw = _mm256_mul_ps(s, e);
                                    __m256 ne = _mm256_mul_ps(s, w);
                                    __m256 sw = _mm256_mul_ps(n, e);
                                    __m256 se = _mm256_mul_ps(n, w);

                                    tnw = _mm256_mul_ps(b, nw);
                                    tne = _mm256_mul_ps(b, ne);
                                    tsw = _mm256_mul_ps(b, sw);
                                    tse = _mm256_mul_ps(b, se);

                                    bnw = _mm256_mul_ps(t, nw);
                                    bne = _mm256_mul_ps(t, ne);
                                    bsw = _mm256_mul_ps(t, sw);
                                    bse = _mm256_mul_ps(t, se);
                                }

                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                                __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

                                __m256 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                                {
                                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v110_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v011_in_range = _mm256_and_ps(y1_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_ps(x1_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                                }

                                __m256 tnw_offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t),
                                                                  _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w));
                                __m256 tne_offset = _mm256_add_ps(tnw_offset, *(__m256*)_ps256_1);
                                __m256 tsw_offset = _mm256_add_ps(tnw_offset, vImgWf);
                                __m256 tse_offset = _mm256_add_ps(tsw_offset, *(__m256*)_ps256_1);

                                __m256 bnw_offset = _mm256_add_ps(_mm256_mul_ps(vImgWf, vImgHf), tnw_offset);
                                __m256 bne_offset = _mm256_add_ps(bnw_offset, *(__m256*)_ps256_1);
                                __m256 bsw_offset = _mm256_add_ps(bnw_offset, vImgWf);
                                __m256 bse_offset = _mm256_add_ps(bsw_offset, *(__m256*)_ps256_1);

                                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, *(__m256*)_ps256_n1);
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, x1_in_range);
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, y1_in_range);
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, v110_in_range);

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, z1_in_range);
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, v101_in_range);
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, v011_in_range);
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, v111_in_range);

                                    __m256 _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = ((gx + 1) * w - 1) / 2.f;
                                gy = ((gy + 1) * h - 1) / 2.f;
                                gz = ((gz + 1) * d - 1) / 2.f;

                                gx = abs(gx + 0.5f);
                                gx = w - abs(gx - w) - 0.5;

                                gy = abs(gy + 0.5f);
                                gy = h - abs(gy - h) - 0.5;

                                gz = abs(gz + 0.5f);
                                gz = d - abs(gz - d) - 0.5;

                                gx = std::min(w - 1.0f, std::max(gx, 0.0f));
                                gy = std::min(h - 1.0f, std::max(gy, 0.0f));
                                gz = std::min(d - 1.0f, std::max(gz, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(gx);
                                int y0 = (int)floor(gy);
                                int z0 = (int)floor(gz);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool z1_in_range = (z1 > -1) & (z1 < d);

                                bool v11_in_range = x1_in_range & y1_in_range;

                                bool v110_in_range = y1_in_range & z1_in_range;

                                bool v101_in_range = x1_in_range & z1_in_range;
                                bool v111_in_range = v11_in_range & z1_in_range;

                                float alpha = gx - x0;
                                float beta = gy - y0;
                                float gamma = gz - z0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v000 = image.depth(z0).row(y0)[x0];
                                    float v010 = y1_in_range ? image.depth(z0).row(y1)[x0] : 0;
                                    float v100 = z1_in_range ? image.depth(z1).row(y0)[x0] : 0;
                                    float v110 = v110_in_range ? image.depth(z1).row(y1)[x0] : 0;

                                    float v001 = x1_in_range ? image.depth(z0).row(y0)[x1] : 0;
                                    float v011 = v11_in_range ? image.depth(z0).row(y1)[x1] : 0;
                                    float v101 = v101_in_range ? image.depth(z1).row(y0)[x1] : 0;
                                    float v111 = v111_in_range ? image.depth(z1).row(y1)[x1] : 0;

                                    float v00 = v000 * (1 - alpha) + v001 * alpha;
                                    float v01 = v010 * (1 - alpha) + v011 * alpha;
                                    float v10 = v100 * (1 - alpha) + v101 * alpha;
                                    float v11 = v110 * (1 - alpha) + v111 * alpha;

                                    float v0 = v00 * (1 - beta) + v01 * beta;
                                    float v1 = v10 * (1 - beta) + v11 * beta;

                                    top_blob.channel(q).depth(y)[x / 3] = v0 * (1 - gamma) + v1 * gamma;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(border_x, reflectx_v);

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(border_y, reflecty_v);

                                    // z
                                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));
                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, border_z), *(__m256*)_ps256_inv_sign_mask);
                                    gz = _mm256_sub_ps(border_z, reflectz_v);
                                }

                                __m256 x_w = _mm256_floor_ps(gx);
                                __m256 y_n = _mm256_floor_ps(gy);
                                __m256 z_t = _mm256_floor_ps(gz);

                                __m256 w = _mm256_sub_ps(gx, x_w);
                                __m256 e = _mm256_sub_ps(*(__m256*)_ps256_1, w);
                                __m256 n = _mm256_sub_ps(gy, y_n);
                                __m256 s = _mm256_sub_ps(*(__m256*)_ps256_1, n);
                                __m256 t = _mm256_sub_ps(gz, z_t);
                                __m256 b = _mm256_sub_ps(*(__m256*)_ps256_1, t);

                                __m256 tnw, tne, tsw, tse, bnw, bne, bsw, bse;
                                {
                                    __m256 nw = _mm256_mul_ps(s, e);
                                    __m256 ne = _mm256_mul_ps(s, w);
                                    __m256 sw = _mm256_mul_ps(n, e);
                                    __m256 se = _mm256_mul_ps(n, w);

                                    tnw = _mm256_mul_ps(b, nw);
                                    tne = _mm256_mul_ps(b, ne);
                                    tsw = _mm256_mul_ps(b, sw);
                                    tse = _mm256_mul_ps(b, se);

                                    bnw = _mm256_mul_ps(t, nw);
                                    bne = _mm256_mul_ps(t, ne);
                                    bsw = _mm256_mul_ps(t, sw);
                                    bse = _mm256_mul_ps(t, se);
                                }

                                __m256 x1 = _mm256_add_ps(x_w, *(__m256*)_ps256_1);
                                __m256 y1 = _mm256_add_ps(y_n, *(__m256*)_ps256_1);
                                __m256 z1 = _mm256_add_ps(z_t, *(__m256*)_ps256_1);

                                __m256 x1_in_range = _mm256_and_ps(_mm256_cmp_ps(x1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, x1, _CMP_GT_OS));
                                __m256 y1_in_range = _mm256_and_ps(_mm256_cmp_ps(y1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, y1, _CMP_GT_OS));
                                __m256 z1_in_range = _mm256_and_ps(_mm256_cmp_ps(z1, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, z1, _CMP_GT_OS));

                                __m256 v110_in_range, v011_in_range, v101_in_range, v111_in_range;
                                {
                                    __m256 v11_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v110_in_range = _mm256_and_ps(x1_in_range, y1_in_range);

                                    v011_in_range = _mm256_and_ps(y1_in_range, z1_in_range);
                                    v101_in_range = _mm256_and_ps(x1_in_range, z1_in_range);
                                    v111_in_range = _mm256_and_ps(v11_in_range, z1_in_range);
                                }

                                __m256 tnw_offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), z_t),
                                                                  _mm256_add_ps(_mm256_mul_ps(y_n, vImgWf), x_w));
                                __m256 tne_offset = _mm256_add_ps(tnw_offset, *(__m256*)_ps256_1);
                                __m256 tsw_offset = _mm256_add_ps(tnw_offset, vImgWf);
                                __m256 tse_offset = _mm256_add_ps(tsw_offset, *(__m256*)_ps256_1);

                                __m256 bnw_offset = _mm256_add_ps(_mm256_mul_ps(vImgWf, vImgHf), tnw_offset);
                                __m256 bne_offset = _mm256_add_ps(bnw_offset, *(__m256*)_ps256_1);
                                __m256 bsw_offset = _mm256_add_ps(bnw_offset, vImgWf);
                                __m256 bse_offset = _mm256_add_ps(bsw_offset, *(__m256*)_ps256_1);

                                __m256i i_tnw_offset = _mm256_cvtps_epi32(tnw_offset);
                                __m256i i_tne_offset = _mm256_cvtps_epi32(tne_offset);
                                __m256i i_tsw_offset = _mm256_cvtps_epi32(tsw_offset);
                                __m256i i_tse_offset = _mm256_cvtps_epi32(tse_offset);

                                __m256i i_bnw_offset = _mm256_cvtps_epi32(bnw_offset);
                                __m256i i_bne_offset = _mm256_cvtps_epi32(bne_offset);
                                __m256i i_bsw_offset = _mm256_cvtps_epi32(bsw_offset);
                                __m256i i_bse_offset = _mm256_cvtps_epi32(bse_offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    __m256 tnw_val = mask_gather_ps256(image, i_tnw_offset, *(__m256*)_ps256_n1);
                                    __m256 tne_val = mask_gather_ps256(image, i_tne_offset, x1_in_range);
                                    __m256 tsw_val = mask_gather_ps256(image, i_tsw_offset, y1_in_range);
                                    __m256 tse_val = mask_gather_ps256(image, i_tse_offset, v110_in_range);

                                    __m256 bnw_val = mask_gather_ps256(image, i_bnw_offset, z1_in_range);
                                    __m256 bne_val = mask_gather_ps256(image, i_bne_offset, v101_in_range);
                                    __m256 bsw_val = mask_gather_ps256(image, i_bsw_offset, v011_in_range);
                                    __m256 bse_val = mask_gather_ps256(image, i_bse_offset, v111_in_range);

                                    __m256 _v = _mm256_mul_ps(tnw_val, tnw);
                                    _v = _mm256_comp_fmadd_ps(tne_val, tne, _v);
                                    _v = _mm256_comp_fmadd_ps(tsw_val, tsw, _v);
                                    _v = _mm256_comp_fmadd_ps(tse_val, tse, _v);

                                    _v = _mm256_comp_fmadd_ps(bnw_val, bnw, _v);
                                    _v = _mm256_comp_fmadd_ps(bne_val, bne, _v);
                                    _v = _mm256_comp_fmadd_ps(bsw_val, bsw, _v);
                                    _v = _mm256_comp_fmadd_ps(bse_val, bse, _v);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = (gx + 1) / 2.f * (w - 1);
                                gy = (gy + 1) / 2.f * (h - 1);
                                gz = (gz + 1) / 2.f * (d - 1);

                                gx = abs(gx);
                                gx = (w - 1) - abs(gx - (w - 1));

                                gy = abs(gy);
                                gy = (h - 1) - abs(gy - (h - 1));

                                gz = abs(gz);
                                gz = (d - 1) - abs(gz - (d - 1));

                                gx = std::min(w - 1.0f, std::max(gx, 0.0f));
                                gy = std::min(h - 1.0f, std::max(gy, 0.0f));
                                gz = std::min(d - 1.0f, std::max(gz, 0.0f));

                                // bilinear interpolate
                                int x0 = (int)floor(gx);
                                int y0 = (int)floor(gy);
                                int z0 = (int)floor(gz);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                bool x1_in_range = (x1 > -1) & (x1 < w);
                                bool y1_in_range = (y1 > -1) & (y1 < h);
                                bool z1_in_range = (z1 > -1) & (z1 < d);

                                bool v11_in_range = x1_in_range & y1_in_range;

                                bool v110_in_range = y1_in_range & z1_in_range;

                                bool v101_in_range = x1_in_range & z1_in_range;
                                bool v111_in_range = v11_in_range & z1_in_range;

                                float alpha = gx - x0;
                                float beta = gy - y0;
                                float gamma = gz - z0;

                                for (int q = 0; q < channels; q++)
                                {
                                    const Mat& image = bottom_blob.channel(q);
                                    float v000 = image.depth(z0).row(y0)[x0];
                                    float v010 = y1_in_range ? image.depth(z0).row(y1)[x0] : 0;
                                    float v100 = z1_in_range ? image.depth(z1).row(y0)[x0] : 0;
                                    float v110 = v110_in_range ? image.depth(z1).row(y1)[x0] : 0;

                                    float v001 = x1_in_range ? image.depth(z0).row(y0)[x1] : 0;
                                    float v011 = v11_in_range ? image.depth(z0).row(y1)[x1] : 0;
                                    float v101 = v101_in_range ? image.depth(z1).row(y0)[x1] : 0;
                                    float v111 = v111_in_range ? image.depth(z1).row(y1)[x1] : 0;

                                    float v00 = v000 * (1 - alpha) + v001 * alpha;
                                    float v01 = v010 * (1 - alpha) + v011 * alpha;
                                    float v10 = v100 * (1 - alpha) + v101 * alpha;
                                    float v11 = v110 * (1 - alpha) + v111 * alpha;

                                    float v0 = v00 * (1 - beta) + v01 * beta;
                                    float v1 = v10 * (1 - beta) + v11 * beta;

                                    top_blob.channel(q).depth(y)[x / 3] = v0 * (1 - gamma) + v1 * gamma;
                                }
                            }
                        }
                    }
                }
            }
            else if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                //upzip (3)
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    // z
                                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));
                                v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, gz, _CMP_GT_OS)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                                              _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, v_in_range);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }

                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = ((gx + 1) * w - 1) / 2.f;
                                gy = ((gy + 1) * h - 1) / 2.f;
                                gz = ((gz + 1) * d - 1) / 2.f;

                                // bilinear interpolate
                                int x0 = static_cast<int>(floor(gx + 0.5f));
                                int y0 = static_cast<int>(floor(gy + 0.5f));
                                int z0 = static_cast<int>(floor(gz + 0.5f));

                                bool v_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h) && (z0 > -1) && (z0 < bottom_blob.d);

                                for (int q = 0; q < channels; q++)
                                {
                                    top_blob.channel(q).depth(y)[x / 3] = v_in_range ? bottom_blob.channel(q).depth(z0).row(y0)[x0] : 0;
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    // z
                                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                __m256 v_in_range = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(gx, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgWf, gx, _CMP_GT_OS)),
                                                                  _mm256_and_ps(_mm256_cmp_ps(gy, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgHf, gy, _CMP_GT_OS)));
                                v_in_range = _mm256_and_ps(v_in_range, _mm256_and_ps(_mm256_cmp_ps(gz, *(__m256*)_ps256_n1, _CMP_GT_OS), _mm256_cmp_ps(vImgDf, gz, _CMP_GT_OS)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                                              _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, v_in_range);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = (gx + 1) / 2.f * (w - 1);
                                gy = (gy + 1) / 2.f * (h - 1);
                                gz = (gz + 1) / 2.f * (d - 1);

                                int x0 = static_cast<int>(floor(gx + 0.5f));
                                int y0 = static_cast<int>(floor(gy + 0.5f));
                                int z0 = static_cast<int>(floor(gz + 0.5f));

                                bool v_in_range = (x0 > -1) & (x0 < bottom_blob.w) & (y0 > -1) & (y0 < bottom_blob.h) && (z0 > -1) && (z0 < bottom_blob.d);

                                for (int q = 0; q < channels; q++)
                                {
                                    top_blob.channel(q).depth(y)[x / 3] = v_in_range ? bottom_blob.channel(q).depth(z0).row(y0)[x0] : 0;
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                                    // z
                                    gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                                              _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = ((gx + 1) * w - 1) / 2.f;
                                gy = ((gy + 1) * h - 1) / 2.f;
                                gz = ((gz + 1) * d - 1) / 2.f;

                                gx = std::min(w - 1.0f, std::max(gx, 0.0f));
                                gy = std::min(h - 1.0f, std::max(gy, 0.0f));
                                gz = std::min(d - 1.0f, std::max(gz, 0.0f));

                                int x0 = static_cast<int>(floor(gx + 0.5f));
                                int y0 = static_cast<int>(floor(gy + 0.5f));
                                int z0 = static_cast<int>(floor(gz + 0.5f));

                                for (int q = 0; q < channels; q++)
                                {
                                    top_blob.channel(q).depth(y)[x / 3] = bottom_blob.channel(q).depth(z0).row(y0)[x0];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

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
                                    // x
                                    gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));

                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));

                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                                    // z
                                    gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));

                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                                }

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                                              _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = (gx + 1) / 2.f * (w - 1);
                                gy = (gy + 1) / 2.f * (h - 1);
                                gz = (gz + 1) / 2.f * (d - 1);

                                gx = std::min(w - 1.0f, std::max(gx, 0.0f));
                                gy = std::min(h - 1.0f, std::max(gy, 0.0f));
                                gz = std::min(d - 1.0f, std::max(gz, 0.0f));

                                int x0 = static_cast<int>(floor(gx + 0.5f));
                                int y0 = static_cast<int>(floor(gy + 0.5f));
                                int z0 = static_cast<int>(floor(gz + 0.5f));

                                for (int q = 0; q < channels; q++)
                                {
                                    top_blob.channel(q).depth(y)[x / 3] = bottom_blob.channel(q).depth(z0).row(y0)[x0];
                                }
                            }
                        }
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

                                __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
                                __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
                                gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

                                tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
                                tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

                                gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
                                gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
                                gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

                                gx = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), vImgWf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                gy = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), vImgHf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
                                gz = _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), vImgDf, *(__m256*)_ps256_1), *(__m256*)_ps256_2);

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                {
                                    // x
                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
                                    gx = _mm256_add_ps(gx, v0p5fp8);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, vImgWf), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(vImgWf, reflectx_v);

                                    gx = _mm256_sub_ps(gx, v0p5fp8);

                                    _mm256_sub_ps(gx, v0p5fp8);

                                    gx = _mm256_min_ps(border_x, _mm256_max_ps(gx, _mm256_setzero_ps()));

                                    // y
                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_add_ps(gy, v0p5fp8);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, vImgHf), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(vImgHf, reflecty_v);

                                    gy = _mm256_sub_ps(gy, v0p5fp8);

                                    _mm256_sub_ps(gy, v0p5fp8);

                                    gy = _mm256_min_ps(border_y, _mm256_max_ps(gy, _mm256_setzero_ps()));

                                    // z
                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_add_ps(gz, v0p5fp8);

                                    gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, vImgDf), *(__m256*)_ps256_inv_sign_mask);
                                    gz = _mm256_sub_ps(vImgDf, reflectz_v);

                                    gz = _mm256_sub_ps(gz, v0p5fp8);

                                    _mm256_sub_ps(gz, v0p5fp8);

                                    gz = _mm256_min_ps(border_z, _mm256_max_ps(gz, _mm256_setzero_ps()));
                                }

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                                              _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = ((gx + 1) * w - 1) / 2.f;
                                gy = ((gy + 1) * h - 1) / 2.f;
                                gz = ((gz + 1) * d - 1) / 2.f;

                                gx = floor(gx + 0.5f);
                                gy = floor(gy + 0.5f);
                                gz = floor(gz + 0.5f);

                                gx = abs(gx + 0.5f);
                                gx = w - abs(gx - w) - 0.5;

                                gy = abs(gy + 0.5f);
                                gy = h - abs(gy - h) - 0.5;

                                gz = abs(gz + 0.5f);
                                gz = d - abs(gz - d) - 0.5;

                                int x0 = std::min(w - 1.0f, std::max(gx, 0.0f));
                                int y0 = std::min(h - 1.0f, std::max(gy, 0.0f));
                                int z0 = std::min(d - 1.0f, std::max(gz, 0.0f));

                                for (int q = 0; q < channels; q++)
                                {
                                    top_blob.channel(q).depth(y)[x / 3] = bottom_blob.channel(q).depth(z0).row(y0)[x0];
                                }
                            }
                        }
                    }
                    else
                    {
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int y = 0; y < grid_p1.c; y++)
                        {
                            float* gridptr = grid_p1.channel(y);
                            int nn = grid_size;
#if __AVX__
                            for (int x = 0; x + 23 < nn; x += 24)
                            {
                                __m256 tmp_x = _mm256_loadu_ps(gridptr + x);
                                __m256 tmp_y = _mm256_loadu_ps(gridptr + x + 8);
                                __m256 gz = _mm256_loadu_ps(gridptr + x + 16);

                                __m256 gx = _mm256_permute2f128_ps(tmp_x, tmp_y, 0b00110000);
                                __m256 gy = _mm256_permute2f128_ps(tmp_x, gz, 0b00100001);
                                gz = _mm256_permute2f128_ps(tmp_y, gz, 0b00110000);

                                tmp_x = _mm256_shuffle_ps(gx, gy, 0b01001001);
                                tmp_y = _mm256_shuffle_ps(gy, gz, 0b10011110);

                                gy = _mm256_shuffle_ps(tmp_x, tmp_y, 0b11011000);
                                gx = _mm256_shuffle_ps(gx, tmp_y, 0b10001100);
                                gz = _mm256_shuffle_ps(tmp_x, gz, 0b11001101);

                                gx = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gx, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1));
                                gy = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gy, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1));
                                gz = _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(gz, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1));

                                gx = _mm256_floor_ps(_mm256_add_ps(gx, _mm256_set1_ps(0.5f)));
                                gy = _mm256_floor_ps(_mm256_add_ps(gy, _mm256_set1_ps(0.5f)));
                                gz = _mm256_floor_ps(_mm256_add_ps(gz, _mm256_set1_ps(0.5f)));

                                // compute coord
                                {
                                    // x
                                    const __m256 border_x = _mm256_sub_ps(vImgWf, *(__m256*)_ps256_1);

                                    gx = _mm256_and_ps(gx, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(gx, border_x), *(__m256*)_ps256_inv_sign_mask);
                                    gx = _mm256_sub_ps(border_x, reflectx_v);

                                    // y
                                    const __m256 border_y = _mm256_sub_ps(vImgHf, *(__m256*)_ps256_1);

                                    gy = _mm256_and_ps(gy, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflecty_v = _mm256_and_ps(_mm256_sub_ps(gy, border_y), *(__m256*)_ps256_inv_sign_mask);
                                    gy = _mm256_sub_ps(border_y, reflecty_v);

                                    // z
                                    const __m256 border_z = _mm256_sub_ps(vImgDf, *(__m256*)_ps256_1);

                                    gz = _mm256_and_ps(gz, *(__m256*)_ps256_inv_sign_mask);

                                    __m256 reflectz_v = _mm256_and_ps(_mm256_sub_ps(gz, border_z), *(__m256*)_ps256_inv_sign_mask);
                                    gz = _mm256_sub_ps(border_z, reflectz_v);
                                }

                                __m256 offset = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vImgWf, vImgHf), gz),
                                                              _mm256_add_ps(_mm256_mul_ps(gy, vImgWf), gx));
                                __m256i i_offset = _mm256_cvtps_epi32(offset);

                                for (int q = 0; q < channels; q++)
                                {
                                    __m256 _v = mask_gather_ps256(bottom_blob.channel(q), i_offset, *(__m256*)_ps256_n1);

                                    _mm256_storeu_ps(static_cast<float*>(top_blob.channel(q).depth(y).data) + x / 3, _v);
                                }
                            }
                            nn = grid_size % 24;
#endif // __AVX__
                            for (int x = grid_size - nn; x < grid_size; x += 3)
                            {
                                float gx = gridptr[x];
                                float gy = gridptr[x + 1];
                                float gz = gridptr[x + 2];

                                gx = (gx + 1) / 2.f * (w - 1);
                                gy = (gy + 1) / 2.f * (h - 1);
                                gz = (gz + 1) / 2.f * (d - 1);

                                gx = floor(gx + 0.5f);
                                gy = floor(gy + 0.5f);
                                gz = floor(gz + 0.5f);

                                gx = abs(gx);
                                gx = (w - 1) - abs(gx - (w - 1));

                                gy = abs(gy);
                                gy = (h - 1) - abs(gy - (h - 1));

                                gz = abs(gz);
                                gz = (d - 1) - abs(gz - (d - 1));

                                int x0 = std::min(w - 1.0f, std::max(gx, 0.0f));
                                int y0 = std::min(h - 1.0f, std::max(gy, 0.0f));
                                int z0 = std::min(d - 1.0f, std::max(gz, 0.0f));

                                for (int q = 0; q < channels; q++)
                                {
                                    top_blob.channel(q).depth(y)[x / 3] = bottom_blob.channel(q).depth(z0).row(y0)[x0];
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                NCNN_LOGE("unsupported bicubic when dims == 4");
                return -1;
            }
        }
        return 0;
#else
        return GridSample::forward(bottom_blobs, top_blobs, opt);
#endif // __SSE2__
    }
    return 0;
}

} // namespace ncnn
