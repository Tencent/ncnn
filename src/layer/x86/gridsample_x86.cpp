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

#include "gridsample_bilinear_pack4.h"
#include "gridsample_nearest_pack4.h"
#include "gridsample_bicubic_pack4.h"

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

#include "gridsample_bilinear_pack1.h"
#include "gridsample_nearest_pack1.h"
#include "gridsample_bicubic_pack1.h"

int GridSample_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];
    const int elempack = bottom_blob.elempack;

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

        if (dims == 3)
        {
            top_blob.create(grid_p1.h, grid_p1.c, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bilinear_align0_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_bilinear_align1_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
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
                        gridsample_2d_nearest_align0_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_nearest_align0_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_nearest_align1_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
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
                        gridsample_2d_bicubic_align0_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_2d_bicubic_align0_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_2d_bicubic_align1_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
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
            top_blob.create(grid_p1.h, grid_p1.d, grid_p1.c, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (sample_type == 1)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_bilinear_align0_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_3d_bilinear_align1_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
            }
            else if (sample_type == 2)
            {
                if (padding_mode == 1)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_zeros_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 2)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_border_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                }
                else if (padding_mode == 3)
                {
                    if (align_corner == 0)
                    {
                        gridsample_3d_nearest_align0_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
                    }
                    else
                    {
                        gridsample_3d_nearest_align1_reflection_blob_pack1(bottom_blob, top_blob, grid_p1, opt);
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
