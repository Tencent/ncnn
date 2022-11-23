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
const __m512 v1fp16 = _mm512_set1_ps(1.0f);
const __m512 vn1fp16 = _mm512_set1_ps(-1.0f);
const __m512i v1ip16 = _mm512_set1_epi32(1);
const __m512i vn1ip16 = _mm512_set1_epi32(-1);

#include "gridsample_bilinear_pack16.h"
#include "gridsample_nearest_pack16.h"
#include "gridsample_bicubic_pack16.h"

#endif // __AVX512F__
const __m256 v1fp8 = *(__m256*)_ps256_1;
const __m256 vn1fp8 = _mm256_set1_ps(-1.0f);
const __m256i v1ip8 = _mm256_set1_epi32(1);
const __m256i vn1ip8 = _mm256_set1_epi32(-1);

#include "gridsample_bilinear_pack8.h"
#include "gridsample_nearest_pack8.h"
#include "gridsample_bicubic_pack8.h"

#endif // __AVX__

const __m128 v1fp4 = _mm_set1_ps(1.0f);
const auto vn1fp4 = _mm_set1_ps(-1.0f);
const auto v1ip4 = _mm_set1_epi32(1);
const auto vn1ip4 = _mm_set1_epi32(-1);

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

#include "gridsample_bicubic_pack4.h"
#include "gridsample_bilinear_pack4.h"
#include "gridsample_nearest_pack4.h"

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
                return -1;
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
                return -1;
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
                return GridSample::forward(bottom_blobs, top_blobs, opt);
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
                    int nn = size >> 3;
                    int remain = size;
#if __AVX__
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int j = 0; j < nn; j++)
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
                        for (int q = 0; q < channels; q++)
                        {
                            auto _v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), bottom_blob.channel(q),
                                                               i_offset, *reinterpret_cast<__m256*>(&v_in_range), sizeof(float));

                            _mm256_storeu_ps(top_blob.channel(q).row(0) + j * 8, _v);
                        }
                    }

                    remain = remain & 7;
#endif // __AVX__
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int j = size - remain; j < nn; j++)
                    {
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
                return GridSample::forward(bottom_blobs, top_blobs, opt);
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
            return GridSample::forward(bottom_blobs, top_blobs, opt);
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
