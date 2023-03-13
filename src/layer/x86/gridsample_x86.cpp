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

#include "gridsample_x86.h"

#if __SSE2__
#include <emmintrin.h>
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
#endif // __AVX2__

    return v;
}

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

#endif // __SSE2__

#if _MSC_VER
#define OPT_2
#elif __clang__
#define OPT_2 __attribute__((optnone))
#elif __GNUC__
#define OPT_2 __attribute__((optimize("2")))
#endif

namespace GridSample_x86_kernel {

template<bool align_corner>
struct grid_sample_unormalize;

template<>
struct grid_sample_unormalize</*align_corner*/ true>
{
#if __AVX__
    OPT_2
    __m256 operator()(__m256 length, __m256 coord)
    {
        return _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(coord, *(__m256*)_ps256_1), *(__m256*)_ps256_2), _mm256_sub_ps(length, *(__m256*)_ps256_1));
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        return (coord + 1) / 2.f * (length - 1);
    }
};

template<>
struct grid_sample_unormalize</*align_corner*/ false>
{
#if __AVX__
    OPT_2
    __m256 operator()(__m256 length, __m256 coord)
    {
        return _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(coord, *(__m256*)_ps256_1), length, *(__m256*)_ps256_1), *(__m256*)_ps256_2);
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        return ((coord + 1) * length - 1) / 2.f;
    }
};

template<GridSample::PaddingMode pd, bool align_corner>
struct compute_coord;

template<bool align_corner>
struct compute_coord<GridSample::Border, align_corner>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        const __m256 border_x = _mm256_sub_ps(length, *(__m256*)_ps256_1);

        coord = _mm256_min_ps(border_x, _mm256_max_ps(coord, _mm256_setzero_ps()));

        return coord;
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

template<>
struct compute_coord<GridSample::Reflection, /*align_corner*/ true>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        const __m256 border_x = _mm256_sub_ps(length, *(__m256*)_ps256_1);

        coord = _mm256_and_ps(coord, *(__m256*)_ps256_inv_sign_mask);

        __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(coord, border_x), *(__m256*)_ps256_inv_sign_mask);
        coord = _mm256_sub_ps(border_x, reflectx_v);

        return coord;
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        coord = abs(coord);
        coord = (length - 1) - abs(coord - (length - 1));

        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

template<>
struct compute_coord<GridSample::Reflection, /*align_corner*/ false>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        const __m256 border_x = _mm256_sub_ps(length, *(__m256*)_ps256_1);

        __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
        coord = _mm256_add_ps(coord, v0p5fp8);

        coord = _mm256_and_ps(coord, *(__m256*)_ps256_inv_sign_mask);

        __m256 reflectx_v = _mm256_and_ps(_mm256_sub_ps(coord, length), *(__m256*)_ps256_inv_sign_mask);
        coord = _mm256_sub_ps(length, reflectx_v);

        coord = _mm256_sub_ps(coord, v0p5fp8);

        _mm256_sub_ps(coord, v0p5fp8);

        coord = _mm256_min_ps(border_x, _mm256_max_ps(coord, _mm256_setzero_ps()));

        return coord;
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        coord = abs(coord + 0.5f);
        coord = length - abs(coord - length) - 0.5;

        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

#include "gridsample_bilinear_compute_blob.h"
#include "gridsample_bicubic_compute_blob.h"
#include "gridsample_nearest_compute_blob.h"

} //namespace GridSample_x86_kernel

int GridSample_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    using namespace GridSample_x86_kernel;
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];
    const int elempack = bottom_blob.elempack;

    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw, outh, outd;
    Mat offset_blob, in_bound_blob, value_blob;

    ncnn::Mat grid_tmp;
    if (permute_fusion == 0 && grid.elempack != 1)
    {
        ncnn::convert_packing(grid, grid_tmp, 1, opt);
    }

    ncnn::Mat grid_p1 = (grid.elempack == 1) ? grid : grid_tmp;

    if (dims == 3)
    {
        outw = permute_fusion == 0 ? grid.h : grid.w;
        outh = permute_fusion == 0 ? grid.c : grid.h;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (sample_type == GridSample::Bilinear)
        {
            offset_blob.create(outw, outh, 4, elemsize, 1, opt.blob_allocator);
            in_bound_blob.create(outw, outh, 4, elemsize, 1, opt.blob_allocator);
            value_blob.create(outw, outh, 2, elemsize, 1, opt.blob_allocator);
            if (offset_blob.empty() || in_bound_blob.empty() || value_blob.empty())
                return -100;

            in_bound_blob.fill(-1.0f);

            if (padding_mode == GridSample::Zeros)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Zeros, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Zeros, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Border)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Border, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Border, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Reflection)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Reflection, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Reflection, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Nearest)
        {
            offset_blob.create(outw, outh, 1, elemsize, 1, opt.blob_allocator);
            in_bound_blob.create(outw, outh, 1, elemsize, 1, opt.blob_allocator);
            if (offset_blob.empty() || in_bound_blob.empty())
                return -100;

            in_bound_blob.fill(-1.0f);

            if (padding_mode == GridSample::Zeros)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Zeros, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Zeros, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Border)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Border, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Border, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Reflection)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Reflection, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Reflection, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Bicubic)
        {
            offset_blob.create(outw, outh, 16, elemsize, 1, opt.blob_allocator);
            in_bound_blob.create(outw, outh, 16, elemsize, 1, opt.blob_allocator);
            value_blob.create(outw, outh, 2, elemsize, 1, opt.blob_allocator);
            if (offset_blob.empty() || in_bound_blob.empty() || value_blob.empty())
                return -100;

            if (padding_mode == GridSample::Zeros)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Zeros, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Zeros, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Border)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Border, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Border, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Reflection)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Reflection, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Reflection, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
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
        outw = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        outh = permute_fusion == 0 ? grid_p1.d : grid_p1.h;
        outd = permute_fusion == 0 ? grid_p1.c : grid_p1.d;

        top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (sample_type == GridSample::Bilinear)
        {
            offset_blob.create(outw, outh, outd, 8, elemsize, 1, opt.blob_allocator);
            in_bound_blob.create(outw, outh, outd, 8, elemsize, 1, opt.blob_allocator);
            value_blob.create(outw, outh, outd, 3, elemsize, 1, opt.blob_allocator);
            if (offset_blob.empty() || in_bound_blob.empty() || value_blob.empty())
                return -100;

            in_bound_blob.fill(-1.0f);

            if (padding_mode == GridSample::Zeros)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Zeros, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Zeros, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Border)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Border, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Border, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Reflection)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Reflection, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Reflection, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Nearest)
        {
            offset_blob.create(outw, outh, outd, 1, elemsize, 1, opt.blob_allocator);
            in_bound_blob.create(outw, outh, outd, 1, elemsize, 1, opt.blob_allocator);
            if (offset_blob.empty() || in_bound_blob.empty())
                return -100;

            in_bound_blob.fill(-1.0f);

            if (padding_mode == GridSample::Zeros)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Zeros, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Zeros, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Border)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Border, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Border, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
            }
            else if (padding_mode == GridSample::Reflection)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Reflection, false> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Reflection, true> op;
                    op(bottom_blob, grid_p1, offset_blob, in_bound_blob, value_blob, permute_fusion, opt);
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
            NCNN_LOGE("unsupported bicubic when dims == 4");
            return -100;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_2d_bilinear_apply_interpolation_p16(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p16(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
            else if (sample_type == GridSample::Bicubic)
            {
                gridsample_2d_bicubic_apply_interpolation_p16(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_3d_bilinear_apply_interpolation_p16(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p16(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_2d_bilinear_apply_interpolation_p8(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p8(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
            else if (sample_type == GridSample::Bicubic)
            {
                gridsample_2d_bicubic_apply_interpolation_p8(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_3d_bilinear_apply_interpolation_p8(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p8(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
        }
    }

#endif // __AVX__
    if (elempack == 4)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_2d_bilinear_apply_interpolation_p4(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p4(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
            else if (sample_type == GridSample::Bicubic)
            {
                gridsample_2d_bicubic_apply_interpolation_p4(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_3d_bilinear_apply_interpolation_p4(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p4(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
        }
    }

#endif // __SSE2__

    if (elempack == 1)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_2d_bilinear_apply_interpolation_p1(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p1(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
            else if (sample_type == GridSample::Bicubic)
            {
                gridsample_2d_bicubic_apply_interpolation_p1(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Bilinear)
            {
                gridsample_3d_bilinear_apply_interpolation_p1(bottom_blob, top_blob, offset_blob, in_bound_blob, value_blob, opt);
            }
            else if (sample_type == GridSample::Nearest)
            {
                gridsample_nearest_apply_interpolation_p1(bottom_blob, top_blob, offset_blob, in_bound_blob, opt);
            }
        }
    }

    return 0;
}

} // namespace ncnn
