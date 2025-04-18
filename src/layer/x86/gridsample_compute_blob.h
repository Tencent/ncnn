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

#include "x86_usability.h"

template<bool align_corner>
struct grid_sample_unormalize;

template<>
struct grid_sample_unormalize</*align_corner*/ true>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        return _mm256_mul_ps(_mm256_div_ps(_mm256_add_ps(coord, _mm256_set1_ps(1)), _mm256_set1_ps(2)), _mm256_sub_ps(length, _mm256_set1_ps(1)));
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
    __m256 operator()(__m256 length, __m256 coord)
    {
        return _mm256_div_ps(_mm256_comp_fmsub_ps(_mm256_add_ps(coord, _mm256_set1_ps(1)), length, _mm256_set1_ps(1)), _mm256_set1_ps(2));
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        return ((coord + 1) * length - 1) / 2.f;
    }
};

template<GridSample::PaddingMode pd, bool align_corner>
struct compute_coord
{
#if __AVX__
    __m256 operator()(__m256 /*length*/, __m256 coord)
    {
        return coord;
    }
#endif // __AVX__
    float operator()(int /*length*/, float coord)
    {
        return coord;
    }
};

template<bool align_corner>
struct compute_coord<GridSample::Padding_BORDER, align_corner>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        const __m256 border_x = _mm256_sub_ps(length, _mm256_set1_ps(1));

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
struct compute_coord<GridSample::Padding_REFLECTION, /*align_corner*/ true>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        const __m256 border_x = _mm256_sub_ps(length, _mm256_set1_ps(1));

        coord = abs256_ps(coord);

        __m256 reflectx_v = abs256_ps(_mm256_sub_ps(coord, border_x));
        coord = _mm256_sub_ps(border_x, reflectx_v);

        return coord;
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        coord = fabs(coord);
        coord = (length - 1) - fabs(coord - (length - 1));

        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

template<>
struct compute_coord<GridSample::Padding_REFLECTION, /*align_corner*/ false>
{
#if __AVX__
    __m256 operator()(__m256 length, __m256 coord)
    {
        const __m256 border_x = _mm256_sub_ps(length, _mm256_set1_ps(1));

        __m256 v0p5fp8 = _mm256_set1_ps(0.5f);
        coord = _mm256_add_ps(coord, v0p5fp8);

        coord = abs256_ps(coord);

        __m256 reflectx_v = abs256_ps(_mm256_sub_ps(coord, length));
        coord = _mm256_sub_ps(length, reflectx_v);

        coord = _mm256_sub_ps(coord, v0p5fp8);

        _mm256_sub_ps(coord, v0p5fp8);

        coord = _mm256_min_ps(border_x, _mm256_max_ps(coord, _mm256_setzero_ps()));

        return coord;
    }
#endif // __AVX__
    float operator()(int length, float coord)
    {
        coord = fabs(coord + 0.5f);
        coord = length - fabs(coord - length) - 0.5;

        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

#include "gridsample_bilinear_compute_blob.h"
#include "gridsample_bicubic_compute_blob.h"
#include "gridsample_nearest_compute_blob.h"
