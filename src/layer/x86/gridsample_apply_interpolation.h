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

#if __SSE2__
#if __AVX__
static __m256 mask_gather_ps256(const float* ptr, __m256i offset, __m256 mask)
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

static __m128 mask_gather_ps(const float* ptr, __m128i offset, __m128 mask)
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

#include "gridsample_bilinear_apply_interpolation.h"
#include "gridsample_bicubic_apply_interpolation.h"
#include "gridsample_nearest_apply_interpolation.h"