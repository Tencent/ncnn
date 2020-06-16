// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
#ifndef AVX_USABILITY
#define AVX_USABILITY
#include <immintrin.h>

static inline __m256 HorizontalSums(__m256 v0, __m256 v1, __m256 v2, __m256 v3, __m256 v4,
                                    __m256 v5, __m256 v6, __m256 v7) {
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s45 = _mm256_hadd_ps(v4, v5);
    const __m256 s67 = _mm256_hadd_ps(v6, v7);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);
    const __m256 s4556 = _mm256_hadd_ps(s45, s67);

    // inter-lane shuffle
    v0 = _mm256_blend_ps(s0123, s4556, 0xF0);
    v1 = _mm256_permute2f128_ps(s0123, s4556, 0x21);

    return _mm256_add_ps(v0, v1);
}

static inline __m128 HorizontalSums(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);

    return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                      _mm256_castps256_ps128(s0123));
}
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 =
        _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

#endif