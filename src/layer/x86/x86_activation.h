// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef X86_ACTIVATION_H
#define X86_ACTIVATION_H

#include <math.h>
#include "mat.h"

static inline float activation_ss(float v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        v = fmax(v, 0.f);
    }
    else if (activation_type == 2)
    {
        float slope = activation_params[0];
        v = v > 0.f ? v : v * slope;
    }
    else if (activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (v < min)
            v = min;
        if (v > max)
            v = max;
    }
    else if (activation_type == 4)
    {
        v = 1.f / (1.f + exp(-v));
    }
    else if (activation_type == 5)
    {
        v = v * tanh(log(exp(v) + 1.f));
    }

    return v;
}

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"

static inline __m128 sigmoid_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    return _mm_div_ps(one, _mm_add_ps(one, exp_ps(_mm_sub_ps(_mm_setzero_ps(), inputs))));
}

static inline __m128 tanh_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 two = _mm_set1_ps(2.0f);
    return _mm_sub_ps(_mm_mul_ps(sigmoid_sse(_mm_mul_ps(inputs, two)), two), one);
}

static inline __m128 mish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, tanh_sse(log_ps(_mm_add_ps(exp_ps(inputs), _mm_set1_ps(1.f)))));
}

static inline __m128 abs_sse(__m128 inputs)
{
    return _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), inputs), inputs);
}

static inline __m128 lrelu_sse(__m128 inputs, float slope)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(_mm_set1_ps(slope), neg));
}

static inline __m128 prelu_sse(__m128 inputs, __m128 alphas)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(alphas, neg));
}

static inline __m128 activation_sse(__m128 _v, int activation_type, const ncnn::Mat& activation_params)
{
    // Process fused activations
    if (activation_type == 1)
    {
        // Relu
        return _mm_max_ps(_v, _mm_setzero_ps());
    }
    else if (activation_type == 2)
    {
        // Leaky relu
        return lrelu_sse(_v, activation_params[0]);
    }
    else if (activation_type == 3)
    {
        // min max clip
        __m128 min = _mm_set1_ps(activation_params[0]);
        __m128 max = _mm_set1_ps(activation_params[1]);
        return _mm_min_ps(_mm_max_ps(_v, min), max);
    }
    else if (activation_type == 4)
    {
        // Sigmoid
        return sigmoid_sse(_v);
    }
    else if (activation_type == 5)
    {
        return mish_sse(_v);
    }

    return _v;
}

#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"

static inline __m256 sigmoid_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs))));
}

static inline __m256 tanh_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    return _mm256_fmsub_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two, one);
}

static inline __m256 mish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, tanh_avx(log256_ps(_mm256_add_ps(exp256_ps(inputs), _mm256_set1_ps(1.f)))));
}

static inline __m256 abs_avx(__m256 inputs)
{
    return _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs), inputs);
}

static inline __m256 lrelu_avx(__m256 inputs, float slope)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(_mm256_set1_ps(slope), neg));
}

static inline __m256 prelu_avx(__m256 inputs, __m256 alphas)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(alphas, neg));
}

static inline __m256 activation_avx(__m256 _v, int activation_type, const ncnn::Mat& activation_params)
{
    // Process fused activations
    if (activation_type == 1)
    {
        // Relu
        return _mm256_max_ps(_v, _mm256_setzero_ps());
    }
    else if (activation_type == 2)
    {
        // Leaky relu
        return lrelu_avx(_v, activation_params[0]);
    }
    else if (activation_type == 3)
    {
        // min max clip
        __m256 min = _mm256_set1_ps(activation_params[0]);
        __m256 max = _mm256_set1_ps(activation_params[1]);
        return _mm256_min_ps(_mm256_max_ps(_v, min), max);
    }
    else if (activation_type == 4)
    {
        // Sigmoid
        return sigmoid_avx(_v);
    }
    else if (activation_type == 5)
    {
        return mish_avx(_v);
    }

    return _v;
}
#endif // __AVX__
#endif // __SSE2__

#endif // X86_ACTIVATION_H
