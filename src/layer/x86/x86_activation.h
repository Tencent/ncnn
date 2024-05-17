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

#include "mat.h"
#include "fused_activation.h"
#include "x86_usability.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"

static NCNN_FORCEINLINE __m128 sigmoid_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    return _mm_div_ps(one, _mm_add_ps(one, exp_ps(_mm_sub_ps(_mm_setzero_ps(), inputs))));
}

static NCNN_FORCEINLINE __m128 tanh_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 two = _mm_set1_ps(2.0f);
    return _mm_sub_ps(_mm_mul_ps(sigmoid_sse(_mm_mul_ps(inputs, two)), two), one);
}

static NCNN_FORCEINLINE __m128 mish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, tanh_sse(log_ps(_mm_add_ps(exp_ps(inputs), _mm_set1_ps(1.f)))));
}

static NCNN_FORCEINLINE __m128 swish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, sigmoid_sse(inputs));
}

static NCNN_FORCEINLINE __m128 hardswish_sse(__m128 inputs, __m128 a, __m128 b)
{
    const __m128 one = _mm_set1_ps(1.0f);
    b = _mm_add_ps(_mm_mul_ps(inputs, a), b);
    b = _mm_max_ps(b, _mm_setzero_ps());
    b = _mm_min_ps(b, one);
    return _mm_mul_ps(b, inputs);
}

static NCNN_FORCEINLINE __m128 lrelu_sse(__m128 inputs, float slope)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(_mm_set1_ps(slope), neg));
}

static NCNN_FORCEINLINE __m128 prelu_sse(__m128 inputs, __m128 alphas)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m128 elu_sse(__m128 inputs, __m128 alphas)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    neg = _mm_sub_ps(exp_ps(neg), _mm_set1_ps(1.f));
    return _mm_add_ps(pos, _mm_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m128 activation_sse(__m128 _v, int activation_type, const ncnn::Mat& activation_params)
{
    // Process fused activations
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return _mm_max_ps(_v, _mm_setzero_ps());
    }
    case 2:
    {
        // Leaky relu
        return lrelu_sse(_v, activation_params[0]);
    }
    case 3:
    {
        // min max clip
        __m128 min = _mm_set1_ps(activation_params[0]);
        __m128 max = _mm_set1_ps(activation_params[1]);
        return _mm_min_ps(_mm_max_ps(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_sse(_v);
    }
    case 5:
    {
        return mish_sse(_v);
    }
    case 6:
    {
        __m128 _a = _mm_set1_ps(activation_params[0]);
        __m128 _b = _mm_set1_ps(activation_params[1]);
        return hardswish_sse(_v, _a, _b);
    }
    }

    return _v;
}

#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"

static NCNN_FORCEINLINE __m256 sigmoid_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs))));
}

static NCNN_FORCEINLINE __m256 tanh_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
#if __FMA__
    return _mm256_fmsub_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two, one);
#else
    return _mm256_sub_ps(_mm256_mul_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two), one);
#endif
}

static NCNN_FORCEINLINE __m256 mish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, tanh_avx(log256_ps(_mm256_add_ps(exp256_ps(inputs), _mm256_set1_ps(1.f)))));
}

static NCNN_FORCEINLINE __m256 swish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, sigmoid_avx(inputs));
}

static NCNN_FORCEINLINE __m256 hardswish_avx(__m256 inputs, __m256 a, __m256 b)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    b = _mm256_comp_fmadd_ps(inputs, a, b);
    b = _mm256_max_ps(b, _mm256_setzero_ps());
    b = _mm256_min_ps(b, one);
    return _mm256_mul_ps(b, inputs);
}

static NCNN_FORCEINLINE __m256 lrelu_avx(__m256 inputs, float slope)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(_mm256_set1_ps(slope), neg));
}

static NCNN_FORCEINLINE __m256 prelu_avx(__m256 inputs, __m256 alphas)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m256 elu_avx(__m256 inputs, __m256 alphas)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    neg = _mm256_sub_ps(exp256_ps(neg), _mm256_set1_ps(1.f));
    return _mm256_add_ps(pos, _mm256_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m256 activation_avx(__m256 _v, int activation_type, const ncnn::Mat& activation_params)
{
    // Process fused activations
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return _mm256_max_ps(_v, _mm256_setzero_ps());
    }
    case 2:
    {
        // Leaky relu
        return lrelu_avx(_v, activation_params[0]);
    }
    case 3:
    {
        // min max clip
        __m256 min = _mm256_set1_ps(activation_params[0]);
        __m256 max = _mm256_set1_ps(activation_params[1]);
        return _mm256_min_ps(_mm256_max_ps(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_avx(_v);
    }
    case 5:
    {
        return mish_avx(_v);
    }
    case 6:
    {
        __m256 _a = _mm256_set1_ps(activation_params[0]);
        __m256 _b = _mm256_set1_ps(activation_params[1]);
        return hardswish_avx(_v, _a, _b);
    }
    }

    return _v;
}

#if __AVX512F__
#include "avx512_mathfun.h"

static NCNN_FORCEINLINE __m512 sigmoid_avx512(__m512 inputs)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), inputs))));
}

static NCNN_FORCEINLINE __m512 tanh_avx512(__m512 inputs)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    return _mm512_fmsub_ps(sigmoid_avx512(_mm512_mul_ps(inputs, two)), two, one);
}

static NCNN_FORCEINLINE __m512 mish_avx512(__m512 inputs)
{
    return _mm512_mul_ps(inputs, tanh_avx512(log512_ps(_mm512_add_ps(exp512_ps(inputs), _mm512_set1_ps(1.f)))));
}

static NCNN_FORCEINLINE __m512 swish_avx512(__m512 inputs)
{
    return _mm512_mul_ps(inputs, sigmoid_avx512(inputs));
}

static NCNN_FORCEINLINE __m512 hardswish_avx512(__m512 inputs, __m512 a, __m512 b)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    b = _mm512_fmadd_ps(inputs, a, b);
    b = _mm512_max_ps(b, _mm512_setzero_ps());
    b = _mm512_min_ps(b, one);
    return _mm512_mul_ps(b, inputs);
}

static NCNN_FORCEINLINE __m512 lrelu_avx512(__m512 inputs, float slope)
{
    __mmask16 _is_negative = _mm512_cmp_ps_mask(inputs, _mm512_setzero_ps(), _CMP_LT_OQ);
    return _mm512_mask_mul_ps(inputs, _is_negative, inputs, _mm512_set1_ps(slope));
}

static NCNN_FORCEINLINE __m512 elu_avx512(__m512 inputs, __m512 alphas)
{
    __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), inputs);
    __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), inputs);
    neg = _mm512_sub_ps(exp512_ps(neg), _mm512_set1_ps(1.f));
    return _mm512_add_ps(pos, _mm512_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m512 prelu_avx512(__m512 inputs, __m512 alphas)
{
    __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), inputs);
    __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), inputs);
    return _mm512_add_ps(pos, _mm512_mul_ps(alphas, neg));
}

static NCNN_FORCEINLINE __m512 activation_avx512(__m512 _v, int activation_type, const ncnn::Mat& activation_params)
{
    // Process fused activations
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return _mm512_max_ps(_v, _mm512_setzero_ps());
    }
    case 2:
    {
        // Leaky relu
        return lrelu_avx512(_v, activation_params[0]);
    }
    case 3:
    {
        // min max clip
        __m512 min = _mm512_set1_ps(activation_params[0]);
        __m512 max = _mm512_set1_ps(activation_params[1]);
        return _mm512_min_ps(_mm512_max_ps(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_avx512(_v);
    }
    case 5:
    {
        return mish_avx512(_v);
    }
    case 6:
    {
        __m512 _a = _mm512_set1_ps(activation_params[0]);
        __m512 _b = _mm512_set1_ps(activation_params[1]);
        return hardswish_avx512(_v, _a, _b);
    }
    }

    return _v;
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#endif // X86_ACTIVATION_H
