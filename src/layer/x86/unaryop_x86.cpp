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

#include "unaryop_x86.h"

#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __SSE4_1__
#include <smmintrin.h>
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE4_1__
#endif // __SSE2__
#include "x86_usability.h"
#include "x86_activation.h"

namespace ncnn {

UnaryOp_x86::UnaryOp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = op(_p);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = op(_p);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = op(_p);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = op(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_x86_functor {
struct unary_op_abs
{
    float operator()(const float& x) const
    {
        return (float)fabs(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return abs_sse(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return abs_avx(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return abs_avx512(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_neg
{
    float operator()(const float& x) const
    {
        return -x;
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return _mm_sub_ps(_mm_setzero_ps(), x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_sub_ps(_mm256_setzero_ps(), x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return _mm512_sub_ps(_mm512_setzero_ps(), x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_floor
{
    float operator()(const float& x) const
    {
        return (float)floor(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
#if __SSE4_1__
        return _mm_floor_ps(x);
#endif // __SSE4_1__

        // Use negative zero as the sign bit mask.
        const __m128 magic_negative_zero = _mm_set_ps1(-0.0f);

        // The smallest float number that have no fractional part. (2^23)
        const __m128 magic_smallest_no_fraction = _mm_set_ps1(8388608.0f);

        // absolute = abs(x);
        __m128 absolute = _mm_andnot_ps(magic_negative_zero, x);

        // negative_mask = magic_negative_zero && x;
        __m128 negative_mask = _mm_and_ps(magic_negative_zero, x);

        // no_fraction = (magic_smallest_no_fraction < absolute);
        __m128 no_fraction = _mm_cmplt_ps(magic_smallest_no_fraction, absolute);

        // truncated = static_cast<float>(static_cast<uint32_t>(absolute));
        __m128 truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(absolute));

        // truncated_with_sign = (truncated || negative_mask);
        __m128 truncated_with_sign = _mm_or_ps(truncated, negative_mask);

        // negative_fix = ((x < truncated_with_sign) ? 1.0f : 0.0f);
        __m128 negative_fix = _mm_and_ps(
                                  _mm_cmplt_ps(x, truncated_with_sign),
                                  _mm_set_ps1(1.0f));

        // fixed_result = truncated_with_sign - negative_fix;
        __m128 fixed_result = _mm_sub_ps(truncated_with_sign, negative_fix);

        // return ((x && no_fraction) || (!no_fraction && fixed_result));
        return _mm_or_ps(
                   _mm_and_ps(x, no_fraction),
                   _mm_andnot_ps(no_fraction, fixed_result));
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_floor_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEG_INF);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_ceil
{
    float operator()(const float& x) const
    {
        return (float)ceil(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
#if __SSE4_1__
        return _mm_ceil_ps(x);
#endif // __SSE4_1__

        // Use negative zero as the sign bit mask.
        const __m128 magic_negative_zero = _mm_set_ps1(-0.0f);

        // The smallest float number that have no fractional part. (2^23)
        const __m128 magic_smallest_no_fraction = _mm_set_ps1(8388608.0f);

        // absolute = abs(x);
        __m128 absolute = _mm_andnot_ps(magic_negative_zero, x);

        // negative_mask = magic_negative_zero && x;
        __m128 negative_mask = _mm_and_ps(magic_negative_zero, x);

        // no_fraction = (magic_smallest_no_fraction < absolute);
        __m128 no_fraction = _mm_cmplt_ps(magic_smallest_no_fraction, absolute);

        // truncated = static_cast<float>(static_cast<uint32_t>(absolute));
        __m128 truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(absolute));

        // truncated_with_sign = (truncated || negative_mask);
        __m128 truncated_with_sign = _mm_or_ps(truncated, negative_mask);

        // positive_fix = ((x > -0.0f) && (x > truncated_with_sign) ? -1.0f : 0.0f);
        __m128 positive_fix = _mm_and_ps(
                                  _mm_and_ps(
                                      _mm_cmpgt_ps(x, magic_negative_zero),
                                      _mm_cmpgt_ps(x, truncated_with_sign)),
                                  _mm_set_ps1(-1.0f));

        // fixed_result = truncated_with_sign - positive_fix;
        __m128 fixed_result = _mm_sub_ps(truncated_with_sign, positive_fix);

        // return ((x && no_fraction) || (!no_fraction && fixed_result));
        return _mm_or_ps(
                   _mm_and_ps(x, no_fraction),
                   _mm_andnot_ps(no_fraction, fixed_result));
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_ceil_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_POS_INF);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_square
{
    float operator()(const float& x) const
    {
        return x * x;
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return _mm_mul_ps(x, x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_mul_ps(x, x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return _mm512_mul_ps(x, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_sqrt
{
    float operator()(const float& x) const
    {
        return (float)sqrt(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return _mm_sqrt_ps(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_sqrt_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return _mm512_sqrt_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_rsqrt
{
    float operator()(const float& x) const
    {
        return (float)(1.f / sqrt(x));
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return _mm_rsqrt_ps(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_rsqrt_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        __m256 _x0 = _mm512_extractf32x8_ps(x, 0);
        __m256 _x1 = _mm512_extractf32x8_ps(x, 1);
        _x0 = _mm256_rsqrt_ps(_x0);
        _x1 = _mm256_rsqrt_ps(_x1);
        return _mm512_insertf32x8(_mm512_castps256_ps512(_x0), _x1, 1);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_exp
{
    float operator()(const float& x) const
    {
        return (float)exp(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return exp_ps(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return exp256_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return exp512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_log
{
    float operator()(const float& x) const
    {
        return (float)log(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return log_ps(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return log256_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return log512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_sin
{
    float operator()(const float& x) const
    {
        return (float)sin(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return sin_ps(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return sin256_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return sin512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_cos
{
    float operator()(const float& x) const
    {
        return (float)cos(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return cos_ps(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return cos256_ps(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return cos512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_tan
{
    float operator()(const float& x) const
    {
        return (float)tan(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        //TODO sse optimize
        float tmp[4];
        _mm_storeu_ps(tmp, x);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        return _mm_loadu_ps(tmp);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        //TODO avx optimize
        float tmp[8];
        _mm256_storeu_ps(tmp, x);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        tmp[4] = tan(tmp[4]);
        tmp[5] = tan(tmp[5]);
        tmp[6] = tan(tmp[6]);
        tmp[7] = tan(tmp[7]);
        return _mm256_loadu_ps(tmp);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        //TODO avx512 optimize
        float tmp[16];
        _mm512_storeu_ps(tmp, x);
        for (int i = 0; i < 16; i++)
            tmp[i] = tan(tmp[i]);
        return _mm512_loadu_ps(tmp);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_asin
{
    float operator()(const float& x) const
    {
        return (float)asin(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        //TODO sse optimize
        float tmp[4];
        _mm_storeu_ps(tmp, x);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        return _mm_loadu_ps(tmp);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        //TODO avx optimize
        float tmp[8];
        _mm256_storeu_ps(tmp, x);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        tmp[4] = asin(tmp[4]);
        tmp[5] = asin(tmp[5]);
        tmp[6] = asin(tmp[6]);
        tmp[7] = asin(tmp[7]);
        return _mm256_loadu_ps(tmp);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        //TODO avx512 optimize
        float tmp[16];
        _mm512_storeu_ps(tmp, x);
        for (int i = 0; i < 16; i++)
            tmp[i] = asin(tmp[i]);
        return _mm512_loadu_ps(tmp);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_acos
{
    float operator()(const float& x) const
    {
        return (float)acos(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        //TODO sse optimize
        float tmp[4];
        _mm_storeu_ps(tmp, x);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        return _mm_loadu_ps(tmp);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        //TODO avx optimize
        float tmp[8];
        _mm256_storeu_ps(tmp, x);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        tmp[4] = acos(tmp[4]);
        tmp[5] = acos(tmp[5]);
        tmp[6] = acos(tmp[6]);
        tmp[7] = acos(tmp[7]);
        return _mm256_loadu_ps(tmp);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        //TODO avx512 optimize
        float tmp[16];
        _mm512_storeu_ps(tmp, x);
        for (int i = 0; i < 16; i++)
            tmp[i] = acos(tmp[i]);
        return _mm512_loadu_ps(tmp);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_atan
{
    float operator()(const float& x) const
    {
        return (float)atan(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        //TODO sse optimize
        float tmp[4];
        _mm_storeu_ps(tmp, x);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
        return _mm_loadu_ps(tmp);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        //TODO avx optimize
        float tmp[8];
        _mm256_storeu_ps(tmp, x);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
        tmp[4] = atan(tmp[4]);
        tmp[5] = atan(tmp[5]);
        tmp[6] = atan(tmp[6]);
        tmp[7] = atan(tmp[7]);
        return _mm256_loadu_ps(tmp);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        //TODO avx512 optimize
        float tmp[16];
        _mm512_storeu_ps(tmp, x);
        for (int i = 0; i < 16; i++)
            tmp[i] = atan(tmp[i]);
        return _mm512_loadu_ps(tmp);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_reciprocal
{
    float operator()(const float& x) const
    {
        return 1.f / x;
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return _mm_div_ps(*(__m128*)_ps_1, x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return _mm256_div_ps(*(__m256*)_ps256_1, x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return _mm512_div_ps(*(__m512*)_ps512_1, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_tanh
{
    float operator()(const float& x) const
    {
        return (float)tanh(x);
    }
#if __SSE2__
    __m128 operator()(const __m128& x) const
    {
        return tanh_sse(x);
    }
#if __AVX__
    __m256 operator()(const __m256& x) const
    {
        return tanh_avx(x);
    }
#if __AVX512F__
    __m512 operator()(const __m512& x) const
    {
        return tanh_avx512(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

} // namespace UnaryOp_x86_functor

int UnaryOp_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_x86_functor;
    if (op_type == Operation_ABS)
        return unary_op_inplace<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace<unary_op_square>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace<unary_op_log>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace<unary_op_tanh>(bottom_top_blob, opt);

    return 0;
}

} // namespace ncnn
