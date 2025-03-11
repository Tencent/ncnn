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

#include "softmax_x86.h"

#include <float.h>

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

Softmax_x86::Softmax_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _max_avx512 = _mm512_set1_ps(-FLT_MAX);
#endif // __AVX512F__
    __m256 _max_avx = _mm256_set1_ps(-FLT_MAX);
#endif // __AVX__
    __m128 _max = _mm_set1_ps(-FLT_MAX);
#endif // __SSE2__
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _max_avx512 = _mm512_max_ps(_max_avx512, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _max_avx = _mm256_max_ps(_max_avx, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _max = _mm_max_ps(_max, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            max = std::max(max, ptr[0]);
            ptr++;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 8)
    {
        {
            __m256 _max0 = _mm512_castps512_ps256(_max_avx512);
            __m256 _max1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_max_avx512), 1));
            _max_avx = _mm256_max_ps(_max_avx, _max0);
            _max_avx = _mm256_max_ps(_max_avx, _max1);
        }

        _max_avx512 = combine8x2_ps(_max_avx, _max_avx);
    }
#endif // __AVX512F__
    if (elempack == 4)
    {
#if __AVX512F__
        {
            __m256 _max0 = _mm512_castps512_ps256(_max_avx512);
            __m256 _max1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_max_avx512), 1));
            _max_avx = _mm256_max_ps(_max_avx, _max0);
            _max_avx = _mm256_max_ps(_max_avx, _max1);
        }
#endif // __AVX512F__
        {
            __m128 _max0 = _mm256_castps256_ps128(_max_avx);
            __m128 _max1 = _mm256_extractf128_ps(_max_avx, 1);
            _max = _mm_max_ps(_max, _max0);
            _max = _mm_max_ps(_max, _max1);
        }

        _max_avx = combine4x2_ps(_max, _max);
#if __AVX512F__
        _max_avx512 = combine8x2_ps(_max_avx, _max_avx);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 1)
    {
#if __AVX__
#if __AVX512F__
        max = std::max(max, _mm512_comp_reduce_max_ps(_max_avx512));
#endif // __AVX512F__
        max = std::max(max, _mm256_reduce_max_ps(_max_avx));
#endif // __AVX__
        max = std::max(max, _mm_reduce_max_ps(_max));

        _max = _mm_set1_ps(max);
#if __AVX__
        _max_avx = combine4x2_ps(_max, _max);
#if __AVX512F__
        _max_avx512 = combine8x2_ps(_max_avx, _max_avx);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__

    // reduce exp(x - max)
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
    __m128 _sum = _mm_set1_ps(0.f);
#endif // __SSE2__
    float sum = 0.f;
    {
        float* ptr = _ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_sub_ps(_p, _max_avx512);
            _p = exp512_ps(_p);
            _mm512_storeu_ps(ptr, _p);
            _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_sub_ps(_p, _max_avx);
            _p = exp256_ps(_p);
            _mm256_storeu_ps(ptr, _p);
            _sum_avx = _mm256_add_ps(_sum_avx, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_sub_ps(_p, _max);
            _p = exp_ps(_p);
            _mm_storeu_ps(ptr, _p);
            _sum = _mm_add_ps(_sum, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        _sum_avx512 = _mm512_div_ps(_mm512_set1_ps(1.f), _sum_avx512);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
#if __AVX512F__
        {
            __m256 _sum0 = _mm512_castps512_ps256(_sum_avx512);
            __m256 _sum1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_sum_avx512), 1));
            _sum_avx = _mm256_add_ps(_sum_avx, _sum0);
            _sum_avx = _mm256_add_ps(_sum_avx, _sum1);
        }
#endif // __AVX512F__

        _sum_avx = _mm256_div_ps(_mm256_set1_ps(1.f), _sum_avx);

#if __AVX512F__
        _sum_avx512 = combine8x2_ps(_sum_avx, _sum_avx);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 4)
    {
#if __AVX__
#if __AVX512F__
        {
            __m256 _sum0 = _mm512_castps512_ps256(_sum_avx512);
            __m256 _sum1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_sum_avx512), 1));
            _sum_avx = _mm256_add_ps(_sum_avx, _sum0);
            _sum_avx = _mm256_add_ps(_sum_avx, _sum1);
        }
#endif // __AVX512F__
        {
            __m128 _sum0 = _mm256_castps256_ps128(_sum_avx);
            __m128 _sum1 = _mm256_extractf128_ps(_sum_avx, 1);
            _sum = _mm_add_ps(_sum, _sum0);
            _sum = _mm_add_ps(_sum, _sum1);
        }
#endif // __AVX__

        _sum = _mm_div_ps(_mm_set1_ps(1.f), _sum);

#if __AVX__
        _sum_avx = combine4x2_ps(_sum, _sum);
#if __AVX512F__
        _sum_avx512 = combine8x2_ps(_sum_avx, _sum_avx);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__
    if (elempack == 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
        sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
        sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

        sum = 1.f / sum;

#if __SSE2__
        _sum = _mm_set1_ps(sum);
#if __AVX__
        _sum_avx = combine4x2_ps(_sum, _sum);
#if __AVX512F__
        _sum_avx512 = combine8x2_ps(_sum_avx, _sum_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    // div sum
    {
        float* ptr = _ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_mul_ps(_p, _sum_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_mul_ps(_p, _sum_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_mul_ps(_p, _sum);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

#if __SSE2__
#if __AVX__
#if __AVX512F__
static void softmax_unroll16(float* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    __m512 _max_avx512 = _mm512_set1_ps(-FLT_MAX);
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _max_avx512 = _mm512_max_ps(_max_avx512, _p);
            ptr += stride;
        }
    }

    if (elempack == 16)
    {
        // reduce max 16 to 1
        // broadcast 1 to 16
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_permute_ps(_max_avx512, _MM_PERM_CDAB));
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_permute_ps(_max_avx512, _MM_PERM_BADC));
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_shuffle_f32x4(_max_avx512, _max_avx512, _MM_SHUFFLE(2, 3, 0, 1)));
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_shuffle_f32x4(_max_avx512, _max_avx512, _MM_SHUFFLE(1, 0, 3, 2)));
    }
    if (elempack == 8)
    {
        // reduce max 8,8 to 1,1
        // broadcast 1,1 to 8,8
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_permute_ps(_max_avx512, _MM_PERM_CDAB));
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_permute_ps(_max_avx512, _MM_PERM_BADC));
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_shuffle_f32x4(_max_avx512, _max_avx512, _MM_SHUFFLE(2, 3, 0, 1)));
    }
    if (elempack == 4)
    {
        // reduce max 4,4,4,4 to 1,1,1,1
        // broadcast 1,1,1,1 to 4,4,4,4
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_permute_ps(_max_avx512, _MM_PERM_CDAB));
        _max_avx512 = _mm512_max_ps(_max_avx512, _mm512_permute_ps(_max_avx512, _MM_PERM_BADC));
    }

    // reduce exp(x - max)
    __m512 _sum_avx512 = _mm512_set1_ps(0.f);
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_sub_ps(_p, _max_avx512);
            _p = exp512_ps(_p);
            _mm512_storeu_ps(ptr, _p);
            _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
            ptr += stride;
        }
    }

    if (elempack == 16)
    {
        // reduce sum 16 to 1
        // broadcast 1 to 16
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_permute_ps(_sum_avx512, _MM_PERM_CDAB));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_permute_ps(_sum_avx512, _MM_PERM_BADC));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_shuffle_f32x4(_sum_avx512, _sum_avx512, _MM_SHUFFLE(2, 3, 0, 1)));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_shuffle_f32x4(_sum_avx512, _sum_avx512, _MM_SHUFFLE(1, 0, 3, 2)));
    }
    if (elempack == 8)
    {
        // reduce sum 8,8 to 1,1
        // broadcast 1,1 to 8,8
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_permute_ps(_sum_avx512, _MM_PERM_CDAB));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_permute_ps(_sum_avx512, _MM_PERM_BADC));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_shuffle_f32x4(_sum_avx512, _sum_avx512, _MM_SHUFFLE(2, 3, 0, 1)));
    }
    if (elempack == 4)
    {
        // reduce sum 4,4,4,4 to 1,1,1,1
        // broadcast 1,1,1,1 to 4,4,4,4
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_permute_ps(_sum_avx512, _MM_PERM_CDAB));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_permute_ps(_sum_avx512, _MM_PERM_BADC));
    }

    _sum_avx512 = _mm512_div_ps(_mm512_set1_ps(1.f), _sum_avx512);

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_mul_ps(_p, _sum_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += stride;
        }
    }
}
#endif // __AVX512F__

static void softmax_unroll8(float* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    __m256 _max_avx = _mm256_set1_ps(-FLT_MAX);
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _max_avx = _mm256_max_ps(_max_avx, _p);
            ptr += stride;
        }
    }

    if (elempack == 8)
    {
        // reduce max 8 to 1
        // broadcast 1 to 8
        _max_avx = _mm256_max_ps(_max_avx, _mm256_permute_ps(_max_avx, _MM_SHUFFLE(2, 3, 0, 1)));
        _max_avx = _mm256_max_ps(_max_avx, _mm256_permute_ps(_max_avx, _MM_SHUFFLE(1, 0, 3, 2)));
        _max_avx = _mm256_max_ps(_max_avx, _mm256_permute2f128_ps(_max_avx, _max_avx, _MM_SHUFFLE(0, 0, 0, 1)));
    }
    if (elempack == 4)
    {
        // reduce max 4,4 to 1,1
        // broadcast 1,1 to 4,4
        _max_avx = _mm256_max_ps(_max_avx, _mm256_permute_ps(_max_avx, _MM_SHUFFLE(2, 3, 0, 1)));
        _max_avx = _mm256_max_ps(_max_avx, _mm256_permute_ps(_max_avx, _MM_SHUFFLE(1, 0, 3, 2)));
    }

    __m256 _sum_avx = _mm256_set1_ps(0.f);
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_sub_ps(_p, _max_avx);
            _p = exp256_ps(_p);
            _mm256_storeu_ps(ptr, _p);
            _sum_avx = _mm256_add_ps(_sum_avx, _p);
            ptr += stride;
        }
    }

    if (elempack == 8)
    {
        // reduce sum 8 to 1
        // broadcast 1 to 8
        _sum_avx = _mm256_add_ps(_sum_avx, _mm256_permute_ps(_sum_avx, _MM_SHUFFLE(2, 3, 0, 1)));
        _sum_avx = _mm256_add_ps(_sum_avx, _mm256_permute_ps(_sum_avx, _MM_SHUFFLE(1, 0, 3, 2)));
        _sum_avx = _mm256_add_ps(_sum_avx, _mm256_permute2f128_ps(_sum_avx, _sum_avx, _MM_SHUFFLE(0, 0, 0, 1)));
    }
    if (elempack == 4)
    {
        // reduce sum 4,4 to 1,1
        // broadcast 1,1 to 4,4
        _sum_avx = _mm256_add_ps(_sum_avx, _mm256_permute_ps(_sum_avx, _MM_SHUFFLE(2, 3, 0, 1)));
        _sum_avx = _mm256_add_ps(_sum_avx, _mm256_permute_ps(_sum_avx, _MM_SHUFFLE(1, 0, 3, 2)));
    }

    _sum_avx = _mm256_div_ps(_mm256_set1_ps(1.f), _sum_avx);

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_mul_ps(_p, _sum_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += stride;
        }
    }
}
#endif // __AVX__

static void softmax_unroll4(float* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    __m128 _max = _mm_set1_ps(-FLT_MAX);
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _max = _mm_max_ps(_max, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce max 4 to 1
        // broadcast 1 to 4
        _max = _mm_max_ps(_max, _mm_shuffle_ps(_max, _max, _MM_SHUFFLE(2, 3, 0, 1)));
        _max = _mm_max_ps(_max, _mm_shuffle_ps(_max, _max, _MM_SHUFFLE(1, 0, 3, 2)));
    }

    // reduce exp(x - max)
    __m128 _sum = _mm_set1_ps(0.f);
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_sub_ps(_p, _max);
            _p = exp_ps(_p);
            _mm_storeu_ps(ptr, _p);
            _sum = _mm_add_ps(_sum, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce sum 4 to 1
        // broadcast 1 to 4
        _sum = _mm_add_ps(_sum, _mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(2, 3, 0, 1)));
        _sum = _mm_add_ps(_sum, _mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(1, 0, 3, 2)));
    }

    _sum = _mm_div_ps(_mm_set1_ps(1.f), _sum);

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_mul_ps(_p, _sum);
            _mm_storeu_ps(ptr, _p);
            ptr += stride;
        }
    }
}
#endif // __SSE2__

static void softmax_unroll2(float* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    float max0 = -FLT_MAX;
    float max1 = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max0 = std::max(max0, ptr[0]);
            max1 = std::max(max1, ptr[1]);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum0 = 0.f;
    float sum1 = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v0 = expf(ptr[0] - max0);
            float v1 = expf(ptr[1] - max1);
            ptr[0] = v0;
            ptr[1] = v1;
            sum0 += v0;
            sum1 += v1;
            ptr += stride;
        }
    }

    sum0 = 1.f / sum0;
    sum1 = 1.f / sum1;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            ptr[0] *= sum0;
            ptr[1] *= sum1;
            ptr += stride;
        }
    }
}

static void softmax(float* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max = std::max(max, *ptr);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr += stride;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            *ptr *= sum;
            ptr += stride;
        }
    }
}

int Softmax_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax(ptr, size, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w * elempack;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll16(ptr, h, elempack, size);
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll8(ptr, h, elempack, size);
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll4(ptr, h, elempack, size);
        }
#endif // __SSE2__
        for (; i + 1 < size; i += 2)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll2(ptr, h, elempack, size);
        }
        for (; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, h, elempack, size);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            softmax(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d * elempack;
        const int stride = bottom_top_blob.cstep * elempack;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll16(ptr, channels, elempack, stride);
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll8(ptr, channels, elempack, stride);
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll4(ptr, channels, elempack, stride);
        }
#endif // __SSE2__
        for (; i + 1 < size; i += 2)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll2(ptr, channels, elempack, stride);
        }
        for (; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, channels, elempack, stride);
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            const int size = w * h * elempack;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                softmax_unroll16(ptr, d, 1, size);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                softmax_unroll8(ptr, d, 1, size);
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                softmax_unroll4(ptr, d, 1, size);
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                softmax(ptr, d, 1, size);
                ptr++;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                const int size = w * elempack;

                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; j + 15 < size; j += 16)
                {
                    softmax_unroll16(ptr, h, 1, size);
                    ptr += 16;
                }
#endif // __AVX512F__
                for (; j + 7 < size; j += 8)
                {
                    softmax_unroll8(ptr, h, 1, size);
                    ptr += 8;
                }
#endif // __AVX__
                for (; j + 3 < size; j += 4)
                {
                    softmax_unroll4(ptr, h, 1, size);
                    ptr += 4;
                }
#endif // __SSE2__
                for (; j < size; j++)
                {
                    softmax(ptr, h, 1, size);
                    ptr++;
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
