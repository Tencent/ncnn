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

#include "groupnorm_x86.h"

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

namespace ncnn {

GroupNorm_x86::GroupNorm_x86()
{
    // groupnorm slove in the packing dimension
    support_packing = false;
}

int GroupNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int channels_per_group = channels / group;

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            float sum = 0.f;
            float* ptr = bottom_top_blob_g;
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                for (; i + 15 < channels_per_group; i += 16)
                {
                    _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_loadu_ps(ptr));
                    ptr += 16;
                }
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                for (; i + 7 < channels_per_group; i += 8)
                {
                    _sum_avx = _mm256_add_ps(_sum_avx, _mm256_loadu_ps(ptr));
                    ptr += 8;
                }
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                for (; i + 3 < channels_per_group; i += 4)
                {
                    _sum = _mm_add_ps(_sum, _mm_loadu_ps(ptr));
                    ptr += 4;
                }
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                for (; i < channels_per_group; i++)
                {
                    sum += *ptr;
                    ptr++;
                }
            }

            float mean = sum / channels_per_group;

            float sqsum = 0.f;
            ptr = bottom_top_blob_g;
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sqsum_avx512 = _mm512_setzero_ps();
                __m512 _mean_avx512 = _mm512_set1_ps(mean);
                for (; i + 15 < channels_per_group; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_sub_ps(_p, _mean_avx512);
                    _p = _mm512_mul_ps(_p, _p);
                    _sqsum_avx512 = _mm512_add_ps(_p, _sqsum_avx512);
                    ptr += 16;
                }
                sqsum += _mm512_comp_reduce_add_ps(_sqsum_avx512);
#endif // __AVX512F__
                __m256 _sqsum_avx = _mm256_setzero_ps();
                __m256 _mean_avx = _mm256_set1_ps(mean);
                for (; i + 7 < channels_per_group; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_sub_ps(_p, _mean_avx);
                    _sqsum_avx = _mm256_comp_fmadd_ps(_p, _p, _sqsum_avx);
                    ptr += 8;
                }
                sqsum += _mm256_reduce_add_ps(_sqsum_avx);
#endif // __AVX__
                __m128 _sqsum = _mm_setzero_ps();
                __m128 _mean = _mm_set1_ps(mean);
                for (; i + 3 < channels_per_group; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_sub_ps(_p, _mean);
                    _sqsum = _mm_comp_fmadd_ps(_p, _p, _sqsum);
                    ptr += 4;
                }
                sqsum += _mm_reduce_add_ps(_sqsum);
#endif // __SSE2__
                for (; i < channels_per_group; i++)
                {
                    float tmp = *ptr - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }
            }

            float scale1 = 1.f / sqrtf(sqsum / channels_per_group + eps);
            float scale2 = -mean * scale1;

            ptr = bottom_top_blob_g;
            if (affine)
            {
                int i = 0;
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _scale1_avx512 = _mm512_set1_ps(scale1);
                __m512 _scale2_avx512 = _mm512_set1_ps(scale2);
                for (; i + 15 < channels_per_group; i += 16)
                {
                    __m512 _gamma = _mm512_loadu_ps(gamma);
                    __m512 _beta = _mm512_loadu_ps(beta);
                    __m512 _p = _mm512_loadu_ps(ptr);
                    __m512 _a = _mm512_mul_ps(_gamma, _scale1_avx512);
                    __m512 _b = _mm512_add_ps(_mm512_mul_ps(_gamma, _scale2_avx512), _beta);
                    _p = _mm512_add_ps(_mm512_mul_ps(_p, _a), _b);
                    _mm512_storeu_ps(ptr, _p);
                    gamma += 16;
                    beta += 16;
                    ptr += 16;
                }
#endif // __AVX512F__
                __m256 _scale1_avx = _mm256_set1_ps(scale1);
                __m256 _scale2_avx = _mm256_set1_ps(scale2);
                for (; i + 7 < channels_per_group; i += 8)
                {
                    __m256 _gamma = _mm256_loadu_ps(gamma);
                    __m256 _beta = _mm256_loadu_ps(beta);
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _a = _mm256_mul_ps(_gamma, _scale1_avx);
                    __m256 _b = _mm256_comp_fmadd_ps(_gamma, _scale2_avx, _beta);
                    _p = _mm256_comp_fmadd_ps(_p, _a, _b);
                    _mm256_storeu_ps(ptr, _p);
                    gamma += 8;
                    beta += 8;
                    ptr += 8;
                }
#endif // __AVX__
                __m128 _scale1 = _mm_set1_ps(scale1);
                __m128 _scale2 = _mm_set1_ps(scale2);
                for (; i + 3 < channels_per_group; i += 4)
                {
                    __m128 _gamma = _mm_loadu_ps(gamma);
                    __m128 _beta = _mm_loadu_ps(beta);
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _a = _mm_mul_ps(_gamma, _scale1);
                    __m128 _b = _mm_comp_fmadd_ps(_gamma, _scale2, _beta);
                    _p = _mm_comp_fmadd_ps(_p, _a, _b);
                    _mm_storeu_ps(ptr, _p);
                    gamma += 4;
                    beta += 4;
                    ptr += 4;
                }
#endif // __SSE2__
                for (; i < channels_per_group; i++)
                {
                    float a = *gamma * scale1;
                    float b = *gamma * scale2 + *beta;
                    *ptr = *ptr * a + b;
                    gamma++;
                    beta++;
                    ptr++;
                }
            }
            else
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _scale1_avx512 = _mm512_set1_ps(scale1);
                __m512 _scale2_avx512 = _mm512_set1_ps(scale2);
                for (; i + 15 < channels_per_group; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_add_ps(_mm512_mul_ps(_p, _scale1_avx512), _scale2_avx512);
                    _mm512_storeu_ps(ptr, _p);
                    ptr += 16;
                }
#endif // __AVX512F__
                __m256 _scale1_avx = _mm256_set1_ps(scale1);
                __m256 _scale2_avx = _mm256_set1_ps(scale2);
                for (; i + 7 < channels_per_group; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_comp_fmadd_ps(_p, _scale1_avx, _scale2_avx);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                }
#endif // __AVX__
                __m128 _scale1 = _mm_set1_ps(scale1);
                __m128 _scale2 = _mm_set1_ps(scale2);
                for (; i + 3 < channels_per_group; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_comp_fmadd_ps(_p, _scale1, _scale2);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                }
#endif // __SSE2__
                for (; i < channels_per_group; i++)
                {
                    *ptr = *ptr * scale1 + scale2;
                    ptr++;
                }
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int size = channels_per_group * w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.row_range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            // mean and var
            float sum = 0.f;
            float* ptr = bottom_top_blob_g;
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                for (; i + 15 < size; i += 16)
                {
                    _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_loadu_ps(ptr));
                    ptr += 16;
                }
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                for (; i + 7 < size; i += 8)
                {
                    _sum_avx = _mm256_add_ps(_sum_avx, _mm256_loadu_ps(ptr));
                    ptr += 8;
                }
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                for (; i + 3 < size; i += 4)
                {
                    _sum = _mm_add_ps(_sum, _mm_loadu_ps(ptr));
                    ptr += 4;
                }
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                for (; i < size; i++)
                {
                    sum += *ptr;
                    ptr++;
                }
            }

            float mean = sum / size;

            float sqsum = 0.f;
            ptr = bottom_top_blob_g;
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sqsum_avx512 = _mm512_setzero_ps();
                __m512 _mean_avx512 = _mm512_set1_ps(mean);
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_sub_ps(_p, _mean_avx512);
                    _p = _mm512_mul_ps(_p, _p);
                    _sqsum_avx512 = _mm512_add_ps(_p, _sqsum_avx512);
                    ptr += 16;
                }
                sqsum += _mm512_comp_reduce_add_ps(_sqsum_avx512);
#endif // __AVX512F__
                __m256 _sqsum_avx = _mm256_setzero_ps();
                __m256 _mean_avx = _mm256_set1_ps(mean);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_sub_ps(_p, _mean_avx);
                    _sqsum_avx = _mm256_comp_fmadd_ps(_p, _p, _sqsum_avx);
                    ptr += 8;
                }
                sqsum += _mm256_reduce_add_ps(_sqsum_avx);
#endif // __AVX__
                __m128 _sqsum = _mm_setzero_ps();
                __m128 _mean = _mm_set1_ps(mean);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_sub_ps(_p, _mean);
                    _sqsum = _mm_comp_fmadd_ps(_p, _p, _sqsum);
                    ptr += 4;
                }
                sqsum += _mm_reduce_add_ps(_sqsum);
#endif // __SSE2__
                for (; i < size; i++)
                {
                    float tmp = *ptr - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }
            }

            float scale1 = 1.f / sqrtf(sqsum / size + eps);
            float scale2 = -mean * scale1;

            ptr = bottom_top_blob_g;
            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                for (int q = 0; q < channels_per_group; q++)
                {
                    float a = *gamma * scale1;
                    float b = *gamma * scale2 + *beta;
                    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _a_avx512 = _mm512_set1_ps(a);
                    __m512 _b_avx512 = _mm512_set1_ps(b);
                    for (; i + 15 < w; i += 16)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr);
                        _p = _mm512_add_ps(_mm512_mul_ps(_p, _a_avx512), _b_avx512);
                        _mm512_storeu_ps(ptr, _p);
                        ptr += 16;
                    }
#endif // __AVX512F__
                    __m256 _a_avx = _mm256_set1_ps(a);
                    __m256 _b_avx = _mm256_set1_ps(b);
                    for (; i + 7 < w; i += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_comp_fmadd_ps(_p, _a_avx, _b_avx);
                        _mm256_storeu_ps(ptr, _p);
                        ptr += 8;
                    }
#endif // __AVX__
                    __m128 _a = _mm_set1_ps(a);
                    __m128 _b = _mm_set1_ps(b);
                    for (; i + 3 < w; i += 4)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_comp_fmadd_ps(_p, _a, _b);
                        _mm_storeu_ps(ptr, _p);
                        ptr += 4;
                    }
#endif // __SSE2__
                    for (; i < w; i++)
                    {
                        *ptr = *ptr * a + b;
                        ptr++;
                    }
                    gamma++;
                    beta++;
                }
            }
            else
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _scale1_avx512 = _mm512_set1_ps(scale1);
                __m512 _scale2_avx512 = _mm512_set1_ps(scale2);
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_add_ps(_mm512_mul_ps(_p, _scale1_avx512), _scale2_avx512);
                    _mm512_storeu_ps(ptr, _p);
                    ptr += 16;
                }
#endif // __AVX512F__
                __m256 _scale1_avx = _mm256_set1_ps(scale1);
                __m256 _scale2_avx = _mm256_set1_ps(scale2);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_comp_fmadd_ps(_p, _scale1_avx, _scale2_avx);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                }
#endif // __AVX__
                __m128 _scale1 = _mm_set1_ps(scale1);
                __m128 _scale2 = _mm_set1_ps(scale2);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_comp_fmadd_ps(_p, _scale1, _scale2);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *ptr = *ptr * scale1 + scale2;
                    ptr++;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.channel_range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            // mean and var
            float sum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr = bottom_top_blob_g.channel(q);
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                for (; i + 15 < size; i += 16)
                {
                    _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_loadu_ps(ptr));
                    ptr += 16;
                }
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                for (; i + 7 < size; i += 8)
                {
                    _sum_avx = _mm256_add_ps(_sum_avx, _mm256_loadu_ps(ptr));
                    ptr += 8;
                }
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                for (; i + 3 < size; i += 4)
                {
                    _sum = _mm_add_ps(_sum, _mm_loadu_ps(ptr));
                    ptr += 4;
                }
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                for (; i < size; i++)
                {
                    sum += *ptr;
                    ptr++;
                }
            }

            float mean = sum / (channels_per_group * size);

            float sqsum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr = bottom_top_blob_g.channel(q);
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sqsum_avx512 = _mm512_setzero_ps();
                __m512 _mean_avx512 = _mm512_set1_ps(mean);
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_sub_ps(_p, _mean_avx512);
                    _p = _mm512_mul_ps(_p, _p);
                    _sqsum_avx512 = _mm512_add_ps(_p, _sqsum_avx512);
                    ptr += 16;
                }
                sqsum += _mm512_comp_reduce_add_ps(_sqsum_avx512);
#endif // __AVX512F__
                __m256 _sqsum_avx = _mm256_setzero_ps();
                __m256 _mean_avx = _mm256_set1_ps(mean);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_sub_ps(_p, _mean_avx);
                    _sqsum_avx = _mm256_comp_fmadd_ps(_p, _p, _sqsum_avx);
                    ptr += 8;
                }
                sqsum += _mm256_reduce_add_ps(_sqsum_avx);
#endif // __AVX__
                __m128 _sqsum = _mm_setzero_ps();
                __m128 _mean = _mm_set1_ps(mean);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_sub_ps(_p, _mean);
                    _sqsum = _mm_comp_fmadd_ps(_p, _p, _sqsum);
                    ptr += 4;
                }
                sqsum += _mm_reduce_add_ps(_sqsum);
#endif // __SSE2__
                for (; i < size; i++)
                {
                    float tmp = *ptr - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }
            }

            float scale1 = 1.f / sqrtf(sqsum / (channels_per_group * size) + eps);
            float scale2 = -mean * scale1;

            const float* gamma = gamma_data_g;
            const float* beta = beta_data_g;
            for (int q = 0; q < channels_per_group; q++)
            {
                float a = scale1;
                float b = scale2;
                if (affine)
                {
                    a = *gamma * a;
                    b = *gamma * b + *beta;
                }

                float* ptr = bottom_top_blob_g.channel(q);
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _a_avx512 = _mm512_set1_ps(a);
                __m512 _b_avx512 = _mm512_set1_ps(b);
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_add_ps(_mm512_mul_ps(_p, _a_avx512), _b_avx512);
                    _mm512_storeu_ps(ptr, _p);
                    ptr += 16;
                }
#endif // __AVX512F__
                __m256 _a_avx = _mm256_set1_ps(a);
                __m256 _b_avx = _mm256_set1_ps(b);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_comp_fmadd_ps(_p, _a_avx, _b_avx);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                }
#endif // __AVX__
                __m128 _a = _mm_set1_ps(a);
                __m128 _b = _mm_set1_ps(b);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_comp_fmadd_ps(_p, _a, _b);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *ptr = *ptr * a + b;
                    ptr++;
                }

                gamma++;
                beta++;
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
