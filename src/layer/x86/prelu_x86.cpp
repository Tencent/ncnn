// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "prelu_x86.h"

#include "x86_activation.h"

namespace ncnn {

PReLU_x86::PReLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int PReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        const int size = w * elempack;

        if (num_slope > 1)
        {
            const float* slope = slope_data;
            int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size - 15; j += 16)
            {
                float* ptr = (float*)bottom_top_blob + j;
                __m512 _p512 = _mm512_loadu_ps(ptr);
                __m512 _slope512 = _mm512_loadu_ps(slope + j);
                _mm512_storeu_ps(ptr, prelu_avx512(_p512, _slope512));
            }
            i = size / 16 * 16;
#endif // __AVX512F__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size - 7; j += 8)
            {
                float* ptr = (float*)bottom_top_blob + j;
                __m256 _p256 = _mm256_loadu_ps(ptr);
                __m256 _slope256 = _mm256_loadu_ps(slope + j);
                _mm256_storeu_ps(ptr, prelu_avx(_p256, _slope256));
            }
            i = size / 8 * 8;
#endif // __AVX__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size - 3; j += 4)
            {
                float* ptr = (float*)bottom_top_blob + j;
                __m128 _p128 = _mm_loadu_ps(ptr);
                __m128 _slope128 = _mm_loadu_ps(slope + j);
                _mm_storeu_ps(ptr, prelu_sse(_p128, _slope128));
            }
            i = size / 4 * 4;
#endif // __SSE2__
            float* ptr = (float*)bottom_top_blob;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope_data[j];
            }
        }
        else
        {
            int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _slope512 = _mm512_set1_ps(slope_data[0]);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size - 15; j += 16)
            {
                float* ptr = (float*)bottom_top_blob + j;
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, prelu_avx512(_p512, _slope512));
            }
            i = size / 16 * 16;
#endif // __AVX512F__
            __m256 _slope256 = _mm256_set1_ps(slope_data[0]);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size - 7; j += 8)
            {
                float* ptr = (float*)bottom_top_blob + j;
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, prelu_avx(_p256, _slope256));
            }
            i = size / 8 * 8;
#endif // __AVX__
            __m128 _slope128 = _mm_set1_ps(slope_data[0]);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size - 3; j += 4)
            {
                float* ptr = (float*)bottom_top_blob + j;
                __m128 _p128 = _mm_loadu_ps(ptr);
                _mm_storeu_ps(ptr, prelu_sse(_p128, _slope128));
            }
            i = size / 4 * 4;
#endif // __SSE2__
            float slope = slope_data[0];
            float* ptr = (float*)bottom_top_blob;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = i; j < size; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope;
            }
        }

#if __SSE2__
#if __AVX__
#if __AVX512F__
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }
    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            int j = 0;

            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];
#if __SSE2__
            __m128 _slope128 = num_slope > 1 && (elempack == 4) ? _mm_loadu_ps((const float*)slope_data + i * 4) : _mm_set1_ps(slope);
#if __AVX__
            __m256 _slope256 = num_slope > 1 && (elempack == 8) ? _mm256_loadu_ps((const float*)slope_data + i * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_slope128), _slope128, 1);
#if __AVX512F__
            __m512 _slope512 = num_slope > 1 && (elempack == 16) ? _mm512_loadu_ps((const float*)slope_data + i * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_slope256), _slope256, 1);

            for (; j + 15 < size; j += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, prelu_avx512(_p512, _slope512));

                ptr += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < size; j += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, prelu_avx(_p256, _slope256));

                ptr += 8;
            }
#endif // __AVX__
            for (; j + 3 < size; j += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _mm_storeu_ps(ptr, prelu_sse(_p128, _slope128));

                ptr += 4;
            }
#endif // __SSE2__

            for (; j < size; j++)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }
    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];
#if __SSE2__
            __m128 _slope128 = num_slope > 1 && (elempack == 4) ? _mm_loadu_ps((const float*)slope_data + q * 4) : _mm_set1_ps(slope);
#if __AVX__
            __m256 _slope256 = num_slope > 1 && (elempack == 8) ? _mm256_loadu_ps((const float*)slope_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_slope128), _slope128, 1);
#if __AVX512F__
            __m512 _slope512 = num_slope > 1 && (elempack == 16) ? _mm512_loadu_ps((const float*)slope_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_slope256), _slope256, 1);
            for (; i + 15 < size; i += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, prelu_avx512(_p512, _slope512));

                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, prelu_avx(_p256, _slope256));

                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _mm_storeu_ps(ptr, prelu_sse(_p128, _slope128));

                ptr += 4;
            }
#endif // __SSE2__

            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
