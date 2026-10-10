// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

int Bias_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_ptr[q];

        int i = 0;
#if __SSE2__
#if __AVX__
        {
            __m256 _bias256 = _mm256_set1_ps(bias);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _outp = _mm256_add_ps(_p, _bias256);
                _mm256_storeu_ps(ptr, _outp);

                ptr += 8;
            }
        }
#endif // __AVX__
        {
            __m128 _bias = _mm_set1_ps(bias);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _outp = _mm_add_ps(_p, _bias);
                _mm_storeu_ps(ptr, _outp);

                ptr += 4;
            }
        }
#endif // __SSE2__

        for (; i < size; i++)
        {
            *ptr = *ptr + bias;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
