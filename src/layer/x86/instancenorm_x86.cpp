// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

int InstanceNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        const float* ptr0 = ptr;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _sum_avx512 = _mm512_setzero_ps();
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr0);
            _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
            ptr0 += 16;
        }
        sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
        __m256 _sum_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr0);
            _sum_avx = _mm256_add_ps(_sum_avx, _p);
            ptr0 += 8;
        }
        sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
        __m128 _sum = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr0);
            _sum = _mm_add_ps(_sum, _p);
            ptr0 += 4;
        }
        sum += _mm_reduce_add_ps(_sum);
#endif //__SSE2__
        for (; i < size; i++)
        {
            sum += ptr0[0];
            ptr0++;
        }

        float mean = sum / size;
        float tmp = 0.f;

        ptr0 = ptr;
        i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _sqsum_avx512 = _mm512_setzero_ps();
        __m512 _mean_avx512 = _mm512_set1_ps(mean);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr0);
            __m512 _diff = _mm512_sub_ps(_p, _mean_avx512);
            _sqsum_avx512 = _mm512_fmadd_ps(_diff, _diff, _sqsum_avx512);
            ptr0 += 16;
        }
        sqsum += _mm512_comp_reduce_add_ps(_sqsum_avx512);
#endif // __AVX512F
        __m256 _sqsum_avx = _mm256_setzero_ps();
        __m256 _mean_avx = _mm256_set1_ps(mean);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr0);
            __m256 _diff = _mm256_sub_ps(_p, _mean_avx);
            _sqsum_avx = _mm256_comp_fmadd_ps(_diff, _diff, _sqsum_avx);
            ptr0 += 8;
        }
        sqsum += _mm256_reduce_add_ps(_sqsum_avx);
#endif // __AVX__
        __m128 _sqsum = _mm_setzero_ps();
        __m128 _mean = _mm_set1_ps(mean);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr0);
            __m128 _diff = _mm_sub_ps(_p, _mean);
            _sqsum = _mm_comp_fmadd_ps(_diff, _diff, _sqsum);
            ptr0 += 4;
        }
        sqsum += _mm_reduce_add_ps(_sqsum);
#endif // __SSE2__
        for (; i < size; i++)
        {
            tmp = ptr0[0] - mean;
            sqsum += tmp * tmp;
            ptr0++;
        }
        float var = sqsum / size;
        // the var maybe minus due to accuracy

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
            b = -mean * a;
        }

        i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _va_avx512 = _mm512_set1_ps(a);
        __m512 _vb_avx512 = _mm512_set1_ps(b);
        for (; i + 16 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_fmadd_ps(_p, _va_avx512, _vb_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _va_avx = _mm256_set1_ps(a);
        __m256 _vb_vax = _mm256_set1_ps(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_comp_fmadd_ps(_p, _va_avx, _vb_vax);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _va = _mm_set1_ps(a);
        __m128 _vb = _mm_set1_ps(b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_comp_fmadd_ps(_p, _va, _vb);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * a + b;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
