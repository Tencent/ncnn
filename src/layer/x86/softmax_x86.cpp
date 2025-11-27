// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
#include "cpu.h"

namespace ncnn {

#if __SSE2__
static NCNN_FORCEINLINE __m128 _mm_rcp_nr_ps(const __m128& x)
{
    __m128 y = _mm_rcp_ps(x);                               // approx
    __m128 t = _mm_comp_fnmadd_ps(x, y, _mm_set1_ps(2.0f)); // (2 - x*y)
    y = _mm_mul_ps(y, t);
    return y; // 1 NR step
}
#endif

#if __AVX__
static NCNN_FORCEINLINE __m256 _mm256_rcp_nr_ps(const __m256& x)
{
    __m256 y = _mm256_rcp_ps(x);
    __m256 t = _mm256_comp_fnmadd_ps(x, y, _mm256_set1_ps(2.0f));
    y = _mm256_mul_ps(y, t);
    return y;
}
#endif

#if __AVX512F__
static NCNN_FORCEINLINE __m512 _mm512_rcp_nr_ps(const __m512& x)
{
    __m512 y = _mm512_rcp14_ps(x);
    __m512 t = _mm512_fnmadd_ps(x, y, _mm512_set1_ps(2.0f));
    return _mm512_mul_ps(y, t);
}
#endif

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
            max = std::max(max, *ptr++);
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
        _sum_avx512 = _mm512_rcp_nr_ps(_sum_avx512);
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

        _sum_avx = _mm256_rcp_nr_ps(_sum_avx);

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

        _sum = _mm_rcp_nr_ps(_sum);

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
static void softmax_pack16(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 16 * 2);
            __m512 _p3 = _mm512_loadu_ps(ptr + 16 * 3);
            __m512 _p4 = _mm512_loadu_ps(ptr + 16 * 4);
            __m512 _p5 = _mm512_loadu_ps(ptr + 16 * 5);
            __m512 _p6 = _mm512_loadu_ps(ptr + 16 * 6);
            __m512 _p7 = _mm512_loadu_ps(ptr + 16 * 7);
            __m512 _p8 = _mm512_loadu_ps(ptr + 16 * 8);
            __m512 _p9 = _mm512_loadu_ps(ptr + 16 * 9);
            __m512 _pa = _mm512_loadu_ps(ptr + 16 * 10);
            __m512 _pb = _mm512_loadu_ps(ptr + 16 * 11);
            __m512 _pc = _mm512_loadu_ps(ptr + 16 * 12);
            __m512 _pd = _mm512_loadu_ps(ptr + 16 * 13);
            __m512 _pe = _mm512_loadu_ps(ptr + 16 * 14);
            __m512 _pf = _mm512_loadu_ps(ptr + 16 * 15);

            __m512 _tmp0 = _mm512_unpacklo_ps(_p0, _p1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_p0, _p1);
            __m512 _tmp2 = _mm512_unpacklo_ps(_p2, _p3);
            __m512 _tmp3 = _mm512_unpackhi_ps(_p2, _p3);
            __m512 _tmp4 = _mm512_unpacklo_ps(_p4, _p5);
            __m512 _tmp5 = _mm512_unpackhi_ps(_p4, _p5);
            __m512 _tmp6 = _mm512_unpacklo_ps(_p6, _p7);
            __m512 _tmp7 = _mm512_unpackhi_ps(_p6, _p7);
            __m512 _tmp8 = _mm512_unpacklo_ps(_p8, _p9);
            __m512 _tmp9 = _mm512_unpackhi_ps(_p8, _p9);
            __m512 _tmpa = _mm512_unpacklo_ps(_pa, _pb);
            __m512 _tmpb = _mm512_unpackhi_ps(_pa, _pb);
            __m512 _tmpc = _mm512_unpacklo_ps(_pc, _pd);
            __m512 _tmpd = _mm512_unpackhi_ps(_pc, _pd);
            __m512 _tmpe = _mm512_unpacklo_ps(_pe, _pf);
            __m512 _tmpf = _mm512_unpackhi_ps(_pe, _pf);

            __m512 _max01 = _mm512_max_ps(_tmp0, _tmp1);
            __m512 _max23 = _mm512_max_ps(_tmp2, _tmp3);
            __m512 _max45 = _mm512_max_ps(_tmp4, _tmp5);
            __m512 _max67 = _mm512_max_ps(_tmp6, _tmp7);
            __m512 _max89 = _mm512_max_ps(_tmp8, _tmp9);
            __m512 _maxab = _mm512_max_ps(_tmpa, _tmpb);
            __m512 _maxcd = _mm512_max_ps(_tmpc, _tmpd);
            __m512 _maxef = _mm512_max_ps(_tmpe, _tmpf);

            _tmp0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_max01), _mm512_castps_pd(_max23)));
            _tmp1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_max01), _mm512_castps_pd(_max23)));
            _tmp2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_max45), _mm512_castps_pd(_max67)));
            _tmp3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_max45), _mm512_castps_pd(_max67)));
            _tmp4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_max89), _mm512_castps_pd(_maxab)));
            _tmp5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_max89), _mm512_castps_pd(_maxab)));
            _tmp6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_maxcd), _mm512_castps_pd(_maxef)));
            _tmp7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_maxcd), _mm512_castps_pd(_maxef)));

            __m512 _max0123 = _mm512_max_ps(_tmp0, _tmp1);
            __m512 _max4567 = _mm512_max_ps(_tmp2, _tmp3);
            __m512 _max89ab = _mm512_max_ps(_tmp4, _tmp5);
            __m512 _maxcdef = _mm512_max_ps(_tmp6, _tmp7);

            _tmp0 = _mm512_shuffle_f32x4(_max0123, _max4567, _MM_SHUFFLE(1, 0, 1, 0));
            _tmp1 = _mm512_shuffle_f32x4(_max0123, _max4567, _MM_SHUFFLE(3, 2, 3, 2));
            _tmp2 = _mm512_shuffle_f32x4(_max89ab, _maxcdef, _MM_SHUFFLE(1, 0, 1, 0));
            _tmp3 = _mm512_shuffle_f32x4(_max89ab, _maxcdef, _MM_SHUFFLE(3, 2, 3, 2));

            __m512 _max01234567 = _mm512_max_ps(_tmp0, _tmp1);
            __m512 _max89abcdef = _mm512_max_ps(_tmp2, _tmp3);

            _tmp0 = _mm512_shuffle_f32x4(_max01234567, _max89abcdef, _MM_SHUFFLE(2, 0, 2, 0));
            _tmp1 = _mm512_shuffle_f32x4(_max01234567, _max89abcdef, _MM_SHUFFLE(3, 1, 3, 1));

            __m512 _max1 = _mm512_max_ps(_tmp0, _tmp1);

            __m512 _max = _mm512_loadu_ps(maxptr);
            _max = _mm512_max_ps(_max, _max1);
            _mm512_storeu_ps(maxptr, _max);
            ptr += 256;
            maxptr += 16;
        }
        for (; j < size1; j++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            *maxptr = std::max(*maxptr, _mm512_comp_reduce_max_ps(_p));
            ptr += 16;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 16 * 2);
            __m512 _p3 = _mm512_loadu_ps(ptr + 16 * 3);
            __m512 _p4 = _mm512_loadu_ps(ptr + 16 * 4);
            __m512 _p5 = _mm512_loadu_ps(ptr + 16 * 5);
            __m512 _p6 = _mm512_loadu_ps(ptr + 16 * 6);
            __m512 _p7 = _mm512_loadu_ps(ptr + 16 * 7);
            __m512 _p8 = _mm512_loadu_ps(ptr + 16 * 8);
            __m512 _p9 = _mm512_loadu_ps(ptr + 16 * 9);
            __m512 _pa = _mm512_loadu_ps(ptr + 16 * 10);
            __m512 _pb = _mm512_loadu_ps(ptr + 16 * 11);
            __m512 _pc = _mm512_loadu_ps(ptr + 16 * 12);
            __m512 _pd = _mm512_loadu_ps(ptr + 16 * 13);
            __m512 _pe = _mm512_loadu_ps(ptr + 16 * 14);
            __m512 _pf = _mm512_loadu_ps(ptr + 16 * 15);
            _p0 = exp512_ps(_mm512_sub_ps(_p0, _mm512_set1_ps(maxptr[0])));
            _p1 = exp512_ps(_mm512_sub_ps(_p1, _mm512_set1_ps(maxptr[1])));
            _p2 = exp512_ps(_mm512_sub_ps(_p2, _mm512_set1_ps(maxptr[2])));
            _p3 = exp512_ps(_mm512_sub_ps(_p3, _mm512_set1_ps(maxptr[3])));
            _p4 = exp512_ps(_mm512_sub_ps(_p4, _mm512_set1_ps(maxptr[4])));
            _p5 = exp512_ps(_mm512_sub_ps(_p5, _mm512_set1_ps(maxptr[5])));
            _p6 = exp512_ps(_mm512_sub_ps(_p6, _mm512_set1_ps(maxptr[6])));
            _p7 = exp512_ps(_mm512_sub_ps(_p7, _mm512_set1_ps(maxptr[7])));
            _p8 = exp512_ps(_mm512_sub_ps(_p8, _mm512_set1_ps(maxptr[8])));
            _p9 = exp512_ps(_mm512_sub_ps(_p9, _mm512_set1_ps(maxptr[9])));
            _pa = exp512_ps(_mm512_sub_ps(_pa, _mm512_set1_ps(maxptr[10])));
            _pb = exp512_ps(_mm512_sub_ps(_pb, _mm512_set1_ps(maxptr[11])));
            _pc = exp512_ps(_mm512_sub_ps(_pc, _mm512_set1_ps(maxptr[12])));
            _pd = exp512_ps(_mm512_sub_ps(_pd, _mm512_set1_ps(maxptr[13])));
            _pe = exp512_ps(_mm512_sub_ps(_pe, _mm512_set1_ps(maxptr[14])));
            _pf = exp512_ps(_mm512_sub_ps(_pf, _mm512_set1_ps(maxptr[15])));
            _mm512_storeu_ps(ptr, _p0);
            _mm512_storeu_ps(ptr + 16, _p1);
            _mm512_storeu_ps(ptr + 16 * 2, _p2);
            _mm512_storeu_ps(ptr + 16 * 3, _p3);
            _mm512_storeu_ps(ptr + 16 * 4, _p4);
            _mm512_storeu_ps(ptr + 16 * 5, _p5);
            _mm512_storeu_ps(ptr + 16 * 6, _p6);
            _mm512_storeu_ps(ptr + 16 * 7, _p7);
            _mm512_storeu_ps(ptr + 16 * 8, _p8);
            _mm512_storeu_ps(ptr + 16 * 9, _p9);
            _mm512_storeu_ps(ptr + 16 * 10, _pa);
            _mm512_storeu_ps(ptr + 16 * 11, _pb);
            _mm512_storeu_ps(ptr + 16 * 12, _pc);
            _mm512_storeu_ps(ptr + 16 * 13, _pd);
            _mm512_storeu_ps(ptr + 16 * 14, _pe);
            _mm512_storeu_ps(ptr + 16 * 15, _pf);

            __m512 _tmp0 = _mm512_unpacklo_ps(_p0, _p1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_p0, _p1);
            __m512 _tmp2 = _mm512_unpacklo_ps(_p2, _p3);
            __m512 _tmp3 = _mm512_unpackhi_ps(_p2, _p3);
            __m512 _tmp4 = _mm512_unpacklo_ps(_p4, _p5);
            __m512 _tmp5 = _mm512_unpackhi_ps(_p4, _p5);
            __m512 _tmp6 = _mm512_unpacklo_ps(_p6, _p7);
            __m512 _tmp7 = _mm512_unpackhi_ps(_p6, _p7);
            __m512 _tmp8 = _mm512_unpacklo_ps(_p8, _p9);
            __m512 _tmp9 = _mm512_unpackhi_ps(_p8, _p9);
            __m512 _tmpa = _mm512_unpacklo_ps(_pa, _pb);
            __m512 _tmpb = _mm512_unpackhi_ps(_pa, _pb);
            __m512 _tmpc = _mm512_unpacklo_ps(_pc, _pd);
            __m512 _tmpd = _mm512_unpackhi_ps(_pc, _pd);
            __m512 _tmpe = _mm512_unpacklo_ps(_pe, _pf);
            __m512 _tmpf = _mm512_unpackhi_ps(_pe, _pf);

            __m512 _sum01 = _mm512_add_ps(_tmp0, _tmp1);
            __m512 _sum23 = _mm512_add_ps(_tmp2, _tmp3);
            __m512 _sum45 = _mm512_add_ps(_tmp4, _tmp5);
            __m512 _sum67 = _mm512_add_ps(_tmp6, _tmp7);
            __m512 _sum89 = _mm512_add_ps(_tmp8, _tmp9);
            __m512 _sumab = _mm512_add_ps(_tmpa, _tmpb);
            __m512 _sumcd = _mm512_add_ps(_tmpc, _tmpd);
            __m512 _sumef = _mm512_add_ps(_tmpe, _tmpf);

            _tmp0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_sum01), _mm512_castps_pd(_sum23)));
            _tmp1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_sum01), _mm512_castps_pd(_sum23)));
            _tmp2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_sum45), _mm512_castps_pd(_sum67)));
            _tmp3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_sum45), _mm512_castps_pd(_sum67)));
            _tmp4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_sum89), _mm512_castps_pd(_sumab)));
            _tmp5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_sum89), _mm512_castps_pd(_sumab)));
            _tmp6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_sumcd), _mm512_castps_pd(_sumef)));
            _tmp7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_sumcd), _mm512_castps_pd(_sumef)));

            __m512 _sum0123 = _mm512_add_ps(_tmp0, _tmp1);
            __m512 _sum4567 = _mm512_add_ps(_tmp2, _tmp3);
            __m512 _sum89ab = _mm512_add_ps(_tmp4, _tmp5);
            __m512 _sumcdef = _mm512_add_ps(_tmp6, _tmp7);

            _tmp0 = _mm512_shuffle_f32x4(_sum0123, _sum4567, _MM_SHUFFLE(1, 0, 1, 0));
            _tmp1 = _mm512_shuffle_f32x4(_sum0123, _sum4567, _MM_SHUFFLE(3, 2, 3, 2));
            _tmp2 = _mm512_shuffle_f32x4(_sum89ab, _sumcdef, _MM_SHUFFLE(1, 0, 1, 0));
            _tmp3 = _mm512_shuffle_f32x4(_sum89ab, _sumcdef, _MM_SHUFFLE(3, 2, 3, 2));

            __m512 _sum01234567 = _mm512_add_ps(_tmp0, _tmp1);
            __m512 _sum89abcdef = _mm512_add_ps(_tmp2, _tmp3);

            _tmp0 = _mm512_shuffle_f32x4(_sum01234567, _sum89abcdef, _MM_SHUFFLE(2, 0, 2, 0));
            _tmp1 = _mm512_shuffle_f32x4(_sum01234567, _sum89abcdef, _MM_SHUFFLE(3, 1, 3, 1));

            __m512 _sum1 = _mm512_add_ps(_tmp0, _tmp1);

            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_add_ps(_sum, _sum1);
            _mm512_storeu_ps(sumptr, _sum);
            ptr += 256;
            maxptr += 16;
            sumptr += 16;
        }
        for (; j < size1; j++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _max = _mm512_set1_ps(*maxptr);
            _p = exp512_ps(_mm512_sub_ps(_p, _max));
            _mm512_storeu_ps(ptr, _p);
            *sumptr += _mm512_comp_reduce_add_ps(_p);
            ptr += 16;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 32);
            __m512 _p3 = _mm512_loadu_ps(ptr + 48);
            _p0 = _mm512_mul_ps(_p0, _mm512_set1_ps(sumptr[0]));
            _p1 = _mm512_mul_ps(_p1, _mm512_set1_ps(sumptr[1]));
            _p2 = _mm512_mul_ps(_p2, _mm512_set1_ps(sumptr[2]));
            _p3 = _mm512_mul_ps(_p3, _mm512_set1_ps(sumptr[3]));
            _mm512_storeu_ps(ptr, _p0);
            _mm512_storeu_ps(ptr + 16, _p1);
            _mm512_storeu_ps(ptr + 32, _p2);
            _mm512_storeu_ps(ptr + 48, _p3);
            ptr += 64;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _sum = _mm512_set1_ps(*sumptr);
            _p = _mm512_mul_ps(_p, _sum);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
            sumptr++;
        }
    }
}
#endif // __AVX512F__

static void softmax_pack8(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __AVX512F__
        __m512i _pidx = _mm512_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15);
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 16 * 2);
            __m512 _p3 = _mm512_loadu_ps(ptr + 16 * 3);
            __m512 _p4 = _mm512_loadu_ps(ptr + 16 * 4);
            __m512 _p5 = _mm512_loadu_ps(ptr + 16 * 5);
            __m512 _p6 = _mm512_loadu_ps(ptr + 16 * 6);
            __m512 _p7 = _mm512_loadu_ps(ptr + 16 * 7);

            __m512 _tmp0 = _mm512_unpacklo_ps(_p0, _p1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_p0, _p1);
            __m512 _tmp2 = _mm512_unpacklo_ps(_p2, _p3);
            __m512 _tmp3 = _mm512_unpackhi_ps(_p2, _p3);
            __m512 _tmp4 = _mm512_unpacklo_ps(_p4, _p5);
            __m512 _tmp5 = _mm512_unpackhi_ps(_p4, _p5);
            __m512 _tmp6 = _mm512_unpacklo_ps(_p6, _p7);
            __m512 _tmp7 = _mm512_unpackhi_ps(_p6, _p7);

            __m512 _max01 = _mm512_max_ps(_tmp0, _tmp1);
            __m512 _max23 = _mm512_max_ps(_tmp2, _tmp3);
            __m512 _max45 = _mm512_max_ps(_tmp4, _tmp5);
            __m512 _max67 = _mm512_max_ps(_tmp6, _tmp7);

            _tmp0 = _mm512_unpacklo_ps(_max01, _max23);
            _tmp1 = _mm512_unpackhi_ps(_max01, _max23);
            _tmp2 = _mm512_unpacklo_ps(_max45, _max67);
            _tmp3 = _mm512_unpackhi_ps(_max45, _max67);

            __m512 _max0123 = _mm512_max_ps(_tmp0, _tmp1);
            __m512 _max4567 = _mm512_max_ps(_tmp2, _tmp3);

            _tmp0 = _mm512_shuffle_f32x4(_max0123, _max4567, _MM_SHUFFLE(2, 0, 2, 0));
            _tmp1 = _mm512_shuffle_f32x4(_max0123, _max4567, _MM_SHUFFLE(3, 1, 3, 1));

            __m512 _max1 = _mm512_max_ps(_tmp0, _tmp1);

            _max1 = _mm512_permutexvar_ps(_pidx, _max1);

            __m512 _max = _mm512_loadu_ps(maxptr);
            _max = _mm512_max_ps(_max, _max1);
            _mm512_storeu_ps(maxptr, _max);
            ptr += 128;
            maxptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 8 * 2);
            __m256 _p3 = _mm256_loadu_ps(ptr + 8 * 3);
            __m256 _p4 = _mm256_loadu_ps(ptr + 8 * 4);
            __m256 _p5 = _mm256_loadu_ps(ptr + 8 * 5);
            __m256 _p6 = _mm256_loadu_ps(ptr + 8 * 6);
            __m256 _p7 = _mm256_loadu_ps(ptr + 8 * 7);

            __m256 _tmp0 = _mm256_unpacklo_ps(_p0, _p4);
            __m256 _tmp1 = _mm256_unpackhi_ps(_p0, _p4);
            __m256 _tmp2 = _mm256_unpacklo_ps(_p2, _p6);
            __m256 _tmp3 = _mm256_unpackhi_ps(_p2, _p6);
            __m256 _tmp4 = _mm256_unpacklo_ps(_p1, _p5);
            __m256 _tmp5 = _mm256_unpackhi_ps(_p1, _p5);
            __m256 _tmp6 = _mm256_unpacklo_ps(_p3, _p7);
            __m256 _tmp7 = _mm256_unpackhi_ps(_p3, _p7);

            __m256 _max01 = _mm256_max_ps(_tmp0, _tmp1);
            __m256 _max23 = _mm256_max_ps(_tmp2, _tmp3);
            __m256 _max45 = _mm256_max_ps(_tmp4, _tmp5);
            __m256 _max67 = _mm256_max_ps(_tmp6, _tmp7);

            _tmp0 = _mm256_unpacklo_ps(_max01, _max23);
            _tmp1 = _mm256_unpackhi_ps(_max01, _max23);
            _tmp2 = _mm256_unpacklo_ps(_max45, _max67);
            _tmp3 = _mm256_unpackhi_ps(_max45, _max67);

            __m256 _max0123 = _mm256_max_ps(_tmp0, _tmp1);
            __m256 _max4567 = _mm256_max_ps(_tmp2, _tmp3);

            _tmp0 = _mm256_unpacklo_ps(_max0123, _max4567);
            _tmp1 = _mm256_unpackhi_ps(_max0123, _max4567);

            _max0123 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
            _max4567 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

            __m256 _max01234567 = _mm256_max_ps(_max0123, _max4567);

            __m256 _max = _mm256_loadu_ps(maxptr);
            _max = _mm256_max_ps(_max, _max01234567);
            _mm256_storeu_ps(maxptr, _max);
            ptr += 64;
            maxptr += 8;
        }
        for (; j < size1; j++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            *maxptr = std::max(*maxptr, _mm256_reduce_max_ps(_p));
            ptr += 8;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __AVX512F__
        __m512i _pidx = _mm512_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15);
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 16 * 2);
            __m512 _p3 = _mm512_loadu_ps(ptr + 16 * 3);
            __m512 _p4 = _mm512_loadu_ps(ptr + 16 * 4);
            __m512 _p5 = _mm512_loadu_ps(ptr + 16 * 5);
            __m512 _p6 = _mm512_loadu_ps(ptr + 16 * 6);
            __m512 _p7 = _mm512_loadu_ps(ptr + 16 * 7);

            __m512 _max0 = combine8x2_ps(_mm256_set1_ps(maxptr[0]), _mm256_set1_ps(maxptr[1]));
            __m512 _max1 = combine8x2_ps(_mm256_set1_ps(maxptr[2]), _mm256_set1_ps(maxptr[3]));
            __m512 _max2 = combine8x2_ps(_mm256_set1_ps(maxptr[4]), _mm256_set1_ps(maxptr[5]));
            __m512 _max3 = combine8x2_ps(_mm256_set1_ps(maxptr[6]), _mm256_set1_ps(maxptr[7]));
            __m512 _max4 = combine8x2_ps(_mm256_set1_ps(maxptr[8]), _mm256_set1_ps(maxptr[9]));
            __m512 _max5 = combine8x2_ps(_mm256_set1_ps(maxptr[10]), _mm256_set1_ps(maxptr[11]));
            __m512 _max6 = combine8x2_ps(_mm256_set1_ps(maxptr[12]), _mm256_set1_ps(maxptr[13]));
            __m512 _max7 = combine8x2_ps(_mm256_set1_ps(maxptr[14]), _mm256_set1_ps(maxptr[15]));

            _p0 = exp512_ps(_mm512_sub_ps(_p0, _max0));
            _p1 = exp512_ps(_mm512_sub_ps(_p1, _max1));
            _p2 = exp512_ps(_mm512_sub_ps(_p2, _max2));
            _p3 = exp512_ps(_mm512_sub_ps(_p3, _max3));
            _p4 = exp512_ps(_mm512_sub_ps(_p4, _max4));
            _p5 = exp512_ps(_mm512_sub_ps(_p5, _max5));
            _p6 = exp512_ps(_mm512_sub_ps(_p6, _max6));
            _p7 = exp512_ps(_mm512_sub_ps(_p7, _max7));
            _mm512_storeu_ps(ptr, _p0);
            _mm512_storeu_ps(ptr + 16, _p1);
            _mm512_storeu_ps(ptr + 16 * 2, _p2);
            _mm512_storeu_ps(ptr + 16 * 3, _p3);
            _mm512_storeu_ps(ptr + 16 * 4, _p4);
            _mm512_storeu_ps(ptr + 16 * 5, _p5);
            _mm512_storeu_ps(ptr + 16 * 6, _p6);
            _mm512_storeu_ps(ptr + 16 * 7, _p7);

            __m512 _tmp0 = _mm512_unpacklo_ps(_p0, _p1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_p0, _p1);
            __m512 _tmp2 = _mm512_unpacklo_ps(_p2, _p3);
            __m512 _tmp3 = _mm512_unpackhi_ps(_p2, _p3);
            __m512 _tmp4 = _mm512_unpacklo_ps(_p4, _p5);
            __m512 _tmp5 = _mm512_unpackhi_ps(_p4, _p5);
            __m512 _tmp6 = _mm512_unpacklo_ps(_p6, _p7);
            __m512 _tmp7 = _mm512_unpackhi_ps(_p6, _p7);

            __m512 _sum01 = _mm512_add_ps(_tmp0, _tmp1);
            __m512 _sum23 = _mm512_add_ps(_tmp2, _tmp3);
            __m512 _sum45 = _mm512_add_ps(_tmp4, _tmp5);
            __m512 _sum67 = _mm512_add_ps(_tmp6, _tmp7);

            _tmp0 = _mm512_unpacklo_ps(_sum01, _sum23);
            _tmp1 = _mm512_unpackhi_ps(_sum01, _sum23);
            _tmp2 = _mm512_unpacklo_ps(_sum45, _sum67);
            _tmp3 = _mm512_unpackhi_ps(_sum45, _sum67);

            __m512 _sum0123 = _mm512_add_ps(_tmp0, _tmp1);
            __m512 _sum4567 = _mm512_add_ps(_tmp2, _tmp3);

            _tmp0 = _mm512_shuffle_f32x4(_sum0123, _sum4567, _MM_SHUFFLE(2, 0, 2, 0));
            _tmp1 = _mm512_shuffle_f32x4(_sum0123, _sum4567, _MM_SHUFFLE(3, 1, 3, 1));

            __m512 _sum1 = _mm512_add_ps(_tmp0, _tmp1);

            _sum1 = _mm512_permutexvar_ps(_pidx, _sum1);

            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_add_ps(_sum, _sum1);
            _mm512_storeu_ps(sumptr, _sum);
            ptr += 128;
            maxptr += 16;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 8 * 2);
            __m256 _p3 = _mm256_loadu_ps(ptr + 8 * 3);
            __m256 _p4 = _mm256_loadu_ps(ptr + 8 * 4);
            __m256 _p5 = _mm256_loadu_ps(ptr + 8 * 5);
            __m256 _p6 = _mm256_loadu_ps(ptr + 8 * 6);
            __m256 _p7 = _mm256_loadu_ps(ptr + 8 * 7);
            _p0 = exp256_ps(_mm256_sub_ps(_p0, _mm256_set1_ps(maxptr[0])));
            _p1 = exp256_ps(_mm256_sub_ps(_p1, _mm256_set1_ps(maxptr[1])));
            _p2 = exp256_ps(_mm256_sub_ps(_p2, _mm256_set1_ps(maxptr[2])));
            _p3 = exp256_ps(_mm256_sub_ps(_p3, _mm256_set1_ps(maxptr[3])));
            _p4 = exp256_ps(_mm256_sub_ps(_p4, _mm256_set1_ps(maxptr[4])));
            _p5 = exp256_ps(_mm256_sub_ps(_p5, _mm256_set1_ps(maxptr[5])));
            _p6 = exp256_ps(_mm256_sub_ps(_p6, _mm256_set1_ps(maxptr[6])));
            _p7 = exp256_ps(_mm256_sub_ps(_p7, _mm256_set1_ps(maxptr[7])));
            _mm256_storeu_ps(ptr, _p0);
            _mm256_storeu_ps(ptr + 8, _p1);
            _mm256_storeu_ps(ptr + 8 * 2, _p2);
            _mm256_storeu_ps(ptr + 8 * 3, _p3);
            _mm256_storeu_ps(ptr + 8 * 4, _p4);
            _mm256_storeu_ps(ptr + 8 * 5, _p5);
            _mm256_storeu_ps(ptr + 8 * 6, _p6);
            _mm256_storeu_ps(ptr + 8 * 7, _p7);

            __m256 _tmp0 = _mm256_unpacklo_ps(_p0, _p4);
            __m256 _tmp1 = _mm256_unpackhi_ps(_p0, _p4);
            __m256 _tmp2 = _mm256_unpacklo_ps(_p2, _p6);
            __m256 _tmp3 = _mm256_unpackhi_ps(_p2, _p6);
            __m256 _tmp4 = _mm256_unpacklo_ps(_p1, _p5);
            __m256 _tmp5 = _mm256_unpackhi_ps(_p1, _p5);
            __m256 _tmp6 = _mm256_unpacklo_ps(_p3, _p7);
            __m256 _tmp7 = _mm256_unpackhi_ps(_p3, _p7);

            __m256 _sum01 = _mm256_add_ps(_tmp0, _tmp1);
            __m256 _sum23 = _mm256_add_ps(_tmp2, _tmp3);
            __m256 _sum45 = _mm256_add_ps(_tmp4, _tmp5);
            __m256 _sum67 = _mm256_add_ps(_tmp6, _tmp7);

            _tmp0 = _mm256_unpacklo_ps(_sum01, _sum23);
            _tmp1 = _mm256_unpackhi_ps(_sum01, _sum23);
            _tmp2 = _mm256_unpacklo_ps(_sum45, _sum67);
            _tmp3 = _mm256_unpackhi_ps(_sum45, _sum67);

            __m256 _sum0123 = _mm256_add_ps(_tmp0, _tmp1);
            __m256 _sum4567 = _mm256_add_ps(_tmp2, _tmp3);

            _tmp0 = _mm256_unpacklo_ps(_sum0123, _sum4567);
            _tmp1 = _mm256_unpackhi_ps(_sum0123, _sum4567);

            _sum0123 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
            _sum4567 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

            __m256 _sum01234567 = _mm256_add_ps(_sum0123, _sum4567);

            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_add_ps(_sum, _sum01234567);
            _mm256_storeu_ps(sumptr, _sum);
            ptr += 64;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j < size1; j++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _max = _mm256_set1_ps(*maxptr);
            _p = exp256_ps(_mm256_sub_ps(_p, _max));
            _mm256_storeu_ps(ptr, _p);
            *sumptr += _mm256_reduce_add_ps(_p);
            ptr += 8;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 16 * 2);
            __m512 _p3 = _mm512_loadu_ps(ptr + 16 * 3);
            __m512 _p4 = _mm512_loadu_ps(ptr + 16 * 4);
            __m512 _p5 = _mm512_loadu_ps(ptr + 16 * 5);
            __m512 _p6 = _mm512_loadu_ps(ptr + 16 * 6);
            __m512 _p7 = _mm512_loadu_ps(ptr + 16 * 7);
            __m512 _sum0 = combine8x2_ps(_mm256_set1_ps(sumptr[0]), _mm256_set1_ps(sumptr[1]));
            __m512 _sum1 = combine8x2_ps(_mm256_set1_ps(sumptr[2]), _mm256_set1_ps(sumptr[3]));
            __m512 _sum2 = combine8x2_ps(_mm256_set1_ps(sumptr[4]), _mm256_set1_ps(sumptr[5]));
            __m512 _sum3 = combine8x2_ps(_mm256_set1_ps(sumptr[6]), _mm256_set1_ps(sumptr[7]));
            __m512 _sum4 = combine8x2_ps(_mm256_set1_ps(sumptr[8]), _mm256_set1_ps(sumptr[9]));
            __m512 _sum5 = combine8x2_ps(_mm256_set1_ps(sumptr[10]), _mm256_set1_ps(sumptr[11]));
            __m512 _sum6 = combine8x2_ps(_mm256_set1_ps(sumptr[12]), _mm256_set1_ps(sumptr[13]));
            __m512 _sum7 = combine8x2_ps(_mm256_set1_ps(sumptr[14]), _mm256_set1_ps(sumptr[15]));
            _p0 = _mm512_mul_ps(_p0, _sum0);
            _p1 = _mm512_mul_ps(_p1, _sum1);
            _p2 = _mm512_mul_ps(_p2, _sum2);
            _p3 = _mm512_mul_ps(_p3, _sum3);
            _p4 = _mm512_mul_ps(_p4, _sum4);
            _p5 = _mm512_mul_ps(_p5, _sum5);
            _p6 = _mm512_mul_ps(_p6, _sum6);
            _p7 = _mm512_mul_ps(_p7, _sum7);
            _mm512_storeu_ps(ptr, _p0);
            _mm512_storeu_ps(ptr + 16, _p1);
            _mm512_storeu_ps(ptr + 16 * 2, _p2);
            _mm512_storeu_ps(ptr + 16 * 3, _p3);
            _mm512_storeu_ps(ptr + 16 * 4, _p4);
            _mm512_storeu_ps(ptr + 16 * 5, _p5);
            _mm512_storeu_ps(ptr + 16 * 6, _p6);
            _mm512_storeu_ps(ptr + 16 * 7, _p7);
            ptr += 128;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 3 < size1; j += 4)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 16);
            __m256 _p3 = _mm256_loadu_ps(ptr + 24);
            _p0 = _mm256_mul_ps(_p0, _mm256_set1_ps(sumptr[0]));
            _p1 = _mm256_mul_ps(_p1, _mm256_set1_ps(sumptr[1]));
            _p2 = _mm256_mul_ps(_p2, _mm256_set1_ps(sumptr[2]));
            _p3 = _mm256_mul_ps(_p3, _mm256_set1_ps(sumptr[3]));
            _mm256_storeu_ps(ptr, _p0);
            _mm256_storeu_ps(ptr + 8, _p1);
            _mm256_storeu_ps(ptr + 16, _p2);
            _mm256_storeu_ps(ptr + 24, _p3);
            ptr += 32;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _sum = _mm256_set1_ps(*sumptr);
            _p = _mm256_mul_ps(_p, _sum);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
            sumptr++;
        }
    }
}
#endif // __AVX__

static void softmax_pack4(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __AVX__
#if __AVX512F__
        __m512i _pidx = _mm512_setr_epi32(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 32);
            __m512 _p3 = _mm512_loadu_ps(ptr + 48);

            __m512 _tmp0 = _mm512_unpacklo_ps(_p0, _p1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_p0, _p1);
            __m512 _tmp2 = _mm512_unpacklo_ps(_p2, _p3);
            __m512 _tmp3 = _mm512_unpackhi_ps(_p2, _p3);

            __m512 _max01 = _mm512_max_ps(_tmp0, _tmp1);
            __m512 _max23 = _mm512_max_ps(_tmp2, _tmp3);

            _tmp0 = _mm512_unpacklo_ps(_max01, _max23);
            _tmp1 = _mm512_unpackhi_ps(_max01, _max23);

            __m512 _max0123 = _mm512_max_ps(_tmp0, _tmp1);

            _max0123 = _mm512_permutexvar_ps(_pidx, _max0123);

            __m512 _max = _mm512_loadu_ps(maxptr);
            _max = _mm512_max_ps(_max, _max0123);
            _mm512_storeu_ps(maxptr, _max);
            ptr += 64;
            maxptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 16);
            __m256 _p3 = _mm256_loadu_ps(ptr + 24);

            __m256 _tmp0 = _mm256_unpacklo_ps(_p0, _p1);
            __m256 _tmp1 = _mm256_unpackhi_ps(_p0, _p1);
            __m256 _tmp2 = _mm256_unpacklo_ps(_p2, _p3);
            __m256 _tmp3 = _mm256_unpackhi_ps(_p2, _p3);
            __m256 _max01 = _mm256_max_ps(_tmp0, _tmp1);
            __m256 _max23 = _mm256_max_ps(_tmp2, _tmp3);

            _tmp0 = _mm256_permute2f128_ps(_max01, _max23, _MM_SHUFFLE(0, 2, 0, 0));
            _tmp1 = _mm256_permute2f128_ps(_max01, _max23, _MM_SHUFFLE(0, 3, 0, 1));
            _max01 = _mm256_unpacklo_ps(_tmp0, _tmp1);
            _max23 = _mm256_unpackhi_ps(_tmp0, _tmp1);
            __m256 _max0123 = _mm256_max_ps(_max01, _max23);
            __m256 _max = _mm256_loadu_ps(maxptr);
            _max = _mm256_max_ps(_max, _max0123);
            _mm256_storeu_ps(maxptr, _max);
            ptr += 32;
            maxptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + 4);
            __m128 _p2 = _mm_loadu_ps(ptr + 8);
            __m128 _p3 = _mm_loadu_ps(ptr + 12);
            _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
            __m128 _max01 = _mm_max_ps(_p0, _p1);
            __m128 _max23 = _mm_max_ps(_p2, _p3);
            __m128 _max0123 = _mm_max_ps(_max01, _max23);
            __m128 _max = _mm_loadu_ps(maxptr);
            _max = _mm_max_ps(_max, _max0123);
            _mm_storeu_ps(maxptr, _max);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            *maxptr = std::max(*maxptr, _mm_reduce_max_ps(_p));
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __AVX__
#if __AVX512F__
        __m512i _pidx0 = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        __m512i _pidx1 = _mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
        __m512i _pidx2 = _mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11);
        __m512i _pidx3 = _mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15);
        __m512i _pidx = _mm512_setr_epi32(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 32);
            __m512 _p3 = _mm512_loadu_ps(ptr + 48);

            __m512 _max = _mm512_loadu_ps(maxptr);
            __m512 _max0 = _mm512_permutexvar_ps(_pidx0, _max);
            __m512 _max1 = _mm512_permutexvar_ps(_pidx1, _max);
            __m512 _max2 = _mm512_permutexvar_ps(_pidx2, _max);
            __m512 _max3 = _mm512_permutexvar_ps(_pidx3, _max);

            _p0 = exp512_ps(_mm512_sub_ps(_p0, _max0));
            _p1 = exp512_ps(_mm512_sub_ps(_p1, _max1));
            _p2 = exp512_ps(_mm512_sub_ps(_p2, _max2));
            _p3 = exp512_ps(_mm512_sub_ps(_p3, _max3));

            _mm512_storeu_ps(ptr, _p0);
            _mm512_storeu_ps(ptr + 16, _p1);
            _mm512_storeu_ps(ptr + 32, _p2);
            _mm512_storeu_ps(ptr + 48, _p3);

            __m512 _tmp0 = _mm512_unpacklo_ps(_p0, _p1);
            __m512 _tmp1 = _mm512_unpackhi_ps(_p0, _p1);
            __m512 _tmp2 = _mm512_unpacklo_ps(_p2, _p3);
            __m512 _tmp3 = _mm512_unpackhi_ps(_p2, _p3);

            __m512 _sum01 = _mm512_add_ps(_tmp0, _tmp1);
            __m512 _sum23 = _mm512_add_ps(_tmp2, _tmp3);

            _tmp0 = _mm512_unpacklo_ps(_sum01, _sum23);
            _tmp1 = _mm512_unpackhi_ps(_sum01, _sum23);

            __m512 _sum0123 = _mm512_add_ps(_tmp0, _tmp1);

            _sum0123 = _mm512_permutexvar_ps(_pidx, _sum0123);

            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_add_ps(_sum, _sum0123);
            _mm512_storeu_ps(sumptr, _sum);
            ptr += 64;
            maxptr += 16;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 16);
            __m256 _p3 = _mm256_loadu_ps(ptr + 24);
            _p0 = exp256_ps(_mm256_sub_ps(_p0, combine4x2_ps(_mm_set1_ps(maxptr[0]), _mm_set1_ps(maxptr[1]))));
            _p1 = exp256_ps(_mm256_sub_ps(_p1, combine4x2_ps(_mm_set1_ps(maxptr[2]), _mm_set1_ps(maxptr[3]))));
            _p2 = exp256_ps(_mm256_sub_ps(_p2, combine4x2_ps(_mm_set1_ps(maxptr[4]), _mm_set1_ps(maxptr[5]))));
            _p3 = exp256_ps(_mm256_sub_ps(_p3, combine4x2_ps(_mm_set1_ps(maxptr[6]), _mm_set1_ps(maxptr[7]))));
            _mm256_storeu_ps(ptr, _p0);
            _mm256_storeu_ps(ptr + 8, _p1);
            _mm256_storeu_ps(ptr + 16, _p2);
            _mm256_storeu_ps(ptr + 24, _p3);
            __m256 _tmp0 = _mm256_unpacklo_ps(_p0, _p1);
            __m256 _tmp1 = _mm256_unpackhi_ps(_p0, _p1);
            __m256 _tmp2 = _mm256_unpacklo_ps(_p2, _p3);
            __m256 _tmp3 = _mm256_unpackhi_ps(_p2, _p3);
            __m256 _sum01 = _mm256_add_ps(_tmp0, _tmp1);
            __m256 _sum23 = _mm256_add_ps(_tmp2, _tmp3);
            _tmp0 = _mm256_permute2f128_ps(_sum01, _sum23, _MM_SHUFFLE(0, 2, 0, 0));
            _tmp1 = _mm256_permute2f128_ps(_sum01, _sum23, _MM_SHUFFLE(0, 3, 0, 1));
            _sum01 = _mm256_unpacklo_ps(_tmp0, _tmp1);
            _sum23 = _mm256_unpackhi_ps(_tmp0, _tmp1);
            __m256 _sum0123 = _mm256_add_ps(_sum01, _sum23);
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_add_ps(_sum, _sum0123);
            _mm256_storeu_ps(sumptr, _sum);
            ptr += 32;
            maxptr += 8;
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + 4);
            __m128 _p2 = _mm_loadu_ps(ptr + 8);
            __m128 _p3 = _mm_loadu_ps(ptr + 12);
            _p0 = exp_ps(_mm_sub_ps(_p0, _mm_set1_ps(maxptr[0])));
            _p1 = exp_ps(_mm_sub_ps(_p1, _mm_set1_ps(maxptr[1])));
            _p2 = exp_ps(_mm_sub_ps(_p2, _mm_set1_ps(maxptr[2])));
            _p3 = exp_ps(_mm_sub_ps(_p3, _mm_set1_ps(maxptr[3])));
            _mm_storeu_ps(ptr, _p0);
            _mm_storeu_ps(ptr + 4, _p1);
            _mm_storeu_ps(ptr + 8, _p2);
            _mm_storeu_ps(ptr + 12, _p3);
            _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
            __m128 _sum01 = _mm_add_ps(_p0, _p1);
            __m128 _sum23 = _mm_add_ps(_p2, _p3);
            __m128 _sum0123 = _mm_add_ps(_sum01, _sum23);
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_add_ps(_sum, _sum0123);
            _mm_storeu_ps(sumptr, _sum);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _max = _mm_set1_ps(*maxptr);
            _p = exp_ps(_mm_sub_ps(_p, _max));
            _mm_storeu_ps(ptr, _p);
            *sumptr += _mm_reduce_add_ps(_p);
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __AVX__
#if __AVX512F__
        __m512i _pidx0 = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        __m512i _pidx1 = _mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
        __m512i _pidx2 = _mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11);
        __m512i _pidx3 = _mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15);
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p0 = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr + 16);
            __m512 _p2 = _mm512_loadu_ps(ptr + 32);
            __m512 _p3 = _mm512_loadu_ps(ptr + 48);

            __m512 _sum = _mm512_loadu_ps(sumptr);
            __m512 _sum0 = _mm512_permutexvar_ps(_pidx0, _sum);
            __m512 _sum1 = _mm512_permutexvar_ps(_pidx1, _sum);
            __m512 _sum2 = _mm512_permutexvar_ps(_pidx2, _sum);
            __m512 _sum3 = _mm512_permutexvar_ps(_pidx3, _sum);

            _p0 = _mm512_mul_ps(_p0, _sum0);
            _p1 = _mm512_mul_ps(_p1, _sum1);
            _p2 = _mm512_mul_ps(_p2, _sum2);
            _p3 = _mm512_mul_ps(_p3, _sum3);
            _mm512_storeu_ps(ptr, _p0);
            _mm512_storeu_ps(ptr + 16, _p1);
            _mm512_storeu_ps(ptr + 32, _p2);
            _mm512_storeu_ps(ptr + 48, _p3);
            ptr += 64;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr + 8);
            __m256 _p2 = _mm256_loadu_ps(ptr + 16);
            __m256 _p3 = _mm256_loadu_ps(ptr + 24);
            _p0 = _mm256_mul_ps(_p0, combine4x2_ps(_mm_set1_ps(sumptr[0]), _mm_set1_ps(sumptr[1])));
            _p1 = _mm256_mul_ps(_p1, combine4x2_ps(_mm_set1_ps(sumptr[2]), _mm_set1_ps(sumptr[3])));
            _p2 = _mm256_mul_ps(_p2, combine4x2_ps(_mm_set1_ps(sumptr[4]), _mm_set1_ps(sumptr[5])));
            _p3 = _mm256_mul_ps(_p3, combine4x2_ps(_mm_set1_ps(sumptr[6]), _mm_set1_ps(sumptr[7])));
            _mm256_storeu_ps(ptr, _p0);
            _mm256_storeu_ps(ptr + 8, _p1);
            _mm256_storeu_ps(ptr + 16, _p2);
            _mm256_storeu_ps(ptr + 24, _p3);
            ptr += 32;
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p0 = _mm_loadu_ps(ptr);
            __m128 _p1 = _mm_loadu_ps(ptr + 4);
            __m128 _p2 = _mm_loadu_ps(ptr + 8);
            __m128 _p3 = _mm_loadu_ps(ptr + 12);
            _p0 = _mm_mul_ps(_p0, _mm_set1_ps(sumptr[0]));
            _p1 = _mm_mul_ps(_p1, _mm_set1_ps(sumptr[1]));
            _p2 = _mm_mul_ps(_p2, _mm_set1_ps(sumptr[2]));
            _p3 = _mm_mul_ps(_p3, _mm_set1_ps(sumptr[3]));
            _mm_storeu_ps(ptr, _p0);
            _mm_storeu_ps(ptr + 4, _p1);
            _mm_storeu_ps(ptr + 8, _p2);
            _mm_storeu_ps(ptr + 12, _p3);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _sum = _mm_set1_ps(*sumptr);
            _p = _mm_mul_ps(_p, _sum);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __SSE2__

static void softmax_pack1(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _max = _mm512_loadu_ps(maxptr);
            _max = _mm512_max_ps(_max, _p);
            _mm512_storeu_ps(maxptr, _max);
            ptr += 16;
            maxptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _max = _mm256_loadu_ps(maxptr);
            _max = _mm256_max_ps(_max, _p);
            _mm256_storeu_ps(maxptr, _max);
            ptr += 8;
            maxptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _max = _mm_loadu_ps(maxptr);
            _max = _mm_max_ps(_max, _p);
            _mm_storeu_ps(maxptr, _max);
            ptr += 4;
            maxptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, *ptr);
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _max = _mm512_loadu_ps(maxptr);
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _p = _mm512_sub_ps(_p, _max);
            _p = exp512_ps(_p);
            _mm512_storeu_ps(ptr, _p);
            _sum = _mm512_add_ps(_sum, _p);
            _mm512_storeu_ps(sumptr, _sum);
            ptr += 16;
            maxptr += 16;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _max = _mm256_loadu_ps(maxptr);
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _p = _mm256_sub_ps(_p, _max);
            _p = exp256_ps(_p);
            _mm256_storeu_ps(ptr, _p);
            _sum = _mm256_add_ps(_sum, _p);
            _mm256_storeu_ps(sumptr, _sum);
            ptr += 8;
            maxptr += 8;
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _max = _mm_loadu_ps(maxptr);
            __m128 _sum = _mm_loadu_ps(sumptr);
            _p = _mm_sub_ps(_p, _max);
            _p = exp_ps(_p);
            _mm_storeu_ps(ptr, _p);
            _sum = _mm_add_ps(_sum, _p);
            _mm_storeu_ps(sumptr, _sum);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            float v = expf(*ptr - *maxptr);
            *ptr = v;
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _sum = _mm512_rcp_nr_ps(_sum);
            _mm512_storeu_ps(sumptr, _sum);
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _sum = _mm256_rcp_nr_ps(_sum);
            _mm256_storeu_ps(sumptr, _sum);
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = _mm_loadu_ps(sumptr);
            _sum = _mm_rcp_nr_ps(_sum);
            _mm_storeu_ps(sumptr, _sum);
            sumptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size1; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _sum = _mm512_loadu_ps(sumptr);
            _p = _mm512_mul_ps(_p, _sum);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
            sumptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _sum = _mm256_loadu_ps(sumptr);
            _p = _mm256_mul_ps(_p, _sum);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
            sumptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _sum = _mm_loadu_ps(sumptr);
            _p = _mm_mul_ps(_p, _sum);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
            sumptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *ptr *= *sumptr;
            ptr++;
            sumptr++;
        }
    }
}

static void softmax(float* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _negmax_avx512 = _mm512_set1_ps(-FLT_MAX);
        for (; j + 15 < size1; j += 16)
        {
            _mm512_storeu_ps(maxptr, _negmax_avx512);
            maxptr += 16;
        }
#endif // __AVX512F__
        __m256 _negmax_avx = _mm256_set1_ps(-FLT_MAX);
        for (; j + 7 < size1; j += 8)
        {
            _mm256_storeu_ps(maxptr, _negmax_avx);
            maxptr += 8;
        }
#endif // __AVX__
        __m128 _negmax = _mm_set1_ps(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            _mm_storeu_ps(maxptr, _negmax);
            maxptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // reduce exp(x - max)
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _zero_avx512 = _mm512_set1_ps(0.f);
        for (; j + 15 < size1; j += 16)
        {
            _mm512_storeu_ps(sumptr, _zero_avx512);
            sumptr += 16;
        }
#endif // __AVX512F__
        __m256 _zero_avx = _mm256_set1_ps(0.f);
        for (; j + 7 < size1; j += 8)
        {
            _mm256_storeu_ps(sumptr, _zero_avx);
            sumptr += 8;
        }
#endif // __AVX__
        __m128 _zero = _mm_set1_ps(0.f);
        for (; j + 3 < size1; j += 4)
        {
            _mm_storeu_ps(sumptr, _zero);
            sumptr += 4;
        }
#endif // __SSE2__
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        softmax_pack16(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        softmax_pack8(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __AVX__
    if (elempack == 4)
    {
        softmax_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
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
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = (size_t)w * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax(ptr, h, elempack, stride, size1, maxptr, sumptr);
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
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax(ptr, h, 1, size, size, maxptr, sumptr);
            }
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
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax(ptr, d, 1, size, size, maxptr, sumptr);
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
