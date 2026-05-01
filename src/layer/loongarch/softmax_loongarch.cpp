// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_loongarch.h"

#include <float.h>
#include <math.h>

#include "cpu.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#include "loongarch_usability.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

#if NCNN_BF16
#include "softmax_bf16s.h"
#endif

Softmax_loongarch::Softmax_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __loongarch_sx
#if __loongarch_asx
    __m256 _max_lasx = (__m256)__lasx_xvreplfr2vr_s(-FLT_MAX);
#endif // __loongarch_asx
    __m128 _max = (__m128)__lsx_vreplfr2vr_s(-FLT_MAX);
#endif // __loongarch_sx
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _max_lasx = __lasx_xvfmax_s(_max_lasx, _p);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _max = __lsx_vfmax_s(_max, _p);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

#if __loongarch_sx
    if (elempack == 4)
    {
#if __loongarch_asx
        {
            __m128 _max0 = __lasx_extract_128_lo_s(_max_lasx);
            __m128 _max1 = __lasx_extract_128_hi_s(_max_lasx);
            _max = __lsx_vfmax_s(_max, _max0);
            _max = __lsx_vfmax_s(_max, _max1);
        }

        _max_lasx = __lasx_concat_128_s(_max, _max);
#endif // __loongarch_asx
    }
    if (elempack == 1)
    {
#if __loongarch_asx
        {
            __m128 _max0 = __lasx_extract_128_lo_s(_max_lasx);
            __m128 _max1 = __lasx_extract_128_hi_s(_max_lasx);
            _max = __lsx_vfmax_s(_max, _max0);
            _max = __lsx_vfmax_s(_max, _max1);
        }
#endif // __loongarch_asx
        max = std::max(max, __lsx_reduce_fmax_s(_max));

        _max = (__m128)__lsx_vreplfr2vr_s(max);
#if __loongarch_asx
        _max_lasx = __lasx_concat_128_s(_max, _max);
#endif // __loongarch_asx
    }
#endif // __loongarch_sx

    // reduce exp(x - max)
#if __loongarch_sx
#if __loongarch_asx
    __m256 _sum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
    __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // __loongarch_sx
    float sum = 0.f;
    {
        float* ptr = _ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfsub_s(_p, _max_lasx);
            _p = exp256_ps(_p);
            __lasx_xvst((__m256i)_p, ptr, 0);
            _sum_lasx = __lasx_xvfadd_s(_sum_lasx, _p);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfsub_s(_p, _max);
            _p = exp_ps(_p);
            __lsx_vst(_p, ptr, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
        _sum_lasx = __lasx_xvfdiv_s(_one, _sum_lasx);
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
#if __loongarch_asx
        {
            __m128 _sum0 = __lasx_extract_128_lo_s(_sum_lasx);
            __m128 _sum1 = __lasx_extract_128_hi_s(_sum_lasx);
            _sum = __lsx_vfadd_s(_sum, _sum0);
            _sum = __lsx_vfadd_s(_sum, _sum1);
        }
#endif // __loongarch_asx

        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        _sum = __lsx_vfdiv_s(_one, _sum);

#if __loongarch_asx
        _sum_lasx = __lasx_concat_128_s(_sum, _sum);
#endif // __loongarch_asx
    }
#endif // __loongarch_sx
    if (elempack == 1)
    {
#if __loongarch_sx
#if __loongarch_asx
        {
            __m128 _sum0 = __lasx_extract_128_lo_s(_sum_lasx);
            __m128 _sum1 = __lasx_extract_128_hi_s(_sum_lasx);
            _sum = __lsx_vfadd_s(_sum, _sum0);
            _sum = __lsx_vfadd_s(_sum, _sum1);
        }
#endif // __loongarch_asx
        sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx

        sum = 1.f / sum;

#if __loongarch_sx
        _sum = (__m128)__lsx_vreplfr2vr_s(sum);
#if __loongarch_asx
        _sum_lasx = __lasx_concat_128_s(_sum, _sum);
#endif // __loongarch_asx
#endif // __loongarch_sx
    }

    // div sum
    {
        float* ptr = _ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmul_s(_p, _sum_lasx);
            __lasx_xvst((__m256i)_p, ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmul_s(_p, _sum);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

#if __loongarch_sx
#if __loongarch_asx
static void softmax_pack8(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = (__m256)__lasx_xvld(ptr, 0);
            __m256 _p1 = (__m256)__lasx_xvld(ptr + 8, 0);
            __m256 _p2 = (__m256)__lasx_xvld(ptr + 16, 0);
            __m256 _p3 = (__m256)__lasx_xvld(ptr + 24, 0);
            __m256 _p4 = (__m256)__lasx_xvld(ptr + 32, 0);
            __m256 _p5 = (__m256)__lasx_xvld(ptr + 40, 0);
            __m256 _p6 = (__m256)__lasx_xvld(ptr + 48, 0);
            __m256 _p7 = (__m256)__lasx_xvld(ptr + 56, 0);
            transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
            __m256 _max01 = __lasx_xvfmax_s(_p0, _p1);
            __m256 _max23 = __lasx_xvfmax_s(_p2, _p3);
            __m256 _max45 = __lasx_xvfmax_s(_p4, _p5);
            __m256 _max67 = __lasx_xvfmax_s(_p6, _p7);
            __m256 _max0123 = __lasx_xvfmax_s(_max01, _max23);
            __m256 _max4567 = __lasx_xvfmax_s(_max45, _max67);
            __m256 _max01234567 = __lasx_xvfmax_s(_max0123, _max4567);
            __m256 _max = (__m256)__lasx_xvld(maxptr, 0);
            _max = __lasx_xvfmax_s(_max, _max01234567);
            __lasx_xvst((__m256i)_max, maxptr, 0);
            ptr += 64;
            maxptr += 8;
        }
        for (; j < size1; j++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            *maxptr = std::max(*maxptr, __lasx_reduce_fmax_s(_p));
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
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = (__m256)__lasx_xvld(ptr, 0);
            __m256 _p1 = (__m256)__lasx_xvld(ptr + 8, 0);
            __m256 _p2 = (__m256)__lasx_xvld(ptr + 16, 0);
            __m256 _p3 = (__m256)__lasx_xvld(ptr + 24, 0);
            __m256 _p4 = (__m256)__lasx_xvld(ptr + 32, 0);
            __m256 _p5 = (__m256)__lasx_xvld(ptr + 40, 0);
            __m256 _p6 = (__m256)__lasx_xvld(ptr + 48, 0);
            __m256 _p7 = (__m256)__lasx_xvld(ptr + 56, 0);
            _p0 = exp256_ps(__lasx_xvfsub_s(_p0, (__m256)__lasx_xvreplfr2vr_s(maxptr[0])));
            _p1 = exp256_ps(__lasx_xvfsub_s(_p1, (__m256)__lasx_xvreplfr2vr_s(maxptr[1])));
            _p2 = exp256_ps(__lasx_xvfsub_s(_p2, (__m256)__lasx_xvreplfr2vr_s(maxptr[2])));
            _p3 = exp256_ps(__lasx_xvfsub_s(_p3, (__m256)__lasx_xvreplfr2vr_s(maxptr[3])));
            _p4 = exp256_ps(__lasx_xvfsub_s(_p4, (__m256)__lasx_xvreplfr2vr_s(maxptr[4])));
            _p5 = exp256_ps(__lasx_xvfsub_s(_p5, (__m256)__lasx_xvreplfr2vr_s(maxptr[5])));
            _p6 = exp256_ps(__lasx_xvfsub_s(_p6, (__m256)__lasx_xvreplfr2vr_s(maxptr[6])));
            _p7 = exp256_ps(__lasx_xvfsub_s(_p7, (__m256)__lasx_xvreplfr2vr_s(maxptr[7])));
            __lasx_xvst((__m256i)_p0, ptr, 0);
            __lasx_xvst((__m256i)_p1, ptr + 8, 0);
            __lasx_xvst((__m256i)_p2, ptr + 16, 0);
            __lasx_xvst((__m256i)_p3, ptr + 24, 0);
            __lasx_xvst((__m256i)_p4, ptr + 32, 0);
            __lasx_xvst((__m256i)_p5, ptr + 40, 0);
            __lasx_xvst((__m256i)_p6, ptr + 48, 0);
            __lasx_xvst((__m256i)_p7, ptr + 56, 0);
            transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
            __m256 _sum01 = __lasx_xvfadd_s(_p0, _p1);
            __m256 _sum23 = __lasx_xvfadd_s(_p2, _p3);
            __m256 _sum45 = __lasx_xvfadd_s(_p4, _p5);
            __m256 _sum67 = __lasx_xvfadd_s(_p6, _p7);
            __m256 _sum0123 = __lasx_xvfadd_s(_sum01, _sum23);
            __m256 _sum4567 = __lasx_xvfadd_s(_sum45, _sum67);
            __m256 _sum01234567 = __lasx_xvfadd_s(_sum0123, _sum4567);
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            _sum = __lasx_xvfadd_s(_sum, _sum01234567);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            ptr += 64;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j < size1; j++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _max = (__m256)__lasx_xvreplfr2vr_s(*maxptr);
            _p = exp256_ps(__lasx_xvfsub_s(_p, _max));
            __lasx_xvst((__m256i)_p, ptr, 0);
            *sumptr += __lasx_reduce_fadd_s(_p);
            ptr += 8;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            _sum = __lasx_xvfdiv_s(_one, _sum);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            _sum = __lsx_vfdiv_s(_one, _sum);
            __lsx_vst(_sum, sumptr, 0);
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
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p0 = (__m256)__lasx_xvld(ptr, 0);
            __m256 _p1 = (__m256)__lasx_xvld(ptr + 8, 0);
            __m256 _p2 = (__m256)__lasx_xvld(ptr + 16, 0);
            __m256 _p3 = (__m256)__lasx_xvld(ptr + 24, 0);
            __m256 _p4 = (__m256)__lasx_xvld(ptr + 32, 0);
            __m256 _p5 = (__m256)__lasx_xvld(ptr + 40, 0);
            __m256 _p6 = (__m256)__lasx_xvld(ptr + 48, 0);
            __m256 _p7 = (__m256)__lasx_xvld(ptr + 56, 0);
            _p0 = __lasx_xvfmul_s(_p0, (__m256)__lasx_xvreplfr2vr_s(sumptr[0]));
            _p1 = __lasx_xvfmul_s(_p1, (__m256)__lasx_xvreplfr2vr_s(sumptr[1]));
            _p2 = __lasx_xvfmul_s(_p2, (__m256)__lasx_xvreplfr2vr_s(sumptr[2]));
            _p3 = __lasx_xvfmul_s(_p3, (__m256)__lasx_xvreplfr2vr_s(sumptr[3]));
            _p4 = __lasx_xvfmul_s(_p4, (__m256)__lasx_xvreplfr2vr_s(sumptr[4]));
            _p5 = __lasx_xvfmul_s(_p5, (__m256)__lasx_xvreplfr2vr_s(sumptr[5]));
            _p6 = __lasx_xvfmul_s(_p6, (__m256)__lasx_xvreplfr2vr_s(sumptr[6]));
            _p7 = __lasx_xvfmul_s(_p7, (__m256)__lasx_xvreplfr2vr_s(sumptr[7]));
            __lasx_xvst((__m256i)_p0, ptr, 0);
            __lasx_xvst((__m256i)_p1, ptr + 8, 0);
            __lasx_xvst((__m256i)_p2, ptr + 16, 0);
            __lasx_xvst((__m256i)_p3, ptr + 24, 0);
            __lasx_xvst((__m256i)_p4, ptr + 32, 0);
            __lasx_xvst((__m256i)_p5, ptr + 40, 0);
            __lasx_xvst((__m256i)_p6, ptr + 48, 0);
            __lasx_xvst((__m256i)_p7, ptr + 56, 0);
            ptr += 64;
            sumptr += 8;
        }
        for (; j < size1; j++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _sum = (__m256)__lasx_xvreplfr2vr_s(*sumptr);
            _p = __lasx_xvfmul_s(_p, _sum);
            __lasx_xvst((__m256i)_p, ptr, 0);
            ptr += 8;
            sumptr++;
        }
    }
}
#endif // __loongarch_asx

static void softmax_pack4(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p0 = (__m128)__lsx_vld(ptr, 0);
            __m128 _p1 = (__m128)__lsx_vld(ptr + 4, 0);
            __m128 _p2 = (__m128)__lsx_vld(ptr + 8, 0);
            __m128 _p3 = (__m128)__lsx_vld(ptr + 12, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            __m128 _max01 = __lsx_vfmax_s(_p0, _p1);
            __m128 _max23 = __lsx_vfmax_s(_p2, _p3);
            __m128 _max0123 = __lsx_vfmax_s(_max01, _max23);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);
            _max = __lsx_vfmax_s(_max, _max0123);
            __lsx_vst(_max, maxptr, 0);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            *maxptr = std::max(*maxptr, __lsx_reduce_fmax_s(_p));
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
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p0 = (__m128)__lsx_vld(ptr, 0);
            __m128 _p1 = (__m128)__lsx_vld(ptr + 4, 0);
            __m128 _p2 = (__m128)__lsx_vld(ptr + 8, 0);
            __m128 _p3 = (__m128)__lsx_vld(ptr + 12, 0);
            _p0 = exp_ps(__lsx_vfsub_s(_p0, (__m128)__lsx_vreplfr2vr_s(maxptr[0])));
            _p1 = exp_ps(__lsx_vfsub_s(_p1, (__m128)__lsx_vreplfr2vr_s(maxptr[1])));
            _p2 = exp_ps(__lsx_vfsub_s(_p2, (__m128)__lsx_vreplfr2vr_s(maxptr[2])));
            _p3 = exp_ps(__lsx_vfsub_s(_p3, (__m128)__lsx_vreplfr2vr_s(maxptr[3])));
            __lsx_vst(_p0, ptr, 0);
            __lsx_vst(_p1, ptr + 4, 0);
            __lsx_vst(_p2, ptr + 8, 0);
            __lsx_vst(_p3, ptr + 12, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            __m128 _sum01 = __lsx_vfadd_s(_p0, _p1);
            __m128 _sum23 = __lsx_vfadd_s(_p2, _p3);
            __m128 _sum0123 = __lsx_vfadd_s(_sum01, _sum23);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _sum = __lsx_vfadd_s(_sum, _sum0123);
            __lsx_vst(_sum, sumptr, 0);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _max = (__m128)__lsx_vreplfr2vr_s(*maxptr);
            _p = exp_ps(__lsx_vfsub_s(_p, _max));
            __lsx_vst(_p, ptr, 0);
            *sumptr += __lsx_reduce_fadd_s(_p);
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            _sum = __lasx_xvfdiv_s(_one, _sum);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            _sum = __lsx_vfdiv_s(_one, _sum);
            __lsx_vst(_sum, sumptr, 0);
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
            __m128 _p0 = (__m128)__lsx_vld(ptr, 0);
            __m128 _p1 = (__m128)__lsx_vld(ptr + 4, 0);
            __m128 _p2 = (__m128)__lsx_vld(ptr + 8, 0);
            __m128 _p3 = (__m128)__lsx_vld(ptr + 12, 0);
            _p0 = __lsx_vfmul_s(_p0, (__m128)__lsx_vreplfr2vr_s(sumptr[0]));
            _p1 = __lsx_vfmul_s(_p1, (__m128)__lsx_vreplfr2vr_s(sumptr[1]));
            _p2 = __lsx_vfmul_s(_p2, (__m128)__lsx_vreplfr2vr_s(sumptr[2]));
            _p3 = __lsx_vfmul_s(_p3, (__m128)__lsx_vreplfr2vr_s(sumptr[3]));
            __lsx_vst(_p0, ptr, 0);
            __lsx_vst(_p1, ptr + 4, 0);
            __lsx_vst(_p2, ptr + 8, 0);
            __lsx_vst(_p3, ptr + 12, 0);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _sum = (__m128)__lsx_vreplfr2vr_s(*sumptr);
            _p = __lsx_vfmul_s(_p, _sum);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __loongarch_sx

static void softmax_pack1(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _max = (__m256)__lasx_xvld(maxptr, 0);
            _max = __lasx_xvfmax_s(_max, _p);
            __lasx_xvst((__m256i)_max, maxptr, 0);
            ptr += 8;
            maxptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);
            _max = __lsx_vfmax_s(_max, _p);
            __lsx_vst(_max, maxptr, 0);
            ptr += 4;
            maxptr += 4;
        }
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _max = (__m256)__lasx_xvld(maxptr, 0);
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            _p = __lasx_xvfsub_s(_p, _max);
            _p = exp256_ps(_p);
            __lasx_xvst((__m256i)_p, ptr, 0);
            _sum = __lasx_xvfadd_s(_sum, _p);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            ptr += 8;
            maxptr += 8;
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _p = __lsx_vfsub_s(_p, _max);
            _p = exp_ps(_p);
            __lsx_vst(_p, ptr, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            __lsx_vst(_sum, sumptr, 0);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            _sum = __lasx_xvfdiv_s(_one, _sum);
            __lasx_xvst((__m256i)_sum, sumptr, 0);
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            _sum = __lsx_vfdiv_s(_one, _sum);
            __lsx_vst(_sum, sumptr, 0);
            sumptr += 4;
        }
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        for (; j + 7 < size1; j += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _sum = (__m256)__lasx_xvld(sumptr, 0);
            _p = __lasx_xvfmul_s(_p, _sum);
            __lasx_xvst((__m256i)_p, ptr, 0);
            ptr += 8;
            sumptr += 8;
        }
#endif // __loongarch_asx
        for (; j + 3 < size1; j += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _p = __lsx_vfmul_s(_p, _sum);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx
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
    // init max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _negmax_lasx = (__m256)__lasx_xvreplfr2vr_s(-FLT_MAX);
        for (; j + 7 < size1; j += 8)
        {
            __lasx_xvst((__m256i)_negmax_lasx, maxptr, 0);
            maxptr += 8;
        }
#endif // __loongarch_asx
        __m128 _negmax = (__m128)__lsx_vreplfr2vr_s(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            __lsx_vst(_negmax, maxptr, 0);
            maxptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // init sum
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256i _zero_lasx = __lasx_xvldi(0);
        for (; j + 7 < size1; j += 8)
        {
            __lasx_xvst(_zero_lasx, sumptr, 0);
            sumptr += 8;
        }
#endif // __loongarch_asx
        __m128i _zero = __lsx_vldi(0);
        for (; j + 3 < size1; j += 4)
        {
            __lsx_vst(_zero, sumptr, 0);
            sumptr += 4;
        }
#endif // __loongarch_sx
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        softmax_pack8(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
        softmax_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __loongarch_sx
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}

int Softmax_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

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

#if NCNN_BF16
int Softmax_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        unsigned short* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_bf16s_lsx(ptr, size, 1);
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

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

            softmax_bf16s_lsx_dispatch(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            softmax_bf16s_lsx(ptr, w, elempack);
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

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

            softmax_bf16s_lsx_dispatch(ptr, channels, elempack, stride, size1, maxptr, sumptr);
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
                unsigned short* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax_bf16s_lsx_dispatch(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax_bf16s_lsx(ptr, w, elempack);
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
            unsigned short* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax_bf16s_lsx_dispatch(ptr, d, 1, size, size, maxptr, sumptr);
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_bf16s_lsx(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
