// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_riscv.h"
#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Softmax_riscv::Softmax_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline))
#endif // __riscv_xtheadvector
static vfloat32m8_t
reset_tails(vfloat32m8_t x, size_t vl, float v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    vbool4_t _vl_mask = __riscv_vmsgeu_vx_u32m8_b4(__riscv_vid_v_u32m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f32m8(x, v, _vl_mask, vlm8);
    return x;
}
#endif // __riscv_vector

static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // NCNN_LOGE("softmax %d %d  %d", elemcount, elempack, size);

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    // reduce max
    vfloat32m8_t _max = __riscv_vfmv_v_f_f32m8(-FLT_MAX, __riscv_vsetvlmax_e32m8());
    {
        const float* ptr = _ptr;

        int n = size / __riscv_vsetvlmax_e32m8() * __riscv_vsetvlmax_e32m8();
        const size_t vl = __riscv_vsetvlmax_e32m8();
        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % __riscv_vsetvlmax_e32m8();
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = reset_tails(_p, vlr, -FLT_MAX);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
#else  // __riscv_xtheadvector
            _max = __riscv_vfmax_vv_f32m8_tu(_max, _max, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    if (elempack == packn)
    {
        // reduce max n,n,n,n,n,n,n,n to n
        // broadcast n to n,n,n,n,n,n,n,n

        vfloat32m4_t _max0 = __riscv_vfmax_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_max, 0), __riscv_vget_v_f32m8_f32m4(_max, 1), __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _max2 = __riscv_vfmax_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_max0, 0), __riscv_vget_v_f32m4_f32m2(_max0, 1), __riscv_vsetvlmax_e32m2());
        vfloat32m1_t _max4 = __riscv_vfmax_vv_f32m1(__riscv_vget_v_f32m2_f32m1(_max2, 0), __riscv_vget_v_f32m2_f32m1(_max2, 1), __riscv_vsetvlmax_e32m1());
        _max = __riscv_vcreate_v_f32m1_f32m8(_max4, _max4, _max4, _max4, _max4, _max4, _max4, _max4);
    }
    if (elempack == 1)
    {
        // reduce max n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _max0 = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
        _max0 = __riscv_vfredmax_vs_f32m8_f32m1(_max, _max0, __riscv_vsetvlmax_e32m8());
        _max = __riscv_vset_v_f32m1_f32m8(_max, 0, _max0);
        _max = __riscv_vrgather_vx_f32m8(_max, 0, __riscv_vsetvlmax_e32m8());
    }

    // reduce exp(x - max)
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
    {
        float* ptr = _ptr;

        int n = size / __riscv_vsetvlmax_e32m8() * __riscv_vsetvlmax_e32m8();
        const size_t vl = __riscv_vsetvlmax_e32m8();
        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfsub_vv_f32m8(_p, _max, vl);
            _p = exp_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % __riscv_vsetvlmax_e32m8();
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vlr);
            _p = __riscv_vfsub_vv_f32m8(_p, _max, vlr);
            _p = exp_ps(_p, vlr);
            __riscv_vse32_v_f32m8(ptr, _p, vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = reset_tails(_p, vlr, 0.f);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
#else  // __riscv_xtheadvector
            _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    if (elempack == packn)
    {
        // reduce sum n,n,n,n,n,n,n,n to n
        // broadcast n to n,n,n,n,n,n,n,n

        vfloat32m4_t _sum0 = __riscv_vfadd_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_sum, 0), __riscv_vget_v_f32m8_f32m4(_sum, 1), __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _sum2 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sum0, 0), __riscv_vget_v_f32m4_f32m2(_sum0, 1), __riscv_vsetvlmax_e32m2());
        vfloat32m1_t _sum4 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(_sum2, 0), __riscv_vget_v_f32m2_f32m1(_sum2, 1), __riscv_vsetvlmax_e32m1());
        _sum = __riscv_vcreate_v_f32m1_f32m8(_sum4, _sum4, _sum4, _sum4, _sum4, _sum4, _sum4, _sum4);
    }
    if (elempack == 1)
    {
        // reduce sum n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, __riscv_vsetvlmax_e32m8());
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 0, _sum0);
        _sum = __riscv_vrgather_vx_f32m8(_sum, 0, __riscv_vsetvlmax_e32m8());
    }

    _sum = __riscv_vfrdiv_vf_f32m8(_sum, 1.f, __riscv_vsetvlmax_e32m8());

    // div sum
    {
        float* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            n -= vl;
            ptr += vl;
        }
    }
#else  // __riscv_vector
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
#endif // __riscv_vector
}

#if __riscv_vector
static void softmax_packn(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    const size_t vlm4 = __riscv_vsetvlmax_e32m4();
    const size_t vlm2 = __riscv_vsetvlmax_e32m2();
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vlm8);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_v_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _m2 = __riscv_vfmv_v_f_f32m1(maxptr[2], vlm1);
            vfloat32m1_t _m3 = __riscv_vfmv_v_f_f32m1(maxptr[3], vlm1);
            vfloat32m1_t _m4 = __riscv_vfmv_v_f_f32m1(maxptr[4], vlm1);
            vfloat32m1_t _m5 = __riscv_vfmv_v_f_f32m1(maxptr[5], vlm1);
            vfloat32m1_t _m6 = __riscv_vfmv_v_f_f32m1(maxptr[6], vlm1);
            vfloat32m1_t _m7 = __riscv_vfmv_v_f_f32m1(maxptr[7], vlm1);
            vfloat32m1_t _max0 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 0), _m0, vlm1);
            vfloat32m1_t _max1 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 1), _m1, vlm1);
            vfloat32m1_t _max2 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 2), _m2, vlm1);
            vfloat32m1_t _max3 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 3), _m3, vlm1);
            vfloat32m1_t _max4 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 4), _m4, vlm1);
            vfloat32m1_t _max5 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 5), _m5, vlm1);
            vfloat32m1_t _max6 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 6), _m6, vlm1);
            vfloat32m1_t _max7 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 7), _m7, vlm1);
            maxptr[0] = __riscv_vfmv_f_s_f32m1_f32(_max0);
            maxptr[1] = __riscv_vfmv_f_s_f32m1_f32(_max1);
            maxptr[2] = __riscv_vfmv_f_s_f32m1_f32(_max2);
            maxptr[3] = __riscv_vfmv_f_s_f32m1_f32(_max3);
            maxptr[4] = __riscv_vfmv_f_s_f32m1_f32(_max4);
            maxptr[5] = __riscv_vfmv_f_s_f32m1_f32(_max5);
            maxptr[6] = __riscv_vfmv_f_s_f32m1_f32(_max6);
            maxptr[7] = __riscv_vfmv_f_s_f32m1_f32(_max7);
            ptr += vlm8;
            maxptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vlm4);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_v_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _m2 = __riscv_vfmv_v_f_f32m1(maxptr[2], vlm1);
            vfloat32m1_t _m3 = __riscv_vfmv_v_f_f32m1(maxptr[3], vlm1);
            vfloat32m1_t _max0 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 0), _m0, vlm1);
            vfloat32m1_t _max1 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 1), _m1, vlm1);
            vfloat32m1_t _max2 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 2), _m2, vlm1);
            vfloat32m1_t _max3 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 3), _m3, vlm1);
            maxptr[0] = __riscv_vfmv_f_s_f32m1_f32(_max0);
            maxptr[1] = __riscv_vfmv_f_s_f32m1_f32(_max1);
            maxptr[2] = __riscv_vfmv_f_s_f32m1_f32(_max2);
            maxptr[3] = __riscv_vfmv_f_s_f32m1_f32(_max3);
            ptr += vlm4;
            maxptr += 4;
        }
        for (; j + 1 < size1; j += 2)
        {
            vfloat32m2_t _p = __riscv_vle32_v_f32m2(ptr, vlm2);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_v_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _max0 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m2_f32m1(_p, 0), _m0, vlm1);
            vfloat32m1_t _max1 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m2_f32m1(_p, 1), _m1, vlm1);
            maxptr[0] = __riscv_vfmv_f_s_f32m1_f32(_max0);
            maxptr[1] = __riscv_vfmv_f_s_f32m1_f32(_max1);
            ptr += vlm2;
            maxptr += 2;
        }
        for (; j < size1; j++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vlm1);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(*maxptr, vlm1);
            _p = __riscv_vfredmax_vs_f32m1_f32m1(_p, _m0, vlm1);
            *maxptr = __riscv_vfmv_f_s_f32m1_f32(_p);
            ptr += vlm1;
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
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vlm8);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_v_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _m2 = __riscv_vfmv_v_f_f32m1(maxptr[2], vlm1);
            vfloat32m1_t _m3 = __riscv_vfmv_v_f_f32m1(maxptr[3], vlm1);
            vfloat32m1_t _m4 = __riscv_vfmv_v_f_f32m1(maxptr[4], vlm1);
            vfloat32m1_t _m5 = __riscv_vfmv_v_f_f32m1(maxptr[5], vlm1);
            vfloat32m1_t _m6 = __riscv_vfmv_v_f_f32m1(maxptr[6], vlm1);
            vfloat32m1_t _m7 = __riscv_vfmv_v_f_f32m1(maxptr[7], vlm1);
            vfloat32m8_t _max = __riscv_vcreate_v_f32m1_f32m8(_m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7);
            _p = exp_ps(__riscv_vfsub_vv_f32m8(_p, _max, vlm8), vlm8);
            __riscv_vse32_v_f32m8(ptr, _p, vlm8);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_v_f_f32m1(sumptr[1], vlm1);
            vfloat32m1_t _s2 = __riscv_vfmv_v_f_f32m1(sumptr[2], vlm1);
            vfloat32m1_t _s3 = __riscv_vfmv_v_f_f32m1(sumptr[3], vlm1);
            vfloat32m1_t _s4 = __riscv_vfmv_v_f_f32m1(sumptr[4], vlm1);
            vfloat32m1_t _s5 = __riscv_vfmv_v_f_f32m1(sumptr[5], vlm1);
            vfloat32m1_t _s6 = __riscv_vfmv_v_f_f32m1(sumptr[6], vlm1);
            vfloat32m1_t _s7 = __riscv_vfmv_v_f_f32m1(sumptr[7], vlm1);
            vfloat32m1_t _sum0 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 0), _s0, vlm1);
            vfloat32m1_t _sum1 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 1), _s1, vlm1);
            vfloat32m1_t _sum2 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 2), _s2, vlm1);
            vfloat32m1_t _sum3 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 3), _s3, vlm1);
            vfloat32m1_t _sum4 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 4), _s4, vlm1);
            vfloat32m1_t _sum5 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 5), _s5, vlm1);
            vfloat32m1_t _sum6 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 6), _s6, vlm1);
            vfloat32m1_t _sum7 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_p, 7), _s7, vlm1);
            sumptr[0] = __riscv_vfmv_f_s_f32m1_f32(_sum0);
            sumptr[1] = __riscv_vfmv_f_s_f32m1_f32(_sum1);
            sumptr[2] = __riscv_vfmv_f_s_f32m1_f32(_sum2);
            sumptr[3] = __riscv_vfmv_f_s_f32m1_f32(_sum3);
            sumptr[4] = __riscv_vfmv_f_s_f32m1_f32(_sum4);
            sumptr[5] = __riscv_vfmv_f_s_f32m1_f32(_sum5);
            sumptr[6] = __riscv_vfmv_f_s_f32m1_f32(_sum6);
            sumptr[7] = __riscv_vfmv_f_s_f32m1_f32(_sum7);

            ptr += vlm8;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vlm4);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_v_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _m2 = __riscv_vfmv_v_f_f32m1(maxptr[2], vlm1);
            vfloat32m1_t _m3 = __riscv_vfmv_v_f_f32m1(maxptr[3], vlm1);
            vfloat32m4_t _max = __riscv_vcreate_v_f32m1_f32m4(_m0, _m1, _m2, _m3);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vlm4), vlm4);
            __riscv_vse32_v_f32m4(ptr, _p, vlm4);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_v_f_f32m1(sumptr[1], vlm1);
            vfloat32m1_t _s2 = __riscv_vfmv_v_f_f32m1(sumptr[2], vlm1);
            vfloat32m1_t _s3 = __riscv_vfmv_v_f_f32m1(sumptr[3], vlm1);
            vfloat32m1_t _sum0 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 0), _s0, vlm1);
            vfloat32m1_t _sum1 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 1), _s1, vlm1);
            vfloat32m1_t _sum2 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 2), _s2, vlm1);
            vfloat32m1_t _sum3 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m4_f32m1(_p, 3), _s3, vlm1);
            sumptr[0] = __riscv_vfmv_f_s_f32m1_f32(_sum0);
            sumptr[1] = __riscv_vfmv_f_s_f32m1_f32(_sum1);
            sumptr[2] = __riscv_vfmv_f_s_f32m1_f32(_sum2);
            sumptr[3] = __riscv_vfmv_f_s_f32m1_f32(_sum3);
            ptr += vlm4;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j + 1 < size1; j += 2)
        {
            vfloat32m2_t _p = __riscv_vle32_v_f32m2(ptr, vlm2);
            vfloat32m1_t _m0 = __riscv_vfmv_v_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_v_f_f32m1(maxptr[1], vlm1);
            vfloat32m2_t _max = __riscv_vcreate_v_f32m1_f32m2(_m0, _m1);
            _p = exp_ps(__riscv_vfsub_vv_f32m2(_p, _max, vlm2), vlm2);
            __riscv_vse32_v_f32m2(ptr, _p, vlm2);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_v_f_f32m1(sumptr[1], vlm1);
            vfloat32m1_t _sum0 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m2_f32m1(_p, 0), _s0, vlm1);
            vfloat32m1_t _sum1 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m2_f32m1(_p, 1), _s1, vlm1);
            sumptr[0] = __riscv_vfmv_f_s_f32m1_f32(_sum0);
            sumptr[1] = __riscv_vfmv_f_s_f32m1_f32(_sum1);
            ptr += vlm2;
            maxptr += 2;
            sumptr += 2;
        }
        for (; j < size1; j++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vlm1);
            _p = exp_ps(__riscv_vfsub_vf_f32m1(_p, *maxptr, vlm1), vlm1);
            __riscv_vse32_v_f32m1(ptr, _p, vlm1);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(*sumptr, vlm1);
            vfloat32m1_t _sum = __riscv_vfredusum_vs_f32m1_f32m1(_p, _s0, vlm1);
            *sumptr = __riscv_vfmv_f_s_f32m1_f32(_sum);
            ptr += vlm1;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _sum = __riscv_vfrdiv_vf_f32m8(_sum, 1.f, vl);
            __riscv_vse32_v_f32m8(sumptr, _sum, vl);
            n -= vl;
            sumptr += vl;
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
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vlm8);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_v_f_f32m1(sumptr[1], vlm1);
            vfloat32m1_t _s2 = __riscv_vfmv_v_f_f32m1(sumptr[2], vlm1);
            vfloat32m1_t _s3 = __riscv_vfmv_v_f_f32m1(sumptr[3], vlm1);
            vfloat32m1_t _s4 = __riscv_vfmv_v_f_f32m1(sumptr[4], vlm1);
            vfloat32m1_t _s5 = __riscv_vfmv_v_f_f32m1(sumptr[5], vlm1);
            vfloat32m1_t _s6 = __riscv_vfmv_v_f_f32m1(sumptr[6], vlm1);
            vfloat32m1_t _s7 = __riscv_vfmv_v_f_f32m1(sumptr[7], vlm1);
            vfloat32m8_t _sum = __riscv_vcreate_v_f32m1_f32m8(_s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vlm8);
            __riscv_vse32_v_f32m8(ptr, _p, vlm8);
            ptr += vlm8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vlm4);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_v_f_f32m1(sumptr[1], vlm1);
            vfloat32m1_t _s2 = __riscv_vfmv_v_f_f32m1(sumptr[2], vlm1);
            vfloat32m1_t _s3 = __riscv_vfmv_v_f_f32m1(sumptr[3], vlm1);
            vfloat32m4_t _sum = __riscv_vcreate_v_f32m1_f32m4(_s0, _s1, _s2, _s3);
            _p = __riscv_vfmul_vv_f32m4(_p, _sum, vlm4);
            __riscv_vse32_v_f32m4(ptr, _p, vlm4);
            ptr += vlm4;
            sumptr += 4;
        }
        for (; j + 1 < size1; j += 2)
        {
            vfloat32m2_t _p = __riscv_vle32_v_f32m2(ptr, vlm2);
            vfloat32m1_t _s0 = __riscv_vfmv_v_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_v_f_f32m1(sumptr[1], vlm1);
            vfloat32m2_t _sum = __riscv_vcreate_v_f32m1_f32m2(_s0, _s1);
            _p = __riscv_vfmul_vv_f32m2(_p, _sum, vlm2);
            __riscv_vse32_v_f32m2(ptr, _p, vlm2);
            ptr += vlm2;
            sumptr += 2;
        }
        for (; j < size1; j++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vlm1);
            _p = __riscv_vfmul_vf_f32m1(_p, *sumptr, vlm1);
            __riscv_vse32_v_f32m1(ptr, _p, vlm1);
            ptr += vlm1;
            sumptr++;
        }
    }
}
#endif // __riscv_vector

static void softmax_pack1(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _max = __riscv_vle32_v_f32m8(maxptr, vl);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
            __riscv_vse32_v_f32m8(maxptr, _max, vl);
            n -= vl;
            ptr += vl;
            maxptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, *ptr);
            ptr++;
            maxptr++;
        }
#endif // __riscv_vector
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _max = __riscv_vle32_v_f32m8(maxptr, vl);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _p = __riscv_vfsub_vv_f32m8(_p, _max, vl);
            _p = exp_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            __riscv_vse32_v_f32m8(sumptr, _sum, vl);
            n -= vl;
            ptr += vl;
            maxptr += vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            float v = expf(*ptr - *maxptr);
            *ptr = v;
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
#endif // __riscv_vector
    }

    {
        float* sumptr = _sumptr;
#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _sum = __riscv_vfrdiv_vf_f32m8(_sum, 1.f, vl);
            __riscv_vse32_v_f32m8(sumptr, _sum, vl);
            n -= vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
#endif // __riscv_vector
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            n -= vl;
            ptr += vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *ptr *= *sumptr;
            ptr++;
            sumptr++;
        }
#endif // __riscv_vector
    }
}

static void softmax(float* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    {
        float* maxptr = _maxptr;

#if __riscv_vector
        vfloat32m8_t _negmax = __riscv_vfmv_v_f_f32m8(-FLT_MAX, __riscv_vsetvlmax_e32m8());
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            __riscv_vse32_v_f32m8(maxptr, _negmax, vl);
            n -= vl;
            maxptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
#endif // __riscv_vector
    }

    // reduce exp(x - max)
    {
        float* sumptr = _sumptr;

#if __riscv_vector
        vfloat32m8_t _zero = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            __riscv_vse32_v_f32m8(sumptr, _zero, vl);
            n -= vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
#endif // __riscv_vector
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    if (elempack == packn)
    {
        softmax_packn(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __riscv_vector
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}

int Softmax_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
