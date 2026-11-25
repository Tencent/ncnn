// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
int softmax_bf16s_zvfbfmin(Mat& bottom_top_blob, int axis, const Option& opt);
#endif

#if __riscv_vector
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline)) static vfloat32m4_t
softmax_bf16s_reset_tails(vfloat32m4_t x, size_t vl, float v)
{
    const size_t vlm4 = __riscv_vsetvlmax_e32m4();
    vbool8_t _vl_mask = __riscv_vmsgeu_vx_u32m4_b8(__riscv_vid_v_u32m4(vlm4), vl, vlm4);
    x = __riscv_vfmerge_vfm_f32m4(x, v, _vl_mask, vlm4);
    return x;
}
#endif // __riscv_xtheadvector
#endif // __riscv_vector

static void softmax_bf16s(unsigned short* _ptr, int elemcount, int elempack)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    const int size = elemcount * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;

    // full vl chunks keep every chunk aligned with the packn lanes of _max and _sum
    const size_t vl = __riscv_vsetvlmax_e16m2();

    // reduce max
    vfloat32m4_t _max = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vl);
    {
        const unsigned short* ptr = _ptr;

        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _max = __riscv_vfmax_vv_f32m4(_max, _p, vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m2(remain);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlr), vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = softmax_bf16s_reset_tails(_p, vlr, -FLT_MAX);
            _max = __riscv_vfmax_vv_f32m4(_max, _p, vl);
#else  // __riscv_xtheadvector
            _max = __riscv_vfmax_vv_f32m4_tu(_max, _max, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    if (elempack == packn)
    {
        // reduce max nn,nn to nn
        // broadcast nn to nn,nn

        vfloat32m2_t _max0 = __riscv_vfmax_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_max, 0), __riscv_vget_v_f32m4_f32m2(_max, 1), __riscv_vsetvlmax_e32m2());
        _max = __riscv_vcreate_v_f32m2_f32m4(_max0, _max0);
    }
    if (elempack == 1)
    {
        // reduce max n,n,n,n to 1
        // broadcast 1 to n,n,n,n

        vfloat32m1_t _max0 = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
        _max0 = __riscv_vfredmax_vs_f32m4_f32m1(_max, _max0, vl);
        _max = __riscv_vfmv_v_f_f32m4(__riscv_vfmv_f_s_f32m1_f32(_max0), vl);
    }

    // reduce exp(x - max) and store back to bf16
    vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vl);
    {
        unsigned short* ptr = _ptr;

        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vl), vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
            _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m2(remain);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlr), vlr);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vlr), vlr);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vlr), vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = softmax_bf16s_reset_tails(_p, vlr, 0.f);
            _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vl);
#else  // __riscv_xtheadvector
            _sum = __riscv_vfadd_vv_f32m4_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    if (elempack == packn)
    {
        // reduce sum nn,nn to nn
        // broadcast nn to nn,nn

        vfloat32m2_t _sum0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sum, 0), __riscv_vget_v_f32m4_f32m2(_sum, 1), __riscv_vsetvlmax_e32m2());
        _sum = __riscv_vcreate_v_f32m2_f32m4(_sum0, _sum0);
    }
    if (elempack == 1)
    {
        // reduce sum n,n,n,n to 1
        // broadcast 1 to n,n,n,n

        vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m4_f32m1(_sum, _sum0, vl);
        _sum = __riscv_vfmv_v_f_f32m4(__riscv_vfmv_f_s_f32m1_f32(_sum0), vl);
    }

    _sum = __riscv_vfrdiv_vf_f32m4(_sum, 1.f, vl);

    // div sum
    {
        unsigned short* ptr = _ptr;

        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = __riscv_vfmul_vv_f32m4(_p, _sum, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m2(remain);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlr), vlr);
            _p = __riscv_vfmul_vv_f32m4(_p, _sum, vlr);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vlr), vlr);
        }
    }
#else  // __riscv_vector
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr++));
        }
    }

    // reduce exp(x - max) and store back to bf16
    float sum = 0.f;
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max);
            *ptr = float32_to_bfloat16(v);
            sum += v;
            ptr++;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * sum);
            ptr++;
        }
    }
#endif // __riscv_vector

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

#if __riscv_vector
static void softmax_bf16s_packn(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
#if !__riscv_xtheadvector
    // xtheadvector mixing vl in fp16 intrinsics leads to incorrect result
    // ref https://github.com/XUANTIE-RV/xuantie-gnu-toolchain/issues/28
    const size_t vlm4 = __riscv_vsetvlmax_e16m4();
    const size_t vlm2 = __riscv_vsetvlmax_e16m2();
    const size_t vlm1 = __riscv_vsetvlmax_e16m1();
#endif // !__riscv_xtheadvector

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if !__riscv_xtheadvector
        for (; j + 3 < size1; j += 4)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vlm4), vlm4);
            vfloat32m1_t _m0 = __riscv_vfmv_s_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_s_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _m2 = __riscv_vfmv_s_f_f32m1(maxptr[2], vlm1);
            vfloat32m1_t _m3 = __riscv_vfmv_s_f_f32m1(maxptr[3], vlm1);
            vfloat32m1_t _max0 = __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vget_v_f32m8_f32m2(_p, 0), _m0, vlm1);
            vfloat32m1_t _max1 = __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vget_v_f32m8_f32m2(_p, 1), _m1, vlm1);
            vfloat32m1_t _max2 = __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vget_v_f32m8_f32m2(_p, 2), _m2, vlm1);
            vfloat32m1_t _max3 = __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vget_v_f32m8_f32m2(_p, 3), _m3, vlm1);
            maxptr[0] = __riscv_vfmv_f_s_f32m1_f32(_max0);
            maxptr[1] = __riscv_vfmv_f_s_f32m1_f32(_max1);
            maxptr[2] = __riscv_vfmv_f_s_f32m1_f32(_max2);
            maxptr[3] = __riscv_vfmv_f_s_f32m1_f32(_max3);
            ptr += vlm4;
            maxptr += 4;
        }
        for (; j + 1 < size1; j += 2)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlm2), vlm2);
            vfloat32m1_t _m0 = __riscv_vfmv_s_f_f32m1(maxptr[0], vlm1);
            vfloat32m1_t _m1 = __riscv_vfmv_s_f_f32m1(maxptr[1], vlm1);
            vfloat32m1_t _max0 = __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vget_v_f32m4_f32m2(_p, 0), _m0, vlm1);
            vfloat32m1_t _max1 = __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vget_v_f32m4_f32m2(_p, 1), _m1, vlm1);
            maxptr[0] = __riscv_vfmv_f_s_f32m1_f32(_max0);
            maxptr[1] = __riscv_vfmv_f_s_f32m1_f32(_max1);
            ptr += vlm2;
            maxptr += 2;
        }
#endif // !__riscv_xtheadvector
        for (; j < size1; j++)
        {
#if __riscv_xtheadvector
            size_t vlm1 = __riscv_vsetvlmax_e16m1();
#endif // __riscv_xtheadvector
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vlm1), vlm1);
            vfloat32m1_t _max = __riscv_vfredmax_vs_f32m2_f32m1(_p, __riscv_vfmv_s_f_f32m1(*maxptr, vlm1), vlm1);
            *maxptr = __riscv_vfmv_f_s_f32m1_f32(_max);
            ptr += vlm1;
            maxptr++;
        }
    }

    // reduce exp(x - max) and store back to bf16
    // at most two positions per step, exp_ps on f32m8 runs out of vector registers
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if !__riscv_xtheadvector
        for (; j + 1 < size1; j += 2)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlm2), vlm2);
            vfloat32m2_t _m0 = __riscv_vfmv_v_f_f32m2(maxptr[0], vlm1);
            vfloat32m2_t _m1 = __riscv_vfmv_v_f_f32m2(maxptr[1], vlm1);
            vfloat32m4_t _max = __riscv_vcreate_v_f32m2_f32m4(_m0, _m1);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vlm2), vlm2);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vlm2), vlm2);
            vfloat32m1_t _s0 = __riscv_vfmv_s_f_f32m1(sumptr[0], vlm1);
            vfloat32m1_t _s1 = __riscv_vfmv_s_f_f32m1(sumptr[1], vlm1);
            vfloat32m1_t _sum0 = __riscv_vfredusum_vs_f32m2_f32m1(__riscv_vget_v_f32m4_f32m2(_p, 0), _s0, vlm1);
            vfloat32m1_t _sum1 = __riscv_vfredusum_vs_f32m2_f32m1(__riscv_vget_v_f32m4_f32m2(_p, 1), _s1, vlm1);
            sumptr[0] = __riscv_vfmv_f_s_f32m1_f32(_sum0);
            sumptr[1] = __riscv_vfmv_f_s_f32m1_f32(_sum1);
            ptr += vlm2;
            maxptr += 2;
            sumptr += 2;
        }
#endif // !__riscv_xtheadvector
        for (; j < size1; j++)
        {
#if __riscv_xtheadvector
            size_t vlm1 = __riscv_vsetvlmax_e16m1();
#endif // __riscv_xtheadvector
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vlm1), vlm1);
            _p = exp_ps(__riscv_vfsub_vf_f32m2(_p, *maxptr, vlm1), vlm1);
            __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_p, vlm1), vlm1);
            vfloat32m1_t _sum = __riscv_vfredusum_vs_f32m2_f32m1(_p, __riscv_vfmv_s_f_f32m1(*sumptr, vlm1), vlm1);
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
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if !__riscv_xtheadvector
        for (; j + 3 < size1; j += 4)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vlm4), vlm4);
            vfloat32m2_t _s0 = __riscv_vfmv_v_f_f32m2(sumptr[0], vlm1);
            vfloat32m2_t _s1 = __riscv_vfmv_v_f_f32m2(sumptr[1], vlm1);
            vfloat32m2_t _s2 = __riscv_vfmv_v_f_f32m2(sumptr[2], vlm1);
            vfloat32m2_t _s3 = __riscv_vfmv_v_f_f32m2(sumptr[3], vlm1);
            vfloat32m8_t _sum = __riscv_vcreate_v_f32m2_f32m8(_s0, _s1, _s2, _s3);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vlm4);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vlm4), vlm4);
            ptr += vlm4;
            sumptr += 4;
        }
        for (; j + 1 < size1; j += 2)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlm2), vlm2);
            vfloat32m2_t _s0 = __riscv_vfmv_v_f_f32m2(sumptr[0], vlm1);
            vfloat32m2_t _s1 = __riscv_vfmv_v_f_f32m2(sumptr[1], vlm1);
            vfloat32m4_t _sum = __riscv_vcreate_v_f32m2_f32m4(_s0, _s1);
            _p = __riscv_vfmul_vv_f32m4(_p, _sum, vlm2);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vlm2), vlm2);
            ptr += vlm2;
            sumptr += 2;
        }
#endif // !__riscv_xtheadvector
        for (; j < size1; j++)
        {
#if __riscv_xtheadvector
            size_t vlm1 = __riscv_vsetvlmax_e16m1();
#endif // __riscv_xtheadvector
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vlm1), vlm1);
            _p = __riscv_vfmul_vf_f32m2(_p, *sumptr, vlm1);
            __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_p, vlm1), vlm1);
            ptr += vlm1;
            sumptr++;
        }
    }
}
#endif // __riscv_vector

static void softmax_bf16s_pack1(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
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
            *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
            ptr++;
            maxptr++;
        }
#endif // __riscv_vector
    }

    // reduce exp(x - max) and store back to bf16
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _max = __riscv_vle32_v_f32m4(maxptr, vl);
            vfloat32m4_t _sum = __riscv_vle32_v_f32m4(sumptr, vl);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vl), vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
            _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vl);
            __riscv_vse32_v_f32m4(sumptr, _sum, vl);
            n -= vl;
            ptr += vl;
            maxptr += vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - *maxptr);
            *ptr = float32_to_bfloat16(v);
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
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

#if __riscv_vector
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vl);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vl), vl);
            n -= vl;
            ptr += vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * *sumptr);
            ptr++;
            sumptr++;
        }
#endif // __riscv_vector
    }
}

static void softmax_bf16s(unsigned short* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    // init max / sum
    {
        float* maxptr = _maxptr;
        float* sumptr = _sumptr;

#if __riscv_vector
        vfloat32m8_t _negmax = __riscv_vfmv_v_f_f32m8(-FLT_MAX, __riscv_vsetvlmax_e32m8());
        vfloat32m8_t _zero = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            __riscv_vse32_v_f32m8(maxptr, _negmax, vl);
            __riscv_vse32_v_f32m8(sumptr, _zero, vl);
            n -= vl;
            maxptr += vl;
            sumptr += vl;
        }
#else  // __riscv_vector
        for (int j = 0; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
            *sumptr++ = 0.f;
        }
#endif // __riscv_vector
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;

    if (elempack == packn)
    {
        softmax_bf16s_packn(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __riscv_vector
    if (elempack == 1)
    {
        softmax_bf16s_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

static int softmax_bf16s(Mat& bottom_top_blob, int axis, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        return softmax_bf16s_zvfbfmin(bottom_top_blob, axis, opt);
    }
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
        unsigned short* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_bf16s(ptr, size, 1);
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

            softmax_bf16s(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            softmax_bf16s(ptr, w, elempack);
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

            softmax_bf16s(ptr, channels, elempack, stride, size1, maxptr, sumptr);
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

                softmax_bf16s(ptr, h, 1, size, size, maxptr, sumptr);
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
                softmax_bf16s(ptr, w, elempack);
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

            softmax_bf16s(ptr, d, 1, size, size, maxptr, sumptr);
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
                    softmax_bf16s(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
