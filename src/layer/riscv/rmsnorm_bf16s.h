// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void rmsnorm_bf16s_zvfbfmin(Mat& bottom_top_blob, const float* gamma_data, float eps, int affine_size, const Option& opt);
#endif

#if __riscv_vector
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline)) static vfloat32m8_t
rmsnorm_bf16s_reset_tails(vfloat32m8_t x, size_t vl, float v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    vbool4_t _vl_mask = __riscv_vmsgeu_vx_u32m8_b4(__riscv_vid_v_u32m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f32m8(x, v, _vl_mask, vlm8);
    return x;
}
#endif // __riscv_xtheadvector
#endif // __riscv_vector

static void rmsnorm_bf16s(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    const int size = elemcount * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;

    // reduce sqsum
    vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
    {
        const unsigned short* ptr0 = ptr;

        const size_t vl = __riscv_vsetvlmax_e16m4();
        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr0, vl), vl);
            _sqsum = __riscv_vfmacc_vv_f32m8(_sqsum, _p, _p, vl);
            ptr0 += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m4(remain);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr0, vlr), vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = rmsnorm_bf16s_reset_tails(_p, vlr, 0.f);
            _sqsum = __riscv_vfmacc_vv_f32m8(_sqsum, _p, _p, vl);
#else  // __riscv_xtheadvector
            _sqsum = __riscv_vfmacc_vv_f32m8_tu(_sqsum, _p, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    // a = 1.f / sqrtf(sqsum / elemcount + eps);
    vfloat32m8_t _a;
    if (elempack == packn)
    {
        // reduce sqsum nn,nn,nn,nn to nn
        // broadcast nn to nn,nn,nn,nn

        const size_t vlm2 = __riscv_vsetvlmax_e32m2();

        vfloat32m4_t _sqsum0 = __riscv_vfadd_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_sqsum, 0), __riscv_vget_v_f32m8_f32m4(_sqsum, 1), __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _sqsum2 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sqsum0, 0), __riscv_vget_v_f32m4_f32m2(_sqsum0, 1), vlm2);
        vfloat32m2_t _rms2 = __riscv_vfadd_vf_f32m2(__riscv_vfdiv_vf_f32m2(_sqsum2, elemcount, vlm2), eps, vlm2);
        vfloat32m2_t _a2 = __riscv_vfrdiv_vf_f32m2(__riscv_vfsqrt_v_f32m2(_rms2, vlm2), 1.f, vlm2);
        _a = __riscv_vcreate_v_f32m2_f32m8(_a2, _a2, _a2, _a2);
    }
    else // if (elempack == 1)
    {
        // reduce sqsum n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _sqsum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, __riscv_vsetvlmax_e32m8());
        const float sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum0);
        const float a = 1.f / sqrtf(sqsum / elemcount + eps);
        _a = __riscv_vfmv_v_f_f32m8(a, __riscv_vsetvlmax_e32m8());
    }

    if (gamma_ptr && elempack == packn)
    {
#if !__riscv_xtheadvector
        // xtheadvector mixing vl in fp16 intrinsics leads to incorrect result
        // ref https://github.com/XUANTIE-RV/xuantie-gnu-toolchain/issues/28
        const size_t vlm4 = __riscv_vsetvlmax_e16m4();
        const size_t vlm2 = __riscv_vsetvlmax_e16m2();
        const size_t vlm1 = __riscv_vsetvlmax_e16m1();
#endif // !__riscv_xtheadvector

        int i = 0;
#if !__riscv_xtheadvector
        for (; i + 3 < elemcount; i += 4)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vlm4), vlm4);
            _p = __riscv_vfmul_vv_f32m8(_p, _a, vlm4);
            vfloat32m2_t _p0 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(_p, 0), gamma_ptr[0], vlm1);
            vfloat32m2_t _p1 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(_p, 1), gamma_ptr[1], vlm1);
            vfloat32m2_t _p2 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(_p, 2), gamma_ptr[2], vlm1);
            vfloat32m2_t _p3 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(_p, 3), gamma_ptr[3], vlm1);
            _p = __riscv_vcreate_v_f32m2_f32m8(_p0, _p1, _p2, _p3);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vlm4), vlm4);
            ptr += vlm4;
            gamma_ptr += 4;
        }
        for (; i + 1 < elemcount; i += 2)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vlm2), vlm2);
            _p = __riscv_vfmul_vv_f32m4(_p, __riscv_vget_v_f32m8_f32m4(_a, 0), vlm2);
            vfloat32m2_t _p0 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m4_f32m2(_p, 0), gamma_ptr[0], vlm1);
            vfloat32m2_t _p1 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m4_f32m2(_p, 1), gamma_ptr[1], vlm1);
            _p = __riscv_vcreate_v_f32m2_f32m4(_p0, _p1);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vlm2), vlm2);
            ptr += vlm2;
            gamma_ptr += 2;
        }
#endif // !__riscv_xtheadvector
        for (; i < elemcount; i++)
        {
#if __riscv_xtheadvector
            size_t vlm1 = __riscv_vsetvlmax_e16m1();
#endif // __riscv_xtheadvector
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vlm1), vlm1);
            _p = __riscv_vfmul_vv_f32m2(_p, __riscv_vget_v_f32m8_f32m2(_a, 0), vlm1);
            _p = __riscv_vfmul_vf_f32m2(_p, gamma_ptr[0], vlm1);
            __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_p, vlm1), vlm1);
            ptr += vlm1;
            gamma_ptr += 1;
        }
    }
    else if (gamma_ptr) // if (elempack == 1)
    {
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(gamma_ptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _a, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _gamma, vl);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vl), vl);
            n -= vl;
            ptr += vl;
            gamma_ptr += vl;
        }
    }
    else
    {
        // full vl chunks keep every chunk aligned with the packn lanes of _a
        const size_t vl = __riscv_vsetvlmax_e16m4();
        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _a, vl);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vl), vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m4(remain);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vlr), vlr);
            _p = __riscv_vfmul_vv_f32m8(_p, _a, vlr);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vlr), vlr);
        }
    }
#else  // __riscv_vector
    float sqsum = 0.f;
    for (int i = 0; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        sqsum += v * v;
    }

    float rms = sqsum / size;

    float a = 1.f / sqrtf(rms + eps);

    if (gamma_ptr)
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = float32_to_bfloat16((bfloat16_to_float32(ptr[i]) * a) * gamma_ptr[i]);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * a);
        }
    }
#endif // __riscv_vector

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

static void rmsnorm_bf16s(Mat& bottom_top_blob, const float* gamma_data, float eps, int affine_size, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        rmsnorm_bf16s_zvfbfmin(bottom_top_blob, gamma_data, eps, affine_size, opt);
        return;
    }
#endif

    const int elempack = bottom_top_blob.elempack;
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        // assert affine_size == w

        unsigned short* ptr = bottom_top_blob;
        rmsnorm_bf16s(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            rmsnorm_bf16s(ptr, gamma_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = bottom_top_blob.channel(q).row<unsigned short>(i);
                    rmsnorm_bf16s(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                rmsnorm_bf16s(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    if (dims == 4)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        unsigned short* ptr = bottom_top_blob.channel(q).depth(z).row<unsigned short>(i);
                        rmsnorm_bf16s(ptr, gamma_data, eps, w, elempack);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    unsigned short* ptr = bottom_top_blob.channel(q).depth(z);
                    rmsnorm_bf16s(ptr, gamma_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                rmsnorm_bf16s(ptr, gamma_data, eps, w * h * d, elempack);
            }
        }
    }
}
