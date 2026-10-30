// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void selu_bf16s_zvfbfmin(Mat& bottom_top_blob, float alpha, float lambda, const Option& opt);
#endif

static void selu_bf16s(Mat& bottom_top_blob, float alpha, float lambda, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        selu_bf16s_zvfbfmin(bottom_top_blob, alpha, lambda, opt);
        return;
    }
#endif

    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;
    const float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vbool8_t _lower = __riscv_vmflt_vf_f32m4_b8(_p, 0.f, vl);
            // Clamp unused positive inputs and saturated negative tails before the polynomial.
            vfloat32m4_t _x = __riscv_vfmax_vf_f32m4(__riscv_vfmin_vf_f32m4(_p, 0.f, vl), -88.f, vl);
            vfloat32m4_t _negative = __riscv_vfmul_vf_f32m4(expm1_ps(_x, vl), alphaxlambda, vl);
            _p = __riscv_vfmul_vf_f32m4_mu(__riscv_vmnot_m_b8(_lower, vl), _p, _p, lambda, vl);
            _p = __riscv_vmerge_vvm_f32m4(_p, _negative, _lower, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
            ptr += vl;
            n -= vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
        for (int i = 0; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[i]);
            v = v < 0.f ? expm1f(v) * alphaxlambda : v * lambda;
            ptr[i] = float32_to_bfloat16(v);
        }
#endif
    }
}
