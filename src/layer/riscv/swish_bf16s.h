// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void swish_bf16s_zvfbfmin(Mat& bottom_top_blob, const Option& opt);
#endif

static void swish_bf16s(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        swish_bf16s_zvfbfmin(bottom_top_blob, opt);
        return;
    }
#endif

    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;

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
            vfloat32m4_t _x = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vbool8_t _negative = __riscv_vmflt_vf_f32m4_b8(_x, 0.f, vl);
            // Halving the exponent keeps the negative tail in the FP32 normal range.
            vfloat32m4_t _e = exp_ps(__riscv_vfmul_vf_f32m4(__riscv_vfabs_v_f32m4(_x, vl), -0.5f, vl), vl);
            vfloat32m4_t _p = __riscv_vfdiv_vv_f32m4(_x, __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(_e, _e, vl), 1.f, vl), vl);
            _p = __riscv_vfmul_vv_f32m4_mu(_negative, _p, _p, _e, vl);
            _p = __riscv_vfmul_vv_f32m4_mu(_negative, _p, _p, _e, vl);
            // All smaller negative inputs round to BF16 zero; avoid exp_ps clamping.
            _p = __riscv_vfmerge_vfm_f32m4(_p, -0.f, __riscv_vmflt_vf_f32m4_b8(_x, -104.f, vl), vl);
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
            float e = expf(-fabsf(v) * 0.5f);
            float y = v / (1.f + e * e);
            if (v < 0.f)
                y = (y * e) * e;
            ptr[i] = float32_to_bfloat16(y);
        }
#endif
    }
}
