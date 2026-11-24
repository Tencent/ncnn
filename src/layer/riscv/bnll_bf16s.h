// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void bnll_bf16s_zvfbfmin(Mat& bottom_top_blob, const Option& opt);
#endif

static void bnll_bf16s(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        bnll_bf16s_zvfbfmin(bottom_top_blob, opt);
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
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vbool8_t _mask = __riscv_vmfgt_vf_f32m4_b8(_p, 0.f, vl);

#if __riscv_xtheadvector
            vfloat32m4_t _comm = __riscv_vfsgnjx_vv_f32m4(_p, _p, vl);
            _comm = __riscv_vfsgnjn_vv_f32m4(_comm, _comm, vl);
#else
            vfloat32m4_t _comm = __riscv_vfsgnjn_vv_f32m4_mu(_mask, _p, _p, _p, vl);
#endif
            _comm = exp_ps(_comm, vl);
            _comm = __riscv_vfadd_vf_f32m4(_comm, 1.f, vl);
            _comm = log_ps(_comm, vl);

#if __riscv_xtheadvector
            vfloat32m4_t _res = __riscv_vfadd_vv_f32m4(_comm, _p, vl);
            _res = __riscv_vmerge_vvm_f32m4(_comm, _res, _mask, vl);
#else
            vfloat32m4_t _res = __riscv_vfadd_vv_f32m4_mu(_mask, _comm, _comm, _p, vl);
#endif
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_res, vl), vl);
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
            if (v > 0.f)
                v = v + logf(1.f + expf(-v));
            else
                v = logf(1.f + expf(v));
            ptr[i] = float32_to_bfloat16(v);
        }
#endif
    }
}
