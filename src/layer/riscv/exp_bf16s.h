// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void exp_bf16s_zvfbfmin(Mat& bottom_top_blob, float scale, float shift, float base, const Option& opt);
#endif

static void exp_bf16s(Mat& bottom_top_blob, float scale, float shift, float base, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        exp_bf16s_zvfbfmin(bottom_top_blob, scale, shift, base, opt);
        return;
    }
#endif

    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;

    if (base == -1.f)
    {
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
                _p = __riscv_vfmul_vf_f32m4(_p, scale, vl);
                _p = __riscv_vfadd_vf_f32m4(_p, shift, vl);
                _p = exp_ps(_p, vl);
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
                ptr[i] = float32_to_bfloat16(expf(shift + v * scale));
            }
#endif
        }

        return;
    }

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
            _p = __riscv_vfmul_vf_f32m4(_p, scale, vl);
            _p = __riscv_vfadd_vf_f32m4(_p, shift, vl);
            _p = pow_ps(__riscv_vfmv_v_f_f32m4(base, vl), _p, vl);
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
            ptr[i] = float32_to_bfloat16(powf(base, shift + v * scale));
        }
#endif
    }
}
