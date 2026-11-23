// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void gelu_bf16s_zvfbfmin(Mat& bottom_top_blob, int fast_gelu, const Option& opt);
#endif

static void gelu_bf16s(Mat& bottom_top_blob, int fast_gelu, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        gelu_bf16s_zvfbfmin(bottom_top_blob, fast_gelu, opt);
        return;
    }
#endif

    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;

    if (fast_gelu)
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

                vfloat32m4_t _arg = __riscv_vfmul_vf_f32m4(
                                        __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(_p, _p, vl), _p, vl), 0.044715f, vl);

                _arg = __riscv_vfadd_vv_f32m4(_p, _arg, vl);
                _arg = __riscv_vfmul_vf_f32m4(_arg, 0.79788452f, vl);
                vfloat32m4_t _tanharg = tanh_ps(_arg, vl);
                _p = __riscv_vfmul_vf_f32m4(
                         __riscv_vfmul_vv_f32m4(_p, __riscv_vfadd_vf_f32m4(_tanharg, 1.f, vl), vl), .5f, vl);

                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
                n -= vl;
                ptr += vl;
            }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
            for (int i = 0; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
                ptr[i] = float32_to_bfloat16(0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v))));
            }
#endif
        }
    }
    else
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
                vfloat32m4_t _perfc = __riscv_vfmul_vf_f32m4(_p, -.70710678f, vl);
                _p = __riscv_vfmul_vf_f32m4(_p, .5f, vl);
                // y = x * P(X <= x) where X ~ N(0, 1)

                _perfc = erfc_ps(_perfc, vl);

                _p = __riscv_vfmul_vv_f32m4(_p, _perfc, vl);
                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

                n -= vl;
                ptr += vl;
            }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
            for (int i = 0; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                // y = x * P(X <= x) where X ~ N(0, 1)
                ptr[i] = float32_to_bfloat16(0.5f * v * erfcf(-0.70710678f * v));
            }
#endif
        }
    }
}
