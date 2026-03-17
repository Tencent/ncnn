// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int GELU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

#if __riscv_vector
    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            int n = size;
#if __riscv_zvfh
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);

                vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);

                // fast gelu: y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
                vfloat16m4_t _p2 = __riscv_vfmul_vv_f16m4(_p, _p, vl);
                vfloat16m4_t _p3 = __riscv_vfmul_vv_f16m4(_p2, _p, vl);
                vfloat16m4_t _arg = __riscv_vfmul_vf_f16m4(_p3, (__fp16)0.044715f, vl);
                _arg = __riscv_vfadd_vv_f16m4(_p, _arg, vl);
                _arg = __riscv_vfmul_vf_f16m4(_arg, (__fp16)0.79788452f, vl);

                // For tanh, convert to float32
                vfloat32m8_t _arg_f32 = __riscv_vfwcvt_f_f_v_f32m8(_arg, vl);
                vfloat32m8_t _tanharg_f32 = tanh_ps(_arg_f32, vl);
                vfloat16m4_t _tanharg = __riscv_vfncvt_f_f_w_f16m4(_tanharg_f32, vl);

                vfloat16m4_t _one = __riscv_vfmv_v_f_f16m4((__fp16)1.f, vl);
                _tanharg = __riscv_vfadd_vv_f16m4(_tanharg, _one, vl);
                _p = __riscv_vfmul_vv_f16m4(_p, _tanharg, vl);
                _p = __riscv_vfmul_vf_f16m4(_p, (__fp16)0.5f, vl);

                __riscv_vse16_v_f16m4(ptr, _p, vl);
                n -= vl;
                ptr += vl;
            }
#else  // __riscv_zvfh
            for (int i = 0; i < size; i++)
            {
                __fp16 v = ptr[i];
                // fast gelu: y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
                ptr[i] = (__fp16)(0.5f * (float)v * (1.0f + tanhf(0.79788452f * ((float)v + 0.044715f * (float)v * (float)v * (float)v))));
            }
#endif // __riscv_zvfh
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            int n = size;
#if __riscv_zvfh
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);

                // y = 0.5 * x * erfc(-x / sqrt(2))
                // Convert to float32 for erfc
                vfloat32m8_t _p_f32 = __riscv_vfwcvt_f_f_v_f32m8(_p, vl);
                vfloat32m8_t _perfc_f32 = __riscv_vfmul_vf_f32m8(_p_f32, -0.70710678f, vl);
                _p_f32 = __riscv_vfmul_vf_f32m8(_p_f32, 0.5f, vl);

                _perfc_f32 = erfc_ps(_perfc_f32, vl);

                _p_f32 = __riscv_vfmul_vv_f32m8(_p_f32, _perfc_f32, vl);
                _p = __riscv_vfncvt_f_f_w_f16m4(_p_f32, vl);

                __riscv_vse16_v_f16m4(ptr, _p, vl);
                n -= vl;
                ptr += vl;
            }
#else  // __riscv_zvfh
            for (int i = 0; i < size; i++)
            {
                // y = 0.5 * x * erfc(-x / sqrt(2))
                ptr[i] = (__fp16)(0.5f * (float)ptr[i] * erfcf(-0.70710678f * (float)ptr[i]));
            }
#endif // __riscv_zvfh
        }
    }

    return 0;
#endif // __riscv_vector

    return GELU::forward_inplace(bottom_top_blob, opt);
}
#endif // NCNN_ZFH

} // namespace ncnn
