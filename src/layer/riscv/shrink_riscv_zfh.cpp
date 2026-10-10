// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "shrink_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int Shrink_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
            vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, -lambd, vl);
            vbool4_t _higher = __riscv_vmfgt_vf_f32m8_b4(_p, lambd, vl);

            vfloat32m8_t _out = __riscv_vfmv_v_f_f32m8(0.f, vl);
            _out = __riscv_vfadd_vf_f32m8_mu(_lower, _out, _p, bias, vl);
            _out = __riscv_vfsub_vf_f32m8_mu(_higher, _out, _p, bias, vl);
            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_out, vl), vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] < -lambd ? (__fp16)(ptr[i] + bias) : ptr[i] > lambd ? (__fp16)(ptr[i] - bias) : (__fp16)0.f;
        }
#endif // __riscv_zvfh
    }

    return 0;
}

int Shrink_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    const __fp16 _bias = (__fp16)bias;
    const __fp16 _lambd = (__fp16)lambd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);

            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, -_lambd, vl);
            vbool2_t _higher = __riscv_vmfgt_vf_f16m8_b2(_p, _lambd, vl);

            vfloat16m8_t _out = __riscv_vfmv_v_f_f16m8((__fp16)0.f, vl);
            _out = __riscv_vfadd_vf_f16m8_mu(_lower, _out, _p, _bias, vl);
            _out = __riscv_vfsub_vf_f16m8_mu(_higher, _out, _p, _bias, vl);
            __riscv_vse16_v_f16m8(ptr, _out, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] < -_lambd ? ptr[i] + _bias : ptr[i] > _lambd ? ptr[i] - _bias : (__fp16)0.f;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
