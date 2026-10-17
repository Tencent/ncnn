// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardsigmoid_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int HardSigmoid_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        __fp16 _alpha = (__fp16)alpha;
        __fp16 _beta = (__fp16)beta;

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);

            _p = __riscv_vfmul_vf_f16m8(_p, _alpha, vl);
            _p = __riscv_vfadd_vf_f16m8(_p, _beta, vl);
            _p = __riscv_vfmax_vf_f16m8(_p, (__fp16)0.f, vl);
            _p = __riscv_vfmin_vf_f16m8(_p, (__fp16)1.f, vl);
            __riscv_vse16_v_f16m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            // keep the scalar path identical to the vectorized body for alpha <= 0 too
            __fp16 t = ptr[i] * _alpha + _beta;
            if (t < (__fp16)0.f)
                t = (__fp16)0.f;
            if (t > (__fp16)1.f)
                t = (__fp16)1.f;
            ptr[i] = t;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
