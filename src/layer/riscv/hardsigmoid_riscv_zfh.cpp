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

        __fp16 _lower = (__fp16)lower;
        __fp16 _upper = (__fp16)upper;
        __fp16 _alpha = (__fp16)alpha;
        __fp16 _beta = (__fp16)beta;

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);

            vbool2_t _is_lower = __riscv_vmflt_vf_f16m8_b2(_p, _lower, vl);
            vbool2_t _is_higher = __riscv_vmfgt_vf_f16m8_b2(_p, _upper, vl);
            vbool2_t _apply = __riscv_vmnor_mm_b2(_is_lower, _is_higher, vl);
            _p = __riscv_vfmerge_vfm_f16m8(_p, (__fp16)0.f, _is_lower, vl);
            _p = __riscv_vfmerge_vfm_f16m8(_p, (__fp16)1.f, _is_higher, vl);

            _p = __riscv_vfadd_vf_f16m8_mu(_apply, _p, __riscv_vfmul_vf_f16m8_m(_apply, _p, _alpha, vl), _beta, vl);
            __riscv_vse16_v_f16m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < _lower)
                ptr[i] = (__fp16)0.f;
            else if (ptr[i] > _upper)
                ptr[i] = (__fp16)1.f;
            else
                ptr[i] = ptr[i] * _alpha + _beta;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
