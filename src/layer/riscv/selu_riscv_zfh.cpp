// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int SELU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    float alphaxlambda = alpha * lambda;

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
            vbool4_t _lower = __riscv_vmflt_vf_f16m4_b4(_p, (__fp16)0.f, vl);
            vbool4_t _higher = __riscv_vmnot_m_b4(_lower, vl);

            // Positive part: x * lambda
            _p = __riscv_vfmul_vf_f16m4_mu(_higher, _p, _p, (__fp16)lambda, vl);

            // Negative part: (exp(x) - 1) * alphaxlambda
            // Convert to float32 for exp calculation
            vfloat32m8_t _p_f32 = __riscv_vfwcvt_f_f_v_f32m8(_p, vl);
            vfloat32m8_t _nps_f32 = exp_ps(_p_f32, vl);
            _nps_f32 = __riscv_vfsub_vf_f32m8(_nps_f32, 1.f, vl);
            _nps_f32 = __riscv_vfmul_vf_f32m8(_nps_f32, alphaxlambda, vl);
            vfloat16m4_t _nps = __riscv_vfncvt_f_f_w_f16m4(_nps_f32, vl);

            // Merge results
            _p = __riscv_vmerge_vvm_f16m4(_nps, _p, _lower, vl);

            __riscv_vse16_v_f16m4(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < (__fp16)0.f)
                ptr[i] = (__fp16)((expf((float)ptr[i]) - 1.f) * alphaxlambda);
            else
                ptr[i] = ptr[i] * (__fp16)lambda;
        }
#endif // __riscv_zvfh
    }
    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
