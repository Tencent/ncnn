// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

int SELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if C906
    // FIXME -O3 leads illegal instruction
    return SELU::forward_inplace(bottom_top_blob, opt);
#endif

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
        float* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
            vbool4_t _higher = __riscv_vmnot_m_b4(_lower, vl);

            _p = __riscv_vfmul_vf_f32m8_mu(_higher, _p, _p, lambda, vl);
            vfloat32m8_t _nps = exp_ps(_p, vl);
            _nps = __riscv_vfsub_vf_f32m8_mu(_lower, _p, _nps, 1.f, vl);
            _nps = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _nps, alphaxlambda, vl);

            __riscv_vse32_v_f32m8(ptr, _nps, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = (expf(ptr[i]) - 1.f) * alphaxlambda;
            else
                ptr[i] *= lambda;
        }
#endif // __riscv_vector
    }
    return 0;
};

} // namespace ncnn
