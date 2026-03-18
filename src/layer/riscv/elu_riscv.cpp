// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

ELU_riscv::ELU_riscv()
{
    support_packing = true;
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

int ELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if C906
    // FIXME -O3 leads illegal instruction
    return ELU::forward_inplace(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

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

            vfloat32m8_t _exp_v = exp_ps(_p, vl);
            _exp_v = __riscv_vfsub_vf_f32m8(_exp_v, 1.f, vl);
            _exp_v = __riscv_vfmul_vf_f32m8(_exp_v, alpha, vl);
            _p = __riscv_vmerge_vvm_f32m8(_p, _exp_v, _lower, vl);

            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = alpha * (expf(ptr[i]) - 1.f);
        }
#endif // __riscv_vector
    }
    return 0;
}

} // namespace ncnn
