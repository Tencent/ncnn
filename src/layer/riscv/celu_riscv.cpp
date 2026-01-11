// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#include "celu_riscv.h"
#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

CELU_riscv::CELU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

int CELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
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
        // FIXME Suspected upstream bug
        // Incorrect answer on Xuantie c910
#if __riscv_vector && !__riscv_xtheadvector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vbool4_t _mask = __riscv_vmfgt_vf_f32m8_b4(_p, 0.f, vl);

            vfloat32m8_t _q = __riscv_vfdiv_vf_f32m8(_p, alpha, vl);
            _q = exp_ps(_q, vl);
            _q = __riscv_vfsub_vf_f32m8(_q, 1.f, vl);
            _q = __riscv_vfmul_vf_f32m8(_q, alpha, vl);

            vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_q, _p, _mask, vl);
            __riscv_vse32_v_f32m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr < 0)
                *ptr = alpha * (expf(*ptr / alpha) - 1.f);
            ++ptr;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
