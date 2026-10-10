// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

#if NCNN_ZFH
int BNLL_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);

            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            vbool2_t _mask = __riscv_vmfgt_vf_f16m8_b2(_p, (__fp16)0.f, vl);

#if __riscv_xtheadvector
            vfloat16m8_t _comm = __riscv_vfsgnjx_vv_f16m8(_p, _p, vl);
            _comm = __riscv_vfsgnjn_vv_f16m8(_comm, _comm, vl);
#else
            vfloat16m8_t _comm = __riscv_vfsgnjn_vv_f16m8_mu(_mask, _p, _p, _p, vl);
#endif
            _comm = exp_ps(_comm, vl);
            _comm = __riscv_vfadd_vf_f16m8(_comm, (__fp16)1.f, vl);
            _comm = log_ps(_comm, vl);

#if __riscv_xtheadvector
            vfloat16m8_t _res = __riscv_vfadd_vv_f16m8(_comm, _p, vl);
            _res = __riscv_vmerge_vvm_f16m8(_comm, _res, _mask, vl);
#else
            vfloat16m8_t _res = __riscv_vfadd_vv_f16m8_mu(_mask, _comm, _comm, _p, vl);
#endif

            __riscv_vse16_v_f16m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            float v = (float)*ptr;
            if (v > 0)
                *ptr = (__fp16)(v + logf(1.f + expf(v)));
            else
                *ptr = (__fp16)(logf(1.f + expf(v)));
            ++ptr;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
