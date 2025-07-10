// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#include "celu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include "cpu.h"
namespace ncnn {
#if NCNN_ZFH
int CELU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

#if __riscv_vector && !__riscv_xtheadvector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);

            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            vbool2_t _mask = __riscv_vmfgt_vf_f16m8_b2(_p, 0.f, vl);

            vfloat16m8_t _q = __riscv_vfdiv_vf_f16m8(_p, alpha, vl);
            _q = exp_ps(_q, vl);
            _q = __riscv_vfsub_vf_f16m8(_q, 1.f, vl);
            _q = __riscv_vfmul_vf_f16m8(_q, alpha, vl);

            vfloat16m8_t _res = __riscv_vmerge_vvm_f16m8(_q, _p, _mask, vl);
            __riscv_vse16_v_f16m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr < 0)
                *ptr = alpha * (expf((float)(*ptr / alpha)) - __fp16(1.f));
            ++ptr;
        }
#endif // __riscv_vector
    }

    return 0;
}
#endif
} // namespace ncnn
