// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "hardsigmoid_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "hardsigmoid_bf16s.h"
#endif

HardSigmoid_riscv::HardSigmoid_riscv()
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
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int HardSigmoid_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH || NCNN_BF16
    int elembits = bottom_top_blob.elembits();
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
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

            _p = __riscv_vfmul_vf_f32m8(_p, alpha, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, beta, vl);
            _p = __riscv_vfmax_vf_f32m8(_p, .0f, vl);
            _p = __riscv_vfmin_vf_f32m8(_p, 1.f, vl);

            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            // keep the scalar path identical to the vectorized body for alpha <= 0 too
            float t = ptr[i] * alpha + beta;
            if (t < 0.f)
                t = 0.f;
            if (t > 1.f)
                t = 1.f;
            ptr[i] = t;
        }
#endif // __riscv_vector
    }

    return 0;
}

#if NCNN_BF16
int HardSigmoid_riscv::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    hardsigmoid_bf16s(bottom_top_blob, alpha, beta, lower, upper, opt);
    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
