// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int SwiGLU_riscv::forward_fp16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const __fp16* gateptr = (const __fp16*)gate.channel(q) + offset;
        const __fp16* upptr = (const __fp16*)up.channel(q) + offset;
        __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

#if __riscv_zvfh
        int n = len;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat16m2_t _g16 = __riscv_vle16_v_f16m2(gateptr, vl);
            vfloat16m2_t _u16 = __riscv_vle16_v_f16m2(upptr, vl);

            vfloat32m4_t _g = __riscv_vfwcvt_f_f_v_f32m4(_g16, vl);
            vfloat32m4_t _u = __riscv_vfwcvt_f_f_v_f32m4(_u16, vl);

            if (activation_limit > 0.f)
            {
                _g = __riscv_vfmin_vf_f32m4(_g, activation_limit, vl);
                _u = __riscv_vfmax_vf_f32m4(_u, -activation_limit, vl);
                _u = __riscv_vfmin_vf_f32m4(_u, activation_limit, vl);
            }

            // silu(scale * g) = g * sigmoid(scale * g)
            _g = __riscv_vfmul_vv_f32m4(_g, sigmoid_ps(__riscv_vfmul_vf_f32m4(_g, sigmoid_scale, vl), vl), vl);
            _u = __riscv_vfadd_vf_f32m4(_u, up_offset, vl);

            __riscv_vse16_v_f16m2(outptr, __riscv_vfncvt_f_f_w_f16m2(__riscv_vfmul_vv_f32m4(_g, _u, vl), vl), vl);

            gateptr += vl;
            upptr += vl;
            outptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < len; i++)
        {
            float g = (float)gateptr[i];
            float u = (float)upptr[i];
            if (activation_limit > 0.f)
            {
                g = std::min(g, activation_limit);
                u = std::min(std::max(u, -activation_limit), activation_limit);
            }
            outptr[i] = (__fp16)(g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset));
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
