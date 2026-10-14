// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"

#if NCNN_BF16
#include <riscv_vector.h>

namespace ncnn {

int clip_bf16s_zvfbfmin(Mat& bottom_top_blob, float min, float max, const Option& opt)
{
    const int channels = bottom_top_blob.c;
    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vbfloat16m2_t _p = __riscv_vreinterpret_v_u16m2_bf16m2(__riscv_vle16_v_u16m2(ptr, vl));
            vfloat32m4_t _v = __riscv_vfwcvtbf16_f_f_v_f32m4(_p, vl);
            _v = __riscv_vfmax_vf_f32m4(_v, min, vl);
            _v = __riscv_vfmin_vf_f32m4(_v, max, vl);
            _p = __riscv_vfncvtbf16_f_f_w_bf16m2_rm(_v, __RISCV_FRM_RNE, vl);
            __riscv_vse16_v_u16m2(ptr, __riscv_vreinterpret_v_bf16m2_u16m2(_p), vl);
            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}

} // namespace ncnn
#endif // NCNN_BF16
