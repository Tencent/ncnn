// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int Clip_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

        __fp16 _min = (__fp16)min;
        __fp16 _max = (__fp16)max;
#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
            _p = __riscv_vfmax_vf_f32m8(_p, _min, vl);
            _p = __riscv_vfmin_vf_f32m8(_p, _max, vl);
            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr < _min)
                *ptr = _min;

            if (*ptr > _max)
                *ptr = _max;

            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}
#endif // NCNN_ZFH

} //namespace ncnn
