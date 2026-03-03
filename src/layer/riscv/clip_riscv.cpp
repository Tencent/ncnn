// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Clip_riscv::Clip_riscv()
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

int Clip_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmax_vf_f32m8(_p, min, vl);
            _p = __riscv_vfmin_vf_f32m8(_p, max, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr < min)
                *ptr = min;

            if (*ptr > max)
                *ptr = max;

            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} //namespace ncnn
