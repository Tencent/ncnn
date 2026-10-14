// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16 && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_xtheadvector
int clip_bf16s_zvfbfmin(Mat& bottom_top_blob, float min, float max, const Option& opt);
#endif

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
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Clip_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

#if NCNN_BF16
int Clip_riscv::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZVFBFMIN && __riscv_vector && !__riscv_xtheadvector
#if NCNN_RUNTIME_CPU
    if (cpu_support_riscv_zvfbfmin())
#endif
    {
        return clip_bf16s_zvfbfmin(bottom_top_blob, min, max, opt);
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
        unsigned short* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = __riscv_vfmax_vf_f32m4(_p, min, vl);
            _p = __riscv_vfmin_vf_f32m4(_p, max, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < min)
                v = min;

            if (v > max)
                v = max;

            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}
#endif // NCNN_BF16

} //namespace ncnn
