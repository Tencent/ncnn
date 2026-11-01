// Copyright 2021 Xavier Hsinyuan <thelastlinex@hotmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

ReLU_riscv::ReLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
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

int ReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        if (slope == 0.f)
        {
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int i = 0; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;
                ptr++;
            }
#endif // __riscv_vector
        }
        else
        {
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmul_vf_f32m8_mu(__riscv_vmflt_vf_f32m8_b4(_p, .0f, vl), _p, _p, slope, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int i = 0; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
#endif // __riscv_vector
        }
    }

    return 0;
}

#if NCNN_BF16
int ReLU_riscv::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        if (slope == 0.f)
        {
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m2(n);
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                _p = __riscv_vfmax_vf_f32m4(_p, 0.f, vl);
                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
                ptr += vl;
                n -= vl;
            }
        }
        else
        {
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m2(n);
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                _p = __riscv_vfmul_vf_f32m4_mu(__riscv_vmflt_vf_f32m4_b8(_p, 0.f, vl), _p, _p, slope, vl);
                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
                ptr += vl;
                n -= vl;
            }
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
        if (slope == 0.f)
        {
            for (int i = 0; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = 0;
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope);
            }
        }
#endif
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
