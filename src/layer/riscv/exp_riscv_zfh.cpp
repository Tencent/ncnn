// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "exp_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#if __riscv_zvfh
#include "rvv_mathfun_fp16s.h"
#endif
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int Exp_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (base == -1.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

#if __riscv_zvfh
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                _p = __riscv_vfmul_vf_f32m8(_p, scale, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, shift, vl);
                _p = exp_ps(_p, vl);
                __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_zvfh
            for (int i = 0; i < size; i++)
            {
                ptr[i] = (__fp16)expf(shift + (float)ptr[i] * scale);
            }
#endif // __riscv_zvfh
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
            _p = __riscv_vfmul_vf_f32m8(_p, scale, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, shift, vl);
            _p = pow_ps(__riscv_vfmv_v_f_f32m8(base, vl), _p, vl);
            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (__fp16)powf(base, shift + (float)ptr[i] * scale);
        }
#endif // __riscv_zvfh
    }

    return 0;
}

int Exp_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;
    __fp16 _scale = (__fp16)scale;
    __fp16 _shift = (__fp16)shift;

    if (base == -1.f)
    {
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
                _p = __riscv_vfmul_vf_f16m8(_p, _scale, vl);
                _p = __riscv_vfadd_vf_f16m8(_p, _shift, vl);
                _p = exp_ps(_p, vl);
                __riscv_vse16_v_f16m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_zvfh
            for (int i = 0; i < size; i++)
            {
                ptr[i] = (__fp16)expf(shift + (float)ptr[i] * scale);
            }
#endif // __riscv_zvfh
        }

        return 0;
    }

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
            _p = __riscv_vfmul_vf_f16m8(_p, _scale, vl);
            _p = __riscv_vfadd_vf_f16m8(_p, _shift, vl);
            _p = pow_ps(__riscv_vfmv_v_f_f16m8((__fp16)base, vl), _p, vl);
            __riscv_vse16_v_f16m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (__fp16)powf(base, shift + (float)ptr[i] * scale);
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
