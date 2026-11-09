// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
//fp16s
//hint: scale and bias are computed in fp32

int Scale_riscv::forward_inplace_fp16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    // scale_blob may be fp16 (from the second input) or fp32 (from scale_data weight)
    Mat scale_blob_fp32 = scale_blob;
    if (scale_blob.elembits() == 16)
    {
        cast_float16_to_float32(scale_blob, scale_blob_fp32, opt);
        if (scale_blob_fp32.empty())
            return -100;
    }

    const float* scale = scale_blob_fp32;
    const float* bias = bias_data;

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
        const int size = bottom_top_blob.w * elempack;

#if __riscv_zvfh
        int n = size;
        if (bias_term)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);

                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(scale, vl);
                vfloat32m8_t _bias = __riscv_vle32_v_f32m8(bias, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _s, _bias, vl);
                __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                ptr += vl;
                scale += vl;
                bias += vl;
                n -= vl;
            }
        }
        else
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);

                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(scale, vl);
                _p = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                ptr += vl;
                scale += vl;
                n -= vl;
            }
        }
#else // __riscv_zvfh
        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                ptr[i] = (__fp16)((float)ptr[i] * scale[i] + bias[i]);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                ptr[i] = (__fp16)((float)ptr[i] * scale[i]);
            }
        }
#endif // __riscv_zvfh

        return 0;
    }

    const int channels = dims == 2 ? bottom_top_blob.h : bottom_top_blob.c;
    const int size = dims == 2 ? bottom_top_blob.w : bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d;

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = dims == 2 ? bottom_top_blob.row<__fp16>(q) : bottom_top_blob.channel(q);
            const float s = scale[q];

#if __riscv_zvfh
            int n = size;
            if (bias_term)
            {
                vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(bias[q], __riscv_vsetvlmax_e32m8());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m4(n);

                    vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                    _p = __riscv_vfmadd_vf_f32m8(_p, s, _bias, vl);
                    __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
            else
            {
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m4(n);

                    vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, s, vl);
                    __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_zvfh
            if (bias_term)
            {
                const float b = bias[q];
                for (int i = 0; i < size; i++)
                {
                    ptr[i] = (__fp16)((float)ptr[i] * s + b);
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    ptr[i] = (__fp16)((float)ptr[i] * s);
                }
            }
#endif // __riscv_zvfh
        }

        return 0;
    }

#if __riscv_zvfh
    // elempack == packn
    const size_t vl = __riscv_vsetvl_e16m1(elempack);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = dims == 2 ? bottom_top_blob.row<__fp16>(q) : bottom_top_blob.channel(q);

        vfloat32m2_t _s = __riscv_vle32_v_f32m2(scale + q * elempack, vl);

        if (bias_term)
        {
            vfloat32m2_t _bias = __riscv_vle32_v_f32m2(bias + q * elempack, vl);
            for (int i = 0; i < size; i++)
            {
                vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr, vl), vl);
                _p = __riscv_vfmadd_vv_f32m2(_p, _s, _bias, vl);
                __riscv_vse16_v_f16m1(ptr, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);

                ptr += vl;
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr, vl), vl);
                _p = __riscv_vfmul_vv_f32m2(_p, _s, vl);
                __riscv_vse16_v_f16m1(ptr, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);

                ptr += vl;
            }
        }
    }
#endif // __riscv_zvfh

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
