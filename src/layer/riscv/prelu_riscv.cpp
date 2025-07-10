// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

PReLU_riscv::PReLU_riscv()
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

int PReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        float* ptr = bottom_top_blob;
        if (num_slope > 1)
        {
#if __riscv_vector
            const float* ptr_slope = slope_data;

            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _slope = __riscv_vle32_v_f32m8(ptr_slope, vl);
                vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, .0f, vl);

                _p = __riscv_vfmul_vv_f32m8_mu(_lower, _p, _p, _slope, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
#else // __riscv_vector
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope_data[i];
            }
#endif // __riscv_vector
        }
        else
        {
            float slope = slope_data[0];

#if __riscv_vector
            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, .0f, vl);

                _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else // __riscv_vector
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
#endif // __riscv_vector
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
#if __riscv_vector
            if (num_slope > 1)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* ptr_slope = (const float*)slope_data + i * elempack;
                    int n = elempack;

                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _slope = __riscv_vle32_v_f32m8(ptr_slope, vl);

                        vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, .0f, vl);
                        _p = __riscv_vfmul_vv_f32m8_mu(_lower, _p, _p, _slope, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);

                        ptr += vl;
                        ptr_slope += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                float slope = slope_data[0];
                int n = w * elempack;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, .0f, vl);

                    _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_vector
            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            for (int j = 0; j < w; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope;
            }
#endif // __riscv_vector
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
            int n = size * elempack;

            if (num_slope > 1 && elempack != 1)
            {
                while (n > 0)
                {
                    int n1 = elempack;
                    const float* slope_ptr = (const float*)slope_data + q * elempack;
                    while (n1 > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n1);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _slope = __riscv_vle32_v_f32m8(slope_ptr, vl);

                        vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, .0f, vl);
                        _p = __riscv_vfmul_vv_f32m8_mu(_lower, _p, _p, _slope, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);

                        ptr += vl;
                        slope_ptr += vl;
                        n1 -= vl;
                    }
                    n -= elempack;
                }
            }
            else
            {
                // num_slope == 1 or elempack ==1
                float slope = num_slope > 1 ? slope_data[q] : slope_data[0];
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);

                    vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, .0f, vl);
                    _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_vector
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
#endif // __riscv_vector
        }
    }

    return 0;
}

} // namespace ncnn
