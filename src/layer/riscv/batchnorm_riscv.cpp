// Copyright 2022 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

BatchNorm_riscv::BatchNorm_riscv()
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

int BatchNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        float* ptr = bottom_top_blob;
#if __riscv_vector
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        int n = bottom_top_blob.w * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _a = __riscv_vle32_v_f32m8(ptr_a, vl);
            vfloat32m8_t _b = __riscv_vle32_v_f32m8(ptr_b, vl);

            _p = __riscv_vfmadd_vv_f32m8(_p, _b, _a, vl);

            __riscv_vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            ptr_a += vl;
            ptr_b += vl;
            n -= vl;
        }
#else
        int w = bottom_top_blob.w;
        for (int i = 0; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
#endif // __riscv_vector
        return 0;
    }

#if __riscv_vector
    if (elempack == 1)
#endif
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float a = a_data[i];
                float b = b_data[i];

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, a, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
#else
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = b * ptr[j] + a;
                }
#endif // __riscv_vector
            }
        }
        if (dims == 3 || dims == 4)
        {
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float a = a_data[q];
                float b = b_data[q];

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, a, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    ptr[i] = b * ptr[i] + a;
                }
#endif // __riscv_vector
            }
        }
        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        const size_t vl = __riscv_vsetvl_e32m1(packn);
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                const float* ptr_a = a_data;
                ptr_a += i * elempack;
                const float* ptr_b = b_data;
                ptr_b += i * elempack;
                int n = w * elempack;

                vfloat32m1_t _a = __riscv_vle32_v_f32m1(ptr_a, vl);
                vfloat32m1_t _b = __riscv_vle32_v_f32m1(ptr_b, vl);
                while (n > 0)
                {
                    vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                    _p = __riscv_vfmadd_vv_f32m1(_p, _b, _a, vl);
                    __riscv_vse32_v_f32m1(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d * elempack;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                const float* ptr_a = (const float*)a_data + q * elempack;
                const float* ptr_b = (const float*)b_data + q * elempack;

                vfloat32m1_t _a = __riscv_vle32_v_f32m1(ptr_a, vl);
                vfloat32m1_t _b = __riscv_vle32_v_f32m1(ptr_b, vl);

                int n = size;
                while (n > 0)
                {
                    vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                    _p = __riscv_vfmadd_vv_f32m1(_p, _b, _a, vl);
                    __riscv_vse32_v_f32m1(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }
#endif
    return 0;
}

} // namespace ncnn
