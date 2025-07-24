// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int BatchNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
#if __riscv_zvfh
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        int n = bottom_top_blob.w * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
            vfloat32m8_t _a = __riscv_vle32_v_f32m8(ptr_a, vl);
            vfloat32m8_t _b = __riscv_vle32_v_f32m8(ptr_b, vl);

            _p = __riscv_vfmadd_vv_f32m8(_p, _b, _a, vl);

            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

            ptr += vl;
            ptr_a += vl;
            ptr_b += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        int w = bottom_top_blob.w;
        for (int i = 0; i < w; i++)
        {
            ptr[i] = (__fp16)(b_data[i] * (float)ptr[i] + a_data[i]);
        }
#endif // __riscv_zvfh

        return 0;
    }

    if (elempack == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                float a = a_data[i];
                float b = b_data[i];

#if __riscv_zvfh
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m4(n);
                    vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, a, vl);
                    __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
#else  // __riscv_zvfh
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = (__fp16)(b * (float)ptr[j] + a);
                }
#endif // __riscv_zvfh
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
                __fp16* ptr = bottom_top_blob.channel(q);
                float a = a_data[q];
                float b = b_data[q];

#if __riscv_zvfh
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m4(n);
                    vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, a, vl);
                    __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
#else  // __riscv_zvfh
                for (int i = 0; i < size; i++)
                {
                    ptr[i] = (__fp16)(b * (float)ptr[i] + a);
                }
#endif // __riscv_zvfh
            }
        }

        return 0;
    }

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        const size_t vl = __riscv_vsetvl_e16m1(packn);
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                const float* ptr_a = (const float*)a_data + i * elempack;
                const float* ptr_b = (const float*)b_data + i * elempack;
                int n = w * elempack;

                vfloat32m2_t _a = __riscv_vle32_v_f32m2(ptr_a, vl);
                vfloat32m2_t _b = __riscv_vle32_v_f32m2(ptr_b, vl);
                while (n > 0)
                {
                    vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr, vl), vl);
                    _p = __riscv_vfmadd_vv_f32m2(_p, _b, _a, vl);
                    __riscv_vse16_v_f16m1(ptr, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);

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
                __fp16* ptr = bottom_top_blob.channel(q);
                const float* ptr_a = (const float*)a_data + q * elempack;
                const float* ptr_b = (const float*)b_data + q * elempack;

                vfloat32m2_t _a = __riscv_vle32_v_f32m2(ptr_a, vl);
                vfloat32m2_t _b = __riscv_vle32_v_f32m2(ptr_b, vl);

                int n = size;
                while (n > 0)
                {
                    vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr, vl), vl);
                    _p = __riscv_vfmadd_vv_f32m2(_p, _b, _a, vl);
                    __riscv_vse16_v_f16m1(ptr, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }
#endif // __riscv_zvfh

    return 0;
}

int BatchNorm_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
#if __riscv_zvfh
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        int n = bottom_top_blob.w * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);
            vfloat16m4_t _a = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_a, vl), vl);
            vfloat16m4_t _b = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_b, vl), vl);

            _p = __riscv_vfmadd_vv_f16m4(_p, _b, _a, vl);

            __riscv_vse16_v_f16m4(ptr, _p, vl);

            ptr += vl;
            ptr_a += vl;
            ptr_b += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        int w = bottom_top_blob.w;
        for (int i = 0; i < w; i++)
        {
            ptr[i] = (__fp16)b_data[i] * ptr[i] + (__fp16)a_data[i];
        }
#endif // __riscv_zvfh

        return 0;
    }

    if (elempack == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                __fp16 a = (__fp16)a_data[i];
                __fp16 b = (__fp16)b_data[i];

#if __riscv_zvfh
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);
                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    _p = __riscv_vfmul_vf_f16m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f16m8(_p, a, vl);
                    __riscv_vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
#else  // __riscv_zvfh
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = b * ptr[j] + a;
                }
#endif // __riscv_zvfh
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
                __fp16* ptr = bottom_top_blob.channel(q);
                __fp16 a = (__fp16)a_data[q];
                __fp16 b = (__fp16)b_data[q];

#if __riscv_zvfh
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);
                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    _p = __riscv_vfmul_vf_f16m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f16m8(_p, a, vl);
                    __riscv_vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
#else  // __riscv_zvfh
                for (int i = 0; i < size; i++)
                {
                    ptr[i] = b * ptr[i] + a;
                }
#endif // __riscv_zvfh
            }
        }

        return 0;
    }

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        const size_t vl = __riscv_vsetvl_e16m1(packn);
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                const float* ptr_a = (const float*)a_data + i * elempack;
                const float* ptr_b = (const float*)b_data + i * elempack;
                int n = w * elempack;

                vfloat16m1_t _a = __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(ptr_a, vl), vl);
                vfloat16m1_t _b = __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(ptr_b, vl), vl);
                while (n > 0)
                {
                    vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr, vl);
                    _p = __riscv_vfmadd_vv_f16m1(_p, _b, _a, vl);
                    __riscv_vse16_v_f16m1(ptr, _p, vl);

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
                __fp16* ptr = bottom_top_blob.channel(q);
                const float* ptr_a = (const float*)a_data + q * elempack;
                const float* ptr_b = (const float*)b_data + q * elempack;

                vfloat16m1_t _a = __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(ptr_a, vl), vl);
                vfloat16m1_t _b = __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(ptr_b, vl), vl);

                int n = size;
                while (n > 0)
                {
                    vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr, vl);
                    _p = __riscv_vfmadd_vv_f16m1(_p, _b, _a, vl);
                    __riscv_vse16_v_f16m1(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }
#endif // __riscv_zvfh

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
