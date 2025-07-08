// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
//fp16s(a)
//hint: slope always store as fp32

int PReLU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr = bottom_top_blob;
        if (num_slope > 1)
        {
#if __riscv_zvfh
            const float* ptr_slope = slope_data;

            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);

                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                vfloat32m8_t _slope = __riscv_vle32_v_f32m8(ptr_slope, vl);
                vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, (__fp16)0.f, vl);
                _p = __riscv_vfmul_vv_f32m8_mu(_lower, _p, _p, _slope, vl);

                __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);
                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
#else // __riscv_zvfh
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < (__fp16)0.f)
                    ptr[i] = (__fp16)((float)ptr[i] * slope_data[i]);
            }
#endif // __riscv_zvfh
        }
        else
        {
            float slope = slope_data[0];

#if __riscv_zvfh
            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, (__fp16)0.f, vl);

                _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                ptr += vl;
                n -= vl;
            }
#else // __riscv_zvfh
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < (__fp16)0.f)
                    ptr[i] = (__fp16)((float)ptr[i] * slope);
            }
#endif // __riscv_zvfh
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
#if __riscv_zvfh
            if (num_slope > 1)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* ptr_slope = (const float*)slope_data + i * elempack;
                    int n = elempack;

                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e16m4(n);
                        vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                        vfloat32m8_t _slope = __riscv_vle32_v_f32m8(ptr_slope, vl);

                        vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, (__fp16)0.f, vl);
                        _p = __riscv_vfmul_vv_f32m8_mu(_lower, _p, _p, _slope, vl);
                        __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

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
                    size_t vl = __riscv_vsetvl_e16m4(n);
                    vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                    vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, (__fp16)0.f, vl);

                    _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                    __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_zvfh
            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            for (int j = 0; j < w; j++)
            {
                if (ptr[j] < (__fp16)0.f)
                    ptr[j] = (__fp16)((float)ptr[j] * slope);
            }
#endif // __riscv_zvfh
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
            __fp16* ptr = bottom_top_blob.channel(q);
#if __riscv_zvfh
            int n = size * elempack;

            if (num_slope > 1 && elempack != 1)
            {
                while (n > 0)
                {
                    int n1 = elempack;
                    const float* slope_ptr = (const float*)slope_data + q * elempack;
                    while (n1 > 0)
                    {
                        size_t vl = __riscv_vsetvl_e16m4(n1);
                        vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
                        vfloat32m8_t _slope = __riscv_vle32_v_f32m8(slope_ptr, vl);

                        vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, (__fp16)0.f, vl);
                        _p = __riscv_vfmul_vv_f32m8_mu(_lower, _p, _p, _slope, vl);
                        __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

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
                    size_t vl = __riscv_vsetvl_e16m4(n);
                    vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);

                    vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, (__fp16)0.f, vl);
                    _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                    __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_zvfh
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < (__fp16)0.f)
                    ptr[i] = (__fp16)((float)ptr[i] * slope);
            }
#endif // __riscv_zvfh
        }
    }

    return 0;
}

int PReLU_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr = bottom_top_blob;
        if (num_slope > 1)
        {
#if __riscv_zvfh
            const float* ptr_slope = slope_data;

            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);
                vfloat16m4_t _slope = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_slope, vl), vl);
                vbool4_t _lower = __riscv_vmflt_vf_f16m4_b4(_p, (__fp16)0.f, vl);

                _p = __riscv_vfmul_vv_f16m4_mu(_lower, _p, _p, _slope, vl);
                __riscv_vse16_v_f16m4(ptr, _p, vl);

                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
#else // __riscv_zvfh
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < (__fp16)0.f)
                    ptr[i] *= (__fp16)slope_data[i];
            }
#endif // __riscv_zvfh
        }
        else
        {
            __fp16 slope = (__fp16)slope_data[0];

#if __riscv_zvfh
            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);
                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl);

                _p = __riscv_vfmul_vf_f16m8_mu(_lower, _p, _p, slope, vl);
                __riscv_vse16_v_f16m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else // __riscv_zvfh
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < (__fp16)0.f)
                    ptr[i] *= slope;
            }
#endif // __riscv_zvfh
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
#if __riscv_zvfh
            if (num_slope > 1)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* ptr_slope = (const float*)slope_data + i * elempack;
                    int n = elempack;

                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e16m4(n);
                        vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);
                        vfloat16m4_t _slope = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_slope, vl), vl);

                        vbool4_t _lower = __riscv_vmflt_vf_f16m4_b4(_p, (__fp16)0.f, vl);
                        _p = __riscv_vfmul_vv_f16m4_mu(_lower, _p, _p, _slope, vl);
                        __riscv_vse16_v_f16m4(ptr, _p, vl);

                        ptr += vl;
                        ptr_slope += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                __fp16 slope = (__fp16)slope_data[0];
                int n = w * elempack;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);
                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl);

                    _p = __riscv_vfmul_vf_f16m8_mu(_lower, _p, _p, slope, vl);
                    __riscv_vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_zvfh
            __fp16 slope = num_slope > 1 ? (__fp16)slope_data[i] : (__fp16)slope_data[0];

            for (int j = 0; j < w; j++)
            {
                if (ptr[j] < (__fp16)0.f)
                    ptr[j] *= slope;
            }
#endif // __riscv_zvfh
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
            __fp16* ptr = bottom_top_blob.channel(q);
#if __riscv_zvfh
            int n = size * elempack;

            if (num_slope > 1 && elempack != 1)
            {
                while (n > 0)
                {
                    int n1 = elempack;
                    const float* slope_ptr = (const float*)slope_data + q * elempack;
                    while (n1 > 0)
                    {
                        size_t vl = __riscv_vsetvl_e16m4(n1);
                        vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);
                        vfloat16m4_t _slope = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(slope_ptr, vl), vl);

                        vbool4_t _lower = __riscv_vmflt_vf_f16m4_b4(_p, (__fp16)0.f, vl);
                        _p = __riscv_vfmul_vv_f16m4_mu(_lower, _p, _p, _slope, vl);
                        __riscv_vse16_v_f16m4(ptr, _p, vl);

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
                    size_t vl = __riscv_vsetvl_e16m8(n);
                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);

                    vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl);
                    _p = __riscv_vfmul_vf_f16m8_mu(_lower, _p, _p, (__fp16)slope, vl);
                    __riscv_vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
#else  // __riscv_zvfh
            __fp16 slope = num_slope > 1 ? (__fp16)slope_data[q] : (__fp16)slope_data[0];

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < (__fp16)0.f)
                    ptr[i] *= slope;
            }
#endif // __riscv_zvfh
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
