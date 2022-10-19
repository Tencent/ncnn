// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "prelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

PReLU_riscv::PReLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif
}

int PReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
#if __riscv_vector
    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        float* ptr = bottom_top_blob;
        const float* ptr_slope = slope_data;
        if (num_slope > 1)
        {
            int n = w * elempack;

            // #pragma omp parallel for num_threads(opt.num_threads)
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _slope = vle32_v_f32m8(ptr_slope, vl);
                vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);

                _p = vfmul_vv_f32m8_m(_lower, _p, /*op1*/ _p, _slope, vl);
                vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
        }
        else
        {
            float slope = slope_data[0];

            int n = w * elempack;
            // #pragma omp parallel for num_threads(opt.num_threads)
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);

                _p = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
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
            if (num_slope > 1)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* ptr_slope = (const float*)slope_data + i * elempack;
                    int n = elempack;

                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _slope = vle32_v_f32m8(ptr_slope, vl);

                        vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                        _p = vfmul_vv_f32m8_m(_lower, _p, /*op1*/ _p, _slope, vl);
                        vse32_v_f32m8(ptr, _p, vl);

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
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);

                    _p = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                    vse32_v_f32m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
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
            int n = size * elempack;

            if (num_slope > 1 && elempack != 1)
            {
                while (n > 0)
                {
                    int n1 = elempack;
                    const float* slope_ptr = (const float*)slope_data + q * elempack;
                    while (n1 > 0)
                    {
                        size_t vl = vsetvl_e32m8(n1);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _slope = vle32_v_f32m8(slope_ptr, vl);

                        vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                        _p = vfmul_vv_f32m8_m(_lower, _p, /*op1*/ _p, _slope, vl);
                        vse32_v_f32m8(ptr, _p, vl);

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
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);

                    vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                    _p = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                    vse32_v_f32m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }

#else
    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope_data[i];
            }
        }
        else
        {
            float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
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
            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            for (int j = 0; j < w; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope;
            }
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
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

#endif

    return 0;
}

#if __riscv_vector && __riscv_zfh
//fp16s(a)
//hint: slope always store as fp32

int PReLU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr = bottom_top_blob;
        const float* ptr_slope = slope_data;
        if (num_slope > 1)
        {
            int n = w * elempack;

            // #pragma omp parallel for num_threads(opt.num_threads)
            while (n > 0)
            {
                size_t vl = vsetvl_e16m4(n);

                vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                vfloat32m8_t _slope = vle32_v_f32m8(ptr_slope, vl);
                vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                _p = vfmul_vv_f32m8_m(_lower, _p, /*op1*/ _p, _slope, vl);

                vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);
                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
        }
        else
        {
            float slope = slope_data[0];

            int n = w * elempack;
            // #pragma omp parallel for num_threads(opt.num_threads)
            while (n > 0)
            {
                size_t vl = vsetvl_e16m4(n);
                vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);

                _p = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

                ptr += vl;
                n -= vl;
            }
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
            if (num_slope > 1)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* ptr_slope = (const float*)slope_data + i * elempack;
                    int n = elempack;

                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m4(n);
                        vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                        vfloat32m8_t _slope = vle32_v_f32m8(ptr_slope, vl);

                        vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                        _p = vfmul_vv_f32m8_m(_lower, _p, /*op1*/ _p, _slope, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

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
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                    vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);

                    _p = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                    vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
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
            int n = size * elempack;

            if (num_slope > 1 && elempack != 1)
            {
                while (n > 0)
                {
                    int n1 = elempack;
                    const float* slope_ptr = (const float*)slope_data + q * elempack;
                    while (n1 > 0)
                    {
                        size_t vl = vsetvl_e16m4(n1);
                        vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                        vfloat32m8_t _slope = vle32_v_f32m8(slope_ptr, vl);

                        vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                        _p = vfmul_vv_f32m8_m(_lower, _p, /*op1*/ _p, _slope, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

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
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);

                    vbool4_t _lower = vmflt_vf_f32m8_b4(_p, .0f, vl);
                    _p = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                    vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }

    return 0;
}

int PReLU_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr = bottom_top_blob;
        const float* ptr_slope = slope_data;
        if (num_slope > 1)
        {
            int n = w * elempack;

            // #pragma omp parallel for num_threads(opt.num_threads)
            while (n > 0)
            {
                size_t vl = vsetvl_e16m4(n);
                vfloat16m4_t _p = vle16_v_f16m4(ptr, vl);
                vfloat16m4_t _slope = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_slope, vl), vl);
                vbool4_t _lower = vmflt_vf_f16m4_b4(_p, .0f, vl);

                _p = vfmul_vv_f16m4_m(_lower, _p, /*op1*/ _p, _slope, vl);
                vse16_v_f16m4(ptr, _p, vl);

                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
        }
        else
        {
            __fp16 slope = slope_data[0];

            int n = w * elempack;
            // #pragma omp parallel for num_threads(opt.num_threads)
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vbool2_t _lower = vmflt_vf_f16m8_b2(_p, .0f, vl);

                _p = vfmul_vf_f16m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                vse16_v_f16m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
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
            if (num_slope > 1)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* ptr_slope = (const float*)slope_data + i * elempack;
                    int n = elempack;

                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m4(n);
                        vfloat16m4_t _p = vle16_v_f16m4(ptr, vl);
                        vfloat16m4_t _slope = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_slope, vl), vl);

                        vbool4_t _lower = vmflt_vf_f16m4_b4(_p, .0f, vl);
                        _p = vfmul_vv_f16m4_m(_lower, _p, /*op1*/ _p, _slope, vl);
                        vse16_v_f16m4(ptr, _p, vl);

                        ptr += vl;
                        ptr_slope += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                __fp16 slope = slope_data[0];
                int n = w * elempack;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vbool2_t _lower = vmflt_vf_f16m8_b2(_p, .0f, vl);

                    _p = vfmul_vf_f16m8_m(_lower, _p, /*op1*/ _p, slope, vl);
                    vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
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
            int n = size * elempack;

            if (num_slope > 1 && elempack != 1)
            {
                while (n > 0)
                {
                    int n1 = elempack;
                    const float* slope_ptr = (const float*)slope_data + q * elempack;
                    while (n1 > 0)
                    {
                        size_t vl = vsetvl_e16m4(n1);
                        vfloat16m4_t _p = vle16_v_f16m4(ptr, vl);
                        vfloat16m4_t _slope = vfncvt_f_f_w_f16m4(vle32_v_f32m8(slope_ptr, vl), vl);

                        vbool4_t _lower = vmflt_vf_f16m4_b4(_p, .0f, vl);
                        _p = vfmul_vv_f16m4_m(_lower, _p, /*op1*/ _p, _slope, vl);
                        vse16_v_f16m4(ptr, _p, vl);

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
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);

                    vbool2_t _lower = vmflt_vf_f16m8_b2(_p, .0f, vl);
                    _p = vfmul_vf_f16m8_m(_lower, _p, /*op1*/ _p, (__fp16)slope, vl);
                    vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }

    return 0;
}

#endif
} // namespace ncnn
