// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
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

#include "batchnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

BatchNorm_riscv::BatchNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int BatchNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __riscv_vector
    int elembits = bottom_top_blob.elembits();

#if __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif
    int elempack = bottom_top_blob.elempack;
#endif // __riscv_vector
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
            size_t vl = vsetvl_e32m8(n);

            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _a = vle32_v_f32m8(ptr_a, vl);
            vfloat32m8_t _b = vle32_v_f32m8(ptr_b, vl);

            _p = vfmadd_vv_f32m8(_p, _b, _a, vl);

            vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            ptr_a += vl;
            ptr_b += vl;
            n -= vl;
        }
#else
        int w = bottom_top_blob.w;
        #pragma omp parallel for num_threads(opt.num_threads)
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
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    _p = vfmul_vf_f32m8(_p, b, vl);
                    _p = vfadd_vf_f32m8(_p, a, vl);
                    vse32_v_f32m8(ptr, _p, vl);

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
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    _p = vfmul_vf_f32m8(_p, b, vl);
                    _p = vfadd_vf_f32m8(_p, a, vl);
                    vse32_v_f32m8(ptr, _p, vl);

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

        const size_t vl = vsetvl_e32m1(packn);
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

                vfloat32m1_t _a = vle32_v_f32m1(ptr_a, vl);
                vfloat32m1_t _b = vle32_v_f32m1(ptr_b, vl);
                while (n > 0)
                {
                    vfloat32m1_t _p = vle32_v_f32m1(ptr, vl);
                    _p = vfmadd_vv_f32m1(_p, _b, _a, vl);
                    vse32_v_f32m1(ptr, _p, vl);

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

                vfloat32m1_t _a = vle32_v_f32m1(ptr_a, vl);
                vfloat32m1_t _b = vle32_v_f32m1(ptr_b, vl);

                int n = size;
                while (n > 0)
                {
                    vfloat32m1_t _p = vle32_v_f32m1(ptr, vl);
                    _p = vfmadd_vv_f32m1(_p, _b, _a, vl);
                    vse32_v_f32m1(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }
#endif
    return 0;
}

#if __riscv_vector && __riscv_zfh
int BatchNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    if (dims == 1)
    {
        int n = bottom_top_blob.w * elempack;
        __fp16* ptr = bottom_top_blob;
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        while (n > 0)
        {
            size_t vl = vsetvl_e16m4(n);

            vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
            vfloat32m8_t _a = vle32_v_f32m8(ptr_a, vl);
            vfloat32m8_t _b = vle32_v_f32m8(ptr_b, vl);

            _p = vfmadd_vv_f32m8(_p, _b, _a, vl);

            vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

            ptr += vl;
            ptr_a += vl;
            ptr_b += vl;
            n -= vl;
        }

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

                int n = w;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                    _p = vfmul_vf_f32m8(_p, b, vl);
                    _p = vfadd_vf_f32m8(_p, a, vl);
                    vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
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

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                    ;
                    _p = vfmul_vf_f32m8(_p, b, vl);
                    _p = vfadd_vf_f32m8(_p, a, vl);
                    vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }

        return 0;
    }

    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        const size_t vl = vsetvl_e16m1(packn);
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                const float* ptr_a = (const float*)a_data + i * elempack;
                const float* ptr_b = (const float*)b_data + i * elempack;
                int n = w * elempack;

                vfloat32m2_t _a = vle32_v_f32m2(ptr_a, vl);
                vfloat32m2_t _b = vle32_v_f32m2(ptr_b, vl);
                while (n > 0)
                {
                    vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr, vl), vl);
                    _p = vfmadd_vv_f32m2(_p, _b, _a, vl);
                    vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_p, vl), vl);

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

                vfloat32m2_t _a = vle32_v_f32m2(ptr_a, vl);
                vfloat32m2_t _b = vle32_v_f32m2(ptr_b, vl);

                int n = size;
                while (n > 0)
                {
                    vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr, vl), vl);
                    _p = vfmadd_vv_f32m2(_p, _b, _a, vl);
                    vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_p, vl), vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }

    return 0;
}

int BatchNorm_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    if (dims == 1)
    {
        int n = bottom_top_blob.w * elempack;
        __fp16* ptr = bottom_top_blob;
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        while (n > 0)
        {
            size_t vl = vsetvl_e16m4(n);

            vfloat16m4_t _p = vle16_v_f16m4(ptr, vl);
            vfloat16m4_t _a = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_a, vl), vl);
            vfloat16m4_t _b = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_b, vl), vl);

            _p = vfmadd_vv_f16m4(_p, _b, _a, vl);

            vse16_v_f16m4(ptr, _p, vl);

            ptr += vl;
            ptr_a += vl;
            ptr_b += vl;
            n -= vl;
        }

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

                int n = w;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    _p = vfmul_vf_f16m8(_p, b, vl);
                    _p = vfadd_vf_f16m8(_p, a, vl);
                    vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
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

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    ;
                    _p = vfmul_vf_f16m8(_p, b, vl);
                    _p = vfadd_vf_f16m8(_p, a, vl);
                    vse16_v_f16m8(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }

        return 0;
    }

    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        const size_t vl = vsetvl_e16m1(packn);
        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                const float* ptr_a = (const float*)a_data + i * elempack;
                const float* ptr_b = (const float*)b_data + i * elempack;
                int n = w * elempack;

                vfloat16m1_t _a = vfncvt_f_f_w_f16m1(vle32_v_f32m2(ptr_a, vl), vl);
                vfloat16m1_t _b = vfncvt_f_f_w_f16m1(vle32_v_f32m2(ptr_b, vl), vl);
                while (n > 0)
                {
                    vfloat16m1_t _p = vle16_v_f16m1(ptr, vl);
                    _p = vfmadd_vv_f16m1(_p, _b, _a, vl);
                    vse16_v_f16m1(ptr, _p, vl);

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

                vfloat16m1_t _a = vfncvt_f_f_w_f16m1(vle32_v_f32m2(ptr_a, vl), vl);
                vfloat16m1_t _b = vfncvt_f_f_w_f16m1(vle32_v_f32m2(ptr_b, vl), vl);

                int n = size;
                while (n > 0)
                {
                    vfloat16m1_t _p = vle16_v_f16m1(ptr, vl);
                    _p = vfmadd_vv_f16m1(_p, _b, _a, vl);
                    vse16_v_f16m1(ptr, _p, vl);

                    ptr += vl;
                    n -= vl;
                }
            }
        }
    }

    return 0;
}

#endif // __riscv_vector && __riscv_zfh
} // namespace ncnn
