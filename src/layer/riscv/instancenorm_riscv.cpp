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

#include "instancenorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {
InstanceNorm_riscv::InstanceNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int InstanceNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
// x = (x - mean) / (sqrt(var + eps)) * gamma + beta
#if __riscv_vector
    int elembits = bottom_top_blob.elembits();
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
    int elempack = bottom_top_blob.elempack;
#endif // __riscv_vector
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    int dims = bottom_top_blob.dims;
#if __riscv_vector
    if (elempack == 1)
#endif // __riscv_vector
    {
#if __riscv_vector
        size = elempack * size;
#endif
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            // mean and var
            float sum = 0.f;
            float sqsum = 0.f;
#if __riscv_vector
            vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
            vfloat32m1_t _sqsum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
            {
                int n = size;
                float* ptr_sum = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_sum, vl);
                    _sum = vfredusum_vs_f32m8_f32m1(_sum, _p, /* scalar */ _sum, vl);
                    // _sqsum = vfredosum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
                    ptr_sum += vl;
                    n -= vl;
                }
            }
            sum = vfmv_f_s_f32m1_f32(_sum);
#else
            for (int i = 0; i < size; i++)
            {
                sum += ptr[i];
                //sqsum += ptr[i] * ptr[i];
            }
#endif // __riscv_vector
            float mean = sum / size;
#if __riscv_vecotr
            {
                int n = size;
                float* ptr_sqsum = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_sqsum, vl);
                    _p = vfsub_vf_f32m8(_p, mean, vl);
                    _sqsum = vfredosum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
                    n -= vl;
                    ptr_sqsum += vl;
                }
            }
            sqsum = vfmv_f_s_f32m1_f32(_sqsum);
#else
            float tmp = 0.f;
            for (int i = 0; i < size; i++)
            {
                tmp = ptr[i] - mean;
                sqsum += tmp * tmp;
            }
#endif // __riscv_vector
            float var = sqsum / size;
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            float a;
            float b;
            if (affine)
            {
                float gamma = gamma_data[q];
                float beta = beta_data[q];

                a = gamma / (sqrtf(var + eps));
                b = -mean * a + beta;
            }
            else
            {
                a = 1.f / (sqrtf(var + eps));
                b = -mean * a;
            }
#if __riscv_vector
            {
                int n = size;
                float* ptr_store = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_store, vl);
                    _p = vfmul_vf_f32m8(_p, a, vl);
                    _p = vfadd_vf_f32m8(_p, b, vl);
                    vse32_v_f32m8(ptr_store, _p, vl);
                    n -= vl;
                    ptr_store += vl;
                }
            }
#else
            for (int i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * a + b;
            }
#endif // __riscv_vector
        }
        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        const size_t vl = vsetvl_e32m1(packn);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sqsum = vfmv_v_f_f32m1(0.f, vl);

            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = vle32_v_f32m1(ptr + vl * i, vl);
                _sum = vfadd_vv_f32m1(_p, _sum, vl);
                // _sqsum = vfmadd_vv_f32m1(_p,_p,_sqsum,vl);
            }
            vfloat32m1_t _mean = vfdiv_vf_f32m1(_sum, size, vl);
            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = vle32_v_f32m1(ptr + vl * i, vl);
                _p = vfsub_vv_f32m1(_p, _mean, vl);
                _sqsum = vfmadd_vv_f32m1(_p, _p, _sqsum, vl);
            }
            vfloat32m1_t _var = vfdiv_vf_f32m1(_sqsum, size, vl);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            vfloat32m1_t _a;
            vfloat32m1_t _b;
            if (affine)
            {
                vfloat32m1_t _gamma = vle32_v_f32m1((const float*)gamma_data + q * vl, vl);
                vfloat32m1_t _beta = vle32_v_f32m1((const float*)beta_data + q * vl, vl);
                _a = vfdiv_vv_f32m1(_gamma, vfsqrt_v_f32m1(vfadd_vf_f32m1(_var, eps, vl), vl), vl);
                _b = vfnmsub_vv_f32m1(_a, _mean, _beta, vl);
            }
            else
            {
                _a = vfrdiv_vf_f32m1(vfsqrt_v_f32m1(vfadd_vf_f32m1(_var, eps, vl), vl), 1.f, vl);
                _b = vfmul_vv_f32m1(_a, _mean, vl);
                _b = vfsgnjn_vv_f32m1(_b, _b, vl);
            }
            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = vle32_v_f32m1(ptr + i * vl, vl);
                _p = vfmadd_vv_f32m1(_p, _a, _b, vl);
                vse32_v_f32m1(ptr + i * vl, _p, vl);
            }
        }
        return 0;
    }
#endif // __riscv_vector
    return 0;
}

#if __riscv_vector && __riscv_zfh
int InstanceNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int elempack = bottom_top_blob.elempack;

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    int dims = bottom_top_blob.dims;
    if (elempack == 1)
    {
        size = elempack * size;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            // mean and var
            float sum = 0.f;
            float sqsum = 0.f;
            vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
            vfloat32m1_t _sqsum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
            {
                int n = size;
                __fp16* ptr_sum = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_sum, vl), vl);
                    _sum = vfredusum_vs_f32m8_f32m1(_sum, _p, /* scalar */ _sum, vl);
                    // _sqsum = vfredosum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
                    ptr_sum += vl;
                    n -= vl;
                }
            }
            sum = vfmv_f_s_f32m1_f32(_sum);
            float mean = sum / size;
            {
                int n = size;
                __fp16* ptr_sqsum = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_sqsum, vl), vl);
                    _p = vfsub_vf_f32m8(_p, mean, vl);
                    _sqsum = vfredosum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
                    n -= vl;
                    ptr_sqsum += vl;
                }
            }
            sqsum = vfmv_f_s_f32m1_f32(_sqsum);
            float var = sqsum / size;
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            float a;
            float b;
            if (affine)
            {
                float gamma = gamma_data[q];
                float beta = beta_data[q];

                a = gamma / (sqrtf(var + eps));
                b = -mean * a + beta;
            }
            else
            {
                a = 1.f / (sqrtf(var + eps));
                b = -mean * a;
            }
            {
                int n = size;
                __fp16* ptr_store = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_store, vl), vl);
                    _p = vfmul_vf_f32m8(_p, a, vl);
                    _p = vfadd_vf_f32m8(_p, b, vl);
                    vse16_v_f16m4(ptr_store, vfncvt_f_f_w_f16m4(_p, vl), vl);
                    n -= vl;
                    ptr_store += vl;
                }
            }
        }
        return 0;
    }

    const int packn = csrr_vlenb() / 2;
    if (elempack == packn)
    {
        const size_t vl = vsetvl_e16m1(packn);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);
            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);
            vfloat32m2_t _sqsum = vfmv_v_f_f32m2(0.f, vl);

            for (int i = 0; i < size; i++)
            {
                vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + vl * i, vl), vl);
                _sum = vfadd_vv_f32m2(_p, _sum, vl);
                // _sqsum = vfmadd_vv_f32m2(_p,_p,_sqsum,vl);
            }
            vfloat32m2_t _mean = vfdiv_vf_f32m2(_sum, size, vl);
            for (int i = 0; i < size; i++)
            {
                vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + vl * i, vl), vl);
                _p = vfsub_vv_f32m2(_p, _mean, vl);
                _sqsum = vfmadd_vv_f32m2(_p, _p, _sqsum, vl);
            }
            vfloat32m2_t _var = vfdiv_vf_f32m2(_sqsum, size, vl);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            vfloat32m2_t _a;
            vfloat32m2_t _b;
            if (affine)
            {
                vfloat32m2_t _gamma = vle32_v_f32m2((const float*)gamma_data + q * vl, vl);
                vfloat32m2_t _beta = vle32_v_f32m2((const float*)beta_data + q * vl, vl);
                _a = vfdiv_vv_f32m2(_gamma, vfsqrt_v_f32m2(vfadd_vf_f32m2(_var, eps, vl), vl), vl);
                _b = vfnmsub_vv_f32m2(_a, _mean, _beta, vl);
            }
            else
            {
                _a = vfrdiv_vf_f32m2(vfsqrt_v_f32m2(vfadd_vf_f32m2(_var, eps, vl), vl), 1.f, vl);
                _b = vfmul_vv_f32m2(_a, _mean, vl);
                _b = vfsgnjn_vv_f32m2(_b, _b, vl);
            }
            for (int i = 0; i < size; i++)
            {
                vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + i * vl, vl), vl);
                _p = vfmadd_vv_f32m2(_p, _a, _b, vl);
                vse16_v_f16m1(ptr + i * vl, vfncvt_f_f_w_f16m1(_p, vl), vl);
            }
        }
        return 0;
    }
    return 0;
}

int InstanceNorm_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta
    int elempack = bottom_top_blob.elempack;

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    int dims = bottom_top_blob.dims;
    if (elempack == 1)
    {
        size = elempack * size;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            // mean and var
            __fp16 sum = 0.f;
            __fp16 sqsum = 0.f;
            vfloat16m1_t _sum = vfmv_s_f_f16m1(vundefined_f16m1(), 0.f, vsetvlmax_e32m1());
            vfloat16m1_t _sqsum = vfmv_s_f_f16m1(vundefined_f16m1(), 0.f, vsetvlmax_e32m1());
            {
                int n = size;
                __fp16* ptr_sum = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr_sum, vl);
                    _sum = vfredusum_vs_f16m8_f16m1(_sum, _p, /* scalar */ _sum, vl);
                    // _sqsum = vfredosum_vs_f16m8_f16m1(_sqsum, vfmul_vv_f16m8(_p, _p, vl), /* scalar */ _sqsum, vl);
                    ptr_sum += vl;
                    n -= vl;
                }
            }
            sum = vfmv_f_s_f16m1_f16(_sum);
            __fp16 mean = sum / size;
            {
                int n = size;
                __fp16* ptr_sqsum = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr_sqsum, vl);
                    _p = vfsub_vf_f16m8(_p, mean, vl);
                    _sqsum = vfredosum_vs_f16m8_f16m1(_sqsum, vfmul_vv_f16m8(_p, _p, vl), /* scalar */ _sqsum, vl);
                    n -= vl;
                    ptr_sqsum += vl;
                }
            }
            sqsum = vfmv_f_s_f16m1_f16(_sqsum);
            __fp16 var = sqsum / size;
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            __fp16 a;
            __fp16 b;
            if (affine)
            {
                float gamma = gamma_data[q];
                float beta = beta_data[q];

                a = static_cast<__fp16>(gamma / (sqrt(var + eps)));
                b = static_cast<__fp16>(-mean * a + beta);
            }
            else
            {
                a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
                b = static_cast<__fp16>(-mean * a);
            }
            {
                int n = size;
                __fp16* ptr_store = ptr;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr_store, vl);
                    _p = vfmul_vf_f16m8(_p, a, vl);
                    _p = vfadd_vf_f16m8(_p, b, vl);
                    vse16_v_f16m8(ptr_store, _p, vl);
                    n -= vl;
                    ptr_store += vl;
                }
            }
        }
        return 0;
    }

    const int packn = csrr_vlenb() / 2;
    if (elempack == packn)
    {
        const size_t vl = vsetvl_e16m1(packn);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);
            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sqsum = vfmv_v_f_f16m1(0.f, vl);

            for (int i = 0; i < size; i++)
            {
                vfloat16m1_t _p = vle16_v_f16m1(ptr + vl * i, vl);
                _sum = vfadd_vv_f16m1(_p, _sum, vl);
                // _sqsum = vfmadd_vv_f16m1(_p,_p,_sqsum,vl);
            }
            vfloat16m1_t _mean = vfdiv_vf_f16m1(_sum, size, vl);
            for (int i = 0; i < size; i++)
            {
                vfloat16m1_t _p = vle16_v_f16m1(ptr + vl * i, vl);
                _p = vfsub_vv_f16m1(_p, _mean, vl);
                _sqsum = vfmadd_vv_f16m1(_p, _p, _sqsum, vl);
            }
            vfloat16m1_t _var = vfdiv_vf_f16m1(_sqsum, size, vl);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            vfloat16m1_t _a;
            vfloat16m1_t _b;
            if (affine)
            {
                vfloat16m1_t _gamma = vfncvt_f_f_w_f16m1(vle32_v_f32m2((const float*)gamma_data + q * vl, vl), vl);
                vfloat16m1_t _beta = vfncvt_f_f_w_f16m1(vle32_v_f32m2((const float*)beta_data + q * vl, vl), vl);
                _a = vfdiv_vv_f16m1(_gamma, vfsqrt_v_f16m1(vfadd_vf_f16m1(_var, eps, vl), vl), vl);
                _b = vfnmsub_vv_f16m1(_a, _mean, _beta, vl);
            }
            else
            {
                _a = vfrdiv_vf_f16m1(vfsqrt_v_f16m1(vfadd_vf_f16m1(_var, eps, vl), vl), 1.f, vl);
                _b = vfmul_vv_f16m1(_a, _mean, vl);
                _b = vfsgnjn_vv_f16m1(_b, _b, vl);
            }
            for (int i = 0; i < size; i++)
            {
                vfloat16m1_t _p = vle16_v_f16m1(ptr + i * vl, vl);
                _p = vfmadd_vv_f16m1(_p, _a, _b, vl);
                vse16_v_f16m1(ptr + i * vl, _p, vl);
            }
        }
        return 0;
    }
    return 0;
}

#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn