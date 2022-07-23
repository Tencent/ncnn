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

#include <math.h>

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {
InstanceNorm_riscv::InstanceNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int InstanceNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
// x = (x - mean) / (sqrt(var + eps)) * gamma + beta
#if __riscv_vector
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
            float* ptr = bottom_top_blob.channel(q);

            // mean and var
            float sum = 0.f;
            float sqsum = 0.f;
            vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
            vfloat32m1_t _sqsum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
            {
                int n = size;
                float* ptr_sum = ptr;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_sum, vl);
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
                float* ptr_sqsum = ptr;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_sqsum, vl);
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

                a = static_cast<float>(gamma / (sqrt(var + eps)));
                b = -mean * a + beta;
            }
            else
            {
                a = static_cast<float>(1.f / (sqrt(var + eps)));
                b = -mean * a;
            }
            {
                int n = size;
                float* ptr_store = ptr;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_store, vl);
                    _p = vfmul_vf_f32m8(_p, a, vl);
                    _p = vfadd_vf_f32m8(_p, b, vl);
                    vse32_v_f32m8(ptr_store, _p, vl);
                    n -= vl;
                    ptr_store += vl;
                }
            }
        }
        return 0;
    }

    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        const word_type vl = vsetvl_e32m1(packn);
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
#else
    return InstanceNorm::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
    return 0;
}

} // namespace ncnn