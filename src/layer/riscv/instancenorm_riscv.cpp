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

#include "cpu.h"

namespace ncnn {
InstanceNorm_riscv::InstanceNorm_riscv()
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

int InstanceNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int elempack = bottom_top_blob.elempack;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    int dims = bottom_top_blob.dims;
    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            // mean and var
            float sum = 0.f;
            float sqsum = 0.f;
#if __riscv_vector && !defined(C906)
            vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            vfloat32m1_t _sqsum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            {
                int n = size;
                float* ptr_sum = ptr;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl);
                    _sum = __riscv_vfredusum_vs_f32m8_f32m1(_p, _sum, vl);
                    // _sqsum = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_p, _p, vl), _sqsum, vl);
                    ptr_sum += vl;
                    n -= vl;
                }
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(_sum);
#else
            for (int i = 0; i < size; i++)
            {
                sum += ptr[i];
                //sqsum += ptr[i] * ptr[i];
            }
#endif // __riscv_vector
            float mean = sum / size;
#if __riscv_vector && !defined(C906)
            {
                int n = size;
                float* ptr_sqsum = ptr;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sqsum, vl);
                    _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                    _sqsum = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_p, _p, vl), _sqsum, vl);
                    n -= vl;
                    ptr_sqsum += vl;
                }
            }
            sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum);
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
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_store, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                    __riscv_vse32_v_f32m8(ptr_store, _p, vl);
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
        const size_t vl = __riscv_vsetvl_e32m1(packn);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sqsum = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + vl * i, vl);
                _sum = __riscv_vfadd_vv_f32m1(_p, _sum, vl);
                // _sqsum = __riscv_vfmadd_vv_f32m1(_p,_p,_sqsum,vl);
            }
            vfloat32m1_t _mean = __riscv_vfdiv_vf_f32m1(_sum, size, vl);
            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + vl * i, vl);
                _p = __riscv_vfsub_vv_f32m1(_p, _mean, vl);
                _sqsum = __riscv_vfmadd_vv_f32m1(_p, _p, _sqsum, vl);
            }
            vfloat32m1_t _var = __riscv_vfdiv_vf_f32m1(_sqsum, size, vl);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            vfloat32m1_t _a;
            vfloat32m1_t _b;
            if (affine)
            {
                vfloat32m1_t _gamma = __riscv_vle32_v_f32m1((const float*)gamma_data + q * vl, vl);
                vfloat32m1_t _beta = __riscv_vle32_v_f32m1((const float*)beta_data + q * vl, vl);
                _a = __riscv_vfdiv_vv_f32m1(_gamma, __riscv_vfsqrt_v_f32m1(__riscv_vfadd_vf_f32m1(_var, eps, vl), vl), vl);
                _b = __riscv_vfnmsub_vv_f32m1(_a, _mean, _beta, vl);
            }
            else
            {
                _a = __riscv_vfrdiv_vf_f32m1(__riscv_vfsqrt_v_f32m1(__riscv_vfadd_vf_f32m1(_var, eps, vl), vl), 1.f, vl);
                _b = __riscv_vfmul_vv_f32m1(_a, _mean, vl);
                _b = __riscv_vfsgnjn_vv_f32m1(_b, _b, vl);
            }
            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i * vl, vl);
                _p = __riscv_vfmadd_vv_f32m1(_p, _a, _b, vl);
                __riscv_vse32_v_f32m1(ptr + i * vl, _p, vl);
            }
        }
        return 0;
    }
#endif // __riscv_vector
    return 0;
}

} // namespace ncnn
