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
#include "layernorm_riscv.h"
#include <math.h>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

#if __riscv_vector && __riscv_zfh
#include "layernorm_rvv_fp16.h"
#endif

LayerNorm_riscv::LayerNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

#if __riscv_vector
static inline int layernorm_rvv_pack1_procedure(int size, float* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
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
            // _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
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
            _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            n -= vl;
            ptr_sqsum += vl;
        }
    }
    sqsum = vfmv_f_s_f32m1_f32(_sqsum);
    float var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    {
        int n = size;
        float* ptr_store = ptr;
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
        if (affine)
        {
            while (n > 0)
            {
                word_type vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr_store, vl);
                _p = vfmul_vf_f32m8(_p, a, vl);
                vfloat32m8_t _gamma = vle32_v_f32m8(ptr_gamma, vl);
                _p = vfadd_vf_f32m8(_p, b, vl);
                vfloat32m8_t _beta = vle32_v_f32m8(ptr_beta, vl);
                _p = vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                vse32_v_f32m8(ptr_store, _p, vl);

                n -= vl;
                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
        }
        else
        {
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

static inline int layernorm_rvv_packn_procedure(int size, float* ptr, const float* gamma_data, const float* beta_data, float eps, int affine, const word_type vl)
{
    // mean and var
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
        _sqsum = vfmacc_vv_f32m1(_sqsum, _p, _p, vl);
    }
    vfloat32m1_t _var = vfdiv_vf_f32m1(_sqsum, size, vl);

    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    vfloat32m1_t _a = vfrdiv_vf_f32m1(vfsqrt_v_f32m1(vfadd_vf_f32m1(_var, eps, vl), vl), 1.f, vl);
    // how about vfrsqrt7.v?
    vfloat32m1_t _b = vfmul_vv_f32m1(vfsgnjn_vv_f32m1(_mean, _mean, vl), _a, vl);
    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m1_t _p = vle32_v_f32m1(ptr + offset, vl);
            _p = vfmadd_vv_f32m1(_p, _a, _b, vl);
            _p = vfmul_vf_f32m1(_p, gamma_data[i], vl);
            _p = vfadd_vf_f32m1(_p, beta_data[i], vl);
            vse32_v_f32m1(ptr + offset, _p, vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m1_t _p = vle32_v_f32m1(ptr + offset, vl);
            _p = vfmadd_vv_f32m1(_p, _a, _b, vl);
            vse32_v_f32m1(ptr + offset, _p, vl);
        }
    }

    return 0;
}
#else
static inline int layernorm_scalar_procedure(int size, float* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    // mean and var
    float sum = 0.f;
    float sqsum = 0.f;
    for (int i = 0; i < size; i++)
    {
        sum += ptr[i];
        //sqsum += ptr[i] * ptr[i];
    }
    float mean = sum / size;
    float tmp = 0.f;
    for (int i = 0; i < size; i++)
    {
        tmp = ptr[i] - mean;
        sqsum += tmp * tmp;
    }
    float var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;

    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a + b;
        }
    }
    return 0;
}
#endif // __riscv_vector

int LayerNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
// x = (x - mean) / sqrt(var + eps) * gamma + beta
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
#endif // __riscv_zfh

    int elempack = bottom_top_blob.elempack;
    const int packn = csrr_vlenb() / 4;
#endif // __riscv_vector
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
#if __riscv_vector
        return layernorm_rvv_pack1_procedure(w * elempack, ptr, gamma_data, beta_data, eps, affine);
#else
        return layernorm_scalar_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#endif // __riscv_vector
    }
#if __riscv_vector
    if (elempack == 1)
#endif
    {
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
#if __riscv_vector
                layernorm_rvv_pack1_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#else
                layernorm_scalar_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#endif // __riscv_vector
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);
#if __riscv_vector
                        layernorm_rvv_pack1_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#else
                        layernorm_scalar_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#endif // __riscv_vector
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
                    layernorm_rvv_pack1_procedure(size, ptr, gamma_data, beta_data, eps, affine);
#else
                    layernorm_scalar_procedure(size, ptr, gamma_data, beta_data, eps, affine);
#endif // __riscv_vector
                }
            }
        }
    }

#if __riscv_vector
    if (elempack == packn)
    {
        const word_type vl = vsetvl_e32m1(packn);
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                layernorm_rvv_packn_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
            }
        }
        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);

                        layernorm_rvv_packn_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_procedure(size, ptr, gamma_data, beta_data, eps, affine, vl);
                }
            }
        }
    }
#endif // __riscv_vector
    return 0;
}

#if __riscv_vector && __riscv_zfh
int LayerNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / sqrt(var + eps) * gamma + beta
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;

        return layernorm_rvv_pack1_fp16s_procedure(w * elempack, ptr, gamma_data, beta_data, eps, affine);
    }
    if (elempack == 1)
    {
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_pack1_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);

                        layernorm_rvv_pack1_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_pack1_fp16s_procedure(size, ptr, gamma_data, beta_data, eps, affine);
                }
            }
        }

        return 0;
    }

    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        const word_type vl = vsetvl_e16m1(packn);
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_packn_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
            }
        }
        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);

                        layernorm_rvv_packn_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_fp16s_procedure(size, ptr, gamma_data, beta_data, eps, affine, vl);
                }
            }
        }
    }

    return 0;
}

int LayerNorm_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / sqrt(var + eps) * gamma + beta
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;

        return layernorm_rvv_pack1_fp16sa_procedure(w * elempack, ptr, gamma_data, beta_data, eps, affine);
    }
    if (elempack == 1)
    {
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_pack1_fp16sa_procedure(w, ptr, gamma_data, beta_data, eps, affine);
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);

                        layernorm_rvv_pack1_fp16sa_procedure(w, ptr, gamma_data, beta_data, eps, affine);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_pack1_fp16sa_procedure(size, ptr, gamma_data, beta_data, eps, affine);
                }
            }
        }

        return 0;
    }

    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        const word_type vl = vsetvl_e16m1(packn);
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_packn_fp16sa_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
            }
        }
        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);

                        layernorm_rvv_packn_fp16sa_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_fp16sa_procedure(size, ptr, gamma_data, beta_data, eps, affine, vl);
                }
            }
        }
    }

    return 0;
}
#endif

} // namespace ncnn