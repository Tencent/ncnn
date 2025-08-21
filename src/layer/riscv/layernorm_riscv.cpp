// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"
#include <math.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

LayerNorm_riscv::LayerNorm_riscv()
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

#if __riscv_vector
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline))
#endif // __riscv_xtheadvector
static vfloat32m8_t
reset_tails(vfloat32m8_t x, size_t vl, float v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    vbool4_t _vl_mask = __riscv_vmsgeu_vx_u32m8_b4(__riscv_vid_v_u32m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f32m8(x, v, _vl_mask, vlm8);
    return x;
}

static inline int layernorm_rvv_pack1_procedure(int size, float* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    float mean = 0.f;
    float var = 0.f;
    size_t vl_max = __riscv_vsetvlmax_e32m8();

    {
        vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl_max);
        int n = size / vl_max * vl_max;
        float* ptr_sum = ptr;

        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl_max);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl_max);

            ptr_sum += vl_max;
            n -= vl_max;
        }

        int remain = size % vl_max;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vlr);
#if __riscv_xtheadvector
            _p = reset_tails(_p, vlr, 0.f);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl_max);
#else
            _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        }

        vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vl_max);
        mean = __riscv_vfmv_f_s_f32m1_f32(_sum0) / size;
    }

    {
        vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vl_max);
        int n = size / vl_max * vl_max;
        float* ptr_sum = ptr;

        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl_max);
            vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, vl_max);
            _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, vl_max);
            _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl_max);

            ptr_sum += vl_max;
            n -= vl_max;
        }

        int remain = size % vl_max;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vlr);
            vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, vlr);
            _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, vlr);

#if __riscv_xtheadvector
            _temp = reset_tails(_temp, vlr, 0.f);
            _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl_max);
#else
            _sqsum = __riscv_vfadd_vv_f32m8_tu(_sqsum, _sqsum, _temp, vlr);
#endif // __riscv_xtheadvector
        }

        vfloat32m1_t _sqsum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vl_max);
        var = __riscv_vfmv_f_s_f32m1_f32(_sqsum0) / size;
    }

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
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_store, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                __riscv_vse32_v_f32m8(ptr_store, _p, vl);

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
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_store, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                __riscv_vse32_v_f32m8(ptr_store, _p, vl);
                n -= vl;
                ptr_store += vl;
            }
        }
    }
    return 0;
}

static inline int layernorm_rvv_packn_procedure(int size, float* ptr, const float* gamma_data, const float* beta_data, float eps, int affine, const size_t vl)
{
    vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t _sqsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + vl * i, vl);
        _sum = __riscv_vfadd_vv_f32m1(_p, _sum, vl);
        // _sqsum = vfmadd_vv_f32m1(_p,_p,_sqsum,vl);
    }
    vfloat32m1_t _mean = __riscv_vfdiv_vf_f32m1(_sum, size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + vl * i, vl);
        _p = __riscv_vfsub_vv_f32m1(_p, _mean, vl);
        _sqsum = __riscv_vfmacc_vv_f32m1(_sqsum, _p, _p, vl);
    }
    vfloat32m1_t _var = __riscv_vfdiv_vf_f32m1(_sqsum, size, vl);
    vfloat32m1_t _a = __riscv_vfrdiv_vf_f32m1(__riscv_vfsqrt_v_f32m1(__riscv_vfadd_vf_f32m1(_var, eps, vl), vl), 1.f, vl);
    vfloat32m1_t _b = __riscv_vfmul_vv_f32m1(__riscv_vfsgnjn_vv_f32m1(_mean, _mean, vl), _a, vl);
    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + offset, vl);
            _p = __riscv_vfmadd_vv_f32m1(_p, _a, _b, vl);
            _p = __riscv_vfmul_vf_f32m1(_p, gamma_data[i], vl);
            _p = __riscv_vfadd_vf_f32m1(_p, beta_data[i], vl);
            __riscv_vse32_v_f32m1(ptr + offset, _p, vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + offset, vl);
            _p = __riscv_vfmadd_vv_f32m1(_p, _a, _b, vl);
            __riscv_vse32_v_f32m1(ptr + offset, _p, vl);
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
    for (int i = 0; i < size; i++) sum += ptr[i];

    float mean = sum / size;
    float tmp = 0.f;
    for (int i = 0; i < size; i++)
    {
        tmp = ptr[i] - mean;
        sqsum += tmp * tmp;
    }

    float var = sqsum / size;

    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    if (affine)
        for (int i = 0; i < size; i++) ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
    else
        for (int i = 0; i < size; i++) ptr[i] = ptr[i] * a + b;

    return 0;
}
#endif // __riscv_vector

int LayerNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
#endif // NCNN_ZFH

    int elempack = bottom_top_blob.elempack;
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
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e32m1(packn);
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
} // namespace ncnn