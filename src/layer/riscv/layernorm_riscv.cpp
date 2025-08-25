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
#endif // __riscv_vector

static int layernorm(float* ptr, const float* gamma_data, const float* beta_data, float eps, int elementcount, int elementpack)
{
    float mean = 0.f;
    float var = 0.f;

    size_t size = elementcount * elementpack;
    int remain = elementcount;
#if __riscv_vector
    size_t vl_max = __riscv_vsetvlmax_e32m8();
    remain = elementcount % vl_max;
#endif

    int i = 0;
    float* ptr_sum = ptr;
#if __riscv_vector
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl_max);
    for (; i + vl_max - 1 < size; i += vl_max)
    {
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl_max);
        _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl_max);
        ptr_sum += vl_max;
    }

    if (i < size)
    {
        size_t vlr = __riscv_vsetvl_e32m8(remain);
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vlr);
#if __riscv_xtheadvector
        _p = reset_tails(_p, vlr, 0.f);
        _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl_max);
#else
        _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        i += remain;
    }

    vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vl_max);
    mean += __riscv_vfmv_f_s_f32m1_f32(_sum0);
#endif // __riscv_vector
    for (; i < size; i++) mean += *ptr_sum++;
    mean /= elementcount;

    i = 0;
    ptr_sum = ptr;
#if __riscv_vector
    vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vl_max);
    for (; i + vl_max - 1 < size; i += vl_max)
    {
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl_max);
        vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, vl_max);
        _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, vl_max);
        _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl_max);

        ptr_sum += vl_max;
    }

    if (i < size)
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
        i += remain;
    }

    vfloat32m1_t _sqsum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vl_max);
    var += __riscv_vfmv_f_s_f32m1_f32(_sqsum0);
#endif // __riscv_vector
    for (; i < size; i++)
    {
        float tmp = *ptr_sum++ - mean;
        var += tmp * tmp;
    }

    var /= elementcount;
    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    int n = size;
    float* ptr_store = ptr;
    if (gamma_data && beta_data)
    {
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
#if __riscv_vector
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_store, vl);
            _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, b, vl);

            vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vl);
            vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vl);
            _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
            __riscv_vse32_v_f32m8(ptr_store, _p, vl);

            n -= vl;
            ptr_store += vl;
            ptr_gamma += vl;
            ptr_beta += vl;
        }

#endif // __riscv_vector
        while (n-- > 0) *ptr_store++ = (*ptr_store * a + b) * *ptr_gamma++ + *ptr_beta++;
    }
    else
    {
#if __riscv_vector
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
#endif // __riscv_vector
        while (n-- > 0) *ptr_store++ = (*ptr_store * a + b);
    }
    return 0;
}

#if __riscv_vector
static inline int layernorm_rvv_packn_procedure(int size, float* ptr, const float* gamma_data, const float* beta_data, float eps, const size_t vl)
{
    vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
    vfloat32m1_t _sqsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + vl * i, vl);
        _sum = __riscv_vfadd_vv_f32m1(_p, _sum, vl);
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
    if (gamma_data && beta_data)
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
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        return layernorm(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif // __riscv_vector
    if (dims == 2)
    {
        // assert affine_size == w
#if __riscv_vector
        if (elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                layernorm_rvv_packn_procedure(w, ptr, gamma_data, beta_data, eps, vl);
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
#if __riscv_vector
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);
                        layernorm_rvv_packn_procedure(w, ptr, gamma_data, beta_data, eps, vl);
                    }
                }
            }
#endif // __riscv_vector
            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);
                        layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
                    }
                }
            }
        }
        else // if (affine_size == size)
        {
#if __riscv_vector
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_procedure(w * h, ptr, gamma_data, beta_data, eps, vl);
                }
            }
#endif // __riscv_vector
            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    layernorm(ptr, gamma_data, beta_data, eps, w * h, elempack);
                }
            }
        }
    }
    return 0;
}
} // namespace ncnn