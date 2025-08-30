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

static int layernorm(float* ptr, const float* gamma_data, const float* beta_data, float eps, int elementcount, int elementpack, size_t vl)
{
    float mean = 0.f;
    float var = 0.f;
    float a = 0.f;
    float b = 0.f;

    size_t size = elementcount * elementpack;
#if __riscv_vector
    vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t _sqsum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t _a, _b;
    vfloat32m8_t _sum, _sqsum;
#endif

    int i = 0;
    float* ptr_sum = ptr;

#if __riscv_vector
    if (elementpack != 1)
    {
        for (; i < elementcount; i++) _sum0 = __riscv_vfadd_vv_f32m1(_sum0, __riscv_vle32_v_f32m1(ptr + vl * i, vl), vl);
        _sum0 = __riscv_vfdiv_vf_f32m1(_sum0, elementcount, vl);
    }
#endif // __riscv_vector

    if (elementpack == 1)
    {
#if __riscv_vector
        _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
        for (; i + vl - 1 < size; i += vl)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            ptr_sum += vl;
        }

        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vl);
        mean += __riscv_vfmv_f_s_f32m1_f32(_sum0);
#endif // __riscv_vector
        for (; i < size; i++) mean += *ptr_sum++;
        mean /= elementcount;
    }

    i = 0;
    ptr_sum = ptr;

#if __riscv_vector
    if (elementpack != 1)
    {
        for (; i < elementcount; i++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + vl * i, vl);
            _p = __riscv_vfsub_vv_f32m1(_p, _sum0, vl);
            _sqsum0 = __riscv_vfmacc_vv_f32m1(_sqsum0, _p, _p, vl);
        }
        _sqsum0 = __riscv_vfdiv_vf_f32m1(_sqsum0, elementcount, vl);
        _a = __riscv_vfrdiv_vf_f32m1(__riscv_vfsqrt_v_f32m1(__riscv_vfadd_vf_f32m1(_sqsum0, eps, vl), vl), 1.f, vl);
        _b = __riscv_vfmul_vv_f32m1(__riscv_vfsgnjn_vv_f32m1(_sum0, _sum0, vl), _a, vl);
    }
#endif // __riscv_vector

    if (elementpack == 1)
    {
#if __riscv_vector
        _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vl);
        for (; i + vl - 1 < size; i += vl)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_sum, vl);
            vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, vl);
            _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, vl);
            _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl);

            ptr_sum += vl;
        }

        _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vl);
        var += __riscv_vfmv_f_s_f32m1_f32(_sqsum0);
#endif // __riscv_vector
        for (; i < size; i++)
        {
            float tmp = *ptr_sum++ - mean;
            var += tmp * tmp;
        }

        var /= elementcount;
        a = static_cast<float>(1.f / (sqrt(var + eps)));
        b = -mean * a;
    }

    i = 0;
    float* ptr_store = ptr;
    if (gamma_data && beta_data)
    {
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
#if __riscv_vector
        if (elementpack != 1)
        {
            for (; i < elementcount; i++)
            {
                const int offset = vl * i;
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + offset, vl);
                _p = __riscv_vfmadd_vv_f32m1(_p, _a, _b, vl);
                _p = __riscv_vfmul_vf_f32m1(_p, *ptr_gamma++, vl);
                _p = __riscv_vfadd_vf_f32m1(_p, *ptr_beta++, vl);
                __riscv_vse32_v_f32m1(ptr + offset, _p, vl);
            }
        }
#endif // __riscv_vector

        if (elementpack == 1)
        {
#if __riscv_vector
            for (; i + vl - 1 < size; i += vl)
            {
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_store, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);

                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vl);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                __riscv_vse32_v_f32m8(ptr_store, _p, vl);

                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
#endif // __riscv_vector
            for (; i < size; i++) *ptr_store++ = (*ptr_store * a + b) * *ptr_gamma++ + *ptr_beta++;
        }
    }
    else
    {
#if __riscv_vector
        if (elementpack != 1)
        {
            for (; i < elementcount; i++)
            {
                const int offset = vl * i;
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + offset, vl);
                _p = __riscv_vfmadd_vv_f32m1(_p, _a, _b, vl);
                __riscv_vse32_v_f32m1(ptr + offset, _p, vl);
            }
        }
#endif // __riscv_vector

        if (elementpack == 1)
        {
#if __riscv_vector
            for (; i + vl - 1 < size; i += vl)
            {
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr_store, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                __riscv_vse32_v_f32m8(ptr_store, _p, vl);
                ptr_store += vl;
            }
#endif // __riscv_vector
            for (; i < size; i++) *ptr_store++ = (*ptr_store * a + b);
        }
    }
    return 0;
}

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
    size_t vl = 1;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    vl = (elempack == packn) ? __riscv_vsetvl_e32m1(packn) : __riscv_vsetvlmax_e32m8();
#endif // __riscv_vector

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        return layernorm(ptr, gamma_data, beta_data, eps, w * elempack, 1, vl);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm(ptr, gamma_data, beta_data, eps, w, elempack, vl);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    layernorm(ptr, gamma_data, beta_data, eps, w, elempack, vl);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm(ptr, gamma_data, beta_data, eps, w * h, elempack, vl);
            }
        }
    }
    return 0;
}
} // namespace ncnn