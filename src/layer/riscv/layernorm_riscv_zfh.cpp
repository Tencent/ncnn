// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
static void layernorm_fp16s(__fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int elemcount, int elempack)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2; // fp16
    size_t vl = (elempack == packn) ? __riscv_vsetvl_e16m1(packn) : __riscv_vsetvlmax_e16m4();
#endif // __riscv_vector
    float mean = 0.f;
    float var = 0.f;
    float a = 0.f;
    float b = 0.f;

    size_t size = elemcount * elempack;
#if __riscv_vector
    vfloat32m2_t _sum1, _sqsum1;
    vfloat32m2_t _a, _b;
#endif // __riscv_vector

    int i = 0;
    __fp16* ptr_sum = ptr;

#if __riscv_vector
    if (elempack != 1)
    {
        _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        for (; i < elemcount; i++) _sum1 = __riscv_vfadd_vv_f32m2(_sum1, __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl), vl);
        _sum1 = __riscv_vfdiv_vf_f32m2(_sum1, elemcount, vl);
    }
#endif // __riscv_vector

    if (elempack == 1)
    {
#if __riscv_vector
        vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
        for (; i + vl - 1 < size; i += vl)
        {
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vl), vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            ptr_sum += vl;
        }

        vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vl);
        mean += __riscv_vfmv_f_s_f32m1_f32(_sum0);
#endif // __riscv_vector
        for (; i < size; i++) mean += (float)*ptr_sum++;
        mean /= elemcount;
    }

    i = 0;
    ptr_sum = ptr;
#if __riscv_vector
    if (elempack != 1)
    {
        _sqsum1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
            _p = __riscv_vfsub_vv_f32m2(_p, _sum1, vl);
            _sqsum1 = __riscv_vfmacc_vv_f32m2(_sqsum1, _p, _p, vl);
        }
        vfloat32m2_t _var = __riscv_vfdiv_vf_f32m2(_sqsum1, elemcount, vl);
        _a = __riscv_vfrdiv_vf_f32m2(__riscv_vfsqrt_v_f32m2(__riscv_vfadd_vf_f32m2(_var, eps, vl), vl), 1.f, vl);
        _b = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(_sum1, _sum1, vl), _a, vl);
    }
#endif // __riscv_vector

    if (elempack == 1)
    {
#if __riscv_vector
        vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vl);
        for (; i + vl - 1 < size; i += vl)
        {
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vl), vl);
            vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, vl);
            _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, vl);
            _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl);

            ptr_sum += vl;
        }

        vfloat32m1_t _sqsum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vl);
        var += __riscv_vfmv_f_s_f32m1_f32(_sqsum0);
#endif // __riscv_vector
        for (; i < size; i++)
        {
            float tmp = (float)*ptr_sum++ - mean;
            var += tmp * tmp;
        }

        var /= elemcount;
        a = static_cast<float>(1.f / (sqrt(var + eps)));
        b = -mean * a;
    }

    i = 0;
    __fp16* ptr_store = ptr;
    if (gamma_data && beta_data)
    {
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
#if __riscv_vector
        if (elempack != 1)
        {
            for (; i < elemcount; i++)
            {
                vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr_store, vl), vl);
                _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
                _p = __riscv_vfmul_vf_f32m2(_p, *ptr_gamma++, vl);
                _p = __riscv_vfadd_vf_f32m2(_p, *ptr_beta++, vl);
                __riscv_vse16_v_f16m1(ptr_store, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
                ptr_store += vl;
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
#if __riscv_vector
            for (; i + vl - 1 < size; i += vl)
            {
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vl), vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);

                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vl);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
#endif // __riscv_vector
            for (; i < size; i++) *ptr_store++ = (__fp16)((float)*ptr_store * a + b) * *ptr_gamma++ + *ptr_beta++;
        }
    }
    else
    {
#if __riscv_vector
        if (elempack != 1)
        {
            for (; i < elemcount; i++)
            {
                vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr_store, vl), vl);
                _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
                __riscv_vse16_v_f16m1(ptr_store, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
                ptr_store += vl;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
#if __riscv_vector
            for (; i + vl - 1 < size; i += vl)
            {
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vl), vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);
                ptr_store += vl;
            }
#endif // __riscv_vector
            for (; i < size; i++) *ptr_store++ = (__fp16)((float)*ptr_store * a + b);
        }
    }
}

int LayerNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int elempack = bottom_top_blob.elempack;
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
        layernorm_fp16s(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            layernorm_fp16s(ptr, gamma_data, beta_data, eps, w, elempack);
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
                    __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
                    layernorm_fp16s(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                layernorm_fp16s(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
