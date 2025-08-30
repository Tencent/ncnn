// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH

static int layernorm_fp16s(__fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int elementcount, int elementpack, size_t vl)
{
    float mean = 0.f;
    float var = 0.f;
    float a = 0.f;
    float b = 0.f;

    size_t size = elementcount * elementpack;
#if __riscv_vector
    vfloat32m2_t _sum1, _sqsum1;
    vfloat32m2_t _a, _b;
#endif // __riscv_vector

    int i = 0;
    __fp16* ptr_sum = ptr;

#if __riscv_vector
    if (elementpack != 1)
    {
        _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        for (; i < elementcount; i++) _sum1 = __riscv_vfadd_vv_f32m2(_sum1, __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl), vl);
        _sum1 = __riscv_vfdiv_vf_f32m2(_sum1, elementcount, vl);
    }
#endif // __riscv_vector

    if (elementpack == 1)
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
        mean /= elementcount;
    }

    i = 0;
    ptr_sum = ptr;
#if __riscv_vector
    if (elementpack != 1)
    {
        _sqsum1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
        for (int i = 0; i < elementcount; i++)
        {
            vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
            _p = __riscv_vfsub_vv_f32m2(_p, _sum1, vl);
            _sqsum1 = __riscv_vfmacc_vv_f32m2(_sqsum1, _p, _p, vl);
        }
        vfloat32m2_t _var = __riscv_vfdiv_vf_f32m2(_sqsum1, elementcount, vl);
        _a = __riscv_vfrdiv_vf_f32m2(__riscv_vfsqrt_v_f32m2(__riscv_vfadd_vf_f32m2(_var, eps, vl), vl), 1.f, vl);
        _b = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(_sum1, _sum1, vl), _a, vl);
    }
#endif // __riscv_vector

    if (elementpack == 1)
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

        var /= elementcount;
        a = static_cast<float>(1.f / (sqrt(var + eps)));
        b = -mean * a;
    }

    i = 0;
    int n = size;
    __fp16* ptr_store = ptr;
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
                vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + offset, vl), vl);
                _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
                _p = __riscv_vfmul_vf_f32m2(_p, *ptr_gamma++, vl);
                _p = __riscv_vfadd_vf_f32m2(_p, *ptr_beta++, vl);
                __riscv_vse16_v_f16m1(ptr + offset, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
            }
        }
#endif // __riscv_vector

        if (elementpack == 1)
        {
#if __riscv_vector
            while (n > 0)
            {
                size_t vlr = __riscv_vsetvl_e16m4(n);
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vlr), vlr);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vlr);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vlr);

                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vlr);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vlr);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vlr);
                __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vlr), vlr);

                n -= vlr;
                ptr_store += vlr;
                ptr_gamma += vlr;
                ptr_beta += vlr;
            }
#endif // __riscv_vector
            while (n-- > 0) *ptr_store++ = (__fp16)((float)*ptr_store * a + b) * *ptr_gamma++ + *ptr_beta++;
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
                vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + offset, vl), vl);
                _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
                __riscv_vse16_v_f16m1(ptr + offset, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
            }
        }
#endif // __riscv_vector
        if (elementpack == 1)
        {
#if __riscv_vector
            while (n > 0)
            {
                size_t vlr = __riscv_vsetvl_e16m4(n);
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vlr), vlr);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vlr);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vlr);
                __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vlr), vlr);
                n -= vlr;
                ptr_store += vlr;
            }
#endif // __riscv_vector
            while (n-- > 0) *ptr_store++ = (__fp16)((float)*ptr_store * a + b);
        }
    }
    return 0;
}

static int layernorm_fp16sa(__fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int elementcount, int elementpack, size_t vl)
{
    __fp16 mean = 0.f;
    __fp16 var = 0.f;
    __fp16 a = 0.f;
    __fp16 b = 0.f;

    size_t size = elementcount * elementpack;
#if __riscv_zvfh
    vfloat16m1_t _sum0, _sqsum0, _a, _b;
    vfloat16m8_t _sum, _sqsum;
#endif

    int i = 0;
    __fp16* ptr_sum = ptr;
#if __riscv_zvfh
    if (elementpack != 1)
    {
        _sum0 = __riscv_vfmv_v_f_f16m1(0.f, __riscv_vsetvlmax_e16m1());
        for (; i < elementcount; i++) _sum0 = __riscv_vfadd_vv_f16m1(_sum0, __riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
        _sum0 = __riscv_vfdiv_vf_f16m1(_sum0, elementcount, vl);
    }
#endif // __riscv_zvfh

    if (elementpack == 1)
    {
#if __riscv_zvfh
        _sum = __riscv_vfmv_v_f_f16m8(0.f, vl);
        for (; i + vl - 1 < size; i += vl)
        {
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vl);
            _sum = __riscv_vfadd_vv_f16m8(_sum, _p, vl);
            ptr_sum += vl;
        }

        _sum0 = __riscv_vfredusum_vs_f16m8_f16m1(_sum, _sum0, vl);
        mean += __riscv_vfmv_f_s_f16m1_f16(_sum0);
#endif // __riscv_zvfh
        for (; i < size; i++) mean += *ptr_sum++;
        mean /= elementcount;
    }

    i = 0;
    ptr_sum = ptr;
#if __riscv_zvfh
    if (elementpack != 1)
    {
        _sqsum0 = __riscv_vfmv_v_f_f16m1(0.f, __riscv_vsetvlmax_e16m1());
        for (; i < elementcount; i++)
        {
            vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + vl * i, vl);
            _p = __riscv_vfsub_vv_f16m1(_p, _sum0, vl);
            _sqsum0 = __riscv_vfmacc_vv_f16m1(_sqsum0, _p, _p, vl);
        }
        _sqsum0 = __riscv_vfdiv_vf_f16m1(_sqsum0, elementcount, vl);
        _a = __riscv_vfrdiv_vf_f16m1(__riscv_vfsqrt_v_f16m1(__riscv_vfadd_vf_f16m1(_sqsum0, eps, vl), vl), 1.f, vl);
        _b = __riscv_vfmul_vv_f16m1(__riscv_vfsgnjn_vv_f16m1(_sum0, _sum0, vl), _a, vl);
    }
#endif // __riscv_zvfh

    if (elementpack == 1)
    {
#if __riscv_zvfh
        _sqsum = __riscv_vfmv_v_f_f16m8(0.f, vl);
        for (; i + vl - 1 < size; i += vl)
        {
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vl);
            vfloat16m8_t _temp = __riscv_vfsub_vf_f16m8(_p, mean, vl);
            _temp = __riscv_vfmul_vv_f16m8(_temp, _temp, vl);
            _sqsum = __riscv_vfadd_vv_f16m8(_sqsum, _temp, vl);

            ptr_sum += vl;
        }

        _sqsum0 = __riscv_vfredusum_vs_f16m8_f16m1(_sqsum, _sqsum0, vl);
        var += __riscv_vfmv_f_s_f16m1_f16(_sqsum0);
#endif // __riscv_zvfh
        for (; i < size; i++)
        {
            __fp16 tmp = *ptr_sum++ - mean;
            var += tmp * tmp;
        }

        var /= elementcount;
        a = static_cast<__fp16>(1.f / (sqrt((float)var + eps)));
        b = -mean * a;
    }

    i = 0;
    int n = size;
    __fp16* ptr_store = ptr;
    if (gamma_data && beta_data)
    {
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
#if __riscv_zvfh
        if (elementpack != 1)
        {
            for (; i < elementcount; i++)
            {
                const int offset = vl * i;
                vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + offset, vl);
                _p = __riscv_vfmadd_vv_f16m1(_p, _a, _b, vl);
                _p = __riscv_vfmul_vf_f16m1(_p, (__fp16)*ptr_gamma++, vl);
                _p = __riscv_vfadd_vf_f16m1(_p, (__fp16)*ptr_beta++, vl);
                __riscv_vse16_v_f16m1(ptr + offset, _p, vl);
            }
        }
#endif // __riscv_zvfh

        if (elementpack == 1)
        {
#if __riscv_zvfh
            while (n > 0)
            {
                size_t vlr = __riscv_vsetvl_e16m4(n);
                vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr_store, vlr);
                _p = __riscv_vfmul_vf_f16m4(_p, a, vlr);
                _p = __riscv_vfadd_vf_f16m4(_p, b, vlr);

                vfloat16m4_t _gamma = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_gamma, vlr), vlr);
                vfloat16m4_t _beta = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_beta, vlr), vlr);
                _p = __riscv_vfmadd_vv_f16m4(_p, _gamma, _beta, vlr);
                __riscv_vse16_v_f16m4(ptr_store, _p, vlr);

                n -= vlr;
                ptr_store += vlr;
                ptr_gamma += vlr;
                ptr_beta += vlr;
            }
#endif // __riscv_zvfh
            while (n-- > 0) *ptr_store++ = (*ptr_store * a + b) * (__fp16)*ptr_gamma++ + (__fp16)*ptr_beta++;
        }
    }
    else
    {
#if __riscv_zvfh
        if (elementpack != 1)
        {
            for (; i < elementcount; i++)
            {
                const int offset = vl * i;
                vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + offset, vl);
                _p = __riscv_vfmadd_vv_f16m1(_p, _a, _b, vl);
                __riscv_vse16_v_f16m1(ptr + offset, _p, vl);
            }
        }
#endif // __riscv_zvfh

        if (elementpack == 1)
        {
#if __riscv_zvfh
            while (n > 0)
            {
                size_t vlr = __riscv_vsetvl_e16m8(n);
                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_store, vlr);
                _p = __riscv_vfmul_vf_f16m8(_p, a, vlr);
                _p = __riscv_vfadd_vf_f16m8(_p, b, vlr);
                __riscv_vse16_v_f16m8(ptr_store, _p, vlr);

                n -= vlr;
                ptr_store += vlr;
            }
#endif // __riscv_zvfh
            while (n-- > 0) *ptr_store++ = (*ptr_store * a + b);
        }
    }
    return 0;
}

int LayerNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;

    size_t vl = 1;
#if __riscv_vector
    const int packn = csrr_vlenb() / 2; // fp16
    vl = (elempack == packn) ? __riscv_vsetvl_e16m1(packn) : __riscv_vsetvlmax_e16m4();
#endif // __riscv_vector

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
        return layernorm_fp16s(ptr, gamma_data, beta_data, eps, w * elempack, 1, vl);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            layernorm_fp16s(ptr, gamma_data, beta_data, eps, w, elempack, vl);
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
                    layernorm_fp16s(ptr, gamma_data, beta_data, eps, w, elempack, vl);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                layernorm_fp16s(ptr, gamma_data, beta_data, eps, w * h, elempack, vl);
            }
        }
    }
    return 0;
}

int LayerNorm_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t vl = 1;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2; // fp16
    vl = (elempack == packn) ? __riscv_vsetvl_e16m1(packn) : __riscv_vsetvlmax_e16m8();
#endif

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
        return layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w * elempack, 1, vl);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w, elempack, vl);
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
                    layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w, elempack, vl);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w * h, elempack, vl);
            }
        }
    }
    return 0;
}
#endif // NCNN_ZFH
} // namespace ncnn
