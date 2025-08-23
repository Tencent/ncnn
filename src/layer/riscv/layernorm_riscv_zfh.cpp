// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
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

static int layernorm_fp16s(__fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int elementcount, int elementpack)
{
    float mean = 0.f;
    float var = 0.f;

    size_t size = elementcount * elementpack;
    int remain = elementcount;
#if __riscv_vector
    size_t vl_max = __riscv_vsetvlmax_e16m4();
    size_t vl_f32 = __riscv_vsetvlmax_e32m8();
    remain = elementcount % vl_max;
#endif

    int i = 0;
    __fp16* ptr_sum = ptr;
#if __riscv_vector
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl_max);
    for (; i + vl_max - 1 < size; i += vl_max)
    {
        vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vl_max), vl_f32);
        _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl_f32);
        ptr_sum += vl_max;
    }

    if (i < size)
    {
        size_t vlr = __riscv_vsetvl_e16m4(remain);
        vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vlr), __riscv_vsetvl_e32m8(remain));
#if __riscv_xtheadvector
        _p = reset_tails(_p, __riscv_vsetvl_e32m8(remain), 0.f);
        _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl_f32);
#else
        _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, __riscv_vsetvl_e32m8(remain));
#endif
        i += remain;
    }

    vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vl_f32);
    mean = __riscv_vfmv_f_s_f32m1_f32(_sum0);
#endif // __riscv_vector
    for (; i < size; i++) mean += (float)*ptr_sum++;
    mean /= elementcount;

    i = 0;
    ptr_sum = ptr;
#if __riscv_vector
    vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vl_f32);
    for (; i + vl_max - 1 < size; i += vl_max)
    {
        vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vl_max), vl_f32);
        vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, vl_f32);
        _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, vl_f32);
        _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl_f32);

        ptr_sum += vl_max;
    }

    if (i < size)
    {
        size_t vlr = __riscv_vsetvl_e16m4(remain);
        vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vlr), __riscv_vsetvl_e32m8(remain));
        vfloat32m8_t _temp = __riscv_vfsub_vf_f32m8(_p, mean, __riscv_vsetvl_e32m8(remain));
        _temp = __riscv_vfmul_vv_f32m8(_temp, _temp, __riscv_vsetvl_e32m8(remain));
#if __riscv_xtheadvector
        _temp = reset_tails(_temp, __riscv_vsetvl_e32m8(remain), 0.f);
        _sqsum = __riscv_vfadd_vv_f32m8(_sqsum, _temp, vl_f32);
#else
        _sqsum = __riscv_vfadd_vv_f32m8_tu(_sqsum, _sqsum, _temp, __riscv_vsetvl_e32m8(remain));
#endif // __riscv_xtheadvector
        i += remain;
    }

    vfloat32m1_t _sqsum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vl_f32);
    var = __riscv_vfmv_f_s_f32m1_f32(_sqsum0);
#endif // __riscv_vector
    for (; i < size; i++)
    {
        float tmp = (float)*ptr_sum++ - mean;
        var += tmp * tmp;
    }

    var /= elementcount;
    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    int n = size;
    __fp16* ptr_store = ptr;
    if (gamma_data && beta_data)
    {
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
#if __riscv_vector
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vl), vl);
            _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, b, vl);

            vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vl);
            vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vl);
            _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
            __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

            n -= vl;
            ptr_store += vl;
            ptr_gamma += vl;
            ptr_beta += vl;
        }

#endif // __riscv_vector
        while (n-- > 0) *ptr_store++ = (__fp16)((float)*ptr_store * a + b) * *ptr_gamma++ + *ptr_beta++;
    }
    else
    {
#if __riscv_vector
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vl), vl);
            _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
            __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);
            n -= vl;
            ptr_store += vl;
        }
#endif // __riscv_vector
        while (n-- > 0) *ptr_store++ = (__fp16)((float)*ptr_store * a + b);
    }
    return 0;
}

#if __riscv_vector
static inline int layernorm_rvv_packn_fp16s_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, const size_t vl)
{
    // f16m1 => f32m2
    vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);
    vfloat32m2_t _sqsum = __riscv_vfmv_v_f_f32m2(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
        _sum = __riscv_vfadd_vv_f32m2(_p, _sum, vl);
    }
    vfloat32m2_t _mean = __riscv_vfdiv_vf_f32m2(_sum, (float)size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
        _p = __riscv_vfsub_vv_f32m2(_p, _mean, vl);
        _sqsum = __riscv_vfmacc_vv_f32m2(_sqsum, _p, _p, vl);
    }
    vfloat32m2_t _var = __riscv_vfdiv_vf_f32m2(_sqsum, (float)size, vl);
    vfloat32m2_t _a = __riscv_vfrdiv_vf_f32m2(__riscv_vfsqrt_v_f32m2(__riscv_vfadd_vf_f32m2(_var, eps, vl), vl), 1.f, vl);
    vfloat32m2_t _b = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(_mean, _mean, vl), _a, vl);
    if (gamma_data && beta_data)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + offset, vl), vl);
            _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
            _p = __riscv_vfmul_vf_f32m2(_p, gamma_data[i], vl);
            _p = __riscv_vfadd_vf_f32m2(_p, beta_data[i], vl);
            __riscv_vse16_v_f16m1(ptr + offset, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + offset, vl), vl);
            _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
            __riscv_vse16_v_f16m1(ptr + offset, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
        }
    }

    return 0;
}
#endif

#if __riscv_zvfh
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline))
#endif // __riscv_xtheadvector
static vfloat16m8_t
reset_tails(vfloat16m8_t x, size_t vl, __fp16 v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e16m8();
    vbool2_t _vl_mask = __riscv_vmsgeu_vx_u16m8_b2(__riscv_vid_v_u16m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f16m8(x, v, _vl_mask, vlm8);
    return x;
}
#endif // __riscv_zvfh

static int layernorm_fp16sa(__fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int elementcount, int elementpack)
{
    __fp16 mean = 0.f;
    __fp16 var = 0.f;

    size_t size = elementcount * elementpack;
    int remain = elementcount;
#if __riscv_zvfh
    size_t vl_max = __riscv_vsetvlmax_e16m8();
    remain = elementcount % vl_max;
#endif // __riscv_zvfh

    int i = 0;
    __fp16* ptr_sum = ptr;
#if __riscv_zvfh
    vfloat16m8_t _sum = __riscv_vfmv_v_f_f16m8(0.f, vl_max);
    for (; i + vl_max - 1 < size; i += vl_max)
    {
        vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vl_max);
        _sum = __riscv_vfadd_vv_f16m8(_sum, _p, vl_max);
        ptr_sum += vl_max;
    }

    if (i < size)
    {
        size_t vlr = __riscv_vsetvl_e16m8(remain);
        vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vlr);
#if __riscv_xtheadvector
        _p = reset_tails(_p, vlr, (__fp16)0.f);
        _sum = __riscv_vfadd_vv_f16m8(_sum, _p, vl_max);
#else
        _sum = __riscv_vfadd_vv_f16m8_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        i += remain;
    }

    vfloat16m1_t _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, __riscv_vsetvlmax_e16m1());
    _sum0 = __riscv_vfredusum_vs_f16m8_f16m1(_sum, _sum0, vl_max);
    mean += __riscv_vfmv_f_s_f16m1_f16(_sum0);
#endif // __riscv_zvfh
    for (; i < size; i++) mean += *ptr_sum++;
    mean /= elementcount;

    i = 0;
    ptr_sum = ptr;
#if __riscv_zvfh
    vfloat16m8_t _sqsum = __riscv_vfmv_v_f_f16m8((__fp16)0.f, vl_max);
    for (; i + vl_max - 1 < size; i += vl_max)
    {
        vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vl_max);
        vfloat16m8_t _temp = __riscv_vfsub_vf_f16m8(_p, mean, vl_max);
        _temp = __riscv_vfmul_vv_f16m8(_temp, _temp, vl_max);
        _sqsum = __riscv_vfadd_vv_f16m8(_sqsum, _temp, vl_max);

        ptr_sum += vl_max;
    }

    if (i < size)
    {
        size_t vlr = __riscv_vsetvl_e16m8(remain);
        vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vlr);
        vfloat16m8_t _temp = __riscv_vfsub_vf_f16m8(_p, mean, vlr);
        _temp = __riscv_vfmul_vv_f16m8(_temp, _temp, vlr);

#if __riscv_xtheadvector
        _temp = reset_tails(_temp, vlr, (__fp16)0.f);
        _sqsum = __riscv_vfadd_vv_f16m8(_sqsum, _temp, vl_max);
#else
        _sqsum = __riscv_vfadd_vv_f16m8_tu(_sqsum, _sqsum, _temp, vlr);
#endif // __riscv_xtheadvector
        i += remain;
    }

    vfloat16m1_t _sqsum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, __riscv_vsetvlmax_e16m1());
    _sqsum0 = __riscv_vfredusum_vs_f16m8_f16m1(_sqsum, _sqsum0, vl_max);
    var += __riscv_vfmv_f_s_f16m1_f16(_sqsum0);
#endif // __riscv_zvfh
    for (; i < size; i++)
    {
        __fp16 tmp = *ptr_sum++ - mean;
        var += tmp * tmp;
    }

    var /= elementcount;
    __fp16 a = static_cast<__fp16>(1.f / (sqrt((float)var + eps)));
    __fp16 b = -mean * a;

    int n = size;
    __fp16* ptr_store = ptr;
    if (gamma_data && beta_data)
    {
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
#if __riscv_zvfh
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr_store, vl);
            _p = __riscv_vfmul_vf_f16m4(_p, a, vl);
            _p = __riscv_vfadd_vf_f16m4(_p, b, vl);

            vfloat16m4_t _gamma = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_gamma, vl), vl);
            vfloat16m4_t _beta = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_beta, vl), vl);
            _p = __riscv_vfmadd_vv_f16m4(_p, _gamma, _beta, vl);
            __riscv_vse16_v_f16m4(ptr_store, _p, vl);

            n -= vl;
            ptr_store += vl;
            ptr_gamma += vl;
            ptr_beta += vl;
        }

#endif // __riscv_zvfh
        while (n-- > 0) *ptr_store++ = (*ptr_store * a + b) * (__fp16)*ptr_gamma++ + (__fp16)*ptr_beta++;
    }
    else
    {
#if __riscv_zvfh
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_store, vl);
            _p = __riscv_vfmul_vf_f16m8(_p, a, vl);
            _p = __riscv_vfadd_vf_f16m8(_p, b, vl);
            __riscv_vse16_v_f16m8(ptr_store, _p, vl);
            n -= vl;
            ptr_store += vl;
        }
#endif // __riscv_zvfh
        while (n-- > 0) *ptr_store++ = (*ptr_store * a + b);
    }
    return 0;
}

#if __riscv_zvfh
static inline int layernorm_rvv_packn_fp16sa_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, const size_t vl)
{
    vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1(0.f, vl);
    vfloat16m1_t _sqsum = __riscv_vfmv_v_f_f16m1(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + vl * i, vl);
        _sum = __riscv_vfadd_vv_f16m1(_p, _sum, vl);
    }

    vfloat16m1_t _mean = __riscv_vfdiv_vf_f16m1(_sum, size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + vl * i, vl);
        _p = __riscv_vfsub_vv_f16m1(_p, _mean, vl);
        _sqsum = __riscv_vfmacc_vv_f16m1(_sqsum, _p, _p, vl);
    }

    vfloat16m1_t _var = __riscv_vfdiv_vf_f16m1(_sqsum, size, vl);
    vfloat16m1_t _a = __riscv_vfrdiv_vf_f16m1(__riscv_vfsqrt_v_f16m1(__riscv_vfadd_vf_f16m1(_var, eps, vl), vl), 1.f, vl);
    vfloat16m1_t _b = __riscv_vfmul_vv_f16m1(__riscv_vfsgnjn_vv_f16m1(_mean, _mean, vl), _a, vl);
    if (gamma_data && beta_data)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + offset, vl);
            _p = __riscv_vfmadd_vv_f16m1(_p, _a, _b, vl);
            _p = __riscv_vfmul_vf_f16m1(_p, gamma_data[i], vl);
            _p = __riscv_vfadd_vf_f16m1(_p, beta_data[i], vl);
            __riscv_vse16_v_f16m1(ptr + offset, _p, vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + offset, vl);
            _p = __riscv_vfmadd_vv_f16m1(_p, _a, _b, vl);
            __riscv_vse16_v_f16m1(ptr + offset, _p, vl);
        }
    }

    return 0;
}
#endif // __riscv_zvfh

int LayerNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2; // fp16
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif // __riscv_vector

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
        return layernorm_fp16s(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_packn_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, vl);
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            // assert affine_size == w
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_fp16s(ptr, gamma_data, beta_data, eps, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
                        layernorm_rvv_packn_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, vl);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_fp16s_procedure(w * h, ptr, gamma_data, beta_data, eps, vl);
                }
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
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

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2; // fp16
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
        return layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
#if __riscv_zvfh
        if (elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_packn_fp16sa_procedure(w, ptr, gamma_data, beta_data, eps, vl);
            }
        }
#endif // __riscv_zvfh
        if (elempack == 1)
        {
            // assert affine_size == w
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
#if __riscv_zvfh
        if (elempack == packn)
        {
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
                        layernorm_rvv_packn_fp16sa_procedure(w, ptr, gamma_data, beta_data, eps, vl);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_fp16sa_procedure(w * h, ptr, gamma_data, beta_data, eps, vl);
                }
            }
        }
#endif // __riscv_zvfh
        if (elempack == 1)
        {
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
                        layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w, elempack);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_fp16sa(ptr, gamma_data, beta_data, eps, w * h, elempack);
                }
            }
        }
    }
    return 0;
}
#endif // NCNN_ZFH
} // namespace ncnn
