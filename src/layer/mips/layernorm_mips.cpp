// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

LayerNorm_mips::LayerNorm_mips()
{
#if __mips_msa
    support_packing = true;
    support_any_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void layernorm_mips(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute mean
#if __mips_msa
    v4f32 _mean = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float mean = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _mean = __msa_fadd_w(_mean, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        _mean = __msa_fdiv_w(_mean, _elemcount);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        mean += __msa_reduce_fadd_w(_mean);
#endif // __mips_msa

        mean = mean / elemcount;
#if __mips_msa
        _mean = __msa_fill_w_f32(mean);
#endif // __mips_msa
    }
    // compute variance
#if __mips_msa
    v4f32 _var = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float var = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _p = __msa_fsub_w(_p, _mean);
            _var = __ncnn_msa_fmadd_w(_var, _p, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        v4f32 _eps = __msa_fill_w_f32(eps);
        _var = __msa_fdiv_w(_var, _elemcount);
        _var = __msa_fadd_w(_var, _eps);
        _var = __msa_frsqrt_w(_var);
        _mean = __msa_fmul_w(_mean, _var);
        _mean = __msa_fsub_w((v4f32)__msa_fill_w(0), _mean);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        var += __msa_reduce_fadd_w(_var);
#endif // __mips_msa

        var = 1.f / sqrtf(var / elemcount + eps);
        mean = -mean * var;
#if __mips_msa
        _var = __msa_fill_w_f32(var);
        _mean = __msa_fill_w_f32(mean);
#endif // __mips_msa
    }
    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __mips_msa
        if (elempack == 4)
        {
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);

                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                v4f32 _beta = __msa_fill_w_f32(beta_ptr[0]);
                _p = __ncnn_msa_fmadd_w(_mean, _p, _var);
                _p = __ncnn_msa_fmadd_w(_beta, _p, _gamma);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);
                __builtin_prefetch(beta_ptr + 16);

                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
                v4f32 _beta = (v4f32)__msa_ld_w(beta_ptr, 0);
                _p = __ncnn_msa_fmadd_w(_mean, _p, _var);
                _p = __ncnn_msa_fmadd_w(_beta, _p, _gamma);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * var + mean) * gamma_ptr[0] + beta_ptr[0];
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __ncnn_msa_fmadd_w(_mean, _p, _var);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * var + mean;
            ptr++;
        }
    }
}

int LayerNorm_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        layernorm_mips(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm_mips(ptr, gamma_data, beta_data, eps, w, elempack);
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
                    layernorm_mips(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm_mips(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    if (dims == 4)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).depth(z).row(i);
                        layernorm_mips(ptr, gamma_data, beta_data, eps, w, elempack);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    float* ptr = bottom_top_blob.channel(q).depth(z);
                    layernorm_mips(ptr, gamma_data, beta_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm_mips(ptr, gamma_data, beta_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void layernorm_mips_bf16(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute mean
#if __mips_msa
    v4f32 _mean0 = (v4f32)__msa_fill_w(0);
    v4f32 _mean1 = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float mean = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v8i16 _zero_bf16 = __msa_fill_h(0);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr0 + 16);

            v8i16 _p01 = __msa_ld_h(ptr0, 0);
            v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
            v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
            _mean0 = __msa_fadd_w(_mean0, _p0);
            _mean1 = __msa_fadd_w(_mean1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = bfloat2float_msa(ptr0);
            _mean0 = __msa_fadd_w(_mean0, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            mean += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }
    }

#if __mips_msa
    if (elempack == 8)
    {
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        _mean0 = __msa_fdiv_w(_mean0, _elemcount);
        _mean1 = __msa_fdiv_w(_mean1, _elemcount);
    }
    if (elempack == 4)
    {
        _mean0 = __msa_fadd_w(_mean0, _mean1);
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        _mean0 = __msa_fdiv_w(_mean0, _elemcount);
        _mean1 = _mean0;
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        mean += __msa_reduce_fadd_w(_mean0);
        mean += __msa_reduce_fadd_w(_mean1);
#endif // __mips_msa

        mean = mean / elemcount;
#if __mips_msa
        _mean0 = __msa_fill_w_f32(mean);
        _mean1 = _mean0;
#endif // __mips_msa
    }
    // compute variance
#if __mips_msa
    v4f32 _var0 = (v4f32)__msa_fill_w(0);
    v4f32 _var1 = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float var = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v8i16 _zero_bf16 = __msa_fill_h(0);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr0 + 16);

            v8i16 _p01 = __msa_ld_h(ptr0, 0);
            v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
            v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
            _p0 = __msa_fsub_w(_p0, _mean0);
            _p1 = __msa_fsub_w(_p1, _mean1);
            _var0 = __ncnn_msa_fmadd_w(_var0, _p0, _p0);
            _var1 = __ncnn_msa_fmadd_w(_var1, _p1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = bfloat2float_msa(ptr0);
            _p = __msa_fsub_w(_p, _mean0);
            _var0 = __ncnn_msa_fmadd_w(_var0, _p, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]) - mean;
            var += v * v;
            ptr0++;
        }
    }

#if __mips_msa
    if (elempack == 8)
    {
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        v4f32 _eps = __msa_fill_w_f32(eps);
        _var0 = __msa_fdiv_w(_var0, _elemcount);
        _var1 = __msa_fdiv_w(_var1, _elemcount);
        _var0 = __msa_fadd_w(_var0, _eps);
        _var1 = __msa_fadd_w(_var1, _eps);
        _var0 = __msa_frsqrt_w(_var0);
        _var1 = __msa_frsqrt_w(_var1);
        _mean0 = __msa_fmul_w(_mean0, _var0);
        _mean1 = __msa_fmul_w(_mean1, _var1);
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        _mean0 = __msa_fsub_w(_zero, _mean0);
        _mean1 = __msa_fsub_w(_zero, _mean1);
    }
    if (elempack == 4)
    {
        _var0 = __msa_fadd_w(_var0, _var1);

        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        v4f32 _eps = __msa_fill_w_f32(eps);
        _var0 = __msa_fdiv_w(_var0, _elemcount);
        _var0 = __msa_fadd_w(_var0, _eps);
        _var0 = __msa_frsqrt_w(_var0);
        _mean0 = __msa_fmul_w(_mean0, _var0);
        _mean0 = __msa_fsub_w((v4f32)__msa_fill_w(0), _mean0);
        _var1 = _var0;
        _mean1 = _mean0;
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        var += __msa_reduce_fadd_w(_var0);
        var += __msa_reduce_fadd_w(_var1);
#endif // __mips_msa

        var = 1.f / sqrtf(var / elemcount + eps);
        mean = -mean * var;
#if __mips_msa
        _var0 = __msa_fill_w_f32(var);
        _var1 = _var0;
        _mean0 = __msa_fill_w_f32(mean);
        _mean1 = _mean0;
#endif // __mips_msa
    }
    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __mips_msa
        v8i16 _zero_bf16 = __msa_fill_h(0);
        if (elempack == 8)
        {
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 16);

                v8i16 _p01 = __msa_ld_h(ptr, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                v4f32 _beta = __msa_fill_w_f32(beta_ptr[0]);
                _p0 = __ncnn_msa_fmadd_w(_mean0, _p0, _var0);
                _p1 = __ncnn_msa_fmadd_w(_mean1, _p1, _var1);
                _p0 = __ncnn_msa_fmadd_w(_beta, _p0, _gamma);
                _p1 = __ncnn_msa_fmadd_w(_beta, _p1, _gamma);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 4)
        {
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 16);

                v8i16 _p01 = __msa_ld_h(ptr, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4f32 _gamma0 = __msa_fill_w_f32(gamma_ptr[0]);
                v4f32 _gamma1 = __msa_fill_w_f32(gamma_ptr[1]);
                v4f32 _beta0 = __msa_fill_w_f32(beta_ptr[0]);
                v4f32 _beta1 = __msa_fill_w_f32(beta_ptr[1]);
                _p0 = __ncnn_msa_fmadd_w(_mean0, _p0, _var0);
                _p1 = __ncnn_msa_fmadd_w(_mean1, _p1, _var1);
                _p0 = __ncnn_msa_fmadd_w(_beta0, _p0, _gamma0);
                _p1 = __ncnn_msa_fmadd_w(_beta1, _p1, _gamma1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 2;
                beta_ptr += 2;
            }
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);

                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                v4f32 _beta = __msa_fill_w_f32(beta_ptr[0]);
                _p = __ncnn_msa_fmadd_w(_mean0, _p, _var0);
                _p = __ncnn_msa_fmadd_w(_beta, _p, _gamma);
                *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);
                __builtin_prefetch(beta_ptr + 16);

                v8i16 _p01 = __msa_ld_h(ptr, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4f32 _gamma0 = (v4f32)__msa_ld_w(gamma_ptr, 0);
                v4f32 _gamma1 = (v4f32)__msa_ld_w(gamma_ptr + 4, 0);
                v4f32 _beta0 = (v4f32)__msa_ld_w(beta_ptr, 0);
                v4f32 _beta1 = (v4f32)__msa_ld_w(beta_ptr + 4, 0);
                _p0 = __ncnn_msa_fmadd_w(_mean0, _p0, _var0);
                _p1 = __ncnn_msa_fmadd_w(_mean1, _p1, _var1);
                _p0 = __ncnn_msa_fmadd_w(_beta0, _p0, _gamma0);
                _p1 = __ncnn_msa_fmadd_w(_beta1, _p1, _gamma1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 8;
                beta_ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);
                __builtin_prefetch(beta_ptr + 16);

                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
                v4f32 _beta = (v4f32)__msa_ld_w(beta_ptr, 0);
                _p = __ncnn_msa_fmadd_w(_mean0, _p, _var0);
                _p = __ncnn_msa_fmadd_w(_beta, _p, _gamma);
                *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16((v * var + mean) * gamma_ptr[0] + beta_ptr[0]);
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __mips_msa
        v8i16 _zero_bf16 = __msa_fill_h(0);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 16);

            v8i16 _p01 = __msa_ld_h(ptr, 0);
            v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
            v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
            _p0 = __ncnn_msa_fmadd_w(_mean0, _p0, _var0);
            _p1 = __ncnn_msa_fmadd_w(_mean1, _p1, _var1);
            __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            v4f32 _p = bfloat2float_msa(ptr);
            _p = __ncnn_msa_fmadd_w(_mean0, _p, _var0);
            *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16(v * var + mean);
            ptr++;
        }
    }
}

int LayerNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w, elempack);
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
                    unsigned short* ptr = bottom_top_blob.channel(q).row<unsigned short>(i);
                    layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    if (dims == 4)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        unsigned short* ptr = bottom_top_blob.channel(q).depth(z).row<unsigned short>(i);
                        layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w, elempack);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    unsigned short* ptr = bottom_top_blob.channel(q).depth(z);
                    layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
