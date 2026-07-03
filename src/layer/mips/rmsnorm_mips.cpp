// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

RMSNorm_mips::RMSNorm_mips()
{
#if __mips_msa
    support_packing = true;
    support_any_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void rmsnorm_mips(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute rms
#if __mips_msa
    v4f32 _rms = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float rms = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _rms = __ncnn_msa_fmadd_w(_rms, _p, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            rms += ptr0[0] * ptr0[0];
            ptr0++;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        v4f32 _eps = __msa_fill_w_f32(eps);
        _rms = __msa_fdiv_w(_rms, _elemcount);
        _rms = __msa_fadd_w(_rms, _eps);
        _rms = __msa_frsqrt_w(_rms);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        rms += __msa_reduce_fadd_w(_rms);
#endif // __mips_msa

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __mips_msa
        _rms = __msa_fill_w_f32(rms);
#endif // __mips_msa
    }
    if (gamma_ptr)
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
                _p = __msa_fmul_w(_p, _rms);
                _p = __msa_fmul_w(_p, _gamma);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);

                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
                _p = __msa_fmul_w(_p, _rms);
                _p = __msa_fmul_w(_p, _gamma);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * rms) * gamma_ptr[0];
            ptr++;
            gamma_ptr++;
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
            _p = __msa_fmul_w(_p, _rms);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * rms;
            ptr++;
        }
    }
}

int RMSNorm_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        rmsnorm_mips(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm_mips(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm_mips(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_mips(ptr, gamma_data, eps, w * h, elempack);
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
                        rmsnorm_mips(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm_mips(ptr, gamma_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_mips(ptr, gamma_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void rmsnorm_mips_bf16(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute rms
#if __mips_msa
    v4f32 _rms0 = (v4f32)__msa_fill_w(0);
    v4f32 _rms1 = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float rms = 0.f;
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
            _rms0 = __ncnn_msa_fmadd_w(_rms0, _p0, _p0);
            _rms1 = __ncnn_msa_fmadd_w(_rms1, _p1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = bfloat2float_msa(ptr0);
            _rms0 = __ncnn_msa_fmadd_w(_rms0, _p, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]);
            rms += v * v;
            ptr0++;
        }
    }

#if __mips_msa
    if (elempack == 8)
    {
        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        v4f32 _eps = __msa_fill_w_f32(eps);
        _rms0 = __msa_fdiv_w(_rms0, _elemcount);
        _rms1 = __msa_fdiv_w(_rms1, _elemcount);
        _rms0 = __msa_fadd_w(_rms0, _eps);
        _rms1 = __msa_fadd_w(_rms1, _eps);
        _rms0 = __msa_frsqrt_w(_rms0);
        _rms1 = __msa_frsqrt_w(_rms1);
    }
    if (elempack == 4)
    {
        _rms0 = __msa_fadd_w(_rms0, _rms1);

        v4f32 _elemcount = __msa_fill_w_f32((float)elemcount);
        v4f32 _eps = __msa_fill_w_f32(eps);
        _rms0 = __msa_fdiv_w(_rms0, _elemcount);
        _rms0 = __msa_fadd_w(_rms0, _eps);
        _rms0 = __msa_frsqrt_w(_rms0);
        _rms1 = _rms0;
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        rms += __msa_reduce_fadd_w(_rms0);
        rms += __msa_reduce_fadd_w(_rms1);
#endif // __mips_msa

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __mips_msa
        _rms0 = __msa_fill_w_f32(rms);
        _rms1 = _rms0;
#endif // __mips_msa
    }
    if (gamma_ptr)
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
                _p0 = __msa_fmul_w(_p0, _rms0);
                _p1 = __msa_fmul_w(_p1, _rms1);
                _p0 = __msa_fmul_w(_p0, _gamma);
                _p1 = __msa_fmul_w(_p1, _gamma);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
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
                _p0 = __msa_fmul_w(_p0, _rms0);
                _p1 = __msa_fmul_w(_p1, _rms1);
                _p0 = __msa_fmul_w(_p0, _gamma0);
                _p1 = __msa_fmul_w(_p1, _gamma1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 2;
            }
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);

                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                _p = __msa_fmul_w(_p, _rms0);
                _p = __msa_fmul_w(_p, _gamma);
                *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);

                v8i16 _p01 = __msa_ld_h(ptr, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4f32 _gamma0 = (v4f32)__msa_ld_w(gamma_ptr, 0);
                v4f32 _gamma1 = (v4f32)__msa_ld_w(gamma_ptr + 4, 0);
                _p0 = __msa_fmul_w(_p0, _rms0);
                _p1 = __msa_fmul_w(_p1, _rms1);
                _p0 = __msa_fmul_w(_p0, _gamma0);
                _p1 = __msa_fmul_w(_p1, _gamma1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);

                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
                _p = __msa_fmul_w(_p, _rms0);
                _p = __msa_fmul_w(_p, _gamma);
                *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * rms * gamma_ptr[0]);
            ptr++;
            gamma_ptr++;
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
            _p0 = __msa_fmul_w(_p0, _rms0);
            _p1 = __msa_fmul_w(_p1, _rms1);
            __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            v4f32 _p = bfloat2float_msa(ptr);
            _p = __msa_fmul_w(_p, _rms0);
            *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16(v * rms);
            ptr++;
        }
    }
}

int RMSNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        rmsnorm_mips_bf16(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            rmsnorm_mips_bf16(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm_mips_bf16(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                rmsnorm_mips_bf16(ptr, gamma_data, eps, w * h, elempack);
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
                        rmsnorm_mips_bf16(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm_mips_bf16(ptr, gamma_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                rmsnorm_mips_bf16(ptr, gamma_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
