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
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void rmsnorm_mips_bf16(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __mips_msa
    if (elempack == 4)
    {
        // compute rms
        v4f32 _rms = (v4f32)__msa_fill_w(0);
        const unsigned short* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr0, 0));
            _rms = __msa_fmadd_w(_rms, _p, _p);
            ptr0 += 4;
        }

        float rms_data[4];
        __msa_st_w((v4i32)_rms, rms_data, 0);
        for (int i = 0; i < 4; i++)
        {
            rms_data[i] = 1.f / sqrtf(rms_data[i] / elemcount + eps);
        }
        _rms = (v4f32)__msa_ld_w(rms_data, 0);

        if (gamma_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                _p = __msa_fmul_w(_p, _rms);
                _p = __msa_fmul_w(_p, _gamma);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                _p = __msa_fmul_w(_p, _rms);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __mips_msa

    // elempack == 1 or no MSA
    float rms = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v4f32 _rms = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr0, 0));
            _rms = __msa_fmadd_w(_rms, _p, _p);
            ptr0 += 4;
        }
        rms += __msa_reduce_fadd_w(_rms);
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]);
            rms += v * v;
            ptr0++;
        }
    }

    rms = 1.f / sqrtf(rms / elemcount + eps);

    if (gamma_ptr)
    {
        int i = 0;
#if __mips_msa
        v4f32 _rms = __msa_fill_w_f32(rms);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
            v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
            _p = __msa_fmul_w(_p, _rms);
            _p = __msa_fmul_w(_p, _gamma);
            __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
            ptr += 4;
            gamma_ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16((v * rms) * gamma_ptr[0]);
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __mips_msa
        v4f32 _rms = __msa_fill_w_f32(rms);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
            _p = __msa_fmul_w(_p, _rms);
            __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
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

static void rmsnorm_mips(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _rms = (v4f32)__msa_fill_w(0);
        const float* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _rms = __msa_fmadd_w(_rms, _p, _p);
            ptr0 += 4;
        }

        float rms_data[4];
        __msa_st_w((v4i32)_rms, rms_data, 0);
        for (int i = 0; i < 4; i++)
        {
            rms_data[i] = 1.f / sqrtf(rms_data[i] / elemcount + eps);
        }
        _rms = (v4f32)__msa_ld_w(rms_data, 0);

        if (gamma_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                _p = __msa_fmul_w(_p, _rms);
                _p = __msa_fmul_w(_p, _gamma);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __msa_fmul_w(_p, _rms);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __mips_msa

    float rms = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v4f32 _rms = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _rms = __msa_fmadd_w(_rms, _p, _p);
            ptr0 += 4;
        }
        rms += __msa_reduce_fadd_w(_rms);
#endif // __mips_msa
        for (; i < size; i++)
        {
            rms += ptr0[0] * ptr0[0];
            ptr0++;
        }
    }

    rms = 1.f / sqrtf(rms / elemcount + eps);

    if (gamma_ptr)
    {
        int i = 0;
#if __mips_msa
        v4f32 _rms = __msa_fill_w_f32(rms);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
            _p = __msa_fmul_w(_p, _rms);
            _p = __msa_fmul_w(_p, _gamma);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
            gamma_ptr += 4;
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
        v4f32 _rms = __msa_fill_w_f32(rms);
        for (; i + 3 < size; i += 4)
        {
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

    return 0;
}

#if NCNN_BF16
int RMSNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
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
                    unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q).row<unsigned short>(i);
                    rmsnorm_mips_bf16(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q);
                rmsnorm_mips_bf16(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
