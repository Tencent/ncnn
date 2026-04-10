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
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void layernorm_mips_bf16(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __mips_msa
    if (elempack == 4)
    {
        // compute mean
        v4f32 _sum = (v4f32)__msa_fill_w(0);
        const unsigned short* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr0, 0));
            _sum = __msa_fadd_w(_sum, _p);
            ptr0 += 4;
        }

        float sum_data[4];
        __msa_st_w((v4i32)_sum, sum_data, 0);

        float mean_data[4];
        for (int i = 0; i < 4; i++)
        {
            mean_data[i] = sum_data[i] / elemcount;
        }
        v4f32 _mean = (v4f32)__msa_ld_w(mean_data, 0);

        // compute variance
        v4f32 _sqsum = (v4f32)__msa_fill_w(0);
        ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr0, 0));
            _p = __msa_fsub_w(_p, _mean);
            _sqsum = __msa_fmadd_w(_sqsum, _p, _p);
            ptr0 += 4;
        }

        float sqsum_data[4];
        __msa_st_w((v4i32)_sqsum, sqsum_data, 0);

        float a_data[4];
        float b_data[4];
        for (int i = 0; i < 4; i++)
        {
            float a = 1.f / sqrtf(sqsum_data[i] / elemcount + eps);
            a_data[i] = a;
            b_data[i] = -mean_data[i] * a;
        }

        v4f32 _a = (v4f32)__msa_ld_w(a_data, 0);
        v4f32 _b = (v4f32)__msa_ld_w(b_data, 0);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                _p = __msa_fmadd_w(_b, _p, _a);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                v4f32 _beta = __msa_fill_w_f32(beta_ptr[0]);
                _p = __msa_fmadd_w(_beta, _p, _gamma);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                _p = __msa_fmadd_w(_b, _p, _a);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __mips_msa

    // elempack == 1 or no MSA
    float mean = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v4f32 _sum = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr0, 0));
            _sum = __msa_fadd_w(_sum, _p);
            ptr0 += 4;
        }
        mean += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            mean += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }
        mean /= size;
    }

    float var = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v4f32 _mean = __msa_fill_w_f32(mean);
        v4f32 _sqsum = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr0, 0));
            _p = __msa_fsub_w(_p, _mean);
            _sqsum = __msa_fmadd_w(_sqsum, _p, _p);
            ptr0 += 4;
        }
        var += __msa_reduce_fadd_w(_sqsum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]) - mean;
            var += v * v;
            ptr0++;
        }
        var = 1.f / sqrtf(var / size + eps);
    }

    const float bias = -mean * var;

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __mips_msa
        v4f32 _a = __msa_fill_w_f32(var);
        v4f32 _b = __msa_fill_w_f32(bias);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
            v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
            v4f32 _beta = (v4f32)__msa_ld_w(beta_ptr, 0);
            _p = __msa_fmadd_w(_b, _p, _a);
            _p = __msa_fmadd_w(_beta, _p, _gamma);
            __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
            ptr += 4;
            gamma_ptr += 4;
            beta_ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16((v * var + bias) * gamma_ptr[0] + beta_ptr[0]);
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __mips_msa
        v4f32 _a = __msa_fill_w_f32(var);
        v4f32 _b = __msa_fill_w_f32(bias);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
            _p = __msa_fmadd_w(_b, _p, _a);
            __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16(v * var + bias);
            ptr++;
        }
    }
}

static void layernorm_mips(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _sum = (v4f32)__msa_fill_w(0);
        const float* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _sum = __msa_fadd_w(_sum, _p);
            ptr0 += 4;
        }

        float sum_data[4];
        __msa_st_w((v4i32)_sum, sum_data, 0);

        float mean_data[4];
        for (int i = 0; i < 4; i++)
        {
            mean_data[i] = sum_data[i] / elemcount;
        }
        v4f32 _mean = (v4f32)__msa_ld_w(mean_data, 0);

        v4f32 _sqsum = (v4f32)__msa_fill_w(0);
        ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _p = __msa_fsub_w(_p, _mean);
            _sqsum = __msa_fmadd_w(_sqsum, _p, _p);
            ptr0 += 4;
        }

        float sqsum_data[4];
        __msa_st_w((v4i32)_sqsum, sqsum_data, 0);

        float a_data[4];
        float b_data[4];
        for (int i = 0; i < 4; i++)
        {
            float a = 1.f / sqrtf(sqsum_data[i] / elemcount + eps);
            a_data[i] = a;
            b_data[i] = -mean_data[i] * a;
        }

        v4f32 _a = (v4f32)__msa_ld_w(a_data, 0);
        v4f32 _b = (v4f32)__msa_ld_w(b_data, 0);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __msa_fmadd_w(_b, _p, _a);
                v4f32 _gamma = __msa_fill_w_f32(gamma_ptr[0]);
                v4f32 _beta = __msa_fill_w_f32(beta_ptr[0]);
                _p = __msa_fmadd_w(_beta, _p, _gamma);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __msa_fmadd_w(_b, _p, _a);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __mips_msa

    float mean = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v4f32 _sum = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _sum = __msa_fadd_w(_sum, _p);
            ptr0 += 4;
        }
        mean += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
        mean /= size;
    }

    float var = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __mips_msa
        v4f32 _mean = __msa_fill_w_f32(mean);
        v4f32 _sqsum = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _p = __msa_fsub_w(_p, _mean);
            _sqsum = __msa_fmadd_w(_sqsum, _p, _p);
            ptr0 += 4;
        }
        var += __msa_reduce_fadd_w(_sqsum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
        var = 1.f / sqrtf(var / size + eps);
    }

    const float bias = -mean * var;

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __mips_msa
        v4f32 _a = __msa_fill_w_f32(var);
        v4f32 _b = __msa_fill_w_f32(bias);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr, 0);
            v4f32 _beta = (v4f32)__msa_ld_w(beta_ptr, 0);
            _p = __msa_fmadd_w(_b, _p, _a);
            _p = __msa_fmadd_w(_beta, _p, _gamma);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
            gamma_ptr += 4;
            beta_ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * var + bias) * gamma_ptr[0] + beta_ptr[0];
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __mips_msa
        v4f32 _a = __msa_fill_w_f32(var);
        v4f32 _b = __msa_fill_w_f32(bias);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __msa_fmadd_w(_b, _p, _a);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * var + bias;
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

    return 0;
}

#if NCNN_BF16
int LayerNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        int elemcount = w * elempack;
        layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, elemcount, elempack);
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
                    unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q).row<unsigned short>(i);
                    layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q);
                layernorm_mips_bf16(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
