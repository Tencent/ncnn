// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SCALE_MIPS_BF16S_H
#define SCALE_MIPS_BF16S_H

static void scale_bf16s_msa(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
#if __mips_msa
    v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale, 0) : (v4f32)__msa_fill_w_f32(scale[0]);
    v4f32 _b = (elempack == 4) ? (v4f32)__msa_ld_w(bias, 0) : (v4f32)__msa_fill_w_f32(bias[0]);
#endif
    float s = scale[0];
    float b = bias[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __msa_fmadd_w(_b, _p, _s);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s + b);
        ptr++;
    }
}

static void scale_bf16s_no_bias_msa(unsigned short* ptr, const float* scale, int size, int elempack)
{
#if __mips_msa
    v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale, 0) : (v4f32)__msa_fill_w_f32(scale[0]);
#endif
    float s = scale[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __msa_fmul_w(_p, _s);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s);
        ptr++;
    }
}

static void scale_bf16s_per_element_msa(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
        v4f32 _b = (v4f32)__msa_ld_w(bias + i, 0);
        _p = __msa_fmadd_w(_b, _p, _s);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
    }
}

static void scale_bf16s_no_bias_per_element_msa(unsigned short* ptr, const float* scale, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
        _p = __msa_fmul_w(_p, _s);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
    }
}

#endif // SCALE_MIPS_BF16S_H
