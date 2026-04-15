// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PRELU_MIPS_BF16S_H
#define PRELU_MIPS_BF16S_H

static void prelu_bf16s_msa(unsigned short* ptr, const float* slope, int size, int elempack)
{
#if __mips_msa
    v4f32 _slope = (elempack == 4) ? (v4f32)__msa_ld_w(slope, 0) : (v4f32)__msa_fill_w_f32(slope[0]);
    v4f32 _zero = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float s = slope[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        v4f32 _ps = __msa_fmul_w(_p, _slope);
        v4i32 _mask = __msa_fclt_w(_p, _zero);
        _p = (v4f32)__msa_bsel_v((v16u8)_mask, (v16u8)_p, (v16u8)_ps);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr);
        if (v < 0.f)
            v *= s;
        *ptr = float32_to_bfloat16(v);
        ptr++;
    }
}

static void prelu_bf16s_per_element_msa(unsigned short* ptr, const float* slope, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _slope = (v4f32)__msa_ld_w(slope + i, 0);
        v4f32 _ps = __msa_fmul_w(_p, _slope);
        v4i32 _mask = __msa_fclt_w(_p, _zero);
        _p = (v4f32)__msa_bsel_v((v16u8)_mask, (v16u8)_p, (v16u8)_ps);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        if (v < 0.f)
            v *= slope[i];
        ptr[i] = float32_to_bfloat16(v);
    }
}

static void prelu_bf16s_single_slope_msa(unsigned short* ptr, float slope, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _ps = __msa_fmul_w(_p, _slope);
        v4i32 _mask = __msa_fclt_w(_p, _zero);
        _p = (v4f32)__msa_bsel_v((v16u8)_mask, (v16u8)_p, (v16u8)_ps);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        if (v < 0.f)
            v *= slope;
        ptr[i] = float32_to_bfloat16(v);
    }
}

#endif // PRELU_MIPS_BF16S_H
