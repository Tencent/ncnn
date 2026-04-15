// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef BATCHNORM_MIPS_BF16S_H
#define BATCHNORM_MIPS_BF16S_H

static void batchnorm_bf16s_msa(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
#if __mips_msa
    v4f32 _a = (elempack == 4) ? (v4f32)__msa_ld_w(a, 0) : (v4f32)__msa_fill_w_f32(a[0]);
    v4f32 _b = (elempack == 4) ? (v4f32)__msa_ld_w(b, 0) : (v4f32)__msa_fill_w_f32(b[0]);
#endif
    float sa = a[0];
    float sb = b[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __msa_fmadd_w(_a, _p, _b);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(sb * bfloat16_to_float32(*ptr) + sa);
        ptr++;
    }
}

static void batchnorm_bf16s_per_element_msa(unsigned short* ptr, const float* a, const float* b, int size, int num_threads)
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
        v4f32 _a0 = (v4f32)__msa_ld_w(a + i, 0);
        v4f32 _b0 = (v4f32)__msa_ld_w(b + i, 0);
        _p = __msa_fmadd_w(_a0, _p, _b0);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(b[i] * bfloat16_to_float32(ptr[i]) + a[i]);
    }
}

#endif // BATCHNORM_MIPS_BF16S_H
