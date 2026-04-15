// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GELU_MIPS_BF16S_H
#define GELU_MIPS_BF16S_H

static void gelu_bf16s(Mat& a, int fast_gelu, const Option& opt)
{
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __mips_msa
        if (fast_gelu)
        {
            v4f32 _half = (v4f32)__msa_fill_w_f32(0.5f);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            v4f32 _fast1c = (v4f32)__msa_fill_w_f32(0.79788452f);
            v4f32 _fast2c = (v4f32)__msa_fill_w_f32(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);

                v4f32 _cube = __msa_fmul_w(_p, _p);
                _cube = __msa_fmul_w(_p, _cube);

                v4f32 _blob = __msa_fmul_w(_fast2c, _cube);
                _blob = __msa_fadd_w(_p, _blob);
                _blob = __msa_fmul_w(_fast1c, _blob);
                _blob = tanh_ps(_blob);
                _blob = __msa_fadd_w(_one, _blob);

                _blob = __msa_fmul_w(_half, __msa_fmul_w(_blob, _p));

                float2bfloat_msa_store(_blob, ptr);
                ptr += 4;
            }
        }
        else
        {
            v4f32 _half = (v4f32)__msa_fill_w_f32(0.5f);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            v4f32 _inv_sqrt2 = (v4f32)__msa_fill_w_f32(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);

                v4f32 _erf = erf_ps(__msa_fmul_w(_p, _inv_sqrt2));
                v4f32 _blob = __msa_fadd_w(_one, _erf);
                _blob = __msa_fmul_w(_half, __msa_fmul_w(_blob, _p));

                float2bfloat_msa_store(_blob, ptr);
                ptr += 4;
            }
        }
#endif // __mips_msa
        if (fast_gelu)
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + erff(0.70710678f * v));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
}

#endif // GELU_MIPS_BF16S_H
