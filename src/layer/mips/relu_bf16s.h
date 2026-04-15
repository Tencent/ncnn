// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef RELU_MIPS_BF16S_H
#define RELU_MIPS_BF16S_H

static void relu_bf16s(Mat& a, float slope, const Option& opt)
{
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = a.channel(q);

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                _p = __msa_fmax_w(_p, _zero);
                float2bfloat_msa_store(_p, ptr);
                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f) v = 0.f;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = a.channel(q);

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _pos = __msa_fmax_w(_zero, _p);
                v4f32 _neg = __msa_fmin_w(_zero, _p);
                _p = __msa_fadd_w(_pos, __msa_fmul_w(_slope, _neg));
                float2bfloat_msa_store(_p, ptr);
                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f) v *= slope;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
}

#endif // RELU_MIPS_BF16S_H
