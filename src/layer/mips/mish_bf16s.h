// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MISH_MIPS_BF16S_H
#define MISH_MIPS_BF16S_H

static void mish_bf16s(Mat& a, const Option& opt)
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
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _p = __msa_fmul_w(_p, tanh_ps(log_ps(__msa_fadd_w(exp_ps(_p), (v4f32)__msa_fill_w_f32(1.f)))));
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v * tanhf(logf(expf(v) + 1.f));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // MISH_MIPS_BF16S_H
