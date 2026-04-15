// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SELU_MIPS_BF16S_H
#define SELU_MIPS_BF16S_H

static void selu_bf16s(Mat& a, float alpha, float lambda, const Option& opt)
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
        v4f32 _zero = (v4f32)__msa_fill_w_f32(0.f);
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(alpha);
        v4f32 _lambda = (v4f32)__msa_fill_w_f32(lambda);
        v4f32 _alpha_lambda = (v4f32)__msa_fill_w_f32(alpha * lambda);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _pos = __msa_fmul_w(_lambda, _p);
            v4f32 _neg = __msa_fmul_w(__msa_fsub_w(__msa_fmul_w(exp_ps(_p), _alpha), _alpha), _lambda);
            v4i32 _mask = __msa_fclt_w(_p, _zero);
            _p = (v4f32)__msa_bsel_v((v16u8)_mask, (v16u8)_pos, (v16u8)_neg);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < 0.f) v = (expf(v) * alpha - alpha) * lambda;
            else v = v * lambda;
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // SELU_MIPS_BF16S_H
