// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ELU_MIPS_BF16S_H
#define ELU_MIPS_BF16S_H

static void elu_bf16s(Mat& a, float alpha, const Option& opt)
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
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        v4f32 _zero = (v4f32)__msa_fill_w_f32(0.f);
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(alpha);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _elu = __msa_fmul_w(_alpha, __msa_fsub_w(exp_ps(_p), _one));
            v4i32 _mask = __msa_fclt_w(_p, _zero);
            _p = (v4f32)__msa_bsel_v((v16u8)_mask, (v16u8)_p, (v16u8)_elu);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < 0.f) v = alpha * (expf(v) - 1.f);
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // ELU_MIPS_BF16S_H
