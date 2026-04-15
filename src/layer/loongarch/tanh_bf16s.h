// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TANH_LOONGARCH_BF16S_H
#define TANH_LOONGARCH_BF16S_H

static void tanh_bf16s(Mat& a, const Option& opt)
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
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
            _p = tanh256_ps(_p);
            __lsx_vst(float2bfloat_avx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
            _p = tanh_ps(_p);
            __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = tanhf(v);
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // TANH_LOONGARCH_BF16S_H
