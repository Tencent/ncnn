// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "absval_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

AbsVal_loongarch::AbsVal_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int AbsVal_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 16)
        return forward_inplace_bf16s_fp16s(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256i _p = __lasx_xvld(ptr, 0);
            __m256i _outp = __lasx_xvbitclri_w(_p, 31);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128i _p = __lsx_vld(ptr, 0);
            __m128i _outp = __lsx_vbitclri_w(_p, 31);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *ptr > 0 ? *ptr : -*ptr;

            ptr++;
        }
    }

    return 0;
}

int AbsVal_loongarch::forward_inplace_bf16s_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    // fp16/bf16 abs: sign bit is bit 15 for both formats.
    // Reinterpret pairs of 16-bit values as 32-bit and apply AND with
    // 0x7fff7fff to clear both sign bits in one 32-bit operation.
    // No fp32 round-trip required.

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256i _sign_mask256 = (__m256i)__lasx_xvreplgr2vr_w(0x7fff7fff);
        for (; i + 15 < size; i += 16)
        {
            __m256i _p = __lasx_xvld(ptr, 0);
            __m256i _outp = __lasx_xvand_v(_p, _sign_mask256);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 16;
        }
#endif // __loongarch_asx
        __m128i _sign_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fff7fff);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p = __lsx_vld(ptr, 0);
            __m128i _outp = __lsx_vand_v(_p, _sign_mask);
            __lsx_vst(_outp, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *ptr & 0x7fffu;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
