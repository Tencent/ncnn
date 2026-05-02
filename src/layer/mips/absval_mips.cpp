// Copyright 2019 Leo <leo@nullptr.com.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "absval_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

namespace ncnn {

AbsVal_mips::AbsVal_mips()
{
#if __mips_msa
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int AbsVal_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 16)
        return forward_inplace_bf16s_fp16s(bottom_top_blob, opt);

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        v4u32 _sign_mask = (v4u32)__msa_fill_w(0x7fffffff);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = (v4f32)__msa_and_v((v16u8)_p, (v16u8)_sign_mask);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *ptr > 0.f ? *ptr : -*ptr;
            ptr++;
        }
    }

    return 0;
}

int AbsVal_mips::forward_inplace_bf16s_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    // fp16/bf16 abs: sign bit is bit 15 for both formats.
    // Reinterpret pairs of 16-bit values as 32-bit and apply AND with
    // 0x7fff7fff to clear both sign bits in one 32-bit operation.
    // No fp32 round-trip required.

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        v4u32 _sign_mask = (v4u32)__msa_fill_w(0x7fff7fff);
        for (; i + 7 < size; i += 8)
        {
            v4u32 _p = (v4u32)__msa_ld_w(ptr, 0);
            _p = (v4u32)__msa_and_v((v16u8)_p, (v16u8)_sign_mask);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 8;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *ptr & 0x7fffu;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
