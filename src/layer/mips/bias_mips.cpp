// Copyright 2019 Leo <leo@nullptr.com.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Bias_mips::Bias_mips()
{
#if __mips_msa
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Bias_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __mips_msa
        v4f32 _bias = (elempack == 4) ? (v4f32)__msa_ld_w(bias_ptr + q * 4, 0) : (v4f32)__msa_fill_w_f32(bias_ptr[q]);
#endif
        float bias = bias_ptr[q];

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __msa_fadd_w(_p, _bias);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr += bias;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Bias_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

#if __mips_msa
        v4f32 _bias = (elempack == 4) ? (v4f32)__msa_ld_w(bias_ptr + q * 4, 0) : (v4f32)__msa_fill_w_f32(bias_ptr[q]);
#endif
        float bias = bias_ptr[q];

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _p = __msa_fadd_w(_p, _bias);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v + bias;
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
