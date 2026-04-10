// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

ReLU_mips::ReLU_mips()
{
#if __mips_msa
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ReLU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        if (slope == 0.f)
        {
            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __msa_fmax_w(_p, _zero);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;
                ptr++;
            }
        }
        else
        {
            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int ReLU_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                _p = __msa_fmax_w(_p, _zero);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
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
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                v4f32 _pos = __msa_fmax_w(_p, _zero);
                v4f32 _neg = __msa_fmin_w(_p, _zero);
                _p = __msa_fadd_w(_pos, __msa_fmul_w(_slope, _neg));
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
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

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
