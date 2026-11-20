// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Dropout_powerpc::Dropout_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Dropout_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ALTIVEC__
        const v4f32 _scale = vec_splats(scale);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            _p = __ncnn_altivec_mul(_p, _scale);
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            *ptr = *ptr * scale;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Dropout_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ALTIVEC__
        const v4f32 _scale = vec_splats(scale);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
            _p = __ncnn_altivec_mul(_p, _scale);
            __ncnn_altivec_f32_to_bf16(ptr, _p);
            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            *ptr = float32_to_bfloat16(v * scale);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
