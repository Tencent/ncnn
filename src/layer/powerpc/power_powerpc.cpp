// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "power_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

Power_powerpc::Power_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Power_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ALTIVEC__
        v4f32 _power = vec_splats(power);
        v4f32 _scale = vec_splats(scale);
        v4f32 _shift = vec_splats(shift);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);

            // (shift + x * scale) ^ power
            v4f32 _v = vec_madd(_p, _scale, _shift);
            _p = pow_ps(_v, _power);

            __ncnn_altivec_store(ptr, _p);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            *ptr = powf((shift + *ptr * scale), power);

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Power_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
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

        for (int i = 0; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[i]);
            ptr[i] = float32_to_bfloat16(powf((shift + v * scale), power));
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
