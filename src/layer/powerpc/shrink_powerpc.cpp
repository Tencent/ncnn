// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shrink_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Shrink_powerpc::Shrink_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Shrink_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        v4f32 _zero = vec_splats(0.f);
        v4f32 _lambd = vec_splats(lambd);
        v4f32 _neg_lambd = vec_sub(_zero, _lambd);
        v4f32 _bias = vec_splats(bias);
        v4f32 _neg_bias = vec_sub(_zero, _bias);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);

            // x < -lambd ? x + bias : x > lambd ? x - bias : 0
            v4i32 _ltmask = __ncnn_altivec_cmplt(_p, _neg_lambd);
            v4i32 _gtmask = __ncnn_altivec_cmpgt(_p, _lambd);
            v4f32 _ltv = vec_add(_p, _bias);
            v4f32 _gtv = vec_add(_p, _neg_bias);
            v4f32 _r = __ncnn_altivec_sel(_gtmask, _gtv, _zero);
            _p = __ncnn_altivec_sel(_ltmask, _ltv, _r);

            __ncnn_altivec_store(ptr, _p);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            *ptr = *ptr < -lambd ? *ptr + bias : *ptr > lambd ? *ptr - bias : 0.f;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Shrink_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
            v = v < -lambd ? v + bias : v > lambd ? v - bias : 0.f;
            ptr[i] = float32_to_bfloat16(v);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
