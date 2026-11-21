// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardswish_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

HardSwish_powerpc::HardSwish_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int HardSwish_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        v4f32 _one = vec_splats(1.f);
        v4f32 _alpha = vec_splats(alpha);
        v4f32 _beta = vec_splats(beta);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4f32 _outp = vec_madd(_p, _alpha, _beta);
            _outp = vec_max(_outp, _zero);
            _outp = vec_min(_outp, _one);
            _outp = __ncnn_altivec_mul(_outp, _p);
            __ncnn_altivec_store(ptr, _outp);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int HardSwish_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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

        int i = 0;
#if __ALTIVEC__
        v4f32 _zero = vec_splats(0.f);
        v4f32 _one = vec_splats(1.f);
        v4f32 _alpha = vec_splats(alpha);
        v4f32 _beta = vec_splats(beta);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
            v4f32 _outp = vec_madd(_p, _alpha, _beta);
            _outp = vec_max(_outp, _zero);
            _outp = vec_min(_outp, _one);
            _outp = __ncnn_altivec_mul(_outp, _p);
            __ncnn_altivec_f32_to_bf16(ptr, _outp);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                ;
            else
                v = v * (v * alpha + beta);
            *ptr = float32_to_bfloat16(v);

            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
