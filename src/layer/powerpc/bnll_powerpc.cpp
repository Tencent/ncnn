// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

BNLL_powerpc::BNLL_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BNLL_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4f32 _abs_p = __ncnn_altivec_clear_sign(_p);
            v4f32 _tmp = log_ps(vec_add(_one, exp_ps(__ncnn_altivec_flip_sign(_abs_p))));
            v4f32 _outp = vec_add(vec_max(_p, _zero), _tmp);
            __ncnn_altivec_store(ptr, _outp);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            if (*ptr > 0.f)
                *ptr = *ptr + logf(1.f + expf(-*ptr));
            else
                *ptr = logf(1.f + expf(*ptr));

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int BNLL_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
            v4f32 _abs_p = __ncnn_altivec_clear_sign(_p);
            v4f32 _tmp = log_ps(vec_add(_one, exp_ps(__ncnn_altivec_flip_sign(_abs_p))));
            v4f32 _outp = vec_add(vec_max(_p, _zero), _tmp);
            __ncnn_altivec_f32_to_bf16(ptr, _outp);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v > 0.f)
                v = v + logf(1.f + expf(-v));
            else
                v = logf(1.f + expf(v));
            *ptr = float32_to_bfloat16(v);

            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
