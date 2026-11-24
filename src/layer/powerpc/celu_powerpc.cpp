// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "celu_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

CELU_powerpc::CELU_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int CELU_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        v4f32 _inv_alpha = __ncnn_altivec_div2(_one, _alpha);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);

            // (exp(x / alpha) - 1) * alpha
            v4f32 _neg = __ncnn_altivec_mul(_alpha, vec_sub(exp_ps(__ncnn_altivec_mul(_p, _inv_alpha)), _one));
            v4i32 _ltmask = __ncnn_altivec_cmplt(_p, _zero);
            _p = __ncnn_altivec_sel(_ltmask, _neg, _p);

            __ncnn_altivec_store(ptr, _p);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr / alpha) - 1.f) * alpha;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int CELU_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
            if (v < 0.f)
                v = (expf(v / alpha) - 1.f) * alpha;
            ptr[i] = float32_to_bfloat16(v);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
