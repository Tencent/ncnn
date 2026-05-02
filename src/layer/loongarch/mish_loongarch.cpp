// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "mish_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Mish_loongarch::Mish_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Mish_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
#if __loongarch_sx
#if __loongarch_asx
        __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmul_s(_p, tanh256_ps(log256_ps(__lasx_xvfadd_s(exp256_ps(_p), _one8))));
            __lasx_xvst(_p, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmul_s(_p, tanh_ps(log_ps(__lsx_vfadd_s(exp_ps(_p), _one4))));
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *ptr * tanhf(logf(expf(*ptr) + 1.f));
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Mish_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr);
            _p = __lasx_xvfmul_s(_p, tanh256_ps(log256_ps(__lasx_xvfadd_s(exp256_ps(_p), (__m256)__lasx_xvreplfr2vr_s(1.f)))));
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i*)ptr);
            _p = __lsx_vfmul_s(_p, tanh_ps(log_ps(__lsx_vfadd_s(exp_ps(_p), (__m128)__lsx_vreplfr2vr_s(1.f)))));
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v * tanhf(logf(expf(v) + 1.f));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
