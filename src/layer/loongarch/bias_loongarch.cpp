// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Bias_loongarch::Bias_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Bias_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_ptr[q];

#if __loongarch_sx
#if __loongarch_asx
        int nn8 = size >> 3;
        int remain8 = size - (nn8 << 3);
#else
        int nn8 = 0;
        int remain8 = size;
#endif // __loongarch_asx
        int nn4 = remain8 >> 2;
        int remain = remain8 - (nn4 << 2);
#else
        int remain = size;
#endif // __loongarch_sx

#if __loongarch_sx
#if __loongarch_asx
        __m256 _bias256 = (__m256)__lasx_xvreplfr2vr_s(bias);
        for (; nn8 > 0; nn8--)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _outp = __lasx_xvfadd_s(_p, _bias256);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _bias = (__m128)__lsx_vreplfr2vr_s(bias);
        for (; nn4 > 0; nn4--)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _outp = __lsx_vfadd_s(_p, _bias);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx

        for (; remain > 0; remain--)
        {
            *ptr = *ptr + bias;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Bias_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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

        float bias = bias_ptr[q];

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _bias256 = (__m256)__lasx_xvreplfr2vr_s(bias);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfadd_s(_p, _bias256);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _bias = (__m128)__lsx_vreplfr2vr_s(bias);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
            _p = __lsx_vfadd_s(_p, _bias);
            __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
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
