// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

ReLU_loongarch::ReLU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ReLU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero256 = (__m256)__lasx_xvreplgr2vr_w(0);
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmax_s(_p, _zero256);
                __lasx_xvst(_p, ptr, 0);

                ptr += 8;
            }
#endif // __loongarch_asx
            __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmax_s(_p, _zero);
                __lsx_vst(_p, ptr, 0);

                ptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero256 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _slope256 = (__m256)__lasx_xvreplfr2vr_s(slope);
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero256);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope256);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(_p, ptr, 0);

                ptr += 8;
            }
#endif // __loongarch_asx
            __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _slope = (__m128)__lsx_vreplfr2vr_s(slope);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero);
                __m128 _ps = __lsx_vfmul_s(_p, _slope);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(_p, ptr, 0);

                ptr += 4;
            }
#endif // __loongarch_sx
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
int ReLU_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                _p = __lasx_xvfmax_s(_p, (__m256)__lasx_xvreplgr2vr_w(0));
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                _p = __lsx_vfmax_s(_p, (__m128)__lsx_vreplgr2vr_w(0));
                __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                ptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0)
                    v = 0;
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
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _slope_lasx = (__m256)__lasx_xvreplfr2vr_s(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope_lasx);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
            }
#endif // __loongarch_asx
            {
                __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _slope4 = (__m128)__lsx_vreplfr2vr_s(slope);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                    __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero4);
                    __m128 _ps = __lsx_vfmul_s(_p, _slope4);
                    _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                    __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                    ptr += 4;
                }
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0)
                    v *= slope;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
