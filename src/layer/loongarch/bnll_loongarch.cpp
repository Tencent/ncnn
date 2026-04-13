// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "loongarch_usability.h"
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

BNLL_loongarch::BNLL_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BNLL_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _zero8 = (__m256)__lasx_xvreplgr2vr_w(0);
        __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        __m256i _abs_mask8 = __lasx_xvreplgr2vr_w(0x7fffffff);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);

            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _abs_p = (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask8);
            __m256 _tmp = log256_ps(__lasx_xvfadd_s(_one8, exp256_ps((__m256)__lasx_xvbitrevi_w((__m256i)_abs_p, 31))));
            __m256 _outp = __lasx_xvfadd_s(__lasx_xvfmax_s(_p, _zero8), _tmp);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif
        __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128i _abs_mask4 = __lsx_vreplgr2vr_w(0x7fffffff);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _abs_p = (__m128)__lsx_vand_v((__m128i)_p, _abs_mask4);
            __m128 _tmp = log_ps(__lsx_vfadd_s(_one4, exp_ps((__m128)__lsx_vbitrevi_w((__m128i)_abs_p, 31))));
            __m128 _outp = __lsx_vfadd_s(__lsx_vfmax_s(_p, _zero4), _tmp);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif
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
int BNLL_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
            __m256 _p = bfloat2float_avx((__m128i*)ptr);
            __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _one = (__m256)__lasx_xvreplfr2vr_s(1.f);
            __m256 _abs_p = (__m256)__lasx_xvand_v((__m256i)_p, __lasx_xvreplgr2vr_w(0x7fffffff));
            __m256 _tmp = log256_ps(__lasx_xvfadd_s(_one, exp256_ps((__m256)__lasx_xvbitrevi_w((__m256i)_abs_p, 31))));
            __m256 _outp = __lasx_xvfadd_s(__lasx_xvfmax_s(_p, _zero), _tmp);
            __lsx_vst(float2bfloat_avx(_outp), ptr, 0);
            ptr += 8;
        }
#endif
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i*)ptr);
            __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _abs_p = (__m128)__lsx_vand_v((__m128i)_p, __lsx_vreplgr2vr_w(0x7fffffff));
            __m128 _tmp = log_ps(__lsx_vfadd_s(_one, exp_ps((__m128)__lsx_vbitrevi_w((__m128i)_abs_p, 31))));
            __m128 _outp = __lsx_vfadd_s(__lsx_vfmax_s(_p, _zero), _tmp);
            __lsx_vstelm_d(float2bfloat_sse(_outp), ptr, 0, 0);
            ptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v > 0)
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
