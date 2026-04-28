// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

RotaryEmbed_loongarch::RotaryEmbed_loongarch()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int RotaryEmbed_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    return RotaryEmbed::forward(bottom_blobs, top_blobs, opt);
}

#if NCNN_BF16
int RotaryEmbed_loongarch::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            if (interleaved)
            {
                const unsigned short* ptr = head.row<const unsigned short>(i);
                const unsigned short* cos_ptr = cos_cache.row<const unsigned short>(i);
                const unsigned short* sin_ptr = sin_cache.row<const unsigned short>(i);
                unsigned short* outptr = out_head.row<unsigned short>(i);

                int j = 0;
#if __loongarch_sx
#if __loongarch_asx
                __m256i _signmask256 = __lasx_xvreplgr2vr_w(0);
                _signmask256 = __lasx_xvinsgr2vr_w(_signmask256, -1, 1);
                _signmask256 = __lasx_xvinsgr2vr_w(_signmask256, -1, 3);
                _signmask256 = __lasx_xvinsgr2vr_w(_signmask256, -1, 5);
                _signmask256 = __lasx_xvinsgr2vr_w(_signmask256, -1, 7);

                for (; j + 7 < embed_dim / 2; j += 8)
                {
                    __m256 _a0 = bfloat2float_lasx((const __m128i*)ptr);
                    __m256 _a1 = bfloat2float_lasx((const __m128i*)(ptr + 8));

                    __m256 _c8 = bfloat2float_lasx((const __m128i*)cos_ptr);
                    __m256 _s8 = bfloat2float_lasx((const __m128i*)sin_ptr);

                    __m128 _c4lo = (__m128)__lasx_extract_lo128((__m256i)_c8);
                    __m128 _c4hi = (__m128)__lasx_extract_hi128((__m256i)_c8);
                    __m128 _s4lo = (__m128)__lasx_extract_lo128((__m256i)_s8);
                    __m128 _s4hi = (__m128)__lasx_extract_hi128((__m256i)_s8);

                    __m256 _c0 = combine4x2_ps((__m128)__lsx_vilvr_w((__m128i)_c4lo, (__m128i)_c4lo), (__m128)__lsx_vilvh_w((__m128i)_c4lo, (__m128i)_c4lo));
                    __m256 _c1 = combine4x2_ps((__m128)__lsx_vilvr_w((__m128i)_c4hi, (__m128i)_c4hi), (__m128)__lsx_vilvh_w((__m128i)_c4hi, (__m128i)_c4hi));
                    __m256 _s0 = combine4x2_ps((__m128)__lsx_vilvr_w((__m128i)_s4lo, (__m128i)_s4lo), (__m128)__lsx_vilvh_w((__m128i)_s4lo, (__m128i)_s4lo));
                    __m256 _s1 = combine4x2_ps((__m128)__lsx_vilvr_w((__m128i)_s4hi, (__m128i)_s4hi), (__m128)__lsx_vilvh_w((__m128i)_s4hi, (__m128i)_s4hi));

                    __m256 _swap0 = (__m256)__lasx_xvshuf4i_w((__m256i)_a0, _LSX_SHUFFLE(2, 3, 0, 1));
                    __m256 _swap1 = (__m256)__lasx_xvshuf4i_w((__m256i)_a1, _LSX_SHUFFLE(2, 3, 0, 1));

                    __m256 _ac0 = __lasx_xvfmul_s(_a0, _c0);
                    __m256 _ac1 = __lasx_xvfmul_s(_a1, _c1);
                    __m256 _ss0 = __lasx_xvfmul_s(_swap0, _s0);
                    __m256 _ss1 = __lasx_xvfmul_s(_swap1, _s1);
                    __m256 _y0sub = __lasx_xvfsub_s(_ac0, _ss0);
                    __m256 _y0add = __lasx_xvfadd_s(_ac0, _ss0);
                    __m256 _y1sub = __lasx_xvfsub_s(_ac1, _ss1);
                    __m256 _y1add = __lasx_xvfadd_s(_ac1, _ss1);

                    __m256 _y0 = (__m256)__lasx_xvbitsel_v((__m256i)_y0sub, (__m256i)_y0add, _signmask256);
                    __m256 _y1 = (__m256)__lasx_xvbitsel_v((__m256i)_y1sub, (__m256i)_y1add, _signmask256);

                    __lsx_vst(float2bfloat_lasx(_y0), outptr, 0);
                    __lsx_vst(float2bfloat_lasx(_y1), outptr + 8, 0);

                    ptr += 16;
                    outptr += 16;
                    cos_ptr += 8;
                    sin_ptr += 8;
                }
#endif // __loongarch_asx
                __m128i _signmask = __lsx_vreplgr2vr_w(0);
                _signmask = __lsx_vinsgr2vr_w(_signmask, -1, 1);
                _signmask = __lsx_vinsgr2vr_w(_signmask, -1, 3);

                for (; j + 3 < embed_dim / 2; j += 4)
                {
                    __m128 _a0 = bfloat2float_lsx(ptr);
                    __m128 _a1 = bfloat2float_lsx(ptr + 4);

                    __m128 _c4 = bfloat2float_lsx(cos_ptr);
                    __m128 _s4 = bfloat2float_lsx(sin_ptr);

                    __m128 _clo = (__m128)__lsx_vilvr_w((__m128i)_c4, (__m128i)_c4);
                    __m128 _chi = (__m128)__lsx_vilvh_w((__m128i)_c4, (__m128i)_c4);
                    __m128 _slo = (__m128)__lsx_vilvr_w((__m128i)_s4, (__m128i)_s4);
                    __m128 _shi = (__m128)__lsx_vilvh_w((__m128i)_s4, (__m128i)_s4);

                    __m128 _swap0 = (__m128)__lsx_vshuf4i_w((__m128i)_a0, _LSX_SHUFFLE(2, 3, 0, 1));
                    __m128 _swap1 = (__m128)__lsx_vshuf4i_w((__m128i)_a1, _LSX_SHUFFLE(2, 3, 0, 1));

                    __m128 _ac0 = __lsx_vfmul_s(_a0, _clo);
                    __m128 _ac1 = __lsx_vfmul_s(_a1, _chi);
                    __m128 _ss0 = __lsx_vfmul_s(_swap0, _slo);
                    __m128 _ss1 = __lsx_vfmul_s(_swap1, _shi);
                    __m128 _y0sub = __lsx_vfsub_s(_ac0, _ss0);
                    __m128 _y0add = __lsx_vfadd_s(_ac0, _ss0);
                    __m128 _y1sub = __lsx_vfsub_s(_ac1, _ss1);
                    __m128 _y1add = __lsx_vfadd_s(_ac1, _ss1);

                    __m128 _y0 = (__m128)__lsx_vbitsel_v((__m128i)_y0sub, (__m128i)_y0add, _signmask);
                    __m128 _y1 = (__m128)__lsx_vbitsel_v((__m128i)_y1sub, (__m128i)_y1add, _signmask);

                    __lsx_vst(float2bfloat_lsx(_y0, _y1), outptr, 0);

                    ptr += 8;
                    outptr += 8;
                    cos_ptr += 4;
                    sin_ptr += 4;
                }
#endif // __loongarch_sx
                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = bfloat16_to_float32(ptr[0]);
                    const float x1 = bfloat16_to_float32(ptr[1]);
                    const float cos_val = bfloat16_to_float32(*cos_ptr++);
                    const float sin_val = bfloat16_to_float32(*sin_ptr++);

                    outptr[0] = float32_to_bfloat16(x0 * cos_val - x1 * sin_val);
                    outptr[1] = float32_to_bfloat16(x0 * sin_val + x1 * cos_val);

                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const unsigned short* ptr0 = head.row<const unsigned short>(i);
                const unsigned short* ptr1 = ptr0 + embed_dim / 2;
                const unsigned short* cos_ptr = cos_cache.row<const unsigned short>(i);
                const unsigned short* sin_ptr = sin_cache.row<const unsigned short>(i);
                unsigned short* outptr0 = out_head.row<unsigned short>(i);
                unsigned short* outptr1 = outptr0 + embed_dim / 2;

                int j = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; j + 7 < embed_dim / 2; j += 8)
                {
                    __m256 _x0 = bfloat2float_lasx((const __m128i*)ptr0);
                    __m256 _x1 = bfloat2float_lasx((const __m128i*)ptr1);
                    __m256 _c = bfloat2float_lasx((const __m128i*)cos_ptr);
                    __m256 _s = bfloat2float_lasx((const __m128i*)sin_ptr);

                    __m256 _y0 = __lasx_xvfsub_s(__lasx_xvfmul_s(_x0, _c), __lasx_xvfmul_s(_x1, _s));
                    __m256 _y1 = __lasx_xvfmadd_s(_x0, _s, __lasx_xvfmul_s(_x1, _c));

                    __lsx_vst(float2bfloat_lasx(_y0), outptr0, 0);
                    __lsx_vst(float2bfloat_lasx(_y1), outptr1, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    cos_ptr += 8;
                    sin_ptr += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
#endif // __loongarch_asx
                for (; j + 3 < embed_dim / 2; j += 4)
                {
                    __m128 _x0 = bfloat2float_lsx(ptr0);
                    __m128 _x1 = bfloat2float_lsx(ptr1);
                    __m128 _c = bfloat2float_lsx(cos_ptr);
                    __m128 _s = bfloat2float_lsx(sin_ptr);

                    __m128 _y0 = __lsx_vfsub_s(__lsx_vfmul_s(_x0, _c), __lsx_vfmul_s(_x1, _s));
                    __m128 _y1 = __lsx_vfmadd_s(_x0, _s, __lsx_vfmul_s(_x1, _c));

                    __lsx_vstelm_d(float2bfloat_lsx(_y0), outptr0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_y1), outptr1, 0, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    cos_ptr += 4;
                    sin_ptr += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __loongarch_sx
                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = bfloat16_to_float32(*ptr0++);
                    const float x1 = bfloat16_to_float32(*ptr1++);
                    const float cos_val = bfloat16_to_float32(*cos_ptr++);
                    const float sin_val = bfloat16_to_float32(*sin_ptr++);

                    *outptr0++ = float32_to_bfloat16(x0 * cos_val - x1 * sin_val);
                    *outptr1++ = float32_to_bfloat16(x0 * sin_val + x1 * cos_val);
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
