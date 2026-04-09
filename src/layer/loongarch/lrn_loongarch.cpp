// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

LRN_loongarch::LRN_loongarch()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int LRN_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    Mat square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

        int i = 0;
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _outp = __lasx_xvfmul_s(_p, _p);
            __lasx_xvst(_outp, outptr, 0);

            ptr += 8;
            outptr += 8;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _outp = __lsx_vfmul_s(_p, _p);
            __lsx_vst(_outp, outptr, 0);

            ptr += 4;
            outptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *outptr = *ptr * *ptr;
            ptr++;
            outptr++;
        }
    }

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        Mat square_sum;
        square_sum.create(w, h, channels, elemsize, opt.workspace_allocator);
        if (square_sum.empty())
            return -100;
        square_sum.fill(0.f);

        const float alpha_div_size = alpha / local_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int p = q - local_size / 2; p <= q + local_size / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                float* ssptr = square_sum.channel(q);

                int i = 0;
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __m256 _sp = (__m256)__lasx_xvld(sptr, 0);
                    __m256 _ssp = (__m256)__lasx_xvld(ssptr, 0);
                    _ssp = __lasx_xvfadd_s(_ssp, _sp);
                    __lasx_xvst(_ssp, ssptr, 0);

                    sptr += 8;
                    ssptr += 8;
                }
#endif // __loongarch_asx
#if __loongarch_sx
                for (; i + 3 < size; i += 4)
                {
                    __m128 _sp = (__m128)__lsx_vld(sptr, 0);
                    __m128 _ssp = (__m128)__lsx_vld(ssptr, 0);
                    _ssp = __lsx_vfadd_s(_ssp, _sp);
                    __lsx_vst(_ssp, ssptr, 0);

                    sptr += 4;
                    ssptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
            }

            float* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

            int i = 0;
#if __loongarch_asx
            __m256 _bias = (__m256)__lasx_xvreplfr2vr_s(bias);
            __m256 _ads = (__m256)__lasx_xvreplfr2vr_s(alpha_div_size);
            __m256 _mb = (__m256)__lasx_xvreplfr2vr_s(-beta);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _ssp = (__m256)__lasx_xvld(ssptr, 0);
                _ssp = __lasx_xvfmul_s(_ssp, _ads);
                _ssp = __lasx_xvfadd_s(_ssp, _bias);
                _ssp = pow256_ps(_ssp, _mb);
                _p = __lasx_xvfmul_s(_p, _ssp);
                __lasx_xvst(_p, ptr, 0);

                ssptr += 8;
                ptr += 8;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            __m128 _bias4 = (__m128)__lsx_vreplfr2vr_s(bias);
            __m128 _ads4 = (__m128)__lsx_vreplfr2vr_s(alpha_div_size);
            __m128 _mb4 = (__m128)__lsx_vreplfr2vr_s(-beta);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _ssp = (__m128)__lsx_vld(ssptr, 0);
                _ssp = __lsx_vfmul_s(_ssp, _ads4);
                _ssp = __lsx_vfadd_s(_ssp, _bias4);
                _ssp = pow_ps(_ssp, _mb4);
                _p = __lsx_vfmul_s(_p, _ssp);
                __lsx_vst(_p, ptr, 0);

                ssptr += 4;
                ptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *ptr = *ptr * powf(bias + alpha_div_size * *ssptr, -beta);
                ssptr++;
                ptr++;
            }
        }
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        int outw = w;
        int outh = h;

        Mat square_blob_bordered = square_blob;
        int pad = local_size / 2;
        if (pad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            opt_b.use_packing_layout = false;
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt_b);
            if (square_blob_bordered.empty())
                return -100;

            w = square_blob_bordered.w;
            h = square_blob_bordered.h;
        }

        const int maxk = local_size * local_size;
        const float alpha_div_size = alpha / maxk;

        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - local_size;
            for (int i = 0; i < local_size; i++)
            {
                for (int j = 0; j < local_size; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const Mat m = square_blob_bordered.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i) + j;

                    float ss = 0.f;
                    for (int k = 0; k < maxk; k++)
                    {
                        ss += sptr[space_ofs[k]];
                    }

                    ptr[j] = ptr[j] * powf(bias + alpha_div_size * ss, -beta);
                }

                ptr += outw;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int LRN_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
