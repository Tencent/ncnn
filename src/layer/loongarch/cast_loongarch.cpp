// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cast_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Cast_loongarch::Cast_loongarch()
{
    support_packing = true;
}

int Cast_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        if (type_from == 3)
        {
            Cast::forward(bottom_blob, top_blob, opt);
        }

        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        // bfloat16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 4)
    {
        top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int size = w * h * d * elempack;

    if (type_from == 1 && type_to == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 15 < size; i += 16)
            {
                __builtin_prefetch(ptr + 32);
                __m128 _p0 = (__m128)__lsx_vld(ptr, 0);
                __m128 _p1 = (__m128)__lsx_vld(ptr + 4, 0);
                __m128 _p2 = (__m128)__lsx_vld(ptr + 8, 0);
                __m128 _p3 = (__m128)__lsx_vld(ptr + 12, 0);
                __m128i _p01 = __lsx_vfcvt_h_s(_p1, _p0);
                __m128i _p23 = __lsx_vfcvt_h_s(_p3, _p2);
                __lsx_vst(_p01, outptr, 0);
                __lsx_vst(_p23, outptr + 8, 0);

                ptr += 16;
                outptr += 16;
            }
#endif // __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p0 = (__m128)__lsx_vld(ptr, 0);
                __m128 _p1 = (__m128)__lsx_vld(ptr + 4, 0);
                __m128i _p = __lsx_vfcvt_h_s(_p1, _p0);
                __lsx_vst(_p, outptr, 0);

                ptr += 8;
                outptr += 8;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = float32_to_float16(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 2 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 15 < size; i += 16)
            {
                __builtin_prefetch(ptr + 32);
                __m128i _p = __lsx_vld(ptr, 0);
                __m128i _p_high = __lsx_vld(ptr + 8, 0);
                __m128 _p0_lo = __lsx_vfcvtl_s_h(_p);
                __m128 _p1_lo = __lsx_vfcvth_s_h(_p);
                __m128 _p0_hi = __lsx_vfcvtl_s_h(_p_high);
                __m128 _p1_hi = __lsx_vfcvth_s_h(_p_high);
                __lsx_vst(_p0_lo, outptr, 0);
                __lsx_vst(_p1_lo, outptr + 4, 0);
                __lsx_vst(_p0_hi, outptr + 8, 0);
                __lsx_vst(_p1_hi, outptr + 12, 0);

                ptr += 16;
                outptr += 16;
            }
#endif // __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 16);
                __m128i _p = __lsx_vld(ptr, 0);
                __m128 _p0 = __lsx_vfcvtl_s_h(_p);
                __m128 _p1 = __lsx_vfcvth_s_h(_p);
                __lsx_vst(_p0, outptr, 0);
                __lsx_vst(_p1, outptr + 4, 0);

                ptr += 8;
                outptr += 8;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = float16_to_float32(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 3 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const signed char* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = (float)ptr[i];
            }
        }
    }

    if (type_from == 4 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                __lasx_xvst(_p, outptr, 0);
                ptr += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                __lsx_vst(_p, outptr, 0);
                ptr += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = bfloat16_to_float32(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 1 && type_to == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 15 < size; i += 16)
            {
                __m256 _p0 = (__m256)__lasx_xvld(ptr, 0);
                __m256 _p1 = (__m256)__lasx_xvld(ptr + 8, 0);
                __m256i _bfp = float2bfloat_lasx(_p0, _p1);
                __lasx_xvst(_bfp, outptr, 0);
                ptr += 16;
                outptr += 16;
            }
#endif // __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m128 _p0 = (__m128)__lsx_vld(ptr, 0);
                __m128 _p1 = (__m128)__lsx_vld(ptr + 4, 0);
                __m128i _bfp = float2bfloat_lsx(_p0, _p1);
                __lsx_vst(_bfp, outptr, 0);
                ptr += 8;
                outptr += 8;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = float32_to_bfloat16(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
