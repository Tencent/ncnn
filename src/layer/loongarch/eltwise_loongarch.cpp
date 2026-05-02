// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

#if NCNN_BF16
int Eltwise_loongarch::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == 0) // Operation_PROD
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr1, 0));
                _p = __lasx_xvfmul_s(_p, _p1);
                __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1, 0));
                _p = __lsx_vfmul_s(_p, _p1);
                __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * bfloat16_to_float32(*ptr1));

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob2.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(outptr, 0));
                    __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                    _p = __lasx_xvfmul_s(_p, _p1);
                    __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                    ptr += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(outptr, 0));
                    __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                    _p = __lsx_vfmul_s(_p, _p1);
                    __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*outptr) * bfloat16_to_float32(*ptr));

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == 1) // Operation_SUM
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                    __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr1, 0));
                    _p = __lasx_xvfadd_s(_p, _p1);
                    __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                    __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1, 0));
                    _p = __lsx_vfadd_s(_p, _p1);
                    __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) + bfloat16_to_float32(*ptr1));

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob2.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(outptr, 0));
                        __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                        _p = __lasx_xvfadd_s(_p, _p1);
                        __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __loongarch_asx
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(outptr, 0));
                        __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                        _p = __lsx_vfadd_s(_p, _p1);
                        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __loongarch_sx
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*outptr) + bfloat16_to_float32(*ptr));

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                __m256 _coeff0_lasx = (__m256)__lasx_xvreplfr2vr_s(coeff0);
                __m256 _coeff1_lasx = (__m256)__lasx_xvreplfr2vr_s(coeff1);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                    __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr1, 0));
                    _p = __lasx_xvfmul_s(_p, _coeff0_lasx);
                    _p = __lasx_xvfmadd_s(_coeff1_lasx, _p1, _p);
                    __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                __m128 _coeff0 = (__m128)__lsx_vreplfr2vr_s(coeff0);
                __m128 _coeff1 = (__m128)__lsx_vreplfr2vr_s(coeff1);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                    __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1, 0));
                    _p = __lsx_vfmul_s(_p, _coeff0);
                    _p = __lsx_vfmadd_s(_coeff1, _p1, _p);
                    __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * coeff0 + bfloat16_to_float32(*ptr1) * coeff1);

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob2.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

                    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                    __m256 _coeff_lasx = (__m256)__lasx_xvreplfr2vr_s(coeff);
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(outptr, 0));
                        __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                        _p = __lasx_xvfmadd_s(_coeff_lasx, _p1, _p);
                        __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __loongarch_asx
                    __m128 _coeff = (__m128)__lsx_vreplfr2vr_s(coeff);
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(outptr, 0));
                        __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                        _p = __lsx_vfmadd_s(_coeff, _p1, _p);
                        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __loongarch_sx
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*outptr) + bfloat16_to_float32(*ptr) * coeff);

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == 2) // Operation_MAX
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr1, 0));
                _p = __lasx_xvfmax_s(_p, _p1);
                __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1, 0));
                _p = __lsx_vfmax_s(_p, _p1);
                __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob2.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(outptr, 0));
                    __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
                    _p = __lasx_xvfmax_s(_p, _p1);
                    __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                    ptr += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(outptr, 0));
                    __m128 _p1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
                    _p = __lsx_vfmax_s(_p, _p1);
                    __lsx_vstelm_d(float2bfloat_lsx(_p, _p), outptr, 0, 0);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*outptr)));

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    return 0;
}
#endif // NCNN_BF16

Eltwise_loongarch::Eltwise_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Eltwise_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);
                _p = __lasx_xvfmul_s(_p, _p1);
                __lasx_xvst(_p, outptr, 0);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                _p = __lsx_vfmul_s(_p, _p1);
                __lsx_vst(_p, outptr, 0);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = *ptr * *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __builtin_prefetch(ptr + 32);
                    __m256 _p = (__m256)__lasx_xvld(outptr, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr, 0);
                    _p = __lasx_xvfmul_s(_p, _p1);
                    __lasx_xvst(_p, outptr, 0);

                    ptr += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    __m128 _p = (__m128)__lsx_vld(outptr, 0);
                    __m128 _p1 = (__m128)__lsx_vld(ptr, 0);
                    _p = __lsx_vfmul_s(_p, _p1);
                    __lsx_vst(_p, outptr, 0);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr *= *ptr;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __builtin_prefetch(ptr + 32);
                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);
                    _p = __lasx_xvfadd_s(_p, _p1);
                    __lasx_xvst(_p, outptr, 0);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                    _p = __lsx_vfadd_s(_p, _p1);
                    __lsx_vst(_p, outptr, 0);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = *ptr + *ptr1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                    for (; i + 7 < size; i += 8)
                    {
                        __builtin_prefetch(ptr + 32);
                        __m256 _p = (__m256)__lasx_xvld(outptr, 0);
                        __m256 _p1 = (__m256)__lasx_xvld(ptr, 0);
                        _p = __lasx_xvfadd_s(_p, _p1);
                        __lasx_xvst(_p, outptr, 0);

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __loongarch_asx
                    for (; i + 3 < size; i += 4)
                    {
                        __builtin_prefetch(ptr + 16);
                        __m128 _p = (__m128)__lsx_vld(outptr, 0);
                        __m128 _p1 = (__m128)__lsx_vld(ptr, 0);
                        _p = __lsx_vfadd_s(_p, _p1);
                        __lsx_vst(_p, outptr, 0);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __loongarch_sx
                    for (; i < size; i++)
                    {
                        *outptr += *ptr;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
#if __loongarch_sx
#if __loongarch_asx
            __m256 _coeff0_256 = (__m256)__lasx_xvreplfr2vr_s(coeff0);
            __m256 _coeff1_256 = (__m256)__lasx_xvreplfr2vr_s(coeff1);
#endif // __loongarch_asx
            __m128 _coeff0 = (__m128)__lsx_vreplfr2vr_s(coeff0);
            __m128 _coeff1 = (__m128)__lsx_vreplfr2vr_s(coeff1);
#endif // __loongarch_sx
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __builtin_prefetch(ptr + 32);
                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);
                    _p = __lasx_xvfmul_s(_p, _coeff0_256);
                    _p = __lasx_xvfmadd_s(_coeff1_256, _p1, _p);
                    __lasx_xvst(_p, outptr, 0);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                    _p = __lsx_vfmul_s(_p, _coeff0);
                    _p = __lsx_vfmadd_s(_coeff1, _p1, _p);
                    __lsx_vst(_p, outptr, 0);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
#if __loongarch_sx
#if __loongarch_asx
                __m256 _coeff_256 = (__m256)__lasx_xvreplfr2vr_s(coeff);
#endif // __loongarch_asx
                __m128 _coeff = (__m128)__lsx_vreplfr2vr_s(coeff);
#endif // __loongarch_sx
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                    for (; i + 7 < size; i += 8)
                    {
                        __builtin_prefetch(ptr + 32);
                        __m256 _p = (__m256)__lasx_xvld(outptr, 0);
                        __m256 _p1 = (__m256)__lasx_xvld(ptr, 0);
                        _p = __lasx_xvfmadd_s(_coeff_256, _p1, _p);
                        __lasx_xvst(_p, outptr, 0);

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __loongarch_asx
                    for (; i + 3 < size; i += 4)
                    {
                        __builtin_prefetch(ptr + 16);
                        __m128 _p = (__m128)__lsx_vld(outptr, 0);
                        __m128 _p1 = (__m128)__lsx_vld(ptr, 0);
                        _p = __lsx_vfmadd_s(_coeff, _p1, _p);
                        __lsx_vst(_p, outptr, 0);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __loongarch_sx
                    for (; i < size; i++)
                    {
                        *outptr += *ptr * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);
                _p = __lasx_xvfmax_s(_p, _p1);
                __lasx_xvst(_p, outptr, 0);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                _p = __lsx_vfmax_s(_p, _p1);
                __lsx_vst(_p, outptr, 0);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr = std::max(*ptr, *ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __builtin_prefetch(ptr + 32);
                    __m256 _p = (__m256)__lasx_xvld(outptr, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr, 0);
                    _p = __lasx_xvfmax_s(_p, _p1);
                    __lasx_xvst(_p, outptr, 0);

                    ptr += 8;
                    outptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    __m128 _p = (__m128)__lsx_vld(outptr, 0);
                    __m128 _p1 = (__m128)__lsx_vld(ptr, 0);
                    _p = __lsx_vfmax_s(_p, _p1);
                    __lsx_vst(_p, outptr, 0);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr = std::max(*ptr, *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
