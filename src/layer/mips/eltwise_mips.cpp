// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

#if NCNN_BF16
static void eltwise_bf16s(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

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
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _p1 = bfloat2float_msa(ptr1);
                _p = __msa_fmul_w(_p, _p1);
                float2bfloat_msa_store(_p, outptr);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __mips_msa
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
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = bfloat2float_msa(outptr);
                    v4f32 _p1 = bfloat2float_msa(ptr);
                    _p = __msa_fmul_w(_p, _p1);
                    float2bfloat_msa_store(_p, outptr);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = bfloat2float_msa(ptr);
                    v4f32 _p1 = bfloat2float_msa(ptr1);
                    _p = __msa_fadd_w(_p, _p1);
                    float2bfloat_msa_store(_p, outptr);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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
#if __mips_msa
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = bfloat2float_msa(outptr);
                        v4f32 _p1 = bfloat2float_msa(ptr);
                        _p = __msa_fadd_w(_p, _p1);
                        float2bfloat_msa_store(_p, outptr);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __mips_msa
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
#if __mips_msa
                v4f32 _coeff0 = (v4f32)__msa_fill_w_f32(coeff0);
                v4f32 _coeff1 = (v4f32)__msa_fill_w_f32(coeff1);
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = bfloat2float_msa(ptr);
                    v4f32 _p1 = bfloat2float_msa(ptr1);
                    _p = __msa_fmul_w(_p, _coeff0);
                    _p = __msa_fmadd_w(_p, _p1, _coeff1);
                    float2bfloat_msa_store(_p, outptr);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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
#if __mips_msa
                    v4f32 _coeff = (v4f32)__msa_fill_w_f32(coeff);
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = bfloat2float_msa(outptr);
                        v4f32 _p1 = bfloat2float_msa(ptr);
                        _p = __msa_fmadd_w(_p, _p1, _coeff);
                        float2bfloat_msa_store(_p, outptr);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __mips_msa
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
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                v4f32 _p1 = bfloat2float_msa(ptr1);
                _p = __msa_fmax_w(_p, _p1);
                float2bfloat_msa_store(_p, outptr);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __mips_msa
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
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = bfloat2float_msa(outptr);
                    v4f32 _p1 = bfloat2float_msa(ptr);
                    _p = __msa_fmax_w(_p, _p1);
                    float2bfloat_msa_store(_p, outptr);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*outptr)));

                    ptr++;
                    outptr++;
                }
            }
        }
    }
}
#endif

Eltwise_mips::Eltwise_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Eltwise_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                _p = __msa_fmul_w(_p, _p1);
                __msa_st_w((v4i32)_p, outptr, 0);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __mips_msa
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
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(outptr, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmul_w(_p, _p1);
                    __msa_st_w((v4i32)_p, outptr, 0);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    _p = __msa_fadd_w(_p, _p1);
                    __msa_st_w((v4i32)_p, outptr, 0);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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
#if __mips_msa
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(outptr, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fadd_w(_p, _p1);
                        __msa_st_w((v4i32)_p, outptr, 0);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __mips_msa
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
#if __mips_msa
            v4f32 _coeff0 = (v4f32)__msa_fill_w_f32(coeff0);
            v4f32 _coeff1 = (v4f32)__msa_fill_w_f32(coeff1);
#endif // __mips_msa
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    _p = __msa_fmul_w(_p, _coeff0);
                    _p = __msa_fmadd_w(_p, _p1, _coeff1);
                    __msa_st_w((v4i32)_p, outptr, 0);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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
#if __mips_msa
                v4f32 _coeff = (v4f32)__msa_fill_w_f32(coeff);
#endif // __mips_msa
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __mips_msa
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(outptr, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmadd_w(_p, _p1, _coeff);
                        __msa_st_w((v4i32)_p, outptr, 0);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __mips_msa
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
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                _p = __msa_fmax_w(_p, _p1);
                __msa_st_w((v4i32)_p, outptr, 0);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __mips_msa
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
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(outptr, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmax_w(_p, _p1);
                    __msa_st_w((v4i32)_p, outptr, 0);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __mips_msa
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

#if NCNN_BF16
int Eltwise_mips::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    eltwise_bf16s(bottom_blobs, top_blob, op_type, coeffs, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
