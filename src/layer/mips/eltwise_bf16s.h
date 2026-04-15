// Tencent is pleased to support the open source community by making ncnn available.
//
//                    Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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
