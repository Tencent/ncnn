// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Eltwise_powerpc::Eltwise_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int Eltwise_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __ALTIVEC__
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _p1 = __ncnn_altivec_load(ptr1);
                _p = __ncnn_altivec_mul(_p, _p1);
                __ncnn_altivec_store(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(outptr);
                    v4f32 _p1 = __ncnn_altivec_load(ptr);
                    _p = __ncnn_altivec_mul(_p, _p1);
                    __ncnn_altivec_store(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    v4f32 _p1 = __ncnn_altivec_load(ptr1);
                    _p = vec_add(_p, _p1);
                    __ncnn_altivec_store(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(outptr);
                        v4f32 _p1 = __ncnn_altivec_load(ptr);
                        _p = vec_add(_p, _p1);
                        __ncnn_altivec_store(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
            v4f32 _coeff0 = vec_splats(coeff0);
            v4f32 _coeff1 = vec_splats(coeff1);
#endif // __ALTIVEC__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    v4f32 _p1 = __ncnn_altivec_load(ptr1);
                    _p = __ncnn_altivec_mul(_p, _coeff0);
                    _p = vec_madd(_p1, _coeff1, _p);
                    __ncnn_altivec_store(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
                v4f32 _coeff = vec_splats(coeff);
#endif // __ALTIVEC__
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __ALTIVEC__
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(outptr);
                        v4f32 _p1 = __ncnn_altivec_load(ptr);
                        _p = vec_madd(_p1, _coeff, _p);
                        __ncnn_altivec_store(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _p1 = __ncnn_altivec_load(ptr1);
                _p = vec_max(_p, _p1);
                __ncnn_altivec_store(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __ALTIVEC__
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
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(outptr);
                    v4f32 _p1 = __ncnn_altivec_load(ptr);
                    _p = vec_max(_p, _p1);
                    __ncnn_altivec_store(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    *outptr = std::max(*outptr, *ptr);

                    ptr++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
