// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "eltwise_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Eltwise_loongarch::Eltwise_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

int Eltwise_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __loongarch_sx
            for (; i + 3 < size; i += 4)
            {
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
                for (; i + 3 < size; i += 4)
                {
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
                for (; i + 3 < size; i += 4)
                {
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
                    for (; i + 3 < size; i += 4)
                    {
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
                for (; i + 3 < size; i += 4)
                {
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
                __m128 _coeff = (__m128)__lsx_vreplfr2vr_s(coeff);
#endif // __loongarch_sx
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __loongarch_sx
                    for (; i + 3 < size; i += 4)
                    {
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
            for (; i + 3 < size; i += 4)
            {
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
                for (; i + 3 < size; i += 4)
                {
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
