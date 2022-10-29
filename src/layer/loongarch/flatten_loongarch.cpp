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

#include "flatten_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

namespace ncnn {

Flatten_loongarch::Flatten_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

int Flatten_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = total % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 4
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2)
    {
#if __loongarch_sx
        if (elempack == 4) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + w * i * 4;
                float* outptr1 = (float*)top_blob + w * (i * 4 + 1);
                float* outptr2 = (float*)top_blob + w * (i * 4 + 2);
                float* outptr3 = (float*)top_blob + w * (i * 4 + 3);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    __m128i _r0 = __lsx_vld(ptr, 0);
                    __m128i _r1 = __lsx_vld(ptr + 4, 0);
                    __m128i _r2 = __lsx_vld(ptr + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(ptr + 4 * 3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, outptr0, 0);
                    __lsx_vst(_r0123_1, outptr1, 0);
                    __lsx_vst(_r0123_2, outptr2, 0);
                    __lsx_vst(_r0123_3, outptr3, 0);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }
#endif // __loongarch_sx
    }

    if (dims == 3 || dims == 4)
    {
#if __loongarch_sx
        if (elempack == 4) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + size * q * 4;
                float* outptr1 = (float*)top_blob + size * (q * 4 + 1);
                float* outptr2 = (float*)top_blob + size * (q * 4 + 2);
                float* outptr3 = (float*)top_blob + size * (q * 4 + 3);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    __m128i _r0 = __lsx_vld(ptr, 0);
                    __m128i _r1 = __lsx_vld(ptr + 4, 0);
                    __m128i _r2 = __lsx_vld(ptr + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(ptr + 4 * 3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, outptr0, 0);
                    __lsx_vst(_r0123_1, outptr1, 0);
                    __lsx_vst(_r0123_2, outptr2, 0);
                    __lsx_vst(_r0123_3, outptr3, 0);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }
#endif // __loongarch_sx

        if (elempack == 1) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

                int i = 0;
#if __loongarch_sx
                for (; i + 3 < size; i += 4)
                {
                    __lsx_vst(__lsx_vld(ptr, 0), outptr, 0);
                    ptr += 4;
                    outptr += 4;
                }
#endif // __loongarch_sx
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Flatten_loongarch::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = total % 8 == 0 ? 8 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 8
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2)
    {
#if __loongarch_sx
        if (elempack == 8) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const signed char* ptr = bottom_blob.row<signed char>(i);
                signed char* outptr0 = (signed char*)top_blob + w * i * 8;
                signed char* outptr1 = (signed char*)top_blob + w * (i * 8 + 1);
                signed char* outptr2 = (signed char*)top_blob + w * (i * 8 + 2);
                signed char* outptr3 = (signed char*)top_blob + w * (i * 8 + 3);
                signed char* outptr4 = (signed char*)top_blob + w * (i * 8 + 4);
                signed char* outptr5 = (signed char*)top_blob + w * (i * 8 + 5);
                signed char* outptr6 = (signed char*)top_blob + w * (i * 8 + 6);
                signed char* outptr7 = (signed char*)top_blob + w * (i * 8 + 7);

                int j = 0;
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }
        }
#endif // __loongarch_sx
    }

    if (dims == 3 || dims == 4)
    {
#if __loongarch_sx
        if (elempack == 8) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* ptr = bottom_blob.channel(q);
                signed char* outptr0 = (signed char*)top_blob + size * q * 8;
                signed char* outptr1 = (signed char*)top_blob + size * (q * 8 + 1);
                signed char* outptr2 = (signed char*)top_blob + size * (q * 8 + 2);
                signed char* outptr3 = (signed char*)top_blob + size * (q * 8 + 3);
                signed char* outptr4 = (signed char*)top_blob + size * (q * 8 + 4);
                signed char* outptr5 = (signed char*)top_blob + size * (q * 8 + 5);
                signed char* outptr6 = (signed char*)top_blob + size * (q * 8 + 6);
                signed char* outptr7 = (signed char*)top_blob + size * (q * 8 + 7);

                int i = 0;
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }
        }
#endif // __loongarch_sx

        if (elempack == 1) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* ptr = bottom_blob.channel(q);
                signed char* outptr = (signed char*)top_blob + size * q;

                int i = 0;
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
