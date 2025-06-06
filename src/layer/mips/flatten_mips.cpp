// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "flatten_mips.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

Flatten_mips::Flatten_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int Flatten_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __mips_msa
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
#if __mips_msa
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
                    v4f32 _r0 = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(ptr + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(ptr + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(ptr + 4 * 3, 0);

                    v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                    v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                    __msa_st_w((v4i32)_r0123_0, outptr0, 0);
                    __msa_st_w((v4i32)_r0123_1, outptr1, 0);
                    __msa_st_w((v4i32)_r0123_2, outptr2, 0);
                    __msa_st_w((v4i32)_r0123_3, outptr3, 0);

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
#endif // __mips_msa
    }

    if (dims == 3 || dims == 4)
    {
#if __mips_msa
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
                    v4f32 _r0 = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(ptr + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(ptr + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(ptr + 4 * 3, 0);

                    v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                    v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                    __msa_st_w((v4i32)_r0123_0, outptr0, 0);
                    __msa_st_w((v4i32)_r0123_1, outptr1, 0);
                    __msa_st_w((v4i32)_r0123_2, outptr2, 0);
                    __msa_st_w((v4i32)_r0123_3, outptr3, 0);

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
#endif // __mips_msa

        if (elempack == 1) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    __msa_st_w(__msa_ld_w(ptr, 0), outptr, 0);
                    ptr += 4;
                    outptr += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Flatten_mips::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __mips_msa
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
#if __mips_msa
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
#endif // __mips_msa
    }

    if (dims == 3 || dims == 4)
    {
#if __mips_msa
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
#endif // __mips_msa

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
