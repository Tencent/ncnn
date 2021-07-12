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

#include "flatten_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Flatten_riscv::Flatten_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Flatten_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = total % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == packn
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
#if __riscv_vector
        if (elempack == packn) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr = (float*)top_blob + w * i * packn;

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m1(n);

                    vfloat32m1_t _p = vle32_v_f32m1(ptr, vl);
                    vsse32_v_f32m1(outptr, w * sizeof(float), _p, vl);

                    ptr += vl;
                    outptr += 1;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector
    }

    if (dims == 3)
    {
#if __riscv_vector
        if (elempack == packn) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q * packn;

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m1(n);

                    vfloat32m1_t _p = vle32_v_f32m1(ptr, vl);
                    vsse32_v_f32m1(outptr, size * sizeof(float), _p, vl);

                    ptr += vl;
                    outptr += 1;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

#if __riscv_vector
                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);

                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vse32_v_f32m8(outptr, _p, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
#endif // __riscv_vector
            }
        }
    }

    return 0;
}

int Flatten_riscv::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = total % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == packn
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
#if __riscv_vector
        if (elempack == packn) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* ptr = bottom_blob.row<unsigned short>(i);
                unsigned short* outptr = (unsigned short*)top_blob + w * i * packn;

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m1(n);

                    vuint16m1_t _p = vle16_v_u16m1(ptr, vl);
                    vsse16_v_u16m1(outptr, w * sizeof(unsigned short), _p, vl);

                    ptr += vl;
                    outptr += 1;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector
    }

    if (dims == 3)
    {
#if __riscv_vector
        if (elempack == packn) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = (unsigned short*)top_blob + size * q * packn;

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m1(n);

                    vuint16m1_t _p = vle16_v_u16m1(ptr, vl);
                    vsse16_v_u16m1(outptr, size * sizeof(unsigned short), _p, vl);

                    ptr += vl;
                    outptr += 1;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = (unsigned short*)top_blob + size * q;

#if __riscv_vector
                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);

                    vuint16m8_t _p = vle16_v_u16m8(ptr, vl);
                    vse16_v_u16m8(outptr, _p, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
#endif // __riscv_vector
            }
        }
    }

    return 0;
}

int Flatten_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 1;
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = total % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == packn
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
#if __riscv_vector
        if (elempack == packn) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const signed char* ptr = bottom_blob.row<signed char>(i);
                signed char* outptr = (signed char*)top_blob + w * i * packn;

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e8m1(n);

                    vint8m1_t _p = vle8_v_i8m1(ptr, vl);
                    vsse8_v_i8m1(outptr, w * sizeof(unsigned char), _p, vl);

                    ptr += vl;
                    outptr += 1;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector
    }

    if (dims == 3)
    {
#if __riscv_vector
        if (elempack == packn) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* ptr = bottom_blob.channel(q);
                signed char* outptr = (signed char*)top_blob + size * q * packn;

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e8m1(n);

                    vint8m1_t _p = vle8_v_i8m1(ptr, vl);
                    vsse8_v_i8m1(outptr, size * sizeof(signed char), _p, vl);

                    ptr += vl;
                    outptr += 1;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1) // out_elempack == packn
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* ptr = bottom_blob.channel(q);
                signed char* outptr = (signed char*)top_blob + size * q;

#if __riscv_vector
                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e8m8(n);

                    vint8m8_t _p = vle8_v_i8m8(ptr, vl);
                    vse8_v_i8m8(outptr, _p, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
#endif // __riscv_vector
            }
        }
    }

    return 0;
}

} // namespace ncnn
