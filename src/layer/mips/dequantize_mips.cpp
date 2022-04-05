// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "dequantize_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Dequantize_mips::Dequantize_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

int Dequantize_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // assert bottom_blob.elembits() == 32

    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __mips_msa
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                v4f32 _scale = (v4f32)__msa_fill_w_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmul_w(_v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else if (bias_data_size == 1)
                {
                    v4f32 _bias = (v4f32)__msa_fill_w_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _scale = (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmul_w(_v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else if (bias_data_size == 1)
                {
                    v4f32 _bias = (v4f32)__msa_fill_w_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _scale = (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _scale = (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                        v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 2;

            top_blob.create(w, outh, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    v4f32 _scale0 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + i * 8, 0);
                    v4f32 _scale1 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + i * 8 + 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __builtin_prefetch(intptr + 32);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        _v0 = __msa_fmul_w(_v0, _scale0);
                        _v1 = __msa_fmul_w(_v1, _scale1);
                        __msa_st_w((v4i32)_v0, ptr0, 0);
                        __msa_st_w((v4i32)_v1, ptr1, 0);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    v4f32 _scale0 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + i * 8, 0);
                    v4f32 _scale1 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + i * 8 + 4, 0);
                    v4f32 _bias0 = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + i * 8, 0);
                    v4f32 _bias1 = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + i * 8 + 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __builtin_prefetch(intptr + 32);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                        _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                        __msa_st_w((v4i32)_v0, ptr0, 0);
                        __msa_st_w((v4i32)_v1, ptr1, 0);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    v4f32 _scale0 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + q * 8, 0);
                    v4f32 _scale1 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + q * 8 + 4, 0);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __builtin_prefetch(intptr + 64);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        v4f32 _v2 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 8, 0));
                        v4f32 _v3 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 12, 0));
                        _v0 = __msa_fmul_w(_v0, _scale0);
                        _v1 = __msa_fmul_w(_v1, _scale1);
                        _v2 = __msa_fmul_w(_v2, _scale0);
                        _v3 = __msa_fmul_w(_v3, _scale1);
                        __msa_st_w((v4i32)_v0, ptr0, 0);
                        __msa_st_w((v4i32)_v2, ptr0 + 4, 0);
                        __msa_st_w((v4i32)_v1, ptr1, 0);
                        __msa_st_w((v4i32)_v3, ptr1 + 4, 0);

                        intptr += 16;
                        ptr0 += 8;
                        ptr1 += 8;
                    }
                    for (; i < size; i++)
                    {
                        __builtin_prefetch(intptr + 32);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        _v0 = __msa_fmul_w(_v0, _scale0);
                        _v1 = __msa_fmul_w(_v1, _scale1);
                        __msa_st_w((v4i32)_v0, ptr0, 0);
                        __msa_st_w((v4i32)_v1, ptr1, 0);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    v4f32 _scale0 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + q * 8, 0);
                    v4f32 _scale1 = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + q * 8 + 4, 0);
                    v4f32 _bias0 = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + q * 8, 0);
                    v4f32 _bias1 = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + q * 8 + 4, 0);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __builtin_prefetch(intptr + 64);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        v4f32 _v2 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 8, 0));
                        v4f32 _v3 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 12, 0));
                        _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                        _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                        _v2 = __msa_fmadd_w(_bias0, _v2, _scale0);
                        _v3 = __msa_fmadd_w(_bias1, _v3, _scale1);
                        __msa_st_w((v4i32)_v0, ptr0, 0);
                        __msa_st_w((v4i32)_v2, ptr0 + 4, 0);
                        __msa_st_w((v4i32)_v1, ptr1, 0);
                        __msa_st_w((v4i32)_v3, ptr1 + 4, 0);

                        intptr += 16;
                        ptr0 += 8;
                        ptr1 += 8;
                    }
                    for (; i < size; i++)
                    {
                        __builtin_prefetch(intptr + 32);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                        _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                        __msa_st_w((v4i32)_v0, ptr0, 0);
                        __msa_st_w((v4i32)_v1, ptr1, 0);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                v4f32 _scale = (v4f32)__msa_fill_w_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmul_w(_v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else if (bias_data_size == 1)
                {
                    v4f32 _bias = (v4f32)__msa_fill_w_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _scale = (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmul_w(_v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else if (bias_data_size == 1)
                {
                    v4f32 _bias = (v4f32)__msa_fill_w_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _scale = (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        v4f32 _scale = (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                        v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    v4f32 _scale = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __builtin_prefetch(intptr + 16);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmul_w(_v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    v4f32 _scale = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + i * 4, 0);
                    v4f32 _bias = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __builtin_prefetch(intptr + 16);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    v4f32 _scale = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + q * 4, 0);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __builtin_prefetch(intptr + 32);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        _v0 = __msa_fmul_w(_v0, _scale);
                        _v1 = __msa_fmul_w(_v1, _scale);
                        __msa_st_w((v4i32)_v0, ptr, 0);
                        __msa_st_w((v4i32)_v1, ptr + 4, 0);

                        intptr += 8;
                        ptr += 8;
                    }
                    for (; i < size; i++)
                    {
                        __builtin_prefetch(intptr + 16);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmul_w(_v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    v4f32 _scale = scale_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_data[0]) : (v4f32)__msa_ld_w((const float*)scale_data + q * 4, 0);
                    v4f32 _bias = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + q * 4, 0);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __builtin_prefetch(intptr + 32);
                        v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                        _v0 = __msa_fmadd_w(_bias, _v0, _scale);
                        _v1 = __msa_fmadd_w(_bias, _v1, _scale);
                        __msa_st_w((v4i32)_v0, ptr, 0);
                        __msa_st_w((v4i32)_v1, ptr + 4, 0);

                        intptr += 8;
                        ptr += 8;
                    }
                    for (; i < size; i++)
                    {
                        __builtin_prefetch(intptr + 16);
                        v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                        _v = __msa_fmadd_w(_bias, _v, _scale);
                        __msa_st_w((v4i32)_v, ptr, 0);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __mips_msa

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
#if __mips_msa
                v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
                for (; j + 3 < w; j += 4)
                {
                    __builtin_prefetch(intptr + 16);
                    v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    _v = __msa_fmul_w(_v, _scale);
                    __msa_st_w((v4i32)_v, ptr, 0);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __mips_msa
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
#if __mips_msa
                v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
                v4f32 _bias = (v4f32)__msa_fill_w_f32(bias);
                for (; j + 3 < w; j += 4)
                {
                    __builtin_prefetch(intptr + 16);
                    v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    _v = __msa_fmadd_w(_bias, _v, _scale);
                    __msa_st_w((v4i32)_v, ptr, 0);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __mips_msa
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
#if __mips_msa
                v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
                for (; i + 7 < size; i += 8)
                {
                    __builtin_prefetch(intptr + 32);
                    v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                    _v0 = __msa_fmul_w(_v0, _scale);
                    _v1 = __msa_fmul_w(_v1, _scale);
                    __msa_st_w((v4i32)_v0, ptr, 0);
                    __msa_st_w((v4i32)_v1, ptr + 4, 0);

                    intptr += 8;
                    ptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(intptr + 16);
                    v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    _v = __msa_fmul_w(_v, _scale);
                    __msa_st_w((v4i32)_v, ptr, 0);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
#if __mips_msa
                v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
                v4f32 _bias = (v4f32)__msa_fill_w_f32(bias);
                for (; i + 7 < size; i += 8)
                {
                    __builtin_prefetch(intptr + 32);
                    v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                    _v0 = __msa_fmadd_w(_bias, _v0, _scale);
                    _v1 = __msa_fmadd_w(_bias, _v1, _scale);
                    __msa_st_w((v4i32)_v0, ptr, 0);
                    __msa_st_w((v4i32)_v1, ptr + 4, 0);

                    intptr += 8;
                    ptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(intptr + 16);
                    v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    _v = __msa_fmadd_w(_bias, _v, _scale);
                    __msa_st_w((v4i32)_v, ptr, 0);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
