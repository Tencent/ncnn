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

#include "quantize_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Quantize_mips::Quantize_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

int Quantize_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale);
                    outptr[1] = float2int8(ptr0[1] * scale);
                    outptr[2] = float2int8(ptr0[2] * scale);
                    outptr[3] = float2int8(ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale_data[i * 4]);
                    outptr[1] = float2int8(ptr0[1] * scale_data[i * 4 + 1]);
                    outptr[2] = float2int8(ptr0[2] * scale_data[i * 4 + 2]);
                    outptr[3] = float2int8(ptr0[3] * scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    v4f32 _scale = (v4f32)__msa_fill_w_f32(scale_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 2);
                        const float* ptr1 = bottom_blob.row(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        for (int j = 0; j < w; j++)
                        {
                            __builtin_prefetch(ptr0 + 16);
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _vlow = (v4f32)__msa_ld_w(ptr0, 0);
                            v4f32 _vhigh = (v4f32)__msa_ld_w(ptr1, 0);
                            _vlow = __msa_fmul_w(_vlow, _scale);
                            _vhigh = __msa_fmul_w(_vhigh, _scale);
                            *((int64_t*)outptr) = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 2);
                        const float* ptr1 = bottom_blob.row(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        v4f32 _scale0 = (v4f32)__msa_ld_w((const float*)scale_data + i * 8, 0);
                        v4f32 _scale1 = (v4f32)__msa_ld_w((const float*)scale_data + i * 8 + 4, 0);

                        for (int j = 0; j < w; j++)
                        {
                            __builtin_prefetch(ptr0 + 16);
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _vlow = (v4f32)__msa_ld_w(ptr0, 0);
                            v4f32 _vhigh = (v4f32)__msa_ld_w(ptr1, 0);
                            _vlow = __msa_fmul_w(_vlow, _scale0);
                            _vhigh = __msa_fmul_w(_vhigh, _scale1);
                            *((int64_t*)outptr) = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * scale);
                            outptr1[0] = float2int8(ptr0[1] * scale);
                            outptr2[0] = float2int8(ptr0[2] * scale);
                            outptr3[0] = float2int8(ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        const float s0 = scale_data[i * 4];
                        const float s1 = scale_data[i * 4 + 1];
                        const float s2 = scale_data[i * 4 + 2];
                        const float s3 = scale_data[i * 4 + 3];

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * s0);
                            outptr1[0] = float2int8(ptr0[1] * s1);
                            outptr2[0] = float2int8(ptr0[2] * s2);
                            outptr3[0] = float2int8(ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
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
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    v4f32 _scale = (v4f32)__msa_fill_w_f32(scale_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 2);
                        const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        int i = 0;
                        for (; i + 1 < size; i += 2)
                        {
                            __builtin_prefetch(ptr0 + 32);
                            __builtin_prefetch(ptr1 + 32);
                            v4f32 _v0 = (v4f32)__msa_ld_w(ptr0, 0);
                            v4f32 _v1 = (v4f32)__msa_ld_w(ptr0 + 4, 0);
                            v4f32 _v2 = (v4f32)__msa_ld_w(ptr1, 0);
                            v4f32 _v3 = (v4f32)__msa_ld_w(ptr1 + 4, 0);
                            _v0 = __msa_fmul_w(_v0, _scale);
                            _v1 = __msa_fmul_w(_v1, _scale);
                            _v2 = __msa_fmul_w(_v2, _scale);
                            _v3 = __msa_fmul_w(_v3, _scale);
                            *((int64_t*)outptr) = float2int8(_v0, _v2);
                            *((int64_t*)(outptr + 8)) = float2int8(_v1, _v3);

                            ptr0 += 8;
                            ptr1 += 8;
                            outptr += 16;
                        }
                        for (; i < size; i++)
                        {
                            __builtin_prefetch(ptr0 + 16);
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _vlow = (v4f32)__msa_ld_w(ptr0, 0);
                            v4f32 _vhigh = (v4f32)__msa_ld_w(ptr1, 0);
                            _vlow = __msa_fmul_w(_vlow, _scale);
                            _vhigh = __msa_fmul_w(_vhigh, _scale);
                            *((int64_t*)outptr) = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 2);
                        const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        v4f32 _scale0 = (v4f32)__msa_ld_w((const float*)scale_data + q * 8, 0);
                        v4f32 _scale1 = (v4f32)__msa_ld_w((const float*)scale_data + q * 8 + 4, 0);

                        int i = 0;
                        for (; i < size; i++)
                        {
                            __builtin_prefetch(ptr0 + 16);
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _vlow = (v4f32)__msa_ld_w(ptr0, 0);
                            v4f32 _vhigh = (v4f32)__msa_ld_w(ptr1, 0);
                            _vlow = __msa_fmul_w(_vlow, _scale0);
                            _vhigh = __msa_fmul_w(_vhigh, _scale1);
                            *((int64_t*)outptr) = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * scale);
                            outptr1[0] = float2int8(ptr0[1] * scale);
                            outptr2[0] = float2int8(ptr0[2] * scale);
                            outptr3[0] = float2int8(ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        const float s0 = scale_data[q * 4];
                        const float s1 = scale_data[q * 4 + 1];
                        const float s2 = scale_data[q * 4 + 2];
                        const float s3 = scale_data[q * 4 + 3];

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * s0);
                            outptr1[0] = float2int8(ptr0[1] * s1);
                            outptr2[0] = float2int8(ptr0[2] * s2);
                            outptr3[0] = float2int8(ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
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

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr0 = bottom_blob.row(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8(*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            int i = 0;
#if __mips_msa
            v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
            for (; i + 15 < size; i += 16)
            {
                __builtin_prefetch(ptr + 64);
                v4f32 _v0 = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _v1 = (v4f32)__msa_ld_w(ptr + 4, 0);
                v4f32 _v2 = (v4f32)__msa_ld_w(ptr + 8, 0);
                v4f32 _v3 = (v4f32)__msa_ld_w(ptr + 12, 0);
                _v0 = __msa_fmul_w(_v0, _scale);
                _v1 = __msa_fmul_w(_v1, _scale);
                _v2 = __msa_fmul_w(_v2, _scale);
                _v3 = __msa_fmul_w(_v3, _scale);
                *((int64_t*)outptr) = float2int8(_v0, _v1);
                *((int64_t*)(outptr + 8)) = float2int8(_v2, _v3);

                ptr += 16;
                outptr += 16;
            }
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);
                v4f32 _v0 = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _v1 = (v4f32)__msa_ld_w(ptr + 4, 0);
                _v0 = __msa_fmul_w(_v0, _scale);
                _v1 = __msa_fmul_w(_v1, _scale);
                *((int64_t*)outptr) = float2int8(_v0, _v1);

                ptr += 8;
                outptr += 8;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
    }

    return 0;
}

} // namespace ncnn
