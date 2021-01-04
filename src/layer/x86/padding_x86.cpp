// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "padding_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

#if __SSE2__
#include "padding_pack4.h"
#if __AVX__
#include "padding_pack8.h"
#endif // __AVX__
#endif // __SSE2__

Padding_x86::Padding_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Padding_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (left % 8 == 0 && out_elempack == 8)
            {
                // TODO
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (top % 8 == 0 && out_elempack == 8)
            {
                // TODO
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0))
            {
                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m256 pad_value = per_channel_pad_data_size ? _mm256_loadu_ps((const float*)per_channel_pad_data + q * 8) : _mm256_set1_ps(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_avx(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_avx(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_avx(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

#if __AVX__
            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outw % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (left % 4 == 0 && out_elempack == 4)
            {
                // TODO
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

#if __AVX__
            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (top % 4 == 0 && out_elempack == 4)
            {
                // TODO
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

#if __AVX__
            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0))
            {
                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m128 pad_value = per_channel_pad_data_size ? _mm_loadu_ps((const float*)per_channel_pad_data + q * 4) : _mm_set1_ps(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_sse(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack4_sse(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_sse(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }
    }
#endif // __SSE2__

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

} // namespace ncnn
