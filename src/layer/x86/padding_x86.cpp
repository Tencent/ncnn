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

#if __AVX__
#include <immintrin.h>
#endif // __AVX__

namespace ncnn {

#if __AVX__
#include "padding_pack8.h"
#endif // __AVX__

Padding_x86::Padding_x86()
{
#if __AVX__
    support_packing = true;
#endif // __AVX__
}

int Padding_x86::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Padding_x86::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Padding_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    Mat bottom_blob_unpacked = bottom_blob;

#if __AVX__
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int out_elempack = elempack;
    int outc = channels;

    //Check if channel padding is being applied.
    if (front != 0 || behind != 0)
    {
        int padded_channels = (channels * elempack) + front + behind;
        if (type == 0)
        {
            int offset_elempack = front % 8 == 0 ? 8 : 1;
            int channel_elempack = padded_channels % 8 == 0 ? 8 : 1;
            out_elempack = offset_elempack <= channel_elempack ? offset_elempack : channel_elempack;
        }
        else
        {
            //Reflective padding and edge padding only supports channel padding in elempack 1
            out_elempack = 1;
        }
        outc = padded_channels / out_elempack;
        if (out_elempack != elempack)
        {
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);
        }
    }

    if (elempack == 8 && out_elempack == 8)
    {
        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack8_avx(bottom_blob, top_blob, 0, 0, left, right, _mm256_set1_ps(value));
            if (type == 1)
                padding_replicate_pack8_avx(bottom_blob, top_blob, 0, 0, left, right);
            if (type == 2)
                padding_reflect_pack8_avx(bottom_blob, top_blob, 0, 0, left, right);

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack8_avx(bottom_blob, top_blob, top, bottom, left, right, _mm256_set1_ps(value));
            if (type == 1)
                padding_replicate_pack8_avx(bottom_blob, top_blob, top, bottom, left, right);
            if (type == 2)
                padding_reflect_pack8_avx(bottom_blob, top_blob, top, bottom, left, right);

            return 0;
        }

        if (dims == 3)
        {
            top_blob.create(outw, outh, outc, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;
            int front_ = front / elempack;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
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

        return 0;
    }
#endif // __AVX__

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

} // namespace ncnn
