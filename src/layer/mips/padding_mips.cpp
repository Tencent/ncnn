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

#include "padding_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

#if __mips_msa
#include "padding_pack4.h"
#include "padding_pack8_int8.h"
#endif // __mips_msa

Padding_mips::Padding_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int Padding_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                v4f32 pad_value = __msa_fill_w_f32(value);
                padding_constant_pack4_msa(bottom_blob, top_blob, 0, 0, left / 4, right / 4, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                v4f32 pad_value = __msa_fill_w_f32(value);
                padding_constant_pack4_msa(bottom_blob, top_blob, top / 4, bottom / 4, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    v4f32 pad_value = per_channel_pad_data_size ? (v4f32)__msa_ld_w((const float*)per_channel_pad_data + q * 4, 0) : __msa_fill_w_f32(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_msa(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack4_msa(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_msa(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    v4f32 pad_value = per_channel_pad_data_size ? (v4f32)__msa_ld_w((const float*)per_channel_pad_data + q * 4, 0) : __msa_fill_w_f32(value);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack4_msa(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __mips_msa

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat top_blob_unpacked;
    int ret = Padding::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    int out_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % 4 == 0 ? 4 : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

int Padding_mips::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __mips_msa
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v8 = (int64_t)value;
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);
                padding_constant_pack8_int8_msa(bottom_blob, top_blob, 0, 0, left / 8, right / 8, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v8 = (int64_t)value;
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);
                padding_constant_pack8_int8_msa(bottom_blob, top_blob, top / 8, bottom / 8, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    // TODO perchannel
                    //                     int64_t pad_value = per_channel_pad_data_size ? vld1_s8(per_channel_pad_data + q * 8) : vdup_n_s8((signed char)value);
                    int64_t v8 = (int64_t)value;
                    int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_int8_msa(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_int8_msa(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_int8_msa(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    // TODO perchannel
                    //                     int64_t pad_value = per_channel_pad_data_size ? vld1_s8(per_channel_pad_data + q * 8) : vdup_n_s8((signed char)value);
                    int64_t v8 = (int64_t)value;
                    int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack8_int8_msa(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __mips_msa

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat top_blob_unpacked;
    int ret = Padding::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    int out_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % 8 == 0 ? 8 : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

} // namespace ncnn
