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

#include "padding_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#if __ARM_NEON
#include "padding_pack4.h"
#include "padding_pack4_bf16s_fp16s.h"
#include "padding_pack8_int8.h"
#if NCNN_ARM82
#include "padding_pack8_fp16s.h"
#endif
#endif // __ARM_NEON

Padding_arm::Padding_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Padding_arm::create_pipeline(const Option& opt)
{
#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        value_fp16 = float32_to_float16(value);

        ncnn::cast_float32_to_float16(per_channel_pad_data, per_channel_pad_data_fp16, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        value_bf16 = float32_to_bfloat16(value);

        ncnn::cast_float32_to_bfloat16(per_channel_pad_data, per_channel_pad_data_bf16, opt);
    }
#endif

    return 0;
}

int Padding_arm::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Padding_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
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

                float32x4_t pad_value = vdupq_n_f32(value);
                padding_constant_pack4_neon(bottom_blob, top_blob, 0, 0, left / 4, right / 4, pad_value);

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

                float32x4_t pad_value = vdupq_n_f32(value);
                padding_constant_pack4_neon(bottom_blob, top_blob, top / 4, bottom / 4, left, right, pad_value);

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

                    float32x4_t pad_value = per_channel_pad_data_size ? vld1q_f32((const float*)per_channel_pad_data + q * 4) : vdupq_n_f32(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_neon(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack4_neon(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_neon(m, borderm, top, bottom, left, right);
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
                    float32x4_t pad_value = per_channel_pad_data_size ? vld1q_f32((const float*)per_channel_pad_data + q * 4) : vdupq_n_f32(value);

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
                            padding_constant_pack4_neon(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __ARM_NEON

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
#if NCNN_ARM82
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                uint16x8_t pad_value = vdupq_n_u16(value_fp16);
                padding_constant_pack8_fp16s_neon(bottom_blob, top_blob, 0, 0, left / 8, right / 8, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                uint16x8_t pad_value = vdupq_n_u16(value_fp16);
                padding_constant_pack8_fp16s_neon(bottom_blob, top_blob, top / 8, bottom / 8, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
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

                    uint16x8_t pad_value = per_channel_pad_data_size ? vld1q_u16((const unsigned short*)per_channel_pad_data_fp16 + q * 8) : vdupq_n_u16(value_fp16);

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_fp16s_neon(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_fp16s_neon(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_fp16s_neon(m, borderm, top, bottom, left, right);
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
                    uint16x8_t pad_value = per_channel_pad_data_size ? vld1q_u16((const unsigned short*)per_channel_pad_data_fp16 + q * 8) : vdupq_n_u16(value_fp16);

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
                            padding_constant_pack8_fp16s_neon(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

#if NCNN_ARM82
            int out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outw % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                // clang-format off
                // *INDENT-OFF*
                uint16x4_t pad_value;
#if NCNN_ARM82
                if (support_fp16_storage && opt.use_fp16_storage)
                {
                    pad_value = vdup_n_u16(value_fp16);
                }
                else
#endif
#if NCNN_BF16
                if (opt.use_bf16_storage)
                {
                    pad_value = vdup_n_u16(value_bf16);
                }
                else
#endif
                {
                    // shall never reach here
                    pad_value = vdup_n_u16(0);
                }
                // *INDENT-ON*
                // clang-format on
                padding_constant_pack4_bf16_fp16s_neon(bottom_blob, top_blob, 0, 0, left / 4, right / 4, vcombine_u16(pad_value, pad_value));

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

#if NCNN_ARM82
            int out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                // clang-format off
                // *INDENT-OFF*
                uint16x4_t pad_value;
#if NCNN_ARM82
                if (support_fp16_storage && opt.use_fp16_storage)
                {
                    pad_value = vdup_n_u16(value_fp16);
                }
                else
#endif
#if NCNN_BF16
                if (opt.use_bf16_storage)
                {
                    pad_value = vdup_n_u16(value_bf16);
                }
                else
#endif
                {
                    // shall never reach here
                    pad_value = vdup_n_u16(0);
                }
                // *INDENT-ON*
                // clang-format on
                padding_constant_pack4_bf16_fp16s_neon(bottom_blob, top_blob, top / 4, bottom / 4, left, right, vcombine_u16(pad_value, pad_value));

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

#if NCNN_ARM82
            int out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
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

                    // clang-format off
                    // *INDENT-OFF*
                    uint16x4_t pad_value;
#if NCNN_ARM82
                    if (support_fp16_storage && opt.use_fp16_storage)
                    {
                        pad_value = per_channel_pad_data_size ? vld1_u16((const unsigned short*)per_channel_pad_data_fp16 + q * 4) : vdup_n_u16(value_fp16);
                    }
                    else
#endif
#if NCNN_BF16
                    if (opt.use_bf16_storage)
                    {
                        pad_value = per_channel_pad_data_size ? vld1_u16((const unsigned short*)per_channel_pad_data_bf16 + q * 4) : vdup_n_u16(value_bf16);
                    }
                    else
#endif
                    {
                        // shall never reach here
                        pad_value = vdup_n_u16(0);
                    }
                    // *INDENT-ON*
                    // clang-format on

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_bf16_fp16s_neon(m, borderm, top, bottom, left, right, vcombine_u16(pad_value, pad_value));
                        if (type == 1)
                            padding_replicate_pack4_bf16_fp16s_neon(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_bf16_fp16s_neon(m, borderm, top, bottom, left, right);
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
                    // clang-format off
                    // *INDENT-OFF*
                    uint16x4_t pad_value;
#if NCNN_ARM82
                    if (support_fp16_storage && opt.use_fp16_storage)
                    {
                        pad_value = per_channel_pad_data_size ? vld1_u16((const unsigned short*)per_channel_pad_data_fp16 + q * 4) : vdup_n_u16(value_fp16);
                    }
                    else
#endif
#if NCNN_BF16
                    if (opt.use_bf16_storage)
                    {
                        pad_value = per_channel_pad_data_size ? vld1_u16((const unsigned short*)per_channel_pad_data_bf16 + q * 4) : vdup_n_u16(value_bf16);
                    }
                    else
#endif
                    {
                        // shall never reach here
                        pad_value = vdup_n_u16(0);
                    }
                    // *INDENT-ON*
                    // clang-format on

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
                            padding_constant_pack4_bf16_fp16s_neon(m, borderm, top, bottom, left, right, vcombine_u16(pad_value, pad_value));
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __ARM_NEON

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_arm::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
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

                int8x8_t pad_value = vdup_n_s8((signed char)value);
                padding_constant_pack8_int8_neon(bottom_blob, top_blob, 0, 0, left / 8, right / 8, pad_value);

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

                int8x8_t pad_value = vdup_n_s8((signed char)value);
                padding_constant_pack8_int8_neon(bottom_blob, top_blob, top / 8, bottom / 8, left, right, pad_value);

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

                    // TODO perchannel
                    //                     int8x8_t pad_value = per_channel_pad_data_size ? vld1_s8(per_channel_pad_data + q * 8) : vdup_n_s8((signed char)value);
                    int8x8_t pad_value = vdup_n_s8((signed char)value);

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill<int8x8_t>(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_int8_neon(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_int8_neon(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_int8_neon(m, borderm, top, bottom, left, right);
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

            top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    // TODO perchannel
                    //                     int8x8_t pad_value = per_channel_pad_data_size ? vld1_s8(per_channel_pad_data + q * 8) : vdup_n_s8((signed char)value);
                    int8x8_t pad_value = vdup_n_s8((signed char)value);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill<int8x8_t>(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack8_int8_neon(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __ARM_NEON

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
