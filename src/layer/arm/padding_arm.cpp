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

namespace ncnn {

#if __ARM_NEON
#include "padding_pack4.h"
#include "padding_pack4_bf16s_fp16s.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "padding_pack8_fp16s.h"
#endif
#endif // __ARM_NEON

Padding_arm::Padding_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Padding_arm::create_pipeline(const Option& opt)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage)
    {
        ncnn::cast_float32_to_float16(per_channel_pad_data, per_channel_pad_data_fp16, opt);
    }
#endif

    if (opt.use_bf16_storage)
    {
        value_bf16 = float32_to_bfloat16(value);

        ncnn::cast_float32_to_bfloat16(per_channel_pad_data, per_channel_pad_data_bf16, opt);
    }

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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    Mat bottom_blob_unpacked = bottom_blob;

#if __ARM_NEON
    int out_elempack = elempack;
    int outc = channels;
    //Check if channel padding is being applied.
    if (front != 0 || behind != 0)
    {
        int padded_channels = (channels * elempack) + front + behind;
        if (type == 0)
        {
            int offset_elempack = front % 4 == 0 ? 4 : 1;
            int channel_elempack = padded_channels % 4 == 0 ? 4 : 1;
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

    if (elempack == 4 && out_elempack == 4)
    {
        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack4_neon(bottom_blob, top_blob, 0, 0, left, right, vdupq_n_f32(value));
            if (type == 1)
                padding_replicate_pack4_neon(bottom_blob, top_blob, 0, 0, left, right);
            if (type == 2)
                padding_reflect_pack4_neon(bottom_blob, top_blob, 0, 0, left, right);

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack4_neon(bottom_blob, top_blob, top, bottom, left, right, vdupq_n_f32(value));
            if (type == 1)
                padding_replicate_pack4_neon(bottom_blob, top_blob, top, bottom, left, right);
            if (type == 2)
                padding_reflect_pack4_neon(bottom_blob, top_blob, top, bottom, left, right);

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

        return 0;
    }
#endif // __ARM_NEON
    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    Mat bottom_blob_unpacked = bottom_blob;

#if __ARM_NEON
    int out_elempack = elempack;
    int outc = channels;
    //Check if channel padding is being applied.
    if (front != 0 || behind != 0)
    {
        int padded_channels = (channels * elempack) + front + behind;
        if (type == 0)
        {
            if (opt.use_fp16_arithmetic)
            {
                int offset_elempack = front % 8 == 0 ? 8 : 1;
                int channel_elempack = padded_channels % 8 == 0 ? 8 : 1;
                out_elempack = offset_elempack <= channel_elempack ? offset_elempack : channel_elempack;
            }
            else
            {
                int offset_elempack = front % 4 == 0 ? 4 : 1;
                int channel_elempack = padded_channels % 4 == 0 ? 4 : 1;
                out_elempack = offset_elempack <= channel_elempack ? offset_elempack : channel_elempack;
            }
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (elempack == 8 && out_elempack == 8)
    {
        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
            {
                float16x8_t pad_value = vdupq_n_f16((__fp16)value);

                padding_constant_pack8_fp16s_neon(bottom_blob, top_blob, 0, 0, left, right, pad_value);
            }
            if (type == 1)
                padding_replicate_pack8_fp16s_neon(bottom_blob, top_blob, 0, 0, left, right);
            if (type == 2)
                padding_reflect_pack8_fp16s_neon(bottom_blob, top_blob, 0, 0, left, right);

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
            {
                float16x8_t pad_value = vdupq_n_f16((__fp16)value);

                padding_constant_pack8_fp16s_neon(bottom_blob, top_blob, top, bottom, left, right, pad_value);
            }
            if (type == 1)
                padding_replicate_pack8_fp16s_neon(bottom_blob, top_blob, top, bottom, left, right);
            if (type == 2)
                padding_reflect_pack8_fp16s_neon(bottom_blob, top_blob, top, bottom, left, right);

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

                float16x8_t pad_value = per_channel_pad_data_size ? vld1q_f16((const __fp16*)per_channel_pad_data_fp16 + q * 8) : vdupq_n_f16((__fp16)value);

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

        return 0;
    }
#endif

    if (elempack == 4 && out_elempack == 4)
    {
        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                uint16x8_t pad_value = opt.use_fp16_storage ? vreinterpretq_u16_f16(vdupq_n_f16((__fp16)value)) : vdupq_n_u16(value_bf16);
#else
                uint16x8_t pad_value = vdupq_n_u16(value_bf16);
#endif
                padding_constant_pack4_bf16_fp16s_neon(bottom_blob, top_blob, 0, 0, left, right, pad_value);
            }
            if (type == 1)
                padding_replicate_pack4_bf16_fp16s_neon(bottom_blob, top_blob, 0, 0, left, right);
            if (type == 2)
                padding_reflect_pack4_bf16_fp16s_neon(bottom_blob, top_blob, 0, 0, left, right);

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                uint16x8_t pad_value = opt.use_fp16_storage ? vreinterpretq_u16_f16(vdupq_n_f16((__fp16)value)) : vdupq_n_u16(value_bf16);
#else
                uint16x8_t pad_value = vdupq_n_u16(value_bf16);
#endif
                padding_constant_pack4_bf16_fp16s_neon(bottom_blob, top_blob, top, bottom, left, right, pad_value);
            }
            if (type == 1)
                padding_replicate_pack4_bf16_fp16s_neon(bottom_blob, top_blob, top, bottom, left, right);
            if (type == 2)
                padding_reflect_pack4_bf16_fp16s_neon(bottom_blob, top_blob, top, bottom, left, right);

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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                uint16x4_t pad_value;
                if (opt.use_fp16_storage)
                {
                    pad_value = per_channel_pad_data_size ? vreinterpret_u16_f16(vld1_f16((const __fp16*)per_channel_pad_data_fp16 + q * 4)) : vreinterpret_u16_f16(vdup_n_f16((__fp16)value));
                }
                else
                {
                    pad_value = per_channel_pad_data_size ? vld1_u16((const unsigned short*)per_channel_pad_data_bf16 + q * 4) : vdup_n_u16(value_bf16);
                }
#else
                uint16x4_t pad_value = per_channel_pad_data_size ? vld1_u16((const unsigned short*)per_channel_pad_data_bf16 + q * 4) : vdup_n_u16(value_bf16);
#endif
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

        return 0;
    }
#endif // __ARM_NEON
    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

} // namespace ncnn
