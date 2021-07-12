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

#include "padding_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

#if __riscv_vector
#include "padding_packn.h"
#endif // __riscv_vector

Padding_riscv::Padding_riscv()
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

int Padding_riscv::create_pipeline(const Option& opt)
{
#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage)
    {
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

int Padding_riscv::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Padding_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == packn)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (left % packn == 0 && out_elempack == packn)
            {
                // TODO
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (top % packn == 0 && out_elempack == packn)
            {
                // TODO
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (front % packn == 0 && out_elempack == packn && !(outc != channels * elempack && type != 0))
            {
                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    vfloat32m1_t pad_value = per_channel_pad_data_size ? vle32_v_f32m1((const float*)per_channel_pad_data + q * packn, vl) : vfmv_v_f_f32m1(value, vl);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_packn_float32_rvv(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_packn_float32_rvv(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_packn_float32_rvv(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

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
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % packn == 0 ? packn : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

int Padding_riscv::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const word_type vl = vsetvl_e16m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == packn)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (left % packn == 0 && out_elempack == packn)
            {
                // TODO
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (top % packn == 0 && out_elempack == packn)
            {
                // TODO
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (front % packn == 0 && out_elempack == packn && !(outc != channels * elempack && type != 0))
            {
                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    // clang-format off
                    // *INDENT-OFF*
                    vuint16m1_t pad_value;
#if __riscv_zfh
                    if (opt.use_fp16_storage)
                    {
                        pad_value = per_channel_pad_data_size ? vreinterpret_v_f16m1_u16m1(vle16_v_f16m1((const __fp16*)per_channel_pad_data_fp16 + q * packn, vl)) : vreinterpret_v_f16m1_u16m1(vfmv_v_f_f16m1((__fp16)value, vl));
                    }
                    else
#endif
#if NCNN_BF16
                    if (opt.use_bf16_storage)
                    {
                        pad_value = per_channel_pad_data_size ? vle16_v_u16m1((const unsigned short*)per_channel_pad_data_bf16 + q * packn, vl) : vmv_v_x_u16m1(value_bf16, vl);
                    }
                    else
#endif
                    {
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
                            padding_constant_packn_uint16_rvv(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_packn_uint16_rvv(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_packn_uint16_rvv(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

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
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % packn == 0 ? packn : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

int Padding_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 1;
    const word_type vl = vsetvl_e8m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == packn)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (left % packn == 0 && out_elempack == packn)
            {
                // TODO
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (top % packn == 0 && out_elempack == packn)
            {
                // TODO
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (front % packn == 0 && out_elempack == packn && !(outc != channels * elempack && type != 0))
            {
                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    // TODO perchannel
                    // vint8m1_t pad_value = per_channel_pad_data_size ? vle8_v_i8m1(per_channel_pad_data + q * packn) : vmv_v_x_i8m1((signed char)value);
                    vint8m1_t pad_value = vmv_v_x_i8m1((signed char)value, vl);

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_packn_int8_rvv(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_packn_int8_rvv(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_packn_int8_rvv(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

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
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % packn == 0 ? packn : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

} // namespace ncnn
