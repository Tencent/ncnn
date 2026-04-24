// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shufflechannel_loongarch.h"

namespace ncnn {

ShuffleChannel_loongarch::ShuffleChannel_loongarch()
{
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static int unpack_or_cast_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.empty())
    {
        dst = src;
        return 0;
    }

    Mat unpacked = src;
    if (src.elempack != 1)
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;

        convert_packing(src, unpacked, 1, opt_unpack);
        if (unpacked.empty())
            return -100;
    }

#if NCNN_BF16
    if (opt.use_bf16_storage && unpacked.elembits() == 16)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        cast_bfloat16_to_float32(unpacked, dst, opt_cast);
        if (dst.empty())
            return -100;

        return 0;
    }
#endif

    dst = unpacked;
    return 0;
}

static int restore_layout_from_reference(const Mat& src, Mat& dst, const Mat& reference, const Option& opt)
{
    Mat tmp = src;

    if (opt.use_packing_layout && reference.elempack != 1)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat packed;
        convert_packing(tmp, packed, reference.elempack, opt_pack);
        if (packed.empty())
            return -100;

        tmp = packed;
    }

#if NCNN_BF16
    if (opt.use_bf16_storage && reference.elembits() == 16)
    {
        cast_float32_to_bfloat16(tmp, dst, opt);
        if (dst.empty())
            return -100;

        return 0;
    }
#endif

    dst = tmp;
    return 0;
}

int ShuffleChannel_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int logical_channels = bottom_blob.c * bottom_blob.elempack;
    int _group = reverse ? logical_channels / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    Mat bottom_blob_fp32;
    if (unpack_or_cast_to_float32(bottom_blob, bottom_blob_fp32, opt) != 0)
        return -100;

    Option opt_fp32 = opt;
    opt_fp32.use_packing_layout = false;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    Mat top_blob_fp32;
    int ret = ShuffleChannel::forward(bottom_blob_fp32, top_blob_fp32, opt_fp32);
    if (ret != 0)
        return ret;

    return restore_layout_from_reference(top_blob_fp32, top_blob, bottom_blob, opt);
}

} // namespace ncnn
