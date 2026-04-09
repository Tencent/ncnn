// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_loongarch.h"

namespace ncnn {

DeformableConv2D_loongarch::DeformableConv2D_loongarch()
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

    if (unpacked.elembits() == 16)
    {
#if NCNN_BF16
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        cast_bfloat16_to_float32(unpacked, dst, opt_cast);
        if (dst.empty())
            return -100;

        return 0;
#else
        return -100;
#endif
    }

    dst = unpacked;
    return 0;
}

static int restore_output_layout(const Mat& src, Mat& dst, int out_elempack, bool output_bf16, const Option& opt)
{
    Mat tmp = src;

    if (out_elempack != 1)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat packed;
        convert_packing(tmp, packed, out_elempack, opt_pack);
        if (packed.empty())
            return -100;

        tmp = packed;
    }

    if (output_bf16)
    {
#if NCNN_BF16
        cast_float32_to_bfloat16(tmp, dst, opt);
        if (dst.empty())
            return -100;

        return 0;
#else
        return -100;
#endif
    }

    dst = tmp;
    return 0;
}

int DeformableConv2D_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    bool all_pack1_fp32 = true;
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack != 1 || bottom_blobs[i].elembits() != 32)
        {
            all_pack1_fp32 = false;
            break;
        }
    }

    if (all_pack1_fp32)
        return DeformableConv2D::forward(bottom_blobs, top_blobs, opt);

    Option opt_fp32 = opt;
    opt_fp32.use_packing_layout = false;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    std::vector<Mat> bottom_blobs_fp32(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (unpack_or_cast_to_float32(bottom_blobs[i], bottom_blobs_fp32[i], opt) != 0)
            return -100;
    }

    std::vector<Mat> top_blobs_fp32(1);
    int ret = DeformableConv2D::forward(bottom_blobs_fp32, top_blobs_fp32, opt_fp32);
    if (ret != 0)
        return ret;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }

    top_blobs.resize(1);
    return restore_output_layout(top_blobs_fp32[0], top_blobs[0], out_elempack, bottom_blob.elembits() == 16, opt);
}

} // namespace ncnn
