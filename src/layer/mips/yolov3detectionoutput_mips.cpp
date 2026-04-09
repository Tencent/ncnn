// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "yolov3detectionoutput_mips.h"

namespace ncnn {

Yolov3DetectionOutput_mips::Yolov3DetectionOutput_mips()
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

int Yolov3DetectionOutput_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
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
        return Yolov3DetectionOutput::forward(bottom_blobs, top_blobs, opt);

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

    return Yolov3DetectionOutput::forward(bottom_blobs_fp32, top_blobs, opt_fp32);
}

} // namespace ncnn
