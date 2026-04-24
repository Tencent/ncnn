// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_loongarch.h"

namespace ncnn {

GridSample_loongarch::GridSample_loongarch()
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

static int restore_layout_from_reference(const Mat& src, Mat& dst, const Mat& reference, const Option& opt)
{
    Mat tmp = src;

    if (reference.elempack != 1)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat packed;
        convert_packing(tmp, packed, reference.elempack, opt_pack);
        if (packed.empty())
            return -100;

        tmp = packed;
    }

    if (reference.elembits() == 16)
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

int GridSample_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid_blob = bottom_blobs[1];

    if (bottom_blob.elempack == 1 && grid_blob.elempack == 1 && bottom_blob.elembits() == 32 && grid_blob.elembits() == 32)
        return GridSample::forward(bottom_blobs, top_blobs, opt);

    Option opt_fp32 = opt;
    opt_fp32.use_packing_layout = false;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    Mat bottom_blob_fp32;
    if (unpack_or_cast_to_float32(bottom_blob, bottom_blob_fp32, opt) != 0)
        return -100;

    Mat grid_blob_fp32;
    if (unpack_or_cast_to_float32(grid_blob, grid_blob_fp32, opt) != 0)
        return -100;

    std::vector<Mat> bottom_blobs_fp32(2);
    bottom_blobs_fp32[0] = bottom_blob_fp32;
    bottom_blobs_fp32[1] = grid_blob_fp32;

    std::vector<Mat> top_blobs_fp32(1);
    int ret = GridSample::forward(bottom_blobs_fp32, top_blobs_fp32, opt_fp32);
    if (ret != 0)
        return ret;

    top_blobs.resize(1);
    return restore_layout_from_reference(top_blobs_fp32[0], top_blobs[0], bottom_blob, opt);
}

} // namespace ncnn
