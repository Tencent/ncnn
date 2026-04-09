// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lstm_loongarch.h"

namespace ncnn {

LSTM_loongarch::LSTM_loongarch()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static int cast_to_float32_if_needed(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.elembits() != 16)
    {
        dst = src;
        return 0;
    }

#if NCNN_BF16
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    cast_bfloat16_to_float32(src, dst, opt_cast);
    if (dst.empty())
        return -100;

    return 0;
#else
    return -100;
#endif
}

int LSTM_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (!(opt.use_bf16_storage && bottom_blob.elembits() == 16))
        return LSTM::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_fp32;
    if (cast_to_float32_if_needed(bottom_blob, bottom_blob_fp32, opt) != 0)
        return -100;

    Option opt_fp32 = opt;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    Mat top_blob_fp32;
    int ret = LSTM::forward(bottom_blob_fp32, top_blob_fp32, opt_fp32);
    if (ret != 0)
        return ret;

#if NCNN_BF16
    cast_float32_to_bfloat16(top_blob_fp32, top_blob, opt);
    if (top_blob.empty())
        return -100;
#endif

    return 0;
}

int LSTM_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (!(opt.use_bf16_storage && !bottom_blobs.empty() && bottom_blobs[0].elembits() == 16))
        return LSTM::forward(bottom_blobs, top_blobs, opt);

    std::vector<Mat> bottom_blobs_fp32(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (cast_to_float32_if_needed(bottom_blobs[i], bottom_blobs_fp32[i], opt) != 0)
            return -100;
    }

    Option opt_fp32 = opt;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    std::vector<Mat> top_blobs_fp32(top_blobs.size());
    int ret = LSTM::forward(bottom_blobs_fp32, top_blobs_fp32, opt_fp32);
    if (ret != 0)
        return ret;

    top_blobs.resize(top_blobs_fp32.size());
#if NCNN_BF16
    for (size_t i = 0; i < top_blobs_fp32.size(); i++)
    {
        cast_float32_to_bfloat16(top_blobs_fp32[i], top_blobs[i], opt);
        if (top_blobs[i].empty())
            return -100;
    }
#endif

    return 0;
}

} // namespace ncnn
