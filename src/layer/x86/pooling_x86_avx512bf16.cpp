// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "pooling_bf16s.h"

void pooling_global_max_bf16s_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    pooling_global_max_bf16s_sse(bottom_blob, top_blob, opt);
}

void pooling_global_avg_bf16s_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    pooling_global_avg_bf16s_sse(bottom_blob, top_blob, opt);
}

void pooling_max_bf16s_sse_avx512bf16(const Mat& bottom_blob_bordered, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, const Option& opt)
{
    pooling_max_bf16s_sse(bottom_blob_bordered, top_blob, kernel_w, kernel_h, stride_w, stride_h, opt);
}

void pooling_avg_bf16s_sse_avx512bf16(const Mat& bottom_blob_bordered, const Mat& bottom_blob, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_mode, int avgpool_count_include_pad, const Option& opt)
{
    pooling_avg_bf16s_sse(bottom_blob_bordered, bottom_blob, top_blob, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, pad_mode, avgpool_count_include_pad, opt);
}

} // namespace ncnn
