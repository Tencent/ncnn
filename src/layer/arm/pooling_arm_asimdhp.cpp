// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling_arm.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "pooling_fp16s.h"

int pooling_fp16s_asimdhp(const Mat& bottom_blob, const Mat& bottom_blob_bordered, Mat& top_blob, int pooling_type, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int global_pooling, int pad_mode, int avgpool_count_include_pad, const Option& opt)
{
    return pooling_fp16s(bottom_blob, bottom_blob_bordered, top_blob, pooling_type, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, global_pooling, pad_mode, avgpool_count_include_pad, opt);
}

int pooling_fp16sa_asimdhp(const Mat& bottom_blob, const Mat& bottom_blob_bordered, Mat& top_blob, int pooling_type, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int global_pooling, int pad_mode, int avgpool_count_include_pad, const Option& opt)
{
    return pooling_fp16sa(bottom_blob, bottom_blob_bordered, top_blob, pooling_type, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, global_pooling, pad_mode, avgpool_count_include_pad, opt);
}

} // namespace ncnn
