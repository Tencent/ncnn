// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "prelu_bf16s.h"

void prelu_bf16s_zvfbfmin(Mat& bottom_top_blob, int num_slope, const Mat& slope_data, const Option& opt)
{
    prelu_bf16s(bottom_top_blob, num_slope, slope_data, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
