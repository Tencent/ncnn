// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "batchnorm_bf16s.h"

void batchnorm_bf16s_zvfbfmin(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt)
{
    batchnorm_bf16s(bottom_top_blob, a_data, b_data, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
