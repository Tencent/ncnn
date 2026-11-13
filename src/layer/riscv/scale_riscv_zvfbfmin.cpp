// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "scale_bf16s.h"

void scale_bf16s_zvfbfmin(Mat& bottom_top_blob, const float* scale, const float* bias, bool bias_term, const Option& opt)
{
    scale_bf16s(bottom_top_blob, scale, bias, bias_term, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
