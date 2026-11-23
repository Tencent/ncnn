// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "dropout_bf16s.h"

void dropout_bf16s_zvfbfmin(Mat& bottom_top_blob, float scale, const Option& opt)
{
    dropout_bf16s(bottom_top_blob, scale, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
