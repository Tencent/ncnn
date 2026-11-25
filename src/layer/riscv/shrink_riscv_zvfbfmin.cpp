// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "shrink_bf16s.h"

void shrink_bf16s_zvfbfmin(Mat& bottom_top_blob, float lambd, float bias, const Option& opt)
{
    shrink_bf16s(bottom_top_blob, lambd, bias, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
