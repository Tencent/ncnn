// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_BF16

#include "gelu_bf16s.h"

void gelu_bf16s_zvfbfmin(Mat& bottom_top_blob, int fast_gelu, const Option& opt)
{
    gelu_bf16s(bottom_top_blob, fast_gelu, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
