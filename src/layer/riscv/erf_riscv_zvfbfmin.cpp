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

#include "erf_bf16s.h"

void erf_bf16s_zvfbfmin(Mat& bottom_top_blob, const Option& opt)
{
    erf_bf16s(bottom_top_blob, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
