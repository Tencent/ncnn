// Copyright 2026 Tencent
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

#include "swiglu_bf16s.h"

void swiglu_bf16s_zvfbfmin(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit)
{
    swiglu_bf16s(gate, up, top_blob, opt, scale, offset, limit);
}

#endif // NCNN_BF16

} // namespace ncnn
