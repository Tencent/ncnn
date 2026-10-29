// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"
#include "unaryop.h"

#include <fenv.h>
#include <math.h>
#include <vector>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#include "unaryop_functor.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_BF16

#include "unaryop_bf16s.h"

void unaryop_bf16s_zvfbfmin(Mat& bottom_top_blob, int op_type, const Option& opt)
{
    unaryop_bf16s(bottom_top_blob, op_type, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
