// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "binaryop.h"

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

#if __riscv_vector
#include "rvv_mathfun.h"
#endif

namespace ncnn {

#if NCNN_BF16

#include "binaryop_bf16s.h"

void binaryop_bf16s_zvfbfmin(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    binaryop_bf16s(a, b, c, op_type, opt);
}

void binaryop_scalar_inplace_bf16s_zvfbfmin(Mat& a, unsigned short b, int op_type, const Option& opt)
{
    binary_op_scalar_inplace_bf16s(a, b, op_type, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
