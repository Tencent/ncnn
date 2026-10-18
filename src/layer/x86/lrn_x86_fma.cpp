// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_x86.h"

#include "avx_mathfun.h"

#include "cpu.h"

namespace ncnn {

#include "lrn_fp32.h"

int lrn_fp32_fma(Mat& bottom_top_blob, int region_type, int local_size, float alpha, float beta, float bias, const Option& opt)
{
    return lrn_fp32(bottom_top_blob, region_type, local_size, alpha, beta, bias, opt);
}

} // namespace ncnn
