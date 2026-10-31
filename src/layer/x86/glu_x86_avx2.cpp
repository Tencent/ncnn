// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "glu_fp32.h"

int glu_fp32_avx2(const Mat& input, Mat& output, int axis, const Option& opt)
{
    return glu_fp32(input, output, axis, opt);
}

} // namespace ncnn
