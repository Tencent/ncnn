// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "glu_fp32.h"

void glu_fp32_fma4(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
    glu_fp32(bottom_blob, top_blob, axis, opt);
}

} // namespace ncnn
