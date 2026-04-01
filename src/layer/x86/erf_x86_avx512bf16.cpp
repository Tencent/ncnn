// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "erf_bf16s.h"

void erf_bf16s_avx512bf16(Mat& a, const Option& opt)
{
    erf_bf16s(a, opt);
}

} // namespace ncnn
