// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "elu_bf16s.h"

void elu_bf16s_avx512bf16(Mat& a, float alpha, const Option& opt)
{
    elu_bf16s(a, alpha, opt);
}

} // namespace ncnn
