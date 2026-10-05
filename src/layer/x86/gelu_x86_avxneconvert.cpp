// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "gelu_bf16s.h"

void gelu_bf16s_avxneconvert(Mat& a, int fast_gelu, const Option& opt)
{
    gelu_bf16s(a, fast_gelu, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
