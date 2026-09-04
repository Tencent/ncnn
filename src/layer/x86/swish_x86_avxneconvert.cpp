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

#include "swish_bf16s.h"

void swish_bf16s_avxneconvert(Mat& a, const Option& opt)
{
    swish_bf16s(a, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
